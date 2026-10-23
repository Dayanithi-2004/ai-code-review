"""
engines/ai_request.py — Async LLM request handler with batching and retries.

Handles constructing prompts from diff chunks + RAG context, sending
them to LLM providers (OpenAI/Gemini/Anthropic via LangChain), and
managing retries, rate limits, and token budgets.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

from revelyt.plugins.pr_reviewer.engines.base import FileChunk, PRContext

logger = logging.getLogger(__name__)

# Default model configs
DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 4096
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0


class AIRequestClient:
    """
    Handles all LLM API interactions for the PR reviewer.

    - Constructs structured prompts from diff + RAG context
    - Batches multiple chunks into single API calls when possible
    - Implements exponential backoff retry logic
    - Tracks token usage for budget enforcement
    """

    def __init__(
        self,
        provider: str = "gemini",
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self._total_tokens_used = 0
        self._total_requests = 0
        self._llm = None

    def _get_llm(self):
        """Lazy-initialize the LLM based on provider."""
        if self._llm is not None:
            return self._llm

        if self.provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            self._llm = ChatGoogleGenerativeAI(
                model=self.model,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
        elif self.provider == "openai":
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self._llm = ChatAnthropic(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        else:
            raise ValueError(f"Unsupported AI provider: {self.provider}")

        return self._llm

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_review_prompt(self, chunk: FileChunk, context: PRContext) -> str:
        """
        Build a structured prompt for reviewing a single diff chunk.
        Includes RAG context if available.
        """
        prompt_parts = [
            "You are an expert, pragmatic Staff Software Engineer performing a production code review.",
            "Your goal is to identify ONLY high-impact issues in the provided code diff.",
            "",
            "Prioritize:",
            "- Critical bugs",
            "- Security vulnerabilities",
            "- Race conditions",
            "- Data corruption risks",
            "- Concurrency issues",
            "- Logic errors",
            "- Breaking API changes",
            "- Major architectural problems",
            "- Performance regressions",
            "- Incorrect error handling",
            "",
            "Ignore:",
            "- Formatting",
            "- Naming style",
            "- Missing comments or docstrings",
            "- Minor refactoring suggestions",
            "- Personal coding preferences",
            "- Purely stylistic issues",
            "- Test coverage suggestions",
            "",
            "Strict Review Rules:",
            "1. Only comment on code shown in the diff.",
            "2. Do NOT assume behavior of code that is not visible.",
            "3. If something might be an issue but you cannot confirm from the diff, DO NOT report it.",
            "4. Do NOT speculate about missing context.",
            "5. Prefer silence over uncertain feedback.",
            "",
            "Severity Threshold:",
            "Only report issues if they are HIGH SEVERITY or MEDIUM SEVERITY with clear evidence.",
            "Do NOT report LOW severity issues.",
            "",
            "Evidence Requirement:",
            "For each issue you report, include why the issue is a problem, the real-world impact, and a suggested fix.",
            "If you cannot explain the impact clearly, do not report the issue.",
            "",
            "Confidence Guidelines:",
            "- 0.9-1.0 -> The issue is clearly a bug or vulnerability",
            "- 0.7-0.89 -> Strong evidence but some context may be missing",
            "- 0.5-0.69 -> Possible issue but uncertain",
            "Never report issues below 0.5 confidence.",
            "Lower confidence if behavior depends on external code, assumptions are required, or the diff lacks context.",
            "",
            "Diff Context Instruction:",
            "Review the following Git diff.",
            "Lines starting with + are added, - are removed, context lines are unchanged.",
            "Focus your review primarily on added or modified lines.",
            "If the diff exceeds 500 lines, focus only on the most risky areas.",
        ]

        # Security Focus Mode
        if any(keyword in chunk.file_path.lower() for keyword in ["auth", "crypto", "token", "password", "session", "jwt", "security"]):
            prompt_parts.extend([
                "",
                "SECURITY FOCUS MODE: This file appears to handle sensitive operations. Prioritize finding security vulnerabilities.",
            ])

        prompt_parts.extend([
            "",
            f"FILE: {chunk.file_path}",
            f"LANGUAGE: {chunk.language}",
            f"LINES: {chunk.start_line}-{chunk.end_line}",
            f"CHANGE TYPE: {chunk.change_type.value}",
        ])

        # Add RAG context if available
        if context.symbol_definitions:
            prompt_parts.extend([
                "",
                "RELATED CODE CONTEXT:",
            ])
            for symbol, definition in list(context.symbol_definitions.items())[:5]:
                prompt_parts.extend([f"  {symbol}:", f"  ```", definition[:500], "  ```"])

        if context.related_tests:
            prompt_parts.extend([
                "",
                f"RELATED TESTS: {json.dumps(list(context.related_tests.keys())[:5])}",
            ])

        prompt_parts.extend([
            "",
            "Output Format:",
            "Respond in JSON format. If no high-confidence issues are found, respond with an empty issues list.",
            '{"summary": "Short summary of the overall review", "issues": [{"severity": "high|medium|critical", "type": "security|bug|performance|quality", "line": int, "message": "[Title] Explanation of why this is a problem AND real-world impact.", "suggestion": "Concrete suggested fix", "confidence": float}]}',
            "",
            "Final Instruction:",
            "If no high-confidence issues are found, respond with:",
            '{"summary": "No critical issues found in this diff.", "issues": []}',
            "",
            "GIT DIFF:",
            "```",
            chunk.diff_content,
            "```",
        ])

        return "\n".join(prompt_parts)

    # ------------------------------------------------------------------
    # API execution
    # ------------------------------------------------------------------

    def request_review(self, chunk: FileChunk, context: PRContext) -> Dict[str, Any]:
        """
        Send a single chunk for AI review. Returns parsed JSON response.
        Includes retry logic with exponential backoff.
        """
        prompt = self.build_review_prompt(chunk, context)
        return self._call_llm_with_retry(prompt)

    def request_batch_review(
        self, chunks: List[FileChunk], context: PRContext
    ) -> List[Dict[str, Any]]:
        """Review multiple chunks sequentially (async version below)."""
        results = []
        for chunk in chunks:
            result = self.request_review(chunk, context)
            results.append(result)
        return results

    async def request_batch_review_async(
        self, chunks: List[FileChunk], context: PRContext, max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Review multiple chunks concurrently with a semaphore to
        prevent overwhelming the API.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results: List[Dict[str, Any]] = []

        async def _review_one(chunk: FileChunk) -> Dict[str, Any]:
            async with semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.request_review, chunk, context)

        tasks = [_review_one(c) for c in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Replace exceptions with empty results
        return [
            r if isinstance(r, dict) else {"issues": [], "error": str(r)}
            for r in results
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_llm_with_retry(self, prompt: str) -> Dict[str, Any]:
        """Call LLM with exponential backoff retry."""
        llm = self._get_llm()

        for attempt in range(MAX_RETRIES):
            try:
                response = llm.invoke(prompt)
                self._total_requests += 1

                # Extract content
                content = response.content if hasattr(response, "content") else str(response)

                # Try to parse as JSON
                return self._parse_ai_response(content)

            except Exception as e:
                wait_time = RETRY_BACKOFF ** (attempt + 1)
                logger.warning(
                    "AI request failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1, MAX_RETRIES, e, wait_time,
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait_time)
                else:
                    logger.error("AI request failed after %d attempts", MAX_RETRIES)
                    return {"issues": [], "error": str(e)}

        return {"issues": []}

    def _parse_ai_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response, extracting JSON from markdown code blocks if needed."""
        content = content.strip()

        # Strip markdown code fences
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            import re
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            logger.warning("Failed to parse AI response as JSON")
            return {"issues": [], "raw_response": content}

    @property
    def tokens_used(self) -> int:
        return self._total_tokens_used

    @property
    def request_count(self) -> int:
        return self._total_requests

    def get_prompt_hash(self, chunk: FileChunk, context: PRContext) -> str:
        """Generate a deterministic hash for caching purposes."""
        prompt = self.build_review_prompt(chunk, context)
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
