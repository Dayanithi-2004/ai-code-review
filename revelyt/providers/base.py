import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Type, TypeVar, Any
from pydantic import BaseModel, ValidationError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

T = TypeVar('T', bound=BaseModel)

# Compact schema instructions to save ~200 tokens per LLM call vs Pydantic's verbose default.
# The full Pydantic parser is still used for PARSING responses — only the PROMPT is compressed.
COMPACT_SCHEMA_INSTRUCTIONS = {
    "FindingList": '{"findings": [{"severity": "CRITICAL|HIGH|MEDIUM|LOW|INFO", "category": "max 3 words", "title": "3-7 word summary", "line_number": 42, "class_or_method": "ClassName.method or Global", "description": "max 2 sentences", "suggestion": "max 2 sentences", "confidence": 0.0-1.0}]}',
    "SummaryOutput": '{"purpose": "1 sentence", "key_functions": ["funcName1", "funcName2"]}',
    "BatchSummaryOutput": '{"summaries": [{"file_path": "relative/path", "purpose": "1 sentence", "key_functions": ["func — what it does"]}]}',
    "RequirementList": '{"requirements": [{"id": int, "description": "str", "category": "str"}]}',
    "VerificationResultList": '{"results": [{"requirement_id": int, "status": "MET|NOT_MET|PARTIAL", "justification": "1 sentence", "evidence_files": ["path"]}]}',
    "TriageDecision": '{"complexity_score": 1-10, "is_trivial": true/false, "needs_security_review": true/false, "needs_qa_review": true/false}',
    "BatchFindingList": '{"files": [{"file_path": "relative/path", "purpose": "1 short sentence about core file responsibility", "key_functions": ["exact_function_name"], "findings": [{"severity": "CRITICAL|HIGH|MEDIUM|LOW|INFO", "category": "max 3 words", "title": "3-7 word summary", "line_number": 42, "class_or_method": "ClassName.method or Global", "description": "max 2 sentences", "suggestion": "max 2 sentences", "confidence": 0.0-1.0}]}]}',
}

# Concrete one-shot examples for small models that struggle with schema-only instructions
JSON_EXAMPLES = {
    "FindingList": '{"findings": [{"severity": "HIGH", "category": "Security", "title": "Hardcoded database password", "line_number": 15, "class_or_method": "Global", "description": "The database password is hardcoded as a string literal.", "suggestion": "Use environment variables via os.getenv().", "confidence": 0.95}]}',
    "RequirementList": '{"requirements": [{"id": 1, "description": "The system must support user authentication", "category": "Security"}]}',
    "BatchSummaryOutput": '{"summaries": [{"file_path": "app/main.py", "purpose": "FastAPI application entry point", "key_functions": ["create_app — initializes the FastAPI app"]}]}',
    "BatchFindingList": '{"files": [{"file_path": "routes/auth.js", "purpose": "Handles user registration and session tokens", "key_functions": ["POST /login", "verifyDatabaseUser"], "findings": [{"severity": "HIGH", "category": "Security", "title": "Missing input validation", "line_number": 15, "class_or_method": "Global", "description": "User input is passed directly to the database query without validation.", "suggestion": "Add input validation using express-validator.", "confidence": 0.9}]}]}',
}

logger = logging.getLogger(__name__)



def _count_tokens_tiktoken(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Counts tokens using tiktoken.
    Falls back to character heuristic only if tiktoken is unavailable.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4

def _get_model_input_budget(model_name: str) -> int:
    """
    Safe input token budget for local quantized models.

    Qwen2.5-Coder 3B and 14B both have a 32K context window.
    We budget roughly half for the prompt, leaving room for KV cache
    overhead and output generation tokens.

        3B  → 16K  (32K × 0.50)
        7B  → 16K  (32K × 0.50)
        14B → 20K  (32K × 0.625)
        32B → 24K  (32K × 0.75)

    Cloud models (Gemini, GPT, Claude) have 128K–1M context and are
    not constrained here.
    """
    name = model_name.lower()
    if any(tag in name for tag in ("3b", "-3b", "_3b")):
        return 16_000
    if any(tag in name for tag in ("7b", "-7b", "_7b")):
        return 16_000
    if any(tag in name for tag in ("14b", "-14b", "_14b")):
        return 20_000
    if any(tag in name for tag in ("32b", "-32b", "_32b")):
        return 24_000
    return 100_000  # Cloud / unknown: no practical limit


class BaseProvider(ABC):
    """
    Strategy interface for all LLM providers.

    Every concrete subclass MUST declare these class-level attributes so the
    `providers` CLI command can introspect the registry without any hardcoding:

        ENV_VAR          — environment variable name for the API key
        STRUCTURED_OUTPUT — "native" | "repair"
        NOTE             — one-line description shown in the providers table
        MODEL_ALIASES    — dict of cli_alias → resolved_model_name
        MODEL_COSTS      — dict of resolved_model_name → (input_$/1M, output_$/1M)
        TIKTOKEN_ENCODING — tiktoken encoding name used for token estimation
    """

    # ── Self-describing class attributes (subclasses override these) ─────────
    ENV_VAR:           str  = ""
    STRUCTURED_OUTPUT: str  = "repair"   # "native" or "repair"
    NOTE:              str  = ""
    MODEL_ALIASES:     dict = {}
    MODEL_COSTS:       dict = {}
    TIKTOKEN_ENCODING: str  = "cl100k_base"
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, api_key: str = None, model_name: str = None):
        self.api_key = api_key
        self.model_name = model_name
        self.llm: BaseChatModel = self.get_llm()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.cost_per_1m_input = 0.0
        self.cost_per_1m_output = 0.0

    @abstractmethod
    def get_llm(self) -> BaseChatModel:
        pass

    def _invoke_with_backoff(self, prompt, max_retries: int = 3, base_delay: float = 2.0):
        """
        Wrapper around self.llm.invoke that retries on 429 rate-limit errors
        with exponential backoff (2s, 4s, 8s).
        Non-429 errors are raised immediately.
        """
        for attempt in range(max_retries):
            try:
                return self.llm.invoke(prompt)
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "429" in error_str or "rate" in error_str or "too many" in error_str
                if is_rate_limit and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # 2s, 4s, 8s
                    logger.info(
                        "Rate limited (429), retrying in %.1fs (attempt %d/%d)",
                        delay, attempt + 1, max_retries,
                    )
                    time.sleep(delay)
                else:
                    raise
        # Should never reach here, but just in case
        return self.llm.invoke(prompt)  # pragma: no cover

    def invoke_structured(
        self,
        prompt: ChatPromptTemplate,
        output_schema: Type[T],
        **kwargs: Any,
    ) -> T:
        parser = PydanticOutputParser(pydantic_object=output_schema)

        safe_kwargs = {
            k: v.replace("{", "{{").replace("}", "}}") if isinstance(v, str) else v
            for k, v in kwargs.items()
        }

        formatted_prompt = prompt.invoke(safe_kwargs)
        schema_name = output_schema.__name__# ── Token budget guard ─────────────────────────────────────────────
        # Local quantized models (Qwen 3B, 14B) have a 32K context window.
        # Count prompt tokens now so the warning appears before any LLM call,
        # giving the caller a chance to see the issue in logs rather than
        # getting a silent truncation or OOM from the server.
        _prompt_tokens = _count_tokens_tiktoken(
            formatted_prompt.to_string(), self.TIKTOKEN_ENCODING
        )
        _budget = _get_model_input_budget(self.model_name)
        if _prompt_tokens > _budget:
            logger.warning(
                "Prompt for %s exceeds safe input budget: %d tokens > %d limit "
                "(model=%s). Code content may need further truncation.",
                schema_name, _prompt_tokens, _budget, self.model_name,
            )
        # ──────────────────────────────────────────────────────────────────
        compact = COMPACT_SCHEMA_INSTRUCTIONS.get(schema_name)
        example = JSON_EXAMPLES.get(schema_name)
        
        # Build the JSON instruction suffix
        json_suffix = "\n\nCRITICAL: Return ONLY valid JSON. No markdown, no explanation, no ```json fences."
        if compact:
            json_suffix += f"\nRequired schema: {compact}"
        else:
            json_suffix += f"\n{parser.get_format_instructions()}"
        if example:
            json_suffix += f"\nExample output: {example}"
        
        formatted_prompt.messages[-1].content += json_suffix

        first_error = None
        broken_output = ""

        # ---------- Pass 1 ----------
        try:
            response = self._invoke_with_backoff(formatted_prompt)
            raw_content = getattr(response, "content", str(response))
            self._track_usage(formatted_prompt.to_string(), raw_content)
            logger.debug("Pass 1 raw output (first 500 chars): %s", raw_content[:500])
            clean_json = self._extract_json(raw_content)
            return parser.parse(clean_json)

        except Exception as exc:
            first_error = exc
            broken_output = locals().get("clean_json", str(exc))
            logger.warning("Pass 1 failed for %s: %s", schema_name, exc)

        # ---------- Pass 2: Repair ----------
        try:
            repair_content = (
                f"Your previous JSON response could not be parsed.\n\n"
                f"Parse error: {first_error}\n\n"
                f"Your broken output was:\n{broken_output[:2000]}\n\n"
                f"Return ONLY the corrected valid JSON matching this schema: {compact or 'see above'}\n"
                f"No explanation, no markdown, no ```json fences. Just the raw JSON object."
            )
            repair_response = self._invoke_with_backoff(repair_content)
            repair_raw = getattr(repair_response, "content", str(repair_response))
            self._track_usage(repair_content, repair_raw)
            logger.debug("Pass 2 raw output (first 500 chars): %s", repair_raw[:500])
            clean_repair = self._extract_json(repair_raw)
            return parser.parse(clean_repair)

        except Exception as second_error:
            logger.warning("Pass 2 failed for %s: %s", schema_name, second_error)

        # ---------- Pass 3: Template fill (for small models) ----------
        if example:
            try:
                template_prompt = (
                    f"Fill in this JSON template based on your earlier analysis. "
                    f"Replace the example values with real values from your review.\n\n"
                    f"Template:\n{example}\n\n"
                    f"Return ONLY the filled JSON. Nothing else."
                )
                template_response = self._invoke_with_backoff(template_prompt)
                template_raw = getattr(template_response, "content", str(template_response))
                self._track_usage(template_prompt, template_raw)
                logger.debug("Pass 3 raw output (first 500 chars): %s", template_raw[:500])
                clean_template = self._extract_json(template_raw)
                return parser.parse(clean_template)
            except Exception as third_error:
                logger.warning("Pass 3 (template) failed for %s: %s", schema_name, third_error)

        logger.error(
            "All 3 passes failed for %s. First error: %s",
            schema_name,
            first_error,
        )
        return self._empty_result(output_schema)

    def _is_local_model(self) -> bool:
        """Check if this provider uses a local model (Ollama, HuggingFace etc.)."""
        return self.cost_per_1m_input == 0.0 and self.cost_per_1m_output == 0.0

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    def _track_usage(self, prompt_text: str, response_text: str) -> None:
        self.total_input_tokens  += _count_tokens_tiktoken(prompt_text,  self.TIKTOKEN_ENCODING)
        self.total_output_tokens += _count_tokens_tiktoken(response_text, self.TIKTOKEN_ENCODING)

    def get_session_cost(self) -> float:
        input_cost  = (self.total_input_tokens  / 1_000_000) * self.cost_per_1m_input
        output_cost = (self.total_output_tokens / 1_000_000) * self.cost_per_1m_output
        return input_cost + output_cost

    def get_token_summary(self) -> dict:
        return {
            "input_tokens":   self.total_input_tokens,
            "output_tokens":  self.total_output_tokens,
            "total_tokens":   self.total_input_tokens + self.total_output_tokens,
            "estimated_cost": self.get_session_cost(),
            "encoding":       self.TIKTOKEN_ENCODING,
        }

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _extract_json(self, raw_text: str) -> str:
        """Aggressively extract JSON from model output, handling thinking tags,
        markdown fences, and other wrapping that small models produce."""
        # Strip <think>...</think> blocks (Qwen/DeepSeek thinking mode)
        text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
        
        # Extract just the JSON block from the LLM's messy string (user suggested regex)
        # Greedy matching ensures we get from the first brace to the last brace, dropping prefix/suffix filler.
        match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
        if match:
            candidate = match.group(0).strip()
            repaired = self._try_parse_or_repair(candidate)
            if repaired:
                return repaired
                
        # Fallback for severely truncated JSON (missing closing braces entirely)
        obj_match = re.search(r"(\{.*|\[.*)", text, flags=re.DOTALL)
        if obj_match:
            candidate = obj_match.group(1).strip()
            repaired = self._try_parse_or_repair(candidate)
            if repaired:
                return repaired
            return candidate
            
        return text.strip()

    def _try_parse_or_repair(self, candidate: str) -> str | None:
        """Try to parse JSON, repairing trailing commas and truncated brackets."""
        # 1. Try as-is
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
        # 2. Fix trailing commas
        fixed = re.sub(r",\s*([\]}])", r"\1", candidate)
        try:
            json.loads(fixed)
            return fixed
        except json.JSONDecodeError:
            pass
        # 3. Auto-close truncated JSON (e.g. model hit token limit)
        repaired = self._repair_truncated_json(fixed)
        try:
            json.loads(repaired)
            logger.info("Auto-repaired truncated JSON (%d→%d chars)", len(candidate), len(repaired))
            return repaired
        except json.JSONDecodeError:
            pass
        return None

    @staticmethod
    def _repair_truncated_json(text: str) -> str:
        """Auto-close truncated JSON by balancing brackets and braces."""
        # Remove any trailing incomplete string value (after last complete key-value)
        # e.g. '"confidence": 0.7' with no closing }, ], }
        cleaned = text.rstrip()
        # Remove trailing comma if present
        cleaned = re.sub(r",\s*$", "", cleaned)
        # Close them in reverse nesting order
        # Walk the string to figure out nesting order
        stack = []
        for ch in cleaned:
            if ch in "{[":
                stack.append(ch)
            elif ch == "}" and stack and stack[-1] == "{":
                stack.pop()
            elif ch == "]" and stack and stack[-1] == "[":
                stack.pop()
        # Close remaining opens in reverse order
        suffix = ""
        for opener in reversed(stack):
            suffix += "}" if opener == "{" else "]"
        return cleaned + suffix

    @staticmethod
    def _empty_result(output_schema: Type[T]) -> T:
        name = output_schema.__name__
        if name == "FindingList":
            return None  # Signal total failure — agent distinguishes from clean
        if name == "SummaryOutput":
            return output_schema(
                purpose="Summary unavailable — model did not return valid output.",
                key_functions=[],
            )
        if name == "RequirementList":
            return output_schema(requirements=[])
        if name == "VerificationResult":
            return output_schema(
                requirement_id=-1,
                status="PARTIAL",
                justification="Verification unavailable — model did not return valid output.",
                evidence_files=[],
            )
        try:
            return output_schema()
        except Exception:
            return None