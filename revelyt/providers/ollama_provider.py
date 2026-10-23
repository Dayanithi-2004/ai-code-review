"""
OllamaProvider — Local model execution via Ollama.

OSS MODEL ACCURACY FIX — ROOT CAUSE 5 (DeepSeek R1 <think> contamination):

DeepSeek R1 on Ollama generates thousands of tokens of chain-of-thought reasoning
BEFORE the JSON output. Two problems:
  1. The thinking tokens consume context budget, leaving less room for code analysis.
  2. If the model reasons itself into a false conclusion during <think>, it commits
     to it in the JSON output.

Fixes applied here:
  a) For deepseek-r1 models: prepend "/no_think" to the system prompt.
     This is the official Ollama/DeepSeek instruction to suppress chain-of-thought
     and output JSON directly. Reduces token usage by ~60% for R1 models.
  b) The base _extract_json() already strips <think>...</think> via regex.
     We add an additional inline stripper here as a defence-in-depth measure
     for cases where Ollama emits thinking tokens WITHOUT proper XML tags.
"""

import os
import re
from typing import Any, Type

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from revelyt.providers.base import BaseProvider, T
from revelyt.providers.factory import register


@register("llama", "llama3", "mistral", "phi", "codestral", "omnicoder", "qwen2.5-coder", "qwen2.5-32b", "kimi", "qwen2.5-auditor")
class OllamaProvider(BaseProvider):

    # ── Self-describing metadata (read by `revelyt providers`) ─────────────
    ENV_VAR           = "None (local)"
    STRUCTURED_OUTPUT = "repair"
    NOTE              = "Requires: ollama serve"
    TIKTOKEN_ENCODING = "cl100k_base"

    MODEL_ALIASES = {
        "llama":          "llama3",
        "llama3":         "llama3",
        "mistral":        "mistral",
        "phi":            "phi3",
        "codestral":      "codestral",
        "omnicoder":      "carstenuhlig/omnicoder-9b",
        "qwen2.5-coder":  "qwen2.5-coder:14b",
        "qwen2.5-32b":    "qwen2.5-coder:32b",
        "kimi": "kimi-k2.5:cloud",
        "qwen2.5-auditor": "qwen2.5-auditor-16k",
    }

    MODEL_COSTS = {}   # local — all $0
    # ─────────────────────────────────────────────────────────────────────────

    DEFAULT_HOST = "http://localhost:11434"

    # Models that produce chain-of-thought tokens we need to suppress.
    # /no_think is the official DeepSeek instruction; other reasoning models
    # may need similar treatment.
    _THINKING_MODELS = {"deepseek-r1", "deepseek-r1:14b", "deepseek-r1:7b",
                        "deepseek-r1:32b", "deepseek-r1:70b", "qwq",
                        "kimi-k2.5:cloud"}

    def __init__(self, api_key: str = None, model_name: str = "ollama", **kwargs):
        resolved = self.MODEL_ALIASES.get(model_name.lower(), model_name)
        self._host           = os.getenv("OLLAMA_HOST", self.DEFAULT_HOST)
        self._resolved_model = resolved
        # Flag: does this model emit reasoning tokens we need to suppress?
        self._is_thinking_model = any(
            tm in resolved.lower() for tm in self._THINKING_MODELS
        )

        # Enable parallel inference by default (user can override via env var)
        # M3 Pro 18GB can handle 4 parallel slots for small models (<8B)
        os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")

        super().__init__(api_key=None, model_name=resolved)
        self.cost_per_1m_input  = 0.0
        self.cost_per_1m_output = 0.0

    def get_llm(self) -> ChatOllama:
        temp = 0.3 if "omnicoder" in self._resolved_model else 0.1
        return ChatOllama(
            model=self._resolved_model,
            base_url=self._host,
            temperature=temp,
            num_predict=8192,
            keep_alive=-1,  # Keep model loaded in GPU memory (no unload between requests)
        )

    def invoke_structured(
        self,
        prompt: ChatPromptTemplate,
        output_schema: Type[T],
        **kwargs: Any,
    ) -> T:
        """
        Override to inject /no_think for DeepSeek R1 and strip inline
        thinking tokens that arrive without proper XML tags.
        """
        if self._is_thinking_model:
            prompt = self._inject_no_think(prompt)

        # Use the base class repair-loop implementation
        return super().invoke_structured(prompt, output_schema, **kwargs)

    def _inject_no_think(self, prompt: ChatPromptTemplate) -> ChatPromptTemplate:
        """
        Prepend /no_think to the first system message so DeepSeek R1
        skips chain-of-thought generation and outputs JSON directly.

        This is the official Ollama instruction supported by deepseek-r1
        models. It reduces token usage by ~60% and eliminates cases where
        the model reasons itself into false conclusions before the JSON.
        """
        messages = list(prompt.messages)
        if not messages:
            return prompt

        # Find the first system message and prepend /no_think
        new_messages = []
        injected = False
        for msg in messages:
            if not injected and hasattr(msg, "role") and msg.role == "system":
                # Prepend the directive without altering the rest of the prompt
                from langchain_core.messages import SystemMessage
                original_content = msg.content if hasattr(msg, "content") else str(msg)
                new_msg = type(msg)(content=f"/no_think\n\n{original_content}")
                new_messages.append(new_msg)
                injected = True
            else:
                new_messages.append(msg)

        if not injected:
            # No system message found — insert one at the start
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.messages import SystemMessage
            new_messages.insert(0, SystemMessage(content="/no_think"))

        return ChatPromptTemplate.from_messages(new_messages)

    def _extract_json(self, raw_text: str) -> str:
        """
        Extended JSON extractor for Ollama models.

        DeepSeek R1 sometimes emits thinking content inline (without <think> tags)
        when running in certain Ollama versions. We strip common thinking preamble
        patterns before delegating to the base extractor.
        """
        text = raw_text

        # Strip <think>...</think> blocks (handled by base class too, but be explicit)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # Strip thinking preamble patterns that Ollama R1 emits without tags.
        # These always appear BEFORE the JSON and follow recognisable discourse markers.
        thinking_preamble_patterns = [
            r"^(?:Let me (?:analyze|think|review|consider|look at).*?)(?=\{|\[)",
            r"^(?:I (?:need to|should|will|can see|notice|observe).*?)(?=\{|\[)",
            r"^(?:Looking at (?:the|this) (?:code|file|function).*?)(?=\{|\[)",
            r"^(?:After (?:analyzing|reviewing|examining).*?)(?=\{|\[)",
            r"^(?:The (?:code|file|function) (?:has|contains|shows|appears).*?)(?=\{|\[)",
            r"^(?:Based on (?:my analysis|the code|reviewing).*?)(?=\{|\[)",
        ]
        for pattern in thinking_preamble_patterns:
            text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

        # Delegate to base class which handles markdown fences, truncated JSON etc.
        return super()._extract_json(text)