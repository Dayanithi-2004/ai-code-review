"""
LlamaCppProvider — Local model execution via llama.cpp server with Metal acceleration.

Provides 2-3x faster inference than Ollama on Mac by:
1. Direct Metal API access (no abstraction layers)
2. Better memory management (no Go runtime overhead)
3. Optimized KV cache handling
4. Lower CPU overhead

Setup:
  1. Install: brew install llama.cpp
  2. Download model: huggingface-cli download Qwen/Qwen2.5-Coder-14B-Instruct-GGUF qwen2.5-coder-14b-instruct-q4_k_m.gguf
  3. Start server: llama-server --model <path> --threads 10 --gpu-layers 40 --port 8080
  4. Use: revelyt scan ./repo --provider llama-cpp --llama-cpp-url http://localhost:8080
"""

import os
from typing import Any, Type, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from revelyt.providers.base import BaseProvider, T, _count_tokens_tiktoken
from revelyt.providers.factory import register


@register("llama-cpp")
class LlamaCppProvider(BaseProvider):
    # ── Self-describing metadata (read by `revelyt providers`) ─────────────
    ENV_VAR = "None (local)"
    STRUCTURED_OUTPUT = "repair"
    NOTE = "Requires: llama-server (Metal acceleration)"
    TIKTOKEN_ENCODING = "cl100k_base"

    MODEL_ALIASES = {
        "llama-cpp": "qwen2.5-coder-14b-instruct-q4_k_m",
    }

    MODEL_COSTS = {}  # local — all $0
    # ─────────────────────────────────────────────────────────────────────────

    DEFAULT_URL = "http://localhost:8080"

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "llama-cpp",
        base_url: Optional[str] = None,
        **kwargs,
    ):
        resolved = self.MODEL_ALIASES.get(model_name.lower(), model_name)
        self._base_url = base_url or os.getenv("LLAMA_CPP_URL", self.DEFAULT_URL)
        self._resolved_model = resolved

        super().__init__(api_key=None, model_name=resolved)
        self.cost_per_1m_input = 0.0
        self.cost_per_1m_output = 0.0

    def get_llm(self) -> ChatOpenAI:
        """
        Use ChatOpenAI because llama-server exposes an OpenAI-compatible API.
        This is faster than Ollama because:
        - Direct Metal API access
        - No Go runtime overhead
        - Better memory management
        """
        return ChatOpenAI(
            model=self._resolved_model,
            base_url=f"{self._base_url}/v1",
            api_key="not-needed",  # llama.cpp doesn't require auth
            temperature=0.1,
            max_tokens=8192,
            timeout=120,
            max_retries=2,
        )

    def invoke_structured(
        self,
        prompt: ChatPromptTemplate,
        output_schema: Type[T],
        **kwargs: Any,
    ) -> T:
        """
        Use the base class repair-loop implementation.
        llama.cpp server doesn't support native structured output,
        but it's fast enough that the repair loop rarely triggers.
        """
        return super().invoke_structured(prompt, output_schema, **kwargs)
