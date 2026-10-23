import os
from typing import Any, Type
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from revelyt.providers.base import BaseProvider, T, _count_tokens_tiktoken
from revelyt.providers.factory import register


# Model name constants (avoids SonarQube S1192 literal duplication)
_GEMINI_31_PRO = "gemini-3.1-pro-preview"
_GEMINI_3_FLASH = "gemini-3-flash-preview"
_GEMINI_31_FLASH = "gemini-3.1-flash-lite-preview"
_GEMINI_25_PRO = "gemini-2.5-pro"

# Retry configuration for rate limiting
_MAX_RETRIES = 3
_INITIAL_BACKOFF = 1.0  # seconds


@register("gemini", _GEMINI_31_PRO, _GEMINI_3_FLASH, _GEMINI_31_FLASH, _GEMINI_25_PRO)
class GoogleProvider(BaseProvider):
    # ── Self-describing metadata (read by `revelyt providers`) ─────────────
    ENV_VAR = "GOOGLE_API_KEY"
    STRUCTURED_OUTPUT = "native"
    NOTE = "Recommended for demos — generous free tier"
    TIKTOKEN_ENCODING = "cl100k_base"  # approximation — Gemini uses SentencePiece

    MODEL_ALIASES = {
        "gemini": _GEMINI_25_PRO,  # default alias for CLI
        _GEMINI_31_PRO: _GEMINI_31_PRO,
        _GEMINI_3_FLASH: _GEMINI_3_FLASH,
        _GEMINI_31_FLASH: _GEMINI_31_FLASH,
        _GEMINI_25_PRO: _GEMINI_25_PRO,
    }

    # (input_cost, output_cost) per 1M tokens
    MODEL_COSTS = {
        _GEMINI_31_PRO: (2.00, 12.00),
        _GEMINI_3_FLASH: (0.50, 3.00),
        _GEMINI_31_FLASH: (0.10, 0.40),
        _GEMINI_25_PRO: (1.25, 10.00),
    }
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, api_key: str = None, model_name: str = "gemini", **kwargs):
        resolved = self.MODEL_ALIASES.get(model_name.lower(), model_name)

        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key is missing. "
                "Get a free key at https://aistudio.google.com and set GOOGLE_API_KEY."
            )

        self._resolved_model = resolved
        super().__init__(api_key=api_key, model_name=resolved)

        self.cost_per_1m_input, self.cost_per_1m_output = self.MODEL_COSTS.get(
            resolved, (0.10, 0.40)
        )

    def get_llm(self) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=self._resolved_model,
            google_api_key=self.api_key,
            temperature=0.1,
            convert_system_message_to_human=True,
        )

    def invoke_structured(
        self,
        prompt: ChatPromptTemplate,
        output_schema: Type[T],
        **kwargs: Any,
    ) -> T:
        import time
        import random

        formatted_prompt = prompt.invoke(kwargs)
        structured_llm = self.llm.with_structured_output(
            output_schema, include_raw=True
        )

        for attempt in range(_MAX_RETRIES):
            # Snapshot tokens before this attempt
            input_snap = self.total_input_tokens
            output_snap = self.total_output_tokens

            try:
                raw = structured_llm.invoke(formatted_prompt)
                ai_message = raw.get("raw")
                result = raw.get("parsed")

                if result is None:
                    # Parsed result is None - this counts as a failed attempt
                    raise ValueError("Model returned empty response")

                if (
                    ai_message
                    and hasattr(ai_message, "usage_metadata")
                    and ai_message.usage_metadata
                ):
                    usage = ai_message.usage_metadata
                    self.total_input_tokens = input_snap + usage.get("input_tokens", 0)
                    self.total_output_tokens = output_snap + usage.get(
                        "output_tokens", 0
                    )
                else:
                    self.total_input_tokens = input_snap + _count_tokens_tiktoken(
                        formatted_prompt.to_string(), self.TIKTOKEN_ENCODING
                    )
                    self.total_output_tokens = output_snap + _count_tokens_tiktoken(
                        str(result), self.TIKTOKEN_ENCODING
                    )

                return result

            except Exception as e:
                err_str = str(e)

                # Restore to pre-attempt snapshot; fallback will add its own tokens
                self.total_input_tokens = input_snap
                self.total_output_tokens = output_snap

                # Check for fatal errors - don't retry
                if "404" in err_str or "NOT_FOUND" in err_str:
                    raise RuntimeError(
                        f"Gemini model '{self._resolved_model}' not found (404). "
                        f"Check MODEL_ALIASES in google_provider.py.\nOriginal error: {e}"
                    ) from e

                # Check if this is a rate limit error
                is_rate_limit = "429" in err_str or "ResourceExhausted" in err_str

                if is_rate_limit:
                    if attempt < _MAX_RETRIES - 1:
                        # Exponential backoff with jitter
                        backoff = _INITIAL_BACKOFF * (2**attempt) + random.uniform(0, 1)
                        print(
                            f"[!] Gemini rate limited (429). Retrying in {backoff:.1f}s... (attempt {attempt + 1}/{_MAX_RETRIES})"
                        )
                        time.sleep(backoff)
                        continue
                    else:
                        print(
                            f"[!] Gemini rate limited after {_MAX_RETRIES} attempts. Falling back to repair loop."
                        )
                else:
                    if attempt < _MAX_RETRIES - 1:
                        print(
                            f"[!] Gemini structured output failed ({e}). Retrying... (attempt {attempt + 1}/{_MAX_RETRIES})"
                        )
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    print(
                        f"[!] Gemini structured output failed after {_MAX_RETRIES} attempts ({e}). Falling back to repair loop."
                    )

                # All retries exhausted - fall back to repair loop
                return super().invoke_structured(prompt, output_schema, **kwargs)

        # Should not reach here, but just in case
        return super().invoke_structured(prompt, output_schema, **kwargs)
