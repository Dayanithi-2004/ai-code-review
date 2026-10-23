import os
from typing import Any, Type
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from revelyt.providers.base import BaseProvider, T, _count_tokens_tiktoken
from revelyt.providers.factory import register


@register("claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5")
class AnthropicProvider(BaseProvider):
    # ── Self-describing metadata (read by `revelyt providers`) ─────────────
    ENV_VAR = "ANTHROPIC_API_KEY"
    STRUCTURED_OUTPUT = "native"
    NOTE = ""
    TIKTOKEN_ENCODING = (
        "cl100k_base"  # approximation — Anthropic has no tiktoken encoder
    )

    MODEL_ALIASES = {
        "claude-opus-4-6": "claude-opus-4-6",
        "claude-sonnet-4-6": "claude-sonnet-4-6",
        "claude-haiku-4-5": "claude-haiku-4-5",
    }

    MODEL_COSTS = {
        "claude-opus-4-6": (5.00, 25.00),
        "claude-sonnet-4-6": (3.00, 15.00),
        "claude-haiku-4-5": (1.00, 5.00),
    }
    # ─────────────────────────────────────────────────────────────────────────

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self, api_key: str = None, model_name: str = DEFAULT_MODEL, **kwargs):
        resolved = self.MODEL_ALIASES.get(model_name.lower(), model_name)

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key is missing. Set ANTHROPIC_API_KEY in your environment."
            )

        self._resolved_model = resolved
        super().__init__(api_key=api_key, model_name=resolved)

        self.cost_per_1m_input, self.cost_per_1m_output = self.MODEL_COSTS.get(
            resolved, (3.00, 15.00)
        )

    def get_llm(self) -> ChatAnthropic:
        return ChatAnthropic(
            model=self._resolved_model,
            api_key=self.api_key,
            temperature=0.1,
            max_tokens=4096,
        )

    def invoke_structured(
        self,
        prompt: ChatPromptTemplate,
        output_schema: Type[T],
        **kwargs: Any,
    ) -> T:
        formatted_prompt = prompt.invoke(kwargs)
        structured_llm = self.llm.with_structured_output(
            output_schema, include_raw=True
        )

        # Snapshot tokens before native attempt
        input_snap = self.total_input_tokens
        output_snap = self.total_output_tokens

        try:
            raw = structured_llm.invoke(formatted_prompt)
            ai_message = raw.get("raw")
            result = raw.get("parsed")

            if result is None:
                raise ValueError("Model returned empty response")

            if (
                ai_message
                and hasattr(ai_message, "usage_metadata")
                and ai_message.usage_metadata
            ):
                usage = ai_message.usage_metadata
                self.total_input_tokens = input_snap + usage.get("input_tokens", 0)
                self.total_output_tokens = output_snap + usage.get("output_tokens", 0)
            else:
                self.total_input_tokens = input_snap + _count_tokens_tiktoken(
                    formatted_prompt.to_string(), self.TIKTOKEN_ENCODING
                )
                self.total_output_tokens = output_snap + _count_tokens_tiktoken(
                    str(result), self.TIKTOKEN_ENCODING
                )

            return result

        except Exception as e:
            # Restore to pre-attempt snapshot; fallback will add its own tokens
            self.total_input_tokens = input_snap
            self.total_output_tokens = output_snap
            print(
                f"[!] Anthropic structured output failed ({e}), falling back to repair loop..."
            )
            return super().invoke_structured(prompt, output_schema, **kwargs)
