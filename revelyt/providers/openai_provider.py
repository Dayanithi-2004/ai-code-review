import os
from typing import Any, Type
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from revelyt.providers.base import BaseProvider, T, _count_tokens_tiktoken
from revelyt.providers.factory import register

# Model name constants (avoids SonarQube S1192 literal duplication)
_GPT_52     = "gpt-5.2"
_GPT_52_PRO = "gpt-5.2-pro"
_GPT_5      = "gpt-5"
_GPT_5_MINI = "gpt-5-mini"
_GPT_5_NANO = "gpt-5-nano"
_GPT_41     = "gpt-4.1"


@register(_GPT_52, _GPT_52_PRO, _GPT_5, _GPT_5_MINI, _GPT_5_NANO, _GPT_41)
class OpenAIProvider(BaseProvider):

    # ── Self-describing metadata (read by `revelyt providers`) ─────────────
    ENV_VAR           = "OPENAI_API_KEY"
    STRUCTURED_OUTPUT = "native"
    NOTE              = "Best JSON reliability"
    TIKTOKEN_ENCODING = "o200k_base"

    MODEL_ALIASES = {
        _GPT_52:     _GPT_52,
        _GPT_52_PRO: _GPT_52_PRO,
        _GPT_5:      _GPT_5,
        _GPT_5_MINI: _GPT_5_MINI,
        _GPT_5_NANO: _GPT_5_NANO,
        _GPT_41:     _GPT_41,
    }

    MODEL_COSTS = {
        _GPT_52:     (1.75,  14.00),
        _GPT_52_PRO: (21.00, 168.00),
        _GPT_5:      (1.25,  10.00),
        _GPT_5_MINI: (0.25,   2.00),
        _GPT_5_NANO: (0.05,   0.40),
        _GPT_41:     (2.00,   8.00),
    }
    # ─────────────────────────────────────────────────────────────────────────

    DEFAULT_MODEL = _GPT_52

    def __init__(self, api_key: str = None, model_name: str = DEFAULT_MODEL, **kwargs):
        resolved = self.MODEL_ALIASES.get(model_name.lower(), model_name)

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is missing. Set OPENAI_API_KEY in your environment."
            )

        super().__init__(api_key=self.api_key, model_name=resolved)

        self.cost_per_1m_input, self.cost_per_1m_output = self.MODEL_COSTS.get(
            resolved, (1.75, 14.00)
        )

    def get_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            temperature=0.1,
        )

    def invoke_structured(
        self,
        prompt: ChatPromptTemplate,
        output_schema: Type[T],
        **kwargs: Any,
    ) -> T:
        formatted_prompt = prompt.invoke(kwargs)
        structured_llm   = self.llm.with_structured_output(output_schema, include_raw=True)

        try:
            raw        = structured_llm.invoke(formatted_prompt)
            ai_message = raw.get("raw")
            result     = raw.get("parsed")

            if ai_message and hasattr(ai_message, "usage_metadata") and ai_message.usage_metadata:
                usage = ai_message.usage_metadata
                self.total_input_tokens  += usage.get("input_tokens",  0)
                self.total_output_tokens += usage.get("output_tokens", 0)
            else:
                self.total_input_tokens  += _count_tokens_tiktoken(
                    formatted_prompt.to_string(), self.TIKTOKEN_ENCODING)
                self.total_output_tokens += _count_tokens_tiktoken(
                    str(result), self.TIKTOKEN_ENCODING)

            return result

        except Exception as e:
            print(f"[!] OpenAI structured output failed: {e}")
            return self._empty_result(output_schema)