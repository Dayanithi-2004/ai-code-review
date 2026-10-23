import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from revelyt.providers.base import BaseProvider
from revelyt.providers.factory import register


@register("starcoder2", "codellama", "qwen2.5", "qwen3", "deepseek", "stablecode", "kimi")
class HuggingFaceProvider(BaseProvider):

    # ── Self-describing metadata (read by `revelyt providers`) ─────────────
    ENV_VAR           = "HUGGINGFACEHUB_API_TOKEN"
    STRUCTURED_OUTPUT = "repair"
    NOTE              = "Budget / TDD mode — weaker JSON"
    TIKTOKEN_ENCODING = "cl100k_base"

    MODEL_ALIASES = {
        "starcoder2":  "bigcode/starcoder2-15b",
        "codellama":   "codellama/CodeLlama-7b-Instruct-hf",
        "qwen2.5":     "Qwen/Qwen2.5-Coder-7B-Instruct",
        "qwen3":       "Qwen/Qwen3-Coder-Next",
        "deepseek":    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "stablecode":  "stabilityai/stable-code-instruct-3b",
        "kimi":        "moonshotai/kimi-k2.5",
    }

    MODEL_COSTS = {}   # free tier — all $0
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, api_key: str = None, model_name: str = "starcoder2", **kwargs):
        repo_id = self.MODEL_ALIASES.get(model_name.lower(), model_name)

        api_key = api_key or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not api_key:
            raise ValueError(
                "Hugging Face API token is missing. "
                "Set HUGGINGFACEHUB_API_TOKEN or pass it via --api-key."
            )

        self._repo_id = repo_id
        super().__init__(api_key=api_key, model_name=repo_id)
        self.cost_per_1m_input  = 0.0
        self.cost_per_1m_output = 0.0

    def get_llm(self) -> ChatHuggingFace:
        endpoint = HuggingFaceEndpoint(
            repo_id=self._repo_id,
            huggingfacehub_api_token=self.api_key,
            task="text-generation",
            max_new_tokens=4096,
            temperature=0.1,
            repetition_penalty=1.03,
        )
        return ChatHuggingFace(llm=endpoint)