import os
from typing import Any, Type

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console

from revelyt.providers.base import BaseProvider, T
from revelyt.providers.factory import register

logger = __import__('logging').getLogger(__name__)


@register(
    "openrouter",
    "openrouter/auto",
    "deepseek-chat",
    "deepseek/deepseek-chat",
    "nemotron",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen3-coder",
    "qwen/qwen3-coder:free",
    "gpt-oss",
    "openai/gpt-oss-120b:free"
)
class OpenRouterProvider(BaseProvider):
    """
    OpenRouter Provider — allows access to dozens of models via a single API.
    Compatible with OpenAI's Chat Completion API.
    """

    # ── Self-describing metadata ─────────────
    ENV_VAR           = "OPENROUTER_API_KEY"
    STRUCTURED_OUTPUT = "openai-json"
    NOTE              = "Unified access to massive free agentic models."
    TIKTOKEN_ENCODING = "cl100k_base"

    MODEL_ALIASES = {
        "deepseek-chat": "deepseek/deepseek-chat",
        "nemotron":      "nvidia/nemotron-3-super-120b-a12b:free",
        "qwen3-coder":   "qwen/qwen3-coder:free",
        "gpt-oss":       "openai/gpt-oss-120b:free",
        "llama-3-70b":   "meta-llama/llama-3-70b-instruct",
        "openrouter":    "openrouter/auto",
    }

    # Default costs for unknown models (conservative estimate)
    DEFAULT_COSTS = (0.0, 0.0) # Assume free for these big tests

    MODEL_COSTS = {
        "deepseek/deepseek-chat": (0.14, 0.28),
    }
    # ─────────────────────────────────────────

    DEFAULT_MODEL = "openrouter/auto"

    def __init__(self, api_key: str = None, model_name: str = DEFAULT_MODEL, **kwargs):
        resolved = self.MODEL_ALIASES.get(model_name.lower(), model_name)

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is missing. Set OPENROUTER_API_KEY in your environment."
            )

        super().__init__(api_key=self.api_key, model_name=resolved)

        self.cost_per_1m_input, self.cost_per_1m_output = self.MODEL_COSTS.get(
            resolved, self.DEFAULT_COSTS
        )

    def get_llm(self) -> ChatOpenAI:
        # OpenRouter uses the OpenAI-compatible endpoint
        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/microsoft/revelyt",
                "X-Title": "Revelyt",
            },
            temperature=0.1,
        )

    SUPPORTS_STRUCTURED_OUTPUT = True

    def invoke_structured(
        self,
        prompt: ChatPromptTemplate,
        output_schema: Type[T],
        **kwargs: Any,
    ) -> T:
        """
        Uses OpenAI-style JSON response format which most OpenRouter models support.
        Includes schema cleaning and normalization patterns.
        """
        formatted_prompt = prompt.format(**kwargs)

        try:
            # Note: OpenRouter supports the 'json_object' response_format
            response = self.llm.bind(
                response_format={"type": "json_object"}
            ).invoke(formatted_prompt)

            raw_content = getattr(response, "content", str(response))
            self._track_usage(formatted_prompt, raw_content)

            import json
            # 1. Clean response markers
            text = str(raw_content)
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            # Pass through the aggressive JSON extractor from BaseProvider
            # to handle <think> blocks and truncated JSON if any
            text = self._extract_json(text)
            
            data = json.loads(text)

            # 2. Data Normalization
            # ROOT CAUSE FIX: If LLM returns a raw list, wrap it in a dict so .get() works.
            
            # Map schema names to their expected root array keys
            schema_array_keys = {
                "FindingList": "findings",
                "TriageFileList": "files",
                "RequirementList": "requirements",
                "BatchSummaryOutput": "summaries",
                "VerificationResultList": "results"
            }
            
            expected_key = schema_array_keys.get(output_schema.__name__)

            if isinstance(data, list) and expected_key:
                data = {expected_key: data}
            elif isinstance(data, list):
                data = {"data": data}
            
            # Fuzzy root key matching: If Nemotron returned {"files": [...]} but schema expects {"summaries": [...]}, remap it.
            if isinstance(data, dict) and expected_key and expected_key not in data:
                # Find the first list in the dict and assume it's the data we want
                for k, v in data.items():
                    if isinstance(v, list):
                        data[expected_key] = v
                        break
                
                # Ultimate fallback: if there are no lists at all, inject an empty one
                if expected_key not in data:
                    data[expected_key] = []

            if output_schema.__name__ == "FindingList" and isinstance(data.get("findings"), list):
                for f in data["findings"]:
                    # Ensure all required fields exist to prevent Pydantic crashes
                    if "category" not in f:
                        f["category"] = "General"
                    if "title" not in f:
                        f["title"] = f.get("category", "General Finding")
                    if "severity" not in f:
                        f["severity"] = "LOW"
                    if "description" not in f:
                        f["description"] = "No description provided."
                    if "suggestion" not in f:
                        f["suggestion"] = "Review the affected code."
                    if "confidence" not in f:
                        f["confidence"] = 0.8
                    
                    if isinstance(f.get("confidence"), str):
                        f["confidence"] = {"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.5}.get(f["confidence"].upper(), 0.8)
                    if isinstance(f.get("line_start"), str): f["line_start"] = int(f["line_start"]) if f["line_start"].isdigit() else 0
                    if isinstance(f.get("line_end"), str): f["line_end"] = int(f["line_end"]) if f["line_end"].isdigit() else 0

            if output_schema.__name__ == "RequirementList" and isinstance(data.get("requirements"), list):
                for i, r in enumerate(data["requirements"]):
                    if "id" not in r: r["id"] = i + 1
                    if "description" not in r: r["description"] = "No description provided."
                    if "category" not in r: r["category"] = "General"

            if output_schema.__name__ == "VerificationResultList" and isinstance(data.get("results"), list):
                for i, r in enumerate(data["results"]):
                    if "requirement_id" not in r: r["requirement_id"] = i + 1
                    if "status" not in r: r["status"] = "PARTIAL"
                    if "justification" not in r: r["justification"] = "Missing justification."
                    if "evidence_files" not in r: r["evidence_files"] = []

            if output_schema.__name__ == "BatchSummaryOutput" and isinstance(data.get("summaries"), list):
                for s in data["summaries"]:
                    if "file_path" not in s: s["file_path"] = "unknown"
                    if "purpose" not in s: s["purpose"] = "Logic for this module."
                    if "key_functions" not in s: s["key_functions"] = []
                    if "dependant_count" not in s: s["dependant_count"] = 0

            if output_schema.__name__ == "SummaryOutput":
                if "purpose" not in data: data["purpose"] = "Logic for this module."
                if "key_functions" not in data: data["key_functions"] = []

            # 3. Pydantic validation
            from typing import cast
            schema_any = cast(Any, output_schema)
            if hasattr(schema_any, "model_validate"):
                return schema_any.model_validate(data)
            return schema_any(**data)

        except Exception as e:
            # Fallback to repair loop
            Console().print(f"   [yellow]⚠ OpenRouter structured output failed for {self.model_name}: {e}[/yellow]")
            logger.debug(f"OpenRouter exact error: {e}")
            return super().invoke_structured(prompt, output_schema, **kwargs)
