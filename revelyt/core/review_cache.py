import json
import hashlib
import logging
import threading
from pathlib import Path
from typing import Optional

from revelyt.core.schemas import FindingList, Finding

logger = logging.getLogger(__name__)


class ReviewCache:
    """
    Semantic Cache for AI Reviews.
    Hashes the role, filename, and exact prompt payload (e.g. pruned AST content).
    If a developer adds a comment to a PR without changing the code inside a function,
    the AST pruner outputs the exact same snippet. The semantic cache catches this
    and returns the previous LLM findings instantly at 0 token cost.

    Cache key includes:
    - role_name: agent role (Architect, Developer, QA, etc.)
    - file_path: source file path
    - skills_context: wave-level skills rules (changes when rules are updated)
    - code_content: actual code content (changes when code changes)
    - model_name: LLM model (different models may produce different outputs)
    - prohibition_hash: agent prohibition rules (changes when prohibitions are updated)
    """

    def __init__(self, target_dir: Path):
        self.cache_file = target_dir / ".revelyt_cache.json"
        self._cache = self._load()
        self._lock = threading.Lock()

    def _load(self) -> dict:
        if not self.cache_file.exists():
            return {}
        try:
            with open(self.cache_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not load review cache: %s", e)
            return {}

    def _save(self) -> None:
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self._cache, f)
        except Exception as e:
            logger.warning("Could not save review cache: %s", e)

    def _get_key(
        self,
        role_name: str,
        file_path: str,
        skills_context: str,
        code_content: str,
        model_name: str,
        prohibition_hash: str,
    ) -> str:
        payload = (
            f"{role_name}::{file_path}::{skills_context}::{code_content}"
            f"::{model_name}::{prohibition_hash}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(
        self,
        role_name: str,
        file_path: str,
        skills_context: str,
        code_content: str,
        model_name: str,
        prohibition_hash: str,
    ) -> Optional[FindingList]:
        key = self._get_key(
            role_name,
            file_path,
            skills_context,
            code_content,
            model_name,
            prohibition_hash,
        )
        if key in self._cache:
            data = self._cache[key]
            try:
                return FindingList(findings=[Finding(**f) for f in data])
            except Exception:
                return None
        return None

    def set(
        self,
        role_name: str,
        file_path: str,
        skills_context: str,
        code_content: str,
        result: FindingList,
        model_name: str,
        prohibition_hash: str,
    ) -> None:
        key = self._get_key(
            role_name,
            file_path,
            skills_context,
            code_content,
            model_name,
            prohibition_hash,
        )
        with self._lock:
            self._cache[key] = [f.model_dump() for f in result.findings]
            self._save()
