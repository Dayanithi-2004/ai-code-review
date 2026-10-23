"""
cache/ai_result_cache.py — LLM result caching to prevent duplicate API costs.

Caches AI findings keyed on a hash of (code content + prompt version + context),
so identical code reviewed with the same prompt returns cached results instantly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 43200  # 12 hours


class AIResultCache:
    """
    Caches LLM analysis results to avoid paying for the same chunk+prompt
    combination twice.  Keyed on a combined hash of code content, prompt
    version, and context inputs.
    """

    def __init__(self, cache_dir: str = ".revelyt_cache", ttl: int = DEFAULT_TTL_SECONDS):
        self._cache_path = Path(cache_dir) / "ai_result_cache.json"
        self._ttl = ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0
        self._load()

    def _load(self) -> None:
        if self._cache_path.exists():
            try:
                with open(self._cache_path, "r") as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._cache = {}

    def _save(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "w") as f:
            json.dump(self._cache, f)

    @staticmethod
    def compute_key(code_content: str, prompt_version: str = "v1", context_hash: str = "") -> str:
        """Compute a deterministic cache key."""
        raw = f"{code_content}|{prompt_version}|{context_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get(self, key: str) -> Optional[List[Dict]]:
        """Retrieve cached AI findings."""
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        if time.time() - entry.get("timestamp", 0) > self._ttl:
            del self._cache[key]
            self._misses += 1
            return None
        self._hits += 1
        logger.debug("AI cache HIT (key=%s, savings=$estimated)", key[:8])
        return entry.get("findings")

    def store(self, key: str, findings: List[Dict], tokens_used: int = 0) -> None:
        """Store AI findings in cache."""
        self._cache[key] = {
            "findings": findings,
            "timestamp": time.time(),
            "tokens_used": tokens_used,
        }
        self._save()

    def get_stats(self) -> Dict[str, Any]:
        """Return cache performance statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        estimated_savings = sum(
            v.get("tokens_used", 0) for v in self._cache.values()
        )
        return {
            "total_entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(hit_rate, 1),
            "estimated_token_savings": estimated_savings,
        }

    def prune_expired(self) -> int:
        now = time.time()
        expired = [k for k, v in self._cache.items() if now - v.get("timestamp", 0) > self._ttl]
        for k in expired:
            del self._cache[k]
        if expired:
            self._save()
        return len(expired)
