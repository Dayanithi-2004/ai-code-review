"""
cache/diff_cache.py — Content-addressed caching for diff analysis.

Prevents re-analyzing identical diffs across force-pushes, re-runs,
and incremental commits within the same PR.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default cache TTL: 24 hours
DEFAULT_TTL_SECONDS = 86400


class DiffCache:
    """
    Content-addressed cache backed by a local JSON file.

    Cache keys are SHA-256 hashes of diff content, ensuring identical
    diffs are detected regardless of commit SHA.
    """

    def __init__(self, cache_dir: str = ".revelyt_cache", ttl: int = DEFAULT_TTL_SECONDS):
        self._cache_path = Path(cache_dir) / "diff_cache.json"
        self._ttl = ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        if self._cache_path.exists():
            try:
                with open(self._cache_path, "r") as f:
                    self._cache = json.load(f)
                logger.debug("Loaded diff cache with %d entries", len(self._cache))
            except (json.JSONDecodeError, IOError):
                self._cache = {}

    def _save(self) -> None:
        """Persist cache to disk."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "w") as f:
            json.dump(self._cache, f)

    @staticmethod
    def _hash_content(content: str) -> str:
        """Generate a SHA-256 hash of diff content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def is_cached(self, diff_content: str) -> bool:
        """Check if this diff has already been analyzed."""
        key = self._hash_content(diff_content)
        entry = self._cache.get(key)
        if entry is None:
            return False
        # Check TTL
        if time.time() - entry.get("timestamp", 0) > self._ttl:
            del self._cache[key]
            return False
        return True

    def get(self, diff_content: str) -> Optional[List[Dict]]:
        """Retrieve cached findings for a diff."""
        key = self._hash_content(diff_content)
        entry = self._cache.get(key)
        if entry and time.time() - entry.get("timestamp", 0) <= self._ttl:
            logger.debug("Cache HIT for diff (key=%s)", key[:8])
            return entry.get("findings", [])
        return None

    def store(self, diff_content: str, findings: List[Dict]) -> None:
        """Store findings for a diff."""
        key = self._hash_content(diff_content)
        self._cache[key] = {
            "findings": findings,
            "timestamp": time.time(),
        }
        self._save()
        logger.debug("Cached findings for diff (key=%s)", key[:8])

    def invalidate(self, diff_content: str) -> None:
        """Remove a specific entry from cache."""
        key = self._hash_content(diff_content)
        self._cache.pop(key, None)
        self._save()

    def invalidate_repo(self, repo_name: str) -> int:
        """Invalidate all cache entries for a specific repo."""
        keys_to_remove = [
            k for k, v in self._cache.items()
            if v.get("repo") == repo_name
        ]
        for k in keys_to_remove:
            del self._cache[k]
        self._save()
        return len(keys_to_remove)

    def prune_expired(self) -> int:
        """Remove all expired entries."""
        now = time.time()
        expired = [k for k, v in self._cache.items() if now - v.get("timestamp", 0) > self._ttl]
        for k in expired:
            del self._cache[k]
        self._save()
        return len(expired)

    @property
    def size(self) -> int:
        return len(self._cache)
