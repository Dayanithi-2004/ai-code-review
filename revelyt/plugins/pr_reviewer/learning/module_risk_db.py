"""
learning/module_risk_db.py — Historical bug tracking per module.

Maintains simple file-level issue counts across PRs to detect
historically error-prone modules and boost their risk scores.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ModuleRiskDB:
    """
    Lightweight JSON-backed database tracking how many issues
    each file/module has had historically.

    This data feeds into the ImpactAnalyzer's ``historical_bug_weight``
    field on FileChunk, flagging files that are chronically bug-prone.
    """

    def __init__(self, db_path: str = ".revelyt_cache/module_risk.json"):
        self._db_path = Path(db_path)
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self._db_path.exists():
            try:
                with open(self._db_path, "r") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._data = {}

    def _save(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._db_path, "w") as f:
            json.dump(self._data, f, indent=2)

    def track_issues(self, file_path: str, issue_count: int, pr_number: int = 0) -> None:
        """Record that a file had N issues in a PR."""
        if file_path not in self._data:
            self._data[file_path] = {
                "total_issues": 0,
                "pr_count": 0,
                "last_pr": 0,
                "history": [],
            }

        entry = self._data[file_path]
        entry["total_issues"] += issue_count
        entry["pr_count"] += 1
        entry["last_pr"] = pr_number
        entry["history"].append({"pr": pr_number, "issues": issue_count})
        # Keep only last 50 entries
        entry["history"] = entry["history"][-50:]
        self._save()

    def get_bug_weight(self, file_path: str) -> float:
        """
        Return a risk weight (0.0 – 5.0) based on historical issues.

        Files with many past issues get a higher weight, which feeds
        into the ImpactAnalyzer.
        """
        entry = self._data.get(file_path)
        if entry is None:
            return 0.0

        total = entry["total_issues"]
        if total >= 20:
            return 5.0
        elif total >= 10:
            return 3.0
        elif total >= 5:
            return 2.0
        elif total >= 2:
            return 1.0
        else:
            return 0.5

    def get_high_risk_modules(self, threshold: float = 3.0) -> List[str]:
        """Return file paths with a bug weight above the threshold."""
        return [
            path for path in self._data
            if self.get_bug_weight(path) >= threshold
        ]

    def get_stats(self, file_path: str) -> Dict[str, Any]:
        """Return full history for a specific module."""
        return self._data.get(file_path, {})
