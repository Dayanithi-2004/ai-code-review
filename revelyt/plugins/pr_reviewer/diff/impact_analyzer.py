"""
diff/impact_analyzer.py — Heuristic risk scoring for diff chunks.

Scores each FileChunk based on:
  - Path sensitivity (auth, security, payment, config dirs)
  - Change magnitude (lines changed, complexity)
  - File type (test files get de-risked, lockfiles elevated)
  - Historical bug weight (from learning engine)
"""

from __future__ import annotations

import os
from typing import Dict, List, Set

from revelyt.plugins.pr_reviewer.engines.base import FileChunk


# Path patterns and their risk weights — positive = risky, negative = safe
_PATH_RISK_RULES: List[Dict] = [
    {"patterns": {"auth", "security", "crypto", "session", "token", "password", "secret"}, "weight": 4.0},
    {"patterns": {"payment", "billing", "stripe", "checkout"}, "weight": 3.5},
    {"patterns": {"admin", "permission", "rbac", "role"}, "weight": 3.0},
    {"patterns": {"database", "migration", "schema", "model"}, "weight": 2.5},
    {"patterns": {"api", "endpoint", "route", "controller"}, "weight": 2.0},
    {"patterns": {"config", "settings", "env"}, "weight": 1.5},
    {"patterns": {"test", "spec", "fixture", "mock", "conftest"}, "weight": -2.0},
    {"patterns": {"docs", "readme", "changelog", "license"}, "weight": -3.0},
]

# Extensions that are inherently low-risk or should be skipped
_LOW_RISK_EXTENSIONS: Set[str] = {".md", ".txt", ".rst", ".csv", ".svg", ".png", ".jpg"}

# Extensions signaling dependency changes (elevated risk)
_DEPENDENCY_EXTENSIONS: Set[str] = {".lock", ".toml", ".cfg"}
_DEPENDENCY_FILES: Set[str] = {
    "requirements.txt", "package.json", "package-lock.json", "yarn.lock",
    "Pipfile.lock", "poetry.lock", "go.sum", "Cargo.lock", "Gemfile.lock",
}


class ImpactAnalyzer:
    """
    Assigns ``risk_score`` to FileChunks so downstream engines can
    prioritise expensive analysis (AI) on high-impact changes and
    skip low-value diffs entirely.

    Scores are cumulative floats; a higher score means higher risk.
    """

    def analyze(self, chunks: List[FileChunk]) -> List[FileChunk]:
        """Enrich each chunk with a risk_score."""
        for chunk in chunks:
            chunk.risk_score = self._calculate_risk(chunk)
        return chunks

    def _calculate_risk(self, chunk: FileChunk) -> float:
        score = 0.0

        # 1. Path-based risk
        path_lower = chunk.file_path.lower()
        for rule in _PATH_RISK_RULES:
            if any(p in path_lower for p in rule["patterns"]):
                score += rule["weight"]

        # 2. Low-risk extensions
        if chunk.extension in _LOW_RISK_EXTENSIONS:
            score -= 2.0

        # 3. Dependency file changes
        basename = os.path.basename(chunk.file_path)
        if basename in _DEPENDENCY_FILES or chunk.extension in _DEPENDENCY_EXTENSIONS:
            score += 2.5

        # 4. Change magnitude
        total = chunk.total_changed_lines
        if total > 200:
            score += 3.0
        elif total > 100:
            score += 2.0
        elif total > 50:
            score += 1.0

        # 5. Complexity (already computed by SemanticDiffAnalyzer)
        if chunk.complexity > 15:
            score += 3.0
        elif chunk.complexity > 10:
            score += 2.0
        elif chunk.complexity > 5:
            score += 1.0

        # 6. Historical bug weight (injected by learning engine)
        score += chunk.historical_bug_weight

        return max(score, 0.0)  # Floor at 0

    def sort_by_risk(self, chunks: List[FileChunk], descending: bool = True) -> List[FileChunk]:
        """Return chunks sorted by risk_score."""
        return sorted(chunks, key=lambda c: c.risk_score, reverse=descending)
