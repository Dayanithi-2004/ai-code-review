"""
engines/risk_engine.py — PR Risk Score calculator.

Synthesises diff metrics (lines changed, file sensitivity, complexity,
dependency updates, test coverage gaps) into a crisp 0–10 risk score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from revelyt.plugins.pr_reviewer.engines.base import FileChunk, Finding, Severity

logger = logging.getLogger(__name__)


@dataclass
class RiskFactor:
    name: str
    score: float
    description: str


@dataclass
class RiskReport:
    """Final risk assessment for a PR."""
    score: float                     # 0.0 – 10.0
    level: str                       # "low", "medium", "high", "critical"
    factors: List[RiskFactor] = field(default_factory=list)
    recommendation: str = ""

    def to_markdown(self) -> str:
        emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(self.level, "⚪")
        lines = [
            f"## {emoji} PR Risk Score: {self.score:.1f} / 10.0 ({self.level.upper()})",
            "",
        ]
        if self.factors:
            lines.append("| Factor | Impact | Detail |")
            lines.append("|--------|--------|--------|")
            for f in self.factors:
                lines.append(f"| {f.name} | +{f.score:.1f} | {f.description} |")
        if self.recommendation:
            lines.extend(["", f"**Recommendation:** {self.recommendation}"])
        return "\n".join(lines)


class RiskEngine:
    """
    Calculates a holistic PR Risk Score by aggregating multiple factors.
    """

    def calculate(
        self,
        chunks: List[FileChunk],
        findings: List[Finding],
    ) -> RiskReport:
        factors: List[RiskFactor] = []

        # 1. Change magnitude
        total_files = len({c.file_path for c in chunks})
        total_lines = sum(c.total_changed_lines for c in chunks)
        magnitude_score = self._score_magnitude(total_files, total_lines)
        if magnitude_score > 0:
            factors.append(RiskFactor(
                "Change Size", magnitude_score,
                f"{total_files} files, {total_lines} lines changed",
            ))

        # 2. Sensitive file changes
        sensitive_score = self._score_sensitive_files(chunks)
        if sensitive_score > 0:
            sensitive_files = [c.file_path for c in chunks if c.risk_score >= 4.0]
            factors.append(RiskFactor(
                "Sensitive Files", sensitive_score,
                f"Modified: {', '.join(sensitive_files[:3])}",
            ))

        # 3. Complexity score
        max_complexity = max((c.complexity for c in chunks), default=0)
        avg_complexity = sum(c.complexity for c in chunks) / max(len(chunks), 1)
        complexity_score = self._score_complexity(max_complexity, avg_complexity)
        if complexity_score > 0:
            factors.append(RiskFactor(
                "Code Complexity", complexity_score,
                f"Max: {max_complexity:.0f}, Avg: {avg_complexity:.1f}",
            ))

        # 4. Security findings
        critical_findings = [f for f in findings if f.severity in (Severity.CRITICAL, Severity.HIGH)]
        security_score = min(len(critical_findings) * 1.5, 3.0)
        if security_score > 0:
            factors.append(RiskFactor(
                "Security Issues", security_score,
                f"{len(critical_findings)} critical/high findings",
            ))

        # 5. Test coverage gaps
        missing_tests = [c for c in chunks if c.test_coverage_status == "missing"]
        test_score = min(len(missing_tests) * 0.5, 2.0)
        if test_score > 0:
            factors.append(RiskFactor(
                "Missing Tests", test_score,
                f"{len(missing_tests)} files without test updates",
            ))

        # 6. Dependency changes
        dep_chunks = [c for c in chunks if any(
            kw in c.file_path.lower() for kw in ("requirements", "package.json", "go.mod", "cargo.toml", "pipfile")
        )]
        dep_score = min(len(dep_chunks) * 1.0, 2.0)
        if dep_score > 0:
            factors.append(RiskFactor(
                "Dependency Changes", dep_score,
                f"{len(dep_chunks)} dependency file(s) modified",
            ))

        # Aggregate
        raw_score = sum(f.score for f in factors)
        final_score = min(raw_score, 10.0)

        level = self._score_to_level(final_score)
        recommendation = self._get_recommendation(level, factors)

        return RiskReport(
            score=final_score,
            level=level,
            factors=factors,
            recommendation=recommendation,
        )

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_magnitude(self, files: int, lines: int) -> float:
        score = 0.0
        if lines > 500:
            score += 2.0
        elif lines > 200:
            score += 1.5
        elif lines > 100:
            score += 1.0
        elif lines > 50:
            score += 0.5

        if files > 30:
            score += 1.5
        elif files > 15:
            score += 1.0
        elif files > 5:
            score += 0.5

        return score

    def _score_sensitive_files(self, chunks: List[FileChunk]) -> float:
        sensitive = [c for c in chunks if c.risk_score >= 4.0]
        return min(len(sensitive) * 1.0, 3.0)

    def _score_complexity(self, max_c: float, avg_c: float) -> float:
        score = 0.0
        if max_c > 20:
            score += 2.0
        elif max_c > 10:
            score += 1.0
        if avg_c > 8:
            score += 1.0
        return score

    def _score_to_level(self, score: float) -> str:
        if score >= 7.0:
            return "critical"
        elif score >= 5.0:
            return "high"
        elif score >= 3.0:
            return "medium"
        else:
            return "low"

    def _get_recommendation(self, level: str, factors: List[RiskFactor]) -> str:
        if level == "critical":
            return "🔴 This PR requires senior/security team review before merging."
        elif level == "high":
            return "🟠 Additional review recommended. Consider breaking into smaller PRs."
        elif level == "medium":
            return "🟡 Standard review process is sufficient."
        else:
            return "🟢 Low risk. Proceed with normal review."
