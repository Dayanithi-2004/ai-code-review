"""
review/summary.py — Generates the top-level Revelyt PR Summary.

Combines risk score, finding counts, key issues, and actionable
recommendations into a polished markdown comment for the PR.
"""

from __future__ import annotations

from typing import List, Optional

from revelyt.plugins.pr_reviewer.engines.base import Finding, Severity
from revelyt.plugins.pr_reviewer.engines.risk_engine import RiskReport
from revelyt.plugins.pr_reviewer.review.formatter import ReviewFormatter


class PRSummaryGenerator:
    """
    Generates the overarching Revelyt PR Summary posted as
    the top-level comment on the pull request.
    """

    def __init__(self):
        self.formatter = ReviewFormatter()

    def generate(
        self,
        findings: List[Finding],
        risk_report: RiskReport,
        pr_title: str = "",
        total_files: int = 0,
        total_lines: int = 0,
        review_mode: str = "deep",
        ai_enabled: bool = True,
    ) -> str:
        """Generate the full PR summary markdown."""
        parts = [
            "# 🛡️ Revelyt PR Review Summary",
            "",
        ]

        # PR metadata
        if pr_title:
            parts.append(f"**PR:** {pr_title}")
        parts.append(f"**Files analyzed:** {total_files} | **Lines changed:** {total_lines}")
        parts.append(f"**Review mode:** `{review_mode}` | **AI:** {'enabled' if ai_enabled else 'disabled'}")
        parts.append("")

        # Risk score
        parts.append(risk_report.to_markdown())
        parts.append("")

        # Finding summary
        parts.append("## 📊 Findings Summary")
        parts.append("")
        parts.append(self.formatter.format_summary_table(findings))
        parts.append("")

        # Key issues
        issues_section = self.formatter.format_findings_list(findings)
        if issues_section:
            parts.append(issues_section)
            parts.append("")

        # Actionable recommendations
        recs = self._generate_recommendations(findings, risk_report)
        if recs:
            parts.append("## 🎯 Actions Required")
            parts.append("")
            for rec in recs:
                parts.append(f"- {rec}")
            parts.append("")

        # Footer
        parts.extend([
            "---",
            "_Review powered by [Revelyt](https://github.com/revelyt) — Enterprise PR Analysis_",
        ])

        return "\n".join(parts)

    def _generate_recommendations(
        self, findings: List[Finding], risk_report: RiskReport
    ) -> List[str]:
        """Generate actionable recommendations based on findings."""
        recs: List[str] = []

        critical = [f for f in findings if f.severity in (Severity.CRITICAL, Severity.HIGH)]
        secrets = [f for f in findings if f.type.value == "secret"]
        deps = [f for f in findings if f.type.value == "dependency"]
        missing_tests = [f for f in findings if f.rule_id == "test-impact-missing"]

        if secrets:
            recs.append(f"🔴 **{len(secrets)} secret(s) detected** — Remove immediately before merge.")

        if critical:
            recs.append(f"🔶 **{len(critical)} critical/high issue(s)** — Must be resolved before approval.")

        if deps:
            recs.append(f"📦 **{len(deps)} dependency vulnerability/issue(s)** — Upgrade affected packages.")

        if missing_tests:
            recs.append(f"🧪 **{len(missing_tests)} file(s) lacking test updates** — Add test coverage.")

        if risk_report.level in ("critical", "high"):
            recs.append("⚠️ PR risk is elevated — Consider requesting additional reviewer.")

        if not recs:
            recs.append("✅ No critical actions required. Proceed with standard review.")

        return recs
