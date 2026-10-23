"""
review/formatter.py — Finding output formatter with OWASP severity mapping.

Converts Finding objects into rich GitHub Markdown for inline comments
and PR summary sections with severity badges, CWE references, and
actionable suggestions.
"""

from __future__ import annotations

from typing import Dict, List

from revelyt.plugins.pr_reviewer.engines.base import Finding, FindingType, Severity


_SEVERITY_EMOJI: Dict[Severity, str] = {
    Severity.INFORMATIONAL: "ℹ️",
    Severity.LOW: "💡",
    Severity.MEDIUM: "⚠️",
    Severity.HIGH: "🔶",
    Severity.CRITICAL: "🔴",
}

_SEVERITY_LABEL: Dict[Severity, str] = {
    Severity.INFORMATIONAL: "Informational",
    Severity.LOW: "Low",
    Severity.MEDIUM: "Medium",
    Severity.HIGH: "High",
    Severity.CRITICAL: "Critical",
}

_TYPE_LABEL: Dict[FindingType, str] = {
    FindingType.SECURITY: "🔐 Security",
    FindingType.BUG: "🐛 Bug",
    FindingType.PERFORMANCE: "🚀 Performance",
    FindingType.QUALITY: "📏 Quality",
    FindingType.STYLE: "🎨 Style",
    FindingType.DEPENDENCY: "📦 Dependency",
    FindingType.SECRET: "🔑 Secret",
}


class ReviewFormatter:
    """
    Formats Finding objects into GitHub-compatible markdown.
    """

    def format_inline_comment(self, finding: Finding) -> str:
        """Format a single finding as an inline PR review comment."""
        emoji = _SEVERITY_EMOJI.get(finding.severity, "⚠️")
        sev_label = _SEVERITY_LABEL.get(finding.severity, "Medium")
        type_label = _TYPE_LABEL.get(finding.type, "Quality")

        parts = [
            f"**{emoji} {type_label}** — {sev_label} (confidence: {finding.confidence:.0%})",
            "",
            finding.message,
        ]

        if finding.suggestion:
            parts.extend(["", f"**💡 Suggestion:** {finding.suggestion}"])

        if finding.suggested_patch:
            parts.extend(["", "**Suggested fix:**", "```suggestion", finding.suggested_patch, "```"])

        if finding.cwe_id:
            parts.append(f"\n📚 Reference: [{finding.cwe_id}](https://cwe.mitre.org/data/definitions/{finding.cwe_id.split('-')[1]}.html)")

        if finding.owasp_ref:
            parts.append(f"📚 OWASP: {finding.owasp_ref}")

        parts.append(f"\n`{finding.engine}:{finding.rule_id}`")
        return "\n".join(parts)

    def format_summary_table(self, findings: List[Finding]) -> str:
        """Format a summary table of all findings."""
        if not findings:
            return "✅ No issues found."

        # Count by severity
        counts: Dict[Severity, int] = {}
        for f in findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1

        lines = [
            "| Severity | Count |",
            "|----------|-------|",
        ]
        for sev in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFORMATIONAL):
            count = counts.get(sev, 0)
            if count > 0:
                emoji = _SEVERITY_EMOJI[sev]
                label = _SEVERITY_LABEL[sev]
                lines.append(f"| {emoji} {label} | {count} |")

        lines.append(f"\n**Total: {len(findings)} findings**")
        return "\n".join(lines)

    def format_findings_list(self, findings: List[Finding], max_items: int = 20) -> str:
        """Format a bullet list of key findings."""
        if not findings:
            return ""

        # Sort by severity (critical first)
        severity_order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3, Severity.INFORMATIONAL: 4}
        sorted_findings = sorted(findings, key=lambda f: severity_order.get(f.severity, 5))

        lines = ["### Key Issues", ""]
        for f in sorted_findings[:max_items]:
            emoji = _SEVERITY_EMOJI.get(f.severity, "⚠️")
            lines.append(f"- {emoji} **{f.file}:{f.line}** — {f.message}")

        if len(findings) > max_items:
            lines.append(f"\n_...and {len(findings) - max_items} more issues._")

        return "\n".join(lines)

    def format_github_annotation(self, finding: Finding) -> Dict:
        """Format a finding as a GitHub Check Run annotation."""
        level_map = {
            Severity.INFORMATIONAL: "notice",
            Severity.LOW: "notice",
            Severity.MEDIUM: "warning",
            Severity.HIGH: "failure",
            Severity.CRITICAL: "failure",
        }
        return {
            "path": finding.file,
            "start_line": finding.line,
            "end_line": finding.end_line or finding.line,
            "annotation_level": level_map.get(finding.severity, "warning"),
            "title": f"[{finding.rule_id}] {_SEVERITY_LABEL.get(finding.severity, 'Medium')}",
            "message": finding.message,
        }
