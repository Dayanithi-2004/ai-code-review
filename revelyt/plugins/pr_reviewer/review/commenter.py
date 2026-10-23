"""
review/commenter.py — GitHub comment and Check Run posting.

Posts inline review comments, PR summary comments, and GitHub Check
Runs with annotations. Supports streaming-style incremental posting.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from revelyt.plugins.pr_reviewer.engines.base import Finding
from revelyt.plugins.pr_reviewer.github_client import GitHubClient
from revelyt.plugins.pr_reviewer.review.formatter import ReviewFormatter

logger = logging.getLogger(__name__)

# Maximum inline comments per review to avoid GitHub API limits
MAX_INLINE_COMMENTS = 30


class GitHubCommenter:
    """
    Posts Revelyt findings to GitHub as:
      - Inline review comments on specific lines
      - A top-level PR summary comment
      - GitHub Check Run with annotations
    """

    def __init__(self, client: GitHubClient, formatter: ReviewFormatter):
        self.client = client
        self.formatter = formatter

    def post_full_review(
        self,
        repo_name: str,
        pr_number: int,
        findings: List[Finding],
        risk_markdown: str = "",
        summary_markdown: str = "",
    ) -> None:
        """
        Post a complete review: inline comments + summary + check run.

        Parameters
        ----------
        repo_name : str
            e.g. "owner/repo"
        pr_number : int
        findings : List[Finding]
        risk_markdown : str
            Risk score section from the RiskEngine.
        summary_markdown : str
            Overall summary section.
        """
        # 1. Post inline comments (limited to top N by severity)
        actionable = [f for f in findings if f.is_actionable]
        self._post_inline_comments(repo_name, pr_number, actionable)

        # 2. Post top-level summary
        self._post_summary(repo_name, pr_number, findings, risk_markdown, summary_markdown)

        # 3. Create Check Run (if possible)
        self._create_check_run(repo_name, pr_number, findings)

    def _post_inline_comments(
        self, repo_name: str, pr_number: int, findings: List[Finding]
    ) -> None:
        """Post inline review comments on specific diff lines."""
        # Sort by severity (critical first) and limit
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "informational": 4}
        sorted_findings = sorted(
            findings,
            key=lambda f: severity_order.get(f.severity.value, 5),
        )
        limited = sorted_findings[:MAX_INLINE_COMMENTS]

        comments = []
        for finding in limited:
            body = self.formatter.format_inline_comment(finding)
            comments.append({
                "path": finding.file,
                "line": finding.line,
                "body": body,
            })

        if comments:
            try:
                event = "REQUEST_CHANGES" if any(
                    f.severity.value in ("critical", "high") for f in limited
                ) else "COMMENT"

                self.client.submit_review(
                    repo_name,
                    pr_number,
                    body="Revelyt automated review — see inline comments below.",
                    event=event,
                    comments=comments,
                )
                logger.info("Posted %d inline comments (%s)", len(comments), event)
            except Exception as e:
                logger.error("Failed to post review: %s. Falling back to individual comments.", e)
                # Fallback: post comments individually
                for finding in limited:
                    body = self.formatter.format_inline_comment(finding)
                    self.client.post_review_comment(
                        repo_name, pr_number,
                        finding.file, finding.line, body,
                    )

    def _post_summary(
        self,
        repo_name: str,
        pr_number: int,
        findings: List[Finding],
        risk_markdown: str,
        summary_markdown: str,
    ) -> None:
        """Post the top-level PR summary comment."""
        parts = [
            "# 🛡️ Revelyt PR Review",
            "",
        ]

        if risk_markdown:
            parts.append(risk_markdown)
            parts.append("")

        # Finding summary table
        table = self.formatter.format_summary_table(findings)
        parts.append(table)

        # Key issues list
        issues_list = self.formatter.format_findings_list(findings)
        if issues_list:
            parts.extend(["", issues_list])

        if summary_markdown:
            parts.extend(["", summary_markdown])

        parts.extend([
            "",
            "---",
            "_Powered by [Revelyt](https://github.com/revelyt) — AI-powered code review_",
        ])

        body = "\n".join(parts)
        self.client.post_pr_comment(repo_name, pr_number, body)
        logger.info("Posted PR summary comment")

    def _create_check_run(
        self, repo_name: str, pr_number: int, findings: List[Finding]
    ) -> None:
        """Create a GitHub Check Run with annotations."""
        try:
            pr = self.client.get_pr(repo_name, pr_number)
            head_sha = pr.head.sha

            # Build annotations
            annotations = [
                self.formatter.format_github_annotation(f) for f in findings[:50]
            ]

            critical_count = sum(1 for f in findings if f.severity.value in ("critical", "high"))
            conclusion = "failure" if critical_count > 0 else "success" if not findings else "neutral"

            summary = self.formatter.format_summary_table(findings)

            self.client.create_check_run(
                repo_name=repo_name,
                head_sha=head_sha,
                name="Revelyt PR Review",
                status="completed",
                conclusion=conclusion,
                title=f"Revelyt: {len(findings)} findings ({critical_count} critical)",
                summary=summary,
                annotations=annotations,
            )
            logger.info("Created GitHub Check Run (conclusion=%s)", conclusion)
        except Exception as e:
            logger.warning("Could not create Check Run (permissions?): %s", e)
