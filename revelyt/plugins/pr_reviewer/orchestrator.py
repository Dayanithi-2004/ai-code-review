"""
orchestrator.py — Main entry point for Revelyt PR Review.

Bridges the CLI / CI/CD interface with the review pipeline.
Handles GitHub API access, diff fetching, pipeline execution,
and result posting.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from revelyt.plugins.pr_reviewer.engines.base import PRContext
from revelyt.plugins.pr_reviewer.engines.ai_request import AIRequestClient
from revelyt.plugins.pr_reviewer.github_client import GitHubClient
from revelyt.plugins.pr_reviewer.pipeline.review_pipeline import ReviewPipeline, PipelineResult
from revelyt.plugins.pr_reviewer.filters.limits import ReviewMode
from revelyt.plugins.pr_reviewer.review.formatter import ReviewFormatter
from revelyt.plugins.pr_reviewer.review.commenter import GitHubCommenter
from revelyt.plugins.pr_reviewer.review.summary import PRSummaryGenerator

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class ReviewConfig:
    """Configuration for a PR review run."""
    token: str = ""
    repo_name: str = ""
    pr_number: int = 0
    mode: ReviewMode = ReviewMode.DEEP
    enable_ai: bool = True
    ai_provider: str = "gemini"
    ai_model: str = "gemini-2.0-flash"
    post_comments: bool = True
    dry_run: bool = False
    repo_root: str = "."
    github_base_url: Optional[str] = None  # For GitHub Enterprise


class PROrchestrator:
    """
    Main orchestrator for Revelyt PR reviews.

    Usage::

        config = ReviewConfig(token="ghp_xxx", repo_name="owner/repo", pr_number=42)
        orchestrator = PROrchestrator(config)
        result = orchestrator.run()
    """

    def __init__(self, config: ReviewConfig):
        self.config = config

        # GitHub client
        self.github = GitHubClient(
            token=config.token,
            base_url=config.github_base_url,
        )

        # AI client (optional)
        self.ai_client: Optional[AIRequestClient] = None
        if config.enable_ai:
            self.ai_client = AIRequestClient(
                provider=config.ai_provider,
                model=config.ai_model,
            )

        # Pipeline
        self.pipeline = ReviewPipeline(
            ai_client=self.ai_client,
            enable_ai=config.enable_ai,
            review_mode=config.mode,
            repo_root=config.repo_root,
        )

        # Review posting
        self.formatter = ReviewFormatter()
        self.commenter = GitHubCommenter(self.github, self.formatter)
        self.summary_gen = PRSummaryGenerator()

    def run(self) -> PipelineResult:
        """Execute the full review workflow."""
        config = self.config

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # ── Step 1: Fetch PR data ─────────────────────────────────
            task = progress.add_task("Fetching PR metadata...", total=None)
            metadata = self.github.get_pr_metadata(config.repo_name, config.pr_number)
            progress.update(task, description=f"PR #{config.pr_number}: {metadata['title']}")

            # ── Step 2: Fetch diff ────────────────────────────────────
            progress.update(task, description="Fetching PR diff...")
            raw_diff = self.github.get_pr_diff(config.repo_name, config.pr_number)

            if not raw_diff:
                console.print("[yellow]No diff content found for this PR.[/yellow]")
                return PipelineResult()

            # ── Step 3: Build context ─────────────────────────────────
            progress.update(task, description="Building PR context...")
            context = PRContext(
                repo_name=config.repo_name,
                pr_number=config.pr_number,
                pr_title=metadata["title"],
                pr_description=metadata["description"],
                base_branch=metadata["base_branch"],
                head_branch=metadata["head_branch"],
                author=metadata["author"],
            )

            # ── Step 4: Execute pipeline ──────────────────────────────
            progress.update(task, description="Analyzing PR...")
            result = self.pipeline.execute(raw_diff, context)

            # ── Step 5: Display results ───────────────────────────────
            progress.update(task, description="Formatting results...")

        # Print CLI output
        self._print_cli_results(result, context)

        # ── Step 6: Post to GitHub ────────────────────────────────────
        if config.post_comments and not config.dry_run:
            self._post_to_github(result, context)
        elif config.dry_run:
            console.print("\n[yellow]🔶 DRY RUN — No comments posted to GitHub.[/yellow]")

        return result

    def run_local_diff(self, raw_diff: str) -> PipelineResult:
        """
        Review a local diff (e.g., from ``git diff HEAD~1``).
        Does not interact with GitHub.
        """
        context = PRContext(
            repo_name="local",
            pr_title="Local diff review",
        )

        console.print("[bold blue]🔍 Revelyt Local Review[/bold blue]\n")

        result = self.pipeline.execute(raw_diff, context)
        self._print_cli_results(result, context)
        return result

    # ------------------------------------------------------------------
    # CLI output
    # ------------------------------------------------------------------

    def _print_cli_results(self, result: PipelineResult, context: PRContext) -> None:
        """Print formatted results to the terminal."""
        # Risk score panel
        if result.risk_report:
            console.print(Panel(
                result.risk_report.to_markdown(),
                title="📊 Risk Assessment",
                border_style="blue",
            ))

        # Findings
        if result.findings:
            console.print(f"\n[bold]Found {len(result.findings)} issue(s):[/bold]\n")

            for i, finding in enumerate(result.findings[:20], 1):
                sev_colors = {
                    "critical": "red", "high": "dark_orange",
                    "medium": "yellow", "low": "green", "informational": "dim",
                }
                color = sev_colors.get(finding.severity.value, "white")
                console.print(
                    f"  [{color}]{i}. [{finding.severity.value.upper()}][/{color}] "
                    f"[bold]{finding.file}:{finding.line}[/bold] — {finding.message}"
                )
                if finding.suggestion:
                    console.print(f"     💡 {finding.suggestion}")

            if len(result.findings) > 20:
                console.print(f"\n  ... and {len(result.findings) - 20} more findings.")
        else:
            console.print("\n[green]✅ No issues found! Clean PR.[/green]")

        # Metrics
        console.print(f"\n⏱️  Analysis completed in {result.execution_time_ms:.0f}ms")
        console.print(f"📁 {result.chunks_analyzed} chunks analyzed | Mode: {result.review_mode.value}")

    # ------------------------------------------------------------------
    # GitHub posting
    # ------------------------------------------------------------------

    def _post_to_github(self, result: PipelineResult, context: PRContext) -> None:
        """Post findings and summary to GitHub."""
        if not result.findings and result.risk_report and result.risk_report.level == "low":
            # Post a simple approval
            self.github.post_pr_comment(
                self.config.repo_name,
                self.config.pr_number,
                "# 🛡️ Revelyt PR Review\n\n✅ No issues found. LGTM! 🎉",
            )
            console.print("[green]✅ Posted clean review to GitHub.[/green]")
            return

        # Generate full summary
        risk_md = result.risk_report.to_markdown() if result.risk_report else ""
        summary_md = self.summary_gen.generate(
            findings=result.findings,
            risk_report=result.risk_report,
            pr_title=context.pr_title,
            total_files=result.stats.total_files if result.stats else 0,
            total_lines=result.stats.total_lines if result.stats else 0,
            review_mode=result.review_mode.value,
            ai_enabled=result.ai_enabled,
        )

        # Post full review (inline + summary + check run)
        self.commenter.post_full_review(
            repo_name=self.config.repo_name,
            pr_number=self.config.pr_number,
            findings=result.findings,
            risk_markdown=risk_md,
            summary_markdown=summary_md,
        )

        console.print(f"[green]✅ Posted {len(result.findings)} findings to GitHub.[/green]")
