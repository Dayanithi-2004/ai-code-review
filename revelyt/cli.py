import asyncio
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

# Suppress harmless LangChain/Pydantic serializer warnings that spam the console
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

# ── Root logger configuration ───────────────────────────────────────────────
_log_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
_file_handler = logging.FileHandler("revelyt.log", mode="w")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(_log_formatter)

_root = logging.getLogger("revelyt")
_root.setLevel(logging.DEBUG)
_root.addHandler(_file_handler)

logger = logging.getLogger(__name__)
# ─────────────────────────────────────────────────────────────────────────────

import typer
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel

from revelyt.core.orchestrator import RevelytOrchestrator
from revelyt.providers.factory import ProviderFactory

app = typer.Typer(
    help=(
        "Revelyt — Senior-Level AI Code Reviewer.\n\n"
        "Scans Python and Node.js codebases with a team of LLM agents and produces "
        "a severity-ranked PDF/HTML report.\n\n"
        "Run [bold cyan]revelyt <command> --help[/bold cyan] for full options on any command."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# ── SCAN COMMAND ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def scan(
    target_path: str = typer.Argument(
        ..., help="Local directory path or GitHub URL to scan"
    ),
    brd: Optional[str] = typer.Option(
        None,
        "--brd",
        "-b",
        help="Path to a Markdown BRD file for Phase 4 requirement verification",
    ),
    provider: str = typer.Option(
        "gemini",
        "--provider",
        "-p",
        help="LLM provider alias to use. Run `revelyt providers` to see all options.",
    ),
    oss_provider: Optional[str] = typer.Option(
        None,
        "--oss-provider",
        help="OSS/local model for Wave 2 + PM summarisation (e.g. qwen2.5-coder). "
        "Defaults to --provider if not set.",
    ),
    diff: bool = typer.Option(
        False,
        "--diff",
        "-d",
        help="Enable diff-only mode: analyse only changed + impacted files",
    ),
    base: str = typer.Option(
        "main",
        "--base",
        help="Git ref to diff against (branch, tag, or commit). Only used with --diff.",
    ),
    head: str = typer.Option(
        "HEAD", "--head", help="Git ref for current state. Only used with --diff."
    ),
    agents: Optional[str] = typer.Option(
        None,
        "--agents",
        "-a",
        help="Comma-separated agents to run (architect,developer,qa,devexp,doc). Default: all.",
    ),
    report_format: str = typer.Option(
        "pdf", "--format", "-f", help="Output report format: pdf or html."
    ),
    concurrency: int = typer.Option(
        4,
        "--concurrency",
        "-c",
        help="Max parallel LLM calls. Increase for faster scans (if RAM allows). Default: 4.",
    ),
    skip_wave2_clean: bool = typer.Option(
        True,
        "--skip-wave2-clean/--no-skip-wave2-clean",
        help="Skip Wave 2 on files with no Wave 1 findings. Saves time on clean files. Default: True.",
    ),
    llama_cpp_url: Optional[str] = typer.Option(
        None,
        "--llama-cpp-url",
        help="URL for llama.cpp server (e.g. http://localhost:8080). "
        "Auto-enables Metal-accelerated inference when --provider llama-cpp.",
    ),
):
    """
    Run the full 6-Phase Revelyt pipeline on a target repository.

    Phases: Ingestion → Dependency Mapping → AI Review → BRD Verification →
    Severity Elevation → PDF/HTML Report.

    Examples:\n
      revelyt scan ./my-project --provider gemini-2.5-pro\n
      revelyt scan https://github.com/org/repo -b requirements.md -p gpt-5\n
      revelyt scan ./my-project --diff --base develop\n
      revelyt scan ./my-project --agents architect,qa --provider gemini-2.5-pro\n
      revelyt scan ./my-project --provider gemini-2.5-pro --oss-provider qwen2.5-coder\n
      revelyt scan ./my-project --format html\n
      revelyt scan ./my-project -c 8 --provider qwen2.5-coder:14b\n
      revelyt scan ./my-project --skip-wave2-clean --provider qwen2.5-coder:14b
    """
    logger.info(
        "CLI scan: target=%s, provider=%s, agents=%s, format=%s, concurrency=%d, skip_wave2_clean=%s, llama_cpp_url=%s",
        target_path,
        provider,
        agents,
        report_format,
        concurrency,
        skip_wave2_clean,
        llama_cpp_url,
    )
    try:
        orchestrator = RevelytOrchestrator(
            target_path=target_path,
            brd_path=brd,
            provider_name=provider,
            oss_provider_name=oss_provider,
            diff_base=base if diff else None,
            diff_head=head if diff else None,
            active_agents=agents,
            report_format=report_format,
            concurrency=concurrency,
            skip_wave2_clean=skip_wave2_clean,
            llama_cpp_url=llama_cpp_url,
        )
        asyncio.run(orchestrator.run_scan())

    except FileNotFoundError as e:
        console.print(f"\n❌ [bold red]File Error:[/bold red] {e}")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"\n❌ [bold red]Configuration Error:[/bold red] {e}")
        console.print(
            "   Run [cyan]revelyt providers[/cyan] to see available providers."
        )
        raise typer.Exit(1)
    except Exception as e:
        logger.error("Critical pipeline failure: %s", e, exc_info=True)
        console.print(f"\n💥 [bold red]Critical Pipeline Failure:[/bold red] {e}")
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# ── VERIFY COMMAND ───────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def verify(
    brd: str = typer.Argument(..., help="Path to the .md BRD file"),
    provider: str = typer.Option(
        "gemini",
        "--provider",
        "-p",
        help="LLM provider to use for parsing and verification",
    ),
    session_db: Optional[str] = typer.Option(
        None,
        "--session",
        "-s",
        help="Path to an existing session SQLite database to verify against. "
        "If omitted, starts a fresh in-memory session.",
    ),
):
    """
    Standalone Phase 4: parse a BRD and verify it against an existing scan session.

    Re-runs BRD verification without paying for a full codebase re-scan.
    Useful when requirements change but code has not.

    Examples:\n
      revelyt verify requirements.md\n
      revelyt verify requirements.md --session ./previous_scan.db\n
      revelyt verify requirements.md -p gpt-5 -s ./scan.db
    """
    from revelyt.core.session import SessionManager
    from revelyt.agents.product_manager_agent import ProductManagerAgent

    brd_path = Path(brd).resolve()
    if not brd_path.exists():
        console.print(f"❌ [bold red]BRD file not found:[/bold red] {brd_path}")
        raise typer.Exit(1)

    try:
        provider_instance = ProviderFactory.create(provider)
    except ValueError as e:
        console.print(f"❌ [bold red]Provider Error:[/bold red] {e}")
        raise typer.Exit(1)

    db_path = session_db if session_db else ":memory:"
    session = SessionManager(db_path=db_path)

    console.print("\n⚖️  [bold blue]Standalone BRD Verification[/bold blue]")
    console.print(f"   ↳ Provider:  [cyan]{type(provider_instance).__name__}[/cyan]")
    console.print(f"   ↳ BRD file:  [dim]{brd_path}[/dim]")
    console.print(f"   ↳ Session:   [dim]{db_path}[/dim]")

    pm_agent = ProductManagerAgent(session, provider_instance, brd_path.parent)
    pm_agent.verify_brd(brd_path)


# ─────────────────────────────────────────────────────────────────────────────
# providers
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def providers():
    """
    List all registered LLM providers and their CLI aliases.

    Reads directly from the provider registry — adding a new provider file
    automatically makes it appear here with no changes needed in cli.py.

    Example:\n
      revelyt providers
    """
    registry = ProviderFactory.list_providers()

    # Deduplicate: registry maps alias → class, collapse to class → [aliases]
    seen: dict = {}
    for alias, cls in sorted(registry.items()):
        seen.setdefault(cls, []).append(alias)

    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Revelyt — Registered LLM Providers[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print()

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold white",
        border_style="dim",
        expand=False,
    )
    table.add_column("Provider", style="bold cyan", no_wrap=True)
    table.add_column("CLI Aliases", style="green", no_wrap=False, overflow="fold")
    table.add_column("Env Variable", style="yellow", no_wrap=True)
    table.add_column("Structured Output", style="white", no_wrap=True)
    table.add_column("Notes", style="dim", no_wrap=False)

    for cls, aliases in seen.items():
        generic = sorted(a for a in aliases if cls.MODEL_ALIASES.get(a) != a)
        model = sorted(a for a in aliases if a not in generic)
        ordered = generic + model

        structured_label = (
            "✅ Native (tool calling)"
            if cls.STRUCTURED_OUTPUT == "native"
            else "⚠️  Repair loop"
        )

        alias_display = "  ".join(
            f"[bold]{a}[/bold]" if i == 0 else a for i, a in enumerate(ordered)
        )

        table.add_row(
            cls.__name__.replace("Provider", ""),
            alias_display,
            cls.ENV_VAR,
            structured_label,
            cls.NOTE,
        )

    console.print(table)
    console.print()
    console.print(
        "Usage:   [bold green]revelyt scan <target> --provider [/bold green]"
        "[bold cyan]<alias>[/bold cyan]"
    )
    console.print(
        "Example: revelyt scan ./my-repo --provider [cyan]gemini-2.5-pro[/cyan]"
    )
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# ── TRIAGE COMMAND ───────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def triage(
    target_path: str = typer.Argument(..., help="Local directory path or GitHub URL"),
    top: int = typer.Option(
        20, "--top", "-n", help="Number of highest-priority files to display"
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output raw JSON instead of a table (useful for scripting)",
    ),
):
    """
    Dry-run Phase 1 & 2 only — show which files would be scanned and their
    dependency impact scores, without spending any LLM tokens.

    Files are ranked by dependant_count (blast radius). High-count files are
    scanned first and get severity elevation during a real scan.

    Examples:\n
      revelyt triage ./my-project\n
      revelyt triage https://github.com/org/repo --top 10\n
      revelyt triage ./my-project --json
    """
    from revelyt.core.session import SessionManager
    from revelyt.core.sandbox import sandbox_session
    from revelyt.ingestion.loader import ProjectLoader
    from revelyt.ingestion.detector import ProjectDetector
    from revelyt.core.mapper import CodeMapper

    console.print(
        "\n🔍 [bold blue]Revelyt Triage (dry-run — no LLM calls)[/bold blue]"
    )
    console.print(f"   ↳ Target: [dim]{target_path}[/dim]\n")

    with sandbox_session() as (sandbox_dir, _):
        loader = ProjectLoader(sandbox_dir)
        project_root = loader.load(target_path)

        detector = ProjectDetector(project_root)
        tech_stack = detector.detect()
        console.print(
            f"   ↳ Detected Stack: [green]{tech_stack['language'].upper()}[/green]"
            f" / [green]{tech_stack['framework'].upper()}[/green]"
        )

        session = SessionManager()
        mapper = CodeMapper(session, project_root)

        summary_files = list(loader.get_summary_manifest())
        analysis_files = list(loader.get_file_manifest())

        for f in summary_files:
            mapper.build_graph(f)
        mapper.update_dependant_counts()

        with session.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT ms.file_path, ms.dependant_count,
                       COUNT(dg.source_file) as imports_count
                FROM module_summaries ms
                LEFT JOIN dep_graph dg ON dg.source_file = ms.file_path
                GROUP BY ms.file_path
                ORDER BY ms.dependant_count DESC, imports_count DESC
                LIMIT ?
            """,
                (top,),
            ).fetchall()

        analysis_set = {str(f.relative_to(project_root)) for f in analysis_files}

        if json_output:
            output = [
                {
                    "file": row["file_path"],
                    "dependant_count": row["dependant_count"],
                    "imports_count": row["imports_count"],
                    "in_analysis": row["file_path"] in analysis_set,
                }
                for row in rows
            ]
            console.print(json.dumps(output, indent=2))
            return

        table = Table(
            title=f"Top {top} files by blast radius (dependant_count)",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold white",
            border_style="dim",
            expand=False,
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("File", style="cyan", no_wrap=False)
        table.add_column("Dependants ↑", style="bold yellow", justify="right", width=12)
        table.add_column("Imports", style="dim", justify="right", width=9)
        table.add_column("Elevation Risk", style="bold", width=16)
        table.add_column("Analysis Chain", style="dim", width=15)

        for i, row in enumerate(rows, 1):
            count = row["dependant_count"]
            risk = _risk_label(count)
            in_analysis = row["file_path"] in analysis_set
            chain_label = (
                "[green]✅ included[/green]"
                if in_analysis
                else "[dim]summary only[/dim]"
            )

            table.add_row(
                str(i),
                row["file_path"],
                str(count),
                str(row["imports_count"]),
                risk,
                chain_label,
            )

        console.print(table)
        console.print()

        total_analysis = len(analysis_files)
        total_summary = len(summary_files)
        total_tasks = total_analysis * 3 + total_summary

        console.print("   📊 [bold]Scan estimate:[/bold]")
        console.print(
            f"      Analysis files (quality/security/perf): [cyan]{total_analysis}[/cyan]"
        )
        console.print(
            f"      Summary files (all incl. tests/alembic): [cyan]{total_summary}[/cyan]"
        )
        console.print(
            f"      Total LLM tasks if scanned:              [yellow]{total_tasks}[/yellow]"
        )
        console.print()
        console.print(
            "   ℹ️  Files with [yellow]dependant_count ≥ 5[/yellow] will have HIGH findings "
            "elevated to [red]CRITICAL[/red] during a real scan."
        )
        console.print()


# ─────────────────────────────────────────────────────────────────────────────
# ── REPORT COMMAND ───────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def report(
    session_db: str = typer.Argument(
        ..., help="Path to an existing session SQLite database"
    ),
    output: str = typer.Option(
        "revelyt_report.pdf",
        "--output",
        "-o",
        help="Output path for the regenerated PDF report",
    ),
):
    """
    Regenerate a PDF report from an existing scan session database — no LLM calls.

    Example:\n
      revelyt report ./my_scan_session.db\n
      revelyt report ./my_scan_session.db --output final_audit.pdf
    """
    from revelyt.core.session import SessionManager
    from collections import defaultdict
    import markdown
    from xhtml2pdf import pisa

    db_path = Path(session_db).resolve()
    if not db_path.exists():
        console.print(f"❌ [bold red]Session database not found:[/bold red] {db_path}")
        raise typer.Exit(1)

    console.print("\n📄 [bold blue]Regenerating PDF Report[/bold blue]")
    console.print(f"   ↳ Session: [dim]{db_path}[/dim]")

    session = SessionManager(db_path=str(db_path))

    with session.get_connection() as conn:
        findings = conn.execute("""
            SELECT * FROM findings
            ORDER BY file_path ASC,
                CASE severity
                    WHEN 'CRITICAL' THEN 1 WHEN 'HIGH'   THEN 2
                    WHEN 'MEDIUM'   THEN 3 WHEN 'LOW'    THEN 4
                    ELSE 5
                END
        """).fetchall()

        critical = conn.execute(
            "SELECT COUNT(*) FROM findings WHERE severity='CRITICAL'"
        ).fetchone()[0]
        high = conn.execute(
            "SELECT COUNT(*) FROM findings WHERE severity='HIGH'"
        ).fetchone()[0]
        brd_results = conn.execute("SELECT * FROM brd_requirements").fetchall()

    if not findings and not brd_results:
        console.print(
            "   ↳ [yellow]No findings or BRD results in this session.[/yellow]"
        )
        raise typer.Exit(0)

    grouped = defaultdict(list)
    for row in findings:
        grouped[str(row["file_path"])].append(row)

    md = _build_report_markdown(grouped, critical, high, len(findings), brd_results)

    html = markdown.markdown(md, extensions=["tables", "fenced_code"])
    styled = f"""<html><head><style>
        body  {{ font-family: Helvetica, sans-serif; font-size: 12px; }}
        h1    {{ color: #2C3E50; }}
        h3    {{ color: #2980B9; border-bottom: 1px solid #BDC3C7; padding-bottom: 5px; }}
        h4    {{ margin-bottom: 2px; }}
        code  {{ background-color: #F4F6F7; padding: 2px 4px; border-radius: 4px;
                 font-family: Courier, monospace; }}
        pre   {{ background-color: #F4F6F7; padding: 10px; border-left: 4px solid #3498DB;
                 white-space: pre-wrap; word-wrap: break-word; }}
    </style></head><body>{html}</body></html>"""

    out_path = Path(output).resolve()
    with open(out_path, "w+b") as f:
        pisa.CreatePDF(styled, dest=f)

    console.print(f"   ↳ ✅ [bold green]Report saved to: {out_path.name}[/bold green]")


def _risk_label(count: int) -> str:
    if count >= 10:
        return "[red]CRITICAL risk[/red]"
    if count >= 5:
        return "[yellow]HIGH risk[/yellow]"
    if count >= 2:
        return "[blue]MEDIUM risk[/blue]"
    return "[dim]LOW[/dim]"


def _status_color(status: str) -> str:
    if status == "MET":
        return "green"
    if status == "NOT_MET":
        return "red"
    return "orange"


def _build_report_markdown(
    grouped: dict, critical: int, high: int, total: int, brd_results: list
) -> str:
    md = f"""# 🛡️ Revelyt Audit Report

## 📊 Scan Metrics
- **CRITICAL Issues:** {critical}
- **HIGH Issues:** {high}
- **Total Issues Found:** {total}

---

## 📂 Detailed Findings by File
"""
    for path, issues in grouped.items():
        md += f"\n### 📄 `{path}`\n"
        for issue in issues:
            color = "red" if issue["severity"] in ("CRITICAL", "HIGH") else "black"

            # Safely extract new keys (handles older DB records gracefully)
            line_no = issue["line_number"] if "line_number" in issue.keys() else None
            class_method = (
                issue["class_or_method"] if "class_or_method" in issue.keys() else None
            )

            # Build the location string
            location_str = ""
            if class_method and class_method != "Global":
                location_str += f" in `{class_method}`"
            if line_no:
                location_str += f" at line **{line_no}**"

            md += (
                f"#### <span style='color:{color}'>[{issue['severity']}]</span> "
                f"{issue['title']}{location_str}\n"
                f"- **Category:** {issue['category']} | **Impact:** {issue['impact_scope']}\n\n"
                f"**Description:**\n{issue['description']}\n\n"
                f"**Suggested Fix:**\n```\n{issue['suggestion']}\n```\n\n"
            )
        md += "---\n"

    if brd_results:
        md += "\n## ⚖️ Business Requirements (BRD) Verification\n\n"
        for req in brd_results:
            color = _status_color(req["status"])
            md += (
                f"### Req #{req['id']}: "
                f"<span style='color:{color}'>[{req['status']}]</span>\n"
                f"**Requirement:** {req['requirement']}\n\n"
                f"**Evidence:**\n```text\n{req['evidence']}\n```\n\n---\n"
            )

    return md


# ─────────────────────────────────────────────────────────────────────────────
# pr (Enterprise Plugin)
# ─────────────────────────────────────────────────────────────────────────────

pr_app = typer.Typer(
    help=(
        "Revelyt PR Reviewer — Enterprise Pull Request Analysis.\n\n"
        "Combines deterministic rule engines (instant, free) with optional AI analysis "
        "on high-risk chunks.\n\n"
        "[bold]Sub-commands:[/bold]\n"
        "  check   Review a GitHub Pull Request\n"
        "  diff    Review a local git diff from stdin or a file"
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(pr_app, name="pr")


@pr_app.command("check")
def pr_check(
    repo: str = typer.Argument(..., help="GitHub repository (e.g., owner/repo)"),
    pr_number: int = typer.Argument(..., help="Pull Request number"),
    mode: str = typer.Option("deep", help="Review depth: fast, balanced, or deep"),
    no_ai: bool = typer.Option(
        False,
        "--no-ai",
        help="Disable AI review (deterministic rules only — zero cost)",
    ),
    provider: str = typer.Option(
        "gemini", "--provider", "-p", help="LLM provider alias"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview results without posting a GitHub comment"
    ),
    token: Optional[str] = typer.Option(
        None, envvar=["GITHUB_TOKEN", "GH_TOKEN"], help="GitHub personal access token"
    ),
):
    """
    Full PR review on a GitHub Pull Request (deterministic rules + optional AI).

    Posts inline comments, a summary comment, and a GitHub Check Run.

    Examples:\n
      revelyt pr check owner/repo 42\n
      revelyt pr check owner/repo 42 --mode deep\n
      revelyt pr check owner/repo 42 --no-ai\n
      revelyt pr check owner/repo 42 --dry-run --mode balanced
    """
    if not token and not dry_run:
        console.print(
            "❌ [bold red]GitHub token is required (set GITHUB_TOKEN environment variable).[/bold red]"
        )
        raise typer.Exit(1)

    from revelyt.plugins.pr_reviewer.orchestrator import PROrchestrator, ReviewConfig
    from revelyt.plugins.pr_reviewer.filters.limits import ReviewMode

    try:
        review_mode = ReviewMode(mode.lower())
    except ValueError:
        console.print(
            f"❌ [bold red]Invalid mode: '{mode}'. Valid modes are: fast, balanced, deep.[/bold red]"
        )
        raise typer.Exit(1)

    config = ReviewConfig(
        token=token or "",
        repo_name=repo,
        pr_number=pr_number,
        mode=review_mode,
        enable_ai=not no_ai,
        ai_provider=provider,
        dry_run=dry_run,
        post_comments=True,
    )

    orchestrator = PROrchestrator(config)
    orchestrator.run()


@pr_app.command("diff")
def pr_diff(
    diff_file: Optional[typer.FileText] = typer.Argument(
        None, help="File containing the diff (defaults to stdin)"
    ),
    mode: str = typer.Option("balanced", help="Review depth: fast, balanced, or deep"),
    no_ai: bool = typer.Option(
        False, "--no-ai", help="Disable AI review (rules only — zero cost)"
    ),
    provider: str = typer.Option(
        "gemini", "--provider", "-p", help="LLM provider alias"
    ),
):
    """
    Review a local git diff from stdin or a file. Does not post to GitHub.

    Examples:\n
      git diff HEAD~1 | revelyt pr diff\n
      git diff HEAD~1 | revelyt pr diff --mode deep\n
      git diff HEAD~1 | revelyt pr diff --no-ai\n
      revelyt pr diff ./changes.patch --mode balanced
    """
    import sys
    from revelyt.plugins.pr_reviewer.orchestrator import PROrchestrator, ReviewConfig
    from revelyt.plugins.pr_reviewer.filters.limits import ReviewMode

    if diff_file is None:
        if sys.stdin.isatty():
            console.print(
                "❌ [bold red]No diff provided. Pipe a diff to stdin or specify a diff file.[/bold red]"
            )
            raise typer.Exit(1)
        raw_diff = sys.stdin.read()
    else:
        raw_diff = diff_file.read()

    try:
        review_mode = ReviewMode(mode.lower())
    except ValueError:
        console.print(
            f"❌ [bold red]Invalid mode: '{mode}'. Valid modes are: fast, balanced, deep.[/bold red]"
        )
        raise typer.Exit(1)

    config = ReviewConfig(
        mode=review_mode,
        enable_ai=not no_ai,
        ai_provider=provider,
    )

    orchestrator = PROrchestrator(config)
    orchestrator.run_local_diff(raw_diff)


# ─────────────────────────────────────────────────────────────────────────────
# ── ENTRYPOINT ───────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
