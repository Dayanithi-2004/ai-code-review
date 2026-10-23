import logging
import time
import re
from revelyt.core.schemas import RequirementList
from langchain_core.prompts import ChatPromptTemplate
import asyncio
from pathlib import Path
from typing import Optional, Set, Dict, List, Tuple
from collections import defaultdict
import markdown
from xhtml2pdf import pisa
from rich.console import Console
from revelyt.providers.factory import ProviderFactory
from revelyt.core.session import SessionManager
from revelyt.providers.base import BaseProvider
from revelyt.core.diff import DiffEngine
from revelyt.core.sandbox import sandbox_session
from revelyt.core.skills_engine import get_skills_engine

CONFIG_FILES_TO_SKIP = {".env", "requirements.txt", "pyproject.toml"}

from revelyt.core.schemas import FindingList
from revelyt.core.code_optimizer import should_skip_file
from revelyt.ingestion.loader import ProjectLoader
from revelyt.ingestion.detector import ProjectDetector
from revelyt.core.mapper import CodeMapper
from revelyt.agents import (
    ProductManagerAgent,
    CombinedWave1Agent,
    CombinedWave2Agent,
)
from revelyt.agents.base_agent import (
    is_analyzable,
    set_scan_start_time,
    _is_oss_model,
)
from revelyt.core.code_router import (
    classify_file,
    classify_files,
    get_analysis_config,
)

logger = logging.getLogger(__name__)
console = Console()


# ── File role detection ────────────────────────────────────────────────────
# Used by _deduplicate_cross_agent_findings() to pick the most authoritative
# file when the same finding appears across multiple files.
# Priority: lower number = more preferred. Covers FastAPI, Flask, Django,
# Express, NestJS, and generic Python/JS/TS naming conventions.

_ROLE_PATTERNS: List[Tuple[int, str, List[str]]] = [
    (
        1,
        "ROUTE",
        [
            r"(^|/)(main|app|server|application)\.(py|ts|js)$",
            r"(^|/)(routes?|router|routers|views?|api|endpoints?|handlers?|controllers?)(\w*)\.(py|ts|js)$",
            r"\w+\.controller\.(ts|js)$",
            r"/(routes?|controllers?|views?|handlers?|endpoints?|api)(/[\w/]+)*\.\w+$",
        ],
    ),
    (
        2,
        "SERVICE",
        [
            r"(^|/)(crud|service|services|repository|repositories|repo|repos|"
            r"manager|managers|use_case|use_cases|interactor)(\w*)\.(py|ts|js)$",
            r"\w+\.service\.(ts|js)$",
            r"/(services?|repositories?|repos?|use_cases?)(/[\w/]+)*\.\w+$",
        ],
    ),
    (
        3,
        "DB_CFG",
        [
            r"(^|/)(database|db|config|configuration|settings|engine|connection|"
            r"datasource|orm)(\w*)\.(py|ts|js)$",
            r"\w+\.module\.(ts|js)$",
            r"(knexfile|prisma|drizzle\.config)\.(ts|js)$",
        ],
    ),
    (
        4,
        "MODEL",
        [
            r"(^|/)(model|models|entity|entities|domain|aggregate)(\w*)\.(py|ts|js)$",
            r"\w+\.(entity|model)\.(ts|js)$",
            r"/(models?|entities?|domain)(/[\w/]+)*\.\w+$",
        ],
    ),
    (
        5,
        "SCHEMA",
        [
            r"(^|/)(schema|schemas|serializer|serializers|dto|dtos|"
            r"type|types|interface|interfaces|form|forms|validator|validators)(\w*)\.(py|ts|js)$",
            r"\w+\.(dto|interface|schema)\.(ts|js)$",
            r"/(schemas?|serializers?|dtos?|types?|interfaces?)(/[\w/]+)*\.\w+$",
        ],
    ),
    (
        10,
        "TEST",
        [
            r"(test_\w+|_test\b|\bspec\b|\.test\.|\.spec\.|conftest|fixtures?|factor)",
        ],
    ),
]


def _detect_file_role(file_path: str) -> Tuple[int, str]:
    """
    Detect the architectural role of a file from its path.

    Returns (priority, label). Lower priority = more preferred location
    when the same finding appears on multiple files.

    Falls back to (6, 'OTHER') for unrecognised files.
    """
    full = file_path.lower()
    for priority, label, patterns in _ROLE_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, full):
                return priority, label
    return 6, "OTHER"


# ── Frontend file detection ────────────────────────────────────────────────
# Always-on filter: frontend files (.jsx, .tsx, frontend dirs) are skipped
# by default since they almost never contain security-relevant logic.
# Backend .js/.ts files in Express/Node/Fastify are NOT affected.

_FRONTEND_EXTENSIONS = {".jsx", ".tsx"}
_FRONTEND_DIR_PATTERNS = re.compile(
    r"(^|/)("
    r"frontend|client|web-app|webapp|ui"
    r")/",
    re.IGNORECASE,
)
_FRONTEND_SUBDIR_PATTERNS = re.compile(
    r"(^|/)("
    r"components|pages|views|screens|layouts|hooks|contexts|redux|store"
    r")/",
    re.IGNORECASE,
)


def _is_frontend_file(file_path: Path, project_root: Path) -> bool:
    """
    Return True if a file is a frontend component/page that should be skipped.

    Rules:
      1. .jsx / .tsx → always frontend (these extensions don't exist in backend)
      2. .js / .ts under frontend/, client/, ui/ → frontend
      3. .js / .ts under src/components/, src/pages/, src/views/ → frontend
    """
    if file_path.suffix in _FRONTEND_EXTENSIONS:
        return True

    # For .js/.ts, check directory patterns
    if file_path.suffix in (".js", ".ts", ".mjs", ".cjs"):
        try:
            rel = str(file_path.relative_to(project_root))
        except ValueError:
            rel = str(file_path)
        if _FRONTEND_DIR_PATTERNS.search(rel):
            return True

    return False


class RevelytOrchestrator:
    """
    The main engine for Revelyt. Coordinates the 6-Phase Pipeline.
    """

    # Valid agent aliases for --agents flag
    VALID_AGENTS = {"architect", "developer", "qa", "devexp", "doc"}

    def __init__(
        self,
        target_path: str,
        brd_path: str = None,
        provider_name: str = "gemini",
        oss_provider_name: Optional[str] = None,
        diff_base: Optional[str] = None,
        diff_head: Optional[str] = None,
        active_agents: Optional[str] = None,
        report_format: str = "pdf",
        concurrency: int = 4,
        skip_wave2_clean: bool = True,
        llama_cpp_url: Optional[str] = None,
    ):
        self.target_path = target_path
        self.brd_path = Path(brd_path).resolve() if brd_path else None
        self.session = SessionManager()

        # Diff mode: when set, only changed + impacted files are analysed
        self.diff_base = diff_base
        self.diff_head = diff_head or "HEAD"
        self._diff_files: Optional[Set[str]] = None  # populated after Phase 2

        # Concurrency: max parallel LLM calls
        # Cloud providers can handle much higher concurrency (network-bound)
        # Local providers (Ollama) are GPU-bound, so keep it lower
        self.concurrency = concurrency
        self._auto_concurrency = concurrency == 4  # track if user overrode it

        # Skip Wave 2 on files with no Wave 1 findings (saves time on clean files)
        self.skip_wave2_clean = skip_wave2_clean

        # Agent selector: comma-separated list or None (all agents)
        if active_agents:
            self.active_agents = {a.strip().lower() for a in active_agents.split(",")}
            invalid = self.active_agents - self.VALID_AGENTS
            if invalid:
                raise ValueError(
                    f"Unknown agent(s): {invalid}. "
                    f"Valid: {', '.join(sorted(self.VALID_AGENTS))}"
                )
        else:
            self.active_agents = None  # None = all agents

        # Report output format
        self.report_format = report_format

        self.provider = ProviderFactory.create(provider_name, base_url=llama_cpp_url)

        # Secondary OSS provider for Wave 2 and PM summarisation.
        # If not specified, falls back to self.provider — fully backward-compatible.
        # When set (e.g. "qwen2.5-coder:7b"), Wave 2 and PM run on the OSS model
        # (free, runs locally) while Wave 1 and BRD run on the frontier model.
        if oss_provider_name:
            self.oss_provider = ProviderFactory.create(oss_provider_name)
        else:
            self.oss_provider = self.provider  # single-model mode (default)

        # llama.cpp URL for Metal-accelerated inference
        self.llama_cpp_url = llama_cpp_url

        logger.info(
            "Orchestrator init: target=%s, provider=%s, model=%s, "
            "oss_provider=%s, agents=%s, format=%s",
            target_path,
            type(self.provider).__name__,
            self.provider.model_name,
            self.oss_provider.model_name if oss_provider_name else "same",
            self.active_agents or "all",
            report_format,
        )

        console.print(
            f"   ↳ Provider (Wave 1 + BRD): [cyan]{type(self.provider).__name__}[/cyan] "
            f"([dim]{self.provider.model_name}[/dim])"
        )
        if oss_provider_name:
            console.print(
                f"   ↳ OSS Provider (Wave 2 + PM): [cyan]{type(self.oss_provider).__name__}[/cyan] "
                f"([dim]{self.oss_provider.model_name}[/dim])"
            )
        if self.active_agents:
            console.print(
                f"   ↳ Active agents: [green]{', '.join(sorted(self.active_agents))}[/green]"
            )
        if self.diff_base:
            console.print(
                f"   ↳ Diff mode: [yellow]{self.diff_base}...{self.diff_head}[/yellow]"
            )

        # Auto-boost concurrency for cloud-backed providers
        if self._auto_concurrency:
            provider_cls = type(self.provider).__name__
            if provider_cls in (
                "GoogleProvider",
                "OpenAIProvider",
                "AnthropicProvider",
                "OpenRouterProvider",
                "HuggingFaceProvider",
            ):
                # Direct cloud APIs: generous rate limits
                self.concurrency = 10
                console.print(
                    f"   ↳ ⚡ Cloud API detected — concurrency boosted to "
                    f"[cyan]{self.concurrency}[/cyan]"
                )
            elif (
                provider_cls == "OllamaProvider"
                and ":cloud" in self.provider.model_name
            ):
                # Ollama cloud proxies (e.g. kimi-k2.5:cloud): strict rate limits
                self.concurrency = 3
                console.print(
                    f"   ↳ ⚡ Ollama cloud proxy detected — concurrency set to "
                    f"[cyan]{self.concurrency}[/cyan] (rate limit safe)"
                )
            elif provider_cls == "OllamaProvider":
                # Local Ollama: GPU handles NUM_PARALLEL=3, queue 2x to avoid idle time
                self.concurrency = 6
                console.print(
                    f"   ↳ ⚡ Local Ollama detected — concurrency set to "
                    f"[cyan]{self.concurrency}[/cyan] (GPU queue depth)"
                )
            elif provider_cls in ("LlamaCppProvider", "LlamaCppAuditorProvider"):
                # llama.cpp parses --ctx-size 32K across all parallel slots.
                # 4 concurrency = 8K context per internal slot
                self.concurrency = 5
                console.print(
                    f"   ↳ ⚡ llama.cpp (Metal) detected — concurrency set to "
                    f"[cyan]{self.concurrency}[/cyan] (matches --parallel 5 with 4-bit KV)"
                )

    async def run_scan(self):
        scan_start = time.monotonic()
        set_scan_start_time(scan_start)
        logger.info(
            "=== Scan started: %s (session=%s) ===",
            self.target_path,
            self.session.session_id,
        )
        console.print(
            f"🚀 [bold cyan]Initializing Revelyt Scan for:[/bold cyan] {self.target_path}"
        )
        console.print(f"   ↳ Session ID: [dim]{self.session.session_id}[/dim]")

        # Auto-manage llama.cpp server lifecycle if using llama-cpp provider
        self.provider_cls_name = type(self.provider).__name__
        self._is_llamacpp = self.provider_cls_name in ("LlamaCppProvider", "LlamaCppAuditorProvider")
        self._auto_started_llamacpp = False
        self.llamacpp_manager = None

        if self._is_llamacpp and not self.llama_cpp_url:
            from revelyt.core.llamacpp_manager import LlamaCppManager
            
            if self.provider_cls_name == "LlamaCppAuditorProvider":
                self.llamacpp_manager = LlamaCppManager(
                    parallel=self.concurrency,
                    model_repo="Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
                    model_file="qwen2.5-coder-14b-instruct-q4_k_m.gguf"
                )
            else:
                self.llamacpp_manager = LlamaCppManager(parallel=self.concurrency)
                
            self.llama_cpp_url = self.llamacpp_manager.ensure_ready()
            self._auto_started_llamacpp = True
            
            from revelyt.providers.factory import ProviderFactory
            if self.provider_cls_name == "LlamaCppProvider":
                alias = "llama-cpp"
            else:
                alias = "llama-cpp-auditor"
            provider_instance = ProviderFactory.create(alias, base_url=self.llama_cpp_url)
            self.provider = provider_instance
            self.oss_provider = provider_instance

        try:
            with sandbox_session() as (sandbox_dir, _sandbox_id):
                # PHASE 1: INGESTION & TRIAGE
                project_root = self._phase_ingestion(sandbox_dir)
                if not project_root:
                    return

                # PHASE 2: DEPENDENCY MAPPING (always runs on ALL files)
                mapper = self._phase_mapping(project_root)

                # DIFF RESOLUTION: compute changed + impacted file set
                if self.diff_base:
                    self._resolve_diff(project_root, mapper)

                # PHASE 3: PARALLEL REVIEW CHAINS
                await self._phase_review(project_root)

                # PHASE 4: BRD VERIFICATION (only if a BRD file was provided)
                if self.brd_path:
                    self._pm_agent.verify_brd(self.brd_path)

                # PHASE 5 & 6: SEVERITY ELEVATION & REPORTING
                self._phase_elevation_and_reporting(project_root)
        finally:
            # Shutdown llama.cpp server if we auto-started it
            if self._auto_started_llamacpp and self.llamacpp_manager:
                self.llamacpp_manager.shutdown()

        # Print total scan time
        elapsed = time.monotonic() - scan_start
        hours, rem = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(rem, 60)
        parts = []
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        parts.append(f"{seconds}s")
        console.print(
            f"\n⏱️  [bold]Total scan time:[/bold] [cyan]{' '.join(parts)}[/cyan]"
        )

    def _is_cloud_provider(self) -> bool:
        """Check if the primary provider is cloud-backed (not local GPU)."""
        provider_cls = type(self.provider).__name__
        # These providers always hit remote APIs
        if provider_cls in (
            "GoogleProvider",
            "OpenAIProvider",
            "AnthropicProvider",
            "OpenRouterProvider",
            "HuggingFaceProvider",
        ):
            return True
        # Ollama cloud proxies (e.g. kimi-k2.5:cloud)
        if provider_cls == "OllamaProvider" and ":cloud" in self.provider.model_name:
            return True
        return False

    def _get_review_batch_size(self) -> int:
        """Model-tier-aware batch size for directory-level review batching."""
        name = self.provider.model_name.lower().split(":")[0]
        if _is_oss_model(name):
            return (
                3  # Local 14B model: keep batches small for fast inference and reliable JSON
            )
        if "pro" in name:
            return 8  # 1M context, strong file tracking
        if "lite" in name:
            return 3  # 1M context, weaker file tracking
        return 4  # Flash / cloud default

    def _group_files_by_directory(
        self, files: list, agents: list, batch_size: int
    ) -> list:
        """
        Group files by parent directory, pre-filter skippable files (Gap 5),
        then chunk each directory group into batches of batch_size.
        """
        from collections import defaultdict

        dir_groups: dict[Path, list[Path]] = defaultdict(list)
        for fp in files:
            # Only include files that at least one agent can handle (Gap 5)
            if any(not self._should_skip_agent_on_file(a, fp) for a in agents):
                dir_groups[fp.parent].append(fp)

        batches = []
        for _dir_path, dir_files in dir_groups.items():
            for i in range(0, len(dir_files), batch_size):
                batches.append(dir_files[i : i + batch_size])

        return batches

    async def _run_wave_batched(
        self,
        agents: list,
        files: list,
        semaphore,
        *,
        skills_context: str = "",
        pruner=None,
        static_covered_lines=None,
        cache=None,
    ) -> dict:
        """
        Run Wave 1 using directory-level batched review.

        Groups files by directory, sends batches of 3-8 files per LLM call,
        and collects per-file results. Falls back to individual review if
        a batch fails.

        For local models: processes batches sequentially to avoid
        overwhelming the llama-server (which has limited --parallel slots).
        """
        console.print("\n   ── [bold]Wave 1: Reviewer (batched) [/bold] ──")

        batch_size = self._get_review_batch_size()
        batches = self._group_files_by_directory(files, agents, batch_size)

        total_files = sum(len(b) for b in batches)
        console.print(
            f"   ↳ Firing {len(batches)} batch tasks "
            f"({total_files} files, ×{batch_size} per batch)..."
        )

        # Build covered_lines map for batch access
        covered_map = {}
        if static_covered_lines:
            covered_map = static_covered_lines

        all_results: dict = {}

        async def _run_batch_task(agent, batch_files):
            async with semaphore:
                result = await asyncio.to_thread(
                    agent.analyze_batch,
                    batch_files,
                    skills_context=skills_context,
                    pruner=pruner,
                    covered_lines_by_file=covered_map,
                    cache=cache,
                )
                return result

        # Build all tasks
        tasks = []
        for agent in agents:
            for batch in batches:
                tasks.append((agent, batch))

        # For local models: process sequentially so we don't flood the server
        if self._is_llamacpp:
            for agent, batch in tasks:
                try:
                    batch_results = await _run_batch_task(agent, batch)
                    if batch_results:
                        all_results.update(batch_results)
                except Exception as exc:
                    logger.error("Batch task failed: %s", exc)
        else:
            # Cloud providers: fire all concurrently
            coros = [_run_batch_task(a, b) for a, b in tasks]
            for coro in asyncio.as_completed(coros):
                try:
                    batch_results = await coro
                    if batch_results:
                        all_results.update(batch_results)
                except Exception as exc:
                    logger.error("Batch task failed: %s", exc)

        return all_results

    def _phase_ingestion(self, sandbox_dir: Path) -> Path:
        logger.info("Phase 1: Ingestion started (sandbox=%s)", sandbox_dir)
        console.print("\n📦 [bold blue]Phase 1: Ingestion & Triage[/bold blue]")
        loader = ProjectLoader(sandbox_dir)
        project_root = loader.load(self.target_path)

        detector = ProjectDetector(project_root)
        tech_stack = detector.detect()
        logger.info(
            "Stack detected: %s / %s", tech_stack["language"], tech_stack["framework"]
        )
        console.print(
            f"   ↳ Detected Stack: [green]{tech_stack['language'].upper()}[/green]"
            f" / [green]{tech_stack['framework'].upper()}[/green]"
        )
        return project_root

    def _phase_mapping(self, project_root: Path) -> CodeMapper:
        logger.info("Phase 2: Dependency mapping started")
        console.print("\n🗺️  [bold blue]Phase 2: Dependency Mapping[/bold blue]")
        mapper = CodeMapper(self.session, project_root)

        # Full manifest for mapping — we want test/migration imports in the dep graph too
        loader = ProjectLoader(project_root.parent)
        for file_path in loader.get_summary_manifest():
            mapper.build_graph(file_path)

        mapper.update_dependant_counts()
        logger.info("Phase 2: Dependency graph and dependant counts updated")
        console.print("   ↳ ✅ Dependency graph and topological metrics updated.")
        return mapper

    def _resolve_diff(self, project_root: Path, mapper: CodeMapper):
        """
        Computes the set of files that need analysis in diff mode.
        Uses DiffEngine + mapper's impact analysis (reverse dependencies).
        """
        console.print("\n🔀 [bold blue]Diff Resolution[/bold blue]")

        try:
            changed = DiffEngine.get_changed_files(
                repo_path=project_root,
                base=self.diff_base,
                head=self.diff_head,
            )
        except ValueError as e:
            console.print(f"   ↳ [yellow]Diff failed: {e}[/yellow]")
            console.print("   ↳ [yellow]Falling back to full scan.[/yellow]")
            return

        if not changed:
            console.print("   ↳ [green]No files changed — nothing to analyse.[/green]")
            self._diff_files = set()  # empty set = skip analysis
            return

        console.print(f"   ↳ 📝 {len(changed)} file(s) changed in diff")
        for f in sorted(changed):
            console.print(f"      [cyan]M[/cyan] {f}")

        # Expand: find files that import any changed file
        impacted = mapper.get_impacted_files(changed)
        impact_only = impacted - changed

        if impact_only:
            console.print(
                f"   ↳ 💥 {len(impact_only)} additional file(s) impacted (reverse deps)"
            )
            for f in sorted(impact_only):
                console.print(f"      [yellow]I[/yellow] {f}")

        self._diff_files = impacted
        console.print(
            f"   ↳ ✅ Diff scope: [bold]{len(self._diff_files)}[/bold] files "
            f"({len(changed)} changed + {len(impact_only)} impacted)"
        )

    def _is_agent_active(self, alias: str) -> bool:
        """Check if an agent alias is enabled by the --agents flag."""
        return self.active_agents is None or alias in self.active_agents

    def _build_agents(self):
        """
        Instantiate review agents.

        Wave 1 (architectural pass) always uses self.provider — the strongest
        model configured, responsible for security and architectural findings.

        Wave 2 (DX + docs pass) uses self.oss_provider — a cheaper/faster
        model sufficient for pattern-recognition tasks like missing docstrings,
        type hints, and API ergonomics. Defaults to self.provider if no OSS
        provider was configured (single-model mode).
        """
        wave1, wave2 = [], []

        # Wave 1: Architect + Developer — frontier model, full code or skeleton
        if self._is_agent_active("architect") or self._is_agent_active("developer"):
            wave1.append(CombinedWave1Agent(self.session, self.provider))

        # Wave 2: QA + DevExp + DocReviewer — OSS model sufficient for DX/docs
        if (
            self._is_agent_active("qa")
            or self._is_agent_active("devexp")
            or self._is_agent_active("doc")
        ):
            wave2.append(CombinedWave2Agent(self.session, self.oss_provider))

        return wave1, wave2

    def _apply_diff_filter(self, analysis_files, project_root):
        """Narrow analysis_files to only those in the diff scope."""
        if self._diff_files is None:
            return analysis_files
        if not self._diff_files:
            console.print(
                "   ↳ [green]No files in diff scope — skipping analysis agents.[/green]"
            )
            return []
        full_count = len(analysis_files)
        filtered = [
            f
            for f in analysis_files
            if str(f.relative_to(project_root)) in self._diff_files
        ]
        console.print(
            f"   ↳ \U0001f500 Diff mode: analysing [bold]{len(filtered)}[/bold] "
            f"files instead of {full_count}"
        )
        return filtered

    async def _phase_review(self, project_root: Path):
        console.print(
            "\n\U0001f575\ufe0f  [bold blue]Phase 3: AI Agent Review & Summarization[/bold blue]"
        )

        loader = ProjectLoader(project_root.parent)
        analysis_files = list(loader.get_file_manifest())
        summary_files = list(loader.get_summary_manifest())

        # Discard files not worth LLM token cost (.lock, binary, empty files, trivial __init__.py)
        analysis_files = [f for f in analysis_files if not should_skip_file(f)]
        summary_files = [
            f
            for f in summary_files
            if not should_skip_file(f) and f.name not in CONFIG_FILES_TO_SKIP
        ]

        if not summary_files:
            console.print(
                "   ↳ [yellow]No files found. Did Phase 1 complete correctly?[/yellow]"
            )
            return

        analysis_files = self._apply_diff_filter(analysis_files, project_root)

        # Drop files no agent can analyze (unknown language: .html, .css, .md, .json, etc.)
        # so they don't waste semaphore slots or inflate the "Firing N tasks" count.
        pre_lang_count = len(analysis_files)
        analysis_files = [f for f in analysis_files if is_analyzable(f)]
        skipped_lang = pre_lang_count - len(analysis_files)
        if skipped_lang:
            console.print(
                f"   ↳ Skipped [yellow]{skipped_lang}[/yellow] non-analyzable files "
                f"([dim].html, .css, .md, etc.[/dim])"
            )

        # Drop frontend files — backend-only by default.
        # Frontend (.jsx, .tsx, frontend/ dirs) almost never has security-relevant logic.
        pre_fe_count = len(analysis_files)
        analysis_files = [f for f in analysis_files if not _is_frontend_file(f, project_root)]
        summary_files = [f for f in summary_files if not _is_frontend_file(f, project_root)]
        skipped_fe = pre_fe_count - len(analysis_files)
        if skipped_fe:
            console.print(
                f"   ↳ ⚡ Skipped [yellow]{skipped_fe}[/yellow] frontend files "
                f"([dim].jsx, .tsx, frontend/ dirs[/dim])"
            )

        # ── Smart File Routing ──
        file_classification = classify_files(analysis_files)
        skipped_count = len(file_classification["SKIP"])
        heavy_count = len(file_classification["HEAVY"])
        light_count = len(file_classification["LIGHT"])
        normal_count = len(file_classification["NORMAL"])

        if skipped_count:
            console.print(
                f"   ↳ ⚡ [green]Skipping {skipped_count} generated/vendor files[/green]"
            )
        if heavy_count or light_count:
            console.print(
                f"   ↳ ⚡ File routing: [red]{heavy_count} HEAVY[/red] | "
                f"[cyan]{normal_count} NORMAL[/cyan] | "
                f"[green]{light_count} LIGHT[/green] | "
                f"[dim]{skipped_count} SKIP[/dim]"
            )

        # Remove SKIP files from analysis
        analysis_files = [fp for fp in analysis_files if classify_file(fp)[0] != "SKIP"]

        console.print(
            f"   ↳ {len(analysis_files)} files for analysis  |  "
            f"{len(summary_files)} files for summarisation"
        )

        # ── Triage & Model Routing ──
        if self.diff_base and self.diff_head:
            from revelyt.core.triage import TriageAgent
            import subprocess

            diff_patch = ""
            try:
                res = subprocess.run(
                    ["git", "diff", f"{self.diff_base}...{self.diff_head}"],
                    cwd=str(project_root),
                    capture_output=True,
                    text=True,
                )
                diff_patch = res.stdout[:150000]  # type: ignore[index]
            except Exception:
                pass

            if diff_patch:
                console.print("\n⚕️  [bold blue]Triage & Model Routing[/bold blue]")
                from revelyt.providers.factory import ProviderFactory

                triage_provider = self.provider
                if type(self.provider).__name__ == "GoogleProvider":
                    triage_provider = ProviderFactory.create(
                        "gemini-3-flash-preview", api_key=self.provider.api_key
                    )

                triage_agent = TriageAgent(triage_provider)
                decision = triage_agent.analyze_diff(diff_patch)

                console.print(
                    f"   ↳ Complexity: [cyan]{decision.complexity_score}/10[/cyan]  |  "
                    f"Trivial: [yellow]{decision.is_trivial}[/yellow]  |  "
                    f"Security: [red]{decision.needs_security_review}[/red]"
                )

                if decision.is_trivial:
                    console.print(
                        "   ↳ [green]Trivial PR detected (docs/formatting only). Skipping AI Review Phase.[/green]"
                    )
                    return

                if (
                    decision.complexity_score <= 4
                    and not decision.needs_security_review
                ):
                    console.print(
                        "   ↳ [green]Low complexity PR. Routing agents to fast/cheap model to save tokens.[/green]"
                    )
                    self.provider = triage_provider

        wave1_agents, wave2_agents = self._build_agents()
        
        # If llama-cpp, provider and oss_provider are technically the same Provider instance,
        # but the LlamacppManager underneath handles the local inference.
        pm_agent = ProductManagerAgent(self.session, self.oss_provider, project_root)
        self._pm_agent = pm_agent

        semaphore = asyncio.Semaphore(self.concurrency)

        # ── Initialize Diff Pruner if in diff mode ──
        pruner = None
        if self.diff_base and self.diff_head:
            from revelyt.core.diff_pruner import DiffPruner

            pruner = DiffPruner(
                repo_path=project_root, base=self.diff_base, head=self.diff_head
            )

        # ── Initialize Semantic Cache ──
        from revelyt.core.review_cache import ReviewCache

        review_cache = ReviewCache(project_root)

        # ── Initialize Function-Level Cache ──
        from revelyt.core.function_cache import FunctionCache

        func_cache = FunctionCache(project_root)
        console.print("   ↳ ⚡ Function-level caching enabled")

        # ── Initialize Static Gate ──
        from revelyt.core.static_gate import StaticGate

        static_gate = StaticGate(project_root)
        static_covered_lines = {}
        static_findings_total = 0

        console.print("   ↳ Running Static Gate (linters/security)...")
        for fp in analysis_files:
            s_res = static_gate.run_on_file(fp)
            if s_res.findings:
                with self.session.get_connection() as conn:
                    for finding in s_res.findings:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO findings
                            (session_id, file_path, agent_name, severity, category, title,
                             description, suggestion, confidence, impact_scope,
                             line_number, class_or_method)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                self.session.session_id,
                                str(fp),
                                "StaticGate",
                                finding.severity,
                                finding.category,
                                finding.title,
                                finding.description,
                                finding.suggestion,
                                finding.confidence,
                                "LOCAL",
                                finding.line_number,
                                finding.class_or_method,
                            ),
                        )
                static_findings_total += len(s_res.findings)
            static_covered_lines[str(fp)] = s_res.covered_lines

        if static_findings_total:
            console.print(
                f"   ↳ [green]Static Gate caught {static_findings_total} issues for free.[/green]"
            )

        # ── Initialize Skills Engine (wave-level injection) ──
        skills_engine = get_skills_engine()

        # ── Sequential Execution for Local Llama-cpp ──
        # To save RAM and use models efficiently, we run Phase 3A (PM summarization) 
        # with a small 3B model, and then swap to the 14B model for the heavy security review.
        if self._is_llamacpp:
            console.print("\n   ── [bold]Phase 3A: Sequential local summarization (3B model)[/bold] ──")
            from revelyt.core.llamacpp_manager import SUMMARY_MODEL_REPO, SUMMARY_MODEL_FILE, DEFAULT_MODEL_REPO, DEFAULT_MODEL_FILE
            
            # 1. Load 3B model
            self.llamacpp_manager.load_model(SUMMARY_MODEL_REPO, SUMMARY_MODEL_FILE)
            
            # 2. Run summarization
            console.print("   ↳ 🧠 Running PM summarisation with 3B model...")
            await asyncio.to_thread(pm_agent.summarize_batch, summary_files)
            
            # 3. Load 14B model
            console.print("\n   ── [bold]Phase 3B: Deep Security Review (14B model)[/bold] ──")
            self.llamacpp_manager.load_model(DEFAULT_MODEL_REPO, DEFAULT_MODEL_FILE)
            
            batch_size = self._get_review_batch_size()
            wave1_context = skills_engine.get_wave_compact_rules("wave1")

            if batch_size > 1:
                wave1_results = await self._run_wave_batched(
                    wave1_agents,
                    analysis_files,
                    semaphore,
                    skills_context=wave1_context,
                    pruner=pruner,
                    static_covered_lines=static_covered_lines,
                    cache=review_cache,
                )
            else:
                wave1_results = await self._run_wave(
                    wave1_agents,
                    analysis_files,
                    semaphore,
                    wave_num=1,
                    skills_context=wave1_context,
                    pruner=pruner,
                    static_covered_lines=static_covered_lines,
                    cache=review_cache,
                    func_cache=func_cache,
                )
        else:
            # ── Concurrent execution for Cloud/External APIs ──
            console.print("\n   ── [bold]Phase 3A: Single-Pass Security Audit & Mapping (Wave 1)[/bold] ──")
            batch_size = self._get_review_batch_size()
            
            wave1_context = skills_engine.get_wave_compact_rules("wave1")

            # Fire PM agent concurrently with Wave 1 agents
            pm_task = asyncio.to_thread(pm_agent.summarize_batch, summary_files)

            if batch_size > 1:
                wave1_task = self._run_wave_batched(
                    wave1_agents,
                    analysis_files,
                    semaphore,
                    skills_context=wave1_context,
                    pruner=pruner,
                    static_covered_lines=static_covered_lines,
                    cache=review_cache,
                )
            else:
                wave1_task = self._run_wave(
                    wave1_agents,
                    analysis_files,
                    semaphore,
                    wave_num=1,
                    skills_context=wave1_context,
                    pruner=pruner,
                    static_covered_lines=static_covered_lines,
                    cache=review_cache,
                    func_cache=func_cache,
                )

            # Wait for both review and summarization to finish concurrently
            wave1_results, _ = await asyncio.gather(wave1_task, pm_task)

        # ── Wave 2 is automatically dropped for LlamaCpp runs to save massive latency ──
        if wave2_agents and not self._is_llamacpp:
            await self._run_wave(
                wave2_agents,
                analysis_files,
                semaphore,
                wave_num=2,
                wave1_results=wave1_results,
                skills_context=skills_engine.get_wave_compact_rules("wave2"),
                pruner=pruner,
                static_covered_lines=static_covered_lines,
                cache=review_cache,
                func_cache=func_cache,
            )

    def _should_skip_agent_on_file(self, agent, file_path: Path) -> bool:
        """
        Evaluate secondary skip logic for deep mode optimizations.

        Wave 2 (DeepReviewer) is skipped on:
          1. Config files, __init__.py, test files, migrations (all depths)
          2. Definition-only files (MODEL/SCHEMA role with no function bodies)
             — these return at most docstring findings which Wave 1 already covers.
             Applied regardless of depth since it's a content-based decision.
        """
        filename = file_path.name
        filepath_str = str(file_path)

        if agent.role_name == "DeepReviewer" and (
            filename in CONFIG_FILES_TO_SKIP
            or filename == "__init__.py"
            or "test_" in filename
            or "migrations" in filepath_str
        ):
            return True

        # Skip Wave 2 on definition-only files (models, schemas, DTOs).
        # These files contain only class definitions with no logic — Wave 2
        # adds ~2900 input tokens per file for typically 0-1 duplicate findings.
        if agent.wave == 2 and agent.role_name == "DeepReviewer":
            role_priority, role_label = _detect_file_role(str(file_path))
            if role_label in ("MODEL", "SCHEMA"):
                try:
                    content = file_path.read_text(errors="ignore")
                    # Has no function or method definitions — pure class declarations
                    import re as _re

                    has_functions = bool(
                        _re.search(r"^    def ", content, _re.MULTILINE)
                    )
                    if not has_functions:
                        logger.debug(
                            "Skipping Wave 2 on definition-only file: %s (%s)",
                            file_path.name,
                            role_label,
                        )
                        return True
                except Exception:
                    pass  # Can't read file — don't skip, let Wave 2 try

        return False

    async def _run_single_task(
        self,
        agent,
        file_path,
        semaphore,
        wave1_results,
        skills_context="",
        pruner=None,
        covered_lines=None,
        cache=None,
        func_cache=None,
    ):
        """Execute a single agent analysis task."""
        async with semaphore:
            covered = covered_lines or set()
            if wave1_results is not None:
                ctx = wave1_results.get(str(file_path), FindingList(findings=[]))
                result = await asyncio.to_thread(
                    agent.analyze,
                    file_path,
                    ctx,
                    skills_context=skills_context,
                    pruner=pruner,
                    covered_lines=covered,
                    cache=cache,
                    func_cache=func_cache,
                )
            else:
                result = await asyncio.to_thread(
                    agent.analyze,
                    file_path,
                    skills_context=skills_context,
                    pruner=pruner,
                    covered_lines=covered,
                    cache=cache,
                    func_cache=func_cache,
                )
            return str(file_path), result

    async def _run_wave(
        self,
        agents,
        files,
        semaphore,
        *,
        wave_num,
        wave1_results=None,
        skills_context="",
        pruner=None,
        static_covered_lines=None,
        cache=None,
        func_cache=None,
    ) -> Dict[str, FindingList]:
        """Dispatch a wave of agents across all files."""
        if not agents:
            return {}

        agent_names = ", ".join(a.role_name for a in agents)
        label = "independent" if wave_num == 1 else "with Wave 1 context"
        console.print(
            f"\n   ── [bold]Wave {wave_num}[/bold]: {agent_names} ({label}) ──"
        )

        results: Dict[str, FindingList] = {}

        # Skip Wave 2 on files with no Wave 1 findings (saves time on clean files)
        clean_files_skipped = 0
        if wave_num == 2 and self.skip_wave2_clean and wave1_results:
            files_with_findings = {
                fp for fp, res in wave1_results.items() if res and res.findings
            }
            original_count = len(files)
            files = [fp for fp in files if str(fp) in files_with_findings]
            clean_files_skipped = original_count - len(files)
            if clean_files_skipped:
                console.print(
                    f"   ↳ ⚡ [green]Skipping Wave 2 on {clean_files_skipped} clean files[/green]"
                )

        tasks = []
        for fp in files:
            for agent in agents:
                if self._should_skip_agent_on_file(agent, fp):
                    continue

                covered = (
                    static_covered_lines.get(str(fp), set())
                    if static_covered_lines
                    else set()
                )
                tasks.append(
                    self._run_single_task(
                        agent,
                        fp,
                        semaphore,
                        wave1_results,
                        skills_context,
                        pruner=pruner,
                        covered_lines=covered,
                        cache=cache,
                        func_cache=func_cache,
                    )
                )

        logger.info(
            "Wave %d: dispatching %d tasks (%s)", wave_num, len(tasks), agent_names
        )
        console.print(f"   ↳ Firing {len(tasks)} Wave {wave_num} tasks...")

        # ── Stream results with early abort on consecutive failures ──
        ABORT_THRESHOLD = 5  # abort after this many consecutive empty results
        consecutive_empty = 0
        completed = 0
        aborted = False

        for coro in asyncio.as_completed(tasks):
            file_key, result = await coro
            completed += 1

            if file_key not in results:
                results[file_key] = FindingList(findings=[])

            if result and result.findings:
                results[file_key].findings.extend(result.findings)
                consecutive_empty = 0  # reset on success
            else:
                consecutive_empty += 1

            # Early abort: if the first N consecutive files all fail,
            # the provider is likely broken (auth error, rate limit, etc.)
            if (
                consecutive_empty >= ABORT_THRESHOLD
                and completed <= ABORT_THRESHOLD + 2
            ):
                console.print(
                    f"\n   🛑 [bold red]ABORTING Wave {wave_num}: "
                    f"first {consecutive_empty} files returned no results.[/bold red]\n"
                    f"   [red]Provider is likely failing (auth error / rate limit). "
                    f"Check revelyt.log.[/red]\n"
                    f"   [dim]Skipping remaining {len(tasks) - completed} tasks.[/dim]"
                )
                logger.error(
                    "Wave %d aborted: %d consecutive empty results in first %d tasks",
                    wave_num,
                    consecutive_empty,
                    completed,
                )
                aborted = True
                break

        if not aborted:
            logger.info("Wave %d: all %d tasks complete", wave_num, completed)

        # ── Post-wave sanity check (if not already aborted) ──
        if not aborted:
            total_findings = sum(len(r.findings) for r in results.values())
            if len(tasks) > 10 and total_findings == 0:
                console.print(
                    f"\n   ⚠️  [bold yellow]WARNING: Wave {wave_num} produced 0 findings "
                    f"across {completed} files.[/bold yellow]\n"
                    f"   [yellow]This almost certainly indicates a provider failure "
                    f"(auth error, rate limit, or network issue).\n"
                    f"   Check revelyt.log for details.[/yellow]"
                )

        return results

    def _deduplicate_cross_agent_findings(self) -> int:
        """
        Two-pass deduplication of findings for the current session.

        Pass 1 — same-file: removes near-duplicate titles within the same file.
          Catches Wave 1 + Wave 2 filing the same issue with slightly different
          wording on the same file (e.g. "echo=True in create_engine" vs
          "Echo enabled in create_engine").

        Pass 2 — cross-file: removes the same finding duplicated across files,
          keeping only the instance on the most architecturally appropriate file.
          Priority is determined by _detect_file_role() — a regex-based detector
          covering FastAPI, Flask, Django, Express, NestJS naming conventions.
          Tiebreaker: higher dependant_count (more imports = more authoritative).

        Near-duplicate detection: Jaccard >= 0.35 on word tokens (splitting on
        = _ - before stripping so "echo=True" tokenises to {"echo","true"})
        OR identical leading 3-word trigram.

        Returns total number of findings removed.
        """
        SEV_SCORE = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "INFO": 0}

        def _tokens(title: str) -> set:
            # Split on = _ - so "echo=True" → {"echo","true"}, "create_engine" → {"create","engine"}
            t = re.sub(r"[=_\-]", " ", title)
            clean = re.sub(r"[^a-z0-9\s]", "", t.lower())
            return set(w for w in clean.split() if len(w) > 2)

        def _trigram(title: str) -> tuple:
            t = re.sub(r"[=_\-]", " ", title)
            words = [
                w for w in re.sub(r"[^a-z0-9\s]", "", t.lower()).split() if len(w) > 2
            ]
            return tuple(words[:3])

        def _jaccard(a: str, b: str) -> float:
            ta, tb = _tokens(a), _tokens(b)
            return len(ta & tb) / len(ta | tb) if ta and tb else 0.0

        def _near_dup(a: str, b: str, threshold: float = 0.35) -> bool:
            if _jaccard(a, b) >= threshold:
                return True
            la, lb = _trigram(a), _trigram(b)
            return len(la) >= 2 and la == lb

        ids_to_delete: List[int] = []

        # Fetch findings + dependant_count for cross-file tiebreaking
        with self.session.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT f.id, f.file_path, f.title, f.severity, f.agent_name,
                       COALESCE(m.dependant_count, 0) AS dependant_count
                FROM findings f
                LEFT JOIN module_summaries m
                    ON f.file_path = m.file_path
                WHERE f.session_id = ?
                ORDER BY
                    CASE f.severity
                        WHEN 'CRITICAL' THEN 4 WHEN 'HIGH'   THEN 3
                        WHEN 'MEDIUM'   THEN 2 WHEN 'LOW'    THEN 1
                        ELSE 0
                    END DESC,
                    f.file_path ASC
            """,
                (self.session.session_id,),
            ).fetchall()

        all_findings = [dict(r) for r in rows]

        # ── PASS 1: same-file near-duplicate removal ───────────────────
        by_file: Dict[str, List[dict]] = {}
        for f in all_findings:
            by_file.setdefault(f["file_path"], []).append(f)

        pass1_deleted: Set[int] = set()
        for fp, file_findings in by_file.items():
            kept: List[dict] = []
            for finding in file_findings:
                if finding["id"] in pass1_deleted:
                    continue
                if any(_near_dup(finding["title"], k["title"]) for k in kept):
                    pass1_deleted.add(finding["id"])
                    ids_to_delete.append(finding["id"])
                else:
                    kept.append(finding)

        # ── PASS 2: cross-file near-duplicate removal ──────────────────
        surviving = [f for f in all_findings if f["id"] not in pass1_deleted]

        # Group by near-duplicate title across all files
        groups: List[List[dict]] = []
        for finding in surviving:
            placed = False
            for group in groups:
                if _near_dup(finding["title"], group[0]["title"]):
                    group.append(finding)
                    placed = True
                    break
            if not placed:
                groups.append([finding])

        for group in groups:
            if len(group) <= 1:
                continue
            # Sort: best severity → best role priority → most depended-upon file
            group.sort(
                key=lambda f: (
                    -SEV_SCORE.get(f["severity"], 0),
                    _detect_file_role(f["file_path"])[
                        0
                    ],  # lower role priority = more preferred
                    -f.get(
                        "dependant_count", 0
                    ),  # more dependants = more authoritative
                )
            )
            for dup in group[1:]:
                ids_to_delete.append(dup["id"])

        # ── Delete (deduplicate id list first) ────────────────────────
        ids_to_delete = list(set(ids_to_delete))

        if ids_to_delete:
            with self.session.get_connection() as conn:
                placeholders = ",".join("?" * len(ids_to_delete))
                conn.execute(
                    f"DELETE FROM findings WHERE id IN ({placeholders})",
                    ids_to_delete,
                )
            logger.info("Deduplication removed %d findings", len(ids_to_delete))
            console.print(
                f"   ↳ 🔄 Deduplication: removed [yellow]{len(ids_to_delete)}[/yellow] "
                f"near-duplicate findings "
                f"([dim]pass1: same-file · pass2: cross-file[/dim])"
            )

        return len(ids_to_delete)

    def _phase_elevation_and_reporting(self, project_root: Path):
        console.print(
            "\n📈 [bold blue]Phase 5 & 6: Severity Elevation & Reporting[/bold blue]"
        )

        # ── Cross-agent deduplication (before elevation so we don't promote ──
        # ── duplicates that are about to be deleted)                         ──
        self._deduplicate_cross_agent_findings()

        with self.session.get_connection() as conn:
            # ── Blast-radius elevation ──
            conn.execute("""
                UPDATE findings
                SET severity = 'CRITICAL', impact_scope = 'System-Wide'
                WHERE severity = 'HIGH' AND file_path IN (
                    SELECT file_path FROM module_summaries WHERE dependant_count >= 5
                )
            """)

            # ── Corroboration Engine ──
            corroborated = conn.execute("""
                SELECT file_path, title, COUNT(DISTINCT agent_name) as agent_count,
                       GROUP_CONCAT(DISTINCT agent_name) as agents
                FROM findings
                WHERE agent_name != ''
                GROUP BY file_path, title
                HAVING agent_count >= 2
            """).fetchall()

            corroboration_count = 0
            for row in corroborated:
                conn.execute(
                    """
                    UPDATE findings
                    SET severity = 'CRITICAL',
                        impact_scope = ?
                    WHERE file_path = ? AND title = ?
                """,
                    (
                        f"🔁 Corroborated ({row['agent_count']} agents: {row['agents']})",
                        row["file_path"],
                        row["title"],
                    ),
                )
                corroboration_count += 1

            if corroboration_count:
                logger.info(
                    "Corroboration: %d findings elevated by multi-agent agreement",
                    corroboration_count,
                )
                console.print(
                    f"   ↳ 🔁 [bold magenta]Corroborated: {corroboration_count} "
                    f"findings confirmed by multiple agents[/bold magenta]"
                )

            # ── Gather final metrics ──
            critical = conn.execute(
                "SELECT COUNT(*) FROM findings WHERE severity = 'CRITICAL'"
            ).fetchone()[0]
            high = conn.execute(
                "SELECT COUNT(*) FROM findings WHERE severity = 'HIGH'"
            ).fetchone()[0]
            medium = conn.execute(
                "SELECT COUNT(*) FROM findings WHERE severity = 'MEDIUM'"
            ).fetchone()[0]
            low = conn.execute(
                "SELECT COUNT(*) FROM findings WHERE severity = 'LOW'"
            ).fetchone()[0]

            console.print(f"   ↳ 🚨 [red]CRITICAL Issues: {critical}[/red]")
            console.print(f"   ↳ 🟠 [yellow]HIGH Issues: {high}[/yellow]")

            findings = conn.execute("""
                SELECT * FROM findings
                ORDER BY file_path ASC,
                    CASE severity
                        WHEN 'CRITICAL' THEN 1 WHEN 'HIGH'   THEN 2
                        WHEN 'MEDIUM'   THEN 3 WHEN 'LOW'    THEN 4
                        ELSE 5
                    END
            """).fetchall()

            # ── Top 3 riskiest files for executive summary ──
            riskiest_files = conn.execute("""
                SELECT file_path,
                       COUNT(*) as issue_count,
                       SUM(CASE severity
                           WHEN 'CRITICAL' THEN 4 WHEN 'HIGH' THEN 3
                           WHEN 'MEDIUM' THEN 2 WHEN 'LOW' THEN 1
                           ELSE 0 END) as risk_score
                FROM findings
                GROUP BY file_path
                ORDER BY risk_score DESC
                LIMIT 3
            """).fetchall()

        if not findings:
            console.print(
                "   ↳ ✅ [bold green]No issues found! Code is perfectly clean.[/bold green]"
            )
            return

        repo_root_str = str(project_root)
        grouped_findings = _group_findings(findings, repo_root_str)

        brd_results = []
        if self.brd_path:
            with self.session.get_connection() as conn:
                brd_results = conn.execute("SELECT * FROM brd_requirements").fetchall()

        md_content = _build_scan_report_markdown(
            target_path=self.target_path,
            critical=critical,
            high=high,
            medium=medium,
            low=low,
            total=len(findings),
            corroboration_count=corroboration_count,
            riskiest_files=riskiest_files,
            provider=self.provider,
            grouped_findings=grouped_findings,
            brd_results=brd_results,
            repo_root_str=repo_root_str,
        )

        html_content = markdown.markdown(
            md_content, extensions=["tables", "fenced_code"]
        )
        styled_html = _wrap_html(html_content)

        # ── Output (PDF or HTML) ──
        if getattr(self, "report_format", "pdf") == "html":
            report_path = Path("revelyt_report.html").resolve()
            logger.info("Generating HTML report: %s", report_path)
            report_path.write_text(styled_html, encoding="utf-8")
        else:
            report_path = Path("revelyt_report.pdf").resolve()
            logger.info("Generating PDF report: %s", report_path)
            with open(report_path, "w+b") as pdf_file:
                pisa.CreatePDF(styled_html, dest=pdf_file)
        logger.info("Report saved successfully: %s", report_path.name)

        in_tokens = getattr(self.provider, "total_input_tokens", 0)
        out_tokens = getattr(self.provider, "total_output_tokens", 0)
        total_cost = (
            self.provider.get_session_cost()
            if hasattr(self.provider, "get_session_cost")
            else 0.0
        )
        if in_tokens > 0:
            console.print(
                f"   ↳ 🪙  Tokens Used: [cyan]{in_tokens + out_tokens:,}[/cyan]"
                f" | Est Cost: [green]${total_cost:.4f}[/green]"
            )
        console.print(
            f"   ↳ ✅ [bold green]Report saved to: {report_path.name}[/bold green]"
        )


def _group_findings(findings: list, repo_root_str: str) -> dict:
    """Group findings by clean (sandbox-stripped) file path."""
    grouped = defaultdict(list)
    for row in findings:
        raw_path = str(row["file_path"])
        clean_path = raw_path.replace(repo_root_str, "").lstrip("/")
        grouped[clean_path].append(row)
    return grouped


def _wrap_html(html_body: str) -> str:
    """Wrap rendered HTML content in a styled HTML document."""
    return f"""
    <html><head><style>
        body  {{ font-family: Helvetica, sans-serif; font-size: 11px; }}
        h1    {{ color: #2C3E50; font-size: 18px; }}
        h2    {{ color: #2C3E50; border-bottom: 2px solid #3498DB;
                 padding-bottom: 5px; font-size: 14px; }}
        h3    {{ color: #2980B9; border-bottom: 1px solid #BDC3C7;
                 padding-bottom: 5px; font-size: 13px; }}
        h4    {{ margin-bottom: 2px; }}
        code  {{ background-color: #F4F6F7; padding: 1px 3px;
                 border-radius: 3px; font-family: Courier, monospace; font-size: 10px; }}
        pre   {{ background-color: #F4F6F7; padding: 8px;
                 border-left: 4px solid #3498DB;
                 white-space: pre-wrap; word-wrap: break-word; font-size: 10px; }}
        .exec-summary {{ background: #F8F9FA; padding: 12px; border-left: 4px solid #2980B9;
                         margin-bottom: 16px; }}
        /* Table layout — explicit widths prevent column squashing in xhtml2pdf */
        table {{ width: 100%; border-collapse: collapse; table-layout: fixed;
                 margin-bottom: 12px; }}
        th    {{ background-color: #2C3E50; color: white; padding: 6px 8px;
                 text-align: left; font-size: 11px; }}
        td    {{ padding: 5px 8px; border-bottom: 1px solid #E0E0E0;
                 vertical-align: top; word-wrap: break-word; font-size: 10px; }}
        tr:nth-child(even) {{ background-color: #F8F9FA; }}
        /* Findings table: File+Line 20%, Issue 35%, Fix 45% */
        table td:nth-child(1) {{ width: 20%; }}
        table td:nth-child(2) {{ width: 37%; }}
        table td:nth-child(3) {{ width: 43%; }}
        /* BRD table: # 4%, Status 10%, Requirement 50%, Evidence 36% */
        details {{ margin: 5px 0; }}
        summary {{ cursor: pointer; font-weight: bold; }}
    </style></head><body>
    {html_body}
    </body></html>
    """


def _build_scan_report_markdown(
    *,
    target_path,
    critical,
    high,
    medium,
    low,
    total,
    corroboration_count,
    riskiest_files,
    provider,
    grouped_findings,
    brd_results,
    repo_root_str,
) -> str:
    """Assemble the full Markdown content for the scan report."""
    in_tokens = getattr(provider, "total_input_tokens", 0)
    out_tokens = getattr(provider, "total_output_tokens", 0)
    total_cost = (
        provider.get_session_cost() if hasattr(provider, "get_session_cost") else 0.0
    )

    risk_word = "minimal"
    if critical > 0:
        risk_word = "critical"
    elif high > 0:
        risk_word = "significant"
    elif medium > 0:
        risk_word = "moderate"

    top_risk_lines = ""
    for rf in riskiest_files:
        clean_risk = str(rf["file_path"]).replace(repo_root_str, "").lstrip("/")
        top_risk_lines += (
            f"- `{clean_risk}` — {rf['issue_count']} issues "
            f"(risk score: {rf['risk_score']})\n"
        )

    corroboration_line = ""
    if corroboration_count:
        corroboration_line = (
            f"**{corroboration_count} finding(s)** were independently confirmed by "
            f"multiple AI agents (🔁 Corroborated), significantly increasing confidence.\n\n"
        )

    md = f"""
# \U0001f6e1\ufe0f Revelyt Audit Report

**Target Repository:** `{target_path}`

---

## \U0001f4cb Executive Summary

Your codebase has **{total} issue(s)** with an overall risk level of **{risk_word}**.

- \U0001f6a8 **CRITICAL:** <span style="color:red">{critical}</span>
- \U0001f7e0 **HIGH:** <span style="color:orange">{high}</span>
- \U0001f7e1 **MEDIUM:** {medium}
- \U0001f7e2 **LOW:** {low}

{corroboration_line}**Top 3 Riskiest Files:**
{top_risk_lines}

---

## \U0001f4ca Scan Metrics
- **CRITICAL Issues:** <span style="color:red">{critical}</span>
- **HIGH Issues:** <span style="color:orange">{high}</span>
- **Total Issues Found:** {total}

## \U0001f4b8 LLM Cost & Token Usage
- **Input Tokens:** {in_tokens:,}
- **Output Tokens:** {out_tokens:,}
- **Estimated Cloud Cost:** **${total_cost:.4f}** *(Saved by using Open-Weight local execution)*

---

## \U0001f4c1 Detailed Findings by Directory
"""
    md += _build_findings_md(grouped_findings)
    md += _build_brd_md(brd_results, repo_root_str)
    return md


def _build_findings_md(grouped_findings: dict) -> str:
    """Render findings as compact severity-grouped tables."""
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
    all_findings = []
    for clean_path, issues in grouped_findings.items():
        for issue in issues:
            row = dict(issue) if not isinstance(issue, dict) else issue
            row["_clean_path"] = clean_path
            all_findings.append(row)

    all_findings.sort(key=lambda f: severity_order.get(f.get("severity", "INFO"), 5))

    MAX_FINDINGS = 100
    truncated_count = 0
    if len(all_findings) > MAX_FINDINGS:
        important = [
            f
            for f in all_findings
            if f.get("severity") in ("CRITICAL", "HIGH", "MEDIUM")
        ]
        low = [
            f
            for f in all_findings
            if f.get("severity") not in ("CRITICAL", "HIGH", "MEDIUM")
        ]
        budget = max(MAX_FINDINGS - len(important), 0)
        truncated_count = len(low) - budget
        all_findings = important + low[:budget]

    if not all_findings:
        return "\n*No issues found.*\n"

    tiers = {}
    for f in all_findings:
        tiers.setdefault(f.get("severity", "INFO"), []).append(f)

    LABELS = {
        "CRITICAL": "\U0001f6a8 CRITICAL",
        "HIGH": "\U0001f7e0 HIGH",
        "MEDIUM": "\U0001f7e1 MEDIUM",
        "LOW": "\U0001f7e2 LOW",
        "INFO": "\u2139\ufe0f INFO",
    }

    md = ""
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
        findings = tiers.get(sev, [])
        if not findings:
            continue
        md += f"\n### {LABELS.get(sev, sev)} ({len(findings)})\n\n"
        # Three-column layout: File + Line, Issue, Suggested Fix.
        # Merging File and Line into one column gives Issue and Fix more horizontal
        # space in the PDF, preventing mid-sentence truncation.
        md += "| File | Issue | Suggested Fix |\n"
        md += "|------|-------|---------------|\n"
        for f in findings:
            fname = f["_clean_path"].rsplit("/", 1)[-1]

            line_no = f.get("line_number")
            loc = f" L{line_no}" if line_no else ""
            file_cell = f"`{fname}`{loc}"

            class_method = f.get("class_or_method")
            if class_method and class_method not in ("Global", "", None):
                file_cell += f" · {class_method}"  # plain text, no HTML

            title = str(f.get("title", "")).replace("|", "\u2014")
            suggestion = (
                str(f.get("suggestion", "")).replace("|", "\u2014").replace("\n", " ")
            )
            md += f"| {file_cell} | {title} | {suggestion} |\n"
        md += "\n"

    if truncated_count > 0:
        md += f"\n*{truncated_count} LOW/INFO findings omitted for brevity.*\n"
    return md


def _build_brd_md(brd_results: list, repo_root_str: str) -> str:
    """Render BRD verification as a compact table."""
    if not brd_results:
        return ""
    STATUS_ICON = {"MET": "\u2705", "NOT_MET": "\u274c", "PARTIAL": "\u26a0\ufe0f"}
    md = "\n## \u2696\ufe0f Business Requirements (BRD) Verification\n\n"
    md += "| # | Status | Requirement | Evidence |\n"
    md += "|---|--------|-------------|----------|\n"
    for r in brd_results:
        req = dict(r) if not isinstance(r, dict) else r
        icon = STATUS_ICON.get(req["status"], "\u2753")
        req_text = str(req["requirement"]).replace("|", "\u2014")
        evidence = (
            str(req.get("evidence", ""))
            .replace(repo_root_str, "")
            .replace("//", "/")
            .replace("|", "\u2014")
            .replace("\n", " ")
        )
        md += f"| {req['id']} | {icon} {req['status']} | {req_text} | {evidence} |\n"
    md += "\n"
    return md
