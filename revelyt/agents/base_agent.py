"""
BaseAgent — Abstract base class for all Revelyt review agents.

Every agent inherits from BaseAgent and overrides ``_build_system_prompt``
to supply its role-specific instructions.  The shared ``analyze()`` loop,
language detection, finding persistence, and Wave-1 context injection are
handled here so concrete agents stay focused on prompt engineering.

OSS MODEL ACCURACY FIXES applied here:
- User message template now includes an EVIDENCE REQUIREMENT block that forces
  the model to cite specific function names / line content per finding.
  This eliminates "I assume this is missing" hallucinations.
- Wave-2 context injection is tightened: only CRITICAL/HIGH findings forwarded
  (unchanged), but the "corroborate" instruction is made more explicit about
  NOT inventing new issues without code evidence.
"""

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any, Set

from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console

from revelyt.core.schemas import FindingList
from revelyt.core.session import SessionManager
from revelyt.providers.base import BaseProvider
from revelyt.core.code_optimizer import (
    prepare_code_for_analysis,
    compress_wave1_context,
    compute_token_budget,
    extract_code_skeleton,
)

logger = logging.getLogger(__name__)
console = Console()

# ── Elapsed timer (set by orchestrator at scan start) ──────────────────────
_scan_start_time: float | None = None


def set_scan_start_time(t: float) -> None:
    """Called by the orchestrator to mark scan start for elapsed display."""
    global _scan_start_time
    _scan_start_time = t


def _elapsed() -> str:
    """Return [MM:SS] or [HH:MM:SS] elapsed since scan start."""
    if _scan_start_time is None:
        return ""
    secs = int(time.monotonic() - _scan_start_time)
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    tag = f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
    return f"[dim][{tag}][/dim] "


PYTHON_EXTENSIONS = {".py", ".pyi"}
JS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
CONFIG_FILENAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.staging",
    ".env.development",
    ".env.test",
}
CONFIG_SECRET_FILENAMES = {
    "secrets.json",
    "config.json",
    "app.config.json",
    "appsettings.json",
    "application.yaml",
    "application.yml",
}


# ── Model tier detection ───────────────────────────────────────────────────
# Frontier models (Gemini, GPT, Claude) have strong built-in code knowledge
# and don't need OSS-specific guardrails (prohibitions, SAFE skill rules).
# OSS models (Qwen, LLaMA, Mistral etc.) need the full guardrail stack.

_OSS_MODEL_PREFIXES = (
    "qwen",
    "llama",
    "llama2",
    "llama3",
    "mistral",
    "mixtral",
    "deepseek",
    "phi",
    "phi3",
    "phi4",
    "gemma",
    "codellama",
    "starcoder",
    "wizardcoder",
    "codegemma",
    "granite",
)

_FRONTIER_MODEL_PREFIXES = (
    "gemini",
    "gpt",
    "gpt-3",
    "gpt-4",
    "o1",
    "o3",
    "claude",
    "text-davinci",
    "command",
    "kimi",
    "moonshot",
)


def _is_oss_model(model_name: str) -> bool:
    """
    Return True if model_name is an open-weight OSS model that needs
    hallucination guardrails (prohibitions + SAFE skill rules).
    Returns True for unknown models (conservative — apply guardrails by default).
    """
    name = model_name.lower().split(":")[0]  # strip Ollama tag e.g. "qwen2.5-coder:14b"
    if any(name.startswith(p) for p in _FRONTIER_MODEL_PREFIXES):
        return False
    # Unknown models: default to OSS guardrails (safe default)
    return True


# ── Analyzable-file check ──────────────────────────────────────────────────
# Exposed as a public module-level function so the orchestrator can pre-filter
# files before creating tasks — avoiding wasted semaphore slots / sleep time
# for files no agent can review (e.g. .html, .css, .md, .json).

_ANALYZABLE_EXTENSIONS = (
    PYTHON_EXTENSIONS
    | JS_EXTENSIONS
    | {
        ".ini",
        ".cfg",
        ".toml",
        ".yaml",
        ".yml",
    }
)
_ANALYZABLE_FILENAMES = CONFIG_FILENAMES | CONFIG_SECRET_FILENAMES


def is_analyzable(file_path: Path) -> bool:
    """
    Return True if any agent can analyze this file (known language).

    Uses the same extension / filename logic as BaseAgent._get_language()
    so the orchestrator can skip files before they consume concurrency slots.
    """
    if file_path.suffix in _ANALYZABLE_EXTENSIONS:
        return True
    if file_path.name in _ANALYZABLE_FILENAMES:
        return True
    return False


# Skeleton threshold by model tier.
# Pro-class models (high reasoning capacity) invent completions for truncated
# code more readily than Flash/Flash-Lite, producing false positives at low
# thresholds. Higher threshold = less truncation = fewer boundary artifacts.
#
# Tier detection uses substring matching on the normalised model name:
#   "pro"  → Pro tier  → threshold 150 lines
#   "lite" → Lite tier → threshold  60 lines
#   else   → Flash tier → threshold  80 lines (default for frontier models)
#
# For OSS models (Qwen, LLaMA, etc.), we use AST-based skeleton extraction
# which is MORE accurate than model inference on partial code, so we can use
# a LOWER threshold to save tokens.

_SKELETON_THRESHOLDS = {
    "pro": 150,  # Pro: powerful reasoner, invents completions — needs more context
    "lite": 60,  # Flash Lite: conservative reasoner, fine with aggressive skipping
    "flash": 80,  # Flash: default frontier threshold
}

# OSS models benefit from skeleton mode even more than frontier models because:
# 1. AST-based extraction is deterministic and accurate
# 2. OSS models have smaller context windows
# 3. Skeleton mode captures function signatures perfectly
_OSS_SKELETON_THRESHOLD = 50  # Files >50 lines use skeleton mode for OSS models


def _get_skeleton_threshold(model_name: str) -> int:
    """
    Return the appropriate skeleton line-count threshold for any model.

    Frontier models (Gemini, GPT, Claude): Higher thresholds because they
    hallucinate at skeleton boundaries more than Flash models.

    OSS models (Qwen, LLaMA, etc.): Lower threshold because AST-based
    skeleton extraction is deterministic and captures all function signatures.
    """
    name = model_name.lower().split(":")[0]  # strip Ollama tag

    if _is_oss_model(name):
        return _OSS_SKELETON_THRESHOLD

    # Frontier model tier detection
    if "lite" in name:
        return _SKELETON_THRESHOLDS["lite"]
    if "pro" in name:
        return _SKELETON_THRESHOLDS["pro"]
    return _SKELETON_THRESHOLDS["flash"]


class BaseAgent(ABC):
    """
    Abstract base for every Revelyt review agent.

    Attributes:
        wave:       1 (independent) or 2 (receives Wave-1 context).
        role_name:  Human-readable role, e.g. "Software Architect".
        emoji:      Rich-console prefix, e.g. "🏗️".
        category:   Default ``category`` value stored in the findings table.
    """

    wave: int = 0
    role_name: str = ""
    emoji: str = ""
    category: str = ""
    relevant_languages: set | None = None
    use_skeleton_only: bool = False

    # Enable tier-aware skeleton mode for frontier models on this agent.
    # When set to any non-None value, analyze() will skeleton files that exceed
    # _get_skeleton_threshold(model_name) — which automatically returns
    # Pro=150, Flash=80, Flash-Lite=60 line thresholds.
    # Set to None (default) to always use full code (Wave 2 behaviour).
    # Set to any int (e.g. 80) to enable on Wave 1.
    # OSS models always receive full code regardless of this setting.
    frontier_skeleton_threshold: int | None = None

    def __init__(self, session: SessionManager, provider: BaseProvider):
        self.session = session
        self.provider = provider
        
    def _get_batch_summaries(self, file_paths: list[Path]) -> list[dict]:
        """Fetch module summaries from the DB for context injection."""
        summaries = []
        with self.session.get_connection() as conn:
            for fp in file_paths:
                try:
                    rel = str(fp.relative_to(fp.parents[2]))
                except Exception:
                    rel = str(fp)
                
                row = conn.execute(
                    "SELECT file_path, purpose FROM module_summaries WHERE file_path = ?",
                    (rel,)
                ).fetchone()
                
                if row and row["purpose"]:
                    summaries.append({
                        "file_path": row["file_path"],
                        "purpose": row["purpose"]
                    })
        return summaries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        file_path: Path,
        context: Optional[FindingList] = None,
        *,
        skills_context: str = "",
        pruner: Optional[Any] = None,
        covered_lines: Optional[Set[int]] = None,
        cache: Optional[Any] = None,
        func_cache: Optional[Any] = None,
    ) -> FindingList:
        """
        Review a single file and persist any findings.

        Uses function-level caching when func_cache is provided:
        1. Parse file into function/class blocks
        2. Check cache for each block
        3. Only send uncached blocks to LLM
        4. Merge cached + new findings
        """
        language = self._get_language(file_path)
        if language == "unknown":
            return FindingList(findings=[])
        if (
            self.relevant_languages is not None
            and language not in self.relevant_languages
        ):
            return FindingList(findings=[])

        try:
            raw_content = file_path.read_text(errors="ignore")

            if pruner:
                raw_content = pruner.prune(file_path, raw_content)

            # Determine whether to use skeleton mode for this file.
            # Priority: explicit use_skeleton_only > skeleton threshold > full code.
            # - Frontier models (Gemini, GPT, Claude): Pro=150, Flash=80, Lite=60
            # - OSS models (Qwen, LLaMA): 50 lines (AST skeleton is MORE accurate)
            use_skeleton = self.use_skeleton_only
            if not use_skeleton and self.frontier_skeleton_threshold is not None:
                threshold = _get_skeleton_threshold(self.provider.model_name)
                line_count = raw_content.count("\n") + 1
                if line_count > threshold:
                    use_skeleton = True
                    logger.debug(
                        "%s [%s] %s: %d lines > %s threshold %d, using skeleton",
                        self.emoji,
                        self.role_name,
                        file_path.name,
                        line_count,
                        self.provider.model_name,
                        threshold,
                    )

            if use_skeleton:
                code_content = extract_code_skeleton(raw_content, str(file_path))
            else:
                is_oss = _is_oss_model(self.provider.model_name)
                budget = compute_token_budget(raw_content, is_oss_model=is_oss)
                code_content = prepare_code_for_analysis(
                    raw_content, str(file_path), max_chars=budget
                )

            if cache:
                # Compute prohibition hash for cache key
                prohibition_hash: str = hashlib.sha256(
                    self._get_inline_prohibitions(language).encode()
                ).hexdigest()[:8]
                cached_res = cache.get(
                    self.role_name,
                    str(file_path),
                    skills_context,
                    code_content,
                    self.provider.model_name,
                    prohibition_hash,
                )
                if cached_res:
                    console.print(
                        f"   ↳ ⚡ [green]Cache hit for {file_path.name} ({self.role_name}). 0 tokens used![/green]"
                    )
                    self._save_findings(
                        str(file_path), cached_res.findings, self.role_name
                    )
                    return cached_res
        except Exception as exc:
            logger.warning(
                "%s [%s] Skipping %s: %s",
                self.emoji,
                self.role_name,
                file_path.name,
                exc,
            )
            console.print(
                f"   ↳ [yellow]{self.emoji} [{self.role_name}] "
                f"Skipping {file_path.name}: {exc}[/yellow]"
            )
            return FindingList(findings=[])

        if covered_lines:
            lines_str = ", ".join(map(str, sorted(covered_lines)))
            skills_context += (
                f"\n\n🚨 IMPORTANT: The following lines have already been flagged "
                f"by static analysis tools (Ruff/Bandit): {lines_str}. "
                f"DO NOT report trivial/linting issues on these lines."
            )

        # ── Function-level caching ──
        if func_cache and not use_skeleton:
            from revelyt.core.function_cache import (
                extract_function_blocks,
                merge_findings,
            )
            from revelyt.core.schemas import FindingList as FL

            blocks = extract_function_blocks(raw_content, str(file_path))
            prohibition_hash = hashlib.sha256(
                self._get_inline_prohibitions(language).encode()
            ).hexdigest()[:8]

            cached_blocks = {}
            uncached_blocks = []

            for block in blocks:
                bname = block["name"]
                bcontent = block["content"]
                cached = func_cache.get(
                    self.role_name,
                    str(file_path),
                    bname,
                    bcontent,
                    self.provider.model_name,
                    prohibition_hash,
                )
                if cached:
                    cached_blocks[bname] = cached
                else:
                    uncached_blocks.append(block)

            import time
            start_t = time.time()
            # If all blocks cached, return merged cached findings
            if not uncached_blocks:
                merged = merge_findings(cached_blocks, {}, blocks)
                elapsed = time.time() - start_t
                if merged.findings:
                    console.print(
                        f"   ↳ ⚡ [green]Function cache hit for {file_path.name} ({self.role_name}). 0 tokens used![/green] ({elapsed:.1f}s)"
                    )
                self._save_findings(str(file_path), merged.findings, self.role_name)
                return merged

            # Build prompt with only uncached blocks
            uncached_content = "\n\n".join(b["content"] for b in uncached_blocks)
            prompt = self._build_prompt(
                language, str(file_path), uncached_content, context, skills_context
            )

            result: FindingList = self.provider.invoke_structured(
                prompt=prompt,
                output_schema=FindingList,
                file_path=str(file_path),
                code_content=uncached_content,
            )

            if result is None or not result.findings:
                # No new findings — merge cached findings only
                merged = merge_findings(cached_blocks, {}, blocks)
                self._save_findings(str(file_path), merged.findings, self.role_name)
                return merged

            # Cache the new findings per block
            new_findings_by_block = {}
            for block in uncached_blocks:
                # Associate findings with their block (simplified: all findings go to first uncached block)
                new_findings_by_block[block["name"]] = result

            # Cache each new finding block
            func_cache.set(
                self.role_name,
                str(file_path),
                uncached_blocks[0]["name"],
                uncached_blocks[0]["content"],
                result,
                self.provider.model_name,
                prohibition_hash,
            )

            # Merge cached + new findings
            merged = merge_findings(cached_blocks, new_findings_by_block, blocks)
            self._save_findings(str(file_path), merged.findings, self.role_name)

            cached_count = len(cached_blocks)
            new_count = len(uncached_blocks)
            if cached_count:
                console.print(
                    f"   ↳ ⚡ [green]{cached_count} functions cached, {new_count} analyzed in {file_path.name}[/green]"
                )

            return merged

        # ── Standard (non-function-level) path ──
        prompt = self._build_prompt(language, str(file_path), code_content, context, skills_context)

        try:
            result: FindingList = self.provider.invoke_structured(
                prompt=prompt,
                output_schema=FindingList,
                file_path=str(file_path),
                code_content=code_content,
            )

            import time
            start_t_non_cache = time.time()
            if result is None:
                # Total failure: all LLM passes failed (auth error, rate limit, etc.)
                elapsed = time.time() - start_t_non_cache
                console.print(
                    f"   ↳ [bold red]{self.emoji} [{self.role_name}]  "
                    f"{file_path.name:<25} → "
                    f"FAILED (provider returned no result — check logs)[/bold red] ({elapsed:.1f}s)"
                )
                return FindingList(findings=[])

            if result and result.findings:
                if cache:
                    # Compute prohibition hash for cache key (same as above)
                    ph_hash: str = hashlib.sha256(
                        self._get_inline_prohibitions(language).encode()
                    ).hexdigest()[:8]
                    cache.set(
                        self.role_name,
                        str(file_path),
                        skills_context,
                        code_content,
                        result,
                        self.provider.model_name,
                        ph_hash,
                    )

                self._save_findings(str(file_path), result.findings, self.role_name)
                logger.info(
                    "%s [%s] %s → %d findings",
                    self.emoji,
                    self.role_name,
                    file_path.name,
                    len(result.findings),
                )
                elapsed = time.time() - start_t_non_cache
                console.print(
                    f"   ↳ {self.emoji} [{self.role_name}]  "
                    f"{file_path.name:<25} → "
                    f"[red]{len(result.findings)}[/red] findings ({elapsed:.1f}s)"
                )
            else:
                elapsed = time.time() - start_t_non_cache
                console.print(
                    f"   ↳ {self.emoji} [{self.role_name}]  "
                    f"{file_path.name:<25} → "
                    f"[green]clean[/green] ({elapsed:.1f}s)"
                )
                result = result or FindingList(findings=[])

            return result

        except Exception as exc:
            logger.error(
                "%s [%s] Failed on %s: %s",
                self.emoji,
                self.role_name,
                file_path.name,
                exc,
                exc_info=True,
            )
            elapsed_err = time.time() - start_t_non_cache if 'start_t_non_cache' in locals() else 0
            console.print(
                f"   ↳ [bold red]{self.emoji} [{self.role_name}] "
                f"Failed on {file_path.name}: {exc}[/bold red] ({elapsed_err:.1f}s)"
            )
            return FindingList(findings=[])

    def analyze_batch(
        self,
        file_paths: list[Path],
        *,
        skills_context: str = "",
        pruner: Optional[Any] = None,
        covered_lines_by_file: Optional[dict[str, set[int]]] = None,
        cache: Optional[Any] = None,
    ) -> dict[str, FindingList]:
        """
        Analyze multiple related files in a single LLM call (Wave 1 only).

        Follows the same pattern as ProductManagerAgent.summarize_batch():
          - Builds multi-file prompt with --- FILE: path --- separators
          - Invokes LLM with BatchFindingList schema
          - Maps findings back to individual files
          - Falls back to individual analyze() per-file on failure

        Args:
            file_paths: Files from the same directory to review together.
            skills_context: SAFE skill rules injected into the prompt.
            pruner: Optional DiffPruner — called per-file before batching.
            covered_lines_by_file: Map of file_path → set of lines already
                flagged by Ruff/Bandit (Gap 2: per-file static coverage).
            cache: Optional ReviewCache for hit/miss checking.

        Returns:
            Dict mapping str(file_path) → FindingList.
        """
        from revelyt.core.schemas import BatchFindingList
        from revelyt.core.code_optimizer import extract_code_skeleton
        import time

        results: dict[str, FindingList] = {}
        covered_map = covered_lines_by_file or {}

        # ── Build file blocks with per-file metadata ──
        file_entries: list[tuple[Path, str, str]] = []  # (Path, rel_path, block)
        for fp in file_paths:
            language = self._get_language(fp)
            if language == "unknown":
                continue
            if (
                self.relevant_languages is not None
                and language not in self.relevant_languages
            ):
                continue

            try:
                raw = fp.read_text(errors="ignore")
                if pruner:
                    raw = pruner.prune(fp, raw)
                skeleton = extract_code_skeleton(raw, str(fp))
                rel = fp.name  # Use filename for compact prompts
                try:
                    rel = str(fp.relative_to(fp.parents[2]))  # e.g. "routes/auth.js"
                except (ValueError, IndexError):
                    pass

                # Per-file covered lines annotation (Gap 2)
                covered = covered_map.get(str(fp), set())
                if covered:
                    lines_str = ", ".join(map(str, sorted(covered)))
                    header = (
                        f"--- FILE: {rel} (Ruff/Bandit flagged lines: {lines_str}) ---"
                    )
                else:
                    header = f"--- FILE: {rel} ---"

                file_entries.append((fp, rel, f"{header}\n{skeleton}"))
            except Exception as exc:
                logger.warning("Batch skipping %s: %s", fp.name, exc)

        # Not enough files for batching — fall back to individual
        if len(file_entries) <= 1:
            for fp in file_paths:
                covered = covered_map.get(str(fp), set())
                result = self.analyze(
                    fp,
                    skills_context=skills_context,
                    pruner=pruner,
                    covered_lines=covered,
                    cache=cache,
                )
                results[str(fp)] = result
            return results

        combined = "\n\n".join(block for _, _, block in file_entries)

        # Build batch prompt using the agent's system prompt
        language = self._get_language(file_entries[0][0])
        system_prompt = self._build_system_prompt(language)

        # Inject skills context and prohibitions
        prohibitions = self._get_inline_prohibitions(language)
        if prohibitions:
            system_prompt += f"\n\n{prohibitions}"

        # Inject summaries for context (especially critical for local 14B execution)
        summaries = self._get_batch_summaries(file_paths)
        if summaries:
            summary_context = "--- FILE PURPOSES ---\n" + "\n".join(
                f"• {s['file_path']}: {s['purpose']}" for s in summaries
            ) + "\n\n"
            system_prompt += f"\n\n{summary_context}"

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "user",
                    "Review ALL files below for security vulnerabilities, bugs, and code quality issues.\n"
                    "Return findings grouped by file using the BatchFindingList schema.\n"
                    "Lines already flagged by static analysis (shown in parentheses) should NOT be re-reported.\n\n"
                    "{code_content}",
                ),
            ]
        )

        try:
            batch_start = time.time()
            batch_result: BatchFindingList = self.provider.invoke_structured(
                prompt=prompt,
                output_schema=BatchFindingList,
                code_content=combined,
            )
            batch_elapsed = time.time() - batch_start

            if batch_result is None or not batch_result.files:
                raise ValueError("Batch returned empty result")

            # Map findings back to individual files
            for file_finding in batch_result.files:
                # Try to match the file_path from LLM output to our entries
                matched_fp = None
                for fp, rel, _ in file_entries:
                    if file_finding.file_path in (rel, str(fp), fp.name):
                        matched_fp = fp
                        break
                if not matched_fp:
                    # LLM returned a path we don't recognize — skip
                    logger.warning(
                        "Batch: unrecognized file_path '%s'", file_finding.file_path
                    )
                    continue

                fp_key = str(matched_fp)
                finding_list = FindingList(findings=file_finding.findings)
                
                # Single-Pass Architecture: Write the BRD summary to the database
                if file_finding.purpose or file_finding.key_functions:
                    import json
                    try:
                        with self.session.get_connection() as conn:
                            # Use relative path since that's what PM agent used
                            rel_path = str(matched_fp.relative_to(matched_fp.parents[2]))
                            conn.execute(
                                "UPDATE module_summaries SET purpose = ?, key_functions = ? "
                                "WHERE file_path = ?",
                                (file_finding.purpose, json.dumps(file_finding.key_functions), rel_path),
                            )
                    except Exception as e:
                        # Fallback for relative path extraction issues just in case
                        with self.session.get_connection() as conn:
                            conn.execute(
                                "UPDATE module_summaries SET purpose = ?, key_functions = ? "
                                "WHERE file_path = ?",
                                (file_finding.purpose, json.dumps(file_finding.key_functions), fp_key),
                            )

                if finding_list.findings:
                    self._save_findings(fp_key, finding_list.findings, self.role_name)
                    console.print(
                        f"   ↳ {self.emoji} [{self.role_name}]  "
                        f"{matched_fp.name:<25} → "
                        f"[bold red]{len(finding_list.findings)} findings[/bold red] ({batch_elapsed:.1f}s)"
                    )
                else:
                    console.print(
                        f"   ↳ {self.emoji} [{self.role_name}]  "
                        f"{matched_fp.name:<25} → [green]clean[/green] ({batch_elapsed:.1f}s)"
                    )

                results[fp_key] = finding_list

            # Gap-fill: any files not in the LLM response
            for fp, rel, _ in file_entries:
                if str(fp) not in results:
                    results[str(fp)] = FindingList(findings=[])

            logger.info(
                "%s [%s] Batch reviewed %d files → %d total findings",
                self.emoji,
                self.role_name,
                len(file_entries),
                sum(len(r.findings) for r in results.values()),
            )
            return results

        except Exception as exc:
            logger.warning(
                "%s [%s] Batch failed (%s), falling back to individual",
                self.emoji,
                self.role_name,
                exc,
            )
            # Fall back to individual analysis for all files in this batch
            for fp in file_paths:
                covered = covered_map.get(str(fp), set())
                result = self.analyze(
                    fp,
                    skills_context=skills_context,
                    pruner=pruner,
                    covered_lines=covered,
                    cache=cache,
                )
                results[str(fp)] = result
            return results

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        language: str,
        file_path_str: str,
        file_content: str,
        context: Optional[FindingList] = None,
        skills_context: str = "",
    ) -> ChatPromptTemplate:
        """
        Assemble the full prompt for this agent.

        Message order (recency = influence for local 14B models):
          [0] system  — role + framework safety rules
          [1] user    — skills compact_rules
          [2] user    — wave-1 findings context (wave-2 only)
          [3] user    — the actual code
          [4] user    — PROHIBITIONS + evidence requirement  <- LAST, highest recency

        The prohibitions are injected as the FINAL message so they are
        the highest-recency tokens the model reads before generating output.
        For autoregressive local models, final-message instructions have
        the strongest influence on the first generated tokens — this is why
        prohibitions in the system prompt (position 0) were being ignored
        while the model pattern-matched against the code (position 3).

        _get_inline_prohibitions() is a hook wave agents override to return
        their NEVER-flag rules. BaseAgent returns "" so agents without
        prohibitions are unaffected.
        """
        system_prompt = self._build_system_prompt(language)
        
        # Inject summary for context (critical for local 14B execution)
        fp_path = Path(file_path_str)
        summaries = self._get_batch_summaries([fp_path])
        if summaries:
            summary_context = f"--- PRIOR KNOWLEDGE ---\nFile Purpose: {summaries[0]['purpose']}\n\n"
            system_prompt += f"\n\n{summary_context}"
            
        messages = [("system", system_prompt)]

        # Only inject skills guardrails for OSS models — frontier models
        # (Gemini, GPT, Claude) have built-in knowledge of framework patterns
        # and the SAFE: rules add token cost without accuracy benefit.
        if skills_context and _is_oss_model(self.provider.model_name):
            skills_safe = skills_context.replace("{", "{{").replace("}", "}}")
            messages.append(("user", skills_safe))

        # Context injection for Wave-2 agents
        if context and context.findings:
            relevant = sorted(
                [
                    f
                    for f in context.findings
                    if f.severity in ("CRITICAL", "HIGH")
                    or f.category == getattr(self, "category", "")
                ],
                key=lambda f: {
                    "CRITICAL": 0,
                    "HIGH": 1,
                    "MEDIUM": 2,
                    "LOW": 3,
                    "INFO": 4,
                }.get(getattr(f, "severity", ""), 5),
            )
            filtered_context = FindingList(findings=list(relevant)[:5])
            compressed = compress_wave1_context(filtered_context)
            if compressed:
                findings_json_safe = compressed.replace("{", "{{").replace("}", "}}")
                messages.append(
                    (
                        "user",
                        f"{findings_json_safe}\n\n"
                        f"Corroborate high-risk Wave 1 findings ONLY if you can see supporting "
                        f"evidence in the code below. Do NOT invent corroboration. "
                        f"Add new insight only — do not duplicate.",
                    )
                )

        # Code message — separate from prohibitions so the model reads the
        # code first, then immediately hits the prohibition checklist.
        messages.append(
            (
                "user",
                "File Path: {file_path}\n\nCode Content:\n```\n{code_content}\n```",
            )
        )

        # Final message — prohibitions + evidence gate.
        # Position: last in context = highest recency = strongest influence.
        # The model reads this immediately before generating the first output token.
        inline_prohibitions = self._get_inline_prohibitions(language)
        prohibition_block = ""
        if inline_prohibitions:
            prohibition_block = (
                inline_prohibitions.replace("{", "{{").replace("}", "}}") + "\n\n"
            )

        messages.append(
            (
                "user",
                f"{prohibition_block}"
                "Return at most 5 findings, prioritised by severity. "
                "Keep each description and suggestion to 1-2 sentences max.\n\n"
                "BEFORE reporting each finding, verify all three:\n"
                "  1. Can I name the exact function or variable with this problem?\n"
                "  2. Is the problematic code literally present in the code shown above?\n"
                "  3. Is this NOT one of the prohibited or SAFE patterns listed above?\n"
                "If any answer is NO, do not include the finding.",
            )
        )

        return ChatPromptTemplate.from_messages(messages)

    def _get_inline_prohibitions(self, language: str) -> str:
        """
        Return agent-specific NEVER-flag rules for inline injection.

        Called by _build_prompt() and injected as the FINAL user message
        so prohibitions have maximum recency when the model generates output.

        Frontier models (Gemini, GPT, Claude) do not need prohibitions —
        they don't hallucinate the OSS-specific false positive patterns.
        Returns "" for frontier models regardless of subclass override.
        """
        if not _is_oss_model(self.provider.model_name):
            return ""
        return ""

    @abstractmethod
    def _build_system_prompt(self, language: str) -> str:
        """
        Return the system-prompt string specific to this agent's role.
        """
        ...

    # ------------------------------------------------------------------
    # Language detection
    # ------------------------------------------------------------------

    @staticmethod
    def _get_language(file_path: Path) -> str:
        """Detect source language from file extension or name."""
        if file_path.suffix in PYTHON_EXTENSIONS:
            return "python"
        if file_path.suffix in JS_EXTENSIONS:
            return "javascript"
        if file_path.name in CONFIG_FILENAMES:
            return "config"
        if file_path.name in CONFIG_SECRET_FILENAMES:
            return "config"
        if file_path.suffix in {".ini", ".cfg", ".toml", ".yaml", ".yml"}:
            return "config"
        return "unknown"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_findings(
        self, file_path: str, findings: list, agent_name: str = ""
    ) -> None:
        """Insert findings into the session database using a single executemany call."""
        if not findings:
            return
        rows = [
            (
                self.session.session_id,
                file_path,
                agent_name,
                finding.severity,
                self.category,
                finding.title,
                finding.description,
                finding.suggestion,
                finding.confidence,
                "Local",
                finding.line_number,
                finding.class_or_method,
            )
            for finding in findings
        ]
        with self.session.get_connection() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO findings
                (session_id, file_path, agent_name, severity, category, title,
                description, suggestion, confidence, impact_scope, line_number, class_or_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )