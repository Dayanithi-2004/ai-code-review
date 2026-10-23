"""
pipeline/review_pipeline.py — Core review pipeline orchestrator.

Sequences the full review flow:
  Diff Parse → Filter → Chunk → Semantic Analysis → Impact Score →
  Context Build → Engine Execution → Dedup → Format → Post

This is the heart of Revelyt's PR review system.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from revelyt.plugins.pr_reviewer.engines.base import FileChunk, Finding, PRContext
from revelyt.plugins.pr_reviewer.diff.parser import DiffParser
from revelyt.plugins.pr_reviewer.diff.semantic_diff import SemanticDiffAnalyzer
from revelyt.plugins.pr_reviewer.diff.chunker import DiffChunker
from revelyt.plugins.pr_reviewer.diff.impact_analyzer import ImpactAnalyzer
from revelyt.plugins.pr_reviewer.diff.test_impact import TestImpactAnalyzer
from revelyt.plugins.pr_reviewer.diff.change_graph import ChangeGraph
from revelyt.plugins.pr_reviewer.filters.file_filter import FileFilter
from revelyt.plugins.pr_reviewer.filters.limits import PRLimitsGuard, PRStats, ReviewMode
from revelyt.plugins.pr_reviewer.engines.hybrid_engine import HybridEngine
from revelyt.plugins.pr_reviewer.engines.risk_engine import RiskEngine, RiskReport
from revelyt.plugins.pr_reviewer.engines.ai_request import AIRequestClient
from revelyt.plugins.pr_reviewer.context.retrieval import ContextRetrieval
from revelyt.plugins.pr_reviewer.cache.diff_cache import DiffCache
from revelyt.plugins.pr_reviewer.cache.ai_result_cache import AIResultCache
from revelyt.plugins.pr_reviewer.learning.module_risk_db import ModuleRiskDB
from revelyt.plugins.pr_reviewer.review.deduplicator import Deduplicator

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete output of a pipeline run."""
    findings: List[Finding] = field(default_factory=list)
    risk_report: Optional[RiskReport] = None
    stats: Optional[PRStats] = None
    review_mode: ReviewMode = ReviewMode.DEEP
    execution_time_ms: float = 0.0
    chunks_analyzed: int = 0
    chunks_from_cache: int = 0
    ai_enabled: bool = True
    metrics: Dict[str, Any] = field(default_factory=dict)


class ReviewPipeline:
    """
    Executes the full PR review pipeline.

    Stages:
        1. Parse    — Convert raw diff into FileChunks
        2. Filter   — Drop binaries, classify scan tiers
        3. Limit    — Check PR size, set review mode
        4. Chunk    — Split large diffs at logical boundaries
        5. Enrich   — Semantic analysis, impact scoring, learning weights
        6. Context  — Build RAG context for AI
        7. Analyze  — Run hybrid engine (Rules + AI)
        8. Score    — Calculate PR risk
        9. Dedup    — Remove duplicate findings
    """

    def __init__(
        self,
        ai_client: Optional[AIRequestClient] = None,
        enable_ai: bool = True,
        review_mode: ReviewMode = ReviewMode.DEEP,
        repo_root: str = ".",
    ):
        # Stage components
        self.parser = DiffParser()
        self.semantic = SemanticDiffAnalyzer()
        self.chunker = DiffChunker()
        self.impact = ImpactAnalyzer()
        self.test_impact = TestImpactAnalyzer()
        self.change_graph = ChangeGraph()
        self.file_filter = FileFilter()
        self.limits_guard = PRLimitsGuard()
        self.risk_engine = RiskEngine()
        self.deduplicator = Deduplicator()
        self.context_retrieval = ContextRetrieval(repo_root=repo_root)

        # Caches
        self.diff_cache = DiffCache()
        self.ai_cache = AIResultCache()

        # Learning
        self.module_risk_db = ModuleRiskDB()

        # Engine
        self.hybrid_engine = HybridEngine(
            ai_client=ai_client,
            enable_ai=enable_ai,
        )

        self._requested_mode = review_mode

    def execute(self, raw_diff: str, context: PRContext) -> PipelineResult:
        """
        Execute the full pipeline on a raw diff string.

        Parameters
        ----------
        raw_diff : str
            The unified diff from GitHub.
        context : PRContext
            PR metadata (title, author, branches, etc.).

        Returns
        -------
        PipelineResult
        """
        start = time.time()
        result = PipelineResult(ai_enabled=self.hybrid_engine.enable_ai)

        # ── Stage 1: Parse ────────────────────────────────────────────
        logger.info("Stage 1: Parsing diff")
        grouped = self.parser.parse(raw_diff)
        all_chunks = [c for chunks in grouped.values() for c in chunks]
        logger.info("Parsed %d chunks across %d file types", len(all_chunks), len(grouped))

        if not all_chunks:
            logger.warning("No parseable diff content found")
            result.execution_time_ms = (time.time() - start) * 1000
            return result

        # ── Stage 2: Filter ───────────────────────────────────────────
        logger.info("Stage 2: Filtering files")
        filtered = self.file_filter.filter(all_chunks)
        deep_chunks, light_chunks = self.file_filter.partition(filtered)
        logger.info("After filter: %d deep, %d light (dropped %d)", len(deep_chunks), len(light_chunks), len(all_chunks) - len(filtered))

        # ── Stage 3: Limits ───────────────────────────────────────────
        logger.info("Stage 3: Checking PR limits")
        stats = PRStats.from_chunks(filtered)
        result.stats = stats
        effective_mode = self.limits_guard.determine_mode(stats, self._requested_mode)
        result.review_mode = effective_mode
        logger.info("Review mode: %s (requested: %s)", effective_mode.value, self._requested_mode.value)

        # ── Stage 4: Chunk ────────────────────────────────────────────
        logger.info("Stage 4: Chunking large diffs")
        chunked = self.chunker.chunk(deep_chunks)
        result.chunks_analyzed = len(chunked)
        logger.info("Chunked: %d → %d chunks", len(deep_chunks), len(chunked))

        # ── Stage 5: Enrich ───────────────────────────────────────────
        logger.info("Stage 5: Enriching with semantic analysis")

        # 5a. Semantic analysis (complexity, affected functions)
        chunked = self.semantic.analyze(chunked)

        # 5b. Inject historical bug weights from learning DB
        for chunk in chunked:
            chunk.historical_bug_weight = self.module_risk_db.get_bug_weight(chunk.file_path)

        # 5c. Impact scoring
        chunked = self.impact.analyze(chunked)
        chunked = self.impact.sort_by_risk(chunked)

        # ── Stage 6: Context ──────────────────────────────────────────
        logger.info("Stage 6: Building code context (RAG)")
        context = self.context_retrieval.build_context(chunked, context)

        # ── Stage 7: Analyze ──────────────────────────────────────────
        logger.info("Stage 7: Running hybrid engine")
        findings = self.hybrid_engine.run(chunked, context)

        # Also run test impact analysis
        test_findings = self.test_impact.analyze(chunked)
        findings.extend(test_findings)

        # Also check change graph for ripple effects
        if effective_mode == ReviewMode.DEEP:
            graph_findings = self.change_graph.generate_findings(chunked)
            findings.extend(graph_findings)

        logger.info("Total raw findings: %d", len(findings))

        # ── Stage 8: Score ────────────────────────────────────────────
        logger.info("Stage 8: Calculating risk score")
        risk_report = self.risk_engine.calculate(chunked, findings)
        result.risk_report = risk_report
        logger.info("Risk score: %.1f (%s)", risk_report.score, risk_report.level)

        # ── Stage 9: Dedup ────────────────────────────────────────────
        logger.info("Stage 9: Deduplicating findings")
        findings = self.deduplicator.deduplicate(findings)
        findings = self.deduplicator.consolidate_same_line(findings)
        result.findings = findings
        logger.info("Final findings after dedup: %d", len(findings))

        # ── Record learning data ──────────────────────────────────────
        for chunk in chunked:
            file_findings = [f for f in findings if f.file == chunk.file_path]
            if file_findings:
                self.module_risk_db.track_issues(
                    chunk.file_path, len(file_findings), context.pr_number
                )

        # ── Metrics ───────────────────────────────────────────────────
        elapsed = (time.time() - start) * 1000
        result.execution_time_ms = elapsed
        result.metrics = {
            "parse_chunks": len(all_chunks),
            "filtered_chunks": len(filtered),
            "analyzed_chunks": len(chunked),
            "findings_before_dedup": len(findings),
            "findings_final": len(result.findings),
            "risk_score": risk_report.score,
            "execution_time_ms": round(elapsed, 1),
        }

        logger.info("Pipeline complete in %.0fms", elapsed)
        return result
