"""
engines/hybrid_engine.py — Orchestrates the Rule → AI validation pipeline.

Runs deterministic engines first (rules, secrets, dependencies), then
sends remaining high-risk chunks to the AI engine. Combines all findings,
deduplicates, and applies the patch validator to any AI-suggested fixes.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from revelyt.plugins.pr_reviewer.engines.base import (
    Engine,
    FileChunk,
    Finding,
    PRContext,
)
from revelyt.plugins.pr_reviewer.engines.rule_engine import RuleEngine
from revelyt.plugins.pr_reviewer.engines.sensitive_data import SensitiveDataEngine
from revelyt.plugins.pr_reviewer.engines.dependency_scanner import DependencyScanner
from revelyt.plugins.pr_reviewer.engines.ai_request import AIRequestClient
from revelyt.plugins.pr_reviewer.engines.ai_validation import AIValidationEngine
from revelyt.plugins.pr_reviewer.engines.patch_validator import PatchValidator

logger = logging.getLogger(__name__)


class HybridEngine(Engine):
    """
    The hybrid engine coordinates the full analysis pipeline:

    1. **Deterministic pass** (instant, zero-cost):
       - Rule engine (regex + AST)
       - Sensitive data scanner
       - Dependency scanner

    2. **AI pass** (on high-risk chunks only):
       - AI validation engine (with two-pass self-consistency)
       - Patch validation for any suggested fixes

    This approach minimises AI costs while maximising detection coverage.
    """

    name = "hybrid"

    def __init__(
        self,
        ai_client: Optional[AIRequestClient] = None,
        enable_ai: bool = True,
        ai_risk_threshold: float = 2.0,
    ):
        """
        Parameters
        ----------
        ai_client : AIRequestClient, optional
            LLM client. If None and enable_ai is True, a default is created.
        enable_ai : bool
            Set False for --no-ai mode (deterministic rules only).
        ai_risk_threshold : float
            Minimum risk_score for a chunk to be sent to AI analysis.
        """
        self.enable_ai = enable_ai
        self.ai_risk_threshold = ai_risk_threshold

        # Deterministic engines (always run)
        self.rule_engine = RuleEngine()
        self.sensitive_engine = SensitiveDataEngine()
        self.dependency_engine = DependencyScanner()
        self.patch_validator = PatchValidator()

        # AI engines (optional)
        self.ai_client = ai_client
        self.ai_validation: Optional[AIValidationEngine] = None
        if enable_ai and ai_client:
            self.ai_validation = AIValidationEngine(ai_client=ai_client)

    def run(self, chunks: List[FileChunk], context: PRContext) -> List[Finding]:
        """Execute the full hybrid pipeline."""
        all_findings: List[Finding] = []

        # ----------------------------------------------------------
        # Phase 1: Deterministic analysis (ALL chunks)
        # ----------------------------------------------------------
        logger.info("Phase 1: Running deterministic engines on %d chunks", len(chunks))

        rule_findings = self.rule_engine.run(chunks, context)
        secret_findings = self.sensitive_engine.run(chunks, context)
        dep_findings = self.dependency_engine.run(chunks, context)

        all_findings.extend(rule_findings)
        all_findings.extend(secret_findings)
        all_findings.extend(dep_findings)

        logger.info(
            "Deterministic results: %d rule, %d secret, %d dependency findings",
            len(rule_findings), len(secret_findings), len(dep_findings),
        )

        # ----------------------------------------------------------
        # Phase 2: AI analysis (high-risk chunks only)
        # ----------------------------------------------------------
        if self.enable_ai and self.ai_validation:
            # Filter to high-risk chunks
            ai_chunks = [c for c in chunks if c.risk_score >= self.ai_risk_threshold]

            if ai_chunks:
                logger.info(
                    "Phase 2: Running AI analysis on %d/%d high-risk chunks (threshold=%.1f)",
                    len(ai_chunks), len(chunks), self.ai_risk_threshold,
                )
                ai_findings = self.ai_validation.run(ai_chunks, context)

                # Validate any suggested patches
                for finding in ai_findings:
                    if finding.suggested_patch:
                        result = self.patch_validator.validate(
                            "", finding.suggested_patch, "python"
                        )
                        if not result.is_valid:
                            logger.warning(
                                "AI autofix rejected for %s:%d — %s",
                                finding.file, finding.line, result.error,
                            )
                            finding.suggested_patch = None
                            finding.metadata["patch_rejected"] = result.error

                all_findings.extend(ai_findings)
                logger.info("AI results: %d findings", len(ai_findings))
            else:
                logger.info("Phase 2: No chunks above AI risk threshold. Skipping AI.")
        elif not self.enable_ai:
            logger.info("Phase 2: AI disabled (--no-ai mode). Skipping.")

        return all_findings
