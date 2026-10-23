"""
engines/ai_validation.py — Two-pass self-consistency and confidence scoring.

Validates AI findings by:
  1. Running the AI twice and comparing results (self-consistency)
  2. Applying safety filters to strip uncertain language
  3. Computing risk-weighted confidence scores
  4. Rejecting findings below the confidence threshold
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from revelyt.plugins.pr_reviewer.engines.base import (
    Engine,
    FileChunk,
    Finding,
    FindingType,
    PRContext,
    Severity,
)
from revelyt.plugins.pr_reviewer.engines.ai_request import AIRequestClient

logger = logging.getLogger(__name__)

# Confidence threshold — findings below this are dropped
CONFIDENCE_THRESHOLD = 0.6

# Phrases that indicate AI uncertainty (hallucination markers)
_UNCERTAIN_PHRASES = [
    "might be", "could be", "possibly", "perhaps", "maybe",
    "not sure", "it seems", "appears to", "likely", "unlikely",
    "I think", "I believe", "may cause", "potential", "potentially",
]

# Severity string to enum mapping
_SEVERITY_MAP = {
    "low": Severity.LOW,
    "medium": Severity.MEDIUM,
    "high": Severity.HIGH,
    "critical": Severity.CRITICAL,
    "informational": Severity.INFORMATIONAL,
}

# FindingType string to enum mapping
_TYPE_MAP = {
    "security": FindingType.SECURITY,
    "bug": FindingType.BUG,
    "performance": FindingType.PERFORMANCE,
    "quality": FindingType.QUALITY,
    "style": FindingType.STYLE,
}


class AIValidationEngine(Engine):
    """
    Two-pass AI validation engine.

    Pass 1: Generate initial findings from the AI.
    Pass 2: Re-query the AI to verify findings (self-consistency check).

    Findings that appear in both passes with similar messages get
    a confidence boost. Findings that only appear once get penalised.
    """

    name = "ai_validation"

    def __init__(
        self,
        ai_client: AIRequestClient,
        two_pass: bool = True,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.ai_client = ai_client
        self.two_pass = two_pass
        self.confidence_threshold = confidence_threshold

    def run(self, chunks: List[FileChunk], context: PRContext) -> List[Finding]:
        """Run AI analysis with optional two-pass validation."""
        all_findings: List[Finding] = []

        for chunk in chunks:
            # Skip very small or low-risk chunks to save tokens
            if chunk.total_changed_lines < 3 and chunk.risk_score < 2.0:
                continue

            findings = self._analyze_chunk(chunk, context)
            all_findings.extend(findings)

        return all_findings

    def _analyze_chunk(self, chunk: FileChunk, context: PRContext) -> List[Finding]:
        """Analyze a single chunk with two-pass validation."""
        # Pass 1
        pass1_raw = self.ai_client.request_review(chunk, context)
        pass1_findings = self._parse_findings(pass1_raw, chunk)

        if not self.two_pass or not pass1_findings:
            return self._filter_by_confidence(pass1_findings)

        # Pass 2 — Re-query for consistency check
        pass2_raw = self.ai_client.request_review(chunk, context)
        pass2_findings = self._parse_findings(pass2_raw, chunk)

        # Cross-validate
        validated = self._cross_validate(pass1_findings, pass2_findings)
        return self._filter_by_confidence(validated)

    def _parse_findings(self, raw: Dict[str, Any], chunk: FileChunk) -> List[Finding]:
        """Convert raw AI JSON response into Finding objects."""
        findings: List[Finding] = []
        issues = raw.get("issues", [])

        for issue in issues:
            if not isinstance(issue, dict):
                continue

            severity_str = str(issue.get("severity", "medium")).lower()
            type_str = str(issue.get("type", "quality")).lower()
            message = str(issue.get("message", ""))
            suggestion = issue.get("suggestion", "")
            confidence = float(issue.get("confidence", 0.7))
            line = int(issue.get("line", chunk.start_line))

            # Apply safety filters
            confidence = self._apply_safety_filters(message, confidence)
            confidence = self._apply_safety_filters(suggestion, confidence)

            # Apply risk-weighted adjustment
            confidence = self._risk_weighted_confidence(confidence, chunk)

            findings.append(
                Finding(
                    file=chunk.file_path,
                    line=line,
                    severity=_SEVERITY_MAP.get(severity_str, Severity.MEDIUM),
                    type=_TYPE_MAP.get(type_str, FindingType.QUALITY),
                    message=message,
                    suggestion=suggestion if suggestion else None,
                    confidence=min(confidence, 1.0),
                    rule_id=f"ai-{type_str}-{severity_str}",
                    engine=self.name,
                )
            )

        return findings

    def _apply_safety_filters(self, text: str, confidence: float) -> float:
        """Penalise confidence if the AI's language is uncertain."""
        if not text:
            return confidence

        text_lower = text.lower()
        uncertainty_count = sum(1 for phrase in _UNCERTAIN_PHRASES if phrase in text_lower)

        # Each uncertain phrase reduces confidence by 10%
        penalty = uncertainty_count * 0.10
        return max(confidence - penalty, 0.0)

    def _risk_weighted_confidence(self, confidence: float, chunk: FileChunk) -> float:
        """
        Combine AI confidence with chunk metadata to compute
        a risk-weighted score.

        High-risk chunks with high complexity get a slight boost;
        low-risk chunks get a slight penalty (to bias toward precision).
        """
        if chunk.risk_score >= 5.0:
            confidence += 0.05  # Boost: high-risk code deserves attention
        elif chunk.risk_score <= 1.0:
            confidence -= 0.05  # Penalise: findings in low-risk code need more proof

        if chunk.complexity >= 10:
            confidence += 0.03  # Complex code is more likely to have bugs

        return max(min(confidence, 1.0), 0.0)

    def _cross_validate(
        self, pass1: List[Finding], pass2: List[Finding]
    ) -> List[Finding]:
        """
        Cross-validate findings from two AI passes.

        Findings that appear in both passes get a confidence boost.
        Findings that only appear once get a confidence penalty.
        """
        validated: List[Finding] = []

        # Build lookup for pass2 by (file, approximate_line, type)
        pass2_keys = set()
        for f in pass2:
            key = (f.file, f.line // 5, f.type)  # Line bucket for fuzzy matching
            pass2_keys.add(key)

        for finding in pass1:
            key = (finding.file, finding.line // 5, finding.type)
            if key in pass2_keys:
                # Confirmed by both passes — boost confidence
                finding.confidence = min(finding.confidence + 0.15, 1.0)
                finding.metadata["validated"] = "two-pass-confirmed"
            else:
                # Only in pass 1 — penalise confidence
                finding.confidence = finding.confidence * 0.7
                finding.metadata["validated"] = "single-pass-only"

            validated.append(finding)

        return validated

    def _filter_by_confidence(self, findings: List[Finding]) -> List[Finding]:
        """Drop findings below the confidence threshold."""
        kept = [f for f in findings if f.confidence >= self.confidence_threshold]
        dropped = len(findings) - len(kept)
        if dropped > 0:
            logger.info("Dropped %d low-confidence AI findings (threshold=%.2f)", dropped, self.confidence_threshold)
        return kept
