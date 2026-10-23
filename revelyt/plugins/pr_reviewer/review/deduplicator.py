"""
review/deduplicator.py — Intelligent deduplication of review findings.

Prevents spamming PRs with duplicate comments by using fingerprinting
to detect same file+line+rule combinations across engines and commits.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Set

from revelyt.plugins.pr_reviewer.engines.base import Finding

logger = logging.getLogger(__name__)


class Deduplicator:
    """
    Deduplicates findings to prevent posting the same comment multiple
    times across engines, commits, or re-runs.

    Uses Finding.fingerprint() for hash-based dedup and also consolidates
    findings that target the same file+line.
    """

    def __init__(self):
        self._seen_fingerprints: Set[str] = set()

    def deduplicate(self, findings: List[Finding]) -> List[Finding]:
        """Remove duplicate findings based on fingerprint hashing."""
        unique: List[Finding] = []

        for finding in findings:
            fp = finding.fingerprint()
            if fp not in self._seen_fingerprints:
                self._seen_fingerprints.add(fp)
                unique.append(finding)

        dropped = len(findings) - len(unique)
        if dropped > 0:
            logger.info("Deduplicated: removed %d duplicate findings", dropped)
        return unique

    def deduplicate_with_existing(
        self, findings: List[Finding], existing_comments: List[str]
    ) -> List[Finding]:
        """
        Also filter out findings that match already-posted PR comments.

        Parameters
        ----------
        existing_comments : list of str
            Bodies of existing PR comments (fetched from GitHub API).
        """
        # Build a set of existing comment signatures
        existing_sigs: Set[str] = set()
        for comment in existing_comments:
            # Extract rule_id from comment footer if present
            if ":" in comment:
                # Look for `engine:rule_id` pattern at the end
                for line in comment.splitlines():
                    stripped = line.strip().strip("`")
                    if ":" in stripped and len(stripped) < 60:
                        existing_sigs.add(stripped)

        fresh: List[Finding] = []
        for finding in findings:
            sig = f"{finding.engine}:{finding.rule_id}"
            # Also check for the specific message in existing comments
            message_exists = any(finding.message[:50] in comment for comment in existing_comments)
            if sig not in existing_sigs and not message_exists:
                fresh.append(finding)

        skipped = len(findings) - len(fresh)
        if skipped > 0:
            logger.info("Skipped %d findings already posted as comments", skipped)
        return fresh

    def consolidate_same_line(self, findings: List[Finding]) -> List[Finding]:
        """
        Merge multiple findings on the same file+line into a single
        finding to avoid comment clutter.
        """
        by_location: Dict[str, List[Finding]] = {}
        for f in findings:
            key = f"{f.file}:{f.line}"
            by_location.setdefault(key, []).append(f)

        result: List[Finding] = []
        for key, group in by_location.items():
            if len(group) == 1:
                result.append(group[0])
            else:
                # Keep the highest-severity finding, merge messages
                group.sort(key=lambda f: f.confidence, reverse=True)
                primary = group[0]

                # Append other messages
                extra_messages = [f"• [{f.rule_id}] {f.message}" for f in group[1:]]
                primary.message = primary.message + "\n\n**Additional issues on this line:**\n" + "\n".join(extra_messages)
                primary.metadata["consolidated_count"] = len(group)
                result.append(primary)

        return result

    def reset(self) -> None:
        """Clear the fingerprint cache (e.g., for a new PR)."""
        self._seen_fingerprints.clear()
