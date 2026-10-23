"""
filters/limits.py — PR size protection and mode determination.

Prevents the system from choking on massive PRs by enforcing
hard limits and automatically downgrading to fast mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from revelyt.plugins.pr_reviewer.engines.base import FileChunk

logger = logging.getLogger(__name__)


class ReviewMode(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"


@dataclass
class PRStats:
    """Aggregate statistics about a PR's diff."""
    total_files: int = 0
    total_lines_added: int = 0
    total_lines_deleted: int = 0
    total_lines: int = 0
    total_chunks: int = 0
    largest_file_lines: int = 0
    languages: set = field(default_factory=set)

    @classmethod
    def from_chunks(cls, chunks: List[FileChunk]) -> "PRStats":
        stats = cls()
        files_seen = set()
        file_sizes: dict[str, int] = {}

        for chunk in chunks:
            files_seen.add(chunk.file_path)
            stats.total_lines_added += len(chunk.added_lines)
            stats.total_lines_deleted += len(chunk.deleted_lines)
            stats.total_chunks += 1
            stats.languages.add(chunk.language)

            file_sizes[chunk.file_path] = file_sizes.get(chunk.file_path, 0) + chunk.total_changed_lines

        stats.total_files = len(files_seen)
        stats.total_lines = stats.total_lines_added + stats.total_lines_deleted
        stats.largest_file_lines = max(file_sizes.values()) if file_sizes else 0
        return stats


# Limits
_MAX_FILES_DEEP = 100
_MAX_FILES_BALANCED = 200
_MAX_FILES_ABSOLUTE = 500

_MAX_LINES_DEEP = 5000
_MAX_LINES_BALANCED = 10000
_MAX_LINES_ABSOLUTE = 25000

_MAX_AI_CHUNKS = 150


class PRLimitsGuard:
    """
    Evaluates PR size against hard limits and determines the
    appropriate review mode.

    If a user requests ``--mode deep`` but the PR exceeds deep-mode
    thresholds, the guard downgrades automatically.
    """

    def determine_mode(
        self,
        stats: PRStats,
        requested_mode: ReviewMode = ReviewMode.DEEP,
    ) -> ReviewMode:
        """
        Returns the effective review mode after applying limits.

        Parameters
        ----------
        stats : PRStats
            Aggregate PR statistics.
        requested_mode : ReviewMode
            The mode the user originally requested.
        """
        # Absolute hard limits — refuse or force fast
        if stats.total_files > _MAX_FILES_ABSOLUTE or stats.total_lines > _MAX_LINES_ABSOLUTE:
            logger.warning(
                "PR exceeds absolute limits (%d files, %d lines). Forcing FAST mode.",
                stats.total_files, stats.total_lines,
            )
            return ReviewMode.FAST

        if requested_mode == ReviewMode.DEEP:
            if stats.total_files > _MAX_FILES_DEEP or stats.total_lines > _MAX_LINES_DEEP:
                logger.info("PR too large for DEEP mode. Downgrading to BALANCED.")
                return ReviewMode.BALANCED

        if requested_mode in (ReviewMode.DEEP, ReviewMode.BALANCED):
            if stats.total_files > _MAX_FILES_BALANCED or stats.total_lines > _MAX_LINES_BALANCED:
                logger.info("PR too large for BALANCED mode. Downgrading to FAST.")
                return ReviewMode.FAST

        return requested_mode

    def should_skip_ai(self, stats: PRStats) -> bool:
        """Return True if PR is too large for AI analysis."""
        return stats.total_chunks > _MAX_AI_CHUNKS

    def get_ai_chunk_budget(self, stats: PRStats, mode: ReviewMode) -> int:
        """Max number of chunks to send to AI based on mode."""
        if mode == ReviewMode.FAST:
            return min(stats.total_chunks, 20)
        elif mode == ReviewMode.BALANCED:
            return min(stats.total_chunks, 80)
        else:
            return min(stats.total_chunks, _MAX_AI_CHUNKS)
