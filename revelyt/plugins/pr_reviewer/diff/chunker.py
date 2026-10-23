"""
diff/chunker.py — Intelligent diff chunking for LLM context windows.

Splits large diffs into manageable chunks that respect logical boundaries
(functions, classes) rather than cutting mid-statement, and adds surrounding
context lines so the AI can understand the code properly.
"""

from __future__ import annotations

import re
from typing import List

from revelyt.plugins.pr_reviewer.engines.base import FileChunk


# Default limits — tuned for typical LLM context windows
DEFAULT_MAX_CHUNK_LINES = 300
DEFAULT_CONTEXT_LINES = 15


class DiffChunker:
    """
    Breaks large FileChunks into smaller, LLM-friendly pieces that
    respect logical boundaries and carry surrounding context.

    Parameters
    ----------
    max_chunk_lines : int
        Maximum number of diff lines per chunk.
    context_lines : int
        Number of surrounding context lines to include.
    """

    def __init__(
        self,
        max_chunk_lines: int = DEFAULT_MAX_CHUNK_LINES,
        context_lines: int = DEFAULT_CONTEXT_LINES,
    ):
        self.max_chunk_lines = max_chunk_lines
        self.context_lines = context_lines

    def chunk(self, chunks: List[FileChunk]) -> List[FileChunk]:
        """
        Takes a list of FileChunks (one per hunk) and re-splits any
        oversized ones into smaller logical pieces.
        """
        result: List[FileChunk] = []
        for chunk in chunks:
            if chunk.total_changed_lines <= self.max_chunk_lines:
                result.append(chunk)
            else:
                result.extend(self._split_chunk(chunk))
        return result

    def _split_chunk(self, chunk: FileChunk) -> List[FileChunk]:
        """Split a large chunk at logical boundaries."""
        diff_lines = chunk.diff_content.splitlines()
        if not diff_lines:
            return [chunk]

        # Try to split at function/class boundaries
        boundaries = self._find_logical_boundaries(diff_lines, chunk.language)

        if not boundaries:
            # Fall back to fixed-size splitting
            return self._split_fixed(chunk, diff_lines)

        return self._split_at_boundaries(chunk, diff_lines, boundaries)

    def _find_logical_boundaries(self, lines: List[str], language: str) -> List[int]:
        """Find line indices that are good split points (function/class defs)."""
        if language == "python":
            pattern = re.compile(r"^[+-]?\s*(?:class |def |async def )")
        elif language in ("javascript", "typescript"):
            pattern = re.compile(r"^[+-]?\s*(?:function |class |const \w+ = (?:async )?(?:function|\())")
        elif language == "go":
            pattern = re.compile(r"^[+-]?\s*func ")
        elif language == "java":
            pattern = re.compile(r"^[+-]?\s*(?:public |private |protected |static |void |class )")
        else:
            pattern = re.compile(r"^[+-]?\s*(?:def |func |function |class )")

        boundaries: List[int] = []
        for i, line in enumerate(lines):
            if pattern.match(line) and i > 0:
                boundaries.append(i)
        return boundaries

    def _split_at_boundaries(
        self, original: FileChunk, lines: List[str], boundaries: List[int]
    ) -> List[FileChunk]:
        """Split diff at logical boundaries, creating new FileChunks."""
        result: List[FileChunk] = []
        boundaries = [0] + boundaries + [len(lines)]

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            segment = lines[start_idx:end_idx]

            if not segment:
                continue

            added = [l[1:] for l in segment if l.startswith("+")]
            deleted = [l[1:] for l in segment if l.startswith("-")]
            context = [l[1:] if l.startswith(" ") else l for l in segment if l.startswith(" ")]

            result.append(
                FileChunk(
                    file_path=original.file_path,
                    extension=original.extension,
                    language=original.language,
                    start_line=original.start_line + start_idx,
                    end_line=original.start_line + end_idx - 1,
                    content="\n".join(l.lstrip("+-  ") for l in segment),
                    diff_content="\n".join(segment),
                    added_lines=added,
                    deleted_lines=deleted,
                    context_lines=context,
                    change_type=original.change_type,
                    risk_score=original.risk_score,
                )
            )

        return result if result else [original]

    def _split_fixed(self, original: FileChunk, lines: List[str]) -> List[FileChunk]:
        """Fixed-size splitting as a fallback when no logical boundaries exist."""
        result: List[FileChunk] = []

        for i in range(0, len(lines), self.max_chunk_lines):
            segment = lines[i : i + self.max_chunk_lines]

            added = [l[1:] for l in segment if l.startswith("+")]
            deleted = [l[1:] for l in segment if l.startswith("-")]
            context = [l[1:] if l.startswith(" ") else l for l in segment if l.startswith(" ")]

            result.append(
                FileChunk(
                    file_path=original.file_path,
                    extension=original.extension,
                    language=original.language,
                    start_line=original.start_line + i,
                    end_line=original.start_line + i + len(segment) - 1,
                    content="\n".join(l.lstrip("+-  ") for l in segment),
                    diff_content="\n".join(segment),
                    added_lines=added,
                    deleted_lines=deleted,
                    context_lines=context,
                    change_type=original.change_type,
                    risk_score=original.risk_score,
                )
            )

        return result if result else [original]
