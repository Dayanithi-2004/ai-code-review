"""
diff/parser.py — Production-grade unified diff parser.

Parses raw unified diff output (from GitHub API or `git diff`) into
a structured ``dict[str, list[FileChunk]]`` grouped by file extension.
Handles binary files, renames, and multi-hunk diffs.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Dict, List, Optional

from revelyt.plugins.pr_reviewer.engines.base import ChangeType, FileChunk

# Regex patterns for unified diff parsing
_FILE_HEADER = re.compile(r"^diff --git a/(.*?) b/(.*?)$")
_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")
_RENAME_FROM = re.compile(r"^rename from (.*)$")
_RENAME_TO = re.compile(r"^rename to (.*)$")
_BINARY_MARKER = re.compile(r"^Binary files .* differ$")

# Map file extensions to language identifiers
EXTENSION_LANGUAGE_MAP: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".sql": "sql",
    ".sh": "shell",
    ".dockerfile": "dockerfile",
}


class DiffParser:
    """
    Parses raw unified diff text into structured FileChunk objects
    grouped by file extension.

    Usage::

        parser = DiffParser()
        grouped = parser.parse(raw_diff_string)
        # grouped == {".py": [FileChunk(...), ...], ".js": [...]}
    """

    def parse(self, raw_diff: str) -> Dict[str, List[FileChunk]]:
        """
        Parse a full unified diff string and return chunks grouped by extension.

        Parameters
        ----------
        raw_diff : str
            The complete unified diff output.

        Returns
        -------
        dict mapping file extension to a list of ``FileChunk`` objects.
        """
        if not raw_diff or not raw_diff.strip():
            return {}

        files = self._split_into_file_diffs(raw_diff)
        grouped: Dict[str, List[FileChunk]] = defaultdict(list)

        for file_diff in files:
            chunks = self._parse_file_diff(file_diff)
            for chunk in chunks:
                grouped[chunk.extension].append(chunk)

        return dict(grouped)

    def parse_flat(self, raw_diff: str) -> List[FileChunk]:
        """Return a flat list of all FileChunks (not grouped)."""
        grouped = self.parse(raw_diff)
        return [chunk for chunks in grouped.values() for chunk in chunks]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_into_file_diffs(self, raw_diff: str) -> List[str]:
        """Split a full diff into per-file diff blocks."""
        parts: List[str] = []
        current: List[str] = []

        for line in raw_diff.splitlines(keepends=True):
            if _FILE_HEADER.match(line) and current:
                parts.append("".join(current))
                current = []
            current.append(line)

        if current:
            parts.append("".join(current))
        return parts

    def _parse_file_diff(self, file_diff: str) -> List[FileChunk]:
        """Parse a single file's diff into one or more FileChunks (one per hunk)."""
        lines = file_diff.splitlines()
        if not lines:
            return []

        # Determine file path
        file_path = self._extract_file_path(lines)
        if not file_path:
            return []

        # Skip binary files
        if any(_BINARY_MARKER.match(l) for l in lines):
            return []

        # Detect renames
        change_type = ChangeType.MODIFIED
        for line in lines:
            if _RENAME_FROM.match(line):
                change_type = ChangeType.RENAMED
                break
            if line.startswith("new file"):
                change_type = ChangeType.ADDED
                break
            if line.startswith("deleted file"):
                change_type = ChangeType.DELETED
                break

        ext = os.path.splitext(file_path)[1].lower()
        language = EXTENSION_LANGUAGE_MAP.get(ext, "unknown")

        # Parse hunks
        hunks = self._extract_hunks(lines)
        if not hunks:
            # File-level change with no diff hunks (e.g., mode change)
            return [
                FileChunk(
                    file_path=file_path,
                    extension=ext,
                    language=language,
                    diff_content=file_diff,
                    change_type=change_type,
                )
            ]

        chunks: List[FileChunk] = []
        for hunk in hunks:
            added = [l[1:] for l in hunk["lines"] if l.startswith("+")]
            deleted = [l[1:] for l in hunk["lines"] if l.startswith("-")]
            context = [l[1:] if l.startswith(" ") else l for l in hunk["lines"] if l.startswith(" ")]

            chunks.append(
                FileChunk(
                    file_path=file_path,
                    extension=ext,
                    language=language,
                    start_line=hunk["new_start"],
                    end_line=hunk["new_start"] + hunk["new_count"] - 1,
                    content="\n".join(l.lstrip("+-  ") for l in hunk["lines"]),
                    diff_content="\n".join(hunk["lines"]),
                    added_lines=added,
                    deleted_lines=deleted,
                    context_lines=context,
                    change_type=change_type,
                )
            )

        return chunks

    def _extract_file_path(self, lines: List[str]) -> Optional[str]:
        """Extract the target file path from diff headers."""
        for line in lines:
            m = _FILE_HEADER.match(line)
            if m:
                return m.group(2)  # "b/..." path
        return None

    def _extract_hunks(self, lines: List[str]) -> List[dict]:
        """Extract individual hunks with their line ranges and content."""
        hunks: List[dict] = []
        current_hunk: Optional[dict] = None

        for line in lines:
            hunk_match = _HUNK_HEADER.match(line)
            if hunk_match:
                if current_hunk:
                    hunks.append(current_hunk)
                current_hunk = {
                    "old_start": int(hunk_match.group(1)),
                    "old_count": int(hunk_match.group(2) or 1),
                    "new_start": int(hunk_match.group(3)),
                    "new_count": int(hunk_match.group(4) or 1),
                    "header": hunk_match.group(5).strip(),
                    "lines": [],
                }
            elif current_hunk is not None:
                if line.startswith(("+", "-", " ")):
                    current_hunk["lines"].append(line)

        if current_hunk:
            hunks.append(current_hunk)

        return hunks
