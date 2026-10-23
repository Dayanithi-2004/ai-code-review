"""
context/retrieval.py — Context Retrieval Engine (RAG for Code).

Assembles rich context for AI prompts by searching the repository
for definitions, callers, related tests, and cross-branch history.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set

from revelyt.plugins.pr_reviewer.engines.base import FileChunk, PRContext

logger = logging.getLogger(__name__)


class ContextRetrieval:
    """
    Builds deep code context for AI prompts by:
      1. Finding definitions of modified functions/classes
      2. Finding callers of modified functions
      3. Correlating source files to test files
      4. Extracting import chains

    This provides the AI with a rich understanding of the codebase,
    dramatically reducing hallucinations.
    """

    def __init__(self, repo_root: str = "."):
        self.repo_root = repo_root
        self._file_cache: Dict[str, str] = {}

    def build_context(self, chunks: List[FileChunk], base_context: PRContext) -> PRContext:
        """
        Enrich the PRContext with symbol definitions, caller maps,
        and related test files for all modified chunks.
        """
        modified_files = list({c.file_path for c in chunks})
        all_functions = set()
        all_classes = set()

        for chunk in chunks:
            all_functions.update(chunk.affected_functions)
            all_classes.update(chunk.affected_classes)

        # 1. Find definitions
        for func_name in all_functions:
            definition = self._find_definition(func_name)
            if definition:
                base_context.symbol_definitions[func_name] = definition

        for cls_name in all_classes:
            definition = self._find_definition(cls_name)
            if definition:
                base_context.symbol_definitions[cls_name] = definition

        # 2. Find callers
        for func_name in all_functions:
            callers = self._find_callers(func_name, exclude_files=set(modified_files))
            if callers:
                base_context.caller_map[func_name] = callers

        # 3. Find related tests
        for file_path in modified_files:
            tests = self._find_related_tests(file_path)
            if tests:
                base_context.related_tests[file_path] = tests

        # 4. Build file tree (for AI context)
        base_context.file_tree = self._get_file_tree(max_depth=3)

        logger.info(
            "Context built: %d definitions, %d caller maps, %d test mappings",
            len(base_context.symbol_definitions),
            len(base_context.caller_map),
            len(base_context.related_tests),
        )

        return base_context

    # ------------------------------------------------------------------
    # Definition search
    # ------------------------------------------------------------------

    def _find_definition(self, symbol_name: str, max_lines: int = 30) -> Optional[str]:
        """Search the repo for a function/class definition."""
        for dirpath, _, filenames in os.walk(self.repo_root):
            if self._should_skip_dir(dirpath):
                continue
            for fname in filenames:
                if not fname.endswith(".py"):
                    continue
                full_path = os.path.join(dirpath, fname)
                content = self._read_file(full_path)
                if content is None:
                    continue

                # Search for def or class
                pattern = re.compile(
                    rf"^((?:async\s+)?def\s+{re.escape(symbol_name)}\s*\(.*|class\s+{re.escape(symbol_name)}\b.*)",
                    re.MULTILINE,
                )
                match = pattern.search(content)
                if match:
                    start = match.start()
                    # Extract up to max_lines from the definition
                    lines_from_start = content[start:].split("\n")[:max_lines]
                    return "\n".join(lines_from_start)

        return None

    # ------------------------------------------------------------------
    # Caller search
    # ------------------------------------------------------------------

    def _find_callers(self, func_name: str, exclude_files: Set[str] = None) -> List[str]:
        """Find files that call the given function."""
        if exclude_files is None:
            exclude_files = set()

        callers: List[str] = []
        pattern = re.compile(rf"\b{re.escape(func_name)}\s*\(")

        for dirpath, _, filenames in os.walk(self.repo_root):
            if self._should_skip_dir(dirpath):
                continue
            for fname in filenames:
                if not fname.endswith((".py", ".js", ".ts")):
                    continue
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, self.repo_root)

                if rel_path in exclude_files:
                    continue

                content = self._read_file(full_path)
                if content and pattern.search(content):
                    callers.append(rel_path)

        return callers[:10]  # Limit to avoid bloating context

    # ------------------------------------------------------------------
    # Test file search
    # ------------------------------------------------------------------

    def _find_related_tests(self, source_path: str) -> List[str]:
        """Find test files related to a source file."""
        basename = os.path.basename(source_path)
        name, ext = os.path.splitext(basename)

        test_patterns = [
            f"test_{name}{ext}",
            f"{name}_test{ext}",
            f"{name}.test{ext}",
            f"{name}.spec{ext}",
        ]

        found: List[str] = []
        for dirpath, _, filenames in os.walk(self.repo_root):
            if self._should_skip_dir(dirpath):
                continue
            for fname in filenames:
                if fname in test_patterns:
                    found.append(os.path.relpath(os.path.join(dirpath, fname), self.repo_root))

        return found

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_file(self, path: str) -> Optional[str]:
        """Read a file with caching."""
        if path in self._file_cache:
            return self._file_cache[path]
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            self._file_cache[path] = content
            return content
        except (IOError, OSError):
            return None

    def _should_skip_dir(self, dirpath: str) -> bool:
        """Skip non-source directories."""
        skip_patterns = {
            "__pycache__", ".git", "node_modules", ".venv", "venv",
            ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
            ".eggs", "*.egg-info",
        }
        parts = dirpath.split(os.sep)
        return any(p in skip_patterns for p in parts)

    def _get_file_tree(self, max_depth: int = 3) -> List[str]:
        """Get a shallow file tree of the repository."""
        tree: List[str] = []
        for dirpath, dirnames, filenames in os.walk(self.repo_root):
            depth = dirpath.replace(self.repo_root, "").count(os.sep)
            if depth >= max_depth:
                dirnames.clear()
                continue
            if self._should_skip_dir(dirpath):
                dirnames.clear()
                continue
            for fname in filenames:
                rel = os.path.relpath(os.path.join(dirpath, fname), self.repo_root)
                tree.append(rel)
        return tree[:200]  # Cap size
