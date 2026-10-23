"""
diff/change_graph.py — Cross-module dependency graph for ripple-effect detection.

Maps how a change in one function propagates to downstream callers,
allowing the reviewer to warn about broader impact.
"""

from __future__ import annotations

import ast
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set

from revelyt.plugins.pr_reviewer.engines.base import FileChunk, Finding, FindingType, Severity


class ChangeGraph:
    """
    Builds a dependency graph from a repository's source files
    and determines which modules are affected by changes in a PR.
    """

    def __init__(self):
        self._import_graph: Dict[str, Set[str]] = defaultdict(set)  # file -> set of files it imports
        self._reverse_graph: Dict[str, Set[str]] = defaultdict(set)  # file -> set of files that import it
        self._symbol_locations: Dict[str, str] = {}  # "module.func" -> file_path

    def build_from_repo(self, repo_root: str, extensions: Set[str] = None) -> None:
        """
        Walk a repository and build the import dependency graph.
        Only processes Python files by default.
        """
        if extensions is None:
            extensions = {".py"}

        for dirpath, _, filenames in os.walk(repo_root):
            # Skip hidden dirs, venvs, node_modules
            if any(skip in dirpath for skip in ("__pycache__", ".git", "node_modules", ".venv", "venv")):
                continue
            for fname in filenames:
                _, ext = os.path.splitext(fname)
                if ext in extensions:
                    full_path = os.path.join(dirpath, fname)
                    rel_path = os.path.relpath(full_path, repo_root)
                    self._index_python_file(rel_path, full_path)

    def get_affected_files(self, changed_files: List[str]) -> Dict[str, List[str]]:
        """
        Given a list of changed files, return a mapping of
        changed_file -> [list of files that import/depend on it].
        """
        result: Dict[str, List[str]] = {}
        for f in changed_files:
            dependents = self._reverse_graph.get(f, set())
            if dependents:
                result[f] = sorted(dependents)
        return result

    def generate_findings(self, chunks: List[FileChunk]) -> List[Finding]:
        """Generate findings for changes that have significant downstream impact."""
        findings: List[Finding] = []
        changed_files = list({c.file_path for c in chunks})
        affected = self.get_affected_files(changed_files)

        for source_file, dependents in affected.items():
            if len(dependents) >= 3:  # Only flag if 3+ dependents
                findings.append(
                    Finding(
                        file=source_file,
                        line=1,
                        severity=Severity.MEDIUM,
                        type=FindingType.QUALITY,
                        message=(
                            f"This file is imported by {len(dependents)} other module(s). "
                            f"Changes here may have ripple effects."
                        ),
                        suggestion=f"Affected modules: {', '.join(dependents[:5])}",
                        confidence=0.85,
                        rule_id="change-graph-ripple",
                        engine="change_graph",
                    )
                )

        return findings

    # ------------------------------------------------------------------
    # Internal indexing
    # ------------------------------------------------------------------

    def _index_python_file(self, rel_path: str, full_path: str) -> None:
        """Parse a Python file and record its imports."""
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                source = f.read()
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            return

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imported_modules = self._resolve_import(node, rel_path)
                for mod in imported_modules:
                    self._import_graph[rel_path].add(mod)
                    self._reverse_graph[mod].add(rel_path)

            # Index top-level function/class definitions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                module_name = rel_path.replace("/", ".").replace(".py", "")
                self._symbol_locations[f"{module_name}.{node.name}"] = rel_path
            elif isinstance(node, ast.ClassDef):
                module_name = rel_path.replace("/", ".").replace(".py", "")
                self._symbol_locations[f"{module_name}.{node.name}"] = rel_path

    def _resolve_import(self, node: ast.AST, current_file: str) -> List[str]:
        """Best-effort resolution of import statements to file paths."""
        results: List[str] = []

        if isinstance(node, ast.Import):
            for alias in node.names:
                candidate = alias.name.replace(".", "/") + ".py"
                results.append(candidate)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                candidate = node.module.replace(".", "/") + ".py"
                results.append(candidate)
                # Also try as a package __init__.py
                results.append(node.module.replace(".", "/") + "/__init__.py")

        return results
