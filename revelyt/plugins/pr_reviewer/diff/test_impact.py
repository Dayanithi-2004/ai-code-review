"""
diff/test_impact.py — Test coverage impact analysis.

Correlates modified source files with their test files to detect
missing test updates. Suggests specific test commands for CI.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set

from revelyt.plugins.pr_reviewer.engines.base import FileChunk, Finding, FindingType, Severity


# Common test directory / file naming conventions
_TEST_PREFIXES = ("test_", "tests_", "spec_")
_TEST_SUFFIXES = ("_test", "_tests", "_spec", ".test", ".spec")
_TEST_DIRS = {"tests", "test", "spec", "__tests__", "specs"}


class TestImpactAnalyzer:
    """
    Detects if modified production code has accompanying test
    updates in the same PR.

    Capabilities:
      - Maps source files to expected test files
      - Flags missing test coverage updates
      - Suggests specific test commands (pytest / jest / go test)
    """

    def analyze(
        self,
        chunks: List[FileChunk],
        all_changed_files: Optional[Set[str]] = None,
    ) -> List[Finding]:
        """
        Run test impact analysis.

        Parameters
        ----------
        chunks : List[FileChunk]
            All diff chunks for the PR.
        all_changed_files : set, optional
            If provided, the full set of files changed in the PR
            (used to check if a test file was also modified).

        Returns
        -------
        List[Finding]
        """
        if all_changed_files is None:
            all_changed_files = {c.file_path for c in chunks}

        findings: List[Finding] = []
        source_chunks = [c for c in chunks if not self._is_test_file(c.file_path)]
        changed_test_files = {f for f in all_changed_files if self._is_test_file(f)}

        for chunk in source_chunks:
            # Skip non-code files
            if chunk.extension not in (".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".java", ".rb"):
                continue

            expected_tests = self._find_expected_test_files(chunk.file_path)
            has_test_update = any(t in changed_test_files for t in expected_tests)

            if not has_test_update and chunk.total_changed_lines > 5:
                test_cmd = self._suggest_test_command(chunk, expected_tests)
                findings.append(
                    Finding(
                        file=chunk.file_path,
                        line=chunk.start_line,
                        severity=Severity.MEDIUM,
                        type=FindingType.QUALITY,
                        message=(
                            f"Source file modified ({chunk.total_changed_lines} lines changed) "
                            f"but no corresponding test file was updated in this PR."
                        ),
                        suggestion=f"Consider updating: {', '.join(expected_tests[:3]) or 'a relevant test file'}",
                        confidence=0.75,
                        rule_id="test-impact-missing",
                        engine="test_impact",
                        explanation=test_cmd,
                    )
                )

                # Update chunk metadata
                chunk.test_coverage_status = "missing"
            else:
                chunk.test_coverage_status = "covered" if has_test_update else "unknown"

        return findings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_test_file(self, file_path: str) -> bool:
        """Determine if a file is a test file based on naming conventions."""
        basename = os.path.basename(file_path).lower()
        name_no_ext = os.path.splitext(basename)[0]

        if any(basename.startswith(p) for p in _TEST_PREFIXES):
            return True
        if any(name_no_ext.endswith(s) for s in _TEST_SUFFIXES):
            return True

        parts = file_path.replace("\\", "/").split("/")
        if any(d in _TEST_DIRS for d in parts):
            return True

        return False

    def _find_expected_test_files(self, source_path: str) -> List[str]:
        """
        Given a source file, generate candidate test file paths.
        e.g. ``src/auth/login.py`` → ``tests/auth/test_login.py``
        """
        dirname = os.path.dirname(source_path)
        basename = os.path.basename(source_path)
        name, ext = os.path.splitext(basename)

        candidates: List[str] = []

        # Same directory: test_<name>.py
        candidates.append(os.path.join(dirname, f"test_{name}{ext}"))
        # Same directory: <name>_test.py
        candidates.append(os.path.join(dirname, f"{name}_test{ext}"))

        # Tests directory sibling
        parent = os.path.dirname(dirname)
        dir_name = os.path.basename(dirname)
        candidates.append(os.path.join(parent, "tests", dir_name, f"test_{name}{ext}"))
        candidates.append(os.path.join("tests", f"test_{name}{ext}"))

        # JS/TS conventions
        if ext in (".js", ".ts", ".tsx", ".jsx"):
            candidates.append(os.path.join(dirname, f"{name}.test{ext}"))
            candidates.append(os.path.join(dirname, f"{name}.spec{ext}"))
            candidates.append(os.path.join(dirname, "__tests__", f"{name}{ext}"))

        return candidates

    def _suggest_test_command(self, chunk: FileChunk, expected_tests: List[str]) -> str:
        """Generate a suggested test command based on language."""
        if chunk.language == "python":
            if expected_tests:
                return f"Run: pytest {expected_tests[0]}"
            module = chunk.file_path.replace("/", ".").replace(".py", "")
            return f"Run: pytest -k '{os.path.basename(chunk.file_path).replace('.py', '')}'"
        elif chunk.language in ("javascript", "typescript"):
            return f"Run: npx jest --testPathPattern='{os.path.basename(chunk.file_path).split('.')[0]}'"
        elif chunk.language == "go":
            pkg_dir = os.path.dirname(chunk.file_path)
            return f"Run: go test ./{pkg_dir}/..."
        else:
            return "Ensure relevant tests are updated for this change."
