"""
diff/semantic_diff.py — Semantic diff analysis and cyclomatic complexity.

Goes beyond line-level diffs to understand *what* changed semantically:
function signatures, class hierarchies, control flow complexity, etc.
"""

from __future__ import annotations

import ast
import re
from typing import List, Optional

from revelyt.plugins.pr_reviewer.engines.base import ChangeType, FileChunk


class SemanticDiffAnalyzer:
    """
    Enriches FileChunks with semantic metadata:
      - Cyclomatic complexity of changed code
      - Affected function / class names
      - Semantic change classification (signature change, logic change, etc.)
    """

    def analyze(self, chunks: List[FileChunk]) -> List[FileChunk]:
        """Mutate and return chunks with enriched semantic metadata."""
        for chunk in chunks:
            if chunk.language == "python":
                self._analyze_python(chunk)
            elif chunk.language in ("javascript", "typescript"):
                self._analyze_js(chunk)
            else:
                self._analyze_generic(chunk)
        return chunks

    # ------------------------------------------------------------------
    # Python-specific analysis
    # ------------------------------------------------------------------

    def _analyze_python(self, chunk: FileChunk) -> None:
        """Use Python AST to extract affected symbols and complexity."""
        # Extract function/class names from added lines
        chunk.affected_functions = self._extract_python_functions(chunk.added_lines + chunk.deleted_lines)
        chunk.affected_classes = self._extract_python_classes(chunk.added_lines + chunk.deleted_lines)

        # Calculate cyclomatic complexity of addded code
        added_block = "\n".join(chunk.added_lines)
        chunk.complexity = self._python_cyclomatic_complexity(added_block)

        # Detect signature changes
        old_sigs = self._extract_python_signatures(chunk.deleted_lines)
        new_sigs = self._extract_python_signatures(chunk.added_lines)
        if old_sigs and new_sigs and old_sigs != new_sigs:
            chunk.change_type = ChangeType.SIGNATURE_CHANGED
            chunk.risk_score += 3.0  # Signature changes are high risk
        elif chunk.complexity > 10:
            chunk.change_type = ChangeType.LOGIC_CHANGED
            chunk.risk_score += 2.0

    def _extract_python_functions(self, lines: List[str]) -> List[str]:
        """Extract function names from lines using regex."""
        pattern = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(")
        return list({m.group(1) for line in lines if (m := pattern.match(line))})

    def _extract_python_classes(self, lines: List[str]) -> List[str]:
        """Extract class names from lines using regex."""
        pattern = re.compile(r"^\s*class\s+(\w+)")
        return list({m.group(1) for line in lines if (m := pattern.match(line))})

    def _extract_python_signatures(self, lines: List[str]) -> List[str]:
        """Extract full function signatures (def name(args)) from lines."""
        pattern = re.compile(r"^\s*(?:async\s+)?def\s+\w+\s*\(.*?\)")
        return [m.group(0).strip() for line in lines if (m := pattern.match(line))]

    def _python_cyclomatic_complexity(self, code: str) -> float:
        """
        Approximate cyclomatic complexity by counting decision points.
        Uses AST when possible, falls back to regex counting.
        """
        try:
            tree = ast.parse(code)
            return self._ast_complexity(tree)
        except SyntaxError:
            return self._regex_complexity(code)

    def _ast_complexity(self, tree: ast.AST) -> float:
        """Count decision nodes in a Python AST."""
        complexity = 1.0  # Base complexity
        decision_nodes = (
            ast.If, ast.For, ast.While, ast.ExceptHandler,
            ast.With, ast.Assert, ast.BoolOp,
        )
        for node in ast.walk(tree):
            if isinstance(node, decision_nodes):
                complexity += 1
            # BoolOp with 'and'/'or' adds branching
            if isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity

    def _regex_complexity(self, code: str) -> float:
        """Fallback: count decision keywords in non-parseable code."""
        keywords = ["if ", "elif ", "else:", "for ", "while ", "except ", "and ", "or ", "try:", "assert "]
        complexity = 1.0
        for keyword in keywords:
            complexity += code.count(keyword)
        return complexity

    # ------------------------------------------------------------------
    # JS / TS analysis (regex-based)
    # ------------------------------------------------------------------

    def _analyze_js(self, chunk: FileChunk) -> None:
        """Regex-based analysis for JavaScript / TypeScript."""
        func_pattern = re.compile(
            r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\(|function))"
        )
        all_lines = chunk.added_lines + chunk.deleted_lines
        names = []
        for line in all_lines:
            m = func_pattern.search(line)
            if m:
                names.append(m.group(1) or m.group(2))
        chunk.affected_functions = list(set(names))

        class_pattern = re.compile(r"class\s+(\w+)")
        chunk.affected_classes = list({m.group(1) for line in all_lines if (m := class_pattern.search(line))})

        code = "\n".join(chunk.added_lines)
        chunk.complexity = self._regex_complexity(code)

    # ------------------------------------------------------------------
    # Generic fallback
    # ------------------------------------------------------------------

    def _analyze_generic(self, chunk: FileChunk) -> None:
        """Best-effort analysis for unknown languages."""
        code = "\n".join(chunk.added_lines)
        chunk.complexity = self._regex_complexity(code)
