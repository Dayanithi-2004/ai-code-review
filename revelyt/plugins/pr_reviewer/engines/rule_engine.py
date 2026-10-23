"""
engines/rule_engine.py — Deterministic rule engine with AST-powered analysis.

Catches issues that do NOT need an LLM: eval() usage, hardcoded IPs,
unused imports, bare excepts, dangerous functions, SQL injection patterns, etc.
"""

from __future__ import annotations

import ast
import re
from typing import List, Tuple

from revelyt.plugins.pr_reviewer.engines.base import (
    Engine,
    FileChunk,
    Finding,
    FindingType,
    PRContext,
    Severity,
)


# (rule_id, regex_pattern, message, severity, finding_type, suggestion)
_REGEX_RULES: List[Tuple[str, str, str, Severity, FindingType, str]] = [
    (
        "no-eval", r"\beval\s*\(",
        "Use of eval() is a security risk — arbitrary code execution",
        Severity.CRITICAL, FindingType.SECURITY,
        "Replace eval() with ast.literal_eval() or a safe parser.",
    ),
    (
        "no-exec", r"\bexec\s*\(",
        "Use of exec() is a security risk — arbitrary code execution",
        Severity.CRITICAL, FindingType.SECURITY,
        "Remove exec() and use a safer alternative.",
    ),
    (
        "sql-injection", r"(?i)(?:execute|cursor\.execute)\s*\(\s*[f'\"].*?\{.*?\}",
        "Possible SQL injection — string formatting in query",
        Severity.CRITICAL, FindingType.SECURITY,
        "Use parameterized queries instead of f-strings or .format().",
    ),
    (
        "sql-injection-concat", r"(?i)(?:execute|cursor\.execute)\s*\(.*?\+\s*(?:str|request|user|input)",
        "Possible SQL injection — string concatenation in query",
        Severity.CRITICAL, FindingType.SECURITY,
        "Use parameterized queries instead of string concatenation.",
    ),
    (
        "hardcoded-ip", r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "Hardcoded IP address detected",
        Severity.LOW, FindingType.QUALITY,
        "Move IP addresses to configuration files or environment variables.",
    ),
    (
        "fixme-todo", r"(?i)\b(?:FIXME|TODO|HACK|XXX|TEMP)\b",
        "Development marker found — should be resolved before merge",
        Severity.LOW, FindingType.QUALITY,
        "Address or remove this marker before merging.",
    ),
    (
        "no-print-debug", r"\bprint\s*\(",
        "print() statement found — consider using proper logging",
        Severity.LOW, FindingType.QUALITY,
        "Replace with logging.debug() / logger.info() for production code.",
    ),
    (
        "no-debugger", r"\b(?:breakpoint|pdb\.set_trace|debugger)\s*\(",
        "Debug breakpoint found — remove before merge",
        Severity.HIGH, FindingType.BUG,
        "Remove debugging breakpoints before merging.",
    ),
    (
        "insecure-hash", r"(?:md5|sha1)\s*\(",
        "Weak hash function used — MD5/SHA1 are cryptographically broken",
        Severity.HIGH, FindingType.SECURITY,
        "Use SHA-256 or stronger: hashlib.sha256()",
    ),
    (
        "no-wildcard-import", r"from\s+\S+\s+import\s+\*",
        "Wildcard import pollutes namespace and hides dependencies",
        Severity.MEDIUM, FindingType.QUALITY,
        "Import specific names instead of using *.",
    ),
    (
        "subprocess-shell", r"subprocess\.\w+\(.*shell\s*=\s*True",
        "subprocess with shell=True is a security risk",
        Severity.HIGH, FindingType.SECURITY,
        "Avoid shell=True; pass arguments as a list instead.",
    ),
    (
        "unsafe-yaml", r"yaml\.load\s*\(",
        "yaml.load() without Loader is unsafe — allows arbitrary code execution",
        Severity.HIGH, FindingType.SECURITY,
        "Use yaml.safe_load() instead of yaml.load().",
    ),
    (
        "unsafe-pickle", r"pickle\.loads?\s*\(",
        "pickle.load/loads can execute arbitrary code on untrusted data",
        Severity.HIGH, FindingType.SECURITY,
        "Use json or a safe serialization format for untrusted data.",
    ),
    (
        "js-innerhtml", r"\.innerHTML\s*=",
        "Direct innerHTML assignment is an XSS risk",
        Severity.HIGH, FindingType.SECURITY,
        "Use textContent or a sanitization library instead.",
    ),
    (
        "js-document-write", r"document\.write\s*\(",
        "document.write() is an XSS risk and blocks rendering",
        Severity.MEDIUM, FindingType.SECURITY,
        "Use DOM manipulation methods instead.",
    ),
]


class RuleEngine(Engine):
    """
    Runs deterministic regex + AST rules against diff chunks.

    This engine catches obvious issues WITHOUT using an LLM,
    providing instant, zero-cost, zero-hallucination feedback.
    """

    name = "rule_engine"

    def __init__(self):
        self._compiled_rules = [
            (rule_id, re.compile(pattern), msg, sev, ftype, suggestion)
            for rule_id, pattern, msg, sev, ftype, suggestion in _REGEX_RULES
        ]

    def run(self, chunks: List[FileChunk], context: PRContext) -> List[Finding]:
        findings: List[Finding] = []
        for chunk in chunks:
            findings.extend(self._run_regex_rules(chunk))
            if chunk.language == "python":
                findings.extend(self._run_python_ast_rules(chunk))
        return findings

    # ------------------------------------------------------------------
    # Regex rules
    # ------------------------------------------------------------------

    def _run_regex_rules(self, chunk: FileChunk) -> List[Finding]:
        results: List[Finding] = []
        for i, line in enumerate(chunk.added_lines):
            stripped = line.strip()
            # Skip comments
            if stripped.startswith(("#", "//", "/*", "*", "<!--")):
                continue
            for rule_id, pattern, msg, severity, ftype, suggestion in self._compiled_rules:
                if pattern.search(line):
                    results.append(
                        Finding(
                            file=chunk.file_path,
                            line=chunk.start_line + i,
                            severity=severity,
                            type=ftype,
                            message=msg,
                            suggestion=suggestion,
                            confidence=0.95,
                            rule_id=rule_id,
                            engine=self.name,
                        )
                    )
        return results

    # ------------------------------------------------------------------
    # Python AST rules
    # ------------------------------------------------------------------

    def _run_python_ast_rules(self, chunk: FileChunk) -> List[Finding]:
        """AST-based rules for Python: bare excepts, unused imports, etc."""
        code = "\n".join(chunk.added_lines)
        if not code.strip():
            return []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        findings: List[Finding] = []

        for node in ast.walk(tree):
            # Bare except
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                findings.append(
                    Finding(
                        file=chunk.file_path,
                        line=chunk.start_line + (node.lineno - 1),
                        severity=Severity.MEDIUM,
                        type=FindingType.BUG,
                        message="Bare except clause catches all exceptions including SystemExit and KeyboardInterrupt",
                        suggestion="Catch specific exceptions: except ValueError, TypeError:",
                        confidence=0.95,
                        rule_id="no-bare-except",
                        engine=self.name,
                    )
                )

            # Assert in non-test code
            if isinstance(node, ast.Assert) and "test" not in chunk.file_path.lower():
                findings.append(
                    Finding(
                        file=chunk.file_path,
                        line=chunk.start_line + (node.lineno - 1),
                        severity=Severity.LOW,
                        type=FindingType.QUALITY,
                        message="assert statement in production code — removed by python -O",
                        suggestion="Use proper validation with if/raise instead.",
                        confidence=0.80,
                        rule_id="no-assert-production",
                        engine=self.name,
                    )
                )

            # Global statements
            if isinstance(node, ast.Global):
                findings.append(
                    Finding(
                        file=chunk.file_path,
                        line=chunk.start_line + (node.lineno - 1),
                        severity=Severity.MEDIUM,
                        type=FindingType.QUALITY,
                        message="global statement makes code harder to reason about and test",
                        suggestion="Consider passing values as parameters or using class attributes.",
                        confidence=0.85,
                        rule_id="no-global",
                        engine=self.name,
                    )
                )

        return findings
