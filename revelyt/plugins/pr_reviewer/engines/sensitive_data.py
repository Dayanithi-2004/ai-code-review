"""
engines/sensitive_data.py — Enterprise-grade secret and PII detection.

Uses high-fidelity regex patterns (modeled after GitGuardian / TruffleHog)
to detect AWS keys, JWT secrets, private keys, API tokens, Slack tokens,
Stripe keys, PII like emails/SSNs, and configurable enterprise patterns.
"""

from __future__ import annotations

import re
from typing import Dict, List, Pattern, Tuple

from revelyt.plugins.pr_reviewer.engines.base import (
    Engine,
    FileChunk,
    Finding,
    FindingType,
    PRContext,
    Severity,
)


# (rule_id, display_name, regex, severity, cwe_id)
_SECRET_PATTERNS: List[Tuple[str, str, str, Severity, str]] = [
    # Cloud provider keys
    ("aws-access-key", "AWS Access Key ID", r"(?:^|[^A-Za-z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?:[^A-Za-z0-9]|$)", Severity.CRITICAL, "CWE-798"),
    ("aws-secret-key", "AWS Secret Access Key", r"(?i)(?:aws_secret_access_key|aws_secret)\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{40}['\"]?", Severity.CRITICAL, "CWE-798"),
    ("gcp-api-key", "Google Cloud API Key", r"AIza[0-9A-Za-z_-]{35}", Severity.CRITICAL, "CWE-798"),
    ("azure-storage-key", "Azure Storage Account Key", r"(?i)(?:DefaultEndpointsProtocol|AccountKey)\s*=\s*[A-Za-z0-9+/=]{88}", Severity.CRITICAL, "CWE-798"),

    # Authentication tokens
    ("github-token", "GitHub Token", r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}", Severity.CRITICAL, "CWE-798"),
    ("gitlab-token", "GitLab Token", r"glpat-[A-Za-z0-9_-]{20,}", Severity.CRITICAL, "CWE-798"),
    ("slack-token", "Slack Token", r"xox[bpors]-[0-9a-zA-Z]{10,}", Severity.HIGH, "CWE-798"),
    ("slack-webhook", "Slack Webhook URL", r"https://hooks\.slack\.com/services/T[A-Z0-9]{8,}/B[A-Z0-9]{8,}/[A-Za-z0-9]{24,}", Severity.HIGH, "CWE-798"),
    ("stripe-live-key", "Stripe Live Key", r"sk_live_[0-9a-zA-Z]{24,}", Severity.CRITICAL, "CWE-798"),
    ("stripe-test-key", "Stripe Test Key", r"sk_test_[0-9a-zA-Z]{24,}", Severity.LOW, "CWE-798"),
    ("twilio-api-key", "Twilio API Key", r"SK[0-9a-fA-F]{32}", Severity.HIGH, "CWE-798"),
    ("sendgrid-api-key", "SendGrid API Key", r"SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}", Severity.HIGH, "CWE-798"),

    # Cryptographic material
    ("private-key", "Private Key", r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", Severity.CRITICAL, "CWE-321"),
    ("jwt-secret", "Hardcoded JWT Secret", r"(?i)(?:jwt_secret|jwt_key|signing_key)\s*[=:]\s*['\"][A-Za-z0-9\-_]{16,}['\"]", Severity.CRITICAL, "CWE-798"),

    # Generic patterns
    ("generic-password", "Hardcoded Password", r"(?i)(?:password|passwd|pwd)\s*[=:]\s*['\"][^\s'\"]{8,}['\"]", Severity.HIGH, "CWE-798"),
    ("generic-secret", "Hardcoded Secret", r"(?i)(?:secret|api_key|apikey|access_token|auth_token)\s*[=:]\s*['\"][A-Za-z0-9\-_]{16,}['\"]", Severity.HIGH, "CWE-798"),
    ("connection-string", "Database Connection String", r"(?i)(?:mongodb\+srv|postgresql|mysql|redis)://[^\s'\"]+", Severity.HIGH, "CWE-798"),
]

_PII_PATTERNS: List[Tuple[str, str, str, Severity, str]] = [
    ("pii-email", "Email Address in Code", r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", Severity.MEDIUM, "CWE-359"),
    ("pii-ssn", "Social Security Number", r"\b\d{3}-\d{2}-\d{4}\b", Severity.CRITICAL, "CWE-359"),
    ("pii-phone", "Phone Number", r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", Severity.LOW, "CWE-359"),
    ("pii-credit-card", "Credit Card Number", r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b", Severity.CRITICAL, "CWE-359"),
]

# Files where secrets are expected (e.g. .env.example, test fixtures)
_ALLOWLISTED_FILES = {"env.example", ".env.example", ".env.sample", "conftest.py"}


class SensitiveDataEngine(Engine):
    """
    Scans diff chunks for hardcoded secrets, API keys, PII, and
    cryptographic material.  Supports custom enterprise patterns.
    """

    name = "sensitive_data"

    def __init__(self, extra_patterns: List[Tuple[str, str, str, Severity, str]] | None = None):
        self._compiled: List[Tuple[str, str, Pattern, Severity, str]] = []
        all_patterns = _SECRET_PATTERNS + _PII_PATTERNS + (extra_patterns or [])
        for rule_id, name, regex, severity, cwe in all_patterns:
            self._compiled.append((rule_id, name, re.compile(regex), severity, cwe))

    def run(self, chunks: List[FileChunk], context: PRContext) -> List[Finding]:
        findings: List[Finding] = []
        for chunk in chunks:
            if self._is_allowlisted(chunk.file_path):
                continue
            findings.extend(self._scan_chunk(chunk))
        return findings

    def _scan_chunk(self, chunk: FileChunk) -> List[Finding]:
        findings: List[Finding] = []
        # Only scan added lines (not deleted lines — those are being removed)
        for i, line in enumerate(chunk.added_lines):
            for rule_id, name, pattern, severity, cwe in self._compiled:
                if pattern.search(line):
                    # Skip false positives: comments, example/placeholder values
                    stripped = line.strip()
                    if stripped.startswith(("#", "//", "/*", "*", "<!--")):
                        continue
                    if any(placeholder in line.lower() for placeholder in ("example", "placeholder", "xxx", "your_", "change_me")):
                        continue

                    findings.append(
                        Finding(
                            file=chunk.file_path,
                            line=chunk.start_line + i,
                            severity=severity,
                            type=FindingType.SECRET,
                            message=f"🔐 {name} detected in code",
                            suggestion=f"Move this value to an environment variable or secrets manager. Never commit secrets to version control.",
                            confidence=0.92,
                            rule_id=rule_id,
                            engine=self.name,
                            cwe_id=cwe,
                            owasp_ref="A02:2021 Cryptographic Failures",
                        )
                    )
        return findings

    def _is_allowlisted(self, file_path: str) -> bool:
        """Skip files where secrets are expected (examples, fixtures)."""
        import os
        basename = os.path.basename(file_path)
        return basename in _ALLOWLISTED_FILES
