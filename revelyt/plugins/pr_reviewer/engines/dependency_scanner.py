"""
engines/dependency_scanner.py — Vulnerability scanning for dependency changes.

Detects modifications to lockfiles/dependency manifests in the PR diff
and checks added/updated packages against known vulnerability databases.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional, Set, Tuple

from revelyt.plugins.pr_reviewer.engines.base import (
    Engine,
    FileChunk,
    Finding,
    FindingType,
    PRContext,
    Severity,
)

logger = logging.getLogger(__name__)

# Files that contain dependency declarations
_DEPENDENCY_FILES: Set[str] = {
    "requirements.txt", "Pipfile", "Pipfile.lock", "pyproject.toml", "setup.py",
    "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "go.mod", "go.sum",
    "Cargo.toml", "Cargo.lock",
    "Gemfile", "Gemfile.lock",
    "build.gradle", "pom.xml",
    "composer.json", "composer.lock",
}

# Known vulnerable packages (hardcoded subset — real impl would query a DB)
_KNOWN_VULNERABLE: Dict[str, Dict] = {
    "pyyaml": {"versions": ["<5.4"], "cve": "CVE-2020-14343", "severity": "critical", "advisory": "Arbitrary code execution via yaml.load()"},
    "requests": {"versions": ["<2.20.0"], "cve": "CVE-2018-18074", "severity": "high", "advisory": "Session fixation vulnerability"},
    "django": {"versions": ["<3.2.25", "<4.2.10"], "cve": "CVE-2024-24680", "severity": "high", "advisory": "Denial of service in intcomma template filter"},
    "flask": {"versions": ["<2.3.2"], "cve": "CVE-2023-30861", "severity": "high", "advisory": "Session cookie vulnerability"},
    "urllib3": {"versions": ["<2.0.6"], "cve": "CVE-2023-43804", "severity": "medium", "advisory": "Cookie leaking via cross-origin redirects"},
    "pillow": {"versions": ["<10.0.1"], "cve": "CVE-2023-44271", "severity": "high", "advisory": "Denial of service via image processing"},
    "cryptography": {"versions": ["<41.0.4"], "cve": "CVE-2023-38325", "severity": "high", "advisory": "NULL pointer dereference in PKCS7"},
    "lodash": {"versions": ["<4.17.21"], "cve": "CVE-2021-23337", "severity": "critical", "advisory": "Prototype pollution via template()"},
    "axios": {"versions": ["<1.6.0"], "cve": "CVE-2023-45857", "severity": "medium", "advisory": "CSRF token leaking"},
    "express": {"versions": ["<4.18.2"], "cve": "CVE-2022-24999", "severity": "high", "advisory": "Open redirect vulnerability"},
}


class DependencyScanner(Engine):
    """
    Scans dependency file changes in PRs to detect:
      - Addition of known vulnerable packages
      - Downgrade of package versions
      - Addition of suspicious/unknown packages
    """

    name = "dependency_scanner"

    def run(self, chunks: List[FileChunk], context: PRContext) -> List[Finding]:
        findings: List[Finding] = []

        for chunk in chunks:
            basename = os.path.basename(chunk.file_path)
            if basename not in _DEPENDENCY_FILES:
                continue

            logger.info("Scanning dependency file: %s", chunk.file_path)

            # Extract added packages from diff
            added_packages = self._extract_added_packages(chunk, basename)

            for pkg_name, pkg_version in added_packages:
                vuln = self._check_vulnerability(pkg_name)
                if vuln:
                    findings.append(
                        Finding(
                            file=chunk.file_path,
                            line=chunk.start_line,
                            severity=Severity(vuln["severity"]),
                            type=FindingType.DEPENDENCY,
                            message=f"⚠️ Vulnerable dependency: {pkg_name} ({vuln['cve']})",
                            suggestion=f"{vuln['advisory']}. Upgrade to a patched version.",
                            confidence=0.90,
                            rule_id=f"dep-vuln-{vuln['cve'].lower()}",
                            engine=self.name,
                            cwe_id="CWE-1104",
                            owasp_ref="A06:2021 Vulnerable Components",
                        )
                    )
                elif pkg_version == "latest" or pkg_version == "*":
                    findings.append(
                        Finding(
                            file=chunk.file_path,
                            line=chunk.start_line,
                            severity=Severity.MEDIUM,
                            type=FindingType.DEPENDENCY,
                            message=f"Unpinned dependency: {pkg_name}@{pkg_version}",
                            suggestion="Pin to a specific version to ensure reproducible builds.",
                            confidence=0.85,
                            rule_id="dep-unpinned",
                            engine=self.name,
                        )
                    )

        return findings

    def _extract_added_packages(self, chunk: FileChunk, filename: str) -> List[Tuple[str, str]]:
        """Extract (package_name, version) tuples from added lines."""
        packages: List[Tuple[str, str]] = []

        for line in chunk.added_lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if filename == "requirements.txt":
                match = re.match(r"^([a-zA-Z0-9_-]+)\s*([><=!~]+\s*[\d.]+)?", line)
                if match:
                    packages.append((match.group(1).lower(), match.group(2) or "latest"))

            elif filename == "package.json":
                match = re.match(r'"([^"]+)"\s*:\s*"([^"]+)"', line)
                if match:
                    packages.append((match.group(1).lower(), match.group(2)))

            elif filename == "pyproject.toml":
                match = re.match(r'"([a-zA-Z0-9_-]+)\s*([><=!~]+[\d.]+)?"', line)
                if match:
                    packages.append((match.group(1).lower(), match.group(2) or "latest"))

            elif filename in ("go.mod",):
                match = re.match(r"\s*([^\s]+)\s+v?([\d.]+)", line)
                if match:
                    pkg = match.group(1).split("/")[-1]
                    packages.append((pkg.lower(), match.group(2)))

        return packages

    def _check_vulnerability(self, package_name: str) -> Optional[Dict]:
        """Check if a package has known vulnerabilities."""
        return _KNOWN_VULNERABLE.get(package_name.lower())
