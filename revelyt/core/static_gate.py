import json
import logging
import subprocess
from pathlib import Path
from typing import List, Set

from revelyt.core.schemas import Finding

logger = logging.getLogger(__name__)

class StaticResult:
    def __init__(self, findings: List[Finding], covered_lines: Set[int]):
        self.findings = findings
        self.covered_lines = covered_lines

class StaticGate:
    """
    Runs deterministic linters (ruff, bandit) before LLM agents to
    catch basic issues at zero token cost.
    """
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path

    def run_on_file(self, file_path: Path) -> StaticResult:
        if file_path.name.startswith(".env"):
            return self._check_env_file(file_path)

        if file_path.suffix != ".py":
            return StaticResult([], set())

        findings = []
        covered = set()
        rel_path = str(file_path.relative_to(self.repo_path))

        # Run Ruff
        try:
            res = subprocess.run(
                ["ruff", "check", "--output-format=json", rel_path],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True
            )
            if res.stdout.strip():
                data = json.loads(res.stdout)
                for item in data:
                    line = item.get("location", {}).get("row", 0)
                    if line:
                        covered.add(line)
                    code = item.get("code", "Ruff")
                    msg = item.get("message", "")
                    title = f"Ruff: {code}"
                    
                    # keep it clean
                    findings.append(Finding(
                        severity="LOW",
                        category=f"lint_{code.lower()}",
                        title=title[:50],
                        description=f"Line {line}: {msg}",
                        suggestion="Fix this style/linting issue per Ruff guidelines.",
                        confidence=1.0
                    ))
        except Exception as e:
            logger.debug("Ruff skipped or failed on %s: %s", rel_path, e)

        # Run Bandit
        try:
            res = subprocess.run(
                ["bandit", "-r", "-f", "json", rel_path],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True
            )
            if res.stdout.strip():
                try:
                    data = json.loads(res.stdout)
                    for item in data.get("results", []):
                        line = item.get("line_number", 0)
                        if line:
                            covered.add(line)
                        
                        sev_map = {"HIGH": "HIGH", "MEDIUM": "MEDIUM", "LOW": "LOW"}
                        sev = sev_map.get(item.get("issue_severity", "LOW").upper(), "LOW")
                        
                        findings.append(Finding(
                            severity=sev,
                            category="security_bandit",
                            title=item.get("test_name", "Bandit Issue")[:50],
                            description=f"Line {line}: {item.get('issue_text', '')}",
                            suggestion="Review and address the security warning identified by Bandit.",
                            confidence=1.0
                        ))
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            logger.debug("Bandit skipped or failed on %s: %s", rel_path, e)

        return StaticResult(findings=findings, covered_lines=covered)

    # ------------------------------------------------------------------
    # .env file detection
    # ------------------------------------------------------------------

    _SECRET_PATTERNS = (
        "password", "passwd", "secret", "api_key", "apikey",
        "token", "private_key", "access_key", "auth",
        "database_url", "db_url", "connection_string",
    )

    def _check_env_file(self, file_path: Path) -> StaticResult:
        """
        Flag committed .env files as CRITICAL security issues.
        Scans for common secret patterns (API keys, passwords, tokens).
        Zero token cost — runs entirely in static analysis.
        """
        findings = []
        covered = set()

        rel_path = str(file_path.relative_to(self.repo_path))

        # The mere existence of a committed .env file is CRITICAL
        findings.append(Finding(
            severity="CRITICAL",
            category="Secrets Exposure",
            title=f"Committed .env file: {file_path.name}",
            line_number=1,
            class_or_method="Global",
            description=(
                f"{rel_path} is committed to version control. "
                f"Environment files typically contain secrets, API keys, "
                f"and database credentials that should never be in a repository."
            ),
            suggestion=(
                "Add .env to .gitignore immediately. Rotate all secrets "
                "found in this file. Use a secrets manager or CI/CD env vars."
            ),
            confidence=1.0,
        ))
        covered.add(1)

        # Scan for specific secrets line by line
        try:
            content = file_path.read_text(errors="ignore")
            for line_num, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" not in stripped:
                    continue

                key = stripped.split("=", 1)[0].strip().lower()
                value = stripped.split("=", 1)[1].strip().strip("'\"")

                # Skip empty values and placeholders
                if not value or value in ("", "xxx", "your_key_here",
                                          "changeme", "TODO", "REPLACE_ME"):
                    continue

                for pattern in self._SECRET_PATTERNS:
                    if pattern in key:
                        findings.append(Finding(
                            severity="CRITICAL",
                            category="Hardcoded Secret",
                            title=f"Secret in .env: {key.upper()}",
                            line_number=line_num,
                            class_or_method="Global",
                            description=(
                                f"Line {line_num}: {key.upper()} contains a "
                                f"hardcoded value that appears to be a secret."
                            ),
                            suggestion="Rotate this credential immediately and remove from version control.",
                            confidence=0.95,
                        ))
                        covered.add(line_num)
                        break
        except Exception as exc:
            logger.debug("Failed to read .env file %s: %s", file_path.name, exc)

        return StaticResult(findings=findings, covered_lines=covered)
