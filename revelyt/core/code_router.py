"""
code_router — Smart file classification for Revelyt.

Routes files to appropriate analysis depth:
  HEAVY   → Security-critical, auth, DB, core business logic (full analysis, large model)
  NORMAL  → Standard business logic (skeleton mode, large model)
  LIGHT   → Tests, configs, utilities (small model, Wave 1 only)
  SKIP    → Generated code, migrations, vendor files (no analysis)

This avoids wasting expensive LLM tokens on files that don't need deep review.
"""

import re
from pathlib import Path
from typing import Tuple, Set

# ── File classification patterns ─────────────────────────────────────────────

# Files that are auto-generated — never worth analyzing
_GENERATED_PATTERNS = [
    r"migrations?/",  # DB migrations
    r"alembic/",  # Alembic migrations
    r"\.generated\.",  # Explicitly generated
    r"_pb2\.",  # Protocol buffers
    r"\.min\.",  # Minified files
    r"vendor/",  # Vendored dependencies
    r"node_modules/",  # Node dependencies
    r"__pycache__/",  # Python cache
    r"\.egg-info/",  # Python package metadata
    r"dist/",  # Build output
    r"build/",  # Build output
    r"target/",  # Java/Rust build output
    r"\.next/",  # Next.js build
    r"\.nuxt/",  # Nuxt build
    r"\.svelte-kit/",  # SvelteKit build
    r"coverage/",  # Test coverage reports
    r"\.coverage",  # Coverage data
    r"htmlcov/",  # HTML coverage
    r"generated/",  # Explicitly generated
    r"autogen/",  # Auto-generated
]

# Security-critical files that need HEAVY analysis
_SECURITY_CRITICAL_PATTERNS = [
    r"(^|/)(auth|login|oauth|jwt|token|session|password|secret|crypto|encrypt|decrypt)",
    r"(^|/)(security|permission|access_control|rbac|policy)",
    r"(^|/)(db|database|connection|pool|transaction|migration)",
    r"(^|/)(config|settings|env|dotenv)",
    r"(^|/)(middleware|interceptor|filter|guard)",
    r"\.env(\.\w+)?$",  # Environment files
    r"secrets?\.",  # Secret files
    r"credentials?\.",  # Credential files
]

# Files that are tests, configs, or utilities — LIGHT analysis only
_LIGHT_PATTERNS = [
    r"(^|/)test_",  # Python tests
    r"(^|/)_test\.",  # Go-style tests
    r"(^|/)\.test\.",  # JS/TS tests
    r"(^|/)\.spec\.",  # JS/TS specs
    r"(^|/)conftest",  # Pytest fixtures
    r"(^|/)fixtures?/",  # Test fixtures
    r"(^|/)mock",  # Mock files
    r"(^|/)stub",  # Stub files
    r"(^|/)__init__\.py$",  # Package inits
    r"(^|/)utils?\.",  # Utility files
    r"(^|/)helpers?\.",  # Helper files
    r"(^|/)constants?\.",  # Constant files
    r"(^|/)types?\.",  # Type definitions
    r"(^|/)interfaces?\.",  # Interface definitions
    r"(^|/)schemas?\.",  # Schema files (not models)
    r"(^|/)validators?\.",  # Validator files
    r"(^|/)serializers?\.",  # Serializer files
    r"(^|/)formatters?\.",  # Formatter files
    r"(^|/)loggers?\.",  # Logger files
    r"(^|/)hooks?\.",  # Hook files
    r"(^|/)hooks/",  # Hook directories
    r"(^|/)scripts/",  # Script files
    r"(^|/)bin/",  # Binary scripts
    r"(^|/)cli\.",  # CLI entry points
    r"(^|/)main\.",  # Main entry points (usually simple)
    r"(^|/)app\.",  # App entry points
    r"(^|/)server\.",  # Server entry points
]

# Extensions that indicate LIGHT analysis
_LIGHT_EXTENSIONS = {
    ".cfg",
    ".ini",
    ".toml",
    ".yaml",
    ".yml",  # Config files
    ".json",
    ".xml",  # Data files
    ".sh",
    ".bash",  # Shell scripts
    ".bat",
    ".cmd",  # Windows scripts
    ".sql",  # SQL files (static analysis only)
    ".html",
    ".css",
    ".scss",
    ".sass",  # Frontend (basic review)
    ".md",
    ".rst",  # Documentation
}


def classify_file(file_path: Path) -> Tuple[str, str]:
    """
    Classify a file for analysis routing.

    Returns:
        (category, reason) where category is one of:
            "HEAVY"   — Security-critical, needs full analysis with large model
            "NORMAL"  — Standard business logic, skeleton mode with large model
            "LIGHT"   — Tests/configs/utils, small model, Wave 1 only
            "SKIP"    — Generated/vendor files, no analysis needed

        reason is a human-readable explanation.
    """
    filepath_str = str(file_path).replace("\\", "/")
    filename = file_path.name
    suffix = file_path.suffix.lower()

    # Check for generated/vendor files first
    for pattern in _GENERATED_PATTERNS:
        if re.search(pattern, filepath_str, re.IGNORECASE):
            return ("SKIP", f"Generated/vendor file (matched: {pattern})")

    # Check for security-critical files
    for pattern in _SECURITY_CRITICAL_PATTERNS:
        if re.search(pattern, filepath_str, re.IGNORECASE):
            return ("HEAVY", f"Security-critical file (matched: {pattern})")

    # Check for light analysis files
    for pattern in _LIGHT_PATTERNS:
        if re.search(pattern, filepath_str, re.IGNORECASE):
            return ("LIGHT", f"Test/config/utility file (matched: {pattern})")

    # Check for light extensions
    if suffix in _LIGHT_EXTENSIONS:
        return ("LIGHT", f"Config/data file type ({suffix})")

    # Default: NORMAL analysis
    return ("NORMAL", "Standard source file")


def get_analysis_config(category: str) -> dict:
    """
    Get analysis configuration for a file category.

    Returns dict with:
        use_skeleton: Whether to use skeleton mode
        use_oss_model: Whether to use OSS model (vs primary model)
        run_wave2: Whether to run Wave 2 analysis
        priority: Execution priority (lower = first)
    """
    configs = {
        "HEAVY": {
            "use_skeleton": False,
            "use_oss_model": False,
            "run_wave2": True,
            "priority": 1,
        },
        "NORMAL": {
            "use_skeleton": True,
            "use_oss_model": False,
            "run_wave2": True,
            "priority": 2,
        },
        "LIGHT": {
            "use_skeleton": True,
            "use_oss_model": True,
            "run_wave2": False,
            "priority": 3,
        },
        "SKIP": {
            "use_skeleton": False,
            "use_oss_model": False,
            "run_wave2": False,
            "priority": 99,
        },
    }
    return configs.get(category, configs["NORMAL"])


def classify_files(file_paths: list) -> dict:
    """
    Classify a batch of files and return grouped results.

    Args:
        file_paths: List of Path objects

    Returns:
        dict with keys: HEAVY, NORMAL, LIGHT, SKIP
        Each value is a list of (Path, category, reason) tuples
    """
    results = {"HEAVY": [], "NORMAL": [], "LIGHT": [], "SKIP": []}

    for fp in file_paths:
        category, reason = classify_file(fp)
        results[category].append((fp, category, reason))

    return results
