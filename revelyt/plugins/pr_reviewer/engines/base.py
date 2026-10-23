"""
engines/base.py — Standardized interfaces for all PR Review engines.

Every engine in the system MUST inherit from Engine and return List[Finding].
FileChunk carries rich metadata so downstream engines avoid recomputation.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """OWASP-aligned severity levels."""
    INFORMATIONAL = "informational"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FindingType(str, Enum):
    SECURITY = "security"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    BUG = "bug"
    STYLE = "style"
    DEPENDENCY = "dependency"
    SECRET = "secret"


class ChangeType(str, Enum):
    """Semantic classification of a diff hunk."""
    ADDED = "added"
    DELETED = "deleted"
    MODIFIED = "modified"
    RENAMED = "renamed"
    SIGNATURE_CHANGED = "signature_changed"
    LOGIC_CHANGED = "logic_changed"


# ---------------------------------------------------------------------------
# Core Data Models
# ---------------------------------------------------------------------------

class Finding(BaseModel):
    """Standardized finding output for all PR Review engines."""

    file: str
    line: int
    end_line: Optional[int] = None
    severity: Severity = Severity.MEDIUM
    type: FindingType = FindingType.QUALITY
    message: str
    suggestion: Optional[str] = None
    suggested_patch: Optional[str] = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    rule_id: str = "generic"
    engine: str = "unknown"
    owasp_ref: Optional[str] = None
    cwe_id: Optional[str] = None
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """Findings below 0.6 confidence are considered noise."""
        return self.confidence >= 0.6

    def fingerprint(self) -> str:
        """Unique hash for deduplication across commits."""
        raw = f"{self.file}:{self.line}:{self.rule_id}:{self.message}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class FileChunk(BaseModel):
    """Represents a logical chunk of a modified file within a PR diff."""

    file_path: str
    extension: str = ""
    language: str = "unknown"
    start_line: int = 1
    end_line: int = 1
    content: str = ""
    diff_content: str = ""
    added_lines: List[str] = Field(default_factory=list)
    deleted_lines: List[str] = Field(default_factory=list)
    context_lines: List[str] = Field(default_factory=list)

    # Enriched metadata (set by diff analyzers, reused by engines)
    risk_score: float = 0.0
    complexity: float = 0.0
    change_type: ChangeType = ChangeType.MODIFIED
    affected_functions: List[str] = Field(default_factory=list)
    affected_classes: List[str] = Field(default_factory=list)
    test_coverage_status: Optional[str] = None  # "covered" | "missing" | "unknown"
    historical_bug_weight: float = 0.0

    @property
    def total_changed_lines(self) -> int:
        return len(self.added_lines) + len(self.deleted_lines)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.diff_content.encode()).hexdigest()[:16]


class PRContext(BaseModel):
    """Full context assembled by the Context Retrieval Engine for AI prompts."""

    repo_name: str = ""
    pr_number: int = 0
    pr_title: str = ""
    pr_description: str = ""
    base_branch: str = "main"
    head_branch: str = ""
    author: str = ""
    file_tree: List[str] = Field(default_factory=list)
    symbol_definitions: Dict[str, str] = Field(default_factory=dict)
    caller_map: Dict[str, List[str]] = Field(default_factory=dict)
    related_tests: Dict[str, List[str]] = Field(default_factory=dict)
    cross_pr_history: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine ABC
# ---------------------------------------------------------------------------

class Engine(ABC):
    """
    Base interface for all analysis engines.

    Every engine (Rule, AI, AST, Security, Dependency, Hybrid) MUST
    implement ``run()`` and return a ``List[Finding]``.  The pipeline
    iterates engines in order and aggregates all findings.
    """

    name: str = "base"

    @abstractmethod
    def run(self, chunks: List[FileChunk], context: PRContext) -> List[Finding]:
        """Execute analysis and return findings."""
        ...

    def supports_language(self, language: str) -> bool:  # noqa: ARG002
        """Override to restrict engine to specific languages."""
        return True
