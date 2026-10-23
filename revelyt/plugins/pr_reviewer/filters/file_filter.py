"""
filters/file_filter.py — File type classification and filtering.

Drops binary/image/media files before they hit the analysis pipeline.
Classifies remaining files into scan tiers (deep, light, skip).
"""

from __future__ import annotations

import os
from enum import Enum
from typing import List, Set

from revelyt.plugins.pr_reviewer.engines.base import FileChunk


class ScanTier(str, Enum):
    """Determines how deeply a file should be analyzed."""
    DEEP = "deep"       # Full AST + AI + Security scan
    LIGHT = "light"     # Regex rules only, no AI
    SKIP = "skip"       # Drop entirely


# Extensions that should never be analyzed
_SKIP_EXTENSIONS: Set[str] = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".bmp", ".webp",
    ".mp4", ".mp3", ".wav", ".avi", ".mov",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".bin", ".exe", ".dll", ".so", ".dylib",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".pyc", ".pyo", ".class",
}

# Extensions that get light analysis only (no AI, basic rules)
_LIGHT_EXTENSIONS: Set[str] = {
    ".md", ".txt", ".rst", ".csv",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".html", ".css", ".scss", ".less",
    ".xml", ".svg",
}

# Deep analysis: actual source code
_DEEP_EXTENSIONS: Set[str] = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".java", ".rb", ".rs", ".c", ".cpp", ".h",
    ".cs", ".swift", ".kt", ".scala",
    ".sql", ".sh", ".bash", ".zsh",
    ".dockerfile",
}

# Files that are always important regardless of extension
_ALWAYS_DEEP_FILES: Set[str] = {
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    ".github/workflows", "Makefile", "Jenkinsfile",
    ".env", ".env.production", ".env.staging",
    "requirements.txt", "package.json", "Pipfile", "Cargo.toml",
    "go.mod", "Gemfile", "build.gradle",
}


class FileFilter:
    """
    Classifies and filters files before they enter the analysis pipeline.

    Usage::

        filt = FileFilter()
        processable = filt.filter(all_chunks)
        deep, light = filt.partition(processable)
    """

    def classify(self, file_path: str) -> ScanTier:
        """Determine the scan tier for a given file path."""
        basename = os.path.basename(file_path)
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        # Always-deep files
        if basename in _ALWAYS_DEEP_FILES:
            return ScanTier.DEEP
        for pattern in _ALWAYS_DEEP_FILES:
            if pattern in file_path:
                return ScanTier.DEEP

        # Extension-based classification
        if ext in _SKIP_EXTENSIONS:
            return ScanTier.SKIP
        if ext in _DEEP_EXTENSIONS:
            return ScanTier.DEEP
        if ext in _LIGHT_EXTENSIONS:
            return ScanTier.LIGHT

        # Unknown extensions default to light
        return ScanTier.LIGHT

    def filter(self, chunks: List[FileChunk]) -> List[FileChunk]:
        """Remove all chunks classified as SKIP."""
        return [c for c in chunks if self.classify(c.file_path) != ScanTier.SKIP]

    def partition(self, chunks: List[FileChunk]) -> tuple[List[FileChunk], List[FileChunk]]:
        """Split chunks into (deep_analysis, light_analysis) lists."""
        deep: List[FileChunk] = []
        light: List[FileChunk] = []
        for chunk in chunks:
            tier = self.classify(chunk.file_path)
            if tier == ScanTier.DEEP:
                deep.append(chunk)
            elif tier == ScanTier.LIGHT:
                light.append(chunk)
        return deep, light
