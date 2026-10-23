import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class ProjectDetector:
    """
    Phase 1: Detector.
    Identifies the primary language and framework to gate analysis chains.

    Key fix: Python and Node.js checks are independent (no elif) so monorepos
    or projects with both Python and JS config files are detected correctly.
    """
    def __init__(self, project_root: Path):
        self.project_root = project_root

    def detect(self) -> Dict[str, str]:
        """
        Returns a dict with:
          language           — primary language ("python", "javascript/typescript", "unknown")
          framework          — primary framework
          secondary_language — second language if both detected (e.g. "javascript/typescript")
          secondary_framework
        """
        results = {
            "language": "unknown",
            "framework": "none",
            "secondary_language": "none",
            "secondary_framework": "none",
        }

        has_python = (
            any(self.project_root.glob("**/requirements.txt")) or
            any(self.project_root.glob("**/pyproject.toml")) or
            any(self.project_root.glob("**/Pipfile")) or
            any(self.project_root.glob("**/setup.cfg")) or
            any(self.project_root.glob("**/*.py"))
        )

        has_node = (
            any(self.project_root.glob("**/package.json")) or
            any(self.project_root.glob("**/tsconfig.json")) or
            any(self.project_root.glob("**/*.js")) or
            any(self.project_root.glob("**/*.ts")) or
            any(self.project_root.glob("**/*.jsx")) or
            any(self.project_root.glob("**/*.tsx"))
        )

        # Determine primary language — Python wins if both present (most common case)
        if has_python:
            results["language"] = "python"
            results["framework"] = self._detect_python_framework()
            if has_node:
                results["secondary_language"] = "javascript/typescript"
                results["secondary_framework"] = self._detect_node_framework()
        elif has_node:
            results["language"] = "javascript/typescript"
            results["framework"] = self._detect_node_framework()

        return results

    def _detect_python_framework(self) -> str:
        if self._has_signature(["import flask", "from flask"]):
            return "flask"
        if self._has_signature(["import django", "from django", "django.conf"]):
            return "django"
        if self._has_signature(["import fastapi", "from fastapi"]):
            return "fastapi"
        if self._has_signature(["import tornado", "from tornado"]):
            return "tornado"
        if self._has_signature(["import aiohttp", "from aiohttp"]):
            return "aiohttp"
        return "none"

    def _detect_node_framework(self) -> str:
        if self._has_signature(["require('express')", 'require("express")', "from 'express'", 'from "express"']):
            return "express"
        if self._has_signature(["@nestjs/core", "@nestjs/common"]):
            return "nestjs"
        if self._has_signature(["from 'next'", 'from "next"', "next/app", "next/router"]):
            return "nextjs"
        if self._has_signature(["createApp", "defineComponent", "from 'vue'", 'from "vue"']):
            return "vue"
        if self._has_signature(["from 'nuxt'", 'from "nuxt"', "defineNuxtConfig"]):
            return "nuxt"
        if self._has_signature(["from 'react'", 'from "react"', "React.createElement"]):
            return "react"
        return "none"

    def _has_signature(self, keywords: list) -> bool:
        """Checks priority entry-point files first, then falls back to full scan."""
        priority_files = [
            "app.py", "main.py", "app.ts", "main.ts",
            "server.js", "server.ts", "index.js", "index.ts",
            "app.js", "next.config.js", "next.config.ts",
        ]

        for name in priority_files:
            for path in self.project_root.rglob(name):
                if self._check_file(path, keywords):
                    return True

        for path in self.project_root.rglob("*"):
            if path.suffix in ('.py', '.js', '.ts', '.jsx', '.tsx', '.mjs', '.cjs') and path.is_file():
                if self._check_file(path, keywords):
                    return True
        return False

    def _check_file(self, path: Path, keywords: list) -> bool:
        try:
            content = path.read_text(errors='ignore').lower()
            return any(k.lower() in content for k in keywords)
        except Exception:
            return False