"""
ProjectLoader — Phase 1: Ingestion.

Manifest split:
  get_file_manifest()    → ANALYSIS chains (Wave 1, Wave 2).
                           Excludes tests/, alembic/, migrations/, and all
                           generated/tooling dirs. Only production source code.

  get_summary_manifest() → PM agent summarisation.
                           Also excludes tests/, alembic/, migrations/.
                           Test and migration evidence for BRD verification is
                           provided by ProductManagerAgent._detect_infra_for_brd(),
                           a zero-token filesystem probe that runs separately and
                           injects synthetic summary entries — no LLM call needed.

Why tests are excluded from BOTH manifests:
  Test files contain scaffolding, fixtures, and mocks that are irrelevant to
  architectural review. Summarising them wastes tokens and produces noise.
  The BRD verifier gets test-framework evidence through the filesystem probe
  in ProductManagerAgent instead.
"""

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Generator, Tuple, Optional
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

_HTTPS_SCHEME = "https://"

# Matches GitHub/GitLab/Bitbucket web-UI paths like /tree/<branch> or /blob/<branch>
_TREE_BLOB_RE = re.compile(r"(/tree/|/blob/)(.+?)/?$")


class ProjectLoader:
    """
    Phase 1: Ingestion.
    Handles loading the codebase from local or remote sources and applies triage filters.
    Supports Python and Node.js/TypeScript projects.
    """

    def __init__(self, sandbox_dir: Path):
        self.sandbox_dir = sandbox_dir
        self.project_root = sandbox_dir / "repo"

        # ── Dirs excluded from ANALYSIS chains ────────────────────────────────
        # Quality, security, and performance agents never see these.
        # Tests and migrations have no analysis value and waste tokens.
        self.analysis_ignored_dirs = {
            # Python
            'node_modules', '.venv', 'venv', 'env', '.git', '__pycache__',
            'dist', 'build', 'bin', 'obj', '.idea', '.vscode', '.pytest_cache',
            'alembic', 'migrations', 'tests', 'test',
            '.egg-info', '.eggs', 'htmlcov', '.mypy_cache', '.ruff_cache', '.github',
            # Node.js / frontend
            '.next', '.nuxt', 'coverage', '.turbo', 'out', '.svelte-kit',
            'storybook-static', '.parcel-cache', '.cache', '__tests__',
        }

        # ── Dirs excluded from EVERYTHING ─────────────────────────────────────
        # Tests and migrations are intentionally in this set.
        # BRD evidence for testing/migration requirements is gathered by
        # ProductManagerAgent._detect_infra_for_brd() — a zero-token filesystem
        # probe that does NOT summarise file content, so no tokens are wasted.
        self.always_ignored_dirs = {
            'node_modules', '.venv', 'venv', 'env', '.git', '__pycache__',
            'dist', 'build', 'bin', 'obj', '.idea', '.vscode', '.pytest_cache',
            '.egg-info', '.eggs', 'htmlcov', '.mypy_cache', '.ruff_cache',
            '.next', '.nuxt', 'coverage', '.turbo', 'out', '.svelte-kit',
            'storybook-static', '.parcel-cache', '.cache',
            # Test directories — excluded from summarisation.
            # BRD evidence injected via filesystem probe in ProductManagerAgent.
            'tests', 'test', '__tests__', 'spec', 'specs', 'e2e',
            # Migration directories — same rationale.
            'alembic', 'migrations', 'migrate',
        }

        self.ignored_extensions = {
            # Binaries and media
            '.exe', '.pyc', '.pyo', '.png', '.jpg', '.jpeg', '.gif', '.svg',
            '.ico', '.pdf', '.zip', '.tar', '.gz', '.db', '.sqlite', '.wasm',
            '.ttf', '.woff', '.woff2', '.eot', '.mp4', '.mp3', '.webp',
            # Lock files — huge, zero analysis value
            '.lock',
        }

        # Filenames always skipped regardless of extension
        self.ignored_filenames = {
            'package-lock.json',
            'yarn.lock',
            'pnpm-lock.yaml',
            'poetry.lock',
            'alembic.ini',
            'requirements.txt',
            'pyproject.toml',
            'Pipfile.lock',
            # Developer tooling config — irrelevant to code quality review
            '.gitignore',
            '.gitattributes',
            '.dockerignore',
            '.editorconfig',
            '.prettierrc',
            '.prettierrc.json',
            '.prettierrc.js',
            '.eslintrc',
            '.eslintrc.json',
            '.eslintrc.js',
            '.eslintrc.cjs',
            '.eslintignore',
            '.babelrc',
            'babel.config.js',
            'babel.config.ts',
            'jest.config.js',
            'jest.config.ts',
            'vitest.config.ts',
            'vitest.config.js',
            'playwright.config.ts',
            'cypress.config.ts',
            'tsconfig.json',
            'jsconfig.json',
            'webpack.config.js',
            'vite.config.ts',
            'vite.config.js',
            'rollup.config.js',
            'tailwind.config.js',
            'tailwind.config.ts',
            'postcss.config.js',
            'next.config.js',
            'next.config.ts',
            'nuxt.config.ts',
            'Makefile',
            'Dockerfile',
            'docker-compose.yml',
            'docker-compose.yaml',
        }

        self.minified_suffixes = {'.min.js', '.min.css', '.min.mjs'}

    # ------------------------------------------------------------------
    # Source loading
    # ------------------------------------------------------------------

    def load(self, source: str, token: str = None) -> Path:
        """Entry point to clone or copy the project into the sandbox."""
        if source.startswith(("http://", _HTTPS_SCHEME, "git@")):
            return self._clone_github(source, token)
        else:
            return self._copy_local(source)

    def _clone_github(self, url: str, token: str = None) -> Path:
        clone_url, branch = self._sanitize_git_url(url)
        logger.info("Cloning repository: %s (branch: %s)", clone_url, branch or "default")
        console.print(f"🌐 [blue]Cloning repository:[/blue] {clone_url}")

        if token and clone_url.startswith(_HTTPS_SCHEME):
            clone_url = clone_url.replace(_HTTPS_SCHEME, f"{_HTTPS_SCHEME}{token}@")

        cmd = ["git", "clone", "--depth", "1"]
        if branch:
            cmd += ["--branch", branch]
        cmd += [clone_url, str(self.project_root)]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Clone complete: %s", self.project_root)
            return self.project_root
        except subprocess.CalledProcessError as e:
            logger.error("Error cloning repo: %s", e.stderr, exc_info=True)
            console.print(f"[bold red]Error cloning repo:[/bold red] {e.stderr}")
            raise

    @staticmethod
    def _sanitize_git_url(url: str) -> Tuple[str, Optional[str]]:
        """
        Strip GitHub/GitLab web-UI path segments (/tree/<branch>, /blob/<branch>)
        from a URL and return (clone_url, branch | None).

        Examples
        --------
        >>> ProjectLoader._sanitize_git_url("https://github.com/user/repo/tree/master")
        ('https://github.com/user/repo', 'master')
        >>> ProjectLoader._sanitize_git_url("https://github.com/user/repo")
        ('https://github.com/user/repo', None)
        """
        match = _TREE_BLOB_RE.search(url)
        if match:
            branch = match.group(2)
            clone_url = url[:match.start()]
            return clone_url, branch
        return url, None

    def _copy_local(self, path: str) -> Path:
        src_path = Path(path).resolve()
        if not src_path.exists():
            raise FileNotFoundError(f"Local path not found: {src_path}")

        logger.info("Loading local project from: %s", src_path)
        console.print(f"📂 [blue]Loading local project from:[/blue] {src_path}")
        shutil.copytree(
            src_path, self.project_root,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*self.always_ignored_dirs),
        )
        return self.project_root

    # ------------------------------------------------------------------
    # Manifests
    # ------------------------------------------------------------------

    def get_file_manifest(self) -> Generator[Path, None, None]:
        """
        ANALYSIS manifest — Wave 1 and Wave 2 quality/security agents.
        Production source code only. No tests, migrations, generated dirs.
        """
        yield from self._walk(self.analysis_ignored_dirs)

    def get_summary_manifest(self) -> Generator[Path, None, None]:
        """
        SUMMARY manifest — PM agent module summarisation.
        Also excludes tests/ and migrations/ — test/migration evidence for
        BRD verification is injected by ProductManagerAgent._detect_infra_for_brd()
        as zero-token synthetic summary entries, not by summarising file content.
        """
        yield from self._walk(self.always_ignored_dirs)

    # ------------------------------------------------------------------
    # Shared walker
    # ------------------------------------------------------------------

    def _walk(self, ignored_dirs: set) -> Generator[Path, None, None]:
        """Shared directory walker with configurable ignore set."""
        for root, dirs, files in os.walk(self.project_root):
            # Prune in-place — prevents os.walk descending into ignored dirs
            dirs[:] = [d for d in dirs if d not in ignored_dirs]

            for file in files:
                file_path = Path(root) / file

                if file_path.name in self.ignored_filenames:
                    continue

                if file_path.suffix.lower() in self.ignored_extensions:
                    continue

                name_lower = file_path.name.lower()
                if any(name_lower.endswith(s) for s in self.minified_suffixes):
                    continue

                # Skip individual test files that leak outside test directories
                # (e.g. test_utils.py sitting next to utils.py in a source dir)
                if self._is_test_filename(file_path.name):
                    continue

                if file_path.stat().st_size > 500 * 1024 * 1024:
                    continue

                yield file_path

    @staticmethod
    def _is_test_filename(name: str) -> bool:
        """
        Return True for test files that sit outside a dedicated test directory.
        Covers pytest (test_*.py / *_test.py), Jest/Vitest (*.test.* / *.spec.*),
        and conftest.py.
        """
        lower = name.lower()
        # Python: test_foo.py, foo_test.py, conftest.py
        if lower.startswith('test_') and lower.endswith('.py'):
            return True
        if lower.endswith('_test.py'):
            return True
        if lower == 'conftest.py':
            return True
        # JS/TS: foo.test.ts, foo.spec.ts, foo.test.js, foo.spec.js
        for ext in ('.test.ts', '.test.tsx', '.test.js', '.test.jsx',
                    '.spec.ts', '.spec.tsx', '.spec.js', '.spec.jsx'):
            if lower.endswith(ext):
                return True
        return False