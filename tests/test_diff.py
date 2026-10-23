"""
Tests for Phase 1.1: Git-Aware Incremental Diff Scanning.

Covers:
  - DiffEngine.get_changed_files() with a real temp git repo
  - CodeMapper.get_impacted_files() with seeded dep_graph data
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from revelyt.core.diff import DiffEngine
from revelyt.core.mapper import CodeMapper
from revelyt.core.session import SessionManager


# Fixtures

@pytest.fixture
def git_repo(tmp_path):
    """
    Creates a minimal git repo with two branches:
      main   → contains base.py, shared.py
      feature → adds new.py, modifies base.py (shared.py unchanged)
    """
    repo = tmp_path / "repo"
    repo.mkdir()

    def _run(*args):
        subprocess.run(
            ["git"] + list(args),
            cwd=str(repo),
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t.com",
                 "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t.com"},
        )

    # Initialise repo on 'main'
    _run("init", "-b", "main")
    (repo / "base.py").write_text("# base v1\n")
    (repo / "shared.py").write_text("# shared v1\n")
    _run("add", ".")
    _run("commit", "-m", "initial commit")

    # Create feature branch with changes
    _run("checkout", "-b", "feature")
    (repo / "base.py").write_text("# base v2 — modified\n")
    (repo / "new.py").write_text("# brand new file\n")
    _run("add", ".")
    _run("commit", "-m", "feature changes")

    return repo


@pytest.fixture
def session():
    """Fresh in-memory SessionManager for mapper tests."""
    return SessionManager()


# ─────────────────────────────────────────────────────────────────────────────
# DiffEngine tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDiffEngine:

    def test_get_changed_files(self, git_repo):
        """Changed files between main and feature branch are detected."""
        changed = DiffEngine.get_changed_files(git_repo, base="main", head="feature")
        assert "base.py" in changed, "Modified file should appear in diff"
        assert "new.py" in changed, "Added file should appear in diff"
        assert "shared.py" not in changed, "Unchanged file should NOT appear"

    def test_get_changed_files_not_a_git_repo(self, tmp_path):
        """Non-git directory raises ValueError."""
        plain_dir = tmp_path / "not_a_repo"
        plain_dir.mkdir()
        with pytest.raises(ValueError, match="not a git repository"):
            DiffEngine.get_changed_files(plain_dir, base="main")

    def test_get_changed_files_bad_ref(self, git_repo):
        """Nonexistent branch name raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            DiffEngine.get_changed_files(git_repo, base="nonexistent-branch")

    def test_deleted_files_excluded(self, git_repo):
        """
        Files that were deleted in the head ref should NOT appear
        in the changeset (nothing to analyse for a deleted file).
        """
        # Delete shared.py on the feature branch
        subprocess.run(
            ["git", "checkout", "feature"],
            cwd=str(git_repo), check=True, capture_output=True,
        )
        (git_repo / "shared.py").unlink()
        subprocess.run(
            ["git", "add", "-A"],
            cwd=str(git_repo), check=True, capture_output=True,
            env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t.com",
                 "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t.com"},
        )
        subprocess.run(
            ["git", "commit", "-m", "delete shared.py"],
            cwd=str(git_repo), check=True, capture_output=True,
            env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t.com",
                 "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t.com"},
        )

        changed = DiffEngine.get_changed_files(git_repo, base="main", head="feature")
        assert "shared.py" not in changed, "Deleted files must be excluded (--diff-filter=ACMR)"

    def test_empty_diff(self, git_repo):
        """Diffing a branch against itself returns an empty set."""
        changed = DiffEngine.get_changed_files(git_repo, base="main", head="main")
        assert changed == set()


# ─────────────────────────────────────────────────────────────────────────────
# CodeMapper.get_impacted_files tests
# ─────────────────────────────────────────────────────────────────────────────

class TestImpactAnalysis:

    def _seed_dep_graph(self, session, edges):
        """Helper: insert dependency edges into the session database."""
        with session.get_connection() as conn:
            for source, target in edges:
                conn.execute(
                    "INSERT OR IGNORE INTO dep_graph (source_file, target_file) VALUES (?, ?)",
                    (source, target),
                )

    def test_get_impacted_files(self, session, tmp_path):
        """Changed file + its reverse dependencies are returned."""
        # Dependency graph:
        #   payments.py → auth.py  (payments imports auth)
        #   profile.py  → auth.py  (profile imports auth)
        #   auth.py     → utils.py (auth imports utils)
        self._seed_dep_graph(session, [
            ("payments.py", "auth.py"),
            ("profile.py", "auth.py"),
            ("auth.py", "utils.py"),
        ])

        mapper = CodeMapper(session, tmp_path)
        impacted = mapper.get_impacted_files({"auth.py"})

        # auth.py changed → payments.py and profile.py import it → both impacted
        assert "auth.py" in impacted, "Changed file itself must be in the set"
        assert "payments.py" in impacted, "payments.py imports auth.py"
        assert "profile.py" in impacted, "profile.py imports auth.py"
        # utils.py is imported BY auth.py, not a reverse dep
        assert "utils.py" not in impacted, "utils.py is a forward dep, not a reverse dep"

    def test_get_impacted_files_no_deps(self, session, tmp_path):
        """File with zero dependants returns just the changed file."""
        self._seed_dep_graph(session, [
            ("auth.py", "utils.py"),
        ])

        mapper = CodeMapper(session, tmp_path)
        impacted = mapper.get_impacted_files({"standalone.py"})

        assert impacted == {"standalone.py"}

    def test_get_impacted_files_empty_input(self, session, tmp_path):
        """Empty changeset returns empty set."""
        mapper = CodeMapper(session, tmp_path)
        assert mapper.get_impacted_files(set()) == set()
