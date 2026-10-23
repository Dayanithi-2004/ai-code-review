import logging
import subprocess
from pathlib import Path
from typing import Set

from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


class DiffEngine:
    """
    Git-aware diff engine for incremental scanning.

    Identifies which files changed between two git refs so the analysis
    chains can focus on what matters — like a Senior Architect reviewing
    a Pull Request instead of re-reading the entire codebase.
    """

    @staticmethod
    def get_changed_files(
        repo_path: Path,
        base: str = "main",
        head: str = "HEAD",
    ) -> Set[str]:
        """
        Returns project-relative paths of files that changed between two refs.

        Uses --diff-filter=ACMR to include Added, Copied, Modified, Renamed
        files only.  Deleted files are excluded — there's nothing to analyse.

        Args:
            repo_path: Root of the git repository (the cloned/copied project).
            base:      Git ref to diff against (branch, tag, or commit). Default: "main".
            head:      Git ref for the current state. Default: "HEAD".

        Returns:
            Set of project-relative file paths, e.g. {"src/auth.py", "lib/utils.py"}.

        Raises:
            ValueError: If ``repo_path`` is not a git repository or ``base`` is invalid.
        """
        # ── Guard: is this actually a git repo? ──────────────────────────
        if not (repo_path / ".git").is_dir():
            logger.error("Not a git repository: %s", repo_path)
            raise ValueError(
                f"'{repo_path}' is not a git repository (no .git directory). "
                f"The --diff flag requires a git history to compute the changeset."
            )

        # ── Guard: does the base ref exist? ──────────────────────────────
        try:
            subprocess.run(
                ["git", "rev-parse", "--verify", base],
                cwd=str(repo_path),
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            raise ValueError(
                f"Git ref '{base}' does not exist in the repository. "
                f"Available branches can be listed with: git branch -a"
            )

        # ── Run the diff ─────────────────────────────────────────────────
        # Three-dot diff (`base...head`) finds the merge-base automatically,
        # which is the correct behaviour for PR-style reviews.
        # Two-dot diff (`base..head`) is used as fallback if three-dot fails
        # (e.g. unrelated histories).
        changed_files: Set[str] = set()

        for separator in ("...", ".."):
            try:
                result = subprocess.run(
                    [
                        "git", "diff",
                        "--name-only",
                        "--diff-filter=ACMR",
                        f"{base}{separator}{head}",
                    ],
                    cwd=str(repo_path),
                    check=True,
                    capture_output=True,
                    text=True,
                )
                changed_files = {
                    line.strip()
                    for line in result.stdout.strip().splitlines()
                    if line.strip()
                }
                logger.info("git diff %s%s%s: %d files changed", base, separator, head, len(changed_files))
                break  # success — no need to try the fallback

            except subprocess.CalledProcessError:
                if separator == "...":
                    logger.debug("Three-dot diff failed, trying two-dot fallback")
                    continue  # try two-dot fallback
                logger.error("git diff failed for %s..%s", base, head)
                raise ValueError(
                    f"git diff failed for '{base}..{head}'. "
                    f"Check that both refs exist and share history."
                )

        return changed_files
