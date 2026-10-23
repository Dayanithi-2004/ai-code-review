"""
github_client.py — Production PyGithub wrapper for PR operations.

Handles: fetching PR diff, metadata, file lists, posting inline comments,
creating review summaries, and submitting GitHub Check Runs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from github import Github, GithubException
from github.PullRequest import PullRequest
from github.Repository import Repository

logger = logging.getLogger(__name__)


class GitHubClient:
    """
    Wraps PyGithub to provide a clean interface for PR review operations.

    Usage::

        client = GitHubClient(token="ghp_xxx")
        diff = client.get_pr_diff("owner/repo", 42)
        client.post_review_comment("owner/repo", 42, "file.py", 10, "Issue here")
    """

    def __init__(self, token: str, base_url: Optional[str] = None):
        """
        Parameters
        ----------
        token : str
            GitHub personal access token or GitHub App token.
        base_url : str, optional
            GitHub Enterprise base URL. Omit for github.com.
        """
        if base_url:
            self._gh = Github(base_url=base_url, login_or_token=token)
        else:
            self._gh = Github(login_or_token=token)

    # ------------------------------------------------------------------
    # Repository / PR access
    # ------------------------------------------------------------------

    def get_repo(self, repo_name: str) -> Repository:
        return self._gh.get_repo(repo_name)

    def get_pr(self, repo_name: str, pr_number: int) -> PullRequest:
        repo = self.get_repo(repo_name)
        return repo.get_pull(pr_number)

    def get_pr_diff(self, repo_name: str, pr_number: int) -> str:
        """Fetch the raw unified diff for a PR."""
        import requests

        pr = self.get_pr(repo_name, pr_number)
        headers = {
            "Authorization": f"token {self._gh._Github__requester._Requester__authorizationHeader.split(' ')[1] if hasattr(self._gh, '_Github__requester') else ''}",
            "Accept": "application/vnd.github.v3.diff",
        }
        # Use the diff_url provided by GitHub API
        resp = requests.get(pr.diff_url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.text

    def get_pr_metadata(self, repo_name: str, pr_number: int) -> Dict[str, Any]:
        """Fetch structured metadata for the PR."""
        pr = self.get_pr(repo_name, pr_number)
        return {
            "title": pr.title,
            "description": pr.body or "",
            "author": pr.user.login,
            "base_branch": pr.base.ref,
            "head_branch": pr.head.ref,
            "head_sha": pr.head.sha,
            "changed_files": pr.changed_files,
            "additions": pr.additions,
            "deletions": pr.deletions,
            "labels": [l.name for l in pr.labels],
            "state": pr.state,
            "mergeable": pr.mergeable,
        }

    def get_changed_files(self, repo_name: str, pr_number: int) -> List[str]:
        """Return list of file paths changed in the PR."""
        pr = self.get_pr(repo_name, pr_number)
        return [f.filename for f in pr.get_files()]

    # ------------------------------------------------------------------
    # Posting reviews
    # ------------------------------------------------------------------

    def post_review_comment(
        self,
        repo_name: str,
        pr_number: int,
        file_path: str,
        line: int,
        body: str,
        commit_sha: Optional[str] = None,
    ) -> None:
        """Post an inline review comment on a specific line of a file."""
        try:
            pr = self.get_pr(repo_name, pr_number)
            commit = pr.head.sha if commit_sha is None else commit_sha
            repo = self.get_repo(repo_name)
            commit_obj = repo.get_commit(commit)
            pr.create_review_comment(
                body=body,
                commit=commit_obj,
                path=file_path,
                line=line,
            )
        except GithubException as e:
            logger.error("Failed to post review comment on %s:%d — %s", file_path, line, e)

    def post_pr_comment(self, repo_name: str, pr_number: int, body: str) -> None:
        """Post a top-level comment (e.g., the PR Summary)."""
        try:
            pr = self.get_pr(repo_name, pr_number)
            pr.create_issue_comment(body)
        except GithubException as e:
            logger.error("Failed to post PR comment — %s", e)

    def submit_review(
        self,
        repo_name: str,
        pr_number: int,
        body: str,
        event: str = "COMMENT",
        comments: Optional[List[Dict]] = None,
    ) -> None:
        """
        Submit a full review with optional inline comments.

        Parameters
        ----------
        event : str
            One of 'APPROVE', 'REQUEST_CHANGES', 'COMMENT'.
        comments : list of dict
            Each dict has keys: path, position (or line), body.
        """
        try:
            pr = self.get_pr(repo_name, pr_number)
            pr.create_review(
                body=body,
                event=event,
                comments=comments or [],
            )
        except GithubException as e:
            logger.error("Failed to submit review — %s", e)

    # ------------------------------------------------------------------
    # GitHub Checks API
    # ------------------------------------------------------------------

    def create_check_run(
        self,
        repo_name: str,
        head_sha: str,
        name: str = "Revelyt PR Review",
        status: str = "in_progress",
        conclusion: Optional[str] = None,
        title: str = "",
        summary: str = "",
        annotations: Optional[List[Dict]] = None,
    ) -> int:
        """
        Create or update a GitHub Check Run for richer UI integration.

        Returns the check run ID.
        """
        try:
            repo = self.get_repo(repo_name)
            output = {"title": title or name, "summary": summary}
            if annotations:
                # GitHub limits to 50 annotations per API call
                output["annotations"] = annotations[:50]

            kwargs: Dict[str, Any] = {
                "name": name,
                "head_sha": head_sha,
                "status": status,
                "output": output,
            }
            if conclusion:
                kwargs["conclusion"] = conclusion

            check_run = repo.create_check_run(**kwargs)
            return check_run.id
        except GithubException as e:
            logger.error("Failed to create check run — %s", e)
            return -1

    def update_check_run(
        self,
        repo_name: str,
        check_run_id: int,
        status: str = "completed",
        conclusion: str = "neutral",
        title: str = "",
        summary: str = "",
        annotations: Optional[List[Dict]] = None,
    ) -> None:
        """Update an existing Check Run (e.g., mark completed)."""
        try:
            repo = self.get_repo(repo_name)
            check_run = repo.get_check_run(check_run_id)
            output = {"title": title, "summary": summary}
            if annotations:
                output["annotations"] = annotations[:50]
            check_run.edit(
                status=status,
                conclusion=conclusion,
                output=output,
            )
        except GithubException as e:
            logger.error("Failed to update check run — %s", e)
