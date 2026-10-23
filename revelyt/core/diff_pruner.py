import ast
import re
from pathlib import Path
from typing import Set, List, Optional
import logging
import subprocess

logger = logging.getLogger(__name__)

SKIP_PATTERNS = [
    r"package-lock\.json", r"yarn\.lock", r"poetry\.lock",
    r"\.min\.js$", r"_pb2\.py$",
    r"migrations/\d+_",
]

class DiffPruner:
    """
    AST-aware diff pruner. Extracts only the modified functions/classes 
    (with minimal context) from a file, reducing token sizes by up to 65%.
    """

    def __init__(self, repo_path: Path, base: str = "main", head: str = "HEAD"):
        self.repo_path = repo_path
        self.base = base
        self.head = head

    def should_skip(self, file_path: str) -> bool:
        for p in SKIP_PATTERNS:
            if re.search(p, file_path):
                return True
        return False

    def get_changed_lines(self, file_path: Path) -> Set[int]:
        """Runs git diff to find which lines changed in the target file."""
        try:
            rel_path = str(file_path.relative_to(self.repo_path))
            # try 3-dot first
            for separator in ("...", ".."):
                try:
                    result = subprocess.run(
                        ["git", "diff", "-U0", f"{self.base}{separator}{self.head}", "--", rel_path],
                        cwd=str(self.repo_path),
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    break
                except subprocess.CalledProcessError:
                    if separator == "..":
                        raise
            
            changed_lines = set()
            for line in result.stdout.splitlines():
                if line.startswith("@@"):
                    match = re.search(r"\+([0-9]+)(?:,([0-9]+))?", line)
                    if match:
                        start = int(match.group(1))
                        count = int(match.group(2)) if match.group(2) else 1
                        # Git diff line ranges: +[start],[count]
                        # "count" defaults to 1 if omitted.
                        for i in range(start, start + count):
                            changed_lines.add(i)
            return changed_lines
        except Exception as e:
            logger.warning("Could not get changed lines for %s: %s", file_path.name, e)
            return set()

    def prune(self, file_path: Path, file_content: str) -> str:
        """
        Prune the file content down to only the functions/classes that encompass 
        the changed lines. If it's not a Python file or AST parsing fails, 
        returns the unified diff patch.
        """
        if self.should_skip(file_path.name):
            return file_content

        changed_lines = self.get_changed_lines(file_path)
        if not changed_lines:
            return file_content  # For impacted files (no core changes), return full file

        if file_path.suffix != ".py":
            return self._get_hunks_only(file_path, file_content)

        try:
            tree = ast.parse(file_content)
        except SyntaxError:
            return self._get_hunks_only(file_path, file_content)

        nodes_to_keep: Set[ast.AST] = set()
        container_nodes: List[ast.AST] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    container_nodes.append(node)
        
        container_nodes.sort(key=lambda n: n.end_lineno - n.lineno)

        lines_covered = set()
        
        for line in changed_lines:
            if line in lines_covered:
                continue
            
            found_container = False
            for node in container_nodes:
                if node.lineno <= line <= node.end_lineno:
                    nodes_to_keep.add(node)
                    for l in range(node.lineno, node.end_lineno + 1):
                        lines_covered.add(l)
                    found_container = True
                    break
            
            if not found_container:
                # Changes outside any function/class (e.g. module level import)
                # If there are module level changes, fallback to hunk-based view
                # because it's hard to extract just pure module-level expressions safely
                return self._get_hunks_only(file_path, file_content)

        if not nodes_to_keep:
            return self._get_hunks_only(file_path, file_content)

        return self._reconstruct_file(file_content, nodes_to_keep)

    def _reconstruct_file(self, file_content: str, nodes_to_keep: Set[ast.AST]) -> str:
        all_lines = file_content.splitlines(keepends=True)
        pruned_lines: List[str] = []
        
        for i, line in enumerate(all_lines):
            if line.startswith("import ") or line.startswith("from "):
                pruned_lines.append(line)
        
        pruned_lines.append("\n# ... [pruned un-changed code] ...\n\n")

        sorted_nodes = sorted(list(nodes_to_keep), key=lambda n: getattr(n, 'lineno', 0))
        
        last_end = -1
        for node in sorted_nodes:
            # ast is 1-indexed, list is 0-indexed
            start_context = max(0, node.lineno - 3)
            end_context = min(len(all_lines), node.end_lineno + 2)
            
            start_idx = max(start_context, last_end)
            if start_idx > last_end and last_end != -1:
                pruned_lines.append("\n# ... [pruned] ...\n\n")
                
            for i in range(start_idx, end_context):
                pruned_lines.append(all_lines[i]) # type: ignore
                
            last_end = max(last_end, end_context)

        return "".join(pruned_lines)

    def _get_hunks_only(self, file_path: Path, file_content: str) -> str:
        """Fallback: run git diff -U5 and return the patch."""
        try:
            rel_path = str(file_path.relative_to(self.repo_path))
            for separator in ("...", ".."):
                try:
                    result = subprocess.run(
                        ["git", "diff", "-U5", f"{self.base}{separator}{self.head}", "--", rel_path],
                        cwd=str(self.repo_path),
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    patch = result.stdout.strip()
                    if patch:
                        return f"// DIFF HUNKS ONLY (Full file omitted for brevity):\n{patch}"
                    break
                except subprocess.CalledProcessError:
                    if separator == "..":
                        raise
            return file_content
        except Exception:
            return file_content
