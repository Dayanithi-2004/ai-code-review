"""
function_cache — Function/class-level caching for Revelyt.

Instead of caching entire files, this caches findings per function/class.
If only 2 functions changed in a 50-function file, only re-analyze those 2.

How it works:
1. Parse file into function/class blocks with line ranges
2. Hash each block's content independently
3. Check cache for each block
4. Only send uncached blocks to LLM
5. Merge cached + new findings
"""

import ast
import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from revelyt.core.schemas import FindingList, Finding

logger = logging.getLogger(__name__)


class FunctionCache:
    """
    Cache findings at function/class granularity.

    Cache key: hash(role_name + file_path + function_name + function_content + model_name + prohibition_hash)
    """

    def __init__(self, target_dir: Path):
        self.cache_file = target_dir / ".revelyt_func_cache.json"
        self._cache: Dict[str, list] = self._load()
        self._lock = threading.Lock()

    def _load(self) -> dict:
        if not self.cache_file.exists():
            return {}
        try:
            with open(self.cache_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not load function cache: %s", e)
            return {}

    def _save(self) -> None:
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self._cache, f)
        except Exception as e:
            logger.warning("Could not save function cache: %s", e)

    def _get_key(
        self,
        role_name: str,
        file_path: str,
        func_name: str,
        func_content: str,
        model_name: str,
        prohibition_hash: str,
    ) -> str:
        payload = f"{role_name}::{file_path}::{func_name}::{func_content}::{model_name}::{prohibition_hash}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(
        self,
        role_name: str,
        file_path: str,
        func_name: str,
        func_content: str,
        model_name: str,
        prohibition_hash: str,
    ) -> Optional[FindingList]:
        key = self._get_key(
            role_name, file_path, func_name, func_content, model_name, prohibition_hash
        )
        if key in self._cache:
            data = self._cache[key]
            try:
                return FindingList(findings=[Finding(**f) for f in data])
            except Exception:
                return None
        return None

    def set(
        self,
        role_name: str,
        file_path: str,
        func_name: str,
        func_content: str,
        result: FindingList,
        model_name: str,
        prohibition_hash: str,
    ) -> None:
        key = self._get_key(
            role_name, file_path, func_name, func_content, model_name, prohibition_hash
        )
        with self._lock:
            self._cache[key] = [f.model_dump() for f in result.findings]
            self._save()


def extract_function_blocks(code: str, file_path: str) -> List[Dict]:
    """
    Extract function/class blocks from source code with line ranges.

    Returns list of dicts:
        {
            "name": "function_name",
            "type": "function" | "class" | "global",
            "start_line": int,
            "end_line": int,
            "content": str,
        }

    For non-Python files, returns a single "global" block.
    """
    suffix = Path(file_path).suffix.lower() if file_path else ""

    if suffix not in {".py", ".pyi"}:
        # Non-Python: treat entire file as one block
        lines = code.splitlines()
        return [
            {
                "name": "__file__",
                "type": "global",
                "start_line": 1,
                "end_line": len(lines),
                "content": code,
            }
        ]

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Can't parse: treat as single block
        lines = code.splitlines()
        return [
            {
                "name": "__file__",
                "type": "global",
                "start_line": 1,
                "end_line": len(lines),
                "content": code,
            }
        ]

    lines = code.splitlines()
    blocks = []

    # Collect all top-level functions and classes
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = node.end_lineno or start
            # Include decorators in the block
            if node.decorator_list:
                start = min(d.lineno for d in node.decorator_list)
            content = "\n".join(lines[start - 1 : end])
            blocks.append(
                {
                    "name": node.name,
                    "type": "function",
                    "start_line": start,
                    "end_line": end,
                    "content": content,
                }
            )
        elif isinstance(node, ast.ClassDef):
            start = node.lineno
            end = node.end_lineno or start
            if node.decorator_list:
                start = min(d.lineno for d in node.decorator_list)
            content = "\n".join(lines[start - 1 : end])
            blocks.append(
                {
                    "name": node.name,
                    "type": "class",
                    "start_line": start,
                    "end_line": end,
                    "content": content,
                }
            )

    # If no functions/classes found, treat as global block
    if not blocks:
        blocks.append(
            {
                "name": "__file__",
                "type": "global",
                "start_line": 1,
                "end_line": len(lines),
                "content": code,
            }
        )

    return blocks


def merge_findings(
    cached_findings: Dict[str, FindingList],
    new_findings: Dict[str, FindingList],
    all_blocks: List[Dict],
) -> FindingList:
    """
    Merge cached and new findings from different function blocks.

    Args:
        cached_findings: {func_name: FindingList} from cache
        new_findings: {func_name: FindingList} from LLM
        all_blocks: List of all function blocks in the file

    Returns:
        Combined FindingList with all findings
    """
    all_findings = []

    for block in all_blocks:
        name = block["name"]
        if name in new_findings and new_findings[name].findings:
            all_findings.extend(new_findings[name].findings)
        elif name in cached_findings and cached_findings[name].findings:
            all_findings.extend(cached_findings[name].findings)

    return FindingList(findings=all_findings)
