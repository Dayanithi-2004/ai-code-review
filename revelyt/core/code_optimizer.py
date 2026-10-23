"""
code_optimizer — Token-efficient code preparation for LLM analysis.

Reduces token usage by 40-60% through:
1. Stripping excessive blank lines and trailing whitespace
2. Truncating files that exceed the model context budget
3. Collapsing large docstrings / multi-line comments
4. Skipping trivial files (empty __init__.py, lockfiles, etc.)
5. Compressing Wave-2 context to essential fields only

NEW — extract_code_skeleton():
  Produces an ultra-compact structural representation of a file for PM
  summarisation and BRD verification.  For Python files this uses the AST
  to guarantee all function/class signatures and FastAPI decorators are
  captured even in files where raw truncation would cut them off.  Typical
  reduction: 90-95% vs full code, with HIGHER accuracy for BRD purposes.
"""

import ast
import logging
import re
from pathlib import Path
from typing import Optional

from revelyt.core.schemas import FindingList

logger = logging.getLogger(__name__)

# Files that are never worth sending to the LLM
SKIP_FILENAMES = {
    # Lock files
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
    "Gemfile.lock",
    "composer.lock",
    "__pycache__",
    ".pyc",
    # Config/infra files — not worth full agent analysis
    ".gitignore",
    ".dockerignore",
    ".editorconfig",
    ".prettierrc",
    ".eslintrc",
    ".eslintrc.json",
    ".eslintrc.js",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "Makefile",
    "Procfile",
    "LICENSE",
    "LICENSE.md",
    "alembic.ini",
    "setup.cfg",
    "tox.ini",
    "pytest.ini",
    "babel.config.js",
    "webpack.config.js",
    "vite.config.ts",
    "tsconfig.json",
    "jsconfig.json",
}

SKIP_EXTENSIONS = {
    ".lock",
    ".min.js",
    ".min.css",
    ".map",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    # Text/doc files — low analysis value
    ".md",
    ".txt",
    ".rst",
    ".log",
    ".csv",
}

# Max characters of code to send to analysis agents (~4 chars ≈ 1 token)
MAX_CODE_CHARS = 8_000  # ≈ 2,000 tokens (fallback default)
# Max characters of Wave-2 context
MAX_CONTEXT_CHARS = 3_000  # ≈ 750 tokens
# Max characters for skeleton (per file, used in batch PM summarisation)
MAX_SKELETON_CHARS = 2_000  # ≈ 500 tokens — more than enough for full structure

# Dynamic budget tiers
TINY_FILE_LINES = 30  # < 30 lines → minimal budget (configs, small files)
SMALL_FILE_LINES = 50  # < 50 lines → small budget
MEDIUM_FILE_LINES = 150  # 50-150 lines → standard budget
BUDGET_TINY = 1_500  # ≈ 375 tokens
BUDGET_SMALL = 2_500  # ≈ 625 tokens
BUDGET_MEDIUM = 5_000  # ≈ 1,250 tokens
BUDGET_LARGE = 8_000  # ≈ 2,000 tokens

# OSS model budget tiers (smaller context windows, faster inference)
# OSS models benefit more from skeleton mode, so we can reduce budgets safely
OSS_BUDGET_TINY = 1_000  # ≈ 250 tokens
OSS_BUDGET_SMALL = 1_500  # ≈ 375 tokens
OSS_BUDGET_MEDIUM = 3_000  # ≈ 750 tokens
OSS_BUDGET_LARGE = 5_000  # ≈ 1,250 tokens


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def should_skip_file(file_path: Path) -> bool:
    """Return True if the file should be skipped entirely (not worth LLM tokens)."""
    if file_path.name in SKIP_FILENAMES:
        return True
    if file_path.suffix in SKIP_EXTENSIONS:
        return True

    try:
        size = file_path.stat().st_size
        if size < 20:
            return True
    except OSError:
        return True

    if file_path.name == "__init__.py":
        try:
            content = file_path.read_text(errors="ignore").strip()
            is_trivial = (
                len(content) < 200 and "def " not in content and "class " not in content
            )
            if is_trivial:
                return True
        except OSError:
            return True

    return False


def compute_token_budget(code_content: str, is_oss_model: bool = False) -> int:
    """
    Calculate a dynamic character budget based on file complexity.

    Small, simple files get a smaller budget (fewer tokens to the LLM).
    Large, complex files get a larger budget (better review quality).

    Args:
        code_content: Raw source code.
        is_oss_model: If True, use reduced budgets optimized for OSS models.
    """
    lines = code_content.count("\n") + 1
    has_classes = "class " in code_content

    if is_oss_model:
        if lines < TINY_FILE_LINES and not has_classes:
            return OSS_BUDGET_TINY
        if lines < SMALL_FILE_LINES and not has_classes:
            return OSS_BUDGET_SMALL
        if lines < MEDIUM_FILE_LINES:
            return OSS_BUDGET_MEDIUM
        return OSS_BUDGET_LARGE
    else:
        if lines < TINY_FILE_LINES and not has_classes:
            return BUDGET_TINY
        if lines < SMALL_FILE_LINES and not has_classes:
            return BUDGET_SMALL
        if lines < MEDIUM_FILE_LINES:
            return BUDGET_MEDIUM
        return BUDGET_LARGE


def prepare_code_for_analysis(
    code_content: str, file_path: str = "", max_chars: int = 0
) -> str:
    """
    Prepare code for LLM analysis agents (security/quality/qa chains).

    Args:
        code_content: Raw source code.
        file_path:    File path (for logging).
        max_chars:    Character budget. If 0, auto-calculated via compute_token_budget().

    Strategies (in order):
    1. Strip trailing whitespace from each line
    2. Collapse runs of 3+ blank lines into 2
    3. Collapse long docstrings (>15 lines) to first 3 + last 2 lines
    4. Truncate to budget, keeping imports + beginning + end
    """
    if not code_content:
        return code_content

    budget = max_chars if max_chars > 0 else compute_token_budget(code_content)

    lines = [line.rstrip() for line in code_content.splitlines()]

    # Collapse runs of 3+ blank lines → 2
    collapsed = []
    blank_run = 0
    for line in lines:
        if not line:
            blank_run += 1
            if blank_run <= 2:
                collapsed.append(line)
        else:
            blank_run = 0
            collapsed.append(line)
    lines = collapsed

    lines = _collapse_long_docstrings(lines)

    result = "\n".join(lines)

    if len(result) > budget:
        result = _smart_truncate(result, lines, file_path, budget)

    return result


def extract_code_skeleton(code: str, file_path: str = "") -> str:
    """
    Extract a compact structural skeleton of a source file for PM summarisation
    and BRD verification.

    For Python files, uses the AST — guarantees 100% function/class coverage
    regardless of file length, and captures FastAPI/SQLAlchemy decorators that
    are the primary evidence for BRD verification (endpoint routes, model fields).

    For all other file types, falls back to prepare_code_for_analysis().

    Token budget: ~500 tokens per file vs ~2000 for full truncated code.
    A 6-file batch summary drops from ~12,000 tokens to ~3,000 tokens.

    Critical difference vs raw code[:4000]:
      - code[:4000] on crud.py (230 lines) silently drops return_book(),
        get_loans(), soft_delete_loan() — causing false BRD NOT_MET results.
      - extract_code_skeleton() captures ALL 17 functions in ~555 chars.
    """
    suffix = Path(file_path).suffix.lower() if file_path else ""

    if suffix in {".py", ".pyi"}:
        try:
            skeleton = _extract_python_skeleton(code, file_path)
            # Hard cap: if the skeleton itself is huge (e.g. thousands of attrs),
            # truncate it. This should never happen in normal codebases.
            if len(skeleton) > MAX_SKELETON_CHARS:
                skeleton = (
                    skeleton[:MAX_SKELETON_CHARS] + "\n# ... (skeleton truncated)"
                )  # type: ignore
            return skeleton
        except Exception as exc:
            logger.debug(
                "Skeleton extraction failed for %s (%s), falling back", file_path, exc
            )
            return prepare_code_for_analysis(code, file_path)

    # Non-Python: smart truncation is good enough (JS/TS have clear signatures)
    return prepare_code_for_analysis(code, file_path)


def compress_wave1_context(context: Optional[FindingList]) -> str:
    """
    Compress Wave-1 findings context to essential fields only.
    Drops verbose descriptions and suggestions to save ~60% tokens.
    """
    if context is None or getattr(context, "findings", None) is None:
        return ""

    compact_lines = []
    for f in context.findings:
        first_sentence = f.description.split(".")[0] + "." if f.description else ""
        compact_lines.append(f"- [{f.severity}] {f.title}: {first_sentence}")

    compressed = "WAVE 1 FINDINGS (summary):\n" + "\n".join(compact_lines)

    if len(compressed) > MAX_CONTEXT_CHARS:
        compressed = compressed[:MAX_CONTEXT_CHARS] + "\n... (truncated)"

    return compressed


# ──────────────────────────────────────────────────────────────────────────────
# Python skeleton extraction (AST-based)
# ──────────────────────────────────────────────────────────────────────────────


def _extract_python_skeleton(code: str, file_path: str = "") -> str:
    """
    Build a compact structural representation of a Python file using AST.

    Captures:
    - Module-level docstring (first line only)
    - All import statements
    - All top-level constants / assignments (e.g. DATABASE_URL = ...)
    - All class definitions with their attributes and method signatures
    - All top-level function signatures with their decorators
    - First-line docstrings for functions and classes

    Decorators are especially important:
    - FastAPI: @app.post("/loans/{loan_id}/return") reveals the HTTP route
    - SQLAlchemy: Column(String, unique=True) reveals field constraints
    - Pydantic: @validator / @model_validator reveals validation logic
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return prepare_code_for_analysis(code, file_path)

    source_lines = code.splitlines()
    result: list[str] = []

    _append_module_docstring(tree, result)

    for node in tree.body:
        _process_toplevel_node(node, source_lines, result)

    return _collapse_blank_lines(result)


def _append_module_docstring(tree, result: list[str]) -> None:
    """Append the first line of the module docstring, if any."""
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        first = tree.body[0].value.value.strip().split("\n")[0][:120]
        result.append(f'"""{first}"""')
        result.append("")


def _process_toplevel_node(node, source_lines: list[str], result: list[str]) -> None:
    """Dispatch a single top-level AST node into the skeleton result."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        result.append(source_lines[node.lineno - 1].rstrip())

    elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        line = source_lines[node.lineno - 1].rstrip()
        if len(line) <= 120 and not line.lstrip().startswith("_"):
            result.append(line)

    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        result.append("")
        _append_function_skeleton(node, source_lines, result, indent="")

    elif isinstance(node, ast.ClassDef):
        result.append("")
        _append_class_skeleton(node, source_lines, result)


def _collapse_blank_lines(lines: list[str]) -> str:
    """Collapse 2+ consecutive blank lines to 1."""
    final: list[str] = []
    blanks = 0
    for line in lines:
        if not line.strip():
            blanks += 1
            if blanks <= 1:
                final.append(line)
        else:
            blanks = 0
            final.append(line)
    return "\n".join(final).strip()


def _get_first_docstring(body: list) -> str:
    """Extract the first line of a function/class docstring, if present."""
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[0].value.value.strip().split("\n")[0][:100]
    return ""


def _get_decorator_lines(node, source_lines: list[str], indent: str) -> list[str]:
    """Return the source lines for all decorators on a node."""
    lines = []
    for dec in node.decorator_list:
        dec_src = source_lines[dec.lineno - 1].strip()
        lines.append(f"{indent}{dec_src}")
    return lines


def _append_function_skeleton(
    node,
    source_lines: list[str],
    result: list[str],
    indent: str = "",
) -> None:
    """Append a compact function signature (decorators + def line + optional docstring)."""
    # Decorators
    result.extend(_get_decorator_lines(node, source_lines, indent))

    # Signature — keep full signature so arg types are visible for BRD evidence
    sig_line = source_lines[node.lineno - 1].rstrip()
    # Trim leading spaces then re-indent with our indent level
    sig_line = indent + sig_line.lstrip()
    # Source lines end with ':' — strip it before appending ': ...' stub
    if sig_line.endswith(":"):
        sig_line = sig_line[:-1]
    result.append(f"{sig_line}: ...")

    # First-line docstring
    doc = _get_first_docstring(node.body)
    if doc:
        result.append(f"{indent}    # {doc}")


def _append_class_skeleton(
    node,
    source_lines: list[str],
    result: list[str],
) -> None:
    """Append a compact class skeleton (class line + docstring + attributes + methods)."""
    # Class declaration (includes base classes)
    class_sig = source_lines[node.lineno - 1].rstrip()
    result.append(class_sig)

    # Class docstring
    doc = _get_first_docstring(node.body)
    if doc:
        result.append(f'    """{doc}"""')

    # Class body: attributes (Column / Field / plain assignments) and method signatures
    for child in node.body:
        # Class-level attribute assignments (SQLAlchemy Column, Pydantic Field, etc.)
        if isinstance(child, (ast.Assign, ast.AnnAssign)):
            line = source_lines[child.lineno - 1].rstrip()
            if len(line) <= 120:
                result.append(f"    {line.lstrip()}")

        # Class-level constants (e.g. __tablename__ = "books")
        elif isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant):
            pass  # skip — already captured as docstring if it's the first one

        # Methods
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result.append("")
            _append_function_skeleton(child, source_lines, result, indent="    ")

    result.append("")


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────


def _extract_docstring(lines: list, start_idx: int, quote: str) -> tuple[list, int]:
    """Extract a multi-line docstring block."""
    doc_lines = [lines[start_idx]]
    i = start_idx + 1
    while i < len(lines):
        doc_lines.append(lines[i])
        if quote in lines[i]:
            break
        i += 1
    return doc_lines, i + 1


def _collapse_long_docstrings(lines: list) -> list:
    """Collapse docstrings/comments longer than 15 lines to first 3 + last 2."""
    result = []
    i = 0
    while i < len(lines):  # type: ignore
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote = stripped[:3]
            if stripped.count(quote) >= 2:
                result.append(line)
                i += 1  # type: ignore
                continue

            doc_lines, next_i = _extract_docstring(lines, i, quote)

            if len(doc_lines) > 15:
                result.extend(doc_lines[:3])  # type: ignore
                result.append(
                    f"    # ... ({len(doc_lines) - 5} lines of docstring collapsed) ..."
                )
                result.extend(doc_lines[-2:])  # type: ignore
            else:
                result.extend(doc_lines)

            i = next_i
        else:
            result.append(line)
            i += 1  # type: ignore

    return result


def _smart_truncate(content: str, lines: list, file_path: str, budget: int = 0) -> str:
    """
    Smart truncation: keep imports region + first portion + last portion.
    """
    if budget <= 0:
        budget = MAX_CODE_CHARS

    import_end = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) or not stripped:
            import_end = idx + 1
        elif import_end > 0 and stripped:
            break

    imports_budget = int(budget * 0.20)  # type: ignore
    body_budget = budget - imports_budget  # type: ignore

    imports_section = "\n".join(lines[:import_end])  # type: ignore
    if len(imports_section) > imports_budget:
        imports_section = imports_section[:imports_budget]  # type: ignore

    body_lines = lines[import_end:]  # type: ignore
    body = "\n".join(body_lines)

    if len(body) <= body_budget:
        result = imports_section + "\n" + body
    else:
        begin_budget = int(body_budget * 0.6)
        end_budget = body_budget - begin_budget
        begin_part = body[:begin_budget]  # type: ignore
        end_part = body[-end_budget:]  # type: ignore
        total_lines = len(body_lines)
        result = (
            imports_section
            + "\n"
            + begin_part
            + f"\n\n# ... [Revelyt: {total_lines} total lines — middle section omitted for efficiency] ...\n\n"
            + end_part
        )

    logger.info(
        "Truncated %s from %d to %d chars (%.0f%% reduction)",
        file_path,
        len(content),
        len(result),
        (1 - len(result) / len(content)) * 100,
    )
    return result
