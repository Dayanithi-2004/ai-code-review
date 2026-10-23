import ast
import logging
import re
from pathlib import Path
from typing import Set
from rich.console import Console
from .session import SessionManager

logger = logging.getLogger(__name__)
console = Console()

# Extensions dispatched to each parser
PYTHON_EXTENSIONS = {'.py', '.pyi'}
JS_EXTENSIONS     = {'.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs'}


class CodeMapper:
    """
    Phase 2: Mapping.
    Builds the dependency graph for Python and Node.js/TypeScript files.
    """
    def __init__(self, session: SessionManager, project_root: Path):
        self.session = session
        self.project_root = project_root

        # Matches static ESM and CJS imports only.
        # Deliberately excludes template-literal requires like require(`${var}`)
        # to avoid garbage edges from dynamic imports.
        self.js_ts_import_pattern = re.compile(
            r"(?:import|from)\s+['\"](?P<module>[@\w\/\.\-]+)['\"]"
            r"|require\(['\"](?P<require_module>[@\w\/\.\-]+)['\"]\)"
        )

    def build_graph(self, file_path: Path):
        """Dispatches to the correct parser based on file extension."""
        if file_path.suffix in PYTHON_EXTENSIONS:
            self._parse_python(file_path)
        elif file_path.suffix in JS_EXTENSIONS:
            self._parse_js_ts(file_path)

    def _parse_python(self, file_path: Path):
        """
        Uses stdlib ast for accurate Python import parsing.
        Handles both absolute and relative imports.
        Relative imports (from . import x, from ..utils import y) are resolved
        to a best-effort file path rather than stored as '.' or '..utils'.
        """
        try:
            source = file_path.read_text(errors='ignore')
            tree   = ast.parse(source)
        except SyntaxError:
            logger.debug("Skipping %s: SyntaxError", file_path.name)
            return  # Invalid Python — skip silently
        except Exception as e:
            logger.warning("Mapper skipping %s: %s", file_path.name, e)
            console.print(f"   ↳ [yellow]Mapper skipping {file_path.name}: {e}[/yellow]")
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._record_import(file_path, alias.name)
            elif isinstance(node, ast.ImportFrom):
                self._handle_import_from(file_path, node)

    def _handle_import_from(self, file_path: Path, node: ast.ImportFrom):
        """Process a single ImportFrom node, resolving relative imports."""
        if node.module is None:
            return
        assert isinstance(node.module, str)

        if node.level and node.level > 0:
            resolved = self._resolve_relative_import(
                file_path, node.module, node.level
            )
            self._record_import(file_path, resolved)
        else:
            self._record_import(file_path, node.module)

    def _resolve_relative_import(
        self, source_file: Path, module: str, level: int
    ) -> str:
        """
        Converts a relative import to a project-relative path string.
        e.g. in libro_api/crud.py:
          from . import models   (level=1, module='models')
          -> 'libro_api/models'
          from ..utils import x  (level=2, module='utils')
          -> 'utils'  (best effort — goes up 2 levels from file)
        """
        # Start from the file's directory and go up `level` times
        base = source_file.parent
        for _ in range(level - 1):
            base = base.parent

        try:
            rel_base = base.relative_to(self.project_root)
            resolved = str(rel_base / module.replace('.', '/'))
        except ValueError:
            # Outside project root — fall back to module name
            resolved = module

        return resolved

    def _parse_js_ts(self, file_path: Path):
        """
        Parses static ESM (import/from) and CJS (require()) imports.
        Dynamic requires (template literals) are intentionally skipped
        to avoid storing garbage edges in the dep graph.
        """
        try:
            content = file_path.read_text(errors='ignore')
        except Exception as e:
            logger.warning("Mapper skipping %s: %s", file_path.name, e)
            console.print(f"   ↳ [yellow]Mapper skipping {file_path.name}: {e}[/yellow]")
            return

        for match in self.js_ts_import_pattern.finditer(content):
            module = match.group("module") or match.group("require_module")
            if module:
                self._record_import(file_path, module)

    def _record_import(self, source: Path, target: str):
        """Saves a dependency edge to SQLite for Phase 3 severity elevation."""
        rel_source = str(source.relative_to(self.project_root))
        with self.session.get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO dep_graph (source_file, target_file) VALUES (?, ?)",
                (rel_source, target)
            )
        # Context manager auto-commits on exit

    def update_dependant_counts(self):
        """
        Calculates how many other files import each file (the blast radius).
        Updates module_summaries so the Orchestrator knows which files are
        most critical for severity elevation.
        """
        with self.session.get_connection() as conn:
            # Seed module_summaries with every file seen in the dep graph.
            # INSERT OR IGNORE preserves any real summaries already written
            # by the summary chain if it ran before this completes.
            conn.execute("""
                INSERT OR IGNORE INTO module_summaries
                    (file_path, purpose, key_functions, dependant_count)
                SELECT DISTINCT target_file, 'Pending summarisation', '[]', 0
                FROM dep_graph
            """)
            conn.execute("""
                INSERT OR IGNORE INTO module_summaries
                    (file_path, purpose, key_functions, dependant_count)
                SELECT DISTINCT source_file, 'Pending summarisation', '[]', 0
                FROM dep_graph
            """)

            # Count inbound edges for every file — the blast radius score
            conn.execute("""
                UPDATE module_summaries
                SET dependant_count = (
                    SELECT COUNT(source_file)
                    FROM dep_graph
                    WHERE target_file = module_summaries.file_path
                )
            """)
        # Context manager auto-commits on exit

    def get_impacted_files(self, changed_files: Set[str]) -> Set[str]:
        """
        Given a set of changed files, returns the union of:
          1. The changed files themselves
          2. All files that IMPORT any changed file (reverse dependencies)

        This catches the "individually correct but together broken" scenario:
        if auth.py changed, and payments.py imports auth.py, payments.py
        needs to be re-reviewed even though it didn't change.

        Args:
            changed_files: Set of project-relative paths from DiffEngine.

        Returns:
            Union of changed files + all files that depend on them.
        """
        if not changed_files:
            return set()

        impacted = set(changed_files)

        with self.session.get_connection() as conn:
            # dep_graph stores: source_file (the importer) → target_file (the imported)
            # We need: "who imports any of these changed files?" → reverse lookup
            placeholders = ",".join("?" for _ in changed_files)

            # Direct reverse deps: files that import a changed file
            rows = conn.execute(
                f"SELECT DISTINCT source_file FROM dep_graph "
                f"WHERE target_file IN ({placeholders})",
                list(changed_files),
            ).fetchall()

            for row in rows:
                impacted.add(row["source_file"])

        return impacted