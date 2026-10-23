"""
ProductManagerAgent — Post-analysis: business alignment, summaries & BRD verification.

Key design decision: test files and migration files are NEVER summarised by the LLM.
BRD evidence for "must include pytest", "must use Alembic", etc. is gathered by
_detect_infra_for_brd() — a zero-token filesystem probe that:
  1. Walks the project root looking for test files and migration dirs
  2. Extracts test function names via fast regex (no LLM, no file content sent)
  3. Injects one synthetic summary entry per detected framework into module_summaries
  4. Costs 0 tokens and runs in <10ms regardless of repo size

The BRD verifier then sees entries like:
  file_path:     "tests/ [auto-detected]"
  purpose:       "pytest test suite — 14 test files found"
  key_functions: ["pytest test suite", "test_create_user", "test_borrow_book", ...]

This satisfies both constraints:
  - DeepSeek never reads or summarises test/migration content (no tokens wasted)
  - BRD requirements like "must include pytest testing" are correctly marked MET
"""

import json
import logging
import os
import re
from pathlib import Path
import typing
from typing import List, Dict

from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console

from revelyt.agents.base_agent import BaseAgent
from revelyt.core.schemas import FindingList, RequirementList, VerificationResult, VerificationResultList
from revelyt.core.session import SessionManager
from revelyt.providers.base import BaseProvider
from revelyt.core.code_optimizer import extract_code_skeleton, prepare_code_for_analysis

logger = logging.getLogger(__name__)
console = Console()


from pydantic import BaseModel, Field


class SummaryOutput(BaseModel):
    purpose: str = Field(
        default="",
        description="1-2 sentence description of the file's main responsibility.",
    )
    key_functions: List[str] = Field(
        default_factory=list,
        description="List of function/class names using exact source-code names.",
    )


class FileSummary(BaseModel):
    file_path: str = Field(description="Relative path of the file")
    purpose: str = Field(description="1-2 sentence description including frameworks used")
    key_functions: List[str] = Field(
        description=(
            "ALL function/class names using exact source-code names. "
            "Include transaction patterns and HTTP endpoints."
        )
    )


class BatchSummaryOutput(BaseModel):
    summaries: List[FileSummary] = Field(description="One summary per file")


# ─────────────────────────────────────────────────────────────────────────────
# BRD semantic resolution map (framework-general, not domain-specific)
# ─────────────────────────────────────────────────────────────────────────────

_BRD_SEMANTIC_RESOLUTION = """\
SEMANTIC FUNCTION NAME RESOLUTION:
Requirement text uses business language; code uses technical names.
Use these vocabulary mappings when matching requirements to code evidence:

Authentication / Authorization:
  Keywords: authentication, login, sign in, secure access, user verification,
    access control, authorization, permissions, roles
  Code evidence: login, authenticate, verify_token, verify_password, jwt_required,
    JwtAuthGuard, @login_required, IsAuthenticated, AuthGuard, create_access_token,
    decode_token, get_current_user, session.user, passport.use, NextAuth, auth.js,
    middleware.ts with auth logic

Database Transactions / Data Consistency:
  Keywords: transactional, atomic, data consistency, rollback, ACID, concurrent,
    race condition prevention, pessimistic locking
  Code evidence: begin_nested, with_for_update, session.begin, transaction(),
    db.transaction(), sequelize.transaction, prisma.$transaction,
    TypeORM QueryRunner, @Transactional, atomic(), select_for_update(), SERIALIZABLE

Pagination / Filtering:
  Keywords: pagination, paging, filtering, search, sorting, list with parameters
  Code evidence: skip, limit, offset, page, per_page, cursor, paginate, Paginator,
    PageNumberPagination, take, first, after, before, ?page=, ?limit=, ?cursor=

Testing:
  Keywords: unit tests, integration tests, test coverage, automated testing,
    pytest, jest, vitest, test suite, testing framework
  Code evidence: pytest test suite, jest test suite, vitest test suite, mocha test suite,
    test_ prefix in filenames, describe(, it(, test(, @pytest.fixture, TestCase,
    jest.mock, supertest, TestClient, APIClient, factory_boy

Migrations / Schema Management:
  Keywords: database migrations, schema changes, schema management, Alembic,
    Prisma migrate, versioned schema
  Code evidence: alembic migrations, upgrade, downgrade, revision, prisma migrate,
    knex migrations, sequelize migrations, django migrations

Soft Delete / Archiving:
  Keywords: soft delete, archive, deactivate, logical delete, restore, trash
  Code evidence: is_deleted, deleted_at, is_active, is_archived, soft_delete,
    archive, deactivate, restore, deletedAt, isDeleted

Audit Logging:
  Keywords: audit trail, activity log, change history, who changed what
  Code evidence: audit, log_action, record_action, AuditLog, ActivityLog,
    create_audit, _record_audit, event_log, history, changelog

Email / Notifications:
  Keywords: email notification, send email, notify user, welcome email
  Code evidence: send_email, send_mail, EmailMessage, SMTP, sendgrid, mailgun,
    nodemailer, ses.sendEmail, notify, push_notification

File Upload / Storage:
  Keywords: file upload, image upload, document storage, S3, cloud storage
  Code evidence: upload, multipart, UploadFile, S3Client, put_object, presigned_url,
    multer, formData, storage.bucket, GridFS

Rate Limiting:
  Keywords: rate limiting, throttling, request limits
  Code evidence: rate_limit, throttle, RateLimiter, express-rate-limit, @Throttle,
    RATE_LIMIT, redis-rate-limit, token bucket

VERDICT RULES:
  MET:     Clear evidence found — function, class, endpoint, or key_functions entry
           matches the vocabulary above. Auto-detected framework entries count.
  PARTIAL: Some evidence found but implementation appears incomplete or partial.
  NOT_MET: Searched ALL summaries (including auto-detected entries) with the
           vocabulary list above and found zero matching evidence.
  When uncertain between NOT_MET and PARTIAL — choose PARTIAL."""


# ─────────────────────────────────────────────────────────────────────────────
# Infrastructure detection patterns (filesystem probe — no LLM)
# ─────────────────────────────────────────────────────────────────────────────

# Test directory names to probe for
_TEST_DIRS = {'tests', 'test', '__tests__', 'spec', 'specs', 'e2e', 'integration'}

# File patterns that identify test frameworks
_PYTEST_FILE_PAT = re.compile(r'^(test_.+|.+_test)\.py$')
_JEST_FILE_PAT = re.compile(r'\.(test|spec)\.(js|ts|jsx|tsx)$')
_CONFTEST_PAT = re.compile(r'^conftest\.py$')
_MOCHA_MARKER = re.compile(r'\.mocharc\.(js|cjs|yml|yaml|json)$')
_VITEST_MARKER = re.compile(r'^vitest\.config\.(ts|js)$')
_PLAYWRIGHT_MARKER = re.compile(r'^playwright\.config\.(ts|js)$')
_CYPRESS_MARKER = re.compile(r'^cypress\.config\.(ts|js)$')

# Regex to extract test function names from file content (fast, no AST)
_PYTEST_FN_PAT = re.compile(r'^(?:async )?def (test_\w+)', re.MULTILINE)
_JEST_FN_PAT = re.compile(r"(?:it|test)\(['\"]([^'\"]{3,60})['\"]", re.MULTILINE)

# Migration detection
_ALEMBIC_MARKER = re.compile(r'^(alembic\.ini|env\.py)$')
_DJANGO_MIGRATION = re.compile(r'^\d{4}_\w+\.py$')
_PRISMA_MIGRATION_DIR = 'prisma'

# Max test function names to extract per framework (keep synthetic entry concise)
_MAX_FN_NAMES = 20


class ProductManagerAgent(BaseAgent):
    """
    Post-analysis agent — generates module summaries and runs BRD verification.

    Test and migration files are NEVER passed to the LLM.
    BRD evidence for testing/migration requirements is gathered by
    _detect_infra_for_brd() — a zero-token filesystem probe.
    """

    wave = 1
    role_name = "PM"
    emoji = "📋"
    category = "Business Alignment"

    def __init__(self, session: SessionManager, provider: BaseProvider, project_root: Path):
        super().__init__(session, provider)
        self.project_root = project_root

    # ------------------------------------------------------------------
    # 1. Module Summarisation (batched)
    # ------------------------------------------------------------------

    def summarize_batch(self, file_paths: list[Path]) -> None:
        """
        Summarize files concurrently (1 file per LLM call) instead of Map-Reduce.
        Small models (3B) reliably generate flat `SummaryOutput` JSONs but fail at deep
        arrrays (`BatchSummaryOutput`), which triggers retry loops and massive slowdowns.
        Using ThreadPoolExecutor we maintain inference throughput entirely via parallel requests.
        """
        if not file_paths:
            self._detect_infra_for_brd()
            return

        IGNORE_EXTS = {".css", ".html", ".svg", ".json", ".md", ".txt", ".lock", ".env", ".yaml", ".yml", ".png", ".jpg", ".jpeg"}
        IGNORE_NAMES = {".gitignore", ".prettierignore", ".ds_store", ".eslintrc", "package.json"}

        # Build list of valid files
        valid_files = []
        for fp in file_paths:
            if fp.suffix.lower() in IGNORE_EXTS or fp.name.lower() in IGNORE_NAMES or "temp." in fp.name.lower():
                continue
            valid_files.append(fp)

        if not valid_files:
            self._detect_infra_for_brd()
            return

        console.print(
            f"   ↳ {self.emoji} Summarising {len(valid_files)} core files using highly concurrent flat generation..."
        )

        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        db_lock = threading.Lock()

        # We assume 5-10 slots. The llama provider handles its own queuing if it gets overwhelmed.
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self._summarize_single, db_lock, fp) for fp in valid_files]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    logger.error("Error in concurrent PM summarisation: %s", exc)

        self._detect_infra_for_brd()

    def _summarize_single(self, db_lock: "threading.Lock", file_path: Path) -> None:
        import time
        start_t = time.time()
        try:
            raw = file_path.read_text(errors="ignore")
            skeleton = extract_code_skeleton(raw, str(file_path))
        except Exception as exc:
            logger.warning("PM single-file skipping %s: %s", file_path.name, exc)
            return

        rel_path = str(file_path.relative_to(self.project_root))

        # Minimal prompt — fewer instructions = fewer hallucinated field names
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                'Return ONLY this JSON, nothing else:\n'
                '{{"purpose": "ONE sentence", "key_functions": ["name1", "name2"]}}\n\n'
                "Rules: purpose = what the file does in 1 sentence. "
                "key_functions = exact function/class names from the code. "
                "No markdown. No explanation. No extra fields.",
            ),
            (
                "user",
                "File: {file_path}\n\n```\n{code_content}\n```\n\nJSON output:",
            ),
        ])

        purpose: str = ""
        key_functions: list = []
        result = None

        try:
            result: SummaryOutput = self.provider.invoke_structured(
                prompt=prompt,
                output_schema=SummaryOutput,
                file_path=rel_path,
                code_content=skeleton,
            )
            if result is not None:
                purpose = result.purpose or ""
                key_functions = result.key_functions or []
        except Exception as exc:
            logger.warning("PM summary LLM failed on %s: %s", file_path.name, exc)

        # Regex fallback — derive from skeleton when LLM fails, costs 0 tokens
        if not purpose:
            purpose = self._extract_purpose_from_skeleton(skeleton, file_path.name)
        if not key_functions:
            key_functions = self._extract_functions_from_skeleton(skeleton)

        if purpose:
            with db_lock, self.session.get_connection() as conn:
                conn.execute(
                    "UPDATE module_summaries SET purpose = ?, key_functions = ? "
                    "WHERE file_path = ?",
                    (purpose, json.dumps(key_functions), rel_path),
                )
            elapsed = time.time() - start_t
            tag = "summarised" if result is not None else "regex fallback"
            console.print(
                f"   ↳ {self.emoji} [{self.role_name}]  "
                f"{rel_path:<40} → [italic]{tag}[/italic] ({elapsed:.1f}s)"
            )

    @staticmethod
    def _extract_purpose_from_skeleton(skeleton: str, filename: str) -> str:
        import re as _re
        # Try module docstring first
        doc_match = _re.search(r'"""(.+?)"""', skeleton, _re.DOTALL)
        if doc_match:
            first_line = doc_match.group(1).strip().split("\n")[0][:150]
            if len(first_line) > 10:
                return first_line
        # Fallback: infer from filename
        stem = filename.rsplit(".", 1)[0]
        stem = _re.sub(r"([A-Z])", r" \1", stem).strip()
        return f"Module: {stem}"

    @staticmethod
    def _extract_functions_from_skeleton(skeleton: str) -> list:
        import re as _re
        names = []
        # Python
        for m in _re.finditer(r"^(?:async )?def (\w+)|^class (\w+)", skeleton, _re.MULTILINE):
            name = m.group(1) or m.group(2)
            if name and not name.startswith("_"):
                names.append(name)
        # JS/TS
        for m in _re.finditer(
            r"(?:export\s+)?(?:async\s+)?function\s+(\w+)|"
            r"export\s+(?:const|let)\s+(\w+)\s*=|"
            r"router\.\w+\(['\"]([^'\"]+)['\"]",
            skeleton, _re.MULTILINE
        ):
            name = m.group(1) or m.group(2) or m.group(3)
            if name:
                names.append(name)
        return list(dict.fromkeys(names))[:20]

    # ------------------------------------------------------------------
    # 2. Zero-token infrastructure probe (test + migration detection)
    # ------------------------------------------------------------------

    def _detect_infra_for_brd(self) -> None:
        """
        Walk the project root and inject synthetic BRD-evidence entries for
        test suites and migration scripts — without reading file content into
        an LLM prompt.

        Strategy:
          - Detect test framework by file patterns (test_*.py, *.test.ts, etc.)
          - Extract test function names via fast regex on file content
          - Detect migration framework by directory/file patterns
          - Insert one synthetic module_summaries row per detected framework

        Cost: 0 tokens. Runtime: <10ms for any repo size.

        The BRD verifier treats these entries identically to real summaries:
        "pytest test suite" in key_functions → MET for "must include pytest testing"
        "alembic migrations" in key_functions → MET for "must use Alembic migrations"
        """
        entries = []
        entries.extend(self._probe_test_frameworks())
        entries.extend(self._probe_migration_frameworks())

        if not entries:
            return

        with self.session.get_connection() as conn:
            for entry in entries:
                # Use INSERT OR REPLACE so re-runs don't duplicate entries
                conn.execute(
                    """
                    INSERT OR REPLACE INTO module_summaries
                        (file_path, purpose, key_functions, dependant_count)
                    VALUES (?, ?, ?, 0)
                    """,
                    (entry["file_path"], entry["purpose"], json.dumps(entry["key_functions"])),
                )
                console.print(
                    f"   ↳ {self.emoji} [{self.role_name}]  "
                    f"{entry['file_path']:<35} → [dim italic]infra probe (0 tokens)[/dim italic]"
                )

    def _probe_test_frameworks(self) -> List[Dict]:
        """
        Detect test frameworks present in the repository by scanning file patterns.
        Returns a list of synthetic summary dicts, one per detected framework.
        """
        pytest_files: List[Path] = []
        jest_files: List[Path] = []
        vitest_found = False
        mocha_found = False
        playwright_found = False
        cypress_found = False

        for root, dirs, files in os.walk(self.project_root):
            # Skip dirs that would never contain tests
            dirs_filtered = [d for d in dirs if d not in {
                'node_modules', '.venv', 'venv', '.git', '__pycache__',
                'dist', 'build', '.next', '.nuxt',
            }]
            dirs.clear()
            dirs.extend(dirs_filtered)

            for fname in files:
                fpath = Path(root) / fname
                if _PYTEST_FILE_PAT.match(fname) or _CONFTEST_PAT.match(fname):
                    pytest_files.append(fpath)
                elif _JEST_FILE_PAT.search(fname):
                    jest_files.append(fpath)
                if _VITEST_MARKER.match(fname):
                    vitest_found = True
                if _MOCHA_MARKER.match(fname):
                    mocha_found = True
                if _PLAYWRIGHT_MARKER.match(fname):
                    playwright_found = True
                if _CYPRESS_MARKER.match(fname):
                    cypress_found = True

        results = []

        # ── pytest ──
        if pytest_files:
            fn_names = self._extract_pytest_fns(pytest_files)
            framework_label = "pytest"
            key_fns = (
                [f"pytest test suite — {len(pytest_files)} test files"] +
                [fn_names[i] for i in range(min(len(fn_names), _MAX_FN_NAMES))]
            )
            results.append({
                "file_path": "tests/ [auto-detected]",
                "purpose": (
                    f"{framework_label} test suite — {len(pytest_files)} test files detected. "
                    "Evidence for BRD testing requirements."
                ),
                "key_functions": key_fns,
            })
            logger.info("Infra probe: pytest — %d files, %d functions", len(pytest_files), len(fn_names))

        # ── jest / vitest ──
        if jest_files:
            fn_names = self._extract_jest_fns(jest_files)
            framework_label = "jest + vitest" if vitest_found else "jest"
            key_fns = (
                [f"{framework_label} test suite — {len(jest_files)} test files"] +
                [fn_names[i] for i in range(min(len(fn_names), _MAX_FN_NAMES))]
            )
            results.append({
                "file_path": "__tests__/ [auto-detected]",
                "purpose": (
                    f"{framework_label} test suite — {len(jest_files)} test files detected. "
                    "Evidence for BRD testing requirements."
                ),
                "key_functions": key_fns,
            })
            logger.info("Infra probe: jest — %d files, %d functions", len(jest_files), len(fn_names))

        # ── mocha ──
        if mocha_found:
            results.append({
                "file_path": "mocha [auto-detected]",
                "purpose": "Mocha test framework configuration detected.",
                "key_functions": ["mocha test suite"],
            })

        # ── playwright ──
        if playwright_found:
            results.append({
                "file_path": "playwright [auto-detected]",
                "purpose": "Playwright end-to-end test framework detected.",
                "key_functions": ["playwright e2e test suite", "end-to-end testing"],
            })

        # ── cypress ──
        if cypress_found:
            results.append({
                "file_path": "cypress [auto-detected]",
                "purpose": "Cypress end-to-end test framework detected.",
                "key_functions": ["cypress e2e test suite", "end-to-end testing"],
            })

        return results

    def _probe_migration_frameworks(self) -> List[Dict]:
        """
        Detect database migration frameworks by directory and file patterns.
        Returns synthetic summary dicts for BRD migration-requirement evidence.
        """
        results = []

        # ── Alembic ──
        alembic_dir = self.project_root / "alembic"
        alembic_ini = self.project_root / "alembic.ini"
        if alembic_dir.is_dir() or alembic_ini.exists():
            revision_files = list(alembic_dir.rglob("*.py")) if alembic_dir.is_dir() else []
            revision_count = len([f for f in revision_files if f.name != "env.py"])
            results.append({
                "file_path": "alembic/ [auto-detected]",
                "purpose": (
                    f"Alembic database migration system — {revision_count} revision files. "
                    "Evidence for BRD migration/schema-management requirements."
                ),
                "key_functions": [
                    "alembic migrations",
                    f"alembic — {revision_count} schema revisions",
                    "upgrade", "downgrade", "revision",
                ],
            })
            logger.info("Infra probe: alembic — %d revisions", revision_count)

        # ── Django migrations ──
        django_migration_files: List[Path] = []
        for root, dirs, files in os.walk(self.project_root):
            dirs_filtered = [d for d in dirs if d not in {'node_modules', '.venv', 'venv', '.git'}]
            dirs.clear()
            dirs.extend(dirs_filtered)
            if os.path.basename(root) == "migrations":
                django_migration_files.extend(
                    Path(root) / f for f in files
                    if _DJANGO_MIGRATION.match(f)
                )
        if django_migration_files:
            results.append({
                "file_path": "migrations/ [auto-detected]",
                "purpose": (
                    f"Django database migrations — {len(django_migration_files)} migration files. "
                    "Evidence for BRD migration requirements."
                ),
                "key_functions": [
                    "django migrations",
                    f"django — {len(django_migration_files)} migration files",
                    "RunMigrations", "makemigrations",
                ],
            })
            logger.info("Infra probe: django migrations — %d files", len(django_migration_files))

        # ── Prisma migrations ──
        prisma_dir = self.project_root / "prisma"
        if prisma_dir.is_dir():
            migration_dirs = [
                d for d in (prisma_dir / "migrations").iterdir()
                if d.is_dir()
            ] if (prisma_dir / "migrations").is_dir() else []
            results.append({
                "file_path": "prisma/ [auto-detected]",
                "purpose": (
                    f"Prisma ORM schema and migrations — "
                    f"{len(migration_dirs)} migration(s) detected."
                ),
                "key_functions": [
                    "prisma migrations",
                    "prisma migrate deploy",
                    "prisma migrate dev",
                    f"prisma — {len(migration_dirs)} migration files",
                ],
            })
            logger.info("Infra probe: prisma migrations — %d dirs", len(migration_dirs))

        return results

    def _extract_pytest_fns(self, files: List[Path]) -> List[str]:
        """Extract test function names from pytest files via regex (no LLM)."""
        names: List[str] = []
        for fpath in files:
            if len(names) >= _MAX_FN_NAMES:
                break
            try:
                content = fpath.read_text(errors="ignore")
                found = _PYTEST_FN_PAT.findall(content)
                names.extend(found)
            except OSError:
                continue
        return list(dict.fromkeys(names))  # deduplicate preserving order

    def _extract_jest_fns(self, files: List[Path]) -> List[str]:
        """Extract test description strings from Jest/Vitest files via regex (no LLM)."""
        names: List[str] = []
        for fpath in files:
            if len(names) >= _MAX_FN_NAMES:
                break
            try:
                content = fpath.read_text(errors="ignore")
                found = _JEST_FN_PAT.findall(content)
                names.extend(found)
            except OSError:
                continue
        return list(dict.fromkeys(names))

    # ------------------------------------------------------------------
    # 3. BRD Verification (holistic, after all summaries + probe complete)
    # ------------------------------------------------------------------

    def verify_brd(self, brd_path: Path) -> None:
        """
        Parse a BRD and verify each requirement against module_summaries.
        Synthetic entries from _detect_infra_for_brd() are included automatically
        since they are stored in the same module_summaries table.
        """
        logger.info("Phase 4: BRD Verification started (%s)", brd_path.name)
        console.print("\n🔍 [bold blue]Phase 4: BRD Verification[/bold blue]")

        try:
            brd_text = brd_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.error("Error reading BRD file: %s", exc, exc_info=True)
            console.print(f"   ↳ [red]Error reading BRD file: {exc}[/red]")
            return

        console.print(f"   ↳ 📄 Parsing {brd_path.name} into atomic requirements...")

        parse_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Extract all business, security, and technical requirements from "
                "the provided text. Return ONLY valid JSON matching the schema.",
            ),
            ("user", "Document Text:\n{text}"),
        ])

        try:
            parsed: RequirementList = self.provider.invoke_structured(
                prompt=parse_prompt,
                output_schema=RequirementList,
                text=brd_text[:15000], # type: ignore
            )

            with self.session.get_connection() as conn:
                for req in parsed.requirements:
                    conn.execute(
                        "INSERT INTO brd_requirements (requirement, status, evidence) "
                        "VALUES (?, ?, ?)",
                        (f"[{req.category}] {req.description}", "PENDING", ""),
                    )

            console.print(f"   ↳ ✅ Extracted {len(parsed.requirements)} requirements.")

        except Exception as exc:
            console.print(f"   ↳ [red]BRD parsing failed: {exc}[/red]")
            return

        summaries = self._get_module_summaries()
        if not summaries:
            console.print("   ↳ [yellow]No module summaries found. Did Phase 3 complete?[/yellow]")
            return

        requirements = self._get_pending_requirements()
        if not requirements:
            console.print("   ↳ [green]No pending requirements to verify.[/green]")
            return

        # Log how many entries are auto-detected vs real summaries
        auto_entries = [s for s in summaries if "[auto-detected]" in s.get("file_path", "")]
        console.print(
            f"   ↳ 📋 Verifying {len(requirements)} requirements against "
            f"{len(summaries)} entries "
            f"([dim]{len(auto_entries)} auto-detected infra entries[/dim])"
        )

        self._verify_all_requirements(requirements, summaries)
        console.print("\n   ↳ ✅ [bold green]BRD Verification Complete.[/bold green]")

    # ------------------------------------------------------------------
    # BRD helpers
    # ------------------------------------------------------------------

    def _compact_summaries_for_brd(self, requirements: list, summaries: list) -> list:
        """Only include summaries whose content matches any requirement keyword."""
        req_text = " ".join(r["requirement"].lower() for r in requirements)
        keywords = set(re.findall(r'[a-z]{4,}', req_text))

        # Always include auto-detected infra entries — they may match
        # requirement keywords that aren't obvious from file paths alone
        filtered = []
        for s in summaries:
            if "[auto-detected]" in s.get("file_path", ""):
                filtered.append(s)
                continue
            s_text = (
                s["file_path"] + " " +
                s.get("purpose", "") + " " +
                " ".join(s.get("key_functions", []))
            ).lower()
            if any(kw in s_text for kw in keywords):
                filtered.append(s)

        return filtered

    def _verify_all_requirements(self, requirements: list, summaries: list) -> None:
        # Process 1 requirement at a time — gives maximum context budget for summaries
        console.print(f"   ↳ ▶️  Verifying {len(requirements)} requirements one at a time...")

        results_all = []
        for req in requirements:
            try:
                # Per-requirement filtering: only include summaries relevant to THIS requirement
                relevant = self._filter_summaries_for_requirement(req, summaries)
                batch_results = self._verify_batch_llm([req], relevant)
                if batch_results:
                    results_all.extend(batch_results)
            except Exception as exc:
                console.print(f"      [bold red]Verification error: {exc}[/bold red]")

        # Display and save all results
        for result, req_text in results_all:
            req_text_str = str(req_text)
            short_desc = req_text_str[:60] + "..." if len(req_text_str) > 60 else req_text_str  # type: ignore[index]
            STATUS_COLORS = {"MET": "green", "NOT_MET": "red"}
            color = STATUS_COLORS.get(result.status, "yellow")
            console.print(
                f"      [{color} bold]{result.status:<8}[/{color} bold] "
                f"| {short_desc:<60} | {result.justification}"
            )

    def _filter_summaries_for_requirement(self, req: dict, summaries: list) -> list:
        """Filter summaries to only those relevant to a single requirement."""
        req_text = req.get("requirement", "").lower()
        keywords = set(re.findall(r'[a-z]{4,}', req_text))

        scored: list[tuple[int, dict]] = []
        for s in summaries:
            # Always include auto-detected infra entries
            if "[auto-detected]" in s.get("file_path", ""):
                scored.append((100, s))
                continue
            s_text = (
                s.get("file_path", "") + " " +
                s.get("purpose", "") + " " +
                " ".join(s.get("key_functions", []))
            ).lower()
            hits = sum(1 for kw in keywords if kw in s_text)
            if hits > 0:
                scored.append((hits, s))

        # Sort by relevance (most keyword hits first), cap at 200
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:200]]

    def _verify_batch_llm(self, batch_reqs: list, summaries: list) -> list:
        """Run a single LLM verification call for a batch of requirements."""

        # Ultra-compact summaries: just file path and key function names
        compact_lines = []
        for s in summaries:
            fp = s.get("file_path", "")
            kf = s.get("key_functions", [])
            kf_str = ", ".join(kf[:5]) if kf else s.get("purpose", "")[:60]
            compact_lines.append(f"{fp}: {kf_str}")
        compact_summaries = "\n".join(compact_lines)

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are verifying whether business requirements are implemented in a codebase.\n"
                "You are given a list of files and their key functions.\n"
                "If a function/endpoint name matches or closely relates to the requirement, status=MET.\n"
                "If partial evidence exists, status=PARTIAL.\n"
                "If no evidence at all, status=NOT_MET.\n"
                "Return ONLY valid JSON matching the VerificationResultList schema.",
            ),
            (
                "user",
                "Requirement:\n{requirements}\n\n"
                "Codebase files and functions:\n{summaries}",
            ),
        ])

        try:
            result_list: VerificationResultList = self.provider.invoke_structured(
                prompt=prompt,
                output_schema=VerificationResultList,
                requirements=json.dumps(batch_reqs),
                summaries=compact_summaries,
            )

            if not result_list or not getattr(result_list, "results", None):
                return []

            paired = []
            for result in result_list.results:
                self._save_verification_result(result.requirement_id, result)
                req_text = next(
                    (r["requirement"] for r in batch_reqs if r["id"] == result.requirement_id),
                    f"Req #{result.requirement_id}",
                )
                paired.append((result, req_text))
            return paired

        except Exception as exc:
            console.print(f"      [bold red]Error during batch verification: {exc}[/bold red]")
            return []

    def _get_module_summaries(self) -> list:
        with self.session.get_connection() as conn:
            rows = conn.execute("SELECT * FROM module_summaries").fetchall()
        summaries = []
        for row in rows:
            entry: dict[str, typing.Any] = dict(row) # type: ignore
            try:
                entry["key_functions"] = json.loads(entry.get("key_functions", "[]") or "[]") # type: ignore
            except (json.JSONDecodeError, TypeError):
                entry["key_functions"] = []
            summaries.append(entry)
        return summaries

    def _get_pending_requirements(self) -> list:
        with self.session.get_connection() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT id, requirement FROM brd_requirements WHERE status = 'PENDING'"
                ).fetchall()
            ]

    def _save_verification_result(self, req_id: int, result: VerificationResult) -> None:
        with self.session.get_connection() as conn:
            conn.execute(
                "UPDATE brd_requirements SET status = ?, evidence = ? WHERE id = ?",
                (result.status, json.dumps(result.evidence_files), req_id),
            )

    def _build_system_prompt(self, language: str) -> str:
        """PM agent uses summarize_batch/verify_brd, not the standard analyze() flow."""
        return (
            "You are a Product Manager reviewing this file. Identify dead code — "
            "functions or classes with no callers and no evident user story. "
            "Return ONLY valid JSON matching the schema."
        )