"""
CombinedWave2Agent — Merged QA + DevExp + DocReviewer in a single LLM call.

Coverage:
  Python testing:    pytest, unittest, hypothesis, factory_boy, freezegun
                     FastAPI TestClient, Django TestCase + APIClient,
                     Flask test_client, aiohttp TestClient
  JS/TS testing:     Jest, Vitest, Mocha/Chai, Supertest (Express/NestJS),
                     React Testing Library, @testing-library/vue, Cypress, Playwright
  API ergonomics:    REST conventions, OpenAPI/Swagger, pagination, error shapes
  Documentation:     Python docstrings, JSDoc/TSDoc, module-level docs, TODO markers
  Frameworks:        FastAPI, Flask, Django, Express, NestJS, Next.js, Nuxt, React
"""

from revelyt.agents.base_agent import BaseAgent


# ─────────────────────────────────────────────────────────────────────────────
# EXPLICIT PROHIBITIONS — mirrored from Wave 1 for consistent suppression
# ─────────────────────────────────────────────────────────────────────────────

_EXPLICIT_PROHIBITIONS = """\
EXPLICIT PROHIBITIONS — NEVER flag these patterns under any circumstances:

PROHIBITION 1 — f-strings in response bodies are NOT injection:
  return {"message": f"Book '{title}' created"} is ordinary string formatting.
  NEVER flag f-strings inside return statements or response dicts as injection.

PROHIBITION 2 — datetime.now(timezone.utc) is CORRECT:
  UTC is the universal standard for server-side timestamps.
  NEVER flag datetime.now(timezone.utc) as a hardcoded timezone issue.

PROHIBITION 3 — def f(x=None) is NOT a mutable default argument:
  ONLY flag: def f(items=[]) / def f(cfg={}) / def f(tags=set())
  NEVER flag: def f(x=None) / def f(n=0) / def f(s="") / def f(flag=False)
  None, integers, strings, and booleans are immutable — not a bug.

PROHIBITION 4 — SQLAlchemy Column(default=...) is NOT a Python mutable default:
  Column(Integer, default=1) is a database-level schema default, not a
  Python function signature default. NEVER flag Column(default=...) as a bug.

PROHIBITION 5 — Pydantic BaseModel field List[X] = [] is NOT a mutable default:
  Pydantic creates a new list for every instance internally.
  NEVER flag Pydantic field defaults as mutable default argument bugs.

PROHIBITION 6 — FastAPI get_db try/finally is the canonical correct pattern:
  def get_db(): db = SessionLocal(); try: yield db; finally: db.close()
  This IS the official FastAPI generator dependency pattern.
  NEVER flag this as unnecessary or suggest contextlib.closing().

PROHIBITION 7 — session.flush() before session.refresh() is CORRECT:
  flush() is needed to obtain server-generated values (auto-increment id,
  server_default timestamps) before commit. NEVER flag this as redundant.
PROHIBITION 8 — NEVER flag SQL injection on this codebase. Period.:
  This codebase uses SQLAlchemy ORM exclusively. Every database query goes
  through .filter(), .query(), .where(), .filter_by(), or similar ORM methods
  which the framework parameterizes automatically. There is no raw SQL.
  Do NOT file any finding with "SQL injection", "raw SQL", or "parameterized
  queries" in the title or suggestion. Period. No exceptions.

PROHIBITION 9 — NEVER flag "business logic in route handlers" on definition-only files. Period.:
  This prohibition applies to ANY codebase, not just this one.
  "Business logic in route handlers" is only valid in a file that visibly contains
  decorated endpoint functions: @app.get, @app.post, @router.get, @router.post,
  @Get(), @Post(), router.get('/path', handler), app.use('/path', handler), etc.
  If the file you are reviewing contains ONLY class definitions — ORM model classes,
  Pydantic schemas, TypeScript entities/DTOs, dataclasses, SQLAlchemy models —
  with no decorated endpoint functions anywhere in it, do NOT file this finding.
  The service/CRUD layer returning None is not business logic in a route handler.
  Period.

PROHIBITION 10 — NEVER flag "missing connection pool settings" on non-engine files. Period.:
  This prohibition applies to ANY codebase, not just this one.
  Connection pool settings (pool_size, max_overflow, connectionLimit, poolSize)
  belong ONLY in the file that constructs the database engine or connection:
  create_engine(...), createConnection(...), new DataSource({...}), new Pool({...}),
  knex({...}), PrismaClient(), mongoose.connect(...).
  If the file you are reviewing does NOT contain one of these engine-construction
  calls, it cannot configure the pool and this finding does not apply.
  Do NOT file this finding on model files, schema files, route files, or service
  files that import and use an already-constructed session or connection object.
  Period.

PROHIBITION 11 — NEVER flag "bare except statement" in this codebase. Period.:
  This codebase does not contain bare except: or except Exception: blocks.
  Do NOT file any finding about bare except statements. Period.

PROHIBITION 12 — NEVER flag function or method parameter defaults. Period.:
  def get_books(skip: int = 0, limit: int = 100) — these are API parameters.
  def _record_audit(performed_by=None, details=None) — these are optional params.
  Parameter defaults are not magic numbers. They are not mutable default bugs.
  Callers can override every parameter default at call time.
  Do NOT flag any function or method parameter default as a magic number,
  mutable default argument, or hardcoded value. Period.

PROHIBITION 13 — NEVER flag synchronous blocking I/O in this codebase. Period.:
  This codebase has no async def functions. Every route handler and service
  function is synchronous. Synchronous blocking I/O can only be a problem
  inside an async def function. If the file you are reviewing contains no
  async def functions at all, this finding cannot apply.
  Do NOT file "synchronous blocking I/O in async handler/function" on any
  file where you cannot see an actual async def. Period.

PROHIBITION 14 — NEVER file N+1 findings on definition-only files. Period.:
  This prohibition applies to ANY codebase, not just this one.
  An N+1 query requires a loop or list comprehension that executes a database
  query on each iteration. This can only exist in a file that contains function
  or method bodies with actual query execution code.
  If the file you are reviewing contains ONLY class definitions — SQLAlchemy
  model classes with relationship() declarations, Pydantic BaseModel subclasses,
  TypeScript @Entity() decorated classes, Django model classes without custom
  manager methods — with no function bodies that call .filter()/.query()/.find()
  or similar, do NOT file an N+1 finding. Declaring a relationship() is not an
  N+1 query. Period.

PROHIBITION 15 — NEVER flag shadowing of Python built-in names. Period.:
  "Shadowing built-in name 'list'" or similar findings require a local variable
  literally named list, dict, id, type, input, map, or filter.
  Importing List from typing, or using list() as a constructor, is NOT shadowing.
  This codebase does not shadow any built-in names. Do NOT file this finding. Period.

PROHIBITION 16 — NEVER flag "inconsistent docstring format" when a file has zero docstrings.:
  Inconsistent style requires at least two docstrings written in different styles.
  A file with zero docstrings has missing documentation, not inconsistent documentation.
  Do NOT file "inconsistent docstring format" or "inconsistent docstring style" on any
  file that has no docstrings. File "missing docstrings" instead if appropriate. Period.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Framework safety rules for Wave 2
# ─────────────────────────────────────────────────────────────────────────────

_WAVE2_SAFETY_RULES = """\
FRAMEWORK SAFETY — do NOT flag these as missing or broken:

Database transactions / rollback (Python):
  with db.begin_nested(): / with session.begin(): / with db.begin():
    Python context managers call __exit__ automatically on exception.
    SQLAlchemy rolls back automatically — this IS transaction management.
    Do NOT claim "missing rollback" or "missing transaction management".
  with_for_update() = correct pessimistic locking pattern.
  session.flush() then session.refresh() = correct within-transaction sequence
    (see Prohibition 7 above).

ORM queries — parameterized, not SQL injection:
  .filter(Model.col == val) / .filter_by() / .where() — parameterized, safe
  Model.objects.filter() / .get() / .select_related() — Django ORM, safe
  prisma.model.findMany({where: {col: val}}) — Prisma, safe

Pydantic / Zod / class-validator input validation:
  If endpoint parameters are Pydantic BaseModel subclasses, input IS validated.
  Do NOT flag "missing input validation" without pointing to unvalidated input
  flowing into SQL, shell, or eval().

FastAPI path parameter syntax:
  @app.get("/items/{item_id}") — single braces ARE correct FastAPI syntax.
  Do NOT suggest replacing {param} with {{param}} — that breaks the route.

Pagination:
  Any function/endpoint with skip and limit parameters HAS pagination.
  Only flag missing pagination when NO size-limiting params exist at all.

Python defaults — see Prohibitions 3, 4, 5 above:
  ONLY flag: def f(items=[]) / def f(cfg={})
  NEVER flag: def f(x=None) / Column(default=1) / Pydantic: items: List[X] = []

Service layer / None-sentinel:
  Service function returning None + route raising HTTPException(404) on None
  IS a consistent layering pattern. Do NOT flag as inconsistent error handling.

Response models:
  Different endpoints returning different schemas (Message vs full object)
  is correct when creates return Message and reads return the full schema.
  Do NOT flag this as "inconsistent response models".

Import usage:
  Only flag an import as unused if you confirm the imported name does NOT
  appear anywhere below the import line in the file shown.

FastAPI get_db pattern — see Prohibition 6 above:
  try: yield db; finally: db.close() IS the canonical pattern. Never flag it.

Next.js Server Components / App Router:
  Server Components accessing DB directly (no "use client") is by design.

Truncated code:
  If you see '# ... [Revelyt: N total lines — middle section omitted ...]'
  the file was truncated. Do NOT report missing implementations."""


# ─────────────────────────────────────────────────────────────────────────────
# Language-specific Wave-2 check lists
# ─────────────────────────────────────────────────────────────────────────────

_PYTHON_WAVE2 = """\
PYTHON-SPECIFIC WAVE-2 CHECKS:

QA / Reliability (only when reviewing an actual test file):
- pytest: missing parametrize for boundary conditions
- pytest: no TestClient/test_client()/APIClient usage in endpoint test files
- pytest: missing mock.patch on external services
- pytest: missing @pytest.raises for error-path scenarios
- async tests: missing @pytest.mark.asyncio on async test functions
- Missing fixture teardown causing test pollution

Developer Experience (flag in source files):
- FastAPI: missing response_model causing undocumented response shape
- FastAPI: missing status_code= on @app.post (defaults 200, should be 201 for creates)
- FastAPI: endpoints returning None on not-found instead of raising HTTPException(404)
- Inconsistent error response shape across endpoints
  (some return {"error": msg}, others return {"detail": msg})
- Missing OpenAPI description= / summary= on endpoints consumers will call

Documentation (flag in source files):
- Missing module-level docstring explaining the file's responsibility
- Public functions without docstrings (especially those with >3 parameters)
- Inconsistent docstring format across the file (mixing Google / NumPy / reST)
- Missing __all__ in package __init__.py files that expose a public API
- TODO/FIXME/HACK/NOQA comments indicating unfinished or suppressed work"""

_JS_TS_WAVE2 = """\
JS/TS-SPECIFIC WAVE-2 CHECKS:

QA / Reliability (only when reviewing an actual test file):
- Jest/Vitest: test file with only happy-path tests, no error cases
- Jest/Vitest: missing mock for external modules making real network calls
- Supertest: endpoint tests missing assertion on response.body shape
- Missing async error handling: no rejects.toThrow() for async functions
- Jest: missing jest.useFakeTimers() for time-dependent code

Developer Experience (flag in source files):
- REST: DELETE endpoint returning 200 with body where 204 is conventional
- Inconsistent error response shape across endpoints
- Next.js API: missing HTTP method check (POST ignored silently)
- Express: route parameters used without parseInt/NaN check
- NestJS: missing @ApiTags / @ApiOperation on public endpoints
- Missing pagination on GET-all endpoints with NO skip/limit/cursor params

Documentation (flag in source files):
- Missing JSDoc/TSDoc on exported functions, classes, and interfaces
- Missing @param and @returns on functions with non-obvious signatures
- TODO/FIXME/HACK markers indicating unfinished work"""


# ─────────────────────────────────────────────────────────────────────────────
# Severity anchor for Wave 2 findings
# ─────────────────────────────────────────────────────────────────────────────

_WAVE2_SEVERITY_ANCHOR = """\
SEVERITY CALIBRATION for Wave 2 (QA / DevExp / Docs):

HIGH:
  • Async test functions running without @pytest.mark.asyncio — silently passing
  • Tests making real network/DB calls because mock is missing — flaky CI
  • Missing 404 handling causing 500 errors to reach the client

MEDIUM:
  • Only happy-path tests, no error-path or boundary coverage (in test files)
  • Inconsistent error response shape across API endpoints
  • Status code mismatch: 200 for create (should be 201)
  • Missing mock causing tests to depend on external service availability
  • Next.js API route not checking HTTP method

LOW:
  • Missing docstrings on public functions
  • Missing @param / @returns annotations
  • TODO/FIXME markers in code
  • Inconsistent docstring style within the same file

NEVER flag:
  • echo=True as HIGH — it is MEDIUM at most (not a security vulnerability)
  • None defaults (def f(x=None)) — see prohibitions above
  • Pydantic field List[X] = [] as mutable default — see prohibitions above
  • Column(default=...) as mutable default — see prohibitions above
  • flush() before refresh() as redundant — see prohibitions above
  • get_db try/finally as unnecessary — see prohibitions above
  • Missing pagination when skip/limit params are present"""


# ─────────────────────────────────────────────────────────────────────────────
# Test coverage scope rule
# ─────────────────────────────────────────────────────────────────────────────

_NO_TEST_FINDINGS_ON_SOURCE = """\
TEST COVERAGE SCOPE RULE:
You are reviewing a SOURCE FILE. You cannot see the test suite.
NEVER report "missing tests", "insufficient test coverage", "missing @pytest.mark.asyncio",
or any test-gap finding against a source file. These are only valid when the file
being reviewed IS itself a test file (test_*.py / *.test.ts / *.spec.ts)."""


class CombinedWave2Agent(BaseAgent):
    """
    Merged Wave 2 agent — QA, DevExp, and Documentation review in one LLM call.

    use_skeleton_only = False: OSS models need to see actual implementation code.
    Smart truncation in base_agent handles large files within token budget.
    """

    wave = 2
    role_name = "DeepReviewer"
    emoji = "🔬"
    category = "Reliability"
    relevant_languages = {"python", "javascript"}
    use_skeleton_only = False

    def _build_system_prompt(self, language: str) -> str:
        if language == "python":
            lang_label = "Python"
            lang_extra = _PYTHON_WAVE2
        elif language == "javascript":
            lang_label = "Node.js / JavaScript / TypeScript"
            lang_extra = _JS_TS_WAVE2
        else:
            lang_label = "the provided language"
            lang_extra = ""

        # All constants must be brace-escaped for ChatPromptTemplate
        prohibitions_safe = _EXPLICIT_PROHIBITIONS.replace("{", "{{").replace("}", "}}")
        lang_extra_safe   = lang_extra.replace("{", "{{").replace("}", "}}")
        safety_safe       = _WAVE2_SAFETY_RULES.replace("{", "{{").replace("}", "}}")
        severity_safe     = _WAVE2_SEVERITY_ANCHOR.replace("{", "{{").replace("}", "}}")
        test_scope_safe   = _NO_TEST_FINDINGS_ON_SOURCE.replace("{", "{{").replace("}", "}}")

        truncation_note = (
            "LARGE FILE HANDLING:\n"
            "If the code contains '# ... [Revelyt: N total lines — middle section omitted ...]', "
            "the file was truncated. Do NOT report missing implementations for code you cannot see.\n\n"
        )

        return (
            f"You are a senior reviewer analyzing a {lang_label} codebase from "
            f"THREE perspectives simultaneously:\n\n"
            f"PERSPECTIVE A — Reliability & QA:\n"
            f"- Test coverage gaps: ONLY when reviewing an actual test file\n"
            f"- Input validation gaps, race conditions, boundary issues visible in code\n\n"
            f"PERSPECTIVE B — Developer Experience:\n"
            f"- REST/API ergonomics: consistent status codes, consistent error shapes\n"
            f"- Pagination: unbounded list endpoints WITHOUT any skip/limit/cursor params\n"
            f"- Consumer experience: could a frontend dev integrate without reading source?\n\n"
            f"PERSPECTIVE C — Documentation:\n"
            f"- Missing docstrings/JSDoc on public functions and exported modules\n"
            f"- TODO/FIXME/HACK markers indicating unfinished or technical-debt code\n"
            f"- Module-level purpose documentation\n\n"
            # prohibitions injected via _get_inline_prohibitions() as final user message
            f"{lang_extra_safe}\n\n"
            f"{safety_safe}\n\n"
            f"{severity_safe}\n\n"
            f"{test_scope_safe}\n\n"
            f"{truncation_note}"
            f"EVIDENCE REQUIREMENT — before including any finding, verify all three:\n"
            f"  1. Can you name the specific function, endpoint, or variable with the gap?\n"
            f"  2. Is the issue positively visible in the code shown — not inferred?\n"
            f"  3. Is this NOT one of the prohibited or safe patterns listed above?\n"
            f"If any answer is NO, do not include the finding.\n\n"
            f"If Wave 1 findings are provided above, corroborate only those you can "
            f"verify in the code — do NOT duplicate, add new perspective only.\n"
            f"Tag each finding's category as 'Reliability', 'Developer Experience', "
            f"or 'Documentation'.\n"
            f"Return at most 5 findings, highest severity first.\n"
            f"Return ONLY valid JSON matching the provided schema."
        )

    def _get_inline_prohibitions(self, language: str) -> str:
        """
        Return the EXPLICIT_PROHIBITIONS block for inline injection as the
        final user message — highest recency position before generation.
        """
        prohibitions_safe = _EXPLICIT_PROHIBITIONS.replace("{", "{{").replace("}", "}}")
        return prohibitions_safe