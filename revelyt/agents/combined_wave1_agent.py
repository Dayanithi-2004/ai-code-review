"""
CombinedWave1Agent — Merged Architect + Developer in a single LLM call.

Coverage:
  Python frameworks:  FastAPI, Flask, Django (+ DRF), Starlette, aiohttp, Celery
  Python DB layers:   SQLAlchemy (Core + ORM), Django ORM, Tortoise-ORM, databases
  Python validation:  Pydantic v1/v2, Marshmallow, WTForms, Django Forms/Serializers
  Python auth:        FastAPI Security/Depends, Django auth, python-jose, passlib
  Python testing:     pytest, unittest, hypothesis

  JS/TS frameworks:   React, Next.js (App + Pages router), Nuxt 3, Vue 3,
                      Express, NestJS, Fastify, Koa, Hapi, Node.js (native)
  JS/TS DB layers:    Prisma, TypeORM, Sequelize, Mongoose, Drizzle, Knex, pg/mysql2
  JS/TS validation:   Zod, Joi, Yup, class-validator, express-validator, valibot
  JS/TS auth:         next-auth, Passport.js, NestJS Guards, jsonwebtoken, Auth.js
  JS/TS testing:      Jest, Vitest, Mocha, Cypress, Playwright
"""

from revelyt.agents.base_agent import BaseAgent


# ─────────────────────────────────────────────────────────────────────────────
# EXPLICIT PROHIBITIONS — placed first so the model hits these before reasoning
#
# Strategy: these are NOT in the whitelist (which models can rationalize around).
# They are hard NEVER-FLAG rules for patterns the model has repeatedly hallucinated
# across multiple runs. Each rule names the exact false positive pattern observed.
# ─────────────────────────────────────────────────────────────────────────────

_EXPLICIT_PROHIBITIONS = """\
EXPLICIT PROHIBITIONS — NEVER flag these patterns under any circumstances:

PROHIBITION 1 — f-strings in response bodies are NOT injection:
  return {"message": f"Book '{title}' created"} — the {title} here is a
  Python variable substitution in a string literal returned to the caller.
  It is NOT a URL path parameter, NOT SQL, NOT a shell command.
  NEVER flag f-strings inside return statements or response dicts as injection,
  path traversal, or security issues. They are ordinary string formatting.

PROHIBITION 2 — datetime.now(timezone.utc) is CORRECT, not a config issue:
  UTC is the universal standard for server-side timestamps. Using timezone.utc
  is intentional best practice — it does not need to be "configurable".
  NEVER flag datetime.now(timezone.utc) or timezone.utc as a hardcoded timezone
  issue or suggest injecting the timezone as a dependency.

PROHIBITION 3 — def f(x=None) is NOT a mutable default argument:
  The Python mutable default argument bug ONLY applies to mutable containers:
    ONLY flag: def f(x=[])  /  def f(cfg={})  /  def f(tags=set())
  NEVER flag: def f(x=None) / def f(n=0) / def f(s="") / def f(flag=False)
  None, integers, strings, and booleans are immutable in Python.
  If every default in a function is None or a primitive, there is NO bug.

PROHIBITION 4 — SQLAlchemy Column(default=...) is NOT a Python mutable default:
  Column(Integer, default=1) is a DATABASE-LEVEL column default stored in the
  schema definition. It is not a Python function default argument at all.
  The Python mutable default gotcha only applies to `def f(x=value)` signatures.
  NEVER flag Column(default=anything) as a mutable default argument.

PROHIBITION 5 — Pydantic BaseModel field List[X] = [] is NOT a mutable default:
  class MyModel(BaseModel): items: List[X] = []
  Pydantic's metaclass creates a new list for every model instance internally.
  The Python mutable default gotcha does NOT apply to Pydantic BaseModel fields.
  NEVER flag Pydantic model field defaults as mutable default argument bugs.

PROHIBITION 6 — FastAPI get_db try/finally is the canonical correct pattern:
  def get_db():
      db = SessionLocal()
      try:
          yield db
      finally:
          db.close()
  This IS the official FastAPI/Starlette generator dependency pattern.
  contextlib.closing() CANNOT be used with yield-based dependencies.
  NEVER flag this as "unnecessary try/finally" or suggest refactoring it.

PROHIBITION 7 — session.flush() before session.refresh() is CORRECT:
  flush() sends pending SQL to the DB without committing, which is required to
  obtain server-generated values (auto-increment id, server_default timestamps)
  before the transaction commits. refresh() then loads those values into the
  Python object. This is the correct SQLAlchemy pattern for within-transaction
  value retrieval. NEVER flag flush() before refresh() as redundant or unnecessary.
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
# Anti-pattern checklists (what to actively look for)
# ─────────────────────────────────────────────────────────────────────────────

_PYTHON_PATTERNS = """\
PYTHON ANTI-PATTERNS TO DETECT:

Security:
- eval() / exec() with any non-literal, non-constant input
- subprocess.run/Popen/call with shell=True + user-controlled string
- pickle.loads() / pickle.load() on untrusted or network-received data
- yaml.load(data) without Loader=yaml.SafeLoader (use yaml.safe_load)
- Raw SQL with f-strings or string concat: cursor.execute(f"...{val}") or
  session.execute(text(f"...{val}")) or "SELECT..." + user_input
- Hardcoded secrets: passwords, API keys, JWT secrets, tokens in source files
- Insecure hash: hashlib.md5/sha1 for password hashing (use bcrypt/argon2/pbkdf2)

Architecture:
- Business logic in route handlers/views instead of service/repository layer
- Missing connection pool config on production DB engine (pool_size, max_overflow)
- Synchronous blocking I/O inside async def: requests.get(), time.sleep(),
  open() for large files, without asyncio.to_thread() or run_in_executor()
- N+1 query: ORM query inside a loop without joinedload/selectinload/prefetch_related
- Celery tasks calling ORM directly without proper session management
- Missing circuit breaker / retry on external HTTP calls

Code Quality:
- Mutable default arguments: def f(items=[]) / def f(cfg={}) — see Prohibition 3
- Bare except: except: or except Exception: swallowing all errors silently
- Not using context managers for DB sessions, file handles, locks
- Missing type hints on public API function signatures and return types
- Shadowing Python builtins: list, dict, id, type, input, map, filter as variable names
- Star imports polluting module namespace: from module import *
- Magic numbers without named constants — ONLY unexplained NUMERIC literals"""

_JS_TS_PATTERNS = """\
JS/TS ANTI-PATTERNS TO DETECT:

Security:
- Template literal SQL without tagged-template safety:
  db.query(`SELECT ... WHERE id=${req.params.id}`) — injection risk
- innerHTML / outerHTML / document.write set to user-controlled value — XSS
- eval(userInput) / new Function(userCode) — code injection
- child_process.exec/execSync with shell string containing user data
- dangerouslySetInnerHTML={{__html: userValue}} without DOMPurify sanitization
- Hardcoded API keys, JWT secrets, database URLs, tokens in source files
- Prototype pollution: Object.assign({}, req.body) / {...req.body} into shared state

Architecture:
- Business logic mixed into route handlers / controllers instead of service classes
- Blocking synchronous calls inside async handlers:
  fs.readFileSync, crypto.pbkdf2Sync, JSON.parse of huge payloads on main thread
- await inside for...of loop when calls are independent — use Promise.all()
- Unhandled promise rejections: .then() without .catch(), async without try/catch
- Missing error handling middleware in Express (4-param handler at the end)
- Circular imports between modules causing runtime errors in large codebases

Code Quality:
- var declarations — use const or let
- == instead of === loose equality comparisons
- any type in TypeScript on auth, payment, or data-pipeline functions
- Missing await on async function calls (returns Promise instead of resolved value)
- React useEffect with missing dependency array causing stale closures or infinite loops
- console.log / console.debug left in production route handlers
- Magic numbers inline (setTimeout(fn, 86400000) vs TWENTY_FOUR_HOURS_MS)
- Empty catch blocks: catch(e) {} silently discarding errors"""

_CONFIG_PATTERNS = """\
CONFIG/ENV FILE ANTI-PATTERNS TO DETECT:
- Any password, secret, API key, token, or private key as a plain string value
- Database connection strings with credentials embedded: postgresql://user:pass@host/db
- AWS_SECRET_ACCESS_KEY, STRIPE_SECRET_KEY, SENDGRID_API_KEY, TWILIO_AUTH_TOKEN, etc.
- JWT_SECRET or SESSION_SECRET set to a short, guessable, or hardcoded value
- DEBUG = True / NODE_ENV = development / echo = True in a production config file
- Private RSA/EC key or certificate pasted inline
- CORS origin set to * in a config file intended for production"""


# ─────────────────────────────────────────────────────────────────────────────
# Framework safety rules (what NOT to flag)
# ─────────────────────────────────────────────────────────────────────────────

_PYTHON_FRAMEWORK_SAFETY = """\
PYTHON FRAMEWORK SAFETY — the following are CORRECT patterns, do NOT flag them:

SQLAlchemy ORM / Core (Flask-SQLAlchemy, FastAPI+SA, SQLModel, Alembic):
  SAFE — parameterized, not SQL injection:
    session.query(Model).filter(Model.col == value)
    session.query(Model).filter_by(col=value)
    select(Model).where(Model.col == value)
    session.execute(text("SELECT ... WHERE id = :id"), {"id": user_id})
  SAFE — correct transaction / rollback patterns:
    with db.begin_nested(): / with session.begin(): / with db.begin():
      Python context managers call __exit__ automatically on exception.
      SQLAlchemy rolls back the transaction — this IS transaction management.
      Do NOT flag as "missing rollback" or "missing transaction management".
    with_for_update() / with_for_update(skip_locked=True) — pessimistic row lock.
    async with AsyncSession() as session: — correct async session management.
  SAFE — correct session lifecycle (see Prohibitions 6 and 7 above):
    session.flush() then session.refresh() — correct within-transaction sequence.
    expire_on_commit=False in sessionmaker — intentional, avoids lazy-load errors.
  SAFE — correct enum/column config:
    Column(SQLAEnum(MyEnum, native_enum=False)) — VARCHAR-backed enum, intentional.
  UNSAFE (only flag these):
    cursor.execute("SELECT ... WHERE id=" + str(val))
    session.execute(text(f"SELECT ... WHERE id={val}"))

Django ORM:
  SAFE: Model.objects.filter(col=value) / .get(pk=pk) / .exclude() / .annotate()
  SAFE: Q(col=value) | Q(other=val)
  SAFE: Model.objects.raw("SELECT ... WHERE col = %s", [value])
  SAFE: select_related("fk"), prefetch_related("m2m") — N+1 fix, not a problem
  UNSAFE (only flag): Model.objects.raw(f"... WHERE col = {value}")

Pydantic v1 / v2:
  SAFE — IS input validation:
    Any function parameter typed as a Pydantic BaseModel subclass
    @validator / @field_validator / @model_validator
    Field(..., ge=0, le=100, min_length=1)
  RULE: Do NOT flag "missing input validation" unless you can point to a specific
  line where raw unvalidated user input flows directly into SQL, shell, or eval().

FastAPI path parameter syntax:
  SAFE: @app.get("/items/{item_id}") — SINGLE braces are CORRECT FastAPI syntax.
  CRITICAL: Do NOT suggest replacing {param} with {{param}} — that breaks the route.

FastAPI general:
  SAFE: Depends(get_db) / Depends(get_current_user) / Security(oauth2_scheme)
  SAFE: HTTPException(status_code=..., detail="...") — correct error handling
  SAFE: response_model=SomeSchema — output serialization/filtering

Flask:
  SAFE: @app.route("/items/<int:item_id>") — type converter validates
  NOTE: Only flag request.args/form/json if raw value directly reaches SQL/eval.

Django (web framework + DRF):
  SAFE: Django Forms / ModelForms / DRF Serializers — IS validation
  SAFE: @login_required / @permission_required / IsAuthenticated — IS auth guard

Python default argument rules — see Prohibitions 3, 4, 5 above:
  ONLY flag: def f(items=[]) / def f(cfg={}) / def f(tags=set())
  NEVER flag: def f(x=None) / Column(default=1) / Pydantic field: x: List = []

Library API string keys — not magic numbers:
  SAFE: {"check_same_thread": False} — SQLite C driver parameter, not magic.
  SAFE: {"Content-Type": "application/json"} — HTTP header name, not magic.
  RULE: Only flag magic numbers for unexplained NUMERIC literals.

Service layer / None-sentinel error handling:
  SAFE: Service function returning None + route raising HTTPException(404) on None.
  This IS a consistent deliberate layering pattern. Do NOT flag as "missing error
  handling in service layer" or "inconsistent error handling".

Pagination rules:
  SAFE: Any function/endpoint with skip and limit parameters HAS pagination.
  Do NOT flag "missing pagination" when skip/limit params are present.

Import usage:
  RULE: Only flag an import as unused if the imported name does NOT appear
  anywhere in the file body. Scan the entire file before making this claim.

Python async / concurrency:
  SAFE: async def with await db.execute() — correct async I/O
  SAFE: asyncio.gather(*tasks) / asyncio.create_task() — correct concurrency
  UNSAFE: requests.get() / time.sleep() directly inside async def (blocking)"""

_JS_TS_FRAMEWORK_SAFETY = """\
JS/TS FRAMEWORK SAFETY — the following are CORRECT patterns, do NOT flag them:

Prisma:
  SAFE: prisma.model.findMany({where: {col: value}}) — parameterized
  SAFE: prisma.$queryRaw`SELECT ... WHERE id = ${id}` — tagged template IS safe
  UNSAFE (only flag): prisma.$queryRawUnsafe(`SELECT ... WHERE id = ${id}`)

TypeORM:
  SAFE: repo.find({where: {col: value}})
  SAFE: repo.createQueryBuilder("e").where("e.col = :val", {val: value})
  UNSAFE: .where(`e.col = ${value}`) — string interpolation in query builder

Sequelize:
  SAFE: Model.findAll({where: {col: value}}) / {where: {[Op.eq]: value}}
  UNSAFE: sequelize.query("SELECT ... WHERE col = " + value)

Mongoose:
  SAFE: Model.find({col: value}) / .findById(id) / .findOne({col: value})
  FLAG ONLY: Model.find(req.body) — entire body as query (NoSQL injection risk)

Zod / Joi / Yup / class-validator:
  SAFE — IS validation: z.object({...}).parse(req.body), Joi.object({...}).validate()
  RULE: Only flag missing validation if raw user input demonstrably reaches SQL/shell/eval.

NestJS:
  SAFE: @Controller() / @Get() / @Post() / @Body() / @Param() / @Query()
  SAFE: @UseGuards(JwtAuthGuard) / @Roles("admin") — IS authentication/authorization
  SAFE: @UsePipes(ValidationPipe) — IS validation
  SAFE: @Injectable() / @Module({...}) — correct DI

Express:
  SAFE: router.get("/:id", handler) + parseInt(req.params.id, 10)
  SAFE: express-validator / helmet() / cors({origin: allowList})
  NOTE: req.params/query/body without validation is weaker — flag ONLY if it
        directly reaches SQL string, shell command, or eval().

Next.js (App Router + Pages Router):
  SAFE: export async function GET/POST(req: NextRequest) — App Router handlers
  SAFE: Server Components accessing DB directly (no "use client") — by design

React:
  SAFE: {variable} in JSX — React escapes HTML by default, NOT XSS
  SAFE: useState / useEffect / useRef / useCallback / useMemo / useContext
  UNSAFE: dangerouslySetInnerHTML={{__html: userValue}} — only flag with user input
  REAL BUG: useEffect with missing dependency array — DO flag this

Auth patterns:
  SAFE: jwt.verify(token, secret, options) — correct signature verification
  SAFE: bcrypt.compare(plain, hash) — correct password check
  FLAG: jwt.decode() without verify() — skips signature verification"""


# ─────────────────────────────────────────────────────────────────────────────
# Severity calibration anchor
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY_ANCHOR = """\
SEVERITY CALIBRATION — apply this scale exactly, do not inflate:

CRITICAL — directly exploitable, attacker needs no special privileges:
  • Hardcoded credential: DATABASE_URL="postgresql://admin:pass@host/db" in source
  • JWT_SECRET = "mysecret" / STRIPE_SECRET_KEY = "sk_live_abc" in code
  • eval(request.data) or exec(user_input) — arbitrary code execution
  • subprocess.run(f"cmd {url}", shell=True) — shell injection with user input
  • dangerouslySetInnerHTML={{__html: req.body.content}} — XSS with user data
  • Private RSA/EC/SSH key committed to repository

HIGH — real security vulnerability, requires attacker context:
  • Raw SQL string concatenation: cursor.execute("...WHERE id=" + str(val))
  • Template literal in non-tagged SQL: db.query(`SELECT...WHERE id=${val}`)
  • yaml.load(data) without SafeLoader on external data
  • Missing authentication on a route that modifies records or exposes PII
  • jwt.decode() used without jwt.verify() — signature not checked

MEDIUM — architectural flaw, not directly exploitable:
  • DRY violation (duplicate CRUD blocks, repeated route patterns)
  • echo=True / DEBUG=True hardcoded in application code
  • Missing connection pool settings (pool_size, max_overflow) on production DB
  • Synchronous blocking I/O inside async handler (requests.get in async def)
  • N+1 query: ORM call inside a loop without batch loading
  • Incorrect HTTP status codes: 200 for created resource (should be 201)
  • Missing error handling on external API calls

LOW — style, documentation, minor convention:
  • Missing docstrings / JSDoc on public functions or exported classes
  • Missing type hints on internal helper functions
  • console.log / print() in production code
  • Magic number without named constant (NUMERIC literals only)
  • Mutable default argument: def f(x=[]) or def f(cfg={})
  • TODO/FIXME markers indicating incomplete work

NEVER inflate these:
  DRY/SRP violation → max MEDIUM, never CRITICAL or HIGH
  Missing docstrings → LOW only
  echo=True → MEDIUM only, never HIGH (not a security vulnerability)
  console.log → LOW only
  Missing pagination → MEDIUM only, and ONLY when skip/limit are absent
  Code style → LOW only"""


# ─────────────────────────────────────────────────────────────────────────────
# Test coverage scope rule
# ─────────────────────────────────────────────────────────────────────────────

_NO_TEST_FINDINGS_ON_SOURCE = """\
TEST COVERAGE SCOPE RULE:
You are reviewing a SOURCE FILE. You cannot see the test suite.
NEVER report "missing tests", "insufficient test coverage", "missing @pytest.mark.asyncio",
or any test-gap finding against a source file. These are only valid when the file
being reviewed IS itself a test file (test_*.py / *.test.ts / *.spec.ts)."""


class CombinedWave1Agent(BaseAgent):
    """
    Merged Wave 1 agent — architectural review + code quality in one LLM call.
    """

    wave = 1
    role_name = "Reviewer"
    emoji = "🔍"
    category = "Architecture"

    # Frontier skeleton threshold: files over 80 lines get skeleton mode
    # on frontier models (Gemini, GPT, Claude). Wave 1 is an architectural
    # pass — function signatures + decorators carry the structural signal.
    # Wave 2 (CombinedWave2Agent) leaves this at None (needs full bodies).
    frontier_skeleton_threshold = 80

    def _build_system_prompt(self, language: str) -> str:
        if language == "python":
            lang_label = "Python"
            lang_patterns = _PYTHON_PATTERNS
            framework_safety = _PYTHON_FRAMEWORK_SAFETY
        elif language == "javascript":
            lang_label = "Node.js / JavaScript / TypeScript"
            lang_patterns = _JS_TS_PATTERNS
            framework_safety = _JS_TS_FRAMEWORK_SAFETY
        elif language == "config":
            lang_label = "configuration / environment"
            lang_patterns = _CONFIG_PATTERNS
            framework_safety = ""
        else:
            lang_label = "the provided language"
            lang_patterns = ""
            framework_safety = ""

        # All constants must be brace-escaped for ChatPromptTemplate
        prohibitions_safe   = _EXPLICIT_PROHIBITIONS.replace("{", "{{").replace("}", "}}")
        lang_patterns_safe  = lang_patterns.replace("{", "{{").replace("}", "}}")
        framework_safe      = framework_safety.replace("{", "{{").replace("}", "}}")
        severity_safe       = _SEVERITY_ANCHOR.replace("{", "{{").replace("}", "}}")
        test_scope_safe     = _NO_TEST_FINDINGS_ON_SOURCE.replace("{", "{{").replace("}", "}}")

        truncation_note = (
            "LARGE FILE HANDLING:\n"
            "If the code contains '# ... [Revelyt: N total lines — middle section omitted ...]', "
            "the file was truncated. Do NOT report missing features or implementations based on "
            "truncated output. Only report issues positively visible in the code shown.\n\n"
        )

        # In combined_wave1_agent.py, update the bottom return statement:

        return (
            f"You are a senior code reviewer analyzing a {lang_label} codebase. "
            f"Review from TWO perspectives simultaneously:\n\n"
            f"PERSPECTIVE A — Architecture & Security:\n"
            f"- Security: hardcoded secrets, injection vectors, OWASP Top 10\n"
            f"- Layering: business logic leaking into route/controller layer\n"
            f"- Scalability: N+1 queries, missing connection pools, blocking I/O in async\n"
            f"- Deployment: debug flags, echo=True, env-specific values hardcoded\n\n"
            f"PERSPECTIVE B — Code Quality:\n"
            f"- SOLID: SRP (mixed-concern files), DRY (duplicated logic blocks)\n"
            f"- Anti-patterns: magic numbers, God objects, bare excepts\n"
            f"- Error handling: silently swallowed exceptions, missing retries\n"
            f"- Language best practices for the detected framework\n\n"
            f"{lang_patterns_safe}\n\n"
            f"{framework_safe}\n\n"
            f"{severity_safe}\n\n"
            f"{test_scope_safe}\n\n"
            f"{truncation_note}"
            f"EVIDENCE REQUIREMENT — before including any finding, verify all three:\n"
            f"  1. Can you name the exact function, class, or variable with the problem?\n"
            f"  2. Is the problematic pattern literally present in the code shown above?\n"
            f"  3. Is this NOT one of the prohibited or safe patterns listed above?\n"
            f"If any answer is NO, do not include the finding.\n\n"
            f"For each finding, you MUST provide:\n"
            f"- `line_number`: The exact integer line number where the issue occurs.\n"
            f"- `class_or_method`: The class and/or method name where the issue resides (use 'Global' if module-level).\n\n"
            f"Tag each finding's category as 'Architecture' or 'Code Quality'.\n"
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