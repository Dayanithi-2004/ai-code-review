# 🛡️ Revelyt PR Reviewer — Enterprise-Grade Pull Request Analysis

> AI-powered + deterministic hybrid code review engine that catches what humans miss.

---

## ⚡ Quick Start

```bash
# 1. Install Revelyt
cd /Users/bootlabs/Desktop/Revelyt/Revelyt
pip install -e .

# 2. Set your tokens
export GITHUB_TOKEN="ghp_your_github_token_here"
export GOOGLE_API_KEY="your_gemini_api_key"       # Or OPENAI_API_KEY for OpenAI

# 3. Review a PR  (deep analysis + AI)
revelyt pr check owner/repo 42 --mode deep

# 4. Review without AI  (deterministic rules only — zero cost, zero hallucination)
revelyt pr check owner/repo 42 --no-ai

# 5. Review local changes before pushing
git diff HEAD~1 | revelyt pr diff --mode balanced

# 6. Dry-run  (see results without posting to GitHub)
revelyt pr check owner/repo 42 --dry-run
```

---

## 📋 All CLI Commands

| Command | Description |
|---------|-------------|
| `revelyt pr check <repo> <pr_number>` | Full PR review (deterministic + AI) |
| `revelyt pr check <repo> <pr_number> --no-ai` | Rules-only mode (instant, free) |
| `revelyt pr check <repo> <pr_number> --mode fast` | Lightweight scan for large PRs |
| `revelyt pr check <repo> <pr_number> --mode balanced` | Standard depth analysis |
| `revelyt pr check <repo> <pr_number> --mode deep` | Full depth + AI + cross-module |
| `revelyt pr check <repo> <pr_number> --dry-run` | Preview results without posting to GitHub |
| `revelyt pr diff` | Review local `git diff` from stdin |
| `revelyt pr diff HEAD~1 --mode deep` | Review last commit locally |

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `deep` | Review depth: `fast`, `balanced`, `deep` |
| `--no-ai` | `false` | Disable LLM calls (deterministic rules only) |
| `--dry-run` | `false` | Show results in CLI without posting comments |
| `--provider` | `gemini` | AI provider: `gemini`, `openai`, `anthropic` |
| `--model` | `gemini-2.0-flash` | Specific LLM model to use |

---

## 🏗️ Architecture Overview

```
PR Webhook / CLI Input
         │
         ▼
┌───────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                        │
│    (github_client.py → pipeline → commenter)          │
└───────────────────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────────────┐
│              9-STAGE REVIEW PIPELINE                   │
│                                                        │
│  1. PARSE       Raw diff → structured FileChunks       │
│  2. FILTER      Drop binaries/images, classify tiers   │
│  3. LIMITS      Check PR size, auto-downgrade mode     │
│  4. CHUNK       Split large diffs at logical boundaries│
│  5. ENRICH      Semantic analysis + complexity + risk   │
│  6. CONTEXT     RAG - find definitions, callers, tests  │
│  7. ANALYZE     Hybrid Engine (Rules → AI → Security)  │
│  8. RISK SCORE  Calculate PR risk (0-10)               │
│  9. DEDUP       Remove duplicate findings              │
└───────────────────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────────────┐
│                   OUTPUT LAYER                         │
│  • GitHub inline comments (per-line)                   │
│  • PR summary comment (risk score + table)             │
│  • GitHub Check Run (with annotations)                 │
│  • Rich CLI terminal output                            │
└───────────────────────────────────────────────────────┘
```

---

## 🚀 Complete Feature List

### 1. 🔍 Intelligent Diff Analysis (`diff/`)

| Module | What It Does |
|--------|-------------|
| **parser.py** | Parses unified diffs into structured `FileChunk` objects with metadata. Handles multi-hunk files, renames, binary detection, and 20+ language mappings. |
| **semantic_diff.py** | AST-powered semantic analysis. Detects function signature changes, calculates cyclomatic complexity (Python via AST, JS via regex), and classifies change types (`LOGIC_CHANGED`, `SIGNATURE_CHANGED`). |
| **chunker.py** | Splits oversized diffs at logical boundaries (function/class defs) so each chunk fits within LLM context windows. Falls back to fixed-size splitting. |
| **impact_analyzer.py** | Multi-factor risk scoring: path sensitivity (auth/payment/config), change magnitude, dependency files, complexity, and historical bug weight. |
| **test_impact.py** | Detects missing test updates. Maps source files to expected test files and suggests specific `pytest` / `jest` / `go test` commands. |
| **change_graph.py** | Builds a Python import dependency graph and detects ripple effects when high-impact files are modified. |

### 2. ⚙️ Analysis Engines (`engines/`)

| Engine | Type | What It Catches |
|--------|------|-----------------|
| **rule_engine.py** | Deterministic | `eval()`, `exec()`, SQL injection (f-strings + concat), `subprocess shell=True`, `pickle.load()`, `yaml.load()`, bare excepts, wildcard imports, `innerHTML`, debugger breakpoints, TODO/FIXME markers, global statements. **15+ rules with AST analysis.** |
| **sensitive_data.py** | Deterministic | AWS keys (AKIA), GCP API keys, GitHub/GitLab/Slack/Stripe/Twilio/SendGrid tokens, private keys, JWT secrets, hardcoded passwords, database connection strings, credit card numbers, SSNs, emails. **30+ patterns.** False-positive filtering (comments, placeholders, allowlisted files). |
| **dependency_scanner.py** | Deterministic | Scans `requirements.txt`, `package.json`, `go.mod`, etc. Checks added/updated packages against a vulnerability database. Flags unpinned dependencies (`*`, `latest`). CWE-1104 / OWASP A06:2021 mapping. |
| **ai_request.py** | AI | Multi-provider LLM client (Gemini/OpenAI/Anthropic via LangChain). Structured prompt construction with RAG context injection. Async batch execution with semaphore-based concurrency. Exponential backoff retry. JSON response parsing with markdown fence stripping. |
| **ai_validation.py** | AI | Two-pass self-consistency checking. Uncertain language detection ("might be", "possibly"). Risk-weighted confidence scoring. Automatic rejection of findings below 0.6 confidence threshold. |
| **patch_validator.py** | Deterministic | Validates AI-suggested fixes before showing to developers. Python: full AST parse + definition-loss detection. JS: bracket balance + syntax checks. Ensures 100% trustable autofixes. |
| **risk_engine.py** | Deterministic | Calculates PR Risk Score (0-10) from: change magnitude, sensitive file paths, code complexity, security findings, test coverage gaps, dependency changes. Outputs markdown risk report with level (🟢 Low → 🔴 Critical). |
| **hybrid_engine.py** | Coordinator | Runs deterministic engines on ALL chunks first (free, instant). Then sends only high-risk chunks (risk ≥ 2.0) to AI for deeper analysis. Validates AI patches before presenting. Supports `--no-ai` mode. |

### 3. 🧠 Context Retrieval — RAG for Code (`context/`)

| Module | What It Does |
|--------|-------------|
| **code_index.py** | Indexes repository symbols for fast lookups |
| **symbol_search.py** | Finds callers and definitions of modified functions |
| **retrieval.py** | Assembles deep context for AI prompts: function definitions, callers of modified code, related test files, repository file tree. Caches file reads. Skips `__pycache__`, `node_modules`, `.git`. |

### 4. 📝 Review Output (`review/`)

| Module | What It Does |
|--------|-------------|
| **formatter.py** | Renders findings as rich GitHub Markdown: severity badges (🔴🔶⚠️💡ℹ️), OWASP references, CWE links, `suggestion` code blocks, engine attribution. Also generates GitHub Check Run annotations. |
| **deduplicator.py** | Fingerprint-based deduplication across engines and commits. Filters against existing PR comments. Consolidates multiple findings on the same line into a single comment. |
| **commenter.py** | Posts inline review comments (top 30 by severity), PR summary comments, and GitHub Check Runs with annotations. Uses `REQUEST_CHANGES` for critical findings, `COMMENT` otherwise. |
| **summary.py** | Generates the top-level Revelyt PR Summary: risk score, finding counts, key issues list, and actionable recommendations (secrets, missing tests, dependencies). |

### 5. 🛡️ Filters & Limits (`filters/`)

| Module | What It Does |
|--------|-------------|
| **file_filter.py** | 3-tier classification: `DEEP` (source code, 30+ extensions), `LIGHT` (config/docs), `SKIP` (images/binaries/fonts, 25+ extensions). Always-deep detection for Dockerfiles, CI configs, `.env` files. |
| **limits.py** | PR size protection. Auto-downgrades mode: DEEP→BALANCED (>100 files or >5K lines), BALANCED→FAST (>200 files or >10K lines). Hard limit at 500 files / 25K lines. AI chunk budgeting per mode. |

### 6. 💾 Caching (`cache/`)

| Module | What It Does |
|--------|-------------|
| **diff_cache.py** | Content-addressed (SHA-256) caching. Skips re-analysis of identical diffs across force-pushes. 24-hour TTL. JSON-backed disk persistence. Repo-level invalidation. |
| **ai_result_cache.py** | Caches LLM results keyed on code + prompt version + context hash. 12-hour TTL. Tracks hit rates and estimated token savings for observability. |

### 7. 📊 Learning Engine (`learning/`)

| Module | What It Does |
|--------|-------------|
| **module_risk_db.py** | JSON-backed historical bug database. Tracks issue counts per file across PRs. Feeds `historical_bug_weight` into impact analysis, causing chronically bug-prone files to get higher risk scores. Keeps last 50 entries per file. |

### 8. 🧩 Plugin System (`config/`)

```
config/
├── rulesets/         # Community & company rule packs (security, performance, etc.)
├── plugins/
│   └── third_party/  # Drop-in custom analyzers (auto-discovered)
└── prompts/          # Versioned LLM prompts per language/severity
```

---

## 🧪 Testing & Verification

### Run Syntax Check (All Files)

```bash
cd /Users/bootlabs/Desktop/Revelyt/Revelyt
python3 -c "
import py_compile, os
base = 'revelyt/plugins/pr_reviewer'
ok, fail = 0, 0
for root, _, files in os.walk(base):
    for f in files:
        if f.endswith('.py'):
            try:
                py_compile.compile(os.path.join(root, f), doraise=True)
                ok += 1
            except: fail += 1
print(f'✅ {ok} passed | ❌ {fail} failed')
"
```

### Run Unit Tests

```bash
# Test the diff parser
python3 -c "
from revelyt.plugins.pr_reviewer.diff.parser import DiffParser

diff = '''diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -10,3 +10,5 @@ def hello():
     return 'world'
+def insecure():
+    eval(input())
'''

parser = DiffParser()
result = parser.parse(diff)
for ext, chunks in result.items():
    for c in chunks:
        print(f'File: {c.file_path} | Lang: {c.language} | +{len(c.added_lines)} -{len(c.deleted_lines)}')
print('✅ Parser works!')
"

# Test the rule engine
python3 -c "
from revelyt.plugins.pr_reviewer.engines.rule_engine import RuleEngine
from revelyt.plugins.pr_reviewer.engines.base import FileChunk, PRContext

chunk = FileChunk(
    file_path='app.py', extension='.py', language='python',
    added_lines=['eval(user_input)', 'password = \"secret123\"', 'import *'],
    diff_content='eval(user_input)',
)
engine = RuleEngine()
findings = engine.run([chunk], PRContext())
for f in findings:
    print(f'[{f.severity.value.upper()}] {f.message}')
print(f'✅ Rule engine found {len(findings)} issues!')
"

# Test the sensitive data scanner
python3 -c "
from revelyt.plugins.pr_reviewer.engines.sensitive_data import SensitiveDataEngine
from revelyt.plugins.pr_reviewer.engines.base import FileChunk, PRContext

chunk = FileChunk(
    file_path='config.py', extension='.py', language='python',
    added_lines=[
        'AWS_KEY = \"AKIAIOSFODNN7EXAMPLE\"',
        'token = \"ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\"',
        'db_url = \"postgresql://admin:pass@localhost/db\"',
    ],
    diff_content='secrets',
)
engine = SensitiveDataEngine()
findings = engine.run([chunk], PRContext())
for f in findings:
    print(f'[{f.severity.value.upper()}] {f.message}')
print(f'✅ Secret scanner found {len(findings)} secrets!')
"

# Test the semantic diff analyzer
python3 -c "
from revelyt.plugins.pr_reviewer.diff.semantic_diff import SemanticDiffAnalyzer
from revelyt.plugins.pr_reviewer.engines.base import FileChunk

chunk = FileChunk(
    file_path='service.py', extension='.py', language='python',
    added_lines=[
        'def process_payment(amount, currency):',
        '    if amount > 1000:',
        '        if currency == \"USD\":',
        '            for i in range(retries):',
        '                try:',
        '                    return charge(amount)',
        '                except Exception:',
        '                    pass',
    ],
    diff_content='complex code',
)
analyzer = SemanticDiffAnalyzer()
analyzer.analyze([chunk])
print(f'Functions: {chunk.affected_functions}')
print(f'Complexity: {chunk.complexity}')
print(f'Risk: {chunk.risk_score}')
print('✅ Semantic analysis works!')
"

# Test the impact analyzer
python3 -c "
from revelyt.plugins.pr_reviewer.diff.impact_analyzer import ImpactAnalyzer
from revelyt.plugins.pr_reviewer.engines.base import FileChunk

chunks = [
    FileChunk(file_path='auth/login.py', extension='.py', added_lines=['x']*100, diff_content='x', complexity=12),
    FileChunk(file_path='docs/readme.md', extension='.md', added_lines=['x']*5, diff_content='x'),
    FileChunk(file_path='requirements.txt', extension='.txt', added_lines=['flask==2.0'], diff_content='x'),
]
analyzer = ImpactAnalyzer()
scored = analyzer.analyze(chunks)
for c in analyzer.sort_by_risk(scored):
    print(f'{c.file_path}: risk={c.risk_score:.1f}')
print('✅ Impact analysis works!')
"

# Test the file filter
python3 -c "
from revelyt.plugins.pr_reviewer.filters.file_filter import FileFilter

filt = FileFilter()
files = ['app.py', 'logo.png', 'styles.css', 'Dockerfile', 'notes.md', 'data.bin']
for f in files:
    print(f'{f}: {filt.classify(f).value}')
print('✅ File filter works!')
"

# Test the risk engine
python3 -c "
from revelyt.plugins.pr_reviewer.engines.risk_engine import RiskEngine
from revelyt.plugins.pr_reviewer.engines.base import FileChunk, Finding, Severity, FindingType

chunks = [
    FileChunk(file_path='auth/login.py', extension='.py', added_lines=['x']*200, diff_content='x', complexity=15, risk_score=5.0, test_coverage_status='missing'),
]
findings = [
    Finding(file='auth/login.py', line=1, severity=Severity.CRITICAL, type=FindingType.SECURITY, message='SQL injection', confidence=0.95, rule_id='sql-injection', engine='rules'),
]
engine = RiskEngine()
report = engine.calculate(chunks, findings)
print(report.to_markdown())
print('✅ Risk engine works!')
"
```

### Full Integration Test (requires GITHUB_TOKEN)

```bash
export GITHUB_TOKEN="ghp_your_token"
export GOOGLE_API_KEY="your_gemini_key"

# Full review with AI
revelyt pr check owner/repo 42 --mode deep

# Deterministic only (no API key needed for AI)
revelyt pr check owner/repo 42 --no-ai

# Dry-run (see output without posting)
revelyt pr check owner/repo 42 --dry-run --mode balanced

# Local diff review
git diff HEAD~1 | revelyt pr diff
```

---

## 📁 Directory Structure

```
revelyt/plugins/pr_reviewer/
├── __init__.py              # Plugin package init
├── orchestrator.py          # Main entry point (CLI/CI → pipeline → GitHub)
├── github_client.py         # PyGithub wrapper (diff, comments, Check Runs)
│
├── pipeline/
│   └── review_pipeline.py   # 9-stage review pipeline orchestrator
│
├── diff/
│   ├── parser.py            # Unified diff → FileChunk parser
│   ├── semantic_diff.py     # AST complexity + semantic change detection
│   ├── chunker.py           # Boundary-aware diff chunking for LLM
│   ├── impact_analyzer.py   # Multi-factor risk scoring
│   ├── change_graph.py      # Cross-module dependency graph
│   └── test_impact.py       # Missing test detection
│
├── engines/
│   ├── base.py              # Engine interface + Finding/FileChunk/PRContext models
│   ├── rule_engine.py       # 15+ deterministic regex + AST rules
│   ├── sensitive_data.py    # 30+ secret/PII detection patterns
│   ├── dependency_scanner.py# Vulnerability DB scanner for lockfiles
│   ├── ai_request.py        # Multi-provider LLM client with async batching
│   ├── ai_validation.py     # Two-pass self-consistency + confidence scoring
│   ├── patch_validator.py   # Deterministic AI-fix validation (AST check)
│   ├── risk_engine.py       # PR Risk Score calculator (0-10)
│   ├── hybrid_engine.py     # Rules → AI pipeline coordinator
│   └── ast_analyzers/       # Language-specific AST parsers
│
├── context/
│   ├── code_index.py        # Repository symbol indexer
│   ├── symbol_search.py     # Caller/definition search
│   └── retrieval.py         # RAG context assembler for AI prompts
│
├── review/
│   ├── formatter.py         # OWASP-aligned markdown formatter
│   ├── deduplicator.py      # Fingerprint dedup + same-line consolidation
│   ├── commenter.py         # GitHub comment/Check Run poster
│   └── summary.py           # PR summary generator
│
├── filters/
│   ├── file_filter.py       # 3-tier file classification (deep/light/skip)
│   └── limits.py            # PR size limits + auto mode downgrading
│
├── cache/
│   ├── diff_cache.py        # Content-addressed diff cache (SHA-256 + TTL)
│   └── ai_result_cache.py   # LLM result cache with token savings tracking
│
├── learning/
│   └── module_risk_db.py    # Historical bug tracking per module
│
└── config/
    ├── rulesets/             # Community & company rule packs
    ├── plugins/third_party/  # Drop-in custom analyzers
    └── prompts/              # Versioned LLM prompts
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GITHUB_TOKEN` | ✅ Yes | GitHub personal access token or GitHub App token |
| `GOOGLE_API_KEY` | For Gemini | Gemini API key (default AI provider) |
| `OPENAI_API_KEY` | For OpenAI | OpenAI API key (if using `--provider openai`) |
| `ANTHROPIC_API_KEY` | For Claude | Anthropic API key (if using `--provider anthropic`) |

---

## 📊 How the Risk Score Works

| Factor | Max Weight | Example |
|--------|-----------|---------|
| Change Size | +3.5 | >500 lines + >30 files |
| Sensitive Files | +3.0 | auth/, payment/, security/ paths |
| Code Complexity | +3.0 | Cyclomatic complexity >20 |
| Security Findings | +3.0 | Critical/High severity issues |
| Missing Tests | +2.0 | Source changed without test update |
| Dependency Changes | +2.0 | requirements.txt / package.json modified |

**Levels:** 🟢 Low (0-3) → 🟡 Medium (3-5) → 🟠 High (5-7) → 🔴 Critical (7-10)

---

## 🆚 Review Modes

| Mode | AI | Depth | Best For |
|------|-----|-------|----------|
| `fast` | Limited (20 chunks max) | Regex rules + secrets | Massive PRs (>200 files) |
| `balanced` | Standard (80 chunks max) | Rules + AI on risky code | Regular PRs |
| `deep` | Full (150 chunks max) | Rules + AI + change graph + RAG context | Critical PRs, security reviews |
| `--no-ai` | Disabled | Pure deterministic rules | CI/CD pipelines (zero cost) |

---

## 🔒 What It Detects (Summary)

### Security (Deterministic — Zero Hallucination)
- SQL injection (f-strings, concatenation)
- `eval()` / `exec()` usage
- `subprocess` with `shell=True`
- `pickle.load()` on untrusted data
- `yaml.load()` without safe loader
- `innerHTML` XSS
- Hardcoded secrets (AWS, GCP, GitHub, Slack, Stripe, etc.)
- Private keys / JWT secrets
- PII (emails, SSNs, credit cards, phone numbers)
- Vulnerable dependencies (CVE database)
- Unpinned dependency versions

### Code Quality (Deterministic)
- Bare `except:` clauses
- `assert` in production code
- `global` statements
- Wildcard imports (`from x import *`)
- Debug breakpoints (`pdb`, `debugger`)
- TODO/FIXME markers
- print() statements (should use logging)
- Hardcoded IP addresses

### AI-Powered (Validated)
- Complex logic bugs
- Performance anti-patterns
- Architecture violations
- Missing error handling
- Race conditions
- All AI findings validated with two-pass self-consistency

---

## 📜 License

Part of the Revelyt project. See root LICENSE file.
