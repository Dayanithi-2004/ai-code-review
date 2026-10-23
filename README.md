<div align="center">

# 🛡️ Revelyt

**Senior-Level AI Code Reviewer — Powered by an Engineering Team of LLM Agents**

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

Revelyt scans any Python or Node.js codebase using **five AI agents** — each representing a role on a real engineering team — and produces a detailed PDF report with severity-ranked findings, BRD compliance verification, and per-file architectural summaries.

## ✨ Features

- **🔍 Wave 1 Reviewer** — security architecture, OWASP Top 10, SOLID principles, anti-patterns, naming, error handling, clean code
- **🔬 Wave 2 Deep Reviewer** — test coverage gaps, regression risk, API ergonomics, usability, documentation quality
- **📋 Product Manager** — module summaries, holistic BRD verification
- **⚙️ Static Gate** — zero-token Ruff + Bandit pre-scan before LLM agents fire
- **🧠 Semantic Cache** — skip re-analysis of unchanged code across runs at zero token cost

### Two-Wave Execution Model

```
Static Gate (Ruff + Bandit, 0 tokens)
              ↓
Wave 1 (independent):   CombinedWave1Agent (Architecture + Security + Dev Quality)
                                    ↓
Wave 2 + PM (concurrent): CombinedWave2Agent (QA + DevExp + Docs)
                           ProductManagerAgent (Summaries + BRD verification)
                                    ↓
Post-analysis:          Severity Elevation + Corroboration Engine
```

Wave 2 agents receive Wave 1 findings as context — they **corroborate** rather than duplicate, providing multi-perspective confirmation of critical issues.

### Additional Capabilities

- 🔀 **Git-aware incremental scanning** — only analyze changed + impacted files with `--diff`
- 📄 **BRD verification** — parse business requirements and verify each against the codebase
- 📊 **PDF/HTML report generation** — severity-ranked findings with cost tracking
- 🔁 **Corroboration Engine** — findings confirmed by multiple agents are elevated to CRITICAL
- 🔌 **Multi-provider support** — Google Gemini, OpenAI, Anthropic, Ollama, HuggingFace, OpenRouter

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/your-org/Revelyt.git
cd Revelyt
pip install -e .
```

### 2. Set your API key

```bash
# Pick one — depending on which provider you want to use
export GOOGLE_API_KEY="AIza..."
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Run a scan

```bash
# Scan a local project
revelyt scan ./my-project --provider gemini-2.5-pro

# Scan a GitHub repo
revelyt scan https://github.com/org/repo --provider gemini-2.5-pro

# Scan with BRD verification
revelyt scan ./my-project -b requirements.md --provider gemini-2.5-pro

# See all available commands
revelyt --help
```

---

## 📖 Commands

Running `revelyt --help` or `revelyt` with no arguments lists every available command with a short description. Use `revelyt <command> --help` for full option details on any individual command.

```
 Usage: revelyt [OPTIONS] COMMAND [ARGS]...

 Revelyt: Senior-Level AI Code Reviewer

╭─ Commands ──────────────────────────────────────────────────────────────────╮
│ scan       Run the full 6-Phase Revelyt pipeline on a repository          │
│ verify     Standalone BRD verification against an existing scan session     │
│ triage     Dry-run Phase 1 & 2 only — no LLM calls                         │
│ report     Regenerate a PDF report from an existing session database        │
│ providers  List all registered LLM providers and their CLI aliases          │
│ pr         Enterprise PR Reviewer Plugin (sub-commands: check, diff)        │
╰─────────────────────────────────────────────────────────────────────────────╯
```

---

### `scan` — Full 6-Phase Pipeline

Run the complete Revelyt analysis pipeline on a target repository.

```bash
revelyt scan [OPTIONS] TARGET_PATH
revelyt scan --help   # show full option list
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `TARGET_PATH` | | *(required)* | Local directory path or GitHub URL to scan |
| `--brd` | `-b` | | Path to a Markdown BRD file for Phase 4 verification |
| `--provider` | `-p` | `gemini` | LLM provider alias. Run `revelyt providers` to see all options |
| `--oss-provider` | | | OSS/local model for Wave 2 + PM (e.g. `qwen2.5-coder`). Defaults to `--provider` if not set |
| `--diff` | `-d` | `False` | Enable diff mode — only analyze changed + impacted files |
| `--base` | | `main` | Git ref to diff against (branch, tag, or commit). Used with `--diff` |
| `--head` | | `HEAD` | Git ref for current state. Used with `--diff` |
| `--agents` | `-a` | *(all)* | Comma-separated agents to run: `architect`, `developer`, `qa`, `devexp`, `doc` |
| `--format` | `-f` | `pdf` | Output report format: `pdf` or `html` |

**Examples:**

```bash
# Full scan of a local project
revelyt scan ./my-fastapi-app --provider gemini-2.5-pro

# Scan a GitHub repo with GPT-5 and BRD verification
revelyt scan https://github.com/org/repo -b requirements.md -p gpt-5

# Run only security and architecture agents
revelyt scan ./my-project --agents architect,developer --provider gemini-2.5-pro

# Incremental scan — only files changed since the develop branch
revelyt scan ./my-project --diff --base develop --provider gemini-2.5-pro

# Incremental scan — last 3 commits
revelyt scan ./my-project --diff --base HEAD~3 --provider gemini-2.5-pro

# Use a local OSS model for Wave 2 + PM to save costs
revelyt scan ./my-project --provider gemini-2.5-pro --oss-provider qwen2.5-coder

# Generate an HTML report instead of PDF
revelyt scan ./my-project --provider gemini-2.5-pro --format html
```

### The 6-Phase Pipeline

| Phase | Name | Description |
|-------|------|-------------|
| 1 | **Ingestion & Triage** | Clone/copy repo, detect stack (Python/Node.js) |
| 2 | **Dependency Mapping** | Build import graph, calculate blast radius per file |
| 3 | **AI Agent Review** | Static Gate → Wave 1 → Wave 2 + PM summarisation (concurrent) |
| 4 | **BRD Verification** | Parse BRD, verify each requirement against summaries |
| 5 | **Severity Elevation** | Elevate severity based on blast radius + corroboration |
| 6 | **Reporting** | Generate `revelyt_report.pdf` (or `.html`) with all findings |

---

### `verify` — Standalone BRD Verification

Re-run Phase 4 against an existing scan session without paying for a full re-scan. Useful after editing requirements without changing code.

```bash
revelyt verify [OPTIONS] BRD
revelyt verify --help
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `BRD` | | *(required)* | Path to the `.md` BRD file |
| `--provider` | `-p` | `gemini` | LLM provider alias |
| `--session` | `-s` | | Path to an existing session `.db` file. If omitted, starts a fresh in-memory session |

```bash
# Verify against a previous scan session
revelyt verify requirements.md --session ./previous_scan.db

# Verify with a different provider
revelyt verify requirements.md -p gpt-5 -s ./scan.db

# Standalone verification (no prior scan session needed)
revelyt verify requirements.md --provider gemini-2.5-pro
```

---

### `triage` — Dry Run (No LLM Calls)

Run Phase 1 & 2 only — see which files would be scanned and their dependency impact scores, without spending any tokens.

Files are ranked by `dependant_count` (blast radius). High-count files are scanned first and receive severity elevation during a real scan.

```bash
revelyt triage [OPTIONS] TARGET_PATH
revelyt triage --help
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `TARGET_PATH` | | *(required)* | Local directory path or GitHub URL |
| `--top` | `-n` | `20` | Number of highest-priority files to display |
| `--json` | | `False` | Output raw JSON instead of a table (useful for scripting) |

```bash
# See file blast radius scores before committing to a full scan
revelyt triage ./my-project

# Show only the top 10 riskiest files
revelyt triage ./my-project --top 10

# Machine-readable JSON output
revelyt triage ./my-project --json
```

---

### `report` — Regenerate Report

Regenerate a PDF report from an existing scan session database — no LLM calls needed.

```bash
revelyt report [OPTIONS] SESSION_DB
revelyt report --help
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `SESSION_DB` | | *(required)* | Path to an existing session `.db` file |
| `--output` | `-o` | `revelyt_report.pdf` | Output path for the regenerated report |

```bash
# Regenerate the PDF report
revelyt report ./my_scan_session.db

# Save to a custom path
revelyt report ./my_scan_session.db --output final_audit.pdf
```

---

### `providers` — List Available LLM Providers

List all registered providers and their CLI aliases. Reads directly from the provider registry — adding a new provider file automatically makes it appear here.

```bash
revelyt providers
```

No options required.

---

### `pr` — Enterprise PR Reviewer Plugin

Run an AI-powered code review on a GitHub Pull Request or local `git diff`. Combines deterministic rule engines (instant, zero-cost) with optional AI analysis on high-risk chunks only.

```bash
revelyt pr --help          # list pr sub-commands
revelyt pr check --help    # options for GitHub PR review
revelyt pr diff --help     # options for local diff review
```

#### `pr check` — Review a GitHub Pull Request

```bash
revelyt pr check [OPTIONS] REPO PR_NUMBER
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `REPO` | | *(required)* | GitHub repository, e.g. `owner/repo` |
| `PR_NUMBER` | | *(required)* | Pull Request number |
| `--mode` | | `deep` | Review depth: `fast`, `balanced`, or `deep` |
| `--no-ai` | | `False` | Disable AI review — deterministic rules only, zero cost |
| `--provider` | `-p` | `gemini` | LLM provider alias |
| `--dry-run` | | `False` | Preview results in CLI without posting comments to GitHub |
| `--token` | | `$GITHUB_TOKEN` | GitHub personal access token (or set `GITHUB_TOKEN` env var) |

```bash
# Full deep review with AI
revelyt pr check owner/repo 42 --mode deep

# Rules-only — no API key needed, instant, free
revelyt pr check owner/repo 42 --no-ai

# Preview results without posting to GitHub
revelyt pr check owner/repo 42 --dry-run --mode balanced

# Use a specific provider
revelyt pr check owner/repo 42 --provider claude-sonnet-4-6
```

#### `pr diff` — Review a Local Git Diff

```bash
revelyt pr diff [OPTIONS] [DIFF_FILE]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `DIFF_FILE` | | stdin | File containing the diff. Defaults to stdin if not provided |
| `--mode` | | `balanced` | Review depth: `fast`, `balanced`, or `deep` |
| `--no-ai` | | `False` | Disable AI review (rules only) |
| `--provider` | `-p` | `gemini` | LLM provider alias |

```bash
# Review uncommitted changes from stdin
git diff HEAD~1 | revelyt pr diff

# Deep analysis on last commit
git diff HEAD~1 | revelyt pr diff --mode deep

# Rules-only, no API cost
git diff HEAD~1 | revelyt pr diff --no-ai

# Review from a saved patch file
revelyt pr diff ./changes.patch --mode balanced
```

#### PR Review Modes

| Mode | AI Chunks | Best For |
|------|-----------|----------|
| `fast` | 20 max | Massive PRs (>200 files), quick pre-merge check |
| `balanced` | 80 max | Regular PRs, standard depth |
| `deep` | 150 max | Critical PRs, security reviews, architecture changes |
| `--no-ai` | 0 | CI/CD pipelines (zero cost, zero hallucination) |

---

## 🏗️ Architecture

```
revelyt/
├── agents/                       # AI review agents
│   ├── base_agent.py             # BaseAgent ABC — shared analyze/prompt/save logic
│   ├── combined_wave1_agent.py   # 🔍 Wave 1 — architecture, security, SOLID
│   ├── combined_wave2_agent.py   # 🔬 Wave 2 — QA, DevExp, docs, test gaps
│   └── product_manager_agent.py  # 📋 Summaries + BRD verification
├── core/
│   ├── orchestrator.py           # 6-phase pipeline with two-wave agent execution
│   ├── session.py                # SQLite session manager (thread-safe RLock)
│   ├── mapper.py                 # Dependency graph builder + blast radius
│   ├── schemas.py                # Pydantic schemas (FindingList, etc.)
│   ├── code_optimizer.py         # Token-efficient code prep + AST skeleton extraction
│   ├── diff.py                   # Git diff engine
│   ├── diff_pruner.py            # AST-aware diff pruner for incremental scans
│   ├── review_cache.py           # SHA-256 semantic cache (0 tokens on cache hit)
│   ├── sandbox.py                # Temp directory sandbox with guaranteed cleanup
│   ├── skills_engine.py          # Modular skill injection (wave-level, not per-agent)
│   ├── static_gate.py            # Ruff + Bandit pre-scan before LLM agents
│   ├── triage.py                 # Fast model triage for diff complexity routing
│   └── meta.py                   # Provider metaclass interface enforcement
├── ingestion/
│   ├── loader.py                 # Git clone / local copy + analysis/summary manifest split
│   └── detector.py               # Stack detection (Python/Node.js/TypeScript)
├── providers/
│   ├── base.py                   # BaseProvider ABC + 3-pass JSON repair loop
│   ├── google_provider.py        # Google Gemini (native structured output)
│   ├── openai_provider.py        # OpenAI GPT (native structured output)
│   ├── anthropic_provider.py     # Anthropic Claude (native structured output)
│   ├── ollama_provider.py        # Local Ollama models + /no_think for DeepSeek R1
│   ├── huggingface_provider.py   # HuggingFace Inference API
│   ├── openrouter_provider.py    # OpenRouter (unified access to many models)
│   └── factory.py                # Self-registering provider registry + factory
├── brd/
│   └── parser.py                 # BRD Markdown document parser
├── plugins/
│   └── pr_reviewer/              # Enterprise PR review plugin
│       ├── orchestrator.py       # PR review entry point (CLI → pipeline → GitHub)
│       ├── pipeline/             # 9-stage review pipeline
│       ├── diff/                 # Diff parsing, semantic analysis, impact scoring
│       ├── engines/              # Rule engine, secret scanner, AI engine, hybrid
│       ├── context/              # RAG context retrieval for AI prompts
│       ├── review/               # GitHub comment formatting and deduplication
│       ├── filters/              # File classification and PR size limits
│       ├── cache/                # Diff cache and AI result cache
│       └── learning/             # Historical bug tracking per module
└── cli.py                        # Typer CLI entry point
```

---

## 🧪 Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v
```

Current test coverage:
- `tests/test_agents.py` — 19 tests covering all agents (prompt construction, context injection, language detection, finding persistence, failure resilience)
- `tests/test_diff.py` — 8 tests covering Git diff engine and impact analysis

---

## ⚙️ Supported Providers

Run `revelyt providers` for the full up-to-date list with all aliases, environment variables, and structured-output support details.

| Provider | CLI Aliases | Env Var | Structured Output |
|----------|-------------|---------|-------------------|
| **Google Gemini** | `gemini-2.5-pro` `gemini-3-flash-preview` `gemini-3.1-pro-preview` `gemini-3.1-flash-lite-preview` | `GOOGLE_API_KEY` | ✅ Native |
| **OpenAI** | `gpt-5` `gpt-5-mini` `gpt-5-nano` `gpt-5.2` `gpt-5.2-pro` `gpt-4.1` | `OPENAI_API_KEY` | ✅ Native |
| **Anthropic** | `claude-sonnet-4-6` `claude-opus-4-6` `claude-haiku-4-5` | `ANTHROPIC_API_KEY` | ✅ Native |
| **Ollama** (local) | `llama3` `mistral` `qwen2.5-coder` `deepseek-r1` `codestral` `phi` `omnicoder` | *(none — local)* | ⚠️ Repair loop |
| **HuggingFace** | `qwen2.5` `qwen3` `deepseek` `starcoder2` `codellama` `stablecode` `kimi` | `HUGGINGFACEHUB_API_TOKEN` | ⚠️ Repair loop |
| **OpenRouter** | `openrouter` `deepseek-chat` `qwen3-coder` `nemotron` `gpt-oss` | `OPENROUTER_API_KEY` | ⚠️ Repair loop |

### Dual-Provider Mode (Cost Optimisation)

Use a frontier model for Wave 1 (security/architecture) and a free or local model for Wave 2 + PM summarisation:

```bash
# Frontier model for security review, local Qwen for docs/DX
revelyt scan ./my-project \
  --provider gemini-2.5-pro \
  --oss-provider qwen2.5-coder

# Cloud frontier + OpenRouter free model for Wave 2
revelyt scan ./my-project \
  --provider gpt-5 \
  --oss-provider qwen3-coder
```

---

## 🔍 What Revelyt Detects

### Security (Wave 1 + Static Gate)
- Hardcoded secrets, API keys, JWT secrets, database credentials
- SQL injection via f-strings or string concatenation on raw queries
- `eval()` / `exec()` with user-controlled input
- `subprocess` with `shell=True` and user-supplied data
- `yaml.load()` without `SafeLoader`, `pickle.load()` on untrusted data
- Missing authentication on data-modifying routes
- `jwt.decode()` used without `jwt.verify()` (signature not checked)
- Insecure hash functions (MD5/SHA1 for passwords)
- `dangerouslySetInnerHTML` with user-controlled values (XSS)

### Architecture (Wave 1)
- Business logic leaking into route/controller handlers
- N+1 queries — ORM calls inside loops without eager loading
- Synchronous blocking I/O inside `async def` functions
- Missing connection pool settings on production DB engines
- DRY violations — duplicated CRUD/service code blocks

### Code Quality (Wave 1)
- Mutable default arguments (`def f(items=[])`, `def f(cfg={})`)
- Bare `except:` clauses swallowing all exceptions silently
- Magic numbers without named constants (numeric literals only)
- Missing type hints on public API signatures
- Star imports polluting module namespace

### Developer Experience & Docs (Wave 2)
- Missing docstrings / JSDoc on public functions and exported modules
- Inconsistent HTTP status codes (200 returned for creates, should be 201)
- Unbounded list endpoints without skip/limit/cursor pagination
- Inconsistent error response shapes across endpoints
- TODO/FIXME/HACK markers indicating incomplete work

### Reliability / QA (Wave 2 — test files only)
- Test files with only happy-path coverage, no error-path tests
- `async` test functions missing `@pytest.mark.asyncio`
- Tests with real network/DB calls that are missing mocks
- Missing `@pytest.raises` for error-path scenarios

---

## 📊 Severity Calibration

| Level | Meaning | Examples |
|-------|---------|---------|
| **CRITICAL** | Directly exploitable, no special privileges needed | Hardcoded credential in source, `eval(user_input)`, private key committed |
| **HIGH** | Real vulnerability, requires attacker context | Raw SQL concat, missing auth on data route, `jwt.decode()` without verify |
| **MEDIUM** | Architectural flaw, not directly exploitable | N+1 query, `echo=True` hardcoded, blocking I/O in async handler |
| **LOW** | Style, documentation, minor convention | Missing docstrings, `console.log` in production, magic numbers |

Findings on files with `dependant_count ≥ 5` (high blast radius) are automatically elevated one severity tier. Findings confirmed independently by multiple agents are elevated to **CRITICAL** with a 🔁 Corroborated label.

---

## 🔀 Incremental Scanning

In `--diff` mode, Revelyt only analyzes files that changed plus any files that import them:

```bash
# Only scan what changed since the main branch
revelyt scan ./my-project --diff --base main --provider gemini-2.5-pro

# Only scan what changed since a specific commit
revelyt scan ./my-project --diff --base abc1234 --provider gemini-2.5-pro

# Preview what would be scanned before spending any tokens
revelyt triage ./my-project
```

The dependency mapper tracks which files import which, so if `auth.py` changed and `payments.py` imports it, both are included in the analysis scope even though only `auth.py` was directly modified.

---

## 📄 BRD Verification

Provide a Markdown requirements document to have Revelyt verify each requirement against the codebase:

```bash
revelyt scan ./my-project --brd requirements.md --provider gemini-2.5-pro
```

Each requirement is classified as `MET`, `PARTIAL`, or `NOT_MET` with evidence files. Requirements for testing frameworks and database migrations are verified via a zero-token filesystem probe — no LLM tokens are spent reading test or migration file content.

To re-run verification against an existing scan without re-scanning the whole codebase:

```bash
revelyt verify requirements.md --session ./previous_scan.db
```