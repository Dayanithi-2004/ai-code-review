---
name: code-quality
description: Error handling, naming, magic numbers, SOLID, mutation discipline, dependency hygiene.
compact_rules: |
  YES: named constants for unexplained numeric literals in logic, context managers for resources, type hints on public APIs, descriptive naming (is_/has_/can_ for bools)
  SAFE (do NOT flag): service/CRUD function that returns None for not-found — route layer raises HTTPException(404); NEVER flag this as "missing error handling", "swallowed error", or any variant regardless of wording
  SAFE (do NOT flag): def f(x=None) / def f(n=0) / def f(s="") — None, int, str, bool defaults are immutable and safe; only def f(x=[]) / def f(cfg={}) / def f(tags=set()) are bugs
  SAFE (do NOT flag): function signature defaults like def get_books(skip=0, limit=100) — these are API parameters, not magic numbers
  SAFE (do NOT flag): try/finally for resource cleanup (db.close(), file.close())
  SAFE (do NOT flag): SQLAlchemy Column(default=1) or Column(default=False) — DB-level schema defaults, not Python function defaults, not magic numbers
  SAFE (do NOT flag): boolean literals True/False as values — not magic numbers
  SAFE (do NOT flag): Pydantic BaseModel fields with List[X] = [] — Pydantic creates a new list per instance; mutable default bug does not apply
  SAFE (do NOT flag): audit/event log dicts with different keys per action — {"new":...} for create, {"before":...,"after":...} for update is intentional design---

# Code Quality Discipline

Merged: `coding-discipline` + `error-handling`

Covers SOLID principles, clean code practices, error handling contracts,
naming conventions, and dependency hygiene.

## Key Rules

1. **Error handling is a contract** — pick return-type OR exception, never mix in the same layer
2. **Never swallow errors** — no bare `except: pass`, no log-and-swallow
3. **No magic numbers** — named constants for unexplained numeric literals in logic only
4. **50-line check** — extract if function exceeds 50 lines
5. **Never silently mutate inputs** — return modified copy, don't alter caller's data
6. **Naming is load-bearing** — functions: what they DO; booleans: `is_/has_/can_`
7. **Dependency hygiene** — check stdlib first, verify package is maintained

## None-Sentinel Pattern

service/CRUD returning None + route raising HTTPException(404) is a standard
deliberate FastAPI/Flask layering pattern. The service layer is not supposed to
know about HTTP. NEVER flag this in any form — not as "missing error handling",
"swallowed exception", "inconsistent error handling", or any variant.

## Mutable Default Note

Only `[]`, `{}`, and `set()` as function defaults are bugs.
`None`, `0`, `""`, `False`, `True` are immutable and always safe.
SQLAlchemy Column defaults and Pydantic field defaults are framework-managed.
Function parameter defaults (skip=0, limit=100) are API parameters, not magic numbers.