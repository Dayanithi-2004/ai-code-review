---
name: security
description: Hardcoded secrets, injection flaws, missing auth, insecure crypto, env config.
compact_rules: |
  YES: env vars for all secrets, parameterized queries only, auth→authz→ownership check order, bcrypt/argon2 for passwords
  SAFE (do NOT flag as injection): ORM .filter()/.where()/.filter_by() calls — SQLAlchemy, Django ORM, Prisma, TypeORM parameterize these automatically
  SAFE (do NOT flag as injection): FastAPI route path parameters ({book_id: int}) — framework-validated before reaching any DB call; not SQL injection vectors
  SAFE (do NOT flag as injection): any file whose DB access goes via crud.* or service.* function calls — the route layer is not where injection is constructed
  SAFE (do NOT flag): Pydantic BaseModel parameters — these ARE validated input
  SAFE (do NOT flag): f-strings inside return statements, response dicts, log messages — these are not SQL or shell commands---

# Security Discipline

Merged: `security-discipline` + `env-config-discipline`

Covers OWASP Top 10 from a structural perspective, hardcoded credential detection,
injection prevention, authentication/authorization patterns, and environment
configuration hygiene.

## Key Rules

1. **Never trust input** — validate type, range, length, format at boundary
2. **No secrets in code** — environment variables only, fail loud at startup
3. **File paths from user input** — `realpath` + directory containment check
4. **Auth order** — authenticate → authorize → ownership check (return 404, not 403)
5. **Passwords** — bcrypt/argon2 only, never MD5/SHA256 without salt
6. **Never log sensitive data** — mask tokens, card numbers, passwords
7. **Config from one place** — central config module, never scattered `os.environ`
8. **Fail loud on missing config** — crash at startup, not silently at runtime

## ORM Safety Note

ORM query methods (.filter, .where, .filter_by, findMany, find) are parameterized
by the framework and are NOT SQL injection vectors. Only flag injection when
user-controlled data is concatenated directly into a raw SQL string.