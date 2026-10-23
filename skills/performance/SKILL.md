---
name: performance
description: N+1 queries, missing indexes, unbounded results, work inside loops.
compact_rules: |
  YES: joinedload/select_related for ORM relationships accessed in loops, .count()/.first() for existence checks, pagination on list route handlers, index on FK columns
  SAFE (do NOT flag): database.py, models.py, schemas.py for "missing pagination" or "missing connection pool" — these are not endpoint files
  SAFE (do NOT flag): get_db() / session factory functions — not list endpoints, no pagination needed
  SAFE (do NOT flag): ORM relationship definitions in model classes — N+1 risk is at query time in service code, not at model definition---

# Performance Awareness

Catches the most common performance problems at the point of review.

## Key Rules

1. **N+1 queries** — any loop accessing ORM relationships needs eager loading
2. **Load only what's needed** — `.count()` not `len(.all())`, select specific columns
3. **Missing indexes** — every WHERE/ORDER BY/JOIN column needs an index
4. **No work in loops** — regex compile, API calls, DB writes belong outside loops
5. **Pagination required** — every list route handler must have skip/limit
6. **The 100× question** — what happens with 100× more records?

## Scope Note

Pagination rules apply to HTTP route handler functions only — functions decorated
with @app.get, @app.post, @router.get etc. that return List[Model].
database.py contains session setup (not endpoints), models.py contains schema
definitions (not endpoints), schemas.py contains Pydantic models (not endpoints).
Never flag these files for missing pagination.

N+1 rules apply at the point where ORM relationships are accessed inside a loop
in service/CRUD code — not at the model definition where relationships are declared.