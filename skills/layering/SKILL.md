---
name: layering
description: Separation of concerns — routing, service, repository, model layers must not bleed.
compact_rules: |
  YES: router→service→repository→DB flow, services return domain objects or None, repositories own all queries, transaction boundaries in service/repository layer
  SAFE (do NOT flag): crud.py / service files for "business logic in route handlers" — crud.py IS the service layer, not a route handler file
  SAFE (do NOT flag): models.py and schemas.py for "business logic in route handlers" — these files define data structures, not routes
  SAFE (do NOT flag): service/CRUD function returning None for not-found + route raising HTTPException(404) — correct layering, not a violation
  SAFE (do NOT flag): AuditMixin / SoftDeleteMixin / TimestampMixin applied to models — mixins ARE the DRY solution; do not flag as a DRY violation
  SAFE (do NOT flag): models.py importing from database.py (Base, engine) — correct dependency direction---

# Layering Discipline

Enforces strict separation between code layers.

## Key Rules

1. **Router** — parse request, auth check, call service, format response
2. **Service** — business rules, orchestrate repos, transaction boundaries
3. **Repository** — all DB queries, ORM calls, cache reads
4. **Model** — field definitions, computed properties, validation
5. **Background job test** — if logic can't be called from a background job, it's in the wrong layer

## None-Sentinel Is Correct Layering

service returning None + route raising HTTPException is the textbook correct
layering pattern for FastAPI/Flask. The service doesn't know about HTTP codes.
The route doesn't know about DB internals. Never flag either side of this pattern.

## Mixin Pattern Is the DRY Solution

AuditMixin, SoftDeleteMixin, TimestampMixin applied to models IS the
correct DRY solution. Do not flag classes that correctly use mixins as
having a DRY violation — they are already following DRY.