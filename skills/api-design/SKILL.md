---
name: api-design
description: API ergonomics, REST conventions, consistency, error handling patterns.
compact_rules: |
  YES: validate required parameters at API boundary, use correct HTTP status codes, paginate large collections, document side effects
  FLAG: missing error responses (500 without description), inconsistent status codes across similar endpoints, unbounded list endpoints
  FLAG: missing docstrings on public functions, undocumented required parameters, inconsistent error shapes
  FLAG: synchronous I/O in request handlers without timeout, blocking calls in async context
  SAFE (do NOT flag): internal helper functions not exposed as API, private methods
---

# API Design Rules

## Key Principles

1. **Consistency** — Similar operations should have similar signatures across the API
2. **Clarity** — Purpose, parameters, and side effects should be self-documenting
3. **Robustness** — Validate inputs, handle errors gracefully, set timeouts

## REST Conventions

| Pattern | Correct | Incorrect |
|---------|---------|-----------|
| Create resource | `POST /users` → 201 | `POST /users` → 200 |
| Delete missing | `DELETE /users/123` → 404 | `DELETE /users/123` → 200 |
| List resources | `GET /users?page=1&limit=20` | `GET /users` (unbounded) |

## Error Responses

All errors should include:
- HTTP status code (4xx for client errors, 5xx for server errors)
- Machine-readable error code
- Human-readable message
- (Optional) Resolution hint

```python
# Good
return JSONResponse(
    status_code=400,
    content={"error": "VALIDATION_ERROR", "message": "email is required", "field": "email"}
)

# Bad
return {"error": "bad request"}  # wrong status code
```
