---
name: test-quality
description: Tests that prove behavior, not implementation. Edge cases, failure paths, independence.
compact_rules: |
  NO: testing implementation (mocking the code under test), always-true assertions, shared state between tests, only happy-path tests, broad except in assertions
  YES: test behavior (inputs→outputs), one assertion per test, happy path + primary failure + boundary case, descriptive test names that explain failure
  FLAG: any test with assert x is not None only, any pytest.raises(Exception) without specific type
---

# Test Quality

Every test must be able to fail. If you can't imagine a code change that would
make the test fail, the test is not testing anything.

## Key Rules

1. **Test behavior, not implementation** — don't mock the thing being tested
2. **One assertion per test** — focused, clear failure signal
3. **Test failure cases** — happy path + primary failure + boundary
4. **Independent tests** — no shared state, use fixtures
5. **Naming** — test name completes: "This test fails when..."
