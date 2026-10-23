---
name: dead-code
description: Unused vars, imports, functions, commented-out code, unreachable branches.
compact_rules: |
  NO: commented-out code blocks, unreachable branches after return, TODO stubs with pass body, dead else branches
  YES: delete unused code (trust git), grep before deleting functions, clean the immediate area when modifying
  FLAG: commented-out code blocks (lines starting with # that contain executable code)
  FLAG: variables assigned but never read within the same scope
  FLAG: functions defined but provably never called within the visible file
  FLAG: imports where you have confirmed the imported name does not appear anywhere in the file body below the import line — scan the ENTIRE visible file before flagging; if uncertain, do NOT flag
  SAFE (do NOT flag): any import where the imported name appears anywhere in the file, even once — load_dotenv(), Base, SessionLocal etc. used later in the file are NOT unused
---

# Dead Code Rule

Code that doesn't run gets deleted, not commented out.

## Key Rules

1. **Commented-out code** — delete it, that's what git is for
2. **Unused variables/imports** — delete immediately, but VERIFY first
3. **Unreachable code** — code after return, impossible conditions
4. **Unused functions** — grep first, delete if only definition found
5. **When modifying** — clean dead code in the function you touched

## Import Verification Note

Before flagging an import as unused, scan the entire visible file content
for any occurrence of the imported name. A single use anywhere in the file
means the import is NOT unused. If you cannot see the full file (truncated),
do not flag imports as unused — you cannot make this claim without complete
visibility.