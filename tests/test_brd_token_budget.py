"""
Test that BRD verification prompts fit within the 8192-token per-slot context limit.
"""
import json
import re
import tiktoken

# Simulate the exact BRD verification flow
enc = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 8192
# Leave headroom for LLM response generation tokens
PROMPT_BUDGET = MAX_TOKENS - 500  # 7692 tokens max for the prompt


def _make_summaries(n: int) -> list:
    """Generate realistic-looking module summaries."""
    summaries = []
    dirs = [
        "backend/src/controllers/appControllers/invoiceController",
        "backend/src/controllers/appControllers/paymentController",
        "backend/src/controllers/middlewaresControllers/createAuthMiddleware",
        "backend/src/controllers/middlewaresControllers/createCRUDController",
        "backend/src/controllers/coreControllers/settingController",
        "backend/src/models/appModels",
        "backend/src/routes/coreRoutes",
        "backend/src/middlewares/settings",
        "backend/src/middlewares/uploadMiddleware",
    ]
    funcs_pool = [
        "create", "read", "update", "remove", "paginatedList", "sendMail",
        "login", "logout", "verifyToken", "resetPassword", "forgetPassword",
        "listAll", "filter", "search", "summary", "index", "schemaValidate",
        "generateUniqueNumber", "downloadPdf", "catchErrors",
    ]
    for i in range(n):
        d = dirs[i % len(dirs)]
        fname = funcs_pool[i % len(funcs_pool)]
        summaries.append({
            "file_path": f"{d}/{fname}.js",
            "purpose": f"Handles {fname} operations for the application module {i}",
            "key_functions": [fname, f"helper_{fname}", f"validate_{fname}"],
        })
    # Add some auto-detected entries
    for infra in ["jest [auto-detected]", "prisma [auto-detected]"]:
        summaries.append({
            "file_path": infra,
            "purpose": "Infrastructure probe",
            "key_functions": ["test_suite", "migration"],
        })
    return summaries


def _filter_summaries_for_requirement(req: dict, summaries: list) -> list:
    """Mirrors the actual _filter_summaries_for_requirement method."""
    req_text = req.get("requirement", "").lower()
    keywords = set(re.findall(r'[a-z]{4,}', req_text))

    scored = []
    for s in summaries:
        if "[auto-detected]" in s.get("file_path", ""):
            scored.append((100, s))
            continue
        s_text = (
            s.get("file_path", "") + " " +
            s.get("purpose", "") + " " +
            " ".join(s.get("key_functions", []))
        ).lower()
        hits = sum(1 for kw in keywords if kw in s_text)
        if hits > 0:
            scored.append((hits, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:200]]


def _build_prompt_tokens(req: dict, summaries: list) -> int:
    """Build the exact prompt that would be sent and count tokens."""
    # System prompt (matches _verify_batch_llm)
    system = (
        "You are verifying whether business requirements are implemented in a codebase.\n"
        "You are given a list of files and their key functions.\n"
        "If a function/endpoint name matches or closely relates to the requirement, status=MET.\n"
        "If partial evidence exists, status=PARTIAL.\n"
        "If no evidence at all, status=NOT_MET.\n"
        "Return ONLY valid JSON matching the VerificationResultList schema."
    )

    # Compact summaries (matches _verify_batch_llm)
    compact_lines = []
    for s in summaries:
        fp = s.get("file_path", "")
        kf = s.get("key_functions", [])
        kf_str = ", ".join(kf[:5]) if kf else s.get("purpose", "")[:60]
        compact_lines.append(f"{fp}: {kf_str}")
    compact_summaries = "\n".join(compact_lines)

    # User prompt
    user = (
        f"Requirement:\n{json.dumps([req])}\n\n"
        f"Codebase files and functions:\n{compact_summaries}"
    )

    # Format instructions (compact schema)
    fmt = '{"results": [{"requirement_id": int, "status": "MET|NOT_MET|PARTIAL", "justification": "1 sentence", "evidence_files": ["path"]}]}'

    total = system + "\n" + fmt + "\n" + user
    return len(enc.encode(total))


def test_single_requirement_fits_in_context():
    """Single requirement with 532 summaries (after filtering) must fit in 8K context."""
    summaries = _make_summaries(532)

    test_requirements = [
        {"id": 1, "requirement": "The system shall authenticate administrators via email address and password using bcrypt hashing", "category": "Auth"},
        {"id": 2, "requirement": "Upon successful login the system shall issue a JSON Web Token JWT with configurable expiry", "category": "Auth"},
        {"id": 3, "requirement": "The system shall support invoice creation with line items tax calculations and currency formatting", "category": "Invoicing"},
        {"id": 4, "requirement": "File uploads shall be validated for type and size before storage", "category": "Upload"},
        {"id": 5, "requirement": "The system shall send email notifications for password reset requests", "category": "Email"},
    ]

    print("\n=== BRD Token Budget Test ===")
    print(f"Total summaries: {len(summaries)}")
    print(f"Budget: {PROMPT_BUDGET} tokens (8192 - 500 headroom)\n")

    all_passed = True
    for req in test_requirements:
        filtered = _filter_summaries_for_requirement(req, summaries)
        tokens = _build_prompt_tokens(req, filtered)
        status = "✅ PASS" if tokens <= PROMPT_BUDGET else "❌ FAIL"
        if tokens > PROMPT_BUDGET:
            all_passed = False
        print(f"  {status} Req #{req['id']}: {tokens:,} tokens ({len(filtered)} summaries) — {req['category']}")

    print(f"\n{'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    assert all_passed, "Some prompts exceed the 8192-token context budget!"


def test_worst_case_200_summaries():
    """Even with max 200 summaries (cap), prompt must fit."""
    summaries = _make_summaries(200)
    req = {"id": 1, "requirement": "The system shall create update remove read list filter search paginate validate login logout reset verify authenticate", "category": "Everything"}

    tokens = _build_prompt_tokens(req, summaries)
    print(f"\n=== Worst Case (200 summaries, broad requirement) ===")
    print(f"  Tokens: {tokens:,} / {PROMPT_BUDGET}")
    assert tokens <= PROMPT_BUDGET, f"200 summaries worst case exceeds budget: {tokens} > {PROMPT_BUDGET}"
    print("  ✅ PASS")


def test_filter_reduces_count():
    """Per-requirement filtering should significantly reduce summary count."""
    summaries = _make_summaries(532)
    req = {"id": 1, "requirement": "JWT authentication login", "category": "Auth"}
    filtered = _filter_summaries_for_requirement(req, summaries)
    print(f"\n=== Filter Reduction Test ===")
    print(f"  Input: {len(summaries)} → Filtered: {len(filtered)} (for 'JWT authentication login')")
    assert len(filtered) < len(summaries), "Filter should reduce summary count"
    assert len(filtered) <= 200, "Filter should cap at 200"
    print("  ✅ PASS")


if __name__ == "__main__":
    test_single_requirement_fits_in_context()
    test_worst_case_200_summaries()
    test_filter_reduces_count()
    print("\n🎉 All BRD token budget tests passed!")
