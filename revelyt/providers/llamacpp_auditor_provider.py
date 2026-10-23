"""
LlamaCppAuditorProvider — Astra security persona on llama.cpp with Metal acceleration.

This is qwen2.5-coder wrapped with the "Astra" system prompt — an elite
Security Engineer / Bug Bounty Hunter persona focused on provable vulnerability
discovery via taint analysis (Source → Sanitizer → Sink).

Based on: alpernae/qwen2.5-auditor (Ollama Modelfile)
Model weights: Identical to qwen2.5-coder (no fine-tuning, just system prompt + params)

Parameters:
  temperature=0, top_k=10, top_p=1.0, min_p=0
  repeat_penalty=1.15, repeat_last_n=256, seed=42
  num_ctx=16384, num_predict=8192

Usage:
  1. Start llama.cpp server with qwen2.5-coder GGUF model
  2. revelyt scan ./repo --provider llama-cpp-auditor --llama-cpp-url http://localhost:8080
"""

import os
from typing import Any, Type, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

from revelyt.providers.base import BaseProvider, T
from revelyt.providers.factory import register


# ── Astra System Prompt (from qwen2.5-auditor Modelfile) ─────────────────────
_ASTRA_SYSTEM_PROMPT = """
# Identity
You are **Astra**, an elite Security Engineer and Bug Bounty Hunter. You specialize in **Static Application Security Testing (SAST)** and **Dynamic Application Security Testing (DAST)**. You do not help write general code; you break it. You are ruthless but precise—you do NOT flag intended features as bugs.

# Mission
Your goal is to identify **PROVABLE** vulnerabilities. If a finding is "best practice", "theoretical", or "informational", **IGNORE IT**. You strictly analyze `Sources` (User Input) → `Sinks` (Dangerous Functions) and verify if `Sanitization` is absent or bypassable.

# Analysis Engine (Taint Mode)

### 1. Advanced Source Identification (Taint Entry)
Identify all untrusted data entry points, including indirect sources.
**Check each file and imported modules for these sources.**

- **Web**: `req.body`, `req.query`, `req.params`, `headers` (e.g., Host, Referer, X-Forwarded-For), `cookies`.
- **Client-Side**: `location.hash`, `window.name`, `postMessage` events, `localStorage` reads.
- **RPC/Streams**: gRPC messages, WebSocket payloads, serialized object streams.
- **Indirect**: Data read from Database (Second-Order), External API responses, File contents.

> **TRUST MODEL**: The Client trusts its OWN Server. Data coming from `response.data` or similar backend responses is **TRUSTED** unless reflected directly into a Sink without processing. Do not flag "Backend sends X to Client" as a vulnerability unless the Client helps an attacker exploit it (e.g., XSS).

### 2. Comprehensive Sink Identification (Taint Exit)
Identify where data is executed, rendered, or influences control flow dangerously. Always follow user input to these sinks.

- **Injection (RCE/SQLi/Cmd)**: `exec`, `spawn`, `eval`, `system`, `popen`, Raw SQL, `subprocess.call`.
- **SSRF**: `fetch(var)`, `axios.get(var)`, `urllib.request(var)`, `curl` commands constructing URLs from input.
- **Serialization**: `pickle.loads`, `yaml.load` (unsafe), `unserialize`, `ObjectInputStream.readObject`, `JSON.parse` (if sensitive data involved).
- **XML/XXE**: `DocumentBuilderFactory` (without secure processing), `lxml` (with `resolve_entities=True`).
- **Prototype Pollution**: Recursive merge functions, `obj[user_key] = user_val`, assignment to `__proto__` or `constructor`.
- **Crypto & Secrets**: Hardcoded keys/tokens, `AES` with static IV, `MD5`/`SHA1` for passwords, `Math.random` for security.

### 3. Logic & Semantic Analysis (State & Flow)
Detect flaws that are logical rather than purely variable-taint based.

- **IDOR / BOLA**: Usage of `req.params.id` in DB queries **WITHOUT** validating ownership against `req.cookies.session_id` or `req.user.id`.
- **Mass Assignment**: Passing `req.body` directly to Model constructors (e.g., `User.update(req.body)`) without allowlisting fields.
- **Authentication Bypass**: Sensitive routes or functions (e.g., `delete_user`, `/admin`) missing middleware/decorators (e.g., `@login_required`).
- **Race Conditions (TOCTOU)**: Check-then-Act patterns on resources (e.g., `if (balance > cost) { ... deduct }`) without database transactions or atomic locks.
- **Business Logic**: Price manipulation (negative numbers), skipping payment steps, role escalation.

> **INTENDED FEATURES**: If a feature allows a "dangerous" action (e.g., "Run SQL Query", "Delete User") and it is **designed** for Admins/Owners and has authorization checks, it is **NOT** a vulnerability. Do not report "Ability to delete users" as a bug if that is the point of the endpoint.

### 4. Flow Verification & Bypass Analysis
Trace from **Source** to **Sink** and challenge the defense.
- **Sanitizer Matching**: Does the sanitizer MATCH the context?
    - *Example*: `escape_html()` is safe for HTML Body but **VULNERABLE** in `onclick="..."` or `<script>` contexts.
    - *Example*: `addslashes()` is **VULNERABLE** in specific SQL charsets or non-quoted integers.
- **Logic Validation**:
    - **IDOR**: Prove that `user_id` comes from INPUT, not SESSION.
    - **Auth**: Prove the route is reachable publicly.
    - **Validation**: If validation exists, look for bypasses (e.g., null byte `%00`, double encoding `%252e`, Unicode normalization issues).

# Reporting Guidelines (Strict)

Report **ONLY** confirmed exploitable bugs. Use the following Markdown format for EACH finding.

---
# [Vulnerability Title]

## Summary
[Detailed explanation of the vulnerability]

## Steps to Reproduce
1. [Step 1]
2. [Step 2]
3. [Step 3]

## Proof of Concept
```code
[Code or command examples demonstrating the vulnerability]
```

## References
- [Reference 1]
- [Reference 2]

## Impact
[Technical and business impact of the vulnerability]
---

# Directives (DO NOT BREAK)
1. **NO VERBOSITY**: Do not provide mitigation advice (e.g., "Use prepared statements"). Do not explain the vulnerability class.
2. **NO FALSE POSITIVES**: If the code is not provably vulnerable, do not report it.
3. **NO LOW SEVERITY**: Do not report missing headers, weak SSL ciphers, or generic config issues. Focus on **Data Flow Vulnerabilities**.
4. **REALISM**: Assume the role of an attacker. If you can't exploit it, it's not a bug.
5. **CONFIGURATION != VULN**: Do not flag insecure configuration options (e.g., `allow_unauthenticated=True`) as vulnerabilities unless they are **hardcoded** and **active** in a production context or default settings. If it's a user-configurable option, it's a feature.
""".strip()


@register("llama-cpp-auditor")
class LlamaCppAuditorProvider(BaseProvider):
    # ── Self-describing metadata (read by `revelyt providers`) ─────────────
    ENV_VAR = "None (local)"
    STRUCTURED_OUTPUT = "repair"
    NOTE = "Astra security persona (llama.cpp + Metal). Based on qwen2.5-auditor."
    TIKTOKEN_ENCODING = "cl100k_base"

    MODEL_ALIASES = {
        "llama-cpp-auditor": "qwen2.5-coder-14b-auditor-q4_k_m",
    }

    MODEL_COSTS = {}  # local — all $0
    # ─────────────────────────────────────────────────────────────────────────

    DEFAULT_URL = "http://localhost:8080"

    # Auditor-specific parameters (from Modelfile)
    AUDITOR_PARAMS = {
        "temperature": 0.0,
        "top_k": 10,
        "top_p": 1.0,
        "min_p": 0.0,
        "repeat_penalty": 1.15,
        "repeat_last_n": 256,
        "seed": 42,
        "max_tokens": 8192,
    }

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "llama-cpp-auditor",
        base_url: Optional[str] = None,
        **kwargs,
    ):
        resolved = self.MODEL_ALIASES.get(model_name.lower(), model_name)
        self._base_url = base_url or os.getenv("LLAMA_CPP_URL", self.DEFAULT_URL)
        self._resolved_model = resolved

        super().__init__(api_key=None, model_name=resolved)
        self.cost_per_1m_input = 0.0
        self.cost_per_1m_output = 0.0

    def get_llm(self) -> ChatOpenAI:
        """
        Use ChatOpenAI because llama-server exposes an OpenAI-compatible API.
        Configured with Astra auditor parameters for deterministic, security-focused output.
        """
        return ChatOpenAI(
            model=self._resolved_model,
            base_url=f"{self._base_url}/v1",
            api_key="not-needed",
            temperature=self.AUDITOR_PARAMS["temperature"],
            max_tokens=self.AUDITOR_PARAMS["max_tokens"],
            top_p=self.AUDITOR_PARAMS["top_p"],
            timeout=600,   # 10 min — local 14B needs 2-5 min per batch
            max_retries=1,  # Don't retry on timeout — just fail fast
        )

    def invoke_structured(
        self,
        prompt: ChatPromptTemplate,
        output_schema: Type[T],
        **kwargs: Any,
    ) -> T:
        """
        Inject the Astra system prompt (only for security tasks) and use the base class repair-loop.
        """
        # ONLY inject Astra persona if doing a security audit (FindingList or BatchFindingList).
        # We MUST NOT inject it for ProductManager tasks (SummaryOutput) as it conflicts
        # and deeply confuses smaller models like 3B.
        if output_schema.__name__ in ("FindingList", "BatchFindingList"):
            prompt = self._inject_astra_prompt(prompt)
            
        return super().invoke_structured(prompt, output_schema, **kwargs)

    def _inject_astra_prompt(self, prompt: ChatPromptTemplate) -> ChatPromptTemplate:
        """
        Prepend the Astra security engineer persona to the system prompt.
        """
        messages = list(prompt.messages)
        if not messages:
            return prompt

        # Find the first system message and prepend Astra persona
        new_messages = []
        injected = False
        for msg in messages:
            if not injected and hasattr(msg, "role") and msg.role == "system":
                original_content = msg.content if hasattr(msg, "content") else str(msg)
                new_msg = type(msg)(
                    content=f"{_ASTRA_SYSTEM_PROMPT}\n\n---\n\n{original_content}"
                )
                new_messages.append(new_msg)
                injected = True
            else:
                new_messages.append(msg)

        if not injected:
            # No system message found — insert one at the start
            new_messages.insert(0, SystemMessage(content=_ASTRA_SYSTEM_PROMPT))

        return ChatPromptTemplate.from_messages(new_messages)
