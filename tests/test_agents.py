"""
Unit tests for the Revelyt Agent system.

Tests cover:
 - BaseAgent prompt construction (with and without Wave 1 context)
 - Language detection
 - Finding persistence
 - CombinedWave1Agent prompt content (architecture + security)
 - CombinedWave2Agent prompt content (QA + DevExp + Docs)
 - ProductManagerAgent summary and BRD verification
 - Graceful failure on provider errors
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from revelyt.core.schemas import Finding, FindingList, VerificationResult
from revelyt.core.session import SessionManager
from revelyt.agents.base_agent import BaseAgent
from revelyt.agents.combined_wave1_agent import CombinedWave1Agent
from revelyt.agents.combined_wave2_agent import CombinedWave2Agent
from revelyt.agents.product_manager_agent import ProductManagerAgent, SummaryOutput


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def session():
    """In-memory session for testing."""
    return SessionManager(db_path=":memory:")


@pytest.fixture
def mock_provider():
    """A mock provider that returns empty FindingList by default."""
    provider = MagicMock()
    provider.invoke_structured.return_value = FindingList(findings=[])
    provider.model_name = "gemini-2.0-flash"
    return provider


@pytest.fixture
def tmp_py_file(tmp_path):
    """Create a temporary Python file."""
    f = tmp_path / "example.py"
    f.write_text("def hello():\n    return 'world'\n")
    return f


@pytest.fixture
def tmp_js_file(tmp_path):
    """Create a temporary JavaScript file."""
    f = tmp_path / "app.js"
    f.write_text("const express = require('express');\n")
    return f


@pytest.fixture
def tmp_env_file(tmp_path):
    """Create a temporary .env file."""
    f = tmp_path / ".env"
    f.write_text("SECRET_KEY=abc123\n")
    return f


@pytest.fixture
def tmp_md_file(tmp_path):
    """Create a temporary markdown file (unknown language)."""
    f = tmp_path / "README.md"
    f.write_text("# Hello\n")
    return f


# ── BaseAgent Tests ─────────────────────────────────────────────────────────

class TestBaseAgentPrompt:
    def test_build_prompt_no_context(self, session, mock_provider, tmp_py_file):
        """Verify _build_prompt() without context returns system + user messages."""
        agent = CombinedWave1Agent(session, mock_provider)
        prompt = agent._build_prompt("python", "def foo(): pass")
        messages = prompt.format_messages(file_path="test.py", code_content="def foo(): pass")
        # System + code + prohibitions/evidence (no Wave 1 context block)
        assert len(messages) >= 2

    def test_build_prompt_with_context(self, session, mock_provider):
        """Verify _build_prompt() with FindingList injects WAVE 1 FINDINGS block."""
        agent = CombinedWave2Agent(session, mock_provider)
        context = FindingList(findings=[
            Finding(
                severity="HIGH",
                category="Code Quality",
                title="Missing error handling",
                description="No try/except around DB call",
                suggestion="Add try/except",
                confidence=0.9,
            )
        ])
        prompt = agent._build_prompt("python", "def foo(): pass", context=context)
        messages = prompt.format_messages(file_path="test.py", code_content="def foo(): pass")
        # Should have more messages than without context
        assert len(messages) >= 3
        # One of the messages should contain Wave 1 findings reference
        all_content = " ".join(m.content for m in messages)
        assert "Missing error handling" in all_content


class TestBaseAgentLanguageDetection:
    def test_python_detection(self, tmp_py_file):
        assert BaseAgent._get_language(tmp_py_file) == "python"

    def test_javascript_detection(self, tmp_js_file):
        assert BaseAgent._get_language(tmp_js_file) == "javascript"

    def test_config_detection(self, tmp_env_file):
        assert BaseAgent._get_language(tmp_env_file) == "config"

    def test_unknown_detection(self, tmp_md_file):
        assert BaseAgent._get_language(tmp_md_file) == "unknown"

    def test_typescript_detection(self, tmp_path):
        f = tmp_path / "index.tsx"
        f.write_text("export default function App() {}")
        assert BaseAgent._get_language(f) == "javascript"

    def test_yaml_config_detection(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("key: value")
        assert BaseAgent._get_language(f) == "config"


class TestBaseAgentSaveFindings:
    def test_save_findings_inserts_rows(self, session, mock_provider):
        """Verify _save_findings() inserts correct number of rows."""
        agent = CombinedWave1Agent(session, mock_provider)
        findings = [
            Finding(
                severity="HIGH", category="Architecture",
                title="Missing index", description="No index on user_id",
                suggestion="Add index", confidence=0.85,
            ),
            Finding(
                severity="MEDIUM", category="Architecture",
                title="Blocking I/O", description="sync in async",
                suggestion="Use async version", confidence=0.7,
            ),
        ]
        agent._save_findings("/path/to/file.py", findings)

        with session.get_connection() as conn:
            rows = conn.execute("SELECT * FROM findings").fetchall()
        assert len(rows) == 2
        assert rows[0]["severity"] == "HIGH"
        assert rows[0]["category"] == "Architecture"
        assert rows[1]["severity"] == "MEDIUM"


# ── Combined Agent Tests ────────────────────────────────────────────────────

class TestCombinedWave1Agent:
    def test_prompt_contains_security_and_architecture(self, session, mock_provider):
        """CombinedWave1Agent system prompt references security and architecture."""
        agent = CombinedWave1Agent(session, mock_provider)
        prompt_text = agent._build_system_prompt("python")
        assert "Security" in prompt_text or "security" in prompt_text.lower()
        assert "Architecture" in prompt_text or "Scalability" in prompt_text

    def test_prompt_contains_code_quality(self, session, mock_provider):
        """CombinedWave1Agent system prompt references code quality."""
        agent = CombinedWave1Agent(session, mock_provider)
        prompt_text = agent._build_system_prompt("python")
        assert "SOLID" in prompt_text or "Code Quality" in prompt_text

    def test_wave_attribute(self, session, mock_provider):
        agent = CombinedWave1Agent(session, mock_provider)
        assert agent.wave == 1

    def test_role_name(self, session, mock_provider):
        agent = CombinedWave1Agent(session, mock_provider)
        assert agent.role_name == "Reviewer"


class TestCombinedWave2Agent:
    def test_prompt_contains_reliability_and_devex(self, session, mock_provider):
        """CombinedWave2Agent system prompt references QA, DevExp, and Docs."""
        agent = CombinedWave2Agent(session, mock_provider)
        prompt_text = agent._build_system_prompt("python")
        assert "Reliability" in prompt_text or "QA" in prompt_text
        assert "Developer Experience" in prompt_text or "ergonomics" in prompt_text

    def test_prompt_contains_documentation(self, session, mock_provider):
        """CombinedWave2Agent system prompt references documentation review."""
        agent = CombinedWave2Agent(session, mock_provider)
        prompt_text = agent._build_system_prompt("python")
        assert "Documentation" in prompt_text or "docstring" in prompt_text.lower()

    def test_receives_wave1_context(self, session, mock_provider, tmp_py_file):
        """CombinedWave2Agent.analyze() passes context through to _build_prompt()."""
        context = FindingList(findings=[
            Finding(
                severity="HIGH", category="Code Quality",
                title="Risky function", description="No validation",
                suggestion="Add validation", confidence=0.9,
            )
        ])
        agent = CombinedWave2Agent(session, mock_provider)
        result = agent.analyze(tmp_py_file, context=context)
        assert isinstance(result, FindingList)
        # Provider was called (the prompt was built and invoked)
        mock_provider.invoke_structured.assert_called_once()

    def test_wave_attribute(self, session, mock_provider):
        agent = CombinedWave2Agent(session, mock_provider)
        assert agent.wave == 2

    def test_role_name(self, session, mock_provider):
        agent = CombinedWave2Agent(session, mock_provider)
        assert agent.role_name == "DeepReviewer"

    def test_javascript_prompt_variant(self, session, mock_provider):
        """CombinedWave2Agent adapts prompt for JavaScript/frontend."""
        agent = CombinedWave2Agent(session, mock_provider)
        prompt_text = agent._build_system_prompt("javascript")
        assert "JavaScript" in prompt_text or "JS/TS" in prompt_text


# ── ProductManagerAgent Tests ───────────────────────────────────────────────

class TestProductManagerAgent:
    def test_summarize_writes_module_summary(self, session, mock_provider, tmp_py_file):
        """Mock provider returning valid SummaryOutput. Verify module_summaries populated."""
        mock_provider.invoke_structured.return_value = SummaryOutput(
            purpose="A utility module for greeting users.",
            key_functions=["hello() — returns greeting string"],
        )

        # Pre-populate module_summaries with the file path
        rel_path = tmp_py_file.name
        with session.get_connection() as conn:
            conn.execute(
                "INSERT INTO module_summaries (file_path) VALUES (?)",
                (rel_path,)
            )

        agent = ProductManagerAgent(session, mock_provider, tmp_py_file.parent)
        agent._summarize_single(tmp_py_file)

        with session.get_connection() as conn:
            row = conn.execute(
                "SELECT purpose, key_functions FROM module_summaries WHERE file_path = ?",
                (rel_path,)
            ).fetchone()
        assert row is not None
        assert "utility module" in row["purpose"]
        assert "hello" in row["key_functions"]

    def test_verify_brd_reads_summaries(self, session, mock_provider, tmp_path):
        """Pre-populate module_summaries. Verify verify_brd() reads and produces results."""
        # Pre-populate summaries
        with session.get_connection() as conn:
            conn.execute(
                "INSERT INTO module_summaries (file_path, purpose, key_functions) VALUES (?, ?, ?)",
                ("app.py", "Main application entry point", json.dumps(["main()"])),
            )

        # Create a BRD file
        brd_file = tmp_path / "requirements.md"
        brd_file.write_text("The system must have a main entry point.")

        # Mock: first call returns RequirementList, subsequent calls return VerificationResultList
        from revelyt.core.schemas import RequirementList, Requirement, VerificationResultList, VerificationResult
        mock_provider.invoke_structured.side_effect = [
            RequirementList(requirements=[
                Requirement(id=1, description="Must have entry point", category="Logic")
            ]),
            VerificationResultList(results=[
                VerificationResult(
                    requirement_id=1, status="MET",
                    justification="app.py contains main()",
                    evidence_files=["app.py"],
                )
            ]),
        ]

        agent = ProductManagerAgent(session, mock_provider, tmp_path)
        agent.verify_brd(brd_file)

        # Verify the requirement was saved
        with session.get_connection() as conn:
            rows = conn.execute("SELECT * FROM brd_requirements").fetchall()
        assert len(rows) == 1
        assert rows[0]["status"] == "MET"


# ── Failure Resilience ──────────────────────────────────────────────────────

class TestAgentFailureResilience:
    def test_all_agents_empty_result_on_provider_failure(self, session, tmp_py_file):
        """Force provider to raise. Verify each agent returns empty FindingList."""
        failing_provider = MagicMock()
        failing_provider.invoke_structured.side_effect = Exception("LLM is down")
        failing_provider.model_name = "gemini-2.0-flash"

        agents = [
            CombinedWave1Agent(session, failing_provider),
            CombinedWave2Agent(session, failing_provider),
        ]

        for agent in agents:
            result = agent.analyze(tmp_py_file)
            assert isinstance(result, FindingList), f"{agent.role_name} didn't return FindingList"
            assert len(result.findings) == 0, f"{agent.role_name} returned findings on failure"
