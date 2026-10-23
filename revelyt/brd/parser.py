import logging
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from revelyt.core.session import SessionManager
from revelyt.providers.base import BaseProvider
from revelyt.core.schemas import Requirement, RequirementList

logger = logging.getLogger(__name__)


class BRDParser:
    """
    Parses a Markdown Business Requirement Document (.md) and extracts
    verifiable requirements into the SQLite database.

    Extraction design principles
    ----------------------------
    1. No hard cap — the model extracts every requirement it finds. The
       verifier handles batching so a larger requirement list is not a problem.
       A cap causes lossy extraction: different models choose different subsets,
       producing requirement counts that vary between runs (15 vs 32 vs 42).

    2. One behaviour per requirement — the prompt enforces strict atomicity.
       "The system must allow CRUD for books" is split into four requirements,
       not merged into one. This makes verification more precise and prevents
       a single ambiguous requirement from masking multiple gaps.

    3. Explicit category schema — every requirement is labelled with one of
       seven categories (Business, Logic, Technical, Database, Security, UI,
       Performance). The category anchors what the model considers "verifiable",
       producing consistent granularity across models and runs.

    4. Idempotent storage — _save_to_db() clears existing requirements for the
       current session before inserting, so retrying a failed scan does not
       duplicate rows.
    """

    # The seven categories every extracted requirement must belong to.
    # Listing them explicitly in the prompt prevents models from inventing
    # their own categorisation scheme (which varies between runs).
    _CATEGORIES = (
        "Business",     # user-facing feature or workflow
        "Logic",        # algorithmic rule or business rule
        "Technical",    # framework, language, library, or infrastructure choice
        "Database",     # data model, constraint, migration, or persistence rule
        "Security",     # credential handling, auth, encryption, data protection
        "UI",           # API contract, endpoint shape, HTTP status code, response format
        "Performance",  # concurrency, throughput, latency, scalability requirement
    )

    _SYSTEM_PROMPT = (
        "You are an expert Business Analyst extracting software requirements "
        "from a specification document.\n\n"
        "EXTRACTION RULES — follow every rule exactly:\n\n"
        "1. ATOMICITY: Each requirement must describe exactly ONE testable "
        "behaviour or constraint. Do NOT merge separate behaviours into one "
        "item. Do NOT split a single behaviour into multiple items.\n\n"
        "2. COMPLETENESS: Extract every distinct, testable requirement in the "
        "document. Do not apply any limit on the number of requirements. "
        "Include functional requirements, data model constraints, technical "
        "stack choices, security requirements, and API contract requirements.\n\n"
        "3. VERIFIABILITY: Each requirement must be verifiable by inspecting "
        "source code or running tests. Omit vague goals like 'the system must "
        "be maintainable' that cannot be confirmed from code.\n\n"
        "4. CATEGORY: Classify each requirement as exactly one of: "
        f"{', '.join(_CATEGORIES)}.\n\n"
        "5. LANGUAGE: Write each requirement as a complete sentence starting "
        "with 'The system must...' or 'The [component] must...'.\n\n"
        "Return the full list as structured output."
    )

    def __init__(self, session: SessionManager, provider: BaseProvider):
        self.session = session
        self.provider = provider

    def parse_and_store(self, brd_path: Path):
        logger.info("Reading BRD from: %s", brd_path)
        if not brd_path.exists():
            logger.error("BRD file not found: %s", brd_path)
            raise FileNotFoundError(f"BRD file not found: {brd_path}")

        content = brd_path.read_text(errors="ignore")

        prompt = ChatPromptTemplate.from_messages([
            ("system", self._SYSTEM_PROMPT),
            ("user",   "Document:\n\n{document}"),
        ])

        logger.info("Extracting requirements via LLM...")
        result: RequirementList = self.provider.invoke_structured(
            prompt=prompt,
            output_schema=RequirementList,
            document=content,
        )

        self._save_to_db(result)
        logger.info("Extracted and saved %d requirements", len(result.requirements))

    def _save_to_db(self, req_list: RequirementList) -> None:
        """
        Persist extracted requirements, replacing any from a previous attempt.

        Idempotent: clears existing brd_requirements rows before inserting
        so that retrying a failed scan does not produce duplicate rows.
        The context manager in get_connection() handles commit/rollback.
        """
        with self.session.get_connection() as conn:
            # Clear any requirements stored by an earlier attempt this session.
            # brd_requirements has no session_id column (it's session-scoped by
            # design — one scan = one set of BRD results), so a full clear is safe.
            conn.execute("DELETE FROM brd_requirements")

            for req in req_list.requirements:
                # Guard: some providers return dicts instead of Requirement objects
                # when structured output parsing partially fails.
                if isinstance(req, dict):
                    logger.warning("Requirement item was a dict; coercing to Requirement model.")
                    req = Requirement(**req)
                elif not isinstance(req, Requirement):
                    logger.warning(
                        "Skipping unexpected requirement item of type %s: %r",
                        type(req).__name__, req,
                    )
                    continue

                # Use the category prefix in the stored requirement text so the
                # verifier prompt has context without a separate column.
                category = getattr(req, "category", "")
                prefix = f"[{category}] " if category else ""
                conn.execute(
                    "INSERT INTO brd_requirements (requirement, status) VALUES (?, ?)",
                    (f"{prefix}{req.description}", "PENDING"),
                )
            # No explicit conn.commit() needed — get_connection() commits on exit.