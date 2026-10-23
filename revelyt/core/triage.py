import logging
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console

from revelyt.core.schemas import TriageDecision
from revelyt.providers.base import BaseProvider

logger = logging.getLogger(__name__)
console = Console()

class TriageAgent:
    """
    Uses a fast, cheap model (e.g. flash) to quickly assess a diff
    and decide which expensive review agents need to be run.
    """
    def __init__(self, provider: BaseProvider):
        self.provider = provider
        self.emoji = "⚕️ "

    def analyze_diff(self, diff_content: str) -> TriageDecision:
        if not diff_content or not diff_content.strip():
            return TriageDecision(
                complexity_score=1, 
                is_trivial=True,
                needs_security_review=False, 
                needs_qa_review=False
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert technical lead triaging a code diff. "
                "Your job is to quickly assess the complexity and risk of the changes. "
                "1. If the diff only contains formatting, comments, or documentation changes, "
                "mark it as trivial. "
                "2. If the diff touches authentication, authorization, secrets, or raw SQL, "
                "mark needs_security_review as true. "
                "3. If the diff adds major features or touches core logic, "
                "mark needs_qa_review as true. "
                "Score complexity from 1 (trivial) to 10 (architectural rewrite)."
            )),
            ("user", (
                "Here is the git diff:\n\n{diff_content}\n\n"
                "Return a structured triage decision."
            ))
        ])

        try:
            # We want this to be extremely brief.
            result = self.provider.invoke_structured(
                prompt=prompt,
                output_schema=TriageDecision,
                diff_content=diff_content,
            )
            return result
        except Exception as e:
            logger.warning("Triage failed: %s", e)
            return TriageDecision(
                complexity_score=10, 
                is_trivial=False,
                needs_security_review=True, 
                needs_qa_review=True
            )
