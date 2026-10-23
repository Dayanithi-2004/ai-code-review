"""
SkillsEngine — Modular, token-efficient code review skill injection.

Design principles (from user's Token Reduction Plan):
- Skills injected ONCE per wave, not per agent (10× reduction)
- Directive format: NO/YES/FLAG keyphrases, not sentences (~40% shorter)
- Wave-level mapping: each wave gets a curated subset of skills
"""

import yaml
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class SkillsEngine:
    """
    Loads code review skills and serves compact directive rules
    for wave-level injection into LLM context.

    A new instance is created per scan so skill files are always
    read fresh from disk — no stale cache across runs.
    """

    def __init__(self, skills_root: Optional[Path] = None):
        self.skills_root = skills_root or (Path(__file__).parent.parent.parent / "skills")
        self.config_path = self.skills_root / "config.yaml"
        self._config = self._load_config()
        # Per-instance cache: lives only for the duration of one scan.
        # Cleared on every new scan because get_skills_engine() always
        # returns a fresh SkillsEngine instance.
        self._skill_cache: Dict[str, str] = {}

    def _load_config(self) -> dict:
        """Load skills/config.yaml with role→skill mapping."""
        if not self.config_path.exists():
            logger.warning("Skills config not found at %s", self.config_path)
            return {"wave1_skills": [], "wave2_skills": []}
        try:
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f) or {}
                logger.debug("Skills config loaded from %s", self.config_path)
                return data
        except Exception as e:
            logger.error("Failed to load skills config: %s", e)
            return {"wave1_skills": [], "wave2_skills": []}

    def _load_compact_rules(self, skill_name: str) -> str:
        """
        Load the compact_rules from a skill's SKILL.md frontmatter.

        Results are cached within this instance (i.e. within one scan)
        so wave1 and wave2 don't re-read the same file if a skill is
        listed in both waves. The cache is NOT shared across scans.
        """
        if skill_name in self._skill_cache:
            return self._skill_cache[skill_name]

        skill_path = self.skills_root / skill_name / "SKILL.md"
        if not skill_path.exists():
            logger.warning("Skill not found: %s", skill_path)
            self._skill_cache[skill_name] = ""
            return ""

        try:
            content = skill_path.read_text(encoding="utf-8")
            # Parse YAML frontmatter between --- markers
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])
                    rules = frontmatter.get("compact_rules", "").strip()
                    logger.debug(
                        "Loaded skill '%s' (%d chars) from %s",
                        skill_name, len(rules), skill_path,
                    )
                    self._skill_cache[skill_name] = rules
                    return rules
        except Exception as e:
            logger.error("Failed to load skill %s: %s", skill_name, e)

        self._skill_cache[skill_name] = ""
        return ""

    def get_wave_compact_rules(self, wave: str) -> str:
        """
        Return a single compact rules string for an entire wave.
        This is injected ONCE per wave, not per agent.

        Args:
            wave: "wave1" or "wave2"

        Returns:
            A directive-format skills block, typically <150 tokens.
        """
        key = f"{wave}_skills"
        skill_names = self._config.get(key, [])

        if not skill_names:
            return ""

        parts = []
        for name in skill_names:
            rules = self._load_compact_rules(name)
            if rules:
                parts.append(rules)

        if not parts:
            return ""

        return "<skills>\n" + "\n".join(parts) + "\n</skills>"


# ─── Factory function — fresh instance every call ─────────────────────────
#
# Previously this was a module-level singleton (_engine global) which meant
# skill file content was cached in memory for the entire process lifetime.
# After editing a SKILL.md on disk, the old content stayed loaded until the
# process restarted.
#
# Fix: get_skills_engine() now always returns a new SkillsEngine instance.
# The per-instance _skill_cache still deduplicates reads within a single
# scan (wave1 + wave2 share the instance via orchestrator), but the next
# scan always starts cold.

def get_skills_engine(root: Optional[Path] = None) -> SkillsEngine:
    """
    Return a fresh SkillsEngine instance.

    A new instance is created on every call so SKILL.md files are
    always read from disk at the start of each scan. This ensures
    that editing a skill file takes effect immediately on the next
    run without needing to restart the process.
    """
    return SkillsEngine(root)