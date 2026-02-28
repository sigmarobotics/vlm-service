"""Modular prompt builder for VLM requests.

Composes system prompts from named sections with placeholder injection.
Pattern from kachaka-gemini's gemini_text_service.py.

Sections: [Role] [Instructions] [Knowledge Base] [Output Format] [Language]
Placeholders: {{CURRENT_TIME}}, {{ZONE}}, or any custom key.
"""

from __future__ import annotations


class PromptBuilder:
    """Build system prompts from ordered sections."""

    def __init__(self):
        self._sections: list[tuple[str, str]] = []

    def add_section(self, title: str, content: str | None) -> PromptBuilder:
        """Add a named section. Empty/None content is silently skipped at build time."""
        self._sections.append((title, content or ""))
        return self

    def build(self, placeholders: dict[str, str] | None = None) -> str:
        """Assemble all non-empty sections and inject placeholders.

        Args:
            placeholders: dict of {{KEY}} -> value replacements.

        Returns:
            Assembled prompt string.
        """
        parts = []
        for title, content in self._sections:
            if content.strip():
                parts.append(f"[{title}]\n{content}")

        prompt = "\n\n".join(parts)

        if placeholders:
            for key, value in placeholders.items():
                prompt = prompt.replace(f"{{{{{key}}}}}", value)

        return prompt

    @classmethod
    def from_venue_config(
        cls,
        venue_config: dict,
        role: str = "",
        instructions: str = "",
        knowledge: str = "",
        output_format: str = "",
    ) -> PromptBuilder:
        """Create a PromptBuilder pre-populated from a venue config.

        Args:
            venue_config: Venue config dict (must have 'language' key).
            role: Role description text.
            instructions: Inspection instructions text.
            knowledge: Domain knowledge text.
            output_format: Output format description (auto-generated if empty).
        """
        builder = cls()
        builder.add_section("Role", role)
        builder.add_section("Instructions", instructions)
        builder.add_section("Knowledge Base", knowledge)
        builder.add_section("Output Format", output_format)

        language = venue_config.get("language", "")
        if language:
            builder.add_section("Language", f"Respond in {language}")

        return builder
