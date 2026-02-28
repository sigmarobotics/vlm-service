from vlm_service.prompt import PromptBuilder


class TestPromptBuilder:

    def test_empty_builder(self):
        builder = PromptBuilder()
        assert builder.build() == ""

    def test_single_section(self):
        builder = PromptBuilder()
        builder.add_section("Role", "You are a patrol inspector.")
        result = builder.build()
        assert "[Role]" in result
        assert "You are a patrol inspector." in result

    def test_multiple_sections(self):
        builder = PromptBuilder()
        builder.add_section("Role", "Inspector")
        builder.add_section("Instructions", "Check for safety issues")
        result = builder.build()
        assert "[Role]" in result
        assert "[Instructions]" in result
        assert result.index("[Role]") < result.index("[Instructions]")

    def test_skip_empty_content(self):
        builder = PromptBuilder()
        builder.add_section("Role", "Inspector")
        builder.add_section("Empty", "")
        builder.add_section("Instructions", "Check safety")
        result = builder.build()
        assert "[Empty]" not in result
        assert "[Role]" in result
        assert "[Instructions]" in result

    def test_skip_none_content(self):
        builder = PromptBuilder()
        builder.add_section("Role", "Inspector")
        builder.add_section("Empty", None)
        result = builder.build()
        assert "[Empty]" not in result

    def test_placeholder_injection(self):
        builder = PromptBuilder()
        builder.add_section("Instructions", "Current time: {{CURRENT_TIME}}. Check area {{ZONE}}.")
        result = builder.build(placeholders={"CURRENT_TIME": "2026-02-28 10:00:00", "ZONE": "A1"})
        assert "2026-02-28 10:00:00" in result
        assert "A1" in result
        assert "{{CURRENT_TIME}}" not in result
        assert "{{ZONE}}" not in result

    def test_placeholder_missing_left_as_is(self):
        builder = PromptBuilder()
        builder.add_section("Instructions", "Time: {{CURRENT_TIME}}. Zone: {{ZONE}}.")
        result = builder.build(placeholders={"CURRENT_TIME": "2026-02-28"})
        assert "2026-02-28" in result
        assert "{{ZONE}}" in result  # Not replaced

    def test_output_format_section(self):
        builder = PromptBuilder()
        schema_desc = '{"is_ng": bool, "analysis": str, "hygiene": int}'
        builder.add_section("Output Format", f"Respond with JSON matching: {schema_desc}")
        result = builder.build()
        assert "[Output Format]" in result
        assert "is_ng" in result

    def test_language_section(self):
        builder = PromptBuilder()
        builder.add_section("Language", "Respond in Traditional Chinese (zh-TW)")
        result = builder.build()
        assert "zh-TW" in result

    def test_from_venue_config(self, sample_venue_config):
        """Convenience factory from a venue config dict."""
        builder = PromptBuilder.from_venue_config(
            venue_config=sample_venue_config,
            role="You are a hospital patrol inspector.",
            instructions="Check hygiene and safety.",
            knowledge="Standard hospital hygiene protocol: ...",
        )
        result = builder.build()
        assert "[Role]" in result
        assert "[Instructions]" in result
        assert "[Knowledge Base]" in result
        assert "[Language]" in result
        assert "zh-TW" in result
