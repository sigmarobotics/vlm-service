# vlm-service Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract a generic, pip-installable VLM image/video analysis + report generation package from existing patrol projects.

**Architecture:** Thin wrapper around Google Gemini SDK. Dynamic JSON schemas built at runtime from venue config dicts — no hardcoded Pydantic models. Modular prompt composition with section-based templates and placeholder injection. Files API lifecycle management for knowledge base documents. Sync API throughout (matching VP-V1 base).

**Tech Stack:** `google-genai` (Gemini SDK), Python 3.10+, `pytest` + `unittest.mock`

**Source References:**
- Extraction base: `/home/snaken/CodeBase/visual-patrol/src/backend/cloud_ai_service.py` (VP-V1, 277 lines)
- Schema reference: `/home/snaken/CodeBase/visual-patrol-v2/src/backend/services/ai_service.py` (VP-V2, 678 lines)
- Prompt reference: `/home/snaken/CodeBase/kachaka-gemini/src/backend/gemini_text_service.py` (lines 44-61)
- Files API reference: `/home/snaken/CodeBase/kachaka-gemini/.claude/worktrees/gemini-files-api/src/backend/file_service.py` (269 lines)

**Key Design Decisions:**
- `response_json_schema` (dict) — NOT `response_schema` (Pydantic). Enables runtime dynamic schemas.
- Fixed core schema: only `is_ng` + `analysis` (sent to model). Token tracking from `response.usage_metadata` (separate layer).
- Dynamic schema: venue config drives scores, categories, custom fields — all generated as JSON schema dicts.
- JSON repair: NOT needed (structured output guarantees valid JSON). Keep regex extraction as thin fallback only.
- Provider: Gemini only. No VILA.
- Video analysis: file upload + polling (from VP-V1).
- Files API: full lifecycle (upload / list / delete / toggle / reupload / get_enabled).
- Report generation: uses same dynamic schema approach for structured reports.
- Sync API: all public methods are synchronous (Files API uses `threading` for lock, no `asyncio`).

---

### Task 1: Package Skeleton + Result Types

**Files:**
- Create: `vlm-service/pyproject.toml`
- Create: `vlm-service/vlm_service/__init__.py`
- Create: `vlm-service/vlm_service/types.py`
- Create: `vlm-service/tests/__init__.py`
- Create: `vlm-service/tests/conftest.py`
- Test: `vlm-service/tests/test_types.py`

**Step 1: Create project directory and pyproject.toml**

```bash
mkdir -p vlm-service/vlm_service vlm-service/tests
```

```toml
# vlm-service/pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "vlm-service"
version = "0.1.0"
description = "Generic VLM image/video analysis and report generation service"
requires-python = ">=3.10"
dependencies = [
    "google-genai>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]
```

**Step 2: Create vlm_service/__init__.py**

```python
"""vlm-service — Generic VLM image/video analysis + report generation."""
```

**Step 3: Write the failing test for result types**

```python
# tests/test_types.py
from vlm_service.types import InspectionResult, ReportResult, VideoResult


def test_inspection_result_from_response():
    raw = {"is_ng": True, "analysis": "Fire hazard detected"}
    usage = {"prompt_token_count": 100, "candidates_token_count": 50, "total_token_count": 150}
    result = InspectionResult(raw_result=raw, usage=usage)
    assert result.is_ng is True
    assert result.analysis == "Fire hazard detected"
    assert result.prompt_tokens == 100
    assert result.output_tokens == 50
    assert result.total_tokens == 150
    assert result.raw_result == raw


def test_inspection_result_defaults():
    result = InspectionResult(raw_result={}, usage={})
    assert result.is_ng is False
    assert result.analysis == ""
    assert result.total_tokens == 0


def test_inspection_result_with_dynamic_fields():
    raw = {"is_ng": False, "analysis": "All clear", "hygiene_score": 9, "fire_safety": "OK"}
    result = InspectionResult(raw_result=raw, usage={})
    assert result.raw_result["hygiene_score"] == 9
    assert result.raw_result["fire_safety"] == "OK"


def test_report_result():
    raw = {"summary": "All locations normal", "sections": []}
    usage = {"prompt_token_count": 200, "candidates_token_count": 300, "total_token_count": 500}
    result = ReportResult(raw_result=raw, usage=usage)
    assert result.raw_result["summary"] == "All locations normal"
    assert result.total_tokens == 500


def test_video_result():
    raw = {"is_ng": False, "analysis": "No issues in video"}
    usage = {"prompt_token_count": 1000, "candidates_token_count": 100, "total_token_count": 1100}
    result = VideoResult(raw_result=raw, usage=usage)
    assert result.is_ng is False
    assert result.analysis == "No issues in video"
    assert result.total_tokens == 1100
```

**Step 4: Run test to verify it fails**

Run: `cd vlm-service && pip install -e ".[dev]" && pytest tests/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vlm_service.types'`

**Step 5: Write minimal implementation**

```python
# vlm_service/types.py
"""Result types for VLM service responses.

These are plain data containers — NOT Pydantic models.
The schema sent to Gemini is a dict (response_json_schema), not Pydantic.
These types wrap the raw dict response + usage metadata for convenience.
"""

from dataclasses import dataclass, field


@dataclass
class _BaseResult:
    """Common fields for all VLM results."""
    raw_result: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_token_count", 0)

    @property
    def output_tokens(self) -> int:
        return self.usage.get("candidates_token_count", 0)

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_token_count", 0)


@dataclass
class InspectionResult(_BaseResult):
    """Result of an image inspection."""

    @property
    def is_ng(self) -> bool:
        return self.raw_result.get("is_ng", False)

    @property
    def analysis(self) -> str:
        return self.raw_result.get("analysis", "")


@dataclass
class VideoResult(_BaseResult):
    """Result of a video analysis."""

    @property
    def is_ng(self) -> bool:
        return self.raw_result.get("is_ng", False)

    @property
    def analysis(self) -> str:
        return self.raw_result.get("analysis", "")


@dataclass
class ReportResult(_BaseResult):
    """Result of a report generation."""
    pass
```

**Step 6: Run test to verify it passes**

Run: `pytest tests/test_types.py -v`
Expected: 5 passed

**Step 7: Create conftest.py and tests/__init__.py**

```python
# tests/__init__.py
```

```python
# tests/conftest.py
"""Shared test fixtures for vlm-service."""
import pytest


@pytest.fixture
def dummy_api_key():
    return "test-api-key-not-real"


@pytest.fixture
def sample_venue_config():
    """Example venue config for a hospital patrol."""
    return {
        "venue_type": "hospital",
        "scores": {
            "hygiene": {"type": "integer", "description": "Hygiene score 1-10", "minimum": 1, "maximum": 10},
            "safety": {"type": "integer", "description": "Safety score 1-10", "minimum": 1, "maximum": 10},
        },
        "categories": ["cleanliness", "equipment", "fire_safety", "access"],
        "custom_fields": {
            "ward_id": {"type": "string", "description": "Ward identifier"},
        },
        "language": "zh-TW",
    }


@pytest.fixture
def sample_factory_config():
    """Example venue config for a factory patrol."""
    return {
        "venue_type": "factory",
        "scores": {
            "fire_risk": {"type": "integer", "description": "Fire risk score 1-10", "minimum": 1, "maximum": 10},
        },
        "categories": ["fire_prevention", "machinery", "ppe", "chemical_storage"],
        "custom_fields": {
            "zone": {"type": "string", "description": "Factory zone"},
            "line_number": {"type": "integer", "description": "Production line number"},
        },
        "language": "en",
    }
```

**Step 8: Commit**

```bash
git init
git add -A
git commit -m "feat: package skeleton with result types and test fixtures"
```

---

### Task 2: Dynamic JSON Schema Builder

**Files:**
- Create: `vlm-service/vlm_service/schema.py`
- Test: `vlm-service/tests/test_schema.py`

**Step 1: Write the failing test**

```python
# tests/test_schema.py
import json
from vlm_service.schema import build_inspection_schema, build_report_schema


class TestBuildInspectionSchema:
    """Test dynamic inspection schema construction."""

    def test_core_only(self):
        """No venue config → only is_ng + analysis."""
        schema = build_inspection_schema()
        assert schema["type"] == "OBJECT"
        props = schema["properties"]
        assert "is_ng" in props
        assert props["is_ng"]["type"] == "BOOLEAN"
        assert "analysis" in props
        assert props["analysis"]["type"] == "STRING"
        assert set(schema["required"]) == {"is_ng", "analysis"}

    def test_with_scores(self, sample_venue_config):
        """Venue config with scores → adds score properties."""
        schema = build_inspection_schema(sample_venue_config)
        props = schema["properties"]
        assert "is_ng" in props
        assert "analysis" in props
        assert "hygiene" in props
        assert props["hygiene"]["type"] == "INTEGER"
        assert "safety" in props

    def test_with_categories(self, sample_venue_config):
        """Venue config with categories → adds category enum field."""
        schema = build_inspection_schema(sample_venue_config)
        props = schema["properties"]
        assert "categories" in props
        assert props["categories"]["type"] == "ARRAY"
        items = props["categories"]["items"]
        assert set(items["enum"]) == {"cleanliness", "equipment", "fire_safety", "access"}

    def test_with_custom_fields(self, sample_venue_config):
        """Venue config with custom_fields → adds them to properties."""
        schema = build_inspection_schema(sample_venue_config)
        props = schema["properties"]
        assert "ward_id" in props
        assert props["ward_id"]["type"] == "STRING"

    def test_different_venues_produce_different_schemas(self, sample_venue_config, sample_factory_config):
        """Two different venue configs produce different schemas."""
        hospital = build_inspection_schema(sample_venue_config)
        factory = build_inspection_schema(sample_factory_config)
        assert "hygiene" in hospital["properties"]
        assert "hygiene" not in factory["properties"]
        assert "fire_risk" in factory["properties"]
        assert "fire_risk" not in hospital["properties"]

    def test_core_fields_always_required(self, sample_venue_config):
        """is_ng and analysis are always in required."""
        schema = build_inspection_schema(sample_venue_config)
        assert "is_ng" in schema["required"]
        assert "analysis" in schema["required"]


class TestBuildReportSchema:
    """Test dynamic report schema construction."""

    def test_default_report_schema(self):
        """No config → basic summary + details structure."""
        schema = build_report_schema()
        props = schema["properties"]
        assert "summary" in props
        assert "details" in props
        assert schema["type"] == "OBJECT"

    def test_report_with_sections(self):
        """Config with sections → adds section fields."""
        config = {
            "sections": ["overview", "findings", "recommendations"],
            "custom_sections": {
                "risk_assessment": {"type": "STRING", "description": "Risk assessment summary"},
            },
        }
        schema = build_report_schema(config)
        props = schema["properties"]
        assert "overview" in props
        assert "findings" in props
        assert "recommendations" in props
        assert "risk_assessment" in props
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vlm_service.schema'`

**Step 3: Write minimal implementation**

```python
# vlm_service/schema.py
"""Dynamic JSON schema builder for Gemini structured output.

Constructs `response_json_schema` dicts at runtime from venue config.
Uses Gemini's JSON schema subset (OBJECT, STRING, BOOLEAN, INTEGER, NUMBER, ARRAY).

Key principle: fixed core (is_ng + analysis) + dynamic fields from config.
"""

# Gemini uses uppercase type names in its JSON schema format
_TYPE_MAP = {
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}


def _map_type(t: str) -> str:
    """Map standard JSON schema types to Gemini uppercase format."""
    return _TYPE_MAP.get(t.lower(), "STRING")


def build_inspection_schema(venue_config: dict | None = None) -> dict:
    """Build a Gemini response_json_schema for image inspection.

    Args:
        venue_config: Optional venue-specific config with keys:
            - scores: dict of {name: {type, description, ...}}
            - categories: list of category strings (becomes enum array)
            - custom_fields: dict of {name: {type, description}}

    Returns:
        dict suitable for passing as response_json_schema to Gemini.
    """
    properties = {
        "is_ng": {
            "type": "BOOLEAN",
            "description": "True if abnormal/NG condition detected, False if normal/OK",
        },
        "analysis": {
            "type": "STRING",
            "description": "Detailed analysis description. Empty string if OK.",
        },
    }
    required = ["is_ng", "analysis"]

    if venue_config:
        # Add score fields
        for name, spec in venue_config.get("scores", {}).items():
            prop = {
                "type": _map_type(spec.get("type", "integer")),
                "description": spec.get("description", name),
            }
            if "minimum" in spec:
                prop["minimum"] = spec["minimum"]
            if "maximum" in spec:
                prop["maximum"] = spec["maximum"]
            properties[name] = prop

        # Add categories enum array
        cats = venue_config.get("categories", [])
        if cats:
            properties["categories"] = {
                "type": "ARRAY",
                "description": "Applicable issue categories",
                "items": {
                    "type": "STRING",
                    "enum": cats,
                },
            }

        # Add custom fields
        for name, spec in venue_config.get("custom_fields", {}).items():
            properties[name] = {
                "type": _map_type(spec.get("type", "string")),
                "description": spec.get("description", name),
            }

    return {
        "type": "OBJECT",
        "properties": properties,
        "required": required,
    }


def build_report_schema(report_config: dict | None = None) -> dict:
    """Build a Gemini response_json_schema for report generation.

    Args:
        report_config: Optional config with keys:
            - sections: list of section names (each becomes a STRING property)
            - custom_sections: dict of {name: {type, description}}

    Returns:
        dict suitable for passing as response_json_schema to Gemini.
    """
    properties = {
        "summary": {
            "type": "STRING",
            "description": "Executive summary of the patrol report",
        },
        "details": {
            "type": "STRING",
            "description": "Detailed findings and observations",
        },
    }
    required = ["summary", "details"]

    if report_config:
        for section_name in report_config.get("sections", []):
            properties[section_name] = {
                "type": "STRING",
                "description": f"{section_name.replace('_', ' ').title()} section",
            }
            required.append(section_name)

        for name, spec in report_config.get("custom_sections", {}).items():
            properties[name] = {
                "type": _map_type(spec.get("type", "string")),
                "description": spec.get("description", name),
            }

    return {
        "type": "OBJECT",
        "properties": properties,
        "required": required,
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_schema.py -v`
Expected: 8 passed

**Step 5: Commit**

```bash
git add vlm_service/schema.py tests/test_schema.py
git commit -m "feat: dynamic JSON schema builder for inspection and report"
```

---

### Task 3: Modular Prompt Builder

**Files:**
- Create: `vlm-service/vlm_service/prompt.py`
- Test: `vlm-service/tests/test_prompt.py`

**Step 1: Write the failing test**

```python
# tests/test_prompt.py
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vlm_service.prompt'`

**Step 3: Write minimal implementation**

```python
# vlm_service/prompt.py
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompt.py -v`
Expected: 10 passed

**Step 5: Commit**

```bash
git add vlm_service/prompt.py tests/test_prompt.py
git commit -m "feat: modular prompt builder with section composition and placeholders"
```

---

### Task 4: Gemini Provider — Image Inspection

**Files:**
- Create: `vlm-service/vlm_service/provider.py`
- Test: `vlm-service/tests/test_provider.py`

**Step 1: Write the failing test**

```python
# tests/test_provider.py
import json
from unittest.mock import MagicMock, patch
import pytest
from vlm_service.provider import GeminiProvider
from vlm_service.types import InspectionResult


class TestGeminiProviderConfigure:

    def test_configure_creates_client(self):
        with patch("vlm_service.provider.genai") as mock_genai:
            provider = GeminiProvider()
            provider.configure(api_key="test-key", model="gemini-2.0-flash")
            mock_genai.Client.assert_called_once_with(api_key="test-key")
            assert provider.is_configured()

    def test_configure_without_key(self):
        provider = GeminiProvider()
        provider.configure(api_key="", model="gemini-2.0-flash")
        assert not provider.is_configured()

    def test_reconfigure_only_on_change(self):
        with patch("vlm_service.provider.genai") as mock_genai:
            provider = GeminiProvider()
            provider.configure(api_key="key1", model="gemini-2.0-flash")
            provider.configure(api_key="key1", model="gemini-2.0-flash")
            assert mock_genai.Client.call_count == 1

    def test_reconfigure_on_key_change(self):
        with patch("vlm_service.provider.genai") as mock_genai:
            provider = GeminiProvider()
            provider.configure(api_key="key1", model="gemini-2.0-flash")
            provider.configure(api_key="key2", model="gemini-2.0-flash")
            assert mock_genai.Client.call_count == 2


class TestGeminiProviderInspection:

    def _make_mock_response(self, result_dict, prompt_tokens=100, output_tokens=50):
        response = MagicMock()
        response.text = json.dumps(result_dict)
        response.usage_metadata.prompt_token_count = prompt_tokens
        response.usage_metadata.candidates_token_count = output_tokens
        response.usage_metadata.total_token_count = prompt_tokens + output_tokens
        return response

    def test_generate_inspection_basic(self):
        with patch("vlm_service.provider.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            result_data = {"is_ng": True, "analysis": "Fire hazard"}
            mock_client.models.generate_content.return_value = self._make_mock_response(result_data)

            provider = GeminiProvider()
            provider.configure(api_key="test-key", model="gemini-2.0-flash")

            schema = {"type": "OBJECT", "properties": {"is_ng": {"type": "BOOLEAN"}, "analysis": {"type": "STRING"}}, "required": ["is_ng", "analysis"]}
            result = provider.generate_inspection(
                image=b"fake-image-bytes",
                user_prompt="Check this area",
                schema=schema,
            )

            assert isinstance(result, InspectionResult)
            assert result.is_ng is True
            assert result.analysis == "Fire hazard"
            assert result.total_tokens == 150

    def test_generate_inspection_with_system_prompt(self):
        with patch("vlm_service.provider.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.models.generate_content.return_value = self._make_mock_response({"is_ng": False, "analysis": ""})

            provider = GeminiProvider()
            provider.configure(api_key="test-key", model="gemini-2.0-flash")

            schema = {"type": "OBJECT", "properties": {"is_ng": {"type": "BOOLEAN"}, "analysis": {"type": "STRING"}}, "required": ["is_ng", "analysis"]}
            provider.generate_inspection(
                image=b"fake-image-bytes",
                user_prompt="Check this area",
                system_prompt="You are an inspector",
                schema=schema,
            )

            call_kwargs = mock_client.models.generate_content.call_args
            config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
            assert config.system_instruction == "You are an inspector"

    def test_generate_inspection_not_configured_raises(self):
        provider = GeminiProvider()
        with pytest.raises(RuntimeError, match="not configured"):
            provider.generate_inspection(image=b"data", user_prompt="test", schema={})

    def test_generate_inspection_with_file_parts(self):
        """Test that file Parts (from Files API) are included in contents."""
        with patch("vlm_service.provider.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.models.generate_content.return_value = self._make_mock_response({"is_ng": False, "analysis": ""})

            provider = GeminiProvider()
            provider.configure(api_key="test-key", model="gemini-2.0-flash")

            file_parts = [MagicMock(), MagicMock()]  # Simulated Part.from_uri() objects
            schema = {"type": "OBJECT", "properties": {"is_ng": {"type": "BOOLEAN"}, "analysis": {"type": "STRING"}}, "required": ["is_ng", "analysis"]}
            provider.generate_inspection(
                image=b"data",
                user_prompt="Check",
                schema=schema,
                file_parts=file_parts,
            )

            call_args = mock_client.models.generate_content.call_args
            contents = call_args.kwargs.get("contents") or call_args[1].get("contents")
            # file_parts should be in the contents list
            assert len(contents) >= 4  # file_part1 + file_part2 + user_prompt + image
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_provider.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vlm_service.provider'`

**Step 3: Write minimal implementation**

```python
# vlm_service/provider.py
"""Gemini VLM provider — image inspection, video analysis, report generation.

Extracted from visual-patrol/src/backend/cloud_ai_service.py (VP-V1).
Key change: uses response_json_schema (dict) instead of response_schema (Pydantic).
"""

import json
import logging
import time

from google import genai
from google.genai import types

from vlm_service.types import InspectionResult, ReportResult, VideoResult

logger = logging.getLogger("vlm_service.provider")


class GeminiProvider:
    """Google Gemini VLM provider with structured output."""

    def __init__(self):
        self._client: genai.Client | None = None
        self._api_key: str = ""
        self._model: str = "gemini-2.0-flash"

    def configure(self, api_key: str, model: str = "gemini-2.0-flash"):
        """Configure or reconfigure the Gemini client.

        Only recreates the client if api_key or model changed.
        """
        if api_key == self._api_key and model == self._model and self._client is not None:
            return

        self._api_key = api_key
        self._model = model

        if self._api_key:
            self._client = genai.Client(api_key=self._api_key)
            logger.info("Gemini configured: model=%s", self._model)
        else:
            self._client = None
            logger.warning("Gemini configured without API key")

    def is_configured(self) -> bool:
        return self._client is not None

    @property
    def model_name(self) -> str:
        return self._model

    def _check_configured(self):
        if not self._client:
            raise RuntimeError("Gemini provider not configured. Call configure() with a valid API key first.")

    @staticmethod
    def _extract_usage(response) -> dict:
        try:
            usage = response.usage_metadata
            return {
                "prompt_token_count": usage.prompt_token_count,
                "candidates_token_count": usage.candidates_token_count,
                "total_token_count": usage.total_token_count,
            }
        except Exception as e:
            logger.warning("Could not extract token usage: %s", e)
            return {}

    def generate_inspection(
        self,
        image: bytes,
        user_prompt: str,
        schema: dict,
        system_prompt: str | None = None,
        file_parts: list | None = None,
    ) -> InspectionResult:
        """Run image inspection with structured output.

        Args:
            image: Raw image bytes (JPEG/PNG).
            user_prompt: Inspection prompt text.
            schema: response_json_schema dict (from schema.build_inspection_schema).
            system_prompt: Optional system instruction (from prompt.PromptBuilder.build).
            file_parts: Optional list of Part objects from Files API for knowledge injection.

        Returns:
            InspectionResult with raw_result dict and usage metadata.
        """
        self._check_configured()

        contents = []
        if file_parts:
            contents.extend(file_parts)
        contents.append(user_prompt)
        contents.append(image)

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema,
            system_instruction=system_prompt,
        )

        logger.info("Inspection request to %s", self._model)
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        usage = self._extract_usage(response)
        raw = json.loads(response.text) if response.text else {}
        logger.info("Inspection tokens: %s", usage)
        return InspectionResult(raw_result=raw, usage=usage)

    def generate_report(
        self,
        report_prompt: str,
        schema: dict,
        system_prompt: str | None = None,
        file_parts: list | None = None,
    ) -> ReportResult:
        """Generate a structured patrol report.

        Args:
            report_prompt: Report generation prompt with patrol data.
            schema: response_json_schema dict (from schema.build_report_schema).
            system_prompt: Optional system instruction.
            file_parts: Optional Files API parts for context.

        Returns:
            ReportResult with raw_result dict and usage metadata.
        """
        self._check_configured()

        contents = []
        if file_parts:
            contents.extend(file_parts)
        contents.append(report_prompt)

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema,
            system_instruction=system_prompt,
        )

        logger.info("Report request to %s", self._model)
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        usage = self._extract_usage(response)
        raw = json.loads(response.text) if response.text else {}
        logger.info("Report tokens: %s", usage)
        return ReportResult(raw_result=raw, usage=usage)

    def analyze_video(
        self,
        video_path: str,
        user_prompt: str,
        schema: dict | None = None,
        system_prompt: str | None = None,
        poll_interval: float = 2.0,
    ) -> VideoResult:
        """Upload and analyze a video file.

        Uploads the video to Gemini Files API, waits for processing,
        then runs analysis with optional structured output.

        Args:
            video_path: Local path to video file.
            user_prompt: Analysis prompt text.
            schema: Optional response_json_schema dict. If None, returns free-text.
            system_prompt: Optional system instruction.
            poll_interval: Seconds between processing status polls.

        Returns:
            VideoResult with raw_result dict and usage metadata.
        """
        self._check_configured()

        logger.info("Uploading video: %s", video_path)
        video_file = self._client.files.upload(file=video_path)

        while video_file.state.name == "PROCESSING":
            time.sleep(poll_interval)
            video_file = self._client.files.get(name=video_file.name)

        if video_file.state.name == "FAILED":
            raise RuntimeError(f"Video processing failed: {video_path}")

        logger.info("Video ready, analyzing")
        contents = [video_file, user_prompt]

        config_kwargs = {}
        if schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_json_schema"] = schema
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        usage = self._extract_usage(response)

        if schema and response.text:
            raw = json.loads(response.text)
        else:
            raw = {"is_ng": False, "analysis": response.text or ""}

        logger.info("Video tokens: %s", usage)
        return VideoResult(raw_result=raw, usage=usage)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_provider.py -v`
Expected: 6 passed

**Step 5: Commit**

```bash
git add vlm_service/provider.py tests/test_provider.py
git commit -m "feat: Gemini provider with image inspection, report, and video analysis"
```

---

### Task 5: Files API Management

**Files:**
- Create: `vlm-service/vlm_service/files.py`
- Test: `vlm-service/tests/test_files.py`

**Step 1: Write the failing test**

```python
# tests/test_files.py
import json
import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest
from vlm_service.files import FileManager


@pytest.fixture
def tmp_data_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def file_manager(tmp_data_dir):
    return FileManager(data_dir=tmp_data_dir)


class TestFileManagerUpload:

    def test_upload_stores_metadata(self, file_manager):
        with patch("vlm_service.files.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_gemini_file = MagicMock()
            mock_gemini_file.name = "files/abc123"
            mock_gemini_file.uri = "https://gemini.googleapis.com/files/abc123"
            mock_client.files.upload.return_value = mock_gemini_file

            record = file_manager.upload(
                file_bytes=b"test content",
                filename="test.pdf",
                mime_type="application/pdf",
                api_key="test-key",
            )

            assert record["filename"] == "test.pdf"
            assert record["gemini_name"] == "files/abc123"
            assert record["gemini_uri"] == "https://gemini.googleapis.com/files/abc123"
            assert record["enabled"] is True
            assert "id" in record
            assert "expires_at" in record

    def test_upload_saves_local_file(self, file_manager):
        with patch("vlm_service.files.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_file = MagicMock()
            mock_file.name = "files/x"
            mock_file.uri = "uri://x"
            mock_client.files.upload.return_value = mock_file

            record = file_manager.upload(b"hello", "test.txt", "text/plain", "key")
            assert os.path.exists(record["local_path"])
            with open(record["local_path"], "rb") as f:
                assert f.read() == b"hello"


class TestFileManagerList:

    def test_list_empty(self, file_manager):
        assert file_manager.list_files() == []

    def test_list_with_files(self, file_manager):
        with patch("vlm_service.files.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_file = MagicMock()
            mock_file.name = "files/a"
            mock_file.uri = "uri://a"
            mock_client.files.upload.return_value = mock_file

            file_manager.upload(b"data", "a.pdf", "application/pdf", "key")
            files = file_manager.list_files()
            assert len(files) == 1
            assert "expired" in files[0]


class TestFileManagerToggle:

    def test_toggle_disable(self, file_manager):
        with patch("vlm_service.files.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_file = MagicMock()
            mock_file.name = "files/a"
            mock_file.uri = "uri://a"
            mock_client.files.upload.return_value = mock_file

            record = file_manager.upload(b"data", "a.pdf", "application/pdf", "key")
            assert file_manager.toggle(record["id"], enabled=False) is True
            files = file_manager.list_files()
            assert files[0]["enabled"] is False

    def test_toggle_not_found(self, file_manager):
        assert file_manager.toggle("nonexistent", enabled=True) is False


class TestFileManagerGetEnabled:

    def test_get_enabled_filters_disabled(self, file_manager):
        with patch("vlm_service.files.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_file = MagicMock()
            mock_file.name = "files/a"
            mock_file.uri = "uri://a"
            mock_client.files.upload.return_value = mock_file

            record = file_manager.upload(b"data", "a.pdf", "application/pdf", "key")
            file_manager.toggle(record["id"], enabled=False)
            enabled = file_manager.get_enabled_files()
            assert len(enabled) == 0


class TestFileManagerDelete:

    def test_delete_removes_record(self, file_manager):
        with patch("vlm_service.files.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_file = MagicMock()
            mock_file.name = "files/a"
            mock_file.uri = "uri://a"
            mock_client.files.upload.return_value = mock_file

            record = file_manager.upload(b"data", "a.pdf", "application/pdf", "key")
            assert file_manager.delete(record["id"], api_key="key") is True
            assert file_manager.list_files() == []

    def test_delete_not_found(self, file_manager):
        assert file_manager.delete("nonexistent") is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_files.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vlm_service.files'`

**Step 3: Write minimal implementation**

```python
# vlm_service/files.py
"""Gemini Files API lifecycle manager.

Handles upload, metadata tracking, expiry detection, toggle, delete, and reupload.
Sync API — uses threading.Lock for metadata access, no asyncio.

Extracted from kachaka-gemini file_service.py (gemini-files-api branch).
Key change: sync instead of async, class-based instead of module-level globals.
"""

import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timedelta, timezone

from google import genai

logger = logging.getLogger("vlm_service.files")

EXPIRY_HOURS = 48


class FileManager:
    """Manage files uploaded to Gemini Files API with local metadata tracking."""

    def __init__(self, data_dir: str):
        """Initialize with a data directory for local file + metadata storage.

        Args:
            data_dir: Directory path for uploads/ and files.json.
        """
        self._data_dir = data_dir
        self._uploads_dir = os.path.join(data_dir, "uploads")
        self._meta_file = os.path.join(data_dir, "files.json")
        self._lock = threading.Lock()

    def _ensure_dirs(self):
        os.makedirs(self._uploads_dir, exist_ok=True)

    def _load_meta(self) -> list[dict]:
        if not os.path.exists(self._meta_file):
            return []
        with open(self._meta_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                return []
        return data if isinstance(data, list) else []

    def _save_meta(self, records: list[dict]):
        self._ensure_dirs()
        tmp = self._meta_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._meta_file)

    @staticmethod
    def _is_expired(record: dict) -> bool:
        expires_at = record.get("expires_at", "")
        if not expires_at:
            return True
        try:
            exp = datetime.fromisoformat(expires_at)
            return datetime.now(timezone.utc) >= exp
        except (ValueError, TypeError):
            return True

    def upload(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        api_key: str,
    ) -> dict:
        """Save locally, upload to Gemini, store metadata. Returns new record."""
        self._ensure_dirs()

        file_id = uuid.uuid4().hex[:12]
        base_name = os.path.basename(filename)
        safe_name = f"{file_id}_{re.sub(r'[^a-zA-Z0-9._-]', '_', base_name)}"
        local_path = os.path.join(self._uploads_dir, safe_name)

        with open(local_path, "wb") as f:
            f.write(file_bytes)

        client = genai.Client(api_key=api_key)
        try:
            gemini_file = client.files.upload(file=local_path)
        except Exception:
            if os.path.exists(local_path):
                os.remove(local_path)
            raise

        now = datetime.now(timezone.utc)
        record = {
            "id": file_id,
            "filename": filename,
            "mime_type": mime_type,
            "size_bytes": len(file_bytes),
            "local_path": local_path,
            "gemini_name": gemini_file.name,
            "gemini_uri": gemini_file.uri,
            "uploaded_at": now.isoformat(),
            "expires_at": (now + timedelta(hours=EXPIRY_HOURS)).isoformat(),
            "enabled": True,
        }

        with self._lock:
            records = self._load_meta()
            records.append(record)
            self._save_meta(records)

        logger.info("Uploaded %s -> %s", filename, gemini_file.name)
        return record

    def list_files(self) -> list[dict]:
        """Return all records with computed 'expired' field."""
        records = self._load_meta()
        for r in records:
            r["expired"] = self._is_expired(r)
        return records

    def toggle(self, file_id: str, enabled: bool) -> bool:
        """Toggle enabled flag. Returns True if found."""
        with self._lock:
            records = self._load_meta()
            found = False
            for r in records:
                if r["id"] == file_id:
                    r["enabled"] = enabled
                    found = True
                    break
            if found:
                self._save_meta(records)
        return found

    def delete(self, file_id: str, api_key: str = "") -> bool:
        """Delete from Gemini + local disk + metadata. Returns True if found."""
        records = self._load_meta()
        target = None
        for r in records:
            if r["id"] == file_id:
                target = r
                break
        if target is None:
            return False

        # Gemini-side deletion (best-effort)
        gemini_name = target.get("gemini_name", "")
        if gemini_name and api_key:
            try:
                client = genai.Client(api_key=api_key)
                client.files.delete(name=gemini_name)
            except Exception as exc:
                logger.warning("Gemini delete failed for %s: %s", gemini_name, exc)

        # Local file removal
        local_path = target.get("local_path", "")
        if local_path and os.path.exists(local_path):
            try:
                os.remove(local_path)
            except OSError as exc:
                logger.warning("Local delete failed for %s: %s", local_path, exc)

        with self._lock:
            records = self._load_meta()
            records = [r for r in records if r["id"] != file_id]
            self._save_meta(records)

        logger.info("Deleted file %s", file_id)
        return True

    def reupload(self, file_id: str, api_key: str) -> dict | None:
        """Re-upload a local file to Gemini (e.g. after expiry). Returns updated record."""
        records = self._load_meta()
        target = None
        for r in records:
            if r["id"] == file_id:
                target = r
                break
        if target is None:
            return None

        local_path = target.get("local_path", "")
        if not local_path or not os.path.exists(local_path):
            logger.warning("Local file missing for %s: %s", file_id, local_path)
            return None

        client = genai.Client(api_key=api_key)
        gemini_file = client.files.upload(file=local_path)

        now = datetime.now(timezone.utc)
        target["gemini_name"] = gemini_file.name
        target["gemini_uri"] = gemini_file.uri
        target["uploaded_at"] = now.isoformat()
        target["expires_at"] = (now + timedelta(hours=EXPIRY_HOURS)).isoformat()

        with self._lock:
            records = self._load_meta()
            for i, r in enumerate(records):
                if r["id"] == file_id:
                    records[i] = target
                    break
            self._save_meta(records)

        logger.info("Re-uploaded %s -> %s", file_id, gemini_file.name)
        return target

    def get_enabled_files(self) -> list[dict]:
        """Return Gemini refs (name, uri, mime_type) for enabled, non-expired files."""
        records = self._load_meta()
        result = []
        for r in records:
            if r.get("enabled") and not self._is_expired(r):
                name = r.get("gemini_name", "")
                uri = r.get("gemini_uri", "")
                if name and uri:
                    result.append({
                        "name": name,
                        "uri": uri,
                        "mime_type": r.get("mime_type", "application/octet-stream"),
                    })
        return result
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_files.py -v`
Expected: 8 passed

**Step 5: Commit**

```bash
git add vlm_service/files.py tests/test_files.py
git commit -m "feat: Files API lifecycle manager with upload, toggle, delete, reupload"
```

---

### Task 6: Public API + __init__ Exports

**Files:**
- Modify: `vlm-service/vlm_service/__init__.py`
- Create: `vlm-service/tests/test_init.py`

**Step 1: Write the failing test**

```python
# tests/test_init.py
"""Test that the public API is accessible from vlm_service."""


def test_imports():
    from vlm_service import GeminiProvider, PromptBuilder, FileManager
    from vlm_service import build_inspection_schema, build_report_schema
    from vlm_service import InspectionResult, ReportResult, VideoResult

    assert GeminiProvider is not None
    assert PromptBuilder is not None
    assert FileManager is not None
    assert build_inspection_schema is not None
    assert build_report_schema is not None
    assert InspectionResult is not None


def test_version():
    import vlm_service
    assert hasattr(vlm_service, "__version__")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_init.py -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
# vlm_service/__init__.py
"""vlm-service — Generic VLM image/video analysis + report generation."""

__version__ = "0.1.0"

from vlm_service.types import InspectionResult, ReportResult, VideoResult
from vlm_service.schema import build_inspection_schema, build_report_schema
from vlm_service.prompt import PromptBuilder
from vlm_service.provider import GeminiProvider
from vlm_service.files import FileManager

__all__ = [
    "GeminiProvider",
    "PromptBuilder",
    "FileManager",
    "build_inspection_schema",
    "build_report_schema",
    "InspectionResult",
    "ReportResult",
    "VideoResult",
]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_init.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add vlm_service/__init__.py tests/test_init.py
git commit -m "feat: public API exports from vlm_service package"
```

---

### Task 7: Integration Test — End-to-End Inspection Flow

**Files:**
- Create: `vlm-service/tests/test_integration.py`

**Step 1: Write the failing test**

```python
# tests/test_integration.py
"""Integration test: schema + prompt + provider wired together."""

import json
from unittest.mock import MagicMock, patch

from vlm_service import (
    GeminiProvider,
    PromptBuilder,
    build_inspection_schema,
    build_report_schema,
)


class TestInspectionFlow:
    """Full flow: venue config → schema → prompt → provider → result."""

    def _mock_response(self, result_dict, tokens=100):
        resp = MagicMock()
        resp.text = json.dumps(result_dict)
        resp.usage_metadata.prompt_token_count = tokens
        resp.usage_metadata.candidates_token_count = tokens // 2
        resp.usage_metadata.total_token_count = tokens + tokens // 2
        return resp

    def test_hospital_inspection(self, sample_venue_config):
        schema = build_inspection_schema(sample_venue_config)
        assert "hygiene" in schema["properties"]

        prompt = PromptBuilder.from_venue_config(
            venue_config=sample_venue_config,
            role="Hospital patrol inspector",
            instructions="Check hygiene and safety",
        )
        system_prompt = prompt.build(placeholders={"CURRENT_TIME": "2026-02-28 10:00"})
        assert "Hospital patrol inspector" in system_prompt

        with patch("vlm_service.provider.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            result_data = {
                "is_ng": True,
                "analysis": "Dirty floor in ward B",
                "hygiene": 3,
                "safety": 8,
                "categories": ["cleanliness"],
                "ward_id": "B-201",
            }
            mock_client.models.generate_content.return_value = self._mock_response(result_data)

            provider = GeminiProvider()
            provider.configure(api_key="test", model="gemini-2.0-flash")
            result = provider.generate_inspection(
                image=b"fake-image",
                user_prompt="Inspect this ward",
                schema=schema,
                system_prompt=system_prompt,
            )

            assert result.is_ng is True
            assert result.analysis == "Dirty floor in ward B"
            assert result.raw_result["hygiene"] == 3
            assert result.raw_result["ward_id"] == "B-201"
            assert result.total_tokens == 150

    def test_factory_inspection(self, sample_factory_config):
        schema = build_inspection_schema(sample_factory_config)
        assert "fire_risk" in schema["properties"]
        assert "hygiene" not in schema["properties"]

        prompt = PromptBuilder.from_venue_config(
            venue_config=sample_factory_config,
            role="Factory patrol inspector",
            instructions="Check fire prevention and PPE",
        )
        system_prompt = prompt.build()
        assert "en" in system_prompt

        with patch("vlm_service.provider.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            result_data = {
                "is_ng": False,
                "analysis": "All clear",
                "fire_risk": 2,
                "categories": ["fire_prevention"],
                "zone": "A",
                "line_number": 3,
            }
            mock_client.models.generate_content.return_value = self._mock_response(result_data)

            provider = GeminiProvider()
            provider.configure(api_key="test", model="gemini-2.0-flash")
            result = provider.generate_inspection(
                image=b"fake",
                user_prompt="Inspect zone A",
                schema=schema,
                system_prompt=system_prompt,
            )

            assert result.is_ng is False
            assert result.raw_result["fire_risk"] == 2
            assert result.raw_result["zone"] == "A"


class TestReportFlow:
    """Full flow: report config → schema → provider → result."""

    def test_structured_report(self):
        report_config = {
            "sections": ["overview", "findings", "recommendations"],
            "custom_sections": {
                "risk_assessment": {"type": "STRING", "description": "Risk summary"},
            },
        }
        schema = build_report_schema(report_config)
        assert "overview" in schema["properties"]
        assert "risk_assessment" in schema["properties"]

        with patch("vlm_service.provider.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            report_data = {
                "summary": "2 issues found",
                "details": "Ward B hygiene, Zone C fire risk",
                "overview": "Morning patrol completed",
                "findings": "Two NG locations",
                "recommendations": "Schedule deep cleaning",
                "risk_assessment": "Medium risk overall",
            }
            resp = MagicMock()
            resp.text = json.dumps(report_data)
            resp.usage_metadata.prompt_token_count = 500
            resp.usage_metadata.candidates_token_count = 300
            resp.usage_metadata.total_token_count = 800
            mock_client.models.generate_content.return_value = resp

            provider = GeminiProvider()
            provider.configure(api_key="test", model="gemini-2.0-flash")
            result = provider.generate_report(
                report_prompt="Generate patrol report from: ...",
                schema=schema,
            )

            assert result.raw_result["summary"] == "2 issues found"
            assert result.raw_result["risk_assessment"] == "Medium risk overall"
            assert result.total_tokens == 800
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_integration.py -v`
Expected: FAIL (if earlier tasks incomplete) or PASS (if all modules exist)

**Step 3: Run test to verify it passes**

Run: `pytest tests/test_integration.py -v`
Expected: 3 passed

**Step 4: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests passed (types: 5, schema: 8, prompt: 10, provider: 6, files: 8, init: 2, integration: 3 = ~42 total)

**Step 5: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end integration tests for inspection and report flows"
```

---

### Task 8: Final Verification + README

**Files:**
- Create: `vlm-service/README.md`
- Verify: all tests pass, package installs cleanly

**Step 1: Run full test suite**

Run: `cd vlm-service && pytest tests/ -v --tb=short`
Expected: All ~42 tests pass

**Step 2: Verify clean install**

Run: `pip install -e ".[dev]" && python -c "from vlm_service import GeminiProvider, build_inspection_schema; print('OK')"`
Expected: `OK`

**Step 3: Write README.md**

```markdown
# vlm-service

Generic VLM (Vision Language Model) image/video analysis and report generation service.

## Install

pip install vlm-service

## Quick Start

from vlm_service import GeminiProvider, PromptBuilder, build_inspection_schema

# 1. Build schema from venue config
schema = build_inspection_schema({
    "scores": {"hygiene": {"type": "integer", "description": "1-10"}},
    "categories": ["cleanliness", "safety"],
})

# 2. Build prompt
prompt = PromptBuilder()
prompt.add_section("Role", "Hospital patrol inspector")
prompt.add_section("Instructions", "Check hygiene conditions")
system_prompt = prompt.build()

# 3. Run inspection
provider = GeminiProvider()
provider.configure(api_key="YOUR_KEY", model="gemini-2.0-flash")
result = provider.generate_inspection(
    image=open("photo.jpg", "rb").read(),
    user_prompt="Inspect this area",
    schema=schema,
    system_prompt=system_prompt,
)

print(result.is_ng, result.analysis, result.raw_result)
```

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add README with quick start example"
```

**Step 5: Final full test run**

Run: `pytest tests/ -v`
Expected: All tests pass. Package is ready.

---

## File Structure Summary

```
vlm-service/
├── pyproject.toml
├── README.md
├── vlm_service/
│   ├── __init__.py         # Public API exports
│   ├── types.py            # InspectionResult, ReportResult, VideoResult (dataclasses)
│   ├── schema.py           # build_inspection_schema(), build_report_schema()
│   ├── prompt.py           # PromptBuilder (section composition + placeholders)
│   ├── provider.py         # GeminiProvider (inspection + report + video)
│   └── files.py            # FileManager (Gemini Files API lifecycle)
└── tests/
    ├── __init__.py
    ├── conftest.py          # Shared fixtures (venue configs)
    ├── test_types.py        # Result type tests
    ├── test_schema.py       # Schema builder tests
    ├── test_prompt.py       # Prompt builder tests
    ├── test_provider.py     # Provider tests (mocked Gemini)
    ├── test_files.py        # File manager tests (mocked Gemini)
    ├── test_init.py         # Import / export tests
    └── test_integration.py  # End-to-end flow tests
```
