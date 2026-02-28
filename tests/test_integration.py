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
