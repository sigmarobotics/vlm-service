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
