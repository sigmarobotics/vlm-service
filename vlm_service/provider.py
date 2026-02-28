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
        self._model: str = "gemini-2.5-flash"

    def configure(self, api_key: str, model: str = "gemini-2.5-flash"):
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
        contents.append(types.Part.from_bytes(data=image, mime_type="image/jpeg"))

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
