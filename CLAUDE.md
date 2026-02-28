# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`vlm-service` is a pip-installable Python package that wraps Google Gemini SDK for image/video analysis and structured report generation. It is extracted from existing patrol inspection projects (visual-patrol, kachaka-gemini) into a generic, reusable library.

## Environment

- Use `python3` (not `python`) — no `python` alias on this system
- GitHub: `Sigma-Snaken/vlm-service` (private), default branch: `main`

## Build & Development Commands

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run all tests
python3 -m pytest -v

# Run a single test file
python3 -m pytest tests/test_schema.py -v

# Run a single test class/method
python3 -m pytest tests/test_provider.py::TestGeminiProviderInspection::test_generate_inspection_basic -v
```

## Real Integration Tests

`scripts/real_test.py` — CLI for testing with real Gemini API calls and patrol photos/videos.
```bash
python3 scripts/real_test.py --module schema          # no API key needed
python3 scripts/real_test.py --module inspect --api-key KEY
python3 scripts/real_test.py --all --api-key KEY
```
Test data lives in sibling project: `/home/snaken/CodeBase/visual-patrol/data/robot-a/report/`

## Architecture

The package is a thin, sync-only wrapper around `google-genai`. All modules are independent and compose together at the caller level.

```
vlm_service/
├── types.py      # Dataclass result containers (InspectionResult, VideoResult, ReportResult)
├── schema.py     # Runtime JSON schema builder from venue config dicts
├── prompt.py     # Section-based prompt composer with placeholder injection
├── provider.py   # GeminiProvider — image inspection, video analysis, report generation
└── files.py      # FileManager — Gemini Files API lifecycle (upload/list/toggle/delete/reupload)
```

**Data flow:** Caller builds a schema dict (`schema.py`) + prompt string (`prompt.py`) → passes both to `GeminiProvider` methods (`provider.py`) → gets back typed result containers (`types.py`). `FileManager` (`files.py`) handles knowledge-base document lifecycle separately and produces `Part` objects that plug into provider calls via `file_parts`.

## Key Design Decisions

- **`response_json_schema` (dict), NOT `response_schema` (Pydantic)** — enables runtime-dynamic schemas from venue config dicts. No hardcoded Pydantic models.
- **Fixed core schema:** `is_ng` + `analysis` only. Dynamic fields (scores, categories, custom fields) are added by `build_inspection_schema()` from venue config.
- **Sync API throughout** — no asyncio. FileManager uses `threading.Lock` for concurrency safety.
- **Gemini only** — no VILA or other providers.
- **No JSON repair** — structured output guarantees valid JSON. Only thin regex fallback if needed.
- **Gemini uppercase type names** — schema types use `STRING`, `BOOLEAN`, `INTEGER` etc. (Gemini's format), mapped from standard JSON Schema lowercase via `_TYPE_MAP`.

## Testing Conventions

- Tests use `unittest.mock` to patch `google.genai` — no real API calls.
- Shared fixtures in `tests/conftest.py`: `dummy_api_key`, `sample_venue_config`, `sample_factory_config`.
- Provider tests mock `genai.Client` and verify call arguments (contents, config, system_instruction).
- FileManager tests use `tmp_path` for local file storage.

## Gotchas

- **Image bytes must be wrapped**: `types.Part.from_bytes(data=image, mime_type="image/jpeg")` — SDK rejects raw `bytes` in contents list. Mock tests won't catch this.
- **Model deprecation**: `gemini-2.0-flash` is deprecated. Default model is `gemini-2.5-flash`.

## Implementation Plan

The full implementation plan is in `docs/plans/2026-02-28-vlm-service.md`. It uses TDD (test-first) with 7 sequential tasks. Follow it with `superpowers:executing-plans`.
