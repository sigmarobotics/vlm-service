#!/usr/bin/env python3
"""Real integration test for vlm-service modules.

Usage:
    python scripts/real_test.py --module schema
    python scripts/real_test.py --module prompt
    python scripts/real_test.py --module inspect --api-key KEY
    python scripts/real_test.py --module video --api-key KEY
    python scripts/real_test.py --module report --api-key KEY
    python scripts/real_test.py --all --api-key KEY
"""

import argparse
import json
import sys
from datetime import datetime

from vlm_service import (
    GeminiProvider,
    PromptBuilder,
    build_inspection_schema,
    build_report_schema,
    FileManager,
)

# ── Test Data ──────────────────────────────────────────────────────────────

IMAGE_DIR = "/home/snaken/CodeBase/visual-patrol/data/robot-a/report/images/42_20260213_093558"
NG_PHOTO = f"{IMAGE_DIR}/__1_NG_ffb8d5f8-facf-4c3e-85a2-2c67396f9111.jpg"
OK_PHOTO = f"{IMAGE_DIR}/AED1_OK_889d98cf-4de1-43d0-9a4f-35047b4d3a98.jpg"
VIDEO_FILE = "/home/snaken/CodeBase/visual-patrol/data/robot-a/report/video/44_20260213_234753.mp4"

# ── Venue Configs ──────────────────────────────────────────────────────────

HOSPITAL_CONFIG = {
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

FACTORY_CONFIG = {
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

OFFICE_CONFIG = {
    "venue_type": "office",
    "scores": {
        "tidiness": {"type": "integer", "description": "Tidiness score 1-10", "minimum": 1, "maximum": 10},
    },
    "categories": ["desk_area", "meeting_room", "entrance"],
    "custom_fields": {},
    "language": "ja",
}

ALL_CONFIGS = {"hospital": HOSPITAL_CONFIG, "factory": FACTORY_CONFIG, "office": OFFICE_CONFIG}


# ── Helpers ────────────────────────────────────────────────────────────────

def header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def dump(obj, indent=2):
    print(json.dumps(obj, indent=indent, ensure_ascii=False))


# ── Module: schema ─────────────────────────────────────────────────────────

def test_schema():
    """Build and display inspection + report schemas for all venue configs."""
    header("SCHEMA TEST")

    for name, config in ALL_CONFIGS.items():
        print(f"--- Inspection schema: {name} ---")
        schema = build_inspection_schema(config)
        dump(schema)
        props = list(schema["properties"].keys())
        required = schema["required"]
        print(f"  Properties: {props}")
        print(f"  Required:   {required}")
        assert "is_ng" in props, "FAIL: is_ng missing"
        assert "analysis" in props, "FAIL: analysis missing"
        print(f"  ✓ Core fields present\n")

    print("--- Report schema (with sections) ---")
    report_config = {
        "sections": ["overview", "findings", "recommendations"],
        "custom_sections": {
            "risk_assessment": {"type": "STRING", "description": "Risk summary"},
        },
    }
    schema = build_report_schema(report_config)
    dump(schema)
    print(f"  Properties: {list(schema['properties'].keys())}")
    print(f"  ✓ Report schema OK\n")


# ── Module: prompt ─────────────────────────────────────────────────────────

def test_prompt():
    """Build and display prompts for all venue configs."""
    header("PROMPT TEST")

    for name, config in ALL_CONFIGS.items():
        print(f"--- Prompt: {name} ---")
        builder = PromptBuilder.from_venue_config(
            venue_config=config,
            role=f"You are a {name} patrol inspector.",
            instructions=f"Inspect this {name} location for safety and cleanliness issues.",
            knowledge=f"Standard {name} inspection protocol applies.",
        )
        prompt = builder.build(placeholders={
            "CURRENT_TIME": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        print(prompt)
        assert "[Role]" in prompt, "FAIL: [Role] section missing"
        assert "[Language]" in prompt, "FAIL: [Language] section missing"
        print(f"\n  ✓ Prompt assembled ({len(prompt)} chars)\n")


# ── Module: inspect ────────────────────────────────────────────────────────

def test_inspect(api_key: str):
    """Run real image inspection on NG + OK photos."""
    header("INSPECT TEST (real API)")

    schema = build_inspection_schema(HOSPITAL_CONFIG)
    builder = PromptBuilder.from_venue_config(
        venue_config=HOSPITAL_CONFIG,
        role="You are a hospital patrol inspector.",
        instructions="Analyze this photo. Determine if there are any abnormal conditions (NG) or if everything is normal (OK). Score hygiene and safety 1-10. Identify applicable categories.",
    )
    system_prompt = builder.build(placeholders={
        "CURRENT_TIME": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    provider = GeminiProvider()
    provider.configure(api_key=api_key, model="gemini-2.5-flash")

    for label, photo_path in [("NG", NG_PHOTO), ("OK", OK_PHOTO)]:
        print(f"--- Inspecting: {label} photo ---")
        print(f"  File: {photo_path}")

        with open(photo_path, "rb") as f:
            image_bytes = f.read()

        result = provider.generate_inspection(
            image=image_bytes,
            user_prompt="請檢查這個區域。",
            schema=schema,
            system_prompt=system_prompt,
        )

        print(f"  is_ng:    {result.is_ng}")
        print(f"  analysis: {result.analysis}")
        print(f"  tokens:   prompt={result.prompt_tokens}, output={result.output_tokens}, total={result.total_tokens}")
        print(f"  raw:")
        dump(result.raw_result)

        expected_ng = (label == "NG")
        match = "✓ MATCH" if result.is_ng == expected_ng else "✗ MISMATCH"
        print(f"  Filename says {label}, model says {'NG' if result.is_ng else 'OK'} → {match}\n")


# ── Module: video ──────────────────────────────────────────────────────────

def test_video(api_key: str):
    """Upload and analyze a real patrol video."""
    header("VIDEO TEST (real API)")

    schema = build_inspection_schema(FACTORY_CONFIG)

    provider = GeminiProvider()
    provider.configure(api_key=api_key, model="gemini-2.5-flash")

    print(f"  File: {VIDEO_FILE}")
    print(f"  Uploading and analyzing (this may take a minute)...")

    result = provider.analyze_video(
        video_path=VIDEO_FILE,
        user_prompt="Analyze this patrol video. Report any abnormal conditions found.",
        schema=schema,
        system_prompt="You are a factory patrol inspector reviewing security camera footage.",
    )

    print(f"  is_ng:    {result.is_ng}")
    print(f"  analysis: {result.analysis}")
    print(f"  tokens:   prompt={result.prompt_tokens}, output={result.output_tokens}, total={result.total_tokens}")
    print(f"  raw:")
    dump(result.raw_result)
    print(f"  ✓ Video analysis complete\n")


# ── Module: report ─────────────────────────────────────────────────────────

def test_report(api_key: str):
    """Generate a structured report from mock inspection data."""
    header("REPORT TEST (real API)")

    report_config = {
        "sections": ["overview", "findings", "recommendations"],
        "custom_sections": {
            "risk_assessment": {"type": "STRING", "description": "Overall risk assessment"},
        },
    }
    schema = build_report_schema(report_config)

    provider = GeminiProvider()
    provider.configure(api_key=api_key, model="gemini-2.5-flash")

    report_prompt = """Generate a patrol inspection report based on the following data:

Patrol Date: 2026-02-13
Location: Hospital Building A
Inspector: Robot Patrol Unit #1

Inspection Results:
1. Ward B-201: NG — Dirty floor, hygiene=3, safety=8
2. AED Station 1: OK — Equipment present, hygiene=9, safety=10
3. Corridor 3F: NG — Obstruction in fire exit, hygiene=7, safety=3
4. Entrance Lobby: OK — Clean and clear, hygiene=9, safety=9

Please write the report in Traditional Chinese (zh-TW)."""

    result = provider.generate_report(
        report_prompt=report_prompt,
        schema=schema,
        system_prompt="You are a hospital patrol report writer. Generate comprehensive reports in the requested language.",
    )

    print(f"  tokens: prompt={result.prompt_tokens}, output={result.output_tokens}, total={result.total_tokens}")
    print(f"  raw:")
    dump(result.raw_result)
    print(f"  ✓ Report generated\n")


# ── Main ───────────────────────────────────────────────────────────────────

MODULES = {
    "schema": (test_schema, False),
    "prompt": (test_prompt, False),
    "inspect": (test_inspect, True),
    "video": (test_video, True),
    "report": (test_report, True),
}


def main():
    parser = argparse.ArgumentParser(description="Real integration test for vlm-service")
    parser.add_argument("--module", choices=list(MODULES.keys()), help="Module to test")
    parser.add_argument("--all", action="store_true", help="Run all modules")
    parser.add_argument("--api-key", help="Gemini API key (required for inspect/video/report)")
    args = parser.parse_args()

    if not args.module and not args.all:
        parser.print_help()
        sys.exit(1)

    modules_to_run = list(MODULES.keys()) if args.all else [args.module]

    for name in modules_to_run:
        func, needs_key = MODULES[name]
        if needs_key:
            if not args.api_key:
                print(f"ERROR: --api-key required for module '{name}'")
                sys.exit(1)
            func(args.api_key)
        else:
            func()

    header("ALL DONE")


if __name__ == "__main__":
    main()
