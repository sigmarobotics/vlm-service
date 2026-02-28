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
