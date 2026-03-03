# vlm-service

Generic VLM (Vision Language Model) image/video analysis and report generation service.

## Install

```bash
pip install vlm-service
```

## Quick Start

```python
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

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Copyright 2026 Sigma Robotics
