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
