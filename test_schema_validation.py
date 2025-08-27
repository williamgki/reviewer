import pytest
from validators import validate_review_card


def test_missing_quotes_fails():
    data = {
        "paper_id": "P1",
        "tl_dr": "test",
        "verdict": "reject",
        "strengths": [],
        "weaknesses": [],
        "key_claims": [
            {
                "id": "C1",
                "statement": "claim",
                "importance": "low",
                "confidence": 0.5,
                "evidence": {
                    "paper_quotes": [],
                    "corpus_quotes": [],
                    "coverage_score": 0.0,
                },
                "risks": [],
            }
        ],
        "suggested_experiments": [],
        "related_work_gaps": [],
        "open_questions": [],
        "confidence_calibration": "",
    }
    with pytest.raises(Exception):
        validate_review_card(data)
