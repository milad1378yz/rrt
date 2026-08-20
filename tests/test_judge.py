from types import SimpleNamespace

import pytest

from core.judge import (
    JudgeConfig,
    RubricJudge,
    criterion_score,
    normalized_score,
    parse_verdict,
    strip_reasoning,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("PRESENT", True), ("NOT_PRESENT", False), ("not present", False), ("maybe", None)],
)
def test_single_criterion_parser(raw, expected):
    assert parse_verdict(raw) is expected


def test_signed_scores_penalize_present_pitfalls():
    rubrics = [
        {"criterion": "Correct", "points": 2.0},
        {"criterion": "Commits pitfall", "points": -1.0},
    ]
    assert normalized_score(rubrics, [True, False]) == pytest.approx(1.0)
    assert normalized_score(rubrics, [True, True]) == pytest.approx(2.0 / 3.0)
    assert criterion_score(rubrics, [True, False]) == pytest.approx(1.0)


def test_reasoning_prefix_is_removed_before_judging():
    assert strip_reasoning("hidden reasoning</think>Final answer") == "Final answer"
    assert strip_reasoning("Final answer") == "Final answer"


class FakeResponses:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.prompts = []

    def create(self, **kwargs):
        self.prompts.append(kwargs["input"])
        return SimpleNamespace(output_text=next(self.outputs))


def test_judge_uses_pitfall_prompt_for_negative_points():
    responses = FakeResponses(["NOT_PRESENT"])
    client = SimpleNamespace(responses=responses)
    judge = RubricJudge(JudgeConfig(retries=1), client=client)
    verdict = judge.grade_criterion(
        "question",
        "answer",
        {"criterion": "Invents evidence", "points": -1.0},
    )
    assert verdict is False
    assert "pitfall" in responses.prompts[0].lower()


def test_judge_retries_unparseable_content():
    responses = FakeResponses(["unclear", "PRESENT"])
    client = SimpleNamespace(responses=responses)
    judge = RubricJudge(JudgeConfig(retries=2), client=client)
    assert judge.grade_criterion(
        "question", "answer", {"criterion": "Correct", "points": 1.0}
    )
