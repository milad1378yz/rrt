from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from build_rollout_cache import build_parser as cache_parser
from build_rollout_cache import cache_summary
from core.cache import cache_rollouts, load_cache, save_cache
from data_prep.convert_rar_science import _build_parser as rar_parser
from data_prep.convert_rar_science import normalize_row as normalize_rar
from data_prep.convert_rubricbench import _build_parser as rubricbench_parser
from data_prep.convert_rubrichub import _build_parser as rubrichub_parser
from data_prep.convert_rubrichub import normalize_row as normalize_rubrichub
from data_prep.utils import default_dataset_dir, write_parquet
from evaluate import bootstrap_mean, parse_cache_specs


def test_paper_split_defaults():
    rubrichub = rubrichub_parser().parse_args([])
    rar = rar_parser().parse_args([])
    rubricbench = rubricbench_parser().parse_args([])
    assert (rubrichub.train, rubrichub.val, rubrichub.test, rubrichub.seed) == (5000, 500, 500, 42)
    assert (rar.train, rar.val, rar.test, rar.seed) == (5000, 500, 500, 42)
    assert (rubricbench.train, rubricbench.val, rubricbench.test, rubricbench.seed) == (
        847,
        150,
        150,
        42,
    )


def test_cache_defaults_match_paper_generation_settings():
    args = cache_parser().parse_args(
        ["--data-dir", "data", "--output", "cache", "--model", "model"]
    )
    assert args.rollouts == 8
    assert args.max_prompt_tokens == 6144
    assert args.max_response_tokens == 32768
    assert args.repetition_penalty == 1.1
    assert args.top_k == 0


def test_cache_summary_uses_signed_normalized_points():
    cache = {
        "train": [
            {
                "criteria": [
                    {"text": "Correct", "points": 2.0},
                    {"text": "Pitfall", "points": -1.0},
                ],
                "presence": [[True, True], [True, False]],
            }
        ]
    }
    assert cache_summary(cache)["train"]["mean_normalized_score"] == pytest.approx(5.0 / 6.0)


def test_rar_normalization_preserves_signed_pitfall_weights():
    row = pd.Series(
        {
            "question": "Why?",
            "reference_answer": "Because.",
            "question_source": "fixture",
            "_source_split": "train",
            "rubric": [
                {"description": "Explains the cause", "weight": 2},
                {"description": "Invents evidence", "weight": -1},
            ],
        }
    )
    normalized = normalize_rar(row, split="test", source_index=7)
    assert [item["points"] for item in normalized["reward_model"]["rubrics"]] == [2.0, -1.0]
    assert normalized["extra_info"]["split"] == "test"


def test_rubrichub_normalization_accepts_numpy_message_columns():
    row = pd.Series(
        {
            "prompt": np.array([{"role": "user", "content": "Question"}], dtype=object),
            "reward_model": {
                "ground_truth": "Answer",
                "rubrics": [{"criterion": "Correct", "points": 1}],
            },
            "data_source": "source",
            "ability": "science",
        }
    )
    normalized = normalize_rubrichub(
        row,
        data_source="rubrichub_science",
        domain="science",
        split="train",
        source_index=3,
    )
    assert normalized["prompt"] == [{"role": "user", "content": "Question"}]
    assert normalized["extra_info"]["user_prompt"] == "Question"


def test_atomic_parquet_write_respects_force(tmp_path: Path):
    destination = tmp_path / "split.parquet"
    write_parquet([{"value": 1}], destination)
    write_parquet([{"value": 2}], destination)
    assert pd.read_parquet(destination)["value"].tolist() == [1]
    write_parquet([{"value": 2}], destination, force=True)
    assert pd.read_parquet(destination)["value"].tolist() == [2]
    assert not list(tmp_path.glob(".*.tmp"))


def test_default_dataset_dir_respects_data_root(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert default_dataset_dir("science") == tmp_path / "datasets" / "science"


def test_cache_round_trip_and_rollout_conversion(tmp_path):
    cache = {
        "train": [
            {
                "prompt": "q",
                "criteria": [
                    {"text": "correct", "points": 1.0},
                    {"text": "pitfall", "points": -1.0},
                ],
                "presence": [[True, False], [False, True]],
            }
        ]
    }
    path = tmp_path / "rollout_cache.pkl"
    save_cache(cache, path)
    loaded = load_cache(tmp_path)
    rollouts = cache_rollouts(loaded["train"])
    np.testing.assert_array_equal(rollouts[0]["verdicts"], [1.0, 1.0])
    np.testing.assert_array_equal(rollouts[1]["verdicts"], [0.0, 0.0])


def test_evaluation_helpers_are_deterministic():
    assert parse_cache_specs(["base=a.pkl", "rrt=b.pkl"]) == {
        "base": "a.pkl",
        "rrt": "b.pkl",
    }
    first = bootstrap_mean([0.1, 0.2, 0.3], samples=100, seed=4)
    second = bootstrap_mean([0.1, 0.2, 0.3], samples=100, seed=4)
    assert first == second
