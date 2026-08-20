"""Shared dataset and judged-rollout cache helpers."""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.rrt import oriented_labels


def coerce_messages(value: Any) -> list[dict]:
    """Normalize a parquet message column into OpenAI-style dictionaries."""

    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return []
    messages = []
    for item in value:
        if hasattr(item, "as_py"):
            item = item.as_py()
        if not isinstance(item, dict):
            continue
        messages.append(
            {
                "role": str(item.get("role", "")),
                "content": str(item.get("content", "")),
            }
        )
    return messages


def extract_rubrics(row) -> list[dict]:
    """Read normalized ``{criterion, points}`` entries from one dataset row."""

    reward_model = dict(row.get("reward_model") or {})
    extra_info = dict(row.get("extra_info") or {})
    raw = reward_model.get("rubrics") or extra_info.get("rubrics_full") or []
    rubrics = []
    for item in raw:
        if not isinstance(item, dict) or not str(item.get("criterion", "")).strip():
            continue
        rubrics.append(
            {
                "criterion": str(item["criterion"]).strip(),
                "points": float(item.get("points", 0.0)),
            }
        )
    return rubrics


def prompt_text(row) -> str:
    """Return the human-readable prompt recorded for judging and the RPN."""

    extra_info = dict(row.get("extra_info") or {})
    if extra_info.get("user_prompt"):
        return str(extra_info["user_prompt"])
    messages = coerce_messages(extra_info.get("prompt_messages") or row.get("prompt"))
    return "\n".join(
        f"{message['role']}: {message['content']}"
        for message in messages
        if message["role"] != "system"
    )


def load_splits(data_dir, splits=("train", "val"), limit=0):
    """Load parquet splits as row dictionaries."""

    rows = {}
    for split in splits:
        path = Path(data_dir) / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        if limit:
            frame = frame.iloc[:limit]
        rows[split] = [frame.iloc[index].to_dict() for index in range(len(frame))]
    return rows


def judged_example(row, presence):
    """Build the compact cache record shared by fitting and evaluation."""

    return {
        "prompt": prompt_text(row),
        "criteria": [
            {"text": item["criterion"], "points": item["points"]} for item in extract_rubrics(row)
        ],
        "presence": presence,
    }


def save_cache(cache, path):
    """Write a complete judged cache."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL))


def load_cache(path):
    """Load ``{split: [judged prompt, ...]}`` from a cache file or directory."""

    path = Path(path)
    if path.is_dir():
        path = path / "rollout_cache.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle)


def cache_rollouts(examples, *, max_per_prompt=0):
    """Convert judged prompt records into flat RPN training rollouts."""

    rollouts = []
    for example in examples:
        criteria = example.get("criteria") or []
        if not criteria:
            continue
        points = [criterion["points"] for criterion in criteria]
        presence = [
            row
            for row in (example.get("presence") or [])
            if row is not None and len(row) == len(criteria)
        ]
        if max_per_prompt:
            presence = presence[:max_per_prompt]
        if not presence:
            continue
        favorable, _ = oriented_labels(presence, points)
        for verdicts in favorable:
            rollouts.append(
                {
                    "prompt": example["prompt"],
                    "criteria": [criterion["text"] for criterion in criteria],
                    "verdicts": verdicts,
                }
            )
    return rollouts
