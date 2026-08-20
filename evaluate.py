"""Paired held-out evaluation of judged policy caches."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.cache import extract_rubrics, load_cache, prompt_text
from core.judge import criterion_score, normalized_score


def bootstrap_mean(values, *, samples=10000, seed=0):
    """Mean and percentile bootstrap confidence interval."""

    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def parse_cache_specs(specs):
    caches = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"cache must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        if not name or name in caches:
            raise ValueError(f"invalid or duplicate cache name {name!r}")
        caches[name] = path
    return caches


def identity_from_row(row):
    return (
        prompt_text(row),
        tuple((rubric["criterion"], rubric["points"]) for rubric in extract_rubrics(row)),
    )


def identity_from_cache(example):
    return (
        example["prompt"],
        tuple((criterion["text"], criterion["points"]) for criterion in example["criteria"]),
    )


def score_example(example, rollouts):
    rubrics = [
        {"criterion": criterion["text"], "points": criterion["points"]}
        for criterion in example["criteria"]
    ]
    presence = [row for row in example["presence"] if row is not None][:rollouts]
    if len(presence) != rollouts:
        return None
    return {
        "criterion_score": float(np.mean([criterion_score(rubrics, row) for row in presence])),
        "normalized_score": float(np.mean([normalized_score(rubrics, row) for row in presence])),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Held-out test parquet.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--reference", required=True, help="Name of the comparison reference.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("caches", nargs="+", help="Judged caches as NAME=PATH.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.rollouts <= 0 or args.bootstrap_samples <= 0:
        raise SystemExit("rollouts and bootstrap samples must be positive")
    cache_paths = parse_cache_specs(args.caches)
    if args.reference not in cache_paths:
        raise KeyError(f"reference {args.reference!r} is not one of {list(cache_paths)}")

    frame = pd.read_parquet(args.data)
    rows = [frame.iloc[index].to_dict() for index in range(len(frame))]
    expected_identities = [identity_from_row(row) for row in rows]
    caches = {}
    for name, path in cache_paths.items():
        cache = load_cache(path)
        if args.split not in cache:
            raise KeyError(f"{name} cache has no {args.split!r} split")
        examples = cache[args.split]
        if len(examples) != len(rows):
            raise ValueError(f"{name} has {len(examples)} prompts; expected {len(rows)}")
        if [identity_from_cache(example) for example in examples] != expected_identities:
            raise ValueError(f"{name} cache does not match the held-out parquet order")
        caches[name] = examples

    per_prompt = []
    metric_values = {
        metric: {name: [] for name in caches} for metric in ("criterion_score", "normalized_score")
    }
    for index in range(len(rows)):
        scores = {
            name: score_example(examples[index], args.rollouts) for name, examples in caches.items()
        }
        if any(score is None for score in scores.values()):
            continue
        record = {"row_index": index}
        for name, values in scores.items():
            for metric, value in values.items():
                metric_values[metric][name].append(value)
                record[f"{name}_{metric}"] = value
        per_prompt.append(record)
    if not per_prompt:
        raise RuntimeError("no prompt has complete verdicts across every cache")

    report = {
        "split": args.split,
        "rollouts_per_prompt": args.rollouts,
        "evaluated_prompts": len(per_prompt),
        "reference": args.reference,
        "means": {
            metric: {name: float(np.mean(values)) for name, values in by_name.items()}
            for metric, by_name in metric_values.items()
        },
        "paired_differences": {},
    }
    for name in caches:
        if name == args.reference:
            continue
        report["paired_differences"][name] = {}
        for metric, by_name in metric_values.items():
            difference = np.asarray(by_name[name]) - np.asarray(by_name[args.reference])
            report["paired_differences"][name][metric] = bootstrap_mean(
                difference,
                samples=args.bootstrap_samples,
                seed=args.seed,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    per_prompt_path = output.with_suffix(".per_prompt.parquet")
    pd.DataFrame(per_prompt).to_parquet(per_prompt_path, index=False)
    report["per_prompt"] = str(per_prompt_path)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
