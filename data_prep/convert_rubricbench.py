"""Create the paper's fixed RubricBench train/validation/test split.

All 1,147 released prompts are shuffled with seed 42 and assigned to 847 training, 150 validation, and 150 test prompts.

Example::

    python -m data_prep.convert_rubricbench
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from data_prep.utils import default_dataset_dir, write_parquet

logger = logging.getLogger(__name__)

REPO_ID = "DonJoey/rubricbench"
DATA_SOURCE = "rubricbench"


def download_rubricbench(cache_dir=None) -> pd.DataFrame:
    """Download and concatenate all released RubricBench parquet shards."""
    files = sorted(
        filename
        for filename in HfApi().list_repo_files(REPO_ID, repo_type="dataset")
        if filename.startswith("data/") and filename.endswith(".parquet")
    )
    if not files:
        raise RuntimeError(f"no parquet files found in {REPO_ID}")
    return pd.concat(
        [
            pd.read_parquet(
                hf_hub_download(REPO_ID, filename, repo_type="dataset", cache_dir=cache_dir)
            )
            for filename in files
        ],
        ignore_index=True,
    )


def normalize_row(row, *, split: str, source_index: int) -> dict:
    """Convert a RubricBench row to the training and judge format."""
    instruction = str(row["instruction"])
    raw_rubrics = row["rubrics"]
    lines = (
        raw_rubrics
        if isinstance(raw_rubrics, (list, tuple, np.ndarray))
        else str(raw_rubrics).splitlines()
    )
    rubrics = [
        {"criterion": str(line).strip(), "points": 1.0}
        for line in lines
        if str(line).strip()
    ]
    messages = [{"role": "user", "content": instruction}]
    domain = str(row.get("domain", "") or "")
    return {
        "prompt": messages,
        "data_source": DATA_SOURCE,
        "ability": domain,
        "reward_model": {"ground_truth": "", "rubrics": rubrics, "style": "rubric"},
        "extra_info": {
            "domain": domain,
            "split": split,
            "source_index": source_index,
            "source_data_source": str(row.get("source", "") or ""),
            "case_id": str(row.get("case_id", "") or ""),
            "rubrics_full": rubrics,
            "user_prompt": instruction,
            "prompt_messages": messages,
        },
    }


def build_subset(args) -> None:
    out_dir = (
        Path(args.out_dir).expanduser()
        if args.out_dir
        else default_dataset_dir("rubricbench_irt")
    )
    sizes = {"train": args.train, "val": args.val, "test": args.test}
    destinations = {split: out_dir / f"{split}.parquet" for split in sizes}
    if not args.force and all(path.exists() for path in destinations.values()):
        logger.info("[skip] paper split already exists in %s", out_dir)
        return

    frame = download_rubricbench(args.cache_dir)
    required = sum(sizes.values())
    if len(frame) != required:
        raise SystemExit(
            f"RubricBench has {len(frame)} rows, but the requested splits total {required}"
        )

    permutation = np.random.default_rng(args.seed).permutation(len(frame))
    start = 0
    for split, size in sizes.items():
        indices = permutation[start : start + size]
        rows = [
            normalize_row(frame.iloc[index], split=split, source_index=int(index))
            for index in indices
        ]
        write_parquet(rows, destinations[split], force=args.force)
        start += size
    logger.info("[done] RubricBench -> %s", out_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--train", type=int, default=847)
    parser.add_argument("--val", type=int, default=150)
    parser.add_argument("--test", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    if min(args.train, args.val, args.test) <= 0:
        raise SystemExit("split sizes must be positive")
    build_subset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
