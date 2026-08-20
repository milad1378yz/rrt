"""Create the paper's fixed RaR Science train/validation/test split.

The released train and validation shards form one source pool. The converter
samples 6,000 prompts without replacement with seed 42, then writes 5,000
training, 500 validation, and 500 test prompts.

Example::

    python -m data_prep.convert_rar_science
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from data_prep.utils import default_dataset_dir, write_parquet

logger = logging.getLogger(__name__)

REPO_ID = "anisha2102/RaR-Science"
DATA_SOURCE = "rar_science"


def download_split(split: str, cache_dir=None) -> str:
    """Download one released RaR Science parquet shard."""
    return hf_hub_download(
        REPO_ID,
        f"data/{split}-00000-of-00001.parquet",
        repo_type="dataset",
        cache_dir=cache_dir,
    )


def _rubrics(row) -> list[dict]:
    """Convert RaR criteria while preserving signed pitfall weights."""
    source = row["rubric"] if row["rubric"] is not None else []
    return [
        {"criterion": str(item["description"]).strip(), "points": float(item["weight"])}
        for item in source
        if isinstance(item, dict) and item.get("description")
    ]


def normalize_row(row, *, split: str, source_index: int) -> dict:
    """Convert a RaR Science row to the training and judge format."""
    question = str(row["question"])
    prompt = [{"role": "user", "content": question}]
    rubrics = _rubrics(row)
    reward_model = {
        "ground_truth": str(row.get("reference_answer", "") or ""),
        "rubrics": rubrics,
        "style": "rubric",
    }
    extra_info = {
        "domain": DATA_SOURCE,
        "split": split,
        "source_index": source_index,
        "source_data_source": str(row.get("question_source", "") or ""),
        "source_split": str(row["_source_split"]),
        "rubrics_full": rubrics,
        "user_prompt": question,
        "prompt_messages": prompt,
    }
    return {
        "prompt": prompt,
        "data_source": DATA_SOURCE,
        "ability": DATA_SOURCE,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


def _source_pool(cache_dir=None) -> pd.DataFrame:
    frames = []
    for split in ("train", "val"):
        frame = pd.read_parquet(download_split(split, cache_dir)).copy()
        frame["_source_split"] = split
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_subset(args) -> None:
    out_dir = (
        Path(args.out_dir).expanduser()
        if args.out_dir
        else default_dataset_dir("rar_science_irt")
    )
    sizes = {"train": args.train, "val": args.val, "test": args.test}
    destinations = {split: out_dir / f"{split}.parquet" for split in sizes}
    if not args.force and all(path.exists() for path in destinations.values()):
        logger.info("[skip] paper split already exists in %s", out_dir)
        return

    frame = _source_pool(args.cache_dir)
    required = sum(sizes.values())
    if len(frame) < required:
        raise SystemExit(f"RaR Science has {len(frame)} rows; {required} are required")

    permutation = np.random.default_rng(args.seed).permutation(len(frame))[:required]
    start = 0
    for split, size in sizes.items():
        indices = permutation[start : start + size]
        rows = [
            normalize_row(frame.iloc[index], split=split, source_index=int(index))
            for index in indices
        ]
        write_parquet(rows, destinations[split], force=args.force)
        start += size
    logger.info("[done] RaR Science -> %s", out_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--val", type=int, default=500)
    parser.add_argument("--test", type=int, default=500)
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
