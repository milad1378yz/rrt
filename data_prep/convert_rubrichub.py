"""Create the paper's fixed RubricHub train/validation/test splits.

Medical and Science each use 5,000 training prompts, 500 validation
prompts, and 500 test prompts sampled without replacement with seed 42.

Example::

    python -m data_prep.convert_rubrichub --domain Science
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from core.cache import coerce_messages
from data_prep.utils import default_dataset_dir, write_parquet

logger = logging.getLogger(__name__)

REPO_ID = "sojuL/RubricHub_v1"
DOMAINS = ("Medical", "Science")


def rubrichub_fields(domain: str) -> tuple[str, str, str]:
    """Return the source path, output slug, and training data-source name."""
    slug = domain.lower()
    return f"RuRL/rurbichub_v1_{domain}.parquet", slug, f"rubrichub_{slug}"


def download_rubrichub(domain: str, cache_dir=None) -> str:
    """Download one RubricHub domain parquet and return its local path."""
    return hf_hub_download(
        REPO_ID,
        rubrichub_fields(domain)[0],
        repo_type="dataset",
        cache_dir=cache_dir,
    )


def normalize_row(row, *, data_source: str, domain: str, split: str, source_index: int) -> dict:
    """Convert a RubricHub row to the training and judge format."""
    prompt = coerce_messages(row["prompt"])
    reward_model = dict(row["reward_model"])
    user_prompt = "\n\n".join(
        turn["content"]
        for turn in prompt
        if isinstance(turn, dict)
        and turn.get("role") == "user"
        and isinstance(turn.get("content"), str)
        and turn["content"]
    )
    messages = [
        {"role": str(turn.get("role", "")), "content": str(turn.get("content", ""))}
        for turn in prompt
        if isinstance(turn, dict)
    ]
    extra_info = {
        "domain": domain,
        "split": split,
        "source_index": source_index,
        "source_data_source": row.get("data_source", ""),
        "rubrics_full": reward_model.get("rubrics", []),
        "user_prompt": user_prompt,
        "prompt_messages": messages,
    }
    return {
        "prompt": prompt,
        "data_source": data_source,
        "ability": row.get("ability", domain),
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


def build_subset(args) -> None:
    _, domain, data_source = rubrichub_fields(args.domain)
    out_dir = (
        Path(args.out_dir).expanduser()
        if args.out_dir
        else default_dataset_dir(f"rubrichub_{domain}_irt")
    )
    sizes = {"train": args.train, "val": args.val, "test": args.test}
    destinations = {split: out_dir / f"{split}.parquet" for split in sizes}
    if not args.force and all(path.exists() for path in destinations.values()):
        logger.info("[skip] paper split already exists in %s", out_dir)
        return

    frame = pd.read_parquet(download_rubrichub(args.domain, args.cache_dir))
    required = sum(sizes.values())
    if len(frame) < required:
        raise SystemExit(f"{args.domain} has {len(frame)} rows; {required} are required")

    permutation = np.random.default_rng(args.seed).permutation(len(frame))[:required]
    start = 0
    for split, size in sizes.items():
        indices = permutation[start : start + size]
        rows = [
            normalize_row(
                frame.iloc[index],
                data_source=data_source,
                domain=domain,
                split=split,
                source_index=int(index),
            )
            for index in indices
        ]
        write_parquet(rows, destinations[split], force=args.force)
        start += size
    logger.info("[done] %s -> %s", args.domain, out_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=DOMAINS, default="Science")
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
