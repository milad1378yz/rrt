"""Small helpers shared by the dataset converters."""

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_dataset_dir(name: str) -> Path:
    """Return ``$DATA_ROOT/datasets/<name>`` (or the repository's ``data/``)."""

    data_root = Path(os.environ.get("DATA_ROOT") or PROJECT_ROOT / "data")
    return data_root / "datasets" / name


def write_parquet(rows, destination, *, force: bool = False) -> None:
    """Write rows atomically, leaving an existing file untouched by default."""
    destination = Path(destination)
    if destination.exists() and not force:
        logger.info("[skip] %s already exists (use --force to overwrite)", destination)
        return

    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    logger.info("[write] %s: %d rows", destination, len(frame))
