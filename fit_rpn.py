"""Warm-start the paper's prompt-conditioned two-parameter RPN."""

import argparse
import json
import logging
from pathlib import Path

from scipy.special import log_ndtr
import torch

from core.cache import cache_rollouts, load_cache
from core.rpn import DEFAULT_EMBED_MODEL, ResponseParameterNetwork
from core.rrt import estimate_quality, flatten_rollouts, partial_m_step

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_nll(model, rollouts, batch_size=64):
    """Evaluate held-out verdict likelihood using MAP rollout qualities."""

    if not rollouts:
        return float("nan")
    model.eval()
    loss_sum = 0.0
    observations = 0
    for start in range(0, len(rollouts), batch_size):
        batch = rollouts[start : start + batch_size]
        criteria, prompts, verdicts, groups = flatten_rollouts(batch)
        a, b = model.parameters_for_text(criteria, prompts)
        a_np = a.cpu().numpy()
        b_np = b.cpu().numpy()
        quality = estimate_quality(
            verdicts,
            a_np,
            b_np,
            groups,
            len(batch),
            model.rrt_config,
        )
        logits = a_np * (quality[groups] - b_np)
        loss_sum += float(
            -(verdicts * log_ndtr(logits) + (1.0 - verdicts) * log_ndtr(-logits)).sum()
        )
        observations += len(verdicts)
    return loss_sum / observations


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="rollout_cache.pkl or its directory.")
    parser.add_argument("--output", required=True, help="Directory for rpn.pt and metrics.json.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--lambda-a", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--train-rollouts-per-prompt", type=int, default=0)
    return parser


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if min(args.epochs, args.batch_size, args.accumulation_steps) <= 0:
        raise SystemExit("epochs, batch size, and accumulation steps must be positive")

    cache = load_cache(args.cache)
    if "train" not in cache or "val" not in cache:
        raise KeyError("the RPN cache must contain train and val splits")
    train_rollouts = cache_rollouts(cache["train"], max_per_prompt=args.train_rollouts_per_prompt)
    val_rollouts = cache_rollouts(cache["val"])
    if not train_rollouts or not val_rollouts:
        raise ValueError("the cache contains no complete train/validation verdicts")

    model = ResponseParameterNetwork(
        args.embed_model,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        embed_batch_size=args.embed_batch_size,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "rpn.pt"
    history = []
    best_validation = float("inf")
    for epoch in range(args.epochs):
        train_loss = partial_m_step(
            model,
            optimizer,
            train_rollouts,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            lambda_a=args.lambda_a,
            grad_clip=args.grad_clip,
            seed=epoch,
        )
        validation_loss = evaluate_nll(model, val_rollouts)
        record = {
            "epoch": epoch + 1,
            "train_nll": train_loss,
            "validation_nll": validation_loss,
        }
        history.append(record)
        logger.info(
            "epoch %d train_nll=%.4f validation_nll=%.4f",
            epoch + 1,
            train_loss,
            validation_loss,
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            model.save(checkpoint_path, epoch=epoch + 1, validation_nll=validation_loss)

    metrics = {
        "best_validation_nll": best_validation,
        "train_rollouts": len(train_rollouts),
        "validation_rollouts": len(val_rollouts),
        "history": history,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    logger.info("wrote %s", checkpoint_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
