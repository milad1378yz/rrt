"""Rubric Response Theory inference and the online hard-EM update.

The paper's standard model is a two-parameter probit item-response model:

    P(G_ij = 1 | z_i) = Phi(a_ij (z_i - b_ij)).

Criterion verdicts are first oriented so that one always means favorable.  A
standard-normal prior on rollout quality makes the one-dimensional posterior
strictly concave, so its mode is found reliably by expanding-bracket bisection.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import log_ndtr, ndtr
import torch


@dataclass(frozen=True)
class RRTConfig:
    """Numerical and parameter bounds used by the standard RRT model."""

    a_min: float = 0.05
    b_bound: float = 4.0
    initial_z_bound: float = 4.0
    sigma_z: float = 1.0
    bisection_steps: int = 40


def oriented_labels(presence, points, *, use_point_weights: bool = False):
    """Return favorable verdicts and normalized criterion weights.

    ``presence`` has shape ``[..., criteria]``.  Positive-point criteria are
    favorable when present; negative-point pitfalls are favorable when absent.
    RRT itself uses uniform weights.  Point weights are returned for the matched
    points-based reward and evaluation metric.
    """

    presence = np.asarray(presence, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 1 or not len(points) or presence.ndim == 0:
        raise ValueError("presence and points must describe a non-empty rubric")
    if presence.shape[-1] != len(points):
        raise ValueError("presence and points must contain the same number of criteria")
    favorable = np.where(points >= 0.0, presence, 1.0 - presence)
    if use_point_weights:
        weights = np.abs(points)
        total = float(weights.sum())
        weights = weights / total if total else np.full(len(points), 1.0 / len(points))
    else:
        weights = np.full(len(points), 1.0 / len(points))
    return favorable, weights


def pass_probability(a, b, quality):
    """Two-parameter probit probability ``Phi(a * (quality - b))``."""

    return ndtr(np.asarray(a) * (np.asarray(quality) - np.asarray(b)))


def _posterior_gradient(quality, verdicts, a, b, prior_precision):
    """Stable derivative of the probit log posterior."""

    logits = a * (quality - b)
    log_density = -0.5 * logits**2 - 0.5 * math.log(2.0 * math.pi)
    pass_mills = np.exp(log_density - log_ndtr(logits))
    fail_mills = np.exp(log_density - log_ndtr(-logits))
    evidence = a * (verdicts * pass_mills - (1.0 - verdicts) * fail_mills)
    return float(evidence.sum() - prior_precision * quality)


def estimate_quality(
    verdicts,
    discrimination,
    difficulty,
    group_index=None,
    n_groups=None,
    config: RRTConfig = RRTConfig(),
):
    """Infer one MAP quality per rollout with expanding-bracket bisection.

    Inputs are flat criterion observations. ``group_index`` maps each observation
    to a rollout.  It may be omitted for a single rollout.
    """

    verdicts = np.asarray(verdicts, dtype=np.float64)
    discrimination = np.asarray(discrimination, dtype=np.float64)
    difficulty = np.asarray(difficulty, dtype=np.float64)
    if not (verdicts.shape == discrimination.shape == difficulty.shape):
        raise ValueError("verdicts, discrimination, and difficulty must have equal shapes")
    if verdicts.ndim != 1 or not len(verdicts):
        raise ValueError("RRT requires a non-empty one-dimensional observation array")
    if ((verdicts < 0.0) | (verdicts > 1.0)).any() or not np.isfinite(verdicts).all():
        raise ValueError("verdicts must be finite values in [0, 1]")
    if (discrimination <= 0.0).any() or not np.isfinite(discrimination).all():
        raise ValueError("discrimination values must be finite and positive")

    if group_index is None:
        group_index = np.zeros(len(verdicts), dtype=np.int64)
        n_groups = 1
    else:
        group_index = np.asarray(group_index, dtype=np.int64)
        if group_index.shape != verdicts.shape:
            raise ValueError("group_index must match the observation shape")
        if n_groups is None:
            n_groups = int(group_index.max()) + 1
    if n_groups is None or n_groups <= 0:
        raise ValueError("n_groups must be positive")

    order = np.argsort(group_index, kind="stable")
    group_index = group_index[order]
    verdicts = verdicts[order]
    discrimination = discrimination[order]
    difficulty = difficulty[order]
    boundaries = np.searchsorted(group_index, np.arange(n_groups + 1))
    if np.any(boundaries[1:] == boundaries[:-1]):
        raise ValueError("every rollout group must contain at least one criterion")

    prior_precision = 1.0 / config.sigma_z**2
    quality = np.empty(n_groups, dtype=np.float64)
    for group, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        G = verdicts[start:end]
        a = discrimination[start:end]
        b = difficulty[start:end]
        lo, hi = -config.initial_z_bound, config.initial_z_bound
        for _ in range(64):
            if _posterior_gradient(lo, G, a, b, prior_precision) >= 0.0:
                break
            lo *= 2.0
        else:
            raise RuntimeError("failed to bracket MAP quality from below")
        for _ in range(64):
            if _posterior_gradient(hi, G, a, b, prior_precision) <= 0.0:
                break
            hi *= 2.0
        else:
            raise RuntimeError("failed to bracket MAP quality from above")

        for _ in range(config.bisection_steps):
            midpoint = 0.5 * (lo + hi)
            if _posterior_gradient(midpoint, G, a, b, prior_precision) > 0.0:
                lo = midpoint
            else:
                hi = midpoint
        quality[group] = 0.5 * (lo + hi)
    return quality


def fisher_information(discrimination, difficulty, quality):
    """Expected Fisher information from each criterion at ``quality``."""

    a = np.asarray(discrimination, dtype=np.float64)
    b = np.asarray(difficulty, dtype=np.float64)
    z = np.asarray(quality, dtype=np.float64)
    logits = a * (z - b)
    probability = np.clip(ndtr(logits), 1e-12, 1.0 - 1e-12)
    density = np.exp(-0.5 * logits**2) / math.sqrt(2.0 * math.pi)
    return a**2 * density**2 / (probability * (1.0 - probability))


def select_by_fisher(discrimination, difficulty, qualities, budget):
    """Rank criteria by total information over a group of rollout qualities."""

    a = np.asarray(discrimination, dtype=np.float64)
    b = np.asarray(difficulty, dtype=np.float64)
    qualities = np.atleast_1d(np.asarray(qualities, dtype=np.float64))
    budget = min(max(int(budget), 0), len(a))
    information = fisher_information(a[None, :], b[None, :], qualities[:, None]).sum(axis=0)
    return np.argsort(-information, kind="stable")[:budget]


def posterior_sd(discrimination, difficulty, quality, config: RRTConfig = RRTConfig()):
    """Laplace posterior standard deviation at a rollout's MAP quality."""

    information = fisher_information(discrimination, difficulty, quality).sum()
    precision = 1.0 / config.sigma_z**2 + float(information)
    return float(1.0 / np.sqrt(precision))


def criterion_nll(discrimination, difficulty, quality, verdicts, *, reduction="mean"):
    """Stable differentiable probit negative log likelihood."""

    logits = discrimination * (quality - difficulty)
    log_pass = torch.special.log_ndtr(logits)
    log_fail = torch.special.log_ndtr(-logits)
    losses = -(verdicts * log_pass + (1.0 - verdicts) * log_fail)
    if reduction == "sum":
        return losses.sum()
    if reduction == "none":
        return losses
    return losses.mean()


def flatten_rollouts(rollouts):
    """Flatten ``{prompt, criteria, verdicts}`` records for one RPN forward."""

    criteria, prompts, verdicts, groups = [], [], [], []
    for group, rollout in enumerate(rollouts):
        n_criteria = len(rollout["criteria"])
        criteria.extend(rollout["criteria"])
        prompts.extend([rollout["prompt"]] * n_criteria)
        verdicts.extend(float(value) for value in rollout["verdicts"])
        groups.extend([group] * n_criteria)
    return criteria, prompts, np.asarray(verdicts), np.asarray(groups, dtype=np.int64)


def partial_m_step(
    model,
    optimizer,
    rollouts,
    *,
    batch_size: int = 16,
    accumulation_steps: int = 0,
    lambda_a: float = 0.05,
    grad_clip: float = 1.0,
    seed: int = 0,
):
    """Run a hard-EM M-step over one rollout pool.

    Dropout is active.  Every mini-batch recomputes detached MAP targets, while
    gradients are accumulated for ``accumulation_steps`` mini-batches. Passing
    zero accumulates the whole pool into the paper's one online AdamW step.
    """

    if not rollouts:
        raise ValueError("partial_m_step requires at least one rollout")
    if batch_size <= 0 or accumulation_steps < 0:
        raise ValueError("batch_size must be positive and accumulation_steps non-negative")
    order = np.random.default_rng(seed).permutation(len(rollouts))
    batches = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
    accumulation_steps = int(accumulation_steps) or len(batches)
    total_observations = sum(len(rollout["criteria"]) for rollout in rollouts)
    total_loss = 0.0
    model.train()
    for group_start in range(0, len(batches), accumulation_steps):
        batch_group = batches[group_start : group_start + accumulation_steps]
        group_observations = sum(
            len(rollouts[index]["criteria"]) for batch in batch_group for index in batch
        )
        optimizer.zero_grad()
        for batch_indices in batch_group:
            batch = [rollouts[index] for index in batch_indices]
            criteria, prompts, verdicts, groups = flatten_rollouts(batch)
            a, b = model.parameters_for_text(criteria, prompts)
            quality = estimate_quality(
                verdicts,
                a.detach().cpu().numpy(),
                b.detach().cpu().numpy(),
                groups,
                len(batch),
                model.rrt_config,
            )
            z = torch.as_tensor(quality[groups], dtype=torch.float32, device=model.device)
            G = torch.as_tensor(verdicts, dtype=torch.float32, device=model.device)
            loss_sum = criterion_nll(a, b, z, G, reduction="sum")
            if lambda_a:
                loss_sum = loss_sum + lambda_a * torch.log(a).pow(2).sum()
            (loss_sum / group_observations).backward()
            total_loss += float(loss_sum.detach())
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
        optimizer.step()
    return total_loss / total_observations
