"""Readable RRT reward used inside the policy-training loop.

The training framework only needs to call ``score_from_verdicts`` for each rollout group, use ``record.reward`` in GRPO, and call ``update`` once after the policy step.  Framework-specific distributed orchestration is intentionally not part of this repository.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from core.judge import RubricJudge
from core.rpn import ResponseParameterNetwork
from core.rrt import (
    estimate_quality,
    oriented_labels,
    partial_m_step,
    posterior_sd,
    select_by_fisher,
)


@dataclass
class RewardRecord:
    prompt: str
    criteria: list[str]
    verdicts: np.ndarray
    quality: float
    uncertainty: float
    reward: float

    def as_training_rollout(self):
        return {
            "prompt": self.prompt,
            "criteria": self.criteria,
            "verdicts": self.verdicts,
        }


class RRTReward:
    """Frozen RRT scoring with an optional online hard-EM update."""

    def __init__(
        self,
        checkpoint,
        *,
        device=None,
        online=True,
        learning_rate=2e-5,
        weight_decay=0.1,
        batch_size=16,
        lambda_a=0.05,
        grad_clip=1.0,
        judge: RubricJudge | None = None,
    ):
        self.model, metadata = ResponseParameterNetwork.load(checkpoint, device=device)
        self.online = bool(online)
        self.batch_size = int(batch_size)
        self.lambda_a = float(lambda_a)
        self.grad_clip = float(grad_clip)
        self.judge = judge
        self.step = int(metadata.get("step", 0))
        self.optimizer = None
        if self.online:
            self.optimizer = torch.optim.AdamW(
                self.model.trainable_parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            if metadata.get("optimizer") is not None:
                self.optimizer.load_state_dict(metadata["optimizer"])

    def score_from_verdicts(self, prompt, rubrics, presence):
        """Convert a complete rubric verdict vector into the RRT reward."""

        criteria = [str(rubric["criterion"]) for rubric in rubrics]
        points = [float(rubric["points"]) for rubric in rubrics]
        favorable, _ = oriented_labels(np.asarray([presence], float), points)
        verdicts = favorable[0]
        a, b = self.model.predict(criteria, prompt)
        quality = float(estimate_quality(verdicts, a, b, config=self.model.rrt_config)[0])
        uncertainty = posterior_sd(a, b, quality, self.model.rrt_config)
        return RewardRecord(
            prompt=str(prompt),
            criteria=criteria,
            verdicts=verdicts,
            quality=quality,
            uncertainty=uncertainty,
            reward=quality,
        )

    def score_response(self, prompt, response, rubrics):
        """Judge and score one response. Pass a judge when constructing the class."""

        if self.judge is None:
            raise RuntimeError("score_response requires a RubricJudge")
        presence = self.judge.grade(prompt, response, rubrics)
        return self.score_from_verdicts(prompt, rubrics, presence)

    def update(self, records):
        """Run one online stochastic partial M-step after a policy update."""

        if not self.online:
            return None
        rollouts = [record.as_training_rollout() for record in records]
        loss = partial_m_step(
            self.model,
            self.optimizer,
            rollouts,
            batch_size=self.batch_size,
            accumulation_steps=0,
            lambda_a=self.lambda_a,
            grad_clip=self.grad_clip,
            seed=self.step,
        )
        self.step += 1
        return loss

    def rank_criteria(self, prompt, rubrics, current_qualities, budget):
        """Run one group-total Fisher ranking step."""

        criteria = [str(rubric["criterion"]) for rubric in rubrics]
        a, b = self.model.predict(criteria, prompt)
        return select_by_fisher(a, b, current_qualities, budget)

    def score_group_adaptive(
        self,
        prompt,
        responses,
        rubrics,
        budget,
    ):
        """Judge a rollout group with sequential adaptive Fisher selection.

        After each selected criterion, the method updates every rollout's MAP quality and chooses the unjudged criterion with the largest group-total Fisher information. The paper uses this path with a frozen RPN.
        """

        if self.online:
            raise ValueError("adaptive Fisher judging uses a frozen RPN")
        if self.judge is None:
            raise RuntimeError("adaptive judging requires a RubricJudge")
        if not responses or not rubrics:
            raise ValueError("responses and rubrics must be non-empty")

        budget = min(max(int(budget), 1), len(rubrics))
        criteria = [str(rubric["criterion"]) for rubric in rubrics]
        points = np.asarray([float(rubric["points"]) for rubric in rubrics])
        a, b = self.model.predict(criteria, prompt)
        qualities = np.zeros(len(responses), dtype=np.float64)
        raw_presence = np.zeros((len(responses), len(rubrics)), dtype=bool)
        selected = []

        for _ in range(budget):
            remaining = np.asarray(
                [index for index in range(len(rubrics)) if index not in selected],
                dtype=np.int64,
            )
            local_index = select_by_fisher(a[remaining], b[remaining], qualities, 1)[0]
            criterion_index = int(remaining[local_index])
            workers = max(1, min(self.judge.config.concurrency, len(responses)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                verdicts = list(
                    pool.map(
                        lambda response: self.judge.grade_criterion(
                            prompt, response, rubrics[criterion_index]
                        ),
                        responses,
                    )
                )
            raw_presence[:, criterion_index] = verdicts
            selected.append(criterion_index)
            favorable, _ = oriented_labels(
                raw_presence[:, selected],
                points[selected],
            )
            for rollout_index in range(len(responses)):
                qualities[rollout_index] = estimate_quality(
                    favorable[rollout_index],
                    a[selected],
                    b[selected],
                    config=self.model.rrt_config,
                )[0]

        records = []
        favorable, _ = oriented_labels(raw_presence[:, selected], points[selected])
        for rollout_index, quality in enumerate(qualities):
            records.append(
                RewardRecord(
                    prompt=str(prompt),
                    criteria=[criteria[index] for index in selected],
                    verdicts=favorable[rollout_index],
                    quality=float(quality),
                    uncertainty=posterior_sd(
                        a[selected], b[selected], quality, self.model.rrt_config
                    ),
                    reward=float(quality),
                )
            )
        return records, selected

    def save(self, path):
        """Checkpoint the online RPN and optimizer beside the policy checkpoint."""

        metadata = {"step": self.step}
        if self.optimizer is not None:
            metadata["optimizer"] = self.optimizer.state_dict()
        self.model.save(Path(path), **metadata)
