from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.rrt import (
    RRTConfig,
    criterion_nll,
    estimate_quality,
    fisher_information,
    oriented_labels,
    partial_m_step,
    pass_probability,
    posterior_sd,
    select_by_fisher,
)
from reward import RRTReward, soft_length_penalty


def _estimate(verdicts, discrimination, difficulty):
    verdicts = np.asarray(verdicts, dtype=float)
    n_rollouts, n_criteria = verdicts.shape
    return estimate_quality(
        verdicts.ravel(),
        np.tile(discrimination, n_rollouts),
        np.tile(difficulty, n_rollouts),
        np.repeat(np.arange(n_rollouts), n_criteria),
        n_rollouts,
    )


def test_probit_is_centered_at_item_difficulty():
    assert pass_probability(1.7, 0.4, 0.4) == pytest.approx(0.5)


def test_map_quality_respects_verdict_dominance():
    verdicts = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
        ]
    )
    quality = _estimate(
        verdicts,
        discrimination=np.array([0.7, 1.0, 1.4, 2.0]),
        difficulty=np.array([-1.0, -0.2, 0.4, 1.2]),
    )
    assert np.all(np.diff(quality) > 0.0)


def test_bisection_is_symmetric_and_finite_in_the_tails():
    quality = _estimate(
        np.array([[1, 1, 1], [0, 0, 0]], dtype=float),
        np.full(3, 50.0),
        np.zeros(3),
    )
    assert np.isfinite(quality).all()
    assert quality[0] == pytest.approx(-quality[1], abs=1e-10)


def test_bisection_expands_its_initial_bracket():
    quality = _estimate(np.ones((1, 100)), np.full(100, 2.0), np.full(100, 4.0))
    assert quality[0] > RRTConfig().initial_z_bound


def test_unsorted_group_indices_are_supported():
    quality = estimate_quality(
        [1, 0, 1, 0],
        np.ones(4),
        np.zeros(4),
        group_index=[1, 0, 1, 0],
        n_groups=2,
    )
    assert quality[1] > quality[0]


def test_negative_point_criteria_are_oriented_as_favorable():
    presence = np.array([[1, 1, 0], [0, 0, 1]], dtype=float)
    favorable, weights = oriented_labels(presence, [2, -1, -3], use_point_weights=True)
    np.testing.assert_array_equal(favorable, [[1, 0, 1], [0, 1, 0]])
    np.testing.assert_allclose(weights, [2 / 6, 1 / 6, 3 / 6])


def test_more_information_reduces_local_uncertainty():
    one = posterior_sd(np.ones(1), np.zeros(1), 0.0)
    eight = posterior_sd(np.ones(8), np.zeros(8), 0.0)
    assert eight < one < 1.0
    assert np.all(fisher_information([1.0], [0.0], [0.0]) > 0.0)


def test_fisher_selection_uses_group_total_information():
    selected = select_by_fisher(
        discrimination=[0.5, 2.0, 1.0],
        difficulty=[0.0, 0.0, 3.0],
        qualities=[-0.2, 0.2],
        budget=1,
    )
    np.testing.assert_array_equal(selected, [1])


def test_tail_nll_is_finite_and_differentiable():
    discrimination = torch.tensor([10.0], requires_grad=True)
    loss = criterion_nll(
        discrimination,
        torch.tensor([5.0]),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(discrimination.grad).all()


class TinyRPN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_a = torch.nn.Parameter(torch.tensor(0.0))
        self.raw_b = torch.nn.Parameter(torch.tensor(0.0))
        self.rrt_config = RRTConfig()

    @property
    def device(self):
        return self.raw_a.device

    def parameters_for_text(self, criteria, prompts):
        n = len(criteria)
        a = torch.nn.functional.softplus(self.raw_a).repeat(n) + 0.05
        b = self.raw_b.repeat(n)
        return a, b

    def trainable_parameters(self):
        return list(self.parameters())


def test_partial_m_step_updates_a_tiny_rpn():
    model = TinyRPN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = model.raw_b.detach().clone()
    loss = partial_m_step(
        model,
        optimizer,
        [
            {"prompt": "q", "criteria": ["a", "b"], "verdicts": np.array([1.0, 0.0])},
            {"prompt": "q", "criteria": ["a", "b"], "verdicts": np.array([1.0, 1.0])},
        ],
        batch_size=1,
    )
    assert np.isfinite(loss)
    assert not torch.equal(before, model.raw_b.detach())


def test_soft_length_penalty_uses_the_paper_buffer():
    assert soft_length_penalty(28672) == 0.0
    assert soft_length_penalty(32768) == pytest.approx(4.0)


def test_adaptive_fisher_recomputes_group_quality():
    class FakeModel:
        rrt_config = RRTConfig()

        def predict(self, criteria, prompt):
            return np.array([0.5, 2.0]), np.array([0.0, 0.0])

    class FakeJudge:
        config = SimpleNamespace(concurrency=2)

        def grade_criterion(self, prompt, response, rubric):
            return response == "good"

    reward = RRTReward.__new__(RRTReward)
    reward.online = False
    reward.model = FakeModel()
    reward.judge = FakeJudge()
    records, selected = reward.score_group_adaptive(
        "question",
        ["good", "bad"],
        [
            {"criterion": "weak", "points": 1.0},
            {"criterion": "strong", "points": 1.0},
        ],
        budget=1,
    )
    assert selected == [1]
    assert records[0].quality > records[1].quality
