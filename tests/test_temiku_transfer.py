"""Contracts for the team's Temiku-derived policy; no hidden effect constants."""
import numpy as np
import pandas as pd

from agent import Agent, TransferPosterior, allocate
from test_agent import PublicTestEnv, candidate


def test_history_keeps_transition_share_separate_from_effect(tmp_path):
    path = tmp_path / "history.csv"
    pd.DataFrame({
        "ID_NUMBER": range(100),
        "tariff_plan_code_from": ["tariff_a"] * 100,
        "tariff_plan_code_to": ["tariff_b"] * 80 + ["tariff_c"] * 20,
        "AVG_ARPU_PREV_3M": [2000.0] * 100,
        "AVG_ARPU_NEXT_3M": [6000.0] * 100,
    }).to_csv(path, index=False)
    history = Agent(history_path=path)._transfer_history()
    common_signal, common_scale = history["tariff_a", "MID", "tariff_b"]
    rare_signal, rare_scale = history["tariff_a", "MID", "tariff_c"]
    assert common_signal > rare_signal > 0
    assert common_scale > rare_scale > 0
    assert np.isclose(common_signal / rare_signal, common_scale / rare_scale)


def test_history_transfer_is_learned_across_targets():
    arms = [candidate(0), candidate(1, target="tariff_c")]
    posterior = TransferPosterior(arms, [0.5, 0.4], [0.1, 0.1])
    before = posterior.mean.copy()
    posterior.update(0, -0.2, 200, 0.65)
    assert posterior.mean[0] < before[0]
    assert posterior.mean[1] < before[1]
    assert np.linalg.eigvalsh(posterior.cov).min() > 0


def test_transition_uncertainty_scales_with_frequency():
    arms = [candidate(0), candidate(1)]
    posterior = TransferPosterior(arms, [0, 0], [0.3, 0.03])
    assert posterior.variance[0] > posterior.variance[1] > 0


def test_missing_direction_does_not_mean_missing_entire_history(tmp_path):
    path = tmp_path / "history.csv"
    pd.DataFrame({
        "tariff_plan_code_from": ["old_unseen_tariff"] * 10,
        "tariff_plan_code_to": ["tariff_b"] * 10,
        "AVG_ARPU_PREV_3M": np.arange(6000.0, 6010.0),
        "AVG_ARPU_NEXT_3M": np.arange(7200.0, 7210.0),
    }).to_csv(path, index=False)
    model = Agent(history_path=path)
    candidates = model._candidates(PublicTestEnv())
    model._prepare_transfer(candidates)
    known = next(i for i, c in enumerate(model.arms) if c.target == "tariff_b")
    unknown = next(i for i, c in enumerate(model.arms) if c.target == "tariff_c")
    assert model.posterior.mean[known] > 0  # segment-level prior for a new cell
    assert model.posterior.mean[unknown] == 0
    assert 0 < model.posterior.variance[unknown] < 0.01


def test_material_shared_evidence_can_support_unpiloted_target(tmp_path):
    model = Agent(history_path=tmp_path / "absent")
    candidates = model._candidates(PublicTestEnv())
    model._prepare_transfer(candidates)
    model.posterior.counts[0] = 1
    model.posterior.cov *= 0.1
    model.posterior.mean[:] = 0.5
    model._refresh_transfer(candidates)
    pool, _ = model._transfer_pool(candidates, {})
    assert any(model.posterior.counts[model.arm_index[c.cell, c.target]] == 0 for c in pool)


def test_negative_fallback_preserves_smaller_feasible_tested_cell(tmp_path):
    model = Agent(history_path=tmp_path / "absent")
    candidates = model._candidates(PublicTestEnv())
    model._prepare_transfer(candidates)
    model.posterior.counts[:] = 1
    model.posterior.cov[:] = 0
    model.posterior.mean[:] = [-0.0001 if c.segment == "HIGH" else -0.3 for c in model.arms]
    model._refresh_transfer(candidates)
    pool, values = model._transfer_pool(candidates, {})
    assert len({c.cell for c in pool}) == 2
    selected = allocate(pool, values, budget=0, contacts=100)
    assert selected and all(pool[i].n <= 100 for i in selected)
