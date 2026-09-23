"""Synthetic contracts, not copies of the organizer's hidden effects."""
import math

import numpy as np
import pandas as pd
import pytest

from agent import Agent, Candidate, TransferPosterior, allocate, feasible
from test_agent import PublicTestEnv, candidate


def test_transfer_updates_similar_hypotheses_but_keeps_residual_uncertainty():
    arms = [candidate(0), candidate(1), candidate(2, target="tariff_c")]
    model = TransferPosterior(arms, [0, 0, 0])
    before = model.variance.copy()
    model.update(0, 0.25, 200, 0.65)
    assert model.mean[0] > model.mean[1] > model.mean[2] > 0
    assert 0 < model.variance[0] < before[0]
    assert 0 < model.variance[1] < before[1]
    assert model.variance[1] > model.variance[0]
    assert model.counts.tolist() == [1, 0, 0]
    assert np.allclose(model.cov, model.cov.T)
    assert np.linalg.eigvalsh(model.cov).min() > 0


def test_pilots_can_overturn_a_wrong_positive_historical_hint():
    model = TransferPosterior([candidate(0)], [0.6])
    assert model.mean[0] > 0
    for _ in range(3):
        model.update(0, -0.20, 200, 0.65)
    assert model.mean[0] < 0


def test_all_tariffs_are_candidates_not_just_top_three(tmp_path):
    env = PublicTestEnv()
    env.tariffs = pd.DataFrame({"tariff_plan_code": ["tariff_a"] + [f"target_{i}" for i in range(8)]})
    model = Agent(mode="transfer", history_path=tmp_path / "absent")
    candidates = model._candidates(env)
    assert {c.target for c in candidates} == {f"target_{i}" for i in range(8)}


@pytest.mark.parametrize("effect", [-0.3, 0, 0.25])
def test_transfer_contract_on_positive_and_negative_pilots(effect, tmp_path):
    env = PublicTestEnv(effect=effect)
    model = Agent(mode="transfer", max_pilots=8, history_path=tmp_path / "absent")
    result = model.act(env)
    assert 1 <= len(result) <= 10
    assert 1 <= len(model.trace["pilots"]) <= 8
    assert model.trace["final_contacts"] <= env.remaining_contacts
    assert model.trace["final_cost"] <= env.remaining_budget
    assert all(math.isfinite(item["incremental_net_proxy"]) for item in model.trace["decisions"])
    assert model.trace["pilot_attempts"] <= 8


def test_channel_is_selected_for_value_not_always_sms(tmp_path):
    env = PublicTestEnv(effect=0.25)
    model = Agent(mode="transfer", risk_weight=0.0, max_pilots=6, history_path=tmp_path / "absent")
    result = model.act(env)
    assert any(c["channel"] in {"digital_ads", "call"} for c in result)


def test_lookahead_does_not_count_unvalidated_historical_plans(tmp_path):
    env = PublicTestEnv()
    model = Agent(mode="transfer", history_path=tmp_path / "absent")
    candidates = model._candidates(env)
    model._prepare_transfer(candidates)
    mean = np.full(len(model.arms), 0.5)
    variance = model.posterior.variance
    assert model._relaxed_value(mean, variance, {}, 100000, 15000) == 0
    assert model._relaxed_value(mean, variance, {}, 100000, 15000, extra=0) > 0


def test_bundled_allocator_prevents_overlap_and_double_charging():
    a, b, c = candidate(0), candidate(1), candidate(2)
    bundle = Candidate(0, "tariff_a;tariff_b", "HIGH", "tariff_c", "sms", 100, 100000, .1, .01,
                       members=(0, 1))
    pool = [a, b, c, bundle]
    indices = allocate(pool, [100, 100, 50, 250], 600, 150)
    assert set(indices) == {2, 3}
    assert feasible(indices, pool, 600, 150)
    assert not feasible([0, 3], pool, 1000, 1000)


def test_packed_filters_match_real_disjoint_audiences_and_size_limit(tmp_path):
    env = PublicTestEnv(effect=0.25)
    chunks = []
    for current in ("tariff_a", "tariff_b", "tariff_c"):
        rows = pd.DataFrame({"ID_NUMBER": range(len(chunks) * 1800, (len(chunks) + 1) * 1800),
                             "current_tariff": current, "arpu_segment": "HIGH", "predicted_arpu": 6000.0})
        chunks.append(rows)
    env.customer_profile = pd.concat(chunks, ignore_index=True)
    model = Agent(mode="transfer", history_path=tmp_path / "absent")
    candidates = model._candidates(env)
    model._prepare_transfer(candidates)
    # Direct synthetic evidence makes multiple public cells bundle-eligible.
    for i in range(len(model.arms)):
        model.posterior.update(i, .3, 200, .65)
    model._refresh_transfer(candidates)
    pool, values = model._transfer_pool(candidates, {}, pack=True)
    bundles = [c for c in pool if len(c.members) > 1]
    assert bundles
    for bundle in bundles:
        actual = env.customer_profile[env.customer_profile.current_tariff.isin(bundle.current.split(';'))]
        assert len(actual) == bundle.n <= 5000
        assert len(set(bundle.members)) == len(bundle.members)
    chosen = allocate(pool, values, 100000, 15000)
    assert feasible(chosen, pool, 100000, 15000)


def test_transfer_selection_and_submission_are_deterministic(tmp_path):
    settings = dict(mode="transfer", max_pilots=6, history_path=tmp_path / "absent")
    first, second = Agent(**settings), Agent(**settings)
    assert first.act(PublicTestEnv()) == second.act(PublicTestEnv())
    assert [(p['target_tariff'], p['n']) for p in first.trace['pilots']] == [
        (p['target_tariff'], p['n']) for p in second.trace['pilots']]
