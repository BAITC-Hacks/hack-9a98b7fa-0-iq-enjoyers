"""Independent contract tests; no organizer implementation or hidden effects."""
import itertools
import math
import socket

import numpy as np
import pandas as pd
import pytest

from agent import Agent, Candidate, CHANNELS, allocate, feasible, update_estimate


def candidate(cell, n=50, channel="sms", target="tariff_b"):
    return Candidate(cell, "tariff_a", "HIGH", target, channel, n, n * 1000, 0.01, 0.04)


def test_allocator_respects_budget_contacts_and_cell_exclusivity():
    pool = [candidate(0), candidate(0, channel="push"), candidate(1), candidate(2)]
    values = [500, 400, 450, 300]
    picked = allocate(pool, values, budget=300, contacts=100)
    assert feasible(picked, pool, 300, 100)
    assert set(picked) == {1, 2}


def test_allocator_negative_mandatory_campaign_and_free_push():
    pool = [candidate(0), candidate(1, channel="push")]
    assert allocate(pool, [-100, -5], 0, 100) == [1]


def test_allocator_enforces_ten_campaigns_and_oversize_rejection():
    pool = [candidate(i, n=10, channel="push") for i in range(15)]
    pool.append(candidate(16, n=5001, channel="push"))
    picked = allocate(pool, [10] * 15 + [10000], 0, 15000)
    assert len(picked) == 10
    assert 15 not in picked


def test_allocator_matches_exhaustive_small_instances():
    rng = np.random.default_rng(123)  # Test fixture, not a judge seed.
    for _ in range(12):
        pool = [candidate(i // 2, n=int(rng.integers(10, 100)),
                          channel="push" if i % 2 else "sms") for i in range(8)]
        values = rng.uniform(-100, 1000, len(pool))
        picked = allocate(pool, values, 600, 180)
        possible = [subset for size in range(1, 5)
                    for subset in itertools.combinations(range(len(pool)), size)
                    if feasible(subset, pool, 600, 180)]
        optimum = max(sum(values[i] for i in subset) for subset in possible)
        assert sum(values[i] for i in picked) == pytest.approx(optimum, abs=1e-6)


def test_update_uses_evidence_and_reduces_working_variance():
    mean, var = update_estimate(0.1, 0.04, -0.2, 200)
    assert -0.2 < mean < 0
    assert 0 < var < 0.04


class PublicTestEnv:
    """Deliberately tiny independent environment with a public API only."""
    def __init__(self, effect=0.05, invalid=False):
        self.customer_profile = pd.DataFrame([
            {"ID_NUMBER": i, "current_tariff": "tariff_a", "arpu_segment": segment,
             "predicted_arpu": arpu}
            for segment, count, arpu, start in [("HIGH", 300, 9000, 0), ("MID", 60, 2000, 300)]
            for i in range(start, start + count)])
        self.tariffs = pd.DataFrame({"tariff_plan_code": ["tariff_a", "tariff_b", "tariff_c"]})
        self.remaining_budget = 100000.0
        self.remaining_contacts = 15000
        self.pilots_left = 20
        self.pilot_history = []
        self.effect = effect
        self.invalid = invalid

    def run_pilot(self, *, target_tariff, channel, n_customers,
                  filter_current_tariff, filter_arpu_segment):
        group = self.customer_profile[
            (self.customer_profile.current_tariff == filter_current_tariff)
            & (self.customer_profile.arpu_segment == filter_arpu_segment)]
        assert 10 <= n_customers <= min(200, len(group))
        assert self.pilots_left > 0
        cost = n_customers * CHANNELS[channel][0]
        assert cost <= self.remaining_budget and n_customers <= self.remaining_contacts
        self.remaining_budget -= cost
        self.remaining_contacts -= n_customers
        self.pilots_left -= 1
        self.pilot_history.append((target_tariff, channel, n_customers))
        return {"observed_lift_ratio": float("nan") if self.invalid else self.effect}


@pytest.mark.parametrize("mode", ["adaptive", "fixed"])
@pytest.mark.parametrize("effect", [0.10, -0.10, 0.0])
def test_agent_public_contract_offline(mode, effect, tmp_path, monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network is forbidden")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    env = PublicTestEnv(effect=effect)
    agent = Agent(mode=mode, history_path=tmp_path / "missing.csv", max_pilots=6)
    campaigns = agent.act(env)
    assert 1 <= len(campaigns) <= 10
    assert 1 <= len(env.pilot_history) <= 6
    assert agent.trace["final_cost"] <= env.remaining_budget
    assert agent.trace["final_contacts"] <= env.remaining_contacts
    assert all(item["target_tariff"] != item["filter_current_tariff"] for item in campaigns)
    assert all(math.isfinite(item["incremental_net_proxy"]) for item in agent.trace["decisions"])
    assert agent.trace["elapsed_seconds"] < 5


def test_invalid_pilot_result_does_not_crash(tmp_path):
    env = PublicTestEnv(invalid=True)
    agent = Agent(history_path=tmp_path / "missing.csv")
    assert len(agent.act(env)) >= 1
    assert agent.trace["pilot_error"] == "nonfinite_observed_lift_ratio"


def test_history_is_robust_to_zero_denominators_and_duplicate_rows(tmp_path):
    path = tmp_path / "history.csv"
    frame = pd.DataFrame({"AVG_ARPU_PREV_3M": [0, 1000, 1000],
                          "AVG_ARPU_NEXT_3M": [5000, 1100, 1100],
                          "tariff_plan_code_from": ["tariff_a"] * 3,
                          "tariff_plan_code_to": ["tariff_b"] * 3})
    frame.to_csv(path, index=False)
    history = Agent(history_path=path)._history()
    assert history[("tariff_a", "tariff_b")] == pytest.approx(0.1 / 31)


def test_solver_failure_has_valid_fallback(monkeypatch):
    import agent
    def failed(*args, **kwargs):
        raise RuntimeError("solver unavailable")
    monkeypatch.setattr(agent, "milp", failed)
    pool = [candidate(0), candidate(1, channel="push")]
    assert feasible(allocate(pool, [100, 200], 200, 100), pool, 200, 100)
