"""Safety invariants for the combined team strategy, not mock-effect fixtures."""
import socket

import pytest

from agent import Agent
from test_agent import PublicTestEnv


@pytest.mark.parametrize("charged", [False, True])
def test_transfer_recovers_first_failure_without_network(tmp_path, monkeypatch, charged):
    def blocked(*args, **kwargs):
        raise AssertionError("Default strategy must not access the network")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    env = PublicTestEnv(effect=0.15)
    original = env.run_pilot
    attempts = []

    def intermittent(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            if charged:
                original(**kwargs)
            raise RuntimeError("synthetic temporary error")
        return original(**kwargs)

    env.run_pilot = intermittent
    model = Agent(mode="transfer", history_path=tmp_path / "missing.csv", max_pilots=8)
    assert model.act(env)
    assert model.trace["pilots"]
    assert 2 <= model.trace["pilot_attempts"] <= 8
    failure = model.trace["pilot_failures"][0]
    assert failure["retry_same_action"] is (not charged)
    assert bool(failure["contacts"]) is charged
    assert model.trace["final_cost"] <= env.remaining_budget
    assert model.trace["final_contacts"] <= env.remaining_contacts


def test_transfer_full_run_survives_solver_failure(tmp_path, monkeypatch):
    import agent

    def unavailable(*args, **kwargs):
        raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr(agent, "milp", unavailable)
    env = PublicTestEnv(effect=0.20)
    model = Agent(mode="transfer", history_path=tmp_path / "missing.csv", max_pilots=6)
    result = model.act(env)
    assert 1 <= len(result) <= 10
    assert model.trace["status"] == "ready"
    assert model.trace["final_cost"] <= env.remaining_budget
    assert model.trace["final_contacts"] <= env.remaining_contacts
