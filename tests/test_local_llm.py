"""Contract fixtures, NOT tests of the quality of a real language model."""
import json

import pytest

import agent
from agent import Agent, local_pilot_priority


OPTIONS = [{"candidate_id": "allowed_one", "customers": 20}]


def responder(advice):
    def request(path, payload=None, timeout=8):
        if path == "/api/show":
            return {"details": {"format": "gguf"}, "model_info": {"general.architecture": "test"}}
        assert payload["stream"] is False
        assert payload["format"]["properties"]["candidate_ids"]["items"]["enum"] == ["allowed_one"]
        assert "ID_NUMBER" not in payload["prompt"]
        return {"done": True, "response": json.dumps(advice)}
    return request


def test_disabled_llm_never_contacts_a_server(monkeypatch):
    monkeypatch.setattr(agent, "_ollama_request", lambda *a, **k: pytest.fail("Unexpected local network request"))
    ids, status = local_pilot_priority(OPTIONS)
    assert ids == [] and status["status"] == "disabled"


def test_valid_local_model_priority_is_accepted(monkeypatch):
    monkeypatch.setattr(agent, "_ollama_request", responder({"candidate_ids": ["allowed_one"], "rationale": "Hypothesis only"}))
    ids, status = local_pilot_priority(OPTIONS, "test-local")
    assert ids == ["allowed_one"] and status["used"]


@pytest.mark.parametrize("advice", [
    {"candidate_ids": ["invented_id"], "rationale": "x"},
    {"candidate_ids": ["allowed_one", "allowed_one"], "rationale": "x"},
    {"candidate_ids": [], "rationale": "x"},
    {"candidate_ids": ["allowed_one"], "rationale": "x", "effect": 900},
    {"candidate_ids": ["allowed_one"], "rationale": "x" * 501},
    {"candidate_ids": [{}], "rationale": "x"},
    {"candidate_ids": "allowed_one", "rationale": "x"},
    [],
])
def test_invalid_model_output_falls_back_without_using_it(monkeypatch, advice):
    monkeypatch.setattr(agent, "_ollama_request", responder(advice))
    ids, status = local_pilot_priority(OPTIONS, "test-local")
    assert ids == [] and status["status"] == "fallback" and not status["used"]


@pytest.mark.parametrize("error", [TimeoutError(), ConnectionRefusedError(), ValueError(), KeyError()])
def test_local_server_errors_do_not_stop_the_agent(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(agent, "_ollama_request", fail)
    ids, status = local_pilot_priority(OPTIONS, "test-local")
    assert not ids and not status["used"]


def test_cloud_models_are_rejected_before_data_is_sent(monkeypatch):
    monkeypatch.setattr(agent, "_ollama_request", lambda *a, **k: pytest.fail("Cloud model request"))
    assert local_pilot_priority(OPTIONS, "test:cloud")[1]["status"] == "fallback"


def test_remote_alias_is_rejected_before_generation(monkeypatch):
    requests = []
    def request(path, *args):
        requests.append(path)
        return {"remote_model": "cloud_alias", "details": {"format": "gguf"}, "model_info": {"a": 1}}
    monkeypatch.setattr(agent, "_ollama_request", request)
    assert local_pilot_priority(OPTIONS, "alias")[1]["status"] == "fallback"
    assert requests == ["/api/show"]


def test_literal_loopback_no_redirect_and_response_size_cap(monkeypatch):
    class Connection:
        def __init__(self, host, port, timeout):
            assert host == "127.0.0.1" and port == 11434 and timeout == 8
        def request(self, method, path, body, headers):
            assert "Authorization" not in headers
        def getresponse(self):
            return self
        status = 302
        def close(self):
            pass
    monkeypatch.setattr(agent.http.client, "HTTPConnection", Connection)
    with pytest.raises(ValueError, match="http_error"):
        agent._ollama_request("/api/show", {})
    Connection.status = 200
    Connection.read = lambda self, n: b"a" * n
    with pytest.raises(ValueError, match="too_large"):
        agent._ollama_request("/api/show", {})


def test_portable_settings_are_loaded_beside_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "__file__", str(tmp_path / "agent.py"))
    (tmp_path / "profitpilot_config.json").write_text(json.dumps({"risk_weight": 1.2, "max_pilots": 4}))
    configured = Agent()
    assert configured.risk_weight == 1.2 and configured.max_pilots == 4
    assert Agent(risk_weight=0.2).risk_weight == 0.2
