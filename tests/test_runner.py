"""Startup and independent validation tests; no organizer files or credentials."""
import csv
import json
import zipfile
from argparse import Namespace

import pytest

import run
from scripts.validate_run import ValidationError, validate_run
from scripts.run_case import strategy_settings
from scripts.render_report import render_report


def test_bundle_autodetection_and_explicit_path_with_spaces(tmp_path):
    archive = tmp_path / "beeline_case_participants (1).zip"
    archive.touch()
    assert run.choose_bundle(search_dirs=[tmp_path, tmp_path]) == archive
    assert run.choose_bundle(archive) == archive


def test_ambiguous_bundle_requires_explicit_choice(tmp_path):
    (tmp_path / "beeline_case_participants.zip").touch()
    (tmp_path / "beeline_case_participants (1).zip").touch()
    with pytest.raises(ValueError, match="Найдено архивов: 2"):
        run.choose_bundle(search_dirs=[tmp_path])


def test_missing_bundle_is_actionable(tmp_path):
    with pytest.raises(ValueError, match="--bundle"):
        run.choose_bundle(tmp_path / "missing.zip")


def test_unexpected_zip_files_are_rejected_before_extraction(tmp_path):
    archive = tmp_path / "case.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escaped.py", "fixture")
    with pytest.raises(ValueError, match="Состав ZIP"):
        run.inspect_bundle(archive)
    assert not (tmp_path.parent / "escaped.py").exists()


def test_subprocess_environment_does_not_forward_credentials(monkeypatch):
    for key in ("OPENAI_API_KEY", "GITHUB_TOKEN", "NVIDIA_API_KEY", "PYTHONPATH", "PIP_INDEX_URL"):
        monkeypatch.setenv(key, "private-test-fixture")
    assert "private-test-fixture" not in run.safe_env().values()


def test_requirements_are_fully_pinned():
    versions = run.expected_versions(run.ROOT / "requirements-dev.txt")
    assert {"numpy", "pandas", "scipy", "pytest"} <= versions.keys()
    assert all(value and ">" not in value and "<" not in value for value in versions.values())


def test_offline_bootstrap_never_creates_environment_or_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("Offline startup attempted a subprocess")
    monkeypatch.setattr(run.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="--prepare-only"):
        run.prepare_environment(offline=True)
    assert not (tmp_path / ".venv-agent").exists()


@pytest.fixture
def public_run(tmp_path):
    work = tmp_path / "bundle"
    work.mkdir()
    with (work / "customer_profile.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["ID_NUMBER", "current_tariff", "arpu_segment"])
        writer.writeheader()
        writer.writerows({"ID_NUMBER": i, "current_tariff": "tariff_a", "arpu_segment": "HIGH"}
                         for i in range(20))
    (work / "tariff_dictionary.csv").write_text("tariff_plan_code\ntariff_a\ntariff_b\n")
    campaign = {"campaign_name": "test", "target_tariff": "tariff_b", "channel": "sms",
                "filter_current_tariff": "tariff_a", "filter_arpu_segment": "HIGH"}
    trace = {"status": "ready", "pilots": [{**campaign, "n": 10, "observed_lift_ratio": 0.1, "spent": 40}],
             "pilot_failures": [], "pilot_attempts": 1,
             "decisions": [{**campaign, "customers": 20, "cost": 80}],
             "remaining_budget_before_final": 99960, "remaining_contacts_before_final": 14990,
             "final_cost": 80, "final_contacts": 20, "final_campaigns": 1, "elapsed_seconds": 0.1}
    traces = tmp_path / "traces"
    traces.mkdir()
    def persist():
        (traces / "trace-test.json").write_text(json.dumps(trace), encoding="utf-8")
    persist()
    with (work / "submission.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(campaign))
        writer.writeheader()
        writer.writerow(campaign)
    return work, traces, trace, persist


def test_validator_checks_public_audiences_costs_and_submission(public_run):
    work, traces, _, _ = public_run
    result = validate_run(work, traces, submission=True)
    assert result["valid"] and result["submission_checked"]
    assert result["max_total_cost"] == 120
    assert result["max_total_contacts"] == 30


@pytest.mark.parametrize("field,value,message", [
    ("status", "no_usable_pilot", "ready"),
    ("pilots", [], "successful pilot"),
    ("pilot_attempts", 21, "request limit"),
    ("remaining_budget_before_final", 99999, "budget accounting"),
    ("final_contacts", 19, "contacts mismatch"),
    ("elapsed_seconds", 301, "runtime limit"),
])
def test_invalid_trace_is_not_a_successful_run(public_run, field, value, message):
    work, traces, trace, persist = public_run
    trace[field] = value
    persist()
    with pytest.raises(ValidationError, match=message):
        validate_run(work, traces)


def test_validator_counts_resources_consumed_by_failed_pilot(public_run):
    work, traces, trace, persist = public_run
    trace["pilot_failures"] = [{"spent": 40, "contacts": 10}]
    trace["pilot_attempts"] = 2
    trace["remaining_budget_before_final"] -= 40
    trace["remaining_contacts_before_final"] -= 10
    persist()
    result = validate_run(work, traces)
    assert result["max_total_cost"] == 160
    assert result["max_total_contacts"] == 40


def test_validator_rejects_final_overlap(public_run):
    work, traces, trace, persist = public_run
    trace["decisions"].append(dict(trace["decisions"][0]))
    persist()
    with pytest.raises(ValidationError, match="Overlapping"):
        validate_run(work, traces)


def test_validator_rejects_different_submission(public_run):
    work, traces, _, _ = public_run
    (work / "submission.csv").write_text("campaign_name\nwrong\n")
    with pytest.raises(ValidationError, match="differs"):
        validate_run(work, traces, submission=True)


def test_validator_requires_all_traces(public_run):
    work, traces, _, _ = public_run
    with pytest.raises(ValidationError, match="Missing"):
        validate_run(work, traces, expected_runs=2)


@pytest.mark.parametrize("field,value", [("risk_weight", float("nan")), ("risk_weight", -1),
                                       ("max_pilots", 21), ("llm_timeout", 60)])
def test_strategy_cli_rejects_invalid_settings(field, value):
    args = Namespace(mode="fixed", risk_weight=0.75, max_pilots=20, llm_model="", llm_timeout=8)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        strategy_settings(args)


def test_report_escapes_untrusted_model_text_and_is_self_contained():
    report = json.loads((run.ROOT / "demo/example_report.json").read_text())
    report["trace"]["llm"] = {"status": "accepted", "rationale": '<script>alert("x")</script>', "used": True}
    rendered = render_report(report)
    assert "&lt;script&gt;" in rendered and "<script>" not in rendered
    assert "<script src=" not in rendered and "<link" not in rendered
    assert "сохранённый пример" in rendered
    assert "не фактическая прибыль" in rendered
