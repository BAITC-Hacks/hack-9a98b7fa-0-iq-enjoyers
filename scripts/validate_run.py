"""Independently validate our output against public CSVs, not organizer internals."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

PRICES = {"push": 0, "sms": 4, "digital_ads": 22, "call": 160}
FILTERS = ("current_tariff", "arpu_segment", "data_segment", "call_segment")


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def number(value):
    value = float(value)
    require(math.isfinite(value), "Non-finite numeric output")
    return value


def normalize(campaign):
    def clean(value):
        return None if value is None or str(value) in {"", "None", "nan"} else str(value)
    return tuple(clean(campaign.get(key)) for key in
                 ("campaign_name", "target_tariff", "channel", *("filter_" + f for f in FILTERS)))


def validate_run(work: Path, traces_dir: Path, expected_runs=1, submission=False):
    require(expected_runs >= 1 and (not submission or expected_runs == 1), "Invalid expected run count")
    with (work / "customer_profile.csv").open(encoding="utf-8-sig", newline="") as source:
        profile = list(csv.DictReader(source))
    with (work / "tariff_dictionary.csv").open(encoding="utf-8-sig", newline="") as source:
        tariffs = {row["tariff_plan_code"] for row in csv.DictReader(source)}
    require(len({r["ID_NUMBER"] for r in profile}) == len(profile), "Duplicate profile IDs")
    paths = sorted(traces_dir.glob("trace-*.json"))
    require(len(paths) == expected_runs, "Missing or unexpected agent traces")
    audience_cache = {}

    def audience(campaign):
        require(campaign.get("target_tariff") in tariffs, "Unknown target tariff")
        require(campaign.get("channel") in PRICES, "Unknown channel")
        filters = tuple((field, str(campaign["filter_" + field])) for field in FILTERS
                        if campaign.get("filter_" + field) not in (None, ""))
        if filters not in audience_cache:
            audience_cache[filters] = {row["ID_NUMBER"] for row in profile
                                      if all(row.get(field) in value.split(";") for field, value in filters)}
        return audience_cache[filters]

    totals, trace = [], None
    for path in paths:
        trace = json.loads(path.read_text(encoding="utf-8"))
        require(trace.get("status") == "ready", "Agent did not produce a ready plan")
        pilots, decisions, failures = trace["pilots"], trace["decisions"], trace.get("pilot_failures", [])
        require(1 <= len(pilots) <= 20, "At least one successful pilot is required")
        require(1 <= len(decisions) <= 10, "Expected 1..10 final campaigns")
        require(len(pilots) + len(failures) == trace["pilot_attempts"] <= 20, "Pilot request limit exceeded or mismatched")
        cost, contacts = 0.0, 0
        for pilot in pilots:
            n = pilot["n"]
            require(isinstance(n, int) and 10 <= n <= min(200, len(audience(pilot))), "Invalid pilot size")
            number(pilot["observed_lift_ratio"])
            expected = n * PRICES[pilot["channel"]]
            require(abs(expected - number(pilot["spent"])) < 1e-6, "Unexpected pilot cost")
            cost += expected
            contacts += n
        for failure in failures:
            spent, consumed = number(failure["spent"]), failure["contacts"]
            require(spent >= 0 and isinstance(consumed, int) and consumed >= 0, "Invalid failed-pilot resources")
            cost += spent
            contacts += consumed
        require(abs(cost + number(trace["remaining_budget_before_final"]) - 100000) < 1e-6, "Pilot budget accounting mismatch")
        require(contacts + trace["remaining_contacts_before_final"] == 15000, "Pilot contact accounting mismatch")
        used = set()
        final_cost, final_contacts = 0.0, 0
        for decision in decisions:
            ids = audience(decision)
            require(1 <= len(ids) <= 5000, "Invalid final campaign size")
            require(len(ids) == decision["customers"], "Final audience count mismatch")
            require(not used.intersection(ids), "Overlapping final campaigns")
            used.update(ids)
            expected = len(ids) * PRICES[decision["channel"]]
            require(abs(expected - number(decision["cost"])) < 1e-6, "Final cost mismatch")
            final_cost += expected
            final_contacts += len(ids)
        require(abs(final_cost - number(trace["final_cost"])) < 1e-6, "Final total cost mismatch")
        require(final_contacts == trace["final_contacts"], "Final total contacts mismatch")
        require(len(decisions) == trace["final_campaigns"], "Final campaign count mismatch")
        require(cost + final_cost <= 100000 + 1e-6, "Budget exceeded")
        require(contacts + final_contacts <= 15000, "Contact limit exceeded")
        require(0 <= number(trace["elapsed_seconds"]) <= 300, "Five-minute runtime limit exceeded")
        totals.append({"total_cost": cost + final_cost, "total_contacts": contacts + final_contacts,
                       "final_campaigns": len(decisions), "successful_pilots": len(pilots)})
    if submission:
        with (work / "submission.csv").open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        require(sorted(map(normalize, rows), key=str) == sorted(map(normalize, trace["decisions"]), key=str),
                "submission.csv differs from the validated plan")
    return {"valid": True, "checked_runs": len(totals),
            "max_total_cost": max(t["total_cost"] for t in totals),
            "max_total_contacts": max(t["total_contacts"] for t in totals),
            "max_final_campaigns": max(t["final_campaigns"] for t in totals),
            "submission_checked": submission}
