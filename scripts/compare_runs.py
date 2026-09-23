"""Aggregate official printed scores, keeping development and unseen seeds separate."""
import argparse
import json
from pathlib import Path
import re
import statistics


def summarize(path):
    manifest = json.loads((path / "manifest.json").read_text())
    output = (path / "output.txt").read_text()
    pairs = [(int(seed), int(net.replace(",", ""))) for seed, net in
             re.findall(r"seed\s+(\d+):\s*чистый результат\s+(-?[\d,]+)", output)]
    if manifest["returncode"] or len(pairs) != manifest["runs"] or len(dict(pairs)) != len(pairs):
        raise ValueError(f"Incomplete run: {path.name}")
    traces = [json.loads(p.read_text()) for p in sorted((path / "traces").glob("*.json"))]
    if not manifest["baseline"]:
        assert len(traces) == len(pairs)
        for trace in traces:
            assert 1 <= len(trace["pilots"]) <= 20
            assert all(10 <= p["n"] <= 200 for p in trace["pilots"])
            assert 1 <= trace["final_campaigns"] <= 10
            assert all(1 <= d["customers"] <= 5000 for d in trace["decisions"])
            assert trace["final_cost"] <= trace["remaining_budget_before_final"] + 1e-6
            assert trace["final_contacts"] <= trace["remaining_contacts_before_final"]
            assert "pilot_error" not in trace
    result = {"run": path.name, "agent_sha256": manifest["agent_sha256"],
              "bundle_sha256": manifest["bundle_sha256"], "runtime_seconds": manifest["elapsed_seconds"],
              "validated_agent_traces": len(traces), "scores": dict(pairs)}
    for label, subset in (("development_0_9", [v for s, v in pairs if s < 10]),
                          ("holdout_10_plus", [v for s, v in pairs if s >= 10]),
                          ("all", [v for _, v in pairs])):
        if subset:
            result[label] = {"n": len(subset), "mean": round(statistics.mean(subset), 1),
                             "median": statistics.median(subset), "minimum": min(subset),
                             "maximum": max(subset), "positive": sum(v > 0 for v in subset)}
    if traces:
        result["max_agent_seconds"] = max(t["elapsed_seconds"] for t in traces)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    results = [summarize(path) for path in args.runs]
    if len({tuple(result["scores"]) for result in results}) != 1:
        raise ValueError("Runs must use identical seed lists")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
