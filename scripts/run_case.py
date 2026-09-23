"""Run the unmodified organizer harness in a fresh, ignored directory.

Organizer internals are staged as opaque bytes, never inspected by the agent.
Only agent.py and the supplied bundle enter the run; credentials do not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "scoring_core.py", "make_submission.py", "customer_profile.csv",
    "tariff_dictionary.csv", "feature_dictionary.csv", "local_eval.py",
    "environment.py", "mock_environment.py", "PARTICIPANT_GUIDE.md",
    "PARTICIPANT_GUIDE.pdf", "data/dict_tariff.csv", "data/traffic.csv",
    "data/arpu_monthly.csv", "data/change_tariff.csv", "agent_template.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--agent", type=Path, default=ROOT / "agent.py")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--mode", choices=["adaptive", "fixed"], default="fixed")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--make-submission", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="new directory for this run (must not exist)")
    args = parser.parse_args()
    if not 1 <= args.runs <= 100:
        parser.error("--runs must be between 1 and 100")
    if not args.bundle.is_file():
        parser.error("Bundle not found. Supply the organizer ZIP with --bundle; no source-code edits are needed.")
    if not args.baseline and not args.agent.is_file():
        parser.error("Agent file not found")
    run_dir = (args.output_dir or (ROOT / "artifacts" / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]))).resolve()
    if run_dir.exists():
        parser.error("Output directory already exists; existing results will not be overwritten")
    work = run_dir / "bundle"
    work.mkdir(parents=True)
    with zipfile.ZipFile(args.bundle) as archive:
        entries = [info for info in archive.infolist() if not info.is_dir()]
        names = [info.filename for info in entries]
        if len(set(names)) != len(names) or set(names) != FILES:
            raise ValueError("Unexpected bundle contents; inspect the public file manifest first")
        if sum(info.file_size for info in entries) > 100_000_000:
            raise ValueError("Bundle exceeds size limit")
        for info in entries:
            destination = work / info.filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Fixed allowlist prevents traversal, links and unknown executable files.
            destination.write_bytes(archive.read(info))
    source = work / "agent_template.py" if args.baseline else args.agent.resolve()
    shutil.copyfile(source, work / "agent.py")
    manifest = {
        "bundle_sha256": hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
        "agent_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "baseline": args.baseline, "mode": args.mode, "runs": args.runs,
        "python": sys.version, "status": "prepared", "workdir": str(work),
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Run directory: {run_dir}", flush=True)
    if args.prepare_only:
        return
    # No inherited API keys, auth tokens, .env or shell configuration.
    child_env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "TZ") if key in os.environ}
    child_env.update({"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                      "PROFITPILOT_MODE": args.mode, "PROFITPILOT_TRACE_DIR": str(run_dir / "traces")})
    command = [sys.executable, "make_submission.py"] if args.make_submission else [sys.executable, "local_eval.py"]
    if not args.make_submission and args.runs > 1:
        command += ["--runs", str(args.runs)]
    started = time.monotonic()
    try:
        result = subprocess.run(command, cwd=work, env=child_env, capture_output=True,
                                text=True, timeout=300 * args.runs)
        output = result.stdout + result.stderr
        code = result.returncode
    except subprocess.TimeoutExpired as error:
        output = "Official harness exceeded the outer safety timeout.\n"
        for part in (error.stdout, error.stderr):
            if part:
                output += part.decode(errors="replace") if isinstance(part, bytes) else part
        code = 124
    (run_dir / "output.txt").write_text(output, encoding="utf-8")
    manifest.update({"status": "finished", "returncode": code,
                     "elapsed_seconds": round(time.monotonic() - started, 3), "command": command})
    # Exit 0 alone does not mean profit or a successful contest submission.
    manifest["evaluation_status"] = re.findall(r"Статус:\s*(\w+)", output)
    manifest["rejected_campaign_warning"] = bool(re.search(r"Кампания[^\n]*отброшена", output))
    manifest["net_by_seed"] = [{"seed": int(seed), "net": int(net.replace(",", ""))}
                                for seed, net in re.findall(r"seed\s+(\d+):\s*чистый результат\s+(-?[\d,]+)", output)]
    single_net = re.search(r"ЧИСТЫЙ РЕЗУЛЬТАТ \(net\):\s*(-?[\d,]+)", output)
    manifest["single_net"] = int(single_net.group(1).replace(",", "")) if single_net else None
    if code == 0 and not args.baseline:
        try:
            if __package__:
                from .validate_run import validate_run
            else:
                from validate_run import validate_run
            manifest["validation"] = validate_run(work, run_dir / "traces",
                                                  1 if args.make_submission else args.runs,
                                                  submission=args.make_submission)
            if manifest["rejected_campaign_warning"]:
                raise ValueError("Official evaluator rejected a campaign")
        except (ValueError, OSError, KeyError, TypeError) as error:
            code = 3
            manifest["validation"] = {"valid": False, "error": str(error)}
    manifest["returncode"] = code
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output)
    if manifest.get("validation", {}).get("valid") is False:
        print(f"Independent validation FAILED: {manifest['validation']['error']}")
    print(f"Exit: {code}; elapsed: {manifest['elapsed_seconds']}s; report: {manifest_path}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
