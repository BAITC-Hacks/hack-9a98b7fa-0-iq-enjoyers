"""Reject staged/tracked secrets and private organizer artifacts; never print secrets."""
from __future__ import annotations

import argparse
from pathlib import PurePosixPath
import re
import subprocess


PATTERNS = {
    "credential-like value": re.compile(rb"(?:sk-proj-|github_pat_|gh[pousr]_|nvapi-|bak-)[A-Za-z0-9_-]{18,}"),
    "private key": re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"),
}


def check_path(name):
    path = PurePosixPath(name)
    if path.name.startswith(".env") and path.name != ".env.example":
        return "local environment file"
    if any(part in {"artifacts", "data", ".venv", ".venv-agent", "__pycache__", "organizer_questions"} for part in path.parts):
        return "private/generated directory"
    if path.suffix.lower() in {".zip", ".pem", ".key", ".csv"}:
        return "raw data/archive/key file"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracked", action="store_true", help="scan all tracked files (for CI)")
    args = parser.parse_args()
    command = ["git", "ls-files", "-z"] if args.tracked else ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]
    names = subprocess.check_output(command).decode().split("\0")
    problems, checked = [], 0
    for name in filter(None, names):
        checked += 1
        reason = check_path(name)
        if reason:
            problems.append(f"{name}: {reason}")
            continue
        content = subprocess.check_output(["git", "show", f":{name}"])
        for label, pattern in PATTERNS.items():
            if pattern.search(content):
                problems.append(f"{name}: {label} (value redacted)")
    if problems:
        print("Publication check FAILED:\n" + "\n".join(problems))
        raise SystemExit(1)
    print(f"Publication check passed: {checked} files; no matched credentials or forbidden paths.")
    print("Pattern scan is a guardrail, not a complete security guarantee.")


if __name__ == "__main__":
    main()
