"""Read-only preflight. Does not install packages, request credentials or modify Git."""
import json
import shutil
import subprocess
import sys


def diagnose(root, bundle=None):
    import run
    checks = []
    def record(name, ok, detail):
        checks.append({"check": name, "ok": ok, "detail": detail})
    record("Python", sys.version_info >= (3, 11), sys.version.split()[0])
    try:
        source = run.choose_bundle(bundle)
        run.inspect_bundle(source)
        record("Пакет Beeline", True, str(source))
    except (ValueError, OSError) as error:
        record("Пакет Beeline", False, str(error))
    python = root / ".venv-agent" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    expected = run.expected_versions(root / "requirements-dev.txt")
    if python.is_file():
        result = subprocess.run([str(python), "-c", "import importlib.metadata as m,json,sys; print(json.dumps({k:m.version(k) for k in json.loads(sys.argv[1])}))", json.dumps(list(expected))],
                                capture_output=True, text=True, env=run.safe_env(), timeout=15)
        try:
            ready = result.returncode == 0 and json.loads(result.stdout) == expected
        except ValueError:
            ready = False
    else:
        ready = False
    record("Зависимости", ready, "Готовы" if ready else "Подготовьте: python3 run.py --prepare-only")
    git = subprocess.run(["git", "branch", "--show-current"], cwd=root, capture_output=True, text=True, timeout=5) if shutil.which("git") else None
    branch = git.stdout.strip() if git and git.returncode == 0 else "Git недоступен / архив исходников"
    return {"core_ready": all(item["ok"] for item in checks), "checks": checks,
            "branch": branch, "ollama_installed": shutil.which("ollama") is not None,
            "notes": ["LLM необязательна; наличие Ollama не подтверждает наличие весов или качество модели.",
                      "Для сдачи проверьте, что жюри увидит нужную ветку. Doctor не меняет main и не выполняет push.",
                      "Это диагностика запуска, не судейская оценка и не гарантия баллов."]}
