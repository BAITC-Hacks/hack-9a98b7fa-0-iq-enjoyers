#!/usr/bin/env python3
"""One-command setup, offline tests, official evaluation and submission packaging."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import venv
import zipfile

ROOT = Path(__file__).resolve().parent


def safe_env():
    result = {key: os.environ[key] for key in
              ("PATH", "LANG", "LC_ALL", "TZ", "SYSTEMROOT", "WINDIR", "TMP", "TEMP", "TMPDIR")
              if key in os.environ}
    result.update({"PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "PIP_CONFIG_FILE": os.devnull,
                   "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
    return result


def choose_bundle(explicit=None, search_dirs=None):
    if explicit:
        result = Path(explicit).expanduser().resolve()
        if not result.is_file():
            raise ValueError(f"ZIP не найден: {result}. Укажите путь через --bundle.")
        return result
    roots = search_dirs if search_dirs is not None else [ROOT, ROOT.parent, Path.home() / "Downloads"]
    found = sorted({path.resolve() for root in roots for path in Path(root).glob("beeline_case_participants*.zip") if path.is_file()})
    if len(found) != 1:
        raise ValueError("Нужен ZIP организатора. Найдено архивов: " + str(len(found))
                         + '. Укажите нужный: --bundle "/путь/beeline_case_participants.zip"')
    return found[0]


def inspect_bundle(path):
    from scripts.run_case import FILES
    with zipfile.ZipFile(path) as archive:
        files = [info for info in archive.infolist() if not info.is_dir()]
        names = [info.filename for info in files]
        if len(set(names)) != len(names) or set(names) != FILES:
            raise ValueError("Состав ZIP отличается от проверенного пакета Beeline; исходники менять не нужно — проверьте архив.")
        if sum(info.file_size for info in files) > 100_000_000:
            raise ValueError("ZIP превышает допустимый размер")


def expected_versions(path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            result.update(expected_versions(path.parent / line[3:].strip()))
        else:
            name, version = line.split("==", 1)
            result[name] = version
    return result


def execute(command, timeout=300):
    subprocess.run([str(part) for part in command], cwd=ROOT, env=safe_env(), check=True, timeout=timeout)


def prepare_environment(offline=False):
    folder = ROOT / ".venv-agent"
    python = folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        if offline:
            raise ValueError("Offline-окружение ещё не подготовлено. Сначала выполните python3 run.py --prepare-only при доступном интернете.")
        print("[1/4] Создаю отдельное окружение .venv-agent (системные пакеты не меняются)", flush=True)
        venv.EnvBuilder(with_pip=True).create(folder)
    expected = expected_versions(ROOT / "requirements-dev.txt")
    probe = subprocess.run([str(python), "-c",
                            "import importlib.metadata as m,json,sys; "
                            "print(json.dumps({k:m.version(k) for k in json.loads(sys.argv[1])}))",
                            json.dumps(list(expected))], cwd=ROOT, env=safe_env(), capture_output=True, text=True)
    try:
        ready = probe.returncode == 0 and json.loads(probe.stdout) == expected
    except ValueError:
        ready = False
    if not ready:
        if offline:
            raise ValueError("Зависимости не подготовлены или отличаются от зафиксированных. Выполните run.py --prepare-only с интернетом.")
        print("[1/4] Устанавливаю зафиксированные зависимости из PyPI", flush=True)
        execute([python, "-m", "pip", "install", "--index-url", "https://pypi.org/simple",
                 "-r", ROOT / "requirements-dev.txt"], timeout=300)
    execute([python, "-m", "pip", "check"], timeout=30)
    return python


def main():
    from scripts.run_case import add_strategy_arguments, strategy_settings
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bundle", type=Path, help="organizer ZIP; otherwise detect in project/Downloads")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--offline", action="store_true", help="never install or download dependencies")
    parser.add_argument("--prepare-only", action="store_true", help="install dependencies before going offline")
    parser.add_argument("--doctor", action="store_true", help="read-only readiness check, no installs or submission")
    parser.add_argument("--demo", action="store_true", help="openable offline HTML example; no ZIP, packages or LLM needed")
    add_strategy_arguments(parser)
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error("Требуется Python 3.11 или новее; рекомендуется Python 3.13.")
    if not 1 <= args.runs <= 100:
        parser.error("--runs должен быть от 1 до 100")
    settings = strategy_settings(args)
    if sum((args.doctor, args.demo, args.prepare_only)) > 1:
        parser.error("Выберите только один из --doctor / --demo / --prepare-only")
    if args.doctor:
        from scripts.doctor import diagnose
        report = diagnose(ROOT, args.bundle)
        for check in report["checks"]:
            print(f"[{'OK' if check['ok'] else 'TODO'}] {check['check']}: {check['detail']}")
        print(f"Ветка: {report['branch']}; Ollama установлен: {report['ollama_installed']}")
        for note in report["notes"]:
            print(note)
        raise SystemExit(0 if report["core_ready"] else 2)
    if args.demo:
        from scripts.render_report import render_report
        report = json.loads((ROOT / "demo/example_report.json").read_text(encoding="utf-8"))
        folder = ROOT / "artifacts" / ("demo-" + uuid.uuid4().hex[:8])
        folder.mkdir(parents=True)
        destination = folder / "report.html"
        destination.write_text(render_report(report), encoding="utf-8")
        print(f"Демо сохранённого результата (не новый прогон). Откройте: {destination}")
        return
    bundle = None if args.prepare_only else choose_bundle(args.bundle)
    if bundle:
        inspect_bundle(bundle)
    python = prepare_environment(args.offline)
    if args.prepare_only:
        print("Окружение готово. Для запуска без сети: python3 run.py --offline --bundle /path/to/case.zip")
        return
    run_dir = ROOT / "artifacts" / ("ready-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True)
    print("[2/4] Проверяю регрессионные тесты", flush=True)
    execute([python, "-m", "pytest", "tests", "-q"], timeout=120)
    print(f"[3/4] Официальная оценка: {args.runs} запусков", flush=True)
    strategy_cli = [part for key, value in settings.items() for part in ("--" + key.replace("_", "-"), str(value))]
    execute([python, "scripts/run_case.py", "--bundle", bundle, "--runs", args.runs,
             "--output-dir", run_dir / "evaluation", *strategy_cli], timeout=300 * args.runs + 30)
    print("[4/4] Генерирую и проверяю submission.csv", flush=True)
    execute([python, "scripts/run_case.py", "--bundle", bundle, "--make-submission",
             "--output-dir", run_dir / "submission", *strategy_cli], timeout=330)
    evaluation = json.loads((run_dir / "evaluation/manifest.json").read_text())
    submission = json.loads((run_dir / "submission/manifest.json").read_text())
    if not evaluation.get("validation", {}).get("valid") or not submission.get("validation", {}).get("valid"):
        raise ValueError("Артефакты не прошли независимую проверку; пакет для сдачи не создан")
    package = run_dir / "submit"
    package.mkdir()
    for source, name in ((ROOT / "agent.py", "agent.py"),
                         (ROOT / "requirements-agent.txt", "requirements.txt"),
                         (ROOT / "THIRD_PARTY.md", "THIRD_PARTY.md"),
                         (run_dir / "submission/bundle/profitpilot_config.json", "profitpilot_config.json"),
                         (run_dir / "submission/bundle/submission.csv", "submission.csv")):
        shutil.copyfile(source, package / name)
    (package / "START_HERE.txt").write_text(
        "ProfitPilot / HackAlem Beeline\n"
        "Python >=3.11 (tested on 3.13).\n"
        "Install: python3 -m pip install -r requirements.txt\n"
        "Place agent.py and profitpilot_config.json into the organizer bundle; keep its public data/ files.\n"
        "Run: python3 local_eval.py\n"
        "Generate again: python3 make_submission.py\n"
        "No API keys or external API required. Optional LLM needs local Ollama weights prepared in advance.\n"
        "If the local LLM is unavailable, numerical fallback applies and may change the generated plan.\n"
        "Do not publish organizer data. This script never submits or uploads the package.\n",
        encoding="utf-8")
    scores = [r["net"] for r in evaluation.get("net_by_seed", [])]
    if not scores and evaluation.get("single_net") is not None:
        scores = [evaluation["single_net"]]
    from scripts.render_report import report_from_run, render_report
    report = report_from_run(run_dir)
    (run_dir / "report.html").write_text(render_report(report), encoding="utf-8")
    summary = {"technical_checks_passed": True, "mode": args.mode, "settings": settings, "offline": args.offline,
               "evaluation": evaluation["validation"], "submission": submission["validation"],
               "positive_runs": sum(v > 0 for v in scores), "scored_runs": len(scores),
               "profit_guaranteed": False,
               "submission_llm": report["trace"].get("llm"),
               "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in package.iterdir()}}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nГОТОВО. Файлы для сдачи: {package}")
    print(f"Проверки и отчёты: {run_dir / 'summary.json'}")
    print(f"Наглядный локальный отчёт: {run_dir / 'report.html'}")
    if args.llm_model and not report["trace"].get("llm", {}).get("used"):
        print("LLM НЕ использована: выполнен численный fallback. Подробности в журнале; это не тест качества модели.")
    if any(v < 0 for v in scores):
        print("Внимание: есть убыточные сценарии. Техническая корректность не означает гарантии прибыли.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError, zipfile.BadZipFile) as error:
        print(f"Запуск не завершён: {error}", file=sys.stderr)
        print("Код править не требуется. Проверьте путь ZIP, версию Python и установку зависимостей по README.", file=sys.stderr)
        raise SystemExit(2)
