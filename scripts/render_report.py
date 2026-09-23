"""Self-contained offline report: no JS, remote fonts, server or browser upload."""
from __future__ import annotations

import html
import json
from pathlib import Path
import statistics


def report_from_run(folder):
    evaluation = json.loads((folder / "evaluation/manifest.json").read_text())
    submission = json.loads((folder / "submission/manifest.json").read_text())
    trace_files = list((folder / "submission/traces").glob("trace-*.json"))
    if len(trace_files) != 1:
        raise ValueError("Expected one submission trace")
    trace = json.loads(trace_files[0].read_text())
    scores = [r["net"] for r in evaluation.get("net_by_seed", [])]
    if not scores and evaluation.get("single_net") is not None:
        scores = [evaluation["single_net"]]
    return {"demo": False, "source": "Официальный локальный mock, не судейские эффекты",
            "scores": scores, "validation": evaluation["validation"],
            "submission_validation": submission["validation"], "settings": evaluation["settings"],
            "agent_sha256": evaluation["agent_sha256"], "trace": trace}


def render_report(report):
    esc = lambda value: html.escape(str(value), quote=True)
    fmt = lambda value: f"{value:,.0f}".replace(",", " ")
    trace, scores = report["trace"], report["scores"]
    llm = trace.get("llm", {"status": "disabled", "used": False})
    llm_label = {"accepted": "Локальная LLM: гипотезы приняты",
                 "fallback": "LLM недоступна/ответ отклонён — численный режим",
                 "disabled": "Численный режим — LLM выключена"}.get(llm["status"], "Статус LLM неизвестен")
    valid = report["validation"]["valid"] and report["submission_validation"]["valid"]
    cost = 100000 - trace["remaining_budget_before_final"] + trace["final_cost"]
    contacts = 15000 - trace["remaining_contacts_before_final"] + trace["final_contacts"]
    rows = []
    for decision in trace["decisions"]:
        filters = ", ".join(f"{key.removeprefix('filter_')}={value}" for key, value in decision.items()
                            if key.startswith("filter_") and value is not None)
        rows.append("<tr>" + "".join(f"<td>{esc(value)}</td>" for value in
                    (decision["campaign_name"], filters, decision["target_tariff"], decision["channel"],
                     fmt(decision["customers"]), fmt(decision["cost"]),
                     fmt(decision["incremental_net_proxy"]))) + "</tr>")
    pilot_rows = "".join("<tr>" + "".join(f"<td>{esc(value)}</td>" for value in
                         (i, p["target_tariff"], p["channel"], p["n"],
                          f'{p["observed_lift_ratio"]:.4f}', fmt(p["spent"]))) + "</tr>"
                         for i, p in enumerate(trace["pilots"], 1))
    demo = "ДЕМО · сохранённый пример, новый расчёт не выполнялся" if report.get("demo") else "РЕЗУЛЬТАТ ЛОКАЛЬНОГО ПРОГОНА"
    median = fmt(statistics.median(scores)) if scores else "нет оценки"
    score_range = f"{fmt(min(scores))} … {fmt(max(scores))}" if scores else "нет оценки"
    return f'''<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>ProfitPilot — план тарифных кампаний</title>
<style>
:root{{font-family:system-ui,sans-serif;color:#edf2f7;background:#111719;line-height:1.55}}
body{{max-width:1180px;margin:0 auto;padding:32px 24px}}h1{{font-size:clamp(32px,5vw,52px);margin:8px 0}}
h2{{margin-top:32px}}.tag{{color:#ffe066;font-size:13px;letter-spacing:.08em}}
.muted,small{{color:#adbcc2}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}}
.card,section,details{{background:#1b2529;border:1px solid #334448;border-radius:12px;padding:20px;margin-top:14px}}
.card strong{{display:block;font-size:30px;color:#ffe066}}.notice{{border-left:4px solid #ffe066;padding:12px 16px;background:#26291e}}
.table{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{text-align:left;padding:10px;border-bottom:1px solid #334448;vertical-align:top}}th{{color:#ffe066}}
meter{{width:100%;height:20px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}
summary{{cursor:pointer;font-weight:600}}footer{{padding:24px 0;color:#adbcc2}}a{{color:#ffe066}}
@media print{{:root{{color:#111;background:white}}.card,section,details{{background:white;color:#111}}}}
</style></head><body>
<div class="tag">0 IQ ENJOYERS / HACKALEM · {esc(demo)}</div>
<h1>ProfitPilot</h1><p class="muted">Сначала проверить гипотезы. Затем распределить бюджет. Каждый шаг — в журнале.</p>
<p class="notice">{esc(report['source'])}. Техническая корректность не гарантирует прибыль или баллы.</p>
<div class="cards">
<div class="card">Независимая проверка<strong>{'PASS' if valid else 'FAIL'}</strong><small>Пилоты, бюджет, охват, CSV</small></div>
<div class="card">Медианный net<strong>{median}</strong><small>по {len(scores)} сценариям оценки</small></div>
<div class="card">Прибыльные сценарии<strong>{sum(v > 0 for v in scores)} / {len(scores)}</strong><small>Диапазон: {score_range}</small></div>
<div class="card">Финальный план<strong>{len(trace['decisions'])} / 10</strong><small>{len(trace['pilots'])} успешных пилотов; сбоев: {len(trace.get('pilot_failures', []))}</small></div>
</div>
<section><h2>Ресурсы плана для submission.csv</h2>
<p>Бюджет с пилотами: <strong>{fmt(cost)} / 100 000</strong></p><meter min="0" max="100000" value="{cost}"></meter>
<p>Контакты с пилотами: <strong>{fmt(contacts)} / 15 000</strong></p><meter min="0" max="15000" value="{contacts}"></meter>
<p class="muted">Это план отдельного запуска сборки CSV. Медиана выше относится к серии оценочных запусков.</p></section>
<section><h2>Кампании</h2><div class="table"><table><thead><tr><th>Кампания</th><th>Аудитория</th><th>Тариф</th><th>Канал</th><th>Абоненты</th><th>Стоимость</th><th>Оценка net*</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div><p class="muted">*Консервативная рабочая оценка алгоритма, а не фактическая прибыль кампании и не доверительный интервал.</p></section>
<section><h2>Роль модели</h2><p>{esc(llm_label)}</p>
<p>{esc(llm.get('rationale', 'Без внешних API. Итоговые расходы и допустимость плана рассчитывает код.'))}</p>
<p class="muted">Текст LLM — непроверенное объяснение приоритета. Модель не задаёт эффект, бюджет или итоговый результат.</p></section>
<details><summary>Пилоты: проверяемая история решений</summary><div class="table"><table><thead><tr><th>№</th><th>Тариф</th><th>Канал</th><th>Размер</th><th>Наблюдаемый эффект</th><th>Стоимость</th></tr></thead><tbody>{pilot_rows}</tbody></table></div>
<pre>{esc(json.dumps(trace.get('pilot_failures', []), ensure_ascii=False, indent=2))}</pre></details>
<details><summary>Настройки, происхождение и ограничения</summary>
<pre>{esc(json.dumps(report['settings'], ensure_ascii=False, indent=2))}</pre>
<p>SHA256 агента: <code>{esc(report['agent_sha256'])}</code></p>
<ul><li>История относится к другой аудитории; до/после — не причинный uplift.</li>
<li>Шум пилотов может привести к убытку; судейские эффекты отличаются от mock.</li>
<li>LLM-режим экспериментальный, его преимущество требует отдельного сравнения.</li></ul></details>
<footer>Самодостаточный локальный HTML. Нет аналитики, внешних скриптов или отправки данных. Сдачу выполняет участник.</footer>
</body></html>'''
