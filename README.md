# ProfitPilot — агент тарифных кампаний Beeline

Команда **0 iq enjoyers** · репозиторий `hack-9a98b7fa-0-iq-enjoyers`.

Рабочий offline-MVP для HackAlem: изучает разрешённую историю, проводит пилоты
через публичное API среды и возвращает 1–10 кампаний с ограничениями бюджета
и контактов. Все данные кейса синтетические.

## Проверенный результат

| Вариант | Медианный чистый результат, у.е. | Положительные прогоны |
|---|---:|---:|
| Официальный шаблон | −384 911,5 | 0/30 |
| Наш основной режим `fixed` | +593 626 | 30/30 |
| Экспериментальный `adaptive` | +261 064,5 | 19/30 |

Это результаты локального mock, не прогноз реальной прибыли и не гарантия
судейского балла. Расширенная проверка после исправлений: **41 тест прошёл,
100/100 планов допустимы, 99/100 прогонов прибыльны**; минимум −39,
медиана +586 957. [Проверка надёжности](reports/reliability.md).
Ранний сравнительный эксперимент сохранён отдельно:
[методика и ограничения](reports/benchmark.md), [результаты с хешами](reports/benchmark.json).

## Запуск без API-ключей

Требуется Python ≥3.11 с модулем `venv`; локально проверено на Python 3.13/Linux.
Зависимости зафиксированы. Пакет организатора не включён в GitHub: скачайте
его из материалов кейса в Downloads. Если найден ровно один подходящий ZIP,
достаточно из корня репозитория выполнить:

```bash
python3 run.py
```

Скрипт создаёт отдельное `.venv-agent`, устанавливает зависимости, выполняет
тесты, 10 официальных прогонов и собирает `artifacts/ready-*/submit/`:
`agent.py`, `submission.csv`, `requirements.txt`, происхождение кода и инструкцию.
Исходники править не нужно. Первый запуск требует интернета **для установки**;
сам агент и тесты не используют LLM или API-ключи. Архив с пробелами в пути:

```bash
python3 run.py --bundle "/path/to/beeline_case_participants (1).zip" --runs 100
```

Для машины без интернета сначала подготовьте окружение с доступом к PyPI,
затем запускайте в том же окружении без установки пакетов:

```bash
python3 run.py --prepare-only
python3 run.py --offline --bundle "/path/to/beeline_case_participants.zip"
```

`--offline` запрещает установку/скачивание зависимостей, но не является
сетевой песочницей. Код агента не содержит сетевых вызовов.
При нескольких ZIP скрипт попросит указать нужный через `--bundle`.
При отсутствии Python/venv их нужно установить средствами ОС; скрипт
не меняет системные пакеты и не запрашивает пароль администратора.

В `evaluation/` и `submission/` сохраняются `output.txt`, `manifest.json`,
журналы `traces/*.json` и распакованный `bundle/`. `summary.json` содержит
итог проверок и хеши файлов для сдачи. CSV создаёт официальная программа.
Независимый валидатор пересчитывает аудитории, стоимость, охват и лимиты
по публичным CSV. Одного кода завершения 0 недостаточно для отметки «готово».
Runner копирует пакет без изменений и не передаёт процессу ключи.

Для отдельных прогонов доступен `scripts/run_case.py --bundle ... --runs 10`
(запускайте Python из `.venv-agent/bin/python`, на Windows — `Scripts/python.exe`).

Для запуска непосредственно в официальном пакете положите туда наш `agent.py`,
установите `requirements-agent.txt`, затем выполните:

```bash
python3 local_eval.py
python3 local_eval.py --runs 10
python3 make_submission.py
```

## Как работает агент

1. Очищает историю и получает слабые предварительные оценки переходов.
2. Формирует непересекающиеся ячейки текущего тарифа и ARPU-сегмента.
   Ячейки >5000 делит разрешёнными `data_segment`/`call_segment`, не списками ID.
3. Проводит небольшой резервный пилот, затем проверяет тарифные гипотезы;
   до 20 пилотов, до 200 человек, не более 25% исходного бюджета на разведку.
4. Обновляет оценки по фактическим результатам разрешённых пилотов.
5. Выбирает кампании через SciPy/HiGHS: бюджет, контакты, 1–10 кампаний,
   не более одного итогового предложения на ячейку. Есть проверенный жадный
   fallback при отказе/тайм-ауте решателя.
6. Сохраняет журнал пилотов и решений, если задан `PROFITPILOT_TRACE_DIR`.

Временный сбой пилота повторяется только если публичные счётчики не изменились:
до трёх попыток на действие и до 20 запросов суммарно. Списанные при сбое
ресурсы учитываются. Если ни один пригодный пилот не завершился, агент выдаёт
явную ошибку, а не некорректную «готовую» стратегию. Постоянный отказ среды
или аудиторию, невыразимую разрешёнными фильтрами, невозможно исправить обещанием
автоматического запуска. Эти случаи также покрыты тестами.

По умолчанию `fixed`: порядок исследования предопределён, но итоговый план
зависит от результатов пилотов. `adaptive` использует трёхсценарное приближение
пользы следующего пилота для плана. Оно пока уступает основному режиму:

```bash
python3 scripts/run_case.py --bundle /path/to/beeline_case_participants.zip --mode adaptive --runs 30
```

LLM в численном контуре нет. Руководство кейса допускает LLM, но противоречие
«10 минут с LLM / 5 минут без интернета» ещё требует ответа организатора.
Текущая реализация работает локально без ключей. Интерпретацию
`observed_lift_ratio` и правила использования готовых компонентов также нужно
подтвердить; известные предположения перечислены в отчёте тестов.

## Наш вклад и сторонние компоненты

[THIRD_PARTY.md](THIRD_PARTY.md) перечисляет использованные библиотеки,
лицензии, рассмотренные GitHub-проекты и происхождение кода. Численный решатель
создан SciPy/HiGHS; наша часть — логика кандидатов, пилотов, ограничений,
оценок и отчётности. Разработка выполнена с помощью Codex.

Не публикуйте `.env`, ключи, исходные CSV и внутренние файлы организатора.
Перед коммитом: `python3 scripts/check_publish.py`. GitHub Actions подготовлен
для unit-тестов и проверки отслеживаемых файлов; для запуска ему не нужны
секреты или данные организатора. Результат CI проверяйте во вкладке Actions;
локальные проверки перечислены в отчёте тестов.

## Исследования

- [Кейс, архитектура и источники](beeline_research/README_RU.md).
- [Сравнение с компаниями и научными работами](beeline_research/competitive_landscape_ru.md).
- [Аудит публичных данных](beeline_research/downloads_review_ru.md).
- [Роли LLM и табличных моделей](beeline_research/model_selection_ru.md).

## Архив: ранний общий поиск telecom-идей

Следующий инструмент предшествовал выбору кейса Beeline. Он не является
конкурсным агентом и не нужен для теста выше.

This tool ranks ten agentic telecom MVPs and enriches every idea with reusable public
GitHub repositories. It can also ask NVIDIA-hosted GLM-5.3 to critique the ranking.

## Quick start

The GitHub-only offline report needs no Python packages or API keys:

```bash
python3 telecom_idea_research.py --offline
```

For GLM-5.3 analysis:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-research.txt
export NVIDIA_API_KEY="your-key"
export GITHUB_TOKEN="your-optional-token"
python3 telecom_idea_research.py
```

Do not put real keys in `.env.example` or commit them to Git.

Reports are written to:

- `research_output/top_10_ideas.md`
- `research_output/top_10_ideas.json`

## LLM usage

GLM-5.3 is used here only as a research/review model. The supplied Beeline case
guide treats an LLM as optional. Confirm any separate event-wide technology
requirements with the organizers; do not conflate them with this case guide.

## Earlier generic idea (not the selected Beeline case)

The earlier idea was **Agentic NOC Incident Commander**: give an agent mock tools for alarms,
KPIs, topology, remediation simulation and ticket creation. Keep a human approval gate
before a network-changing action. This gives the judges a complete and visible agent loop:

`observe -> investigate -> decide -> request approval -> act -> verify`
