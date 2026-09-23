#!/usr/bin/env python3
"""Rank telecom hackathon ideas and enrich them with public GitHub projects.

The script is useful before implementation starts:
1. It searches GitHub for reusable open-source building blocks.
2. It can ask NVIDIA-hosted GLM-5.3 to critique and rerank the ideas.
3. It writes both a Markdown report and machine-readable JSON.

Secrets are read only from environment variables:
  NVIDIA_API_KEY  required for GLM-5.3 analysis
  GITHUB_TOKEN    optional, but recommended for higher GitHub API limits
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "z-ai/glm-5.3"

SOURCES = [
    {
        "title": "HackAlem AI — official requirements",
        "url": "https://hackalem.ai/",
        "note": "Agentic AI, offline event, teams up to three, Codex is mandatory.",
    },
    {
        "title": "AI & Digital Bridge 2026 programme",
        "url": "https://digitalbridge.ai/en/aimonth/",
        "note": "Five-hour build, ten sectors, Codex and OpenAI API are mandatory.",
    },
    {
        "title": "Ericsson — Agentic AI for autonomous telecom networks",
        "url": "https://www.ericsson.com/en/reports-and-papers/white-papers/adapting-agentic-ai",
        "note": "Closed-loop operations, incident resolution and human oversight.",
    },
    {
        "title": "GSMA — Agentic Network for Mobile AI Era",
        "url": "https://www.gsma.com/solutions-and-impact/technologies/networks/gsma_resources/agentic-network-for-mobile-ai-era/",
        "note": "AI agents for O&M, deterministic SLA and energy efficiency.",
    },
    {
        "title": "TM Forum — AI-native intelligent operations",
        "url": "https://inform.tmforum.org/research-and-analysis/reports/new-generation-intelligent-operations-an-ai-native-reinvention",
        "note": "Agents as digital employees for resilient network operations.",
    },
    {
        "title": "IETF draft — Network Digital Twin and Agentic AI",
        "url": "https://datatracker.ietf.org/doc/draft-wmz-nmrg-agent-ndt-arch/",
        "note": "Testing proposed network actions in a risk-free digital twin.",
    },
]


@dataclass(frozen=True)
class Idea:
    rank: int
    name: str
    score: float
    problem: str
    five_hour_demo: str
    agent_tools: list[str]
    extension: str
    github_query: str


IDEAS = [
    Idea(
        1,
        "Agentic NOC Incident Commander",
        9.7,
        "Engineers lose time correlating alarms, KPIs, topology and customer tickets.",
        "Inject a base-station incident; the agent gathers evidence, finds a root cause, simulates a fix, requests approval and opens a ticket.",
        ["get_alarms", "get_kpis", "get_topology", "simulate_fix", "create_ticket"],
        "Connect Prometheus/Grafana, NetBox, ServiceNow or a real operator OSS.",
        "network monitoring incident observability",
    ),
    Idea(
        2,
        "Customer Impact & Outage Communication Agent",
        9.2,
        "Network teams see alarms, while customers experience an unexplained loss of service.",
        "Map a simulated outage to affected subscribers, prioritize vulnerable users and create personalized Kazakh/Russian status messages.",
        ["detect_outage", "find_affected_users", "estimate_eta", "draft_notification"],
        "Add CRM, geospatial coverage, contact-center and notification integrations.",
        "network outage map status monitoring",
    ),
    Idea(
        3,
        "Enterprise SLA Assurance Agent",
        9.0,
        "B2B customers need proactive protection of latency, availability and throughput SLAs.",
        "Monitor synthetic KPIs, predict an SLA breach, explain the cause and recommend traffic rerouting with human approval.",
        ["read_sla", "query_metrics", "forecast_breach", "simulate_reroute", "notify_owner"],
        "Integrate streaming telemetry, contracts and SD-WAN/network controllers.",
        "SLA monitoring anomaly detection",
    ),
    Idea(
        4,
        "SIM-Swap and Subscription Fraud Response Agent",
        8.7,
        "Fraud signals are fragmented across device, account, location and transaction systems.",
        "Score a suspicious SIM replacement, gather supporting signals, explain the risk and require approval before blocking the account.",
        ["get_account_events", "check_device", "check_location", "score_risk", "freeze_sim"],
        "Add graph analytics, streaming events, KYC and bank/fintech signals.",
        "SIM swap fraud detection",
    ),
    Idea(
        5,
        "Field Technician Dispatch Copilot",
        8.5,
        "Repair teams receive incomplete tickets and inefficient routes.",
        "Turn an alarm into a repair plan, select a technician by skills and distance, prepare a checklist and update the work order.",
        ["diagnose_fault", "list_technicians", "optimize_route", "prepare_checklist", "update_work_order"],
        "Connect inventory, maps, workforce management and computer-vision inspection.",
        "field service dispatch route optimization",
    ),
    Idea(
        6,
        "Event-Aware Network Capacity Agent",
        8.3,
        "Concerts, matches and emergencies create local traffic spikes that static planning misses.",
        "Load an event schedule and traffic history, forecast congestion and propose temporary capacity or load-balancing actions.",
        ["read_events", "forecast_traffic", "find_capacity", "simulate_policy", "apply_policy"],
        "Add mobility data, weather, city events and live RAN telemetry.",
        "network traffic forecasting",
    ),
    Idea(
        7,
        "RAN Energy Optimization Agent",
        8.1,
        "Radio sites consume energy even when local traffic is low.",
        "Forecast hourly load, recommend safe sleep-mode windows and show energy savings without violating coverage constraints.",
        ["forecast_load", "check_coverage", "simulate_sleep_mode", "estimate_savings", "schedule_action"],
        "Use real RAN counters, electricity tariffs, weather and carbon-intensity data.",
        "network energy optimization cellular",
    ),
    Idea(
        8,
        "Rural Coverage Planning Digital Twin",
        7.9,
        "Remote communities need better coverage under strict infrastructure budgets.",
        "Compare candidate tower locations using synthetic terrain, population and cost data, then explain the best investment.",
        ["load_population", "load_terrain", "simulate_coverage", "estimate_cost", "rank_sites"],
        "Add OpenStreetMap, elevation, drive-test and regulator coverage data.",
        "radio coverage planning geospatial",
    ),
    Idea(
        9,
        "Explainable Tariff and Retention Agent",
        7.6,
        "Subscribers often overpay, run out of data or churn because plans do not match real usage.",
        "Analyze a synthetic usage history, recommend a plan, explain savings and execute a plan change only after confirmation.",
        ["get_usage", "compare_tariffs", "predict_churn", "explain_offer", "change_plan"],
        "Add billing, campaign systems, consent management and experimentation.",
        "telecom churn prediction",
    ),
    Idea(
        10,
        "Open Gateway API Composition Agent",
        7.4,
        "Developers struggle to discover and combine telecom capabilities such as number verification and device location.",
        "Describe an application in plain language; the agent selects mock CAMARA APIs, builds a workflow and tests it in a sandbox.",
        ["discover_api", "check_consent", "compose_workflow", "run_sandbox", "generate_sdk_code"],
        "Connect GSMA Open Gateway/CAMARA APIs and publish reusable workflows.",
        "CAMARA telecom API",
    ),
]


def repo(name: str, description: str, language: str = "Unknown") -> dict[str, Any]:
    """Create metadata for a manually reviewed GitHub building block."""
    return {
        "name": name,
        "url": f"https://github.com/{name}",
        "description": description,
        "stars": "check GitHub",
        "language": language,
        "updated_at": "",
        "topics": [],
        "source": "curated",
    }


CURATED_REPOSITORIES: dict[str, list[dict[str, Any]]] = {
    "Agentic NOC Incident Commander": [
        repo("prometheus/prometheus", "Time-series monitoring and alerting toolkit.", "Go"),
        repo("netbox-community/netbox", "Network source of truth with a programmable API.", "Python"),
    ],
    "Customer Impact & Outage Communication Agent": [
        repo("louislam/uptime-kuma", "Self-hosted service monitoring and status UI.", "JavaScript"),
        repo("grafana/grafana", "Dashboards and observability visualization.", "TypeScript"),
    ],
    "Enterprise SLA Assurance Agent": [
        repo("prometheus/prometheus", "Metrics collection, querying and alerting.", "Go"),
        repo("grafana/grafana", "SLA dashboards, alerts and incident visualization.", "TypeScript"),
    ],
    "SIM-Swap and Subscription Fraud Response Agent": [
        repo("camaraproject/DeviceSwap", "CAMARA API definitions for checking device/SIM swap signals.", "Gherkin"),
        repo("camaraproject/NumberVerification", "CAMARA number-verification API definitions.", "Gherkin"),
    ],
    "Field Technician Dispatch Copilot": [
        repo("google/or-tools", "Vehicle routing and combinatorial optimization solvers.", "C++"),
        repo("project-osrm/osrm-backend", "Open-source routing engine for road networks.", "C++"),
    ],
    "Event-Aware Network Capacity Agent": [
        repo("LibCity/Bigscity-LibCity", "Extensible spatial-temporal forecasting library.", "Python"),
        repo("Nixtla/neuralforecast", "Neural forecasting models for time-series data.", "Python"),
    ],
    "RAN Energy Optimization Agent": [
        repo("srsran/srsRAN_4G", "Open-source 4G software radio stack for experiments.", "C++"),
        repo("usnistgov/psc-ns3", "Communication-network simulation tools based on ns-3.", "C++"),
    ],
    "Rural Coverage Planning Digital Twin": [
        repo("meshtastic/meshtastic-site-planner", "Terrain-aware RF coverage planner using SPLAT!/ITM.", "TypeScript"),
        repo("seifreed/geoloc-api", "Cell-tower geolocation REST API compatible with OpenCellID data.", "Python"),
    ],
    "Explainable Tariff and Retention Agent": [
        repo("estefaniabarrosa/IBM-Telco-Customer-Churn", "End-to-end churn modelling and retention prioritization.", "Python"),
        repo("IBM/telco-customer-churn-on-icp4d", "IBM telecom churn data-science code pattern.", "Jupyter Notebook"),
    ],
    "Open Gateway API Composition Agent": [
        repo("camaraproject/Commonalities", "Common design, testing and schema assets for CAMARA APIs.", "Gherkin"),
        repo("camaraproject/ConnectivityInsights", "Connectivity Insights API definitions for network-aware applications.", "Gherkin"),
    ],
}


def github_search(query: str, token: str | None, limit: int) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "q": f"{query} stars:>5 pushed:>=2024-01-01",
            "sort": "stars",
            "order": "desc",
            "per_page": limit,
        }
    )
    request = urllib.request.Request(
        f"https://api.github.com/search/repositories?{params}",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "hackalem-telecom-idea-research/1.0",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.load(response)

    return [
        {
            "name": item["full_name"],
            "url": item["html_url"],
            "description": item.get("description") or "",
            "stars": item.get("stargazers_count", 0),
            "language": item.get("language") or "Unknown",
            "updated_at": item.get("updated_at", ""),
            "topics": item.get("topics", [])[:8],
        }
        for item in payload.get("items", [])
    ]


def enrich_with_github(limit: int) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    token = os.getenv("GITHUB_TOKEN")
    repositories: dict[str, list[dict[str, Any]]] = {}
    warnings: list[str] = []

    if not token:
        warnings.append(
            "GITHUB_TOKEN is not set; using manually reviewed GitHub projects. "
            "Set the token to add live search results."
        )
        return {
            idea.name: CURATED_REPOSITORIES.get(idea.name, [])[:limit] for idea in IDEAS
        }, warnings

    for idea in IDEAS:
        curated = CURATED_REPOSITORIES.get(idea.name, [])
        try:
            live = github_search(idea.github_query, token, limit)
            merged = curated + live
            repositories[idea.name] = list(
                {item["name"].lower(): item for item in merged}.values()
            )[:limit]
        except urllib.error.HTTPError as exc:
            repositories[idea.name] = curated[:limit]
            warnings.append(f"GitHub search failed for {idea.name}: HTTP {exc.code}")
        except (urllib.error.URLError, TimeoutError) as exc:
            repositories[idea.name] = curated[:limit]
            warnings.append(f"GitHub search failed for {idea.name}: {exc}")

    return repositories, warnings


def extract_json(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text[text.find("{") : text.rfind("}") + 1]
    if not candidate:
        raise ValueError("The model did not return a JSON object")
    return json.loads(candidate)


def ask_glm(repositories: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the dependency: python3 -m pip install -r requirements.txt") from exc

    evidence = {
        "hackathon_constraints": {
            "build_time_hours": 5,
            "team_size_max": 3,
            "must_be_agentic": True,
            "must_use_codex": True,
            "must_use_openai_api": True,
            "development_repository": "GitHub",
        },
        "seed_ideas": [asdict(idea) for idea in IDEAS],
        "github_projects": repositories,
        "industry_sources": SOURCES,
    }
    prompt = f"""
You are a telecom product architect and a strict AI hackathon judge.
Analyze the evidence below and rank exactly ten project ideas for a five-hour HackAlem build.

Evaluation weights:
- 30% working end-to-end demo in five hours
- 25% real agent behavior: observe, reason, call tools, act, verify
- 20% telecom business value
- 15% differentiation and demo impact
- 10% credible expansion path using open-source GitHub components

Do not claim that a repository solves a problem unless its metadata supports that claim.
Keep a human approval step before any risky network/account action.
The final hackathon product must use OpenAI API even though you are reviewing it through GLM-5.3.

Return only valid JSON with this shape:
{{
  "winner": "idea name",
  "decision": "short Russian explanation",
  "ideas": [
    {{
      "rank": 1,
      "name": "...",
      "score": 0.0,
      "why": "Russian explanation",
      "mvp": ["step 1", "step 2", "step 3"],
      "risks": ["..."],
      "github_reuse": ["owner/repo"]
    }}
  ]
}}

Evidence:
{json.dumps(evidence, ensure_ascii=False)}
""".strip()

    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)
    completion = client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=[
            {"role": "system", "content": "Be evidence-driven, practical and concise."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        top_p=1,
        max_tokens=6000,
        stream=False,
    )
    content = completion.choices[0].message.content or ""
    return extract_json(content)


def render_markdown(
    repositories: dict[str, list[dict[str, Any]]],
    glm_review: dict[str, Any] | None,
    warnings: list[str],
) -> str:
    lines = [
        "# Top 10 telecom ideas for HackAlem AI",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "> Recommendation: build **Agentic NOC Incident Commander** first. It has a clear",
        "> telecom problem, visible tool use, a safe approval gate and a convincing demo.",
        "",
    ]

    if glm_review:
        lines.extend(
            [
                "## GLM-5.3 review",
                "",
                f"**Winner:** {glm_review.get('winner', 'not specified')}",
                "",
                str(glm_review.get("decision", "")),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Review mode",
                "",
                "GLM-5.3 was not called. Rankings below use the built-in evidence-based baseline.",
                "Set `NVIDIA_API_KEY` and run without `--offline` to add the model review.",
                "",
            ]
        )

    lines.extend(["## Ranked ideas", ""])
    for idea in IDEAS:
        lines.extend(
            [
                f"### {idea.rank}. {idea.name} — {idea.score}/10",
                "",
                f"**Problem:** {idea.problem}",
                "",
                f"**Five-hour demo:** {idea.five_hour_demo}",
                "",
                f"**Agent tools:** {', '.join(f'`{tool}`' for tool in idea.agent_tools)}",
                "",
                f"**Expansion:** {idea.extension}",
                "",
                f"**GitHub search:** `{idea.github_query}`",
                "",
            ]
        )
        repos = repositories.get(idea.name, [])
        if repos:
            lines.append("Reusable projects to inspect:")
            lines.append("")
            for repo in repos:
                description = repo["description"] or "No description"
                lines.append(
                    f"- [{repo['name']}]({repo['url']}) — {description} "
                    f"({repo['stars']} stars, {repo['language']})"
                )
            lines.append("")

    lines.extend(["## Current industry sources", ""])
    for source in SOURCES:
        lines.append(f"- [{source['title']}]({source['url']}) — {source['note']}")
    lines.append("")

    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Skip the GLM-5.3 call")
    parser.add_argument(
        "--repos-per-idea",
        type=int,
        default=3,
        choices=range(1, 6),
        metavar="1-5",
        help="Number of public GitHub repositories to attach to every idea",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research_output"),
        help="Directory for Markdown and JSON reports",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repositories, warnings = enrich_with_github(args.repos_per_idea)

    glm_review = None
    if not args.offline:
        try:
            glm_review = ask_glm(repositories)
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"GLM-5.3 review skipped: {exc}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ideas": [asdict(idea) for idea in IDEAS],
        "github_projects": repositories,
        "glm_review": glm_review,
        "sources": SOURCES,
        "warnings": warnings,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "top_10_ideas.json"
    markdown_path = args.output_dir / "top_10_ideas.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_markdown(repositories, glm_review, warnings), encoding="utf-8"
    )

    print(f"Created {markdown_path}")
    print(f"Created {json_path}")
    if warnings:
        print(f"Completed with {len(warnings)} warning(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
