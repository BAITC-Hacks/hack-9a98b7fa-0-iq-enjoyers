# Top 10 telecom ideas for HackAlem AI

Generated: 2026-09-23T07:42:56+00:00

> Recommendation: build **Agentic NOC Incident Commander** first. It has a clear
> telecom problem, visible tool use, a safe approval gate and a convincing demo.

## Review mode

GLM-5.3 was not called. Rankings below use the built-in evidence-based baseline.
Set `NVIDIA_API_KEY` and run without `--offline` to add the model review.

## Ranked ideas

### 1. Agentic NOC Incident Commander — 9.7/10

**Problem:** Engineers lose time correlating alarms, KPIs, topology and customer tickets.

**Five-hour demo:** Inject a base-station incident; the agent gathers evidence, finds a root cause, simulates a fix, requests approval and opens a ticket.

**Agent tools:** `get_alarms`, `get_kpis`, `get_topology`, `simulate_fix`, `create_ticket`

**Expansion:** Connect Prometheus/Grafana, NetBox, ServiceNow or a real operator OSS.

**GitHub search:** `network monitoring incident observability`

Reusable projects to inspect:

- [prometheus/prometheus](https://github.com/prometheus/prometheus) — Time-series monitoring and alerting toolkit. (check GitHub stars, Go)
- [netbox-community/netbox](https://github.com/netbox-community/netbox) — Network source of truth with a programmable API. (check GitHub stars, Python)

### 2. Customer Impact & Outage Communication Agent — 9.2/10

**Problem:** Network teams see alarms, while customers experience an unexplained loss of service.

**Five-hour demo:** Map a simulated outage to affected subscribers, prioritize vulnerable users and create personalized Kazakh/Russian status messages.

**Agent tools:** `detect_outage`, `find_affected_users`, `estimate_eta`, `draft_notification`

**Expansion:** Add CRM, geospatial coverage, contact-center and notification integrations.

**GitHub search:** `network outage map status monitoring`

Reusable projects to inspect:

- [louislam/uptime-kuma](https://github.com/louislam/uptime-kuma) — Self-hosted service monitoring and status UI. (check GitHub stars, JavaScript)
- [grafana/grafana](https://github.com/grafana/grafana) — Dashboards and observability visualization. (check GitHub stars, TypeScript)

### 3. Enterprise SLA Assurance Agent — 9.0/10

**Problem:** B2B customers need proactive protection of latency, availability and throughput SLAs.

**Five-hour demo:** Monitor synthetic KPIs, predict an SLA breach, explain the cause and recommend traffic rerouting with human approval.

**Agent tools:** `read_sla`, `query_metrics`, `forecast_breach`, `simulate_reroute`, `notify_owner`

**Expansion:** Integrate streaming telemetry, contracts and SD-WAN/network controllers.

**GitHub search:** `SLA monitoring anomaly detection`

Reusable projects to inspect:

- [prometheus/prometheus](https://github.com/prometheus/prometheus) — Metrics collection, querying and alerting. (check GitHub stars, Go)
- [grafana/grafana](https://github.com/grafana/grafana) — SLA dashboards, alerts and incident visualization. (check GitHub stars, TypeScript)

### 4. SIM-Swap and Subscription Fraud Response Agent — 8.7/10

**Problem:** Fraud signals are fragmented across device, account, location and transaction systems.

**Five-hour demo:** Score a suspicious SIM replacement, gather supporting signals, explain the risk and require approval before blocking the account.

**Agent tools:** `get_account_events`, `check_device`, `check_location`, `score_risk`, `freeze_sim`

**Expansion:** Add graph analytics, streaming events, KYC and bank/fintech signals.

**GitHub search:** `SIM swap fraud detection`

Reusable projects to inspect:

- [camaraproject/DeviceSwap](https://github.com/camaraproject/DeviceSwap) — CAMARA API definitions for checking device/SIM swap signals. (check GitHub stars, Gherkin)
- [camaraproject/NumberVerification](https://github.com/camaraproject/NumberVerification) — CAMARA number-verification API definitions. (check GitHub stars, Gherkin)

### 5. Field Technician Dispatch Copilot — 8.5/10

**Problem:** Repair teams receive incomplete tickets and inefficient routes.

**Five-hour demo:** Turn an alarm into a repair plan, select a technician by skills and distance, prepare a checklist and update the work order.

**Agent tools:** `diagnose_fault`, `list_technicians`, `optimize_route`, `prepare_checklist`, `update_work_order`

**Expansion:** Connect inventory, maps, workforce management and computer-vision inspection.

**GitHub search:** `field service dispatch route optimization`

Reusable projects to inspect:

- [google/or-tools](https://github.com/google/or-tools) — Vehicle routing and combinatorial optimization solvers. (check GitHub stars, C++)
- [project-osrm/osrm-backend](https://github.com/project-osrm/osrm-backend) — Open-source routing engine for road networks. (check GitHub stars, C++)

### 6. Event-Aware Network Capacity Agent — 8.3/10

**Problem:** Concerts, matches and emergencies create local traffic spikes that static planning misses.

**Five-hour demo:** Load an event schedule and traffic history, forecast congestion and propose temporary capacity or load-balancing actions.

**Agent tools:** `read_events`, `forecast_traffic`, `find_capacity`, `simulate_policy`, `apply_policy`

**Expansion:** Add mobility data, weather, city events and live RAN telemetry.

**GitHub search:** `network traffic forecasting`

Reusable projects to inspect:

- [LibCity/Bigscity-LibCity](https://github.com/LibCity/Bigscity-LibCity) — Extensible spatial-temporal forecasting library. (check GitHub stars, Python)
- [Nixtla/neuralforecast](https://github.com/Nixtla/neuralforecast) — Neural forecasting models for time-series data. (check GitHub stars, Python)

### 7. RAN Energy Optimization Agent — 8.1/10

**Problem:** Radio sites consume energy even when local traffic is low.

**Five-hour demo:** Forecast hourly load, recommend safe sleep-mode windows and show energy savings without violating coverage constraints.

**Agent tools:** `forecast_load`, `check_coverage`, `simulate_sleep_mode`, `estimate_savings`, `schedule_action`

**Expansion:** Use real RAN counters, electricity tariffs, weather and carbon-intensity data.

**GitHub search:** `network energy optimization cellular`

Reusable projects to inspect:

- [srsran/srsRAN_4G](https://github.com/srsran/srsRAN_4G) — Open-source 4G software radio stack for experiments. (check GitHub stars, C++)
- [usnistgov/psc-ns3](https://github.com/usnistgov/psc-ns3) — Communication-network simulation tools based on ns-3. (check GitHub stars, C++)

### 8. Rural Coverage Planning Digital Twin — 7.9/10

**Problem:** Remote communities need better coverage under strict infrastructure budgets.

**Five-hour demo:** Compare candidate tower locations using synthetic terrain, population and cost data, then explain the best investment.

**Agent tools:** `load_population`, `load_terrain`, `simulate_coverage`, `estimate_cost`, `rank_sites`

**Expansion:** Add OpenStreetMap, elevation, drive-test and regulator coverage data.

**GitHub search:** `radio coverage planning geospatial`

Reusable projects to inspect:

- [meshtastic/meshtastic-site-planner](https://github.com/meshtastic/meshtastic-site-planner) — Terrain-aware RF coverage planner using SPLAT!/ITM. (check GitHub stars, TypeScript)
- [seifreed/geoloc-api](https://github.com/seifreed/geoloc-api) — Cell-tower geolocation REST API compatible with OpenCellID data. (check GitHub stars, Python)

### 9. Explainable Tariff and Retention Agent — 7.6/10

**Problem:** Subscribers often overpay, run out of data or churn because plans do not match real usage.

**Five-hour demo:** Analyze a synthetic usage history, recommend a plan, explain savings and execute a plan change only after confirmation.

**Agent tools:** `get_usage`, `compare_tariffs`, `predict_churn`, `explain_offer`, `change_plan`

**Expansion:** Add billing, campaign systems, consent management and experimentation.

**GitHub search:** `telecom churn prediction`

Reusable projects to inspect:

- [estefaniabarrosa/IBM-Telco-Customer-Churn](https://github.com/estefaniabarrosa/IBM-Telco-Customer-Churn) — End-to-end churn modelling and retention prioritization. (check GitHub stars, Python)
- [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d) — IBM telecom churn data-science code pattern. (check GitHub stars, Jupyter Notebook)

### 10. Open Gateway API Composition Agent — 7.4/10

**Problem:** Developers struggle to discover and combine telecom capabilities such as number verification and device location.

**Five-hour demo:** Describe an application in plain language; the agent selects mock CAMARA APIs, builds a workflow and tests it in a sandbox.

**Agent tools:** `discover_api`, `check_consent`, `compose_workflow`, `run_sandbox`, `generate_sdk_code`

**Expansion:** Connect GSMA Open Gateway/CAMARA APIs and publish reusable workflows.

**GitHub search:** `CAMARA telecom API`

Reusable projects to inspect:

- [camaraproject/Commonalities](https://github.com/camaraproject/Commonalities) — Common design, testing and schema assets for CAMARA APIs. (check GitHub stars, Gherkin)
- [camaraproject/ConnectivityInsights](https://github.com/camaraproject/ConnectivityInsights) — Connectivity Insights API definitions for network-aware applications. (check GitHub stars, Gherkin)

## Current industry sources

- [HackAlem AI — official requirements](https://hackalem.ai/) — Agentic AI, offline event, teams up to three, Codex is mandatory.
- [AI & Digital Bridge 2026 programme](https://digitalbridge.ai/en/aimonth/) — Five-hour build, ten sectors, Codex and OpenAI API are mandatory.
- [Ericsson — Agentic AI for autonomous telecom networks](https://www.ericsson.com/en/reports-and-papers/white-papers/adapting-agentic-ai) — Closed-loop operations, incident resolution and human oversight.
- [GSMA — Agentic Network for Mobile AI Era](https://www.gsma.com/solutions-and-impact/technologies/networks/gsma_resources/agentic-network-for-mobile-ai-era/) — AI agents for O&M, deterministic SLA and energy efficiency.
- [TM Forum — AI-native intelligent operations](https://inform.tmforum.org/research-and-analysis/reports/new-generation-intelligent-operations-an-ai-native-reinvention) — Agents as digital employees for resilient network operations.
- [IETF draft — Network Digital Twin and Agentic AI](https://datatracker.ietf.org/doc/draft-wmz-nmrg-agent-ndt-arch/) — Testing proposed network actions in a risk-free digital twin.

## Warnings

- GITHUB_TOKEN is not set; using manually reviewed GitHub projects. Set the token to add live search results.
