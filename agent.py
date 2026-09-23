"""ProfitPilot: offline pilot selection and constrained campaign allocation.

Only the documented env API is used. No LLM, network, hidden-state inspection,
or mock-specific effects. See README and THIRD_PARTY.md for assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np
import pandas as pd
try:
    from scipy.optimize import Bounds, LinearConstraint, milp
except ImportError:  # Explicitly supported, validated greedy fallback.
    Bounds = LinearConstraint = milp = None

# Public channel prices. Read-only published case parameters, not hidden effects.
CHANNELS = {"push": (0.0, 0.50), "sms": (4.0, 0.65),
            "digital_ads": (22.0, 0.85), "call": (160.0, 1.20)}
SPLIT_FILTERS = (("data_segment", {"NON_USER", "LITE", "HEAVY"}),
                 ("call_segment", {"LOW", "MEDIUM", "HIGH"}))


class AgentInputError(ValueError):
    """The public inputs cannot produce a supported, valid campaign."""


class PilotUnavailableError(RuntimeError):
    """No usable pilot completed; do not pretend a submission is compliant."""


@dataclass
class Candidate:
    cell: int
    current: str
    segment: str
    target: str
    channel: str
    n: int
    arpu: float
    mean: float
    variance: float
    sampled: int = 0
    pilots: int = 0
    data_segment: str | None = None
    call_segment: str | None = None

    @property
    def cost(self):
        return self.n * CHANNELS[self.channel][0]

    def filters(self):
        result = {"filter_current_tariff": self.current, "filter_arpu_segment": self.segment}
        if self.data_segment is not None:
            result["filter_data_segment"] = self.data_segment
        if self.call_segment is not None:
            result["filter_call_segment"] = self.call_segment
        return result

    def campaign(self):
        return {"campaign_name": f"PP_{self.cell}_{self.target}_{self.channel}",
                **self.filters(), "target_tariff": self.target, "channel": self.channel}


def update_estimate(mean, variance, observed, n, noise_scale=1.0):
    """Gaussian working model, not a calibrated confidence interval."""
    observation_variance = noise_scale ** 2 / n
    weight = variance / (variance + observation_variance)
    return mean + weight * (observed - mean), variance * (1.0 - weight)


def feasible(indices, candidates, budget, contacts):
    if not 1 <= len(indices) <= 10 or len(set(indices)) != len(indices):
        return False
    selected = [candidates[i] for i in indices]
    return (len({c.cell for c in selected}) == len(selected)
            and all(1 <= c.n <= 5000 for c in selected)
            and sum(c.cost for c in selected) <= budget + 1e-7
            and sum(c.n for c in selected) <= contacts)


def allocate(candidates, values, budget, contacts, exact=True):
    """One action per disjoint cell, 1..10 campaigns, hard budget/contact caps."""
    if not candidates:
        return []
    values = np.asarray(values, dtype=float)
    valid = [i for i, c in enumerate(candidates)
             if 0 < c.n <= min(5000, contacts) and c.cost <= budget and np.isfinite(values[i])]
    if not valid:
        return []
    # Several deterministic greedy starts, also used for fast pilot lookahead.
    best, best_value = [], -math.inf
    for score in (lambda i: values[i], lambda i: values[i] / candidates[i].n,
                  lambda i: values[i] / (1 + candidates[i].cost)):
        selected, cells, spent, used = [], set(), 0.0, 0
        for i in sorted(valid, key=score, reverse=True):
            c = candidates[i]
            if values[i] <= 0 or len(selected) >= 10 or c.cell in cells:
                continue
            if spent + c.cost <= budget and used + c.n <= contacts:
                selected.append(i)
                cells.add(c.cell)
                spent += c.cost
                used += c.n
        if not selected:
            selected = [max(valid, key=lambda i: values[i])]
        value = float(values[selected].sum())
        if value > best_value:
            best, best_value = selected, value
    if exact and milp is not None:
        subset = [candidates[i] for i in valid]
        cells = sorted({c.cell for c in subset})
        matrix = np.array([[c.cost for c in subset], [c.n for c in subset], [1] * len(subset)]
                          + [[int(c.cell == cell) for c in subset] for cell in cells], dtype=float)
        lower = np.array([0, 0, 1] + [0] * len(cells), dtype=float)
        upper = np.array([budget, contacts, 10] + [1] * len(cells), dtype=float)
        try:
            result = milp(-values[valid], integrality=np.ones(len(valid)), bounds=Bounds(0, 1),
                          constraints=LinearConstraint(matrix, lower, upper),
                          options={"time_limit": 2.0, "mip_rel_gap": 0.005})
            if result.x is not None:
                proposal = [valid[j] for j, x in enumerate(result.x) if x > 0.5]
                if feasible(proposal, candidates, budget, contacts) and values[proposal].sum() > best_value:
                    best = proposal
        except (ValueError, RuntimeError):
            pass  # Validated greedy fallback, never a partial invalid solver result.
    return best if feasible(best, candidates, budget, contacts) else []


class Agent:
    def __init__(self, mode=None, history_path="data/change_tariff.csv", risk_weight=0.75,
                 max_pilots=20, time_limit=240.0, max_cell_size=5000):
        self.mode = mode or os.environ.get("PROFITPILOT_MODE", "fixed")
        if self.mode not in {"adaptive", "fixed"}:
            raise ValueError("mode must be adaptive or fixed")
        self.history_path = Path(history_path)
        if not self.history_path.is_absolute() and not self.history_path.is_file():
            self.history_path = Path(__file__).resolve().parent / self.history_path
        self.risk_weight = risk_weight
        self.max_pilots = min(20, max(1, max_pilots))
        self.time_limit = time_limit
        if not 10 <= max_cell_size <= 5000:
            raise ValueError("max_cell_size must be between 10 and 5000")
        self.max_cell_size = max_cell_size
        self.trace = {}

    def _history(self):
        """Weak, robust associations only; before/after is NOT causal uplift."""
        if not self.history_path.is_file():
            return {}
        try:
            frame = pd.read_csv(self.history_path).drop_duplicates()
            pre = pd.to_numeric(frame["AVG_ARPU_PREV_3M"], errors="coerce")
            post = pd.to_numeric(frame["AVG_ARPU_NEXT_3M"], errors="coerce")
            valid = (pre >= 100) & (post >= 0) & np.isfinite(pre) & np.isfinite(post)
            frame = frame.loc[valid].copy()
            frame["ratio"] = ((post.loc[valid] - pre.loc[valid]) / pre.loc[valid]).clip(-0.5, 0.5)
            grouped = frame.groupby(["tariff_plan_code_from", "tariff_plan_code_to"])["ratio"].agg(["median", "count"])
            return {tuple(map(str, key)): float(row["median"] * row["count"] / (row["count"] + 30))
                    for key, row in grouped.iterrows()}
        except (OSError, KeyError, ValueError):
            return {}

    def _partition(self, group, filters=None, depth=0):
        """Disjoint leaves expressible by public filters; never split by row IDs."""
        filters = {} if filters is None else filters
        if 10 <= len(group) <= self.max_cell_size:
            yield group, filters
        elif len(group) > self.max_cell_size and depth < len(SPLIT_FILTERS):
            column, allowed = SPLIT_FILTERS[depth]
            if column not in group.columns:
                yield from self._partition(group, filters, depth + 1)
            else:
                for value, child in group.groupby(column, observed=True):
                    if str(value) in allowed:
                        yield from self._partition(child, {**filters, column: str(value)}, depth + 1)

    def _candidates(self, env):
        profile = env.customer_profile.copy()
        required = {"current_tariff", "arpu_segment", "predicted_arpu"}
        if not required.issubset(profile.columns) or "tariff_plan_code" not in env.tariffs:
            raise AgentInputError("Missing required public profile/tariff columns")
        tariffs = sorted(set(env.tariffs["tariff_plan_code"].dropna().astype(str)))
        history = self._history()
        profile["predicted_arpu"] = (pd.to_numeric(profile["predicted_arpu"], errors="coerce")
                                     .replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0))
        self.arpu_prefix = {}
        candidates = []
        groups = list(profile.groupby(["current_tariff", "arpu_segment"], observed=True))
        extra_cell = len(groups)
        for base_cell, ((current, segment), group) in enumerate(groups):
            current, segment = str(current), str(segment)
            if current not in tariffs or segment not in {"LOW", "MID", "HIGH"}:
                continue
            alternatives = [t for t in tariffs if t != current]
            # No hard-coded winning tariff IDs; ties remain deterministic.
            targets = sorted(alternatives, key=lambda t: (-history.get((current, t), 0.0), t))[:3]
            for part, (leaf, filters) in enumerate(self._partition(group)):
                cell = base_cell if part == 0 else extra_cell
                extra_cell += int(part > 0)
                arp = np.sort(leaf["predicted_arpu"].to_numpy(dtype=float))[::-1]
                self.arpu_prefix[cell] = np.concatenate(([0.0], np.cumsum(arp)))
                for target in targets:
                    for channel, (_, factor) in CHANNELS.items():
                        prior = 0.20 * history.get((current, target), 0.0) * factor
                        candidates.append(Candidate(cell, current, segment, target, channel,
                                                    len(leaf), float(arp.sum()), prior, 0.20 ** 2,
                                                    data_segment=filters.get("data_segment"),
                                                    call_segment=filters.get("call_segment")))
        if not candidates:
            raise AgentInputError("No eligible segment of 10..5000 customers can be expressed using the public filters")
        return candidates

    def _value(self, c, exposure, mean=None, variance=None):
        mean = c.mean if mean is None else mean
        variance = c.variance if variance is None else variance
        conservative_ratio = mean - self.risk_weight * math.sqrt(max(0.0, variance))
        # Conservative overlap proxy: do not count positive final gain on up to
        # all previous pilot contacts in this cell; remove highest-ARPU people.
        # This is NOT an exact identity-level overlap or causal lower bound.
        covered = min(c.n, exposure.get(c.cell, 0))
        remaining_arpu = c.arpu - self.arpu_prefix[c.cell][covered]
        gain = conservative_ratio * (remaining_arpu if conservative_ratio >= 0 else c.arpu)
        return float(gain - c.cost)

    def _plan(self, candidates, exposure, budget, contacts, extra=None, exact=False):
        pool = [c for c in candidates if c.pilots > 0 or c is extra]
        values = [self._value(c, exposure) for c in pool]
        selected = allocate(pool, values, budget, contacts, exact=exact)
        return [pool[i] for i in selected], sum(values[i] for i in selected)

    def _next_pilot(self, candidates, exposure, budget, contacts, pilot_budget, iteration):
        options = []
        reserve = min(x.n for x in candidates)
        sms_evidence = {(c.cell, c.target): c for c in candidates if c.channel == "sms" and c.pilots}
        for c in candidates:
            # Preserve room for at least one complete, supported final cell.
            n = min(200, c.n, int(contacts) - reserve)
            price = CHANNELS[c.channel][0]
            if price:
                n = min(n, int(min(pilot_budget, budget) // price))
            if n < 10 or c.pilots >= 3:
                continue
            # First establish a tariff hypothesis; do not spend the whole pilot
            # budget retesting its four channels before finding useful targets.
            evidence = sms_evidence.get((c.cell, c.target))
            if iteration > 0 and c.channel != "sms" and not c.pilots:
                if self.mode == "fixed" or evidence is None or evidence.mean <= math.sqrt(evidence.variance):
                    continue
            upper = c.arpu * (c.mean + math.sqrt(c.variance)) - c.cost
            options.append((c, n, upper))
        if not options:
            return None
        if iteration == 0:
            # Mandatory small fallback campaign, shared by both policy variants.
            # This limits exposure if all later large campaigns look unpromising.
            c, n, _ = min(options, key=lambda x: (x[0].arpu, x[0].cost, x[0].target))
            return c, n, None
        if self.mode == "fixed":
            # Same candidates/estimator/allocator, non-adaptive coverage order.
            fresh = [(c, n, score) for c, n, score in options if c.pilots == 0]
            if not fresh:
                return None
            fresh.sort(key=lambda x: (-x[0].arpu, x[0].cell, x[0].target,
                                      list(CHANNELS).index(x[0].channel)))
            # Spread initial checks across cells before checking more channels.
            fresh.sort(key=lambda x: exposure.get(x[0].cell, 0))
            c, n, _ = fresh[0]
            return c, n, None
        _, current_value = self._plan(candidates, exposure, budget, contacts)
        shortlist = sorted(options, key=lambda x: x[2], reverse=True)[:16]
        best = None
        for c, n, _ in shortlist:
            price = n * CHANNELS[c.channel][0]
            predictive_sd = math.sqrt(c.variance + 1.0 / n)
            after_exposure = dict(exposure)
            after_exposure[c.cell] = after_exposure.get(c.cell, 0) + n
            future = 0.0
            original = c.mean, c.variance
            try:
                # Three-point, one-step decision-improvement heuristic. Not exact VOI.
                for z, weight in ((-math.sqrt(2), 0.25), (0.0, 0.5), (math.sqrt(2), 0.25)):
                    outcome = original[0] + z * predictive_sd
                    c.mean, c.variance = update_estimate(*original, outcome, n)
                    _, value = self._plan(candidates, after_exposure, budget - price, contacts - n, extra=c)
                    future += weight * value
            finally:
                c.mean, c.variance = original
            immediate_proxy = self._value(c, exposure, variance=0.0) + c.cost
            score = future - current_value + n / c.n * immediate_proxy - price
            if best is None or score > best[2]:
                best = (c, n, score)
        return best if iteration == 0 or best is None or best[2] > 0 else None

    def act(self, env):
        started = time.monotonic()
        candidates = self._candidates(env)
        self.trace = {"schema": 2, "mode": self.mode, "pilots": [], "pilot_failures": [], "decisions": [],
                      "assumptions": ["observed_lift_ratio is treated as channel-adjusted; confirm with organizers",
                                      "Gaussian working noise scale=1; uncertainty is not calibrated",
                                      "pilot overlap is conservatively approximated, not observed by ID"],
                      "candidate_count": len(candidates)}
        exposure = {}
        initial_budget = float(env.remaining_budget)
        # Cap exploration spending; push pilots still consume contacts.
        pilot_budget = initial_budget * 0.25
        unavailable, failure_counts = set(), {}
        attempts = 0
        while attempts < self.max_pilots:
            eligible = [c for c in candidates if (c.cell, c.target, c.channel) not in unavailable]
            if not eligible or env.pilots_left <= 0 or time.monotonic() - started > self.time_limit - 5:
                break
            choice = self._next_pilot(eligible, exposure, float(env.remaining_budget),
                                      int(env.remaining_contacts), pilot_budget, len(self.trace["pilots"]))
            if choice is None:
                break
            c, n, selection_score = choice
            before_budget = float(env.remaining_budget)
            before_contacts = int(env.remaining_contacts)
            before_quota = int(env.pilots_left)
            attempts += 1
            error_name, retryable, result = None, False, None
            try:
                result = env.run_pilot(target_tariff=c.target, channel=c.channel, n_customers=n,
                                       **c.filters())
            except (RuntimeError, ValueError, TimeoutError, ConnectionError) as error:
                error_name = type(error).__name__
                retryable = not isinstance(error, ValueError)
            # Track resources from public counters, even if a result is unusable.
            spent = max(0.0, before_budget - float(env.remaining_budget))
            consumed = max(0, before_contacts - int(env.remaining_contacts))
            pilot_budget = max(0.0, pilot_budget - spent)
            exposure[c.cell] = exposure.get(c.cell, 0) + consumed
            if error_name is None:
                try:
                    observed = float(result["observed_lift_ratio"])
                    if not math.isfinite(observed):
                        error_name = "nonfinite_observed_lift_ratio"
                except (KeyError, TypeError, ValueError):
                    error_name = "invalid_observed_lift_ratio"
            if error_name is not None:
                key = (c.cell, c.target, c.channel)
                failure_counts[key] = failure_counts.get(key, 0) + 1
                unchanged = spent == 0 and consumed == 0 and int(env.pilots_left) == before_quota
                retry = retryable and unchanged and failure_counts[key] < 3
                if not retry:
                    unavailable.add(key)
                self.trace["pilot_failures"].append({**c.campaign(), "reason": error_name,
                                                       "attempt": attempts, "spent": spent,
                                                       "contacts": consumed, "retry_same_action": retry})
                continue
            c.mean, c.variance = update_estimate(c.mean, c.variance, observed, n)
            c.sampled += n
            c.pilots += 1
            plan, proxy = self._plan(candidates, exposure, float(env.remaining_budget), int(env.remaining_contacts))
            self.trace["pilots"].append({**c.campaign(), "n": n, "observed_lift_ratio": observed,
                                          "posterior_mean": c.mean, "posterior_sd": math.sqrt(c.variance),
                                          "decision_improvement_proxy": selection_score,
                                          "spent": spent, "remaining_budget": float(env.remaining_budget),
                                          "remaining_contacts": int(env.remaining_contacts),
                                          "plan_after": [x.campaign()["campaign_name"] for x in plan],
                                          "plan_incremental_net_proxy": proxy})
        pool = [c for c in candidates if c.pilots > 0]
        self.trace["pilot_attempts"] = attempts
        if not pool:
            self.trace["status"] = "no_usable_pilot"
            self._write_trace()
            raise PilotUnavailableError(f"No usable pilot after {attempts} attempts; refusing an untested submission")
        values = [self._value(c, exposure) for c in pool]
        indices = allocate(pool, values, float(env.remaining_budget), int(env.remaining_contacts), exact=True)
        selected = [pool[i] for i in indices]
        for i in indices:
            c = pool[i]
            self.trace["decisions"].append({**c.campaign(), "customers": c.n, "cost": c.cost,
                                             "pilot_count": c.pilots, "sampled": c.sampled,
                                             "mean": c.mean, "working_sd": math.sqrt(c.variance),
                                             "incremental_net_proxy": values[i]})
        self.trace.update({"elapsed_seconds": time.monotonic() - started,
                           "status": "ready" if selected else "no_feasible_plan",
                           "final_campaigns": len(selected), "final_cost": sum(c.cost for c in selected),
                           "final_contacts": sum(c.n for c in selected),
                           "remaining_budget_before_final": float(env.remaining_budget),
                           "remaining_contacts_before_final": int(env.remaining_contacts)})
        self._write_trace()
        if not selected:
            raise AgentInputError("Remaining resources cannot fund a valid tested campaign")
        return [c.campaign() for c in selected]

    def _write_trace(self):
        trace_dir = os.environ.get("PROFITPILOT_TRACE_DIR")
        if trace_dir:
            try:
                folder = Path(trace_dir)
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"trace-{uuid.uuid4().hex}.json").write_text(
                    json.dumps(self.trace, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            except (OSError, ValueError):
                pass  # Reporting must never invalidate campaigns.
