#!/usr/bin/env python3
"""Audit public participant CSVs without importing/executing environment code.

Reads only the explicit CSV allowlist and the public PDF guide. Never extracts
the archive, reads scoring/environment modules, or prints subscriber IDs.
Stdlib only. Output is JSON on stdout; --output optionally saves that artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import zipfile


ALLOWED_CSVS = (
    "customer_profile.csv",
    "data/change_tariff.csv",
    "data/traffic.csv",
    "data/arpu_monthly.csv",
    "data/dict_tariff.csv",
    "tariff_dictionary.csv",
    "feature_dictionary.csv",
)
NULLS = {"", "nan", "null", "none", "na", "n/a"}


def missing(value):
    return value is None or value.strip().lower() in NULLS


def number(value):
    if missing(value):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def quantile(values, probability):
    """Linear interpolation between sorted order statistics."""
    position = (len(values) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def distribution(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "negative": sum(value < 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "min": values[0],
        "p01": quantile(values, 0.01),
        "p25": quantile(values, 0.25),
        "median": statistics.median(values),
        "p75": quantile(values, 0.75),
        "p99": quantile(values, 0.99),
        "max": values[-1],
        "mean": statistics.mean(values),
    }


def ids(rows):
    return {row["ID_NUMBER"] for row in rows if not missing(row.get("ID_NUMBER"))}


def table_summary(columns, rows):
    result = {
        "rows": len(rows), "column_count": len(columns), "columns": columns,
        "exact_duplicate_extra_rows": sum(
            count - 1 for count in Counter(tuple(row.items()) for row in rows).values()
        ),
        "missing_by_column": {
            column: count for column in columns
            if (count := sum(missing(row.get(column)) for row in rows))
        },
    }
    if "ID_NUMBER" in columns:
        counts = Counter(row["ID_NUMBER"] for row in rows)
        result["unique_ids"] = len(ids(rows))
        result["rows_per_id"] = distribution(list(counts.values()))
    month_column = next((key for key in ("TIME_KEY", "time_key") if key in columns), None)
    if month_column:
        result["months"] = dict(sorted(Counter(row[month_column] for row in rows).items()))
        keys = Counter((row["ID_NUMBER"], row[month_column]) for row in rows)
        result["duplicate_id_month_extra_rows"] = sum(count - 1 for count in keys.values())
        unique_rows_by_key = defaultdict(set)
        for row in rows:
            unique_rows_by_key[(row["ID_NUMBER"], row[month_column])].add(tuple(row.items()))
        result["repeated_id_month_keys"] = sum(count > 1 for count in keys.values())
        result["repeated_id_month_keys_with_different_data"] = sum(
            len(values) > 1 for values in unique_rows_by_key.values()
        )
    return result


def audit(bundle, document=None):
    tables, summary = {}, {}
    with zipfile.ZipFile(bundle) as archive:
        for name in ALLOWED_CSVS:
            with archive.open(name) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
                rows = list(reader)
                tables[name] = rows
                summary[name] = table_summary(reader.fieldnames, rows)
        pdf = archive.read("PARTICIPANT_GUIDE.pdf")

    profile = tables["customer_profile.csv"]
    history = tables["data/change_tariff.csv"]
    traffic = tables["data/traffic.csv"]
    monthly = tables["data/arpu_monthly.csv"]
    tariffs = tables["data/dict_tariff.csv"]
    valid_tariffs = {row["tariff_plan_code"] for row in tariffs}
    profile_ids, history_ids = ids(profile), ids(history)

    with bundle.open("rb") as source:
        bundle_digest = hashlib.file_digest(source, "sha256").hexdigest()
    result = {
        "bundle": str(bundle),
        "bundle_sha256": bundle_digest,
        "method": "Public CSV data only; no environment imports or hidden scoring code reads.",
        "null_tokens_case_insensitive": sorted(NULLS),
        "tables": summary,
        "id_coverage": {
            name: {
                "in_profile": len(ids(rows) & profile_ids),
                "outside_profile": len(ids(rows) - profile_ids),
                "profile_missing_from_table": len(profile_ids - ids(rows)),
                "in_historical_changes": len(ids(rows) & history_ids),
            }
            for name, rows in (
                ("change_tariff", history), ("traffic", traffic), ("arpu_monthly", monthly)
            )
        },
    }
    result["documents"] = {"guide_pdf_sha256": hashlib.sha256(pdf).hexdigest()}
    if document is not None:
        data = document.read_bytes()
        result["documents"].update({
            "supplied_document": str(document),
            "supplied_document_sha256": hashlib.sha256(data).hexdigest(),
            "has_pdf_magic": data.startswith(b"%PDF-"),
            "identical_to_archive_pdf": data == pdf,
        })

    cells = defaultdict(list)
    for row in profile:
        if row["current_tariff"] in valid_tariffs and row["arpu_segment"] in {"LOW", "MID", "HIGH"}:
            cells[(row["current_tariff"], row["arpu_segment"])].append(row)
    result["profile"] = {
        "predicted_arpu_sum_decimal": str(sum(
            (Decimal(row["predicted_arpu"]) for row in profile if number(row["predicted_arpu"]) is not None),
            Decimal(0),
        )),
        "numeric": {column: distribution([number(row[column]) for row in profile]) for column in (
            "ARPU_current", "ARPU_3m_avg", "predicted_arpu", "DATA_VOLUME",
            "OUT_LOC_OFFNET_PAID_MIN", "COUNT_BASE_STATION",
        )},
        "segments": {column: dict(sorted(Counter(row[column] for row in profile).items())) for column in (
            "arpu_segment", "data_segment", "call_segment", "current_tariff", "ARPU_trend",
        )},
        "invalid_nonmissing_current_tariff": sum(
            not missing(row["current_tariff"]) and row["current_tariff"] not in valid_tariffs for row in profile
        ),
        "valid_tariff_arpu_cells": len(cells),
        "audience_in_valid_cells": sum(map(len, cells.values())),
        "max_valid_cell_size": max(map(len, cells.values()), default=0),
        "valid_cells_over_5000": sum(len(rows) > 5000 for rows in cells.values()),
        "valid_cells_under_10": sum(len(rows) < 10 for rows in cells.values()),
        "valid_cells_under_200": sum(len(rows) < 200 for rows in cells.values()),
        "arpu_by_segment": {
            segment: {
                "n": sum(row["arpu_segment"] == segment for row in profile),
                "predicted_arpu_sum": math.fsum(number(row["predicted_arpu"]) or 0 for row in profile if row["arpu_segment"] == segment),
            }
            for segment in sorted({row["arpu_segment"] for row in profile})
        },
    }
    for name, rows in (("traffic", traffic), ("arpu_monthly", monthly)):
        groups = defaultdict(set)
        for row in rows:
            groups[row.get("TIME_KEY", row.get("time_key"))].add(row["ID_NUMBER"])
        result["id_coverage"][name]["profile_ids_per_month"] = {
            month: len(month_ids & profile_ids) for month, month_ids in sorted(groups.items())
        }
    result["profile"]["arpu_segment_threshold_mismatches_nonmissing"] = sum(
        row["arpu_segment"] != ("LOW" if value < 1000 else "MID" if value <= 5000 else "HIGH")
        for row in profile
        if (value := number(row["ARPU_3m_avg"])) is not None and not missing(row["arpu_segment"])
    )
    result["profile"]["data_segment_threshold_mismatches_nonmissing"] = sum(
        row["data_segment"] != ("NON_USER" if value == 0 else "LITE" if value <= 2000 else "HEAVY")
        for row in profile
        if (value := number(row["DATA_VOLUME"])) is not None and value >= 0 and not missing(row["data_segment"])
    )

    pairs = Counter((row["tariff_plan_code_from"], row["tariff_plan_code_to"]) for row in history)
    ratios, directions = [], Counter()
    for row in history:
        before, after = number(row["AVG_ARPU_PREV_3M"]), number(row["AVG_ARPU_NEXT_3M"])
        if before is not None and before > 0 and after is not None:
            ratio = (after - before) / before
            ratios.append(ratio)
            directions["UPSELL" if ratio > 0.1 else "DOWNSELL" if ratio < -0.1 else "FLAT"] += 1
    result["history"] = {
        "numeric": {column: distribution([number(row[column]) for row in history]) for column in (
            "AVG_ARPU_PREV_3M", "AVG_ARPU_NEXT_3M",
        )},
        "rows_with_nonpositive_before": sum(
            (value := number(row["AVG_ARPU_PREV_3M"])) is not None and value <= 0 for row in history
        ),
        "rows_with_positive_before_below_100": sum(
            (value := number(row["AVG_ARPU_PREV_3M"])) is not None and 0 < value < 100 for row in history
        ),
        "same_source_target_rows": sum(row["tariff_plan_code_from"] == row["tariff_plan_code_to"] for row in history),
        "invalid_tariff_rows": sum(
            row["tariff_plan_code_from"] not in valid_tariffs or row["tariff_plan_code_to"] not in valid_tariffs for row in history
        ),
        "distinct_transition_pairs_including_invalid": len(pairs),
        "transition_pair_counts": distribution(list(pairs.values())),
        "pairs_under_10_rows": sum(count < 10 for count in pairs.values()),
        "descriptive_change_ratio_positive_before_only": distribution(ratios),
        "directions_positive_before_only_threshold_10pct": dict(directions),
        "warning": "Before-after differences are descriptive, not identified causal effects.",
    }
    tariff_groups = defaultdict(list)
    parameter_columns = [column for column in tariffs[0] if column != "tariff_plan_code"]
    for row in tariffs:
        tariff_groups[tuple(row[column] for column in parameter_columns)].append(row["tariff_plan_code"])
    extended_tariffs = {row["tariff_plan_code"]: row for row in tables["tariff_dictionary.csv"]}
    result["tariffs"] = {
        "count": len(tariffs),
        "parameter_identical_groups": [group for group in tariff_groups.values() if len(group) > 1],
        "two_dictionaries_agree_on_shared_columns": all(
            all(extended_tariffs.get(row["tariff_plan_code"], {}).get(column) == value for column, value in row.items())
            for row in tariffs
        ),
        "warning": "Equal package parameters do not establish equal campaign effects.",
    }
    result["monthly_arpu"] = distribution([number(row["ARPU_1M"]) for row in monthly])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--document", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source_paths = {args.bundle.resolve()}
    if args.document:
        source_paths.add(args.document.resolve())
    if args.output is not None and args.output.resolve() in source_paths:
        parser.error("Output must not overwrite a source file.")
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new filename or omit --output.")
    report = audit(args.bundle, args.document)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as destination:
            destination.write(rendered)
        print(f"Saved audit to {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
