#!/usr/bin/env python3
from __future__ import annotations

"""Compare machine_compare.tsv outputs from multiple machines.

Expected input files are produced automatically by
scripts/run_phase2_structured_alien_seed12_ab.sh.

Each input row is one (machine, variant, seed) result with:
  best_eval, last_eval, latest_mode, latest_sample, latest_gap

This script pivots those rows into a compact table:
  variant, seed, metric, <machine1>, <machine2>, ..., range

The final "range" column is the quick consistency check across machines.

Example:
  python scripts/compare_seed12_ab_across_machines.py \
    /tmp/ab/local/machine_compare.tsv \
    /tmp/ab/machine87/machine_compare.tsv \
    /tmp/ab/machine2080/machine_compare.tsv \
    --output /tmp/ab/across_machines.tsv

Quick check for the most important metric:
  awk -F'\\t' 'NR==1 || $3=="latest_gap"' /tmp/ab/across_machines.tsv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


METRICS = ["best_eval", "last_eval", "latest_mode", "latest_sample", "latest_gap"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare seed12 A/B result tables across multiple machines."
    )
    parser.add_argument(
        "tables",
        nargs="+",
        help="Paths to machine_compare.tsv files. Machine name defaults to file stem.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output TSV path. Prints to stdout if omitted.",
    )
    return parser.parse_args()


def load_tables(paths: list[str]):
    rows = defaultdict(dict)
    machine_names = []
    for raw_path in paths:
        path = Path(raw_path)
        machine = path.stem
        machine_names.append(machine)
        with path.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                key = (row["variant"], row["seed"])
                rows[key][machine] = row
    return rows, machine_names


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_output(rows, machine_names):
    header = ["variant", "seed", "metric", *machine_names, "range"]
    out_rows = [header]
    for key in sorted(rows):
        by_machine = rows[key]
        for metric in METRICS:
            values = [to_float(by_machine.get(machine, {}).get(metric)) for machine in machine_names]
            nums = [v for v in values if v is not None]
            spread = max(nums) - min(nums) if nums else None
            out_rows.append(
                [
                    key[0],
                    key[1],
                    metric,
                    *[str(v) if v is not None else "" for v in values],
                    "" if spread is None else str(spread),
                ]
            )
    return out_rows


def main():
    args = parse_args()
    rows, machine_names = load_tables(args.tables)
    out_rows = build_output(rows, machine_names)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerows(out_rows)
    else:
        writer = csv.writer(__import__("sys").stdout, delimiter="\t")
        writer.writerows(out_rows)


if __name__ == "__main__":
    main()
