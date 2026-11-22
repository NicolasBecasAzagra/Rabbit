#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Derive the two column table of issue 10 from datasets of the rig.

A dataset is a directory written by `bench/run.py`: `summary.csv` with the
statistics of every kernel and workload, and `summary.json` whose metadata names
the acceleration mode that produced it. This script takes pairs of datasets, one
run under emulation (TCG) and one under KVM, of the same workloads, and writes
two files into the directory given by `--out`:

* `rows.csv`, one row per dataset, accelerator, kernel and workload, with the
  median, the spread and the count of errors and missing samples, and the
  acceleration mode of that row as the metadata of its dataset states it;
* `ratios.csv`, one row per dataset and workload, with the ratio of the Rabbit
  to the reference kernel under each mode, and whether the direction survives.

A ratio above 1 means Rabbit is better: the reference over Rabbit for a cost,
Rabbit over the reference for a rate. Nothing is typed by hand. The script
refuses a pair whose metadata does not say `tcg` for the emulated side and `kvm`
for the accelerated one, because a table that mixes the two would support a claim
that its data does not.

The emulated column is a relative comparison between two kernels under one
emulator. Only the accelerated column can support a statement about wall clock
performance, and the direction column is the check of the first against the
second. The boot rows are context, not a comparison, as docs/BENCHMARKING.md
states, and are marked so.

Usage:
    bench/kvm/build_table.py --out DIR \\
        --pair operations=bench/results/vps-1cpu,DIR/operations \\
        --pair agent=bench/results/agent,DIR/agent
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HIGHER_IS_BETTER = {"infer-tps"}
CONTEXT = {"boot-seconds", "boot-rss-mb"}
REFERENCES = ("mainline-nomitig", "mainline")
ROW_COLUMNS = [
    "dataset", "accel", "kernel", "workload", "unit",
    "samples", "errors", "missing", "median", "p99", "stddev",
]
RATIO_COLUMNS = [
    "dataset", "workload", "kind", "reference",
    "ratio_tcg", "ratio_kvm", "direction",
    "tcg_status", "kvm_status",
]


class TableError(Exception):
    """A dataset that cannot be trusted: the message says which and why."""


def read_dataset(directory: Path, expected_accel: str) -> tuple[str, list[dict[str, str]]]:
    """The accelerator of a dataset and its summary rows, after checking it."""
    summary_json = directory / "summary.json"
    summary_csv = directory / "summary.csv"
    if not summary_json.is_file() or not summary_csv.is_file():
        raise TableError(f"{directory}: summary.json and summary.csv are both needed")
    metadata = json.loads(summary_json.read_text()).get("metadata", {})
    accel = metadata.get("accel")
    if accel != expected_accel:
        raise TableError(
            f"{directory}: its metadata says accel={accel!r}, and this column is "
            f"{expected_accel!r}; the two modes are never mixed in one column"
        )
    with summary_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise TableError(f"{directory}: summary.csv has no rows")
    return accel, rows


def number(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["kernel"], row["workload"]): row for row in rows}


def usable(row: dict[str, str] | None) -> float | None:
    """The median of a row, or None when the row has no measurement to compare."""
    if row is None:
        return None
    if int(row.get("samples") or 0) == 0:
        return None
    return number(row["median"])


def ratio(workload: str, rabbit: float | None, reference: float | None) -> float | None:
    """Above 1 means Rabbit is better, whatever the direction of the unit."""
    if rabbit is None or reference is None or rabbit <= 0 or reference <= 0:
        return None
    if workload in HIGHER_IS_BETTER:
        return rabbit / reference
    return reference / rabbit


def status(row_rabbit: dict[str, str] | None, row_ref: dict[str, str] | None) -> str:
    """Why a ratio is missing or thin, or ok. Losses and gaps are not hidden."""
    for name, row in (("rabbit", row_rabbit), ("reference", row_ref)):
        if row is None:
            return f"no {name} row"
        if int(row.get("samples") or 0) == 0:
            return f"no {name} samples"
    problems = []
    for name, row in (("rabbit", row_rabbit), ("reference", row_ref)):
        assert row is not None
        if int(row.get("errors") or 0) or int(row.get("missing") or 0):
            problems.append(f"{name} errors={row['errors']} missing={row['missing']}")
    return "; ".join(problems) if problems else "ok"


def direction(tcg: float | None, kvm: float | None) -> str:
    if tcg is None or kvm is None:
        return "unknown"
    if (tcg > 1) == (kvm > 1) and tcg != 1 and kvm != 1:
        return "same side"
    if tcg == 1 or kvm == 1:
        return "tie under one mode"
    return "reverses"


def derive(pairs: list[tuple[str, Path, Path]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    out_rows: list[dict[str, str]] = []
    out_ratios: list[dict[str, str]] = []
    for name, tcg_dir, kvm_dir in pairs:
        tcg_accel, tcg_rows = read_dataset(tcg_dir, "tcg")
        kvm_accel, kvm_rows = read_dataset(kvm_dir, "kvm")
        tcg, kvm = index(tcg_rows), index(kvm_rows)
        for accel, rows in ((tcg_accel, tcg_rows), (kvm_accel, kvm_rows)):
            for row in rows:
                # A workload the run was not asked for has no samples and no
                # missing ones: it is not a measurement, so it is not a row.
                if row.get("samples") == "0" and row.get("missing") == "0":
                    continue
                out_rows.append({"dataset": name, "accel": accel, **{
                    key: row.get(key, "") for key in ROW_COLUMNS[2:]
                }})
        workloads = sorted({w for (_, w) in tcg} | {w for (_, w) in kvm})
        present = {k for (k, _) in tcg} | {k for (k, _) in kvm}
        for workload in workloads:
            measured = [
                r for r in tcg_rows + kvm_rows
                if r["workload"] == workload and (int(r.get("samples") or 0) or int(r.get("missing") or 0))
            ]
            if not measured:
                continue
            for reference in REFERENCES:
                r_tcg = ratio(workload, usable(tcg.get(("rabbit", workload))),
                              usable(tcg.get((reference, workload))))
                r_kvm = ratio(workload, usable(kvm.get(("rabbit", workload))),
                              usable(kvm.get((reference, workload))))
                if reference not in present:
                    continue
                out_ratios.append({
                    "dataset": name,
                    "workload": workload,
                    "kind": "context" if workload in CONTEXT else "comparison",
                    "reference": reference,
                    "ratio_tcg": "" if r_tcg is None else f"{r_tcg:.3f}",
                    "ratio_kvm": "" if r_kvm is None else f"{r_kvm:.3f}",
                    "direction": direction(r_tcg, r_kvm),
                    "tcg_status": status(tcg.get(("rabbit", workload)), tcg.get((reference, workload))),
                    "kvm_status": status(kvm.get(("rabbit", workload)), kvm.get((reference, workload))),
                })
    return out_rows, out_ratios


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--pair", action="append", required=True, metavar="NAME=TCG_DIR,KVM_DIR",
        help="a dataset under emulation and the same one under KVM",
    )
    args = parser.parse_args()
    pairs = []
    for spec in args.pair:
        name, _, dirs = spec.partition("=")
        tcg, _, kvm = dirs.partition(",")
        if not (name and tcg and kvm):
            print(f"invalid --pair: {spec}", file=sys.stderr)
            return 2
        pairs.append((name, Path(tcg), Path(kvm)))
    try:
        rows, ratios = derive(pairs)
    except TableError as error:
        print(f"build_table: {error}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "rows.csv", ROW_COLUMNS, rows)
    write_csv(args.out / "ratios.csv", RATIO_COLUMNS, ratios)
    print(f"{len(rows)} rows, {len(ratios)} ratios written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
