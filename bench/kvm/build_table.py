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
  to the reference kernel under each mode, and whether the direction survives, and a tie when the difference under KVM is
  smaller than twice the spread between runs;
* `table.md`, the table of issue 10 in Markdown, with every cell copied from the
  two files above.

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
        --pair operations=bench/results/vps-1cpu,RUN1/operations,RUN2/operations \\
        --pair agent=bench/results/agent,RUN1/agent,RUN2/agent

Several KVM directories are runs of one dataset: the table keeps the median of
their medians and the spread between runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

HIGHER_IS_BETTER = {"infer-tps"}
CONTEXT = {"boot-seconds", "boot-rss-mb"}
REFERENCES = ("mainline-nomitig", "mainline")
ROW_COLUMNS = [
    "dataset", "accel", "runs", "kernel", "workload", "unit",
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


def merge_runs(runs: list[list[dict[str, str]]]) -> list[dict[str, str]]:
    """One row per kernel and workload from several runs of the same dataset.

    The median of a merged row is the median of the medians of the runs, so one
    odd run does not move it, and `p99` is the largest of the runs. `stddev` is
    the spread of the medians between runs, which is the number that says how
    far a repeated measurement can be trusted. `samples`, `errors` and
    `missing` are summed, so a run that lost samples shows up in the total.
    """
    keys: list[tuple[str, str]] = []
    for rows in runs:
        for row in rows:
            key = (row["kernel"], row["workload"])
            if key not in keys:
                keys.append(key)
    merged: list[dict[str, str]] = []
    for kernel, workload in keys:
        found = [index(rows).get((kernel, workload)) for rows in runs]
        present = [row for row in found if row is not None]
        medians = [m for m in (usable(row) for row in present) if m is not None]
        p99s = [v for v in (number(row.get("p99", "")) for row in present) if v is not None]
        merged.append({
            "kernel": kernel,
            "workload": workload,
            "unit": present[0].get("unit", ""),
            "runs": str(len(medians)),
            "samples": str(sum(int(row.get("samples") or 0) for row in present)),
            "errors": str(sum(int(row.get("errors") or 0) for row in present)),
            "missing": str(sum(int(row.get("missing") or 0) for row in present)),
            "median": f"{statistics.median(medians):.2f}" if medians else "",
            "p99": f"{max(p99s):.2f}" if p99s else "",
            "stddev": f"{statistics.pstdev(medians):.2f}" if len(medians) > 1 else "",
        })
    return merged


def read_runs(directories: list[Path], expected_accel: str) -> tuple[str, list[dict[str, str]]]:
    """The accelerator and the merged rows of several runs of one dataset."""
    accel = expected_accel
    all_rows = []
    for directory in directories:
        accel, rows = read_dataset(directory, expected_accel)
        all_rows.append(rows)
    return accel, merge_runs(all_rows)


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


def relative_spread(row: dict[str, str] | None) -> float:
    """Spread between runs as a fraction of the median, zero for one run."""
    if row is None:
        return 0.0
    median, spread = number(row.get("median", "")), number(row.get("stddev", ""))
    if not median or spread is None:
        return 0.0
    return spread / median


def direction(tcg: float | None, kvm: float | None, tolerance: float = 0.0) -> str:
    """Whether the ratio under KVM is on the same side of 1 as under emulation.

    `tolerance` is twice the combined relative spread of the two kernels between
    runs. A ratio under KVM closer to 1 than that is a tie, not a side: it is
    said so, instead of being called a reversal or a win.
    """
    if tcg is None or kvm is None:
        return "unknown"
    if tolerance > 0 and abs(kvm - 1) <= tolerance:
        return "tie under KVM"
    if (tcg > 1) == (kvm > 1) and tcg != 1 and kvm != 1:
        return "same side"
    if tcg == 1 or kvm == 1:
        return "tie under one mode"
    return "reverses"


def derive(
    pairs: list[tuple[str, Path | list[Path], Path | list[Path]]]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Rows and ratios for pairs of datasets.

    Either side is one directory or several runs of the same dataset, merged by
    [`merge_runs`]. The runs column of a row says how many went into it.
    """
    out_rows: list[dict[str, str]] = []
    out_ratios: list[dict[str, str]] = []
    for name, tcg_dirs, kvm_dirs in pairs:
        as_list = lambda value: value if isinstance(value, list) else [value]
        tcg_accel, tcg_rows = read_runs(as_list(tcg_dirs), "tcg")
        kvm_accel, kvm_rows = read_runs(as_list(kvm_dirs), "kvm")
        tcg, kvm = index(tcg_rows), index(kvm_rows)
        for accel, rows in ((tcg_accel, tcg_rows), (kvm_accel, kvm_rows)):
            for row in rows:
                # A workload the run was not asked for has no samples and no
                # missing ones: it is not a measurement, so it is not a row.
                if row.get("samples") == "0" and row.get("missing") == "0":
                    continue
                out_rows.append({"dataset": name, "accel": accel, **{
                    key: row.get(key, "") or ("1" if key == "runs" else "")
                    for key in ROW_COLUMNS[2:]
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
                    "direction": direction(
                        r_tcg, r_kvm,
                        2 * (relative_spread(kvm.get(("rabbit", workload)))
                             + relative_spread(kvm.get((reference, workload)))),
                    ),
                    "tcg_status": status(tcg.get(("rabbit", workload)), tcg.get((reference, workload))),
                    "kvm_status": status(kvm.get(("rabbit", workload)), kvm.get((reference, workload))),
                })
    return out_rows, out_ratios


LABELS = {
    "syscall": "system call",
    "mmap": "map and touch memory, per MiB",
    "weights": "fault in memory, 4 KiB pages, per MiB",
    "weights-tail": "slowest single fault, 4 KiB",
    "weights-huge": "fault in memory, 2 MiB pages, per MiB",
    "weights-huge-tail": "slowest single fault, 2 MiB",
    "futex": "futex round trip",
    "threads": "four threads, spawn to join",
    "infer-ttft": "time to the first token",
    "infer-tps": "tokens per second",
    "agent-seconds": "seconds of an agent loop",
}


def with_unit(value: float, unit: str) -> str:
    """A figure the way a person reads it: four significant figures and a unit
    that keeps it between 1 and 1000 where it can. The rig writes nanoseconds
    and milliseconds, and a cell such as 6.83e+6 says nothing at a glance."""
    if unit.startswith("ns"):
        for limit, scale, name in ((1e3, 1.0, "ns"), (1e6, 1e3, "us"), (1e9, 1e6, "ms")):
            if value < limit:
                return f"{value / scale:.4g} {name}"
        return f"{value / 1e9:.4g} s"
    if unit == "ms" and value >= 1000:
        return f"{value / 1000:.4g} s"
    return f"{value:.4g} {unit}"


def markdown(rows: list[dict[str, str]], ratios: list[dict[str, str]]) -> str:
    """The table of issue 10: emulation and KVM side by side, one row per workload.

    Only rows that are comparisons against `mainline-nomitig` appear, and every
    cell is copied from the derived files, never typed; the unit of a cell is the
    one that keeps its figure readable. The direction column
    says when the ratio under KVM is on the other side of 1 than under
    emulation. Both columns name the mode that produced them.
    """
    by_key = {(r["dataset"], r["accel"], r["kernel"], r["workload"]): r for r in rows}
    lines = [
        "| workload | Rabbit, emulation (TCG) | mainline, emulation | ratio, emulation "
        "| Rabbit, KVM | mainline, KVM | ratio, KVM | direction |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ratio_row in ratios:
        if ratio_row["reference"] != "mainline-nomitig" or ratio_row["kind"] != "comparison":
            continue
        workload = ratio_row["workload"]
        if workload not in LABELS:
            continue
        cells = []
        for accel in ("tcg", "kvm"):
            for kernel in ("rabbit", "mainline-nomitig"):
                row = by_key.get((ratio_row["dataset"], accel, kernel, workload))
                value = number(row["median"]) if row else None
                cells.append(with_unit(value, row["unit"]) if value is not None and row else "-")

        def ratio_cell(text: str) -> str:
            return f"{float(text):.2f}x" if text else "-"

        lines.append(
            f"| {LABELS[workload]} | {cells[0]} | {cells[1]} | {ratio_cell(ratio_row['ratio_tcg'])} "
            f"| {cells[2]} | {cells[3]} | {ratio_cell(ratio_row['ratio_kvm'])} | {ratio_row['direction']} |"
        )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--pair", action="append", required=True, metavar="NAME=TCG_DIR,KVM_DIR[,KVM_DIR...]",
        help=(
            "a dataset under emulation and the same one under KVM; several "
            "KVM directories are runs of that dataset and are merged"
        ),
    )
    args = parser.parse_args()
    pairs = []
    for spec in args.pair:
        name, _, dirs = spec.partition("=")
        tcg, _, kvm = dirs.partition(",")
        if not (name and tcg and kvm):
            print(f"invalid --pair: {spec}", file=sys.stderr)
            return 2
        pairs.append((name, Path(tcg), [Path(path) for path in kvm.split(",") if path]))
    try:
        rows, ratios = derive(pairs)
    except TableError as error:
        print(f"build_table: {error}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "table.md").write_text(markdown(rows, ratios))
    write_csv(args.out / "rows.csv", ROW_COLUMNS, rows)
    write_csv(args.out / "ratios.csv", RATIO_COLUMNS, ratios)
    print(f"{len(rows)} rows, {len(ratios)} ratios written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
