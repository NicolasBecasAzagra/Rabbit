#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Derive the progress of Rabbit under KVM, stage by stage, from the raw runs.

The data of `bench/results/kvm/` is a series of stages. A stage is one version
of the Rabbit kernel, measured by five runs of the same three datasets on the
same host, in a directory `stages/<name>/kvm-<cpu>-r<N>/<dataset>/`. This script
reads every stage listed in `MANIFEST.csv`, merges the runs of each dataset with
the same rule as `build_table.py` (median of the medians, spread between runs,
lost samples summed), and writes:

* `stages.csv`, one row per stage, dataset, kernel and workload;
* `ratios_by_stage.csv`, one row per stage and workload, with the ratio of
  Rabbit to the reference kernel. A ratio above 1 means Rabbit is better,
  whichever the unit, and a ratio below 1 is a loss and stays in the file.

Every stage is checked against `MANIFEST.csv`: the commit the bundle was built
from has to be the one listed, and the metadata of every run has to say `kvm`.
A stage that does not match is an error, never a silent row.

Usage: bench/kvm/stages.py [--dir bench/results/kvm] [--cpu skylake-client]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Run as a script or as an isolated one: the helper lives next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_table  # noqa: E402

DATASETS = ("operations", "inference", "agent")
REFERENCE = "mainline-nomitig"
STAGE_COLUMNS = [
    "stage", "commit", "cpu", "dataset", "kernel", "workload", "unit",
    "runs", "median", "p99", "stddev", "samples", "errors", "missing",
]
RATIO_COLUMNS = [
    "stage", "dataset", "workload", "kind", "rabbit", "reference", "ratio", "note",
]


class StageError(Exception):
    """A stage that does not match its description: the message says how."""


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise StageError(f"{path}: no stages listed")
    return rows


def guard_outputs(manifest: Path, directory: Path) -> None:
    """Refuse to start when an output would overwrite the manifest.

    On a file system that ignores case, `STAGES.csv` and `stages.csv` are one
    file, and a script that reads one and writes the other eats its own input and
    grows it on every run. The manifest and the outputs have names that differ by
    more than case, and this makes it a check and not a habit.
    """
    for output in ("stages.csv", "ratios_by_stage.csv"):
        if (directory / output).resolve() == manifest.resolve() or output.lower() == manifest.name.lower():
            raise StageError(f"{manifest.name} and {output} are the same file on a case insensitive disk")


def run_directories(stage_dir: Path, cpu: str) -> list[Path]:
    """The repeated runs of one CPU model, in order: `kvm-<cpu>-r1`, `-r2`, ..."""
    found = sorted(
        (p for p in stage_dir.glob(f"kvm-{cpu}-r[0-9]*") if p.is_dir()),
        key=lambda p: int(p.name.rsplit("-r", 1)[1]),
    )
    if not found:
        raise StageError(f"{stage_dir}: no runs of {cpu}")
    return found


def check_commit(stage_dir: Path, expected: str) -> None:
    recorded = (stage_dir / "BUNDLE-COMMIT").read_text().strip()
    if not recorded.startswith(expected) and not expected.startswith(recorded):
        raise StageError(
            f"{stage_dir}: the bundle was built from {recorded}, "
            f"and the manifest says {expected}"
        )


def read_stage(stage_dir: Path, name: str, commit: str, cpu: str) -> list[dict[str, str]]:
    check_commit(stage_dir, commit)
    runs = run_directories(stage_dir, cpu)
    rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        try:
            _, merged = build_table.read_runs([run / dataset for run in runs], "kvm")
        except build_table.TableError as error:
            raise StageError(str(error)) from error
        for row in merged:
            if row["samples"] == "0" and row["missing"] == "0":
                continue
            rows.append({"stage": name, "commit": commit, "cpu": cpu, "dataset": dataset, **{
                key: row.get(key, "") for key in STAGE_COLUMNS[4:]
            }})
    return rows


def ratios(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    index = {
        (r["stage"], r["dataset"], r["kernel"], r["workload"]): r for r in rows
    }
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for r in rows:
        key = (r["stage"], r["dataset"], r["workload"])
        if key in seen or r["kernel"] != "rabbit":
            continue
        seen.add(key)
        reference = index.get((r["stage"], r["dataset"], REFERENCE, r["workload"]))
        rabbit_median = build_table.number(r["median"])
        reference_median = build_table.number(reference["median"]) if reference else None
        value = build_table.ratio(r["workload"], rabbit_median, reference_median)
        note = ""
        if reference is None:
            note = "no reference row"
        elif int(r["errors"] or 0) or int(r["missing"] or 0) or int(reference["errors"] or 0) or int(reference["missing"] or 0):
            note = "errors or missing samples in the runs"
        out.append({
            "stage": r["stage"],
            "dataset": r["dataset"],
            "workload": r["workload"],
            "kind": "context" if r["workload"] in build_table.CONTEXT else "comparison",
            "rabbit": r["median"],
            "reference": reference["median"] if reference else "",
            "ratio": "" if value is None else f"{value:.3f}",
            "note": note,
        })
    return out


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "kvm",
    )
    parser.add_argument("--cpu", default="skylake-client")
    args = parser.parse_args()
    try:
        manifest_path = args.dir / "MANIFEST.csv"
        guard_outputs(manifest_path, args.dir)
        manifest = read_manifest(manifest_path)
        rows: list[dict[str, str]] = []
        for entry in manifest:
            rows += read_stage(
                args.dir / "stages" / entry["stage"], entry["stage"], entry["commit"], args.cpu
            )
    except (StageError, OSError) as error:
        print(f"stages: {error}", file=sys.stderr)
        return 1
    write_csv(args.dir / "stages.csv", STAGE_COLUMNS, rows)
    out = ratios(rows)
    write_csv(args.dir / "ratios_by_stage.csv", RATIO_COLUMNS, out)
    print(f"{len(rows)} rows and {len(out)} ratios written to {args.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
