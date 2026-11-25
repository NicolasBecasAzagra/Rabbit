#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Derive the tables of tokens per second per code path and weight format.

`bench/kvm/paths.sh` runs the inference runtime once per run, with the code path
it asks for in `/infer.path`, and leaves a directory with `index.csv` (one row
per run), `meta.txt` (what the numbers were produced on) and one serial log per
run. This script reads the three and writes:

* `runs.csv`, one row per run, with the lines the guest printed;
* `table.csv`, one row per format, path and operating system kernel, with the
  median and the spread of the tokens per second and the median time to the
  first token;
* `ratios.csv`, the path against the scalar one for each format (above 1 means
  faster) and, when a mainline run exists, Rabbit against mainline on avx2;
* `tables.md`, the same two tables for people to read.

A run counts for the path the guest REPORTS on its `INFER path` line, never for
the one that was requested. A processor that lacks the requested path lowers it,
and a row whose reported path differs from the requested one says so in its note
and in the `mismatch` column, so a lowered run is never filed under the wrong
name. A run with no `INFER path` line is labelled `not reported` and does not
count. The `mismatch` column is also `yes` when `INFER path_limit` differs from
the request, which is what a stale `/infer.path` looks like. A run whose weights
are not the format that was asked for, that was killed by the timeout, that
printed an `INFER ERR` or `INFER error` line other than the reference mismatch, or
that printed no `INFER ttft_ms` and `INFER tok_per_s`, is not counted either.

The tokens a run generated (`INFER tokens`) have to equal those of the scalar path
of the same format, taken from the counted scalar runs of that format: a run that
does the same work as the others counts, and the speed of a run that decodes other
tokens is not a speed of the same work. A run whose tokens differ is listed with
the reason `tokens differ from the scalar path`. When the counted scalar runs of a
format disagree with each other, no run of that format counts and the cells say
`scalar runs disagree`; when a format has no counted scalar run, the others count
and the cells say `no scalar reference`.

The comparison with the float reference is information only. The runtime exits
with status 1 and prints `INFER ERR runtime: generated tokens differ from the
reference` when its tokens differ from the published reference, after it printed
the speed lines. That is expected for q4 and is a statement about the quantization,
not about the path, so that error line alone does not exclude a run. The
`reference_match` column (true, false or unknown, from `INFER golden ... match=`)
keeps it.

The exit status is nonzero when no run counted or any cell has no counted run;
the files are written all the same.

Under TCG the files carry `-emulator-not-a-speed` in their names and the tables
say it in their headers: the emulator gives the relative cost of instructions
that it translates, and no statement about the speed of a processor.

Usage: bench/kvm/paths.py DIRECTORY
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path
from typing import NamedTuple

FORMATS = ("bf16", "q8", "q4")
PATHS = ("scalar", "sse2", "avx2")
DTYPE = {"bf16": "BF16", "q8": "Q8B32", "q4": "Q4B32"}
KERNELS = ("rabbit", "mainline")
GOVERNOR_CAVEAT = "CPU governor was not performance"
INDEX_COLUMNS = [
    "format", "requested", "kernel", "rep", "qemu_status", "outcome", "wall_s", "log",
]
NOT_REPORTED = "not reported"
REFERENCE_MISMATCH_ERROR = "INFER ERR runtime: generated tokens differ from the reference"
DIFFERS = "tokens differ from the scalar path"
DISAGREE = "scalar runs disagree"
NO_SCALAR = "no scalar reference"
RUN_COLUMNS = [
    "mode", "guest_cpu", "host", "format", "requested", "reported", "path_limit",
    "mismatch", "kernel", "rep", "weights", "ttft_ms", "tok_per_s", "golden",
    "reference_match", "qemu_status", "outcome", "valid", "problem", "tokens", "log",
]
TABLE_COLUMNS = [
    "mode", "guest_cpu", "host", "format", "path", "requested", "kernel", "runs", "n",
    "median_tok_s", "min_tok_s", "max_tok_s", "median_ttft_ms", "reference_match", "note",
]
RATIO_COLUMNS = [
    "mode", "guest_cpu", "host", "kind", "format", "path", "requested", "against",
    "ratio_tok_s", "ratio_ttft", "reading", "note",
]

# The kernel of the machine can write a line of its own in front of or behind a
# line of the program, because both use the same serial port.
LEADING_KERNEL_LOG = re.compile(r"^\[[^\]]*\]\s*")
TRAILING_KERNEL_LOG = re.compile(r"\s*\[[^\]]*\].*$")
NUMBER = r"-?\d+(?:\.\d+)?"


class PathsError(Exception):
    """A directory that cannot be turned into tables: the message says why."""


class Derived(NamedTuple):
    """What `derive` wrote, and how many runs counted."""

    written: list[Path]
    counted: int
    total: int
    empty_cells: list[str]


def clean(line: str) -> str:
    return LEADING_KERNEL_LOG.sub("", line.replace("\r", "").strip())


def parse_log(text: str) -> dict[str, object]:
    """The lines of one run that the tables use, from the serial log of the guest."""
    found: dict[str, object] = {"ok": False, "error": ""}
    for raw in text.splitlines():
        line = clean(raw)
        if not line.startswith("INFER "):
            continue
        body = TRAILING_KERNEL_LOG.sub("", line) if not line.startswith("INFER text") else line
        for key, pattern in (
            ("path", r"^INFER path (\S+)$"),
            ("path_limit", r"^INFER path_limit (\S+)$"),
            ("ttft_ms", rf"^INFER ttft_ms ({NUMBER})$"),
            ("tok_per_s", rf"^INFER tok_per_s ({NUMBER})$"),
            ("weights", r"^INFER model=\S+ weights=(\S+) tensors=\d+$"),
            ("tokens", r"^INFER tokens ([0-9 ]+)$"),
            ("golden", r"^INFER golden .* match=(true|false)$"),
        ):
            match = re.match(pattern, body)
            if match and key not in found:
                found[key] = match.group(1)
        if body == "INFER OK":
            found["ok"] = True
        elif body == REFERENCE_MISMATCH_ERROR:
            found["reference_mismatch"] = True
        elif body.startswith("INFER error ") or body.startswith("INFER ERR "):
            found["error"] = body
    return found


def read_meta(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    try:
        text = path.read_text()
    except OSError as error:
        raise PathsError(f"cannot read {path}: {error}") from error
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip():
            meta[key.strip()] = value.strip()
    for key in ("accel", "cpu"):
        if not meta.get(key):
            raise PathsError(f"{path} does not say `{key}`, so the numbers have no mode")
    if meta["accel"] not in ("kvm", "tcg"):
        raise PathsError(f"{path}: accel is {meta['accel']!r}, expected kvm or tcg")
    return meta


def mode_label(meta: dict[str, str]) -> str:
    return "kvm" if meta["accel"] == "kvm" else "tcg: emulator, not a speed"


def host_label(meta: dict[str, str]) -> str:
    return f"one host: {meta.get('host_cpu', 'unknown')}"


def suffix(meta: dict[str, str]) -> str:
    return "" if meta["accel"] == "kvm" else "-emulator-not-a-speed"


def read_index(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise PathsError(f"cannot read {path}: {error}") from error
    if not rows:
        raise PathsError(f"{path} has no runs")
    missing = [column for column in INDEX_COLUMNS if column not in rows[0]]
    if missing:
        raise PathsError(f"{path} lacks the columns {', '.join(missing)}")
    return rows


def number(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def run_records(directory: Path, meta: dict[str, str], index: list[dict[str, str]]) -> list[dict[str, str]]:
    """One record per run, with the reported path and why a run does not count."""
    records = []
    problem_lists: list[list[str]] = []
    for row in index:
        log = directory / row["log"]
        try:
            text = log.read_text(errors="replace")
        except OSError:
            text = ""
        parsed = parse_log(text)
        reported = str(parsed.get("path", ""))
        weights = str(parsed.get("weights", ""))
        ttft = number(parsed.get("ttft_ms"))
        tok = number(parsed.get("tok_per_s"))
        limit = str(parsed.get("path_limit", ""))
        problems = []
        if not reported:
            problems.append("no INFER path line")
        elif reported not in PATHS:
            problems.append(f"unknown reported path {reported}")
        if ttft is None or tok is None:
            problems.append("no INFER ttft_ms or tok_per_s line")
        elif tok <= 0:
            problems.append("tok_per_s is not positive")
        if weights != DTYPE.get(row["format"], "?"):
            problems.append(
                f"weights are {weights or 'unreported'}, not {DTYPE.get(row['format'], '?')}"
            )
        if parsed["error"]:
            problems.append(str(parsed["error"]))
        if row["outcome"] == "timeout":
            problems.append("timeout")
        if limit and limit != row["requested"]:
            problems.append(
                f"path_limit {limit} differs from the request {row['requested']}"
            )
        counts = (
            reported in PATHS
            and ttft is not None
            and tok is not None
            and tok > 0
            and weights == DTYPE.get(row["format"], "?")
            and row["outcome"] != "timeout"
            and not parsed["error"]
        )
        records.append({
            "mode": mode_label(meta),
            "guest_cpu": meta["cpu"],
            "host": host_label(meta),
            "format": row["format"],
            "requested": row["requested"],
            "reported": reported or NOT_REPORTED,
            "path_limit": limit,
            "mismatch": "yes" if (
                reported != row["requested"] or (limit and limit != row["requested"])
            ) else "no",
            "kernel": row["kernel"],
            "rep": row["rep"],
            "weights": weights,
            "ttft_ms": "" if ttft is None else f"{ttft:.3f}",
            "tok_per_s": "" if tok is None else f"{tok:.3f}",
            "golden": str(parsed.get("golden", "")),
            "reference_match": str(parsed.get("golden", "unknown")),
            "qemu_status": row["qemu_status"],
            "outcome": row["outcome"],
            "valid": "yes" if counts else "no",
            "problem": "; ".join(problems),
            "tokens": str(parsed.get("tokens", "")),
            "log": row["log"],
        })
        problem_lists.append(problems)
    same_work(records, problem_lists)
    for record, problems in zip(records, problem_lists):
        record["problem"] = "; ".join(problems)
    return records


def same_work(records: list[dict[str, str]], problems: list[list[str]]) -> None:
    """Keep only the runs whose tokens equal those of the scalar path of their format."""
    for name in dict.fromkeys(record["format"] for record in records):
        runs = [
            (record, found) for record, found in zip(records, problems)
            if record["format"] == name and record["valid"] == "yes"
        ]
        tokens = {record["tokens"] for record, _ in runs if record["reported"] == "scalar"}
        if len(tokens) > 1:
            for record, found in runs:
                record["valid"] = "no"
                found.append(DISAGREE)
        elif not tokens:
            for _, found in runs:
                found.append(NO_SCALAR)
        else:
            (want,) = tokens
            for record, found in runs:
                if record["tokens"] != want:
                    record["valid"] = "no"
                    found.append(DIFFERS)


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def summarise(records: list[dict[str, str]], meta: dict[str, str]) -> list[dict[str, str]]:
    """One row per format, requested path, reported path and operating system kernel."""
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for record in records:
        key = (record["format"], record["kernel"], record["requested"], record["reported"])
        groups.setdefault(key, []).append(record)

    def order(key: tuple[str, str, str, str]) -> tuple[int, int, int, int]:
        name, kernel, requested, reported = key
        return (
            FORMATS.index(name) if name in FORMATS else len(FORMATS),
            KERNELS.index(kernel) if kernel in KERNELS else len(KERNELS),
            PATHS.index(requested) if requested in PATHS else len(PATHS),
            PATHS.index(reported) if reported in PATHS else len(PATHS),
        )

    rows = []
    for key in sorted(groups, key=order):
        name, kernel, requested, reported = key
        runs = groups[key]
        counted = [run for run in runs if run["valid"] == "yes"]
        tok = [float(run["tok_per_s"]) for run in counted]
        ttft = [float(run["ttft_ms"]) for run in counted]
        notes = []
        if reported == NOT_REPORTED:
            notes.append(f"NOT REPORTED: no INFER path line, requested {requested}")
        elif reported != requested:
            notes.append(
                f"LOWERED: requested {requested}, the guest reported {reported}"
            )
        stale = [run for run in runs if run["path_limit"] and run["path_limit"] != requested]
        if stale:
            notes.append(
                f"STALE REQUEST: path_limit differs from {requested} in {len(stale)} of {len(runs)} runs"
            )
        for flag in (DISAGREE, NO_SCALAR):
            if any(flag in run["problem"] for run in runs):
                notes.append(flag)
        if len(counted) < len(runs):
            notes.append(f"{len(runs) - len(counted)} of {len(runs)} runs do not count")
        matches = {run["reference_match"] for run in counted}
        rows.append({
            "mode": mode_label(meta),
            "guest_cpu": meta["cpu"],
            "host": host_label(meta),
            "format": name,
            "path": reported,
            "requested": requested,
            "kernel": kernel,
            "runs": str(len(runs)),
            "n": str(len(counted)),
            "median_tok_s": fmt(statistics.median(tok)) if tok else "",
            "min_tok_s": fmt(min(tok)) if tok else "",
            "max_tok_s": fmt(max(tok)) if tok else "",
            "median_ttft_ms": fmt(statistics.median(ttft)) if ttft else "",
            "reference_match": (
                "" if not matches else matches.pop() if len(matches) == 1 else "mixed"
            ),
            "note": "; ".join(notes),
        })
    return rows


def reading(num: dict[str, str], den: dict[str, str], ratio: float) -> str:
    """Tie, faster or slower, by the same rule as the rest of the datasets.

    The ranges of the two rows must not overlap for a difference to count. Under
    three runs a row has no spread worth the name, so no reading is given.
    """
    if int(num["n"]) < 3 or int(den["n"]) < 3:
        return "n under 3, no reading"
    overlap = float(num["min_tok_s"]) <= float(den["max_tok_s"]) and float(
        den["min_tok_s"]
    ) <= float(num["max_tok_s"])
    if overlap:
        return "tie (the ranges overlap)"
    return "faster" if ratio > 1 else "slower"


def ratios(table: list[dict[str, str]], meta: dict[str, str]) -> list[dict[str, str]]:
    """Path against scalar per format and kernel, and Rabbit against mainline."""
    usable = [row for row in table if row["n"] != "0" and row["median_tok_s"]]
    rows = []

    def base(row: dict[str, str]) -> dict[str, str] | None:
        for other in usable:
            if (
                other["format"] == row["format"]
                and other["kernel"] == row["kernel"]
                and other["requested"] == "scalar"
                and other["path"] == "scalar"
            ):
                return other
        return None

    def entry(kind: str, row: dict[str, str], against: str, den: dict[str, str], note: str) -> dict[str, str]:
        ratio = float(row["median_tok_s"]) / float(den["median_tok_s"])
        top, bottom = float(den["median_ttft_ms"]), float(row["median_ttft_ms"])
        ttft = f"{top / bottom:.3f}" if bottom > 0 else "n/a"
        return {
            "mode": mode_label(meta),
            "guest_cpu": meta["cpu"],
            "host": host_label(meta),
            "kind": kind,
            "format": row["format"],
            "path": row["path"],
            "requested": row["requested"],
            "against": against,
            "ratio_tok_s": f"{ratio:.3f}",
            "ratio_ttft": ttft,
            "reading": reading(row, den, ratio),
            "note": note,
        }

    for row in usable:
        scalar = base(row)
        if scalar is None or row is scalar:
            continue
        note = row["note"]
        rows.append(entry(
            "path-vs-scalar", row, f"{row['kernel']} scalar", scalar, note
        ))
    for row in usable:
        if row["kernel"] != "rabbit" or row["requested"] != "avx2":
            continue
        for other in usable:
            if (
                other["kernel"] == "mainline"
                and other["format"] == row["format"]
                and other["requested"] == "avx2"
            ):
                if other["path"] != row["path"]:
                    rows.append({
                        "mode": mode_label(meta),
                        "guest_cpu": meta["cpu"],
                        "host": host_label(meta),
                        "kind": "rabbit-vs-mainline",
                        "format": row["format"],
                        "path": row["path"],
                        "requested": row["requested"],
                        "against": f"mainline {other['path']}",
                        "ratio_tok_s": "",
                        "ratio_ttft": "",
                        "reading": "not comparable",
                        "note": "the two kernels ran different paths",
                    })
                else:
                    rows.append(entry(
                        "rabbit-vs-mainline", row, "mainline same path", other,
                        "; ".join(dict.fromkeys(n for n in (row["note"], other["note"]) if n)),
                    ))
    return rows


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def named(row: dict[str, str]) -> str:
    """The path of a row, with what was requested when the guest ran another."""
    if row["requested"] != row["path"]:
        return f"{row['path']} (requested {row['requested']})"
    return row["path"]


def md_table(columns: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell.replace("|", "/") for cell in row) + " |" for row in rows]
    return lines


def markdown(meta: dict[str, str], table: list[dict[str, str]], ratio_rows: list[dict[str, str]],
             records: list[dict[str, str]], governor: bool = False) -> str:
    emulator = meta["accel"] != "kvm"
    out = []
    if emulator:
        out += [
            "# Tokens per second per code path: EMULATOR, NOT A SPEED",
            "",
            "> **Emulator, not a speed.** These runs are under TCG. They prove that the",
            "> machinery works and say nothing about how fast any path is on a processor.",
            "> Do not quote a number of this file as a rate.",
            "",
        ]
    else:
        out += ["# Tokens per second per code path", ""]
    out += [
        f"* mode: **{mode_label(meta)}**",
        f"* guest CPU model: `{meta['cpu']}`, one virtual processor, {meta.get('memory', '?')} of memory",
        f"* {host_label(meta)} ({meta.get('host_system', 'unknown system')}); {meta.get('qemu', 'QEMU unknown')}",
        f"* runs per cell: {meta.get('reps', '?')}; the model: {meta.get('model', 'unknown')}",
        f"* Rabbit kernel sha256 `{meta.get('rabbit_sha256', 'unknown')}`",
        "* the path of a row is the one the guest reported, not the one requested",
        f"* runs that count: {sum(r['valid'] == 'yes' for r in records)} of {len(records)}",
    ]
    if governor:
        out += [
            "",
            "> **Caveat: the CPU governor of the host was not `performance`**, so the clock",
            "> could move during the runs. Read every rate of this file with that in mind.",
        ]
    out += [
        "",
        "## Tokens per second",
        "",
    ]
    rows = [
        [r["format"], named(r), r["kernel"], r["runs"], r["n"], r["median_tok_s"],
         f"{r['min_tok_s']}..{r['max_tok_s']}" if r["min_tok_s"] else "", r["median_ttft_ms"],
         r["reference_match"], r["note"]]
        for r in table
    ]
    out += md_table(
        ["format", "path", "kernel", "runs", "n", "median tok/s", "min..max", "median ttft ms",
         "reference match", "note"],
        rows,
    )
    out += ["", "## Ratios (above 1 means faster)", ""]
    if ratio_rows:
        out += md_table(
            ["kind", "format", "path", "against", "tok/s ratio", "ttft ratio", "reading", "note"],
            [[r["kind"], r["format"], named(r), r["against"], r["ratio_tok_s"],
              r["ratio_ttft"], r["reading"], r["note"]] for r in ratio_rows],
        )
    else:
        out += ["No ratio could be derived: a scalar row, or a mainline row, is missing."]
    bad = [r for r in records if r["valid"] != "yes"]
    if bad:
        out += ["", "## Runs that do not count", ""]
        out += md_table(
            ["format", "requested", "kernel", "rep", "outcome", "problem"],
            [[r["format"], r["requested"], r["kernel"], r["rep"], r["outcome"], r["problem"]]
             for r in bad],
        )
    out += [
        "",
        "## How to read it",
        "",
        "* A ratio of a path against the scalar one is a statement about the vector",
        "  kernels of the runtime. It is not a statement about the operating system and",
        "  it does not count for the stop rule of the comparison of Rabbit with mainline.",
        "* Rabbit against mainline on the same path and format is the comparison of the",
        "  two operating system kernels with the runtime held fixed.",
        "* A reading of tie means that the ranges of the two rows overlap. A decode",
        "  that is bound by the bandwidth of memory can tie on every path.",
        "* A run counts when its tokens equal those of the scalar path of the same format.",
        "  `reference match` is whether they also equal the published float reference:",
        "  false is expected for q4 and is a statement about the quantization, not about",
        "  the path.",
        "* Everything here is one host, one guest CPU model and one virtual processor.",
        "",
    ]
    return "\n".join(out)


def derive(directory: Path) -> Derived:
    meta = read_meta(directory / "meta.txt")
    index = read_index(directory / "index.csv")
    governor = any(row.get("governor_warning") == "yes" for row in index)
    records = run_records(directory, meta, index)
    table = summarise(records, meta)
    if governor:
        for row in table:
            row["note"] = "; ".join(filter(None, (row["note"], GOVERNOR_CAVEAT)))
    ratio_rows = ratios(table, meta)
    tag = suffix(meta)
    written = []
    for name, columns, rows in (
        ("runs", RUN_COLUMNS, records),
        ("table", TABLE_COLUMNS, table),
        ("ratios", RATIO_COLUMNS, ratio_rows),
    ):
        path = directory / f"{name}{tag}.csv"
        write_csv(path, columns, rows)
        written.append(path)
    md = directory / f"tables{tag}.md"
    md.write_text(markdown(meta, table, ratio_rows, records, governor))
    written.append(md)
    empty = [
        f"{row['format']} {row['requested']} {row['kernel']} (reported {row['path']})"
        for row in table
        if row["n"] == "0"
    ]
    return Derived(
        written, sum(r["valid"] == "yes" for r in records), len(records), empty
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path, help="the output directory of paths.sh")
    args = parser.parse_args()
    try:
        result = derive(args.directory)
    except PathsError as error:
        print(f"paths: {error}", file=sys.stderr)
        return 1
    for path in result.written:
        print(f"paths: wrote {path}")
    print(f"paths: counted {result.counted} of {result.total} runs")
    if result.counted == 0:
        print("paths: no run counted", file=sys.stderr)
        return 2
    if result.empty_cells:
        print("paths: cells with no counted run: " + "; ".join(result.empty_cells), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
