#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Derive the table of the policy comparison from its raw logs.

The raw data is four logs in `bench/results/policy/`, cut from one CI run by
`bench/policy/extract_ci_log.py`: the adversary and the compromised binary on
Rabbit, and the same two on Linux with Landlock and seccomp. This script reads
them and writes, next to them:

* `summary.csv`, one row per case, with the verdict of each kernel, whether the
  data of a network call reached the server on Linux, and a class that is
  computed from the two verdicts and never typed by hand;
* `totals.csv`, the number of cases in each class.

The classes, for a case that both sides ran:

    same          the same verdict on both sides
    policy        Rabbit refused with its policy, and Linux allowed the call or
                  failed it for another reason: a difference of enforcement
    surface       Rabbit answered `unavailable` (ENOSYS) and Linux allowed the
                  call: Rabbit has no such call, which is not enforcement
    linux-denies  Linux denied the call and Rabbit answered `unavailable`
    other         anything else, among them a case only one side ran

The enumerated family of the adversary (`enum_NNNN`) has one row per side, with
the counts, and the invariant that no path of it was allowed.

Usage: bench/policy/build_table.py [--dir DIRECTORY]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "results" / "policy"

LOGS = {
    ("adversary", "rabbit"): "rabbit-adversary.log",
    ("adversary", "linux"): "linux-adversary.log",
    ("compromised", "rabbit"): "rabbit-compromised.log",
    ("compromised", "linux"): "linux-compromised.log",
}
CLASSES = ("same", "policy", "surface", "linux-denies", "other")
FAMILY_PREFIX = "enum_"
COLUMNS = ["table", "case", "rabbit_verdict", "linux_verdict", "reached_linux", "class", "note"]


class TableError(Exception):
    """A log that cannot be trusted: the message says which and why."""


def fields(text: str) -> dict[str, str]:
    """The `key=value` words of a line. `why=` takes the rest of the line."""
    result: dict[str, str] = {}
    words = text.split(" ")
    for index, word in enumerate(words):
        key, separator, value = word.partition("=")
        if not separator:
            continue
        if key == "why":
            result[key] = " ".join([value] + words[index + 1 :])
            break
        result[key] = value
    return result


class Log:
    """What one raw log says."""

    def __init__(self, name: str):
        self.name = name
        self.cases: dict[str, dict[str, str]] = {}
        self.nets: dict[str, dict[str, str]] = {}
        self.existence: dict[str, dict[str, str]] = {}
        self.gaps: dict[str, str] = {}
        self.summary: dict[str, str] | None = None

    def verdict(self, case: str) -> str:
        """The verdict of a case: what the servers say for a network call."""
        net = self.nets.get(case)
        return net["verdict"] if net else self.cases[case]["verdict"]

    def reached(self, case: str) -> str:
        net = self.nets.get(case)
        return net["reached"] if net else "na"


def parse(path: Path) -> Log:
    log = Log(path.name)
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith(("ADV case=", "RAW case=")):
            row = fields(line.split(" ", 1)[1])
            if row["case"] in log.cases and log.cases[row["case"]] != row:
                raise TableError(f"{path.name}:{number}: the case {row['case']} appears twice with different lines")
            log.cases[row["case"]] = row
        elif line.startswith(("ADV net ", "RAW net ")):
            row = fields(line.split(" ", 2)[2])
            log.nets[row["case"]] = row
        elif line.startswith("RAW existence "):
            row = fields(line.split(" ", 2)[2])
            log.existence[row["syscall"]] = row
        elif line.startswith(("ADV summary ", "RAW summary ")):
            if log.summary is not None:
                raise TableError(f"{path.name}:{number}: a second summary line")
            log.summary = fields(line.split(" ", 2)[2])
        elif line.startswith("GAP "):
            row = fields(line[4:])
            log.gaps[row["case"]] = row.get("why", "")
    return log


def check(log: Log, complete: bool, verdicts: tuple[str, ...]) -> None:
    """Fail unless the log has its summary and the cases add up to it."""
    if log.summary is None:
        if complete:
            raise TableError(f"{log.name}: the log has no summary line, and it is not declared incomplete")
        return
    expected = int(log.summary["cases"])
    if len(log.cases) != expected:
        raise TableError(f"{log.name}: the summary counts {expected} cases and the log has {len(log.cases)}")
    counted = Counter(row["verdict"] for row in log.cases.values())
    for verdict in verdicts:
        if counted[verdict] != int(log.summary.get(verdict, "0")):
            raise TableError(
                f"{log.name}: the summary says {verdict}={log.summary.get(verdict)} "
                f"and the cases give {counted[verdict]}"
            )
    unknown = set(counted) - set(verdicts) - {"invalid"}
    if unknown:
        raise TableError(f"{log.name}: unknown verdicts {sorted(unknown)}")


def classify(rabbit: str | None, linux: str | None) -> str:
    """The class of a case from the verdict of each side."""
    if rabbit is None or linux is None:
        return "other"
    if rabbit == linux:
        return "same"
    if rabbit == "denied" and linux in ("allowed", "failed"):
        return "policy"
    if rabbit == "unavailable" and linux == "allowed":
        return "surface"
    if rabbit == "unavailable" and linux == "denied":
        return "linux-denies"
    return "other"


def counts(log: Log, prefix_family: bool) -> Counter:
    return Counter(
        row["verdict"] for name, row in log.cases.items() if name.startswith(FAMILY_PREFIX) == prefix_family
    )


def note_for(case: str, klass: str, rabbit_log: Log, linux_log: Log, rabbit: str | None) -> str:
    notes = []
    why = linux_log.gaps.get(case) or rabbit_log.gaps.get(case)
    if why == "not asserted":
        why = "the test does not assert this answer"
    if why:
        notes.append(why)
    if rabbit is None:
        notes.append("not in the table of Rabbit")
    if klass == "surface":
        notes.append("Rabbit has no such call")
    return "; ".join(notes)


def table_rows(table: str, rabbit_log: Log, linux_log: Log) -> list[list[str]]:
    rows: list[list[str]] = []
    names = [name for name in linux_log.cases if not name.startswith(FAMILY_PREFIX)]
    names += [name for name in rabbit_log.cases if not name.startswith(FAMILY_PREFIX) and name not in names]
    for name in names:
        rabbit = rabbit_log.verdict(name) if name in rabbit_log.cases else None
        linux = linux_log.verdict(name) if name in linux_log.cases else None
        klass = classify(rabbit, linux)
        rows.append(
            [
                table,
                name,
                rabbit or "not run",
                linux or "not run",
                linux_log.reached(name) if linux else "na",
                klass,
                note_for(name, klass, rabbit_log, linux_log, rabbit),
            ]
        )
    return rows


def existence_rows(rabbit_log: Log, linux_log: Log) -> list[list[str]]:
    rows = []
    for syscall, linux in linux_log.existence.items():
        rabbit = rabbit_log.existence.get(syscall)
        if rabbit is None:
            continue
        secret = rabbit_log.cases.get(f"{syscall}_secret", {}).get("verdict")
        if rabbit["oracle"] == linux["oracle"]:
            klass = "same"
        elif rabbit["oracle"] == "no" and secret == "denied":
            klass = "policy"
        elif rabbit["oracle"] == "no" and secret == "unavailable":
            klass = "surface"
        else:
            klass = "other"
        note = (
            f"Rabbit: outside={rabbit['outside']} missing={rabbit['missing']}; "
            f"Linux: outside={linux['outside']} missing={linux['missing']}"
        )
        rows.append(["compromised", f"existence:{syscall}", f"oracle={rabbit['oracle']}",
                     f"oracle={linux['oracle']}", "na", klass, note])
    return rows


def family_row(side: str, log: Log, complete: bool) -> list[str]:
    seen = counts(log, True)
    total = sum(seen.values())
    verdict = ";".join(f"{name}={seen[name]}" for name in ("allowed", "denied", "failed", "invalid"))
    held = seen["allowed"] == 0
    note = (
        f"rows={total}; complete={'yes' if complete else 'no'}; no_path_allowed={'true' if held else 'false'}; "
        "the logs do not carry the paths, and no path of the family names a file inside the read root, "
        "so an allowed row would be a file outside it that was read"
    )
    if not complete:
        note += "; the CI step printed only the first 60 lines of the boot, so this is not the whole family"
    rabbit, linux = (verdict, "-") if side == "rabbit" else ("-", verdict)
    return ["adversary", f"family_{side}", rabbit, linux, "na", "family", note]


def build(directory: Path) -> tuple[list[list[str]], list[list[str]]]:
    metadata = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    completeness = {name: info["complete"] for name, info in metadata["logs"].items()}
    logs: dict[tuple[str, str], Log] = {}
    for key, name in LOGS.items():
        path = directory / name
        if not path.exists():
            raise TableError(f"{name}: the raw log is missing")
        if name not in completeness:
            raise TableError(f"{name}: summary.json does not say whether the log is complete")
        logs[key] = parse(path)
        verdicts = ("allowed", "denied", "failed") if key[0] == "adversary" else (
            "allowed", "denied", "unavailable", "failed")
        check(logs[key], completeness[name], verdicts)

    rows: list[list[str]] = []
    rows += table_rows("adversary", logs[("adversary", "rabbit")], logs[("adversary", "linux")])
    rows += [family_row("rabbit", logs[("adversary", "rabbit")], completeness[LOGS[("adversary", "rabbit")]])]
    rows += [family_row("linux", logs[("adversary", "linux")], completeness[LOGS[("adversary", "linux")]])]
    rows += table_rows("compromised", logs[("compromised", "rabbit")], logs[("compromised", "linux")])
    rows += existence_rows(logs[("compromised", "rabbit")], logs[("compromised", "linux")])

    totals: list[list[str]] = []
    for table in ("adversary", "compromised"):
        counted = Counter(
            row[5] for row in rows if row[0] == table and not row[1].startswith(("family_", "existence:"))
        )
        for klass in CLASSES:
            totals.append([table, klass, str(counted[klass])])
        totals.append([table, "all", str(sum(counted.values()))])
    return rows, totals


def write(directory: Path, rows: list[list[str]], totals: list[list[str]]) -> None:
    with (directory / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    with (directory / "totals.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["table", "class", "count"])
        writer.writerows(totals)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="the directory of the raw logs")
    arguments = parser.parse_args(argv[1:])
    try:
        rows, totals = build(arguments.dir)
    except TableError as error:
        print(f"build_table: {error}", file=sys.stderr)
        return 1
    write(arguments.dir, rows, totals)
    print("totals per class (the named cases, not the family and not the existence rows):")
    for table, klass, count in totals:
        print(f"  {table:12} {klass:13} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
