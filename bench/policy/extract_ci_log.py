#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Cut the lines of the policy comparison out of the log of a CI run.

The log comes from `gh run view RUN --log`. Each line of it is

    JOB <tab> STEP <tab> [BOM]TIMESTAMP PAYLOAD

where the timestamp is `YYYY-MM-DDTHH:MM:SS.fffffffZ` and one space follows it.
This script removes the job, the step, the byte order mark and the timestamp
with the space, and nothing else: the payload is copied as it is. It keeps, in
the order of the log, the payload lines that start with one of `ADV `, `RAW `,
`GAP `, `POLICY ` or `CHILD AGENT policy`, from two steps:

* the step of the Rabbit boots. The first boot runs the adversary and the second
  the compromised binary; the second starts after the line that says the
  archive `compromised.cpio` was written.
* the step of the Linux policy. It runs three tests one after another, each of
  them after a `Finished` line of cargo: the policy test (which prints no line
  of the comparison), the adversary and the compromised binary.

Usage: bench/policy/extract_ci_log.py LOG OUTPUT_DIRECTORY
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

RABBIT_STEP = "Run the adversary and the compromised binary against Rabbit"
LINUX_STEP = "Linux policy on this runner"
PREFIXES = ("ADV ", "RAW ", "GAP ", "POLICY ", "CHILD AGENT policy")
TIMESTAMP = re.compile(r"^﻿?\d{4}-\d\d-\d\dT[\d:.]+Z ?")


def payloads(log: Path, step: str) -> list[str]:
    """The payload of every line of `step`, without the runner's prefix."""
    lines = []
    for line in log.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t", 2)
        if len(fields) < 3 or fields[1] != step:
            continue
        lines.append(TIMESTAMP.sub("", fields[2], count=1))
    return lines


def kept(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith(PREFIXES)]


def split_rabbit(lines: list[str]) -> tuple[list[str], list[str]]:
    """The lines of the adversary boot and of the compromised boot."""
    for index, line in enumerate(lines):
        if line.startswith("mkinitramfs: wrote ") and "compromised.cpio" in line:
            return kept(lines[:index]), kept(lines[index:])
    raise SystemExit("extract: the step of Rabbit has no line for compromised.cpio")


def split_linux(lines: list[str]) -> tuple[list[str], list[str]]:
    """The lines of the adversary test and of the compromised test."""
    sections: list[list[str]] = []
    for line in lines:
        if "Finished" in line and "`test` profile" in line:
            sections.append([])
        elif sections:
            sections[-1].append(line)
    if len(sections) != 3:
        raise SystemExit(f"extract: expected 3 test runs in the Linux step, found {len(sections)}")
    return kept(sections[1]), kept(sections[2])


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    log, out = Path(argv[1]), Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rabbit_adversary, rabbit_compromised = split_rabbit(payloads(log, RABBIT_STEP))
    linux_adversary, linux_compromised = split_linux(payloads(log, LINUX_STEP))
    for name, lines in (
        ("rabbit-adversary.log", rabbit_adversary),
        ("rabbit-compromised.log", rabbit_compromised),
        ("linux-adversary.log", linux_adversary),
        ("linux-compromised.log", linux_compromised),
    ):
        (out / name).write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        print(f"{name}: {len(lines)} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
