#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests of bench/policy/build_table.py, on a miniature set of raw logs.

Run with `python3 -m unittest discover -s bench/policy`.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import build_table


def adversary(kernel: str, named: dict[str, str], family: dict[str, int], summary: bool = True) -> str:
    """The raw log of an adversary: named cases, family rows and the summary."""
    lines = ["POLICY installed bytes=30"] if kernel == "rabbit" else ["CHILD AGENT policy abi8"]
    lines += [f"ADV case={name} verdict={verdict}" for name, verdict in named.items()]
    index = 0
    for verdict, count in family.items():
        for _ in range(count):
            lines.append(f"ADV case=enum_{index:04} verdict={verdict}")
            index += 1
    if summary:
        every = list(named.values()) + [v for v, count in family.items() for _ in range(count)]
        lines.append(
            f"ADV summary cases={len(every)} allowed={every.count('allowed')} "
            f"denied={every.count('denied')} failed={every.count('failed')} invalid=0 kernel={kernel}"
        )
    return "\n".join(lines) + "\n"


def compromised(kernel: str, cases: dict[str, str], existence: str = "", extra: str = "") -> str:
    lines = [f"RAW case={name} syscall=x ret=0 verdict={verdict} reached=na" for name, verdict in cases.items()]
    if existence:
        lines.append(existence)
    if extra:
        lines.append(extra)
    verdicts = list(cases.values())
    lines.append(
        f"RAW summary cases={len(verdicts)} allowed={verdicts.count('allowed')} "
        f"denied={verdicts.count('denied')} unavailable={verdicts.count('unavailable')} "
        f"failed={verdicts.count('failed')} kernel={kernel}"
    )
    return "\n".join(lines) + "\n"


class Fixture:
    """A directory with the four logs and summary.json."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.complete = {name: {"complete": True} for name in build_table.LOGS.values()}
        self.logs = {
            "rabbit-adversary.log": adversary("rabbit", {"read_secret": "denied", "net_other": "denied",
                                                         "read_inside": "allowed", "list": "failed"},
                                              {"denied": 3, "failed": 1}),
            "linux-adversary.log": adversary("linux", {"read_secret": "denied", "net_other": "allowed",
                                                       "read_inside": "allowed", "list": "allowed"},
                                             {"denied": 2, "failed": 2}),
            "rabbit-compromised.log": compromised(
                "rabbit", {"creat": "unavailable", "stat_secret": "unavailable", "listen": "denied",
                           "fork": "unavailable"},
                "RAW existence syscall=stat outside=-38 missing=-38 oracle=no"),
            "linux-compromised.log": compromised(
                "linux", {"creat": "denied", "stat_secret": "allowed", "listen": "denied", "fork": "allowed"},
                "RAW existence syscall=stat outside=0 missing=-2 oracle=yes"),
        }

    def write(self) -> Path:
        for name, text in self.logs.items():
            (self.directory / name).write_text(text, encoding="utf-8")
        (self.directory / "summary.json").write_text(json.dumps({"logs": self.complete}), encoding="utf-8")
        return self.directory


class BuildTableTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = Fixture(Path(self.temporary.name))

    def rows(self) -> dict[tuple[str, str], list[str]]:
        rows, _ = build_table.build(self.fixture.write())
        return {(row[0], row[1]): row for row in rows}

    def test_the_classes_follow_the_two_verdicts(self) -> None:
        classify = build_table.classify
        self.assertEqual(classify("allowed", "allowed"), "same")
        self.assertEqual(classify("denied", "denied"), "same")
        self.assertEqual(classify("unavailable", "unavailable"), "same")
        self.assertEqual(classify("denied", "allowed"), "policy")
        self.assertEqual(classify("denied", "failed"), "policy")
        self.assertEqual(classify("unavailable", "allowed"), "surface")
        self.assertEqual(classify("unavailable", "denied"), "linux-denies")
        self.assertEqual(classify("unavailable", "failed"), "other")
        self.assertEqual(classify("failed", "allowed"), "other")
        self.assertEqual(classify("failed", "denied"), "other")
        self.assertEqual(classify(None, "denied"), "other")
        self.assertEqual(classify("allowed", None), "other")

    def test_a_row_gets_its_class_from_the_logs(self) -> None:
        rows = self.rows()
        self.assertEqual(rows[("adversary", "read_secret")][5], "same")
        self.assertEqual(rows[("adversary", "net_other")][5], "policy")
        self.assertEqual(rows[("adversary", "list")][5], "other")
        self.assertEqual(rows[("compromised", "creat")][5], "linux-denies")
        self.assertEqual(rows[("compromised", "stat_secret")][5], "surface")
        self.assertEqual(rows[("compromised", "listen")][5], "same")
        self.assertEqual(rows[("compromised", "fork")][5], "surface")

    def test_a_network_call_takes_its_verdict_from_what_reached_the_server(self) -> None:
        self.fixture.logs["linux-adversary.log"] = adversary(
            "linux", {"read_secret": "denied", "net_other": "failed", "read_inside": "allowed", "list": "allowed"},
            {"denied": 2, "failed": 2},
        ).replace("ADV summary", "ADV net case=net_other verdict=allowed reached=yes detail=failed\nADV summary")
        row = self.rows()[("adversary", "net_other")]
        self.assertEqual(row[3], "allowed")
        self.assertEqual(row[4], "yes")
        self.assertEqual(row[5], "policy")

    def test_the_family_has_one_row_per_side_and_the_invariant(self) -> None:
        rows = self.rows()
        rabbit = rows[("adversary", "family_rabbit")]
        linux = rows[("adversary", "family_linux")]
        self.assertEqual(rabbit[2], "allowed=0;denied=3;failed=1;invalid=0")
        self.assertEqual(linux[3], "allowed=0;denied=2;failed=2;invalid=0")
        self.assertIn("no_path_allowed=true", rabbit[6])
        self.assertNotIn(("adversary", "enum_0000"), rows)

    def test_an_allowed_path_of_the_family_breaks_the_invariant(self) -> None:
        self.fixture.logs["linux-adversary.log"] = adversary(
            "linux", {"read_secret": "denied", "net_other": "allowed", "read_inside": "allowed", "list": "allowed"},
            {"allowed": 1, "denied": 1},
        )
        self.assertIn("no_path_allowed=false", self.rows()[("adversary", "family_linux")][6])

    def test_the_existence_rows_say_why_rabbit_has_no_oracle(self) -> None:
        row = self.rows()[("compromised", "existence:stat")]
        self.assertEqual((row[2], row[3], row[5]), ("oracle=no", "oracle=yes", "surface"))

    def test_the_totals_count_the_named_cases_only(self) -> None:
        _, totals = build_table.build(self.fixture.write())
        by = {(table, klass): int(count) for table, klass, count in totals}
        self.assertEqual(by[("adversary", "all")], 4)
        self.assertEqual(by[("adversary", "policy")], 1)
        self.assertEqual(by[("compromised", "all")], 4)
        self.assertEqual(by[("compromised", "surface")], 2)
        self.assertEqual(by[("compromised", "linux-denies")], 1)

    def test_a_log_without_a_summary_fails_unless_it_is_declared_incomplete(self) -> None:
        self.fixture.logs["rabbit-compromised.log"] = "\n".join(
            self.fixture.logs["rabbit-compromised.log"].splitlines()[:-1]
        ) + "\n"
        with self.assertRaisesRegex(build_table.TableError, "rabbit-compromised.log: the log has no summary"):
            build_table.build(self.fixture.write())

        self.fixture.logs["rabbit-adversary.log"] = adversary(
            "rabbit", {"read_secret": "denied", "net_other": "denied", "read_inside": "allowed", "list": "failed"},
            {"denied": 3, "failed": 1}, summary=False)
        self.fixture.logs["rabbit-compromised.log"] = compromised(
            "rabbit", {"creat": "unavailable", "stat_secret": "unavailable", "listen": "denied",
                       "fork": "unavailable"})
        self.fixture.complete["rabbit-adversary.log"] = {"complete": False}
        rows = self.rows()
        self.assertIn("complete=no", rows[("adversary", "family_rabbit")][6])

    def test_a_summary_that_does_not_match_the_cases_fails(self) -> None:
        self.fixture.logs["linux-compromised.log"] = self.fixture.logs["linux-compromised.log"].replace(
            "cases=4", "cases=5")
        with self.assertRaisesRegex(build_table.TableError, "summary counts 5 cases and the log has 4"):
            build_table.build(self.fixture.write())

        self.fixture.logs["linux-compromised.log"] = compromised(
            "linux", {"creat": "denied", "stat_secret": "allowed", "listen": "denied", "fork": "allowed"}
        ).replace("allowed=2", "allowed=3")
        with self.assertRaisesRegex(build_table.TableError, "allowed=3 and the cases give 2"):
            build_table.build(self.fixture.write())

    def test_a_missing_log_or_a_missing_declaration_fails(self) -> None:
        directory = self.fixture.write()
        (directory / "linux-adversary.log").unlink()
        with self.assertRaisesRegex(build_table.TableError, "linux-adversary.log: the raw log is missing"):
            build_table.build(directory)

        self.fixture.write()
        (directory / "summary.json").write_text(json.dumps({"logs": {}}), encoding="utf-8")
        with self.assertRaisesRegex(build_table.TableError, "does not say whether the log is complete"):
            build_table.build(directory)

    def test_an_identical_duplicate_is_kept_once_and_a_different_one_fails(self) -> None:
        duplicate = "RAW case=creat syscall=x ret=0 verdict=unavailable reached=na\n"
        text = self.fixture.logs["rabbit-compromised.log"]
        self.fixture.logs["rabbit-compromised.log"] = text.replace("RAW summary", duplicate + "RAW summary", 1)
        self.assertIn(("compromised", "creat"), self.rows())

        self.fixture.logs["rabbit-compromised.log"] = text.replace(
            "RAW summary", duplicate.replace("unavailable", "denied") + "RAW summary", 1)
        with self.assertRaisesRegex(build_table.TableError, "appears twice with different lines"):
            build_table.build(self.fixture.write())

    def test_the_csv_has_the_columns_and_one_row_per_case(self) -> None:
        directory = self.fixture.write()
        rows, totals = build_table.build(directory)
        build_table.write(directory, rows, totals)
        lines = (directory / "summary.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], ",".join(build_table.COLUMNS))
        self.assertEqual(len(lines) - 1, len(rows))
        self.assertEqual((directory / "totals.csv").read_text(encoding="utf-8").splitlines()[0],
                         "table,class,count")


class FieldsTest(unittest.TestCase):
    def test_a_gap_keeps_the_whole_reason(self) -> None:
        row = build_table.fields("case=x rabbit=Is(Denied) linux=allowed reached=yes why=a b = c")
        self.assertEqual(row["why"], "a b = c")
        self.assertEqual(row["reached"], "yes")


if __name__ == "__main__":
    unittest.main()
