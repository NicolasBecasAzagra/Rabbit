#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests of bench/kvm/build_table.py, on miniature datasets.

Run with `python3 -m unittest discover -s bench/kvm`.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import build_table

HEADER = "kernel,workload,unit,samples,errors,missing,median,p99,stddev\n"


def dataset(root: Path, name: str, accel: str, rows: list[str]) -> Path:
    directory = root / name
    directory.mkdir()
    (directory / "summary.csv").write_text(HEADER + "\n".join(rows) + "\n")
    (directory / "summary.json").write_text(json.dumps({"metadata": {"accel": accel}}))
    return directory


def row(kernel: str, workload: str, median: float, samples: int = 30, errors: int = 0) -> str:
    return f"{kernel},{workload},ns/op,{samples},{errors},0,{median},{median},0"


class Derive(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def ratios(self, tcg: list[str], kvm: list[str]) -> list[dict[str, str]]:
        pair = ("ops", dataset(self.root, "t", "tcg", tcg), dataset(self.root, "k", "kvm", kvm))
        return build_table.derive([pair])[1]

    def test_a_cost_ratio_is_the_reference_over_rabbit(self) -> None:
        ratios = self.ratios(
            [row("rabbit", "syscall", 100), row("mainline-nomitig", "syscall", 300)],
            [row("rabbit", "syscall", 50), row("mainline-nomitig", "syscall", 100)],
        )
        self.assertEqual(len(ratios), 1)
        self.assertEqual(ratios[0]["ratio_tcg"], "3.000")
        self.assertEqual(ratios[0]["ratio_kvm"], "2.000")
        self.assertEqual(ratios[0]["direction"], "same side")

    def test_a_rate_ratio_is_rabbit_over_the_reference(self) -> None:
        ratios = self.ratios(
            [row("rabbit", "infer-tps", 0.45), row("mainline", "infer-tps", 0.20)],
            [row("rabbit", "infer-tps", 9.0), row("mainline", "infer-tps", 12.0)],
        )
        self.assertEqual(ratios[0]["ratio_tcg"], "2.250")
        self.assertEqual(ratios[0]["ratio_kvm"], "0.750")
        self.assertEqual(ratios[0]["direction"], "reverses")

    def test_a_loss_under_kvm_is_published_not_dropped(self) -> None:
        ratios = self.ratios(
            [row("rabbit", "futex", 10), row("mainline-nomitig", "futex", 100)],
            [row("rabbit", "futex", 200), row("mainline-nomitig", "futex", 100)],
        )
        self.assertEqual(ratios[0]["direction"], "reverses")
        self.assertEqual(ratios[0]["ratio_kvm"], "0.500")

    def test_a_column_with_the_wrong_mode_is_refused(self) -> None:
        wrong = dataset(self.root, "w", "kvm", [row("rabbit", "syscall", 1)])
        right = dataset(self.root, "r", "kvm", [row("rabbit", "syscall", 1)])
        with self.assertRaises(build_table.TableError) as caught:
            build_table.derive([("ops", wrong, right)])
        self.assertIn("never mixed", str(caught.exception))

    def test_a_dataset_without_a_recorded_mode_is_refused(self) -> None:
        directory = dataset(self.root, "n", "tcg", [row("rabbit", "syscall", 1)])
        (directory / "summary.json").write_text(json.dumps({"metadata": {}}))
        other = dataset(self.root, "k", "kvm", [row("rabbit", "syscall", 1)])
        with self.assertRaises(build_table.TableError):
            build_table.derive([("ops", directory, other)])

    def test_missing_samples_leave_no_ratio_and_say_why(self) -> None:
        ratios = self.ratios(
            [row("rabbit", "syscall", 100), row("mainline-nomitig", "syscall", 300)],
            [row("rabbit", "syscall", 0, samples=0), row("mainline-nomitig", "syscall", 100)],
        )
        self.assertEqual(ratios[0]["ratio_kvm"], "")
        self.assertEqual(ratios[0]["kvm_status"], "no rabbit samples")
        self.assertEqual(ratios[0]["direction"], "unknown")

    def test_errors_are_carried_into_the_status(self) -> None:
        ratios = self.ratios(
            [row("rabbit", "syscall", 100), row("mainline-nomitig", "syscall", 300)],
            [row("rabbit", "syscall", 50), row("mainline-nomitig", "syscall", 100, errors=2)],
        )
        self.assertIn("errors=2", ratios[0]["kvm_status"])

    def test_the_boot_rows_are_context_not_comparison(self) -> None:
        ratios = self.ratios(
            [row("rabbit", "boot-seconds", 10), row("mainline", "boot-seconds", 50)],
            [row("rabbit", "boot-seconds", 5), row("mainline", "boot-seconds", 9)],
        )
        self.assertEqual(ratios[0]["kind"], "context")

    def test_every_row_carries_the_mode_of_its_dataset(self) -> None:
        pair = (
            "ops",
            dataset(self.root, "t", "tcg", [row("rabbit", "syscall", 1)]),
            dataset(self.root, "k", "kvm", [row("rabbit", "syscall", 1)]),
        )
        rows = build_table.derive([pair])[0]
        self.assertEqual(sorted(r["accel"] for r in rows), ["kvm", "tcg"])

    def test_a_workload_nobody_ran_is_not_a_row(self) -> None:
        pair = (
            "ops",
            dataset(self.root, "t", "tcg", [row("rabbit", "syscall", 1), row("mainline", "syscall", 2), row("rabbit", "futex", 0, samples=0)]),
            dataset(self.root, "k", "kvm", [row("rabbit", "syscall", 1), row("mainline", "syscall", 2), row("rabbit", "futex", 0, samples=0)]),
        )
        rows, ratios = build_table.derive([pair])
        self.assertEqual({r["workload"] for r in rows}, {"syscall"})
        self.assertEqual({r["workload"] for r in ratios}, {"syscall"})

    def test_the_files_are_written_with_the_declared_columns(self) -> None:
        pair = (
            "ops",
            dataset(self.root, "t", "tcg", [row("rabbit", "syscall", 1), row("mainline", "syscall", 2)]),
            dataset(self.root, "k", "kvm", [row("rabbit", "syscall", 1), row("mainline", "syscall", 2)]),
        )
        out = self.root / "out"
        out.mkdir()
        rows, ratios = build_table.derive([pair])
        build_table.write_csv(out / "rows.csv", build_table.ROW_COLUMNS, rows)
        build_table.write_csv(out / "ratios.csv", build_table.RATIO_COLUMNS, ratios)
        with (out / "ratios.csv").open() as handle:
            self.assertEqual(next(csv.reader(handle)), build_table.RATIO_COLUMNS)


if __name__ == "__main__":
    unittest.main()
