#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests of bench/kvm/stages.py, on a miniature set of stages.

Run with `python3 -m unittest discover -s bench/kvm`.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import stages

HEADER = "kernel,workload,unit,samples,errors,missing,median,p99,stddev\n"


def make_run(stage: Path, cpu: str, number: int, rabbit: float, reference: float, accel: str = "kvm") -> None:
    for dataset in stages.DATASETS:
        directory = stage / f"kvm-{cpu}-r{number}" / dataset
        directory.mkdir(parents=True)
        rows = [
            f"rabbit,syscall,ns/op,30,0,0,{rabbit},{rabbit},0",
            f"mainline-nomitig,syscall,ns/op,30,0,0,{reference},{reference},0",
        ]
        (directory / "summary.csv").write_text(HEADER + "\n".join(rows) + "\n")
        (directory / "summary.json").write_text(json.dumps({"metadata": {"accel": accel}}))


def make_stage(root: Path, name: str, commit: str, medians: list[float], reference: float = 100.0) -> None:
    stage = root / "stages" / name
    stage.mkdir(parents=True)
    (stage / "BUNDLE-COMMIT").write_text(commit + "\n")
    for number, value in enumerate(medians, start=1):
        make_run(stage, "skylake-client", number, value, reference)


class Stages(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_runs_are_found_in_numeric_order(self) -> None:
        stage = self.root / "s"
        stage.mkdir()
        for number in (1, 2, 10, 3):
            make_run(stage, "skylake-client", number, 1, 1)
        names = [p.name for p in stages.run_directories(stage, "skylake-client")]
        self.assertEqual(names, ["kvm-skylake-client-r1", "kvm-skylake-client-r2", "kvm-skylake-client-r3", "kvm-skylake-client-r10"])

    def test_the_runs_of_another_cpu_are_not_mixed_in(self) -> None:
        stage = self.root / "s"
        stage.mkdir()
        make_run(stage, "skylake-client", 1, 1, 1)
        make_run(stage, "epyc-rome", 1, 99, 1)
        self.assertEqual(len(stages.run_directories(stage, "skylake-client")), 1)

    def test_a_stage_merges_its_runs_and_ratios_follow(self) -> None:
        make_stage(self.root, "after", "abc1234", [40, 50, 60])
        rows = stages.read_stage(self.root / "stages" / "after", "after", "abc1234", "skylake-client")
        out = stages.ratios(rows)
        syscall = [r for r in out if r["dataset"] == "operations"][0]
        self.assertEqual(syscall["rabbit"], "50.00")
        self.assertEqual(syscall["ratio"], "2.000")

    def test_a_loss_is_a_ratio_below_one_and_stays_in_the_file(self) -> None:
        make_stage(self.root, "worse", "abc1234", [200, 200, 200])
        rows = stages.read_stage(self.root / "stages" / "worse", "worse", "abc1234", "skylake-client")
        out = stages.ratios(rows)
        self.assertEqual([r["ratio"] for r in out if r["dataset"] == "agent"], ["0.500"])

    def test_a_bundle_built_from_another_commit_is_refused(self) -> None:
        make_stage(self.root, "s", "abc1234", [1, 1, 1])
        with self.assertRaises(stages.StageError) as caught:
            stages.read_stage(self.root / "stages" / "s", "s", "fff9999", "skylake-client")
        self.assertIn("built from", str(caught.exception))

    def test_a_short_commit_in_the_manifest_matches_the_full_one(self) -> None:
        make_stage(self.root, "s", "abc1234def5678", [1, 1, 1])
        rows = stages.read_stage(self.root / "stages" / "s", "s", "abc1234", "skylake-client")
        self.assertTrue(rows)

    def test_a_run_under_emulation_is_refused_in_a_kvm_stage(self) -> None:
        stage = self.root / "stages" / "s"
        stage.mkdir(parents=True)
        (stage / "BUNDLE-COMMIT").write_text("abc1234\n")
        make_run(stage, "skylake-client", 1, 1, 1)
        make_run(stage, "skylake-client", 2, 1, 1, accel="tcg")
        with self.assertRaises(stages.StageError):
            stages.read_stage(stage, "s", "abc1234", "skylake-client")

    def test_an_output_named_like_the_manifest_is_refused(self) -> None:
        manifest = self.root / "STAGES.csv"
        manifest.write_text("stage,commit,change\nx,abc,y\n")
        with self.assertRaises(stages.StageError):
            stages.guard_outputs(manifest, self.root)

    def test_the_real_manifest_does_not_collide_with_the_outputs(self) -> None:
        manifest = self.root / "MANIFEST.csv"
        manifest.write_text("stage,commit,change\nx,abc,y\n")
        stages.guard_outputs(manifest, self.root)

    def test_the_manifest_cannot_be_empty(self) -> None:
        path = self.root / "STAGES.csv"
        path.write_text("stage,commit,change\n")
        with self.assertRaises(stages.StageError):
            stages.read_manifest(path)


if __name__ == "__main__":
    unittest.main()
