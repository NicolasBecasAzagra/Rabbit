#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests of bench/kvm/paths.py, on miniature directories of runs.

Run with `python3 -m unittest discover -s bench/kvm`.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import paths

DTYPE = {"bf16": "BF16", "q8": "Q8B32", "q4": "Q4B32"}
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "bench" / "kvm" / "paths.sh"


def serial(path: str, tok: float, ttft: float = 10.0, weights: str = "BF16",
           limit: str | None = None, golden: str | None = None, ok: bool = True,
           report_path: bool = True) -> str:
    lines = [
        "Rabbit: x86_64 kernel",
        f"INFER path_limit {limit if limit is not None else path}",
        f"INFER model=/model.safetensors weights={weights} tensors=38",
    ]
    if report_path:
        lines.append(f"INFER path {path}")
    lines += [
        "INFER tokens 1 2 3",
        f"INFER ttft_ms {ttft}",
        f"INFER tok_per_s {tok}",
    ]
    if golden:
        lines.append(f"INFER golden expected=1 2 3 actual=1 2 3 match={golden}")
    if ok:
        lines.append("INFER OK")
    return "\r\n".join(lines) + "\r\n"


class Directory:
    """A directory of runs, built the way paths.sh leaves it."""

    def __init__(self, accel: str = "kvm", cpu: str = "Skylake-Client") -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "logs").mkdir()
        (self.root / "meta.txt").write_text(
            f"accel={accel}\ncpu={cpu}\nhost_cpu=Test CPU\nreps=3\n"
        )
        self.rows: list[list[str]] = []
        self.governor = False

    def add(self, fmt: str, requested: str, kernel: str, rep: int, text: str,
            outcome: str = "exited", status: str = "33") -> None:
        name = f"{fmt}-{requested}-{kernel}-r{rep}.serial.log"
        (self.root / "logs" / name).write_text(text)
        self.rows.append([fmt, requested, kernel, str(rep), status, outcome, "5", f"logs/{name}"])

    def derive(self) -> None:
        with (self.root / "index.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow([*paths.INDEX_COLUMNS, "governor_warning"])
            writer.writerows([[*row, "yes" if self.governor else "no"] for row in self.rows])
        self.result = paths.derive(self.root)

    def read(self, name: str) -> list[dict[str, str]]:
        with (self.root / name).open(newline="") as handle:
            return list(csv.DictReader(handle))

    def close(self) -> None:
        self._tmp.cleanup()


class ParseLog(unittest.TestCase):
    def test_the_lines_of_a_run_are_read(self) -> None:
        parsed = paths.parse_log(serial("avx2", 12.5, 80.25, golden="true"))
        self.assertEqual(parsed["path"], "avx2")
        self.assertEqual(parsed["path_limit"], "avx2")
        self.assertEqual(parsed["weights"], "BF16")
        self.assertEqual(parsed["tok_per_s"], "12.5")
        self.assertEqual(parsed["ttft_ms"], "80.25")
        self.assertEqual(parsed["golden"], "true")
        self.assertEqual(parsed["tokens"], "1 2 3")
        self.assertTrue(parsed["ok"])

    def test_a_kernel_line_in_front_of_or_behind_a_line_does_not_hide_it(self) -> None:
        text = "[    1.5] INFER path sse2\nINFER tok_per_s 3.5 [   2.1] noise\n"
        parsed = paths.parse_log(text)
        self.assertEqual(parsed["path"], "sse2")
        self.assertEqual(parsed["tok_per_s"], "3.5")

    def test_the_error_of_an_unknown_path_is_kept(self) -> None:
        parsed = paths.parse_log("INFER error unknown path\nINFER ERR runtime: bad\n")
        self.assertFalse(parsed["ok"])
        self.assertIn("INFER ERR", str(parsed["error"]))
        self.assertNotIn("tok_per_s", parsed)

    def test_path_limit_is_not_taken_for_the_path(self) -> None:
        parsed = paths.parse_log("INFER path_limit avx2\n")
        self.assertNotIn("path", parsed)
        self.assertEqual(parsed["path_limit"], "avx2")


class Derive(unittest.TestCase):
    def setUp(self) -> None:
        self.d = Directory()
        self.addCleanup(self.d.close)

    def test_a_cell_is_the_median_and_the_spread_of_its_runs(self) -> None:
        for rep, tok in enumerate([10.0, 30.0, 20.0], start=1):
            self.d.add("bf16", "scalar", "rabbit", rep, serial("scalar", tok, ttft=100 + tok))
        self.d.derive()
        (row,) = self.d.read("table.csv")
        self.assertEqual(
            (row["n"], row["median_tok_s"], row["min_tok_s"], row["max_tok_s"], row["median_ttft_ms"]),
            ("3", "20.000", "10.000", "30.000", "120.000"),
        )
        self.assertEqual((row["mode"], row["guest_cpu"]), ("kvm", "Skylake-Client"))
        self.assertEqual(row["host"], "one host: Test CPU")

    def test_a_lowered_path_is_filed_under_what_the_guest_reported(self) -> None:
        self.d.add("q8", "scalar", "rabbit", 1, serial("scalar", 10.0, weights="Q8B32"))
        self.d.add("q8", "avx2", "rabbit", 1, serial("sse2", 15.0, weights="Q8B32", limit="avx2"))
        self.d.derive()
        runs = self.d.read("runs.csv")
        lowered = [r for r in runs if r["requested"] == "avx2"][0]
        self.assertEqual((lowered["reported"], lowered["mismatch"]), ("sse2", "yes"))
        table = self.d.read("table.csv")
        self.assertEqual([r["path"] for r in table if r["requested"] == "avx2"], ["sse2"])
        self.assertIn("LOWERED", [r for r in table if r["requested"] == "avx2"][0]["note"])
        self.assertNotIn("avx2", [r["path"] for r in table])
        ratios = self.d.read("ratios.csv")
        self.assertEqual(len(ratios), 1)
        self.assertEqual(ratios[0]["path"], "sse2")
        self.assertEqual(ratios[0]["requested"], "avx2")
        self.assertIn("LOWERED", ratios[0]["note"])
        self.assertIn("sse2 (requested avx2)", (self.d.root / "tables.md").read_text())

    def test_a_ratio_above_one_means_faster_for_both_columns(self) -> None:
        self.d.add("q4", "scalar", "rabbit", 1, serial("scalar", 10.0, ttft=200.0, weights="Q4B32"))
        self.d.add("q4", "avx2", "rabbit", 1, serial("avx2", 25.0, ttft=100.0, weights="Q4B32"))
        self.d.derive()
        (ratio,) = self.d.read("ratios.csv")
        self.assertEqual(ratio["kind"], "path-vs-scalar")
        self.assertEqual((ratio["ratio_tok_s"], ratio["ratio_ttft"]), ("2.500", "2.000"))
        self.assertEqual(ratio["reading"], "n under 3, no reading")

    def test_overlapping_ranges_are_a_tie_and_separate_ones_are_not(self) -> None:
        for rep, (a, b) in enumerate([(10, 10.5), (11, 11.5), (12, 12.5)], start=1):
            self.d.add("bf16", "scalar", "rabbit", rep, serial("scalar", a))
            self.d.add("bf16", "sse2", "rabbit", rep, serial("sse2", b))
            self.d.add("bf16", "avx2", "rabbit", rep, serial("avx2", 30 + rep))
        self.d.derive()
        readings = {r["path"]: r["reading"] for r in self.d.read("ratios.csv")}
        self.assertEqual(readings["sse2"], "tie (the ranges overlap)")
        self.assertEqual(readings["avx2"], "faster")

    def test_the_weights_have_to_be_the_format_that_was_asked_for(self) -> None:
        self.d.add("q4", "scalar", "rabbit", 1, serial("scalar", 10.0, weights="BF16"))
        self.d.derive()
        (row,) = self.d.read("table.csv")
        self.assertEqual((row["runs"], row["n"], row["median_tok_s"]), ("1", "0", ""))
        self.assertIn("weights are BF16, not Q4B32", self.d.read("runs.csv")[0]["problem"])
        self.assertEqual(self.d.read("ratios.csv"), [])

    def test_a_run_without_numbers_or_with_a_timeout_does_not_count(self) -> None:
        self.d.add("bf16", "scalar", "rabbit", 1, "INFER error unknown path\n", status="41")
        self.d.add("bf16", "scalar", "rabbit", 2, serial("scalar", 5.0, ok=False), outcome="timeout")
        self.d.add("bf16", "scalar", "rabbit", 3, serial("scalar", 8.0))
        self.d.derive()
        table = {r["path"]: r for r in self.d.read("table.csv")}
        row = table["scalar"]
        self.assertEqual((row["runs"], row["n"], row["median_tok_s"]), ("2", "1", "8.000"))
        self.assertIn("1 of 2 runs do not count", row["note"])
        self.assertEqual((table["not reported"]["runs"], table["not reported"]["n"]), ("1", "0"))
        self.assertEqual([r["valid"] for r in self.d.read("runs.csv")], ["no", "no", "yes"])

    def test_a_failed_golden_check_does_not_count_and_says_why(self) -> None:
        self.d.add("bf16", "scalar", "rabbit", 1, serial("scalar", 5.0, golden="false"))
        self.d.add("bf16", "scalar", "rabbit", 2, serial("scalar", 7.0, golden="true"))
        self.d.derive()
        runs = self.d.read("runs.csv")
        self.assertEqual([r["valid"] for r in runs], ["no", "yes"])
        self.assertIn("golden check failed", runs[0]["problem"])
        (row,) = self.d.read("table.csv")
        self.assertEqual((row["runs"], row["n"], row["median_tok_s"]), ("2", "1", "7.000"))
        text = (self.d.root / "tables.md").read_text()
        self.assertIn("Runs that do not count", text)
        self.assertIn("golden check failed", text)

    def test_a_zero_median_ttft_gives_no_ratio_and_no_exception(self) -> None:
        self.d.add("q4", "scalar", "rabbit", 1, serial("scalar", 10.0, ttft=50.0, weights="Q4B32"))
        self.d.add("q4", "avx2", "rabbit", 1, serial("avx2", 25.0, ttft=0.0, weights="Q4B32"))
        self.d.derive()
        (ratio,) = self.d.read("ratios.csv")
        self.assertEqual((ratio["ratio_tok_s"], ratio["ratio_ttft"]), ("2.500", "n/a"))

    def test_a_run_with_no_path_line_is_not_reported_and_does_not_count(self) -> None:
        self.d.add("bf16", "scalar", "rabbit", 1, serial("scalar", 10.0))
        self.d.add("bf16", "avx2", "rabbit", 1, serial("avx2", 30.0, report_path=False))
        self.d.derive()
        runs = self.d.read("runs.csv")
        silent = [r for r in runs if r["requested"] == "avx2"][0]
        self.assertEqual(
            (silent["reported"], silent["mismatch"], silent["valid"]), ("not reported", "yes", "no")
        )
        self.assertIn("no INFER path line", silent["problem"])
        table = self.d.read("table.csv")
        (cell,) = [r for r in table if r["requested"] == "avx2"]
        self.assertEqual((cell["path"], cell["n"], cell["median_tok_s"]), ("not reported", "0", ""))
        self.assertIn("NOT REPORTED", cell["note"])
        self.assertNotIn("avx2", [r["path"] for r in table])
        self.assertEqual(self.d.read("ratios.csv"), [])

    def test_a_path_limit_that_differs_from_the_request_is_a_mismatch(self) -> None:
        self.d.add("bf16", "sse2", "rabbit", 1, serial("sse2", 10.0, limit="scalar"))
        self.d.add("bf16", "scalar", "rabbit", 1, serial("scalar", 10.0))
        self.d.derive()
        stale = [r for r in self.d.read("runs.csv") if r["requested"] == "sse2"][0]
        self.assertEqual((stale["reported"], stale["path_limit"], stale["mismatch"]), ("sse2", "scalar", "yes"))
        self.assertIn("path_limit scalar differs", stale["problem"])
        cell = [r for r in self.d.read("table.csv") if r["requested"] == "sse2"][0]
        self.assertIn("STALE REQUEST", cell["note"])
        clean = [r for r in self.d.read("runs.csv") if r["requested"] == "scalar"][0]
        self.assertEqual(clean["mismatch"], "no")

    def test_the_governor_caveat_of_the_index_reaches_the_table(self) -> None:
        self.d.add("bf16", "scalar", "rabbit", 1, serial("scalar", 10.0))
        self.d.governor = True
        self.d.derive()
        (row,) = self.d.read("table.csv")
        self.assertIn("governor", row["note"])
        self.assertIn("Caveat", (self.d.root / "tables.md").read_text())

    def test_rabbit_against_mainline_on_the_same_path(self) -> None:
        self.d.add("bf16", "avx2", "rabbit", 1, serial("avx2", 30.0, ttft=50.0))
        self.d.add("bf16", "avx2", "mainline", 1, serial("avx2", 20.0, ttft=100.0))
        self.d.derive()
        (ratio,) = self.d.read("ratios.csv")
        self.assertEqual(ratio["kind"], "rabbit-vs-mainline")
        self.assertEqual((ratio["ratio_tok_s"], ratio["ratio_ttft"]), ("1.500", "2.000"))

    def test_two_kernels_that_ran_different_paths_are_not_compared(self) -> None:
        self.d.add("bf16", "avx2", "rabbit", 1, serial("avx2", 30.0))
        self.d.add("bf16", "avx2", "mainline", 1, serial("sse2", 20.0, limit="avx2"))
        self.d.derive()
        (ratio,) = self.d.read("ratios.csv")
        self.assertEqual((ratio["ratio_tok_s"], ratio["reading"]), ("", "not comparable"))


class Mode(unittest.TestCase):
    def test_tcg_says_emulator_in_the_names_and_the_headers(self) -> None:
        d = Directory(accel="tcg", cpu="Skylake-Client,-avx2,-fma")
        self.addCleanup(d.close)
        d.add("bf16", "scalar", "rabbit", 1, serial("scalar", 10.0))
        d.derive()
        for name in ("runs", "table", "ratios"):
            self.assertTrue((d.root / f"{name}-emulator-not-a-speed.csv").is_file())
            self.assertFalse((d.root / f"{name}.csv").exists())
        (row,) = d.read("table-emulator-not-a-speed.csv")
        self.assertEqual(row["mode"], "tcg: emulator, not a speed")
        self.assertEqual(row["guest_cpu"], "Skylake-Client,-avx2,-fma")
        text = (d.root / "tables-emulator-not-a-speed.md").read_text()
        self.assertIn("EMULATOR, NOT A SPEED", text.splitlines()[0])
        self.assertIn("one host", text)

    def test_kvm_has_plain_names_and_names_the_mode_and_the_host(self) -> None:
        d = Directory()
        self.addCleanup(d.close)
        d.add("bf16", "scalar", "rabbit", 1, serial("scalar", 10.0))
        d.derive()
        self.assertTrue((d.root / "table.csv").is_file())
        text = (d.root / "tables.md").read_text()
        self.assertIn("mode: **kvm**", text)
        self.assertIn("one host: Test CPU", text)
        self.assertNotIn("EMULATOR", text)

    def test_a_directory_with_no_mode_is_refused(self) -> None:
        d = Directory()
        self.addCleanup(d.close)
        (d.root / "meta.txt").write_text("cpu=Skylake-Client\n")
        d.add("bf16", "scalar", "rabbit", 1, serial("scalar", 10.0))
        with self.assertRaises(paths.PathsError):
            d.derive()
        (d.root / "meta.txt").write_text("accel=hvf\ncpu=Skylake-Client\n")
        with self.assertRaises(paths.PathsError):
            paths.derive(d.root)


class Exit(unittest.TestCase):
    def test_nothing_counted_is_a_nonzero_exit_and_the_tables_are_still_written(self) -> None:
        d = Directory()
        self.addCleanup(d.close)
        d.add("bf16", "scalar", "rabbit", 1, "INFER error unknown path\n", status="41")
        d.derive()
        self.assertEqual((d.result.counted, d.result.total), (0, 1))
        self.assertTrue((d.root / "table.csv").is_file())
        run = subprocess.run(
            [sys.executable, str(Path(paths.__file__)), str(d.root)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(run.returncode, 2)
        self.assertIn("counted 0 of 1 runs", run.stdout)
        self.assertIn("no run counted", run.stderr)

    def test_a_cell_with_no_counted_run_is_a_nonzero_exit(self) -> None:
        d = Directory()
        self.addCleanup(d.close)
        d.add("bf16", "scalar", "rabbit", 1, serial("scalar", 10.0))
        d.add("bf16", "sse2", "rabbit", 1, serial("sse2", 10.0, golden="false"))
        d.derive()
        self.assertEqual((d.result.counted, d.result.total), (1, 2))
        self.assertEqual(len(d.result.empty_cells), 1)
        run = subprocess.run(
            [sys.executable, str(Path(paths.__file__)), str(d.root)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(run.returncode, 2)
        self.assertIn("cells with no counted run", run.stderr)

    def test_all_cells_counted_is_a_zero_exit(self) -> None:
        d = Directory()
        self.addCleanup(d.close)
        d.add("bf16", "scalar", "rabbit", 1, serial("scalar", 10.0))
        d.derive()
        run = subprocess.run(
            [sys.executable, str(Path(paths.__file__)), str(d.root)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(run.returncode, 0, run.stderr)


class Script(unittest.TestCase):
    """paths.sh itself: it parses, lints, and refuses bad requests before building."""

    def run_script(self, *args: str) -> tuple[subprocess.CompletedProcess[str], Path]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name) / "out"
        run = subprocess.run(
            ["bash", str(SCRIPT), "--synthetic", "--out", str(out), *args],
            capture_output=True, text=True, check=False, cwd=tmp.name, timeout=120,
        )
        return run, out

    def test_the_script_parses(self) -> None:
        run = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_shellcheck_has_nothing_to_say(self) -> None:
        exe = shutil.which("shellcheck")
        if exe is None:
            self.skipTest("shellcheck is not on PATH")
        run = subprocess.run([exe, str(SCRIPT)], capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_a_word_given_twice_is_an_error_and_nothing_is_built(self) -> None:
        for flag, value in (("--formats", "bf16,q8,bf16"), ("--paths", "avx2,avx2")):
            run, out = self.run_script("--accel", "tcg", flag, value)
            self.assertNotEqual(run.returncode, 0, flag)
            self.assertIn("twice", run.stderr)
            self.assertFalse(out.exists(), flag)

    def test_kvm_is_refused_on_a_host_that_cannot_run_it(self) -> None:
        probe = subprocess.run(
            [sys.executable, "-c", "import sys; sys.path.insert(0, 'bench'); import run; run.check_kvm_host()"],
            capture_output=True, text=True, check=False, cwd=ROOT,
        )
        if probe.returncode == 0:
            self.skipTest("this host can run KVM, so it is not refused")
        run, out = self.run_script("--accel", "kvm")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("cannot run on this host", run.stderr)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
