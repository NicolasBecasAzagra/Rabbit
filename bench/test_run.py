#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Host tests of the pure helpers of bench/run.py.

Run with `python3 -m unittest discover -s bench -p 'test_run.py'`.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import run

RIG = Path(run.__file__).resolve()

INTEL = "processor : 0\nflags\t\t: fpu vme sse2 vmx ept\n"
AMD = "processor : 0\nflags\t\t: fpu vme sse2 svm npt\n"
NO_VIRT = "processor : 0\nflags\t\t: fpu vme sse2 hypervisor\n"


class KvmProblems(unittest.TestCase):
    def test_intel_host_with_a_usable_device_is_fine(self) -> None:
        self.assertEqual(run.kvm_problems("Linux", INTEL, True, True), [])

    def test_amd_host_with_a_usable_device_is_fine(self) -> None:
        self.assertEqual(run.kvm_problems("Linux", AMD, True, True), [])

    def test_a_vm_without_nested_virtualisation_is_refused(self) -> None:
        problems = run.kvm_problems("Linux", NO_VIRT, False, False)
        self.assertEqual(len(problems), 2)
        self.assertIn("vmx", problems[0])
        self.assertIn("/dev/kvm does not exist", problems[1])

    def test_a_device_the_user_cannot_open_is_refused(self) -> None:
        problems = run.kvm_problems("Linux", INTEL, True, False)
        self.assertEqual(len(problems), 1)
        self.assertIn("cannot open", problems[0])

    def test_unreadable_cpuinfo_is_refused_not_assumed(self) -> None:
        problems = run.kvm_problems("Linux", None, True, True)
        self.assertEqual(len(problems), 1)
        self.assertIn("unchecked", problems[0])

    def test_a_host_that_is_not_linux_is_refused(self) -> None:
        problems = run.kvm_problems("Darwin", None, False, False)
        self.assertEqual(problems, ["KVM is a Linux feature and this host is Darwin"])


class GuestCpu(unittest.TestCase):
    def command(self, **extra: str) -> list[str]:
        return run.qemu_command(
            "qemu", Path("k"), Path("i"), Path("l"), "tcg", **extra
        )

    def test_the_default_is_the_model_of_every_published_dataset(self) -> None:
        command = self.command()
        self.assertEqual(command[command.index("-cpu") + 1], "Skylake-Client")

    def test_the_model_is_the_one_asked_for(self) -> None:
        command = self.command(cpu="EPYC-Rome")
        self.assertEqual(command[command.index("-cpu") + 1], "EPYC-Rome")

    def test_the_host_cpu_is_named_in_the_metadata(self) -> None:
        text = "vendor_id\t: AuthenticAMD\nmodel name\t: AMD Ryzen 5 PRO 3600\n"
        self.assertEqual(
            run.host_cpu(text),
            {"cpu_model": "AMD Ryzen 5 PRO 3600", "cpu_vendor": "AuthenticAMD"},
        )
        self.assertIn("unavailable", run.host_cpu(None)["cpu_model"])

    def test_cpu_host_is_refused(self) -> None:
        message = NoBuild.refusal_of("--cpu", "host")
        self.assertIn("--cpu host is refused", message)


class NoBuild(unittest.TestCase):
    """The refusals of --no-build happen before anything is built or booted."""

    @staticmethod
    def refusal_of(*arguments: str) -> str:
        result = subprocess.run(
            [sys.executable, "-I", str(RIG), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        return result.stderr

    def refusal(self, *arguments: str) -> str:
        return self.refusal_of("--no-build", *arguments)

    def test_needs_the_kernel_and_the_archive(self) -> None:
        self.assertIn("needs both --rabbit and --initrd", self.refusal())
        self.assertIn(
            "needs both --rabbit and --initrd", self.refusal("--rabbit", "/x")
        )

    def test_a_missing_kernel_image_is_not_built(self) -> None:
        message = self.refusal("--rabbit", "/nonexistent/kernel", "--initrd", "/x")
        self.assertIn("Rabbit image not found", message)
        self.assertNotIn("building", message)


if __name__ == "__main__":
    unittest.main()
