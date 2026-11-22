#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-only
#
# The check that runs first on the rented host, before any measurement. It
# stops, with every reason it found, when the host cannot give a trustworthy
# accelerated number, so that the paid time is not spent on a run that would
# have to be thrown away. It writes nothing outside OUT.
#
# Usage: bench/kvm/preflight.sh [OUT_DIR]      (run from the unpacked bundle)
# Environment: CPU, the guest model, default Skylake-Client as in every dataset.

set -u

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
OUT=${1:-"$ROOT/preflight"}
CPU=${CPU:-Skylake-Client}
QEMU=${QEMU:-qemu-system-x86_64}
mkdir -p "$OUT"
LOG="$OUT/preflight.log"
: > "$LOG"
failed=0

say() { printf '%s\n' "$*" | tee -a "$LOG"; }
fail() { say "FAIL: $*"; failed=1; }

say "== host =="
{ uname -a; lscpu 2>/dev/null | grep -E 'Model name|Vendor ID|^CPU\(s\)|Thread|Core|MHz|Virtualization|Hypervisor' || true; } | tee -a "$LOG"

say "== virtualisation in the CPU =="
if grep -qE '(^| )(vmx|svm)( |$)' /proc/cpuinfo; then
    say "ok: the CPU advertises vmx or svm"
else
    fail "the CPU advertises neither vmx (VT-x) nor svm (AMD-V): stop and do not keep paying"
fi
if command -v systemd-detect-virt >/dev/null 2>&1; then
    kind=$(systemd-detect-virt 2>/dev/null || true)
    say "systemd-detect-virt: ${kind:-none}"
    if [ -n "$kind" ] && [ "$kind" != none ]; then
        fail "this host is itself a virtual machine ($kind): the numbers would be nested virtualisation, not bare metal"
    fi
fi

say "== /dev/kvm =="
if [ ! -e /dev/kvm ]; then
    fail "/dev/kvm does not exist (is the module loaded: sudo modprobe kvm_amd or kvm_intel)"
elif [ ! -r /dev/kvm ] || [ ! -w /dev/kvm ]; then
    fail "/dev/kvm exists but this user cannot open it: sudo usermod -aG kvm \$USER and log in again"
else
    ls -l /dev/kvm | tee -a "$LOG"
fi

say "== tools =="
for tool in "$QEMU" python3 taskset sha256sum timeout; do
    if command -v "$tool" >/dev/null 2>&1; then
        say "ok: $tool"
    else
        fail "missing: $tool"
    fi
done
command -v "$QEMU" >/dev/null 2>&1 && "$QEMU" --version | head -1 | tee -a "$LOG"

say "== artifacts =="
if (cd "$ROOT" && sha256sum -c SHA256SUMS) >> "$LOG" 2>&1; then
    say "ok: SHA256SUMS matches the files that arrived"
else
    fail "SHA256SUMS does not match: the bundle arrived damaged, see $LOG"
fi
[ -f "$ROOT/COMMIT" ] && say "rig commit: $(cat "$ROOT/COMMIT")"

if [ "$failed" -ne 0 ]; then
    say "PREFLIGHT FAILED before the boot test: fix the lines above or stop the server"
    exit 1
fi

say "== a boot of the Rabbit kernel under KVM, cpu $CPU =="
# The kernel exits through isa-debug-exit: status 33 is a boot that finished.
timeout 120 "$QEMU" -M q35 -cpu "$CPU" -accel kvm -m 512M \
    -kernel "$ROOT/artifacts/rabbit-kernel" \
    -display none -monitor none -no-reboot -serial "file:$OUT/boot-rabbit.serial.log" \
    -device isa-debug-exit,iobase=0xf4,iosize=0x04 \
    > "$OUT/boot-rabbit.qemu.log" 2>&1
status=$?
say "qemu exit status: $status (33 means the kernel booted to the end)"
if [ -s "$OUT/boot-rabbit.qemu.log" ]; then
    say "QEMU printed this, read it: a warning about features means the guest CPU is not the model asked for"
    tee -a "$LOG" < "$OUT/boot-rabbit.qemu.log"
fi
if [ "$status" -ne 33 ]; then
    fail "the Rabbit kernel did not boot to the end under KVM (status $status), see $OUT/boot-rabbit.serial.log"
fi

if [ "$failed" -ne 0 ]; then
    say "PREFLIGHT FAILED: do not start the session, send the log and decide whether to keep the server"
    exit 1
fi
say "PREFLIGHT OK"
