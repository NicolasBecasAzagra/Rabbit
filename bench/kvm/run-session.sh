#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only
#
# The measurement session of issue 10, run once on the rented host: the
# preflight, then three datasets with `--accel kvm` and nothing built. It
# stops by itself when the time budget is spent, says which datasets it did
# not reach, and never powers the machine off: the server is the owner's to
# destroy.
#
# Usage: bench/kvm/run-session.sh        (from the unpacked bundle)
# Environment:
#   ACCEL         kvm (default). `tcg` rehearses the whole script on any Linux
#                 host, for free, and writes a dataset that says tcg.
#   MAX_MINUTES   stop starting new datasets after this long (default 150)
#   CORES         cores the guest and QEMU may use (default 2,3)
#   CPU           the guest model (default Skylake-Client)
#   OUT           where the results go (default results/<accel>-<UTC date>)

set -u

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT" || exit 1
ACCEL=${ACCEL:-kvm}
MAX_MINUTES=${MAX_MINUTES:-150}
CORES=${CORES:-2,3}
CPU=${CPU:-Skylake-Client}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=${OUT:-"$ROOT/results/$ACCEL-$STAMP"}
START=$(date +%s)
mkdir -p "$OUT"
SESSION="$OUT/session.log"
exec > >(tee -a "$SESSION") 2>&1

echo "session: accel=$ACCEL cpu=$CPU cores=$CORES budget=${MAX_MINUTES}min start=$STAMP"

if [ "$ACCEL" = kvm ]; then
    CPU=$CPU sh bench/kvm/preflight.sh "$OUT/preflight" || {
        echo "session: preflight failed, nothing was measured"
        echo "REMINDER: destroy the server if you are not going to fix this"
        exit 1
    }
fi

# What the numbers were produced on, next to the numbers.
{
    echo "== date =="; date -u
    echo "== uname =="; uname -a
    echo "== lscpu =="; lscpu 2>/dev/null || true
    echo "== /proc/cpuinfo (first core) =="; awk 'BEGIN{RS="";ORS="\n"} NR==1' /proc/cpuinfo
    echo "== memory =="; free -m 2>/dev/null || true
    echo "== qemu =="; qemu-system-x86_64 --version | head -1
    echo "== governor =="; cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c || echo "no cpufreq"
    echo "== kvm parameters =="; for f in /sys/module/kvm*/parameters/*; do [ -r "$f" ] && echo "$f=$(cat "$f")"; done 2>/dev/null
    echo "== bundle =="; cat COMMIT; cat SHA256SUMS
} > "$OUT/host-info.txt" 2>&1

# The governor is set only when the account may; the log of the host says which
# one the run had, so a run on `powersave` is read as such.
if [ -w /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g" 2>/dev/null || true; done
    echo "session: governor set to performance"
else
    echo "session: cannot set the governor (not root), host-info.txt records the one in use"
fi

A=artifacts
COMMON="--accel $ACCEL --cpu $CPU --no-build --rabbit $A/rabbit-kernel --mainline $A/vmlinux --mainline-variant nomitig=mitigations=off"

elapsed() { echo $(( ($(date +%s) - START) / 60 )); }
done_list=""
skipped_list=""

run_dataset() {
    name=$1; shift
    if [ "$(elapsed)" -ge "$MAX_MINUTES" ]; then
        echo "session: budget spent ($(elapsed) min), not starting $name"
        skipped_list="$skipped_list $name"
        return
    fi
    echo "session: $name starts at minute $(elapsed)"
    # shellcheck disable=SC2086
    if taskset -c "$CORES" python3 bench/run.py $COMMON --out "$OUT/$name" "$@"; then
        done_list="$done_list $name"
    else
        echo "session: $name FAILED, its partial rows and logs are kept in $OUT/$name"
        skipped_list="$skipped_list $name(failed)"
    fi
}

run_dataset operations --initrd "$A/ops.cpio" --memory 512M \
    --only syscall,mmap,weights,weights-tail,weights-huge,weights-huge-tail,futex,threads,boot-seconds,boot-rss-mb
run_dataset inference --initrd "$A/infer.cpio" --memory 1024M \
    --only infer-ttft,infer-tps,boot-seconds,boot-rss-mb
run_dataset agent --initrd "$A/agent.cpio" --memory 1024M \
    --only agent-steps,agent-tokens,agent-seconds,agent-step-ms,boot-seconds,boot-rss-mb

echo "session: done:${done_list:- none}"
echo "session: not done:${skipped_list:- none}"
echo "session: took $(elapsed) minutes"
tar -C "$(dirname "$OUT")" -czf "$OUT.tar.gz" "$(basename "$OUT")"
echo "session: results packed in $OUT.tar.gz, copy it to your machine with scp"
echo
echo "REMINDER: this script does not power anything off. Copy the results, then"
echo "REMINDER: destroy the server in the provider panel and check that it is gone."
