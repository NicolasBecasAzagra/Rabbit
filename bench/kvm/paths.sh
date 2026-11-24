#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only
#
# Tokens per second per code path (scalar, sse2, avx2) and per weight format
# (bf16, q8, q4), so that one day of the rented host can produce the dataset.
#
# For every format and path it builds an initramfs with the model staged as
# /model.safetensors and the path as /infer.path, boots the Rabbit kernel REPS
# times with the QEMU recipe of bench/run.py, and keeps each serial log. With
# --mainline it boots the same archive on the mainline kernel for the avx2 path
# of each format. bench/kvm/paths.py then derives the tables, counting every run
# under the path the guest REPORTS and not the one that was requested.
#
# The runs are interleaved (repetition, then format, then path), so a drift of
# the host over the day is spread over every cell and not over one.
#
# Usage: bench/kvm/paths.sh (--synthetic | --model DIR [--q8 DIR] [--q4 DIR])
#                           [--out DIR] [--accel kvm|tcg] [--cpu MODEL] [--reps N]
#                           [--rabbit KERNEL] [--mainline BZIMAGE]
#                           [--mainline-append ARGS] [--infer ELF] [--memory SIZE]
#                           [--timeout SECONDS] [--cores LIST]
#                           [--formats LIST] [--paths LIST] [--keep-archives]
#
#   --out DIR     where the run is written; a relative DIR resolves from the root
#                 of the repository. By default $PATHS_OUT (else $HOME/.cache/rabbit)
#                 plus paths-<UTC timestamp>, outside the tree so the repository
#                 stays clean
#   --synthetic   the small model of the CI, written and quantized here
#   --model DIR   a checkpoint directory (bench/fetch-model.sh); its quantized
#                 copies are DIR-q8 and DIR-q4 unless --q8 and --q4 say otherwise
#   --accel       kvm (default) refuses a host that cannot run it, exactly as
#                 bench/run.py does; tcg is the emulator and no speed
#   --cpu         the guest model, Skylake-Client by default; `host` is refused
#   --reps        runs per cell, 5 by default
#   --cores       taskset list for the QEMU process, 2,3 under kvm; under tcg
#                 only when given
#   --mainline-append  kernel arguments of the mainline runs, mitigations=off
#
# Environment: MAKE_OPTS (for example `-o musl-std`) is passed to the make calls,
# QEMU names the emulator, PATHS_OUT is the parent of the default --out.
#
# Exit status: nonzero when no run counted or any cell has no counted run (the
# tables are still written), and when the free disk space cannot hold the
# archives: every archive embeds a copy of its weights.

set -u

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT" || exit 1

QEMU=${QEMU:-qemu-system-x86_64}
MAKE_OPTS=${MAKE_OPTS:-}
ACCEL=kvm
CPU=Skylake-Client
REPS=5
MEMORY=1024M
TIMEOUT=600
CORES=
OUT=
SYNTHETIC=0
MODEL=
Q8=
Q4=
RABBIT=
MAINLINE=
MAINLINE_APPEND=mitigations=off
INFER_ELF=
FORMATS="bf16 q8 q4"
PATHS="scalar sse2 avx2"
KEEP=0

die() { echo "paths: $*" >&2; exit 1; }
need() { [ "$#" -ge 2 ] || die "$1 needs a value"; }

while [ "$#" -gt 0 ]; do
    case $1 in
        --synthetic) SYNTHETIC=1 ;;
        --model) need "$@"; MODEL=$2; shift ;;
        --q8) need "$@"; Q8=$2; shift ;;
        --q4) need "$@"; Q4=$2; shift ;;
        --out) need "$@"; OUT=$2; shift ;;
        --accel) need "$@"; ACCEL=$2; shift ;;
        --cpu) need "$@"; CPU=$2; shift ;;
        --reps) need "$@"; REPS=$2; shift ;;
        --rabbit) need "$@"; RABBIT=$2; shift ;;
        --mainline) need "$@"; MAINLINE=$2; shift ;;
        --mainline-append) need "$@"; MAINLINE_APPEND=$2; shift ;;
        --infer) need "$@"; INFER_ELF=$2; shift ;;
        --memory) need "$@"; MEMORY=$2; shift ;;
        --timeout) need "$@"; TIMEOUT=$2; shift ;;
        --cores) need "$@"; CORES=$2; shift ;;
        --formats) need "$@"; FORMATS=$(printf '%s' "$2" | tr ',' ' '); shift ;;
        --paths) need "$@"; PATHS=$(printf '%s' "$2" | tr ',' ' '); shift ;;
        --keep-archives) KEEP=1 ;;
        -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
    shift
done

case $ACCEL in kvm|tcg) ;; *) die "--accel is kvm or tcg" ;; esac
case $CPU in
    host|host,*) die "--cpu host is refused: it changes between machines and invalidates the comparison" ;;
esac
case $REPS in ''|*[!0-9]*|0) die "--reps is a positive number" ;; esac
case $TIMEOUT in ''|*[!0-9]*|0) die "--timeout is a positive number of seconds" ;; esac
seen=" "
for f in $FORMATS; do
    case $f in bf16|q8|q4) ;; *) die "unknown format $f" ;; esac
    case $seen in *" $f "*) die "format $f is given twice in --formats" ;; esac
    seen="$seen$f "
done
seen=" "
for p in $PATHS; do
    case $p in scalar|sse2|avx2) ;; *) die "unknown path $p" ;; esac
    case $seen in *" $p "*) die "path $p is given twice in --paths" ;; esac
    seen="$seen$p "
done
[ -n "$FORMATS" ] || die "--formats names no format"
[ -n "$PATHS" ] || die "--paths names no path"
if [ "$SYNTHETIC" = 1 ] && [ -n "$MODEL" ]; then die "--synthetic and --model exclude each other"; fi
if [ "$SYNTHETIC" = 0 ] && [ -z "$MODEL" ]; then die "give --synthetic or --model DIR"; fi

# KVM is checked by the function of the rig itself, before anything is built.
if [ "$ACCEL" = kvm ]; then
    python3 -c 'import sys; sys.path.insert(0, "bench"); import run; run.check_kvm_host()' \
        || exit 1
    [ -z "$CORES" ] && CORES=2,3
    command -v taskset >/dev/null 2>&1 || die "taskset is missing, and the KVM runs are pinned"
fi
if [ -n "$CORES" ] && ! command -v taskset >/dev/null 2>&1; then
    die "--cores needs taskset"
fi
# One check of the core list, so a host with fewer cores stops here and not at
# the first boot of sixty.
if [ "$ACCEL" = kvm ] && [ "$(uname -s)" = Linux ]; then
    taskset -c "$CORES" true >/dev/null 2>&1 \
        || die "taskset -c $CORES failed: this host lacks those cores, give --cores LIST"
fi
command -v "$QEMU" >/dev/null 2>&1 || die "$QEMU is not installed"
for tool in python3 cpio find; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool is missing"
done

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
[ -n "$OUT" ] || OUT=${PATHS_OUT:-$HOME/.cache/rabbit}/paths-$STAMP
mkdir -p "$OUT/logs" "$OUT/work" || die "cannot create $OUT"
OUT=$(CDPATH='' cd -- "$OUT" && pwd)

if command -v sha256sum >/dev/null 2>&1; then
    sum() { sha256sum "$1" | cut -d' ' -f1; }
else
    sum() { shasum -a 256 "$1" | cut -d' ' -f1; }
fi

# Where the models and the programs come from.
if [ "$SYNTHETIC" = 1 ]; then
    MODEL=$OUT/work/model-bf16
    Q8=$OUT/work/model-q8
    Q4=$OUT/work/model-q4
    echo "paths: writing the synthetic model and its quantized copies under $OUT/work"
    # shellcheck disable=SC2086
    make $MAKE_OPTS --no-print-directory infer-model INFER_MODEL_DIR="$MODEL" >> "$OUT/work/build.log" 2>&1 \
        || die "the synthetic model was not written, see $OUT/work/build.log"
    for bits in 8 4; do
        dest=$OUT/work/model-q$bits
        # shellcheck disable=SC2086
        make $MAKE_OPTS --no-print-directory quantize QUANT_BITS=$bits QUANT_IN="$MODEL" QUANT_OUT="$dest" \
            >> "$OUT/work/build.log" 2>&1 || die "quantize $bits failed, see $OUT/work/build.log"
    done
else
    [ -n "$Q8" ] || Q8=${MODEL%/}-q8
    [ -n "$Q4" ] || Q4=${MODEL%/}-q4
fi
model_dir() {
    case $1 in bf16) echo "$MODEL" ;; q8) echo "$Q8" ;; q4) echo "$Q4" ;; esac
}
for f in $FORMATS; do
    d=$(model_dir "$f")
    [ -f "$d/model.safetensors" ] && [ -f "$d/config.json" ] \
        || die "$d is not a model directory for $f (model.safetensors and config.json)"
done

# Every archive embeds its own copy of the weights (nothing is shared between
# archives), and mkinitramfs.sh makes one more copy of the largest model in
# TMPDIR while it builds. Estimate both from the sizes of the model files and
# stop before building anything if the free space cannot hold them.
MARGIN_KIB=65536
free_kib() { df -Pk "$1" | awk 'NR == 2 { print $4 }'; }
need_kib=0
largest_kib=0
# shellcheck disable=SC2086
npaths=$(set -- $PATHS; echo $#)
for f in $FORMATS; do
    bytes=$(wc -c < "$(model_dir "$f")/model.safetensors" | tr -d ' ')
    kib=$(( (bytes + 1023) / 1024 + 8192 ))
    need_kib=$((need_kib + kib * npaths))
    [ "$kib" -gt "$largest_kib" ] && largest_kib=$kib
done
need_out=$((need_kib + largest_kib + MARGIN_KIB))
have_out=$(free_kib "$OUT")
if [ -z "$have_out" ] || [ "$have_out" -lt "$need_out" ]; then
    die "$OUT needs about $((need_out / 1024)) MiB free for the archives (${npaths} paths x the weights of each format, plus one transient copy) and has $(( ${have_out:-0} / 1024 )) MiB; free space or give --out on a larger disk"
fi
have_tmp=$(free_kib "${TMPDIR:-/tmp}")
if [ -z "$have_tmp" ] || [ "$have_tmp" -lt $((largest_kib + MARGIN_KIB)) ]; then
    die "${TMPDIR:-/tmp} needs about $(( (largest_kib + MARGIN_KIB) / 1024 )) MiB free for the transient copy of the weights and has $(( ${have_tmp:-0} / 1024 )) MiB; set TMPDIR to a larger disk"
fi

if [ -z "$INFER_ELF" ]; then
    if [ -f artifacts/rabbit-infer.elf ]; then INFER_ELF=artifacts/rabbit-infer.elf
    else INFER_ELF=rabbit-infer/rabbit-infer.elf; fi
fi
if [ ! -f "$INFER_ELF" ]; then
    command -v make >/dev/null 2>&1 || die "$INFER_ELF is missing and make is not here"
    # shellcheck disable=SC2086
    make $MAKE_OPTS --no-print-directory infer >> "$OUT/work/build.log" 2>&1 \
        || die "make infer failed, see $OUT/work/build.log"
fi
[ -f "$INFER_ELF" ] || die "the runtime $INFER_ELF does not exist"

if [ -z "$RABBIT" ]; then
    if [ -f artifacts/rabbit-kernel ]; then RABBIT=artifacts/rabbit-kernel
    else
        # shellcheck disable=SC2086
        RABBIT=$(make $MAKE_OPTS -s --no-print-directory print-kernel-path 2>/dev/null)
        if [ ! -f "$RABBIT" ]; then
            # shellcheck disable=SC2086
            make $MAKE_OPTS --no-print-directory build >> "$OUT/work/build.log" 2>&1 \
                || die "make build failed, see $OUT/work/build.log"
        fi
    fi
fi
[ -f "$RABBIT" ] || die "the Rabbit kernel $RABBIT does not exist"
if [ -n "$MAINLINE" ] && [ ! -f "$MAINLINE" ]; then die "the mainline image $MAINLINE does not exist"; fi

# What the numbers were produced on, next to the numbers.
if [ -r /proc/cpuinfo ]; then
    HOST_CPU=$(awk -F: '/^model name/ { sub(/^ +/, "", $2); print $2; exit }' /proc/cpuinfo)
else
    HOST_CPU=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)
fi
[ -n "$HOST_CPU" ] || HOST_CPU=unknown
{
    echo "accel=$ACCEL"
    echo "cpu=$CPU"
    echo "reps=$REPS"
    echo "memory=$MEMORY"
    echo "timeout=$TIMEOUT"
    echo "cores=${CORES:-none}"
    echo "formats=$FORMATS"
    echo "paths=$PATHS"
    echo "model=$([ "$SYNTHETIC" = 1 ] && echo 'synthetic CI model' || echo "$MODEL")"
    echo "host_cpu=$HOST_CPU"
    echo "host_system=$(uname -srm)"
    echo "qemu=$("$QEMU" --version | head -1)"
    echo "date=$STAMP"
    echo "commit=$(git rev-parse HEAD 2>/dev/null || cat COMMIT 2>/dev/null || echo unknown)"
    echo "rabbit_kernel=$RABBIT"
    echo "rabbit_sha256=$(sum "$RABBIT")"
    echo "mainline_kernel=${MAINLINE:-none}"
    echo "mainline_sha256=$([ -n "$MAINLINE" ] && sum "$MAINLINE" || echo none)"
    echo "mainline_append=$MAINLINE_APPEND"
    echo "runtime_sha256=$(sum "$INFER_ELF")"
} > "$OUT/meta.txt"
# A governor other than performance lets the clock move during a run. KVM numbers
# are then taken with a caveat that travels with them, in the index and the table.
GOVERNOR_WARNING=no
governors=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u | tr '\n' ' ')
if [ "$ACCEL" = kvm ] && [ -n "$governors" ] && [ "$governors" != "performance " ]; then
    GOVERNOR_WARNING=yes
    {
        echo "paths: WARNING: the CPU governor is ${governors% }, not performance."
        echo "paths: WARNING: the clock can change during a run and move the numbers."
        echo "paths: WARNING: set it (cpupower frequency-set -g performance) or read the table with the caveat it carries."
    } >&2
fi
{
    echo "== date =="; date -u
    echo "== uname =="; uname -a
    echo "== lscpu =="; lscpu 2>/dev/null || true
    echo "== governor =="; cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c || true
    echo "== kvm parameters =="; for f in /sys/module/kvm*/parameters/*; do [ -r "$f" ] && echo "$f=$(cat "$f")"; done 2>/dev/null
} > "$OUT/host-info.txt" 2>&1

# One archive per format and path: the same files for every path, and the one
# word of /infer.path. The model files are linked, not copied, to the stage.
ARCH=$OUT/work/archives
mkdir -p "$ARCH"
for f in $FORMATS; do
    for p in $PATHS; do
        stage=$OUT/work/stage-$f-$p
        rm -rf "$stage"
        mkdir -p "$stage"
        src=$(CDPATH='' cd -- "$(model_dir "$f")" && pwd)
        for file in "$src"/*; do
            [ -f "$file" ] && ln -s "$file" "$stage/${file##*/}"
        done
        rm -f "$stage/infer.path"
        printf '%s\n' "$p" > "$stage/infer.path"
        EXTRA_DIR=$stage bench/mkinitramfs.sh "$ARCH/$f-$p.cpio" "$INFER_ELF" \
            >> "$OUT/work/build.log" 2>&1 || die "the archive of $f $p was not built, see $OUT/work/build.log"
        rm -rf "$stage"
    done
done

echo "format,requested,kernel,rep,qemu_status,outcome,wall_s,log,governor_warning" > "$OUT/index.csv"

PID=
cleanup() {
    [ -n "$PID" ] && kill "$PID" 2>/dev/null
    [ "$KEEP" = 1 ] || rm -rf "$ARCH"
}
trap cleanup EXIT
trap 'echo "paths: interrupted" >&2; exit 130' INT TERM

# One boot. The command is the one of qemu_command in bench/run.py: -enable-kvm
# first under kvm, -M q35, the guest model, the serial port in a file, the exit
# device the kernels use.
run_one() {
    local fmt=$1 path=$2 label=$3 image=$4 append=$5 rep=$6
    local name="$fmt-$path-$label-r$rep"
    local log="logs/$name.serial.log"
    local args=()
    [ -n "$CORES" ] && args+=(taskset -c "$CORES")
    args+=("$QEMU")
    [ "$ACCEL" = kvm ] && args+=(-enable-kvm)
    local kargs=console=ttyS0
    [ -n "$append" ] && kargs="$kargs $append"
    # shellcheck disable=SC2054
    args+=(-M q35 -cpu "$CPU" -m "$MEMORY" -kernel "$image" -initrd "$ARCH/$fmt-$path.cpio"
        -append "$kargs" -display none -monitor none -no-reboot -serial "file:$OUT/$log"
        -device isa-debug-exit,iobase=0xf4,iosize=0x04)
    : > "$OUT/$log"
    echo "paths: $name: ${args[*]}" >> "$OUT/commands.txt"
    "${args[@]}" > "$OUT/logs/$name.qemu.log" 2>&1 &
    PID=$!
    local start=$SECONDS outcome=timeout status=
    while [ $((SECONDS - start)) -lt "$TIMEOUT" ]; do
        if ! kill -0 "$PID" 2>/dev/null; then outcome=exited; break; fi
        if tr -d '\r' < "$OUT/$log" | grep -Eq 'INFER (OK|ERR|error)'; then
            outcome=marker
            sleep 1
            break
        fi
        sleep 0.5
    done
    local wall=$((SECONDS - start))
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null
        wait "$PID" 2>/dev/null
        status=stopped
    else
        wait "$PID" 2>/dev/null
        status=$?
    fi
    # finished: the marker of success was seen, whether QEMU was stopped here
    # (status stopped) or the guest left by itself (its debug-exit status stays).
    # failed: the guest said INFER ERR or INFER error. exited: QEMU ended with no
    # marker, a real exit whose status says why. timeout: nothing in time.
    if [ "$outcome" != timeout ]; then
        if tr -d '\r' < "$OUT/$log" | grep -Eq 'INFER OK'; then outcome=finished
        elif tr -d '\r' < "$OUT/$log" | grep -Eq 'INFER (ERR|error)'; then outcome=failed
        else outcome=exited; fi
    fi
    PID=
    echo "$fmt,$path,$label,$rep,$status,$outcome,$wall,$log,$GOVERNOR_WARNING" >> "$OUT/index.csv"
    echo "paths: $name outcome=$outcome qemu_status=$status wall=${wall}s"
}

echo "paths: accel=$ACCEL cpu=$CPU reps=$REPS formats=[$FORMATS] paths=[$PATHS] out=$OUT"
[ "$ACCEL" = tcg ] && echo "paths: TCG is the emulator: these runs prove the machinery and are no speed"
rep=1
while [ "$rep" -le "$REPS" ]; do
    for f in $FORMATS; do
        for p in $PATHS; do
            run_one "$f" "$p" rabbit "$RABBIT" "" "$rep"
            if [ -n "$MAINLINE" ] && [ "$p" = avx2 ]; then
                run_one "$f" "$p" mainline "$MAINLINE" "$MAINLINE_APPEND" "$rep"
            fi
        done
    done
    rep=$((rep + 1))
done

# The tables are written even when nothing counted; the status says so.
python3 bench/kvm/paths.py "$OUT"
status=$?
[ "$status" -eq 0 ] || die "paths.py exited $status: no run counted, a cell has no counted run, or the tables were not derived (see above); the files are in $OUT"
