#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-only
#
# Pack, on the machine that has the toolchain, everything the rented host needs
# to run the rig under KVM without building anything: the rig itself at a
# committed revision, the kernel images, the three initramfs archives and a
# SHA256SUMS file. The server then needs no Rust, no model download and no
# compiler, which is most of the paid time that a first run would otherwise take.
#
# Usage: bench/kvm/bundle.sh RABBIT_KERNEL VMLINUX OPS_CPIO INFER_CPIO AGENT_CPIO OUT.tar
#
# Run it from a clean checkout: the commit it records is the one the rig was
# taken from, and a dirty tree would make that commit a lie.

set -eu

if [ "$#" -ne 6 ]; then
    echo "usage: $0 RABBIT_KERNEL VMLINUX OPS_CPIO INFER_CPIO AGENT_CPIO OUT.tar" >&2
    exit 2
fi
KERNEL=$1 VMLINUX=$2 OPS=$3 INFER=$4 AGENT=$5 OUT=$6

REPO=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$REPO"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "bundle: the working tree has uncommitted changes, commit or stash them" >&2
    exit 1
fi
COMMIT=$(git rev-parse HEAD)

for file in "$KERNEL" "$VMLINUX" "$OPS" "$INFER" "$AGENT"; do
    if [ ! -f "$file" ]; then
        echo "bundle: not a file: $file" >&2
        exit 1
    fi
done

# macOS has shasum, Linux has sha256sum.
if command -v sha256sum >/dev/null 2>&1; then
    sum() { sha256sum "$@"; }
else
    sum() { shasum -a 256 "$@"; }
fi

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
ROOT="$STAGE/rabbit-kvm"
mkdir -p "$ROOT/artifacts"
# The rig and its helpers, without the published datasets: they are large and
# the server does not need them.
git archive --format=tar HEAD bench | tar -x -C "$ROOT"
rm -rf "$ROOT/bench/results"
cp "$KERNEL" "$ROOT/artifacts/rabbit-kernel"
cp "$VMLINUX" "$ROOT/artifacts/vmlinux"
cp "$OPS" "$ROOT/artifacts/ops.cpio"
cp "$INFER" "$ROOT/artifacts/infer.cpio"
cp "$AGENT" "$ROOT/artifacts/agent.cpio"
printf '%s\n' "$COMMIT" > "$ROOT/COMMIT"
(cd "$ROOT" && sum artifacts/* > SHA256SUMS)

tar -C "$STAGE" -cf "$OUT" rabbit-kvm
echo "bundle: $OUT"
echo "bundle: commit $COMMIT"
cat "$ROOT/SHA256SUMS"
