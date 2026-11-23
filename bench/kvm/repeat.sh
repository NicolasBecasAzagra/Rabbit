#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-only
#
# Repeat the KVM session for one or more guest CPU models, each repetition in
# its own directory, so the spread between runs is in the data and no claim
# rests on a single run. Run it on the rented host from the unpacked bundle.
#
# Usage: bench/kvm/repeat.sh REPETITIONS MODEL [MODEL...]
# Output: results/kvm-<model lowercased>-r<N>/ for every model and repetition.

set -u

if [ "$#" -lt 2 ]; then
    echo "usage: $0 REPETITIONS MODEL [MODEL...]" >&2
    exit 2
fi
REPS=$1
shift
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT" || exit 1

for model in "$@"; do
    tag=$(printf '%s' "$model" | tr 'A-Z_,' 'a-z--')
    n=1
    while [ "$n" -le "$REPS" ]; do
        echo "repeat: $model run $n of $REPS"
        CPU=$model OUT="$ROOT/results/kvm-$tag-r$n" MAX_MINUTES=${MAX_MINUTES:-150} \
            bench/kvm/run-session.sh > "$ROOT/results/session-$tag-r$n.out" 2>&1 \
            || echo "repeat: $model run $n FAILED, see results/session-$tag-r$n.out"
        n=$((n + 1))
        sleep 5
    done
done
echo "repeat: finished"
