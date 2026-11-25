# Tokens per second per code path, under KVM

The dataset of issue 29: the runtime of `rabbit-infer` on every code path
(`scalar`, `sse2`, `avx2`) and every weight format (`bf16`, `q8`, `q4`), measured
on one host with hardware acceleration. Every number is **one host, one guest CPU
model and one virtual processor**, and it supports no statement about other hardware.

| | |
|---|---|
| machine | one bare metal host, AMD Ryzen 5 PRO 3600 (Zen 2), 6 cores, Ubuntu 24.04, Linux 6.8.0-149, QEMU 8.2.2 |
| guest | `Skylake-Client`, one virtual processor, 1024 MiB, `taskset -c 2,3`, governor `performance`, the RAID resync of the host frozen |
| model | `HuggingFaceTB/SmolLM2-135M`, revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` (the checkpoint of `bench/reference/reference.json`), bfloat16 and its Q8B32 and Q4B32 copies from `rabbit-infer-quantize` |
| rig | `bench/kvm/paths.sh` at commit `64e1bbeb836016df57854b66ad4efe527209b025`, 5 repetitions per cell, interleaved, 60 boots |
| kernels | the Rabbit kernel (sha256 in `meta.txt`) and the mainline `vmlinux` of the other datasets with `mitigations=off`, both booting the same archive |

The tables are derived from the logs by `bench/kvm/paths.py` (`tables.md`,
`table.csv`, `ratios.csv`, `runs.csv`); `index.csv` lists the boots and `logs/`
holds the serial and QEMU output of each. Nothing in the tables was edited by
hand.

What the tables show and what they do not is in
[the report](../../../REPORT.md), section "Results: vector kernels for Q8 and
Q4". In short: the vector paths are 2.0 to 5.6 times faster than the scalar one in
the runtime; Rabbit and mainline tie on the same path, so the gain belongs to the
runtime and not to the operating system; the tokens of bf16 and Q8 equal the
published reference on every path, and those of Q4 do not, as published before.

What it does not show: other processors, more than one virtual processor, a
longer context, the instruct checkpoint, AVX-512, a real accelerator, or any
machine that is not virtualised.
