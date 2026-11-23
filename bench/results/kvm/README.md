# The KVM dataset

The rig of `bench/run.py` run with `--accel kvm` on a bare metal host, which closes
issue 10, and the series of versions of the Rabbit kernel that the run made
possible to measure. Every number of the table is derived by a script from the
raw data in this directory. Every number is one host, one processor model and one
virtual processor, and it supports no statement about other hardware.

**What it contradicts.** The ratios of the emulator do not hold on this host. The
advantages that Rabbit showed under emulation, from 1.3x to 123x, are from 1.2x to
2.3x under KVM, with 7.4x for the slowest fault of 2 MiB. Three rows that Rabbit
won under emulation are losses under KVM: the cost of a fault in memory with 4 KiB
pages and with 2 MiB pages, and the slowest fault of 4 KiB. See
[the report](../../../REPORT.md) for what it means, and "What this does not show"
below.

## The host

| | |
|---|---|
| machine | Scaleway Elastic Metal EM-A215R-HDD, bare metal, hourly, Paris |
| CPU | AMD Ryzen 5 PRO 3600, 6 cores and 12 threads, AMD-V; `systemd-detect-virt` says `none` |
| memory | 16 GB |
| system | Ubuntu 24.04.5 LTS, Linux 6.8.0-146 |
| emulator | QEMU 8.2.2 (Debian 1:8.2.2+ds-0ubuntu1.18), the same version as the datasets under emulation |
| guest | one virtual processor, 512 MiB for the operations and 1024 MiB for the model and the agent, as in every dataset |
| guest CPU | `Skylake-Client` for the table; `EPYC-Rome` and `Skylake-Client,-avx2,-fma` are in the first stage |
| pinning | `taskset -c 2,3`, governor `performance` (recorded in `host-info.txt`), the RAID resync of the host frozen |

The processor of the host is AMD and the guest model is an Intel one, so QEMU
masked five features of the model, which `preflight/` and the `*.qemu.log` files
record: `pcid`, `invpcid`, `hle`, `rtm` and `erms`. The first two are the ones that
the reference kernel uses to avoid flushing the TLB when it isolates the page
tables, so the column of the reference kernel with its mitigations on is probably
not what a Skylake machine would give. That is a reading, and it is not measured. The comparison that the table uses is the
reference with `mitigations=off`, as in every dataset. Between the two guest CPU
models, the medians of the first stage differ by at most 14% (the `mmap` of the
reference kernel, 138 against 119 us per MiB) and the system call of the reference
kernel with its mitigations on goes from 49 to 72 ns; the rest differs by 8% or
less.

## The table

Emulation is the relative comparison of two kernels under one emulator. KVM is the
only column that can support a statement about wall clock time, and only on this
host. A ratio above 1 means Rabbit is better, whichever the unit. A tie means that
the difference is smaller than twice the spread between the runs.

| workload | Rabbit, emulation (TCG) | mainline, emulation | ratio, emulation | Rabbit, KVM | mainline, KVM | ratio, KVM | direction |
|---|---|---|---|---|---|---|---|
| futex round trip | 6.532 us | 97.71 us | 14.96x | 1.308 us | 2.254 us | 1.72x | same side |
| map and touch memory, per MiB | 6.826 ms | 8.685 ms | 1.27x | 61.16 us | 138.4 us | 2.26x | same side |
| system call | 376.5 ns | 1.039 us | 2.76x | 42 ns | 49 ns | 1.17x | same side |
| four threads, spawn to join | 619.5 us | 4.589 ms | 7.41x | 106.6 us | 135.7 us | 1.27x | same side |
| fault in memory, 4 KiB pages, per MiB | 2.011 ms | 13.73 ms | 6.83x | 237 us | 174.3 us | 0.73x | reverses |
| fault in memory, 2 MiB pages, per MiB | 824.8 us | 12.98 ms | 15.73x | 234.5 us | 178.1 us | 0.76x | reverses |
| slowest single fault, 2 MiB | 189.1 us | 23.33 ms | 123.32x | 5.714 us | 42.3 us | 7.40x | same side |
| slowest single fault, 4 KiB | 1.744 ms | 38.89 ms | 22.30x | 79.7 us | 41.06 us | 0.52x | reverses |
| tokens per second | 0.45 tok/s | 0.21 tok/s | 2.14x | 46.41 tok/s | 46.97 tok/s | 0.99x | tie under KVM |
| time to the first token | 11.25 s | 22.82 s | 2.03x | 109.7 ms | 114.4 ms | 1.04x | same side |
| seconds of an agent loop | 821 s | 2570 s | 3.13x | 13.48 s | 13.19 s | 0.98x | reverses |

Derived by `bench/kvm/build_table.py` from `bench/results/vps-1cpu/`,
`bench/results/inference-native/` and `bench/results/agent/` for the emulation, and
from the five runs of the last stage for KVM. The medians of the KVM column are the
median of the medians of five runs. The rows with their spread, the errors and the
missing samples are in `table/rows.csv`, and the ratios against both references in
`table/ratios.csv`.

## The stages

Five versions of the Rabbit kernel, each one measured by five runs of the three
datasets on this host. The ratio is Rabbit against the reference with
`mitigations=off`, above 1 when Rabbit is better. The change of each stage is in
`MANIFEST.csv`, and the commit the bundle was built from is in `BUNDLE-COMMIT` of its
directory.

| stage | system call | futex | four threads | mmap | 4 KiB faults | 2 MiB faults | slowest 4 KiB fault | first token | tokens/s | agent loop |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline-3bfb366 | 1.17x | 2.13x | 1.30x | 0.39x | 0.54x | 0.75x | 0.98x | 0.52x | 0.61x | 0.62x |
| vector-5f73cab | 1.17x | 1.71x | 1.28x | 0.39x | 0.55x | 0.76x | 0.94x | 0.76x | 1.02x | 0.99x |
| pin-131d288 | 1.17x | 1.72x | 1.29x | 0.39x | 0.56x | 0.76x | 0.92x | 0.99x | 1.00x | 0.98x |
| fault-f6b8abf | 1.17x | 1.71x | 1.28x | 1.21x | 0.60x | 0.75x | 0.54x | 1.03x | 0.99x | 0.98x |
| ahead-487051a | 1.17x | 1.72x | 1.27x | 2.26x | 0.73x | 0.76x | 0.52x | 1.04x | 0.99x | 0.98x |

Derived by `bench/kvm/stages.py` into `stages.csv` and `ratios_by_stage.csv`. Read
it with the losses in view:

* `vector` made Rabbit use AVX2 and the same instructions as the reference for the
  model. Before it, the runtime chose its code path by asking the processor, and
  Rabbit did not enable AVX, so **the same binary ran different code on the two
  kernels**. The first stage is therefore not a comparison of kernels for the model.
  With the guest model `Skylake-Client,-avx2,-fma`, where the reference cannot use
  AVX2 either, the tokens per second are 28.5 for Rabbit and 27.8 for the reference
  with its mitigations off (28.9 with them on).
* `vector` also corrected a defect: the kernel did not save the vector registers
  when it switched threads, and a test of two threads found 77 of 80 rounds
  corrupted before it and none after. It cost the futex a part of its advantage, from
  2.13x to 1.71x.
* `fault` and `ahead` made the cost of a fault lower, and the slowest single fault
  of 4 KiB higher, from 42 to 80 us, because a fault now does the work of up to 16.
* The 4 KiB fault is still a loss, and so is the 2 MiB one.

## What this does not show

* It is one host, one processor model and one virtual processor. It does not say
  how Rabbit compares on another processor, with more virtual processors, or on
  hardware that is not virtualised.
* The processors of the guest and the host are from different makers; see above.
* The agent and the model are one run of each kernel in each of five repeats of the
  same task, and the task is short. A ratio of 0.98 or 1.04 on them is a tie or a
  small difference, and not a ranking.
* The 4 KiB fault costs about 3280 cycles per page after `ahead`, and the handler of
  the kernel accounts for about 1300 of them. The rest is not explained here:
  the host has no counter of the cycles of the guest, so it could not be sampled.
  The diagnostics below price what was ruled out.

## The files

| path | what it is |
|---|---|
| `MANIFEST.csv` | the stages, the commit of each and what changed |
| `stages/<stage>/kvm-<cpu>-r<N>/<dataset>/summary.{csv,json}` | the summary of one run of one dataset; the json names the accelerator, the guest CPU, the QEMU version, the host and the command lines |
| `stages/<stage>/kvm-<cpu>-r<N>/host-info.txt` | the host as the run saw it (the first run of each stage; the rest are in the archive) |
| `stages/<stage>/raw-logs.tar.gz` | every serial log, QEMU log, preflight, session log and per iteration CSV of the stage, as the rig wrote them |
| `stages/<stage>/BUNDLE-COMMIT`, `BUNDLE-SHA256SUMS` | the commit the bundle was built from and the digests of the images and archives that ran |
| `stages.csv`, `ratios_by_stage.csv` | derived by `bench/kvm/stages.py` |
| `table/` | the table of this page, derived by `bench/kvm/build_table.py` |
| `diagnostics/` | measurements taken to find a cause, see `diagnostics/README.md` |

The `raw-logs.tar.gz` of a stage holds the files that were removed from its
directories to keep the repository small: on the exFAT disk this was written on, a
file takes a megabyte of space, and the data is more than a thousand files.
Extract it in place to get the layout the rig wrote.

## Reproduce

```sh
python3 bench/kvm/stages.py
F=bench/results/kvm/stages/ahead-487051a
python3 bench/kvm/build_table.py --out bench/results/kvm/table \
    --pair operations=bench/results/vps-1cpu,$F/kvm-skylake-client-r{1,2,3,4,5}/operations \
    --pair inference=bench/results/inference-native,$F/kvm-skylake-client-r{1,2,3,4,5}/inference \
    --pair agent=bench/results/agent,$F/kvm-skylake-client-r{1,2,3,4,5}/agent
```

To measure again, `bench/kvm/README.md` has the procedure for a host with `/dev/kvm`.
