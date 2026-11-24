# The KVM session

Issue 10, closed by the data in `bench/results/kvm/`: the rig run with `--accel kvm`
on a host with `/dev/kvm`, and a table with two columns, one under emulation (a relative comparison) and one under
KVM (the only one that can support a statement about wall clock time).

This directory holds the scripts of that session. Nothing in it spends money by
itself: `run-session.sh` runs on the rented host once the owner has started it,
and it never powers the machine off.

| file | where it runs | what it does |
|---|---|---|
| `bundle.sh` | the machine with the toolchain | packs the rig at a commit, the kernel images, the three archives and `SHA256SUMS` into one tar |
| `preflight.sh` | the rented host, first | refuses a host without VT-x or AMD-V, without a usable `/dev/kvm`, that is itself a virtual machine, with a damaged bundle, or where the Rabbit kernel does not boot under KVM |
| `run-session.sh` | the rented host | the preflight, `host-info.txt`, then the datasets `operations`, `inference` and `agent`, with a time budget |
| `build_table.py` | the machine of the owner | derives `rows.csv` and `ratios.csv` from pairs of datasets, one under TCG and one under KVM |
| `repeat.sh` | the rented host | repeats the session for one or more guest CPU models, each repetition in its own directory |
| `paths.sh` | the rented host | tokens per second per code path and per weight format, one boot per run, with the path asked for in `/infer.path` |
| `paths.py` | the rented host (`paths.sh` calls it at the end); the owner can rerun it later on a copy of the directory | derives `runs`, `table` and `ratios` from the logs of `paths.sh`, counting each run under the path the guest reported |
| `stages.py` | the machine of the owner | derives the progress of the kernel, stage by stage, from the runs of `bench/results/kvm/stages/`, and refuses a stage that does not match its manifest |

## What is measured

| dataset | initramfs | memory | workloads | emulated side |
|---|---|---|---|---|
| `operations` | `ops.cpio` | 512M | the operation workloads and the boot rows | `bench/results/vps-1cpu/` |
| `inference` | `infer.cpio` | 1024M | `infer-ttft`, `infer-tps` | `bench/results/inference-native/` |
| `agent` | `agent.cpio` | 1024M | the agent rows | `bench/results/agent/` |

The network rows are not part of the session. Each of them builds a client and an
archive that carries the port of the run, so they need the toolchain on the host,
and `--no-build` refuses them.

## Rules the data follows

* Every dataset carries `accel` in `summary.json`, and `build_table.py` refuses a
  column whose metadata does not say the mode of that column.
* The guest CPU is `Skylake-Client` unless the preflight shows that QEMU cannot
  present it. `--cpu host` is refused. The model used, the host CPU and the
  warnings of QEMU (`*.qemu.log`) are published with the data.
* A ratio above 1 means Rabbit is better. A ratio below 1 is a loss and stays in
  the table. When the direction of a ratio changes between the two modes, the
  `direction` column says `reverses`, and README, REPORT, the website and the
  issues are updated with it.
* A host that is itself a virtual machine fails the preflight: the number would
  be nested virtualisation and not bare metal.

## The session, in order

On the machine with the toolchain, from a clean checkout:

```sh
bench/kvm/bundle.sh KERNEL VMLINUX OPS_CPIO INFER_CPIO AGENT_CPIO rabbit-kvm.tar
scp rabbit-kvm.tar USER@HOST:                 # key based SSH only
```

On the rented host:

```sh
sudo apt-get update && sudo apt-get install -y qemu-system-x86 python3
tar -xf rabbit-kvm.tar && cd rabbit-kvm
sh bench/kvm/preflight.sh                      # stop here if it says FAILED
bench/kvm/run-session.sh                       # MAX_MINUTES=150 by default
```

Back on the machine of the owner:

```sh
scp USER@HOST:rabbit-kvm/results/kvm-*.tar.gz .
# unpack it into bench/results/kvm-<date>/, then
# several runs of one dataset are merged: list them all after the emulated one
python3 bench/kvm/build_table.py --out bench/results/kvm-<date> \
    --pair operations=bench/results/vps-1cpu,RUN1/operations,RUN2/operations \
    --pair inference=bench/results/inference-native,RUN1/inference,RUN2/inference \
    --pair agent=bench/results/agent,RUN1/agent,RUN2/agent
```

Destroy the server in the panel of the provider when the tar is on your machine,
and check that it no longer runs.

## Rehearsal without paying

`ACCEL=tcg bench/kvm/run-session.sh` runs the whole script on any Linux host with
QEMU, skips the KVM preflight and writes datasets that say `tcg`. It is the way
to check the script before the paid host exists. The tests of the table run with
`python3 -m unittest discover -s bench/kvm`.

## The code paths and the weight formats

`paths.sh` measures the tokens per second of the runtime for each of the three
code paths (`scalar`, `sse2`, `avx2`) and each of the three weight formats
(`bf16`, `q8`, `q4`), so that one rental day can produce the dataset of issue 29.
For each cell it builds an initramfs with the model staged as `/model.safetensors`
and the word of the path in `/infer.path`, and boots the Rabbit kernel `--reps`
times (5 by default) with the QEMU recipe of `bench/run.py`: `-enable-kvm` under
KVM, `-cpu Skylake-Client` unless `--cpu` says otherwise (`host` is refused),
`taskset -c 2,3`. `--accel kvm` is checked by `run.check_kvm_host()`, the same
function as the rig, and is refused on a host that cannot run it. With
`--mainline BZIMAGE` the avx2 archive of each format is booted on the mainline
image too (with `mitigations=off`, as in the datasets; `--mainline-append` changes
it), which gives the ratio between the two operating system kernels. The runs are
interleaved (repetition, format, path), so a drift of the host is spread over
every cell.

Without `/infer.path` the runtime prints one extra line, `INFER path_limit none`,
on every run, and nothing else changes: the limit stays off. With it, the runtime
limits the path of the three products to the word and prints
`INFER path_limit` (what was asked) and, further on, `INFER path` (what will run). A processor
that lacks the path lowers it, and an unknown word stops the run with
`INFER error unknown path`. The tables use the `INFER path` line.

On the machine with the toolchain (`make infer`, `bench/fetch-model.sh`, the
quantizer for `DIR-q8` and `DIR-q4`, the kernel images from the session bundle):

```sh
tar -cf paths-kvm.tar --exclude=bench/results bench rabbit-infer/rabbit-infer.elf COMMIT \
    -C ~/rabbit artifacts/model-instruct artifacts/model-instruct-q8 \
    artifacts/model-instruct-q4 artifacts/rabbit-kernel artifacts/vmlinux
```

On the rented host, after `preflight.sh` passed (no toolchain is needed: the
archives are built by `bench/mkinitramfs.sh`, which needs `cpio` and `python3`):

```sh
bench/kvm/paths.sh --model artifacts/model-instruct --accel kvm \
    --infer rabbit-infer/rabbit-infer.elf --rabbit artifacts/rabbit-kernel \
    --mainline artifacts/vmlinux --out results/paths-kvm --memory 1024M
```

Without `--out` the run goes to `$PATHS_OUT/paths-<UTC timestamp>`, with
`PATHS_OUT` defaulting to `$HOME/.cache/rabbit`, outside the tree so the repository
stays clean. A relative `--out` resolves from the root of the repository.

Disk: each of the nine archives embeds its own copy of the weights, and
`bench/mkinitramfs.sh` makes one transient copy of the model it is packing in
`TMPDIR`. With the 135M checkpoint (`model.safetensors` of 269 MB for `bf16`, about
143 MB for `q8` and 76 MB for `q4`) the archives are 3 x (269 + 143 + 76) = about
1.5 GB, and with the transient copy and the logs the run needs about 1.8 GB free
under `--out`. The script computes the same figure from the sizes of the model
files and `df`, and stops before building anything, with the numbers, if the disk
is smaller; `TMPDIR` is checked for the transient copy alone. `--keep-archives`
keeps the 1.5 GB after the run.

Before the first boot the script checks the core list once (`taskset -c 2,3 true`
under KVM on Linux) and stops if the host lacks those cores. It records the CPU
governor in `host-info.txt`. Under KVM with a governor other than `performance` it
does not refuse: it warns on stderr, writes `governor_warning=yes` in every row of
`index.csv`, and `paths.py` carries the caveat in the note of every row of the table
and at the head of `tables.md`.

The exit status of `paths.sh` is nonzero when no run counted or any cell of the
table has no counted run (`paths.py` reports `counted X of Y runs`); the tables are
written all the same.

Expect nine boots per repetition of Rabbit plus three of mainline, sixty in all
with five repetitions. The time per boot has not been measured under KVM; plan an
hour as an estimate and let the first run replace it. Nothing is powered off.

| file in `--out` | what it is |
|---|---|
| `meta.txt`, `host-info.txt` | mode, guest CPU, host CPU, QEMU, the sha256 of the kernels and of the runtime |
| `commands.txt` | the exact QEMU command of every run |
| `logs/*.serial.log`, `logs/*.qemu.log` | what the guest and the emulator said, one pair per run |
| `index.csv` | what `paths.sh` ran: format, requested path, kernel, repetition, QEMU status, outcome, `governor_warning`. The outcome is `finished` (`INFER OK` seen; QEMU was stopped by the script, status `stopped`, or the guest left by itself and the debug-exit status stays), `failed` (`INFER ERR` or `INFER error`), `exited` (QEMU ended with no marker, a real exit whose status says why) or `timeout` |
| `runs.csv` | per run: reported path (`not reported` when the guest printed no `INFER path`), `INFER path_limit`, weights, ttft, tok/s, tokens, golden check, `mismatch` (the reported path or `path_limit` differs from the request), why a run does not count |
| `table.csv`, `ratios.csv`, `tables.md` | median, min..max and ttft per cell; path against scalar per format (above 1 is faster); Rabbit against mainline on avx2 |

Under `--accel tcg` the same files carry `-emulator-not-a-speed` in their names and
the header of `tables-emulator-not-a-speed.md` says it. A TCG rehearsal proves
the machinery and nothing about speed: never quote it as a rate.

What each table can and cannot support:

* The path against scalar says what the vector kernels of the runtime are worth for
  one format on this host. It is a statement about the runtime, not about the
  operating system, and it does not count for the stop rule of the fair comparison
  of Rabbit with mainline.
* Rabbit against mainline on the same path and format is the comparison of the two
  operating system kernels with the runtime held fixed. It is valid only when both
  report the same path; otherwise the row says `not comparable`.
* If the decode is bound by the bandwidth of memory the avx2 paths may tie with
  sse2, and a q4 copy may gain more from its size than from its kernel. A tie is
  a result, and the `reading` column calls it so when the ranges of the runs
  overlap (with fewer than three runs it gives no reading).
* Everything is one host, one guest CPU model and one virtual processor.

What the first KVM run must also do, with the real checkpoint and not the small
CI model: compare the greedy tokens of every path and format against
`bench/reference/reference.json` (the `tokens` column of `runs.csv` against
`new_tokens`), because the speed of a path that decodes other tokens is not a speed
of the same work. For `bf16` the runtime also prints `INFER golden`, and the
`golden` column carries it; a run whose golden check failed does not count and is
listed under "Runs that do not count" with the reason `golden check failed`. The checkpoint has to be the one that `reference.json`
names (`bench/fetch-model.sh` fetches the Instruct variant by default, so check the
revision), and a quantized format that differs from the reference has to be
explained by the quantization, not hidden.

```sh
python3 - results/paths-kvm/runs.csv <<'PY'
import csv, json, sys
want = " ".join(map(str, json.load(open("bench/reference/reference.json"))["new_tokens"]))
for r in csv.DictReader(open(sys.argv[1])):
    print(r["format"], r["requested"], r["reported"], r["kernel"], r["rep"], r["tokens"] == want)
PY
```

The tests of the deriver run with `python3 -m unittest discover -s bench/kvm`.
