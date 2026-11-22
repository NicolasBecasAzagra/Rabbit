# The KVM session

Issue 10: the rig run with `--accel kvm` on a host with `/dev/kvm`, and a table
with two columns, one under emulation (a relative comparison) and one under
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
python3 bench/kvm/build_table.py --out bench/results/kvm-<date> \
    --pair operations=bench/results/vps-1cpu,bench/results/kvm-<date>/operations \
    --pair inference=bench/results/inference-native,bench/results/kvm-<date>/inference \
    --pair agent=bench/results/agent,bench/results/kvm-<date>/agent
```

Destroy the server in the panel of the provider when the tar is on your machine,
and check that it no longer runs.

## Rehearsal without paying

`ACCEL=tcg bench/kvm/run-session.sh` runs the whole script on any Linux host with
QEMU, skips the KVM preflight and writes datasets that say `tcg`. It is the way
to check the script before the paid host exists. The tests of the table run with
`python3 -m unittest discover -s bench/kvm`.
