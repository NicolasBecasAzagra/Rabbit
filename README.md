# Rabbit

![Rabbit, an operating system for local AI agents](docs/banner.svg)

[![CI](https://github.com/NicolasBecasAzagra/Rabbit/actions/workflows/ci.yml/badge.svg)](https://github.com/NicolasBecasAzagra/Rabbit/actions/workflows/ci.yml)

**An operating system for local AI agents, written in Rust from the kernel up.**

Rabbit is a from-scratch x86_64 kernel that speaks the Linux userspace ABI:
the same system calls, the same `struct` layouts, the same ELF loading rules.
On top of that kernel sit two programs that are not toys: an inference runtime
that serves a real language model with no dependency outside the Rust standard
library, and an agent that asks the model what to do and runs the tools.

Rabbit is an experiment in measured method: the same static binary and the
same initramfs run on Rabbit and on mainline Linux, the model reproduces the
reference implementation token for token, and every number is published with
its raw data, the losses included. It asks what a kernel built for one job
costs, and it answers on an emulator first. Run on hardware acceleration, one
bare metal host, those answers changed: see [the same rig under KVM](#the-same-rig-under-kvm).
Its newest result before that is a policy that the kernel enforces for the tools
of the agent, measured against Landlock and seccomp on Linux: see
[the tool policy](#the-tool-policy).

![The boot log of the Rabbit kernel, from the first line to "boot OK"](docs/boot.gif)

The serial output of one real boot under QEMU, as recorded in
[docs/BOOT.md](docs/BOOT.md).

![The three layers of Rabbit](docs/architecture.svg)

## What we are building

**A machine for local agents, built in the open and measured against Linux.**

An agent of today expects three things from the machine it runs on: it calls
APIs over HTTP, it reads and writes files, and it runs commands. It expects a
model that fits, and it expects the answers to come back quickly. Rabbit does
part of that today and names the rest in the roadmap below.

It is built in three layers, each one measurable against mainline Linux on the
same machine, with the same binary and the same checkpoint.

1. **A kernel that does less.** No page table isolation, no LSM, no audit, no
   cgroups, and no hook framework: the one check on the system call path is the
   tool policy of the agent. Demand paging, large pages, a scheduler with a
   short tail, and real multiprocessing. Every number is published, including
   the losses.
2. **A runtime that serves a model.** The weights are mapped from the
   initramfs without being copied, kept resident, and read by a decoder
   written from scratch: RMSNorm, rotary embeddings, grouped query attention,
   the gated feed forward block and a key value cache. It generates the same
   tokens as the reference implementation, one for one.
3. **An agent that acts.** The model answers with a tool call, the kernel
   runs the tool, and the model answers again. The tools read the files of
   the machine, reach the network and call APIs over HTTP; writing files and
   listing a directory wait for a writable file system. The shape of the call is not a
   request in the prompt but an automaton over bytes: the answer cannot be
   anything but a well formed call of a declared tool, with the keys that tool
   needs, each of them once, with the type the tool expects.

## What it does today

| | Rabbit | mainline | mainline, mitigations off |
|---|---|---|---|
| system call | **376 ns** | 29.3 us | 1.04 us |
| map and touch memory | **6.83 ms/MiB** | 25.8 ms/MiB | 8.69 ms/MiB |
| fault in memory, 4 KiB pages | **2.01 ms/MiB** | 44.6 ms/MiB | 13.7 ms/MiB |
| fault in memory, 2 MiB pages | **825 us/MiB** | 44.9 ms/MiB | 13.0 ms/MiB |
| futex round trip | **6.53 us** | 407 us | 97.7 us |
| four threads, spawn to join | **620 us** | 7.73 ms | 4.59 ms |
| boot to the first result | 12.9 s | 206 s | 63.8 s |
| resident memory of the guest | 319 MB | 476 MB | 483 MB |
| time to the first token | **11.25 s** | 24.67 s | 22.82 s |
| tokens per second | **0.45** | 0.20 | 0.21 |
| seconds of an agent loop | **821** | 2793 | 2570 |
| resident memory with the model | 857 MB | 1021 MB | 1023 MB |

Every row ran under QEMU TCG, an emulator, with no hardware acceleration. The
boot row is not a comparison of the kernels, as
[docs/BENCHMARKING.md](docs/BENCHMARKING.md) states, and the rows of resident
memory depend on how each kernel holds the initramfs, so read them as
descriptions and not as wins.

One processor, 512 MiB for the operations and 1024 MiB for the model and the
agent, QEMU 8.2.2, Linux 7.3.0 built from the mainline tree by
`bench/prepare-mainline.sh`, the same initramfs and the same binary on both
sides. Every number of the table comes from a committed dataset:

| rows | dataset |
|---|---|
| the operations | `bench/results/vps-1cpu/` |
| the model | `bench/results/inference-native/` |
| the agent | `bench/results/agent/` |
| the network | `bench/results/net/` |
| the policy | `bench/results/policy/` |
| the same rig under KVM, and the stages | `bench/results/kvm/` |

![Selected Rabbit benchmark measurements](docs/stats.svg)

The network is the fourth layer: the guest enumerates the bus, claims a
network device, resolves the address of the gateway, sends a datagram to a
server on the host and reads the answer, with the socket calls of the Linux
ABI. The agent has a tool that does it, so the loop of the agent reaches
the network.

### The same rig under KVM

The rig was run again with `--accel kvm` on a bare metal host: an AMD Ryzen 5 PRO
3600, QEMU 8.2.2, one virtual processor, the same initramfs and the same binary,
five runs per dataset and the median of the medians of the runs. It is one host,
one processor model and one virtual processor. The columns of the emulator are
the datasets above, one run each.

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

A ratio above 1 means Rabbit is better, whichever the unit, and a tie is a
difference smaller than twice the spread between the runs. The reference is Linux
7.3.0 with `mitigations=off`. Every figure of the first column ran under QEMU TCG,
an emulator. Every figure of the second ran under KVM on that one host.

What the two columns say:

- **The ratios of the emulator do not hold on this host.** The advantages that
  Rabbit showed under emulation, from 1.3x to 123x, are from 1.2x to 2.3x under
  KVM, with 7.4x for the slowest fault of 2 MiB. Three rows that Rabbit won are
  losses: the cost of a fault in memory with 4 KiB pages (0.73x) and with 2 MiB
  pages (0.76x), and the slowest fault of 4 KiB (0.52x). The tokens per second
  and the agent loop are a tie or a loss of 2%, and the first token is 1.04x.
- **Only the KVM column can support a statement about wall clock time, and only
  on that host.** It does not say how Rabbit compares on another processor, with
  more virtual processors or on hardware that is not virtualised.
- **The first KVM runs found two defects of Rabbit, not of the method.** The
  kernel did not save the vector registers when it switched threads, and a test
  of two threads found 77 of 80 rounds corrupted. And the runtime chooses its
  code path by asking the processor, Rabbit did not enable AVX, so the same binary
  ran different code on the two kernels, and Rabbit looked 0.61x on tokens per
  second until it did. The first stage of the series is not a comparison of kernels
  for the model. Both are fixed, the stages are in
  [`bench/results/kvm/`](bench/results/kvm/), and so is every loss.
- **Every loss is published.** The reference kernel without mitigations is
  measured too, and the batches where a boot did not finish are in
  `bench/results/` with the samples that are missing.

## The tool policy

The agent declares its tools, and that declaration does two things. It
constrains what the model may write, through the grammar, and it generates a
manifest that the kernel enforces: which files the process may read and which
addresses it may reach. The same binary applies the same manifest on Linux with
Landlock and seccomp, so the two sides are compared with two adversaries: one
that makes only the calls the grammar accepts, and a compromised binary that
issues raw system calls under the tools.

On a Linux 7.0 kernel with Landlock ABI 8 and a seccomp filter, the agent could
still:

- send TCP and UDP to **another IP address** on the allowed port, with
  `connect`, `sendto`, `sendmsg` and `sendmmsg`, and the data reached the server
  of the test;
- send UDP to **another port**, because the UDP port rules of Landlock need
  ABI 10;
- tell a file outside its read root from a path that does not exist, with `stat`.

Rabbit refuses the `connect`, the `sendto` and the `newfstatat` of those cases
with the policy of its kernel, and no path of the 1554 of the enumerated family
outside the read root is allowed on either side. Read it with its limits:

- **22 of the 43 raw calls of the compromised binary are system calls that
  Rabbit does not implement.** They count as missing, not as protection: only
  11 are refusals by the policy.
- The UDP port rows depend on the kernel version. The rows about the destination
  address do not, because Landlock names ports and no addresses.
- It is not the first work on policies for the tools of agents, and it runs on
  an emulator against a runner, with one policy per boot and no defence against
  prompt injection.

The method, the table and the list of what it does not show are in
[the section of the report](REPORT.md#the-tool-policy-enforced-by-the-kernel),
the raw lines and the script that derives the table are in
[`bench/results/policy/`](bench/results/policy/), and the gaps that are left are
the milestone v0.8.

## Status

Everything in the table is implemented, verified on the host and asserted in
the continuous integration. One commit per milestone. What is not in the table
is in [the open milestones](https://github.com/NicolasBecasAzagra/Rabbit/milestones),
and the agent does not close a task yet: the model of 135 million parameters
repeats a call it has already made ([issue 14](https://github.com/NicolasBecasAzagra/Rabbit/issues/14)).

| Milestone | What it delivers | State |
|---|---|---|
| M0 | Boot through the Xen PVH handover, serial output, physical memory map | done |
| M1 | GDT, TSS with an IST stack, IDT, recoverable exceptions, 8259 PIC, PIT | done |
| M2 | Buddy allocator, size class heap, higher half kernel, direct map | done |
| M2b | The kernel reaches physical memory only through the direct map | done |
| M2c | Frame metadata carved out of the memory it describes, 32 GiB of guests | done |
| M3 | Kernel threads, preemptive round robin scheduler, interrupt safe locks | done |
| M4 | Ring 3, `SYSCALL`/`SYSRET`, ELF64 loader, 59 Linux system calls | done |
| M4b | A real userland: a standard library program running unmodified | done |
| M4c | Demand paging for anonymous mappings | done |
| M8 | Benchmark rig, seventeen measurements, the comparison published with its raw data | done |
| M9 | SMP: ACPI MADT, INIT-SIPI-SIPI, LAPIC and IOAPIC, IPIs, four processors | done |
| M10 | The network: the bus PCI, a virtio device, IPv4 and UDP, the sockets of the ABI | done |
| M11 | Quantization: a quantized checkpoint decoded by the same runtime, token for token | done |
| M12 | TCP in the kernel, the sockets of a stream, so the agent can speak HTTP | done |
| E1 | The initramfs as a read only file system | done |
| E2 | A file backed mapping and large pages | done |
| E2b | Large pages for file mappings and `mlock` residency of the weights | done |
| E3a | A scheduler with priorities and wakeup preemption | done |
| E3b | The tail of the scheduler measured, and the policy that shortens it | done |
| E4 | The inference runtime: a real model, token for token against the reference | done |
| E5 | The measurement of the runtime on both kernels, same machine, same binary | done |
| E6 | The agent: model, tool call, tool result and model, with a grammar that forces the call | done |
| M14 | The tool policy: the kernel enforces the manifest of the tools, measured against Landlock and seccomp | done |
| M15 | The rig under KVM on a bare metal host, and what it found: the vector state of a thread, a copy in `mlock`, the cost of a fault | done |

## What comes next

In the order it has to happen, one row per milestone of the tracker. The
reasoning and the exit criteria of each one are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and the issues carry the same
criteria.

| Milestone | Issues | Why it is next |
|---|---|---|
| **v0.8 Policy hardening** | [#21](https://github.com/NicolasBecasAzagra/Rabbit/issues/21), [#22](https://github.com/NicolasBecasAzagra/Rabbit/issues/22), [#23](https://github.com/NicolasBecasAzagra/Rabbit/issues/23), [#24](https://github.com/NicolasBecasAzagra/Rabbit/issues/24), [#25](https://github.com/NicolasBecasAzagra/Rabbit/issues/25) | The gaps the policy report names: `bind`, the copies of user memory, the descriptors open before the install, the log of denials, and the same tables on a kernel with Landlock ABI 10. |
| **v0.5 An agent that finishes** | [#14](https://github.com/NicolasBecasAzagra/Rabbit/issues/14), [#15](https://github.com/NicolasBecasAzagra/Rabbit/issues/15) | A checkpoint that closes the conversation, and the tools that need a writable file system: `write_file` and `list_dir`. |
| **v0.6 Reach the world** | [#18](https://github.com/NicolasBecasAzagra/Rabbit/issues/18), [#19](https://github.com/NicolasBecasAzagra/Rabbit/issues/19) | The model context protocol, so the agent works with the clients people already have, and TLS, because almost every API people call is HTTPS. |
| **v0.9 Real hardware** | [#26](https://github.com/NicolasBecasAzagra/Rabbit/issues/26), [#27](https://github.com/NicolasBecasAzagra/Rabbit/issues/27), [#28](https://github.com/NicolasBecasAzagra/Rabbit/issues/28) | Every number so far comes from an emulator. A UEFI entry, a disk driver and a first boot on a named machine are what turn the kernel from an experiment into an operating system. |
| **v0.10 Fast on the CPU** | [#29](https://github.com/NicolasBecasAzagra/Rabbit/issues/29), [#30](https://github.com/NicolasBecasAzagra/Rabbit/issues/30), [#31](https://github.com/NicolasBecasAzagra/Rabbit/issues/31) | Vector kernels, the GGUF format and models of one billion parameters or more, which are the smallest that may finish the tasks of the agent. |
| **v0.11 The fair comparison** | [#32](https://github.com/NicolasBecasAzagra/Rabbit/issues/32), [#33](https://github.com/NicolasBecasAzagra/Rabbit/issues/33) | The same weights on the same machine against llama.cpp over Linux, losses included, and a written stop rule if the operating system turns out not to be where the speed is. |
| **v0.12 One accelerator** | [#34](https://github.com/NicolasBecasAzagra/Rabbit/issues/34), [#35](https://github.com/NicolasBecasAzagra/Rabbit/issues/35) | Most of the speed of local inference is in the accelerator. One is chosen with a written go or no-go before any driver code. |
| **v1.0 The standard** | [#36](https://github.com/NicolasBecasAzagra/Rabbit/issues/36), [#37](https://github.com/NicolasBecasAzagra/Rabbit/issues/37), [#38](https://github.com/NicolasBecasAzagra/Rabbit/issues/38) | A bootable image, the API the clients already speak and a hardware matrix that lists only tested machines. |
| **Deferred** | [#16](https://github.com/NicolasBecasAzagra/Rabbit/issues/16) | Processes, signals, dynamic linking and a shell test suite: named so they are not lost. |

The aim behind the order is to be the fastest system for local AI. That is a
goal, not a result: the only run on hardware acceleration is one host, and on it
Rabbit is ahead on some operations, level on the model and behind on the cost of
a fault in memory. [docs/ROADMAP.md](docs/ROADMAP.md) states the
ladder, the proof that closes each rung and the rule that stops the project if
an operating system turns out not to be where the speed is.

A scheduler with a better tail, for batching inference requests, is the work
after these.

## Build and run

Requirements:

- Rust nightly (pinned in `rust-toolchain.toml`), with the `rust-src` and
  `llvm-tools` components.
- QEMU 8.2 or newer (`qemu-system-x86_64`).
- `xorriso` and `mtools` only if you build disk images.

```sh
rustup toolchain install nightly-2026-10-24 \
    --component rust-src --component llvm-tools --component rustfmt --component clippy
make build          # the kernel
make run            # boot it in QEMU, serial on stdout
make test           # host unit tests of every crate
make fmt-check      # the format of every crate
make clippy         # the clippy of every crate, for its target
make clippy-host    # the clippy of every crate, for the host
```

Boot directly with QEMU:

```sh
qemu-system-x86_64 -M q35 -m 512M \
    -kernel "$(make -s print-kernel-path)" \
    -display none -monitor none -no-reboot -serial stdio \
    -device isa-debug-exit,iobase=0xf4,iosize=0x04
```

Debugging: add `-s -S` to QEMU (gdbstub on port 1234, CPU halted), then attach
`lldb "$(make -s print-kernel-path)"` with `gdb-remote 1234`
(macOS) or `gdb` with `target remote :1234` (Linux).

### The comparison against mainline

```sh
bench/prepare-mainline.sh ~/rabbit/artifacts   # the reference image, on a Linux host
bench/fetch-model.sh ~/rabbit/artifacts/model-instruct   # the checkpoint, verified
python3 bench/run.py \
    --rabbit "$(make -s print-kernel-path)" \
    --mainline ~/rabbit/artifacts/vmlinux \
    --mainline-variant "nomitig=mitigations=off" \
    --out bench/results
```

### The agent

```sh
make agent agent-initramfs
EXTRA_DIR=~/rabbit/artifacts/model-instruct \
    bench/mkinitramfs.sh ~/rabbit/artifacts/agent.cpio rabbit-agent/rabbit-agent.elf
python3 bench/run.py --rabbit "$(make -s print-kernel-path)" \
    --mainline ~/rabbit/artifacts/vmlinux \
    --initrd ~/rabbit/artifacts/agent.cpio --memory 1024M \
    --only agent-steps,agent-tokens,agent-seconds,agent-step-ms
```

The full Linux VM setup (for building mainline Linux and a distro rootfs) is in
[docs/VM-SETUP.md](docs/VM-SETUP.md).

## Repository layout

```text
rabbit-arch/       x86_64 hardware access, port I/O, boot trampoline
rabbit-bootinfo/   pure parsing of the PVH boot handover structures
rabbit-drivers/    16550A UART, isa-debug-exit
rabbit-mm/         paging, the buddy allocator, the heap, the mappings
rabbit-sched/      the scheduler and its policy
rabbit-elf/        the ELF64 reader
rabbit-initramfs/  the reader of the cpio archive of the initramfs
rabbit-kernel/     the no_std kernel binary
rabbit-userland/   the test program that runs unmodified in ring 3
rabbit-net/        Ethernet, ARP, IPv4, UDP and TCP, pure and tested on the host
rabbit-http/       the HTTP/1.1 messages of the client of the agent
rabbit-policy/     the capability policy of a program, shared by the agent and the kernel
rabbit-bench/      the workload that measures the kernel
rabbit-netclient/  the datagram round trip client, the same binary on both kernels
rabbit-netstream/  the stream round trip client, the same binary on both kernels
rabbit-infer/      the inference runtime, the grammar, the merge encoder
rabbit-agent/      the agent loop: model, tool call, tool result
bench/             the rig, the workloads, the reference build, the datasets
docs/              architecture, boot protocol, benchmarking, VM setup
scripts/           the generator of the interrupt stubs
```

Dependencies policy: kernel crates may only use MIT, BSD or ISC licensed code.
`GPL-2.0-only` is incompatible with `Apache-2.0-only`, so Apache-only crates
are forbidden in the kernel.

## What this project is not

Being "better than Linux at everything" is not a goal, and it is not a
measurable one. Rabbit publishes a narrow, defensible claim instead:

- Same userspace, same QEMU version, same machine model, same rootfs, same
  benchmark scripts, raw results committed to the repository.
- Two areas where a purpose-built kernel can plausibly win in a private VM:
  syscall entry cost (no KPTI/retpoline in a trusted single-tenant guest) and
  fork/exec plus tail latency (no LSM, no audit, no cgroups, no hook framework;
  the tool policy is the one check on the system call path).
- Every loss is published too. No cherry-picking, no "mitigations off!" as the
  only comparison point.

The methodology is specified in [docs/BENCHMARKING.md](docs/BENCHMARKING.md).

## Boot protocol

Rabbit boots through the Xen PVH handover (`XEN_ELFNOTE_PHYS32_ENTRY`), which
QEMU supports natively for 64-bit ELF kernels. Multiboot2 is not an option:
QEMU's Multiboot loader rejects 64-bit images. The same path also boots
mainline Linux (`vmlinux` with a PVH note), which is what makes the benchmark
comparison fair. Details in [docs/BOOT.md](docs/BOOT.md).

## How this was built

Rabbit was written with the assistance of an AI agent, and this is stated
here on purpose rather than left for someone to discover.

- The design decisions, the milestone criteria and the verification belong to
  a human. Every milestone has an objective exit criterion that a machine
  checks: a serial log assertion in CI, a unit test on the host, a row in the
  benchmark data. Not "it compiles".
- The code was written by an AI agent against those criteria, and every
  change was checked against the build, the tests and the boot log before it
  was committed. Several of the defects this repository fixed were found by
  that checking, not by the code looking wrong.
- The repository is the evidence: one milestone per commit, host tests for
  everything that can be tested on the host, and a benchmark rig that
  publishes its raw data with the failures included.
- What an agent cannot do is explain why a line is there. The reading path in
  [docs/LEARNING.md](docs/LEARNING.md), with its experiments, is the human
  part of the work.

The method is the reusable part. "How much of an operating system can be
written this way, and how do you know it works" is a question worth an
answer, and this repository is one answer to it.

## Documentation

- **[REPORT.md](REPORT.md)** - the measurements, the findings, the losses and the threats to validity.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - crates, design decisions, milestone ladder, risks, stop rules.
- [docs/BOOT.md](docs/BOOT.md) - PVH boot protocol, trampoline, memory map, debugging.
- [docs/BENCHMARKING.md](docs/BENCHMARKING.md) - the method, and the raw numbers with their caveats.
- [bench/results/kvm/](bench/results/kvm/README.md) - the rig under KVM on a bare metal host: the table, the five stages, and the raw data.
- [docs/LEARNING.md](docs/LEARNING.md) - how to read the kernel and the userland, experiments to run, sources to read next.
- [docs/VM-SETUP.md](docs/VM-SETUP.md) - the Linux build host: mainline Linux, rootfs, and the artifact handoff.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Code, comments and commits are in
English. Every file carries an SPDX identifier, every `unsafe` block carries a
`SAFETY:` justification, and CI must be green.

## License

`GPL-2.0-only`. See [LICENSE](LICENSE).
