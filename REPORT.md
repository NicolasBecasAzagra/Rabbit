# Rabbit: a Linux-ABI kernel in Rust, measured against mainline

**Repository**: <https://github.com/NicolasBecasAzagra/Rabbit>
**Status**: milestones M0 to M4c, M8, M9, E1, E2, E2b, E3a, E3b, E4, E5 and E6 are implemented and
asserted in CI, and the rig has been run under KVM on a bare metal host (M15).
The measurement below was produced by the rig in this repository, from the raw
rows in `bench/results/`.

## Abstract

Rabbit is an x86_64 operating system kernel written entirely in Rust that
speaks the Linux userspace ABI. It boots a real language model, a 135 million
parameter Llama-architecture model, and generates the same tokens as the
reference implementation of that model, token for token, on Rabbit and on
mainline Linux, from the same binary and the same initramfs.

The point of the project is not to be "a faster Linux". It is to measure what
a kernel costs an inference server, with a method that can be checked, and to
publish the losses as carefully as the wins. Four findings came out of it:

1. **Under an emulator, with the mitigations disabled on both sides, Rabbit is
   about two to ten times cheaper per operation** than mainline: 2.2 times per
   system call, 4.4 times per megabyte faulted in with 4 KiB pages, 9.7 times
   with 2 MiB pages, 7.5 times per futex round trip. The difference is not the
   arithmetic of the program, which is the same binary on both sides, but the
   work the kernel does per operation. Finding 4 says how much of it survives
   on hardware acceleration.
2. **Large pages are an improvement on Rabbit and not on the reference
   kernel**: mapping the weights with 2 MiB pages halves the cost per MiB and
   shortens the worst fault by more than an order of magnitude on Rabbit,
   while the reference kernel gets slightly slower. The difference is not that
   one kernel has large pages and the other does not, it is what a large page
   costs when it is faulted in.
3. **The measurement that mattered most was the one we lost.** The first
   inference measurement showed Rabbit twice as slow as the reference kernel,
   and the cause was specific: Rabbit mapped a file with 4 KiB pages, so
   streaming 269 MB of weights took 65536 faults instead of 134. Fixing that
   took the faults to 791. The loss and its cause are published in
   `docs/BENCHMARKING.md` alongside the win.
4. **Hardware acceleration contradicted part of the emulator ratios, and found
   two defects of Rabbit.** The same rig run under KVM on one bare metal host
   gives advantages of 1.2 to 2.3 times where the emulator gave 1.3 to 123, and
   three rows that Rabbit won under emulation are losses: the cost of a fault in
   memory with 4 KiB pages (0.73 times) and with 2 MiB pages (0.76 times), and
   the slowest fault of 4 KiB (0.52 times). The tokens per second and the agent
   loop are level with the reference kernel. The first runs also showed that
   Rabbit did not save the vector registers when it switched threads, and that
   the same binary ran different code on the two kernels. See
   [the same rig under KVM](#results-the-same-rig-under-kvm).

## What is in the box

- **A kernel** (`rabbit-kernel`): boots through the Xen PVH handover, runs
  from the higher half with a direct map of physical memory, manages physical
  memory with a buddy allocator whose metadata is carved out of the memory it
  describes, faults user pages in on demand, backs anonymous and file
  mappings with 2 MiB pages, serves 59 Linux system calls, exposes the
  initramfs as a read only file system, and runs preemptive kernel threads.
- **A runtime** (`rabbit-infer`): reads a model in the `safetensors` format,
  maps it through the initramfs without copying it, and generates tokens with
  a Llama-style forward pass: RMSNorm, rotary embeddings, grouped query
  attention, the gated feed forward block, a key value cache and greedy
  sampling. It also holds a byte pair merge encoder, written from scratch, so
  a prompt built at run time reaches the model as the token ids it was
  trained with, and a byte level grammar that forces the answer to be a well
  formed tool call. No dependency outside the Rust standard library.
- **An agent** (`rabbit-agent`): a loop of model, tool call, tool result and
  model, over seven tools (`read_file`, `list_dir`, `write_file`, `compute`,
  `net_send`, `http_get` and `done`). The model chooses the tool and the
  arguments, the kernel decides how much of that costs. `list_dir` and
  `write_file` are declared but not served by Rabbit yet: the kernel answers
  `getdents64` with `ENOSYS` and the initramfs is read only (issue 15).
- **A rig** (`bench/`): boots a kernel image with a fixed configuration,
  hands it the same initramfs, and records every sample, including the ones
  that failed. It rebuilds the kernel, the runtime and the initramfs on every
  run, and it refuses to report a measurement it could not take.

## Method

The rules were written before the first number was measured, in
`docs/BENCHMARKING.md`, and they were not adjusted afterwards.

- One workload binary, one initramfs, one QEMU configuration. Only the
  kernel image differs.
- The reference kernel is measured in two configurations: its default, and
  with `mitigations=off`. A default mainline kernel reloads CR3 on every
  system call to isolate the page tables, which under emulation invalidates
  the translator cache and is not a kernel property worth measuring on its
  own.
- At least 30 iterations per workload, every iteration recorded, failures
  recorded as failures.
- The median, the 99th percentile and the standard deviation for every
  configuration, plus the raw rows, the SHA-256 of both kernel images and the
  exact command line.
- The reference image is Linux 7.3.0 built with `CONFIG_PVH=y`, so that both
  kernels are started by the same handover with the same command line.

**Environment**: TCG, one processor, 512 MiB of guest memory for the operation
benchmarks and 1024 MiB for the inference ones. The primary host runs QEMU
8.2.2 on an x86_64 server and the cross check runs QEMU 11.1.2 on an arm64
macOS laptop (both are described below). Neither host has hardware
acceleration, which is the most important limitation of everything below
except the section on KVM, which is one run on one bare metal host.

## Results: the cost of an operation

Two datasets, measured with the same rig and the same rules.

The **primary** one is a native x86_64 host: an Ubuntu server with six
cores, QEMU 8.2.2, and the reference kernel built from the mainline tree
by `bench/prepare-mainline.sh`. The **cross check** is the same rig on a
different host, an arm64 laptop with QEMU 11.1.2, where both kernels run
under emulation.

Medians, in the unit of the workload, on the native host:

| Workload | Rabbit | mainline | mainline, mitigations off | Rabbit against mainline with mitigations off |
|---|---|---|---|---|
| system call | 376.5 ns | 29.35 us | 1.04 us | **2.8x cheaper** |
| map and touch memory | 6.83 ms/MiB | 25.80 ms/MiB | 8.69 ms/MiB | **1.3x cheaper** |
| fault in memory, 4 KiB pages | 2.01 ms/MiB | 44.55 ms/MiB | 13.73 ms/MiB | **6.8x cheaper** |
| slowest single fault, 4 KiB | 1.74 ms | 42.60 ms | 38.89 ms | **22x cheaper** |
| fault in memory, 2 MiB pages | 824.8 us/MiB | 44.89 ms/MiB | 12.98 ms/MiB | **15.7x cheaper** |
| slowest single fault, 2 MiB | 189.1 us | 68.32 ms | 23.33 ms | **123x cheaper** |
| futex round trip | 6.53 us | 407.4 us | 97.71 us | **15x cheaper** |
| four threads, spawn to join | 619.5 us | 7.73 ms | 4.59 ms | **7.4x faster** |
| boot to the first result | 12.87 s | 206.29 s | 63.80 s | **5x faster** |
| resident memory of the guest | 319 MB | 476 MB | 483 MB | **1.5x less** |

The same rig on the cross check host:

| Workload | Rabbit | mainline | mainline, mitigations off | Rabbit against mainline with mitigations off |
|---|---|---|---|---|
| system call | 239.5 ns | 9827.5 ns | 516.0 ns | **2.2x cheaper** |
| map and touch memory | 2.72 ms/MiB | 8.34 ms/MiB | 3.23 ms/MiB | **1.2x cheaper** |
| fault in memory, 4 KiB pages | 950.1 us/MiB | 13.33 ms/MiB | 4.16 ms/MiB | **4.4x cheaper** |
| slowest single fault, 4 KiB | 263.5 us | 4.65 ms | 2.55 ms | **9.7x cheaper** |
| fault in memory, 2 MiB pages | 445.5 us/MiB | 14.01 ms/MiB | 4.33 ms/MiB | **9.7x cheaper** |
| slowest single fault, 2 MiB | 28.2 us | 4.80 ms | 1.67 ms | **59x cheaper** |
| futex round trip | 3.70 us | 136.7 us | 27.7 us | **7.5x cheaper** |
| four threads, spawn to join | 228.0 us | 1536.0 us | 970.0 us | **4.3x faster** |

The two hosts agree on the order of the effect and disagree on its size,
which is what a measurement under emulation does. The fault rows move the
most: a fault costs the emulator, and the emulator pays more for the work
mainline does per fault. The full tables with the 99th percentile and the
standard deviation are in `bench/results/summary.csv` and in
`bench/results/vps-1cpu/`.

Against the default reference configuration, which pays for page table
isolation on every system call, the system call row is 78x on the native
host. That number is published but it is not a kernel comparison, and the
report says so.

Where the difference comes from, in the order the size of the effect:

- **No page table isolation, no LSM or auditing hooks, no virtual memory
  accounting.** A system call that does not have to walk an audit path or
  reload CR3 is cheaper. The one check on the system call path is the tool
  policy, which the operation benchmarks run without.
- **Demand paging.** A mapping reserves address space and the frames arrive
  when the program touches them, so a program that reads one byte of a 256 MiB
  mapping does not pay for the rest of it. This single change made the
  thread workload seven times faster and is the reason the thread row is a
  win at all: before it, Rabbit was slower there.
- **A simpler futex path.** Eight waiters in a flat table, no robust list,
  no requeue.
- **Large pages for anonymous and for file mappings**, with the cost of the
  copy documented.

## Results: large pages

On the native host:

| | 4 KiB pages | 2 MiB pages |
|---|---|---|
| Rabbit, cost per MiB | 2.01 ms | 824.8 us |
| Rabbit, slowest fault | 1.74 ms | 189.1 us |
| mainline with mitigations off, cost per MiB | 13.73 ms | 12.98 ms |
| mainline with mitigations off, slowest fault | 38.89 ms | 23.33 ms |

And on the cross check host:

| | 4 KiB pages | 2 MiB pages |
|---|---|---|
| Rabbit, cost per MiB | 950.1 us | 445.5 us |
| Rabbit, slowest fault | 263.5 us | 28.2 us |
| mainline with mitigations off, cost per MiB | 4.16 ms | 4.33 ms |
| mainline with mitigations off, slowest fault | 2.55 ms | 1.67 ms |

The raw rows of the first measurement of the large pages are in
`bench/results/e2/`.

The reference kernel gets slightly slower per MiB when it uses large pages,
and its worst fault does not improve. On Rabbit the same change halves the
cost per MiB and shortens the worst fault by a factor of nine. Both kernels
have large pages; they do not cost the same to fault in.

The reason is what happens when a large page arrives: the folio has to be
allocated, zeroed and accounted for, and the compaction machinery around it
is not free. Rabbit allocates a 2 MiB block from its own buddy allocator,
zeroes it and maps it.

For a file mapping, the data of the file is not aligned to 2 MiB in physical
memory, so the kernel copies it into a large page once, the way the read only
huge pages of a file system do. That doubles the memory a mapped file
occupies, and the cost is in the report.

## Results: the arithmetic, measured against the reference

The kernel part of the story is measured by the rig, which boots two kernels.
The arithmetic part is measured separately, natively, because that is the
comparison a user makes: not our kernel against another kernel, but our
program against the programs people run. Both programs run natively on the
same host, with the same weights, and they produce **the same thirty two
tokens**, so what is compared is the arithmetic.

| Natively, one host, one checkpoint | tokens per second | time to the first token |
|---|---|---|
| Rabbit runtime, Rust, no dispatcher and no autograd | **23.6** | **286 ms** |
| Reference implementation, eager, float32 | 10.9 | (2.93 s for 32 tokens) |

A specialised forward pass written in Rust, with blocked matrix multiplication
and the vector instructions of the host, is 2.2 times faster than the eager
path of the reference implementation on this model, and it pins the weights in
about seventy milliseconds, which is inside the time to the first token.

The claim is narrow on purpose. The reference runs in eager mode, so its
dispatcher and its autograd are part of what it costs; a compiled or quantized
reference would be a different comparison. The host varies by around thirty
per cent between runs, so this is a sample and not a precise measurement. The
raw values are in `bench/results/native/`.

## Results: serving a model

The runtime serves SmolLM2-135M, revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, Apache-2.0, 269 MB of bfloat16
weights. It generates `260 3575 282 260 1798 30 198 198` and the text
"The capital of France is the capital of the country.", which is exactly what
the reference implementation produces for the same prompt, in float32, on an
ordinary Linux host. The reference run is in `bench/reference/`, and the
runtime is checked against it.

**The measurement that does not depend on the host is the number of faults.**
Streaming the weights used to take 65536 faults of 4 KiB. With large pages
for file mappings it takes 791, of which 127 are large pages: a factor of
eighty three, and the same on every run. The runtime now also pins the
weights, and the kernel confirms it: 65688 pages of 269058048 bytes are
resident, pinned in about 96 milliseconds, and that cost is inside the time to
the first token.

**The timings do not support a ratio.** Under emulation the same build
measures between 9.8 and 22.2 seconds to the first token depending on where
the threads land. Three batches of the same measurement are published in
`bench/results/inference/`, `bench/results/inference-before/` and
`bench/results/inference-joint/`:

```text
                         Rabbit    mainline    mainline (mitigations=off)
batch 1, before           21.20 s      23.34 s              9.96 s
batch 2, after            22.17 s      22.83 s             21.92 s
joint batch               10.70 s         -                    -
```

In the first batch the reference kernel without mitigations measured 9.9
seconds and in the second 21.9, so the spread between two runs of the same
build under emulation is larger than the difference between the kernels, and
no ratio can be published from those batches.

**After the arithmetic and the residency were added, the gap is larger than
the variance.** In the batch that is committed, the time to the first token on
Rabbit was 10.70 seconds and the rate 0.52 tokens per second, against 21.9
and 22.8 seconds for the reference kernel in the batches above. A gap of more than three times
survives a variance of two, so this one can be stated. What it measures is a
kernel that faults the weights in large pages, keeps them resident, and runs
a forward pass written in Rust without a single dependency outside the
standard library.

What still cannot be said:

- **The absolute numbers describe an emulator.** They say how the two kernels
  compare under the same translation overhead, not how fast either is on a
  processor.
- **The runtime is deliberately naive.** It is a matrix multiplication in
  plain Rust, without a library of vector kernels, so the absolute rate is
  not what a tuned implementation reaches. The comparison that isolates the
  arithmetic is against the reference implementation on the same host, and
  that number is measured separately.

## Results: the agent

The agent is the last piece of the claim: a model that a program asks what to
do, a tool that does it, and the model again. It runs on the same kernel, with
the same runtime, with the same checkpoint and with the same binary as the
reference kernel.

The checkpoint is `HuggingFaceTB/SmolLM2-135M-Instruct`, revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, Apache-2.0. The agent has seven
tools (`read_file`, `list_dir`, `write_file`, `compute`, `net_send`, `http_get`
and `done`), of which `list_dir` and `write_file` wait for a file system the
guest can write (issue 15). The model chooses which one to call and with which
arguments. The grammar of the
call is not a prompt instruction but an automaton over bytes: the answer of the
model cannot be anything but a well formed call of a declared tool, carrying
the keys that tool needs, each of them once, with the type the tool expects.

The task is to read `/vocab.json` from the initramfs and report its size. On
the native host, one processor, 1024 MiB, the same initramfs for both kernels:

| | Rabbit | mainline | mainline, mitigations off |
|---|---|---|---|
| Steps of the loop | 2 | 2 | 2 |
| Tokens the model generated | 39 | 39 | 39 |
| Seconds of the loop | **821.0** | 2792.7 | 2570.1 |
| Seconds of a step | **410.5** | 1396.4 | 1285.0 |
| Boot to the last result | **1200.8 s** | 2807.6 s | 2584.3 s |
| Resident memory of the guest | **857 MB** | 1021 MB | 1023 MB |

The loop is three times faster on Rabbit than on the reference kernel with the
same binary and the same checkpoint, and the guest needs 164 megabytes less.
The raw rows are in `bench/results/agent/`. The rows of the runtime on the
same host are in `bench/results/inference-native/`, and the rows of the
operations are in `bench/results/vps-1cpu/`.

The transcript of the run is the honest part:

```text
AGENT step=1 tool=read_file args={"path":"/vocab.json"} result=bytes=800662 read=4096 content=…
AGENT step=2 tool=read_file args={"path":"/vocab.json"} result=the model repeated the call
AGENT steps=2 tokens=39 seconds=821.017 tool_calls=2
```

The model of 135 million parameters reads the file and answers with its real
size, 800662 bytes, and then asks for the same file again. The loop notices
the repetition and stops there instead of spending the rest of its budget on
the same call. A model that knew when to stop would answer instead of asking
again; that is a property of the checkpoint, not of the kernel, and the number
the kernel is responsible for is the one in the table.

## Results: quantization

A model that does not fit in the memory of the machine cannot be served, so
the runtime reads a quantized checkpoint: a block of thirty two weights with
one scale per block, the scale in binary16, and the weights in four bits or
in eight. The layout is in `docs/QUANTIZATION.md`. The block is decoded
inside the matrix by vector product, without turning the tensor into floating
point first.

Measured on the development checkpoint of the repository, the same one for
both kernels:

| Format | Size | Pinned weights | Tokens of the published prompt |
|---|---|---|---|
| float | 269 060 544 B | 269 058 048 B | the reference ones |
| `Q8B32`, 8.5 bits per weight | 142 985 696 B (0.53x) | 142 983 168 B | **the same ones** |
| `Q4B32`, 4.5 bits per weight | 75 745 552 B (0.28x) | 75 743 232 B | **different ones** |

The eight bit form reproduces the reference tokens exactly, and the
difference between the two sets of logits is published with it: a maximum of
0.543618 and a mean of 0.093776788 over the eight steps of the published
prompt.

**The four bit form changes the answer**, and that is the result, not a
detail: the maximum difference of the logits is 11.669730 and the mean is
2.083776259, because the error of the weights is about 0.11 standard
deviations and the margin between the two candidates of a step goes down to
0.052. A reference implementation agrees: eight bits reproduces the reference
tokens and six, five and four do not. The four bit checkpoint is a third of
the size and it is not the same model, so it is published as a loss and not
as a win.

The memory of the guest falls with the format and the speed does not
improve: the block kernels are scalar while the floating point path is
vectorised, so the tokens per second go from 1.51 to 0.65 on Rabbit and
from 1.25 to 0.20 on the reference kernel. The format buys memory, and the
rows say so. That measurement is of the scalar kernels; the vector kernels
that came after it are in the next section, and they are not measured yet.

## Results: vector kernels for Q8 and Q4, checked and not yet measured

The block kernels of the two quantized formats were scalar, which is why the
formats lost speed. Issue 29 gave them SSE2 and AVX2 kernels, and this section
says what was shown about them and what was not. It contains no speed.

**The order of the sums.** The scalar kernels were rewritten with one
canonical order: a block of 32 weights is summed in eight lanes, each lane
starting from zero and adding a separate product, with no fused multiply add,
the lanes are reduced by a fixed tree, and the scale is applied per block in
scalar arithmetic. The SSE2 kernel (two registers of four lanes) and the AVX2
kernel (one of eight) follow the same order, so they are meant to be bit for
bit equal to the scalar one. The bfloat16 kernels keep their fused multiply
add and are not bit for bit.

**What was checked.**

- A check that runs inside the Rabbit guest compares every path with the
  scalar one by the bits of the result over 51 shapes of each format, with
  signed zeros, subnormals, extreme scales, all 256 byte values as weights of
  Q8 and every nibble in both positions of Q4. It passes in the CI, in a guest
  whose emulated processor offers AVX2 and in one without AVX2 and FMA, where it
  is required to check the scalar and SSE2 paths and says that it skipped AVX2.
  The guest runs under the emulator, so this shows that the kernels give the
  right bits and not how fast they are.
- Changing the reduction tree of an AVX2 kernel, or the order of the nibbles,
  makes the check fail. That was tried by hand, and it is the evidence that
  the check can fail.
- A decoder check decodes seven prompts for 8 and for 64 steps on every path
  of a synthetic model in bfloat16 and in the Q8 and Q4 copies that the
  repository's own quantizer writes from it. The greedy tokens are identical
  on every path; the logits of Q8 and Q4 are bit for bit equal; those of
  bfloat16 differ by 2e-6 at most; no step is closer than twice the largest
  difference.

**What was not shown.**

- No speed. The runtime can now be limited to a path, and `bench/kvm/paths.sh`
  measures tokens per second for each path and format on a KVM host, but it was
  run only under the emulator, whose numbers say nothing about speed. The first
  run on the host with KVM is still to be done.
- No real checkpoint. The checks use a synthetic model with fixed seed. The
  tokens of the real SmolLM2 checkpoint against the reference tokens are not
  yet compared for the vector paths, nor for the rewritten scalar order, whose
  rounding differs from the previous scalar code (on the synthetic models the
  largest logit difference was 4.3e-6 against a smallest margin of 0.0022).
- Not the whole decoder. Only the matrix by vector products change by path.
  The norms, the attention, the softmax and the activation are the same code
  on every path, so equal tokens do not test them.
- Not the quality of the quantization. Each format is compared with its own
  scalar kernel and not with bfloat16.
- No AVX-512. It cannot run on the only host for the rig (a Zen 2 processor)
  or under the emulator, so it was split off as issue 39.

**What a speedup would mean.** The same user program runs on Rabbit and on
Linux, so a faster kernel speeds up both. If the decode is bound by memory
bandwidth, the vector kernels may tie with the scalar ones. A speedup of a
kernel over the scalar path is a statement about the runtime, and it does not
count for the stop rule of the fair comparison (issue 33).

## Results: the same rig under KVM

Issue 10 asked for the rig to run with `--accel kvm` on a host with `/dev/kvm`,
for every dataset to say which acceleration mode produced it, and for the table
to carry both columns. This section is that run. The data, the scripts that derive
every table and the stages are in
[`bench/results/kvm/`](bench/results/kvm/README.md).

### The setup

A bare metal server, an AMD Ryzen 5 PRO 3600 with 16 GB, rented by the hour; the
host reports that it is not a virtual machine and has `/dev/kvm`. QEMU 8.2.2, the
same version as the emulated datasets, one virtual processor, 512 MiB for the
operations and 1024 MiB for the model and the agent, the same initramfs and the
same binary on both kernels. The governor was `performance`, the guest was pinned
to two cores and the resynchronisation of the disks of the host was frozen. The
kernels were built elsewhere and the host built nothing: the rig takes
`--no-build`, refuses `--accel kvm` on a host that cannot run it, and records the
accelerator, the guest and host CPU and the digests of the images in every
`summary.json`. Each version of Rabbit was measured by five runs of the three
datasets, and the table keeps the median of the medians and the spread between the
runs.

The host is AMD and the guest model is an Intel one, `Skylake-Client`, which every
dataset uses. QEMU masked five features of the model (`pcid`, `invpcid`, `hle`,
`rtm`, `erms`) and recorded the warnings. With the native model `EPYC-Rome`, the
medians of the first version differ by at most 14% (the `mmap` of the reference
kernel) and by 8% or less in the rest, except that the system call of the
reference kernel with its mitigations on goes from 49 to 72 ns.

### The table

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

A ratio above 1 means Rabbit is better, whichever the unit. The reference is Linux
7.3.0 with `mitigations=off`. A tie is a difference smaller than twice the spread
between the runs. The emulated columns are the datasets published above, one run
each, and their ratios are the relative comparison of two kernels under one
emulator. The KVM columns are the only ones of the two that can support a
statement about wall clock time, and only on this host.

### What changed along the way

Five versions of the Rabbit kernel, in the order they were made, each measured by
five runs. The first is the kernel as it was when the first run found what is
below.

| version | system call | futex | four threads | mmap | 4 KiB faults | 2 MiB faults | slowest 4 KiB fault | first token | tokens/s | agent loop |
|---|---|---|---|---|---|---|---|---|---|---|
| before | 1.17x | 2.13x | 1.30x | 0.39x | 0.54x | 0.75x | 0.98x | 0.52x | 0.61x | 0.62x |
| vector state saved | 1.17x | 1.71x | 1.28x | 0.39x | 0.55x | 0.76x | 0.94x | 0.76x | 1.02x | 0.99x |
| `mlock` without a copy | 1.17x | 1.72x | 1.29x | 0.39x | 0.56x | 0.76x | 0.92x | 0.99x | 1.00x | 0.98x |
| the fault path | 1.17x | 1.71x | 1.28x | 1.21x | 0.60x | 0.75x | 0.54x | 1.03x | 0.99x | 0.98x |
| sequential runs | 1.17x | 1.72x | 1.27x | 2.26x | 0.73x | 0.76x | 0.52x | 1.04x | 0.99x | 0.98x |

* **The same binary ran different code on the two kernels.** The runtime chooses
  its multiplication by asking the processor: AVX2 with FMA when the processor
  and the kernel report both, SSE2 otherwise. Linux enables AVX, Rabbit did not,
  so the model ran with eight lanes on one kernel and four on the other, and
  Rabbit was 0.61 times on the tokens per second. With the guest model
  `Skylake-Client,-avx2,-fma`, where the reference cannot use AVX2 either, the
  tokens per second are 28.5 for Rabbit and 27.8 for the reference with its
  mitigations off. The first version is therefore not a comparison of kernels for
  the model, and the emulated model rows above were measured in that state.
* **Rabbit did not save the vector registers on a context switch.** The kernel is
  built without SSE, so it never touches them, and the registers of the thread
  that left stayed in the processor for the next one. A boot test of two threads
  that load different patterns found 77 of 80 rounds corrupted, with one and with
  four processors, and none after the fix. The correction costs the futex a part
  of its advantage, from 2.13 to 1.71 times.
* **`mlock` copied the weights.** A fault in a file that asked for large pages
  copied 2 MiB of the archive into a fresh block, so pinning the 269 MB of the
  model was a pass over every byte of it: 46 ms against 10 ms on the reference
  kernel, and 36 of the 37 ms by which the first token was late. Mapping the pages
  where the archive holds them takes 1.3 ms. The price is that those pages are 4
  KiB, and the multiplication is 5% slower for it (0.255 against 0.242 s).
* **The fault path.** A cycle counter built into the fault path showed about 1825
  cycles in the handler, 308 of them an `invlpg` that a not present page does not
  need, and 579 for the lock and the allocation. Removing the invalidation, giving
  each processor a run of 32 frames and installing the entries already accessed
  and dirty brought `mmap` from 356 to 114 us per MiB and the fault in memory from
  313 to 290. The slowest single fault of 4 KiB went from 43 to 77 us, and to 80
  after the next version. My reading, not measured, is that a refill now takes the
  lock and searches the allocator for a run; it is published as the loss it is.
* **Sequential runs.** A fault that continues a sequential run now maps up to 16
  pages. It took the fault in memory from 290 to 237 us per MiB and `mmap` from 114
  to 61. I had estimated two to three times for the 4 KiB case, and the measurement
  was 1.2: the cost that remains scales with pages, not with exceptions.

### What this does not show

* **One host.** One processor model, one virtual processor, one measurement
  session. It says nothing about other processors, about more virtual processors,
  or about hardware that is not virtualised.
* **About 2000 cycles per page of the 4 KiB fault are not explained.** After the
  last version the workload costs about 3280 cycles per page and the handler about
  1300. KVM does not intervene in a fault (the counters of the host show no exit
  for it, and the same number of faults of its own for both kernels), and touching
  the same memory again costs 5.2 us per MiB on both kernels, so it is not the loop
  nor the translation. The host exposes no counter of the cycles of the guest, so
  it could not be sampled, and no cause is claimed. The measurements are in
  `bench/results/kvm/diagnostics/`.
* **The model and the agent are small and short.** A ratio of 0.98 or 1.04 is a tie
  or a small difference, and not a ranking.
* **The runtime was written for the kernel to be measured.** The tokens per second
  that a real inference stack reaches on this processor are not measured here.

## The tool policy enforced by the kernel

### The question and the setup

Can a kernel enforce what a tool-calling agent is allowed to do, so that the
limit holds even when the model, or the binary, does something else? The
manifest is derived from the tools the agent declares: `read_file` and
`list_dir` need a read capability, `write_file` a write capability, `net_send`
and `http_get` the capability of UDP and of TCP, and `compute` and `done`
none. The agent installs the policy once per boot with a private system call
(500), and Rabbit enforces it at `openat`, `newfstatat` by path, `socket`,
`connect`, `sendto`, `listen` and `accept`. The same binary applies the same
manifest on Linux with Landlock and seccomp when the kernel answers `ENOSYS`
to that call. Two adversaries run against both: one makes only the calls the
grammar of the agent accepts, and one is a compromised binary that issues raw
system calls under the tool layer. The data is in `bench/results/policy/`, cut
from one CI run (44360523565, commit `f94d73a`); every number below is derived
from it by `bench/policy/build_table.py`.

### Results

The class of a case is computed from the two verdicts. `same`: the same verdict
on both sides. `policy`: Rabbit refused the call with its policy and Linux
allowed it or failed it for another reason, a difference of enforcement.
`surface`: Rabbit answered `unavailable` (`ENOSYS`, it has no such call) and
Linux allowed the call, which is a missing call and not enforcement.
`linux-denies`: Linux denied the call and Rabbit answered `unavailable`.
`other`: anything else, among them a case that only one side ran.

| table | same | policy | surface | linux-denies | other | cases |
|---|---|---|---|---|---|---|
| adversary (the grammar) | 16 | 4 | 0 | 0 | 2 | 22 |
| compromised (raw system calls) | 12 | 6 | 10 | 8 | 7 | 43 |

The verdicts behind the classes:

| run | allowed | denied | unavailable | failed | cases |
|---|---|---|---|---|---|
| Rabbit, adversary | 7 | 12 | - | 2 | 21 |
| Linux, adversary | 11 | 9 | - | 2 | 22 |
| Rabbit, compromised | 7 | 11 | 22 | 3 | 43 |
| Linux, compromised | 23 | 15 | 0 | 5 | 43 |

The adversary on Linux has one case more, `read_symlink_to_secret`, because the
initramfs of Rabbit has no symbolic links. On Linux a network call counts as
allowed when its data reached the server of the test, whatever the tool then
read. The enumerated family of the adversary, 1554 paths made of six
components, had no allowed path on either side: on Linux 185 were denied and
1369 failed, and on Rabbit 1304 were denied and 250 failed. The two sides
differ in how a path fails, not in whether it is allowed: a path that does not
exist is refused by the policy of Rabbit before it is looked up, and on Linux it
answers `ENOENT`.

The rows of class `policy`, copied from `summary.csv`:

| table | case | Rabbit | Linux | reached on Linux | note from the log |
|---|---|---|---|---|---|
| adversary | `read_long_outside` | denied | failed | na | a path that does not exist answers ENOENT before Landlock is asked |
| adversary | `net_send_other_ip` | denied | allowed | yes | Landlock network rules name a port and no address |
| adversary | `net_send_other_port` | denied | allowed | yes | Landlock enforces UDP ports from ABI 10 |
| adversary | `http_get_other_ip` | denied | allowed | yes | Landlock network rules name a port and no address |
| compromised | `newfstatat_secret` | denied | allowed | na | Landlock cannot restrict stat or access |
| compromised | `newfstatat_missing` | denied | failed | na | - |
| compromised | `connect_tcp_other_ip` | denied | allowed | yes | Landlock network rules name a port and no address |
| compromised | `connect_udp_other_ip` | denied | allowed | yes | Landlock network rules name a port and no address |
| compromised | `connect_udp_other_port` | denied | allowed | yes | the test does not assert this answer |
| compromised | `sendto_other_ip` | denied | allowed | yes | Landlock network rules name a port and no address |

The existence probes are in the same file. `newfstatat` of a path outside the
root answered `denied` for the secret and for a path that does not exist on
Rabbit, and `0` and `ENOENT` on Linux, which is a way to learn which files
exist. The other four probes (`stat`, `access`, `faccessat`, `statx`) answer
`ENOSYS` on Rabbit and the same `0` and `ENOENT` on Linux: Rabbit has no
oracle there because it has no such calls.

### What this shows

On Linux 7.0.0-1020-azure with Landlock ABI 8 and the seccomp filter of the
agent, a program inside the agent could still reach another IP address on the
port the manifest allows, over TCP and over UDP, with `connect`, `sendto`,
`sendmsg` and `sendmmsg`; the servers of the test recorded the data. It could
reach another UDP port, and it could tell a file outside its read root from a
path that does not exist with `stat`. Rabbit refused with its policy the
`connect`, the `sendto` and the `newfstatat` of those cases, and has no
`sendmsg`, `sendmmsg` or `stat`. What Linux denied and Rabbit did not have is
also in the data:
`creat`, `unlink`, `rename`, `mkdir` and `symlink` inside the root, `execve` and
`execveat` outside it, `io_uring_setup` and sockets of the families `AF_UNIX` and
`AF_PACKET`, the last three by the seccomp filter; Rabbit answered
`unavailable` to the first eight and failed the sockets with `EAFNOSUPPORT`.
Both sides denied the secret, the path with `..`, the writes, the other TCP
port, `listen` and `accept`, and no path of the enumerated family was allowed
on either side.

### What this does not show

- It is not a claim of being first. Others have worked on policies for the
  tools of agents: Governed MCP (arXiv 2604.16870,
  <https://arxiv.org/abs/2604.16870>), Agate at the AgenticOS workshop of SOSP
  2026, and Before the Tool Call (arXiv 2603.20953,
  <https://arxiv.org/abs/2603.20953>). The mechanisms used for the Linux side
  are documented at <https://docs.kernel.org/userspace-api/landlock.html> and
  <https://docs.kernel.org/userspace-api/seccomp_filter.html>.
- `unavailable` is a missing system call and not enforcement: 22 of the 43 cases
  of the table on Rabbit. Only the 11 cases refused by the policy say anything
  about enforcement, and the 22 will become refusals or working calls the day
  the kernel implements them.
- The UDP port rule of Landlock needs ABI 10 and the runner has ABI 8. A newer
  kernel closes the `net_send_other_port` and `connect_udp_other_port` rows, and
  it does not close the rows about the address: Landlock keys its network rules
  by port.
- The two sides run in different environments: the Linux side is native, on a
  runner, against the loopback addresses 127.0.0.x, and Rabbit is in QEMU
  against 10.0.2.x, with no hardware acceleration. The log does not record the
  acceleration of QEMU, its version output, the hash of the kernel or of the
  archives; `summary.json` says `not recorded` for each.
- The kernel and the oracle of the tests share the `rabbit-policy` engine, so the
  tests show that the kernel applies the policy and not that the policy is right.
- There is one policy per boot and one program per boot. The helpers that read
  user memory have no fault fixup, and `bind` is not checked on either side.
- `stat` has no tool of the agent, so the `stat` rows are results of raw system
  calls and not of anything a model can ask for.
- There is no protection against prompt injection: text from a server the agent
  may reach can still steer the model into any call its tools allow.
- It is one deterministic run, counted in verdicts and not in timings, on a
  single kernel version for each side. It supports no claim about performance
  and no claim about the security of a real deployment.

### How to reproduce

In CI, the step `Run the adversary and the compromised binary against Rabbit`
boots Rabbit twice with the agent and the fixtures `/agent/hello.txt` and
`/secret.txt`, and the step `Linux policy on this runner` runs the same tables
in a restricted child, with `RABBIT_REQUIRE_LANDLOCK=1`. Locally, for Rabbit:

```sh
make agent build
# a stage directory with agent/hello.txt, secret.txt and a net.conf that holds
# address=10.0.2.2, port=, http_port=, path=/agent and adversary=1 (or compromised=1),
# with an echo server on the host for the two ports
EXTRA_DIR=stage bench/mkinitramfs.sh out.cpio rabbit-agent/rabbit-agent.elf
qemu-system-x86_64 -M q35 -m 512M -kernel "$(make -s print-kernel-path)" \
    -initrd out.cpio -device virtio-net-pci,netdev=n0 -netdev user,id=n0 \
    -display none -monitor none -no-reboot -serial file:run.log \
    -device isa-debug-exit,iobase=0xf4,iosize=0x04
```

and on Linux:

```sh
RABBIT_REQUIRE_LANDLOCK=1 cargo test --manifest-path rabbit-agent/Cargo.toml \
    adversary::tests::the_adversary_is_bounded_on_linux -- --nocapture --test-threads=1
RABBIT_REQUIRE_LANDLOCK=1 cargo test --manifest-path rabbit-agent/Cargo.toml \
    compromised::tests::the_compromised_binary_is_bounded_on_linux -- --nocapture --test-threads=1
```

The table comes from the raw logs with `python3 bench/policy/build_table.py`,
and its tests with `python3 -m unittest discover -s bench/policy`.

## Losses

- **The thread workload was a loss and then became a win.** With eager
  allocation of the mappings, Rabbit was 1.5 times slower than the reference
  kernel there. Demand paging made it four times faster. The loss was
  published at the time, in `bench/results/demand-paging/`.
- **One boot of the reference kernel did not finish** in an earlier run and
  the rig recorded the twenty seven missing samples instead of hiding them.
  A repeated run completed all thirty iterations, so it was a flake. Both
  runs are in `bench/results/anomaly/` and `bench/results/repeat/`.
- **Under KVM, three rows that Rabbit won under emulation are losses**: the cost
  of a fault in memory with 4 KiB pages (0.73 times), with 2 MiB pages (0.76
  times), and the slowest fault of 4 KiB (0.52 times). The last one got worse with
  the changes that improved the others, from 42 to 80 us, and is published with
  them.
- **Rabbit did not save the vector registers on a thread switch**, and the first
  run of the model under KVM ran different instructions on the two kernels. Both
  were found by the run under KVM, fixed, and are in `bench/results/kvm/`.
- **The first reference image could not be asked for large pages at all**:
  `madvise(MADV_HUGEPAGE)` returned `EINVAL`, because that advice is only
  valid when the kernel is built with transparent huge pages. The workload
  reports what the kernel answers, with `MADV_NORMAL` as the control, so the
  difference was found by a probe and not by reading a configuration file.

## Threats to validity

- **The emulated datasets describe an emulator.** Every absolute number of them
  is an emulator's. The run under KVM showed that the ratios are not preserved:
  advantages of 1.3 to 123 times became 1.2 to 2.3 times, and three rows
  reversed. The KVM column is one host and one processor model, and the
  `Skylake-Client` model of the guest is not the processor of the host.
- **The measurement variance under emulation is large**, up to a factor of
  two for the inference timings, and it is larger than the difference
  between the kernels there.
- **The clock source differs.** The reference kernel reads the time stamp
  counter through the vDSO, Rabbit through a system call, which costs
  Rabbit one system call per measurement point. The measurements that
  timestamp every sample are biased against Rabbit.
- **One processor, 512 MiB.** A server that batches requests is not that.
- **The runtime is deliberately small.** It is a matmul in plain Rust with no
  BLAS, with a SSE2 path and an AVX2 and FMA path that it chooses at run time by
  asking the processor, so that the comparison is about the kernel. The choice
  has a consequence: a kernel that does not enable AVX runs the other path, as
  Rabbit did until the run under KVM showed it. The absolute tokens per second
  are not what a real inference stack would reach.
- **The workload is ours.** It measures the operations the kernel does while
  serving a model, not the model.

## Reproducing it

```sh
# the kernel and the runtime
make build
make run                      # boots and prints the memory map and the ticks
make run-infer                # boots with the inference runtime

# the comparison
bench/prepare-mainline.sh     # on a Linux host: the reference image
bench/mkinitramfs.sh
python3 bench/run.py \
    --rabbit "$(make -s print-kernel-path)" \
    --mainline ~/rabbit/artifacts/vmlinux \
    --mainline-variant "nomitig=mitigations=off" \
    --out bench/results

# the checkpoint, verified
bench/fetch-model.sh ~/rabbit/artifacts/model-instruct

# the agent
make agent
EXTRA_DIR=~/rabbit/artifacts/model-instruct \
    bench/mkinitramfs.sh ~/rabbit/artifacts/agent.cpio rabbit-agent/rabbit-agent.elf
python3 bench/run.py \
    --rabbit "$(make -s print-kernel-path)" \
    --mainline ~/rabbit/artifacts/vmlinux \
    --initrd ~/rabbit/artifacts/agent.cpio --memory 1024M \
    --only agent-steps,agent-tokens,agent-seconds,agent-step-ms \
    --out bench/results/agent

# the oracle for the runtime
python3 bench/reference/reference_tokens.py --out reference.json

# the vector kernels of Q8 and Q4 on every path, in the guest
make matvec-check
make matvec-check-run MATVEC_CPU="Skylake-Client,-avx2,-fma" MATVEC_REQUIRE="scalar sse2"

# tokens per second per path, on a host with KVM (see bench/kvm/README.md)
bench/kvm/paths.sh --accel kvm
```

Every assertion in the continuous integration runs the kernel and checks the
serial log, so the claims in this document are checked on every push.

## What comes next

- **The fault in memory, and what it costs per page.** It is the largest loss left
  under KVM, 0.73 times, and 2000 cycles per page of it are not accounted for. A
  host that exposes the counters of the guest, or a profiler inside the guest, would
  say where they go.
- **The vector kernels of Q8 and Q4 on KVM.** They are written, tested and
  gated in the CI, and not measured. One session on the host with KVM runs
  `bench/kvm/paths.sh` and compares the tokens of the real checkpoint with the
  reference ones.
- **Real hardware.** The numbers under KVM are one virtualised processor. A
  physical machine booting Rabbit is the next rung of the roadmap.
- **A scheduler with a better tail, and more than one processor.** A server
  that batches requests lives on the latency of the tail. The kernel starts
  more processors since M9, but every dataset published here is one
  processor.
- **The same method applied to a real serving stack**, with a model that is
  actually used, rather than one that is served to see what it costs.

## The method, for anyone else

The kernel was written with the assistance of an AI agent, against milestones
whose exit criteria are machine checked: a serial log that has to contain
exact lines, a unit test on the host, a row in the benchmark data. The
design decisions, the criteria and the verification belong to a human; the
code was written by an agent against them.

That is the part that may be worth copying. "How much of an operating system
can be written this way, and how do you know it works" is a question worth an
answer, and this repository is one answer to it, with the losses published
next to the wins.
