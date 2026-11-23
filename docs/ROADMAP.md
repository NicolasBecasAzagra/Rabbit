# Roadmap: from an experiment to the standard for local AI

The aim is for Rabbit to become the system people choose to run language models
on their own hardware, because it runs them fast and says exactly how fast.
That is a goal, not a result. Today every number comes from an emulator, there
is no accelerator driver and no physical machine has booted the kernel.

Each rung has a proof that a machine can check. A rung is not done until its
proof is published with the raw data, and a loss is published like a win.

## Where the speed is

The speed of a model on a machine is decided mostly by memory bandwidth and by
the kernels of the compute, usually on an accelerator. An operating system
matters less than that.

The first run on hardware acceleration agrees with that. On one host, some
operations of the kernel are 1.2 to 2.3 times cheaper than the reference (7.4 times
for the tail of a 2 MiB fault), the model and the agent loop are level with it, and
the cost of a fault in memory is 0.73 times. The calculation is where the time goes, and it is the same code on
both sides. Rabbit can only win where the system decides: mapping
the weights without a copy, large pages, a scheduler that does not disturb the
thread that decodes, the time to the first token from a cold boot, and idle
power. The roadmap measures those, and it has a stop rule (see the end).

## The ladder

| Rung | Milestone | Proof that closes it | Issues |
|---|---|---|---|
| 0 | v0.4 Credible numbers (done) | The rig runs with `--accel kvm` and every dataset names its acceleration mode. Proof: [`bench/results/kvm/`](../bench/results/kvm/README.md), one bare metal host | [#10](https://github.com/NicolasBecasAzagra/Rabbit/issues/10) |
| 1 | v0.9 Real hardware | A named physical machine boots Rabbit from USB and the agent answers a task there | [#26](https://github.com/NicolasBecasAzagra/Rabbit/issues/26), [#27](https://github.com/NicolasBecasAzagra/Rabbit/issues/27), [#28](https://github.com/NicolasBecasAzagra/Rabbit/issues/28) |
| 2 | v0.10 Fast on the CPU | AVX2 and AVX-512 paths token exact with the scalar path, GGUF read, a model of 1B or more running | [#29](https://github.com/NicolasBecasAzagra/Rabbit/issues/29), [#30](https://github.com/NicolasBecasAzagra/Rabbit/issues/30), [#31](https://github.com/NicolasBecasAzagra/Rabbit/issues/31) |
| 3 | v0.11 The fair comparison | Same weights, same machine, against llama.cpp over Linux, with the losses | [#32](https://github.com/NicolasBecasAzagra/Rabbit/issues/32), [#33](https://github.com/NicolasBecasAzagra/Rabbit/issues/33) |
| 4 | v0.12 One accelerator | A written go or no-go, then one compute dispatch that matches the CPU | [#34](https://github.com/NicolasBecasAzagra/Rabbit/issues/34), [#35](https://github.com/NicolasBecasAzagra/Rabbit/issues/35) |
| 5 | v1.0 The standard | A bootable image, an OpenAI compatible API and a matrix of tested machines | [#36](https://github.com/NicolasBecasAzagra/Rabbit/issues/36), [#37](https://github.com/NicolasBecasAzagra/Rabbit/issues/37), [#38](https://github.com/NicolasBecasAzagra/Rabbit/issues/38) |

In parallel and not on the path of speed: the policy hardening of v0.8
(issues 21 to 25), the agent that finishes (v0.5, issues 14 and 15) and the
model context protocol and TLS (v0.6, issues 18 and 19). The first of these is
what makes Rabbit more than a fast runtime: a kernel that enforces what an
agent may do.

## What "the standard" would mean

Nobody can declare a standard. It would mean that a person who buys hardware
for local AI can boot an image, load the models they already have, call the API
their tools already speak, and see a published table in which Rabbit is at
least as fast as the alternative on their machine. Each of those is a rung
above. Support for "all hardware" is the last claim, and the matrix of v1.0
lists only machines that were tested, with the log.

## Stop rule

If the experiments of the fair comparison (issue 33) show no gain beyond the
spread of the runs, the report says that the operating system is not where the
speed is, and the roadmap changes: the project keeps the policy and the method
and drops the claim of speed.

## Risks

| Risk | What it does to the plan |
|---|---|
| The accelerator documentation is not enough | Rung 4 ends at the written no-go, and the project stays a fast CPU system with a policy |
| Physical hardware needs quirks that take longer than the kernel | Rung 1 is limited to one machine until it is stable |
| llama.cpp is faster on every row | The comparison is published as it is, and the claim of speed is dropped |
