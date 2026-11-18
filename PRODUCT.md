# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

The primary visitor, confirmed by the owner, is a person who hires or collaborates: an engineering manager, a tech lead, a recruiter or a possible collaborator who arrives from a LinkedIn post. They skim for seconds and decide whether the author's technical judgment is credible; the job is to assess rigour and depth quickly and then, if interested, verify. Engineers who check numbers against the raw data and builders of local agents are likely secondary visitors, but this is not confirmed.

## Product Purpose

Rabbit is a from-scratch x86_64 kernel written in Rust that speaks the Linux userspace ABI, with an inference runtime and an agent on top. The site presents the project, its measured results, the roadmap and the documentation. Success is that a visitor leaves convinced the work is real, rigorous and honestly reported, and knows how to reach the repository and the author.

## Positioning

Confirmed by the owner: Rabbit is a kernel and operating system for local AI, and the goal the owner wants to reach is to be the fastest in the world at it. That is a goal, not a result: no dataset supports a speed claim beyond ratios measured under QEMU TCG on one host, so the site states it as the aim, next to what is measured today and what is still open (hardware acceleration, issue 10), and never as a current fact. What a neighbour could not truthfully copy is the method and the evidence: the same static binary runs on Rabbit and on mainline Linux, the raw data is committed, the losses are published, and the newest result is a tool policy enforced by the kernel and measured against Linux with Landlock and seccomp (bench/results/policy/, REPORT.md).

## Operating Context

The site is an Astro project in website/, deployed to GitHub Pages from main on every push, in English. The docs pages are synced from the markdown files of the repository. Numbers come from datasets under bench/results/ and from CI logs. The audience arrives from a LinkedIn post.

## Capabilities and Constraints

- Today the kernel boots in QEMU, serves a 135M parameter model on the CPU, and runs an agent whose tool calls are forced by a grammar; all benchmarks ran under QEMU TCG, an emulator, with no hardware acceleration.
- The newest measured result is the tool policy: Rabbit refuses calls outside the manifest of the tools of the agent, and on a Linux 7.0 kernel with Landlock ABI 8 plus seccomp the same agent could still reach another IP on the allowed port, send UDP to another port and stat files outside its read root.
- Terminology: an `unavailable` answer (ENOSYS) is a system call the kernel does not have, never enforcement; 22 of the 43 raw calls of the compromised binary on Rabbit are of that kind.
- The agent does not finish tasks yet (the model repeats a call); the 135M and 360M models fail the probe.
- Undecided: how strongly the site states the positioning "a kernel for local agents" given the evidence. Every statement of that positioning must be bounded by what the datasets show, and the limits stay visible next to it.

## Brand Commitments

- Strictly black and white, like a rabbit; no accent colour. The word Rabbit and two ears as negative space are the mark.
- Honesty rules, binding for all site work: no invented numbers; the context "QEMU TCG, an emulator" stays next to benchmark figures; ENOSYS answers are called unavailable, never protection; never claim to be the first or the only one, and cite prior art (Governed MCP arXiv 2604.16870, Agate at the AgenticOS workshop of SOSP 2026, Before the Tool Call arXiv 2603.20953).
- Only real data and logs: the boot log, tables and charts come from files of the repository, never illustrative.

## Evidence on Hand

- REPORT.md, docs/ARCHITECTURE.md, docs/BENCHMARKING.md and the datasets under bench/results/, including bench/results/policy/ with raw logs, summary.csv, totals.csv and summary.json (CI run 44360523565, commit f94d73a).
- The real boot log in docs/BOOT.md and the CI logs of the repository.
- The repository: github.com/NicolasBecasAzagra/Rabbit, with releases, milestones and issues.
- The author's real contact addresses, given by the owner: GitHub https://github.com/NicolasBecasAzagra and LinkedIn https://www.linkedin.com/in/nicolasbecas. No other social address exists.
- Absences future work must not fabricate: results with KVM or on real hardware, users or testimonials, photographs, any other contact address, and any performance claim beyond the datasets.

## Product Principles

- Evidence before claim: every statement links to the file that supports it.
- The losses and the limits are content, placed next to the results they qualify.
- Say what is missing next to what exists.
- A visitor can verify in one click: raw data and the script that derives each table are one link away.
- The mark stays quiet and exact: the data carries the weight.

## Accessibility & Inclusion

WCAG AA contrast on black and on white, visible keyboard focus, state never carried by colour alone (fill, outline and hatch), and every animation resolved to its final state under prefers-reduced-motion.
