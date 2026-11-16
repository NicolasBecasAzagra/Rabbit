# Security policy

Rabbit is a research kernel. It is not hardened, it has not been audited, and
it is not meant to run anything you cannot afford to lose. The documents say it
plainly: [docs/BENCHMARKING.md](docs/BENCHMARKING.md) lists "more secure" and
"production ready" among the claims that are not made.

That does not make a report less useful. A memory safety bug in the kernel, a
way out of the guest, or a way to make the agent run a tool it did not declare
is exactly what this project wants to hear about.

## Supported versions

Only the `main` branch. There are no releases that receive fixes on their own.

## How to report

Use the private report of GitHub:
[Report a vulnerability](https://github.com/NicolasBecasAzagra/Rabbit/security/advisories/new).

Please do not open a public issue for something that can be exploited. Say in
the report:

- what you found and where, with the file and the commit,
- how to reproduce it: the command line of QEMU and the program that triggers it,
- what you expected and what happened.

## What to expect

This is a personal project, so there is no service level. A report is read, and
the answer says whether it is a bug, whether it is in scope and when a fix is
likely. A fix goes in as one commit, with a test when the bug can be tested, and the
report is credited there unless you prefer otherwise.

## In scope

- The kernel crates: memory safety, the system call interface, the loader, the
  scheduler and the network stack.
- The inference runtime and the agent: the grammar of the tool call, the tools
  and the way they reach the kernel.
- The build and the CI workflows of this repository.

## Out of scope

- Missing mitigations that the documents already name as absent, such as page
  table isolation, which Rabbit leaves out on purpose.
- Anything that needs a hypervisor or a host you do not control to be hostile
  to itself.
- The benchmark numbers. A flaw in the method is welcome as an issue.
