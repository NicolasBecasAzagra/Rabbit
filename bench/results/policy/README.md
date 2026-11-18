# The policy dataset

Raw lines of two adversaries run against the policy that Rabbit enforces in the
kernel and against Landlock and seccomp on Linux, from one CI run, and the table
derived from them. See the section "The tool policy enforced by the kernel" of
`REPORT.md` for what it shows and what it does not.

| file | what it is |
|---|---|
| `rabbit-adversary.log` | the calls the grammar of the agent accepts, on Rabbit in QEMU |
| `rabbit-compromised.log` | raw system calls under the tool layer, on Rabbit in QEMU |
| `linux-adversary.log` | the same adversary on Linux, restricted by Landlock and seccomp |
| `linux-compromised.log` | the same raw system calls on Linux |
| `summary.json` | the run, the commit, the environment as the log shows it, and what is not in the log |
| `summary.csv` | one row per case, derived by `bench/policy/build_table.py` |
| `totals.csv` | the number of cases in each class, derived by the same script |

## Where the raw lines come from

The run is CI run 44357746548 of commit `759128d2d1f1a9a6554ccc42e4db54d5a315a036`.
The log was downloaded with `gh run view 44357746548 --log` and is not in the
repository: it holds the whole build. Each of its lines is

    JOB <tab> STEP <tab> [BOM]TIMESTAMP PAYLOAD

`bench/policy/extract_ci_log.py` removes the job, the step, the byte order mark
and the timestamp (`YYYY-MM-DDTHH:MM:SS.fffffffZ`) with the single space after
it, and nothing else: the payload is copied byte for byte. It keeps the lines of
two steps, in the order of the log, whose payload starts with `ADV `, `RAW `,
`GAP `, `POLICY ` or `CHILD AGENT policy`:

* the step `Run the adversary and the compromised binary against Rabbit`: the
  lines before the line that says `compromised.cpio` was written are the
  adversary boot (`rabbit-adversary.log`), and the lines after it are the
  compromised boot (`rabbit-compromised.log`);
* the step `Linux policy on this runner`: it runs three tests, each after a
  `Finished` line of cargo. The second is the adversary (`linux-adversary.log`)
  and the third the compromised binary (`linux-compromised.log`). The first,
  the test of the policy itself, prints no line of the comparison.

Reading the files:

* `ADV case=` and `RAW case=` are the answers as the child process saw them.
  `ADV net` and `RAW net` are printed by the parent of the Linux test, which owns
  the servers: the verdict of a network call there is what the servers recorded,
  and `reached` says whether the data got to one. The table uses that verdict.
* `GAP` lines are printed by the Linux tests where Linux let a call through and
  Rabbit refuses it or does not have it. `POLICY deny` lines are the log of the
  kernel of Rabbit, at most sixteen per boot.
* The `RAW summary` line of the Linux log counts the answers of the child, before
  the parent has judged the network calls, so its counts differ from the table.

## What is incomplete

`rabbit-adversary.log` is partial. The CI step prints only the first 60 lines of
the ADV and POLICY lines of the boot, so the file has the 21 named cases and the
first 21 rows of the enumerated family, and no summary line. The other three
logs are complete and their cases add up to their summary lines, which the script
checks. `summary.json` lists what the log does not show and is written as
`not recorded`.

## Regenerating the table

```sh
python3 bench/policy/extract_ci_log.py LOG bench/results/policy   # from a downloaded log
python3 bench/policy/build_table.py                               # writes summary.csv and totals.csv
python3 -m unittest discover -s bench/policy                      # the tests of the script
```

The script stops with a message when a log is missing, when a log that is not
declared incomplete in `summary.json` has no summary line, or when the cases do
not add up to the summary.
