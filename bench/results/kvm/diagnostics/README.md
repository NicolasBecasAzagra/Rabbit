# Diagnostics

Measurements taken to find where the cost of a page fault goes, kept with the raw
output they came from. They are not benchmarks: some use a kernel built with the
`fault-profile` feature, which adds timestamp reads to the fault path, or a
benchmark binary patched for the purpose, so their absolute numbers are not the
ones of the table. Each file says what it measured. The host is the one of the
parent directory, with the guest CPU `Skylake-Client` and one virtual processor.

| file | what it is | kernel |
|---|---|---|
| `profile.log` | serial log of one run of the operations workload, with the cycles of each phase of an anonymous 4 KiB fault | `486d75f`, the cycle counters added to `131d288` |
| `profile2.log` | the same, after the invalidation of a not present page was removed and a frame run per processor was added | `f6b8abf` with the counters |
| `p3.log` | the same, after a fault that continues a sequential run maps several pages. The `[profile]` lines divide by faults, not by pages, so with up to 16 pages per fault they are not comparable with the two logs above | `487051a` with the counters |
| `samples.rabbit`, `samples.mainline` | the global counters of KVM sampled once per second while a guest ran the operations workload, one line per second, as the debug file system of the host shows them | the images of the stage `pin-131d288` |
| `kvmstat-rabbit.log`, `kvmstat-mainline.log` | the serial logs of those two runs | same |
| `retouch.patch` | the change to `rabbit-bench` that touches the 256 MiB of the 4 KiB fault workload a second time, now mapped, and prints its cost. Not committed | |
| `retouch-<kernel>-<N>.log` | three runs of the workload built with that patch on each kernel | `487051a` and the reference |

## What they show

**The cycles of a fault** (`profile.log`, `profile2.log`; 3559 MHz, so 3559 cycles
are one microsecond):

| phase | `profile.log` | `profile2.log` |
|---|---|---|
| lookup of the mapping | 59 | 59 |
| lock, allocation and zero fill of the frame | 1244 (zero fill 665) | 959 (zero fill 767) |
| update of the page table | 214 | 182 |
| `invlpg` | 308 | 43, which is the timestamp read: the invalidation was removed |

The mean of the whole handler in those logs includes the 1280 faults of 2 MiB,
which cost about 700 000 cycles each, so only the phases of the 4 KiB path are
compared. The workload costs about 4550 cycles per page at the start of this
series (`profile.log`, 1279 ns per page), so about 2700 cycles were outside the
handler.

**KVM does not intervene in a fault.** In `samples.*` the guest of Rabbit ends with
33 256 exits in all and `pf_guest=0`, and `invlpg=0`, for 359 832 faults, and the
host fixed 734 faults of its own against 736 for the reference kernel. The faults of
Rabbit are taken and resolved inside the guest, at the same host cost as the
reference. The counters of the reference kernel run to 140 samples, because that
kernel does not power the machine off.

**The loop and the TLB are not the difference.** `retouch-*.log`: touching the same
256 MiB again, with every page mapped, costs 5.1 to 5.3 us per MiB on Rabbit and on
the reference kernel, against 237 and 173 us for the first pass, so the cost is in
the fault and not in reaching the memory.

**After `487051a`** the handler costs about 1300 cycles per page (in `p3.log`,
`alloc` 16 649 and `map` 2 984 cycles per fault, 23 862 faults, 358 553 zeroed pages,
so about 1100 and 200 per page; adding the lookup and the checks), and the workload
costs 237 us per MiB, about 3280 cycles per page. About 2000 cycles per page are not
accounted for. The host exposes no counter of the cycles of the guest
(`cycles:G` reads zero), so it could not be sampled, and no cause is claimed.
