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
