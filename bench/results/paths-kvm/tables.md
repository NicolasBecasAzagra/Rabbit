# Tokens per second per code path

* mode: **kvm**
* guest CPU model: `Skylake-Client`, one virtual processor, 1024M of memory
* one host: AMD Ryzen 5 PRO 3600 6-Core Processor (Linux 6.8.0-149-generic x86_64); QEMU emulator version 8.2.2 (Debian 1:8.2.2+ds-0ubuntu1.18)
* runs per cell: 5; the model: artifacts/model-base
* Rabbit kernel sha256 `fcff603a007b2e15112b765998a586fdcccfeefb69cc0e55dfa4c791214cffb8`
* the path of a row is the one the guest reported, not the one requested
* runs that count: 60 of 60

## Tokens per second

| format | path | kernel | runs | n | median tok/s | min..max | median ttft ms | reference match | note |
|---|---|---|---|---|---|---|---|---|---|
| bf16 | scalar | rabbit | 5 | 5 | 8.208 | 8.112..8.260 | 611.121 | true |  |
| bf16 | sse2 | rabbit | 5 | 5 | 26.644 | 26.303..26.801 | 190.093 | true |  |
| bf16 | avx2 | rabbit | 5 | 5 | 46.034 | 45.889..46.434 | 110.878 | true |  |
| bf16 | avx2 | mainline | 5 | 5 | 46.306 | 25.472..46.933 | 115.625 | true |  |
| q8 | scalar | rabbit | 5 | 5 | 12.874 | 12.856..12.944 | 387.965 | true |  |
| q8 | sse2 | rabbit | 5 | 5 | 32.071 | 31.998..32.092 | 155.862 | true |  |
| q8 | avx2 | rabbit | 5 | 5 | 60.368 | 59.980..60.805 | 82.851 | true |  |
| q8 | avx2 | mainline | 5 | 5 | 59.903 | 59.824..60.216 | 84.256 | true |  |
| q4 | scalar | rabbit | 5 | 5 | 11.742 | 11.708..11.803 | 425.677 | false |  |
| q4 | sse2 | rabbit | 5 | 5 | 23.064 | 23.025..23.075 | 216.399 | false |  |
| q4 | avx2 | rabbit | 5 | 5 | 32.413 | 32.369..32.433 | 154.082 | false |  |
| q4 | avx2 | mainline | 5 | 5 | 32.937 | 32.410..32.962 | 156.070 | false |  |

## Ratios (above 1 means faster)

| kind | format | path | against | tok/s ratio | ttft ratio | reading | note |
|---|---|---|---|---|---|---|---|
| path-vs-scalar | bf16 | sse2 | rabbit scalar | 3.246 | 3.215 | faster |  |
| path-vs-scalar | bf16 | avx2 | rabbit scalar | 5.608 | 5.512 | faster |  |
| path-vs-scalar | q8 | sse2 | rabbit scalar | 2.491 | 2.489 | faster |  |
| path-vs-scalar | q8 | avx2 | rabbit scalar | 4.689 | 4.683 | faster |  |
| path-vs-scalar | q4 | sse2 | rabbit scalar | 1.964 | 1.967 | faster |  |
| path-vs-scalar | q4 | avx2 | rabbit scalar | 2.760 | 2.763 | faster |  |
| rabbit-vs-mainline | bf16 | avx2 | mainline same path | 0.994 | 1.043 | tie (the ranges overlap) |  |
| rabbit-vs-mainline | q8 | avx2 | mainline same path | 1.008 | 1.017 | tie (the ranges overlap) |  |
| rabbit-vs-mainline | q4 | avx2 | mainline same path | 0.984 | 1.013 | tie (the ranges overlap) |  |

## How to read it

* A ratio of a path against the scalar one is a statement about the vector
  kernels of the runtime. It is not a statement about the operating system and
  it does not count for the stop rule of the comparison of Rabbit with mainline.
* Rabbit against mainline on the same path and format is the comparison of the
  two operating system kernels with the runtime held fixed.
* A reading of tie means that the ranges of the two rows overlap. A decode
  that is bound by the bandwidth of memory can tie on every path.
* A run counts when its tokens equal those of the scalar path of the same format.
  `reference match` is whether they also equal the published float reference:
  false is expected for q4 and is a statement about the quantization, not about
  the path.
* Everything here is one host, one guest CPU model and one virtual processor.
