# GGUF: the oracle of issue 30

Rabbit reads GGUF files (`rabbit-infer/src/gguf.rs`, `gguf_model.rs`). A reader that
has only been tested against files that it wrote itself proves little, so this
directory holds what **llama.cpp**, the implementation that defines the format,
produced for the same weights: the files that its quantizer wrote, the tokens that
it generates from them, and the values that its Python package dequantises.
Nothing here is Rabbit's own output, and nothing was edited by hand.

Every number is one host, one processor and one build of llama.cpp. The exact
versions, commands and the SHA-256 of every file are in
[`provenance.json`](provenance.json).

## What is here

| path | what it is |
|---|---|
| `llama_golden.cpp` | the harness: loads a GGUF with llama.cpp on the CPU with one thread and the plain kernels (`use_extra_bufts = false`, no repacked layouts), decodes 32 greedy tokens after each prompt of a list of token ids, and prints the token, the best and the second best logit of each step |
| `dump_reference.py` | reads GGUF files with the `gguf` package of the same llama.cpp checkout and writes the type of every tensor, the SHA-256 of the raw bytes and of the dequantised float32 values of each quantized tensor, and a few whole blocks of each type as test vectors (the bytes and the bits of the float32 values) |
| `smollm2/` | the real model: `HuggingFaceTB/SmolLM2-135M` at revision `93efa2f`, converted to F16 by `convert_hf_to_gguf.py` and quantized to `Q8_0` and `Q4_K_M` by `llama-quantize`: eight prompts, the golden tokens of each of the three files, the logs of the quantizer, and `ref/` |
| `synthetic/` | the small model that the repository generates (`rabbit-infer-gen --gguf --hidden 256 --ffn 384 --heads 4 --kv-heads 2 --vocab 512 --layers 4`), quantized by `llama-quantize` to `Q8_0` and `Q4_K_M` (the two files are committed, 2.2 and 1.4 MB), six prompts, the golden tokens of the F32, the Q8_0 and the Q4_K_M files, and `ref/` |

The files of the real model (145 and 105 MB) are not committed. Their SHA-256 is
in `provenance.json`, and `ref/tensors-*.json` has the hash of every tensor.

**What cannot be reproduced from the tree.** Every statement below about the real
SmolLM2 files (the 211 tensors verified, the noise table, the equal and near tie
counts) comes from one run, on one Mac (arm64) for Rabbit and one x86 host for the
oracle (`provenance.json`), and cannot be checked from a clone: the GGUF files are
not in it, and CI does not run them. Only the synthetic files and the golden JSON
files are committed, and the unit tests of `rabbit-infer` run on them. Whoever
wants to reproduce the real numbers has to build llama.cpp at the commit of
`provenance.json`, download `HuggingFaceTB/SmolLM2-135M` at the revision of that
file (and check its hashes), convert and quantize it with the commands of
[Rebuilding it](#rebuilding-it), and then run on the files that result:

```sh
rabbit-infer-golden --verify-tensors smollm2-135m-Q8_0.gguf ref/tensors-smollm2-135m-Q8_0.gguf.json
rabbit-infer-golden smollm2-135m-Q8_0.gguf 32 smollm2/prompts.txt smollm2/golden-Q8_0.json > run-Q8_0.json
rabbit-infer-golden --compare run-Q8_0.json smollm2/golden-Q8_0.json
```

(`dump_reference.py` writes the `ref/` of the files that were converted by the
person who runs it, and the hashes only match if the file is the same: compare
them with `provenance.json` first. The last command needs no model: it reads the
`weights` and `mode` fields of the run document.)

## The tensor types of the files

`llama-quantize` chooses a type per tensor, so a `Q4_K_M` file is a mix. Read from
the files with `gguf-py`:

| file | types |
|---|---|
| SmolLM2 `Q8_0` | 61 `F32` (the norms), 211 `Q8_0` |
| SmolLM2 `Q4_K_M` | 61 `F32`, 15 `Q8_0` (the embedding and 14 of `attn_v`), 166 `Q5_0`, 14 `Q6_K` (of `ffn_down`), 16 `Q4_K` (of `ffn_down`) |
| synthetic `Q8_0` | 9 `F32`, 29 `Q8_0` |
| synthetic `Q4_K_M` | 9 `F32`, 22 `Q4_K`, 3 `Q6_K`, 2 `Q5_0`, 2 `Q8_0` |

The width of SmolLM2 is 576, which is not a multiple of 256, so llama.cpp falls
back from `Q4_K` and `Q6_K` to `Q5_0` and `Q8_0` for every tensor whose rows are
576 long, and `Q4_K` and `Q6_K` appear only in `ffn_down` (1536 long). That was
deduced from its source before the file existed, and the file agrees. To read a
`Q4_K_M` file a reader needs `Q8_0`, `Q5_0`, `Q4_K`, `Q6_K` and `F32`. The
synthetic model has a width of 256, so it holds all five types too.

## What the goldens show

- The golden of the F16 file starts with `260 3575 282 260 1798 30 198 198` for
  the prompt `The capital of France is`: the eight tokens of the float32 reference
  of `bench/reference/reference.json`, produced by another implementation.
- Quantization changes the answer **in llama.cpp itself**. Against the F16 file,
  the `Q8_0` file generates other tokens in 3 of the 8 prompts within 32 steps (the
  first divergence at step 5, 13 and 0), and the `Q4_K_M` file in 6 of the 8. So
  Rabbit has to be compared with llama.cpp on the **same file**, never with the F16
  one, and a divergence is only meaningful next to the margin between the two best
  logits of that step, which the goldens carry.
- llama.cpp quantizes the activations to `Q8_0` (for `Q8_0` and `Q5_0` weights) or to
  `Q8_K` (for the K quants) before it multiplies, in its CPU kernels; Rabbit keeps
  the activations in float32. The logits of the two cannot be equal, only the
  tokens can, and only where the margin allows it. What emulating that quantization
  does and does not explain is measured in the next section.

## The plain and the emulated comparison

`rabbit-infer-golden MODEL.gguf 32 prompts.txt golden.json` decodes with Rabbit's
own arithmetic (the **plain** comparison) and `--emulate-llama` decodes with the
activations that llama.cpp's CPU backend multiplies (the **emulated** one): the
activation of every product with a `Q8_0` or `Q5_0` weight is packed to `Q8_0` and
that of a `Q4_K` or `Q6_K` weight to `Q8_K` (reference quantizers of ggml
`d9fe7bf`, unpacked back to float32), and the keys and values are rounded to
binary16, as the default `f16` cache of `llama_golden.cpp`'s context does. The
`F16` weights keep a float32 activation in that commit (`vec_dot_type` is `F32`).
The integer dot product itself is not emulated. The emulation is a **diagnostic of
the oracle's arithmetic**, off by default; it is not a claim about Rabbit's.

`rabbit-infer-golden --verify-tensors FILE.gguf ref/tensors-FILE.gguf.json` hashes
every quantized tensor (raw bytes, and Rabbit's dequantised values as little endian
float32) and compares with `gguf-py`. In the run of this work, on both real files:
all 211 tensors of each, raw and float32 hashes equal (not reproducible from the
tree, see above). The tool fails when the file has a quantized tensor that the
reference does not list, when it has none and the reference does, when a quantized
tensor has a type that Rabbit does not read, or when the reference names a tensor
twice, and its `VERIFY` line prints `skipped=<n>` with the types of the tensors that
it did not read. In that run the reader and the four dequantisers were bit exact on
the real files, so the difference below is not in the weights.

Top logit noise (largest `|Rabbit - llama.cpp|` over the steps with equal tokens),
8 prompts of 32 steps, the logit scale being about 25. The real rows are one run on one
Mac against one oracle host and cannot be reproduced from the tree; the synthetic rows
are what that Mac gave, and the unit tests assert invariants, not these counts:

| file | plain | emulated |
|---|---|---|
| SmolLM2 `F16` | 0.0085 (8 equal) | 0.0046 (8 equal) |
| SmolLM2 `Q8_0` | 0.61 (7 equal, 1 near tie) | 0.82 (6 equal, 2 near ties) |
| SmolLM2 `Q4_K_M` | 1.54 (4 equal, 4 near ties) | 1.12 (4 equal, 4 near ties) |
| synthetic `Q8_0` | 0.027 (5 equal, 1 near tie) | 0.028 (4 equal, 2 near ties) |
| synthetic `Q4_K_M` | 0.028 (6 equal) | 0.033 (5 equal, 1 near tie) |

**The hypothesis that the activation quantization explains the noise was not
confirmed.** The emulation does not make the noise collapse nor the tokens equal.
What it does show: the `f16` cache explains about half of the `F16` residual; on one
prompt of the synthetic `Q8_0` file the emulated top logit was within 1e-3 of the
golden for 30 steps in a row and the plain run was not (seen once, not tested: it
depends on no rounding decision differing); and the logits move by as much as the
whole noise (0.13 on average) when a single rounding decision of the quantizer
changes, by the oracle's own AVX2 variant of the `Q8_0` quantizer (one `ulp` of the
scale, half-way ties) or by a random one `ulp` change of its input. Real activations
do hit exact ties. The attempt to measure how much of the difference is the oracle's
own arithmetic therefore ended negative: the logits of a quantized model depend on
rounding decisions that cannot be reproduced bit for bit by emulating the values.
Not measured: where in the model that sensitivity sits, and any other processor or
build. The quantization of the activations *may* flip a near tie; it was not shown.

The rule of `golden_compare.rs` (module documentation) follows from it: tokens must
be equal except at a near tie, which is a swap of the two best candidates (the golden
token is Rabbit's second best token, which the run document carries as `second_id`)
whose golden margin and Rabbit's margin are both within twice the observed noise; a
small margin with a golden token that is not Rabbit's second best fails. The noise,
relative to the logit scale, has a cap, because a wrong computation raises its own
noise and with it its excuse. The cap is chosen by the `weights` of the run document
(`quantized` when any matrix of the model is quantized, `float` otherwise) and its
`mode` (`plain` or `emulate-llama`) is printed with it. The cap is twice the largest
relative noise of the emulated real quantized files (4.2%): 8.5% rounded up, and for
float files 7e-4. The right synthetic readings are at about 1.1%; the synthetic
`Q4_K_M` file with the split half RoPE (the negative control) is at 10.1% (emulated
10.7%) on the Mac, which is only 1.2 times the cap, so the tests do not rely on that
factor: they assert that it is more than 5 times noisier than the right reading and
that at least two of its first mismatches are not near ties by the yardstick of the
right reading. The cap and the near tie window were chosen on that Mac and are
re-checked by the CI on x86 Linux; if one flips there, re-measure on both before
touching anything, and never loosen a bound to hide an error. The emulated runs are
not equal to the goldens on the synthetic files either, so no strict comparison is
made of them.

With the identity rule, the plain runs of the real files (one run on the Mac,
8 prompts of 32 steps): `F16` equal 8, `Q8_0` equal 7 and one near tie (prompt 6, step
29, golden token 804 is Rabbit's second best, margins 0.028 and 0.074 against twice
the noise 1.22), `Q4_K_M` equal 4 and four near ties (prompts 0, 1, 6 and 7, every
golden token Rabbit's second best, margins 0.05 to 0.32 against twice the noise 3.07);
no prompt fails.

## Rebuilding it

On a Linux x86 host, with the commit of `provenance.json` of llama.cpp built with
`GGML_NATIVE=OFF` and a Python with the packages that its
`requirements/requirements-convert_hf_to_gguf.txt` pins:

```sh
python convert_hf_to_gguf.py MODEL_DIR --outtype f16 --outfile smollm2-135m-f16.gguf
llama-quantize smollm2-135m-f16.gguf smollm2-135m-Q8_0.gguf Q8_0
llama-quantize smollm2-135m-f16.gguf smollm2-135m-Q4_K_M.gguf Q4_K_M
g++ -O2 -std=c++17 -Wno-deprecated-declarations -Ipath/include -Ipath/ggml/include \
    llama_golden.cpp -o llama_golden -Lpath/build/bin -lllama -lggml -lggml-base -lggml-cpu
llama_golden smollm2-135m-Q8_0.gguf 32 smollm2/prompts.txt > golden-Q8_0.json
python dump_reference.py --gguf-py path/gguf-py --out ref smollm2-135m-Q8_0.gguf smollm2-135m-Q4_K_M.gguf
```

The synthetic F32 file is written by the repository and must hash to the value in
`provenance.json` (`rabbit-infer-gen`, the command above) before it is quantized.

## What this does not show

- Not a speed. The golden runs are for tokens.
- Not another processor, another build of llama.cpp, or another thread count: the
  accumulation order of llama.cpp changes with the kernels it selects.
- Not a file converted by the tools of another project, and not an architecture
  other than `llama` with plain attention.
