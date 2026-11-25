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
  tokens can, and only where the margin allows it. Replicating the quantization of
  the activations is a different decision and is not made here.

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
