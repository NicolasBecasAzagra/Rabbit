// SPDX-License-Identifier: GPL-2.0-only
//! A dependency free Llama-style inference runtime.
//!
//! The runtime reads a `safetensors` model and a vocabulary from the
//! initramfs, maps the weights read only instead of copying them, and runs the
//! decoder one token at a time against a key value cache. The dimensions are
//! parameters: a bigger model is a change of shape in the config file and the
//! tensor names, not a change of architecture.
//!
//! ## Layout of this crate
//!
//! * [`json`] is the small scanner the other parsers share. It is written for
//!   the fixed shape of the files this runtime owns, not as a general JSON
//!   library.
//! * [`safetensors`] parses the tensor directory in a `safetensors` header.
//! * [`vocab`] inverts the token string to id map and decodes bytes. It also
//!   recovers the raw bytes of a token, which the grammar needs.
//! * [`bpe`] is the byte pair merge encoder: the GPT-2 pre-tokenizer, the byte
//!   level alphabet and the `merges.txt` of a checkpoint, stored in one buffer
//!   so a real merge table fits a small heap. It is host tested and needs no
//!   file system.
//! * [`config`] reads the handful of values a `safetensors` file cannot carry:
//!   the head count, the norm epsilon, the RoPE theta and the RoPE pair order.
//! * [`math`] holds the pure kernels: RMSNorm, RoPE, softmax and greedy
//!   argmax.
//! * [`grammar`] is the byte level automaton of a tool call. It answers which
//!   bytes may follow a state, whether a state completes the call, and which
//!   tokens of a vocabulary keep the automaton alive. It has no I/O, so it is
//!   host tested.
//! * [`chat`] writes the SmolLM2 chat prompt from plain strings, including the
//!   tool list, and checks the chat markers against a vocabulary.
//! * [`matvec`] is the blocked vector by matrix multiplication over
//!   `bfloat16` weights, with an AVX2 and FMA path, a NEON path, and a scalar
//!   path that all agree to floating point tolerance. It is host tested and
//!   the `rabbit-infer-matmul` binary measures it without a model.
//! * [`quant`] is the block format of a quantized checkpoint: four or eight
//!   bit weights in blocks of thirty-two with one binary16 scale, decoded
//!   inside the matrix vector product. It is host tested and
//!   `docs/QUANTIZATION.md` describes the file.
//! * [`quantize`] rewrites a `safetensors` checkpoint in that format, with the
//!   SHA-256 of the result. It is the code behind the `rabbit-infer-quantize`
//!   tool and behind the gate test.
//! * [`sha256`] is the digest the quantizer prints, so the tools need no
//!   dependency either.
//! * [`cli`] parses the optional token count of the binary, eight by default.
//! * [`paths`] resolves the model root: the initramfs root in the guest, or
//!   `RABBIT_INFER_ROOT` when a native host process sets it.
//! * [`prompt`] encodes a prompt that the caller built, one byte per token,
//!   for the callers that build a conversation at run time.
//! * [`power_off`] asks the machine to go down before the last process
//!   leaves, so mainline Linux does not panic when PID 1 exits.
//! * [`reference`] is the published SmolLM2-135M run the runtime checks
//!   itself against when it serves that checkpoint.
//! * [`gguf`] is the pure reader of the GGUF container: the metadata, the
//!   tensor directory and the exact bytes of a tensor, borrowed from the
//!   input. It is host tested and checks every size.
//! * [`gguf_model`] reads the `llama` model a GGUF file describes: the config
//!   from the metadata and the tensors under the names of the `safetensors`
//!   layout, with named errors for everything it does not read. It is pure and
//!   host tested.
//! * [`synthetic`] writes the synthetic development model: the checkpoint in
//!   `safetensors` and its twin in GGUF, with the rows of the query and key
//!   projections permuted as the `llama.cpp` converter does.
//! * [`gguf_twin`] is the proof that the permutation and the RoPE convention of
//!   the GGUF reader are right: the two twins decode the same tokens, and two
//!   wrong readings of the GGUF file do not.
//! * [`golden_compare`] reads the document that `bench/gguf/llama_golden.cpp`
//!   prints, writes the same document for a run of Rabbit and compares the two
//!   with a stated rule: the tokens must be equal, except at a swap of the two
//!   best candidates at a near tie, which the activation quantization of
//!   `llama.cpp` may flip (the emulation experiment did not confirm it: the
//!   emulated noise was 0.82 and 1.12 against the plain 0.61 and 1.54 on the real
//!   `Q8_0` and `Q4_K_M` files, and a one `ulp` change of the quantizer input
//!   moves the logits by about 0.13). It is pure and host tested, and the
//!   `rabbit-infer-golden` tool is built on it.
//! * [`llama_emulation`] is a diagnostic, off by default: the activations that
//!   the CPU backend of `llama.cpp` multiplies with a quantized weight (packed
//!   to `Q8_0` or `Q8_K` and unpacked) and its `f16` key value cache. It was an
//!   attempt to measure how much of the difference with the oracle is the
//!   oracle's own arithmetic; the conclusion is that the logits of a quantized
//!   model depend on rounding decisions that cannot be reproduced bit for bit.
//! * [`tensor_verify`] hashes the raw bytes of every quantized tensor of a GGUF
//!   file and its dequantised `f32` values, and compares them with the hashes
//!   that `gguf-py` wrote.
//! * `mapping` and `model` are built on every host: the system calls of the
//!   mapping exist on Linux only, and elsewhere the same mapping is made over
//!   bytes in memory, so the decoder and its proofs run on a developer host.
//!   `runtime` and `compare` need the Linux system calls and are only built
//!   for the Linux target. Everything above is host tested.

#![deny(unsafe_op_in_unsafe_fn)]

pub mod bpe;
pub mod chat;
pub mod cli;
pub mod config;
pub mod error;
pub mod gguf;
pub mod gguf_model;
pub mod gguf_twin;
pub mod golden_compare;
pub mod grammar;
pub mod json;
pub mod llama_emulation;
pub mod mapping;
pub mod math;
pub mod matvec;
pub mod model;
pub mod paths;
pub mod power_off;
pub mod prompt;
pub mod quant;
pub mod quantize;
pub mod reference;
pub mod safetensors;
pub mod sha256;
pub mod synthetic;
pub mod tensor_verify;
pub mod vocab;

#[cfg(target_os = "linux")]
pub mod compare;
#[cfg(target_os = "linux")]
pub mod runtime;

#[cfg(all(test, target_os = "linux"))]
mod gate;
