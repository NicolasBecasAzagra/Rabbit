// SPDX-License-Identifier: GPL-2.0-only
//! The decoder a GGUF file of the `llama` architecture describes.
//!
//! [`load`] reads the metadata and the tensor directory of a parsed
//! [`crate::gguf::Gguf`] and answers two things: the [`Config`] of the model and
//! the tensors the decoder binds, under the names of the `safetensors` layout
//! (`model.layers.N.self_attn.q_proj.weight` and so on) with the shape of that
//! layout. The model code then binds a GGUF file exactly as it binds a
//! `safetensors` file. The module is pure: it touches no file and no mapping, so
//! it is tested on every host.
//!
//! This step only reads the model: [`crate::paths::choose_model`] and
//! `Paths.gguf` are not wired into `runtime.rs` yet, and the runtime still opens
//! only the `safetensors` model. That is intended for this step.
//!
//! ## What was verified, and against what
//!
//! Every name and key below was read in these files of `ggml-org/llama.cpp` at
//! commit `79c69f4b06805fe53c91596de6c958f86dbbdf41`:
//!
//! * `gguf-py/gguf/constants.py`, the tensor name templates (`token_embd`,
//!   `output_norm`, `output`, `rope_freqs`, `blk.{bid}.attn_norm`, `attn_q`,
//!   `attn_k`, `attn_v`, `attn_output`, `ffn_norm`, `ffn_gate`, `ffn_up`,
//!   `ffn_down`), the architecture name `llama` and the metadata keys
//!   `general.architecture`, `{arch}.block_count`, `context_length`,
//!   `embedding_length`, `feed_forward_length`, `attention.head_count`,
//!   `attention.head_count_kv`, `attention.layer_norm_rms_epsilon`,
//!   `attention.key_length`, `rope.freq_base`, `rope.dimension_count`,
//!   `rope.scaling.*` and `vocab_size`:
//!   <https://raw.githubusercontent.com/ggml-org/llama.cpp/79c69f4b06805fe53c91596de6c958f86dbbdf41/gguf-py/gguf/constants.py>
//! * `gguf-py/gguf/gguf_writer.py`, the value types the converter writes: the
//!   counts and lengths are `uint32`, the epsilon and the frequency base are
//!   `float32`:
//!   <https://raw.githubusercontent.com/ggml-org/llama.cpp/79c69f4b06805fe53c91596de6c958f86dbbdf41/gguf-py/gguf/gguf_writer.py>
//! * `conversion/llama.py`, the converter. `LlamaModel` has
//!   `undo_permute = True` and permutes the rows of `q_proj` with
//!   `permute(w, n_head, n_head)` and of `k_proj` with `permute(w, n_head,
//!   n_kv_head)`, which is
//!   `w.reshape(n_head, 2, w.shape[0] // n_head // 2, ...).swapaxes(1, 2)
//!   .reshape(w.shape)`. It writes `rope_dimension_count` and `vocab_size`, and
//!   the `rope_freqs` tensor for `rope_type` `llama3`:
//!   <https://raw.githubusercontent.com/ggml-org/llama.cpp/79c69f4b06805fe53c91596de6c958f86dbbdf41/conversion/llama.py>
//! * `conversion/base.py`, `set_gguf_parameters`, which writes the keys above
//!   and the scaling type only for `linear`, `yarn` and `longrope`:
//!   <https://raw.githubusercontent.com/ggml-org/llama.cpp/79c69f4b06805fe53c91596de6c958f86dbbdf41/conversion/base.py>
//! * `src/models/llama.cpp`, the loader and graph of `LLM_ARCH_LLAMA`: the
//!   dimensions of every tensor (`{n_embd, n_vocab}` for the embedding, an
//!   `output` that is optional and falls back to the embedding), the check that
//!   `n_rot` equals the head size, and the optional `rope_freqs`:
//!   <https://raw.githubusercontent.com/ggml-org/llama.cpp/79c69f4b06805fe53c91596de6c958f86dbbdf41/src/models/llama.cpp>
//! * `src/llama-model.cpp`, which returns `LLAMA_ROPE_TYPE_NORM` for this
//!   architecture, which pairs adjacent elements, and reads the optional keys
//!   with their defaults (`rope.freq_base` 10000, scaling `linear` with no
//!   factor):
//!   <https://raw.githubusercontent.com/ggml-org/llama.cpp/79c69f4b06805fe53c91596de6c958f86dbbdf41/src/llama-model.cpp>
//!
//! ## The RoPE convention and the permutation
//!
//! A Hugging Face checkpoint rotates the pair `(i, i + d/2)` of a head, the
//! split half convention. The converter permutes the rows of `q_proj` and
//! `k_proj` of every head so that the element `i + d/2` lands at `2i + 1`, and
//! `llama.cpp` rotates adjacent pairs `(2i, 2i + 1)` of the permuted rows. The
//! same pairs of the original dimensions are rotated by the same angles, so a
//! GGUF `llama` model runs with `rope_interleaved = true` and with no
//! un-permutation: the model code is given the permuted rows as they are.
//!
//! ## What this commit reads
//!
//! Only the architecture `llama`, with `F32` and `F16` tensors and, for the
//! matrices, `Q8_0`. A key or a tensor the converter writes for a feature this
//! runtime lacks is a named error and never silently ignored: a `rope_freqs`
//! tensor, a RoPE scaling, a bias, a fused `attn_qkv`, the experts of a mixture,
//! a quantized tensor of another type (`Q4_0`, `Q5_0`, `Q4_K`, `Q6_K`... are
//! refused by name even where [`crate::gguf`] knows their sizes) and a quantized
//! norm. The dimensions of every tensor must agree with the config
//! ([`GgufModelError::ShapeMismatch`]).
//!
//! ## `Q8_0` is the `Q8B32` block
//!
//! `ggml-common.h` defines `block_q8_0 { ggml_half d; int8_t qs[QK8_0]; }` with
//! `QK8_0` = 32 and a `static_assert` that the size is `sizeof(ggml_half) + QK8_0`,
//! 34 bytes: a binary16 scale, then 32 signed bytes. The blocks of a tensor
//! follow each other along `ne[0]`, the fastest dimension, so a row of a matrix
//! (`ne[0]` weights) is `ne[0] / 32` consecutive blocks, and `ne[0]` must be a
//! multiple of 32. `dequantize_row_q8_0` of `ggml-quants.c` is
//! `y[i * 32 + j] = x[i].qs[j] * d` with `d` the scale converted to `f32`:
//!
//! * <https://raw.githubusercontent.com/ggml-org/ggml/d9fe7bf/src/ggml-common.h>
//! * <https://raw.githubusercontent.com/ggml-org/ggml/d9fe7bf/src/ggml-quants.c>
//!
//! That is, byte for byte, the block of [`crate::quant::Q8_BLOCK_BYTES`] bytes
//! that [`crate::quant::dequantize_block_q8`] reads, so a `Q8_0` tensor of the
//! file is mapped where it lies onto the `Q8B32` matrix of the model (the model
//! sees the same `[out, in]` shape as for a `safetensors` file): the products go
//! through [`crate::matvec::matvec_q8`] on every path and the embedding row is
//! decoded by [`crate::quant::dequantize_row_q8`], with no new kernel and no copy.
//! The quantizer is not the same function, which only matters to someone who
//! compares files: `quantize_row_q8_0_ref` takes `d = amax / 127` and multiplies
//! by `1 / d` of the unrounded `d`, and [`crate::quant::quantize_block_q8`]
//! multiplies by the inverse of the scale after it was rounded to binary16
//! (see the test of [`crate::synthetic`]). Both store the same scale and some
//! quants of a file made by one differ by one from those of the other.
//!
//! ## The metadata fails closed
//!
//! `load` reads a handful of keys, and a key it does not read could still change
//! what the model computes. Every `llama.*` key is therefore either on
//! [`ACCEPTED_KEYS`] or refused with [`GgufModelError::UnsupportedMetadata`]. The
//! list was built from `LlamaModel.set_gguf_parameters` of `conversion/llama.py`
//! and `TextModel.set_gguf_parameters` of `conversion/base.py`, which write for a
//! plain `llama` model (SmolLM2, LLaMA, Mistral): `block_count`,
//! `context_length`, `embedding_length`, `feed_forward_length`,
//! `attention.head_count`, `attention.head_count_kv`,
//! `attention.layer_norm_rms_epsilon`, `rope.freq_base`,
//! `rope.dimension_count` and `vocab_size`, `attention.key_length` and
//! `attention.value_length` when the config has a `head_dim`, and
//! `attention.layer_norm_epsilon` when it has a `layer_norm_eps`. The keys the
//! converter writes only for a feature this runtime lacks are refused:
//! `attention.causal` (only written as false), `expert_count`,
//! `expert_used_count` and the group keys of a mixture of experts, and the
//! `rope.scaling.*` keys, which keep [`GgufModelError::UnsupportedRopeScaling`]
//! together with the old `rope.scale_linear`.
//!
//! Each accepted key was checked against `src/llama-model.cpp` and
//! `src/models/llama.cpp`: the first twelve of the list are read by `llama.cpp`
//! for this architecture (`vocab_size` only to name the size of the model) and
//! are read here (`rope.dimension_count`, `key_length`, `value_length` and
//! `vocab_size` must agree with the head size and the embedding, as `llama.cpp`
//! requires); `attention.layer_norm_epsilon` is
//! written by the converter but `llama.cpp` never reads it for `llama`, whose
//! graph normalizes with `LLM_NORM_RMS` and `layer_norm_rms_epsilon` only, so it
//! changes nothing. Every other key `llama.cpp` can read under the `llama.`
//! prefix (`logit_scale`, `attn_logit_softcapping`, `final_logit_softcapping`,
//! `attention.clamp_kqv`, `attention.max_alibi_bias`, `attention.scale`,
//! `attention.sliding_window`, `use_parallel_residual`, `embedding_length_out`,
//! `pooling_type`, `rope.dimension_count_swa`, `rope.dimension_sections`...) is
//! refused, even where `llama.cpp` itself would ignore it for this architecture:
//! a file that carries one was not made for a plain `llama` decoder.
//!
//! Outside the `llama.` prefix: `tokenizer.*` is not read except for the end of
//! sequence token. Of `general.*`, `name`, `type`, `quantization_version`,
//! `file_type` (read only to print the description of the model, see
//! `llama_model_loader`) and the authorship keys are informational, and
//! `alignment` is applied by the container parser. Two namespaces do change the
//! computation and are refused: `general.tensor_extra.*`, the activation
//! precision policy that `llama_prec_policy::load` applies, and `prism.hadamard.*`,
//! which says that the weights were folded with a Hadamard transform.
//!
//! Choices the format leaves open:
//!
//! * `llama.attention.head_count_kv` is optional and defaults to the head
//!   count, as `llama.cpp` does. So is `llama.rope.freq_base`, which defaults
//!   to 10000 as there. The other keys of the list above are required.
//! * `llama.rope.dimension_count`, `llama.attention.key_length` and
//!   `llama.attention.value_length`, when present, must equal the head size `embedding_length / head_count`, as
//!   `llama.cpp` requires for this architecture.
//! * The end of sequence token is `tokenizer.ggml.eos_token_id` when present
//!   and 0 otherwise; the prompt is never taken from the file.
//! * The output projection is the embedding when `output.weight` is absent.

use core::fmt;
use std::collections::HashSet;

use crate::config::{Config, Field};
use crate::gguf::{Gguf, GgufError, TensorKind, Value};

/// The architecture this module reads, `MODEL_ARCH.LLAMA` of `constants.py`.
pub const ARCHITECTURE: &str = "llama";

const KEY_ARCHITECTURE: &str = "general.architecture";
const KEY_BLOCK_COUNT: &str = "llama.block_count";
const KEY_CONTEXT_LENGTH: &str = "llama.context_length";
const KEY_EMBEDDING_LENGTH: &str = "llama.embedding_length";
const KEY_FEED_FORWARD_LENGTH: &str = "llama.feed_forward_length";
const KEY_HEAD_COUNT: &str = "llama.attention.head_count";
const KEY_HEAD_COUNT_KV: &str = "llama.attention.head_count_kv";
const KEY_RMS_EPSILON: &str = "llama.attention.layer_norm_rms_epsilon";
const KEY_KEY_LENGTH: &str = "llama.attention.key_length";
const KEY_VALUE_LENGTH: &str = "llama.attention.value_length";
const KEY_LAYER_NORM_EPSILON: &str = "llama.attention.layer_norm_epsilon";
const KEY_ROPE_SCALE_LINEAR: &str = "llama.rope.scale_linear";
const KEY_ROPE_FREQ_BASE: &str = "llama.rope.freq_base";
const KEY_ROPE_DIMENSION_COUNT: &str = "llama.rope.dimension_count";
const KEY_VOCAB_SIZE: &str = "llama.vocab_size";
const KEY_EOS_TOKEN: &str = "tokenizer.ggml.eos_token_id";
const ROPE_SCALING_PREFIX: &str = "llama.rope.scaling.";
const KEY_ROPE_SCALING_TYPE: &str = "llama.rope.scaling.type";
const KEY_ROPE_SCALING_FACTOR: &str = "llama.rope.scaling.factor";

/// Every `llama.*` key that is not a RoPE scaling and does not make the file
/// unsupported: the keys `load` reads and checks, and `attention.layer_norm_epsilon`,
/// which the converter writes and `llama.cpp` never reads for `llama`.
pub const ACCEPTED_KEYS: [&str; 13] = [
    KEY_BLOCK_COUNT,
    KEY_CONTEXT_LENGTH,
    KEY_EMBEDDING_LENGTH,
    KEY_FEED_FORWARD_LENGTH,
    KEY_HEAD_COUNT,
    KEY_HEAD_COUNT_KV,
    KEY_RMS_EPSILON,
    KEY_ROPE_FREQ_BASE,
    KEY_ROPE_DIMENSION_COUNT,
    KEY_KEY_LENGTH,
    KEY_VALUE_LENGTH,
    KEY_VOCAB_SIZE,
    KEY_LAYER_NORM_EPSILON,
];

/// The architecture prefix of the keys that are checked against [`ACCEPTED_KEYS`].
const ARCHITECTURE_PREFIX: &str = "llama.";

/// Namespaces outside `llama.` whose keys change the computation: the activation
/// precision policy of `llama_prec_policy::load` and the folded Hadamard weights
/// of `load_hparams_hadamard`.
const REFUSED_PREFIXES: [&str; 2] = ["general.tensor_extra.", "prism.hadamard."];

/// `LLAMA_DEFAULT_ROPE_FREQ_BASE`: the base when the file has none.
const DEFAULT_ROPE_FREQ_BASE: f32 = 10_000.0;

/// The tensor of the RoPE frequency factors, `MODEL_TENSOR.ROPE_FREQS`.
const ROPE_FREQS_TENSOR: &str = "rope_freqs.weight";

/// Why a GGUF file is not a model this runtime can run.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufModelError {
    /// The container is malformed.
    Container(GgufError),
    /// `general.architecture` is not `llama`.
    UnsupportedArchitecture(String),
    /// A required metadata key is absent.
    MissingKey(String),
    /// A metadata key has another type than the converter writes.
    WrongType { key: String, expected: &'static str },
    /// The file asks for a RoPE scaling; the text names the key and its value.
    UnsupportedRopeScaling(String),
    /// A metadata key that is not on [`ACCEPTED_KEYS`] and could change what the
    /// model computes.
    UnsupportedMetadata { key: String },
    /// The file has a `rope_freqs.weight` tensor, the factors of a RoPE
    /// scaling such as `llama3`.
    UnsupportedRopeFreqs,
    /// A tensor of a type this decoder does not read: other than `F32` and `F16`
    /// and, for a matrix, `Q8_0`.
    UnsupportedTensorType { name: String, ggml_type: u32 },
    /// A one dimensional tensor stored as `Q8_0`, a quantized norm or vector:
    /// a norm stays floating point.
    QuantizedNorm { name: String },
    /// A `Q8_0` tensor of more than two dimensions, which has no layout here.
    UnsupportedRank { name: String, rank: usize },
    /// A tensor the decoder does not use: a bias, a fused projection, an
    /// expert, an extra block.
    UnexpectedTensor(String),
    /// A tensor the decoder needs is absent.
    MissingTensor(String),
    /// A tensor, or a key about its size, disagrees with the config.
    ShapeMismatch { name: String },
    /// The head size cannot be rotated, or a key disagrees with it.
    BadHeadSize { key: String },
    /// The values describe no model; the text is the reason.
    BadConfig(String),
}

impl From<GgufError> for GgufModelError {
    fn from(error: GgufError) -> Self {
        Self::Container(error)
    }
}

impl fmt::Display for GgufModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Container(error) => write!(formatter, "gguf: {error}"),
            Self::UnsupportedArchitecture(name) => write!(
                formatter,
                "gguf: unsupported architecture `{name}`, only `{ARCHITECTURE}` is read"
            ),
            Self::MissingKey(key) => write!(formatter, "gguf: missing metadata key `{key}`"),
            Self::WrongType { key, expected } => {
                write!(
                    formatter,
                    "gguf: metadata key `{key}` is not of type {expected}"
                )
            }
            Self::UnsupportedRopeScaling(what) => {
                write!(formatter, "gguf: unsupported RoPE scaling ({what})")
            }
            Self::UnsupportedMetadata { key } => write!(
                formatter,
                "gguf: metadata key `{key}` is not read by this decoder and could change the \
                 model"
            ),
            Self::UnsupportedRopeFreqs => write!(
                formatter,
                "gguf: the file has a `{ROPE_FREQS_TENSOR}` tensor, a RoPE scaling this runtime \
                 does not apply"
            ),
            Self::UnsupportedTensorType { name, ggml_type } => {
                match crate::gguf::ggml_type_name(*ggml_type) {
                    Some(type_name) => write!(
                        formatter,
                        "gguf: tensor `{name}` has ggml type {type_name} ({ggml_type}), only F32, \
                         F16 and (for a matrix) Q8_0 are read"
                    ),
                    None => write!(
                        formatter,
                        "gguf: tensor `{name}` has unknown ggml type {ggml_type}, only F32, F16 \
                         and (for a matrix) Q8_0 are read"
                    ),
                }
            }
            Self::QuantizedNorm { name } => write!(
                formatter,
                "gguf: tensor `{name}` is a one dimensional Q8_0 tensor, a quantized norm or \
                 vector: a norm is read as F32 or F16"
            ),
            Self::UnsupportedRank { name, rank } => write!(
                formatter,
                "gguf: tensor `{name}` has ggml type Q8_0 and {rank} dimensions, only a matrix of \
                 2 is read"
            ),
            Self::UnexpectedTensor(name) => {
                write!(
                    formatter,
                    "gguf: tensor `{name}` is not used by this decoder"
                )
            }
            Self::MissingTensor(name) => write!(formatter, "gguf: missing tensor `{name}`"),
            Self::ShapeMismatch { name } => {
                write!(
                    formatter,
                    "gguf: `{name}` has a shape that disagrees with the config"
                )
            }
            Self::BadHeadSize { key } => write!(
                formatter,
                "gguf: `{key}` disagrees with the head size, or the head size is odd"
            ),
            Self::BadConfig(reason) => write!(formatter, "gguf: {reason}"),
        }
    }
}

impl std::error::Error for GgufModelError {}

/// One tensor the decoder binds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensor {
    /// The name in the `safetensors` layout, which the model binds by.
    pub name: String,
    /// The name in the file.
    pub gguf_name: String,
    /// `F32` or `F16`, or `Q8_0` for a matrix.
    pub kind: TensorKind,
    /// The shape in the `safetensors` layout: `[out, in]` for a matrix, which
    /// is the dimensions of the file in the opposite order.
    pub shape: Vec<usize>,
    /// First byte, relative to the start of the data section.
    pub start: usize,
    /// Bytes the tensor takes.
    pub byte_len: usize,
}

/// A model read from the directory of a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufModel {
    /// The configuration, with `rope_interleaved` set: the file stores the
    /// rows in the order `llama.cpp` rotates.
    pub config: Config,
    /// The tensors in the order the decoder binds them.
    pub tensors: Vec<GgufTensor>,
    /// Where the data section starts, in bytes from the start of the file.
    pub data_offset: usize,
}

/// The value of a key that must exist, as a `usize`.
fn required_usize(gguf: &Gguf<'_>, key: &str) -> Result<usize, GgufModelError> {
    let value = gguf.get_u32(key).map_err(key_error)?;
    Ok(value as usize)
}

/// The value of a key that may be absent.
fn optional_u32(gguf: &Gguf<'_>, key: &str) -> Result<Option<u32>, GgufModelError> {
    if gguf.get(key).is_none() {
        return Ok(None);
    }
    gguf.get_u32(key).map(Some).map_err(key_error)
}

/// The container error of a lookup, as the error of this module.
fn key_error(error: GgufError) -> GgufModelError {
    match error {
        GgufError::MissingKey(key) => GgufModelError::MissingKey(key),
        GgufError::WrongType { key, expected } => GgufModelError::WrongType { key, expected },
        other => GgufModelError::Container(other),
    }
}

/// Refuse every `llama.rope.scaling.*` key that does anything, and the old
/// `llama.rope.scale_linear`.
///
/// `llama.cpp` defaults to a linear scaling whose factor is 1, which is no
/// scaling. A type `none` and a factor 1 (under either key) are therefore the
/// same as absent, and any other key under the prefix (a linear factor, YaRN, an
/// original context length) changes the angles of the rotation, which this
/// runtime does not.
fn check_rope_scaling(gguf: &Gguf<'_>) -> Result<(), GgufModelError> {
    for (key, value) in gguf.metadata() {
        if !key.starts_with(ROPE_SCALING_PREFIX) && *key != KEY_ROPE_SCALE_LINEAR {
            continue;
        }
        let neutral = match (*key, value) {
            (KEY_ROPE_SCALING_TYPE, Value::Str("none")) => true,
            (KEY_ROPE_SCALING_FACTOR | KEY_ROPE_SCALE_LINEAR, Value::F32(factor)) => *factor == 1.0,
            _ => false,
        };
        if !neutral {
            return Err(GgufModelError::UnsupportedRopeScaling(format!(
                "`{key}` = {value:?}"
            )));
        }
    }
    Ok(())
}

/// Refuse every key that could change the model and that `load` does not read:
/// a `llama.*` key off [`ACCEPTED_KEYS`] and the keys of [`REFUSED_PREFIXES`].
/// The RoPE scaling keys are answered by [`check_rope_scaling`], which runs
/// first and lets only the neutral ones through.
fn check_metadata(gguf: &Gguf<'_>) -> Result<(), GgufModelError> {
    for (key, _) in gguf.metadata() {
        if key.starts_with(ROPE_SCALING_PREFIX) || *key == KEY_ROPE_SCALE_LINEAR {
            continue;
        }
        let refused = if key.starts_with(ARCHITECTURE_PREFIX) {
            !ACCEPTED_KEYS.contains(key)
        } else {
            REFUSED_PREFIXES
                .iter()
                .any(|prefix| key.starts_with(prefix))
        };
        if refused {
            return Err(GgufModelError::UnsupportedMetadata {
                key: (*key).to_string(),
            });
        }
    }
    Ok(())
}

/// What the decoder needs from the file, before the file is looked at.
struct Expected {
    gguf: String,
    name: String,
    /// The dimensions in the order of the file.
    dims: Vec<u64>,
}

/// Read the model out of a parsed GGUF file.
pub fn load(gguf: &Gguf<'_>) -> Result<GgufModel, GgufModelError> {
    let architecture = gguf.get_str(KEY_ARCHITECTURE).map_err(key_error)?;
    if architecture != ARCHITECTURE {
        return Err(GgufModelError::UnsupportedArchitecture(
            architecture.to_string(),
        ));
    }

    let layers = required_usize(gguf, KEY_BLOCK_COUNT)?;
    let _context = required_usize(gguf, KEY_CONTEXT_LENGTH)?;
    let hidden = required_usize(gguf, KEY_EMBEDDING_LENGTH)?;
    let ffn = required_usize(gguf, KEY_FEED_FORWARD_LENGTH)?;
    let heads = required_usize(gguf, KEY_HEAD_COUNT)?;
    let epsilon = gguf.get_f32(KEY_RMS_EPSILON).map_err(key_error)?;
    let kv_heads = optional_u32(gguf, KEY_HEAD_COUNT_KV)?.map(|count| count as usize);
    let theta = match gguf.get(KEY_ROPE_FREQ_BASE) {
        None => None,
        Some(_) => Some(gguf.get_f32(KEY_ROPE_FREQ_BASE).map_err(key_error)?),
    };
    let eos = optional_u32(gguf, KEY_EOS_TOKEN)?;
    let rope_dimension = optional_u32(gguf, KEY_ROPE_DIMENSION_COUNT)?;
    let key_length = optional_u32(gguf, KEY_KEY_LENGTH)?;
    let vocab_key = optional_u32(gguf, KEY_VOCAB_SIZE)?;
    let value_length = optional_u32(gguf, KEY_VALUE_LENGTH)?;
    check_rope_scaling(gguf)?;
    check_metadata(gguf)?;

    if gguf.tensor(ROPE_FREQS_TENSOR).is_some() {
        return Err(GgufModelError::UnsupportedRopeFreqs);
    }

    if heads == 0 || hidden == 0 || !hidden.is_multiple_of(heads) {
        return Err(GgufModelError::BadConfig(format!(
            "the embedding length {hidden} is not a multiple of the head count {heads}"
        )));
    }
    let head_dim = hidden / heads;
    if !head_dim.is_multiple_of(2) {
        return Err(GgufModelError::BadHeadSize {
            key: KEY_EMBEDDING_LENGTH.to_string(),
        });
    }
    for (key, value) in [
        (KEY_ROPE_DIMENSION_COUNT, rope_dimension),
        (KEY_KEY_LENGTH, key_length),
        (KEY_VALUE_LENGTH, value_length),
    ] {
        if value.is_some_and(|value| value as usize != head_dim) {
            return Err(GgufModelError::BadHeadSize {
                key: key.to_string(),
            });
        }
    }
    let kv_count = kv_heads.unwrap_or(heads);
    let kv_dim = kv_count
        .checked_mul(head_dim)
        .ok_or_else(|| GgufModelError::BadConfig("the key value width overflows".to_string()))?;

    // The vocabulary is the number of rows of the embedding, which has to say
    // the same as `llama.vocab_size` when the file carries it.
    let embedding = gguf
        .tensor("token_embd.weight")
        .ok_or_else(|| GgufModelError::MissingTensor("token_embd.weight".to_string()))?;
    let mismatch = |name: &str| GgufModelError::ShapeMismatch {
        name: name.to_string(),
    };
    if embedding.dims.len() != 2 || embedding.dims[0] != hidden as u64 {
        return Err(mismatch("token_embd.weight"));
    }
    let vocab = usize::try_from(embedding.dims[1]).map_err(|_| mismatch("token_embd.weight"))?;
    if vocab_key.is_some_and(|value| value as usize != vocab) {
        return Err(mismatch("token_embd.weight"));
    }

    let h = hidden as u64;
    let f = ffn as u64;
    let kv = kv_dim as u64;
    let mut expected = vec![Expected {
        gguf: "token_embd.weight".to_string(),
        name: "model.embed_tokens.weight".to_string(),
        dims: vec![h, vocab as u64],
    }];
    for layer in 0..layers {
        let per_layer: [(&str, &str, Vec<u64>); 9] = [
            ("attn_norm", "input_layernorm", vec![h]),
            ("attn_q", "self_attn.q_proj", vec![h, h]),
            ("attn_k", "self_attn.k_proj", vec![h, kv]),
            ("attn_v", "self_attn.v_proj", vec![h, kv]),
            ("attn_output", "self_attn.o_proj", vec![h, h]),
            ("ffn_norm", "post_attention_layernorm", vec![h]),
            ("ffn_gate", "mlp.gate_proj", vec![h, f]),
            ("ffn_up", "mlp.up_proj", vec![h, f]),
            ("ffn_down", "mlp.down_proj", vec![f, h]),
        ];
        for (gguf_part, hf_part, dims) in per_layer {
            expected.push(Expected {
                gguf: format!("blk.{layer}.{gguf_part}.weight"),
                name: format!("model.layers.{layer}.{hf_part}.weight"),
                dims,
            });
        }
        // A file that names more blocks than it has tensors for ends at the
        // first block it lacks, before the loop gets long.
        let last = expected.last().map(|entry| entry.gguf.as_str());
        if last.is_some_and(|name| gguf.tensor(name).is_none()) {
            break;
        }
    }
    expected.push(Expected {
        gguf: "output_norm.weight".to_string(),
        name: "model.norm.weight".to_string(),
        dims: vec![h],
    });
    let tied = gguf.tensor("output.weight").is_none();
    if !tied {
        expected.push(Expected {
            gguf: "output.weight".to_string(),
            name: "lm_head.weight".to_string(),
            dims: vec![h, vocab as u64],
        });
    }

    let mut tensors = Vec::with_capacity(expected.len());
    let mut known: HashSet<&str> = HashSet::new();
    for entry in &expected {
        let info = gguf
            .tensor(&entry.gguf)
            .ok_or_else(|| GgufModelError::MissingTensor(entry.gguf.clone()))?;
        let kind = match info.kind() {
            Some(kind @ (TensorKind::F32 | TensorKind::F16)) => kind,
            // A matrix is a sequence of rows of whole blocks, which the parser
            // checks (`ne[0]` a multiple of 32) when it sizes the tensor below.
            Some(TensorKind::Q8_0) if info.dims.len() == 2 && entry.dims.len() == 2 => {
                TensorKind::Q8_0
            }
            // A `Q8_0` tensor of another rank is not a matrix: a vector is
            // a quantized norm, and a tensor above rank 2 has no layout here.
            Some(TensorKind::Q8_0) if info.dims.len() == 1 => {
                return Err(GgufModelError::QuantizedNorm {
                    name: entry.gguf.clone(),
                })
            }
            Some(TensorKind::Q8_0) if info.dims.len() > 2 => {
                return Err(GgufModelError::UnsupportedRank {
                    name: entry.gguf.clone(),
                    rank: info.dims.len(),
                })
            }
            Some(TensorKind::Q8_0) => return Err(mismatch(&entry.gguf)),
            _ => {
                return Err(GgufModelError::UnsupportedTensorType {
                    name: entry.gguf.clone(),
                    ggml_type: info.ggml_type,
                })
            }
        };
        if info.dims != entry.dims {
            return Err(mismatch(&entry.gguf));
        }
        let byte_len = usize::try_from(info.byte_size()?).map_err(|_| GgufError::TooLarge)?;
        let start = usize::try_from(info.offset).map_err(|_| GgufError::TooLarge)?;
        tensors.push(GgufTensor {
            name: entry.name.clone(),
            gguf_name: entry.gguf.clone(),
            kind,
            shape: entry.dims.iter().rev().map(|&dim| dim as usize).collect(),
            start,
            byte_len,
        });
        known.insert(entry.gguf.as_str());
    }
    for info in gguf.tensors() {
        if !known.contains(info.name) {
            return Err(GgufModelError::UnexpectedTensor(info.name.to_string()));
        }
    }

    let mut config = Config::defaults();
    config.hidden_size = hidden;
    config.num_hidden_layers = layers;
    config.num_attention_heads = heads;
    config.num_key_value_heads = kv_count;
    config.intermediate_size = ffn;
    config.rms_norm_eps = epsilon;
    config.rope_theta = theta.unwrap_or(DEFAULT_ROPE_FREQ_BASE);
    config.vocab_size = vocab;
    config.tie_word_embeddings = tied;
    config.eos_token_id = eos.unwrap_or(0);
    config.rope_interleaved = true;
    for field in [
        Field::HiddenSize,
        Field::NumHiddenLayers,
        Field::NumAttentionHeads,
        Field::NumKeyValueHeads,
        Field::IntermediateSize,
        Field::RmsNormEps,
        Field::RopeTheta,
        Field::VocabSize,
        Field::TieWordEmbeddings,
        Field::RopeInterleaved,
    ] {
        config.mark_present(field);
    }
    if eos.is_some() {
        config.mark_present(Field::EosTokenId);
    }
    let config = config
        .validate()
        .map_err(|error| GgufModelError::BadConfig(error.to_string()))?;

    Ok(GgufModel {
        config,
        tensors,
        data_offset: gguf.data_offset(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelFile;
    use crate::synthetic::{
        gguf_builder, weights, GgufBuilder, GgufTensorOut, Meta, Permutation, Spec, Storage,
    };

    /// Four heads over two key value heads: grouped query attention.
    fn spec() -> Spec {
        Spec {
            hidden: 16,
            layers: 2,
            heads: 4,
            kv_heads: 2,
            ffn: 8,
            vocab: 16,
            eps: 1e-5,
            theta: 500_000.0,
        }
    }

    fn builder() -> GgufBuilder {
        let spec = spec();
        gguf_builder(&spec, &weights(&spec), Permutation::Converter, Storage::F32)
    }

    fn load_builder(builder: &GgufBuilder) -> Result<GgufModel, GgufModelError> {
        let bytes = builder.to_bytes();
        let gguf = Gguf::parse(&bytes)?;
        load(&gguf)
    }

    fn zeros(bytes: usize) -> Vec<u8> {
        vec![0; bytes]
    }

    #[test]
    fn reads_the_config_and_binds_the_names_of_the_safetensors_layout() {
        let model = load_builder(&builder()).unwrap();
        let config = &model.config;
        assert_eq!(config.hidden_size, 16);
        assert_eq!(config.num_hidden_layers, 2);
        assert_eq!(config.num_attention_heads, 4);
        assert_eq!(config.num_key_value_heads, 2);
        assert_eq!(config.intermediate_size, 8);
        assert_eq!(config.vocab_size, 16);
        assert_eq!(config.rope_theta, 500_000.0);
        assert_eq!(config.rms_norm_eps, 1e-5);
        assert!(config.rope_interleaved);
        assert!(config.tie_word_embeddings);
        assert_eq!(config.eos_token_id, 0);
        assert_eq!(model.tensors.len(), 2 + 2 * 9);
        let by_name = |name: &str| model.tensors.iter().find(|t| t.name == name).unwrap();
        // A matrix is [out, in] there and [in, out] in the file.
        assert_eq!(by_name("model.embed_tokens.weight").shape, [16, 16]);
        assert_eq!(
            by_name("model.layers.1.self_attn.k_proj.weight").shape,
            [8, 16]
        );
        assert_eq!(
            by_name("model.layers.1.mlp.down_proj.weight").shape,
            [16, 8]
        );
        assert_eq!(by_name("model.norm.weight").shape, [16]);
        assert_eq!(
            by_name("model.layers.0.self_attn.q_proj.weight").gguf_name,
            "blk.0.attn_q.weight"
        );
        assert!(model.tensors.iter().all(|t| t.kind == TensorKind::F32));
        assert_eq!(model.data_offset % 32, 0);
    }

    #[test]
    fn a_separate_output_matrix_unties_the_embedding() {
        let mut file = builder();
        file.tensors.push(GgufTensorOut {
            name: "output.weight".into(),
            dims: vec![16, 16],
            ggml_type: 0,
            data: zeros(16 * 16 * 4),
        });
        let model = load_builder(&file).unwrap();
        assert!(!model.config.tie_word_embeddings);
        assert!(model.tensors.iter().any(|t| t.name == "lm_head.weight"));
        file.tensor("output.weight").unwrap().dims = vec![16, 15];
        assert_eq!(
            load_builder(&file).unwrap_err(),
            GgufModelError::ShapeMismatch {
                name: "output.weight".into()
            }
        );
    }

    #[test]
    fn f16_matrices_are_read() {
        let spec = spec();
        let file = gguf_builder(&spec, &weights(&spec), Permutation::Converter, Storage::F16);
        let model = load_builder(&file).unwrap();
        let kind = |name: &str| model.tensors.iter().find(|t| t.name == name).unwrap().kind;
        assert_eq!(
            kind("model.layers.0.self_attn.q_proj.weight"),
            TensorKind::F16
        );
        assert_eq!(
            kind("model.layers.0.input_layernorm.weight"),
            TensorKind::F32
        );
    }

    #[test]
    fn another_architecture_is_refused_by_name() {
        let mut file = builder();
        *file.meta("general.architecture").unwrap() = Meta::Str("qwen2".into());
        let error = load_builder(&file).unwrap_err();
        assert_eq!(
            error,
            GgufModelError::UnsupportedArchitecture("qwen2".into())
        );
        assert!(error.to_string().contains("qwen2"));
        file.remove_meta("general.architecture");
        assert_eq!(
            load_builder(&file).unwrap_err(),
            GgufModelError::MissingKey("general.architecture".into())
        );
    }

    #[test]
    fn every_required_key_is_named_when_it_is_missing() {
        for key in [
            KEY_BLOCK_COUNT,
            KEY_CONTEXT_LENGTH,
            KEY_EMBEDDING_LENGTH,
            KEY_FEED_FORWARD_LENGTH,
            KEY_HEAD_COUNT,
            KEY_RMS_EPSILON,
        ] {
            let mut file = builder();
            file.remove_meta(key);
            assert_eq!(
                load_builder(&file).unwrap_err(),
                GgufModelError::MissingKey(key.into()),
                "{key}"
            );
        }
    }

    #[test]
    fn a_key_of_another_type_is_named() {
        let mut file = builder();
        *file.meta(KEY_BLOCK_COUNT).unwrap() = Meta::Str("two".into());
        assert_eq!(
            load_builder(&file).unwrap_err(),
            GgufModelError::WrongType {
                key: KEY_BLOCK_COUNT.into(),
                expected: "u32"
            }
        );
        let mut file = builder();
        *file.meta(KEY_RMS_EPSILON).unwrap() = Meta::U32(1);
        assert!(matches!(
            load_builder(&file).unwrap_err(),
            GgufModelError::WrongType { .. }
        ));
    }

    #[test]
    fn the_optional_keys_take_the_defaults_of_llama_cpp() {
        let mut file = builder();
        file.remove_meta(KEY_HEAD_COUNT_KV);
        file.remove_meta(KEY_ROPE_FREQ_BASE);
        file.remove_meta(KEY_ROPE_DIMENSION_COUNT);
        file.remove_meta(KEY_VOCAB_SIZE);
        // Without a key value head count the heads are not grouped, and the
        // key and value matrices of the file no longer fit.
        assert_eq!(
            load_builder(&file).unwrap_err(),
            GgufModelError::ShapeMismatch {
                name: "blk.0.attn_k.weight".into()
            }
        );
        *file.meta(KEY_HEAD_COUNT).unwrap() = Meta::U32(2);
        file.tensor("blk.0.attn_k.weight").unwrap().dims = vec![16, 16];
        file.tensor("blk.0.attn_v.weight").unwrap().dims = vec![16, 16];
        file.tensor("blk.1.attn_k.weight").unwrap().dims = vec![16, 16];
        file.tensor("blk.1.attn_v.weight").unwrap().dims = vec![16, 16];
        for name in [
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.1.attn_k.weight",
            "blk.1.attn_v.weight",
        ] {
            file.tensor(name).unwrap().data = zeros(16 * 16 * 4);
        }
        let model = load_builder(&file).unwrap();
        assert_eq!(model.config.num_key_value_heads, 2);
        assert_eq!(model.config.rope_theta, 10_000.0);
    }

    #[test]
    fn the_end_of_sequence_token_comes_from_the_tokenizer_key_and_must_exist() {
        let mut file = builder();
        file.metadata
            .push((KEY_EOS_TOKEN.to_string(), Meta::U32(3)));
        assert_eq!(load_builder(&file).unwrap().config.eos_token_id, 3);
        *file.meta(KEY_EOS_TOKEN).unwrap() = Meta::U32(16);
        assert!(matches!(
            load_builder(&file).unwrap_err(),
            GgufModelError::BadConfig(_)
        ));
    }

    #[test]
    fn a_rope_scaling_is_refused_and_a_neutral_one_is_not() {
        let mut file = builder();
        file.metadata
            .push((KEY_ROPE_SCALING_TYPE.to_string(), Meta::Str("none".into())));
        file.metadata
            .push((KEY_ROPE_SCALING_FACTOR.to_string(), Meta::F32(1.0)));
        assert!(load_builder(&file).is_ok());
        *file.meta(KEY_ROPE_SCALING_TYPE).unwrap() = Meta::Str("linear".into());
        let error = load_builder(&file).unwrap_err();
        assert!(matches!(error, GgufModelError::UnsupportedRopeScaling(_)));
        assert!(error.to_string().contains("llama.rope.scaling.type"));
        *file.meta(KEY_ROPE_SCALING_TYPE).unwrap() = Meta::Str("none".into());
        *file.meta(KEY_ROPE_SCALING_FACTOR).unwrap() = Meta::F32(4.0);
        assert!(matches!(
            load_builder(&file).unwrap_err(),
            GgufModelError::UnsupportedRopeScaling(_)
        ));
        let mut file = builder();
        file.metadata.push((
            "llama.rope.scaling.original_context_length".to_string(),
            Meta::U32(4096),
        ));
        assert!(matches!(
            load_builder(&file).unwrap_err(),
            GgufModelError::UnsupportedRopeScaling(_)
        ));
    }

    #[test]
    fn the_old_linear_scale_key_is_a_rope_scaling() {
        let mut file = builder();
        file.metadata
            .push((KEY_ROPE_SCALE_LINEAR.to_string(), Meta::F32(1.0)));
        assert!(load_builder(&file).is_ok());
        *file.meta(KEY_ROPE_SCALE_LINEAR).unwrap() = Meta::F32(4.0);
        let error = load_builder(&file).unwrap_err();
        assert!(matches!(error, GgufModelError::UnsupportedRopeScaling(_)));
        assert!(error.to_string().contains("llama.rope.scale_linear"));
    }

    #[test]
    fn a_key_that_could_change_the_model_is_refused_by_name() {
        let refused = [
            ("llama.logit_scale", Meta::F32(0.5)),
            ("llama.expert_count", Meta::U32(8)),
            ("llama.expert_used_count", Meta::U32(2)),
            ("llama.attn_logit_softcapping", Meta::F32(50.0)),
            ("llama.final_logit_softcapping", Meta::F32(30.0)),
            ("llama.attention.clamp_kqv", Meta::F32(8.0)),
            ("llama.attention.max_alibi_bias", Meta::F32(8.0)),
            ("llama.attention.sliding_window", Meta::U32(4096)),
            ("llama.attention.causal", Meta::U32(0)),
            ("llama.use_parallel_residual", Meta::U32(1)),
            ("llama.embedding_length_out", Meta::U32(8)),
            ("llama.rope.dimension_count_swa", Meta::U32(4)),
            ("llama.something_new", Meta::U32(1)),
            (
                "general.tensor_extra.name",
                Meta::Str("blk.0.attn_q".into()),
            ),
            ("prism.hadamard.version", Meta::U32(1)),
        ];
        for (key, value) in refused {
            let mut file = builder();
            file.metadata.push((key.to_string(), value));
            let error = load_builder(&file).unwrap_err();
            assert_eq!(
                error,
                GgufModelError::UnsupportedMetadata { key: key.into() },
                "{key}"
            );
            assert!(error.to_string().contains(key), "{key}");
        }
        // A scaling key keeps its own error, even next to a refused key.
        let mut file = builder();
        file.metadata
            .push(("llama.logit_scale".to_string(), Meta::F32(0.5)));
        file.metadata
            .push((KEY_ROPE_SCALING_FACTOR.to_string(), Meta::F32(2.0)));
        assert!(matches!(
            load_builder(&file).unwrap_err(),
            GgufModelError::UnsupportedRopeScaling(_)
        ));
    }

    #[test]
    fn the_accepted_keys_and_the_informational_ones_pass() {
        // Every key of the list, with the values a file of this model carries.
        let mut file = builder();
        file.metadata
            .push((KEY_KEY_LENGTH.to_string(), Meta::U32(4)));
        file.metadata
            .push((KEY_VALUE_LENGTH.to_string(), Meta::U32(4)));
        file.metadata
            .push((KEY_LAYER_NORM_EPSILON.to_string(), Meta::F32(1e-5)));
        for key in ACCEPTED_KEYS {
            assert!(
                file.metadata.iter().any(|(name, _)| name == key),
                "the file lacks {key}"
            );
        }
        // Keys that only describe the file.
        for (key, value) in [
            ("general.type", Meta::Str("model".into())),
            ("general.file_type", Meta::U32(1)),
            ("general.quantization_version", Meta::U32(2)),
            ("general.size_label", Meta::Str("135M".into())),
            ("tokenizer.ggml.bos_token_id", Meta::U32(1)),
        ] {
            file.metadata.push((key.to_string(), value));
        }
        // The synthetic file already says it has no vocabulary: another value of
        // the same key must not change what is read.
        *file.meta("tokenizer.ggml.model").unwrap() = Meta::Str("gpt2".into());
        load_builder(&file).unwrap();
        // The value length is held to the head size like the key length.
        *file.meta(KEY_VALUE_LENGTH).unwrap() = Meta::U32(8);
        assert_eq!(
            load_builder(&file).unwrap_err(),
            GgufModelError::BadHeadSize {
                key: KEY_VALUE_LENGTH.into()
            }
        );
    }

    #[test]
    fn a_rope_freqs_tensor_is_a_named_error() {
        let mut file = builder();
        file.tensors.push(GgufTensorOut {
            name: "rope_freqs.weight".into(),
            dims: vec![2],
            ggml_type: 0,
            data: zeros(8),
        });
        let error = load_builder(&file).unwrap_err();
        assert_eq!(error, GgufModelError::UnsupportedRopeFreqs);
        assert!(error.to_string().contains("rope_freqs.weight"));
    }

    #[test]
    fn a_quantized_tensor_is_refused_with_its_name_and_type() {
        // Q4_0: 18 bytes per 32 weights.
        let mut file = builder();
        let tensor = file.tensor("blk.0.attn_q.weight").unwrap();
        tensor.ggml_type = 2;
        tensor.data = zeros(16 * 16 / 32 * 18);
        let error = load_builder(&file).unwrap_err();
        assert_eq!(
            error,
            GgufModelError::UnsupportedTensorType {
                name: "blk.0.attn_q.weight".into(),
                ggml_type: 2
            }
        );
        let shown = error.to_string();
        assert!(
            shown.contains("blk.0.attn_q.weight") && shown.contains("Q4_0"),
            "{shown}"
        );
        // A code the enum of ggml does not define is refused by number.
        let mut file = builder();
        let tensor = file.tensor("token_embd.weight").unwrap();
        tensor.ggml_type = 99;
        let error = load_builder(&file).unwrap_err();
        assert_eq!(
            error,
            GgufModelError::UnsupportedTensorType {
                name: "token_embd.weight".into(),
                ggml_type: 99
            }
        );
        assert!(error.to_string().contains("unknown ggml type 99"));
    }

    /// Widths that are multiples of the 32 weights of a `Q8_0` block.
    fn blocks() -> Spec {
        Spec {
            hidden: 64,
            layers: 2,
            heads: 4,
            kv_heads: 2,
            ffn: 96,
            vocab: 32,
            eps: 1e-5,
            theta: 500_000.0,
        }
    }

    fn q8_builder() -> GgufBuilder {
        let spec = blocks();
        gguf_builder(
            &spec,
            &weights(&spec),
            Permutation::Converter,
            Storage::Q8_0,
        )
    }

    #[test]
    fn q8_0_matrices_are_bound_in_place_and_the_norms_stay_f32() {
        let model = load_builder(&q8_builder()).unwrap();
        assert!(model.config.tie_word_embeddings);
        let mut matrices = 0;
        for tensor in &model.tensors {
            if tensor.shape.len() == 2 {
                matrices += 1;
                assert_eq!(tensor.kind, TensorKind::Q8_0, "{}", tensor.name);
                // 34 bytes per 32 weights, rows of whole blocks along `ne[0]`.
                let (rows, columns) = (tensor.shape[0], tensor.shape[1]);
                assert_eq!(tensor.byte_len, rows * columns / 32 * 34, "{}", tensor.name);
                assert_eq!(tensor.start % 32, 0, "{}", tensor.name);
            } else {
                assert_eq!(tensor.kind, TensorKind::F32, "{}", tensor.name);
            }
        }
        // The embedding and the 7 matrices of each of the 2 layers.
        assert_eq!(matrices, 1 + 2 * 7);
        // An output matrix of its own, also `Q8_0`, unties the embedding.
        let mut file = q8_builder();
        let embedding = file.tensor("token_embd.weight").unwrap().clone();
        file.tensors.push(GgufTensorOut {
            name: "output.weight".into(),
            ..embedding
        });
        let model = load_builder(&file).unwrap();
        assert!(!model.config.tie_word_embeddings);
        let head = model
            .tensors
            .iter()
            .find(|t| t.name == "lm_head.weight")
            .unwrap();
        assert_eq!(
            (head.kind, head.shape.clone()),
            (TensorKind::Q8_0, vec![32, 64])
        );
    }

    #[test]
    fn a_q8_0_norm_is_refused_by_name() {
        for norm in [
            "blk.1.attn_norm.weight",
            "blk.0.ffn_norm.weight",
            "output_norm.weight",
        ] {
            let mut file = q8_builder();
            let tensor = file.tensor(norm).unwrap();
            // One block of 32 weights, the size the parser expects of a `Q8_0`
            // vector of the width of the model: 64 weights, two blocks.
            tensor.ggml_type = 8;
            tensor.data = zeros(2 * 34);
            let error = load_builder(&file).unwrap_err();
            assert_eq!(error, GgufModelError::QuantizedNorm { name: norm.into() });
            let shown = error.to_string();
            assert!(shown.contains(norm) && shown.contains("Q8_0"), "{shown}");
        }
    }

    #[test]
    fn a_q8_0_tensor_above_rank_2_is_an_unsupported_rank() {
        let mut file = q8_builder();
        let tensor = file.tensor("blk.0.attn_q.weight").unwrap();
        // 64 x 2 x 32 weights in rows of two blocks.
        tensor.dims = vec![64, 2, 32];
        tensor.data = zeros(64 * 2 * 32 / 32 * 34);
        let error = load_builder(&file).unwrap_err();
        assert_eq!(
            error,
            GgufModelError::UnsupportedRank {
                name: "blk.0.attn_q.weight".into(),
                rank: 3
            }
        );
        let shown = error.to_string();
        assert!(
            shown.contains("blk.0.attn_q.weight") && shown.contains("3 dimensions"),
            "{shown}"
        );
        assert!(!shown.contains("norm"), "{shown}");
    }

    #[test]
    fn a_q8_0_vector_is_a_quantized_norm_and_a_matrix_where_a_norm_goes_is_a_shape_mismatch() {
        let mut file = q8_builder();
        let tensor = file.tensor("blk.0.attn_q.weight").unwrap();
        tensor.dims = vec![64 * 64];
        tensor.data = zeros(64 * 64 / 32 * 34);
        assert_eq!(
            load_builder(&file).unwrap_err(),
            GgufModelError::QuantizedNorm {
                name: "blk.0.attn_q.weight".into()
            }
        );
        let mut file = q8_builder();
        let tensor = file.tensor("blk.0.attn_norm.weight").unwrap();
        tensor.ggml_type = 8;
        tensor.dims = vec![64, 2];
        tensor.data = zeros(64 * 2 / 32 * 34);
        assert_eq!(
            load_builder(&file).unwrap_err(),
            GgufModelError::ShapeMismatch {
                name: "blk.0.attn_norm.weight".into()
            }
        );
    }

    #[test]
    fn the_other_quantized_types_are_refused_by_name() {
        // `Q4_0` and `Q5_0` have blocks of 32 weights, `Q4_K` and `Q6_K` of 256
        // (see `TensorKind::block_bytes`), so the width is a multiple of both.
        let spec = Spec {
            hidden: 256,
            layers: 1,
            heads: 4,
            kv_heads: 2,
            ffn: 256,
            vocab: 16,
            eps: 1e-5,
            theta: 10_000.0,
        };
        for (ggml_type, name, block_elements, block_bytes) in [
            (2u32, "Q4_0", 32usize, 18usize),
            (6, "Q5_0", 32, 22),
            (12, "Q4_K", 256, 144),
            (14, "Q6_K", 256, 210),
        ] {
            for tensor_name in [
                "blk.0.attn_q.weight",
                "token_embd.weight",
                "blk.0.ffn_down.weight",
            ] {
                let mut file =
                    gguf_builder(&spec, &weights(&spec), Permutation::Converter, Storage::F32);
                let tensor = file.tensor(tensor_name).unwrap();
                let elements = tensor.dims.iter().product::<u64>() as usize;
                tensor.ggml_type = ggml_type;
                tensor.data = zeros(elements / block_elements * block_bytes);
                let error = load_builder(&file).unwrap_err();
                assert_eq!(
                    error,
                    GgufModelError::UnsupportedTensorType {
                        name: tensor_name.into(),
                        ggml_type
                    },
                    "{name} {tensor_name}"
                );
                let shown = error.to_string();
                assert!(
                    shown.contains(tensor_name) && shown.contains(name),
                    "{shown}"
                );
            }
        }
    }

    #[test]
    fn a_q8_0_matrix_whose_rows_are_not_whole_blocks_is_refused() {
        // 16 weights per row: the first dimension is not a multiple of 32.
        let mut file = builder();
        let tensor = file.tensor("blk.0.attn_q.weight").unwrap();
        tensor.ggml_type = 8;
        tensor.data = zeros(16 * 16 / 32 * 34);
        assert!(matches!(
            load_builder(&file).unwrap_err(),
            GgufModelError::Container(GgufError::BadTensorShape { .. })
        ));
    }

    #[test]
    fn a_tensor_that_disagrees_with_the_config_is_a_shape_mismatch() {
        let mut file = builder();
        let tensor = file.tensor("blk.1.attn_k.weight").unwrap();
        tensor.dims = vec![16, 16];
        tensor.data = zeros(16 * 16 * 4);
        let error = load_builder(&file).unwrap_err();
        assert_eq!(
            error,
            GgufModelError::ShapeMismatch {
                name: "blk.1.attn_k.weight".into()
            }
        );
        assert!(error.to_string().contains("blk.1.attn_k.weight"));
        // The transposed matrix has the right element count and the wrong shape.
        let mut file = builder();
        file.tensor("blk.0.ffn_down.weight").unwrap().dims = vec![16, 8];
        assert!(matches!(
            load_builder(&file).unwrap_err(),
            GgufModelError::ShapeMismatch { .. }
        ));
        // The vocabulary key must say what the embedding says.
        let mut file = builder();
        *file.meta(KEY_VOCAB_SIZE).unwrap() = Meta::U32(15);
        assert_eq!(
            load_builder(&file).unwrap_err(),
            GgufModelError::ShapeMismatch {
                name: "token_embd.weight".into()
            }
        );
    }

    #[test]
    fn a_missing_or_surplus_tensor_is_named() {
        let mut file = builder();
        file.tensors.retain(|t| t.name != "blk.1.ffn_up.weight");
        assert_eq!(
            load_builder(&file).unwrap_err(),
            GgufModelError::MissingTensor("blk.1.ffn_up.weight".into())
        );
        let mut file = builder();
        file.tensors.retain(|t| t.name != "token_embd.weight");
        assert_eq!(
            load_builder(&file).unwrap_err(),
            GgufModelError::MissingTensor("token_embd.weight".into())
        );
        // A bias would change the model if it were ignored.
        let mut file = builder();
        file.tensors.push(GgufTensorOut {
            name: "blk.0.attn_q.bias".into(),
            dims: vec![16],
            ggml_type: 0,
            data: zeros(64),
        });
        assert_eq!(
            load_builder(&file).unwrap_err(),
            GgufModelError::UnexpectedTensor("blk.0.attn_q.bias".into())
        );
        // A block count above the blocks of the file.
        let mut file = builder();
        *file.meta(KEY_BLOCK_COUNT).unwrap() = Meta::U32(3);
        assert_eq!(
            load_builder(&file).unwrap_err(),
            GgufModelError::MissingTensor("blk.2.attn_norm.weight".into())
        );
        let mut file = builder();
        *file.meta(KEY_BLOCK_COUNT).unwrap() = Meta::U32(u32::MAX);
        assert!(matches!(
            load_builder(&file).unwrap_err(),
            GgufModelError::MissingTensor(_)
        ));
    }

    #[test]
    fn the_head_size_must_be_even_and_agree_with_the_keys() {
        let mut file = builder();
        *file.meta(KEY_ROPE_DIMENSION_COUNT).unwrap() = Meta::U32(2);
        assert_eq!(
            load_builder(&file).unwrap_err(),
            GgufModelError::BadHeadSize {
                key: KEY_ROPE_DIMENSION_COUNT.into()
            }
        );
        let mut file = builder();
        file.metadata
            .push((KEY_KEY_LENGTH.to_string(), Meta::U32(8)));
        assert_eq!(
            load_builder(&file).unwrap_err(),
            GgufModelError::BadHeadSize {
                key: KEY_KEY_LENGTH.into()
            }
        );
        // 16 over 16 heads is a head of one element, which cannot rotate.
        let mut file = builder();
        *file.meta(KEY_HEAD_COUNT).unwrap() = Meta::U32(16);
        assert!(matches!(
            load_builder(&file).unwrap_err(),
            GgufModelError::BadHeadSize { .. }
        ));
        let mut file = builder();
        *file.meta(KEY_HEAD_COUNT).unwrap() = Meta::U32(5);
        assert!(matches!(
            load_builder(&file).unwrap_err(),
            GgufModelError::BadConfig(_)
        ));
    }

    #[test]
    fn a_truncated_file_is_an_error_at_every_cut() {
        let bytes = builder().to_bytes();
        let whole = Gguf::parse(&bytes).unwrap();
        let data_offset = whole.data_offset();
        // Inside the data, at the first byte of it, inside the header and empty.
        for cut in [
            bytes.len() - 1,
            bytes.len() - 40,
            data_offset + 3,
            data_offset,
            100,
            10,
            0,
        ] {
            let error = ModelFile::gguf_from_bytes(bytes[..cut].to_vec())
                .err()
                .unwrap_or_else(|| panic!("a file cut at {cut} was read"));
            assert!(error.to_string().starts_with("gguf:"), "{cut}: {error}");
            assert!(matches!(
                Gguf::parse(&bytes[..cut]),
                Err(GgufError::Truncated | GgufError::TensorOutOfBounds { .. })
            ));
        }
        let error = ModelFile::gguf_from_bytes(b"not a gguf file at all".to_vec())
            .err()
            .unwrap();
        assert!(error.to_string().contains("magic"), "{error}");
        assert!(ModelFile::gguf_from_bytes(bytes).is_ok());
    }
}
