// SPDX-License-Identifier: GPL-2.0-only
//! The synthetic development model and its GGUF twin.
//!
//! One set of weights, drawn from a fixed seed, is written two ways:
//!
//! * [`safetensors_bytes`], the checkpoint in the Hugging Face layout, in
//!   `bfloat16`, with the rows of the query and key projections in the split
//!   half order that `rope_interleaved = false` rotates;
//! * [`gguf_builder`], a GGUF file of the `llama` architecture with the same
//!   values, which is what `convert_hf_to_gguf.py` would write for that
//!   checkpoint: the rows of `q_proj` and `k_proj` are permuted by
//!   [`permute_rows`], a transcription of `LlamaModel.permute`, and the metadata
//!   and tensor names are those of `gguf-py` (see [`crate::gguf_model`] for the
//!   files and the commit they were read in).
//!
//! The values are deterministic, so the files are identical from one run to the
//! next and a test can assert on the tokens they produce. The `bfloat16` values
//! are exactly representable in `f32`, so the GGUF file in `F32` holds the very
//! values of the checkpoint.
//!
//! The matrices can also be written as `Q8_0` ([`Storage::Q8_0`]): the blocks are
//! those of [`crate::quant::quantize_block_q8`], applied to the rows of the file,
//! that is, after the permutation, as `convert_hf_to_gguf.py` followed by
//! `llama-quantize` would write them. [`safetensors_q8_bytes`] is the twin of such
//! a file in the `rabbit-q8b32` format: it holds the very bytes of the GGUF file,
//! with the rows of `q_proj` and `k_proj` put back in the order of the checkpoint.
//! A block lies inside one row and the permutation moves whole rows, so
//! permuting the quantized rows or quantizing the permuted rows is the same
//! operation, and the two files carry exactly the same quantized values.
//!
//! A GGUF file is a header, the metadata, the tensor directory, padding to the
//! alignment (32, the default, so the file does not write `general.alignment`),
//! and the tensors, each at an offset that is a multiple of the alignment. The
//! dimensions are written fastest first: a matrix `[out, in]` of the
//! checkpoint is `[in, out]` in the file.

use crate::config::Config;
use crate::error::{Error, Result};
use crate::quant::{
    f32_to_half, half_to_f32, quantize_block_q8, BLOCK, Q8_BLOCK_BYTES, Q8_FORMAT_NAME, SCALE_NAME,
};

/// Fixed seed of the synthetic weights.
pub const SEED: u64 = 0x4d61_6452_7573_7421;

/// Prompt the synthetic model runs: seven ASCII bytes fixed by the generator.
///
/// It lives in the generated `config.json`, not in the runtime, so the real
/// checkpoint can use a different prompt without rebuilding the binary.
pub const SYNTHETIC_PROMPT: [usize; 7] = [77, 97, 100, 82, 117, 115, 116];

/// The context length the GGUF metadata states. The runtime does not read it.
const CONTEXT_LENGTH: u32 = 2048;

/// The alignment of the GGUF data section and of every tensor in it.
const ALIGNMENT: usize = 32;

/// The dimensions of a synthetic model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spec {
    /// Hidden size.
    pub hidden: usize,
    /// Decoder layers.
    pub layers: usize,
    /// Query heads.
    pub heads: usize,
    /// Key value heads.
    pub kv_heads: usize,
    /// Feed forward width.
    pub ffn: usize,
    /// Vocabulary size.
    pub vocab: usize,
    /// RMSNorm epsilon.
    pub eps: f32,
    /// RoPE theta.
    pub theta: f32,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            hidden: 128,
            layers: 4,
            heads: 8,
            kv_heads: 4,
            ffn: 256,
            // Big enough for the reference prompt ids (the largest is 4649),
            // so the synthetic run exercises the same prompt constant as the
            // real model. The byte tokens still occupy ids 0 to 255.
            vocab: 8192,
            eps: 1e-5,
            theta: 10_000.0,
        }
    }
}

impl Spec {
    /// Elements per head.
    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }

    /// Width of the key and value projections.
    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim()
    }

    /// Check what a model needs: positive sizes, whole heads, an even head size
    /// for the rotation, and a grouping of the query heads.
    pub fn validate(&self) -> Result<()> {
        if self.hidden == 0
            || self.layers == 0
            || self.heads == 0
            || self.kv_heads == 0
            || self.ffn == 0
            || self.vocab < 256
        {
            return Err(Error::new(
                "dimensions must be positive and vocab must cover 256 byte tokens",
            ));
        }
        if !self.hidden.is_multiple_of(self.heads) {
            return Err(Error::new(
                "hidden size must be divisible by the query head count",
            ));
        }
        if !self.heads.is_multiple_of(self.kv_heads) {
            return Err(Error::new(
                "query heads must be divisible by the key value head count",
            ));
        }
        if !self.head_dim().is_multiple_of(2) {
            return Err(Error::new("the head size must be even for RoPE"));
        }
        Ok(())
    }
}

/// Deterministic 64-bit linear congruential generator.
pub struct Rng {
    state: u64,
}

impl Rng {
    /// A generator that starts from `seed`.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// A value in `[-1, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.state >> 40) as u32;
        (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

/// Round an `f32` to the nearest `bfloat16`.
pub fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let least_significant = (bits >> 16) & 1;
    bits.wrapping_add(0x7fff + least_significant)
        .wrapping_shr(16) as u16
}

/// One tensor of the checkpoint, in `bfloat16`.
#[derive(Clone, Debug)]
pub struct Tensor {
    /// Name in the Hugging Face layout.
    pub name: String,
    /// `[out, in]` for a matrix, `[n]` for a norm.
    pub shape: Vec<usize>,
    /// The `bfloat16` bits, row after row.
    pub data: Vec<u16>,
}

impl Tensor {
    /// Random values scaled by `scale`, or norm weights near one when `scale`
    /// is zero.
    fn random(name: String, shape: Vec<usize>, scale: f32, rng: &mut Rng) -> Self {
        let count = shape.iter().product::<usize>();
        let mut data = Vec::with_capacity(count);
        if scale == 0.0 {
            for _ in 0..count {
                data.push(to_bf16(1.0 + 0.01 * rng.next_f32()));
            }
        } else {
            for _ in 0..count {
                data.push(to_bf16(scale * rng.next_f32()));
            }
        }
        Self { name, shape, data }
    }

    /// The values as `f32`, which is exact.
    pub fn values(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|&bits| f32::from_bits((bits as u32) << 16))
            .collect()
    }
}

/// The weights of a model, in the order of the checkpoint.
pub fn weights(spec: &Spec) -> Vec<Tensor> {
    let kv_dim = spec.kv_dim();
    let mut rng = Rng::new(SEED);
    let mut tensors = Vec::new();
    let scale = 1.0 / (spec.hidden as f32).sqrt();
    tensors.push(Tensor::random(
        "model.embed_tokens.weight".into(),
        vec![spec.vocab, spec.hidden],
        0.08,
        &mut rng,
    ));
    for layer in 0..spec.layers {
        let prefix = format!("model.layers.{layer}");
        let shapes: [(&str, Vec<usize>, f32); 9] = [
            ("input_layernorm", vec![spec.hidden], 0.0),
            ("self_attn.q_proj", vec![spec.hidden, spec.hidden], scale),
            ("self_attn.k_proj", vec![kv_dim, spec.hidden], scale),
            ("self_attn.v_proj", vec![kv_dim, spec.hidden], scale),
            ("self_attn.o_proj", vec![spec.hidden, spec.hidden], scale),
            ("post_attention_layernorm", vec![spec.hidden], 0.0),
            ("mlp.gate_proj", vec![spec.ffn, spec.hidden], scale),
            ("mlp.up_proj", vec![spec.ffn, spec.hidden], scale),
            ("mlp.down_proj", vec![spec.hidden, spec.ffn], scale),
        ];
        for (name, shape, scale) in shapes {
            tensors.push(Tensor::random(
                format!("{prefix}.{name}.weight"),
                shape,
                scale,
                &mut rng,
            ));
        }
    }
    tensors.push(Tensor::random(
        "model.norm.weight".into(),
        vec![spec.hidden],
        0.0,
        &mut rng,
    ));
    tensors
}

/// The `config.json` of the checkpoint.
///
/// `interleaved` is the `rope_interleaved` the file states. The twin of the
/// GGUF file has `false`: its rows are not permuted.
pub fn config_json(spec: &Spec, interleaved: bool) -> String {
    let prompt = SYNTHETIC_PROMPT
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"hidden_size\":{},\"num_hidden_layers\":{},\"num_attention_heads\":{},\"num_key_value_heads\":{},\"intermediate_size\":{},\"rms_norm_eps\":{},\"vocab_size\":{},\"tie_word_embeddings\":true,\"eos_token_id\":0,\"prompt_ids\":[{}],\"rope_theta\":{},\"rope_interleaved\":{}}}",
        spec.hidden,
        spec.layers,
        spec.heads,
        spec.kv_heads,
        spec.ffn,
        spec.eps,
        spec.vocab,
        prompt,
        spec.theta,
        interleaved
    )
}

/// The parsed [`config_json`], the config the `safetensors` twin runs with.
pub fn config(spec: &Spec, interleaved: bool) -> Result<Config> {
    Config::parse(config_json(spec, interleaved).as_bytes())
}

/// One tensor of a `safetensors` file: the name, the `dtype` the header spells,
/// the shape and the bytes.
struct Part {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    payload: Vec<u8>,
}

/// The `safetensors` file of `parts`, with the `__metadata__` of a quantized
/// checkpoint when `quantized` is set.
fn safetensors_file(parts: &[Part], quantized: bool) -> Vec<u8> {
    let mut data = Vec::new();
    let mut header = String::from("{");
    if quantized {
        header.push_str("\"__metadata__\":{\"format\":\"");
        header.push_str(Q8_FORMAT_NAME);
        header.push_str("\",\"block\":\"32\",\"scale\":\"");
        header.push_str(SCALE_NAME);
        header.push_str("\"}");
    }
    for (index, part) in parts.iter().enumerate() {
        if index != 0 || quantized {
            header.push(',');
        }
        let start = data.len();
        data.extend_from_slice(&part.payload);
        let end = data.len();
        header.push('"');
        header.push_str(&part.name);
        header.push_str("\":{\"dtype\":\"");
        header.push_str(part.dtype);
        header.push_str("\",\"shape\":[");
        for (position, dimension) in part.shape.iter().enumerate() {
            if position != 0 {
                header.push(',');
            }
            header.push_str(&dimension.to_string());
        }
        header.push_str("],\"data_offsets\":[");
        header.push_str(&start.to_string());
        header.push(',');
        header.push_str(&end.to_string());
        header.push_str("]}");
    }
    header.push('}');

    // Safetensors pads the JSON with spaces so the data starts on an eight byte
    // boundary. The runtime does not require it, but real files do it.
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }

    let mut out = Vec::with_capacity(8 + header.len() + data.len());
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    out
}

fn bf16_bytes(tensor: &Tensor) -> Vec<u8> {
    tensor
        .data
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// The `safetensors` file of the checkpoint, in `BF16`.
pub fn safetensors_bytes(tensors: &[Tensor]) -> Vec<u8> {
    let parts: Vec<Part> = tensors
        .iter()
        .map(|tensor| Part {
            name: tensor.name.clone(),
            dtype: "BF16",
            shape: tensor.shape.clone(),
            payload: bf16_bytes(tensor),
        })
        .collect();
    safetensors_file(&parts, false)
}

/// The blocks of `Q8_0` of a row major matrix whose rows are `columns` long, as
/// `quantize_row_q8_0_ref` lays them out and with the quantizer of the runtime.
///
/// A row is a whole number of blocks, so the blocks of the matrix are those of
/// its rows one after the other.
pub fn quantize_q8_rows(values: &[f32], columns: usize) -> Vec<u8> {
    assert!(
        columns != 0 && columns.is_multiple_of(BLOCK) && values.len().is_multiple_of(columns),
        "rows that are a whole number of blocks"
    );
    let mut bytes = vec![0u8; values.len() / BLOCK * Q8_BLOCK_BYTES];
    let blocks = values.as_chunks::<BLOCK>().0;
    for (block, out) in blocks
        .iter()
        .zip(bytes.as_chunks_mut::<Q8_BLOCK_BYTES>().0.iter_mut())
    {
        quantize_block_q8(block, out).expect("the synthetic weights are finite");
    }
    bytes
}

/// The `Q8_0` bytes of a matrix of the checkpoint as the GGUF file holds them:
/// the rows of `q_proj` and `k_proj` permuted as `permutation` says, then
/// quantized.
fn gguf_q8_bytes(spec: &Spec, tensor: &Tensor, permutation: Permutation) -> Vec<u8> {
    let columns = tensor.shape[1];
    quantize_q8_rows(&gguf_values(spec, tensor, permutation), columns)
}

/// The `safetensors` twin of the `Q8_0` GGUF file written with
/// [`Permutation::Converter`]: the same quantized bytes in the rows of the
/// checkpoint.
///
/// Every matrix is `Q8B32`, which is the block of `Q8_0` byte for byte, and
/// every norm stays `BF16`. The bytes of `q_proj` and `k_proj` are those of the
/// GGUF file, permuted back by [`unpermute_rows`], rows of whole blocks, so no
/// block is quantized twice and the two files hold the same quantized values.
/// The file is also what `rabbit-infer-quantize` writes for the `BF16`
/// checkpoint, which quantizes the rows of the checkpoint without permuting:
/// a test runs `quantize::checkpoint` on the checkpoint and compares the bytes.
pub fn safetensors_q8_bytes(spec: &Spec, tensors: &[Tensor]) -> Vec<u8> {
    let parts: Vec<Part> = tensors
        .iter()
        .map(|tensor| {
            if tensor.shape.len() != 2 {
                return Part {
                    name: tensor.name.clone(),
                    dtype: "BF16",
                    shape: tensor.shape.clone(),
                    payload: bf16_bytes(tensor),
                };
            }
            let mut payload = gguf_q8_bytes(spec, tensor, Permutation::Converter);
            if let Some(heads) = permuted_heads(spec, &tensor.name) {
                let row_bytes = tensor.shape[1] / BLOCK * Q8_BLOCK_BYTES;
                payload = unpermute_rows(&payload, tensor.shape[0], row_bytes, heads);
            }
            Part {
                name: tensor.name.clone(),
                dtype: "Q8B32",
                shape: tensor.shape.clone(),
                payload,
            }
        })
        .collect();
    safetensors_file(&parts, true)
}

/// The permutation of `LlamaModel.permute` in `conversion/llama.py`.
///
/// For a matrix of `rows` rows and `columns` columns, `heads` the number of
/// heads the rows are made of (`n_head` for `q_proj`, `n_kv_head` for
/// `k_proj`), the converter computes
/// `w.reshape(heads, 2, rows // heads // 2, columns).swapaxes(1, 2).reshape(w.shape)`.
/// The element `[h, s, j]` of the reshaped matrix moves to `[h, j, s]`: row
/// `h * d + s * d / 2 + j` of the checkpoint becomes row `h * d + 2 * j + s` of
/// the file, `d` being the rows of one head. The two halves of a head are
/// interleaved, so the pair `(j, j + d/2)` that the checkpoint rotates is the
/// adjacent pair `(2j, 2j + 1)` in the file.
pub fn permute_rows<T: Copy>(values: &[T], rows: usize, columns: usize, heads: usize) -> Vec<T> {
    assert_eq!(values.len(), rows * columns, "a matrix of rows and columns");
    assert!(
        heads != 0 && rows.is_multiple_of(heads * 2),
        "every head has an even number of rows"
    );
    let half = rows / heads / 2;
    let mut out = Vec::with_capacity(values.len());
    for head in 0..heads {
        for j in 0..half {
            for s in 0..2 {
                let source = (head * 2 + s) * half + j;
                out.extend_from_slice(&values[source * columns..(source + 1) * columns]);
            }
        }
    }
    out
}

/// The inverse of [`permute_rows`]: the rows of the file put back in the order
/// of the checkpoint.
pub fn unpermute_rows<T: Copy>(values: &[T], rows: usize, columns: usize, heads: usize) -> Vec<T> {
    // `source[i]` is the row of the checkpoint that row `i` of the file holds.
    let indices: Vec<usize> = (0..rows).collect();
    let source = permute_rows(&indices, rows, 1, heads);
    let mut position = vec![0usize; rows];
    for (file_row, &row) in source.iter().enumerate() {
        position[row] = file_row;
    }
    let mut out = Vec::with_capacity(values.len());
    for &file_row in &position {
        out.extend_from_slice(&values[file_row * columns..(file_row + 1) * columns]);
    }
    out
}

/// A metadata value the builder writes.
#[derive(Clone, Debug, PartialEq)]
pub enum Meta {
    /// `uint32`, type 4.
    U32(u32),
    /// `float32`, type 6.
    F32(f32),
    /// A string, type 8.
    Str(String),
}

/// One tensor of a GGUF file under construction.
#[derive(Clone, Debug)]
pub struct GgufTensorOut {
    /// The name in the file.
    pub name: String,
    /// The dimensions, fastest first.
    pub dims: Vec<u64>,
    /// The `ggml_type` code.
    pub ggml_type: u32,
    /// The bytes of the tensor.
    pub data: Vec<u8>,
}

/// A GGUF file under construction. The fields are public so that a test can
/// change what a faithful file says and write the broken file it needs.
#[derive(Clone, Debug, Default)]
pub struct GgufBuilder {
    /// The metadata in the order of the file.
    pub metadata: Vec<(String, Meta)>,
    /// The tensors in the order of the file.
    pub tensors: Vec<GgufTensorOut>,
}

impl GgufBuilder {
    /// The value of a metadata key.
    pub fn meta(&mut self, key: &str) -> Option<&mut Meta> {
        self.metadata
            .iter_mut()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    }

    /// Remove a metadata key.
    pub fn remove_meta(&mut self, key: &str) {
        self.metadata.retain(|(name, _)| name != key);
    }

    /// The tensor of a name.
    pub fn tensor(&mut self, name: &str) -> Option<&mut GgufTensorOut> {
        self.tensors.iter_mut().find(|tensor| tensor.name == name)
    }

    /// The bytes of the file, version 3, alignment 32.
    pub fn to_bytes(&self) -> Vec<u8> {
        fn string(out: &mut Vec<u8>, text: &str) {
            out.extend_from_slice(&(text.len() as u64).to_le_bytes());
            out.extend_from_slice(text.as_bytes());
        }
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());
        for (key, value) in &self.metadata {
            string(&mut out, key);
            match value {
                Meta::U32(value) => {
                    out.extend_from_slice(&4u32.to_le_bytes());
                    out.extend_from_slice(&value.to_le_bytes());
                }
                Meta::F32(value) => {
                    out.extend_from_slice(&6u32.to_le_bytes());
                    out.extend_from_slice(&value.to_le_bytes());
                }
                Meta::Str(value) => {
                    out.extend_from_slice(&8u32.to_le_bytes());
                    string(&mut out, value);
                }
            }
        }
        let mut offset = 0usize;
        for tensor in &self.tensors {
            string(&mut out, &tensor.name);
            out.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
            for dim in &tensor.dims {
                out.extend_from_slice(&dim.to_le_bytes());
            }
            out.extend_from_slice(&tensor.ggml_type.to_le_bytes());
            out.extend_from_slice(&(offset as u64).to_le_bytes());
            offset += tensor.data.len().next_multiple_of(ALIGNMENT);
        }
        while !out.len().is_multiple_of(ALIGNMENT) {
            out.push(0);
        }
        for tensor in &self.tensors {
            out.extend_from_slice(&tensor.data);
            while !out.len().is_multiple_of(ALIGNMENT) {
                out.push(0);
            }
        }
        out
    }
}

/// How the matrices of the GGUF file are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage {
    /// `F32`, ggml type 0.
    F32,
    /// `F16`, ggml type 1: the values rounded to binary16.
    F16,
    /// `F32` that holds the values `F16` would, so that the two files give the
    /// same products bit for bit.
    F32OfF16,
    /// `Q8_0`, ggml type 8: 34 bytes per 32 weights, a binary16 scale then 32
    /// signed bytes, quantized with [`quantize_block_q8`] after the permutation.
    Q8_0,
}

/// Whether the rows of the query and key projections are permuted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Permutation {
    /// As the converter writes them: the adjacent pairs of the file are the
    /// pairs `(j, j + d/2)` of the checkpoint.
    Converter,
    /// The rows of the checkpoint. A reader that rotates adjacent pairs gets
    /// the wrong model from such a file: it is the mistake to detect.
    None,
}

/// The name of a tensor of the checkpoint in `gguf-py`.
fn gguf_name(name: &str) -> String {
    match name {
        "model.embed_tokens.weight" => return "token_embd.weight".into(),
        "model.norm.weight" => return "output_norm.weight".into(),
        "lm_head.weight" => return "output.weight".into(),
        _ => {}
    }
    let rest = name
        .strip_prefix("model.layers.")
        .expect("a layer tensor of the synthetic model");
    let (layer, tail) = rest.split_once('.').expect("a layer index");
    let part = match tail {
        "input_layernorm.weight" => "attn_norm",
        "self_attn.q_proj.weight" => "attn_q",
        "self_attn.k_proj.weight" => "attn_k",
        "self_attn.v_proj.weight" => "attn_v",
        "self_attn.o_proj.weight" => "attn_output",
        "post_attention_layernorm.weight" => "ffn_norm",
        "mlp.gate_proj.weight" => "ffn_gate",
        "mlp.up_proj.weight" => "ffn_up",
        "mlp.down_proj.weight" => "ffn_down",
        other => panic!("tensor {other} of the synthetic model"),
    };
    format!("blk.{layer}.{part}.weight")
}

/// The heads the rows of a tensor are made of when the converter permutes it:
/// `n_head` for `q_proj` and `n_kv_head` for `k_proj`.
fn permuted_heads(spec: &Spec, name: &str) -> Option<usize> {
    if name.ends_with(".self_attn.q_proj.weight") {
        Some(spec.heads)
    } else if name.ends_with(".self_attn.k_proj.weight") {
        Some(spec.kv_heads)
    } else {
        None
    }
}

/// The values of a tensor in the order of the GGUF file.
fn gguf_values(spec: &Spec, tensor: &Tensor, permutation: Permutation) -> Vec<f32> {
    let values = tensor.values();
    match permuted_heads(spec, &tensor.name) {
        Some(heads) if permutation == Permutation::Converter && tensor.shape.len() == 2 => {
            permute_rows(&values, tensor.shape[0], tensor.shape[1], heads)
        }
        _ => values,
    }
}

/// The GGUF file of the checkpoint `tensors`, as a builder.
///
/// A norm vector is `F32`, as the converter writes every one dimensional
/// tensor. A matrix is stored as `storage` says, and the rows of `q_proj` and
/// `k_proj` are permuted as `permutation` says. The model ties the output to the
/// embedding, so there is no `output.weight`.
pub fn gguf_builder(
    spec: &Spec,
    tensors: &[Tensor],
    permutation: Permutation,
    storage: Storage,
) -> GgufBuilder {
    let mut builder = GgufBuilder::default();
    let mut meta = |key: &str, value: Meta| builder.metadata.push((key.to_string(), value));
    meta("general.architecture", Meta::Str("llama".into()));
    meta("general.name", Meta::Str("rabbit-synthetic".into()));
    // A file with no vocabulary. llama.cpp refuses a file that does not say which
    // tokenizer it has; `no_vocab` is the value its loader accepts for a model that
    // takes token ids, which is what the synthetic model and the runtime do.
    meta("tokenizer.ggml.model", Meta::Str("no_vocab".into()));
    meta("llama.context_length", Meta::U32(CONTEXT_LENGTH));
    meta("llama.embedding_length", Meta::U32(spec.hidden as u32));
    meta("llama.block_count", Meta::U32(spec.layers as u32));
    meta("llama.feed_forward_length", Meta::U32(spec.ffn as u32));
    meta("llama.attention.head_count", Meta::U32(spec.heads as u32));
    meta(
        "llama.attention.head_count_kv",
        Meta::U32(spec.kv_heads as u32),
    );
    meta(
        "llama.rope.dimension_count",
        Meta::U32(spec.head_dim() as u32),
    );
    meta(
        "llama.attention.layer_norm_rms_epsilon",
        Meta::F32(spec.eps),
    );
    meta("llama.rope.freq_base", Meta::F32(spec.theta));
    meta("llama.vocab_size", Meta::U32(spec.vocab as u32));

    for tensor in tensors {
        let mut values = gguf_values(spec, tensor, permutation);
        let is_matrix = tensor.shape.len() == 2;
        let (ggml_type, data) = if is_matrix && storage == Storage::F16 {
            let bytes = values
                .iter()
                .flat_map(|&value| f32_to_half(value).to_le_bytes())
                .collect();
            (1, bytes)
        } else if is_matrix && storage == Storage::Q8_0 {
            (8, quantize_q8_rows(&values, tensor.shape[1]))
        } else {
            if is_matrix && storage == Storage::F32OfF16 {
                for value in &mut values {
                    *value = half_to_f32(f32_to_half(*value));
                }
            }
            (
                0,
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            )
        };
        builder.tensors.push(GgufTensorOut {
            name: gguf_name(&tensor.name),
            dims: tensor.shape.iter().rev().map(|&dim| dim as u64).collect(),
            ggml_type,
            data,
        });
    }
    builder
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::Gguf;
    use crate::gguf_model;

    fn small() -> Spec {
        Spec {
            hidden: 32,
            layers: 2,
            heads: 4,
            kv_heads: 2,
            ffn: 48,
            vocab: 256,
            eps: 1e-5,
            theta: 10_000.0,
        }
    }

    #[test]
    fn the_permutation_interleaves_the_two_halves_of_every_head() {
        // Two heads of four rows, one column: row r holds the value r.
        let rows: Vec<u32> = (0..8).collect();
        // Head 0 holds 0 1 2 3: the pairs (0, 2) and (1, 3) of the split half
        // order become the adjacent pairs 0 2 and 1 3. Head 1 is the same at 4.
        assert_eq!(permute_rows(&rows, 8, 1, 2), [0, 2, 1, 3, 4, 6, 5, 7]);
        // Two columns: whole rows move, and the columns keep their order.
        let matrix: Vec<u32> = (0..8).collect();
        assert_eq!(permute_rows(&matrix, 4, 2, 1), [0, 1, 4, 5, 2, 3, 6, 7]);
        // The result is a permutation.
        let mut sorted = permute_rows(&rows, 8, 1, 2);
        sorted.sort_unstable();
        assert_eq!(sorted, rows);
    }

    #[test]
    fn the_permutation_is_the_reshape_swapaxes_reshape_of_the_converter() {
        // An independent transcription with explicit indices of
        // `w.reshape(n_head, 2, rows // n_head // 2, cols).swapaxes(1, 2)
        //  .reshape(w.shape)`: the array a[h][s][j] becomes b[h][j][s].
        let (heads, rows, cols) = (3usize, 12usize, 2usize);
        let d = rows / heads;
        let half = d / 2;
        let w: Vec<u32> = (0..(rows * cols) as u32).collect();
        let at = |h: usize, s: usize, j: usize, c: usize| w[((h * 2 + s) * half + j) * cols + c];
        let mut expected = Vec::new();
        for h in 0..heads {
            for j in 0..half {
                for s in 0..2 {
                    for c in 0..cols {
                        expected.push(at(h, s, j, c));
                    }
                }
            }
        }
        assert_eq!(permute_rows(&w, rows, cols, heads), expected);
    }

    #[test]
    fn the_weights_are_deterministic_and_the_gguf_loads_back() {
        let spec = small();
        spec.validate().unwrap();
        let first = weights(&spec);
        let second = weights(&spec);
        assert_eq!(first.len(), 2 + 9 * spec.layers);
        for (a, b) in first.iter().zip(&second) {
            assert_eq!(a.data, b.data);
        }
        let bytes = gguf_builder(&spec, &first, Permutation::Converter, Storage::F32).to_bytes();
        assert_eq!(
            bytes,
            gguf_builder(&spec, &second, Permutation::Converter, Storage::F32).to_bytes()
        );
        let gguf = Gguf::parse(&bytes).unwrap();
        assert_eq!(gguf.alignment(), 32);
        assert_eq!(gguf.data_offset() % 32, 0);
        let model = gguf_model::load(&gguf).unwrap();
        assert_eq!(model.config.hidden_size, spec.hidden);
        assert_eq!(model.config.num_key_value_heads, spec.kv_heads);
        assert_eq!(model.config.vocab_size, spec.vocab);
        assert!(model.config.rope_interleaved);
        assert!(model.config.tie_word_embeddings);
        assert_eq!(model.tensors.len(), first.len());
        for (loaded, original) in model.tensors.iter().zip(&first) {
            assert_eq!(loaded.name, original.name);
            assert_eq!(loaded.shape, original.shape);
        }
    }

    #[test]
    fn the_gguf_holds_the_values_of_the_checkpoint_with_only_q_and_k_permuted() {
        let spec = small();
        let tensors = weights(&spec);
        let bytes = gguf_builder(&spec, &tensors, Permutation::Converter, Storage::F32).to_bytes();
        let gguf = Gguf::parse(&bytes).unwrap();
        let read = |name: &str| -> Vec<f32> {
            let info = gguf.tensor(name).unwrap();
            gguf.tensor_data(info)
                .unwrap()
                .chunks(4)
                .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
                .collect()
        };
        let by_name = |name: &str| tensors.iter().find(|t| t.name == name).unwrap();
        let q = by_name("model.layers.1.self_attn.q_proj.weight");
        assert_eq!(
            read("blk.1.attn_q.weight"),
            permute_rows(&q.values(), q.shape[0], q.shape[1], spec.heads)
        );
        assert_ne!(read("blk.1.attn_q.weight"), q.values());
        let k = by_name("model.layers.1.self_attn.k_proj.weight");
        assert_eq!(
            read("blk.1.attn_k.weight"),
            permute_rows(&k.values(), k.shape[0], k.shape[1], spec.kv_heads)
        );
        for (name, hf) in [
            (
                "blk.1.attn_v.weight",
                "model.layers.1.self_attn.v_proj.weight",
            ),
            ("blk.1.ffn_up.weight", "model.layers.1.mlp.up_proj.weight"),
            ("token_embd.weight", "model.embed_tokens.weight"),
            ("output_norm.weight", "model.norm.weight"),
        ] {
            assert_eq!(read(name), by_name(hf).values(), "{name}");
        }
        // Without the permutation the rows are the checkpoint's.
        let plain = gguf_builder(&spec, &tensors, Permutation::None, Storage::F32).to_bytes();
        let plain = Gguf::parse(&plain).unwrap();
        let info = plain.tensor("blk.1.attn_q.weight").unwrap();
        let first_row: Vec<f32> = plain.tensor_data(info).unwrap()[..4 * spec.hidden]
            .chunks(4)
            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect();
        assert_eq!(first_row, q.values()[..spec.hidden]);
    }

    /// A model whose widths are multiples of the 32 weights of a block, as `Q8_0`
    /// needs.
    fn blocks() -> Spec {
        Spec {
            hidden: 64,
            layers: 2,
            heads: 4,
            kv_heads: 2,
            ffn: 96,
            vocab: 256,
            eps: 1e-5,
            theta: 10_000.0,
        }
    }

    #[test]
    fn unpermuting_the_rows_undoes_the_permutation() {
        for (rows, columns, heads) in [(8usize, 1usize, 2usize), (12, 5, 3), (16, 34, 4)] {
            let values: Vec<u32> = (0..(rows * columns) as u32).collect();
            let permuted = permute_rows(&values, rows, columns, heads);
            assert_ne!(permuted, values);
            assert_eq!(unpermute_rows(&permuted, rows, columns, heads), values);
            assert_eq!(
                permute_rows(
                    &unpermute_rows(&values, rows, columns, heads),
                    rows,
                    columns,
                    heads
                ),
                values
            );
        }
    }

    /// The bytes of a tensor of a `safetensors` file.
    fn safetensors_tensor<'a>(bytes: &'a [u8], name: &str) -> (&'static str, &'a [u8]) {
        use crate::safetensors::{parse, DType};
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let data = &bytes[8 + header_len..];
        let header = parse(&bytes[8..8 + header_len], data.len()).unwrap();
        let info = header.find(name).unwrap();
        let dtype = match info.dtype {
            DType::Q8B32 => "Q8B32",
            DType::Bf16 => "BF16",
            _ => "other",
        };
        (dtype, &data[info.start..info.end])
    }

    #[test]
    fn the_q8_gguf_and_its_twin_hold_the_same_quantized_bytes() {
        let spec = blocks();
        spec.validate().unwrap();
        let tensors = weights(&spec);
        let gguf_bytes =
            gguf_builder(&spec, &tensors, Permutation::Converter, Storage::Q8_0).to_bytes();
        let gguf = Gguf::parse(&gguf_bytes).unwrap();
        let twin = safetensors_q8_bytes(&spec, &tensors);
        let mut permuted = 0;
        for tensor in &tensors {
            let name = &tensor.name;
            let info = gguf.tensor(&gguf_name(name)).unwrap();
            let in_file = gguf.tensor_data(info).unwrap();
            let (dtype, in_twin) = safetensors_tensor(&twin, name);
            if tensor.shape.len() == 1 {
                // A norm stays floating point: F32 in the GGUF file, BF16 in the twin.
                assert_eq!((info.ggml_type, dtype), (0, "BF16"), "{name}");
                let widened: Vec<u8> = in_twin
                    .chunks(2)
                    .flat_map(|half| [0, 0, half[0], half[1]])
                    .collect();
                assert_eq!(in_file, widened, "{name}");
                continue;
            }
            assert_eq!((info.ggml_type, dtype), (8, "Q8B32"), "{name}");
            let columns = tensor.shape[1];
            let rows = tensor.shape[0];
            let row_bytes = columns / BLOCK * Q8_BLOCK_BYTES;
            assert_eq!(in_file.len(), rows * row_bytes);
            // What the converter and `llama-quantize` write: quantize the rows
            // of the file, the permuted ones.
            let values = tensor.values();
            match permuted_heads(&spec, name) {
                Some(heads) => {
                    permuted += 1;
                    let rows_of_file = permute_rows(&values, rows, columns, heads);
                    assert_eq!(in_file, quantize_q8_rows(&rows_of_file, columns), "{name}");
                    assert_ne!(in_file, in_twin, "{name}");
                    // The twin is the same bytes with the rows put back, and it
                    // is also what quantizing the rows of the checkpoint gives.
                    assert_eq!(
                        permute_rows(in_twin, rows, row_bytes, heads),
                        in_file,
                        "{name}"
                    );
                }
                None => assert_eq!(in_file, in_twin, "{name}"),
            }
            assert_eq!(in_twin, quantize_q8_rows(&values, columns), "{name}");
        }
        assert_eq!(permuted, 2 * spec.layers);
        // Without the permutation the file holds the rows of the checkpoint.
        let plain = gguf_builder(&spec, &tensors, Permutation::None, Storage::Q8_0).to_bytes();
        let plain = Gguf::parse(&plain).unwrap();
        let q = plain.tensor("blk.1.attn_q.weight").unwrap();
        let (_, twin_q) = safetensors_tensor(&twin, "model.layers.1.self_attn.q_proj.weight");
        assert_eq!(plain.tensor_data(q).unwrap(), twin_q);
    }

    /// `ggml_compute_fp32_to_fp16` of `ggml-impl.h` (ggml `d9fe7bf`), which
    /// `GGML_FP32_TO_FP16` is, transcribed.
    fn ggml_fp32_to_fp16(value: f32) -> u16 {
        let scale_to_inf = f32::from_bits(0x7780_0000);
        let scale_to_zero = f32::from_bits(0x0880_0000);
        let mut base = (value.abs() * scale_to_inf) * scale_to_zero;
        let bits = value.to_bits();
        let shl1 = bits.wrapping_add(bits);
        let sign = bits & 0x8000_0000;
        let mut bias = shl1 & 0xff00_0000;
        if bias < 0x7100_0000 {
            bias = 0x7100_0000;
        }
        base += f32::from_bits((bias >> 1).wrapping_add(0x0780_0000));
        let base = base.to_bits();
        let nonsign = ((base >> 13) & 0x7c00) + (base & 0x0fff);
        ((sign >> 16) | if shl1 > 0xff00_0000 { 0x7e00 } else { nonsign }) as u16
    }

    /// `quantize_row_q8_0_ref` of `ggml-quants.c` (ggml `d9fe7bf`) for one
    /// block, transcribed: the scale is `amax / 127` and the quants are the
    /// weights times `1 / d`, with the `d` that is not yet rounded to binary16.
    fn llama_cpp_q8_0(values: &[f32]) -> [u8; Q8_BLOCK_BYTES] {
        let mut block = [0u8; Q8_BLOCK_BYTES];
        let mut amax = 0.0f32;
        for &value in values {
            amax = if amax > value.abs() {
                amax
            } else {
                value.abs()
            };
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        block[..2].copy_from_slice(&ggml_fp32_to_fp16(d).to_le_bytes());
        for (slot, &value) in block[2..].iter_mut().zip(values) {
            *slot = (value * id).round() as i8 as u8;
        }
        block
    }

    #[test]
    fn the_quantizer_of_llama_cpp_writes_the_same_scale_and_not_always_the_same_quants() {
        // The conversion of ggml to binary16 is the one of the runtime.
        let mut rng = Rng::new(7);
        for _ in 0..200_000 {
            let value = rng.next_f32() * 2.0f32.powi((rng.next_f32() * 12.0) as i32 - 8);
            assert_eq!(ggml_fp32_to_fp16(value), f32_to_half(value), "{value}");
        }
        let mut differing_blocks = 0;
        let mut differing_quants = 0;
        let total = 20_000;
        for _ in 0..total {
            let scale = 2.0f32.powi((rng.next_f32() * 6.0) as i32 - 3);
            let values: Vec<f32> = (0..BLOCK).map(|_| rng.next_f32() * scale).collect();
            let ours = quantize_q8_rows(&values, BLOCK);
            let theirs = llama_cpp_q8_0(&values);
            assert_eq!(ours[..2], theirs[..2], "the same binary16 scale");
            let wrong = ours[2..]
                .iter()
                .zip(&theirs[2..])
                .filter(|(a, b)| a != b)
                .count();
            // The only difference is a quant that rounds the other way.
            assert!(ours[2..]
                .iter()
                .zip(&theirs[2..])
                .all(|(a, b)| (*a as i8 as i32 - *b as i8 as i32).abs() <= 1));
            differing_quants += wrong;
            differing_blocks += usize::from(wrong != 0);
        }
        println!(
            "llama.cpp Q8_0 against quantize_block_q8: {differing_blocks} of {total} blocks \
             differ, {differing_quants} of {} quants",
            total * BLOCK
        );
        // Not the same function: `llama.cpp` multiplies by the inverse of the
        // scale before the rounding to binary16 and the runtime after it.
        assert!(differing_blocks > 0);
        // A zero block and an exact block are the same in both.
        assert_eq!(
            quantize_q8_rows(&[0.0; BLOCK], BLOCK),
            llama_cpp_q8_0(&[0.0; BLOCK])
        );
        assert_eq!(
            quantize_q8_rows(&[127.0; BLOCK], BLOCK),
            llama_cpp_q8_0(&[127.0; BLOCK])
        );
    }

    #[test]
    fn the_safetensors_file_is_the_one_the_generator_always_wrote() {
        let spec = small();
        let bytes = safetensors_bytes(&weights(&spec));
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert_eq!(header_len % 8, 0);
        assert!(std::str::from_utf8(&bytes[8..8 + header_len])
            .unwrap()
            .starts_with("{\"model.embed_tokens.weight\":{\"dtype\":\"BF16\""));
    }

    #[test]
    fn the_q8_twin_is_what_the_quantizer_of_the_binary_writes_for_the_checkpoint() {
        let spec = Spec {
            hidden: 64,
            layers: 2,
            heads: 4,
            kv_heads: 2,
            ffn: 96,
            vocab: 256,
            eps: 1e-5,
            theta: 10_000.0,
        };
        spec.validate().unwrap();
        let tensors = weights(&spec);
        let root =
            std::env::temp_dir().join(format!("rabbit-infer-q8-twin-{}", std::process::id()));
        let (input, output) = (root.join("in"), root.join("out"));
        std::fs::create_dir_all(&input).unwrap();
        std::fs::write(input.join("model.safetensors"), safetensors_bytes(&tensors)).unwrap();
        let report = crate::quantize::checkpoint(&input, &output, crate::quantize::Width::Eight)
            .expect("the checkpoint is quantized");
        let written = std::fs::read(output.join("model.safetensors")).unwrap();
        std::fs::remove_dir_all(&root).unwrap();
        let twin = safetensors_q8_bytes(&spec, &tensors);
        assert_eq!(report.bytes, written.len());
        assert!(
            written == twin,
            "the twin differs from the quantizer output (lengths {} and {})",
            twin.len(),
            written.len()
        );
    }
}
