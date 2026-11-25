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
//! A GGUF file is a header, the metadata, the tensor directory, padding to the
//! alignment (32, the default, so the file does not write `general.alignment`),
//! and the tensors, each at an offset that is a multiple of the alignment. The
//! dimensions are written fastest first: a matrix `[out, in]` of the
//! checkpoint is `[in, out]` in the file.

use crate::config::Config;
use crate::error::{Error, Result};
use crate::quant::{f32_to_half, half_to_f32};

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

/// The `safetensors` file of the checkpoint, in `BF16`.
pub fn safetensors_bytes(tensors: &[Tensor]) -> Vec<u8> {
    let mut data = Vec::new();
    let mut header = String::from("{");
    for (index, tensor) in tensors.iter().enumerate() {
        if index != 0 {
            header.push(',');
        }
        let start = data.len();
        for value in &tensor.data {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let end = data.len();
        header.push('"');
        header.push_str(&tensor.name);
        header.push_str("\":{\"dtype\":\"BF16\",\"shape\":[");
        for (position, dimension) in tensor.shape.iter().enumerate() {
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
        let mut values = tensor.values();
        let is_matrix = tensor.shape.len() == 2;
        if permutation == Permutation::Converter && is_matrix {
            if tensor.name.ends_with(".self_attn.q_proj.weight") {
                values = permute_rows(&values, tensor.shape[0], tensor.shape[1], spec.heads);
            } else if tensor.name.ends_with(".self_attn.k_proj.weight") {
                values = permute_rows(&values, tensor.shape[0], tensor.shape[1], spec.kv_heads);
            }
        }
        let (ggml_type, data) = if is_matrix && storage == Storage::F16 {
            let bytes = values
                .iter()
                .flat_map(|&value| f32_to_half(value).to_le_bytes())
                .collect();
            (1, bytes)
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
}
