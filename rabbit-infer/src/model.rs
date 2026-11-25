// SPDX-License-Identifier: GPL-2.0-only
//! The Llama-style decoder over a mapped `safetensors` or GGUF file.
//!
//! The model keeps the tensor directory and a borrowed view of every matrix it
//! needs. A matrix is `[out_features, in_features]`, so one output element is a
//! dot product of a row with the input vector. The projections have no bias and
//! the embedding matrix is reused as the output projection when the checkpoint
//! has no separate `lm_head.weight`.
//!
//! `bfloat16` is converted at read time with
//! `f32::from_bits((bits as u32) << 16)`. The conversion is exact: a `bfloat16`
//! is the top half of an `f32`, so the shifted bits reproduce the original
//! `f32` with the low sixteen bits zero and no rounding step.
//!
//! A quantized checkpoint binds the same way. A `Q4B32` or `Q8B32` matrix is a
//! sequence of blocks with a binary16 scale, and the kernel of [`crate::matvec`]
//! reads a block, applies the scale and accumulates the products without
//! expanding the tensor. The norm weights stay floating point, and the reader
//! refuses a header whose declaration disagrees with its tensors.
//!
//! The matrix vector products go through [`crate::matvec`]. A tensor that lies
//! inside the mapped prefix is handed to the blocked kernel as one contiguous
//! `bfloat16` range; the last at most 4095 bytes of the file live in a small
//! buffer instead of the mapping, so the few `bfloat16` rows that reach into
//! that tail still use the element accessor. The rows of a quantized matrix
//! that reach into it are copied out of the file one at a time and go through
//! the same kernel as the other rows, so a row gives the same bits wherever the
//! tensor sits in the file. No weight of the mapped rows is copied. The
//! conversion to `f32` happens in registers, which halves the weight traffic
//! against a cached `f32` copy (269 MB against 538 MB for the reference model)
//! at the cost of one shift and widening per weight, per token.
//!
//! A GGUF file of the `llama` architecture binds the same way. [`ModelFile`]
//! reads its directory with [`crate::gguf_model`], which names every tensor as
//! the `safetensors` layout does and gives it that layout's shape, and keeps the
//! [`Config`] the metadata describes ([`ModelFile::gguf_config`]). The rows of
//! the query and key projections are stored in the order `llama.cpp` rotates
//! them, so that config has `rope_interleaved` set and nothing un-permutes the
//! weights. The matrices of such a file are `F32` or `F16` and go through the
//! scalar kernels [`matvec::matvec_f32`] and [`matvec::matvec_f16`], whose order
//! of operations is canonical and whose speed is that of a plain loop: they are
//! a reference, not an optimisation. A matrix of a `safetensors` file takes the
//! paths it always took.
//!
//! A `Q8_0` matrix of a GGUF file (the embedding and the output matrix included)
//! is bound in place as a `Q8B32` matrix: the block is the same 34 bytes (see
//! [`crate::gguf_model`]), so [`matvec::matvec_q8`] multiplies it on every path
//! and [`quant::dequantize_row_q8`] decodes a row of the embedding. The file
//! carries no `__metadata__` declaration, because the type code of the tensor
//! says the block, the scale and the layout.
//!
//! A `Q5_0`, `Q4_K` or `Q6_K` matrix is bound in place too, as [`DType::Q5_0`],
//! [`DType::Q4K`] and [`DType::Q6K`]: [`matvec::matvec_q5_0_with`],
//! [`matvec::matvec_q4_k_with`] and [`matvec::matvec_q6_k_with`] multiply it, and
//! [`quant::dequantize_row_q5_0`], [`quant::dequantize_row_q4_k`] and
//! [`quant::dequantize_row_q6_k`] decode a row of the embedding. The scales of
//! every block are checked to be finite when the tensor is bound, as for
//! `Q8_0`, and the rows that reach into the final partial page are copied out
//! and go through the same kernel as the rows of the mapping.

use std::time::Instant;

use crate::config::Config;
use crate::error::{Error, Result};
use crate::gguf::{Gguf, GgufError};
use crate::gguf_model::{self, GgufModel, GgufModelError};
use crate::llama_emulation::{self, Activation};
use crate::mapping::Mapping;
use crate::math::{argmax, rms_norm, rope, silu, softmax};
use crate::matvec;
use crate::quant;
use crate::safetensors::{self, DType, TensorInfo};

/// The dimensions the runtime read from the config and derived for the model.
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    /// Hidden size.
    pub hidden: usize,
    /// Decoder layers.
    pub layers: usize,
    /// Query heads.
    pub q_heads: usize,
    /// Key value heads.
    pub kv_heads: usize,
    /// Elements per head.
    pub head_dim: usize,
    /// Feed forward width.
    pub ffn: usize,
    /// Vocabulary size.
    pub vocab: usize,
    /// RMSNorm epsilon.
    pub eps: f32,
    /// RoPE theta.
    pub rope_theta: f32,
    /// `true` when the RoPE pairs are interleaved.
    pub rope_interleaved: bool,
    /// `true` when the output projection reuses the embedding matrix.
    pub tie_word_embeddings: bool,
    /// Token that ends generation.
    pub eos_token_id: u32,
}

impl Dims {
    /// Group size of grouped query attention.
    pub fn group_size(&self) -> usize {
        self.q_heads / self.kv_heads
    }
}

/// The most bytes of a GGUF file copied to read a header that ends in the
/// final partial page of the file.
const MAX_GGUF_HEADER_COPY: usize = 256 << 20;

/// A mapped model file and its tensor directory.
pub struct ModelFile {
    mapping: Mapping,
    header: safetensors::Header,
    data_start: usize,
    /// The configuration of a GGUF file, which carries it in its metadata.
    gguf_config: Option<Config>,
}

impl ModelFile {
    /// Open the file, read the header and map the data.
    pub fn open(path: &str) -> Result<Self> {
        Self::from_mapping(Mapping::open(path)?)
    }

    /// Read a `safetensors` file held in memory.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::from_mapping(Mapping::from_bytes(bytes))
    }

    /// Open a GGUF file of the `llama` architecture and map its data.
    ///
    /// The tensors are read in place, through the same mapping as a
    /// `safetensors` file. Only the directory and the metadata are copied.
    pub fn open_gguf(path: &str) -> Result<Self> {
        Self::gguf_from_mapping(Mapping::open(path)?)
    }

    /// Read a GGUF file held in memory.
    pub fn gguf_from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::gguf_from_mapping(Mapping::from_bytes(bytes))
    }

    /// The configuration the metadata of a GGUF file describes, or `None` for a
    /// `safetensors` file, whose configuration is `config.json`.
    pub fn gguf_config(&self) -> Option<&Config> {
        self.gguf_config.as_ref()
    }

    fn gguf_from_mapping(mapping: Mapping) -> Result<Self> {
        let file_len = mapping.file_len();
        let read = |head: &[u8]| -> core::result::Result<GgufModel, GgufModelError> {
            let gguf = Gguf::parse_head(head, file_len)?;
            gguf_model::load(&gguf)
        };
        // The mapping holds whole pages, and the header is nearly always inside
        // them. When it ends in the final partial page the file is the header
        // and less than a page of data, so reading it whole costs the header.
        let model = match read(mapping.prefix()) {
            Err(GgufModelError::Container(GgufError::Truncated))
                if mapping.mapped_len() < file_len && file_len <= MAX_GGUF_HEADER_COPY =>
            {
                read(&mapping.read_bytes(0, file_len)?)
            }
            Err(GgufModelError::Container(GgufError::Truncated))
                if mapping.mapped_len() < file_len =>
            {
                return Err(Error::new(format!(
                    "gguf: the header ends in the final partial page of a file of {file_len} \
                     bytes, over the {MAX_GGUF_HEADER_COPY} bytes this reader copies to read it"
                )));
            }
            other => other,
        }
        .map_err(Error::new)?;
        let mut header = safetensors::Header::empty();
        for tensor in &model.tensors {
            let dtype = match tensor.kind {
                crate::gguf::TensorKind::F32 => DType::F32,
                crate::gguf::TensorKind::F16 => DType::F16,
                crate::gguf::TensorKind::Q8_0 => DType::Q8B32,
                crate::gguf::TensorKind::Q5_0 => DType::Q5_0,
                crate::gguf::TensorKind::Q4K => DType::Q4K,
                crate::gguf::TensorKind::Q6K => DType::Q6K,
            };
            let end = tensor
                .start
                .checked_add(tensor.byte_len)
                .ok_or_else(|| Error::new("model: tensor offset overflow"))?;
            header.push(&tensor.name, dtype, &tensor.shape, tensor.start, end)?;
        }
        Ok(Self {
            mapping,
            header,
            data_start: model.data_offset,
            gguf_config: Some(model.config),
        })
    }

    fn from_mapping(mapping: Mapping) -> Result<Self> {
        let file_len = mapping.file_len();
        if file_len < 8 {
            return Err(Error::new("model: file is shorter than the header length"));
        }
        let length_bytes = mapping.read_bytes(0, 8)?;
        let header_len = u64::from_le_bytes([
            length_bytes[0],
            length_bytes[1],
            length_bytes[2],
            length_bytes[3],
            length_bytes[4],
            length_bytes[5],
            length_bytes[6],
            length_bytes[7],
        ]);
        let header_len = usize::try_from(header_len)
            .map_err(|_| Error::new("model: header length does not fit this target"))?;
        let data_start = 8usize
            .checked_add(header_len)
            .ok_or_else(|| Error::new("model: header length overflow"))?;
        if data_start > file_len {
            return Err(Error::new("model: header is longer than the file"));
        }
        let header_bytes = mapping.read_bytes(8, header_len)?;
        let header = safetensors::parse(&header_bytes, file_len - data_start)?;
        Ok(Self {
            mapping,
            header,
            data_start,
            gguf_config: None,
        })
    }

    /// Number of tensors in the directory.
    pub fn tensor_count(&self) -> usize {
        self.header.tensors().len()
    }

    /// Bytes of the model that are mapped, always a multiple of the page size.
    pub fn mapped_len(&self) -> usize {
        self.mapping.mapped_len()
    }

    /// Total size of the checkpoint file.
    pub fn file_len(&self) -> usize {
        self.mapping.file_len()
    }

    /// `true` when the kernel accepted the large page advice on the mapping.
    pub fn huge_pages_requested(&self) -> bool {
        self.mapping.huge_pages_requested()
    }

    /// Ask the kernel to make the mapped weights resident and keep them there.
    ///
    /// Returns the number of pinned bytes on success. The final partial page
    /// lives in a small buffer rather than the mapping, so it is not part of
    /// this range; every mapped weight byte is.
    pub fn pin(&self) -> core::result::Result<usize, i64> {
        self.mapping.pin()?;
        Ok(self.mapped_len())
    }

    /// Borrow one tensor, validating its dtype, shape and byte range.
    fn tensor(&self, name: &str) -> Result<Tensor<'_>> {
        let info = self
            .header
            .find(name)
            .ok_or_else(|| Error::new(format!("model: missing tensor {name}")))?;
        let tensor_name = self.header.name(info);
        Tensor::new(
            info,
            tensor_name,
            &self.mapping,
            self.data_start,
            self.gguf_config.is_some(),
        )
    }

    /// `true` when at least one tensor uses the quantized format.
    pub fn has_quantized_tensors(&self) -> bool {
        self.header.has_quantized_tensors()
    }

    /// Check that the header declares the format of every quantized tensor.
    ///
    /// A quantized tensor is meaningless without the block size and the scale
    /// type that wrote it, and a file that carries such a tensor without the
    /// declaration is not the file this runtime reads. The error names the
    /// field that is missing or wrong, so the serial log says why the file was
    /// refused.
    pub fn validate_quantization(&self) -> Result<()> {
        // A GGUF tensor of type `Q8_0`, `Q5_0`, `Q4_K` or `Q6_K` is the block of its
        // type code, and `gguf_model` checked the sizes: there is no header to
        // declare it in. So that a type added to the table later does not pass
        // unchecked, only the types this reader maps are accepted.
        if self.gguf_config.is_some() {
            for tensor in self.header.tensors() {
                if !matches!(
                    tensor.dtype,
                    DType::F32 | DType::F16 | DType::Q8B32 | DType::Q5_0 | DType::Q4K | DType::Q6K
                ) {
                    return Err(Error::new(format!(
                        "model: GGUF tensor {} has dtype {}, only F32, F16, Q8_0, Q5_0, Q4_K and \
                         Q6_K are read",
                        self.header.name(tensor),
                        tensor.dtype.name()
                    )));
                }
            }
            return Ok(());
        }
        if !self.header.has_quantized_tensors() {
            return Ok(());
        }
        let metadata = self.header.metadata().ok_or_else(|| {
            Error::new("model: quantized tensors without a __metadata__ declaration")
        })?;
        // One file carries one width. A header that mixes the two has no single
        // format to declare, so it is refused instead of guessed.
        let mut expected = None;
        for tensor in self.header.tensors() {
            match tensor.dtype {
                DType::Q4B32 => match expected {
                    None => expected = Some(quant::FORMAT_NAME),
                    Some(quant::FORMAT_NAME) => {}
                    Some(_) => {
                        return Err(Error::new(
                            "model: the header mixes Q4B32 and Q8B32 tensors",
                        ))
                    }
                },
                DType::Q8B32 => match expected {
                    None => expected = Some(quant::Q8_FORMAT_NAME),
                    Some(quant::Q8_FORMAT_NAME) => {}
                    Some(_) => {
                        return Err(Error::new(
                            "model: the header mixes Q4B32 and Q8B32 tensors",
                        ))
                    }
                },
                _ => {}
            }
        }
        let expected = expected.unwrap_or(quant::FORMAT_NAME);
        if metadata.format.as_deref() != Some(expected) {
            return Err(Error::new(format!(
                "model: quantized tensors declare format {:?}, expected {expected}",
                metadata.format.as_deref().unwrap_or("none"),
            )));
        }
        if metadata.block.as_deref() != Some("32") {
            return Err(Error::new(format!(
                "model: quantized tensors declare block {:?}, expected 32",
                metadata.block.as_deref().unwrap_or("none")
            )));
        }
        if metadata.scale.as_deref() != Some(quant::SCALE_NAME) {
            return Err(Error::new(format!(
                "model: quantized tensors declare scale {:?}, expected {}",
                metadata.scale.as_deref().unwrap_or("none"),
                quant::SCALE_NAME
            )));
        }
        Ok(())
    }
}

/// One tensor viewed through the mapping.
#[derive(Clone, Copy)]
struct Tensor<'a> {
    name: &'a str,
    info: &'a TensorInfo,
    mapping: &'a Mapping,
    file_offset: usize,
    element_size: usize,
    /// `true` for an `F32` or `F16` matrix of a GGUF file, which the scalar
    /// float kernels of [`crate::matvec`] multiply.
    float_kernel: bool,
}

impl<'a> Tensor<'a> {
    fn new(
        info: &'a TensorInfo,
        name: &'a str,
        mapping: &'a Mapping,
        data_start: usize,
        from_gguf: bool,
    ) -> Result<Self> {
        let shape = info.shape();
        let bytes = match info.dtype {
            dtype if dtype.is_quantized() => {
                // The format owns the byte count: one block of a scale and a
                // packed payload per thirty-two weights, not a fixed element
                // size.
                if shape.len() != 2 {
                    return Err(Error::new(format!(
                        "model: quantized tensor {name} is not a matrix"
                    )));
                }
                quantized_bytes(dtype, shape[0], shape[1])?
            }
            dtype => {
                let element_size = dtype.element_size().ok_or_else(|| {
                    Error::new(format!("model: tensor {name} has unsupported dtype"))
                })?;
                let elements = info.element_count();
                elements
                    .checked_mul(element_size)
                    .ok_or_else(|| Error::new(format!("model: tensor {name} is too large")))?
            }
        };
        if bytes != info.byte_len() {
            return Err(Error::new(format!(
                "model: tensor {name} has a byte length that disagrees with its shape"
            )));
        }
        let file_offset = data_start
            .checked_add(info.start)
            .ok_or_else(|| Error::new("model: tensor offset overflow"))?;
        if file_offset + bytes > mapping.file_len() {
            return Err(Error::new(format!(
                "model: tensor {name} reaches past the file"
            )));
        }
        let element_size = info.dtype.element_size().unwrap_or(0);
        let tensor = Self {
            name,
            info,
            mapping,
            file_offset,
            element_size,
            float_kernel: from_gguf && matches!(info.dtype, DType::F32 | DType::F16),
        };
        if info.dtype.is_quantized() {
            tensor.check_scales()?;
        }
        Ok(tensor)
    }

    /// Refuse a block whose scale is not a finite binary16.
    ///
    /// The decoder multiplies a whole block by its scale, so a scale that is an
    /// infinity or a NaN would poison thirty-two weights and the serial log
    /// would only show a wrong token. The check reads one `u16` per scale of a
    /// block (two for `Q4_K`, which has `d` and `dmin`), which is cheaper than
    /// the first token and names the block that is wrong.
    fn check_scales(&self) -> Result<()> {
        let columns = self.shape()[1];
        let (row_bytes, block_bytes) = quantized_layout(self.dtype(), columns)?;
        let blocks = columns / quantized_block_elements(self.dtype());
        let offsets = scale_offsets(self.dtype());
        let mut half = [0u8; 2];
        for row in 0..self.shape()[0] {
            for block in 0..blocks {
                for scale in offsets {
                    let offset = self.file_offset + row * row_bytes + block * block_bytes + scale;
                    self.mapping.copy_bytes(offset, &mut half)?;
                    if !quant::scale_is_finite(u16::from_le_bytes(half)) {
                        return Err(Error::new(format!(
                            "model: tensor {} has a non-finite scale in row {row} block {block}",
                            self.name
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    fn shape(&self) -> &[usize] {
        self.info.shape()
    }

    fn dtype(&self) -> DType {
        self.info.dtype
    }

    /// Read one element as `f32`.
    ///
    /// A quantized tensor has no element accessor: a nibble without its block
    /// scale means nothing. Callers read a whole row instead.
    fn value(&self, index: usize) -> f32 {
        let offset = self.file_offset + index * self.element_size;
        match self.dtype() {
            DType::Bf16 => f32::from_bits((self.mapping.u16_at(offset) as u32) << 16),
            DType::F32 => f32::from_bits(self.mapping.u32_at(offset)),
            DType::F16 => quant::half_to_f32(self.mapping.u16_at(offset)),
            DType::Q4B32 | DType::Q8B32 | DType::Q5_0 | DType::Q4K | DType::Q6K | DType::Other => {
                0.0
            }
        }
    }

    /// The rows of a `bfloat16` matrix that are contiguous in the mapping.
    ///
    /// The returned bytes hold whole rows, so a caller can hand them to the
    /// blocked kernel. `None` means the matrix is not `bfloat16` or starts
    /// outside the mapping.
    fn contiguous_bf16(&self) -> Option<(&[u8], usize)> {
        if self.dtype() != DType::Bf16 || self.shape().len() != 2 {
            return None;
        }
        let rows = self.shape()[0];
        let columns = self.shape()[1];
        let row_bytes = columns.checked_mul(2)?;
        if row_bytes == 0 {
            return None;
        }
        self.contiguous_rows(row_bytes, rows)
    }

    /// The rows of a quantized matrix that are contiguous in the mapping.
    ///
    /// A row is one block per thirty-two weights, so the same slicing as the
    /// `bfloat16` path works; only the row length changes with the width.
    fn contiguous_quantized(&self) -> Option<(&[u8], usize)> {
        if !self.dtype().is_quantized() || self.shape().len() != 2 {
            return None;
        }
        let rows = self.shape()[0];
        let row_bytes = quantized_layout(self.dtype(), self.shape()[1]).ok()?.0;
        self.contiguous_rows(row_bytes, rows)
    }

    /// Slice the whole rows of a matrix that the mapping already holds.
    fn contiguous_rows(&self, row_bytes: usize, rows: usize) -> Option<(&[u8], usize)> {
        if row_bytes == 0 {
            return None;
        }
        let available = self.mapping.bytes_from(self.file_offset);
        let mapped_rows = (available.len() / row_bytes).min(rows);
        if mapped_rows == 0 {
            return None;
        }
        let length = mapped_rows.checked_mul(row_bytes)?;
        Some((available.get(..length)?, mapped_rows))
    }

    /// Dot one row of a matrix with `vector`.
    ///
    /// This is the path for the `bfloat16` rows that reach into the final
    /// partial page; the mapped rows go through the blocked kernels of
    /// [`crate::matvec`], and the quantized rows of that page through
    /// [`project_quantized_tail`].
    fn dot(&self, row: usize, vector: &[f32]) -> Result<f32> {
        let columns = self.shape().last().copied().unwrap_or(0);
        if vector.len() != columns || row >= self.shape()[0] {
            return Err(Error::new(format!(
                "model: cannot dot row {row} of tensor {}",
                self.name
            )));
        }
        let base = row * columns;
        let start = self.file_offset + base * self.element_size;
        let in_prefix = if start < self.mapping.mapped_len() {
            ((self.mapping.mapped_len() - start) / self.element_size).min(columns)
        } else {
            0
        };
        let prefix = self.mapping.prefix();
        let mut sum = 0.0f32;
        match self.dtype() {
            DType::Bf16 => {
                for (index, &input) in vector.iter().enumerate().take(in_prefix) {
                    let offset = start + index * 2;
                    let bits = u16::from_le_bytes([prefix[offset], prefix[offset + 1]]);
                    sum += f32::from_bits((bits as u32) << 16) * input;
                }
            }
            DType::F32 => {
                for (index, &input) in vector.iter().enumerate().take(in_prefix) {
                    let offset = start + index * 4;
                    let bits = u32::from_le_bytes([
                        prefix[offset],
                        prefix[offset + 1],
                        prefix[offset + 2],
                        prefix[offset + 3],
                    ]);
                    sum += f32::from_bits(bits) * input;
                }
            }
            DType::F16
            | DType::Q4B32
            | DType::Q8B32
            | DType::Q5_0
            | DType::Q4K
            | DType::Q6K
            | DType::Other => {
                return Err(Error::new(format!(
                    "model: tensor {} has no scalar element accessor",
                    self.name
                )))
            }
        }
        for (index, &input) in vector.iter().enumerate().skip(in_prefix) {
            sum += self.value(base + index) * input;
        }
        Ok(sum)
    }

    /// Copy one row of a matrix into `out`.
    fn read_row(&self, row: usize, out: &mut [f32]) -> Result<()> {
        let columns = self.shape().last().copied().unwrap_or(0);
        if out.len() != columns || row >= self.shape()[0] {
            return Err(Error::new(format!(
                "model: cannot read row {row} of tensor {}",
                self.name
            )));
        }
        if self.dtype().is_quantized() {
            let row_bytes = quantized_layout(self.dtype(), columns)?.0;
            let mut blocks = vec![0u8; row_bytes];
            self.mapping
                .copy_bytes(self.file_offset + row * row_bytes, &mut blocks)?;
            return match self.dtype() {
                DType::Q8B32 => quant::dequantize_row_q8(&blocks, out),
                DType::Q5_0 => quant::dequantize_row_q5_0(&blocks, out),
                DType::Q4K => quant::dequantize_row_q4_k(&blocks, out),
                DType::Q6K => quant::dequantize_row_q6_k(&blocks, out),
                _ => quant::dequantize_row(&blocks, out),
            };
        }
        let base = row * columns;
        for (index, slot) in out.iter_mut().enumerate() {
            *slot = self.value(base + index);
        }
        Ok(())
    }

    /// The one dimensional weight of a norm.
    fn read_vector(&self, out: &mut [f32]) -> Result<()> {
        if self.shape().len() != 1 || self.shape()[0] != out.len() {
            return Err(Error::new(format!(
                "model: tensor {} is not a vector of the expected length",
                self.name
            )));
        }
        if self.dtype().is_quantized() {
            return Err(Error::new(format!(
                "model: normalisation weight {} is quantized, expected a floating point vector",
                self.name
            )));
        }
        for (index, slot) in out.iter_mut().enumerate() {
            *slot = self.value(index);
        }
        Ok(())
    }
}

/// One decoder layer.
struct Layer<'a> {
    input_norm: Tensor<'a>,
    q_proj: Tensor<'a>,
    k_proj: Tensor<'a>,
    v_proj: Tensor<'a>,
    o_proj: Tensor<'a>,
    post_norm: Tensor<'a>,
    gate_proj: Tensor<'a>,
    up_proj: Tensor<'a>,
    down_proj: Tensor<'a>,
}

/// The decoder weights and the dimensions they imply.
pub struct Model<'a> {
    dims: Dims,
    embed: Tensor<'a>,
    layers: Vec<Layer<'a>>,
    final_norm: Tensor<'a>,
    output: Tensor<'a>,
    tensor_count: usize,
    emulate_llama: bool,
}

impl<'a> Model<'a> {
    /// Bind the tensors of `file` to the dimensions in `config`.
    ///
    /// The config is authoritative. Every tensor shape is checked against it
    /// and a disagreement is an error: a checkpoint and its config are shipped
    /// together, so a mismatch would otherwise silently compute a different
    /// model.
    pub fn new(file: &'a ModelFile, config: &Config) -> Result<Self> {
        // A quantized checkpoint has to say which block format wrote it before
        // a single tensor is bound.
        file.validate_quantization()?;
        let hidden = config.hidden_size;
        let layer_count = config.num_hidden_layers;
        let q_heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads;
        let ffn = config.intermediate_size;
        let vocab = config.vocab_size;
        let head_dim = hidden / q_heads;
        let kv_dim = kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| Error::new("model: key value width overflow"))?;

        let embed = file.tensor("model.embed_tokens.weight")?;
        require_matrix(&embed, vocab, hidden, "model.embed_tokens.weight")?;

        let actual_layers = count_layers(file)?;
        if actual_layers != layer_count {
            return Err(Error::new(format!(
                "model: config says {layer_count} layers, the tensors carry {actual_layers}"
            )));
        }

        let mut layers = Vec::with_capacity(layer_count);
        for index in 0..layer_count {
            let input_norm = layer_tensor(file, index, "input_layernorm.weight")?;
            let q_proj = layer_tensor(file, index, "self_attn.q_proj.weight")?;
            let k_proj = layer_tensor(file, index, "self_attn.k_proj.weight")?;
            let v_proj = layer_tensor(file, index, "self_attn.v_proj.weight")?;
            let o_proj = layer_tensor(file, index, "self_attn.o_proj.weight")?;
            let post_norm = layer_tensor(file, index, "post_attention_layernorm.weight")?;
            let gate_proj = layer_tensor(file, index, "mlp.gate_proj.weight")?;
            let up_proj = layer_tensor(file, index, "mlp.up_proj.weight")?;
            let down_proj = layer_tensor(file, index, "mlp.down_proj.weight")?;

            require_matrix(&q_proj, hidden, hidden, "q_proj")?;
            require_matrix(&k_proj, kv_dim, hidden, "k_proj")?;
            require_matrix(&v_proj, kv_dim, hidden, "v_proj")?;
            require_matrix(&o_proj, hidden, hidden, "o_proj")?;
            require_matrix(&gate_proj, ffn, hidden, "gate_proj")?;
            require_matrix(&up_proj, ffn, hidden, "up_proj")?;
            require_matrix(&down_proj, hidden, ffn, "down_proj")?;
            require_vector(&input_norm, hidden, "input_layernorm.weight")?;
            require_vector(&post_norm, hidden, "post_attention_layernorm.weight")?;

            layers.push(Layer {
                input_norm,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                post_norm,
                gate_proj,
                up_proj,
                down_proj,
            });
        }

        let final_norm = file.tensor("model.norm.weight")?;
        require_vector(&final_norm, hidden, "model.norm.weight")?;

        // The config says whether the output projection is a separate matrix.
        // When it is tied, the embedding matrix is reused and a checkpoint
        // that also carries `lm_head.weight` still uses the tied path.
        let output = if config.tie_word_embeddings {
            embed
        } else {
            let head = file.tensor("lm_head.weight")?;
            require_matrix(&head, vocab, hidden, "lm_head.weight")?;
            head
        };

        Ok(Self {
            dims: Dims {
                hidden,
                layers: layer_count,
                q_heads,
                kv_heads,
                head_dim,
                ffn,
                vocab,
                eps: config.rms_norm_eps,
                rope_theta: config.rope_theta,
                rope_interleaved: config.rope_interleaved,
                tie_word_embeddings: config.tie_word_embeddings,
                eos_token_id: config.eos_token_id,
            },
            embed,
            layers,
            final_norm,
            output,
            tensor_count: file.tensor_count(),
            emulate_llama: false,
        })
    }

    /// Round what the CPU backend of `llama.cpp` rounds: the activation of
    /// every product with a quantized weight is packed into the type of its dot
    /// kernel and unpacked ([`crate::llama_emulation`]), and a key or a value is
    /// rounded to binary16 when it is stored. The integer dot product and the
    /// AVX2 quantiser of the oracle's x86 build are not emulated: the reference
    /// quantisers are used, and they can differ from the AVX2 one by a step on a
    /// half-way tie. Off by default; a diagnostic of the oracle's arithmetic that
    /// changes no kernel, and it applies to the sessions started after the call.
    pub fn set_emulate_llama(&mut self, on: bool) {
        self.emulate_llama = on;
    }

    /// `true` when [`Self::set_emulate_llama`] is on.
    pub fn emulate_llama(&self) -> bool {
        self.emulate_llama
    }

    /// The dimensions read from the model.
    pub fn dims(&self) -> Dims {
        self.dims
    }

    /// Number of tensors the file carries.
    pub fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    /// Name of the tensor element type the model was written with.
    pub fn weight_dtype(&self) -> &'static str {
        self.embed.dtype().name()
    }

    /// `true` when any matrix of the model (the embedding, the output matrix and
    /// the seven matrices of every layer) is quantized. The norms are vectors
    /// and do not count; neither does the type of the embedding alone, which
    /// [`Self::weight_dtype`] reports.
    pub fn has_quantized_matrix(&self) -> bool {
        let layer_matrices = self.layers.iter().flat_map(|layer| {
            [
                &layer.q_proj,
                &layer.k_proj,
                &layer.v_proj,
                &layer.o_proj,
                &layer.gate_proj,
                &layer.up_proj,
                &layer.down_proj,
            ]
        });
        [&self.embed, &self.output]
            .into_iter()
            .chain(layer_matrices)
            .any(|tensor| tensor.dtype().is_quantized())
    }

    /// Start a generation session with room for `capacity` positions.
    ///
    /// The key value cache is one flat allocation per tensor instead of one
    /// vector per layer. A real model has thirty layers, and two vectors per
    /// layer would exceed the kernel's user mapping table before the first
    /// token; two allocations stay constant in the layer count.
    pub fn session(&self, capacity: usize) -> Result<Session<'_>> {
        if capacity == 0 {
            return Err(Error::new("model: the context capacity is zero"));
        }
        let dims = self.dims;
        let kv_dim = dims
            .kv_heads
            .checked_mul(dims.head_dim)
            .ok_or_else(|| Error::new("model: key value width overflow"))?;
        let cache_len = dims
            .layers
            .checked_mul(capacity)
            .and_then(|layers| layers.checked_mul(kv_dim))
            .ok_or_else(|| Error::new("model: key value cache is too large"))?;
        Ok(Session {
            model: self,
            capacity,
            position: 0,
            keys: vec![0.0f32; cache_len],
            values: vec![0.0f32; cache_len],
            x: vec![0.0f32; dims.hidden],
            norm: vec![0.0f32; dims.hidden],
            norm_weight: vec![0.0f32; dims.hidden],
            q: vec![0.0f32; dims.q_heads * dims.head_dim],
            k: vec![0.0f32; kv_dim],
            v: vec![0.0f32; kv_dim],
            attn: vec![0.0f32; dims.q_heads * dims.head_dim],
            projected: vec![0.0f32; dims.hidden],
            gate: vec![0.0f32; dims.ffn],
            up: vec![0.0f32; dims.ffn],
            scores: vec![0.0f32; capacity],
            logits: vec![0.0f32; dims.vocab],
            emulated: vec![
                0.0f32;
                if self.emulate_llama {
                    dims.hidden.max(dims.ffn)
                } else {
                    0
                }
            ],
            emulate_llama: self.emulate_llama,
            matmul_macs: 0,
            matmul_nanos: 0,
        })
    }
}

/// One generation session with its key value cache and scratch buffers.
///
/// Every buffer is allocated once, when the session starts. The forward pass
/// does not allocate, so a kernel that cannot grow a large vector through
/// `realloc` still runs long sequences.
pub struct Session<'a> {
    model: &'a Model<'a>,
    capacity: usize,
    position: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
    x: Vec<f32>,
    norm: Vec<f32>,
    norm_weight: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    projected: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    scores: Vec<f32>,
    logits: Vec<f32>,
    /// The activation as `llama.cpp` packs it, when that is emulated.
    emulated: Vec<f32>,
    emulate_llama: bool,
    matmul_macs: u64,
    matmul_nanos: u128,
}

/// Time and count the matrix vector products of one forward pass.
///
/// The counters are plain fields because the runtime has one thread and one
/// session. They measure only the multiply accumulates: the norms, the
/// attention and the activation are outside the clock, so the reported rate can
/// be compared with the isolated benchmark.
struct MatmulClock {
    macs: u64,
    nanos: u128,
}

impl MatmulClock {
    fn new() -> Self {
        Self { macs: 0, nanos: 0 }
    }

    fn run<F>(&mut self, work: F) -> Result<()>
    where
        F: FnOnce() -> Result<u64>,
    {
        let started = Instant::now();
        let macs = work()?;
        self.nanos = self.nanos.saturating_add(started.elapsed().as_nanos());
        self.macs = self.macs.saturating_add(macs);
        Ok(())
    }
}

impl Session<'_> {
    /// Multiply accumulates the forward passes performed.
    pub fn matmul_macs(&self) -> u64 {
        self.matmul_macs
    }

    /// Nanoseconds spent inside the matrix vector products.
    pub fn matmul_nanos(&self) -> u128 {
        self.matmul_nanos
    }

    /// Run one token through the decoder and keep its logits.
    pub fn forward(&mut self, token: u32) -> Result<()> {
        let dims = self.model.dims;
        if token as usize >= dims.vocab {
            return Err(Error::new(format!(
                "model: token {token} is outside the vocabulary"
            )));
        }
        let positions = self
            .position
            .checked_add(1)
            .ok_or_else(|| Error::new("model: context position overflow"))?;
        if positions > self.capacity {
            return Err(Error::new("model: context exceeds the key value cache"));
        }
        let kv_dim = dims.kv_heads * dims.head_dim;
        let cache_stride = self.capacity * kv_dim;

        self.model.embed.read_row(token as usize, &mut self.x)?;
        let mut clock = MatmulClock::new();
        let emulate = self.emulate_llama;

        for (index, layer) in self.model.layers.iter().enumerate() {
            layer.input_norm.read_vector(&mut self.norm_weight)?;
            rms_norm(&self.x, &self.norm_weight, dims.eps, &mut self.norm)?;
            clock.run(|| {
                project_for(
                    emulate,
                    &layer.q_proj,
                    &self.norm,
                    &mut self.emulated,
                    &mut self.q,
                )
            })?;
            clock.run(|| {
                project_for(
                    emulate,
                    &layer.k_proj,
                    &self.norm,
                    &mut self.emulated,
                    &mut self.k,
                )
            })?;
            clock.run(|| {
                project_for(
                    emulate,
                    &layer.v_proj,
                    &self.norm,
                    &mut self.emulated,
                    &mut self.v,
                )
            })?;

            for head in 0..dims.q_heads {
                let start = head * dims.head_dim;
                rope(
                    &mut self.q[start..start + dims.head_dim],
                    self.position,
                    dims.rope_theta,
                    dims.rope_interleaved,
                )?;
            }
            for head in 0..dims.kv_heads {
                let start = head * dims.head_dim;
                rope(
                    &mut self.k[start..start + dims.head_dim],
                    self.position,
                    dims.rope_theta,
                    dims.rope_interleaved,
                )?;
            }

            let layer_base = index * cache_stride;
            let position_base = layer_base + self.position * kv_dim;
            self.keys[position_base..position_base + kv_dim].copy_from_slice(&self.k);
            self.values[position_base..position_base + kv_dim].copy_from_slice(&self.v);
            if emulate {
                llama_emulation::round_to_half(
                    &mut self.keys[position_base..position_base + kv_dim],
                );
                llama_emulation::round_to_half(
                    &mut self.values[position_base..position_base + kv_dim],
                );
            }

            self.attn.fill(0.0);
            let scale = 1.0 / (dims.head_dim as f32).sqrt();
            let group = dims.group_size();
            for head in 0..dims.q_heads {
                let kv_head = head / group;
                let q_start = head * dims.head_dim;
                let kv_start = kv_head * dims.head_dim;
                for (position, score) in self.scores[..positions].iter_mut().enumerate() {
                    let key =
                        &self.keys[layer_base + position * kv_dim + kv_start..][..dims.head_dim];
                    let mut sum = 0.0f32;
                    for element in 0..dims.head_dim {
                        sum += self.q[q_start + element] * key[element];
                    }
                    *score = sum * scale;
                }
                softmax(&mut self.scores[..positions])?;
                for position in 0..positions {
                    let value =
                        &self.values[layer_base + position * kv_dim + kv_start..][..dims.head_dim];
                    let weight = self.scores[position];
                    for element in 0..dims.head_dim {
                        self.attn[q_start + element] += weight * value[element];
                    }
                }
            }

            clock.run(|| {
                project_for(
                    emulate,
                    &layer.o_proj,
                    &self.attn,
                    &mut self.emulated,
                    &mut self.projected,
                )
            })?;
            for (slot, value) in self.x.iter_mut().zip(&self.projected) {
                *slot += value;
            }

            layer.post_norm.read_vector(&mut self.norm_weight)?;
            rms_norm(&self.x, &self.norm_weight, dims.eps, &mut self.norm)?;
            clock.run(|| {
                project_for(
                    emulate,
                    &layer.gate_proj,
                    &self.norm,
                    &mut self.emulated,
                    &mut self.gate,
                )
            })?;
            clock.run(|| {
                project_for(
                    emulate,
                    &layer.up_proj,
                    &self.norm,
                    &mut self.emulated,
                    &mut self.up,
                )
            })?;
            for (gate, up) in self.gate.iter_mut().zip(&self.up) {
                *gate = silu(*gate) * up;
            }
            clock.run(|| {
                project_for(
                    emulate,
                    &layer.down_proj,
                    &self.gate,
                    &mut self.emulated,
                    &mut self.projected,
                )
            })?;
            for (slot, value) in self.x.iter_mut().zip(&self.projected) {
                *slot += value;
            }
        }

        self.model.final_norm.read_vector(&mut self.norm_weight)?;
        rms_norm(&self.x, &self.norm_weight, dims.eps, &mut self.norm)?;
        clock.run(|| {
            project_for(
                emulate,
                &self.model.output,
                &self.norm,
                &mut self.emulated,
                &mut self.logits,
            )
        })?;
        self.matmul_macs = self.matmul_macs.saturating_add(clock.macs);
        self.matmul_nanos = self.matmul_nanos.saturating_add(clock.nanos);
        self.position = positions;
        Ok(())
    }

    /// Greedy sample: the largest logit wins, the first index wins a tie.
    pub fn sample(&self) -> Result<u32> {
        let index = argmax(&self.logits).ok_or_else(|| Error::new("model: no logits to sample"))?;
        u32::try_from(index).map_err(|_| Error::new("model: sampled token does not fit u32"))
    }

    /// The logits of the last forward pass, indexed by token id.
    ///
    /// The masked sampler in the runtime reads them to pick the largest value
    /// among the tokens a grammar allows; the plain sampler keeps using
    /// [`Self::sample`].
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }
}

/// [`project`], with the activation of `llama.cpp` when `emulate` is on.
///
/// `llama.cpp` packs the activation once per `mul_mat` and the packing is a
/// pure function of the input, so the products that share an input (the query,
/// the key and the value; the gate and the up projection) get the same packed
/// values here too, though each one packs its own copy.
fn project_for(
    emulate: bool,
    matrix: &Tensor<'_>,
    input: &[f32],
    scratch: &mut [f32],
    out: &mut [f32],
) -> Result<u64> {
    if emulate {
        if let Some(activation) = Activation::for_weight(matrix.dtype()) {
            let packed = scratch
                .get_mut(..input.len())
                .ok_or_else(|| Error::new("model: the emulation buffer is too short"))?;
            activation.apply(input, packed)?;
            return project(matrix, packed, out);
        }
    }
    project(matrix, input, out)
}

/// Multiply one matrix by a vector and return the multiply accumulates.
///
/// A `bfloat16` matrix goes to the blocked kernel in one contiguous range per
/// call. Only the rows that reach into the final partial page use the element
/// accessor, because those bytes are not part of the mapping.
fn project(matrix: &Tensor<'_>, input: &[f32], out: &mut [f32]) -> Result<u64> {
    let rows = matrix.shape().first().copied().unwrap_or(0);
    let columns = matrix.shape().last().copied().unwrap_or(0);
    if rows == 0 || columns == 0 || input.len() != columns || out.len() < rows {
        return Err(Error::new(format!(
            "model: projection of {} has a shape that disagrees with its buffers",
            matrix.name
        )));
    }
    if matrix.float_kernel {
        return project_float(matrix, input, out);
    }
    let mut mapped_rows = 0;
    if let Some((weights, rows_in_mapping)) = matrix.contiguous_quantized() {
        quantized_product(
            matrix.dtype(),
            weights,
            rows_in_mapping,
            columns,
            input,
            &mut out[..rows_in_mapping],
        )?;
        mapped_rows = rows_in_mapping;
    } else if let Some((weights, rows_in_mapping)) = matrix.contiguous_bf16() {
        let length = rows_in_mapping
            .checked_mul(columns)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| Error::new("model: contiguous weight range overflow"))?;
        let bytes = weights
            .get(..length)
            .ok_or_else(|| Error::new("model: contiguous weight range is short"))?;
        matvec::matvec_bf16(
            bytes,
            rows_in_mapping,
            columns,
            input,
            &mut out[..rows_in_mapping],
        )?;
        mapped_rows = rows_in_mapping;
    }
    if matrix.dtype().is_quantized() {
        project_quantized_tail(matrix, input, out, mapped_rows)?;
    } else {
        for (row, slot) in out.iter_mut().enumerate().take(rows).skip(mapped_rows) {
            *slot = matrix.dot(row, input)?;
        }
    }
    Ok(matvec::macs(rows, columns))
}

/// The rows of a quantized matrix from `first_row` on, which are not whole in
/// the mapping: each is copied out of the file and goes through the same kernel
/// as the mapped rows, so its bits do not depend on where the tensor sits.
fn project_quantized_tail(
    matrix: &Tensor<'_>,
    input: &[f32],
    out: &mut [f32],
    first_row: usize,
) -> Result<()> {
    let rows = matrix.shape().first().copied().unwrap_or(0);
    let columns = matrix.shape().last().copied().unwrap_or(0);
    let row_bytes = quantized_layout(matrix.dtype(), columns)?.0;
    let mut row_data = vec![0u8; row_bytes];
    for row in first_row..rows {
        matrix
            .mapping
            .copy_bytes(matrix.file_offset + row * row_bytes, &mut row_data)?;
        quantized_product(
            matrix.dtype(),
            &row_data,
            1,
            columns,
            input,
            &mut out[row..=row],
        )?;
    }
    Ok(())
}

/// The product of `rows` rows of a quantized matrix with `input`, by the kernel
/// of its type. The kernels of the GGUF types `Q5_0`, `Q4_K` and `Q6_K` are
/// scalar on every path, so the path only names the one that runs.
fn quantized_product(
    dtype: DType,
    weights: &[u8],
    rows: usize,
    columns: usize,
    input: &[f32],
    out: &mut [f32],
) -> Result<()> {
    match dtype {
        DType::Q4B32 => matvec::matvec_q4(weights, rows, columns, input, out),
        DType::Q8B32 => matvec::matvec_q8(weights, rows, columns, input, out),
        DType::Q5_0 => {
            matvec::matvec_q5_0_with(matvec::active(), weights, rows, columns, input, out)
        }
        DType::Q4K => {
            matvec::matvec_q4_k_with(matvec::active(), weights, rows, columns, input, out)
        }
        DType::Q6K => {
            matvec::matvec_q6_k_with(matvec::active(), weights, rows, columns, input, out)
        }
        _ => Err(Error::new("model: not a quantized matrix")),
    }
}

/// Multiply an `F32` or `F16` matrix of a GGUF file by a vector.
///
/// Every row goes through the same scalar kernel and so has the same order of
/// operations: the rows inside the mapping in one call, and the rows that reach
/// into the final partial page one at a time, copied out of the file first.
fn project_float(matrix: &Tensor<'_>, input: &[f32], out: &mut [f32]) -> Result<u64> {
    let rows = matrix.shape().first().copied().unwrap_or(0);
    let columns = matrix.shape().last().copied().unwrap_or(0);
    let row_bytes = columns
        .checked_mul(matrix.element_size)
        .ok_or_else(|| Error::new("model: row byte length overflow"))?;
    let product = |bytes: &[u8], count: usize, out: &mut [f32]| match matrix.dtype() {
        DType::F16 => matvec::matvec_f16(bytes, count, columns, input, out),
        _ => matvec::matvec_f32(bytes, count, columns, input, out),
    };
    let mut mapped_rows = 0;
    if let Some((weights, rows_in_mapping)) = matrix.contiguous_rows(row_bytes, rows) {
        product(weights, rows_in_mapping, &mut out[..rows_in_mapping])?;
        mapped_rows = rows_in_mapping;
    }
    if mapped_rows < rows {
        let mut row_data = vec![0u8; row_bytes];
        for row in mapped_rows..rows {
            matrix
                .mapping
                .copy_bytes(matrix.file_offset + row * row_bytes, &mut row_data)?;
            product(&row_data, 1, &mut out[row..=row])?;
        }
    }
    Ok(matvec::macs(rows, columns))
}

/// Bytes a quantized matrix occupies.
fn quantized_bytes(dtype: DType, rows: usize, columns: usize) -> Result<usize> {
    match dtype {
        DType::Q8B32 => quant::q8_matrix_bytes(rows, columns),
        DType::Q5_0 => quant::q5_0_matrix_bytes(rows, columns),
        DType::Q4K => quant::q4_k_matrix_bytes(rows, columns),
        DType::Q6K => quant::q6_k_matrix_bytes(rows, columns),
        _ => quant::matrix_bytes(rows, columns),
    }
}

/// Bytes of one row and of one block of a quantized tensor.
fn quantized_layout(dtype: DType, columns: usize) -> Result<(usize, usize)> {
    match dtype {
        DType::Q8B32 => Ok((quant::q8_row_bytes(columns)?, quant::Q8_BLOCK_BYTES)),
        DType::Q5_0 => Ok((quant::q5_0_row_bytes(columns)?, quant::Q5_0_BLOCK_BYTES)),
        DType::Q4K => Ok((quant::q4_k_row_bytes(columns)?, quant::Q4_K_BLOCK_BYTES)),
        DType::Q6K => Ok((quant::q6_k_row_bytes(columns)?, quant::Q6_K_BLOCK_BYTES)),
        _ => Ok((quant::row_bytes(columns)?, quant::BLOCK_BYTES)),
    }
}

/// Weights in one block of a quantized type.
fn quantized_block_elements(dtype: DType) -> usize {
    match dtype {
        DType::Q4K | DType::Q6K => quant::QK_K,
        _ => quant::BLOCK,
    }
}

/// Where the binary16 scales sit in a block of a quantized type: `d` first for
/// `Q4B32`, `Q8B32` and `Q5_0`, `d` and `dmin` for `Q4_K`, and `d` last for
/// `Q6_K`.
fn scale_offsets(dtype: DType) -> &'static [usize] {
    match dtype {
        DType::Q4K => &[0, 2],
        DType::Q6K => &[quant::Q6_K_BLOCK_BYTES - 2],
        _ => &[0],
    }
}

/// Count the decoder layers by the highest `model.layers.N` in the directory.
fn count_layers(file: &ModelFile) -> Result<usize> {
    let mut layers = 0usize;
    for tensor in file.header.tensors() {
        let name = file.header.name(tensor);
        let Some(rest) = name.strip_prefix("model.layers.") else {
            continue;
        };
        let Some((index, _)) = rest.split_once('.') else {
            continue;
        };
        let index = index
            .parse::<usize>()
            .map_err(|_| Error::new(format!("model: bad layer index in {name}")))?;
        layers = layers.max(index + 1);
    }
    Ok(layers)
}

fn layer_tensor<'a>(file: &'a ModelFile, index: usize, suffix: &str) -> Result<Tensor<'a>> {
    let name = format!("model.layers.{index}.{suffix}");
    file.tensor(&name)
}

fn require_matrix(tensor: &Tensor<'_>, rows: usize, cols: usize, name: &str) -> Result<()> {
    if tensor.shape() != [rows, cols] {
        return Err(Error::new(format!(
            "model: {name} has shape {:?}, expected [{rows}, {cols}]",
            tensor.shape()
        )));
    }
    Ok(())
}

fn require_vector(tensor: &Tensor<'_>, length: usize, name: &str) -> Result<()> {
    if tensor.shape() != [length] {
        return Err(Error::new(format!(
            "model: {name} has shape {:?}, expected [{length}]",
            tensor.shape()
        )));
    }
    if tensor.dtype().is_quantized() {
        return Err(Error::new(format!(
            "model: {name} is quantized, a norm weight stays floating point"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{
        dequantize_block_q8, f32_to_half, half_to_f32, BLOCK, Q8_BLOCK_BYTES, SCALE_BYTES,
    };
    use crate::synthetic::{
        gguf_builder, weights, GgufBuilder, GgufTensorOut, Meta, Permutation, Rng, Spec, Storage,
    };

    /// A model whose widths are multiples of the 32 weights of a block.
    fn spec() -> Spec {
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

    fn builder() -> GgufBuilder {
        let spec = spec();
        gguf_builder(
            &spec,
            &weights(&spec),
            Permutation::Converter,
            Storage::Q8_0,
        )
    }

    /// The file padded past its last page, so that every matrix lies inside the
    /// mapping and none is read through the row copy of the final partial page.
    fn mapped_bytes(file: &GgufBuilder) -> Vec<u8> {
        let mut bytes = file.to_bytes();
        bytes.resize(bytes.len().next_multiple_of(4096) + 4096, 0);
        bytes
    }

    /// `count` blocks with every signed byte, the two extremes included, and a
    /// finite scale of either sign that the quantizer would not have chosen.
    fn arbitrary_blocks(count: usize) -> Vec<u8> {
        let mut rng = Rng::new(0x0851_0ead);
        let mut bytes = Vec::with_capacity(count * Q8_BLOCK_BYTES);
        for block in 0..count {
            let scale = f32_to_half(rng.next_f32() * 3.0);
            bytes.extend_from_slice(&scale.to_le_bytes());
            for index in 0..BLOCK {
                bytes.push(match (block, index) {
                    (_, 0) => 0x80,
                    (_, 1) => 0x7f,
                    _ => ((rng.next_f32() * 128.0) as i32).clamp(-128, 127) as i8 as u8,
                });
            }
        }
        bytes
    }

    /// `dequantize_row_q8_0` of `ggml-quants.c` (ggml `d9fe7bf`) for one block:
    /// `y[j] = qs[j] * d`, with `d` the binary16 scale as an `f32`.
    fn ggml_dequantize_q8_0(block: &[u8]) -> Vec<f32> {
        let d = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
        block[SCALE_BYTES..]
            .iter()
            .map(|&quant| f32::from(quant as i8) * d)
            .collect()
    }

    #[test]
    fn a_q8_0_block_of_the_file_decodes_like_a_q8b32_block() {
        let spec = spec();
        let mut file = builder();
        let blocks = arbitrary_blocks(spec.vocab * spec.hidden / BLOCK);
        file.tensor("token_embd.weight").unwrap().data = blocks.clone();
        let model_file = ModelFile::gguf_from_bytes(file.to_bytes()).unwrap();
        let embedding = model_file.tensor("model.embed_tokens.weight").unwrap();
        assert_eq!(embedding.dtype(), DType::Q8B32);
        assert_eq!(embedding.shape(), [spec.vocab, spec.hidden]);
        let row_bytes = spec.hidden / BLOCK * Q8_BLOCK_BYTES;
        for row in 0..spec.vocab {
            let mut decoded = vec![0.0f32; spec.hidden];
            embedding.read_row(row, &mut decoded).unwrap();
            let mut expected = vec![0.0f32; spec.hidden];
            let mut ggml = Vec::new();
            let source = &blocks[row * row_bytes..(row + 1) * row_bytes];
            for (index, block) in source.as_chunks::<Q8_BLOCK_BYTES>().0.iter().enumerate() {
                dequantize_block_q8(block, &mut expected[index * BLOCK..(index + 1) * BLOCK])
                    .unwrap();
                ggml.extend(ggml_dequantize_q8_0(block));
            }
            let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&decoded), bits(&expected), "row {row}");
            assert_eq!(bits(&decoded), bits(&ggml), "row {row}");
        }
    }

    #[test]
    fn the_matrices_of_a_q8_0_file_go_through_the_q8_kernels() {
        let spec = spec();
        let file = builder();
        let source = |name: &str| {
            file.tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .unwrap()
                .data
                .clone()
        };
        let model_file = ModelFile::gguf_from_bytes(mapped_bytes(&file)).unwrap();
        assert!(model_file.mapped_len() >= model_file.file_len() - 4096);
        let mut rng = Rng::new(3);
        for (name, hf, rows, columns) in [
            (
                "blk.0.attn_q.weight",
                "model.layers.0.self_attn.q_proj.weight",
                spec.hidden,
                spec.hidden,
            ),
            (
                "blk.1.attn_k.weight",
                "model.layers.1.self_attn.k_proj.weight",
                spec.kv_dim(),
                spec.hidden,
            ),
            (
                "blk.1.ffn_down.weight",
                "model.layers.1.mlp.down_proj.weight",
                spec.hidden,
                spec.ffn,
            ),
            (
                "token_embd.weight",
                "model.embed_tokens.weight",
                spec.vocab,
                spec.hidden,
            ),
        ] {
            let tensor = model_file.tensor(hf).unwrap();
            assert_eq!(tensor.dtype(), DType::Q8B32, "{name}");
            assert!(!tensor.float_kernel, "{name}");
            let input: Vec<f32> = (0..columns).map(|_| rng.next_f32()).collect();
            let mut projected = vec![0.0f32; rows];
            assert_eq!(
                project(&tensor, &input, &mut projected).unwrap(),
                matvec::macs(rows, columns)
            );
            let mut scalar = vec![0.0f32; rows];
            matvec::matvec_q8_scalar(&source(name), rows, columns, &input, &mut scalar).unwrap();
            let mut best = vec![0.0f32; rows];
            matvec::matvec_q8(&source(name), rows, columns, &input, &mut best).unwrap();
            let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&projected), bits(&scalar), "{name}");
            assert_eq!(bits(&projected), bits(&best), "{name}");
        }
    }

    /// The paths the host offers, from the scalar one up. A path above what the
    /// processor has runs on the best one it has, so the list can name them all.
    fn paths() -> Vec<matvec::Path> {
        let mut paths = vec![matvec::Path::Scalar];
        #[cfg(target_arch = "x86_64")]
        paths.extend([matvec::Path::Sse2, matvec::Path::Avx2]);
        #[cfg(target_arch = "aarch64")]
        paths.push(matvec::Path::Neon);
        paths
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    /// A `Q8_0` file whose last tensor is `blk.1.ffn_down.weight`, with arbitrary
    /// blocks, and whose last row straddles the page the file does not fill: the
    /// final partial page holds the end of that row and no padding follows it.
    /// The header grows by whole alignments until the row starts in the mapping
    /// and ends in the tail.
    fn file_with_a_straddling_last_row() -> (Vec<u8>, Vec<u8>) {
        let spec = spec();
        let mut file = builder();
        let mut last = file.tensors.remove(
            file.tensors
                .iter()
                .position(|tensor| tensor.name == "blk.1.ffn_down.weight")
                .unwrap(),
        );
        last.data = arbitrary_blocks(last.data.len() / Q8_BLOCK_BYTES);
        let source = last.data.clone();
        file.tensors.push(last);
        let row_bytes = spec.ffn / BLOCK * Q8_BLOCK_BYTES;
        for extra in (0..8192).step_by(32) {
            *file.meta("general.name").unwrap() = Meta::Str("x".repeat(extra));
            let bytes = file.to_bytes();
            let model_file = ModelFile::gguf_from_bytes(bytes.clone()).unwrap();
            let tensor = model_file
                .tensor("model.layers.1.mlp.down_proj.weight")
                .unwrap();
            let start = tensor.file_offset + (spec.hidden - 1) * row_bytes;
            let end = start + row_bytes;
            if start < model_file.mapped_len() && model_file.mapped_len() < end {
                assert!(end <= model_file.file_len());
                assert!(model_file.file_len() - end < 32, "no padding past the row");
                return (bytes, source);
            }
        }
        panic!("no header length puts the last row across the end of the mapping");
    }

    #[test]
    fn the_last_row_of_a_q8_0_file_that_straddles_its_end_has_the_bits_of_every_path() {
        let spec = spec();
        let (bytes, source) = file_with_a_straddling_last_row();
        let model_file = ModelFile::gguf_from_bytes(bytes).unwrap();
        let tensor = model_file
            .tensor("model.layers.1.mlp.down_proj.weight")
            .unwrap();
        let (rows, columns) = (spec.hidden, spec.ffn);
        let mapped = tensor.contiguous_quantized().unwrap().1;
        assert_eq!(
            mapped,
            rows - 1,
            "only the last row is not whole in the mapping"
        );
        let mut rng = Rng::new(11);
        let input: Vec<f32> = (0..columns).map(|_| rng.next_f32()).collect();
        let mut projected = vec![0.0f32; rows];
        project(&tensor, &input, &mut projected).unwrap();
        let mut scalar = vec![0.0f32; rows];
        matvec::matvec_q8_scalar(&source, rows, columns, &input, &mut scalar).unwrap();
        assert_eq!(bits(&projected), bits(&scalar));
        let mut best = vec![0.0f32; rows];
        matvec::matvec_q8(&source, rows, columns, &input, &mut best).unwrap();
        assert_eq!(bits(&projected), bits(&best));
        for path in paths() {
            let mut out = vec![0.0f32; rows];
            matvec::matvec_q8_with(path, &source, rows, columns, &input, &mut out).unwrap();
            assert_eq!(bits(&projected), bits(&out), "{}", path.name());
        }
    }

    #[test]
    fn a_q8_0_row_has_the_same_bits_in_the_prefix_and_in_the_tail() {
        let spec = spec();
        let (bytes, _) = file_with_a_straddling_last_row();
        let mut padded = bytes.clone();
        padded.resize(padded.len().next_multiple_of(4096) + 4096, 0);
        let name = "model.layers.1.mlp.down_proj.weight";
        let in_tail = ModelFile::gguf_from_bytes(bytes).unwrap();
        let in_prefix = ModelFile::gguf_from_bytes(padded).unwrap();
        let (tail, prefix) = (
            in_tail.tensor(name).unwrap(),
            in_prefix.tensor(name).unwrap(),
        );
        assert!(tail.contiguous_quantized().unwrap().1 < spec.hidden);
        assert_eq!(prefix.contiguous_quantized().unwrap().1, spec.hidden);
        let mut rng = Rng::new(12);
        let input: Vec<f32> = (0..spec.ffn).map(|_| rng.next_f32()).collect();
        let mut from_tail = vec![0.0f32; spec.hidden];
        let mut from_prefix = vec![0.0f32; spec.hidden];
        project(&tail, &input, &mut from_tail).unwrap();
        project(&prefix, &input, &mut from_prefix).unwrap();
        assert_eq!(bits(&from_tail), bits(&from_prefix));
    }

    #[test]
    fn a_q4b32_row_has_the_bits_of_the_kernel_in_the_prefix_and_in_the_tail() {
        let (rows, columns) = (200usize, 64usize);
        let row_bytes = quant::row_bytes(columns).unwrap();
        let mut rng = Rng::new(0x04b3_2000);
        let mut payload = Vec::new();
        for _ in 0..rows * quant::blocks(columns) {
            payload.extend_from_slice(&f32_to_half(rng.next_f32() * 3.0).to_le_bytes());
            payload.extend((0..quant::PACKED_BYTES).map(|_| (rng.next_f32() * 256.0) as u8));
        }
        let file = |padding: usize| {
            let mut header = format!(
                "{{\"__metadata__\":{{\"format\":\"{}\",\"block\":\"32\",\"scale\":\"{}\"}},\
                 \"w\":{{\"dtype\":\"Q4B32\",\"shape\":[{rows},{columns}],\
                 \"data_offsets\":[0,{}]}}}}",
                quant::FORMAT_NAME,
                quant::SCALE_NAME,
                payload.len()
            );
            header.push_str(&" ".repeat(padding));
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend_from_slice(&payload);
            bytes
        };
        // Grow the header until the last row starts in the mapping and ends in the tail.
        let bytes = (0..4096)
            .map(file)
            .find(|bytes| {
                let end = bytes.len();
                end / 4096 * 4096 > end - row_bytes && end % 4096 != 0
            })
            .expect("a header length that puts the last row across the end of the mapping");
        let mut padded = bytes.clone();
        padded.resize(padded.len().next_multiple_of(4096) + 4096, 0);
        let in_tail = ModelFile::from_bytes(bytes).unwrap();
        let in_prefix = ModelFile::from_bytes(padded).unwrap();
        let (tail, prefix) = (in_tail.tensor("w").unwrap(), in_prefix.tensor("w").unwrap());
        assert_eq!(tail.contiguous_quantized().unwrap().1, rows - 1);
        assert_eq!(prefix.contiguous_quantized().unwrap().1, rows);
        let input: Vec<f32> = (0..columns).map(|_| rng.next_f32()).collect();
        let mut from_tail = vec![0.0f32; rows];
        let mut from_prefix = vec![0.0f32; rows];
        project(&tail, &input, &mut from_tail).unwrap();
        project(&prefix, &input, &mut from_prefix).unwrap();
        let mut scalar = vec![0.0f32; rows];
        matvec::matvec_q4_scalar(&payload, rows, columns, &input, &mut scalar).unwrap();
        assert_eq!(bits(&from_tail), bits(&scalar));
        assert_eq!(bits(&from_prefix), bits(&scalar));
    }

    #[test]
    fn a_gguf_table_with_a_dtype_the_reader_does_not_map_is_refused() {
        let mut model_file = ModelFile::gguf_from_bytes(builder().to_bytes()).unwrap();
        model_file.validate_quantization().unwrap();
        model_file
            .header
            .push("extra.weight", DType::Q4B32, &[1, 32], 0, 18)
            .unwrap();
        let error = model_file.validate_quantization().unwrap_err();
        let shown = error.to_string();
        assert!(
            shown.contains("extra.weight") && shown.contains("Q4B32"),
            "{shown}"
        );
    }

    #[test]
    fn a_q8_0_model_binds_and_runs_tied_and_untied() {
        let spec = spec();
        let model_file = ModelFile::gguf_from_bytes(mapped_bytes(&builder())).unwrap();
        let config = model_file.gguf_config().unwrap();
        assert!(config.tie_word_embeddings);
        let model = Model::new(&model_file, config).unwrap();
        assert_eq!(model.weight_dtype(), "Q8B32");
        let mut session = model.session(8).unwrap();
        for token in [1u32, 2, 3] {
            session.forward(token).unwrap();
        }
        assert_eq!(session.logits().len(), spec.vocab);
        assert!(session.logits().iter().all(|value| value.is_finite()));
        let tied = session.logits().to_vec();

        // The output matrix of its own, also `Q8_0`, with the rows of the embedding.
        let mut file = builder();
        let embedding = file.tensor("token_embd.weight").unwrap().data.clone();
        file.tensors.push(GgufTensorOut {
            name: "output.weight".into(),
            dims: vec![spec.hidden as u64, spec.vocab as u64],
            ggml_type: 8,
            data: embedding,
        });
        let model_file = ModelFile::gguf_from_bytes(mapped_bytes(&file)).unwrap();
        let config = model_file.gguf_config().unwrap();
        assert!(!config.tie_word_embeddings);
        let model = Model::new(&model_file, config).unwrap();
        let mut session = model.session(8).unwrap();
        for token in [1u32, 2, 3] {
            session.forward(token).unwrap();
        }
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(session.logits()), bits(&tied));
    }

    #[test]
    fn a_q8_0_block_with_a_scale_that_is_not_finite_is_refused() {
        let mut file = builder();
        let tensor = file.tensor("blk.1.ffn_up.weight").unwrap();
        tensor.data[34..36].copy_from_slice(&0x7c00u16.to_le_bytes());
        let model_file = ModelFile::gguf_from_bytes(file.to_bytes()).unwrap();
        let config = model_file.gguf_config().unwrap();
        let error = Model::new(&model_file, config).err().unwrap();
        assert!(error.to_string().contains("non-finite scale"), "{error}");
    }

    /// The types of the GGUF files that `llama-quantize` writes besides `Q8_0`:
    /// the `ggml_type` code, the dtype of the model, the weights and the bytes of
    /// a block and where the binary16 scales are in it.
    const BLOCK_TYPES: [(u32, DType, usize, usize, &[usize]); 3] = [
        (6, DType::Q5_0, 32, 22, &[0]),
        (12, DType::Q4K, 256, 144, &[0, 2]),
        (14, DType::Q6K, 256, 210, &[208]),
    ];

    /// A model whose widths are multiples of the 256 weights of a `Q4_K` block.
    fn super_spec() -> Spec {
        Spec {
            hidden: 256,
            layers: 1,
            heads: 4,
            kv_heads: 2,
            ffn: 256,
            vocab: 32,
            eps: 1e-5,
            theta: 10_000.0,
        }
    }

    /// `count` blocks of arbitrary bytes with a finite binary16 at each scale.
    fn random_blocks(count: usize, block_bytes: usize, scales: &[usize], seed: u64) -> Vec<u8> {
        let mut rng = Rng::new(seed);
        let mut bytes = Vec::with_capacity(count * block_bytes);
        for _ in 0..count {
            let start = bytes.len();
            bytes.extend((0..block_bytes).map(|_| (rng.next_f32() * 128.0 + 128.0) as u8));
            for &scale in scales {
                let half = f32_to_half(rng.next_f32() * 0.5).to_le_bytes();
                bytes[start + scale..start + scale + 2].copy_from_slice(&half);
            }
        }
        bytes
    }

    /// The file of `super_spec` with every matrix in the given type, with
    /// arbitrary blocks, and the last tensor `blk.0.ffn_down.weight`.
    fn typed_builder(ggml_type: u32, block: (usize, usize, &[usize])) -> GgufBuilder {
        let spec = super_spec();
        let mut file = gguf_builder(&spec, &weights(&spec), Permutation::Converter, Storage::F32);
        for (index, tensor) in file.tensors.iter_mut().enumerate() {
            if tensor.dims.len() == 2 {
                let elements = tensor.dims.iter().product::<u64>() as usize;
                tensor.ggml_type = ggml_type;
                tensor.data =
                    random_blocks(elements / block.0, block.1, block.2, 0x5eed + index as u64);
            }
        }
        let last = file
            .tensors
            .iter()
            .position(|tensor| tensor.name == "blk.0.ffn_down.weight")
            .unwrap();
        let last = file.tensors.remove(last);
        file.tensors.push(last);
        file
    }

    fn source(file: &GgufBuilder, name: &str) -> Vec<u8> {
        file.tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap()
            .data
            .clone()
    }

    /// The product of the scalar kernel of a type, the definition of the order.
    fn scalar_product(
        dtype: DType,
        weights: &[u8],
        rows: usize,
        columns: usize,
        input: &[f32],
        out: &mut [f32],
    ) {
        match dtype {
            DType::Q5_0 => matvec::matvec_q5_0_scalar(weights, rows, columns, input, out),
            DType::Q4K => matvec::matvec_q4_k_scalar(weights, rows, columns, input, out),
            DType::Q6K => matvec::matvec_q6_k_scalar(weights, rows, columns, input, out),
            other => panic!("no scalar kernel for {}", other.name()),
        }
        .unwrap();
    }

    #[test]
    fn the_rows_of_a_q5_0_q4_k_or_q6_k_embedding_decode_like_gguf_py() {
        use crate::quant::oracle;
        let spec = super_spec();
        for (ggml_type, dtype, elements, block_bytes, scales) in BLOCK_TYPES {
            let file = typed_builder(ggml_type, (elements, block_bytes, scales));
            let model_file = ModelFile::gguf_from_bytes(mapped_bytes(&file)).unwrap();
            let embedding = model_file.tensor("model.embed_tokens.weight").unwrap();
            assert_eq!(embedding.dtype(), dtype);
            assert_eq!(embedding.shape(), [spec.vocab, spec.hidden]);
            let blocks = source(&file, "token_embd.weight");
            let row_bytes = spec.hidden / elements * block_bytes;
            for row in 0..spec.vocab {
                let mut decoded = vec![0.0f32; spec.hidden];
                embedding.read_row(row, &mut decoded).unwrap();
                let source = &blocks[row * row_bytes..(row + 1) * row_bytes];
                let expected: Vec<f32> = (0..spec.hidden)
                    .map(|index| {
                        let block = &source[index / elements * block_bytes..][..block_bytes];
                        match dtype {
                            DType::Q5_0 => oracle::q5_0_element(block, index % elements),
                            DType::Q4K => oracle::q4_k_element(block, index % elements),
                            _ => oracle::q6_k_element(block, index % elements),
                        }
                    })
                    .collect();
                assert_eq!(
                    bits(&decoded),
                    bits(&expected),
                    "{} row {row}",
                    dtype.name()
                );
            }
        }
    }

    #[test]
    fn the_matrices_of_a_q5_0_q4_k_or_q6_k_file_go_through_their_scalar_kernels() {
        let spec = super_spec();
        for (ggml_type, dtype, elements, block_bytes, scales) in BLOCK_TYPES {
            let file = typed_builder(ggml_type, (elements, block_bytes, scales));
            let model_file = ModelFile::gguf_from_bytes(mapped_bytes(&file)).unwrap();
            assert!(model_file.mapped_len() >= model_file.file_len() - 4096);
            let mut rng = Rng::new(21);
            for (name, hf, rows, columns) in [
                (
                    "blk.0.attn_q.weight",
                    "model.layers.0.self_attn.q_proj.weight",
                    spec.hidden,
                    spec.hidden,
                ),
                (
                    "blk.0.attn_k.weight",
                    "model.layers.0.self_attn.k_proj.weight",
                    spec.kv_dim(),
                    spec.hidden,
                ),
                (
                    "blk.0.ffn_down.weight",
                    "model.layers.0.mlp.down_proj.weight",
                    spec.hidden,
                    spec.ffn,
                ),
                (
                    "token_embd.weight",
                    "model.embed_tokens.weight",
                    spec.vocab,
                    spec.hidden,
                ),
            ] {
                let tensor = model_file.tensor(hf).unwrap();
                assert_eq!(tensor.dtype(), dtype, "{name}");
                assert!(!tensor.float_kernel, "{name}");
                assert_eq!(tensor.contiguous_quantized().unwrap().1, rows, "{name}");
                let input: Vec<f32> = (0..columns).map(|_| rng.next_f32()).collect();
                let mut projected = vec![0.0f32; rows];
                assert_eq!(
                    project(&tensor, &input, &mut projected).unwrap(),
                    matvec::macs(rows, columns)
                );
                let mut scalar = vec![0.0f32; rows];
                scalar_product(
                    dtype,
                    &source(&file, name),
                    rows,
                    columns,
                    &input,
                    &mut scalar,
                );
                assert_eq!(bits(&projected), bits(&scalar), "{} {name}", dtype.name());
                assert!(projected.iter().all(|value| value.is_finite()));
            }
        }
    }

    /// A file of the given type whose last tensor, `blk.0.ffn_down.weight`, has its
    /// last row across the end of the mapping, with no padding after it.
    fn file_with_a_straddling_last_row_of(
        ggml_type: u32,
        block: (usize, usize, &[usize]),
    ) -> Vec<u8> {
        let spec = super_spec();
        let mut file = typed_builder(ggml_type, block);
        let row_bytes = spec.ffn / block.0 * block.1;
        for extra in (0..8192).step_by(32) {
            *file.meta("general.name").unwrap() = Meta::Str("x".repeat(extra));
            let bytes = file.to_bytes();
            let model_file = ModelFile::gguf_from_bytes(bytes.clone()).unwrap();
            let tensor = model_file
                .tensor("model.layers.0.mlp.down_proj.weight")
                .unwrap();
            let start = tensor.file_offset + (spec.hidden - 1) * row_bytes;
            let end = start + row_bytes;
            if start < model_file.mapped_len() && model_file.mapped_len() < end {
                assert!(end <= model_file.file_len());
                assert!(model_file.file_len() - end < 32, "no padding past the row");
                return bytes;
            }
        }
        panic!("no header length puts the last row across the end of the mapping");
    }

    #[test]
    fn the_last_row_of_a_q5_0_q4_k_or_q6_k_file_has_the_same_bits_in_the_tail_and_the_prefix() {
        let spec = super_spec();
        let name = "model.layers.0.mlp.down_proj.weight";
        for (ggml_type, dtype, elements, block_bytes, scales) in BLOCK_TYPES {
            let block = (elements, block_bytes, scales);
            let bytes = file_with_a_straddling_last_row_of(ggml_type, block);
            let mut padded = bytes.clone();
            padded.resize(padded.len().next_multiple_of(4096) + 4096, 0);
            let in_tail = ModelFile::gguf_from_bytes(bytes).unwrap();
            let in_prefix = ModelFile::gguf_from_bytes(padded).unwrap();
            let (tail, prefix) = (
                in_tail.tensor(name).unwrap(),
                in_prefix.tensor(name).unwrap(),
            );
            assert_eq!(tail.dtype(), dtype);
            assert_eq!(tail.contiguous_quantized().unwrap().1, spec.hidden - 1);
            assert_eq!(prefix.contiguous_quantized().unwrap().1, spec.hidden);
            let mut rng = Rng::new(22);
            let input: Vec<f32> = (0..spec.ffn).map(|_| rng.next_f32()).collect();
            let mut from_tail = vec![0.0f32; spec.hidden];
            let mut from_prefix = vec![0.0f32; spec.hidden];
            project(&tail, &input, &mut from_tail).unwrap();
            project(&prefix, &input, &mut from_prefix).unwrap();
            assert_eq!(bits(&from_tail), bits(&from_prefix), "{}", dtype.name());
            let file = typed_builder(ggml_type, block);
            let mut scalar = vec![0.0f32; spec.hidden];
            scalar_product(
                dtype,
                &source(&file, "blk.0.ffn_down.weight"),
                spec.hidden,
                spec.ffn,
                &input,
                &mut scalar,
            );
            assert_eq!(bits(&from_tail), bits(&scalar), "{}", dtype.name());
        }
    }

    #[test]
    fn a_q5_0_q4_k_or_q6_k_block_with_a_scale_that_is_not_finite_is_refused() {
        for (ggml_type, dtype, elements, block_bytes, scales) in BLOCK_TYPES {
            for &scale in scales {
                for (name, hf, block_index) in [
                    (
                        "blk.0.ffn_up.weight",
                        "model.layers.0.mlp.up_proj.weight",
                        3usize,
                    ),
                    ("token_embd.weight", "model.embed_tokens.weight", 0),
                ] {
                    let mut file = typed_builder(ggml_type, (elements, block_bytes, scales));
                    let tensor = file.tensor(name).unwrap();
                    let at = block_index * block_bytes + scale;
                    tensor.data[at..at + 2].copy_from_slice(&0x7c00u16.to_le_bytes());
                    let model_file = ModelFile::gguf_from_bytes(file.to_bytes()).unwrap();
                    let config = model_file.gguf_config().unwrap();
                    let error = Model::new(&model_file, config).err().unwrap();
                    let shown = error.to_string();
                    assert!(
                        shown.contains("non-finite scale") && shown.contains(hf),
                        "{} {name} scale at {scale}: {shown}",
                        dtype.name()
                    );
                }
            }
        }
    }

    #[test]
    fn a_q5_0_q4_k_or_q6_k_model_binds_and_runs_tied_and_untied() {
        let spec = super_spec();
        for (ggml_type, dtype, elements, block_bytes, scales) in BLOCK_TYPES {
            let file = typed_builder(ggml_type, (elements, block_bytes, scales));
            let model_file = ModelFile::gguf_from_bytes(mapped_bytes(&file)).unwrap();
            let config = model_file.gguf_config().unwrap();
            assert!(config.tie_word_embeddings);
            let model = Model::new(&model_file, config).unwrap();
            assert_eq!(model.weight_dtype(), dtype.name());
            let mut session = model.session(8).unwrap();
            for token in [1u32, 2, 3] {
                session.forward(token).unwrap();
            }
            assert_eq!(session.logits().len(), spec.vocab);
            assert!(session.logits().iter().all(|value| value.is_finite()));
            let tied = session.logits().to_vec();

            // An output matrix of its own with the rows of the embedding.
            let mut untied = file.clone();
            let embedding = untied.tensor("token_embd.weight").unwrap().clone();
            untied.tensors.push(GgufTensorOut {
                name: "output.weight".into(),
                ..embedding
            });
            let model_file = ModelFile::gguf_from_bytes(mapped_bytes(&untied)).unwrap();
            let config = model_file.gguf_config().unwrap();
            assert!(!config.tie_word_embeddings);
            let model = Model::new(&model_file, config).unwrap();
            let mut session = model.session(8).unwrap();
            for token in [1u32, 2, 3] {
                session.forward(token).unwrap();
            }
            assert_eq!(bits(session.logits()), bits(&tied), "{}", dtype.name());
        }
    }

    #[test]
    fn a_quantized_norm_is_refused_when_the_model_binds_it() {
        // `gguf_model` refuses it first; the model refuses it too if a table
        // reaches it, as a `safetensors` header can name one.
        let file = typed_builder(12, (256, 144, &[0, 2]));
        let mut model_file = ModelFile::gguf_from_bytes(file.to_bytes()).unwrap();
        model_file
            .header
            .push("norm.weight", DType::Q4K, &[1, 256], 0, 144)
            .unwrap();
        model_file.validate_quantization().unwrap();
        let norm = model_file.tensor("norm.weight").unwrap();
        assert!(require_vector(&norm, 256, "norm.weight").is_err());
    }
}
