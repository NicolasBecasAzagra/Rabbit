// SPDX-License-Identifier: GPL-2.0-only
//! The `Q4B32` block format: four bit weights with one scale per block.
//!
//! A matrix is stored as a sequence of blocks of [`BLOCK`] consecutive weights
//! along a row. A block is a two byte IEEE binary16 scale followed by
//! [`PACKED_BYTES`] bytes of nibbles, so it costs 4.5 bits per weight. The
//! quantized weight is `q * scale` with `q` a signed four bit integer in
//! `-8..=7`, and the scale is the largest absolute value of the block divided
//! by seven, so the largest weight of a block lands on `+7` and the smallest
//! step of the block is `max_abs / 7`.
//!
//! The scale is binary16 and not binary32 because the two extra bytes per block
//! would bring the format to five bits per weight, which is the size the
//! milestone rules out for seven billion parameters. Binary16 keeps ten
//! mantissa bits, which is far more precision than the four bits of the weight:
//! rounding the scale moves a block by at most `2^-11` of its range, while
//! rounding the weight moves it by up to `1/14` of its range.
//!
//! The layout is documented for another implementation in
//! `docs/QUANTIZATION.md`. This module owns everything the format means: the
//! half precision conversion, the block packing, the block unpacking, and the
//! exact number of bytes a matrix occupies. It has no I/O and no architecture
//! specific code, so it is tested on the host.
//!
//! # The GGUF block types `Q5_0`, `Q4_K` and `Q6_K`
//!
//! The layouts are those of ggml, at the commit `d9fe7bf` of
//! <https://github.com/ggml-org/ggml>:
//!
//! * `block_q5_0` is `{ d: f16, qh: [u8; 4], qs: [u8; 16] }`, 22 bytes for 32
//!   weights; `block_q4_K` is `{ d: f16, dmin: f16, scales: [u8; 12],
//!   qs: [u8; 128] }`, 144 bytes for 256 weights; `block_q6_K` is
//!   `{ ql: [u8; 128], qh: [u8; 64], scales: [i8; 16], d: f16 }`, 210 bytes for
//!   256 weights. `QK_K` is 256 and `K_SCALE_SIZE` is 12.
//!   <https://raw.githubusercontent.com/ggml-org/ggml/d9fe7bf/src/ggml-common.h>
//! * The element order and the float operations are those of
//!   `dequantize_row_q5_0`, `dequantize_row_q4_K` (with `get_scale_min_k4`) and
//!   `dequantize_row_q6_K`.
//!   <https://raw.githubusercontent.com/ggml-org/ggml/d9fe7bf/src/ggml-quants.c>
//! * The bits that the tests compare with come from `gguf-py` (numpy, `float32`)
//!   at the commit `79c69f4b06805fe53c91596de6c958f86dbbdf41` of llama.cpp, the
//!   `dequantize_blocks` of `Q5_0`, `Q4_K` and `Q6_K`.
//!   <https://raw.githubusercontent.com/ggml-org/llama.cpp/79c69f4b06805fe53c91596de6c958f86dbbdf41/gguf-py/gguf/quants.py>
//!
//! The C source and `gguf-py` read the same `float32` operations in the same
//! order, so they define the same bits: `Q5_0` is `(q - 16) as f32 * d`; `Q4_K`
//! is `(d * sc) * q - (dmin * m)` with `d * sc` and `dmin * m` rounded to `f32`
//! first; `Q6_K` is `(d * scale) * q` with `d * scale` rounded first. A C
//! compiler that contracts the `a * b - c` of `Q4_K` into a fused multiply add
//! could differ from numpy in the last bit; this module never fuses, like numpy,
//! and the vectors of `bench/gguf` come from numpy.

use crate::error::{Error, Result};

/// Weights in one block.
pub const BLOCK: usize = 32;

/// Bytes of the binary16 scale of a block.
pub const SCALE_BYTES: usize = 2;

/// Bytes of the sixteen packed nibbles of a block.
pub const PACKED_BYTES: usize = BLOCK / 2;

/// Bytes of one complete block.
pub const BLOCK_BYTES: usize = SCALE_BYTES + PACKED_BYTES;

/// Bytes of the eight bit payload of a block.
///
/// [`Q8B32`](Q8_BLOCK_BYTES) is the same block layout at one byte per weight.
/// It exists because the four bit block does not keep the greedy answer of
/// every checkpoint: the eight bit form is the one the runtime produces when
/// the answer has to stay identical, and it still halves the weights against
/// `bfloat16`.
pub const Q8_PACKED_BYTES: usize = BLOCK;

/// Bytes of one eight bit block: the binary16 scale and the signed bytes.
pub const Q8_BLOCK_BYTES: usize = SCALE_BYTES + Q8_PACKED_BYTES;

/// Name of the eight bit format as the checkpoint metadata spells it.
pub const Q8_FORMAT_NAME: &str = "rabbit-q8b32";

/// Largest eight bit quantized value.
const Q8_MAX_QUANT: f32 = 127.0;

/// Larger magnitudes are clamped to the positive end of the four bit range.
const MAX_QUANT: f32 = 7.0;

/// Smallest quantized value, kept for the documentation of the range.
const MIN_QUANT: i32 = -8;

/// Largest quantized value.
const MAX_QUANT_INT: i32 = 7;

/// Name of the format as the checkpoint metadata spells it.
pub const FORMAT_NAME: &str = "rabbit-q4b32";

/// Name of the scale type as the checkpoint metadata spells it.
pub const SCALE_NAME: &str = "f16";

/// Number of blocks one row of `columns` weights occupies.
pub fn blocks(columns: usize) -> usize {
    columns.div_ceil(BLOCK)
}

/// Bytes one row of `columns` weights occupies.
///
/// `columns` must be a multiple of [`BLOCK`]: the format has no partial block,
/// so a matrix with a ragged row is refused instead of padded.
pub fn row_bytes(columns: usize) -> Result<usize> {
    if columns == 0 || !columns.is_multiple_of(BLOCK) {
        return Err(Error::new(format!(
            "quant: {columns} columns is not a multiple of the {BLOCK} weight block"
        )));
    }
    Ok(blocks(columns) * BLOCK_BYTES)
}

/// Bytes a `rows` by `columns` matrix occupies.
pub fn matrix_bytes(rows: usize, columns: usize) -> Result<usize> {
    let row = row_bytes(columns)?;
    rows.checked_mul(row)
        .ok_or_else(|| Error::new("quant: matrix byte length overflow"))
}

/// Bytes one row of `columns` weights occupies in the eight bit form.
pub fn q8_row_bytes(columns: usize) -> Result<usize> {
    if columns == 0 || !columns.is_multiple_of(BLOCK) {
        return Err(Error::new(format!(
            "quant: {columns} columns is not a multiple of the {BLOCK} weight block"
        )));
    }
    Ok(blocks(columns) * Q8_BLOCK_BYTES)
}

/// Bytes a `rows` by `columns` eight bit matrix occupies.
pub fn q8_matrix_bytes(rows: usize, columns: usize) -> Result<usize> {
    let row = q8_row_bytes(columns)?;
    rows.checked_mul(row)
        .ok_or_else(|| Error::new("quant: matrix byte length overflow"))
}

/// Convert an IEEE binary16 bit pattern to the nearest `f32`.
///
/// Every binary16 value is exactly representable in `f32`, so the conversion is
/// exact for the whole range, subnormals included.
pub fn half_to_f32(half: u16) -> f32 {
    let sign = ((half as u32) & 0x8000) << 16;
    let exponent = ((half >> 10) & 0x1f) as u32;
    let mantissa = (half & 0x03ff) as u32;
    let bits = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            // Subnormal: normalize the mantissa and lower the exponent.
            let mut value = mantissa;
            let mut shift = 0u32;
            while value & 0x0400 == 0 {
                value <<= 1;
                shift += 1;
            }
            sign | ((113 - shift) << 23) | ((value & 0x03ff) << 13)
        }
    } else if exponent == 0x1f {
        // Infinity or a quiet NaN, both with a non-zero mantissa preserved.
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((exponent + 127 - 15) << 23) | (mantissa << 13)
    };
    f32::from_bits(bits)
}

/// Convert `value` to the nearest binary16 bit pattern, ties to even.
///
/// The quantizer uses the rounded value the decoder will read, so a scale that
/// binary16 cannot hold exactly does not bias the nibbles it produces.
pub fn f32_to_half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0xff {
        return sign
            | 0x7c00
            | if mantissa == 0 {
                0
            } else {
                0x0200 | ((mantissa >> 13) as u16)
            };
    }
    let unbiased = exponent - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased >= -14 {
        let half_exponent = ((unbiased + 15) as u32) << 10;
        let half_mantissa = mantissa >> 13;
        let remainder = mantissa & 0x1fff;
        let mut result = half_exponent | half_mantissa;
        if remainder > 0x1000 || (remainder == 0x1000 && half_mantissa & 1 == 1) {
            result += 1;
        }
        return sign | result as u16;
    }
    if unbiased < -25 {
        return sign;
    }
    // Subnormal: the target mantissa is `m * 2^24`, where `m` is the value.
    let full = mantissa | 0x0080_0000;
    let shift = (-1 - unbiased) as u32;
    let truncated = full >> shift;
    let remainder = full & ((1u32 << shift) - 1);
    let halfway = 1u32 << (shift - 1);
    let mut result = truncated;
    if remainder > halfway || (remainder == halfway && truncated & 1 == 1) {
        result += 1;
    }
    sign | result as u16
}

/// Quantize one block of `BLOCK` values into one block of the file.
///
/// The scale is stored first, as binary16, then the nibbles: the low nibble of
/// byte `2 + i` is weight `2i` and the high nibble is weight `2i + 1`. A nibble
/// is the two's complement four bit value of the quantized weight.
pub fn quantize_block(values: &[f32], out: &mut [u8]) -> Result<()> {
    if values.len() != BLOCK || out.len() != BLOCK_BYTES {
        return Err(Error::new(format!(
            "quant: a block is {BLOCK} values into {BLOCK_BYTES} bytes"
        )));
    }
    let mut largest = 0.0f32;
    for &value in values {
        if !value.is_finite() {
            return Err(Error::new("quant: a weight is not finite"));
        }
        largest = largest.max(value.abs());
    }
    // The scale used for the nibbles is the one the decoder reads back, so the
    // largest weight uses the whole positive range instead of an unrounded
    // approximation of it.
    let scale = if largest == 0.0 {
        0.0
    } else {
        half_to_f32(f32_to_half(largest / MAX_QUANT))
    };
    out[..SCALE_BYTES].copy_from_slice(&f32_to_half(largest / MAX_QUANT).to_le_bytes());
    let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
    out[SCALE_BYTES..].fill(0);
    for (index, &value) in values.iter().enumerate() {
        let quantized = if scale == 0.0 {
            0
        } else {
            (value * inverse)
                .round()
                .clamp(MIN_QUANT as f32, MAX_QUANT_INT as f32) as i32
        };
        let nibble = (quantized & 0x0f) as u8;
        let shift = 4 * (index % 2);
        out[SCALE_BYTES + index / 2] |= nibble << shift;
    }
    Ok(())
}

/// Decode one block into `BLOCK` values.
pub fn dequantize_block(block: &[u8], out: &mut [f32]) -> Result<()> {
    if block.len() != BLOCK_BYTES || out.len() != BLOCK {
        return Err(Error::new(format!(
            "quant: a block is {BLOCK_BYTES} bytes into {BLOCK} values"
        )));
    }
    let scale = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
    if !scale.is_finite() {
        return Err(Error::new("quant: a block scale is not finite"));
    }
    for (index, slot) in out.iter_mut().enumerate() {
        let packed = block[SCALE_BYTES + index / 2];
        let nibble = (packed >> (4 * (index % 2))) & 0x0f;
        let value = ((nibble as i8) << 4 >> 4) as f32;
        *slot = value * scale;
    }
    Ok(())
}

/// Decode a whole row of blocks into `out`.
pub fn dequantize_row(blocks: &[u8], out: &mut [f32]) -> Result<()> {
    let columns = out.len();
    let expected = row_bytes(columns)?;
    if blocks.len() != expected {
        return Err(Error::new(format!(
            "quant: a row of {columns} weights is {expected} bytes, not {}",
            blocks.len()
        )));
    }
    for (index, chunk) in out.as_chunks_mut::<BLOCK>().0.iter_mut().enumerate() {
        let start = index * BLOCK_BYTES;
        dequantize_block(&blocks[start..start + BLOCK_BYTES], chunk)?;
    }
    Ok(())
}

/// `true` when a binary16 bit pattern is finite.
pub fn scale_is_finite(half: u16) -> bool {
    half & 0x7c00 != 0x7c00
}

/// Quantize one block of `BLOCK` values into one eight bit block.
///
/// The scale is the largest magnitude of the block divided by 127, rounded to
/// binary16, and each weight is one signed byte of the block. The layout is the
/// four bit one with a wider payload, so a reader that knows the block size
/// reads both with the same loop.
pub fn quantize_block_q8(values: &[f32], out: &mut [u8]) -> Result<()> {
    if values.len() != BLOCK || out.len() != Q8_BLOCK_BYTES {
        return Err(Error::new(format!(
            "quant: an eight bit block is {BLOCK} values into {Q8_BLOCK_BYTES} bytes"
        )));
    }
    let mut largest = 0.0f32;
    for &value in values {
        if !value.is_finite() {
            return Err(Error::new("quant: a weight is not finite"));
        }
        largest = largest.max(value.abs());
    }
    let scale = if largest == 0.0 {
        0.0
    } else {
        half_to_f32(f32_to_half(largest / Q8_MAX_QUANT))
    };
    out[..SCALE_BYTES].copy_from_slice(&f32_to_half(largest / Q8_MAX_QUANT).to_le_bytes());
    let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
    for (index, &value) in values.iter().enumerate() {
        let quantized = if scale == 0.0 {
            0
        } else {
            (value * inverse).round().clamp(-128.0, 127.0) as i8
        };
        out[SCALE_BYTES + index] = quantized as u8;
    }
    Ok(())
}

/// Decode one eight bit block into `BLOCK` values.
pub fn dequantize_block_q8(block: &[u8], out: &mut [f32]) -> Result<()> {
    if block.len() != Q8_BLOCK_BYTES || out.len() != BLOCK {
        return Err(Error::new(format!(
            "quant: an eight bit block is {Q8_BLOCK_BYTES} bytes into {BLOCK} values"
        )));
    }
    let scale = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
    if !scale.is_finite() {
        return Err(Error::new("quant: a block scale is not finite"));
    }
    for (index, slot) in out.iter_mut().enumerate() {
        *slot = (block[SCALE_BYTES + index] as i8) as f32 * scale;
    }
    Ok(())
}

/// Decode a whole eight bit row of blocks into `out`.
pub fn dequantize_row_q8(blocks: &[u8], out: &mut [f32]) -> Result<()> {
    let columns = out.len();
    let expected = q8_row_bytes(columns)?;
    if blocks.len() != expected {
        return Err(Error::new(format!(
            "quant: an eight bit row of {columns} weights is {expected} bytes, not {}",
            blocks.len()
        )));
    }
    for (index, chunk) in out.as_chunks_mut::<BLOCK>().0.iter_mut().enumerate() {
        let start = index * Q8_BLOCK_BYTES;
        dequantize_block_q8(&blocks[start..start + Q8_BLOCK_BYTES], chunk)?;
    }
    Ok(())
}

/// Weights in one `Q4_K` or `Q6_K` super block (`QK_K` of ggml).
pub const QK_K: usize = 256;

/// Bytes of the packed scales and minimums of a `Q4_K` block (`K_SCALE_SIZE`).
const K_SCALE_SIZE: usize = 12;

/// Bytes of one `Q5_0` block: the binary16 `d`, four bytes of fifth bits and
/// sixteen bytes of nibbles.
pub const Q5_0_BLOCK_BYTES: usize = 22;

/// Bytes of one `Q4_K` block: `d`, `dmin`, the packed scales and 128 bytes of
/// nibbles.
pub const Q4_K_BLOCK_BYTES: usize = 144;

/// Bytes of one `Q6_K` block: 128 bytes that hold the low four bits of the 256
/// six bit weights, two per byte, 64 bytes that hold the high two bits, four per
/// byte, 16 signed scales and the binary16 `d`.
pub const Q6_K_BLOCK_BYTES: usize = 210;

/// Bytes one row of `columns` weights occupies, for a block of `elements`
/// weights in `block_bytes` bytes.
fn blocked_row_bytes(
    name: &str,
    columns: usize,
    elements: usize,
    block_bytes: usize,
) -> Result<usize> {
    if columns == 0 || !columns.is_multiple_of(elements) {
        return Err(Error::new(format!(
            "quant: {columns} columns is not a multiple of the {elements} weight block of {name}"
        )));
    }
    Ok(columns / elements * block_bytes)
}

/// Bytes a `rows` by `columns` matrix occupies, for a row of `row` bytes.
fn blocked_matrix_bytes(rows: usize, row: usize) -> Result<usize> {
    rows.checked_mul(row)
        .ok_or_else(|| Error::new("quant: matrix byte length overflow"))
}

/// Bytes one row of `columns` weights occupies in `Q5_0`.
pub fn q5_0_row_bytes(columns: usize) -> Result<usize> {
    blocked_row_bytes("Q5_0", columns, BLOCK, Q5_0_BLOCK_BYTES)
}

/// Bytes a `rows` by `columns` matrix occupies in `Q5_0`.
pub fn q5_0_matrix_bytes(rows: usize, columns: usize) -> Result<usize> {
    blocked_matrix_bytes(rows, q5_0_row_bytes(columns)?)
}

/// Bytes one row of `columns` weights occupies in `Q4_K`.
pub fn q4_k_row_bytes(columns: usize) -> Result<usize> {
    blocked_row_bytes("Q4_K", columns, QK_K, Q4_K_BLOCK_BYTES)
}

/// Bytes a `rows` by `columns` matrix occupies in `Q4_K`.
pub fn q4_k_matrix_bytes(rows: usize, columns: usize) -> Result<usize> {
    blocked_matrix_bytes(rows, q4_k_row_bytes(columns)?)
}

/// Bytes one row of `columns` weights occupies in `Q6_K`.
pub fn q6_k_row_bytes(columns: usize) -> Result<usize> {
    blocked_row_bytes("Q6_K", columns, QK_K, Q6_K_BLOCK_BYTES)
}

/// Bytes a `rows` by `columns` matrix occupies in `Q6_K`.
pub fn q6_k_matrix_bytes(rows: usize, columns: usize) -> Result<usize> {
    blocked_matrix_bytes(rows, q6_k_row_bytes(columns)?)
}

/// The binary16 at byte `offset` of a block, as `f32`.
#[inline]
fn half_at(block: &[u8], offset: usize) -> f32 {
    half_to_f32(u16::from_le_bytes([block[offset], block[offset + 1]]))
}

/// The scale `d` of a `Q5_0` block, as `f32`, without a finiteness check.
#[inline]
pub(crate) fn q5_0_scale(block: &[u8]) -> f32 {
    half_at(block, 0)
}

/// The thirty-two integer weights `q - 16` of a `Q5_0` block, in `-16..=15`.
///
/// Weight `j < 16` is the low nibble of `qs[j]` and weight `j + 16` is its high
/// nibble; the fifth bit of weight `k` is bit `k` of `qh`, read little endian.
/// `block` must hold the 22 bytes of the block.
pub(crate) fn q5_0_weights(block: &[u8]) -> [i8; BLOCK] {
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    let qs = &block[6..Q5_0_BLOCK_BYTES];
    let mut weights = [0i8; BLOCK];
    for j in 0..BLOCK / 2 {
        let low = (qs[j] & 0x0f) | ((((qh >> j) & 1) as u8) << 4);
        let high = (qs[j] >> 4) | ((((qh >> (j + BLOCK / 2)) & 1) as u8) << 4);
        weights[j] = low as i8 - 16;
        weights[j + BLOCK / 2] = high as i8 - 16;
    }
    weights
}

/// Decode one `Q5_0` block into `BLOCK` values: `(q - 16) * d`, the integer
/// converted to `f32` before the product, as `dequantize_row_q5_0` does.
pub fn dequantize_block_q5_0(block: &[u8], out: &mut [f32]) -> Result<()> {
    if block.len() != Q5_0_BLOCK_BYTES || out.len() != BLOCK {
        return Err(Error::new(format!(
            "quant: a Q5_0 block is {Q5_0_BLOCK_BYTES} bytes into {BLOCK} values"
        )));
    }
    let scale = q5_0_scale(block);
    if !scale.is_finite() {
        return Err(Error::new("quant: a block scale is not finite"));
    }
    for (slot, weight) in out.iter_mut().zip(q5_0_weights(block)) {
        *slot = weight as f32 * scale;
    }
    Ok(())
}

/// Decode a whole `Q5_0` row of blocks into `out`.
pub fn dequantize_row_q5_0(blocks: &[u8], out: &mut [f32]) -> Result<()> {
    let columns = out.len();
    let expected = q5_0_row_bytes(columns)?;
    if blocks.len() != expected {
        return Err(Error::new(format!(
            "quant: a Q5_0 row of {columns} weights is {expected} bytes, not {}",
            blocks.len()
        )));
    }
    for (index, chunk) in out.as_chunks_mut::<BLOCK>().0.iter_mut().enumerate() {
        let start = index * Q5_0_BLOCK_BYTES;
        dequantize_block_q5_0(&blocks[start..start + Q5_0_BLOCK_BYTES], chunk)?;
    }
    Ok(())
}

/// The scale and the minimum `j` of the twelve packed bytes of a `Q4_K` block,
/// 6 bits each (`get_scale_min_k4` of ggml).
#[inline]
fn scale_min_k4(j: usize, packed: &[u8]) -> (u8, u8) {
    if j < 4 {
        (packed[j] & 63, packed[j + 4] & 63)
    } else {
        (
            (packed[j + 4] & 0x0f) | ((packed[j - 4] >> 6) << 4),
            (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4),
        )
    }
}

/// The two binary16 scales of a `Q4_K` block, `d` and `dmin`, as `f32`.
#[inline]
pub(crate) fn q4_k_scales(block: &[u8]) -> (f32, f32) {
    (half_at(block, 0), half_at(block, 2))
}

/// Decode a `Q4_K` block into 256 values without a finiteness check.
///
/// The super block is four groups of 64 weights. For the group `g`, the
/// scales `(sc, m)` of the sub blocks `2g` and `2g + 1` give
/// `d1 = d * sc`, `m1 = dmin * m` (and `d2`, `m2`), each rounded to `f32`; the
/// first 32 values of the group are `d1 * (qs[32g + l] & 15) - m1` and the next
/// 32 are `d2 * (qs[32g + l] >> 4) - m2`, with a separate multiply and
/// subtract. `block` must hold the 144 bytes of the block.
pub(crate) fn decode_q4_k(block: &[u8], out: &mut [f32; QK_K]) {
    let (d, dmin) = q4_k_scales(block);
    let packed = &block[4..4 + K_SCALE_SIZE];
    let qs = &block[4 + K_SCALE_SIZE..Q4_K_BLOCK_BYTES];
    for group in 0..QK_K / 64 {
        let (sc1, m1) = scale_min_k4(2 * group, packed);
        let (sc2, m2) = scale_min_k4(2 * group + 1, packed);
        let d1 = d * sc1 as f32;
        let min1 = dmin * m1 as f32;
        let d2 = d * sc2 as f32;
        let min2 = dmin * m2 as f32;
        for l in 0..32 {
            let byte = qs[32 * group + l];
            out[64 * group + l] = d1 * (byte & 0x0f) as f32 - min1;
            out[64 * group + 32 + l] = d2 * (byte >> 4) as f32 - min2;
        }
    }
}

/// Decode one `Q4_K` block into `QK_K` values.
pub fn dequantize_block_q4_k(block: &[u8], out: &mut [f32]) -> Result<()> {
    let Ok(out) = <&mut [f32; QK_K]>::try_from(out) else {
        return Err(Error::new(format!(
            "quant: a Q4_K block is {Q4_K_BLOCK_BYTES} bytes into {QK_K} values"
        )));
    };
    if block.len() != Q4_K_BLOCK_BYTES {
        return Err(Error::new(format!(
            "quant: a Q4_K block is {Q4_K_BLOCK_BYTES} bytes into {QK_K} values"
        )));
    }
    let (d, dmin) = q4_k_scales(block);
    if !d.is_finite() || !dmin.is_finite() {
        return Err(Error::new("quant: a block scale is not finite"));
    }
    decode_q4_k(block, out);
    Ok(())
}

/// Decode a whole `Q4_K` row of super blocks into `out`.
pub fn dequantize_row_q4_k(blocks: &[u8], out: &mut [f32]) -> Result<()> {
    let columns = out.len();
    let expected = q4_k_row_bytes(columns)?;
    if blocks.len() != expected {
        return Err(Error::new(format!(
            "quant: a Q4_K row of {columns} weights is {expected} bytes, not {}",
            blocks.len()
        )));
    }
    for (index, chunk) in out.as_chunks_mut::<QK_K>().0.iter_mut().enumerate() {
        let start = index * Q4_K_BLOCK_BYTES;
        dequantize_block_q4_k(&blocks[start..start + Q4_K_BLOCK_BYTES], chunk)?;
    }
    Ok(())
}

/// The binary16 `d` of a `Q6_K` block, as `f32`, without a finiteness check.
#[inline]
pub(crate) fn q6_k_scale(block: &[u8]) -> f32 {
    half_at(block, Q6_K_BLOCK_BYTES - 2)
}

/// Decode a `Q6_K` block into 256 values without a finiteness check.
///
/// The super block is two halves of 128 weights, each with 64 bytes of `ql`,
/// 32 bytes of `qh` and 8 of the signed scales. For the position `l` in `0..32`
/// of a half, the four weights `l`, `l + 32`, `l + 64` and `l + 96` are
/// `ql[l] & 15`, `ql[l + 32] & 15`, `ql[l] >> 4` and `ql[l + 32] >> 4`, each
/// with two more bits of `qh[l]` (bits 0-1, 2-3, 4-5 and 6-7) above them, minus
/// 32, and the scale of weight `l + 32 * i` is the signed byte
/// `scales[2 * i + l / 16]` of the half. Each value is `(d * scale) * q` with
/// `d * scale` rounded to `f32` first and `q` converted to `f32`. `block` must
/// hold the 210 bytes of the block.
pub(crate) fn decode_q6_k(block: &[u8], out: &mut [f32; QK_K]) {
    let d = q6_k_scale(block);
    let ql = &block[..QK_K / 2];
    let qh = &block[QK_K / 2..QK_K / 2 + QK_K / 4];
    let scales = &block[QK_K / 2 + QK_K / 4..QK_K / 2 + QK_K / 4 + QK_K / 16];
    for half in 0..QK_K / 128 {
        let ql = &ql[64 * half..];
        let qh = &qh[32 * half..];
        let scales = &scales[8 * half..];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0x0f) | ((qh[l] & 3) << 4)) as i8 - 32;
            let q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
            let base = 128 * half + l;
            out[base] = d * (scales[is] as i8) as f32 * q1 as f32;
            out[base + 32] = d * (scales[is + 2] as i8) as f32 * q2 as f32;
            out[base + 64] = d * (scales[is + 4] as i8) as f32 * q3 as f32;
            out[base + 96] = d * (scales[is + 6] as i8) as f32 * q4 as f32;
        }
    }
}

/// Decode one `Q6_K` block into `QK_K` values.
pub fn dequantize_block_q6_k(block: &[u8], out: &mut [f32]) -> Result<()> {
    let Ok(out) = <&mut [f32; QK_K]>::try_from(out) else {
        return Err(Error::new(format!(
            "quant: a Q6_K block is {Q6_K_BLOCK_BYTES} bytes into {QK_K} values"
        )));
    };
    if block.len() != Q6_K_BLOCK_BYTES {
        return Err(Error::new(format!(
            "quant: a Q6_K block is {Q6_K_BLOCK_BYTES} bytes into {QK_K} values"
        )));
    }
    if !q6_k_scale(block).is_finite() {
        return Err(Error::new("quant: a block scale is not finite"));
    }
    decode_q6_k(block, out);
    Ok(())
}

/// Decode a whole `Q6_K` row of super blocks into `out`.
pub fn dequantize_row_q6_k(blocks: &[u8], out: &mut [f32]) -> Result<()> {
    let columns = out.len();
    let expected = q6_k_row_bytes(columns)?;
    if blocks.len() != expected {
        return Err(Error::new(format!(
            "quant: a Q6_K row of {columns} weights is {expected} bytes, not {}",
            blocks.len()
        )));
    }
    for (index, chunk) in out.as_chunks_mut::<QK_K>().0.iter_mut().enumerate() {
        let start = index * Q6_K_BLOCK_BYTES;
        dequantize_block_q6_k(&blocks[start..start + Q6_K_BLOCK_BYTES], chunk)?;
    }
    Ok(())
}

/// What `llama.cpp` produced for the same weights, read from `bench/gguf`, and
/// the closed form of each element of the three GGUF block types, written from
/// the layout of `gguf-py` instead of the loops of ggml so that the decoders
/// above are not tested against a copy of themselves.
#[cfg(test)]
pub(crate) mod oracle {
    use super::{half_to_f32, BLOCK, QK_K};
    use crate::json::Scanner;

    /// One whole block: its bytes and the `float32` values `gguf-py` gave.
    pub struct Block {
        pub bytes: Vec<u8>,
        pub values: Vec<f32>,
    }

    /// The blocks of one type of one `vectors.json`.
    pub struct Vectors {
        pub block_elements: usize,
        pub block_bytes: usize,
        pub blocks: Vec<Block>,
    }

    /// The two files, by the model they come from.
    pub const SOURCES: [(&str, &str); 2] = [
        (
            "smollm2",
            include_str!("../../bench/gguf/smollm2/ref/vectors.json"),
        ),
        (
            "synthetic",
            include_str!("../../bench/gguf/synthetic/ref/vectors.json"),
        ),
    ];

    /// Walk a JSON object, calling `field` with the key and the scanner placed
    /// on its value, which `field` must consume.
    fn object(scanner: &mut Scanner, mut field: impl FnMut(&mut Scanner, &str)) {
        scanner.skip_whitespace();
        scanner.expect(b'{').unwrap();
        loop {
            scanner.skip_whitespace();
            let key = scanner.parse_string().unwrap();
            scanner.skip_whitespace();
            scanner.expect(b':').unwrap();
            field(scanner, &key);
            scanner.skip_whitespace();
            match scanner.bump().unwrap() {
                b',' => continue,
                b'}' => return,
                other => panic!("json: unexpected {}", other as char),
            }
        }
    }

    /// Walk a JSON array, calling `item` with the scanner placed on each value.
    fn array(scanner: &mut Scanner, mut item: impl FnMut(&mut Scanner)) {
        scanner.skip_whitespace();
        scanner.expect(b'[').unwrap();
        loop {
            item(scanner);
            scanner.skip_whitespace();
            match scanner.bump().unwrap() {
                b',' => continue,
                b']' => return,
                other => panic!("json: unexpected {}", other as char),
            }
        }
    }

    fn hex_bytes(text: &str) -> Vec<u8> {
        assert!(text.len() % 2 == 0);
        (0..text.len() / 2)
            .map(|index| u8::from_str_radix(&text[2 * index..2 * index + 2], 16).unwrap())
            .collect()
    }

    /// The blocks of `kind` (`"Q8_0"`, `"Q5_0"`, `"Q4_K"` or `"Q6_K"`) of one file.
    pub fn load(text: &str, kind: &str) -> Vectors {
        let mut found = None;
        let mut scanner = Scanner::new(text.as_bytes());
        object(&mut scanner, |scanner, key| {
            if key != kind {
                scanner.skip_value().unwrap();
                return;
            }
            let mut vectors = Vectors {
                block_elements: 0,
                block_bytes: 0,
                blocks: Vec::new(),
            };
            object(scanner, |scanner, key| match key {
                "block_elements" => vectors.block_elements = scanner.parse_usize().unwrap(),
                "block_bytes" => vectors.block_bytes = scanner.parse_usize().unwrap(),
                "blocks" => array(scanner, |scanner| {
                    let mut block = Block {
                        bytes: Vec::new(),
                        values: Vec::new(),
                    };
                    object(scanner, |scanner, key| match key {
                        "bytes_hex" => {
                            scanner.skip_whitespace();
                            block.bytes = hex_bytes(&scanner.parse_string().unwrap());
                        }
                        "f32_bits" => array(scanner, |scanner| {
                            scanner.skip_whitespace();
                            let digits = scanner.parse_string().unwrap();
                            assert_eq!(digits.len(), 8);
                            block
                                .values
                                .push(f32::from_bits(u32::from_str_radix(&digits, 16).unwrap()));
                        }),
                        _ => scanner.skip_value().unwrap(),
                    });
                    vectors.blocks.push(block);
                }),
                _ => scanner.skip_value().unwrap(),
            });
            found = Some(vectors);
        });
        found.unwrap_or_else(|| panic!("no {kind} in the vectors"))
    }

    /// One entry of a `tensors-*.json`.
    pub struct Tensor {
        pub name: String,
        pub kind: String,
        pub bytes: usize,
        pub sha256_raw: String,
        pub sha256_f32: String,
    }

    /// The tensors of a `tensors-*.json`.
    pub fn tensors(text: &str) -> Vec<Tensor> {
        let mut tensors = Vec::new();
        let mut scanner = Scanner::new(text.as_bytes());
        array(&mut scanner, |scanner| {
            let mut tensor = Tensor {
                name: String::new(),
                kind: String::new(),
                bytes: 0,
                sha256_raw: String::new(),
                sha256_f32: String::new(),
            };
            object(scanner, |scanner, key| {
                scanner.skip_whitespace();
                match key {
                    "name" => tensor.name = scanner.parse_string().unwrap(),
                    "type" => tensor.kind = scanner.parse_string().unwrap(),
                    "bytes" => tensor.bytes = scanner.parse_usize().unwrap(),
                    "sha256_raw" => tensor.sha256_raw = scanner.parse_string().unwrap(),
                    "sha256_f32" => tensor.sha256_f32 = scanner.parse_string().unwrap(),
                    _ => scanner.skip_value().unwrap(),
                }
            });
            tensors.push(tensor);
        });
        tensors
    }

    /// The scale `d` of a `Q5_0` block.
    pub fn q5_0_scale(block: &[u8]) -> f32 {
        half_to_f32(u16::from_le_bytes([block[0], block[1]]))
    }

    /// The integer weight `i` of a `Q5_0` block, in `-16..=15`.
    pub fn q5_0_weight(block: &[u8], i: usize) -> i32 {
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let byte = block[6 + i % 16];
        let nibble = if i < 16 { byte & 0x0f } else { byte >> 4 };
        nibble as i32 + 16 * ((qh >> i) & 1) as i32 - 16
    }

    /// Element `i` of a `Q5_0` block: `(nibble + 16 * fifth_bit - 16) * d`.
    pub fn q5_0_element(block: &[u8], i: usize) -> f32 {
        assert!(i < BLOCK);
        q5_0_weight(block, i) as f32 * q5_0_scale(block)
    }

    /// Element `i` of a `Q4_K` block, with the scales unpacked as `gguf-py`
    /// does: three rows of four bytes, the scales, the minimums and the
    /// shared high bits.
    pub fn q4_k_element(block: &[u8], i: usize) -> f32 {
        assert!(i < QK_K);
        let d = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = half_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let (scales, qs) = (&block[4..16], &block[16..]);
        let (group, within) = (i / 64, i % 64);
        let sub = 2 * group + within / 32;
        let (low_row, min_row, shared) = (&scales[0..4], &scales[4..8], &scales[8..12]);
        let (scale, min) = if sub < 4 {
            (low_row[sub] & 0x3f, min_row[sub] & 0x3f)
        } else {
            let s = sub - 4;
            (
                (shared[s] & 0x0f) | ((low_row[s] >> 2) & 0x30),
                (shared[s] >> 4) | ((min_row[s] >> 2) & 0x30),
            )
        };
        let byte = qs[32 * group + within % 32];
        let nibble = if within < 32 { byte & 0x0f } else { byte >> 4 };
        let scaled = d * scale as f32;
        let offset = dmin * min as f32;
        scaled * nibble as f32 - offset
    }

    /// Element `i` of a `Q6_K` block.
    pub fn q6_k_element(block: &[u8], i: usize) -> f32 {
        assert!(i < QK_K);
        let (ql, rest) = block.split_at(128);
        let (qh, rest) = rest.split_at(64);
        let (scales, d) = rest.split_at(16);
        let d = half_to_f32(u16::from_le_bytes([d[0], d[1]]));
        let (half, within) = (i / 128, i % 128);
        let (quarter, l) = (within / 32, within % 32);
        let byte = ql[64 * half + 32 * (quarter % 2) + l];
        let low = if quarter < 2 { byte & 0x0f } else { byte >> 4 };
        let high = (qh[32 * half + l] >> (2 * quarter)) & 3;
        let q = (low | (high << 4)) as i32 - 32;
        let scale = scales[8 * half + 2 * quarter + l / 16] as i8;
        let scaled = d * scale as f32;
        scaled * q as f32
    }
}

#[cfg(test)]
mod tests {
    use super::{
        blocks, dequantize_block, dequantize_block_q4_k, dequantize_block_q5_0,
        dequantize_block_q6_k, dequantize_block_q8, dequantize_row, dequantize_row_q4_k,
        dequantize_row_q5_0, dequantize_row_q6_k, dequantize_row_q8, f32_to_half, half_to_f32,
        matrix_bytes, oracle, q4_k_matrix_bytes, q4_k_row_bytes, q5_0_matrix_bytes, q5_0_row_bytes,
        q6_k_matrix_bytes, q6_k_row_bytes, q8_matrix_bytes, q8_row_bytes, quantize_block,
        quantize_block_q8, row_bytes, scale_is_finite, BLOCK, BLOCK_BYTES, Q4_K_BLOCK_BYTES,
        Q5_0_BLOCK_BYTES, Q6_K_BLOCK_BYTES, Q8_BLOCK_BYTES, QK_K,
    };
    use crate::gguf::TensorKind;

    #[test]
    fn half_conversion_covers_the_known_values() {
        // Exact values, the largest finite binary16, the smallest normal, a
        // subnormal and the two infinities.
        for (half, value) in [
            (0x0000u16, 0.0f32),
            (0x3c00, 1.0),
            (0xbc00, -1.0),
            (0x3800, 0.5),
            (0x7bff, 65_504.0),
            (0x0400, 6.103_515_6e-5),
            (0x0001, 5.960_464_5e-8),
            (0x7c00, f32::INFINITY),
            (0xfc00, f32::NEG_INFINITY),
        ] {
            assert_eq!(half_to_f32(half), value, "half {half:#06x}");
        }
        assert_eq!(f32_to_half(1.0), 0x3c00);
        assert_eq!(f32_to_half(-2.0), 0xc000);
        assert_eq!(f32_to_half(65_504.0), 0x7bff);
        assert_eq!(f32_to_half(0.5), 0x3800);
        assert_eq!(f32_to_half(6.0e-8), 0x0001);
        assert_eq!(f32_to_half(1.0e-9), 0x0000);
        assert_eq!(f32_to_half(f32::INFINITY), 0x7c00);
        // A value above the range saturates to infinity instead of wrapping.
        assert!(half_to_f32(f32_to_half(1.0e6)).is_infinite());
        assert_eq!(f32_to_half(f32::NEG_INFINITY), 0xfc00);
        // Every finite binary16 round trips through the conversion.
        for half in (0..0x7c00u16).chain(0x8000..0xfc00) {
            assert_eq!(f32_to_half(half_to_f32(half)), half, "half {half:#06x}");
        }
    }

    #[test]
    fn a_block_with_exact_values_round_trips() {
        // The largest magnitude is seven, so the scale is exactly one and every
        // nibble value is an integer. The round trip is exact.
        let mut values = [0.0f32; BLOCK];
        for (index, slot) in values.iter_mut().enumerate() {
            *slot = ((index as i32) % 15 - 7) as f32;
        }
        let mut block = [0u8; BLOCK_BYTES];
        quantize_block(&values, &mut block).unwrap();
        assert_eq!(u16::from_le_bytes([block[0], block[1]]), 0x3c00);
        let mut decoded = [0.0f32; BLOCK];
        dequantize_block(&block, &mut decoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn a_zero_block_stores_a_zero_scale() {
        let values = [0.0f32; BLOCK];
        let mut block = [0xffu8; BLOCK_BYTES];
        quantize_block(&values, &mut block).unwrap();
        assert_eq!(block, [0u8; BLOCK_BYTES]);
        let mut decoded = [1.0f32; BLOCK];
        dequantize_block(&block, &mut decoded).unwrap();
        assert_eq!(decoded, [0.0f32; BLOCK]);
    }

    #[test]
    fn the_quantization_error_stays_inside_half_a_step() {
        let values: Vec<f32> = (0..BLOCK)
            .map(|index| ((index as f32) * 0.37).sin() * 1.5)
            .collect();
        let mut block = [0u8; BLOCK_BYTES];
        quantize_block(&values, &mut block).unwrap();
        let scale = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let mut decoded = [0.0f32; BLOCK];
        dequantize_block(&block, &mut decoded).unwrap();
        for (original, restored) in values.iter().zip(decoded) {
            assert!(
                (original - restored).abs() <= scale / 2.0 + 1e-6,
                "{} against {} with scale {scale}",
                original,
                restored
            );
        }
    }

    #[test]
    fn a_row_of_blocks_round_trips() {
        let columns = 3 * BLOCK;
        let values: Vec<f32> = (0..columns)
            .map(|index| ((index as f32) * 0.11).cos())
            .collect();
        let mut bytes = vec![0u8; row_bytes(columns).unwrap()];
        for (index, chunk) in values.as_chunks::<BLOCK>().0.iter().enumerate() {
            quantize_block(
                chunk,
                &mut bytes[index * BLOCK_BYTES..(index + 1) * BLOCK_BYTES],
            )
            .unwrap();
        }
        let mut decoded = vec![0.0f32; columns];
        dequantize_row(&bytes, &mut decoded).unwrap();
        for (index, (original, restored)) in values.iter().zip(&decoded).enumerate() {
            // One scale per block, so the bound is half a step of that block.
            let block = &values[index / BLOCK * BLOCK..(index / BLOCK + 1) * BLOCK];
            let mut largest = 0.0f32;
            for &value in block {
                largest = largest.max(value.abs());
            }
            assert!((original - restored).abs() <= largest / 14.0 + 1e-6);
        }
    }

    #[test]
    fn row_and_matrix_sizes_are_the_block_layout() {
        assert_eq!(blocks(BLOCK), 1);
        assert_eq!(blocks(576), 18);
        assert_eq!(row_bytes(576).unwrap(), 18 * 18);
        assert_eq!(matrix_bytes(3, 32).unwrap(), 3 * 18);
        assert!(row_bytes(31).is_err());
        assert!(row_bytes(0).is_err());
        assert!(matrix_bytes(usize::MAX, 32).is_err());
    }

    #[test]
    fn an_eight_bit_block_round_trips_and_bounds_its_error() {
        let mut values = [0.0f32; BLOCK];
        for (index, slot) in values.iter_mut().enumerate() {
            *slot = ((index as f32) * 0.043).sin() * 2.0;
        }
        let mut block = [0u8; Q8_BLOCK_BYTES];
        quantize_block_q8(&values, &mut block).unwrap();
        let scale = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let mut decoded = [0.0f32; BLOCK];
        dequantize_block_q8(&block, &mut decoded).unwrap();
        for (original, restored) in values.iter().zip(decoded) {
            assert!((original - restored).abs() <= scale / 2.0 + 1e-6);
        }
        // A largest magnitude of 127 gives a scale of exactly one, so the
        // integer payload can be read back value by value.
        let exact = [127.0f32; BLOCK];
        quantize_block_q8(&exact, &mut block).unwrap();
        dequantize_block_q8(&block, &mut decoded).unwrap();
        assert_eq!(decoded, exact);
    }

    #[test]
    fn an_eight_bit_row_uses_the_wider_layout() {
        assert_eq!(q8_row_bytes(576).unwrap(), 18 * Q8_BLOCK_BYTES);
        assert_eq!(q8_matrix_bytes(3, 32).unwrap(), 3 * Q8_BLOCK_BYTES);
        let columns = 2 * BLOCK;
        let values: Vec<f32> = (0..columns)
            .map(|index| ((index as f32) * 0.19).cos())
            .collect();
        let mut bytes = vec![0u8; q8_row_bytes(columns).unwrap()];
        for (index, chunk) in values.as_chunks::<BLOCK>().0.iter().enumerate() {
            quantize_block_q8(
                chunk,
                &mut bytes[index * Q8_BLOCK_BYTES..(index + 1) * Q8_BLOCK_BYTES],
            )
            .unwrap();
        }
        let mut decoded = vec![0.0f32; columns];
        dequantize_row_q8(&bytes, &mut decoded).unwrap();
        for (original, restored) in values.iter().zip(&decoded) {
            assert!((original - restored).abs() < 0.02);
        }
    }

    #[test]
    fn rejects_a_block_of_the_wrong_length_and_a_ragged_row() {
        assert!(quantize_block(&[0.0; BLOCK - 1], &mut [0u8; BLOCK_BYTES]).is_err());
        assert!(quantize_block(&[0.0; BLOCK], &mut [0u8; BLOCK_BYTES - 1]).is_err());
        assert!(dequantize_block(&[0u8; BLOCK_BYTES], &mut [0.0; BLOCK - 1]).is_err());
        assert!(dequantize_row(&[0u8; BLOCK_BYTES], &mut [0.0; BLOCK + 1]).is_err());
    }

    #[test]
    fn rejects_a_weight_that_is_not_finite() {
        let mut values = [0.0f32; BLOCK];
        values[3] = f32::NAN;
        assert!(quantize_block(&values, &mut [0u8; BLOCK_BYTES]).is_err());
        // A non-finite scale is refused when the block is decoded.
        let mut block = [0u8; BLOCK_BYTES];
        block[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
        assert!(!scale_is_finite(0x7c00));
        assert!(dequantize_block(&block, &mut [0.0f32; BLOCK]).is_err());
    }

    type Dequantizer = fn(&[u8], &mut [f32]) -> crate::error::Result<()>;

    /// The four dequantisers, by the name of the type in `vectors.json`.
    fn dequantizers() -> [(&'static str, Dequantizer); 4] {
        [
            ("Q8_0", dequantize_block_q8),
            ("Q5_0", dequantize_block_q5_0),
            ("Q4_K", dequantize_block_q4_k),
            ("Q6_K", dequantize_block_q6_k),
        ]
    }

    /// A deterministic stream of bytes.
    fn bytes(seed: u64, count: usize) -> Vec<u8> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 33) as u8
            })
            .collect()
    }

    #[test]
    fn the_gguf_block_sizes_are_those_of_ggml() {
        assert_eq!(Q5_0_BLOCK_BYTES, 2 + 4 + BLOCK / 2);
        assert_eq!(Q4_K_BLOCK_BYTES, 2 + 2 + 12 + QK_K / 2);
        assert_eq!(Q6_K_BLOCK_BYTES, QK_K / 2 + QK_K / 4 + QK_K / 16 + 2);
        assert_eq!(QK_K, 256);
        for (kind, elements, size) in [
            (TensorKind::Q5_0, BLOCK, Q5_0_BLOCK_BYTES),
            (TensorKind::Q4K, QK_K, Q4_K_BLOCK_BYTES),
            (TensorKind::Q6K, QK_K, Q6_K_BLOCK_BYTES),
        ] {
            assert_eq!(kind.block_elements(), elements as u64);
            assert_eq!(kind.block_bytes(), size as u64);
        }
        assert_eq!(q5_0_row_bytes(96).unwrap(), 3 * 22);
        assert_eq!(q4_k_row_bytes(512).unwrap(), 2 * 144);
        assert_eq!(q6_k_row_bytes(768).unwrap(), 3 * 210);
        assert_eq!(q5_0_matrix_bytes(5, 64).unwrap(), 5 * 2 * 22);
        assert_eq!(q4_k_matrix_bytes(5, 256).unwrap(), 5 * 144);
        assert_eq!(q6_k_matrix_bytes(5, 256).unwrap(), 5 * 210);
    }

    #[test]
    fn every_block_of_the_vectors_has_the_bits_of_gguf_py() {
        for (model, text) in oracle::SOURCES {
            for (kind, decode) in dequantizers() {
                let vectors = oracle::load(text, kind);
                assert_eq!(vectors.blocks.len(), 6, "{model} {kind}");
                for (index, block) in vectors.blocks.iter().enumerate() {
                    assert_eq!(block.bytes.len(), vectors.block_bytes);
                    assert_eq!(block.values.len(), vectors.block_elements);
                    let mut out = vec![0.0f32; vectors.block_elements];
                    decode(&block.bytes, &mut out).unwrap();
                    for (element, (actual, expected)) in out.iter().zip(&block.values).enumerate() {
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "{model} {kind} block {index} element {element}: {actual} against {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_vectors_have_the_geometry_of_the_constants() {
        for (_, text) in oracle::SOURCES {
            for (kind, elements, size) in [
                ("Q8_0", BLOCK, Q8_BLOCK_BYTES),
                ("Q5_0", BLOCK, Q5_0_BLOCK_BYTES),
                ("Q4_K", QK_K, Q4_K_BLOCK_BYTES),
                ("Q6_K", QK_K, Q6_K_BLOCK_BYTES),
            ] {
                let vectors = oracle::load(text, kind);
                assert_eq!(vectors.block_elements, elements, "{kind}");
                assert_eq!(vectors.block_bytes, size, "{kind}");
            }
        }
    }

    #[test]
    fn the_closed_form_of_every_element_has_the_bits_of_gguf_py() {
        // The element by element decoders of the oracle module are what the
        // matrix tests of the kernels use on random blocks, so they are pinned
        // to the vectors first.
        for (model, text) in oracle::SOURCES {
            for (kind, element) in [
                ("Q5_0", oracle::q5_0_element as fn(&[u8], usize) -> f32),
                ("Q4_K", oracle::q4_k_element),
                ("Q6_K", oracle::q6_k_element),
            ] {
                for block in &oracle::load(text, kind).blocks {
                    for (index, expected) in block.values.iter().enumerate() {
                        assert_eq!(
                            element(&block.bytes, index).to_bits(),
                            expected.to_bits(),
                            "{model} {kind} element {index}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_closed_form_agrees_with_the_decoders_on_random_blocks() {
        for seed in 0..64u64 {
            let mut block = bytes(0x5eed_0000 + seed, Q5_0_BLOCK_BYTES);
            block[..2].copy_from_slice(&f32_to_half(1.5 - seed as f32 * 0.07).to_le_bytes());
            let mut out = [0.0f32; BLOCK];
            dequantize_block_q5_0(&block, &mut out).unwrap();
            for (index, value) in out.iter().enumerate() {
                assert_eq!(
                    value.to_bits(),
                    oracle::q5_0_element(&block, index).to_bits()
                );
            }

            let mut block = bytes(0x6eed_0000 + seed, Q4_K_BLOCK_BYTES);
            block[..2].copy_from_slice(&f32_to_half(0.02 * (seed as f32 - 20.0)).to_le_bytes());
            block[2..4].copy_from_slice(&f32_to_half(0.03 * (seed as f32 - 40.0)).to_le_bytes());
            let mut out = [0.0f32; QK_K];
            dequantize_block_q4_k(&block, &mut out).unwrap();
            for (index, value) in out.iter().enumerate() {
                assert_eq!(
                    value.to_bits(),
                    oracle::q4_k_element(&block, index).to_bits()
                );
            }

            let mut block = bytes(0x7eed_0000 + seed, Q6_K_BLOCK_BYTES);
            block[208..].copy_from_slice(&f32_to_half(0.001 * (seed as f32 - 30.0)).to_le_bytes());
            dequantize_block_q6_k(&block, &mut out).unwrap();
            for (index, value) in out.iter().enumerate() {
                assert_eq!(
                    value.to_bits(),
                    oracle::q6_k_element(&block, index).to_bits()
                );
            }
        }
    }

    #[test]
    fn a_row_of_every_gguf_type_is_its_blocks_in_order() {
        let mut stream = bytes(0xabcd, 3 * Q6_K_BLOCK_BYTES);
        for chunk in stream.chunks_mut(Q6_K_BLOCK_BYTES) {
            chunk[208..].copy_from_slice(&f32_to_half(0.5).to_le_bytes());
        }
        let mut row = vec![0.0f32; 3 * QK_K];
        dequantize_row_q6_k(&stream, &mut row).unwrap();
        for (block, chunk) in row.chunks(QK_K).enumerate() {
            let mut one = [0.0f32; QK_K];
            dequantize_block_q6_k(
                &stream[block * Q6_K_BLOCK_BYTES..(block + 1) * Q6_K_BLOCK_BYTES],
                &mut one,
            )
            .unwrap();
            assert_eq!(chunk, one);
        }

        let mut stream = bytes(0xbcde, 2 * Q4_K_BLOCK_BYTES);
        for chunk in stream.chunks_mut(Q4_K_BLOCK_BYTES) {
            chunk[..4].copy_from_slice(&[0x00, 0x3c, 0x00, 0x38]);
        }
        let mut row = vec![0.0f32; 2 * QK_K];
        dequantize_row_q4_k(&stream, &mut row).unwrap();
        for (block, chunk) in row.chunks(QK_K).enumerate() {
            let mut one = [0.0f32; QK_K];
            dequantize_block_q4_k(
                &stream[block * Q4_K_BLOCK_BYTES..(block + 1) * Q4_K_BLOCK_BYTES],
                &mut one,
            )
            .unwrap();
            assert_eq!(chunk, one);
        }

        let mut stream = bytes(0xcdef, 5 * Q5_0_BLOCK_BYTES);
        for chunk in stream.chunks_mut(Q5_0_BLOCK_BYTES) {
            chunk[..2].copy_from_slice(&f32_to_half(0.25).to_le_bytes());
        }
        let mut row = vec![0.0f32; 5 * BLOCK];
        dequantize_row_q5_0(&stream, &mut row).unwrap();
        for (block, chunk) in row.chunks(BLOCK).enumerate() {
            let mut one = [0.0f32; BLOCK];
            dequantize_block_q5_0(
                &stream[block * Q5_0_BLOCK_BYTES..(block + 1) * Q5_0_BLOCK_BYTES],
                &mut one,
            )
            .unwrap();
            assert_eq!(chunk, one);
        }
    }

    #[test]
    fn every_tensor_of_the_synthetic_file_has_the_hash_of_gguf_py() {
        // The whole `Q4_K_M` file of the synthetic model is committed, and
        // `gguf-py` hashed the raw bytes and the dequantised `float32` bytes of
        // each of its tensors: every block of `Q5_0`, `Q4_K` and `Q6_K` of the
        // file is checked here, not only the six of the vectors.
        let file = include_bytes!("../../bench/gguf/synthetic/synth-Q4_K_M.gguf");
        let listed = oracle::tensors(include_str!(
            "../../bench/gguf/synthetic/ref/tensors-synth-Q4_K_M.gguf.json"
        ));
        let gguf = crate::gguf::Gguf::parse(file).unwrap();
        let mut checked = [0usize; 3];
        for tensor in &listed {
            let info = gguf.tensor(&tensor.name).unwrap();
            let data = gguf.tensor_data(info).unwrap();
            assert_eq!(data.len(), tensor.bytes, "{}", tensor.name);
            let digest = crate::sha256::digest(data);
            assert_eq!(
                crate::sha256::hex(&digest),
                tensor.sha256_raw,
                "{}",
                tensor.name
            );
            let elements = info.elements().unwrap() as usize;
            let mut values = vec![0.0f32; elements];
            let slot = match tensor.kind.as_str() {
                "Q5_0" => {
                    dequantize_row_q5_0(data, &mut values).unwrap();
                    0
                }
                "Q4_K" => {
                    dequantize_row_q4_k(data, &mut values).unwrap();
                    1
                }
                "Q6_K" => {
                    dequantize_row_q6_k(data, &mut values).unwrap();
                    2
                }
                "Q8_0" => continue,
                other => panic!("unexpected type {other}"),
            };
            checked[slot] += 1;
            let mut raw = Vec::with_capacity(4 * elements);
            for value in &values {
                raw.extend_from_slice(&value.to_le_bytes());
            }
            let digest = crate::sha256::digest(&raw);
            assert_eq!(
                crate::sha256::hex(&digest),
                tensor.sha256_f32,
                "{} ({})",
                tensor.name,
                tensor.kind
            );
        }
        // The file has 2 tensors of `Q5_0`, 22 of `Q4_K` and 3 of `Q6_K`.
        assert_eq!(checked, [2, 22, 3]);
    }

    #[test]
    fn a_gguf_block_of_the_wrong_size_is_refused() {
        for (kind, decode) in dequantizers() {
            let (elements, size) = match kind {
                "Q8_0" => (BLOCK, Q8_BLOCK_BYTES),
                "Q5_0" => (BLOCK, Q5_0_BLOCK_BYTES),
                "Q4_K" => (QK_K, Q4_K_BLOCK_BYTES),
                _ => (QK_K, Q6_K_BLOCK_BYTES),
            };
            let block = vec![0u8; size];
            let mut out = vec![0.0f32; elements];
            assert!(decode(&block, &mut out).is_ok(), "{kind}");
            assert!(
                decode(&block[..size - 1], &mut out).is_err(),
                "{kind} short"
            );
            assert!(
                decode(&vec![0u8; size + 1], &mut out).is_err(),
                "{kind} long"
            );
            assert!(decode(&[], &mut out).is_err(), "{kind} empty");
            assert!(
                decode(&block, &mut out[..elements - 1]).is_err(),
                "{kind} out"
            );
            assert!(
                decode(&block, &mut vec![0.0f32; elements + 1]).is_err(),
                "{kind} out"
            );
        }
    }

    #[test]
    fn a_gguf_row_must_be_whole_blocks_of_the_exact_length() {
        let mut out = [0.0f32; 33];
        assert!(dequantize_row_q5_0(&[0u8; 2 * Q5_0_BLOCK_BYTES], &mut out).is_err());
        assert!(dequantize_row_q5_0(&[0u8; 2 * Q5_0_BLOCK_BYTES], &mut out[..64 - 33]).is_err());
        let mut out = [0.0f32; 64];
        assert!(dequantize_row_q5_0(&[0u8; 2 * Q5_0_BLOCK_BYTES], &mut out).is_ok());
        assert!(dequantize_row_q5_0(&[0u8; 2 * Q5_0_BLOCK_BYTES + 1], &mut out).is_err());
        assert!(dequantize_row_q5_0(&[0u8; Q5_0_BLOCK_BYTES], &mut out).is_err());
        assert!(dequantize_row_q5_0(&[], &mut []).is_err());
        let mut out = [0.0f32; 2 * QK_K - 32];
        assert!(dequantize_row_q4_k(&[0u8; 2 * Q4_K_BLOCK_BYTES], &mut out).is_err());
        assert!(dequantize_row_q6_k(&[0u8; 2 * Q6_K_BLOCK_BYTES], &mut out).is_err());
        let mut out = [0.0f32; 2 * QK_K];
        assert!(dequantize_row_q4_k(&[0u8; 2 * Q4_K_BLOCK_BYTES - 1], &mut out).is_err());
        assert!(dequantize_row_q6_k(&[0u8; 2 * Q6_K_BLOCK_BYTES + 1], &mut out).is_err());
        assert!(dequantize_row_q4_k(&[0u8; 2 * Q4_K_BLOCK_BYTES], &mut out).is_ok());
        assert!(dequantize_row_q6_k(&[0u8; 2 * Q6_K_BLOCK_BYTES], &mut out).is_ok());
        assert!(q5_0_row_bytes(31).is_err() && q5_0_row_bytes(0).is_err());
        assert!(q4_k_row_bytes(128).is_err() && q6_k_row_bytes(32).is_err());
        assert!(q4_k_matrix_bytes(usize::MAX, 256).is_err());
    }

    #[test]
    fn a_gguf_scale_that_is_not_finite_is_refused_like_the_eight_bit_one() {
        let mut out = [0.0f32; QK_K];
        for half in [0x7c00u16, 0xfc00, 0x7e00, 0x7c01, 0xffff] {
            assert!(!scale_is_finite(half));
            let mut block = [0u8; Q5_0_BLOCK_BYTES];
            block[..2].copy_from_slice(&half.to_le_bytes());
            assert!(dequantize_block_q5_0(&block, &mut out[..BLOCK]).is_err());
            let mut block = [0u8; Q4_K_BLOCK_BYTES];
            block[..2].copy_from_slice(&half.to_le_bytes());
            assert!(dequantize_block_q4_k(&block, &mut out).is_err(), "d");
            let mut block = [0u8; Q4_K_BLOCK_BYTES];
            block[2..4].copy_from_slice(&half.to_le_bytes());
            assert!(dequantize_block_q4_k(&block, &mut out).is_err(), "dmin");
            let mut block = [0u8; Q6_K_BLOCK_BYTES];
            block[208..].copy_from_slice(&half.to_le_bytes());
            assert!(dequantize_block_q6_k(&block, &mut out).is_err());
            // The row functions report the same error.
            let mut row = [0u8; 2 * Q6_K_BLOCK_BYTES];
            row[Q6_K_BLOCK_BYTES + 208..].copy_from_slice(&half.to_le_bytes());
            assert!(dequantize_row_q6_k(&row, &mut [0.0f32; 2 * QK_K]).is_err());
        }
        // The largest finite binary16 is not refused.
        let mut block = [0u8; Q4_K_BLOCK_BYTES];
        block[..4].copy_from_slice(&[0xff, 0x7b, 0xff, 0x7b]);
        assert!(dequantize_block_q4_k(&block, &mut out).is_ok());
    }
}
