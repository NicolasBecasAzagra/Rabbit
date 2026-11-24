// SPDX-License-Identifier: GPL-2.0-only
//! Blocked vector by matrix multiplication over `bfloat16` weights.
//!
//! A projection is `out[r] = dot(weight[r], input)` with a row major matrix
//! `[rows, columns]`. The output is tiled by [`ROW_TILE`] rows and the columns
//! are walked in [`COLUMN_TILE`] chunks. A tile of four rows of the widest
//! matrix in the reference model (1536 columns) is 12 KiB of weights plus 2 KiB
//! of input, so the whole tile stays in the first level cache while it is
//! consumed. The input chunk is loaded once per tile chunk and used by four
//! rows, and each weight element is read once, in address order, on the one
//! pass the tile makes over the matrix. Nothing is read twice.
//!
//! The weights stay `bfloat16` in the mapping. The kernel widens them to `f32`
//! in registers as it reads them, which costs one shift and one widening per
//! weight and no extra memory traffic. A cached `f32` copy would double the
//! weight footprint from 269 MB to 538 MB for the reference model and double
//! the bytes streamed per token; the conversion instructions are cheaper than
//! that traffic, and the model is never copied out of the initramfs. The price
//! is that the arithmetic units do the conversion on every token instead of
//! once at load.
//!
//! Three vector paths exist. `x86_64` uses AVX2 and FMA when the processor and
//! the kernel report both, and SSE2 otherwise; SSE2 is part of the x86-64
//! baseline, so a kernel that does not enable the extended register state still
//! gets four wide arithmetic. `aarch64` uses the NEON baseline, so the same
//! kernel can be measured by a host benchmark on an ARM machine. The
//! architecture chooses the module at compile time and, on `x86_64`, the
//! feature detection chooses at run time. The scalar path is always available
//! and produces the same values to floating point tolerance. The `bfloat16`
//! vector paths sum eight (or four) products per accumulator and the scalar
//! path sums one, so the rounding order differs; the tokens the runtime
//! generates do not. The scalar Q8 and Q4 kernels instead define a canonical
//! order (eight lanes, a fixed tree, one scale per block) that the vector
//! kernels of those formats must reproduce bit for bit.

use std::sync::atomic::{AtomicU8, Ordering};

use crate::error::{Error, Result};

/// Rows of the output computed together.
pub const ROW_TILE: usize = 4;

/// Columns walked together. A multiple of the widest vector step, 8.
pub const COLUMN_TILE: usize = 512;

/// Multiply accumulates one matrix vector product performs.
pub fn macs(rows: usize, columns: usize) -> u64 {
    rows as u64 * columns as u64
}

/// A code path of the `bfloat16` product.
///
/// The paths are ordered from the least to the most capable one, so a limit and
/// a detection compare with `<`. The variants that exist depend on the
/// architecture: `x86_64` has the three below, `aarch64` has the scalar path
/// and NEON, and any other architecture has the scalar path alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Path {
    /// One product at a time, on any processor.
    Scalar = 0,
    /// Four wide SSE2, the `x86_64` baseline.
    #[cfg(target_arch = "x86_64")]
    Sse2 = 1,
    /// Eight wide AVX2. The `bfloat16` kernel uses fused multiply add here and
    /// the Q8 kernel does not, so it can match the scalar path bit for bit.
    #[cfg(target_arch = "x86_64")]
    Avx2 = 2,
    /// Four wide NEON, the `aarch64` baseline.
    #[cfg(target_arch = "aarch64")]
    Neon = 1,
}

impl Path {
    /// The name of the path in the serial log.
    pub fn name(self) -> &'static str {
        match self {
            Path::Scalar => "scalar",
            #[cfg(target_arch = "x86_64")]
            Path::Sse2 => "sse2",
            #[cfg(target_arch = "x86_64")]
            Path::Avx2 => "avx2",
            #[cfg(target_arch = "aarch64")]
            Path::Neon => "neon",
        }
    }
}

/// The best path the processor offers. AVX2 needs the processor and the kernel
/// to report both `avx2` and `fma`.
#[cfg(target_arch = "x86_64")]
pub fn detected() -> Path {
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        Path::Avx2
    } else {
        // SSE2 is part of the x86-64 baseline, so no detection is needed. A
        // kernel that does not enable the extended register state reports
        // AVX2 as absent, and this path still uses four wide arithmetic.
        Path::Sse2
    }
}

/// The best path the processor offers.
#[cfg(target_arch = "aarch64")]
pub fn detected() -> Path {
    Path::Neon
}

/// The best path the processor offers.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn detected() -> Path {
    Path::Scalar
}

/// The path to run for a request: the lower of `limit` and [`detected`]. A
/// request never selects a path the processor does not have.
pub fn select(limit: Path) -> Path {
    limit.min(detected())
}

/// The code of "no limit" in [`LIMIT`]; no path has it.
const NO_LIMIT: u8 = u8::MAX;

/// The process wide limit of [`set_path_limit`], as the code of a [`Path`].
static LIMIT: AtomicU8 = AtomicU8::new(NO_LIMIT);

/// Limit the path of [`matvec_bf16`], [`matvec_q8`] and [`matvec_q4`] for the
/// whole process, or lift the limit with `None`.
///
/// Opt in and off by default: with no limit the three products run on
/// [`detected`], exactly as before, and the runtime never calls this. The
/// guest check of the whole model calls it to decode the same prompts on one
/// path after the other, because the decoder reaches the products through
/// those three functions and has no parameter to carry a path.
///
/// The limit is global state. A test that runs in a thread of the parallel test
/// harness must not call it, because every other test would run on the
/// limited path; tests use [`matvec_bf16_with`], [`matvec_q8_with`] and
/// [`matvec_q4_with`] instead. Call it from a single threaded program, before
/// the products it is meant to affect. As with [`select`], a limit above what
/// the processor has runs on [`detected`].
pub fn set_path_limit(limit: Option<Path>) {
    LIMIT.store(limit.map_or(NO_LIMIT, |path| path as u8), Ordering::Relaxed);
}

/// The path the three products of the process run on: [`detected`] when no
/// limit is set, and the limit lowered by [`select`] otherwise.
pub fn active() -> Path {
    match path_from_code(LIMIT.load(Ordering::Relaxed)) {
        Some(limit) => select(limit),
        None => detected(),
    }
}

/// The path with this code, or `None` for [`NO_LIMIT`] and any other value.
fn path_from_code(code: u8) -> Option<Path> {
    match code {
        0 => Some(Path::Scalar),
        #[cfg(target_arch = "x86_64")]
        1 => Some(Path::Sse2),
        #[cfg(target_arch = "x86_64")]
        2 => Some(Path::Avx2),
        #[cfg(target_arch = "aarch64")]
        1 => Some(Path::Neon),
        _ => None,
    }
}

/// Multiply `weights` by `input`, writing one value per row.
///
/// `weights` is a row major `bfloat16` matrix of `rows * columns` elements and
/// `input` has `columns` values. The caller validated the shapes; a length that
/// still disagrees is reported as an error rather than indexed. The product
/// runs on the best path the processor offers, or on the lower path a
/// [`set_path_limit`] asked for.
pub fn matvec_bf16(
    weights: &[u8],
    rows: usize,
    columns: usize,
    input: &[f32],
    out: &mut [f32],
) -> Result<()> {
    matvec_bf16_with(active(), weights, rows, columns, input, out)
}

/// The product of [`matvec_bf16`] on an explicit path, for the tests that
/// compare paths. The path is lowered to what the processor
/// has through [`select`], so a request above it runs on the best path
/// available instead of an instruction the processor lacks.
pub fn matvec_bf16_with(
    path: Path,
    weights: &[u8],
    rows: usize,
    columns: usize,
    input: &[f32],
    out: &mut [f32],
) -> Result<()> {
    validate(weights, rows, columns, input, out)?;
    match select(path) {
        Path::Scalar => matvec_scalar(weights, rows, columns, input, out),
        #[cfg(target_arch = "x86_64")]
        Path::Sse2 => {
            // SAFETY: SSE2 is part of the x86-64 baseline, and `validate`
            // checked every length the kernel indexes.
            unsafe {
                sse2::matvec(weights, rows, columns, input, out);
            }
        }
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 => {
            // SAFETY: `select` only returns this path when the detection
            // reported both features, and `validate` checked every length the
            // kernel indexes.
            unsafe {
                x86::matvec(weights, rows, columns, input, out);
            }
        }
        #[cfg(target_arch = "aarch64")]
        Path::Neon => {
            // SAFETY: NEON is part of the aarch64 baseline, and `validate`
            // checked every length the kernel indexes.
            unsafe {
                neon::matvec(weights, rows, columns, input, out);
            }
        }
    }
    Ok(())
}

/// The scalar kernel, used by the benchmark as the baseline and by a processor
/// without a vector path.
pub fn matvec_scalar(weights: &[u8], rows: usize, columns: usize, input: &[f32], out: &mut [f32]) {
    for (row, slot) in out.iter_mut().enumerate().take(rows) {
        let mut sum = 0.0f32;
        for (column, &value) in input.iter().enumerate().take(columns) {
            let offset = (row * columns + column) * 2;
            let bits = u16::from_le_bytes([weights[offset], weights[offset + 1]]);
            sum += f32::from_bits((bits as u32) << 16) * value;
        }
        *slot = sum;
    }
}

// The block is a whole number of eight wide groups, so the canonical order of
// the block dot products below has no tail.
const _: () = assert!(crate::quant::BLOCK % 8 == 0);

/// Sum eight lane accumulators with the fixed tree of the block dot products.
///
/// `t[j] = lane[j] + lane[j + 4]` for `j` in `0..4`, then
/// `u[j] = t[j] + t[j + 2]` for `j` in `0..2`, then `u[0] + u[1]`. The tree is
/// the one a vector kernel gets by adding the high half of a register to the
/// low half until one lane is left, so a kernel can be tested against this
/// function bit for bit.
#[inline]
fn reduce8(lane: [f32; 8]) -> f32 {
    let t0 = lane[0] + lane[4];
    let t1 = lane[1] + lane[5];
    let t2 = lane[2] + lane[6];
    let t3 = lane[3] + lane[7];
    let u0 = t0 + t2;
    let u1 = t1 + t3;
    u0 + u1
}

/// Multiply a `Q4B32` matrix by `input`, writing one value per row.
///
/// The product runs on the best path the processor offers (or the lower one a
/// [`set_path_limit`] asked for), and every path gives
/// the same bits as [`matvec_q4_scalar`], which defines the order of the
/// floating point operations.
pub fn matvec_q4(
    weights: &[u8],
    rows: usize,
    columns: usize,
    input: &[f32],
    out: &mut [f32],
) -> Result<()> {
    matvec_q4_with(active(), weights, rows, columns, input, out)
}

/// The product of [`matvec_q4`] on an explicit path, for the tests and the
/// guest check that compare paths. The path is lowered to what the processor
/// has through [`select`], so a request above it runs on the best path
/// available instead of an instruction the processor lacks. `aarch64` and the
/// other architectures have no vector kernel for this format yet, so every path
/// runs the scalar one there.
pub fn matvec_q4_with(
    path: Path,
    weights: &[u8],
    rows: usize,
    columns: usize,
    input: &[f32],
    out: &mut [f32],
) -> Result<()> {
    validate_q4(weights, rows, columns, input, out)?;
    match select(path) {
        #[cfg(target_arch = "x86_64")]
        Path::Sse2 => {
            // SAFETY: SSE2 is part of the x86-64 baseline, and `validate_q4`
            // checked every length the kernel indexes.
            unsafe {
                q4_x86::sse2(weights, rows, columns, input, out);
            }
            Ok(())
        }
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 => {
            // SAFETY: `select` only returns this path when the detection
            // reported `avx2`, and `validate_q4` checked every length the
            // kernel indexes.
            unsafe {
                q4_x86::avx2(weights, rows, columns, input, out);
            }
            Ok(())
        }
        _ => matvec_q4_scalar(weights, rows, columns, input, out),
    }
}

/// The scalar four bit kernel: the reference the vector kernels reproduce.
///
/// The weights are never expanded into a floating point matrix: the kernel
/// reads one block of [`crate::quant::BLOCK_BYTES`] bytes, applies the scale of
/// that block to the sum of its thirty-two products, and moves on. The memory
/// traffic is 18 bytes for 32 weights, against 64 bytes for the same weights in
/// `bfloat16`, which is what the format buys.
///
/// The nibble is sign extended before the product and the block sum is scaled
/// once, so the rounding of the scale is not repeated thirty-two times.
///
/// The order of the floating point operations is canonical, so that a vector
/// kernel can reproduce the result bit for bit. Weight `k` of a block is the
/// low nibble of packed byte `k / 2` when `k` is even and the high nibble when
/// it is odd, sign extended and converted to `f32`. For one block, eight lane
/// accumulators `acc[0..8]` start at zero and the block is walked in groups of
/// eight consecutive weights: for each group `g` and lane `j`,
/// `acc[j] = acc[j] + w[8g + j] * x[8g + j]`, with a separate multiply and add
/// (no fused multiply add). The lanes are then summed by [`reduce8`], giving
/// `block_sum`, and the row is `row = row + block_sum * scale` over the blocks
/// in order, starting at `0.0`. The columns are a whole number of blocks and a
/// block is a whole number of groups, so there is no tail.
pub fn matvec_q4_scalar(
    weights: &[u8],
    rows: usize,
    columns: usize,
    input: &[f32],
    out: &mut [f32],
) -> Result<()> {
    validate_q4(weights, rows, columns, input, out)?;
    let row_bytes = crate::quant::row_bytes(columns)?;
    let blocks = crate::quant::blocks(columns);
    for (row, slot) in out.iter_mut().enumerate().take(rows) {
        let base = row * row_bytes;
        let mut sum = 0.0f32;
        for block in 0..blocks {
            let offset = base + block * crate::quant::BLOCK_BYTES;
            let scale = crate::quant::half_to_f32(u16::from_le_bytes([
                weights[offset],
                weights[offset + 1],
            ]));
            let packed =
                &weights[offset + crate::quant::SCALE_BYTES..offset + crate::quant::BLOCK_BYTES];
            let column = block * crate::quant::BLOCK;
            let mut lane = [0.0f32; 8];
            for group in 0..crate::quant::BLOCK / 8 {
                for (j, slot) in lane.iter_mut().enumerate() {
                    let k = group * 8 + j;
                    let nibble = (packed[k / 2] >> (4 * (k % 2))) & 0x0f;
                    let weight = ((nibble as i32) << 28 >> 28) as f32;
                    let product = weight * input[column + k];
                    *slot += product;
                }
            }
            let block_sum = reduce8(lane);
            sum += block_sum * scale;
        }
        *slot = sum;
    }
    Ok(())
}

/// Name of the vector kernel this build selects, for the serial log.
#[cfg(target_arch = "x86_64")]
pub fn kernel_name() -> &'static str {
    match detected() {
        Path::Avx2 => "avx2+fma",
        _ => "sse2",
    }
}

/// Name of the vector kernel this build selects, for the serial log.
#[cfg(target_arch = "aarch64")]
pub fn kernel_name() -> &'static str {
    "neon"
}

/// Name of the vector kernel this build selects, for the serial log.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn kernel_name() -> &'static str {
    "scalar"
}

/// Multiply an eight bit block matrix by `input`, writing one value per row.
///
/// The product runs on the best path the processor offers (or the lower one a
/// [`set_path_limit`] asked for), and every path gives
/// the same bits as [`matvec_q8_scalar`], which defines the order of the
/// floating point operations.
pub fn matvec_q8(
    weights: &[u8],
    rows: usize,
    columns: usize,
    input: &[f32],
    out: &mut [f32],
) -> Result<()> {
    matvec_q8_with(active(), weights, rows, columns, input, out)
}

/// The product of [`matvec_q8`] on an explicit path, for the tests and the
/// guest check that compare paths. The path is lowered to what the processor
/// has through [`select`], so a request above it runs on the best path
/// available instead of an instruction the processor lacks. `aarch64` and the
/// other architectures have no vector kernel for this format yet, so every path
/// runs the scalar one there.
pub fn matvec_q8_with(
    path: Path,
    weights: &[u8],
    rows: usize,
    columns: usize,
    input: &[f32],
    out: &mut [f32],
) -> Result<()> {
    validate_q8(weights, rows, columns, input, out)?;
    match select(path) {
        #[cfg(target_arch = "x86_64")]
        Path::Sse2 => {
            // SAFETY: SSE2 is part of the x86-64 baseline, and `validate_q8`
            // checked every length the kernel indexes.
            unsafe {
                q8_x86::sse2(weights, rows, columns, input, out);
            }
            Ok(())
        }
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 => {
            // SAFETY: `select` only returns this path when the detection
            // reported `avx2`, and `validate_q8` checked every length the
            // kernel indexes.
            unsafe {
                q8_x86::avx2(weights, rows, columns, input, out);
            }
            Ok(())
        }
        _ => matvec_q8_scalar(weights, rows, columns, input, out),
    }
}

/// The scalar eight bit kernel: the reference the vector kernels reproduce.
///
/// The loop is the four bit one with a wider payload: the block is read, its
/// scale is applied to the sum of the thirty-two products, and the weights are
/// never expanded into a floating point matrix. The reason the eight bit form
/// exists is fidelity, not speed: it costs 34 bytes per 32 weights instead of
/// 18, and it is the width that keeps the tokens of the reference checkpoint.
///
/// The order of the floating point operations is canonical, so that a vector
/// kernel can reproduce the result bit for bit. Weight `k` of a block is the
/// signed byte `k` after the scale, converted to `f32`. For one block, eight
/// lane accumulators `acc[0..8]` start at zero and the block is walked in
/// groups of eight consecutive weights: for each group `g` and lane `j`,
/// `acc[j] = acc[j] + w[8g + j] * x[8g + j]`, with a separate multiply and add
/// (no fused multiply add). The lanes are then summed by [`reduce8`], giving
/// `block_sum`, and the row is `row = row + block_sum * scale` over the blocks
/// in order, starting at `0.0`. The columns are a whole number of blocks and a
/// block is a whole number of groups, so there is no tail.
pub fn matvec_q8_scalar(
    weights: &[u8],
    rows: usize,
    columns: usize,
    input: &[f32],
    out: &mut [f32],
) -> Result<()> {
    validate_q8(weights, rows, columns, input, out)?;
    let row_bytes = crate::quant::q8_row_bytes(columns)?;
    let blocks = crate::quant::blocks(columns);
    for (row, slot) in out.iter_mut().enumerate().take(rows) {
        let base = row * row_bytes;
        let mut sum = 0.0f32;
        for block in 0..blocks {
            let offset = base + block * crate::quant::Q8_BLOCK_BYTES;
            let scale = crate::quant::half_to_f32(u16::from_le_bytes([
                weights[offset],
                weights[offset + 1],
            ]));
            let quantized =
                &weights[offset + crate::quant::SCALE_BYTES..offset + crate::quant::Q8_BLOCK_BYTES];
            let column = block * crate::quant::BLOCK;
            let mut lane = [0.0f32; 8];
            for group in 0..crate::quant::BLOCK / 8 {
                for (j, slot) in lane.iter_mut().enumerate() {
                    let k = group * 8 + j;
                    let weight = quantized[k] as i8 as f32;
                    let product = weight * input[column + k];
                    *slot += product;
                }
            }
            let block_sum = reduce8(lane);
            sum += block_sum * scale;
        }
        *slot = sum;
    }
    Ok(())
}

/// Check every length the `bfloat16` kernel indexes.
fn validate(weights: &[u8], rows: usize, columns: usize, input: &[f32], out: &[f32]) -> Result<()> {
    if rows == 0 || columns == 0 {
        return Err(Error::new("matvec: an empty matrix has no product"));
    }
    if input.len() != columns {
        return Err(Error::new(
            "matvec: input length disagrees with the columns",
        ));
    }
    if out.len() < rows {
        return Err(Error::new("matvec: output is shorter than the rows"));
    }
    let needed = rows
        .checked_mul(columns)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| Error::new("matvec: matrix byte length overflow"))?;
    if weights.len() < needed {
        return Err(Error::new("matvec: weights are shorter than the matrix"));
    }
    Ok(())
}

/// Check every length and the block alignment the `Q4B32` kernel indexes.
fn validate_q4(
    weights: &[u8],
    rows: usize,
    columns: usize,
    input: &[f32],
    out: &[f32],
) -> Result<()> {
    if rows == 0 || columns == 0 {
        return Err(Error::new("matvec: an empty matrix has no product"));
    }
    if input.len() != columns {
        return Err(Error::new(
            "matvec: input length disagrees with the columns",
        ));
    }
    if out.len() < rows {
        return Err(Error::new("matvec: output is shorter than the rows"));
    }
    if !columns.is_multiple_of(crate::quant::BLOCK) {
        return Err(Error::new(format!(
            "matvec: {columns} columns is not a multiple of the {} weight block",
            crate::quant::BLOCK
        )));
    }
    let needed = crate::quant::matrix_bytes(rows, columns)?;
    if weights.len() < needed {
        return Err(Error::new(
            "matvec: quantized weights are shorter than the matrix",
        ));
    }
    Ok(())
}

/// Check every length and the block alignment the `Q8B32` kernel indexes.
fn validate_q8(
    weights: &[u8],
    rows: usize,
    columns: usize,
    input: &[f32],
    out: &[f32],
) -> Result<()> {
    if rows == 0 || columns == 0 {
        return Err(Error::new("matvec: an empty matrix has no product"));
    }
    if input.len() != columns {
        return Err(Error::new(
            "matvec: input length disagrees with the columns",
        ));
    }
    if out.len() < rows {
        return Err(Error::new("matvec: output is shorter than the rows"));
    }
    if !columns.is_multiple_of(crate::quant::BLOCK) {
        return Err(Error::new(format!(
            "matvec: {columns} columns is not a multiple of the {} weight block",
            crate::quant::BLOCK
        )));
    }
    let needed = crate::quant::q8_matrix_bytes(rows, columns)?;
    if weights.len() < needed {
        return Err(Error::new(
            "matvec: quantized weights are shorter than the matrix",
        ));
    }
    Ok(())
}

/// The AVX2 and FMA kernel.
#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{COLUMN_TILE, ROW_TILE};
    use core::arch::x86_64::{
        __m128i, __m256, _mm256_castps256_ps128, _mm256_castsi256_ps, _mm256_cvtepu16_epi32,
        _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps,
        _mm256_slli_epi32, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_loadu_si128, _mm_movehdup_ps,
        _mm_movehl_ps,
    };

    /// # Safety
    ///
    /// The caller must have validated the lengths and must run on a processor
    /// that reports `avx2` and `fma`.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn matvec(
        weights: &[u8],
        rows: usize,
        columns: usize,
        input: &[f32],
        out: &mut [f32],
    ) {
        // SAFETY: the caller validated every length, the pointers stay inside
        // the slices for each load, and the target features are present.
        unsafe {
            let aligned = columns & !7;
            let mut row_start = 0;
            while row_start + ROW_TILE <= rows {
                let mut accumulator = [_mm256_setzero_ps(); ROW_TILE];
                let mut column_start = 0;
                while column_start < aligned {
                    let end = (column_start + COLUMN_TILE).min(aligned);
                    let mut column = column_start;
                    while column < end {
                        let input_vector = _mm256_loadu_ps(input.as_ptr().add(column));
                        let mut lane = 0;
                        while lane < ROW_TILE {
                            let offset = ((row_start + lane) * columns + column) * 2;
                            let packed =
                                _mm_loadu_si128(weights.as_ptr().add(offset) as *const __m128i);
                            let widened = _mm256_cvtepu16_epi32(packed);
                            let shifted = _mm256_slli_epi32(widened, 16);
                            let values = _mm256_castsi256_ps(shifted);
                            accumulator[lane] =
                                _mm256_fmadd_ps(values, input_vector, accumulator[lane]);
                            lane += 1;
                        }
                        column += 8;
                    }
                    column_start = end;
                }
                for (lane, accumulator) in accumulator.iter().enumerate() {
                    let mut sum = horizontal(*accumulator);
                    for (column, &input_value) in
                        input.iter().enumerate().take(columns).skip(aligned)
                    {
                        let offset = ((row_start + lane) * columns + column) * 2;
                        let bits = u16::from_le_bytes([weights[offset], weights[offset + 1]]);
                        sum += f32::from_bits((bits as u32) << 16) * input_value;
                    }
                    out[row_start + lane] = sum;
                }
                row_start += ROW_TILE;
            }
            while row_start < rows {
                out[row_start] = dot_row(weights, row_start, columns, input);
                row_start += 1;
            }
        }
    }

    /// One row of the tail, after the four row tiles.
    ///
    /// # Safety
    ///
    /// The lengths are validated by `matvec` and the target features are
    /// present.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_row(weights: &[u8], row: usize, columns: usize, input: &[f32]) -> f32 {
        // SAFETY: the row and column ranges are inside the slices.
        unsafe {
            let aligned = columns & !7;
            let mut accumulator = _mm256_setzero_ps();
            let mut column = 0;
            while column < aligned {
                let offset = (row * columns + column) * 2;
                let packed = _mm_loadu_si128(weights.as_ptr().add(offset) as *const __m128i);
                let widened = _mm256_cvtepu16_epi32(packed);
                let shifted = _mm256_slli_epi32(widened, 16);
                let values = _mm256_castsi256_ps(shifted);
                let input_vector = _mm256_loadu_ps(input.as_ptr().add(column));
                accumulator = _mm256_fmadd_ps(values, input_vector, accumulator);
                column += 8;
            }
            let mut sum = horizontal(accumulator);
            for (column, &input_value) in input.iter().enumerate().take(columns).skip(aligned) {
                let offset = (row * columns + column) * 2;
                let bits = u16::from_le_bytes([weights[offset], weights[offset + 1]]);
                sum += f32::from_bits((bits as u32) << 16) * input_value;
            }
            sum
        }
    }

    /// Sum the eight lanes of one accumulator.
    ///
    /// # Safety
    ///
    /// The caller must run on a processor with AVX2.
    #[target_feature(enable = "avx2")]
    unsafe fn horizontal(value: __m256) -> f32 {
        let high = _mm256_extractf128_ps(value, 1);
        let low = _mm256_castps256_ps128(value);
        let mut sum = _mm_add_ps(low, high);
        let duplicated = _mm_movehdup_ps(sum);
        sum = _mm_add_ps(sum, duplicated);
        let moved = _mm_movehl_ps(sum, sum);
        sum = _mm_add_ss(sum, moved);
        _mm_cvtss_f32(sum)
    }
}

/// The SSE2 kernel, used when AVX2 and FMA are not available.
#[cfg(target_arch = "x86_64")]
mod sse2 {
    use super::{COLUMN_TILE, ROW_TILE};
    use core::arch::x86_64::{
        __m128, __m128i, _mm_add_ps, _mm_add_ss, _mm_castsi128_ps, _mm_cvtss_f32, _mm_loadl_epi64,
        _mm_loadu_ps, _mm_movehl_ps, _mm_mul_ps, _mm_setzero_ps, _mm_setzero_si128, _mm_shuffle_ps,
        _mm_slli_epi32, _mm_unpacklo_epi16,
    };

    /// # Safety
    ///
    /// The caller must have validated the lengths. SSE2 is part of the x86-64
    /// baseline.
    pub(super) unsafe fn matvec(
        weights: &[u8],
        rows: usize,
        columns: usize,
        input: &[f32],
        out: &mut [f32],
    ) {
        // SAFETY: the caller validated every length, and every load stays
        // inside the slices.
        unsafe {
            let zero = _mm_setzero_ps();
            let aligned = columns & !7;
            let mut row_start = 0;
            while row_start + ROW_TILE <= rows {
                let mut accumulator = [[zero; 2]; ROW_TILE];
                let mut column_start = 0;
                while column_start < aligned {
                    let end = (column_start + COLUMN_TILE).min(aligned);
                    let mut column = column_start;
                    while column < end {
                        let input_low = _mm_loadu_ps(input.as_ptr().add(column));
                        let input_high = _mm_loadu_ps(input.as_ptr().add(column + 4));
                        let mut lane = 0;
                        while lane < ROW_TILE {
                            let offset = ((row_start + lane) * columns + column) * 2;
                            let low = widen(&weights[offset..]);
                            let high = widen(&weights[offset + 8..]);
                            accumulator[lane][0] =
                                _mm_add_ps(accumulator[lane][0], _mm_mul_ps(low, input_low));
                            accumulator[lane][1] =
                                _mm_add_ps(accumulator[lane][1], _mm_mul_ps(high, input_high));
                            lane += 1;
                        }
                        column += 8;
                    }
                    column_start = end;
                }
                for (lane, accumulator) in accumulator.iter().enumerate() {
                    let mut sum = horizontal(_mm_add_ps(accumulator[0], accumulator[1]));
                    for (column, &input_value) in
                        input.iter().enumerate().take(columns).skip(aligned)
                    {
                        let offset = ((row_start + lane) * columns + column) * 2;
                        let bits = u16::from_le_bytes([weights[offset], weights[offset + 1]]);
                        sum += f32::from_bits((bits as u32) << 16) * input_value;
                    }
                    out[row_start + lane] = sum;
                }
                row_start += ROW_TILE;
            }
            while row_start < rows {
                let mut sum = 0.0f32;
                for (column, &input_value) in input.iter().enumerate().take(columns) {
                    let offset = (row_start * columns + column) * 2;
                    let bits = u16::from_le_bytes([weights[offset], weights[offset + 1]]);
                    sum += f32::from_bits((bits as u32) << 16) * input_value;
                }
                out[row_start] = sum;
                row_start += 1;
            }
        }
    }

    /// Widen four `bfloat16` values to `f32`.
    ///
    /// # Safety
    ///
    /// `bytes` must hold at least eight bytes.
    #[target_feature(enable = "sse2")]
    unsafe fn widen(bytes: &[u8]) -> __m128 {
        // SAFETY: the caller guarantees the eight bytes the load reads.
        unsafe {
            let packed: __m128i = _mm_loadl_epi64(bytes.as_ptr() as *const __m128i);
            let widened = _mm_unpacklo_epi16(packed, _mm_setzero_si128());
            let shifted = _mm_slli_epi32(widened, 16);
            _mm_castsi128_ps(shifted)
        }
    }

    /// Sum the four lanes of one accumulator.
    ///
    /// # Safety
    ///
    /// The caller must run on a processor with SSE2.
    #[target_feature(enable = "sse2")]
    unsafe fn horizontal(value: __m128) -> f32 {
        let moved = _mm_movehl_ps(value, value);
        let mut sum = _mm_add_ps(value, moved);
        let duplicated = _mm_shuffle_ps::<0x55>(sum, sum);
        sum = _mm_add_ss(sum, duplicated);
        _mm_cvtss_f32(sum)
    }
}

/// The SSE2 and AVX2 kernels of the four bit product. Both reproduce the
/// canonical order of [`matvec_q4_scalar`] bit for bit: eight lane accumulators
/// that start at zero, a separate multiply and add per lane (no fused multiply
/// add, and the `fma` feature is not enabled), the fixed tree of `reduce8`, and
/// the scale and the row sum in scalar `f32` in block order.
#[cfg(target_arch = "x86_64")]
mod q4_x86 {
    use crate::quant::{half_to_f32, BLOCK, BLOCK_BYTES, SCALE_BYTES};
    use core::arch::x86_64::{
        __m128, __m128i, _mm256_add_ps, _mm256_castps256_ps128, _mm256_cvtepi32_ps,
        _mm256_cvtepi8_epi32, _mm256_extractf128_ps, _mm256_loadu_ps, _mm256_mul_ps,
        _mm256_setzero_ps, _mm_add_ps, _mm_add_ss, _mm_and_si128, _mm_cmplt_epi16, _mm_cmplt_epi8,
        _mm_cvtepi32_ps, _mm_cvtsi32_si128, _mm_cvtss_f32, _mm_loadu_ps, _mm_movehl_ps, _mm_mul_ps,
        _mm_set1_epi8, _mm_setzero_ps, _mm_setzero_si128, _mm_shuffle_ps, _mm_srli_epi16,
        _mm_sub_epi8, _mm_unpackhi_epi16, _mm_unpacklo_epi16, _mm_unpacklo_epi8, _mm_xor_si128,
    };

    /// The binary16 scale at the start of the block at `offset`.
    #[inline(always)]
    fn block_scale(weights: &[u8], offset: usize) -> f32 {
        half_to_f32(u16::from_le_bytes([weights[offset], weights[offset + 1]]))
    }

    /// The eight weights of the four packed bytes at `packed` as signed bytes
    /// in the low half of a register: lane `2 i` is the low nibble of byte `i`
    /// and lane `2 i + 1` is the high nibble, each sign extended from four bits.
    ///
    /// # Safety
    ///
    /// `packed` must be valid for reading four bytes.
    #[target_feature(enable = "sse2")]
    unsafe fn nibbles(packed: *const u8) -> __m128i {
        // SAFETY: the caller guarantees the four bytes the load reads, and
        // SSE2 is part of the x86-64 baseline.
        unsafe {
            let four = (packed as *const i32).read_unaligned();
            let v = _mm_cvtsi32_si128(four);
            let mask = _mm_set1_epi8(0x0f);
            let low = _mm_and_si128(v, mask);
            let high = _mm_and_si128(_mm_srli_epi16::<4>(v), mask);
            // Lanes 0 to 7 as the nibbles 0 to 15, then the four bit two's
            // complement value: (n ^ 8) - 8.
            let bytes = _mm_unpacklo_epi8(low, high);
            let eight = _mm_set1_epi8(8);
            _mm_sub_epi8(_mm_xor_si128(bytes, eight), eight)
        }
    }

    /// The AVX2 kernel.
    ///
    /// # Safety
    ///
    /// The caller must have validated the lengths with `validate_q4` and must
    /// run on a processor that reports `avx2`.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn avx2(
        weights: &[u8],
        rows: usize,
        columns: usize,
        input: &[f32],
        out: &mut [f32],
    ) {
        let blocks = columns / BLOCK;
        for (row, slot) in out.iter_mut().enumerate().take(rows) {
            let mut sum = 0.0f32;
            for block in 0..blocks {
                let offset = (row * blocks + block) * BLOCK_BYTES;
                let scale = block_scale(weights, offset);
                // SAFETY: `validate_q4` checked that the matrix holds
                // `rows * blocks` whole blocks and that `input` has
                // `columns` values, so every 4 byte load of the 16 byte
                // payload and every 8 value input load below is inside its
                // slice, and `avx2` is present.
                let block_sum = unsafe {
                    let packed = weights.as_ptr().add(offset + SCALE_BYTES);
                    let x = input.as_ptr().add(block * BLOCK);
                    let mut acc = _mm256_setzero_ps();
                    let mut group = 0;
                    while group < BLOCK / 8 {
                        let bytes = nibbles(packed.add(4 * group));
                        let w = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
                        let v = _mm256_loadu_ps(x.add(8 * group));
                        acc = _mm256_add_ps(acc, _mm256_mul_ps(w, v));
                        group += 1;
                    }
                    // t[j] = acc[j] + acc[j + 4], u[j] = t[j] + t[j + 2],
                    // then u[0] + u[1].
                    let t = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
                    let u = _mm_add_ps(t, _mm_movehl_ps(t, t));
                    _mm_cvtss_f32(_mm_add_ss(u, _mm_shuffle_ps::<0x55>(u, u)))
                };
                sum += block_sum * scale;
            }
            *slot = sum;
        }
    }

    /// Eight signed bytes in the low half of `bytes` as two vectors of four
    /// `f32`.
    #[target_feature(enable = "sse2")]
    fn widen(bytes: __m128i) -> (__m128, __m128) {
        let zero = _mm_setzero_si128();
        // Sign extend by interleaving each byte with its sign mask.
        let words = _mm_unpacklo_epi8(bytes, _mm_cmplt_epi8(bytes, zero));
        let sign = _mm_cmplt_epi16(words, zero);
        let low = _mm_unpacklo_epi16(words, sign);
        let high = _mm_unpackhi_epi16(words, sign);
        (_mm_cvtepi32_ps(low), _mm_cvtepi32_ps(high))
    }

    /// The SSE2 kernel: lanes 0 to 3 in one register, lanes 4 to 7 in another.
    ///
    /// # Safety
    ///
    /// The caller must have validated the lengths with `validate_q4`. SSE2 is
    /// part of the x86-64 baseline.
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn sse2(
        weights: &[u8],
        rows: usize,
        columns: usize,
        input: &[f32],
        out: &mut [f32],
    ) {
        let blocks = columns / BLOCK;
        for (row, slot) in out.iter_mut().enumerate().take(rows) {
            let mut sum = 0.0f32;
            for block in 0..blocks {
                let offset = (row * blocks + block) * BLOCK_BYTES;
                let scale = block_scale(weights, offset);
                // SAFETY: `validate_q4` checked that the matrix holds
                // `rows * blocks` whole blocks and that `input` has
                // `columns` values, so every 4 byte load of the 16 byte
                // payload and every 4 value input load below is inside its
                // slice.
                let block_sum = unsafe {
                    let packed = weights.as_ptr().add(offset + SCALE_BYTES);
                    let x = input.as_ptr().add(block * BLOCK);
                    let mut low = _mm_setzero_ps();
                    let mut high = _mm_setzero_ps();
                    let mut group = 0;
                    while group < BLOCK / 8 {
                        let (w_low, w_high) = widen(nibbles(packed.add(4 * group)));
                        let v_low = _mm_loadu_ps(x.add(8 * group));
                        let v_high = _mm_loadu_ps(x.add(8 * group + 4));
                        low = _mm_add_ps(low, _mm_mul_ps(w_low, v_low));
                        high = _mm_add_ps(high, _mm_mul_ps(w_high, v_high));
                        group += 1;
                    }
                    // t[j] = acc[j] + acc[j + 4], u[j] = t[j] + t[j + 2],
                    // then u[0] + u[1].
                    let t = _mm_add_ps(low, high);
                    let u = _mm_add_ps(t, _mm_movehl_ps(t, t));
                    _mm_cvtss_f32(_mm_add_ss(u, _mm_shuffle_ps::<0x55>(u, u)))
                };
                sum += block_sum * scale;
            }
            *slot = sum;
        }
    }
}

/// The SSE2 and AVX2 kernels of the eight bit product. Both reproduce the
/// canonical order of [`matvec_q8_scalar`] bit for bit: eight lane accumulators
/// that start at zero, a separate multiply and add per lane (no fused multiply
/// add, and the `fma` feature is not enabled), the fixed tree of `reduce8`, and
/// the scale and the row sum in scalar `f32` in block order.
#[cfg(target_arch = "x86_64")]
mod q8_x86 {
    use crate::quant::{half_to_f32, BLOCK, Q8_BLOCK_BYTES, SCALE_BYTES};
    use core::arch::x86_64::{
        __m128, __m128i, _mm256_add_ps, _mm256_castps256_ps128, _mm256_cvtepi32_ps,
        _mm256_cvtepi8_epi32, _mm256_extractf128_ps, _mm256_loadu_ps, _mm256_mul_ps,
        _mm256_setzero_ps, _mm_add_ps, _mm_add_ss, _mm_cmplt_epi16, _mm_cmplt_epi8,
        _mm_cvtepi32_ps, _mm_cvtss_f32, _mm_loadl_epi64, _mm_loadu_ps, _mm_movehl_ps, _mm_mul_ps,
        _mm_setzero_ps, _mm_setzero_si128, _mm_shuffle_ps, _mm_unpackhi_epi16, _mm_unpacklo_epi16,
        _mm_unpacklo_epi8,
    };

    /// The binary16 scale at the start of the block at `offset`.
    #[inline(always)]
    fn block_scale(weights: &[u8], offset: usize) -> f32 {
        half_to_f32(u16::from_le_bytes([weights[offset], weights[offset + 1]]))
    }

    /// The AVX2 kernel.
    ///
    /// # Safety
    ///
    /// The caller must have validated the lengths with `validate_q8` and must
    /// run on a processor that reports `avx2`.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn avx2(
        weights: &[u8],
        rows: usize,
        columns: usize,
        input: &[f32],
        out: &mut [f32],
    ) {
        let blocks = columns / BLOCK;
        for (row, slot) in out.iter_mut().enumerate().take(rows) {
            let mut sum = 0.0f32;
            for block in 0..blocks {
                let offset = (row * blocks + block) * Q8_BLOCK_BYTES;
                let scale = block_scale(weights, offset);
                // SAFETY: `validate_q8` checked that the matrix holds
                // `rows * blocks` whole blocks and that `input` has
                // `columns` values, so every 8 byte weight load and every
                // 8 value input load below is inside its slice, and `avx2` is
                // present.
                let block_sum = unsafe {
                    let quantized = weights.as_ptr().add(offset + SCALE_BYTES);
                    let x = input.as_ptr().add(block * BLOCK);
                    let mut acc = _mm256_setzero_ps();
                    let mut group = 0;
                    while group < BLOCK / 8 {
                        let bytes = _mm_loadl_epi64(quantized.add(8 * group) as *const __m128i);
                        let w = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
                        let v = _mm256_loadu_ps(x.add(8 * group));
                        acc = _mm256_add_ps(acc, _mm256_mul_ps(w, v));
                        group += 1;
                    }
                    // t[j] = acc[j] + acc[j + 4], u[j] = t[j] + t[j + 2],
                    // then u[0] + u[1].
                    let t = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
                    let u = _mm_add_ps(t, _mm_movehl_ps(t, t));
                    _mm_cvtss_f32(_mm_add_ss(u, _mm_shuffle_ps::<0x55>(u, u)))
                };
                sum += block_sum * scale;
            }
            *slot = sum;
        }
    }

    /// Eight signed bytes at `bytes` as two vectors of four `f32`.
    ///
    /// # Safety
    ///
    /// `bytes` must be valid for reading eight bytes.
    #[target_feature(enable = "sse2")]
    unsafe fn widen(bytes: *const u8) -> (__m128, __m128) {
        // SAFETY: the caller guarantees the eight bytes the load reads, and
        // SSE2 is part of the x86-64 baseline.
        unsafe {
            let zero = _mm_setzero_si128();
            let packed = _mm_loadl_epi64(bytes as *const __m128i);
            // Sign extend by interleaving each byte with its sign mask.
            let words = _mm_unpacklo_epi8(packed, _mm_cmplt_epi8(packed, zero));
            let sign = _mm_cmplt_epi16(words, zero);
            let low = _mm_unpacklo_epi16(words, sign);
            let high = _mm_unpackhi_epi16(words, sign);
            (_mm_cvtepi32_ps(low), _mm_cvtepi32_ps(high))
        }
    }

    /// The SSE2 kernel: lanes 0 to 3 in one register, lanes 4 to 7 in another.
    ///
    /// # Safety
    ///
    /// The caller must have validated the lengths with `validate_q8`. SSE2 is
    /// part of the x86-64 baseline.
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn sse2(
        weights: &[u8],
        rows: usize,
        columns: usize,
        input: &[f32],
        out: &mut [f32],
    ) {
        let blocks = columns / BLOCK;
        for (row, slot) in out.iter_mut().enumerate().take(rows) {
            let mut sum = 0.0f32;
            for block in 0..blocks {
                let offset = (row * blocks + block) * Q8_BLOCK_BYTES;
                let scale = block_scale(weights, offset);
                // SAFETY: `validate_q8` checked that the matrix holds
                // `rows * blocks` whole blocks and that `input` has
                // `columns` values, so every 8 byte weight load and every
                // 4 value input load below is inside its slice.
                let block_sum = unsafe {
                    let quantized = weights.as_ptr().add(offset + SCALE_BYTES);
                    let x = input.as_ptr().add(block * BLOCK);
                    let mut low = _mm_setzero_ps();
                    let mut high = _mm_setzero_ps();
                    let mut group = 0;
                    while group < BLOCK / 8 {
                        let (w_low, w_high) = widen(quantized.add(8 * group));
                        let v_low = _mm_loadu_ps(x.add(8 * group));
                        let v_high = _mm_loadu_ps(x.add(8 * group + 4));
                        low = _mm_add_ps(low, _mm_mul_ps(w_low, v_low));
                        high = _mm_add_ps(high, _mm_mul_ps(w_high, v_high));
                        group += 1;
                    }
                    // t[j] = acc[j] + acc[j + 4], u[j] = t[j] + t[j + 2],
                    // then u[0] + u[1].
                    let t = _mm_add_ps(low, high);
                    let u = _mm_add_ps(t, _mm_movehl_ps(t, t));
                    _mm_cvtss_f32(_mm_add_ss(u, _mm_shuffle_ps::<0x55>(u, u)))
                };
                sum += block_sum * scale;
            }
            *slot = sum;
        }
    }
}

/// The NEON kernel.
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{COLUMN_TILE, ROW_TILE};
    use core::arch::aarch64::{
        float32x4_t, uint16x4_t, vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1_u16,
        vld1q_f32, vmovl_u16, vreinterpretq_f32_u32, vshlq_n_u32,
    };

    /// # Safety
    ///
    /// The caller must have validated the lengths. NEON is part of the aarch64
    /// baseline.
    pub(super) unsafe fn matvec(
        weights: &[u8],
        rows: usize,
        columns: usize,
        input: &[f32],
        out: &mut [f32],
    ) {
        // SAFETY: the caller validated every length, and every load stays
        // inside the slices.
        unsafe {
            let zero = vdupq_n_f32(0.0);
            let aligned = columns & !7;
            let mut row_start = 0;
            while row_start + ROW_TILE <= rows {
                let mut accumulator = [[zero; 2]; ROW_TILE];
                let mut column_start = 0;
                while column_start < aligned {
                    let end = (column_start + COLUMN_TILE).min(aligned);
                    let mut column = column_start;
                    while column < end {
                        let input_low = vld1q_f32(input.as_ptr().add(column));
                        let input_high = vld1q_f32(input.as_ptr().add(column + 4));
                        let mut lane = 0;
                        while lane < ROW_TILE {
                            let offset = ((row_start + lane) * columns + column) * 2;
                            let low = widen(&weights[offset..]);
                            let high = widen(&weights[offset + 8..]);
                            accumulator[lane][0] = vfmaq_f32(accumulator[lane][0], low, input_low);
                            accumulator[lane][1] =
                                vfmaq_f32(accumulator[lane][1], high, input_high);
                            lane += 1;
                        }
                        column += 8;
                    }
                    column_start = end;
                }
                for (lane, accumulator) in accumulator.iter().enumerate() {
                    let mut sum = vaddvq_f32(vaddq_f32(accumulator[0], accumulator[1]));
                    for (column, &input_value) in
                        input.iter().enumerate().take(columns).skip(aligned)
                    {
                        let offset = ((row_start + lane) * columns + column) * 2;
                        let bits = u16::from_le_bytes([weights[offset], weights[offset + 1]]);
                        sum += f32::from_bits((bits as u32) << 16) * input_value;
                    }
                    out[row_start + lane] = sum;
                }
                row_start += ROW_TILE;
            }
            while row_start < rows {
                let mut sum = 0.0f32;
                for (column, &input_value) in input.iter().enumerate().take(columns) {
                    let offset = (row_start * columns + column) * 2;
                    let bits = u16::from_le_bytes([weights[offset], weights[offset + 1]]);
                    sum += f32::from_bits((bits as u32) << 16) * input_value;
                }
                out[row_start] = sum;
                row_start += 1;
            }
        }
    }

    /// Widen four `bfloat16` values to `f32`.
    ///
    /// # Safety
    ///
    /// `bytes` must hold at least eight bytes.
    unsafe fn widen(bytes: &[u8]) -> float32x4_t {
        // SAFETY: the caller guarantees the eight bytes the load reads.
        unsafe {
            let packed: uint16x4_t = vld1_u16(bytes.as_ptr() as *const u16);
            let widened = vmovl_u16(packed);
            let shifted = vshlq_n_u32::<16>(widened);
            vreinterpretq_f32_u32(shifted)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        active, detected, matvec_bf16, matvec_bf16_with, matvec_q4, matvec_q4_scalar,
        matvec_q4_with, matvec_q8, matvec_q8_scalar, matvec_q8_with, matvec_scalar, reduce8,
        select, Path, COLUMN_TILE, ROW_TILE,
    };
    use crate::quant::{
        f32_to_half, matrix_bytes, q8_matrix_bytes, BLOCK, BLOCK_BYTES, Q8_BLOCK_BYTES, SCALE_BYTES,
    };

    /// Deterministic weights in a small range, stored as `bfloat16`.
    fn weights(count: usize) -> Vec<u8> {
        let mut state = 0x9e37_79b9u64;
        let mut bytes = Vec::with_capacity(count * 2);
        for _ in 0..count {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let value = ((state >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
            let bits = value.to_bits();
            let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
            bytes.extend_from_slice(&(rounded >> 16).to_le_bytes());
        }
        bytes
    }

    fn input(columns: usize) -> Vec<f32> {
        (0..columns)
            .map(|index| ((index as f32) * 0.031 - 0.7).sin())
            .collect()
    }

    fn error(actual: &[f32], expected: &[f32]) -> f32 {
        actual
            .iter()
            .zip(expected)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn the_vector_path_agrees_with_the_scalar_path() {
        for rows in [1usize, 3, ROW_TILE, ROW_TILE + 1, 37] {
            for columns in [1usize, 7, 8, 31, 576, COLUMN_TILE + 5] {
                let weights = weights(rows * columns);
                let input = input(columns);
                let mut scalar = vec![0.0f32; rows];
                let mut vector = vec![0.0f32; rows];
                matvec_scalar(&weights, rows, columns, &input, &mut scalar);
                matvec_bf16(&weights, rows, columns, &input, &mut vector).expect("shapes agree");
                let tolerance = 1e-4 * columns as f32;
                assert!(
                    error(&vector, &scalar) <= tolerance,
                    "rows {rows} columns {columns} diff {}",
                    error(&vector, &scalar)
                );
                for path in available_paths() {
                    // A request that the processor can serve must run as asked,
                    // or the label of the assertion below would be false.
                    assert_eq!(select(path), path);
                    let mut explicit = vec![0.0f32; rows];
                    matvec_bf16_with(path, &weights, rows, columns, &input, &mut explicit)
                        .expect("shapes agree");
                    assert!(
                        error(&explicit, &scalar) <= tolerance,
                        "path {} rows {rows} columns {columns} diff {}",
                        path.name(),
                        error(&explicit, &scalar)
                    );
                }
            }
        }
    }

    /// Every path of this architecture.
    #[cfg(target_arch = "x86_64")]
    const ALL_PATHS: &[Path] = &[Path::Scalar, Path::Sse2, Path::Avx2];
    #[cfg(target_arch = "aarch64")]
    const ALL_PATHS: &[Path] = &[Path::Scalar, Path::Neon];
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    const ALL_PATHS: &[Path] = &[Path::Scalar];

    /// The paths this processor can run: none above the detected one.
    fn available_paths() -> Vec<Path> {
        ALL_PATHS
            .iter()
            .copied()
            .filter(|path| *path <= detected())
            .collect()
    }

    #[test]
    fn a_selection_never_exceeds_the_limit_or_the_processor() {
        for &limit in ALL_PATHS {
            let chosen = select(limit);
            assert!(chosen <= limit, "{} above the limit", chosen.name());
            assert!(
                chosen <= detected(),
                "{} above the processor",
                chosen.name()
            );
            if limit <= detected() {
                assert_eq!(chosen, limit);
            } else {
                assert_eq!(chosen, detected());
            }
        }
        assert_eq!(select(Path::Scalar), Path::Scalar);
        assert_eq!(select(*ALL_PATHS.last().unwrap()), detected());
    }

    #[test]
    fn without_a_limit_the_products_run_on_the_detected_path() {
        // The limit is process wide, so no test sets it: the tests run in
        // parallel threads and every product of every other test would follow.
        // This one only reads it.
        assert_eq!(active(), detected());
    }

    #[test]
    fn the_paths_are_ordered_and_named() {
        for pair in ALL_PATHS.windows(2) {
            assert!(pair[0] < pair[1]);
        }
        assert_eq!(Path::Scalar.name(), "scalar");
        #[cfg(target_arch = "x86_64")]
        {
            assert!(Path::Scalar < Path::Sse2 && Path::Sse2 < Path::Avx2);
            assert_eq!(Path::Sse2.name(), "sse2");
            assert_eq!(Path::Avx2.name(), "avx2");
            assert!(detected() >= Path::Sse2);
            assert!(available_paths().contains(&Path::Scalar));
            assert!(available_paths().contains(&Path::Sse2));
        }
        #[cfg(target_arch = "aarch64")]
        {
            assert_eq!(Path::Neon.name(), "neon");
            assert_eq!(detected(), Path::Neon);
        }
    }

    #[test]
    fn known_products_are_exact() {
        // Every value is exact in bfloat16 and every product is exact, so the
        // expected sums are a fixed golden value for all three kernels. Eight
        // columns fill one vector step; five rows leave one row for the tail.
        let rows = [
            [1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            [-1.0, 0.0, 1.0, 0.5, -1.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
        ];
        let input = [0.5f32, 1.0, 2.0, -1.0, 0.5, 1.0, 2.0, -1.0];
        let expected = [9.0f32, 2.0, -2.0, 5.0, 2.0];
        let mut weights = Vec::new();
        for row in rows {
            for value in row {
                let bits = value.to_bits();
                weights.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
            }
        }
        let mut scalar = [0.0f32; 5];
        let mut vector = [0.0f32; 5];
        matvec_scalar(&weights, 5, 8, &input, &mut scalar);
        matvec_bf16(&weights, 5, 8, &input, &mut vector).expect("shapes agree");
        assert_eq!(scalar, expected);
        assert_eq!(vector, expected);
    }

    #[test]
    fn a_matrix_of_ones_sums_the_input() {
        let columns = 19usize;
        let rows = 9usize;
        let one = 0x3f80u16.to_le_bytes();
        let mut weights = Vec::with_capacity(rows * columns * 2);
        for _ in 0..rows * columns {
            weights.extend_from_slice(&one);
        }
        let input: Vec<f32> = (0..columns).map(|index| index as f32).collect();
        let expected: f32 = input.iter().sum();
        let mut out = vec![0.0f32; rows];
        matvec_bf16(&weights, rows, columns, &input, &mut out).expect("shapes agree");
        for value in out {
            assert!((value - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn rejects_a_short_output_and_a_short_matrix() {
        let input = [1.0f32, 2.0];
        let mut out = [0.0f32];
        assert!(matvec_bf16(&[0u8; 8], 2, 2, &input, &mut out).is_err());
        assert!(matvec_bf16(&[0u8; 4], 2, 2, &input, &mut out).is_err());
    }

    /// Pack one block from a binary16 scale and thirty-two quantized weights.
    fn q4_block(scale: f32, values: &[i32; BLOCK]) -> Vec<u8> {
        let mut block = vec![0u8; BLOCK_BYTES];
        block[..SCALE_BYTES].copy_from_slice(&f32_to_half(scale).to_le_bytes());
        for (index, &value) in values.iter().enumerate() {
            block[SCALE_BYTES + index / 2] |= ((value & 0x0f) as u8) << (4 * (index % 2));
        }
        block
    }

    #[test]
    fn the_quantized_kernel_matches_the_reference_product() {
        // Five rows of two blocks with an exact scale per block. The quantized
        // weights and the input are dyadic and every product and partial sum is
        // exact in `f32`, so the expected values below are the ones the
        // reference implementation (numpy, float64) gives for the dequantized
        // matrix. A lane or a nibble that is read in the wrong order moves one
        // of them.
        let columns = 2 * BLOCK;
        let rows = 5usize;
        let scales = [0.5f32, 1.5];
        let mut weights = Vec::with_capacity(matrix_bytes(rows, columns).unwrap());
        for row in 0..rows {
            for (block, &scale) in scales.iter().enumerate() {
                let mut values = [0i32; BLOCK];
                for (index, slot) in values.iter_mut().enumerate() {
                    *slot = ((row * 37 + (block * 32 + index) * 11) % 16) as i32 - 8;
                }
                weights.extend_from_slice(&q4_block(scale, &values));
            }
        }
        let mut input = [0.0f32; 2 * BLOCK];
        for (index, slot) in input.iter_mut().enumerate() {
            *slot = [0.25f32, -0.5, 0.75, 1.0, 1.25][index % 5];
        }
        let mut out = [0.0f32; 5];
        matvec_q4(&weights, rows, columns, &input, &mut out).unwrap();
        // The product computed in float64 from the dequantized weights.
        assert_eq!(out, [-24.0, -22.5, -21.0, -19.5, -2.0]);
    }

    #[test]
    fn the_quantized_kernel_agrees_with_the_expanded_product() {
        let rows = 3usize;
        let columns = 5 * BLOCK;
        let values: Vec<f32> = (0..rows * columns)
            .map(|index| ((index as f32) * 0.071 - 0.9).sin())
            .collect();
        let mut weights = vec![0u8; matrix_bytes(rows, columns).unwrap()];
        for (block, chunk) in values.as_chunks::<BLOCK>().0.iter().enumerate() {
            crate::quant::quantize_block(
                chunk,
                &mut weights[block * BLOCK_BYTES..(block + 1) * BLOCK_BYTES],
            )
            .unwrap();
        }
        let input: Vec<f32> = (0..columns)
            .map(|index| ((index as f32) * 0.13).cos())
            .collect();
        let mut out = vec![0.0f32; rows];
        matvec_q4(&weights, rows, columns, &input, &mut out).unwrap();
        let row_bytes = matrix_bytes(1, columns).unwrap();
        for (row, actual) in out.iter().enumerate() {
            // The expected value is the sum over the dequantized row, read
            // back through the format itself.
            let mut decoded = vec![0.0f32; columns];
            crate::quant::dequantize_row(
                &weights[row * row_bytes..(row + 1) * row_bytes],
                &mut decoded,
            )
            .unwrap();
            let expected: f64 = decoded
                .iter()
                .zip(&input)
                .map(|(weight, input)| *weight as f64 * *input as f64)
                .sum();
            assert!(
                (*actual as f64 - expected).abs() < 1e-4,
                "row {row}: {actual} against {expected}"
            );
        }
    }

    /// Pack one eight bit block from a binary16 scale and signed weights.
    fn q8_block(scale: f32, values: &[i32; BLOCK]) -> Vec<u8> {
        let mut block = vec![0u8; Q8_BLOCK_BYTES];
        block[..SCALE_BYTES].copy_from_slice(&f32_to_half(scale).to_le_bytes());
        for (index, &value) in values.iter().enumerate() {
            block[SCALE_BYTES + index] = (value as i8) as u8;
        }
        block
    }

    #[test]
    fn the_eight_bit_kernel_matches_the_reference_product() {
        // The same shape as the four bit golden: five rows, two blocks, one
        // exact scale per block. The expected values are the ones a float64
        // reference gives for the dequantized matrix.
        let columns = 2 * BLOCK;
        let rows = 5usize;
        let mut weights = Vec::with_capacity(q8_matrix_bytes(rows, columns).unwrap());
        for row in 0..rows {
            for (block, &scale) in [1.0f32, 2.0].iter().enumerate() {
                let mut values = [0i32; BLOCK];
                for (index, slot) in values.iter_mut().enumerate() {
                    *slot = ((row * 53 + (block * 32 + index) * 17) % 255) as i32 - 127;
                }
                weights.extend_from_slice(&q8_block(scale, &values));
            }
        }
        let mut input = [0.0f32; 2 * BLOCK];
        for (index, slot) in input.iter_mut().enumerate() {
            *slot = [0.25f32, -0.5, 0.75, 1.0, 1.25][index % 5];
        }
        let mut out = [0.0f32; 5];
        matvec_q8(&weights, rows, columns, &input, &mut out).unwrap();
        assert_eq!(out, [406.75, -1387.5, -121.75, 761.5, -777.75]);
    }

    #[test]
    fn the_eight_bit_kernel_agrees_with_the_expanded_product() {
        let rows = 3usize;
        let columns = 5 * BLOCK;
        let values: Vec<f32> = (0..rows * columns)
            .map(|index| ((index as f32) * 0.067 - 0.4).sin())
            .collect();
        let mut weights = vec![0u8; q8_matrix_bytes(rows, columns).unwrap()];
        for (block, chunk) in values.as_chunks::<BLOCK>().0.iter().enumerate() {
            crate::quant::quantize_block_q8(
                chunk,
                &mut weights[block * Q8_BLOCK_BYTES..(block + 1) * Q8_BLOCK_BYTES],
            )
            .unwrap();
        }
        let input: Vec<f32> = (0..columns)
            .map(|index| ((index as f32) * 0.11).cos())
            .collect();
        let mut out = vec![0.0f32; rows];
        matvec_q8(&weights, rows, columns, &input, &mut out).unwrap();
        let row_bytes = q8_matrix_bytes(1, columns).unwrap();
        for (row, actual) in out.iter().enumerate() {
            let mut decoded = vec![0.0f32; columns];
            crate::quant::dequantize_row_q8(
                &weights[row * row_bytes..(row + 1) * row_bytes],
                &mut decoded,
            )
            .unwrap();
            let expected: f64 = decoded
                .iter()
                .zip(&input)
                .map(|(weight, input)| *weight as f64 * *input as f64)
                .sum();
            assert!(
                (*actual as f64 - expected).abs() < 1e-3,
                "row {row}: {actual} against {expected}"
            );
        }
    }

    /// A deterministic stream of `u64` values.
    struct Stream(u64);

    impl Stream {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 24
        }

        /// A value in `[-1, 1)`, negative about half of the time.
        fn unit(&mut self) -> f32 {
            (self.next() & 0xff_ffff) as f32 / (1u32 << 23) as f32 - 1.0
        }
    }

    /// Scales with both signs, a spread of magnitudes and an exact zero.
    fn scale(stream: &mut Stream) -> f32 {
        match stream.next() % 5 {
            0 => 0.0,
            1 => -stream.unit() * 0.001,
            _ => stream.unit() * 3.0,
        }
    }

    /// The canonical order written out plainly: the weights of a block are
    /// decoded first, then every group of eight is added into its lane, then
    /// the lanes meet in the fixed tree. Only `f32` operations, one multiply
    /// and one add each.
    fn canonical_block(weights: &[f32; BLOCK], input: &[f32]) -> f32 {
        let mut acc = [0.0f32; 8];
        let mut group = 0;
        while group < BLOCK / 8 {
            let mut lane = 0;
            while lane < 8 {
                let product = weights[8 * group + lane] * input[8 * group + lane];
                acc[lane] += product;
                lane += 1;
            }
            group += 1;
        }
        let t0 = acc[0] + acc[4];
        let t1 = acc[1] + acc[5];
        let t2 = acc[2] + acc[6];
        let t3 = acc[3] + acc[7];
        let u0 = t0 + t2;
        let u1 = t1 + t3;
        u0 + u1
    }

    /// The canonical product of a quantized matrix, `decode` turning the bytes
    /// of one block after the scale into its thirty-two weights.
    fn canonical_matvec(
        matrix: &[u8],
        block_bytes: usize,
        decode: fn(&[u8]) -> [f32; BLOCK],
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Vec<f32> {
        let blocks = columns / BLOCK;
        let mut out = Vec::new();
        for row in 0..rows {
            let mut row_sum = 0.0f32;
            for block in 0..blocks {
                let start = (row * blocks + block) * block_bytes;
                let bytes = &matrix[start..start + block_bytes];
                let scale = crate::quant::half_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
                let weights = decode(&bytes[SCALE_BYTES..]);
                let block_sum = canonical_block(&weights, &input[block * BLOCK..]);
                let scaled = block_sum * scale;
                row_sum += scaled;
            }
            out.push(row_sum);
        }
        out
    }

    fn decode_q4(bytes: &[u8]) -> [f32; BLOCK] {
        let mut weights = [0.0f32; BLOCK];
        for (k, weight) in weights.iter_mut().enumerate() {
            let byte = bytes[k / 2];
            let nibble = if k % 2 == 0 { byte & 0x0f } else { byte >> 4 };
            *weight = if nibble >= 8 {
                nibble as f32 - 16.0
            } else {
                nibble as f32
            };
        }
        weights
    }

    fn decode_q8(bytes: &[u8]) -> [f32; BLOCK] {
        let mut weights = [0.0f32; BLOCK];
        for (k, weight) in weights.iter_mut().enumerate() {
            *weight = bytes[k] as i8 as f32;
        }
        weights
    }

    /// A matrix of random blocks: random scales (with zeros and negatives) and
    /// random payload bytes, the payload being `payload` bytes per block.
    fn random_matrix(stream: &mut Stream, blocks: usize, payload: usize) -> Vec<u8> {
        let mut matrix = Vec::new();
        for _ in 0..blocks {
            matrix.extend_from_slice(&f32_to_half(scale(stream)).to_le_bytes());
            for _ in 0..payload {
                matrix.push(stream.next() as u8);
            }
        }
        matrix
    }

    fn random_input(stream: &mut Stream, columns: usize) -> Vec<f32> {
        (0..columns).map(|_| stream.unit() * 4.0).collect()
    }

    #[test]
    fn the_quantized_kernels_follow_the_canonical_order_bit_for_bit() {
        let mut stream = Stream(0x5eed_1234_abcd_ef01);
        for rows in [1usize, 3, ROW_TILE, 37] {
            for blocks in [1usize, 2, 3, 18, 48] {
                let columns = blocks * BLOCK;
                let input = random_input(&mut stream, columns);

                let q4 = random_matrix(&mut stream, rows * blocks, BLOCK / 2);
                let mut actual = vec![0.0f32; rows];
                matvec_q4(&q4, rows, columns, &input, &mut actual).unwrap();
                let expected = canonical_matvec(&q4, BLOCK_BYTES, decode_q4, rows, columns, &input);
                for (row, (a, e)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        e.to_bits(),
                        "q4 rows {rows} blocks {blocks} row {row}: {a} against {e}"
                    );
                }

                let q8 = random_matrix(&mut stream, rows * blocks, BLOCK);
                let mut actual = vec![0.0f32; rows];
                matvec_q8(&q8, rows, columns, &input, &mut actual).unwrap();
                let expected =
                    canonical_matvec(&q8, Q8_BLOCK_BYTES, decode_q8, rows, columns, &input);
                for (row, (a, e)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        e.to_bits(),
                        "q8 rows {rows} blocks {blocks} row {row}: {a} against {e}"
                    );
                }
            }
        }
    }

    /// The input values the vector kernels must not round differently: signed
    /// zeros, subnormals, large magnitudes and ordinary values of both signs.
    fn awkward_input(stream: &mut Stream, columns: usize) -> Vec<f32> {
        (0..columns)
            .map(|index| match stream.next() % 9 {
                0 => -0.0,
                1 => 0.0,
                2 => f32::from_bits(1 + (stream.next() as u32 & 0x3f_ffff)),
                3 => -f32::from_bits(1 + (stream.next() as u32 & 0x3f_ffff)),
                4 => stream.unit() * 1.0e18,
                5 => stream.unit() * 1.0e-30,
                _ => stream.unit() * 4.0 + (index % 3) as f32,
            })
            .collect()
    }

    /// A matrix of random blocks whose scales are zero, negative, subnormal in
    /// binary16 or large, and whose payload hits both ends of the `i8` range.
    fn awkward_q8_matrix(stream: &mut Stream, blocks: usize) -> Vec<u8> {
        let mut matrix = Vec::new();
        for _ in 0..blocks {
            let half = match stream.next() % 6 {
                0 => 0x0000,
                1 => 0x8000,
                2 => 0x0001 + (stream.next() as u16 & 0x03ff),
                3 => 0x7bff,
                _ => f32_to_half(stream.unit() * 2.0),
            };
            matrix.extend_from_slice(&half.to_le_bytes());
            for _ in 0..BLOCK {
                matrix.push(match stream.next() % 8 {
                    0 => 0x80,
                    1 => 0x7f,
                    2 => 0x00,
                    3 => 0xff,
                    _ => stream.next() as u8,
                });
            }
        }
        matrix
    }

    #[test]
    fn every_eight_bit_path_is_bit_exact_with_the_canonical_order() {
        let mut stream = Stream(0x00c0_ffee_f00d_0001);
        for path in available_paths() {
            assert_eq!(select(path), path, "{} was lowered", path.name());
            for rows in [1usize, 3, ROW_TILE, ROW_TILE + 1, 37] {
                for blocks in [1usize, 2, 3, 18, 48] {
                    let columns = blocks * BLOCK;
                    for awkward in [false, true] {
                        let (matrix, input) = if awkward {
                            (
                                awkward_q8_matrix(&mut stream, rows * blocks),
                                awkward_input(&mut stream, columns),
                            )
                        } else {
                            (
                                random_matrix(&mut stream, rows * blocks, BLOCK),
                                random_input(&mut stream, columns),
                            )
                        };
                        let expected = canonical_matvec(
                            &matrix,
                            Q8_BLOCK_BYTES,
                            decode_q8,
                            rows,
                            columns,
                            &input,
                        );
                        let mut scalar = vec![0.0f32; rows];
                        matvec_q8_scalar(&matrix, rows, columns, &input, &mut scalar).unwrap();
                        let mut actual = vec![0.0f32; rows];
                        matvec_q8_with(path, &matrix, rows, columns, &input, &mut actual).unwrap();
                        for row in 0..rows {
                            assert_eq!(
                                scalar[row].to_bits(),
                                expected[row].to_bits(),
                                "scalar rows {rows} blocks {blocks} row {row}"
                            );
                            assert_eq!(
                                actual[row].to_bits(),
                                expected[row].to_bits(),
                                "{} rows {rows} blocks {blocks} awkward {awkward} row {row}: {} against {}",
                                path.name(),
                                actual[row],
                                expected[row]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_eight_bit_paths_keep_the_sign_of_a_zero() {
        // One block whose products are all signed zeros. The row starts from
        // `+0.0`, so every correct path returns `+0.0`; the test pins the bits
        // of the result on every path, and it cannot see a lane that starts from
        // the first product, which the bit for bit comparison with the scalar
        // kernel over random data does catch.
        let mut matrix = vec![0u8; Q8_BLOCK_BYTES];
        matrix[..SCALE_BYTES].copy_from_slice(&f32_to_half(1.0).to_le_bytes());
        for (index, byte) in matrix[SCALE_BYTES..].iter_mut().enumerate() {
            *byte = if index % 2 == 0 { 1 } else { 0 };
        }
        let input = [-0.0f32; BLOCK];
        for path in available_paths() {
            let mut out = [1.0f32; 1];
            matvec_q8_with(path, &matrix, 1, BLOCK, &input, &mut out).unwrap();
            assert_eq!(out[0].to_bits(), 0.0f32.to_bits(), "{}", path.name());
        }
    }

    /// A four bit matrix of random blocks whose scales are zero, negative,
    /// subnormal in binary16 or large, and whose packed bytes often hold the
    /// extreme nibbles.
    fn awkward_q4_matrix(stream: &mut Stream, blocks: usize) -> Vec<u8> {
        let mut matrix = Vec::new();
        for _ in 0..blocks {
            let half = match stream.next() % 6 {
                0 => 0x0000,
                1 => 0x8000,
                2 => 0x0001 + (stream.next() as u16 & 0x03ff),
                3 => 0x7bff,
                _ => f32_to_half(stream.unit() * 2.0),
            };
            matrix.extend_from_slice(&half.to_le_bytes());
            for _ in 0..BLOCK / 2 {
                matrix.push(match stream.next() % 8 {
                    0 => 0x88,
                    1 => 0x77,
                    2 => 0x00,
                    3 => 0xff,
                    _ => stream.next() as u8,
                });
            }
        }
        matrix
    }

    #[test]
    fn every_four_bit_path_is_bit_exact_with_the_canonical_order() {
        let mut stream = Stream(0x00c0_ffee_f00d_0004);
        for path in available_paths() {
            assert_eq!(select(path), path, "{} was lowered", path.name());
            for rows in [1usize, 3, ROW_TILE, ROW_TILE + 1, 37] {
                for blocks in [1usize, 2, 3, 18, 48] {
                    let columns = blocks * BLOCK;
                    for awkward in [false, true] {
                        let (matrix, input) = if awkward {
                            (
                                awkward_q4_matrix(&mut stream, rows * blocks),
                                awkward_input(&mut stream, columns),
                            )
                        } else {
                            (
                                random_matrix(&mut stream, rows * blocks, BLOCK / 2),
                                random_input(&mut stream, columns),
                            )
                        };
                        let expected = canonical_matvec(
                            &matrix,
                            BLOCK_BYTES,
                            decode_q4,
                            rows,
                            columns,
                            &input,
                        );
                        let mut scalar = vec![0.0f32; rows];
                        matvec_q4_scalar(&matrix, rows, columns, &input, &mut scalar).unwrap();
                        let mut actual = vec![0.0f32; rows];
                        matvec_q4_with(path, &matrix, rows, columns, &input, &mut actual).unwrap();
                        for row in 0..rows {
                            assert_eq!(
                                scalar[row].to_bits(),
                                expected[row].to_bits(),
                                "scalar rows {rows} blocks {blocks} row {row}"
                            );
                            assert_eq!(
                                actual[row].to_bits(),
                                expected[row].to_bits(),
                                "{} rows {rows} blocks {blocks} awkward {awkward} row {row}: {} against {}",
                                path.name(),
                                actual[row],
                                expected[row]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn every_four_bit_nibble_is_sign_extended_in_both_positions() {
        // 256 packed bytes over 16 blocks: every pair of nibbles occurs, so
        // each of the 16 values sits in the even and in the odd position.
        let mut matrix = Vec::new();
        for block in 0..16usize {
            matrix.extend_from_slice(&f32_to_half(1.0).to_le_bytes());
            matrix.extend((0..BLOCK / 2).map(|index| (block * BLOCK / 2 + index) as u8));
        }
        let mut stream = Stream(0x0000_0000_00ba_5e11);
        let input = random_input(&mut stream, 16 * BLOCK);
        let mut expected = [0.0f32; 1];
        matvec_q4_scalar(&matrix, 1, 16 * BLOCK, &input, &mut expected).unwrap();
        let canonical = canonical_matvec(&matrix, BLOCK_BYTES, decode_q4, 1, 16 * BLOCK, &input);
        assert_eq!(expected[0].to_bits(), canonical[0].to_bits());
        for path in available_paths() {
            let mut out = [0.0f32; 1];
            matvec_q4_with(path, &matrix, 1, 16 * BLOCK, &input, &mut out).unwrap();
            assert_eq!(out[0].to_bits(), expected[0].to_bits(), "{}", path.name());
        }
    }

    #[test]
    fn the_four_bit_paths_keep_the_sign_of_a_zero() {
        // One block whose products are all signed zeros. The row starts from
        // `+0.0`, so every correct path returns `+0.0`; the test pins the bits
        // of the result on every path, and it cannot see a lane that starts from
        // the first product, which the bit for bit comparison with the scalar
        // kernel over random data does catch.
        let mut matrix = vec![0u8; BLOCK_BYTES];
        matrix[..SCALE_BYTES].copy_from_slice(&f32_to_half(1.0).to_le_bytes());
        for (index, byte) in matrix[SCALE_BYTES..].iter_mut().enumerate() {
            *byte = if index % 2 == 0 { 0x01 } else { 0x10 };
        }
        let input = [-0.0f32; BLOCK];
        for path in available_paths() {
            let mut out = [1.0f32; 1];
            matvec_q4_with(path, &matrix, 1, BLOCK, &input, &mut out).unwrap();
            assert_eq!(out[0].to_bits(), 0.0f32.to_bits(), "{}", path.name());
        }
    }

    #[test]
    fn the_lane_reduction_is_the_fixed_tree() {
        let mut stream = Stream(0x0123_4567_89ab_cdef);
        let mut order_matters = false;
        for _ in 0..256 {
            let mut lane = [0.0f32; 8];
            for value in lane.iter_mut() {
                // Magnitudes over many octaves, so a different order of the
                // additions gives a different rounding.
                *value = stream.unit() * (1u32 << (stream.next() % 24)) as f32;
            }
            let tree = ((lane[0] + lane[4]) + (lane[2] + lane[6]))
                + ((lane[1] + lane[5]) + (lane[3] + lane[7]));
            assert_eq!(reduce8(lane).to_bits(), tree.to_bits());
            let mut sequential = 0.0f32;
            for value in lane {
                sequential += value;
            }
            order_matters |= sequential.to_bits() != tree.to_bits();
        }
        assert!(
            order_matters,
            "the data cannot tell the tree from a running sum"
        );
        assert_eq!(reduce8([1.0; 8]), 8.0);
        assert_eq!(reduce8([0.0; 8]).to_bits(), 0.0f32.to_bits());
        // 2^24 + 1 is not representable, so a running sum loses both ones and
        // gives 2^24. The tree adds the two ones to each other first.
        let lane = [16_777_216.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(reduce8(lane), 16_777_218.0);
    }

    #[test]
    fn the_quantized_kernel_rejects_a_bad_shape() {
        let input = [1.0f32; BLOCK];
        let mut out = [0.0f32; 1];
        // Columns that are not a multiple of the block.
        assert!(matvec_q4(&[0u8; 64], 1, BLOCK - 1, &input[..BLOCK - 1], &mut out).is_err());
        // A matrix shorter than its shape.
        assert!(matvec_q4(&[0u8; BLOCK_BYTES - 1], 1, BLOCK, &input, &mut out).is_err());
        // An input that disagrees with the columns.
        assert!(matvec_q4(&[0u8; BLOCK_BYTES], 1, BLOCK, &input[..1], &mut out).is_err());
    }
    /// The weight format of the small decoder below.
    #[derive(Clone, Copy, Debug)]
    enum Format {
        Bf16,
        Q8,
        Q4,
    }

    /// A deterministic matrix of `rows * columns` values in the given format,
    /// quantized row by row with the quantizer of the crate.
    fn format_matrix(
        format: Format,
        rows: usize,
        columns: usize,
        seed: u64,
        scale: f32,
    ) -> Vec<u8> {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        };
        let mut bytes = Vec::new();
        for _ in 0..rows {
            let row: Vec<f32> = (0..columns).map(|_| next() * scale).collect();
            match format {
                Format::Bf16 => {
                    for value in &row {
                        let bits = value.to_bits();
                        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
                        bytes.extend_from_slice(&((rounded >> 16) as u16).to_le_bytes());
                    }
                }
                Format::Q8 => {
                    for block in row.chunks(BLOCK) {
                        let mut out = [0u8; Q8_BLOCK_BYTES];
                        crate::quant::quantize_block_q8(block, &mut out).expect("a block");
                        bytes.extend_from_slice(&out);
                    }
                }
                Format::Q4 => {
                    for block in row.chunks(BLOCK) {
                        let mut out = [0u8; BLOCK_BYTES];
                        crate::quant::quantize_block(block, &mut out).expect("a block");
                        bytes.extend_from_slice(&out);
                    }
                }
            }
        }
        bytes
    }

    /// The product of the decoder below on an explicit path.
    fn product(
        format: Format,
        path: Path,
        weights: &[u8],
        rows: usize,
        columns: usize,
        input: &[f32],
        out: &mut [f32],
    ) {
        match format {
            Format::Bf16 => matvec_bf16_with(path, weights, rows, columns, input, out),
            Format::Q8 => matvec_q8_with(path, weights, rows, columns, input, out),
            Format::Q4 => matvec_q4_with(path, weights, rows, columns, input, out),
        }
        .expect("the shapes agree");
    }

    /// A small decoder over the three formats: a tied embedding, a few gated
    /// feed forward layers with a residual and a hidden state that carries the
    /// history, so every token depends on the whole prompt. Every product goes
    /// through an explicit path, so no global state is touched.
    struct Decoder {
        format: Format,
        embed: Vec<u8>,
        embed_rows: Vec<Vec<f32>>,
        layers: Vec<[Vec<u8>; 3]>,
    }

    const DECODER_HIDDEN: usize = 64;
    const DECODER_FFN: usize = 96;
    const DECODER_VOCAB: usize = 384;
    const DECODER_LAYERS: usize = 3;

    impl Decoder {
        fn new(format: Format) -> Self {
            let embed = format_matrix(format, DECODER_VOCAB, DECODER_HIDDEN, 11, 0.4);
            // The input rows are the dequantized weights of the same matrix, so
            // the embedding the decoder reads is what the output product reads.
            let row_bytes = match format {
                Format::Bf16 => DECODER_HIDDEN * 2,
                Format::Q8 => DECODER_HIDDEN / BLOCK * Q8_BLOCK_BYTES,
                Format::Q4 => DECODER_HIDDEN / BLOCK * BLOCK_BYTES,
            };
            let embed_rows = embed
                .chunks(row_bytes)
                .map(|bytes| {
                    let mut row = vec![0.0f32; DECODER_HIDDEN];
                    match format {
                        Format::Bf16 => {
                            for (slot, pair) in row.iter_mut().zip(bytes.chunks(2)) {
                                *slot = f32::from_bits(
                                    (u16::from_le_bytes([pair[0], pair[1]]) as u32) << 16,
                                );
                            }
                        }
                        Format::Q8 => {
                            crate::quant::dequantize_row_q8(bytes, &mut row).expect("a row")
                        }
                        Format::Q4 => crate::quant::dequantize_row(bytes, &mut row).expect("a row"),
                    }
                    row
                })
                .collect();
            let layers = (0..DECODER_LAYERS as u64)
                .map(|layer| {
                    [
                        format_matrix(format, DECODER_FFN, DECODER_HIDDEN, 100 + layer, 0.1),
                        format_matrix(format, DECODER_FFN, DECODER_HIDDEN, 200 + layer, 0.1),
                        format_matrix(format, DECODER_HIDDEN, DECODER_FFN, 300 + layer, 0.1),
                    ]
                })
                .collect();
            Self {
                format,
                embed,
                embed_rows,
                layers,
            }
        }

        /// Greedy decoding of `steps` tokens after `prompt` on `path`. Returns
        /// the tokens and the logits every token was sampled from.
        fn decode(&self, path: Path, prompt: &[u32], steps: usize) -> (Vec<u32>, Vec<Vec<f32>>) {
            let ones = vec![1.0f32; DECODER_HIDDEN];
            let mut state = vec![0.0f32; DECODER_HIDDEN];
            let mut tokens = Vec::new();
            let mut all_logits = Vec::new();
            let mut feed: Vec<u32> = prompt.to_vec();
            while tokens.len() < steps {
                let mut logits = vec![0.0f32; DECODER_VOCAB];
                for &token in &feed {
                    let mut x: Vec<f32> = self.embed_rows[token as usize]
                        .iter()
                        .zip(&state)
                        .map(|(embedding, carried)| embedding + 0.5 * carried)
                        .collect();
                    let mut normal = vec![0.0f32; DECODER_HIDDEN];
                    for [gate_weights, up_weights, down_weights] in &self.layers {
                        crate::math::rms_norm(&x, &ones, 1e-5, &mut normal).expect("lengths");
                        let mut gate = vec![0.0f32; DECODER_FFN];
                        let mut up = vec![0.0f32; DECODER_FFN];
                        let mut down = vec![0.0f32; DECODER_HIDDEN];
                        let format = self.format;
                        product(
                            format,
                            path,
                            gate_weights,
                            DECODER_FFN,
                            DECODER_HIDDEN,
                            &normal,
                            &mut gate,
                        );
                        product(
                            format,
                            path,
                            up_weights,
                            DECODER_FFN,
                            DECODER_HIDDEN,
                            &normal,
                            &mut up,
                        );
                        for (gate, up) in gate.iter_mut().zip(&up) {
                            *gate = crate::math::silu(*gate) * up;
                        }
                        product(
                            format,
                            path,
                            down_weights,
                            DECODER_HIDDEN,
                            DECODER_FFN,
                            &gate,
                            &mut down,
                        );
                        for (slot, value) in x.iter_mut().zip(&down) {
                            *slot += value;
                        }
                    }
                    crate::math::rms_norm(&x, &ones, 1e-5, &mut normal).expect("lengths");
                    product(
                        self.format,
                        path,
                        &self.embed,
                        DECODER_VOCAB,
                        DECODER_HIDDEN,
                        &normal,
                        &mut logits,
                    );
                    state.copy_from_slice(&normal);
                }
                let token = crate::math::argmax(&logits).expect("logits") as u32;
                tokens.push(token);
                all_logits.push(logits);
                feed = vec![token];
            }
            (tokens, all_logits)
        }
    }

    /// The margin between the two best logits.
    fn margin(logits: &[f32]) -> f32 {
        let best = crate::math::argmax(logits).expect("logits");
        let second = logits
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != best)
            .map(|(_, value)| *value)
            .fold(f32::NEG_INFINITY, f32::max);
        logits[best] - second
    }

    #[test]
    fn every_path_of_the_host_decodes_the_same_tokens() {
        let prompts: [&[u32]; 4] = [
            &[5],
            &[17, 301, 42],
            &[3, 99, 250, 7, 120, 61, 8, 300, 22],
            &[
                11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            ],
        ];
        let steps = 12;
        for format in [Format::Bf16, Format::Q8, Format::Q4] {
            let decoder = Decoder::new(format);
            for &path in &available_paths() {
                assert_eq!(select(path), path);
                let mut largest = 0.0f32;
                let mut undecided = 0;
                for prompt in prompts {
                    let (expected, expected_logits) = decoder.decode(Path::Scalar, prompt, steps);
                    let (tokens, logits) = decoder.decode(path, prompt, steps);
                    assert_eq!(
                        tokens,
                        expected,
                        "{format:?} on {} changed the tokens of a prompt of {}",
                        path.name(),
                        prompt.len()
                    );
                    for (actual, reference) in logits.iter().zip(&expected_logits) {
                        assert!(
                            actual
                                .iter()
                                .chain(reference)
                                .all(|value| value.is_finite()),
                            "a logit is not finite"
                        );
                        let step = error(actual, reference);
                        largest = largest.max(step);
                        // A tie after the error can flip the first index that
                        // wins the argmax, so the border counts as undecided.
                        if step > 0.0 && margin(reference) <= 2.0 * step {
                            undecided += 1;
                        }
                    }
                }
                println!(
                    "decoder format={format:?} path={} prompts={} steps={steps} max_abs_logit_diff={largest} undecided={undecided}",
                    path.name(),
                    prompts.len()
                );
                assert_eq!(undecided, 0, "a step is too close to call");
                match format {
                    // The quantized kernels are bit for bit with the scalar one.
                    Format::Q8 | Format::Q4 => assert_eq!(largest, 0.0),
                    Format::Bf16 => assert!(largest < 1e-3, "bf16 logits differ by {largest}"),
                }
            }
        }
    }
}
