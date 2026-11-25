// SPDX-License-Identifier: GPL-2.0-only
//! The activations that the CPU backend of `llama.cpp` multiplies with a weight,
//! as a diagnostic of the oracle's arithmetic.
//!
//! Rabbit multiplies a quantized weight with the `f32` activation. `llama.cpp`
//! does not: before `mul_mat` it packs the `f32` activation into the type that
//! its dot kernel of the weight wants (`vec_dot_type` of `type_traits_cpu`,
//! <https://raw.githubusercontent.com/ggml-org/llama.cpp/79c69f4b06805fe53c91596de6c958f86dbbdf41/ggml/src/ggml-cpu/ggml-cpu.c>):
//!
//! | weight | activation |
//! |---|---|
//! | `Q8_0`, `Q5_0` | `Q8_0` |
//! | `Q4_K`, `Q6_K` | `Q8_K` |
//! | `F16`, `F32` | `F32`, unchanged |
//!
//! (`F16` is listed with `vec_dot_type = GGML_TYPE_F32` at that commit, so the
//! activation of a `F16` weight is not rounded to binary16; the attention reads
//! the `F32` query as it is, too.) The functions here give the values of that
//! packed activation as `f32`: quantized with the reference quantizer of the
//! activation type and dequantized back, so that the kernels of Rabbit, which
//! take `f32`, multiply what `llama.cpp` multiplies. The integer dot product
//! itself is not emulated, only its input values.
//!
//! The key value cache of `llama.cpp` is `f16` by default (`type_k` and `type_v`
//! of `llama_context_default_params` are `GGML_TYPE_F16`, in
//! <https://raw.githubusercontent.com/ggml-org/llama.cpp/79c69f4b06805fe53c91596de6c958f86dbbdf41/src/llama-context.cpp>),
//! and `llama_golden.cpp` does not change them: [`round_to_half`] is the
//! rounding of a stored key or value.
//!
//! This was an attempt to measure how much of the difference between Rabbit and
//! the oracle is the oracle's own arithmetic. The conclusion is negative: the
//! logits of a quantized model depend on rounding decisions that cannot be
//! reproduced bit for bit (see [`crate::golden_compare`] for the numbers), so
//! the emulation does not give the oracle's logits and measures nothing exact.
//! It is off by default and is not a claim about the arithmetic of Rabbit.
//!
//! ## Sources
//!
//! The quantizers are those of ggml `d9fe7bf`,
//! <https://raw.githubusercontent.com/ggml-org/ggml/d9fe7bf/src/ggml-quants.c>:
//!
//! * `quantize_row_q8_0_ref`: per block of 32, `d = amax / 127`, `id = 1 / d`
//!   (0 when `d` is 0), the scale stored as binary16 and `qs = roundf(x * id)`
//!   (half away from zero); `dequantize_row_q8_0` gives `qs * d16`.
//! * `quantize_row_q8_K_ref`: per block of 256, `max` is the value of the largest
//!   magnitude with its sign (the first one on a tie), `iscale = -127 / max`,
//!   `qs = min(127, nearest_int(iscale * x))` (`nearest_int` adds `1.5 * 2^23`
//!   and reads the low bits: round half to even), and `d = 1 / iscale` is
//!   stored as an `f32` (`block_q8_K` of
//!   <https://raw.githubusercontent.com/ggml-org/ggml/d9fe7bf/src/ggml-common.h>
//!   is `float d`, `int8_t qs[256]` and `int16_t bsums[16]`); a block of zeros
//!   has `d = 0`. `dequantize_row_q8_K` gives `d * qs`.
//!
//! The x86 build of the oracle (`GGML_NATIVE=OFF` still enables AVX2 by default)
//! packs `Q8_0` with its own AVX2 code (`id = 127 / amax`, round half to even)
//! and `Q8_K` with the reference one; the reference `Q8_0` quantizer is used
//! here and the two can differ by one step of a value that lands on a tie. The
//! AVX2 quantizer and the integer dot product are therefore not emulated.
//!
//! ## What was read, and what pins it
//!
//! On 2026-11-25 these files were fetched from the URLs above and read, and the
//! statements of this module were checked against them:
//!
//! * `ggml-cpu.c` at `llama.cpp` `79c69f4`, `type_traits_cpu`: `vec_dot_type` is
//!   `F32` for `F32` and `F16` (lines 222 and 228), `Q8_0` for `Q5_0` and `Q8_0`
//!   and `Q8_K` for `Q4_K` and `Q6_K`.
//! * `llama-context.cpp` at the same commit, `llama_context_default_params`
//!   (line 4408): `type_k` and `type_v` are `GGML_TYPE_F16` (lines 4433 and 4434).
//! * `ggml-quants.c` at ggml `d9fe7bf`: `quantize_row_q8_0_ref` (`d = amax / 127`,
//!   `id = 1 / d`, `roundf`), `quantize_row_q8_K_ref` (`iscale = -127 / max`,
//!   `nearest_int`, `MIN(127, v)`) and `nearest_int`; `ggml-common.h`:
//!   `block_q8_K`.
//! * `ggml-cpu/arch/x86/quants.c` at `llama.cpp` `79c69f4`, the AVX2
//!   `quantize_row_q8_0` (lines 331 to 346): `d = max / 127`,
//!   `id = 127 / max`, `_mm256_round_ps` to nearest even. The x86 `Q8_K` quantizer
//!   was not read: that it agrees with the reference one is not verified.
//!
//! No test pins any of this: the tests check the code against itself and against
//! the committed goldens, not against these files, so a change upstream is not
//! noticed.

use crate::error::{Error, Result};
use crate::quant::{f32_to_half, half_to_f32};
use crate::safetensors::DType;

/// Values per block of `Q8_0` (`QK8_0`).
const QK8_0: usize = 32;
/// Values per block of `Q8_K` (`QK_K`).
const QK8_K: usize = 256;

/// The type `llama.cpp` packs the activation into for a weight of a type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    /// `Q8_0`: blocks of 32, a binary16 scale and 32 `i8`.
    Q8_0,
    /// `Q8_K`: blocks of 256, an `f32` scale and 256 `i8`.
    Q8K,
}

impl Activation {
    /// The activation type of a weight type, `None` when the activation stays
    /// `f32` (the `F32` and `F16` weights).
    pub fn for_weight(dtype: DType) -> Option<Self> {
        match dtype {
            DType::Q8B32 | DType::Q5_0 | DType::Q4B32 => Some(Self::Q8_0),
            DType::Q4K | DType::Q6K => Some(Self::Q8K),
            _ => None,
        }
    }

    /// Quantize `input` and dequantize it into `out`.
    pub fn apply(self, input: &[f32], out: &mut [f32]) -> Result<()> {
        match self {
            Self::Q8_0 => q8_0_roundtrip(input, out),
            Self::Q8K => q8_k_roundtrip(input, out),
        }
    }
}

fn check_lengths(input: &[f32], out: &[f32], block: usize) -> Result<()> {
    if input.len() != out.len() || !input.len().is_multiple_of(block) {
        return Err(Error::new(format!(
            "llama emulation: {} values into {}, in blocks of {block}",
            input.len(),
            out.len()
        )));
    }
    Ok(())
}

/// `quantize_row_q8_0_ref` followed by `dequantize_row_q8_0`.
pub fn q8_0_roundtrip(input: &[f32], out: &mut [f32]) -> Result<()> {
    check_lengths(input, out, QK8_0)?;
    for (block, target) in input.chunks(QK8_0).zip(out.chunks_mut(QK8_0)) {
        let mut amax = 0.0f32;
        for &value in block {
            amax = amax.max(value.abs());
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d16 = half_to_f32(f32_to_half(d));
        for (&value, slot) in block.iter().zip(target.iter_mut()) {
            let quant = (value * id).round() as i8;
            *slot = f32::from(quant) * d16;
        }
    }
    Ok(())
}

/// `nearest_int` of ggml: round half to even by the magic number `1.5 * 2^23`.
fn nearest_int(value: f32) -> i32 {
    let shifted = value + 12_582_912.0;
    (shifted.to_bits() as i32 & 0x007f_ffff) - 0x0040_0000
}

/// `quantize_row_q8_K_ref` followed by `dequantize_row_q8_K`.
pub fn q8_k_roundtrip(input: &[f32], out: &mut [f32]) -> Result<()> {
    check_lengths(input, out, QK8_K)?;
    for (block, target) in input.chunks(QK8_K).zip(out.chunks_mut(QK8_K)) {
        let mut max = 0.0f32;
        let mut amax = 0.0f32;
        for &value in block {
            let magnitude = value.abs();
            if magnitude > amax {
                amax = magnitude;
                max = value;
            }
        }
        if amax == 0.0 {
            target.fill(0.0);
            continue;
        }
        let iscale = -127.0f32 / max;
        let d = 1.0 / iscale;
        for (&value, slot) in block.iter().zip(target.iter_mut()) {
            let quant = nearest_int(iscale * value).min(127) as i8;
            *slot = d * f32::from(quant);
        }
    }
    Ok(())
}

/// Round every value to binary16 and back: the value of a key or a value that
/// the `f16` cache of `llama.cpp` holds.
pub fn round_to_half(values: &mut [f32]) {
    for value in values {
        *value = half_to_f32(f32_to_half(*value));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_int_rounds_half_to_even() {
        for (value, expected) in [
            (0.5, 0),
            (1.5, 2),
            (2.5, 2),
            (-0.5, 0),
            (-1.5, -2),
            (-2.5, -2),
            (126.4, 126),
            (-126.6, -127),
            (127.0, 127),
        ] {
            assert_eq!(nearest_int(value), expected, "{value}");
        }
    }

    #[test]
    fn a_q8_0_block_is_the_reference_quantizer() {
        // amax 127 -> d = 1, id = 1: the integers survive, the halves go away
        // from zero (roundf).
        let mut input = [0.0f32; 32];
        input[0] = 127.0;
        input[1] = -126.5;
        input[2] = 2.5;
        input[3] = 0.4;
        let mut out = [0.0f32; 32];
        q8_0_roundtrip(&input, &mut out).unwrap();
        assert_eq!(&out[..4], &[127.0, -127.0, 3.0, 0.0]);
        // The scale is held as binary16: amax 1 gives d = 1/127 rounded to f16.
        let mut input = [0.0f32; 32];
        input[5] = 1.0;
        q8_0_roundtrip(&input, &mut out).unwrap();
        let d16 = half_to_f32(f32_to_half(1.0 / 127.0));
        assert_eq!(out[5], 127.0 * d16);
        // A block of zeros stays zeros.
        q8_0_roundtrip(&[0.0; 32], &mut out).unwrap();
        assert!(out.iter().all(|&value| value == 0.0));
    }

    #[test]
    fn a_q8_k_block_scales_by_the_signed_maximum() {
        let mut input = [0.0f32; 256];
        input[0] = -4.0;
        input[1] = 2.0;
        input[2] = 3.999;
        let mut out = [0.0f32; 256];
        q8_k_roundtrip(&input, &mut out).unwrap();
        // max = -4, so iscale = -127 / -4 = 31.75 and the largest magnitude maps
        // to -127; 2 maps to 63.5, which goes to the even 64.
        let d = 1.0 / (-127.0f32 / -4.0);
        assert_eq!(out[0], d * -127.0);
        assert_eq!(out[1], d * 64.0);
        assert_eq!(out[2], d * 127.0);
        assert!(out[3..].iter().all(|&value| value == 0.0));
        q8_k_roundtrip(&[0.0; 256], &mut out).unwrap();
        assert!(out.iter().all(|&value| value == 0.0));
    }

    #[test]
    fn the_roundtrip_error_is_a_fraction_of_a_step() {
        let input: Vec<f32> = (0..512)
            .map(|index| ((index * 37 % 101) as f32 - 50.0) * 0.03)
            .collect();
        let mut q8_0 = vec![0.0f32; 512];
        let mut q8_k = vec![0.0f32; 512];
        q8_0_roundtrip(&input, &mut q8_0).unwrap();
        q8_k_roundtrip(&input, &mut q8_k).unwrap();
        for (index, &value) in input.iter().enumerate() {
            assert!((q8_0[index] - value).abs() <= 1.5 / 127.0 * 1.5, "{index}");
            assert!((q8_k[index] - value).abs() <= 1.5 / 127.0 * 1.5, "{index}");
        }
    }

    #[test]
    fn a_length_that_is_not_a_block_is_an_error() {
        let mut out = [0.0f32; 31];
        assert!(q8_0_roundtrip(&[0.0; 31], &mut out).is_err());
        assert!(q8_k_roundtrip(&[0.0; 32], &mut [0.0; 32]).is_err());
        assert!(q8_0_roundtrip(&[0.0; 32], &mut [0.0; 64]).is_err());
    }

    #[test]
    fn the_activation_of_each_weight_type() {
        assert_eq!(Activation::for_weight(DType::Q8B32), Some(Activation::Q8_0));
        assert_eq!(Activation::for_weight(DType::Q5_0), Some(Activation::Q8_0));
        assert_eq!(Activation::for_weight(DType::Q4K), Some(Activation::Q8K));
        assert_eq!(Activation::for_weight(DType::Q6K), Some(Activation::Q8K));
        assert_eq!(Activation::for_weight(DType::F16), None);
        assert_eq!(Activation::for_weight(DType::F32), None);
    }

    #[test]
    fn the_half_rounding_is_idempotent_and_to_nearest() {
        let mut values = [1.0f32, 1.0 + 1.0 / 4096.0, 0.1, -1.23457, 65504.0, 1e-8];
        round_to_half(&mut values);
        assert_eq!(values[0], 1.0);
        // 1 + 2^-12 is a quarter of the spacing of binary16 at 1: it rounds down.
        assert_eq!(values[1], 1.0);
        assert_eq!(values[2], half_to_f32(f32_to_half(0.1)));
        assert_ne!(values[2], 0.1);
        assert_eq!(values[4], 65504.0);
        let again = values;
        round_to_half(&mut values);
        assert_eq!(values, again);
    }
}
