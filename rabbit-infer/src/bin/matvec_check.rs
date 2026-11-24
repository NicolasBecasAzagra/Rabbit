// SPDX-License-Identifier: GPL-2.0-only
//! Self-check of the vector kernels of the eight bit product, for the guest.
//!
//! For every path the processor offers, the check runs the product over a set
//! of shapes with deterministic pseudo random data and compares the bits of the
//! result with the canonical scalar kernel. It prints one line per path:
//!
//! ```text
//! MATVEC q8 path=<name> shapes=<n> bit_exact=<true|false>
//! MATVEC q4 path=<name> shapes=<n> bit_exact=<true|false>
//! ```
//!
//! and `MATVEC OK checked=<paths> skipped=<paths>` at the end, only when every
//! line is exact. The exit status is
//! nonzero otherwise. The binary is packed into an initramfs by `make
//! matvec-check`, so the x86 kernels run on the processor of the guest.

#![deny(unsafe_op_in_unsafe_fn)]

use rabbit_infer::matvec::{
    detected, matvec_q4_scalar, matvec_q4_with, matvec_q8_scalar, matvec_q8_with, select, Path,
    ROW_TILE,
};
use rabbit_infer::quant::{f32_to_half, BLOCK, BLOCK_BYTES, Q8_BLOCK_BYTES};

/// The paths with a kernel of their own for this format, lowest first.
const PATHS: &[Path] = &[
    Path::Scalar,
    #[cfg(target_arch = "x86_64")]
    Path::Sse2,
    #[cfg(target_arch = "x86_64")]
    Path::Avx2,
];

/// A small linear congruential generator.
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 24
    }

    /// A value in `[-1, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next() & 0xff_ffff) as f32 / (1u32 << 23) as f32 - 1.0
    }
}

/// A matrix of `blocks` random blocks. The awkward form uses zero, negative,
/// subnormal and large scales and both ends of the `i8` range.
fn matrix(stream: &mut Stream, blocks: usize, awkward: bool) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(blocks * Q8_BLOCK_BYTES);
    for _ in 0..blocks {
        let half = match (awkward, stream.next() % 6) {
            (true, 0) => 0x0000,
            (true, 1) => 0x8000,
            (true, 2) => 0x0001 + (stream.next() as u16 & 0x03ff),
            (true, 3) => 0x7bff,
            _ => f32_to_half(stream.unit() * 2.0),
        };
        bytes.extend_from_slice(&half.to_le_bytes());
        for _ in 0..BLOCK {
            bytes.push(match (awkward, stream.next() % 8) {
                (true, 0) => 0x80,
                (true, 1) => 0x7f,
                (true, 2) => 0x00,
                (true, 3) => 0xff,
                _ => stream.next() as u8,
            });
        }
    }
    bytes
}

/// A four bit matrix of `blocks` random blocks. The awkward form uses zero,
/// negative, subnormal and large scales and often the extreme nibbles.
fn matrix_q4(stream: &mut Stream, blocks: usize, awkward: bool) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(blocks * BLOCK_BYTES);
    for _ in 0..blocks {
        let half = match (awkward, stream.next() % 6) {
            (true, 0) => 0x0000,
            (true, 1) => 0x8000,
            (true, 2) => 0x0001 + (stream.next() as u16 & 0x03ff),
            (true, 3) => 0x7bff,
            _ => f32_to_half(stream.unit() * 2.0),
        };
        bytes.extend_from_slice(&half.to_le_bytes());
        for _ in 0..BLOCK / 2 {
            bytes.push(match (awkward, stream.next() % 8) {
                (true, 0) => 0x88,
                (true, 1) => 0x77,
                (true, 2) => 0x00,
                (true, 3) => 0xff,
                _ => stream.next() as u8,
            });
        }
    }
    bytes
}

/// An input of `columns` values. The awkward form has signed zeros,
/// subnormals and large and small magnitudes.
fn input(stream: &mut Stream, columns: usize, awkward: bool) -> Vec<f32> {
    (0..columns)
        .map(|index| match (awkward, stream.next() % 9) {
            (true, 0) => -0.0,
            (true, 1) => 0.0,
            (true, 2) => f32::from_bits(1 + (stream.next() as u32 & 0x3f_ffff)),
            (true, 3) => -f32::from_bits(1 + (stream.next() as u32 & 0x3f_ffff)),
            (true, 4) => stream.unit() * 1.0e18,
            (true, 5) => stream.unit() * 1.0e-30,
            (true, _) => stream.unit() * 4.0 + (index % 3) as f32,
            (false, _) => stream.unit() * 4.0,
        })
        .collect()
}

/// Run every eight bit shape on `path`; the number of shapes and whether all
/// were exact.
fn check_q8(path: Path) -> (usize, bool) {
    let mut stream = Stream(0x00c0_ffee_f00d_0001);
    let mut shapes = 0;
    let mut exact = select(path) == path;
    for rows in [1usize, 3, ROW_TILE, ROW_TILE + 1, 37] {
        for blocks in [1usize, 2, 3, 18, 48] {
            for awkward in [false, true] {
                let columns = blocks * BLOCK;
                let weights = matrix(&mut stream, rows * blocks, awkward);
                let input = input(&mut stream, columns, awkward);
                let mut expected = vec![0.0f32; rows];
                let mut actual = vec![0.0f32; rows];
                shapes += 1;
                let done = matvec_q8_scalar(&weights, rows, columns, &input, &mut expected)
                    .and_then(|()| {
                        matvec_q8_with(path, &weights, rows, columns, &input, &mut actual)
                    });
                if done.is_err()
                    || expected
                        .iter()
                        .zip(&actual)
                        .any(|(left, right)| left.to_bits() != right.to_bits())
                {
                    exact = false;
                }
            }
        }
    }
    // Every one of the 256 byte values as a weight, so the sign extension of
    // each kernel is checked for all of them and not by sampling.
    let sweep: Vec<u8> = (0..8)
        .flat_map(|block| {
            let scale = 0x3c00u16.to_le_bytes(); // 1.0
            let weights = (0..BLOCK).map(move |index| (block * BLOCK + index) as u8);
            scale.into_iter().chain(weights)
        })
        .collect();
    let columns = 8 * BLOCK;
    let input = input(&mut stream, columns, false);
    let mut expected = [0.0f32; 1];
    let mut actual = [0.0f32; 1];
    shapes += 1;
    let done = matvec_q8_scalar(&sweep, 1, columns, &input, &mut expected)
        .and_then(|()| matvec_q8_with(path, &sweep, 1, columns, &input, &mut actual));
    if done.is_err() || expected[0].to_bits() != actual[0].to_bits() {
        exact = false;
    }
    (shapes, exact)
}

/// Run every four bit shape on `path`; the number of shapes and whether all
/// were exact.
fn check_q4(path: Path) -> (usize, bool) {
    let mut stream = Stream(0x00c0_ffee_f00d_0004);
    let mut shapes = 0;
    let mut exact = select(path) == path;
    for rows in [1usize, 3, ROW_TILE, ROW_TILE + 1, 37] {
        for blocks in [1usize, 2, 3, 18, 48] {
            for awkward in [false, true] {
                let columns = blocks * BLOCK;
                let weights = matrix_q4(&mut stream, rows * blocks, awkward);
                let input = input(&mut stream, columns, awkward);
                let mut expected = vec![0.0f32; rows];
                let mut actual = vec![0.0f32; rows];
                shapes += 1;
                let done = matvec_q4_scalar(&weights, rows, columns, &input, &mut expected)
                    .and_then(|()| {
                        matvec_q4_with(path, &weights, rows, columns, &input, &mut actual)
                    });
                if done.is_err()
                    || expected
                        .iter()
                        .zip(&actual)
                        .any(|(left, right)| left.to_bits() != right.to_bits())
                {
                    exact = false;
                }
            }
        }
    }
    // 256 packed bytes over 16 blocks with every byte value once, so each of
    // the 16 nibble values is checked in the even and in the odd position and
    // the sign extension of each is covered exhaustively, not by sampling.
    let sweep: Vec<u8> = (0..16)
        .flat_map(|block| {
            let scale = 0x3c00u16.to_le_bytes(); // 1.0
            let packed = (0..BLOCK / 2).map(move |index| (block * (BLOCK / 2) + index) as u8);
            scale.into_iter().chain(packed)
        })
        .collect();
    let columns = 16 * BLOCK;
    let input = input(&mut stream, columns, false);
    let mut expected = [0.0f32; 1];
    let mut actual = [0.0f32; 1];
    shapes += 1;
    let done = matvec_q4_scalar(&sweep, 1, columns, &input, &mut expected)
        .and_then(|()| matvec_q4_with(path, &sweep, 1, columns, &input, &mut actual));
    if done.is_err() || expected[0].to_bits() != actual[0].to_bits() {
        exact = false;
    }
    (shapes, exact)
}

fn main() {
    let mut all = true;
    let mut checked = Vec::new();
    let mut skipped = Vec::new();
    for &path in PATHS {
        if path > detected() {
            skipped.push(path.name());
            // The guest does not offer this path. The line says so, and the
            // Makefile fails when it requires a path that was skipped.
            println!("MATVEC q8 path={} skipped", path.name());
            println!("MATVEC q4 path={} skipped", path.name());
            continue;
        }
        checked.push(path.name());
        let (shapes, exact) = check_q8(path);
        println!(
            "MATVEC q8 path={} shapes={shapes} bit_exact={exact}",
            path.name()
        );
        all &= exact;
        let (shapes, exact) = check_q4(path);
        println!(
            "MATVEC q4 path={} shapes={shapes} bit_exact={exact}",
            path.name()
        );
        all &= exact;
    }
    if all {
        // The line names what was and was not checked, so an OK that skipped a
        // path cannot be mistaken for one that ran it.
        println!(
            "MATVEC OK checked={} skipped={}",
            checked.join(","),
            if skipped.is_empty() {
                "none".to_string()
            } else {
                skipped.join(",")
            }
        );
    } else {
        println!("MATVEC FAIL");
    }
    use std::io::Write;
    let _ = std::io::stdout().flush();
    rabbit_infer::power_off::power_off();
    std::process::exit(if all { 0 } else { 1 });
}
