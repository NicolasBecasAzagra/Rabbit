// SPDX-License-Identifier: GPL-2.0-only
//! Self-check of the vector kernels, for the guest.
//!
//! The check has two parts. For every path the processor offers, the first
//! runs the eight and four bit products over a set of shapes with
//! deterministic pseudo random data and compares the bits of the result with
//! the canonical scalar kernel. It prints one line per path:
//!
//! ```text
//! MATVEC q8 path=<name> shapes=<n> bit_exact=<true|false>
//! MATVEC q4 path=<name> shapes=<n> bit_exact=<true|false>
//! ```
//!
//! The second decodes whole prompts with the model of the initramfs on every
//! path, for every weight format the initramfs carries (`/model.safetensors`
//! is bfloat16, `/model-q8.safetensors` and `/model-q4.safetensors` are its
//! quantized copies), through the same `Model` and `Session` the runtime uses.
//! The path of the products is set for the process with
//! `matvec::set_path_limit`, which this program may do because it has one
//! thread. Every path is compared with the scalar path:
//!
//! ```text
//! MATVEC model format=<bf16|q8|q4> path=<name> prompts=<n> steps=<n> tokens_equal=<true|false> max_abs_logit_diff=<f> undecided=<n>
//! ```
//!
//! `max_abs_logit_diff` is the largest difference of any logit of any step
//! against the scalar path, 0 when every logit is bit exact. A step is
//! undecided when the margin between the two best logits of the scalar run is
//! below twice the largest difference of that step: equal tokens say nothing
//! there. Both numbers are taken over the steps where the two runs have the
//! same input, that is up to and including the first step whose token differs.
//! A token that differs is printed on a `MATVEC model MISMATCH` line.
//!
//! The last lines are `MATVEC model OK`, only when every token is equal, and
//! `MATVEC OK checked=<paths> skipped=<paths>`, only when both parts passed. The
//! exit status is nonzero otherwise. The binary is packed into an initramfs by
//! `make matvec-check`, so the x86 kernels run on the processor of the guest.

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

/// The decoding of whole prompts on every path, for every weight format.
#[cfg(target_os = "linux")]
mod model_check {
    use std::io::Read;

    use rabbit_infer::config::Config;
    use rabbit_infer::error::{Error, Result};
    use rabbit_infer::mapping::{open_readonly, open_readonly_optional};
    use rabbit_infer::math::argmax;
    use rabbit_infer::matvec::{self, detected, Path};
    use rabbit_infer::model::{Model, ModelFile};
    use rabbit_infer::{paths, reference};

    use super::{Stream, PATHS};

    /// The steps decoded per prompt, and the lengths of the prefix each summary
    /// line covers. The short summary is a prefix of the long one.
    const STEPS: usize = 64;
    const SUMMARIES: [usize; 2] = [8, STEPS];

    /// Lengths of the deterministic synthetic prompts.
    const SYNTHETIC_LENGTHS: [usize; 5] = [1, 3, 12, 30, 61];

    /// The weight formats and the file each one is read from.
    const FORMATS: [(&str, &str); 3] = [
        ("bf16", paths::MODEL_PATH),
        ("q8", "/model-q8.safetensors"),
        ("q4", "/model-q4.safetensors"),
    ];

    /// One greedy decoding: per step the token and the two best logits, and
    /// every logit the token was chosen from.
    struct Run {
        tokens: Vec<u32>,
        top: Vec<f32>,
        second: Vec<f32>,
        logits: Vec<f32>,
    }

    /// The best and the second best of `logits`.
    fn two_best(logits: &[f32]) -> (f32, f32) {
        let best = argmax(logits).unwrap_or(0);
        let second = logits
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != best)
            .map(|(_, value)| *value)
            .fold(f32::NEG_INFINITY, f32::max);
        (logits.get(best).copied().unwrap_or(f32::NAN), second)
    }

    /// Decode `STEPS` tokens after `prompt` on `path`, whatever the token is:
    /// the end of sequence token does not stop the run, so every path decodes
    /// the same number of steps.
    fn decode(model: &Model<'_>, prompt: &[u32], path: Path) -> Result<Run> {
        matvec::set_path_limit(Some(path));
        if matvec::active() != path {
            return Err(Error::new(format!(
                "the limit {} did not take effect, the products run on {}",
                path.name(),
                matvec::active().name()
            )));
        }
        let vocab = model.dims().vocab;
        let mut session = model.session(prompt.len() + STEPS)?;
        for &token in prompt {
            session.forward(token)?;
        }
        let mut run = Run {
            tokens: Vec::with_capacity(STEPS),
            top: Vec::with_capacity(STEPS),
            second: Vec::with_capacity(STEPS),
            logits: Vec::with_capacity(STEPS * vocab),
        };
        for step in 0..STEPS {
            let logits = session.logits();
            let (top, second) = two_best(logits);
            run.top.push(top);
            run.second.push(second);
            run.logits.extend_from_slice(logits);
            let token = session.sample()?;
            run.tokens.push(token);
            if step + 1 < STEPS {
                session.forward(token)?;
            }
        }
        Ok(run)
    }

    /// What one summary line reports.
    #[derive(Default)]
    struct Tally {
        prompts: usize,
        tokens_equal: bool,
        max_abs_logit_diff: f64,
        undecided: usize,
    }

    /// The difference of two logits; a NaN or an infinity on one side only is
    /// an infinite difference, so it can never pass as small.
    fn difference(left: f32, right: f32) -> f64 {
        if left.to_bits() == right.to_bits() {
            return 0.0;
        }
        let difference = (f64::from(left) - f64::from(right)).abs();
        if difference.is_nan() {
            f64::INFINITY
        } else {
            difference
        }
    }

    /// Where a comparison happens, for the lines it prints.
    struct Site<'a> {
        format: &'a str,
        path: &'a str,
        prompt: usize,
        length: usize,
    }

    /// Compare the first `steps` steps of `run` with those of the scalar run of
    /// the same prompt, up to and including the first step with another token.
    /// A different token is printed with where it happened and its margin; the
    /// longest summary covers every step, so only it prints.
    fn add(tally: &mut Tally, site: &Site<'_>, steps: usize, reference: &Run, run: &Run) {
        let vocab = reference.logits.len() / STEPS;
        tally.prompts += 1;
        for step in 0..steps {
            let expected = &reference.logits[step * vocab..(step + 1) * vocab];
            let actual = &run.logits[step * vocab..(step + 1) * vocab];
            let largest = expected
                .iter()
                .zip(actual)
                .map(|(left, right)| difference(*left, *right))
                .fold(0.0f64, f64::max);
            let margin = f64::from(reference.top[step]) - f64::from(reference.second[step]);
            tally.max_abs_logit_diff = tally.max_abs_logit_diff.max(largest);
            // A tie after the error can flip the first index that wins the
            // argmax, so the border counts as undecided.
            if largest > 0.0 && margin <= 2.0 * largest {
                tally.undecided += 1;
            }
            // A degenerate model can give the same NaN on every path and the
            // same token; that is not a pass.
            if expected
                .iter()
                .chain(actual)
                .any(|value| !value.is_finite())
            {
                tally.tokens_equal = false;
                if steps == STEPS {
                    println!(
                        "MATVEC model MISMATCH non_finite format={} path={} prompt={} step={step}",
                        site.format, site.path, site.prompt
                    );
                }
            }
            if reference.tokens[step] != run.tokens[step] {
                tally.tokens_equal = false;
                if steps == STEPS {
                    println!(
                        "MATVEC model MISMATCH format={} path={} prompt={} prompt_len={} step={step} scalar_token={} path_token={} scalar_top={} scalar_second={} path_top={} path_second={} margin={margin} step_logit_diff={largest}",
                        site.format,
                        site.path,
                        site.prompt,
                        site.length,
                        reference.tokens[step],
                        run.tokens[step],
                        reference.top[step],
                        reference.second[step],
                        run.top[step],
                        run.second[step],
                    );
                }
                break;
            }
        }
    }

    /// The prompts: the reference one, the one of the config when it has one,
    /// and deterministic sequences of different lengths over the vocabulary.
    fn prompts(config: &Config, vocab: usize) -> Vec<Vec<u32>> {
        let mut prompts = vec![reference::PROMPT_IDS.to_vec()];
        if let Some(ids) = config.prompt_ids.as_deref() {
            prompts.push(ids.to_vec());
        }
        let mut stream = Stream(0x00c0_ffee_0000_0029);
        for length in SYNTHETIC_LENGTHS {
            prompts.push(
                (0..length)
                    .map(|_| (stream.next() % vocab as u64) as u32)
                    .collect(),
            );
        }
        prompts
    }

    /// Decode every prompt on every path with the model of one format, and
    /// print the summary lines. `true` when the format passed.
    fn check_format(name: &str, model_path: &str, config: &Config) -> Result<bool> {
        let file = ModelFile::open(model_path)?;
        let model = Model::new(&file, config)?;
        let vocab = model.dims().vocab;
        let prompts = prompts(config, vocab);
        if prompts
            .iter()
            .any(|prompt| prompt.iter().any(|&token| token as usize >= vocab))
        {
            return Err(Error::new("a prompt token is outside the vocabulary"));
        }
        let mut tallies: Vec<Vec<Tally>> = PATHS
            .iter()
            .map(|_| {
                SUMMARIES
                    .iter()
                    .map(|_| Tally {
                        tokens_equal: true,
                        ..Tally::default()
                    })
                    .collect()
            })
            .collect();
        for (index, prompt) in prompts.iter().enumerate() {
            let reference = decode(&model, prompt, Path::Scalar)?;
            for (position, &path) in PATHS.iter().enumerate() {
                if path > detected() {
                    continue;
                }
                let run = if path == Path::Scalar {
                    None
                } else {
                    Some(decode(&model, prompt, path)?)
                };
                let site = Site {
                    format: name,
                    path: path.name(),
                    prompt: index,
                    length: prompt.len(),
                };
                for (tally, &steps) in tallies[position].iter_mut().zip(&SUMMARIES) {
                    add(
                        tally,
                        &site,
                        steps,
                        &reference,
                        run.as_ref().unwrap_or(&reference),
                    );
                }
            }
        }
        let mut passed = true;
        for (position, &path) in PATHS.iter().enumerate() {
            if path > detected() {
                println!("MATVEC model format={name} path={} skipped", path.name());
                continue;
            }
            for (tally, &steps) in tallies[position].iter().zip(&SUMMARIES) {
                println!(
                    "MATVEC model format={name} path={} prompts={} steps={steps} tokens_equal={} max_abs_logit_diff={} undecided={}",
                    path.name(),
                    tally.prompts,
                    tally.tokens_equal,
                    tally.max_abs_logit_diff,
                    tally.undecided,
                );
                passed &= tally.tokens_equal;
            }
        }
        Ok(passed)
    }

    /// Run the whole model check. `true` when every format and path passed.
    pub fn run() -> bool {
        let outcome = (|| -> Result<bool> {
            let mut bytes = Vec::new();
            open_readonly(paths::CONFIG_PATH)?.read_to_end(&mut bytes)?;
            let config = Config::parse(&bytes)?;
            let mut all = true;
            for (name, path) in FORMATS {
                if name != "bf16" && open_readonly_optional(path)?.is_none() {
                    println!("MATVEC model format={name} absent");
                    continue;
                }
                println!("MATVEC model format={name} file={path}");
                all &= check_format(name, path, &config)?;
            }
            Ok(all)
        })();
        // The limit is for this check only; the rest of the program and a
        // reader of the code see the products back on the detected path.
        matvec::set_path_limit(None);
        match outcome {
            Ok(true) => {
                println!("MATVEC model OK");
                true
            }
            Ok(false) => {
                println!("MATVEC model FAIL tokens differ");
                false
            }
            Err(error) => {
                println!("MATVEC model ERR {error}");
                false
            }
        }
    }
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
    #[cfg(target_os = "linux")]
    {
        all &= model_check::run();
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
