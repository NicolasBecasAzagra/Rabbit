// SPDX-License-Identifier: GPL-2.0-only
//! The proof that a GGUF `llama` file is read with the right RoPE convention.
//!
//! A checkpoint in the Hugging Face layout and the GGUF file the `llama.cpp`
//! converter writes for it hold the same weights, with the rows of `q_proj` and
//! `k_proj` permuted in the second (see [`crate::synthetic::permute_rows`]).
//! The checkpoint is run with the split half RoPE it was trained with and the
//! GGUF file with adjacent pairs and no un-permutation, which is what
//! `llama.cpp` does with that file. If the permutation and the convention are
//! right, the two decode the same tokens.
//!
//! That is not yet a proof: a model whose position information changes nothing
//! would pass too. So [`prove`] also runs two wrong readings, which must give
//! other tokens:
//!
//! * `split_half_rope`: the permuted GGUF file with the split half RoPE;
//! * `no_permutation`: a GGUF file written without the permutation, with the
//!   adjacent pairs RoPE.
//!
//! A control that is not detected means the model is too insensitive to tell
//! the right reading from a wrong one, and the proof fails.
//!
//! The twin is not bit exact with the GGUF run, and cannot be: the checkpoint
//! is `bfloat16` and goes through the `bfloat16` kernel, the file is `F32` and
//! goes through [`crate::matvec::matvec_f32`], and the attention score of a
//! head is a sum over the elements of the head, which the permutation visits in
//! another order. The logits differ by the rounding of those sums, and
//! [`TOLERANCE`] bounds the difference. The tokens are compared as well, and a
//! step whose margin between the two best logits is not above twice the
//! difference is `undecided`, which fails the proof: equal tokens would say
//! nothing there.
//!
//! The module has no I/O: the caller opens the files, so the same proof runs on
//! a model held in memory on any host and on the files of the guest.

use crate::config::Config;
use crate::error::{Error, Result};
use crate::math::argmax;
use crate::model::{Model, ModelFile};

/// The largest absolute difference of a logit between the twins that is still
/// rounding. The logits of the synthetic model are below one, the rounding of
/// the twins is a few millionths, and the wrong readings move them by tenths.
pub const TOLERANCE: f64 = 1e-5;

/// The steps decoded per prompt.
pub const STEPS: usize = 64;

/// The prefixes of the run each summary line covers.
pub const SUMMARIES: [usize; 2] = [8, STEPS];

/// Lengths of the deterministic prompts of [`prompts`], besides the one of the
/// config.
const PROMPT_LENGTHS: [usize; 6] = [1, 2, 3, 12, 30, 61];

/// The prompts of the proof: the prompt of the config, when it has one, and
/// deterministic sequences of several lengths over the vocabulary.
pub fn prompts(config: &Config) -> Vec<Vec<u32>> {
    let vocab = config.vocab_size as u64;
    let mut prompts = Vec::new();
    if let Some(ids) = config.prompt_ids.as_deref() {
        prompts.push(ids.to_vec());
    }
    let mut state = 0x00c0_ffee_0000_0030u64;
    for length in PROMPT_LENGTHS {
        prompts.push(
            (0..length)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((state >> 24) % vocab) as u32
                })
                .collect(),
        );
    }
    prompts
}

/// One greedy decoding of `STEPS` tokens.
struct Run {
    tokens: Vec<u32>,
    /// The best and the second best logit of every step.
    top: Vec<f32>,
    second: Vec<f32>,
    /// Every logit every token was chosen from.
    logits: Vec<f32>,
}

/// Decode `STEPS` tokens after `prompt`, whatever the token is: the end of
/// sequence token does not stop the run.
fn decode(model: &Model<'_>, prompt: &[u32]) -> Result<Run> {
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
        let best = argmax(logits).unwrap_or(0);
        run.top.push(logits.get(best).copied().unwrap_or(f32::NAN));
        run.second.push(
            logits
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != best)
                .map(|(_, value)| *value)
                .fold(f32::NEG_INFINITY, f32::max),
        );
        run.logits.extend_from_slice(logits);
        let token = session.sample()?;
        run.tokens.push(token);
        if step + 1 < STEPS {
            session.forward(token)?;
        }
    }
    Ok(run)
}

/// The difference of two logits; a NaN or an infinity on one side only is an
/// infinite difference, so it can never pass as small.
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

/// What comparing a run with the reference run found, over some steps.
#[derive(Clone, Copy, Debug, Default)]
struct Tally {
    prompts: usize,
    /// How many prompts had another token somewhere.
    differing: usize,
    max_abs_logit_diff: f64,
    undecided: usize,
    bit_exact: bool,
}

impl Tally {
    fn new() -> Self {
        Self {
            bit_exact: true,
            ..Self::default()
        }
    }

    /// Compare the first `steps` steps of `run` with those of `reference`, up to
    /// and including the first step with another token: after it the two runs
    /// have another input and the logits say nothing.
    fn add(&mut self, steps: usize, reference: &Run, run: &Run) {
        let vocab = reference.logits.len() / STEPS;
        self.prompts += 1;
        let mut differs = false;
        for step in 0..steps {
            let expected = &reference.logits[step * vocab..(step + 1) * vocab];
            let actual = &run.logits[step * vocab..(step + 1) * vocab];
            let largest = expected
                .iter()
                .zip(actual)
                .map(|(left, right)| difference(*left, *right))
                .fold(0.0f64, f64::max);
            let margin = f64::from(reference.top[step]) - f64::from(reference.second[step]);
            self.max_abs_logit_diff = self.max_abs_logit_diff.max(largest);
            if largest > 0.0 {
                self.bit_exact = false;
                // A tie after the error can flip the first index that wins the
                // argmax, so the border counts as undecided.
                if margin <= 2.0 * largest {
                    self.undecided += 1;
                }
            }
            // A degenerate model can give the same NaN on every run and the
            // same token; that is not a pass.
            if expected
                .iter()
                .chain(actual)
                .any(|value| !value.is_finite())
            {
                differs = true;
                self.max_abs_logit_diff = f64::INFINITY;
            }
            if reference.tokens[step] != run.tokens[step] {
                differs = true;
                break;
            }
        }
        if differs {
            self.differing += 1;
        }
    }
}

/// One line of the twin comparison.
#[derive(Clone, Copy, Debug)]
pub struct TwinLine {
    /// The prefix of the run the line covers.
    pub steps: usize,
    /// Prompts compared.
    pub prompts: usize,
    /// `true` when every token of every prompt is the same.
    pub tokens_equal: bool,
    /// The largest difference of any logit.
    pub max_abs_logit_diff: f64,
    /// Steps whose margin does not clear twice the difference.
    pub undecided: usize,
    /// `true` when every logit has the same bits.
    pub bit_exact: bool,
}

impl TwinLine {
    /// `true` when the line proves what it should.
    pub fn holds(&self) -> bool {
        self.tokens_equal
            && self.max_abs_logit_diff <= TOLERANCE
            && self.undecided == 0
            && self.prompts != 0
    }
}

/// One wrong reading of the file and whether the proof told it from the right
/// one.
#[derive(Clone, Copy, Debug)]
pub struct Control {
    /// `split_half_rope` or `no_permutation`.
    pub name: &'static str,
    /// `true` when another token came out for at least one prompt.
    pub detected: bool,
    /// Prompts with another token.
    pub prompts_differing: usize,
    /// The largest difference of a logit, up to the first other token.
    pub max_abs_logit_diff: f64,
}

/// The outcome of [`prove`].
#[derive(Clone, Debug)]
pub struct Report {
    /// The element type of the matrices of the GGUF file.
    pub format: &'static str,
    /// The twin comparison, one line per summary.
    pub twin: Vec<TwinLine>,
    /// The smallest margin between the two best logits of the checkpoint over
    /// every prompt and step.
    pub min_margin: f32,
    /// Prompts run.
    pub prompts: usize,
    /// The wrong readings.
    pub controls: Vec<Control>,
}

impl Report {
    /// `true` when the twins agree and every wrong reading was detected.
    pub fn ok(&self) -> bool {
        self.twin.iter().all(TwinLine::holds)
            && self.min_margin > 0.0
            && self.controls.iter().all(|control| control.detected)
    }

    /// The lines of the guest check, without `GGUF OK`.
    pub fn lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for line in &self.twin {
            lines.push(format!(
                "GGUF twin format={} prompts={} steps={} tokens_equal={} max_abs_logit_diff={} undecided={} bit_exact={}",
                self.format,
                line.prompts,
                line.steps,
                line.tokens_equal,
                line.max_abs_logit_diff,
                line.undecided,
                line.bit_exact
            ));
        }
        lines.push(format!(
            "GGUF margin min_top2={} prompts={} steps={STEPS}",
            self.min_margin, self.prompts
        ));
        for control in &self.controls {
            lines.push(format!(
                "GGUF negative_control {} detected={} prompts_differing={} max_abs_logit_diff={}",
                control.name,
                control.detected,
                control.prompts_differing,
                control.max_abs_logit_diff
            ));
        }
        lines
    }
}

/// The config of a GGUF file, with the RoPE convention of the caller.
fn gguf_config(file: &ModelFile, interleaved: bool) -> Result<Config> {
    let mut config = file
        .gguf_config()
        .cloned()
        .ok_or_else(|| Error::new("gguf twin: the file is not a GGUF file"))?;
    config.rope_interleaved = interleaved;
    Ok(config)
}

/// Run the proof.
///
/// `twin` is the checkpoint and its `config.json` (which must say split half),
/// `gguf` the permuted GGUF file, `unpermuted` the same GGUF file written
/// without the permutation. `format` names the matrices of the GGUF file.
pub fn prove(
    twin: &ModelFile,
    twin_config: &Config,
    gguf: &ModelFile,
    unpermuted: &ModelFile,
    prompts: &[Vec<u32>],
    format: &'static str,
) -> Result<Report> {
    if twin_config.rope_interleaved {
        return Err(Error::new(
            "gguf twin: the checkpoint must use the split half RoPE",
        ));
    }
    let twin_model = Model::new(twin, twin_config)?;
    // The file as it is read: the adjacent pairs the converter permuted for.
    let gguf_model = Model::new(gguf, &gguf_config(gguf, true)?)?;
    // The wrong readings.
    let split_half_config = gguf_config(gguf, false)?;
    let split_half_model = Model::new(gguf, &split_half_config)?;
    let unpermuted_model = Model::new(unpermuted, &gguf_config(unpermuted, true)?)?;

    let vocab = twin_model.dims().vocab;
    if prompts
        .iter()
        .any(|prompt| prompt.iter().any(|&token| token as usize >= vocab))
    {
        return Err(Error::new(
            "gguf twin: a prompt token is outside the vocabulary",
        ));
    }

    let mut twin_tallies = [Tally::new(); SUMMARIES.len()];
    let mut controls = [
        ("split_half_rope", Tally::new()),
        ("no_permutation", Tally::new()),
    ];
    let mut min_margin = f32::INFINITY;
    for prompt in prompts {
        let reference = decode(&twin_model, prompt)?;
        for (top, second) in reference.top.iter().zip(&reference.second) {
            min_margin = min_margin.min(top - second);
        }
        let run = decode(&gguf_model, prompt)?;
        for (tally, &steps) in twin_tallies.iter_mut().zip(&SUMMARIES) {
            tally.add(steps, &reference, &run);
        }
        let wrong = [
            decode(&split_half_model, prompt)?,
            decode(&unpermuted_model, prompt)?,
        ];
        for ((_, tally), run) in controls.iter_mut().zip(&wrong) {
            tally.add(STEPS, &reference, run);
        }
    }

    Ok(Report {
        format,
        twin: twin_tallies
            .iter()
            .zip(&SUMMARIES)
            .map(|(tally, &steps)| TwinLine {
                steps,
                prompts: tally.prompts,
                tokens_equal: tally.differing == 0,
                max_abs_logit_diff: tally.max_abs_logit_diff,
                undecided: tally.undecided,
                bit_exact: tally.bit_exact,
            })
            .collect(),
        min_margin,
        prompts: prompts.len(),
        controls: controls
            .iter()
            .map(|(name, tally)| Control {
                name,
                detected: tally.differing != 0,
                prompts_differing: tally.differing,
                max_abs_logit_diff: tally.max_abs_logit_diff,
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::{
        self, gguf_builder, safetensors_bytes, weights, Permutation, Spec, Storage,
    };

    /// A model small enough for a debug build, with grouped query attention.
    fn grouped() -> Spec {
        Spec {
            hidden: 64,
            layers: 3,
            heads: 8,
            kv_heads: 2,
            ffn: 96,
            vocab: 512,
            eps: 1e-5,
            theta: 10_000.0,
        }
    }

    /// The same with one key value head per query head.
    fn multi_head() -> Spec {
        Spec {
            heads: 4,
            kv_heads: 4,
            ..grouped()
        }
    }

    struct Files {
        twin: ModelFile,
        twin_config: Config,
        gguf: ModelFile,
        unpermuted: ModelFile,
    }

    fn files(spec: &Spec, storage: Storage) -> Files {
        spec.validate().unwrap();
        let tensors = weights(spec);
        Files {
            twin: ModelFile::from_bytes(safetensors_bytes(&tensors)).unwrap(),
            twin_config: synthetic::config(spec, false).unwrap(),
            gguf: ModelFile::gguf_from_bytes(
                gguf_builder(spec, &tensors, Permutation::Converter, storage).to_bytes(),
            )
            .unwrap(),
            unpermuted: ModelFile::gguf_from_bytes(
                gguf_builder(spec, &tensors, Permutation::None, storage).to_bytes(),
            )
            .unwrap(),
        }
    }

    fn prove_files(files: &Files) -> Report {
        let prompts = prompts(&files.twin_config);
        prove(
            &files.twin,
            &files.twin_config,
            &files.gguf,
            &files.unpermuted,
            &prompts,
            "f32",
        )
        .unwrap()
    }

    fn show(name: &str, report: &Report) {
        for line in report.lines() {
            println!("{name}: {line}");
        }
    }

    #[test]
    fn the_twin_decodes_the_same_tokens_and_the_wrong_readings_do_not() {
        for (name, spec) in [("grouped", grouped()), ("multi_head", multi_head())] {
            let files = files(&spec, Storage::F32);
            let report = prove_files(&files);
            show(name, &report);
            assert_eq!(report.prompts, 7);
            for line in &report.twin {
                assert!(line.tokens_equal, "{name}: tokens differ: {line:?}");
                assert!(
                    line.max_abs_logit_diff <= TOLERANCE,
                    "{name}: logits differ by {}",
                    line.max_abs_logit_diff
                );
                assert_eq!(line.undecided, 0, "{name}");
            }
            // The structure of the logits: the smallest margin between the two
            // best logits is far above the rounding of the twin.
            let rounding = report
                .twin
                .iter()
                .map(|l| l.max_abs_logit_diff)
                .fold(0.0, f64::max);
            assert!(
                f64::from(report.min_margin) > 100.0 * rounding,
                "{name}: margin {} against rounding {rounding}",
                report.min_margin
            );
            for control in &report.controls {
                assert!(control.detected, "{name}: {control:?} was not detected");
                // Not a hair of difference: the mistake moves the logits by
                // orders of magnitude more than the rounding of the twin does.
                assert!(
                    control.max_abs_logit_diff > 100.0 * TOLERANCE,
                    "{name}: {control:?}"
                );
            }
            assert!(report.ok(), "{name}");
        }
    }

    #[test]
    fn a_proof_with_a_wrong_twin_fails() {
        // The checkpoint declared interleaved while its rows are not permuted
        // is the very mistake the proof is for.
        let spec = grouped();
        let files = files(&spec, Storage::F32);
        let wrong = synthetic::config(&spec, true).unwrap();
        let prompts = prompts(&wrong);
        assert!(prove(
            &files.twin,
            &wrong,
            &files.gguf,
            &files.unpermuted,
            &prompts,
            "f32"
        )
        .is_err());
    }

    #[test]
    fn the_report_fails_when_a_control_is_not_detected_or_the_twins_differ() {
        let spec = grouped();
        let files = files(&spec, Storage::F32);
        let mut report = prove_files(&files);
        assert!(report.ok());
        report.controls[0].detected = false;
        assert!(!report.ok());
        report.controls[0].detected = true;
        report.twin[1].tokens_equal = false;
        assert!(!report.ok());
        report.twin[1].tokens_equal = true;
        report.twin[1].max_abs_logit_diff = 2.0 * TOLERANCE;
        assert!(!report.ok());
        report.twin[1].max_abs_logit_diff = 0.0;
        report.twin[1].undecided = 1;
        assert!(!report.ok());
        let lines = report.lines();
        assert_eq!(lines.len(), 5);
        assert!(lines[0].starts_with("GGUF twin format=f32 prompts=7 steps=8 tokens_equal=true"));
        assert!(lines[3].starts_with("GGUF negative_control split_half_rope detected=true"));
    }

    #[test]
    fn f16_matrices_give_the_bits_of_the_f32_file_with_the_same_values() {
        let spec = grouped();
        let tensors = weights(&spec);
        let half = ModelFile::gguf_from_bytes(
            gguf_builder(&spec, &tensors, Permutation::Converter, Storage::F16).to_bytes(),
        )
        .unwrap();
        let wide = ModelFile::gguf_from_bytes(
            gguf_builder(&spec, &tensors, Permutation::Converter, Storage::F32OfF16).to_bytes(),
        )
        .unwrap();
        let half_model = Model::new(&half, half.gguf_config().unwrap()).unwrap();
        let wide_model = Model::new(&wide, wide.gguf_config().unwrap()).unwrap();
        assert_eq!(half_model.weight_dtype(), "F16");
        assert_eq!(wide_model.weight_dtype(), "F32");
        let config = synthetic::config(&spec, false).unwrap();
        for prompt in prompts(&config).iter().take(4) {
            let left = decode(&half_model, prompt).unwrap();
            let right = decode(&wide_model, prompt).unwrap();
            assert_eq!(left.tokens, right.tokens);
            assert!(left
                .logits
                .iter()
                .zip(&right.logits)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    /// A model of a few bytes, which has no use but to be read.
    fn tiny() -> Spec {
        Spec {
            hidden: 8,
            layers: 1,
            heads: 2,
            kv_heads: 1,
            ffn: 8,
            vocab: 16,
            eps: 1e-5,
            theta: 10_000.0,
        }
    }

    fn logits_of(model: &Model<'_>, prompt: &[u32]) -> Run {
        decode(model, prompt).unwrap()
    }

    #[test]
    fn a_file_in_the_last_partial_page_reads_like_one_the_mapping_covers() {
        // A model of 3 KiB is smaller than a page: nothing of it is mapped,
        // every row of every matrix goes through the row copy, and the header
        // is read from a copy of the file. The same file padded with zeros past
        // the last tensor lies inside the mapped pages. Both give the bits.
        let spec = tiny();
        let tensors = weights(&spec);
        let bytes = gguf_builder(&spec, &tensors, Permutation::Converter, Storage::F32).to_bytes();
        assert!(bytes.len() < 4096, "{} bytes", bytes.len());
        let mut padded = bytes.clone();
        padded.resize(2 * 4096, 0);
        let tail = ModelFile::gguf_from_bytes(bytes).unwrap();
        let mapped = ModelFile::gguf_from_bytes(padded).unwrap();
        assert_eq!(tail.mapped_len(), 0);
        assert_eq!(mapped.mapped_len(), 2 * 4096);
        let tail_model = Model::new(&tail, tail.gguf_config().unwrap()).unwrap();
        let mapped_model = Model::new(&mapped, mapped.gguf_config().unwrap()).unwrap();
        for prompt in [&[1u32, 2, 3][..], &[5], &[0, 15, 7, 7]] {
            let left = logits_of(&tail_model, prompt);
            let right = logits_of(&mapped_model, prompt);
            assert_eq!(left.tokens, right.tokens);
            assert!(left.logits.iter().all(|value| value.is_finite()));
            assert!(left
                .logits
                .iter()
                .zip(&right.logits)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    #[test]
    fn a_header_that_ends_in_the_last_partial_page_is_read_from_a_copy() {
        // The description pushes the end of the header past the first page of
        // a file of two: the mapped page is cut inside the header, so the first
        // parse is truncated and the whole file is read from a copy.
        let spec = tiny();
        let tensors = weights(&spec);
        let mut builder = gguf_builder(&spec, &tensors, Permutation::Converter, Storage::F32);
        builder.metadata.push((
            "general.description".to_string(),
            synthetic::Meta::Str("x".repeat(4500)),
        ));
        let bytes = builder.to_bytes();
        assert!(bytes.len() > 4096 && bytes.len() < 8192);
        let file = ModelFile::gguf_from_bytes(bytes).unwrap();
        assert_eq!(file.mapped_len(), 4096);
        let model = Model::new(&file, file.gguf_config().unwrap()).unwrap();
        let run = logits_of(&model, &[1, 2, 3]);
        assert!(run.logits.iter().all(|value| value.is_finite()));
    }
}
