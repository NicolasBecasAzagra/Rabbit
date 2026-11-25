// SPDX-License-Identifier: GPL-2.0-only
//! Rabbit's tokens against the golden tokens of `llama.cpp`, on the same GGUF file.
//!
//! `bench/gguf/llama_golden.cpp` loads a GGUF file with `llama.cpp` on the CPU,
//! decodes greedily `steps` tokens after each prompt and prints, per prompt, the
//! chosen tokens and the best and the second best logit of every step. This
//! module reads that document ([`Document::parse`]), makes the same document
//! from Rabbit ([`run`], [`decode`], [`Document::to_json`], whose text has the
//! layout of the harness so that the two outputs can be diffed) and compares them
//! ([`compare`]). It has no I/O and no system call, so it is tested on every host.
//!
//! ## Why tokens and not logits
//!
//! `llama.cpp` quantizes the activations before it multiplies them with a
//! quantized weight (to `Q8_0` for `Q8_0` and `Q5_0` weights and to `Q8_K` for the
//! K types) and keeps its key value cache in `f16`. Rabbit keeps both in `f32`.
//! The two cannot be expected to produce the same logits, only the same tokens,
//! and only where the margin between the best and the second best logit is larger
//! than the difference of the logits.
//!
//! ## What the emulation showed
//!
//! The hypothesis was that this activation quantization explains the logit noise
//! of the real files. [`crate::llama_emulation`] puts it into Rabbit (the
//! reference quantizers of ggml `d9fe7bf` and the `f16` cache, off by default,
//! `rabbit-infer-golden --emulate-llama`) and the same comparison was run on the
//! three real SmolLM2 files, 8 prompts of 32 steps, with the plain and the
//! emulated arithmetic (`max_top_logit_diff`, the logit scale being the mean
//! `|top|` of the golden, about 25.5). These numbers are from one run on one Mac
//! (arm64) against one oracle host (x86, `provenance.json`); the real GGUF files
//! are not committed, so the tree cannot reproduce them (see
//! `bench/gguf/README.md` for who runs what):
//!
//! | file | plain | emulated |
//! |---|---|---|
//! | `F16` | 0.0085 (equal 8 of 8) | 0.0046 (equal 8 of 8) |
//! | `Q8_0` | 0.61 (equal 7, near tie 1) | 0.82 (equal 6, near tie 2) |
//! | `Q4_K_M` | 1.54 (equal 4, near tie 4) | 1.12 (equal 4, near tie 4) |
//!
//! The hypothesis was **not confirmed**: the noise does not collapse, and the
//! tokens do not become equal (nor on the two synthetic files, where the emulated
//! runs have one near tie more than the plain ones on this host). What
//! the emulation does show:
//!
//! * The `f16` cache explains about half of the residual of the `F16` file (mean
//!   `|difference|` 0.0016 to 0.0009).
//! * On one prompt of the synthetic `Q8_0` file the emulated top logit was within
//!   1e-3 of the golden for 30 steps in a row and the plain one was not. That was
//!   seen on this host and is **not** tested: it depends on no rounding decision
//!   differing, which another libm or SIMD width may change.
//! * A single flipped decision changes the logits of the real file by an amount
//!   as large as the whole noise. The two `Q8_0` quantizers of the oracle's x86
//!   build and of the reference differ only by one `ulp` of the scale and by
//!   half-way ties, and the emulated runs with the two differ by 0.14 on average
//!   (the plain and the emulated runs differ from the golden by 0.13); moving the
//!   input of the quantizer by one `ulp` at random does the same (0.13). Real
//!   activations hit exact half-way ties (a one token pass has one position at
//!   which the two quantizers differ, seen by the three products that share it,
//!   and its logits moved by 0.5), and a different last bit before the
//!   quantizer, which another order of float additions gives, may flip a decision
//!   now and then. Where that sensitivity sits in the model was not located.
//!
//! The result is negative: reproducing the activation quantizers of the oracle
//! does not remove the noise, and the logits of a quantized model depend on
//! rounding decisions that no emulation of the values can promise to share bit
//! for bit. A float file has 0.0085 and a quantized one 0.6 to 1.5, so the noise
//! comes with quantized weights and activations, but nothing here shows that it is
//! the oracle's arithmetic and not Rabbit's. It also means that the noise of a file
//! cannot be an absolute number, and that the near tie rule below rests on what
//! the data allow (a noise measured on the file) and not on a proof that the
//! quantization of the activations flips a near tie. It **may**; it was not shown.
//!
//! ## The decision rule
//!
//! For each prompt, the steps are compared in order up to the first step whose
//! tokens differ: until then both runs have fed the same tokens, so the logits are
//! the logits of the same input. After a mismatch the inputs differ and nothing
//! is compared.
//!
//! * The **observed logit noise** `N` of a file is the largest
//!   `|Rabbit top logit - golden top logit|` over the equal steps of every prompt
//!   of the file (the steps before the first mismatch of a prompt, all of them for
//!   a prompt without one).
//! * A prompt **passes as equal** when every step has the same token.
//! * A prompt **passes as a near tie** when, at its first mismatch, (1) the golden
//!   token is Rabbit's second best token, so that the two best candidates were
//!   swapped (`second_id` of the run document), **and** (2) the golden margin
//!   (`top - second` of `llama.cpp`) and Rabbit's own margin are both at most `2N`.
//!   A small margin alone is not enough: a golden token that is not Rabbit's
//!   second best is a different decision, not a rounding tie, and the prompt
//!   fails.
//! * Otherwise the prompt **fails**. A run with any logit that is not finite, in
//!   any position of the vocabulary of any step (`decode` counts them all), fails
//!   its prompt, and that logit does not enter the noise. Such a run cannot be
//!   written as a document ([`Document::to_json`] is an error): `nan` and `inf`
//!   are not JSON and the parser could not read them back.
//! * The **noise cap**: the observed noise is measured on the file that is being
//!   judged, so a wrong computation that moves the logits raises its own excuse.
//!   The noise relative to the logit scale of the golden (`scale`, the mean
//!   `|top|` over all its steps) must therefore not exceed a cap, and a file whose
//!   relative noise is above it **fails by itself**, whatever its prompts say.
//!
//! The cap is chosen by the class of the weights ([`Weights`]), which the run
//! document carries (`weights`): **quantized** when any matrix of the model (the
//! embedding, the output and the seven of each layer) is quantized, **float**
//! otherwise. It is not the type of the embedding, so an `F16` embedding with a
//! `Q8_0` body is quantized. A run document that does not say refuses to be
//! compared unless the class is named explicitly ([`compare_capped`],
//! `--cap float|quantized`), and a name that disagrees with the document is an
//! error. The document also carries the `mode` (`plain` or `emulate-llama`), and
//! the summary line prints both.
//!
//! The caps come from the emulated runs of the three real files, with a safety
//! factor of 2. For the quantized files, the largest relative noise of the
//! emulated runs is 4.2% (`Q4_K_M`, 1.1198 over a scale of 26.55; `Q8_0` is 3.2%),
//! so the cap is [`NOISE_CAP_QUANTIZED`] = 8.5% (twice 4.2% is 8.4%, rounded up);
//! the plain runs of the real files are 2.4% and 5.8%, under it. For the float
//! files it is 2 times the larger relative noise of the two runs of the real `F16`
//! file (3.3e-4, the plain one, the emulation having only the `f16` cache to
//! add), [`NOISE_CAP_FLOAT`] = 7e-4. Against these, the synthetic files read with
//! the right RoPE are at about 1.1% on this host.
//!
//! The cap is loose and it is what the data allow: a model that is that
//! sensitive cannot be held to less. The synthetic `Q4_K_M` file read with the
//! wrong RoPE convention (split half) is at about 10% on this host, which is over
//! the cap only by a factor of 1.2: that factor is **not** what the tests rely on.
//! They assert that the wrong reading's relative noise is more than 5 times that
//! of the right one, that at least two of its first mismatches are not near ties
//! by the yardstick of the right reading (golden margin above twice the right
//! noise, or a golden token that is not Rabbit's second best), and that its
//! noise is above half the cap. The rule is one sided: it never excuses a
//! mismatch by a larger noise than the file shows, and a file in which every step
//! has the same tokens has noise to report and nothing to excuse.
//!
//! No strict comparison (all prompts equal, a small bound on the logits) is made
//! of the emulated runs: they are not equal to the golden on the synthetic files
//! either.
//!
//! ## Other hosts
//!
//! The cap, the factor 2 of the near-tie window and the figures above were chosen
//! and measured on this host (arm64 macOS, NEON). The committed tests run in the CI
//! on x86 Linux, where `libm` (`exp`, `sin`, `cos`) and the AVX2 or NEON kernels can
//! move a rounding decision, and one decision moves the logits of a quantized
//! synthetic model by 0.1 or more. The tests therefore assert invariants (no
//! prompt fails, the file passes the rule and the cap, the tokens are equal up to
//! the first mismatch of every prompt) and never an exact count of equal prompts
//! or near ties, nor a number of steps within a tolerance. If one flips on another
//! host: **re-measure on both hosts** (`cargo test --manifest-path
//! rabbit-infer/Cargo.toml --lib golden_compare -- --nocapture` prints the
//! summary lines) and decide from the two sets of numbers
//! whether the model or the rule needs a change. Do not loosen the cap or the
//! window to hide a mismatch that is an error.
//!
//! The summary line is
//! `GOLDEN file=<name> prompts=<n> steps=<n> equal=<k> near_tie=<m> fail=<f>
//! max_top_logit_diff=<x> smallest_golden_margin=<y> noise_rel=<r>
//! noise_cap=<c> weights=<float|quantized> mode=<plain|emulate-llama|unknown>`, on
//! one line. `x` is the observed logit noise, `y` the smallest golden margin over
//! the compared steps (the equal ones and the mismatching one of every prompt),
//! `r` the relative noise and `c` the cap; a file over its cap also has a
//! `GOLDEN noise_over_cap` line.

use crate::error::{Error, Result};
use crate::json::Scanner;
use crate::math::argmax;
use crate::model::Model;

/// The largest noise of a file with quantized weights, relative to its logit
/// scale, that the rule accepts: twice the largest relative noise of the emulated
/// runs of the real `Q8_0` and `Q4_K_M` files (3.2% and 4.2%: 8.4%), rounded up. See
/// the module documentation.
pub const NOISE_CAP_QUANTIZED: f64 = 0.085;

/// The same for a file whose matrices are `F16` or `F32`: twice the relative
/// noise of the real `F16` file (3.3e-4 plain, 1.8e-4 emulated), rounded up.
pub const NOISE_CAP_FLOAT: f64 = 7e-4;

/// What the matrices of a model are, which decides the cap of the noise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weights {
    /// Every matrix is `F16`, `F32` or `BF16`: [`NOISE_CAP_FLOAT`].
    Float,
    /// At least one matrix is quantized: [`NOISE_CAP_QUANTIZED`].
    Quantized,
}

impl Weights {
    /// The word of the run document and of `--cap`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Float => "float",
            Self::Quantized => "quantized",
        }
    }

    /// The class that a word names.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "float" => Some(Self::Float),
            "quantized" => Some(Self::Quantized),
            _ => None,
        }
    }

    /// The cap of the noise relative to the logit scale for this class.
    pub fn cap(self) -> f64 {
        match self {
            Self::Float => NOISE_CAP_FLOAT,
            Self::Quantized => NOISE_CAP_QUANTIZED,
        }
    }
}

/// The arithmetic a run of Rabbit used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Rabbit's own arithmetic.
    Plain,
    /// The diagnostic of [`crate::llama_emulation`].
    EmulateLlama,
}

impl Mode {
    /// The word of the run document.
    pub fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::EmulateLlama => "emulate-llama",
        }
    }

    /// The mode that a word names.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "plain" => Some(Self::Plain),
            "emulate-llama" => Some(Self::EmulateLlama),
            _ => None,
        }
    }
}

/// One prompt run: the prompt and, per step, the chosen token and the best and
/// the second best logit.
#[derive(Clone, Debug, PartialEq)]
pub struct Trace {
    /// The token ids of the prompt.
    pub prompt_ids: Vec<u32>,
    /// The token of every step.
    pub tokens: Vec<u32>,
    /// The best logit of every step.
    pub top: Vec<f32>,
    /// The second best logit of every step.
    pub second: Vec<f32>,
    /// The id of the second best token of every step (the first one on a tie).
    /// Rabbit's runs have it; the document of `llama_golden` has not, and then
    /// it is empty.
    pub second_id: Vec<u32>,
    /// How many of all the logits of all the steps are not finite. It is counted
    /// by [`decode`] and not written: a run with one cannot be written.
    pub non_finite: usize,
}

impl Trace {
    /// The margin between the two best logits of a step.
    pub fn margin(&self, step: usize) -> f64 {
        f64::from(self.top[step]) - f64::from(self.second[step])
    }
}

/// The document `llama_golden` prints, and Rabbit's equivalent.
#[derive(Clone, Debug, PartialEq)]
pub struct Document {
    /// The number of tokens of the vocabulary.
    pub vocab: usize,
    /// Tokens decoded per prompt.
    pub steps: usize,
    /// The class of the matrices of the model, decided from all of them. Rabbit's
    /// runs have it; the document of `llama_golden` has not.
    pub weights: Option<Weights>,
    /// The arithmetic of the run. Rabbit's runs have it.
    pub mode: Option<Mode>,
    /// One run per prompt.
    pub prompts: Vec<Trace>,
}

/// The token ids of a prompt list: one prompt per line, the ids separated by
/// white space; empty lines and lines that start with `#` are skipped, as in
/// `llama_golden.cpp`.
pub fn parse_prompts(text: &str) -> Result<Vec<Vec<u32>>> {
    let mut prompts = Vec::new();
    for (number, line) in text.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let ids = line
            .split_whitespace()
            .map(|word| {
                word.parse::<u32>().map_err(|_| {
                    Error::new(format!(
                        "golden: line {} has `{word}`, which is not a token id",
                        number + 1
                    ))
                })
            })
            .collect::<Result<Vec<u32>>>()?;
        if !ids.is_empty() {
            prompts.push(ids);
        }
    }
    Ok(prompts)
}

/// Decode `steps` tokens greedily after `prompt`: the end of sequence token does
/// not stop the run and a tie goes to the first index, as in `llama_golden.cpp`.
pub fn decode(model: &Model<'_>, prompt: &[u32], steps: usize) -> Result<Trace> {
    if prompt.is_empty() || steps == 0 {
        return Err(Error::new("golden: a prompt and a step count are needed"));
    }
    let mut session = model.session(prompt.len() + steps)?;
    for &token in prompt {
        session.forward(token)?;
    }
    let mut trace = Trace {
        prompt_ids: prompt.to_vec(),
        tokens: Vec::with_capacity(steps),
        top: Vec::with_capacity(steps),
        second: Vec::with_capacity(steps),
        second_id: Vec::with_capacity(steps),
        non_finite: 0,
    };
    for step in 0..steps {
        let logits = session.logits();
        trace.non_finite += logits.iter().filter(|value| !value.is_finite()).count();
        let best = argmax(logits).ok_or_else(|| Error::new("golden: no logits"))?;
        trace.top.push(logits[best]);
        let (second_id, second) = second_best(logits, best);
        trace.second.push(second);
        trace.second_id.push(second_id);
        let token = session.sample()?;
        trace.tokens.push(token);
        if step + 1 < steps {
            session.forward(token)?;
        }
    }
    Ok(trace)
}

/// The id and the value of the best logit but `best`, the first one on a tie. A
/// `NaN` is never the second best; when no logit is left the value is `-inf`.
fn second_best(logits: &[f32], best: usize) -> (u32, f32) {
    let mut found = (0usize, f32::NEG_INFINITY);
    let mut any = false;
    for (index, &value) in logits.iter().enumerate() {
        if index != best && !value.is_nan() && (!any || value > found.1) {
            found = (index, value);
            any = true;
        }
    }
    (found.0 as u32, found.1)
}

/// Decode every prompt, calling `progress` with the index of a prompt that is
/// done.
pub fn run(
    model: &Model<'_>,
    prompts: &[Vec<u32>],
    steps: usize,
    mut progress: impl FnMut(usize),
) -> Result<Document> {
    let mut traces = Vec::with_capacity(prompts.len());
    for (index, prompt) in prompts.iter().enumerate() {
        traces.push(decode(model, prompt, steps)?);
        progress(index);
    }
    Ok(Document {
        vocab: model.dims().vocab,
        steps,
        weights: Some(if model.has_quantized_matrix() {
            Weights::Quantized
        } else {
            Weights::Float
        }),
        mode: Some(if model.emulate_llama() {
            Mode::EmulateLlama
        } else {
            Mode::Plain
        }),
        prompts: traces,
    })
}

impl Document {
    /// Read a document of `llama_golden`, or the one [`Self::to_json`] writes.
    pub fn parse(text: &str) -> Result<Self> {
        let mut scanner = Scanner::new(text.as_bytes());
        let mut vocab = None;
        let mut steps = None;
        let mut weights = None;
        let mut mode = None;
        let mut prompts = None;
        scanner.skip_whitespace();
        scanner.expect(b'{')?;
        loop {
            scanner.skip_whitespace();
            let key = scanner.parse_string()?;
            scanner.skip_whitespace();
            scanner.expect(b':')?;
            match key.as_str() {
                "vocab" => vocab = Some(scanner.parse_usize()?),
                "steps" => steps = Some(scanner.parse_usize()?),
                "weights" => {
                    scanner.skip_whitespace();
                    let word = scanner.parse_string()?;
                    weights = Some(Weights::from_name(&word).ok_or_else(|| {
                        Error::new(format!(
                            "golden: `weights` is `{word}`, not float or quantized"
                        ))
                    })?);
                }
                "mode" => {
                    scanner.skip_whitespace();
                    let word = scanner.parse_string()?;
                    mode = Some(Mode::from_name(&word).ok_or_else(|| {
                        Error::new(format!(
                            "golden: `mode` is `{word}`, not plain or emulate-llama"
                        ))
                    })?);
                }
                "prompts" => prompts = Some(parse_traces(&mut scanner)?),
                _ => scanner.skip_value()?,
            }
            scanner.skip_whitespace();
            match scanner.bump()? {
                b',' => continue,
                b'}' => break,
                _ => return Err(Error::new("golden: expected ',' or '}'")),
            }
        }
        if !scanner.at_end() {
            return Err(Error::new("golden: trailing bytes after the document"));
        }
        let missing = |key: &str| Error::new(format!("golden: the document has no `{key}`"));
        let document = Self {
            vocab: vocab.ok_or_else(|| missing("vocab"))?,
            steps: steps.ok_or_else(|| missing("steps"))?,
            weights,
            mode,
            prompts: prompts.ok_or_else(|| missing("prompts"))?,
        };
        document.validate()?;
        Ok(document)
    }

    /// Every prompt has `steps` steps, and every token is inside the vocabulary.
    fn validate(&self) -> Result<()> {
        for (index, trace) in self.prompts.iter().enumerate() {
            let lengths = [trace.tokens.len(), trace.top.len(), trace.second.len()];
            if lengths.iter().any(|&length| length != self.steps) {
                return Err(Error::new(format!(
                    "golden: prompt {index} does not have {} tokens and logits",
                    self.steps
                )));
            }
            if !trace.second_id.is_empty() && trace.second_id.len() != self.steps {
                return Err(Error::new(format!(
                    "golden: prompt {index} does not have {} second ids",
                    self.steps
                )));
            }
            if trace
                .tokens
                .iter()
                .chain(&trace.second_id)
                .chain(&trace.prompt_ids)
                .any(|&token| token as usize >= self.vocab)
            {
                return Err(Error::new(format!(
                    "golden: prompt {index} has a token outside the vocabulary of {}",
                    self.vocab
                )));
            }
        }
        Ok(())
    }

    /// The document as text, in the layout of `llama_golden.cpp` (the floats with
    /// `%.9g`) plus the fields of Rabbit's runs that the harness does not have
    /// (`weights`, `mode`, `second_id`, written only when present).
    ///
    /// A logit that is not finite has no JSON text that [`Self::parse`] reads back,
    /// so a document with one, or a run that counted one, is a named error and
    /// not a file that cannot be read.
    pub fn to_json(&self) -> Result<String> {
        for (index, trace) in self.prompts.iter().enumerate() {
            let bad = trace
                .top
                .iter()
                .chain(&trace.second)
                .filter(|value| !value.is_finite())
                .count();
            if bad > 0 || trace.non_finite > 0 {
                return Err(Error::new(format!(
                    "golden: non_finite logits: prompt {index} has {} that are not finite, \
                     so the document is not written",
                    bad.max(trace.non_finite)
                )));
            }
        }
        let mut text = format!(
            "{{\n  \"vocab\": {},\n  \"steps\": {},\n",
            self.vocab, self.steps
        );
        if let Some(weights) = self.weights {
            text.push_str(&format!("  \"weights\": \"{}\",\n", weights.name()));
        }
        if let Some(mode) = self.mode {
            text.push_str(&format!("  \"mode\": \"{}\",\n", mode.name()));
        }
        text.push_str("  \"prompts\": [\n");
        for (index, trace) in self.prompts.iter().enumerate() {
            let ids = |values: &[u32]| join(values.iter().map(u32::to_string));
            let floats = |values: &[f32]| join(values.iter().map(|&value| format_g9(value)));
            text.push_str("    {\n");
            text.push_str(&format!(
                "      \"prompt_ids\": [{}],\n",
                ids(&trace.prompt_ids)
            ));
            text.push_str(&format!("      \"tokens\": [{}],\n", ids(&trace.tokens)));
            text.push_str(&format!("      \"top\": [{}],\n", floats(&trace.top)));
            if trace.second_id.is_empty() {
                text.push_str(&format!("      \"second\": [{}]\n", floats(&trace.second)));
            } else {
                text.push_str(&format!("      \"second\": [{}],\n", floats(&trace.second)));
                text.push_str(&format!(
                    "      \"second_id\": [{}]\n",
                    ids(&trace.second_id)
                ));
            }
            text.push_str(if index + 1 < self.prompts.len() {
                "    },\n"
            } else {
                "    }\n"
            });
        }
        text.push_str("  ]\n}\n");
        Ok(text)
    }
}

/// The items joined by `", "`.
fn join(items: impl Iterator<Item = String>) -> String {
    items.collect::<Vec<_>>().join(", ")
}

/// `printf("%.9g", value)` of the C library, for the value of an `f32`.
fn format_g9(value: f32) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.to_string();
    }
    let value = f64::from(value);
    let scientific = format!("{value:.8e}");
    let Some((mantissa, exponent)) = scientific.split_once('e') else {
        return scientific;
    };
    let Ok(exponent) = exponent.parse::<i32>() else {
        return scientific;
    };
    if !(-4..9).contains(&exponent) {
        let sign = if exponent < 0 { '-' } else { '+' };
        return format!("{}e{sign}{:02}", trim_zeros(mantissa), exponent.abs());
    }
    let decimals = (8 - exponent) as usize;
    trim_zeros(&format!("{value:.decimals$}")).to_string()
}

/// A decimal without its trailing zeros after the point, and without the point
/// when nothing is left of the fraction.
fn trim_zeros(text: &str) -> &str {
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.')
    } else {
        text
    }
}

/// The array of runs under `prompts`.
fn parse_traces(scanner: &mut Scanner<'_>) -> Result<Vec<Trace>> {
    let mut traces = Vec::new();
    scanner.skip_whitespace();
    scanner.expect(b'[')?;
    scanner.skip_whitespace();
    if scanner.peek() == Some(b']') {
        scanner.expect(b']')?;
        return Ok(traces);
    }
    loop {
        traces.push(parse_trace(scanner)?);
        scanner.skip_whitespace();
        match scanner.bump()? {
            b',' => continue,
            b']' => return Ok(traces),
            _ => return Err(Error::new("golden: expected ',' or ']' in the prompts")),
        }
    }
}

/// One object of `prompts`.
fn parse_trace(scanner: &mut Scanner<'_>) -> Result<Trace> {
    let mut prompt_ids = None;
    let mut tokens = None;
    let mut top = None;
    let mut second = None;
    let mut second_id = None;
    scanner.skip_whitespace();
    scanner.expect(b'{')?;
    loop {
        scanner.skip_whitespace();
        let key = scanner.parse_string()?;
        scanner.skip_whitespace();
        scanner.expect(b':')?;
        match key.as_str() {
            "prompt_ids" => prompt_ids = Some(parse_ids(scanner)?),
            "tokens" => tokens = Some(parse_ids(scanner)?),
            "top" => top = Some(parse_floats(scanner)?),
            "second" => second = Some(parse_floats(scanner)?),
            "second_id" => second_id = Some(parse_ids(scanner)?),
            _ => scanner.skip_value()?,
        }
        scanner.skip_whitespace();
        match scanner.bump()? {
            b',' => continue,
            b'}' => break,
            _ => return Err(Error::new("golden: expected ',' or '}' in a prompt")),
        }
    }
    let missing = |key: &str| Error::new(format!("golden: a prompt has no `{key}`"));
    Ok(Trace {
        prompt_ids: prompt_ids.ok_or_else(|| missing("prompt_ids"))?,
        tokens: tokens.ok_or_else(|| missing("tokens"))?,
        top: top.ok_or_else(|| missing("top"))?,
        second: second.ok_or_else(|| missing("second"))?,
        second_id: second_id.unwrap_or_default(),
        non_finite: 0,
    })
}

/// An array of token ids.
fn parse_ids(scanner: &mut Scanner<'_>) -> Result<Vec<u32>> {
    scanner
        .parse_usize_array()?
        .into_iter()
        .map(|id| u32::try_from(id).map_err(|_| Error::new("golden: a token id is too large")))
        .collect()
}

/// An array of numbers, as the `f32` that `%.9g` wrote.
fn parse_floats(scanner: &mut Scanner<'_>) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    scanner.skip_whitespace();
    scanner.expect(b'[')?;
    scanner.skip_whitespace();
    if scanner.peek() == Some(b']') {
        scanner.expect(b']')?;
        return Ok(values);
    }
    loop {
        values.push(scanner.parse_number()? as f32);
        scanner.skip_whitespace();
        match scanner.bump()? {
            b',' => continue,
            b']' => return Ok(values),
            _ => return Err(Error::new("golden: expected ',' or ']' in an array")),
        }
    }
}

/// The first step of a prompt whose token differs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mismatch {
    /// The step, from 0.
    pub step: usize,
    /// The token of `llama.cpp`.
    pub golden_token: u32,
    /// The token of Rabbit.
    pub rabbit_token: u32,
    /// The second best token of Rabbit at that step.
    pub rabbit_second_token: u32,
    /// `top - second` of `llama.cpp` at that step.
    pub golden_margin: f64,
    /// `top - second` of Rabbit at that step.
    pub rabbit_margin: f64,
}

/// What the rule says about a prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Every step has the same token.
    Equal,
    /// The first mismatch is a swap of the two best candidates, and a near tie of
    /// both runs.
    NearTie,
    /// A mismatch the rule does not explain, or a logit that is not finite.
    Fail,
}

impl Verdict {
    /// The word of the report.
    pub fn name(self) -> &'static str {
        match self {
            Self::Equal => "equal",
            Self::NearTie => "near_tie",
            Self::Fail => "fail",
        }
    }
}

/// One prompt of a comparison.
#[derive(Clone, Debug)]
pub struct PromptReport {
    /// The prompt, from 0.
    pub index: usize,
    /// Tokens of the prompt.
    pub prompt_len: usize,
    /// Steps of the run.
    pub steps: usize,
    /// The leading steps with the same token.
    pub equal_steps: usize,
    /// The largest `|Rabbit top - golden top|` over the leading equal steps.
    pub max_top_diff: f64,
    /// The first step with another token, if any.
    pub mismatch: Option<Mismatch>,
    /// `true` when a logit of a compared step is not finite, or when the run
    /// counted one anywhere in its logits.
    pub non_finite: bool,
    /// The smallest golden margin over the compared steps.
    pub smallest_golden_margin: f64,
    /// What the rule says; see [`compare`].
    pub verdict: Verdict,
}

/// The comparison of one file.
#[derive(Clone, Debug)]
pub struct Report {
    /// The name of the file, for the summary line.
    pub name: String,
    /// Tokens decoded per prompt.
    pub steps: usize,
    /// The observed logit noise: the largest `|Rabbit top - golden top|` over the
    /// equal steps of every prompt.
    pub max_top_logit_diff: f64,
    /// The noise the near tie rule used: the observed one.
    pub rule_noise: f64,
    /// The logit scale: the mean `|top|` of the golden over all its steps.
    pub scale: f64,
    /// The cap of the noise relative to the scale that was applied.
    pub noise_cap: f64,
    /// The class of the weights that chose the cap.
    pub weights: Weights,
    /// The arithmetic of the run, when its document says.
    pub mode: Option<Mode>,
    /// The smallest golden margin over the compared steps of every prompt.
    pub smallest_golden_margin: f64,
    /// One entry per prompt.
    pub prompts: Vec<PromptReport>,
}

/// Compare the run of Rabbit with the golden document of the same file, with the
/// cap of the class of weights that the run document says.
///
/// A run document without `weights` is an error that asks for the class
/// explicitly ([`compare_capped`]): there is no default. The two documents must
/// describe the same prompts, steps and vocabulary, and the run must carry
/// `second_id`; otherwise the comparison is an error and not a verdict. See the
/// module documentation for the rule.
pub fn compare(name: &str, golden: &Document, rabbit: &Document) -> Result<Report> {
    let weights = rabbit.weights.ok_or_else(|| {
        Error::new(
            "golden: the run document has no `weights` (float or quantized), so the cap of the \
             noise is not known; pass it explicitly with `--cap float` or `--cap quantized`",
        )
    })?;
    compare_capped(name, golden, rabbit, weights)
}

/// [`compare`] with the class of the weights given, for a run document that
/// does not carry it. A class that disagrees with the document's own is an
/// error, so the cap cannot be loosened by naming another one.
pub fn compare_capped(
    name: &str,
    golden: &Document,
    rabbit: &Document,
    weights: Weights,
) -> Result<Report> {
    if let Some(own) = rabbit.weights {
        if own != weights {
            return Err(Error::new(format!(
                "golden: the run document says its weights are {}, not {}",
                own.name(),
                weights.name()
            )));
        }
    }
    let noise_cap = weights.cap();
    if rabbit
        .prompts
        .iter()
        .any(|trace| trace.second_id.len() != rabbit.steps)
    {
        return Err(Error::new(
            "golden: the run document has no `second_id` for every step, which the near tie rule \
             needs; decode it again with this tool",
        ));
    }
    if golden.vocab != rabbit.vocab
        || golden.steps != rabbit.steps
        || golden.prompts.len() != rabbit.prompts.len()
    {
        return Err(Error::new(format!(
            "golden: the golden has {} prompts of {} steps over a vocabulary of {}, the run {} of \
             {} over {}",
            golden.prompts.len(),
            golden.steps,
            golden.vocab,
            rabbit.prompts.len(),
            rabbit.steps,
            rabbit.vocab
        )));
    }
    let mut prompts = Vec::with_capacity(golden.prompts.len());
    for (index, (expected, actual)) in golden.prompts.iter().zip(&rabbit.prompts).enumerate() {
        if expected.prompt_ids != actual.prompt_ids {
            return Err(Error::new(format!(
                "golden: prompt {index} is not the same prompt in the golden and in the run"
            )));
        }
        prompts.push(compare_prompt(index, expected, actual, golden.steps));
    }
    let observed = prompts
        .iter()
        .filter(|prompt| !prompt.non_finite)
        .fold(0.0f64, |noise, prompt| noise.max(prompt.max_top_diff));
    let noise = observed;
    let (sum, count) = golden
        .prompts
        .iter()
        .flat_map(|prompt| prompt.top.iter())
        .filter(|value| value.is_finite())
        .fold((0.0f64, 0usize), |(sum, count), value| {
            (sum + f64::from(value.abs()), count + 1)
        });
    let scale = if count == 0 { 0.0 } else { sum / count as f64 };
    let smallest = prompts.iter().fold(f64::INFINITY, |smallest, prompt| {
        smallest.min(prompt.smallest_golden_margin)
    });
    for prompt in &mut prompts {
        prompt.verdict = match (prompt.non_finite, prompt.mismatch) {
            (true, _) => Verdict::Fail,
            (false, None) => Verdict::Equal,
            (false, Some(mismatch))
                if mismatch.golden_token == mismatch.rabbit_second_token
                    && mismatch.golden_margin <= 2.0 * noise
                    && mismatch.rabbit_margin <= 2.0 * noise =>
            {
                Verdict::NearTie
            }
            (false, Some(_)) => Verdict::Fail,
        };
    }
    Ok(Report {
        name: name.to_string(),
        steps: golden.steps,
        max_top_logit_diff: observed,
        rule_noise: noise,
        scale,
        noise_cap,
        weights,
        mode: rabbit.mode,
        smallest_golden_margin: smallest,
        prompts,
    })
}

/// The steps of one prompt up to the first mismatch. The verdict is filled in
/// by [`compare`], which knows the noise of the whole file.
fn compare_prompt(index: usize, golden: &Trace, rabbit: &Trace, steps: usize) -> PromptReport {
    let mut report = PromptReport {
        index,
        prompt_len: golden.prompt_ids.len(),
        steps,
        equal_steps: 0,
        max_top_diff: 0.0,
        mismatch: None,
        non_finite: false,
        smallest_golden_margin: f64::INFINITY,
        verdict: Verdict::Equal,
    };
    for step in 0..steps {
        let finite = [
            golden.top[step],
            golden.second[step],
            rabbit.top[step],
            rabbit.second[step],
        ]
        .iter()
        .all(|value| value.is_finite());
        report.non_finite |= !finite || rabbit.non_finite > 0 || golden.non_finite > 0;
        let margin = golden.margin(step);
        if margin < report.smallest_golden_margin {
            report.smallest_golden_margin = margin;
        }
        if golden.tokens[step] != rabbit.tokens[step] {
            report.mismatch = Some(Mismatch {
                step,
                golden_token: golden.tokens[step],
                rabbit_token: rabbit.tokens[step],
                rabbit_second_token: rabbit.second_id[step],
                golden_margin: margin,
                rabbit_margin: rabbit.margin(step),
            });
            return report;
        }
        report.equal_steps += 1;
        if finite {
            let difference = (f64::from(rabbit.top[step]) - f64::from(golden.top[step])).abs();
            report.max_top_diff = report.max_top_diff.max(difference);
        }
    }
    report
}

impl Report {
    /// Prompts whose tokens are all equal.
    pub fn equal(&self) -> usize {
        self.count(Verdict::Equal)
    }

    /// Prompts that pass as a near tie.
    pub fn near_tie(&self) -> usize {
        self.count(Verdict::NearTie)
    }

    /// Prompts that fail.
    pub fn failed(&self) -> usize {
        self.count(Verdict::Fail)
    }

    fn count(&self, verdict: Verdict) -> usize {
        self.prompts
            .iter()
            .filter(|prompt| prompt.verdict == verdict)
            .count()
    }

    /// The observed noise over the logit scale.
    pub fn relative_noise(&self) -> f64 {
        if self.scale == 0.0 {
            0.0
        } else {
            self.max_top_logit_diff / self.scale
        }
    }

    /// `true` when the noise relative to the scale is above the cap, which fails
    /// the file by itself.
    pub fn over_cap(&self) -> bool {
        self.max_top_logit_diff > self.noise_cap * self.scale
    }

    /// `true` when there is a prompt, none fails and the noise is within its cap.
    pub fn passed(&self) -> bool {
        !self.prompts.is_empty() && self.failed() == 0 && !self.over_cap()
    }

    /// The summary line of the file.
    pub fn summary(&self) -> String {
        format!(
            "GOLDEN file={} prompts={} steps={} equal={} near_tie={} fail={} \
             max_top_logit_diff={} smallest_golden_margin={} noise_rel={} noise_cap={} \
             weights={} mode={}",
            self.name,
            self.prompts.len(),
            self.steps,
            self.equal(),
            self.near_tie(),
            self.failed(),
            self.max_top_logit_diff as f32,
            self.smallest_golden_margin as f32,
            self.relative_noise() as f32,
            self.noise_cap as f32,
            self.weights.name(),
            self.mode.map_or("unknown", Mode::name)
        )
    }

    /// One line per prompt, and one more for each that is not fully equal: the
    /// first mismatch with its tokens and its two margins.
    pub fn details(&self) -> Vec<String> {
        let mut lines = Vec::new();
        if self.over_cap() {
            lines.push(format!(
                "GOLDEN noise_over_cap noise={} scale={} noise_rel={} noise_cap={}",
                self.max_top_logit_diff as f32,
                self.scale as f32,
                self.relative_noise() as f32,
                self.noise_cap as f32
            ));
        }
        for prompt in &self.prompts {
            lines.push(format!(
                "GOLDEN prompt={} prompt_len={} equal_steps={}/{} max_top_diff={} verdict={}",
                prompt.index,
                prompt.prompt_len,
                prompt.equal_steps,
                prompt.steps,
                prompt.max_top_diff as f32,
                prompt.verdict.name()
            ));
            if let Some(mismatch) = prompt.mismatch {
                lines.push(format!(
                    "GOLDEN mismatch prompt={} step={} golden_token={} rabbit_token={} \
                     rabbit_second_token={} golden_margin={} rabbit_margin={} two_noise={}",
                    prompt.index,
                    mismatch.step,
                    mismatch.golden_token,
                    mismatch.rabbit_token,
                    mismatch.rabbit_second_token,
                    mismatch.golden_margin as f32,
                    mismatch.rabbit_margin as f32,
                    (2.0 * self.rule_noise) as f32
                ));
            }
            if prompt.non_finite {
                lines.push(format!(
                    "GOLDEN non_finite prompt={}: a logit is not finite, so the prompt fails",
                    prompt.index
                ));
            }
        }
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::model::ModelFile;
    use crate::synthetic::{gguf_builder, weights, Permutation, Spec, Storage};
    use std::sync::OnceLock;

    const SYNTH_PROMPTS: &str = include_str!("../../bench/gguf/synthetic/prompts.txt");
    const SYNTH_Q8_GGUF: &[u8] = include_bytes!("../../bench/gguf/synthetic/synth-Q8_0.gguf");
    const SYNTH_Q4KM_GGUF: &[u8] = include_bytes!("../../bench/gguf/synthetic/synth-Q4_K_M.gguf");
    const SYNTH_Q8_GOLDEN: &str = include_str!("../../bench/gguf/synthetic/golden-Q8_0.json");
    const SYNTH_Q4KM_GOLDEN: &str = include_str!("../../bench/gguf/synthetic/golden-Q4_K_M.json");
    const SMOLLM2_GOLDENS: [&str; 3] = [
        include_str!("../../bench/gguf/smollm2/golden-f16.json"),
        include_str!("../../bench/gguf/smollm2/golden-Q8_0.json"),
        include_str!("../../bench/gguf/smollm2/golden-Q4_K_M.json"),
    ];

    const STEPS: usize = 32;

    /// Rabbit's run of a committed file, with the RoPE convention of the caller
    /// and, when `emulate` is set, the arithmetic of the CPU backend of
    /// `llama.cpp` ([`crate::llama_emulation`]).
    fn rabbit_run(gguf: &[u8], interleaved: bool, emulate: bool) -> Document {
        let file = ModelFile::gguf_from_bytes(gguf.to_vec()).unwrap();
        let mut config: Config = file.gguf_config().unwrap().clone();
        config.rope_interleaved = interleaved;
        let mut model = Model::new(&file, &config).unwrap();
        assert!(!model.emulate_llama(), "the emulation is off by default");
        model.set_emulate_llama(emulate);
        let prompts = parse_prompts(SYNTH_PROMPTS).unwrap();
        run(&model, &prompts, STEPS, |_| {}).unwrap()
    }

    fn q8_run() -> &'static Document {
        static RUN: OnceLock<Document> = OnceLock::new();
        RUN.get_or_init(|| rabbit_run(SYNTH_Q8_GGUF, true, false))
    }

    fn q8_emulated_run() -> &'static Document {
        static RUN: OnceLock<Document> = OnceLock::new();
        RUN.get_or_init(|| rabbit_run(SYNTH_Q8_GGUF, true, true))
    }

    fn q4km_run() -> &'static Document {
        static RUN: OnceLock<Document> = OnceLock::new();
        RUN.get_or_init(|| rabbit_run(SYNTH_Q4KM_GGUF, true, false))
    }

    fn q4km_emulated_run() -> &'static Document {
        static RUN: OnceLock<Document> = OnceLock::new();
        RUN.get_or_init(|| rabbit_run(SYNTH_Q4KM_GGUF, true, true))
    }

    /// A trace of the golden: no `second_id`.
    fn trace(tokens: &[u32], top: &[f32], second: &[f32]) -> Trace {
        Trace {
            prompt_ids: vec![1],
            tokens: tokens.to_vec(),
            top: top.to_vec(),
            second: second.to_vec(),
            second_id: Vec::new(),
            non_finite: 0,
        }
    }

    /// A trace of Rabbit, which has the second best token of every step.
    fn rabbit_trace(tokens: &[u32], top: &[f32], second: &[f32], second_id: &[u32]) -> Trace {
        Trace {
            second_id: second_id.to_vec(),
            ..trace(tokens, top, second)
        }
    }

    fn document(traces: Vec<Trace>) -> Document {
        Document {
            vocab: 16,
            steps: traces[0].tokens.len(),
            weights: None,
            mode: None,
            prompts: traces,
        }
    }

    /// A run document of the quantized class.
    fn quantized(traces: Vec<Trace>) -> Document {
        Document {
            weights: Some(Weights::Quantized),
            mode: Some(Mode::Plain),
            ..document(traces)
        }
    }

    /// What must hold on any host for a file of the synthetic family: no prompt
    /// fails, the file passes the rule and the cap, and the tokens are equal for
    /// every prompt up to its first mismatch. No count of equal prompts or of near
    /// ties is asserted: a rounding decision that libm or the SIMD width of
    /// another host moves can turn an equal prompt into a near tie.
    fn assert_invariants(report: &Report, golden: &Document, rabbit: &Document) {
        assert_eq!(report.prompts.len(), 6);
        assert_eq!(report.failed(), 0, "{:#?}", report.details());
        assert!(!report.over_cap(), "{:#?}", report.details());
        assert!(report.passed(), "{:#?}", report.details());
        for (entry, (expected, actual)) in report
            .prompts
            .iter()
            .zip(golden.prompts.iter().zip(&rabbit.prompts))
        {
            let upto = entry
                .mismatch
                .map_or(golden.steps, |mismatch| mismatch.step);
            assert_eq!(entry.equal_steps, upto);
            assert_eq!(expected.tokens[..upto], actual.tokens[..upto]);
        }
    }

    #[test]
    fn the_prompt_list_skips_comments_and_empty_lines_like_the_harness() {
        let prompts = parse_prompts("# a comment\n\n1 2  3\n   \n4\n").unwrap();
        assert_eq!(prompts, vec![vec![1, 2, 3], vec![4]]);
        assert_eq!(parse_prompts(SYNTH_PROMPTS).unwrap().len(), 6);
        assert!(parse_prompts("1 x\n").is_err());
        assert!(parse_prompts("-1\n").is_err());
    }

    #[test]
    fn g9_is_the_printf_conversion() {
        for (value, text) in [
            (0.0f32, "0"),
            (-0.0, "-0"),
            (1.0, "1"),
            (-2.5, "-2.5"),
            (0.1, "0.100000001"),
            (2.286_561_5, "2.28656149"),
            (123456789.0, "123456792"),
            (1234567890.0, "1.23456794e+09"),
            (0.001, "0.00100000005"),
            (0.0001, "9.99999975e-05"),
            (0.00001, "9.99999975e-06"),
            (1e-10, "1.00000001e-10"),
            (f32::INFINITY, "inf"),
            (f32::NEG_INFINITY, "-inf"),
            (f32::NAN, "nan"),
        ] {
            assert_eq!(format_g9(value), text, "{value:e}");
        }
    }

    #[test]
    fn the_committed_goldens_parse_and_print_back_byte_for_byte() {
        let texts = [SYNTH_Q8_GOLDEN, SYNTH_Q4KM_GOLDEN]
            .into_iter()
            .chain(SMOLLM2_GOLDENS);
        for text in texts {
            let document = Document::parse(text).unwrap();
            assert_eq!(document.steps, STEPS);
            assert_eq!(document.weights, None);
            assert_eq!(document.to_json().unwrap().trim_end(), text.trim_end());
        }
    }

    #[test]
    fn a_run_document_prints_and_reads_back_with_its_fields() {
        let run = Document {
            weights: Some(Weights::Float),
            mode: Some(Mode::EmulateLlama),
            ..document(vec![
                rabbit_trace(&[1, 2], &[5.5, 6.25], &[1.0, 2.0], &[3, 4]),
                rabbit_trace(&[2, 2], &[5.5, 6.25], &[1.0, 2.0], &[0, 15]),
            ])
        };
        let text = run.to_json().unwrap();
        assert!(text.contains("\"weights\": \"float\""), "{text}");
        assert!(text.contains("\"mode\": \"emulate-llama\""), "{text}");
        assert!(text.contains("\"second_id\": [3, 4]"), "{text}");
        assert_eq!(Document::parse(&text).unwrap(), run);
        for bad in [
            r#"{"vocab": 4, "steps": 1, "weights": "half", "prompts": []}"#,
            r#"{"vocab": 4, "steps": 1, "mode": "fast", "prompts": []}"#,
            r#"{"vocab": 4, "steps": 1, "prompts": [{"prompt_ids": [1], "tokens": [1], "top": [1.0], "second": [0.5], "second_id": [1, 2]}]}"#,
            r#"{"vocab": 4, "steps": 1, "prompts": [{"prompt_ids": [1], "tokens": [1], "top": [1.0], "second": [0.5], "second_id": [4]}]}"#,
        ] {
            assert!(Document::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_malformed_document_is_an_error() {
        for text in [
            "",
            "{}",
            r#"{"vocab": 4, "steps": 1}"#,
            r#"{"vocab": 4, "steps": 1, "prompts": [{"prompt_ids": [1], "tokens": [1], "top": [1.0]}]}"#,
            r#"{"vocab": 4, "steps": 2, "prompts": [{"prompt_ids": [1], "tokens": [1], "top": [1.0], "second": [0.5]}]}"#,
            r#"{"vocab": 4, "steps": 1, "prompts": [{"prompt_ids": [1], "tokens": [4], "top": [1.0], "second": [0.5]}]}"#,
            r#"{"vocab": 4, "steps": 1, "prompts": []} x"#,
        ] {
            assert!(Document::parse(text).is_err(), "{text}");
        }
        let document = Document::parse(
            r#"{"vocab": 4, "extra": [1, {"a": null}], "steps": 1, "prompts": [{"prompt_ids": [1], "tokens": [2], "top": [1.5], "second": [0.5]}]}"#,
        )
        .unwrap();
        assert_eq!(document.prompts[0].tokens, [2]);
    }

    #[test]
    fn the_second_best_has_an_id_and_a_nan_is_never_it() {
        assert_eq!(second_best(&[1.0, 5.0, 3.0, 3.0], 1), (2, 3.0));
        assert_eq!(second_best(&[5.0, f32::NAN, 2.0], 0), (2, 2.0));
        let (id, value) = second_best(&[5.0], 0);
        assert_eq!((id, value), (0, f32::NEG_INFINITY));
    }

    #[test]
    fn the_rule_on_made_up_runs() {
        // Equal tokens with logits 0.01 apart: noise 0.01.
        let golden = document(vec![
            trace(&[1, 2, 3], &[5.0, 6.0, 7.0], &[1.0, 1.0, 1.0]),
            // Mismatch at step 1 with margins 0.015 and 0.02 (<= 0.02): near tie.
            trace(&[1, 2, 3], &[5.0, 6.0, 7.0], &[1.0, 5.985, 1.0]),
            // Mismatch at step 1 with a golden margin of 2: fails.
            trace(&[1, 2, 3], &[5.0, 6.0, 7.0], &[1.0, 4.0, 1.0]),
            // The golden margin is a near tie, Rabbit's is not: fails.
            trace(&[1, 2, 3], &[5.0, 6.0, 7.0], &[1.0, 5.99, 1.0]),
            // Both margins are tiny but the golden token is not Rabbit's second
            // best (it is 8, the golden has 2): not a swap, fails.
            trace(&[1, 2, 3], &[5.0, 6.0, 7.0], &[1.0, 5.985, 1.0]),
        ]);
        let rabbit = quantized(vec![
            rabbit_trace(&[1, 2, 3], &[5.01, 6.0, 7.0], &[1.0, 1.0, 1.0], &[0, 0, 0]),
            rabbit_trace(&[1, 9, 3], &[5.0, 6.0, 7.0], &[1.0, 5.98, 1.0], &[0, 2, 0]),
            rabbit_trace(&[1, 9, 3], &[5.0, 6.0, 7.0], &[1.0, 5.99, 1.0], &[0, 2, 0]),
            rabbit_trace(&[1, 9, 3], &[5.0, 6.0, 7.0], &[1.0, 3.0, 1.0], &[0, 2, 0]),
            rabbit_trace(&[1, 9, 3], &[5.0, 6.0, 7.0], &[1.0, 5.98, 1.0], &[0, 8, 0]),
        ]);
        let report = compare("made-up", &golden, &rabbit).unwrap();
        assert!((report.max_top_logit_diff - 0.01).abs() < 1e-6);
        let verdicts: Vec<Verdict> = report.prompts.iter().map(|p| p.verdict).collect();
        assert_eq!(
            verdicts,
            [
                Verdict::Equal,
                Verdict::NearTie,
                Verdict::Fail,
                Verdict::Fail,
                Verdict::Fail
            ]
        );
        assert_eq!(report.prompts[1].equal_steps, 1);
        assert_eq!(report.prompts[1].mismatch.unwrap().step, 1);
        assert_eq!(report.prompts[1].mismatch.unwrap().rabbit_second_token, 2);
        assert!(!report.passed());
        assert_eq!(
            (report.equal(), report.near_tie(), report.failed()),
            (1, 1, 3)
        );
        assert!(report.summary().starts_with(
            "GOLDEN file=made-up prompts=5 steps=3 equal=1 near_tie=1 fail=3 max_top_logit_diff=0.0"
        ));
        assert!(report.summary().ends_with(" weights=quantized mode=plain"));
        assert_eq!(report.details().len(), 5 + 4);
    }

    #[test]
    fn a_file_with_no_noise_excuses_nothing() {
        // The logits are the same bits, so the noise is 0 and a mismatch with a
        // margin above 0 is a failure, however small.
        let golden = document(vec![trace(&[1, 2], &[5.0, 6.0], &[1.0, 5.999])]);
        let rabbit = quantized(vec![rabbit_trace(
            &[1, 3],
            &[5.0, 6.0],
            &[1.0, 5.999],
            &[0, 2],
        )]);
        let report = compare("same", &golden, &rabbit).unwrap();
        assert_eq!(report.max_top_logit_diff, 0.0);
        assert_eq!(report.failed(), 1);
    }

    #[test]
    fn a_logit_that_is_not_finite_fails_and_is_not_noise() {
        let golden = document(vec![trace(&[1, 2], &[5.0, 6.0], &[1.0, 1.0])]);
        for bad in [f32::NAN, f32::INFINITY] {
            let rabbit = quantized(vec![rabbit_trace(
                &[1, 2],
                &[5.0, bad],
                &[1.0, 1.0],
                &[0, 0],
            )]);
            let report = compare("nan", &golden, &rabbit).unwrap();
            assert_eq!(report.failed(), 1);
            assert_eq!(report.max_top_logit_diff, 0.0);
            assert!(!report.passed());
        }
        // A logit that is not finite and is neither the top nor the second, which
        // `decode` counted over all of them, fails a run whose steps look fine.
        let mut rabbit = quantized(vec![rabbit_trace(
            &[1, 2],
            &[5.0, 6.0],
            &[1.0, 1.0],
            &[0, 0],
        )]);
        assert!(compare("fine", &golden, &rabbit).unwrap().passed());
        rabbit.prompts[0].non_finite = 3;
        let report = compare("hidden", &golden, &rabbit).unwrap();
        assert_eq!(report.failed(), 1);
        assert!(!report.passed());
        assert!(report
            .details()
            .iter()
            .any(|line| line.starts_with("GOLDEN non_finite prompt=0")));
        // And such a run has no text that could not be read back: a named error.
        let error = rabbit.to_json().unwrap_err().to_string();
        assert!(error.contains("non_finite"), "{error}");
    }

    #[test]
    fn the_writer_refuses_what_the_parser_cannot_read_back() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for in_top in [true, false] {
                let mut trace = rabbit_trace(&[1, 2], &[5.0, 6.0], &[1.0, 1.0], &[0, 0]);
                if in_top {
                    trace.top[1] = bad;
                } else {
                    trace.second[0] = bad;
                }
                let error = quantized(vec![trace]).to_json().unwrap_err().to_string();
                assert!(error.contains("non_finite"), "{error}");
            }
        }
    }

    #[test]
    fn a_run_is_compared_with_the_cap_of_its_own_weights() {
        let golden = document(vec![trace(&[1, 2, 3], &[5.0, 6.0, 7.0], &[1.0, 1.0, 1.0])]);
        // 0.4 off a scale of 6 is 6.7%: under the quantized cap, over the float one.
        let traces = || {
            vec![rabbit_trace(
                &[1, 2, 3],
                &[5.0, 6.4, 7.0],
                &[1.0, 1.0, 1.0],
                &[0, 0, 0],
            )]
        };
        let mut run = document(traces());
        run.mode = Some(Mode::Plain);
        // No `weights`: refused, and the error says what to pass.
        let error = compare("x", &golden, &run).unwrap_err().to_string();
        assert!(
            error.contains("`weights`") && error.contains("--cap"),
            "{error}"
        );
        run.weights = Some(Weights::Quantized);
        let quantized_report = compare("x", &golden, &run).unwrap();
        assert!(quantized_report.passed());
        assert_eq!(quantized_report.noise_cap, NOISE_CAP_QUANTIZED);
        run.weights = Some(Weights::Float);
        let float_report = compare("x", &golden, &run).unwrap();
        assert!(!float_report.passed());
        assert_eq!(float_report.noise_cap, NOISE_CAP_FLOAT);
        assert!(float_report
            .summary()
            .ends_with(" weights=float mode=plain"));
        // A class named for a document that has none is used, and one that
        // disagrees with the document's own is an error and not a loosening.
        run.weights = None;
        assert!(compare_capped("x", &golden, &run, Weights::Quantized)
            .unwrap()
            .passed());
        assert!(!compare_capped("x", &golden, &run, Weights::Float)
            .unwrap()
            .passed());
        run.weights = Some(Weights::Float);
        assert!(compare_capped("x", &golden, &run, Weights::Quantized).is_err());
        // A run that was not decoded by this tool has no second best token.
        let bare = document(vec![trace(&[1, 2, 3], &[5.0, 6.4, 7.0], &[1.0, 1.0, 1.0])]);
        assert!(compare_capped("x", &golden, &bare, Weights::Float).is_err());
    }

    #[test]
    fn the_weights_of_a_run_come_from_all_its_matrices() {
        let spec = Spec {
            hidden: 64,
            layers: 2,
            heads: 4,
            kv_heads: 2,
            ffn: 96,
            vocab: 128,
            eps: 1e-5,
            theta: 10_000.0,
        };
        let tensors = weights(&spec);
        let build = |storage| gguf_builder(&spec, &tensors, Permutation::Converter, storage);
        let decode_weights = |bytes: Vec<u8>| {
            let file = ModelFile::gguf_from_bytes(bytes).unwrap();
            let model = Model::new(&file, file.gguf_config().unwrap()).unwrap();
            let run = run(&model, &[vec![1, 2]], 2, |_| {}).unwrap();
            assert_eq!(run.mode, Some(Mode::Plain));
            (run.weights.unwrap(), model.weight_dtype())
        };
        assert_eq!(
            decode_weights(build(Storage::F16).to_bytes()),
            (Weights::Float, "F16")
        );
        assert_eq!(
            decode_weights(build(Storage::Q8_0).to_bytes()),
            (Weights::Quantized, "Q8B32")
        );
        // An `F16` embedding with a `Q8_0` body is quantized, which the type of
        // the embedding alone does not say.
        let mut mixed = build(Storage::Q8_0);
        let half = build(Storage::F16);
        let embedding = half
            .tensors
            .iter()
            .find(|tensor| tensor.name == "token_embd.weight")
            .unwrap()
            .clone();
        *mixed.tensor("token_embd.weight").unwrap() = embedding;
        assert_eq!(
            decode_weights(mixed.to_bytes()),
            (Weights::Quantized, "F16")
        );
    }

    #[test]
    fn runs_that_do_not_describe_the_same_prompts_are_an_error() {
        let golden = document(vec![trace(&[1, 2], &[5.0, 6.0], &[1.0, 1.0])]);
        let run = quantized(vec![rabbit_trace(
            &[1, 2],
            &[5.0, 6.0],
            &[1.0, 1.0],
            &[0, 0],
        )]);
        let mut other = run.clone();
        other.prompts[0].prompt_ids = vec![2];
        assert!(compare("x", &golden, &other).is_err());
        let mut other = run.clone();
        other.steps = 3;
        assert!(compare("x", &golden, &other).is_err());
        let mut other = run.clone();
        other.prompts.clear();
        assert!(compare("x", &golden, &other).is_err());
        let empty = Document {
            vocab: 16,
            steps: 2,
            weights: Some(Weights::Float),
            mode: None,
            prompts: Vec::new(),
        };
        assert!(!compare("x", &empty, &empty).unwrap().passed());
    }

    #[test]
    fn the_synthetic_q8_0_file_decodes_the_golden_tokens_of_llama_cpp() {
        let golden = Document::parse(SYNTH_Q8_GOLDEN).unwrap();
        let report = compare("synth-Q8_0.gguf", &golden, q8_run()).unwrap();
        for line in std::iter::once(report.summary()).chain(report.details()) {
            println!("{line}");
        }
        assert_eq!(q8_run().weights, Some(Weights::Quantized));
        assert_eq!(report.noise_cap, NOISE_CAP_QUANTIZED);
        assert_invariants(&report, &golden, q8_run());
    }

    #[test]
    fn the_synthetic_q4_k_m_file_decodes_the_golden_tokens_of_llama_cpp() {
        let golden = Document::parse(SYNTH_Q4KM_GOLDEN).unwrap();
        let report = compare("synth-Q4_K_M.gguf", &golden, q4km_run()).unwrap();
        for line in std::iter::once(report.summary()).chain(report.details()) {
            println!("{line}");
        }
        assert_eq!(q4km_run().weights, Some(Weights::Quantized));
        assert_invariants(&report, &golden, q4km_run());
    }

    /// The step with the largest golden margin among the steps that Rabbit and
    /// the golden decode with the same tokens before it, so a corrupted token
    /// there is the first mismatch of its prompt.
    fn largest_margin(golden: &Document, rabbit: &Document) -> (usize, usize) {
        let mut best = (0, 0, f64::NEG_INFINITY);
        for (index, (expected, actual)) in golden.prompts.iter().zip(&rabbit.prompts).enumerate() {
            for step in 0..golden.steps {
                let same_history = expected.tokens[..step] == actual.tokens[..step];
                if same_history && expected.margin(step) > best.2 {
                    best = (index, step, expected.margin(step));
                }
            }
        }
        (best.0, best.1)
    }

    /// The step where the larger margin of the two runs is the smallest, among
    /// the steps both runs reach with the same history.
    fn tightest_step(golden: &Document, rabbit: &Document) -> (usize, usize) {
        let mut best = (0, 0, f64::INFINITY);
        for (index, (expected, actual)) in golden.prompts.iter().zip(&rabbit.prompts).enumerate() {
            for step in 0..golden.steps {
                let worse = expected.margin(step).max(actual.margin(step));
                let same_history = expected.tokens[..step] == actual.tokens[..step];
                if same_history && worse < best.2 {
                    best = (index, step, worse);
                }
            }
        }
        (best.0, best.1)
    }

    #[test]
    fn a_corrupted_golden_token_at_a_large_margin_fails() {
        for (golden_text, rabbit) in [(SYNTH_Q8_GOLDEN, q8_run()), (SYNTH_Q4KM_GOLDEN, q4km_run())]
        {
            let golden = Document::parse(golden_text).unwrap();
            let (prompt, step) = largest_margin(&golden, rabbit);
            let mut corrupted = golden.clone();
            let token = &mut corrupted.prompts[prompt].tokens[step];
            *token = (*token + 1) % golden.vocab as u32;
            let report = compare("corrupted", &corrupted, rabbit).unwrap();
            let entry = &report.prompts[prompt];
            assert_eq!(entry.verdict, Verdict::Fail, "{:#?}", report.details());
            assert_eq!(entry.mismatch.unwrap().step, step);
            assert!(!report.passed());
            // The margin is far above twice the noise.
            assert!(entry.mismatch.unwrap().golden_margin > 10.0 * report.max_top_logit_diff);
        }
    }

    /// Corrupt the golden token of a step to `token`.
    fn with_token(golden: &Document, prompt: usize, step: usize, token: u32) -> Document {
        let mut corrupted = golden.clone();
        corrupted.prompts[prompt].tokens[step] = token;
        corrupted
    }

    #[test]
    fn a_swap_of_the_two_best_tokens_at_a_tiny_margin_is_a_near_tie() {
        for (golden_text, rabbit) in [(SYNTH_Q8_GOLDEN, q8_run()), (SYNTH_Q4KM_GOLDEN, q4km_run())]
        {
            let golden = Document::parse(golden_text).unwrap();
            let honest = compare("honest", &golden, rabbit).unwrap();
            let (prompt, step) = tightest_step(&golden, rabbit);
            // The golden chose Rabbit's second best: the two best swapped.
            let swapped = with_token(
                &golden,
                prompt,
                step,
                rabbit.prompts[prompt].second_id[step],
            );
            let report = compare("tiny", &swapped, rabbit).unwrap();
            let entry = &report.prompts[prompt];
            let mismatch = entry.mismatch.unwrap();
            println!(
                "tiny margin: prompt {prompt} step {step} golden_margin {} rabbit_margin {} \
                 two_noise {} (honest file: {})",
                mismatch.golden_margin,
                mismatch.rabbit_margin,
                2.0 * report.max_top_logit_diff,
                honest.summary()
            );
            assert_eq!(mismatch.step, step);
            assert_eq!(entry.verdict, Verdict::NearTie, "{:#?}", report.details());
            assert!(report.passed());
        }
    }

    #[test]
    fn a_golden_token_that_is_not_rabbits_second_best_fails_even_at_a_tiny_margin() {
        for (golden_text, rabbit) in [(SYNTH_Q8_GOLDEN, q8_run()), (SYNTH_Q4KM_GOLDEN, q4km_run())]
        {
            let golden = Document::parse(golden_text).unwrap();
            let (prompt, step) = tightest_step(&golden, rabbit);
            let own = &rabbit.prompts[prompt];
            // Any token but Rabbit's best and second best, so that the tiny margins
            // of the two runs are the same as in the swap above.
            let token = (0..golden.vocab as u32)
                .find(|&token| token != own.tokens[step] && token != own.second_id[step])
                .unwrap();
            let corrupted = with_token(&golden, prompt, step, token);
            let report = compare("not second", &corrupted, rabbit).unwrap();
            let entry = &report.prompts[prompt];
            let mismatch = entry.mismatch.unwrap();
            assert_eq!(mismatch.step, step);
            assert_ne!(mismatch.golden_token, mismatch.rabbit_second_token);
            assert!(mismatch.golden_margin <= 2.0 * report.max_top_logit_diff);
            assert!(mismatch.rabbit_margin <= 2.0 * report.max_top_logit_diff);
            assert_eq!(entry.verdict, Verdict::Fail, "{:#?}", report.details());
            assert!(!report.passed());
        }
    }

    #[test]
    fn the_wrong_rope_convention_on_the_synthetic_q4_k_m_file_fails() {
        let golden = Document::parse(SYNTH_Q4KM_GOLDEN).unwrap();
        let right = compare("synth-Q4_K_M.gguf", &golden, q4km_run()).unwrap();
        assert!(right.passed(), "{:#?}", right.details());
        let wrong = rabbit_run(SYNTH_Q4KM_GGUF, false, false);
        let report = compare("synth-Q4_K_M.gguf split-half rope", &golden, &wrong).unwrap();
        for line in std::iter::once(report.summary()).chain(report.details()) {
            println!("{line}");
        }
        println!(
            "right reading: noise {} relative {} cap {}; split half: noise {} relative {}",
            right.max_top_logit_diff,
            right.relative_noise(),
            right.noise_cap,
            report.max_top_logit_diff,
            report.relative_noise()
        );
        // The wrong reading is much noisier than the right one, whatever the exact
        // rounding of the host.
        assert!(report.relative_noise() > 5.0 * right.relative_noise());
        assert!(report.equal() < right.equal());
        // And at least one of its first mismatches is not a near tie by the
        // yardstick of the right reading: the golden margin is more than twice its
        // noise, or the golden token is not Rabbit's second best. This host finds
        // two; one is the bound that does not hinge on a single rounding.
        let not_a_near_tie = report
            .prompts
            .iter()
            .filter_map(|prompt| prompt.mismatch)
            .filter(|mismatch| {
                mismatch.golden_margin > 2.0 * right.max_top_logit_diff
                    || mismatch.golden_token != mismatch.rabbit_second_token
            })
            .count();
        assert!(not_a_near_tie >= 1, "{not_a_near_tie}");
        // A looser statement than the cap itself, which has a factor of 1.2 to the
        // measurement of this host and could flip on another.
        assert!(report.relative_noise() > NOISE_CAP_QUANTIZED / 2.0);
        // The same wrong reading with the arithmetic of llama.cpp emulated.
        let emulated = rabbit_run(SYNTH_Q4KM_GGUF, false, true);
        let report = compare(
            "synth-Q4_K_M.gguf split-half rope emulated",
            &golden,
            &emulated,
        )
        .unwrap();
        println!("{}", report.summary());
        assert!(report.relative_noise() > 5.0 * right.relative_noise());
    }

    #[test]
    fn a_noise_above_the_cap_fails_a_file_whose_tokens_are_all_equal() {
        // Every token equal, the top logit 0.7 off a scale of about 6: 11.7%.
        let golden = document(vec![trace(&[1, 2, 3], &[5.0, 6.0, 7.0], &[1.0, 1.0, 1.0])]);
        let rabbit = quantized(vec![rabbit_trace(
            &[1, 2, 3],
            &[5.0, 6.7, 7.0],
            &[1.0, 1.0, 1.0],
            &[0, 0, 0],
        )]);
        let report = compare("loud", &golden, &rabbit).unwrap();
        assert_eq!(report.equal(), 1);
        assert_eq!(report.failed(), 0);
        assert!(report.over_cap());
        assert!(!report.passed());
        assert!(report.details()[0].starts_with("GOLDEN noise_over_cap"));
        // 0.4 off the same scale is 6.7%: under the cap of the quantized files and
        // over the one of the float files.
        let rabbit = quantized(vec![rabbit_trace(
            &[1, 2, 3],
            &[5.0, 6.4, 7.0],
            &[1.0, 1.0, 1.0],
            &[0, 0, 0],
        )]);
        assert!(compare("quiet", &golden, &rabbit).unwrap().passed());
        let unlabelled = Document {
            weights: None,
            ..rabbit.clone()
        };
        assert!(
            !compare_capped("quiet", &golden, &unlabelled, Weights::Float)
                .unwrap()
                .passed()
        );
        // A summary line names the relative noise and its cap.
        let line = compare("quiet", &golden, &rabbit).unwrap().summary();
        assert!(
            line.contains(" noise_rel=0.06") && line.contains(" noise_cap=0.085"),
            "{line}"
        );
    }

    #[test]
    fn the_right_readings_of_the_synthetic_files_are_under_the_cap() {
        for (golden_text, rabbit) in [(SYNTH_Q8_GOLDEN, q8_run()), (SYNTH_Q4KM_GOLDEN, q4km_run())]
        {
            let golden = Document::parse(golden_text).unwrap();
            let report = compare("synthetic", &golden, rabbit).unwrap();
            println!("{} relative {}", report.summary(), report.relative_noise());
            assert!(!report.over_cap());
        }
    }

    #[test]
    fn the_emulation_is_off_by_default_and_changes_the_logits_when_on() {
        for (plain, emulated) in [
            (q8_run(), q8_emulated_run()),
            (q4km_run(), q4km_emulated_run()),
        ] {
            assert_ne!(plain, emulated);
            assert_eq!(plain.mode, Some(Mode::Plain));
            assert_eq!(emulated.mode, Some(Mode::EmulateLlama));
        }
        let again = rabbit_run(SYNTH_Q8_GGUF, true, false);
        assert_eq!(&again, q8_run(), "the plain run is reproducible");
    }

    #[test]
    fn the_emulated_runs_of_the_synthetic_files_are_judged_by_the_same_rule() {
        // Not strict: the emulation does not give the golden tokens on every
        // prompt (see the module documentation), only the same verdict.
        for (name, golden_text, rabbit) in [
            ("synth-Q8_0.gguf", SYNTH_Q8_GOLDEN, q8_emulated_run()),
            ("synth-Q4_K_M.gguf", SYNTH_Q4KM_GOLDEN, q4km_emulated_run()),
        ] {
            let golden = Document::parse(golden_text).unwrap();
            let report = compare(name, &golden, rabbit).unwrap();
            for line in std::iter::once(report.summary()).chain(report.details()) {
                println!("emulated {line}");
            }
            assert_invariants(&report, &golden, rabbit);
        }
    }
}
