// SPDX-License-Identifier: GPL-2.0-only
//! Decode a GGUF file greedily and print what `llama_golden` prints, to compare.
//!
//! ```sh
//! rabbit-infer-golden [--emulate-llama] MODEL.gguf STEPS PROMPTS.txt [GOLDEN.json]
//! rabbit-infer-golden --compare RUN.json GOLDEN.json [--cap float|quantized]
//! rabbit-infer-golden --verify-tensors MODEL.gguf REFERENCE.json
//! ```
//!
//! The first form loads the GGUF file, decodes `STEPS` greedy tokens after each
//! prompt of `PROMPTS.txt` (one prompt per line, the token ids separated by
//! spaces, as `bench/gguf/llama_golden.cpp` reads it; the end of sequence token
//! does not stop a run and a tie goes to the first index) and prints on standard
//! output the document of that harness: `vocab`, `steps` and, per prompt,
//! `prompt_ids`, `tokens`, `top` and `second`. With a fourth argument, the golden
//! document of `llama.cpp` for the same file, it also compares the run with it by
//! the rule of [`rabbit_infer::golden_compare`] and prints the report on standard
//! error: one `GOLDEN file=... fail=...` summary line, a line per prompt and, for
//! every prompt that is not fully equal, its first mismatch with the tokens and
//! the two margins. The second form compares a document written earlier and
//! prints the report on standard output.
//!
//! The document of a run carries `weights` (`float` or `quantized`, decided from
//! every matrix of the model) and `mode` (`plain` or `emulate-llama`), and
//! `second_id`, the second best token of every step that the near tie rule
//! needs. `--compare` reads them: the cap of the noise is that of the `weights`
//! of the document, and the summary line prints it with the mode. A document
//! without `weights` is refused unless `--cap float|quantized` names the class,
//! and a `--cap` that disagrees with the document's own is refused too. A run
//! with a logit that is not finite is not written (a named error, exit status 1).
//!
//! `--emulate-llama` is a diagnostic of the oracle's arithmetic: the decoder
//! multiplies the activations that the CPU backend of `llama.cpp` multiplies
//! (see [`rabbit_infer::llama_emulation`]) and rounds the keys and values to
//! binary16, and the document and the report say `mode=emulate-llama`. It is off by default and
//! is not a claim about the arithmetic of Rabbit.
//!
//! `--verify-tensors` hashes, for every quantized tensor of the file, the raw
//! bytes and the values that Rabbit's dequantisers give (little endian `f32`)
//! and compares both with the `sha256_raw` and `sha256_f32` that
//! `bench/gguf/dump_reference.py` wrote for the same file. It prints one
//! `VERIFY file=... tensors=... raw_ok=... f32_ok=... mismatches=[...]` line and
//! exits with 1 when anything differs, when the file has quantized tensors that
//! the reference does not list or none at all, or when a tensor of a type that is
//! not read is quantized (it is counted as `skipped=<n>` with its type names).
//!
//! The exit status is 0 when the run was compared and no prompt fails and the
//! noise is within its cap (or when it was not compared), 1 when a prompt fails or
//! the noise is over the cap and 2 for a usage error or a file that cannot be
//! read. The cap is that of the weights of the model for the first form and that
//! of the document (or of `--cap`) for `--compare`. The tool needs no guest: on a
//! host that is not Linux the file is read into memory instead of mapped.

#![deny(unsafe_op_in_unsafe_fn)]

use std::io::Write;
use std::process::ExitCode;
use std::time::Instant;

use rabbit_infer::error::{Error, Result};
use rabbit_infer::golden_compare::{self, Document, Report, Weights};
use rabbit_infer::model::{Model, ModelFile};
use rabbit_infer::tensor_verify;

const USAGE: &str =
    "usage: rabbit-infer-golden [--emulate-llama] MODEL.gguf STEPS PROMPTS.txt [GOLDEN.json]\n       \
                     rabbit-infer-golden --compare RUN.json GOLDEN.json [--cap float|quantized]\n       \
                     rabbit-infer-golden --verify-tensors MODEL.gguf REFERENCE.json";

/// The last component of a path, for the summary line.
fn base_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn read(path: &str) -> Result<String> {
    std::fs::read_to_string(path)
        .map_err(|error| Error::new(format!("cannot read {path}: {error}")))
}

fn decode_file(
    model_path: &str,
    steps: usize,
    prompts_path: &str,
    emulate: bool,
) -> Result<Document> {
    let prompts = golden_compare::parse_prompts(&read(prompts_path)?)?;
    if prompts.is_empty() {
        return Err(Error::new(format!("no prompt in {prompts_path}")));
    }
    let started = Instant::now();
    let file = ModelFile::open_gguf(model_path)?;
    let config = file
        .gguf_config()
        .cloned()
        .ok_or_else(|| Error::new("the file is not a GGUF file"))?;
    let mut model = Model::new(&file, &config)?;
    model.set_emulate_llama(emulate);
    eprintln!(
        "golden: {} loaded in {:.1} s ({} tensors, embedding {}, quantized matrices {})",
        base_name(model_path),
        started.elapsed().as_secs_f64(),
        model.tensor_count(),
        model.weight_dtype(),
        model.has_quantized_matrix()
    );
    let decoding = Instant::now();
    let document = golden_compare::run(&model, &prompts, steps, |index| {
        eprintln!(
            "golden: prompt {}/{} done, {:.1} s since the first",
            index + 1,
            prompts.len(),
            decoding.elapsed().as_secs_f64()
        );
    })?;
    Ok(document)
}

fn parse_document(path: &str) -> Result<Document> {
    Document::parse(&read(path)?).map_err(|error| Error::new(format!("{path}: {error}")))
}

fn report_lines(report: &Report) -> Vec<String> {
    let mut lines = report.details();
    lines.push(report.summary());
    lines
}

/// Remove `name VALUE` from the arguments and return the value.
fn take_option(arguments: &mut Vec<String>, name: &str) -> Result<Option<String>> {
    let Some(position) = arguments.iter().position(|argument| argument == name) else {
        return Ok(None);
    };
    if position + 1 >= arguments.len() {
        return Err(Error::new(format!("{name} needs a value\n{USAGE}")));
    }
    let value = arguments.remove(position + 1);
    arguments.remove(position);
    Ok(Some(value))
}

fn verify_tensors(model_path: &str, reference_path: &str) -> Result<ExitCode> {
    let reference = tensor_verify::parse_reference(&read(reference_path)?)
        .map_err(|error| Error::new(format!("{reference_path}: {error}")))?;
    let bytes = std::fs::read(model_path)
        .map_err(|error| Error::new(format!("cannot read {model_path}: {error}")))?;
    let verification = tensor_verify::verify(&bytes, &reference)?;
    println!("{}", verification.line(base_name(model_path)));
    Ok(if verification.passed() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}

fn run(arguments: &[String]) -> Result<ExitCode> {
    let emulate = arguments
        .iter()
        .any(|argument| argument == "--emulate-llama");
    let mut arguments: Vec<String> = arguments
        .iter()
        .filter(|argument| *argument != "--emulate-llama")
        .cloned()
        .collect();
    let cap = match take_option(&mut arguments, "--cap")? {
        None => None,
        Some(word) => Some(
            Weights::from_name(&word)
                .ok_or_else(|| Error::new(format!("--cap is `{word}`, not float or quantized")))?,
        ),
    };
    if cap.is_some() && arguments.first().map(String::as_str) != Some("--compare") {
        return Err(Error::new(format!("--cap is only for --compare\n{USAGE}")));
    }
    match arguments.as_slice() {
        [flag, model_path, reference_path] if flag == "--verify-tensors" && !emulate => {
            verify_tensors(model_path, reference_path)
        }
        [flag, run_path, golden_path] if flag == "--compare" && !emulate => {
            let golden = parse_document(golden_path)?;
            let run = parse_document(run_path)?;
            let name = base_name(run_path);
            let report = match cap {
                Some(weights) => golden_compare::compare_capped(name, &golden, &run, weights),
                None => golden_compare::compare(name, &golden, &run),
            }
            .map_err(|error| Error::new(format!("{run_path}: {error}")))?;
            for line in report_lines(&report) {
                println!("{line}");
            }
            Ok(exit_code(&report))
        }
        [model_path, steps, prompts_path, rest @ ..] if rest.len() <= 1 => {
            let steps = steps
                .parse::<usize>()
                .ok()
                .filter(|&steps| steps > 0)
                .ok_or_else(|| Error::new("STEPS must be a positive number"))?;
            let document = decode_file(model_path, steps, prompts_path, emulate)?;
            // A run with a logit that is not finite has no document to write: it
            // is reported and the run fails, but it is still compared if it can be.
            let written = document.to_json();
            match &written {
                Ok(text) => {
                    let mut out = std::io::stdout().lock();
                    out.write_all(text.as_bytes())?;
                    out.flush()?;
                }
                Err(error) => eprintln!("rabbit-infer-golden: {error}"),
            }
            let failed_write = if written.is_err() {
                ExitCode::from(1)
            } else {
                ExitCode::SUCCESS
            };
            let Some(golden_path) = rest.first() else {
                return Ok(failed_write);
            };
            let report = golden_compare::compare(
                base_name(model_path),
                &parse_document(golden_path)?,
                &document,
            )?;
            for line in report_lines(&report) {
                eprintln!("{line}");
            }
            if written.is_err() {
                return Ok(failed_write);
            }
            Ok(exit_code(&report))
        }
        _ => Err(Error::new(USAGE)),
    }
}

fn exit_code(report: &Report) -> ExitCode {
    if report.passed() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match run(&arguments) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("rabbit-infer-golden: {error}");
            ExitCode::from(2)
        }
    }
}
