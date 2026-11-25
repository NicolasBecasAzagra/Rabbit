// SPDX-License-Identifier: GPL-2.0-only
//! Synthetic model generator for developing the runtime.
//!
//! The real weights are not available while the runtime is written, so this
//! development tool writes a `safetensors` file with the same tensor names and
//! shapes a Llama-style decoder uses, a `config.json` with the parameters the
//! shapes cannot carry, and a byte level `vocab.json`. The values come from a
//! fixed seed, so every run writes byte-for-byte identical files and the
//! generated token ids can be asserted as a golden value.
//!
//! The weights come from [`rabbit_infer::synthetic`], which also writes the same
//! weights as a GGUF file of the `llama` architecture: `--gguf` adds
//! `model.gguf`, with the rows of the query and key projections permuted as the
//! `llama.cpp` converter does, and `--gguf-unpermuted` adds
//! `model-nopermute.gguf`, the same file without the permutation, which a
//! reader that rotates adjacent pairs must get wrong. The files the generator
//! wrote before these flags existed are unchanged, byte for byte.
//!
//! This binary is not part of the runtime: it is built on the host by
//! `make infer-model` and its output is packed into the test initramfs.

#![deny(unsafe_op_in_unsafe_fn)]

use std::fs;
use std::path::PathBuf;

use rabbit_infer::synthetic::{
    self, gguf_builder, safetensors_bytes, Permutation, Spec, Storage, Tensor,
};
use rabbit_infer::vocab::token_for_byte;

/// Arguments with the dimensions the model is built from.
struct Args {
    spec: Spec,
    interleaved: bool,
    gguf: bool,
    gguf_unpermuted: bool,
    out: PathBuf,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            spec: Spec::default(),
            interleaved: false,
            gguf: false,
            gguf_unpermuted: false,
            out: PathBuf::from("rabbit-infer-model"),
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rabbit-infer-gen: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    args.spec.validate().map_err(|error| error.to_string())?;
    let spec = args.spec;

    fs::create_dir_all(&args.out).map_err(|error| format!("cannot create output: {error}"))?;

    let tensors = synthetic::weights(&spec);

    fs::write(
        args.out.join("model.safetensors"),
        safetensors_bytes(&tensors),
    )
    .map_err(|error| format!("cannot write model: {error}"))?;
    write_vocab(&args.out.join("vocab.json"), spec.vocab)?;
    write_config(&args.out.join("config.json"), &args)?;
    if args.gguf {
        write_gguf(
            &args.out.join("model.gguf"),
            &spec,
            &tensors,
            Permutation::Converter,
        )?;
    }
    if args.gguf_unpermuted {
        write_gguf(
            &args.out.join("model-nopermute.gguf"),
            &spec,
            &tensors,
            Permutation::None,
        )?;
    }

    let params: usize = tensors
        .iter()
        .map(|tensor| tensor.shape.iter().product::<usize>())
        .sum();
    println!(
        "rabbit-infer-gen: wrote {} tensors, {} parameters, dims hidden={} layers={} q_heads={} kv_heads={} head_dim={} ffn={} vocab={}",
        tensors.len(),
        params,
        spec.hidden,
        spec.layers,
        spec.heads,
        spec.kv_heads,
        spec.head_dim(),
        spec.ffn,
        spec.vocab
    );
    Ok(())
}

/// Write the GGUF twin of the checkpoint, in `F32`.
fn write_gguf(
    path: &PathBuf,
    spec: &Spec,
    tensors: &[Tensor],
    permutation: Permutation,
) -> Result<(), String> {
    let bytes = gguf_builder(spec, tensors, permutation, Storage::F32).to_bytes();
    fs::write(path, bytes).map_err(|error| format!("cannot write gguf: {error}"))
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut iterator = std::env::args().skip(1);
    while let Some(argument) = iterator.next() {
        match argument.as_str() {
            "--hidden" => args.spec.hidden = parse_usize(iterator.next(), "--hidden")?,
            "--layers" => args.spec.layers = parse_usize(iterator.next(), "--layers")?,
            "--heads" => args.spec.heads = parse_usize(iterator.next(), "--heads")?,
            "--kv-heads" => args.spec.kv_heads = parse_usize(iterator.next(), "--kv-heads")?,
            "--ffn" => args.spec.ffn = parse_usize(iterator.next(), "--ffn")?,
            "--vocab" => args.spec.vocab = parse_usize(iterator.next(), "--vocab")?,
            "--eps" => args.spec.eps = parse_f32(iterator.next(), "--eps")?,
            "--theta" => args.spec.theta = parse_f32(iterator.next(), "--theta")?,
            "--interleaved" => args.interleaved = true,
            "--gguf" => args.gguf = true,
            "--gguf-unpermuted" => args.gguf_unpermuted = true,
            "--out" => {
                args.out = PathBuf::from(
                    iterator
                        .next()
                        .ok_or_else(|| "--out needs a directory".to_string())?,
                )
            }
            "--help" | "-h" => {
                println!(
                    "usage: rabbit-infer-gen [--hidden N] [--layers N] [--heads N] \
                     [--kv-heads N] [--ffn N] [--vocab N] [--eps F] [--theta F] \
                     [--interleaved] [--gguf] [--gguf-unpermuted] [--out DIR]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(args)
}

fn parse_usize(value: Option<String>, name: &str) -> Result<usize, String> {
    value
        .ok_or_else(|| format!("{name} needs a value"))?
        .parse::<usize>()
        .map_err(|_| format!("{name} needs a non-negative integer"))
}

fn parse_f32(value: Option<String>, name: &str) -> Result<f32, String> {
    value
        .ok_or_else(|| format!("{name} needs a value"))?
        .parse::<f32>()
        .map_err(|_| format!("{name} needs a number"))
}

/// Write the byte level vocabulary.
fn write_vocab(path: &PathBuf, vocab_size: usize) -> Result<(), String> {
    let mut json = String::from("{");
    for id in 0..vocab_size {
        if id != 0 {
            json.push(',');
        }
        let token = if id < 256 {
            token_for_byte(id as u8)
        } else {
            format!("<extra{id}>")
        };
        json.push('"');
        push_escaped(&mut json, &token);
        json.push_str("\":");
        json.push_str(&id.to_string());
    }
    json.push('}');
    fs::write(path, json).map_err(|error| format!("cannot write vocabulary: {error}"))
}

/// Write the parameters the tensor shapes cannot carry.
fn write_config(path: &PathBuf, args: &Args) -> Result<(), String> {
    fs::write(path, synthetic::config_json(&args.spec, args.interleaved))
        .map_err(|error| format!("cannot write config: {error}"))
}

/// Escape the characters JSON strings cannot carry raw.
fn push_escaped(out: &mut String, text: &str) {
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            character if (character as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => out.push(character),
        }
    }
}
