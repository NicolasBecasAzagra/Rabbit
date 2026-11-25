// SPDX-License-Identifier: GPL-2.0-only
//! Whole tensor verification of a GGUF file against the hashes of `gguf-py`.
//!
//! `bench/gguf/dump_reference.py` reads a GGUF file with the `gguf` package of
//! `llama.cpp` and writes, for every quantized tensor, the SHA-256 of its raw
//! bytes (`sha256_raw`) and of its dequantised values as little endian `f32`
//! (`sha256_f32`). [`verify`] does the same with the reader and the
//! dequantisers of Rabbit and compares both, for every quantized tensor of the
//! file: a block that Rabbit decodes to one different bit changes the hash of its
//! tensor. It has no I/O and is tested on the committed synthetic files; the
//! `rabbit-infer-golden --verify-tensors` tool runs it on the real ones.
//!
//! The check fails, and does not only count, when the file has a quantized tensor
//! that the reference does not list, when the reference lists one that the file
//! does not have, when the file has no tensor that is read while the reference
//! is not empty, when the file has a quantized tensor of a type that is not read
//! (it cannot be hashed with the dequantisers of Rabbit), and when the reference
//! names a tensor twice (a named error: the second entry would silently win).
//! The tensors of a type that Rabbit does not read are counted as `skipped`, with
//! the names of their types; the `F32` and `F16` tensors are read, are not
//! quantized and are not part of this check, so they are not counted there.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::error::{Error, Result};
use crate::gguf::{ggml_type_name, Gguf, TensorKind};
use crate::json::Scanner;
use crate::quant;
use crate::sha256;

/// One entry of a `tensors-*.json`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reference {
    /// The name of the tensor in the file.
    pub name: String,
    /// The type, as `gguf-py` names it: `Q8_0`, `Q5_0`, `Q4_K` or `Q6_K`.
    pub kind: String,
    /// The bytes of the tensor in the file.
    pub bytes: usize,
    /// The SHA-256 of the raw bytes, in hexadecimal.
    pub sha256_raw: String,
    /// The SHA-256 of the dequantised values as little endian `f32`.
    pub sha256_f32: String,
}

/// The list that `dump_reference.py` writes: `{name, type, shape, bytes,
/// sha256_raw, sha256_f32}` per quantized tensor.
pub fn parse_reference(text: &str) -> Result<Vec<Reference>> {
    let mut scanner = Scanner::new(text.as_bytes());
    let mut entries = Vec::new();
    scanner.skip_whitespace();
    scanner.expect(b'[')?;
    scanner.skip_whitespace();
    if scanner.peek() == Some(b']') {
        scanner.expect(b']')?;
        return if scanner.at_end() {
            Ok(entries)
        } else {
            Err(Error::new("verify: trailing bytes after the list"))
        };
    }
    loop {
        let mut entry = Reference {
            name: String::new(),
            kind: String::new(),
            bytes: 0,
            sha256_raw: String::new(),
            sha256_f32: String::new(),
        };
        scanner.skip_whitespace();
        scanner.expect(b'{')?;
        loop {
            scanner.skip_whitespace();
            let key = scanner.parse_string()?;
            scanner.skip_whitespace();
            scanner.expect(b':')?;
            scanner.skip_whitespace();
            match key.as_str() {
                "name" => entry.name = scanner.parse_string()?,
                "type" => entry.kind = scanner.parse_string()?,
                "bytes" => entry.bytes = scanner.parse_usize()?,
                "sha256_raw" => entry.sha256_raw = scanner.parse_string()?,
                "sha256_f32" => entry.sha256_f32 = scanner.parse_string()?,
                _ => scanner.skip_value()?,
            }
            scanner.skip_whitespace();
            match scanner.bump()? {
                b',' => continue,
                b'}' => break,
                _ => return Err(Error::new("verify: expected ',' or '}' in an entry")),
            }
        }
        if entry.name.is_empty() || entry.sha256_raw.is_empty() || entry.sha256_f32.is_empty() {
            return Err(Error::new(
                "verify: an entry has no name, sha256_raw or sha256_f32",
            ));
        }
        entries.push(entry);
        scanner.skip_whitespace();
        match scanner.bump()? {
            b',' => continue,
            b']' => break,
            _ => return Err(Error::new("verify: expected ',' or ']' in the list")),
        }
    }
    if !scanner.at_end() {
        return Err(Error::new("verify: trailing bytes after the list"));
    }
    check_unique(&entries)?;
    Ok(entries)
}

/// A reference that names a tensor twice is an error.
fn check_unique(reference: &[Reference]) -> Result<()> {
    let mut names = HashSet::new();
    for entry in reference {
        if !names.insert(entry.name.as_str()) {
            return Err(Error::new(format!(
                "verify: the reference lists the tensor `{}` twice",
                entry.name
            )));
        }
    }
    Ok(())
}

/// What [`verify`] found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verification {
    /// Quantized tensors of the file.
    pub tensors: usize,
    /// Tensors of the reference, to tell an empty file from an empty reference.
    pub listed: usize,
    /// Tensors of the file whose type is not read, per name of the type.
    pub skipped: BTreeMap<String, usize>,
    /// Tensors whose raw bytes have the hash of the reference.
    pub raw_ok: usize,
    /// Tensors whose dequantised values have the hash of the reference.
    pub f32_ok: usize,
    /// One entry per thing that differs: `name:raw`, `name:f32`, `name:type`,
    /// `name:bytes`, `name:missing_in_reference`, `name:missing_in_file`,
    /// `name:unread_quantized_type` or `no_quantized_tensor_in_file`.
    pub mismatches: Vec<String>,
}

impl Verification {
    /// `true` when every tensor matches and nothing is missing.
    pub fn passed(&self) -> bool {
        self.mismatches.is_empty()
            && (self.tensors > 0 || self.listed == 0)
            && self.raw_ok == self.tensors
            && self.f32_ok == self.tensors
    }

    /// The report line.
    pub fn line(&self, file: &str) -> String {
        let skipped: usize = self.skipped.values().sum();
        let types: Vec<String> = self
            .skipped
            .iter()
            .map(|(name, count)| format!("{name}x{count}"))
            .collect();
        format!(
            "VERIFY file={file} tensors={} raw_ok={} f32_ok={} skipped={skipped} \
             skipped_types=[{}] mismatches=[{}]",
            self.tensors,
            self.raw_ok,
            self.f32_ok,
            types.join(","),
            self.mismatches.join(",")
        )
    }
}

/// The name of a quantized type as `gguf-py` writes it, `None` for `F32`/`F16`.
fn quantized_label(kind: TensorKind) -> Option<&'static str> {
    match kind {
        TensorKind::Q8_0 => Some("Q8_0"),
        TensorKind::Q5_0 => Some("Q5_0"),
        TensorKind::Q4K => Some("Q4_K"),
        TensorKind::Q6K => Some("Q6_K"),
        TensorKind::F32 | TensorKind::F16 => None,
    }
}

/// `true` for the `ggml_type` codes of plain numbers that are not quantized and
/// that this check does not hash: `F32`, `F16`, `BF16`, `F64` and the integers.
fn is_plain(code: u32) -> bool {
    matches!(code, 0 | 1 | 24..=28 | 30)
}

/// Hash the raw bytes and the dequantised values of every quantized tensor of
/// `file` and compare them with `reference`.
pub fn verify(file: &[u8], reference: &[Reference]) -> Result<Verification> {
    check_unique(reference)?;
    let gguf = Gguf::parse(file).map_err(|error| Error::new(format!("verify: {error}")))?;
    let listed: HashMap<&str, &Reference> = reference
        .iter()
        .map(|entry| (entry.name.as_str(), entry))
        .collect();
    let mut result = Verification {
        tensors: 0,
        listed: reference.len(),
        skipped: BTreeMap::new(),
        raw_ok: 0,
        f32_ok: 0,
        mismatches: Vec::new(),
    };
    let mut seen = Vec::new();
    for info in gguf.tensors() {
        let Some(label) = info.kind().and_then(quantized_label) else {
            if info.kind().is_none() {
                let type_name = ggml_type_name(info.ggml_type)
                    .map_or_else(|| format!("type{}", info.ggml_type), str::to_string);
                if !is_plain(info.ggml_type) {
                    result
                        .mismatches
                        .push(format!("{}:unread_quantized_type", info.name));
                }
                *result.skipped.entry(type_name).or_insert(0) += 1;
            }
            continue;
        };
        result.tensors += 1;
        seen.push(info.name);
        let Some(entry) = listed.get(info.name) else {
            result
                .mismatches
                .push(format!("{}:missing_in_reference", info.name));
            continue;
        };
        let data = gguf
            .tensor_data(info)
            .map_err(|error| Error::new(format!("verify: {error}")))?;
        if entry.kind != label {
            result.mismatches.push(format!("{}:type", info.name));
        }
        if entry.bytes != data.len() {
            result.mismatches.push(format!("{}:bytes", info.name));
        }
        if sha256::hex(&sha256::digest(data)) == entry.sha256_raw {
            result.raw_ok += 1;
        } else {
            result.mismatches.push(format!("{}:raw", info.name));
        }
        let elements = info
            .elements()
            .and_then(|count| usize::try_from(count).ok())
            .ok_or_else(|| Error::new(format!("verify: {} is too large", info.name)))?;
        let mut values = vec![0.0f32; elements];
        match info.kind() {
            Some(TensorKind::Q8_0) => quant::dequantize_row_q8(data, &mut values)?,
            Some(TensorKind::Q5_0) => quant::dequantize_row_q5_0(data, &mut values)?,
            Some(TensorKind::Q4K) => quant::dequantize_row_q4_k(data, &mut values)?,
            _ => quant::dequantize_row_q6_k(data, &mut values)?,
        }
        let mut bytes = Vec::with_capacity(4 * elements);
        for value in &values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        if sha256::hex(&sha256::digest(&bytes)) == entry.sha256_f32 {
            result.f32_ok += 1;
        } else {
            result.mismatches.push(format!("{}:f32", info.name));
        }
    }
    if result.tensors == 0 && !reference.is_empty() {
        result
            .mismatches
            .push("no_quantized_tensor_in_file".to_string());
    }
    for entry in reference {
        if !seen.contains(&entry.name.as_str()) {
            result
                .mismatches
                .push(format!("{}:missing_in_file", entry.name));
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    const Q4KM_FILE: &[u8] = include_bytes!("../../bench/gguf/synthetic/synth-Q4_K_M.gguf");
    const Q4KM_REF: &str =
        include_str!("../../bench/gguf/synthetic/ref/tensors-synth-Q4_K_M.gguf.json");
    const Q8_FILE: &[u8] = include_bytes!("../../bench/gguf/synthetic/synth-Q8_0.gguf");
    const Q8_REF: &str =
        include_str!("../../bench/gguf/synthetic/ref/tensors-synth-Q8_0.gguf.json");

    #[test]
    fn the_committed_synthetic_files_verify_whole() {
        for (file, reference, tensors) in [(Q4KM_FILE, Q4KM_REF, 29), (Q8_FILE, Q8_REF, 29)] {
            let reference = parse_reference(reference).unwrap();
            let result = verify(file, &reference).unwrap();
            println!("{}", result.line("synth"));
            assert_eq!(result.tensors, tensors);
            assert!(result.passed(), "{}", result.line("synth"));
        }
    }

    #[test]
    fn a_wrong_hash_a_wrong_type_and_a_missing_entry_are_mismatches() {
        let mut reference = parse_reference(Q4KM_REF).unwrap();
        let first = reference[0].name.clone();
        let second = reference[1].name.clone();
        let third = reference[2].name.clone();
        reference[0].sha256_raw = "0".repeat(64);
        reference[1].sha256_f32 = "0".repeat(64);
        reference[2].kind = "Q8_0".to_string();
        let removed = reference.pop().unwrap();
        reference.push(Reference {
            name: "not.in.the.file".to_string(),
            ..removed.clone()
        });
        let result = verify(Q4KM_FILE, &reference).unwrap();
        assert!(!result.passed());
        assert_eq!(result.tensors, 29);
        // One tensor fails the raw hash, another the `f32` hash, and the last
        // has no entry: neither of its hashes is checked.
        assert_eq!(result.raw_ok, 27);
        assert_eq!(result.f32_ok, 27);
        for expected in [
            format!("{first}:raw"),
            format!("{second}:f32"),
            format!("{third}:type"),
            format!("{}:missing_in_reference", removed.name),
            "not.in.the.file:missing_in_file".to_string(),
        ] {
            assert!(result.mismatches.contains(&expected), "{expected}");
        }
    }

    #[test]
    fn a_flipped_bit_of_a_tensor_changes_both_hashes() {
        let reference = parse_reference(Q8_REF).unwrap();
        let gguf = Gguf::parse(Q8_FILE).unwrap();
        let info = gguf.tensor(&reference[0].name).unwrap();
        let start = gguf.data_offset() + info.offset as usize + 5;
        let mut file = Q8_FILE.to_vec();
        file[start] ^= 1;
        let result = verify(&file, &reference).unwrap();
        assert!(!result.passed());
        assert!(result
            .mismatches
            .contains(&format!("{}:raw", reference[0].name)));
        assert!(result
            .mismatches
            .contains(&format!("{}:f32", reference[0].name)));
    }

    /// The synthetic model of the committed files, small and written by the
    /// builder, with the matrices stored as `storage` says.
    fn built(storage: crate::synthetic::Storage) -> crate::synthetic::GgufBuilder {
        use crate::synthetic::{gguf_builder, weights, Permutation, Spec};
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
        gguf_builder(&spec, &weights(&spec), Permutation::Converter, storage)
    }

    #[test]
    fn a_file_with_no_quantized_tensor_fails_against_a_reference_that_lists_some() {
        let reference = parse_reference(Q8_REF).unwrap();
        let file = built(crate::synthetic::Storage::F32).to_bytes();
        let result = verify(&file, &reference).unwrap();
        assert_eq!(result.tensors, 0);
        assert!(!result.passed());
        assert!(result
            .mismatches
            .contains(&"no_quantized_tensor_in_file".to_string()));
        // The `F32` tensors are read and not quantized: not skipped.
        assert!(result.skipped.is_empty());
        let line = result.line("f32.gguf");
        assert!(line.contains(" skipped=0 skipped_types=[]"), "{line}");
        // Nothing to verify against nothing is not a failure of the tool.
        assert!(verify(&file, &[]).unwrap().passed());
    }

    #[test]
    fn quantized_tensors_that_the_reference_does_not_list_fail() {
        let mut reference = parse_reference(Q8_REF).unwrap();
        let file = Q8_FILE;
        assert!(verify(file, &reference).unwrap().passed());
        let dropped = reference.remove(3);
        let result = verify(file, &reference).unwrap();
        assert!(!result.passed());
        assert!(result
            .mismatches
            .contains(&format!("{}:missing_in_reference", dropped.name)));
        let none = verify(file, &[]).unwrap();
        assert!(!none.passed());
        assert_eq!(none.mismatches.len(), none.tensors);
    }

    #[test]
    fn a_tensor_of_a_type_that_is_not_read_is_counted_and_a_quantized_one_fails() {
        let reference = parse_reference(Q8_REF).unwrap();
        let mut builder = built(crate::synthetic::Storage::Q8_0);
        // `Q4_0` (2) is a quantized type of ggml that Rabbit does not read, `I32`
        // (26) is a plain one.
        let name = builder
            .tensors
            .iter()
            .find(|tensor| tensor.ggml_type == 8)
            .unwrap()
            .name
            .clone();
        builder.tensor(&name).unwrap().ggml_type = 2;
        let result = verify(&builder.to_bytes(), &reference).unwrap();
        assert!(!result.passed());
        assert!(result.skipped.contains_key("Q4_0"), "{:?}", result.skipped);
        assert!(result
            .mismatches
            .contains(&format!("{name}:unread_quantized_type")));
        let mut builder = built(crate::synthetic::Storage::Q8_0);
        builder.tensor(&name).unwrap().ggml_type = 26;
        let result = verify(&builder.to_bytes(), &reference).unwrap();
        assert!(result.skipped.contains_key("I32"));
        assert!(!result
            .mismatches
            .contains(&format!("{name}:unread_quantized_type")));
    }

    #[test]
    fn a_reference_that_names_a_tensor_twice_is_a_named_error() {
        let mut reference = parse_reference(Q8_REF).unwrap();
        let twice = reference[0].clone();
        reference.push(twice.clone());
        let error = verify(Q8_FILE, &reference).unwrap_err().to_string();
        assert!(
            error.contains("twice") && error.contains(&twice.name),
            "{error}"
        );
        let text = r#"[{"name": "a", "sha256_raw": "x", "sha256_f32": "y"},
                       {"name": "a", "sha256_raw": "x", "sha256_f32": "y"}]"#;
        let error = parse_reference(text).unwrap_err().to_string();
        assert!(error.contains("twice"), "{error}");
    }

    #[test]
    fn a_malformed_list_is_an_error() {
        for text in ["", "{}", "[{\"name\": \"a\"}]", "[] x", "[{\"name\": \"a\""] {
            assert!(parse_reference(text).is_err(), "{text}");
        }
        assert!(parse_reference("[]").unwrap().is_empty());
    }
}
