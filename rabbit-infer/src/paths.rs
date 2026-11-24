// SPDX-License-Identifier: GPL-2.0-only
//! The model root and the four paths every run reads.
//!
//! In the guest the model, the vocabulary, the config and the merges live at
//! the initramfs root, and the kernel gives a program no environment, so the
//! four constants are the whole story. A native run is an ordinary host
//! process: it sets
//! [`ROOT_ENV`] to the directory that holds the files, and the same binary
//! opens them there. The root is resolved once, at the start of a run, and the
//! log records whether it came from the environment or from the guest.
//!
//! This module has no I/O and no architecture specific code, so the path rules
//! are tested on the host.

use std::ffi::OsString;

use crate::error::{Error, Result};
use crate::matvec::Path;

/// Model path in the initramfs root.
pub const MODEL_PATH: &str = "/model.safetensors";

/// Vocabulary path in the initramfs root.
pub const VOCAB_PATH: &str = "/vocab.json";

/// Config path in the initramfs root.
pub const CONFIG_PATH: &str = "/config.json";

/// Merges path in the initramfs root.
pub const MERGES_PATH: &str = "/merges.txt";

/// Optional file that holds the one word of a requested code path.
pub const INFER_PATH_PATH: &str = "/infer.path";

/// Environment variable that replaces the initramfs root when it is defined.
pub const ROOT_ENV: &str = "RABBIT_INFER_ROOT";

/// The four paths a run reads, plus the root they were derived from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Paths {
    /// Root the other paths were built from.
    pub root: String,
    /// `model.safetensors` under the root.
    pub model: String,
    /// `vocab.json` under the root.
    pub vocab: String,
    /// `config.json` under the root.
    pub config: String,
    /// `merges.txt` under the root.
    pub merges: String,
    /// `infer.path` under the root, the optional request of a code path.
    pub infer_path: String,
}

impl Paths {
    /// The guest root, byte for byte the paths the initramfs has always had.
    pub fn guest() -> Self {
        Self {
            root: "/".to_string(),
            model: MODEL_PATH.to_string(),
            vocab: VOCAB_PATH.to_string(),
            config: CONFIG_PATH.to_string(),
            merges: MERGES_PATH.to_string(),
            infer_path: INFER_PATH_PATH.to_string(),
        }
    }

    /// A host directory that holds the files.
    pub fn under(root: String) -> Self {
        let separator = if root.ends_with('/') { "" } else { "/" };
        Self {
            model: format!("{root}{separator}model.safetensors"),
            vocab: format!("{root}{separator}vocab.json"),
            config: format!("{root}{separator}config.json"),
            merges: format!("{root}{separator}merges.txt"),
            infer_path: format!("{root}{separator}infer.path"),
            root,
        }
    }
}

/// The code path an `infer.path` file asks for: its one word, trimmed.
///
/// The words are the names the serial log prints, so the paths of the other
/// architectures have their own. An unknown word is an error, never a silent
/// default: a measurement of "avx2" that ran on another path is a lie.
pub fn parse_path_request(bytes: &[u8]) -> Result<Path> {
    let word = std::str::from_utf8(bytes)
        .map_err(|_| Error::new("runtime: infer.path is not valid UTF-8"))?
        .trim();
    [
        Path::Scalar,
        #[cfg(target_arch = "x86_64")]
        Path::Sse2,
        #[cfg(target_arch = "x86_64")]
        Path::Avx2,
        #[cfg(target_arch = "aarch64")]
        Path::Neon,
    ]
    .into_iter()
    .find(|path| path.name() == word)
    .ok_or_else(|| Error::new(format!("runtime: unknown path {word:?} in infer.path")))
}

/// Resolve the root and say which source it came from.
pub fn resolve() -> Result<(Paths, &'static str)> {
    resolve_from(std::env::var_os(ROOT_ENV))
}

/// The pure half of [`resolve`], so both branches are tested without touching
/// the process environment.
fn resolve_from(value: Option<OsString>) -> Result<(Paths, &'static str)> {
    match value {
        Some(value) => {
            let root = value
                .into_string()
                .map_err(|_| Error::new("runtime: RABBIT_INFER_ROOT is not valid UTF-8"))?;
            // An empty value names the current directory; keep it visible in
            // the log instead of silently falling back to the guest root.
            let root = if root.is_empty() {
                ".".to_string()
            } else {
                root
            };
            Ok((Paths::under(root), "environment"))
        }
        None => Ok((Paths::guest(), "guest")),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        parse_path_request, resolve_from, Paths, CONFIG_PATH, INFER_PATH_PATH, MERGES_PATH,
        MODEL_PATH, VOCAB_PATH,
    };
    use crate::matvec::Path;
    use std::ffi::OsString;

    #[test]
    fn the_request_file_sits_at_the_root_like_the_others() {
        assert_eq!(Paths::guest().infer_path, INFER_PATH_PATH);
        assert_eq!(
            Paths::under("/models/x".to_string()).infer_path,
            "/models/x/infer.path"
        );
    }

    #[test]
    fn a_request_is_one_trimmed_word() {
        assert_eq!(parse_path_request(b"scalar").unwrap(), Path::Scalar);
        assert_eq!(parse_path_request(b" scalar\n").unwrap(), Path::Scalar);
        #[cfg(target_arch = "x86_64")]
        {
            assert_eq!(parse_path_request(b"sse2\n").unwrap(), Path::Sse2);
            assert_eq!(parse_path_request(b"avx2\r\n").unwrap(), Path::Avx2);
        }
    }

    #[test]
    fn an_unknown_empty_or_odd_word_is_an_error() {
        for word in [&b""[..], b"\n", b"avx512", b"AVX2", b"sse2 avx2", b"\xff"] {
            assert!(parse_path_request(word).is_err(), "{word:?}");
        }
    }

    #[test]
    fn the_guest_paths_are_the_root_constants() {
        let paths = Paths::guest();
        assert_eq!(paths.root, "/");
        assert_eq!(paths.model, MODEL_PATH);
        assert_eq!(paths.vocab, VOCAB_PATH);
        assert_eq!(paths.config, CONFIG_PATH);
        assert_eq!(paths.merges, MERGES_PATH);
    }

    #[test]
    fn an_environment_root_is_prepended() {
        let paths = Paths::under("/models/smollm2".to_string());
        assert_eq!(paths.model, "/models/smollm2/model.safetensors");
        assert_eq!(paths.vocab, "/models/smollm2/vocab.json");
        assert_eq!(paths.config, "/models/smollm2/config.json");
        assert_eq!(paths.merges, "/models/smollm2/merges.txt");
        let trailing = Paths::under("/models/smollm2/".to_string());
        assert_eq!(trailing.model, "/models/smollm2/model.safetensors");
    }

    #[test]
    fn an_absent_variable_names_the_guest_root() {
        let (paths, source) = resolve_from(None).expect("the guest root always resolves");
        assert_eq!(source, "guest");
        assert_eq!(paths, Paths::guest());
    }

    #[test]
    fn a_defined_variable_names_the_environment() {
        let (paths, source) =
            resolve_from(Some(OsString::from("/models/smollm2"))).expect("a UTF-8 root resolves");
        assert_eq!(source, "environment");
        assert_eq!(paths.model, "/models/smollm2/model.safetensors");
        assert_eq!(paths.root, "/models/smollm2");
        let (empty, source) = resolve_from(Some(OsString::from(""))).expect("an empty root");
        assert_eq!(source, "environment");
        assert_eq!(empty.model, "./model.safetensors");
    }
}
