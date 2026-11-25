// SPDX-License-Identifier: GPL-2.0-only
//! A reader of the GGUF container format.
//!
//! The parser is pure: it takes the bytes of a whole file, borrows from them
//! and never copies a tensor. It touches no file system and no `mmap`, so it
//! builds and is tested on every host. It never panics on hostile input: every
//! read is bounds checked, every size is computed with checked arithmetic, no
//! count taken from the file sizes an allocation before it is compared with
//! the bytes that remain, and no allocation made up front from such a count
//! exceeds a small constant: the vectors grow as the entries are read.
//!
//! ## What was verified, and against what
//!
//! The layout, the value types, the alignment rules and the `ggml_type` codes
//! were checked line by line against these files of `ggml-org/ggml` at commit
//! `d9fe7bf`:
//!
//! * `docs/gguf.md`, the format description:
//!   <https://raw.githubusercontent.com/ggml-org/ggml/d9fe7bf/docs/gguf.md>
//! * `include/ggml.h`, the `ggml_type` enum:
//!   <https://raw.githubusercontent.com/ggml-org/ggml/d9fe7bf/include/ggml.h>
//! * `src/ggml-common.h`, the block structs (`QK8_0`, `QK5_0`, `QK_K`,
//!   `K_SCALE_SIZE` and the `static_assert` of every block size):
//!   <https://raw.githubusercontent.com/ggml-org/ggml/d9fe7bf/src/ggml-common.h>
//! * `src/gguf.cpp`, the reference reader, for what the prose leaves open
//!   (versions, duplicate keys, duplicate tensor names, empty keys, the
//!   alignment, at most four dimensions, the multiple of the block size in the
//!   first dimension):
//!   <https://raw.githubusercontent.com/ggml-org/ggml/d9fe7bf/src/gguf.cpp>
//!
//! The file is little endian: magic `GGUF`, a `u32` version that is 2 or 3,
//! the tensor count and the key count as `u64`, the key value pairs, the
//! tensor infos, zero padding up to a multiple of `general.alignment` (32 when
//! absent) if there is at least one tensor, then the tensor data. A string is
//! a `u64` length and that many UTF-8 bytes. A tensor offset is relative to
//! the start of the data and must be a multiple of the alignment. A big endian
//! file has no marker the spec defines, so it is rejected through its
//! version, which reads as a number with the low sixteen bits clear.
//!
//! ## Choices the format leaves open
//!
//! * Versions 2 and 3 are read, as in the reference reader, which refuses
//!   version 0, version 1 ("GGUFv1 is no longer supported"), anything above
//!   `GGUF_VERSION` (3, in `include/gguf.h`) and any version whose low sixteen
//!   bits are clear (a big endian file). The version history of the spec says
//!   of v2 "Most countable values (lengths, etc) were changed from `uint32` to
//!   `uint64`" and of v3 "Adds big-endian support": the layout read here, with
//!   `u64` counts and lengths, is the one of both, and `gguf.cpp` reads them
//!   with the same code. Version 1 and every other value are rejected.
//! * `general.alignment` must be a `u32`, as the reference reader requires
//!   ("key ... must be of type"), and a power of two, which is what it checks:
//!   `alignment == 0 || (alignment & (alignment - 1)) != 0` is an error. The
//!   spec adds "it must be a multiple of 8", so the value must be both, which
//!   means a power of two of at least 8. A zero, a 4 or a 24 is refused.
//! * A file without tensors gets no padding and its data section starts right
//!   after the key value pairs, as in the reference reader, which only seeks
//!   to the padded offset `if n_tensors > 0`.
//! * An empty metadata key is refused, as the reference reader does ("key ...
//!   is empty"). An empty tensor name is not, because the reference does not.
//! * Arrays may nest, up to a depth of 4. The spec says "Arrays can be
//!   nested", but the reference reader reads an array of arrays as "invalid
//!   GGUF type" (its `GGUF_TYPE_ARRAY` case in the element switch), so it is
//!   stricter; a depth limit keeps the walk of a hostile file bounded.
//! * A tensor has one to four dimensions. The spec says "currently at most
//!   4"; the reference reader refuses more.
//! * A tensor of a supported type is checked while parsing: its shape, its
//!   size and that it fits in the file. A tensor of any other type only needs
//!   an aligned offset inside the data and a shape whose element count fits
//!   in a `u64`, because its size is not known.
//! * Offsets are not required to be contiguous or free of overlap, which the
//!   reference reader enforces. Nothing here writes through them.
//! * The limits on key length (65535 bytes) and tensor name length (64 bytes)
//!   and the ASCII rule on keys are not enforced: a reader gains nothing from
//!   refusing a longer name.
//!
//! A large array, such as a tokenizer vocabulary, is walked once to find where
//! it ends and kept as a slice of the input; [`Value::strings`] iterates it.

use core::fmt;
use std::collections::HashMap;

/// The four bytes at the start of every GGUF file.
const MAGIC: [u8; 4] = *b"GGUF";
/// The oldest version this reader accepts: `u64` counts and lengths.
const MIN_VERSION: u32 = 2;
/// The newest version, `GGUF_VERSION` of the reference.
const MAX_VERSION: u32 = 3;
/// The alignment when `general.alignment` is absent.
const DEFAULT_ALIGNMENT: u64 = 32;
/// The key that overrides the alignment.
const ALIGNMENT_KEY: &str = "general.alignment";
/// The most dimensions a tensor may have.
const MAX_DIMS: u32 = 4;
/// How deep arrays may nest before the file is called absurd.
const MAX_ARRAY_DEPTH: u32 = 4;
/// The smallest key value pair: an empty key, the type and a one byte value.
const MIN_KV_BYTES: u64 = 8 + 4 + 1;
/// The smallest tensor info: an empty name, the dimension count, one dimension,
/// the type and the offset.
const MIN_TENSOR_INFO_BYTES: u64 = 8 + 4 + 8 + 4 + 8;
/// The most entries reserved up front from a count that the file claims.
const MAX_PREALLOCATED: u64 = 1024;
/// The smallest string: its length.
const MIN_STRING_BYTES: u64 = 8;
/// The smallest array: the element type and the count.
const MIN_ARRAY_BYTES: u64 = 4 + 8;

/// Why a file was refused, or why a lookup failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufError {
    /// The file does not start with `GGUF`.
    BadMagic,
    /// The version is not 2 or 3. A big endian file shows up here too.
    UnsupportedVersion(u32),
    /// The file ends before a field it announced.
    Truncated,
    /// `general.alignment` is not a power of two that is a multiple of 8.
    BadAlignment(u64),
    /// A size or an offset of the file does not fit in this platform's `usize`.
    TooLarge,
    /// A tensor's bytes do not lie inside the data section.
    TensorOutOfBounds { name: String },
    /// A tensor offset is not a multiple of the alignment.
    MisalignedTensor { name: String },
    /// The tensor has a ggml type this runtime does not read.
    UnsupportedTensorType { name: String, ggml_type: u32 },
    /// No metadata value has this key.
    MissingKey(String),
    /// The value exists but has another type.
    WrongType { key: String, expected: &'static str },
    /// The dimensions of a tensor are impossible for its type.
    BadTensorShape { name: String },
    /// Invalid UTF-8, an empty key, a bool that is not 0 or 1, an unknown
    /// value type or arrays nested absurdly deep.
    BadValue,
    /// Two metadata values share a key.
    DuplicateKey(String),
    /// Two tensors share a name.
    DuplicateTensor(String),
}

impl fmt::Display for GgufError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => formatter.write_str("not a GGUF file: the magic is not `GGUF`"),
            Self::UnsupportedVersion(version) => write!(
                formatter,
                "unsupported GGUF version {version}: only little endian versions {MIN_VERSION} and \
                 {MAX_VERSION} are read"
            ),
            Self::Truncated => formatter.write_str("the GGUF file ends before a field it announces"),
            Self::BadAlignment(alignment) => write!(
                formatter,
                "bad `{ALIGNMENT_KEY}` {alignment}: it must be a power of two and a multiple of 8"
            ),
            Self::TooLarge => formatter
                .write_str("a size or an offset of the GGUF file is too large for this platform"),
            Self::TensorOutOfBounds { name } => {
                write!(formatter, "tensor `{name}` does not fit in the data section")
            }
            Self::MisalignedTensor { name } => write!(
                formatter,
                "tensor `{name}` has an offset that is not a multiple of the alignment"
            ),
            Self::UnsupportedTensorType { name, ggml_type } => match ggml_type_name(*ggml_type) {
                Some(type_name) => write!(
                    formatter,
                    "tensor `{name}` has ggml type {type_name} ({ggml_type}), which is not supported"
                ),
                None => write!(
                    formatter,
                    "tensor `{name}` has unknown ggml type {ggml_type}, which is not supported"
                ),
            },
            Self::MissingKey(key) => write!(formatter, "missing metadata key `{key}`"),
            Self::WrongType { key, expected } => {
                write!(formatter, "metadata key `{key}` is not of type {expected}")
            }
            Self::BadTensorShape { name } => {
                write!(formatter, "tensor `{name}` has a shape its type cannot hold")
            }
            Self::BadValue => formatter.write_str(
                "invalid GGUF value: bad UTF-8, an empty key, a bool that is not 0 or 1, an \
                 unknown value type or arrays nested too deep",
            ),
            Self::DuplicateKey(key) => write!(formatter, "duplicate metadata key `{key}`"),
            Self::DuplicateTensor(name) => write!(formatter, "duplicate tensor name `{name}`"),
        }
    }
}

impl std::error::Error for GgufError {}

/// The type tag of a metadata value, as `gguf_metadata_value_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl ValueType {
    /// The type for a tag of the file, or `None` for a tag the spec lacks.
    pub fn from_u32(tag: u32) -> Option<Self> {
        Some(match tag {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => return None,
        })
    }

    /// The bytes one value takes, or `None` for the types of variable size.
    fn fixed_size(self) -> Option<u64> {
        match self {
            Self::U8 | Self::I8 | Self::Bool => Some(1),
            Self::U16 | Self::I16 => Some(2),
            Self::U32 | Self::I32 | Self::F32 => Some(4),
            Self::U64 | Self::I64 | Self::F64 => Some(8),
            Self::String | Self::Array => None,
        }
    }
}

/// One metadata value. Strings and arrays borrow the input.
#[derive(Debug, Clone, PartialEq)]
pub enum Value<'a> {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(&'a str),
    /// An array that was walked once and checked, and is kept as the bytes of
    /// its elements, without the element type and the count in front of them.
    Array {
        element: ValueType,
        count: u64,
        bytes: &'a [u8],
    },
    U64(u64),
    I64(i64),
    F64(f64),
}

impl<'a> Value<'a> {
    /// The strings of an array of strings, or `None` for any other value.
    pub fn strings(&self) -> Option<StrArray<'a>> {
        match self {
            Self::Array {
                element: ValueType::String,
                count,
                bytes,
            } => Some(StrArray {
                reader: Reader::new(bytes),
                left: *count,
            }),
            _ => None,
        }
    }
}

/// The iterator [`Value::strings`] returns. The bytes were checked when the
/// file was parsed, so it ends early rather than fail if they were not.
#[derive(Debug, Clone)]
pub struct StrArray<'a> {
    reader: Reader<'a>,
    left: u64,
}

impl<'a> Iterator for StrArray<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        if self.left == 0 {
            return None;
        }
        self.left -= 1;
        self.reader.string().ok()
    }
}

/// The tensor types this crate reads, with the codes of `enum ggml_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorKind {
    F32,
    F16,
    Q8_0,
    Q5_0,
    Q4K,
    Q6K,
}

impl TensorKind {
    /// The kind of a `ggml_type` code, or `None` when it is not supported.
    pub fn from_ggml_type(code: u32) -> Option<Self> {
        Some(match code {
            0 => Self::F32,
            1 => Self::F16,
            6 => Self::Q5_0,
            8 => Self::Q8_0,
            12 => Self::Q4K,
            14 => Self::Q6K,
            _ => return None,
        })
    }

    /// The `ggml_type` code.
    pub fn ggml_type(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q5_0 => 6,
            Self::Q8_0 => 8,
            Self::Q4K => 12,
            Self::Q6K => 14,
        }
    }

    /// The elements in one block: `QK8_0`, `QK5_0` and `QK_K`, or 1.
    pub fn block_elements(self) -> u64 {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q8_0 | Self::Q5_0 => 32,
            Self::Q4K | Self::Q6K => 256,
        }
    }

    /// The bytes of one block, as the `static_assert` of `ggml-common.h`.
    pub fn block_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q8_0 => 34,
            Self::Q5_0 => 22,
            Self::Q4K => 144,
            Self::Q6K => 210,
        }
    }
}

/// The name `ggml.h` gives to a `ggml_type` code, or `None` for a code the
/// enum does not define, including the codes that were removed.
pub fn ggml_type_name(code: u32) -> Option<&'static str> {
    Some(match code {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        34 => "TQ1_0",
        35 => "TQ2_0",
        39 => "MXFP4",
        40 => "NVFP4",
        41 => "Q1_0",
        42 => "Q2_0",
        43 => "F8_E4M3",
        _ => return None,
    })
}

/// One entry of the tensor directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo<'a> {
    pub name: &'a str,
    /// The dimensions in the order of the file: `dims[0]` is the fastest.
    pub dims: Vec<u64>,
    /// The `ggml_type` code as the file has it.
    pub ggml_type: u32,
    /// The offset from the start of the data section, as the file has it.
    pub offset: u64,
}

impl TensorInfo<'_> {
    /// The supported kind of this tensor, or `None`.
    pub fn kind(&self) -> Option<TensorKind> {
        TensorKind::from_ggml_type(self.ggml_type)
    }

    /// The number of elements, or `None` when the product overflows a `u64`.
    pub fn elements(&self) -> Option<u64> {
        self.dims
            .iter()
            .try_fold(1u64, |product, &dim| product.checked_mul(dim))
    }

    /// The bytes the tensor takes in the file.
    ///
    /// The first dimension must be a multiple of the block size of the type,
    /// and the size must fit in a `u64`. A type that is not supported has no
    /// known size and is an [`GgufError::UnsupportedTensorType`].
    pub fn byte_size(&self) -> Result<u64, GgufError> {
        let kind = self
            .kind()
            .ok_or_else(|| GgufError::UnsupportedTensorType {
                name: self.name.to_string(),
                ggml_type: self.ggml_type,
            })?;
        let bad_shape = || GgufError::BadTensorShape {
            name: self.name.to_string(),
        };
        let first = *self.dims.first().ok_or_else(bad_shape)?;
        if first % kind.block_elements() != 0 {
            return Err(bad_shape());
        }
        let blocks = self.elements().ok_or_else(bad_shape)? / kind.block_elements();
        blocks.checked_mul(kind.block_bytes()).ok_or_else(bad_shape)
    }
}

/// A parsed GGUF file that borrows its bytes.
#[derive(Debug)]
pub struct Gguf<'a> {
    version: u32,
    alignment: u64,
    metadata: Vec<(&'a str, Value<'a>)>,
    metadata_index: HashMap<&'a str, usize>,
    tensors: Vec<TensorInfo<'a>>,
    tensor_index: HashMap<&'a str, usize>,
    data_offset: usize,
    data: &'a [u8],
}

impl<'a> Gguf<'a> {
    /// Parse a whole file. Nothing is copied but the directory.
    pub fn parse(bytes: &'a [u8]) -> Result<Gguf<'a>, GgufError> {
        let mut reader = Reader::new(bytes);
        if reader.take(MAGIC.len())? != MAGIC {
            return Err(GgufError::BadMagic);
        }
        let version = reader.u32()?;
        if !(MIN_VERSION..=MAX_VERSION).contains(&version) {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let tensor_count = reader.u64()?;
        let kv_count = reader.u64()?;
        if kv_count > reader.remaining() as u64 / MIN_KV_BYTES {
            return Err(GgufError::Truncated);
        }

        let mut metadata = Vec::with_capacity(preallocated(kv_count));
        let mut metadata_index = HashMap::with_capacity(preallocated(kv_count));
        for _ in 0..kv_count {
            let key = reader.string()?;
            if key.is_empty() {
                return Err(GgufError::BadValue);
            }
            let value_type = reader.value_type()?;
            let value = reader.value(value_type, 0)?;
            if metadata_index.insert(key, metadata.len()).is_some() {
                return Err(GgufError::DuplicateKey(key.to_string()));
            }
            metadata.push((key, value));
        }

        let alignment = match metadata_index.get(ALIGNMENT_KEY) {
            None => DEFAULT_ALIGNMENT,
            Some(&index) => match metadata[index].1 {
                Value::U32(alignment) => u64::from(alignment),
                _ => {
                    return Err(GgufError::WrongType {
                        key: ALIGNMENT_KEY.to_string(),
                        expected: "u32",
                    })
                }
            },
        };
        if !alignment.is_power_of_two() || alignment % 8 != 0 {
            return Err(GgufError::BadAlignment(alignment));
        }

        if tensor_count > reader.remaining() as u64 / MIN_TENSOR_INFO_BYTES {
            return Err(GgufError::Truncated);
        }
        let mut tensors = Vec::with_capacity(preallocated(tensor_count));
        let mut tensor_index = HashMap::with_capacity(preallocated(tensor_count));
        for _ in 0..tensor_count {
            let name = reader.string()?;
            let bad_shape = || GgufError::BadTensorShape {
                name: name.to_string(),
            };
            let n_dims = reader.u32()?;
            if n_dims == 0 || n_dims > MAX_DIMS {
                return Err(bad_shape());
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(reader.u64()?);
            }
            let ggml_type = reader.u32()?;
            let offset = reader.u64()?;
            if tensor_index.insert(name, tensors.len()).is_some() {
                return Err(GgufError::DuplicateTensor(name.to_string()));
            }
            tensors.push(TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
            });
        }

        let mut data_offset = reader.position();
        if !tensors.is_empty() {
            data_offset = padded_offset(data_offset, alignment)?;
            if data_offset > bytes.len() {
                return Err(GgufError::Truncated);
            }
        }
        let data = &bytes[data_offset..];

        for tensor in &tensors {
            let out_of_bounds = || GgufError::TensorOutOfBounds {
                name: tensor.name.to_string(),
            };
            if tensor.elements().is_none() {
                return Err(GgufError::BadTensorShape {
                    name: tensor.name.to_string(),
                });
            }
            if tensor.offset % alignment != 0 {
                return Err(GgufError::MisalignedTensor {
                    name: tensor.name.to_string(),
                });
            }
            if tensor.offset > data.len() as u64 {
                return Err(out_of_bounds());
            }
            if tensor.kind().is_some() {
                let end = tensor
                    .offset
                    .checked_add(tensor.byte_size()?)
                    .ok_or_else(out_of_bounds)?;
                if end > data.len() as u64 {
                    return Err(out_of_bounds());
                }
            }
        }

        Ok(Gguf {
            version,
            alignment,
            metadata,
            metadata_index,
            tensors,
            tensor_index,
            data_offset,
            data,
        })
    }

    /// The version of the file, 2 or 3.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// The alignment of the data section and of every tensor offset.
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Where the data section starts, in bytes from the start of the file.
    pub fn data_offset(&self) -> usize {
        self.data_offset
    }

    /// The metadata pairs in the order of the file.
    pub fn metadata(&self) -> &[(&'a str, Value<'a>)] {
        &self.metadata
    }

    /// The value of a metadata key.
    pub fn get(&self, key: &str) -> Option<&Value<'a>> {
        self.metadata_index
            .get(key)
            .map(|&index| &self.metadata[index].1)
    }

    fn typed<T>(
        &self,
        key: &str,
        expected: &'static str,
        pick: impl FnOnce(&Value<'a>) -> Option<T>,
    ) -> Result<T, GgufError> {
        let value = self
            .get(key)
            .ok_or_else(|| GgufError::MissingKey(key.to_string()))?;
        pick(value).ok_or_else(|| GgufError::WrongType {
            key: key.to_string(),
            expected,
        })
    }

    /// A `uint32` value.
    pub fn get_u32(&self, key: &str) -> Result<u32, GgufError> {
        self.typed(key, "u32", |value| match value {
            Value::U32(v) => Some(*v),
            _ => None,
        })
    }

    /// A `uint64` value.
    pub fn get_u64(&self, key: &str) -> Result<u64, GgufError> {
        self.typed(key, "u64", |value| match value {
            Value::U64(v) => Some(*v),
            _ => None,
        })
    }

    /// A `float32` value.
    pub fn get_f32(&self, key: &str) -> Result<f32, GgufError> {
        self.typed(key, "f32", |value| match value {
            Value::F32(v) => Some(*v),
            _ => None,
        })
    }

    /// A string value.
    pub fn get_str(&self, key: &str) -> Result<&'a str, GgufError> {
        self.typed(key, "string", |value| match value {
            Value::Str(v) => Some(*v),
            _ => None,
        })
    }

    /// A bool value.
    pub fn get_bool(&self, key: &str) -> Result<bool, GgufError> {
        self.typed(key, "bool", |value| match value {
            Value::Bool(v) => Some(*v),
            _ => None,
        })
    }

    /// The tensor directory in the order of the file.
    pub fn tensors(&self) -> &[TensorInfo<'a>] {
        &self.tensors
    }

    /// The tensor of this name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo<'a>> {
        self.tensor_index
            .get(name)
            .map(|&index| &self.tensors[index])
    }

    /// The exact bytes of a tensor, borrowed from the input.
    pub fn tensor_data(&self, tensor: &TensorInfo<'_>) -> Result<&'a [u8], GgufError> {
        let size = tensor.byte_size()?;
        let out_of_bounds = || GgufError::TensorOutOfBounds {
            name: tensor.name.to_string(),
        };
        let end = tensor.offset.checked_add(size).ok_or_else(out_of_bounds)?;
        let start = usize::try_from(tensor.offset).map_err(|_| out_of_bounds())?;
        let end = usize::try_from(end).map_err(|_| out_of_bounds())?;
        self.data.get(start..end).ok_or_else(out_of_bounds)
    }

    /// The kind of a tensor, or the error that names its unsupported type.
    pub fn require_supported(&self, tensor: &TensorInfo<'_>) -> Result<TensorKind, GgufError> {
        tensor
            .kind()
            .ok_or_else(|| GgufError::UnsupportedTensorType {
                name: tensor.name.to_string(),
                ggml_type: tensor.ggml_type,
            })
    }
}

/// How many entries to reserve for a count taken from the file.
fn preallocated(count: u64) -> usize {
    count.min(MAX_PREALLOCATED) as usize
}

/// The smallest multiple of `alignment` that is at least `position`.
fn padded_offset(position: usize, alignment: u64) -> Result<usize, GgufError> {
    let position = position as u64;
    let padded = match position % alignment {
        0 => Some(position),
        rest => position.checked_add(alignment - rest),
    };
    padded
        .and_then(|padded| usize::try_from(padded).ok())
        .ok_or(GgufError::TooLarge)
}

/// A cursor over the input whose every read is bounds checked.
#[derive(Debug, Clone)]
struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn position(&self) -> usize {
        self.position
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], GgufError> {
        let end = self
            .position
            .checked_add(count)
            .ok_or(GgufError::Truncated)?;
        let slice = self
            .bytes
            .get(self.position..end)
            .ok_or(GgufError::Truncated)?;
        self.position = end;
        Ok(slice)
    }

    fn take_u64(&mut self, count: u64) -> Result<&'a [u8], GgufError> {
        // A count beyond `usize` is beyond the input too, whatever its size.
        self.take(usize::try_from(count).unwrap_or(usize::MAX))
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], GgufError> {
        <[u8; N]>::try_from(self.take(N)?).map_err(|_| GgufError::Truncated)
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn string(&mut self) -> Result<&'a str, GgufError> {
        let length = self.u64()?;
        let bytes = self.take_u64(length)?;
        core::str::from_utf8(bytes).map_err(|_| GgufError::BadValue)
    }

    fn value_type(&mut self) -> Result<ValueType, GgufError> {
        ValueType::from_u32(self.u32()?).ok_or(GgufError::BadValue)
    }

    fn value(&mut self, value_type: ValueType, depth: u32) -> Result<Value<'a>, GgufError> {
        Ok(match value_type {
            ValueType::U8 => Value::U8(u8::from_le_bytes(self.array()?)),
            ValueType::I8 => Value::I8(i8::from_le_bytes(self.array()?)),
            ValueType::U16 => Value::U16(u16::from_le_bytes(self.array()?)),
            ValueType::I16 => Value::I16(i16::from_le_bytes(self.array()?)),
            ValueType::U32 => Value::U32(u32::from_le_bytes(self.array()?)),
            ValueType::I32 => Value::I32(i32::from_le_bytes(self.array()?)),
            ValueType::F32 => Value::F32(f32::from_le_bytes(self.array()?)),
            ValueType::Bool => match self.array::<1>()?[0] {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                _ => return Err(GgufError::BadValue),
            },
            ValueType::String => Value::Str(self.string()?),
            ValueType::Array => self.array_value(depth)?,
            ValueType::U64 => Value::U64(u64::from_le_bytes(self.array()?)),
            ValueType::I64 => Value::I64(i64::from_le_bytes(self.array()?)),
            ValueType::F64 => Value::F64(f64::from_le_bytes(self.array()?)),
        })
    }

    /// Walk an array once, checking every element, and keep its bytes.
    fn array_value(&mut self, depth: u32) -> Result<Value<'a>, GgufError> {
        if depth >= MAX_ARRAY_DEPTH {
            return Err(GgufError::BadValue);
        }
        let element = self.value_type()?;
        let count = self.u64()?;
        let start = self.position;
        match element.fixed_size() {
            Some(size) => {
                let total = count.checked_mul(size).ok_or(GgufError::Truncated)?;
                let bytes = self.take_u64(total)?;
                if element == ValueType::Bool && bytes.iter().any(|&byte| byte > 1) {
                    return Err(GgufError::BadValue);
                }
            }
            None => {
                let minimum = match element {
                    ValueType::String => MIN_STRING_BYTES,
                    _ => MIN_ARRAY_BYTES,
                };
                if count > self.remaining() as u64 / minimum {
                    return Err(GgufError::Truncated);
                }
                for _ in 0..count {
                    match element {
                        ValueType::String => {
                            self.string()?;
                        }
                        _ => {
                            self.array_value(depth + 1)?;
                        }
                    }
                }
            }
        }
        Ok(Value::Array {
            element,
            count,
            bytes: &self.bytes[start..self.position],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Little endian writer for the files of the tests.
    #[derive(Default)]
    struct Writer {
        bytes: Vec<u8>,
    }

    impl Writer {
        fn u8(&mut self, value: u8) -> &mut Self {
            self.bytes.push(value);
            self
        }

        fn u32(&mut self, value: u32) -> &mut Self {
            self.bytes.extend_from_slice(&value.to_le_bytes());
            self
        }

        fn u64(&mut self, value: u64) -> &mut Self {
            self.bytes.extend_from_slice(&value.to_le_bytes());
            self
        }

        fn raw(&mut self, value: &[u8]) -> &mut Self {
            self.bytes.extend_from_slice(value);
            self
        }

        fn string(&mut self, value: &str) -> &mut Self {
            self.u64(value.len() as u64).raw(value.as_bytes())
        }
    }

    fn kv(key: &str, value_type: u32, value: &[u8]) -> Vec<u8> {
        let mut w = Writer::default();
        w.string(key).u32(value_type).raw(value);
        w.bytes
    }

    fn kv_u32(key: &str, value: u32) -> Vec<u8> {
        kv(key, 4, &value.to_le_bytes())
    }

    fn kv_str(key: &str, value: &str) -> Vec<u8> {
        let mut v = Writer::default();
        v.string(value);
        kv(key, 8, &v.bytes)
    }

    fn tensor_info(name: &str, dims: &[u64], ggml_type: u32, offset: u64) -> Vec<u8> {
        let mut w = Writer::default();
        w.string(name).u32(dims.len() as u32);
        for &dim in dims {
            w.u64(dim);
        }
        w.u32(ggml_type).u64(offset);
        w.bytes
    }

    /// Header, pairs, tensor infos, padding to `alignment`, then the data.
    fn file(
        version: u32,
        kvs: &[Vec<u8>],
        tensors: &[Vec<u8>],
        alignment: usize,
        data: &[u8],
    ) -> Vec<u8> {
        let mut w = Writer::default();
        w.raw(b"GGUF")
            .u32(version)
            .u64(tensors.len() as u64)
            .u64(kvs.len() as u64);
        for pair in kvs {
            w.raw(pair);
        }
        for info in tensors {
            w.raw(info);
        }
        while w.bytes.len() % alignment != 0 {
            w.u8(0);
        }
        w.raw(data);
        w.bytes
    }

    fn pattern(length: usize) -> Vec<u8> {
        (0..length).map(|i| (i * 7 + 3) as u8).collect()
    }

    /// Every value type, an array of strings, two tensors of two types.
    fn sample_kvs() -> Vec<Vec<u8>> {
        let mut tokens = Writer::default();
        tokens.u32(8).u64(3);
        tokens.string("<s>").string("hello").string("caf\u{e9}");
        let mut scores = Writer::default();
        scores.u32(6).u64(3);
        for score in [0.0f32, -1.5, 2.25] {
            scores.raw(&score.to_le_bytes());
        }
        let mut nested = Writer::default();
        nested.u32(9).u64(2);
        nested.u32(4).u64(1).u32(7);
        nested.u32(4).u64(0);
        vec![
            kv_str("general.architecture", "llama"),
            kv("t.u8", 0, &[200]),
            kv("t.i8", 1, &(-5i8).to_le_bytes()),
            kv("t.u16", 2, &60000u16.to_le_bytes()),
            kv("t.i16", 3, &(-30000i16).to_le_bytes()),
            kv_u32("llama.block_count", 2),
            kv("t.i32", 5, &(-7i32).to_le_bytes()),
            kv("llama.rope.freq_base", 6, &10000.0f32.to_le_bytes()),
            kv("t.flag", 7, &[1]),
            kv("t.off", 7, &[0]),
            kv("tokenizer.ggml.tokens", 9, &tokens.bytes),
            kv("tokenizer.ggml.scores", 9, &scores.bytes),
            kv("t.nested", 9, &nested.bytes),
            kv("t.u64", 10, &u64::MAX.to_le_bytes()),
            kv("t.i64", 11, &i64::MIN.to_le_bytes()),
            kv("t.f64", 12, &0.5f64.to_le_bytes()),
        ]
    }

    /// F32 4x3 (48 bytes) at 0 and Q8_0 64x2 (136 bytes) at 64.
    fn sample_with(alignment: usize, extra_kv: Option<Vec<u8>>) -> Vec<u8> {
        let mut kvs = sample_kvs();
        kvs.extend(extra_kv);
        let second = 64u64;
        let tensors = [
            tensor_info("token_embd.weight", &[4, 3], 0, 0),
            tensor_info("blk.0.attn.weight", &[64, 2], 8, second),
        ];
        file(3, &kvs, &tensors, alignment, &pattern(64 + 136))
    }

    fn sample() -> Vec<u8> {
        sample_with(32, None)
    }

    #[test]
    fn parses_every_field_of_a_valid_file() {
        let bytes = sample();
        let gguf = Gguf::parse(&bytes).unwrap();
        assert_eq!(gguf.version(), 3);
        assert_eq!(gguf.alignment(), 32);
        assert_eq!(gguf.metadata().len(), 16);
        assert_eq!(gguf.get_str("general.architecture").unwrap(), "llama");
        assert_eq!(gguf.get_u32("llama.block_count").unwrap(), 2);
        assert_eq!(gguf.get_f32("llama.rope.freq_base").unwrap(), 10000.0);
        assert!(gguf.get_bool("t.flag").unwrap());
        assert!(!gguf.get_bool("t.off").unwrap());
        assert_eq!(gguf.get_u64("t.u64").unwrap(), u64::MAX);
        assert_eq!(gguf.get("t.u8"), Some(&Value::U8(200)));
        assert_eq!(gguf.get("t.i8"), Some(&Value::I8(-5)));
        assert_eq!(gguf.get("t.u16"), Some(&Value::U16(60000)));
        assert_eq!(gguf.get("t.i16"), Some(&Value::I16(-30000)));
        assert_eq!(gguf.get("t.i32"), Some(&Value::I32(-7)));
        assert_eq!(gguf.get("t.i64"), Some(&Value::I64(i64::MIN)));
        assert_eq!(gguf.get("t.f64"), Some(&Value::F64(0.5)));
        assert_eq!(gguf.get("absent"), None);

        let tokens = gguf.get("tokenizer.ggml.tokens").unwrap();
        match tokens {
            Value::Array { element, count, .. } => {
                assert_eq!(*element, ValueType::String);
                assert_eq!(*count, 3);
            }
            other => panic!("not an array: {other:?}"),
        }
        let strings: Vec<&str> = tokens.strings().unwrap().collect();
        assert_eq!(strings, ["<s>", "hello", "caf\u{e9}"]);
        match gguf.get("tokenizer.ggml.scores").unwrap() {
            Value::Array {
                element,
                count,
                bytes,
            } => {
                assert_eq!(*element, ValueType::F32);
                assert_eq!(*count, 3);
                let mut expected = Vec::new();
                for score in [0.0f32, -1.5, 2.25] {
                    expected.extend_from_slice(&score.to_le_bytes());
                }
                assert_eq!(*bytes, expected);
            }
            other => panic!("not an array: {other:?}"),
        }
        assert!(matches!(
            gguf.get("t.nested"),
            Some(Value::Array {
                element: ValueType::Array,
                count: 2,
                ..
            })
        ));
        assert!(gguf.get("t.u8").unwrap().strings().is_none());

        assert_eq!(gguf.tensors().len(), 2);
        let embd = gguf.tensor("token_embd.weight").unwrap();
        assert_eq!(embd.dims, [4, 3]);
        assert_eq!((embd.ggml_type, embd.offset), (0, 0));
        assert_eq!(embd.kind(), Some(TensorKind::F32));
        assert_eq!(embd.elements(), Some(12));
        assert_eq!(embd.byte_size().unwrap(), 48);
        let attn = gguf.tensor("blk.0.attn.weight").unwrap();
        assert_eq!(attn.dims, [64, 2]);
        assert_eq!((attn.ggml_type, attn.offset), (8, 64));
        assert_eq!(attn.kind(), Some(TensorKind::Q8_0));
        assert_eq!(attn.elements(), Some(128));
        assert_eq!(attn.byte_size().unwrap(), 4 * 34);
        assert!(gguf.tensor("nope").is_none());
        assert_eq!(gguf.require_supported(attn).unwrap(), TensorKind::Q8_0);
        assert_eq!(gguf.data_offset() % 32, 0);
    }

    #[test]
    fn tensor_data_returns_the_exact_bytes() {
        let bytes = sample();
        let gguf = Gguf::parse(&bytes).unwrap();
        let data = pattern(64 + 136);
        let embd = gguf.tensor("token_embd.weight").unwrap();
        assert_eq!(gguf.tensor_data(embd).unwrap(), &data[0..48]);
        let attn = gguf.tensor("blk.0.attn.weight").unwrap();
        assert_eq!(gguf.tensor_data(attn).unwrap(), &data[64..200]);
        let start = gguf.data_offset();
        assert_eq!(
            gguf.tensor_data(attn).unwrap().as_ptr(),
            bytes[start + 64..].as_ptr()
        );
    }

    #[test]
    fn block_sizes_match_ggml_common() {
        let size = |kind: TensorKind| (kind.block_elements(), kind.block_bytes());
        assert_eq!(size(TensorKind::F32), (1, 4));
        assert_eq!(size(TensorKind::F16), (1, 2));
        assert_eq!(size(TensorKind::Q8_0), (32, 34));
        assert_eq!(size(TensorKind::Q5_0), (32, 22));
        assert_eq!(size(TensorKind::Q4K), (256, 144));
        assert_eq!(size(TensorKind::Q6K), (256, 210));
        for kind in [
            TensorKind::F32,
            TensorKind::F16,
            TensorKind::Q8_0,
            TensorKind::Q5_0,
            TensorKind::Q4K,
            TensorKind::Q6K,
        ] {
            assert_eq!(TensorKind::from_ggml_type(kind.ggml_type()), Some(kind));
        }
    }

    #[test]
    fn every_supported_type_has_its_size() {
        let dims = |first: u64| TensorInfo {
            name: "t",
            dims: vec![first, 3],
            ggml_type: 0,
            offset: 0,
        };
        let sized = |code: u32, first: u64| {
            let mut info = dims(first);
            info.ggml_type = code;
            info.byte_size()
        };
        assert_eq!(sized(0, 5), Ok(60));
        assert_eq!(sized(1, 5), Ok(30));
        assert_eq!(sized(8, 32), Ok(3 * 34));
        assert_eq!(sized(6, 64), Ok(6 * 22));
        assert_eq!(sized(12, 256), Ok(3 * 144));
        assert_eq!(sized(14, 512), Ok(6 * 210));
    }

    #[test]
    fn custom_alignment_of_64() {
        let alignment = kv_u32("general.alignment", 64);
        let kvs = [alignment];
        let tensors = [tensor_info("a", &[32], 8, 0), tensor_info("b", &[4], 0, 64)];
        let data = pattern(64 + 16);
        let bytes = file(3, &kvs, &tensors, 64, &data);
        let gguf = Gguf::parse(&bytes).unwrap();
        assert_eq!(gguf.alignment(), 64);
        assert_eq!(gguf.data_offset() % 64, 0);
        assert_eq!(
            gguf.tensor_data(gguf.tensor("a").unwrap()).unwrap(),
            &data[..34]
        );
        assert_eq!(
            gguf.tensor_data(gguf.tensor("b").unwrap()).unwrap(),
            &data[64..80]
        );
        assert_eq!(gguf.get_u32("general.alignment").unwrap(), 64);

        // Offset 32 is aligned for 32 but not for 64.
        let tensors = [tensor_info("a", &[32], 8, 32)];
        let bytes = file(3, &kvs, &tensors, 64, &pattern(96));
        assert_eq!(
            Gguf::parse(&bytes).unwrap_err(),
            GgufError::MisalignedTensor { name: "a".into() }
        );
    }

    #[test]
    fn bad_alignments() {
        for (alignment, expected) in [
            (0, 0),
            (12, 12),
            (4, 4),
            (33, 33),
            (24, 24),
            (96, 96),
            (1, 1),
        ] {
            let kvs = [kv_u32("general.alignment", alignment)];
            let bytes = file(3, &kvs, &[], 8, &[]);
            assert_eq!(
                Gguf::parse(&bytes).unwrap_err(),
                GgufError::BadAlignment(expected)
            );
        }
        let kvs = [kv("general.alignment", 10, &32u64.to_le_bytes())];
        let bytes = file(3, &kvs, &[], 8, &[]);
        assert_eq!(
            Gguf::parse(&bytes).unwrap_err(),
            GgufError::WrongType {
                key: "general.alignment".into(),
                expected: "u32"
            }
        );
        // A huge alignment pushes the data past the end of the file.
        let kvs = [kv_u32("general.alignment", 1 << 31)];
        let tensors = [tensor_info("a", &[4], 0, 0)];
        let bytes = file(3, &kvs, &tensors, 8, &[0; 16]);
        assert_eq!(Gguf::parse(&bytes).unwrap_err(), GgufError::Truncated);
    }

    #[test]
    fn powers_of_two_that_are_multiples_of_8_are_alignments() {
        for alignment in [8u32, 16, 32, 64, 4096] {
            let kvs = [kv_u32("general.alignment", alignment)];
            let tensors = [tensor_info("a", &[4], 0, 0)];
            let bytes = file(3, &kvs, &tensors, alignment as usize, &pattern(16));
            let gguf = Gguf::parse(&bytes).unwrap();
            assert_eq!(gguf.alignment(), u64::from(alignment));
            assert_eq!(gguf.data_offset() % alignment as usize, 0);
        }
    }

    #[test]
    fn sizes_beyond_the_platform_are_too_large() {
        assert_eq!(padded_offset(41, 32), Ok(64));
        assert_eq!(padded_offset(64, 32), Ok(64));
        assert_eq!(padded_offset(usize::MAX - 1, 8), Err(GgufError::TooLarge));
        assert_eq!(padded_offset(usize::MAX, 1 << 31), Err(GgufError::TooLarge));
        let shown = GgufError::TooLarge.to_string();
        assert!(shown.contains("too large"), "{shown}");
        assert_ne!(GgufError::TooLarge, GgufError::Truncated);
        assert_ne!(shown, GgufError::Truncated.to_string());
    }

    #[test]
    fn bad_magic() {
        let mut bytes = sample();
        bytes[0] = b'g';
        assert_eq!(Gguf::parse(&bytes).unwrap_err(), GgufError::BadMagic);
        assert_eq!(
            Gguf::parse(b"GGML\x03\0\0\0").unwrap_err(),
            GgufError::BadMagic
        );
        assert_eq!(Gguf::parse(b"GG").unwrap_err(), GgufError::Truncated);
        assert_eq!(Gguf::parse(b"").unwrap_err(), GgufError::Truncated);
    }

    #[test]
    fn versions_2_and_3_are_read() {
        for version in [2u32, 3] {
            let kvs = [kv_u32("a", 7), kv_str("b", "text")];
            let tensors = [tensor_info("t", &[4], 0, 0)];
            let bytes = file(version, &kvs, &tensors, 32, &pattern(16));
            let gguf = Gguf::parse(&bytes).unwrap();
            assert_eq!(gguf.version(), version);
            assert_eq!(gguf.get_u32("a").unwrap(), 7);
            assert_eq!(gguf.get_str("b").unwrap(), "text");
            assert_eq!(
                gguf.tensor_data(gguf.tensor("t").unwrap()).unwrap(),
                &pattern(16)[..]
            );
        }
    }

    #[test]
    fn other_versions_and_big_endian() {
        for version in [0u32, 1, 4, 5, 0x0200_0000, 0x0300_0000, u32::MAX] {
            let bytes = file(version, &[], &[], 8, &[]);
            assert_eq!(
                Gguf::parse(&bytes).unwrap_err(),
                GgufError::UnsupportedVersion(version)
            );
        }
        // A big endian writer puts 3 as 00 00 00 03, which reads as 0x03000000.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_be_bytes());
        bytes.extend_from_slice(&[0; 16]);
        assert_eq!(
            Gguf::parse(&bytes).unwrap_err(),
            GgufError::UnsupportedVersion(0x0300_0000)
        );
    }

    #[test]
    fn every_prefix_of_a_valid_file_is_an_error() {
        for bytes in [
            sample(),
            sample_with(64, Some(kv_u32("general.alignment", 64))),
        ] {
            assert!(Gguf::parse(&bytes).is_ok());
            for length in 0..bytes.len() {
                assert!(
                    Gguf::parse(&bytes[..length]).is_err(),
                    "prefix of {length} bytes parsed"
                );
            }
        }
    }

    fn exercise(gguf: &Gguf<'_>) {
        for (key, value) in gguf.metadata() {
            assert!(gguf.get(key).is_some());
            if let Some(strings) = value.strings() {
                let _ = strings.count();
            }
        }
        for tensor in gguf.tensors() {
            let _ = gguf.tensor_data(tensor);
            let _ = gguf.require_supported(tensor);
            let _ = tensor.elements();
        }
    }

    #[test]
    fn single_byte_mutations_never_panic() {
        let bytes = sample();
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let mut accepted = 0;
        for _ in 0..6000 {
            let mut mutated = bytes.clone();
            let position = next() % mutated.len();
            mutated[position] = next() as u8;
            if let Ok(gguf) = Gguf::parse(&mutated) {
                accepted += 1;
                exercise(&gguf);
            }
        }
        // Most mutations land in data or strings and still parse.
        assert!(accepted > 0);
        // Two byte mutations aimed at the header and the counts.
        for _ in 0..6000 {
            let mut mutated = bytes.clone();
            for _ in 0..2 {
                let position = next() % 64;
                mutated[position] = next() as u8;
            }
            if let Ok(gguf) = Gguf::parse(&mutated) {
                exercise(&gguf);
            }
        }
    }

    #[test]
    fn offset_beyond_the_end() {
        let tensors = [tensor_info("a", &[8], 0, 64)];
        let bytes = file(3, &[], &tensors, 32, &pattern(32));
        assert_eq!(
            Gguf::parse(&bytes).unwrap_err(),
            GgufError::TensorOutOfBounds { name: "a".into() }
        );
        // The offset is inside the data but the tensor is not.
        let tensors = [tensor_info("a", &[16], 0, 32)];
        let bytes = file(3, &[], &tensors, 32, &pattern(64));
        assert_eq!(
            Gguf::parse(&bytes).unwrap_err(),
            GgufError::TensorOutOfBounds { name: "a".into() }
        );
        // An offset that overflows with the size.
        let tensors = [tensor_info("a", &[8], 0, u64::MAX - 7)];
        let bytes = file(3, &[], &tensors, 8, &[0; 8]);
        assert!(Gguf::parse(&bytes).is_err());
        // The same for a type this crate does not read.
        let tensors = [tensor_info("a", &[32], 2, 64)];
        let bytes = file(3, &[], &tensors, 32, &pattern(32));
        assert_eq!(
            Gguf::parse(&bytes).unwrap_err(),
            GgufError::TensorOutOfBounds { name: "a".into() }
        );
    }

    #[test]
    fn tensor_data_refuses_a_foreign_tensor_info() {
        let bytes = sample();
        let gguf = Gguf::parse(&bytes).unwrap();
        let foreign = TensorInfo {
            name: "x",
            dims: vec![1 << 20],
            ggml_type: 0,
            offset: 0,
        };
        assert_eq!(
            gguf.tensor_data(&foreign).unwrap_err(),
            GgufError::TensorOutOfBounds { name: "x".into() }
        );
        let wrapping = TensorInfo {
            offset: u64::MAX,
            dims: vec![4],
            ..foreign
        };
        assert_eq!(
            gguf.tensor_data(&wrapping).unwrap_err(),
            GgufError::TensorOutOfBounds { name: "x".into() }
        );
    }

    #[test]
    fn unaligned_tensor_offset() {
        let tensors = [tensor_info("a", &[4], 0, 0), tensor_info("b", &[4], 0, 16)];
        let bytes = file(3, &[], &tensors, 32, &pattern(64));
        assert_eq!(
            Gguf::parse(&bytes).unwrap_err(),
            GgufError::MisalignedTensor { name: "b".into() }
        );
    }

    #[test]
    fn dims_whose_product_overflows() {
        let tensors = [tensor_info("a", &[1 << 40, 1 << 40], 0, 0)];
        let bytes = file(3, &[], &tensors, 32, &pattern(32));
        assert_eq!(
            Gguf::parse(&bytes).unwrap_err(),
            GgufError::BadTensorShape { name: "a".into() }
        );
        // The same for a type this crate does not read.
        let tensors = [tensor_info("a", &[1 << 40, 1 << 40], 2, 0)];
        let bytes = file(3, &[], &tensors, 32, &pattern(32));
        assert_eq!(
            Gguf::parse(&bytes).unwrap_err(),
            GgufError::BadTensorShape { name: "a".into() }
        );
        // The product fits but the bytes do not.
        let tensors = [tensor_info("a", &[1 << 31, 1 << 31, 3], 0, 0)];
        let bytes = file(3, &[], &tensors, 32, &pattern(32));
        assert_eq!(
            Gguf::parse(&bytes).unwrap_err(),
            GgufError::BadTensorShape { name: "a".into() }
        );
    }

    #[test]
    fn shapes_that_are_not_a_whole_number_of_blocks() {
        for (code, first) in [(8u64, 33u64), (6, 16), (12, 255), (14, 300)] {
            let tensors = [tensor_info("a", &[first, 2], code as u32, 0)];
            let bytes = file(3, &[], &tensors, 32, &pattern(4096));
            assert_eq!(
                Gguf::parse(&bytes).unwrap_err(),
                GgufError::BadTensorShape { name: "a".into() },
                "type {code} first dimension {first}"
            );
        }
        for dims in [vec![], vec![1, 1, 1, 1, 1]] {
            let tensors = [tensor_info("a", &dims, 0, 0)];
            let bytes = file(3, &[], &tensors, 32, &pattern(64));
            assert_eq!(
                Gguf::parse(&bytes).unwrap_err(),
                GgufError::BadTensorShape { name: "a".into() }
            );
        }
    }

    #[test]
    fn huge_counts_with_a_tiny_buffer() {
        for (tensor_count, kv_count) in [(u64::MAX, 0), (0, u64::MAX), (1 << 40, 1 << 40), (3, 0)] {
            let mut w = Writer::default();
            w.raw(b"GGUF")
                .u32(3)
                .u64(tensor_count)
                .u64(kv_count)
                .raw(&[0; 16]);
            assert_eq!(Gguf::parse(&w.bytes).unwrap_err(), GgufError::Truncated);
        }
        // A string or an array that claims more bytes than there are.
        let kvs = [kv("k", 9, &{
            let mut w = Writer::default();
            w.u32(8).u64(u64::MAX);
            w.bytes
        })];
        let bytes = file(3, &kvs, &[], 8, &[]);
        assert_eq!(Gguf::parse(&bytes).unwrap_err(), GgufError::Truncated);
        let kvs = [kv("k", 9, &{
            let mut w = Writer::default();
            w.u32(4).u64(u64::MAX / 2);
            w.bytes
        })];
        let bytes = file(3, &kvs, &[], 8, &[]);
        assert_eq!(Gguf::parse(&bytes).unwrap_err(), GgufError::Truncated);
        let kvs = [kv("k", 8, &u64::MAX.to_le_bytes())];
        let bytes = file(3, &kvs, &[], 8, &[]);
        assert_eq!(Gguf::parse(&bytes).unwrap_err(), GgufError::Truncated);
        let mut w = Writer::default();
        w.raw(b"GGUF").u32(3).u64(0).u64(1).u64(u64::MAX);
        assert_eq!(Gguf::parse(&w.bytes).unwrap_err(), GgufError::Truncated);
    }

    #[test]
    fn an_empty_key_is_refused() {
        let bytes = file(3, &[kv_u32("", 1)], &[], 8, &[]);
        assert_eq!(Gguf::parse(&bytes).unwrap_err(), GgufError::BadValue);
        let bytes = file(3, &[kv_u32("a", 1), kv_u32("", 1)], &[], 8, &[]);
        assert_eq!(Gguf::parse(&bytes).unwrap_err(), GgufError::BadValue);
    }

    #[test]
    fn hostile_counts_that_fit_the_file_reserve_little() {
        // Millions of pairs the file has room for: the first one is refused
        // (an empty key) long before the vectors would have grown.
        let pairs = 2_000_000u64;
        let mut w = Writer::default();
        w.raw(b"GGUF").u32(3).u64(0).u64(pairs);
        w.raw(&vec![0; (pairs * MIN_KV_BYTES) as usize]);
        assert_eq!(Gguf::parse(&w.bytes).unwrap_err(), GgufError::BadValue);
        // The same for tensor infos, which a zero name and zero dimensions
        // refuse at the first one.
        let infos = 500_000u64;
        let mut w = Writer::default();
        w.raw(b"GGUF").u32(3).u64(infos).u64(0);
        w.raw(&vec![0; (infos * MIN_TENSOR_INFO_BYTES) as usize]);
        assert_eq!(
            Gguf::parse(&w.bytes).unwrap_err(),
            GgufError::BadTensorShape {
                name: String::new()
            }
        );
        // One more than the file has room for is Truncated at once.
        let mut w = Writer::default();
        w.raw(b"GGUF").u32(3).u64(0).u64(pairs + 1);
        w.raw(&vec![0; (pairs * MIN_KV_BYTES) as usize]);
        assert_eq!(Gguf::parse(&w.bytes).unwrap_err(), GgufError::Truncated);
        assert_eq!(preallocated(u64::MAX), 1024);
        assert_eq!(preallocated(7), 7);
    }

    #[test]
    fn huge_counts_on_array_elements() {
        // An array of strings with room for `count` empty strings, plus one.
        let strings = 1_000_000u64;
        for (count, expected) in [(strings, true), (strings + 1, false)] {
            let mut array = Writer::default();
            array
                .u32(8)
                .u64(count)
                .raw(&vec![0; (strings * MIN_STRING_BYTES) as usize]);
            let bytes = file(3, &[kv("k", 9, &array.bytes)], &[], 8, &[]);
            assert_eq!(Gguf::parse(&bytes).is_ok(), expected, "count {count}");
            if !expected {
                assert_eq!(Gguf::parse(&bytes).unwrap_err(), GgufError::Truncated);
            }
        }
        // Millions of strings in a file of a few bytes, and of arrays.
        for element in [8u32, 9] {
            let mut array = Writer::default();
            array.u32(element).u64(5_000_000).raw(&[0; 16]);
            let bytes = file(3, &[kv("k", 9, &array.bytes)], &[], 8, &[]);
            assert_eq!(Gguf::parse(&bytes).unwrap_err(), GgufError::Truncated);
        }
    }

    #[test]
    fn invalid_utf8_key_and_value() {
        let mut w = Writer::default();
        w.u64(2).raw(&[0xff, 0xfe]).u32(0).u8(1);
        let bytes = file(3, &[w.bytes], &[], 8, &[]);
        assert_eq!(Gguf::parse(&bytes).unwrap_err(), GgufError::BadValue);

        let mut value = Writer::default();
        value.u64(1).u8(0x80);
        let bytes = file(3, &[kv("k", 8, &value.bytes)], &[], 8, &[]);
        assert_eq!(Gguf::parse(&bytes).unwrap_err(), GgufError::BadValue);

        // Inside an array of strings too.
        let mut array = Writer::default();
        array.u32(8).u64(1).u64(1).u8(0xc0);
        let bytes = file(3, &[kv("k", 9, &array.bytes)], &[], 8, &[]);
        assert_eq!(Gguf::parse(&bytes).unwrap_err(), GgufError::BadValue);

        let mut name = Writer::default();
        name.u64(1).u8(0xff).u32(1).u64(4).u32(0).u64(0);
        let bytes = file(3, &[], &[name.bytes], 8, &[0; 16]);
        assert_eq!(Gguf::parse(&bytes).unwrap_err(), GgufError::BadValue);
    }

    #[test]
    fn bad_bool_and_bad_types() {
        let bytes = file(3, &[kv("k", 7, &[2])], &[], 8, &[]);
        assert_eq!(Gguf::parse(&bytes).unwrap_err(), GgufError::BadValue);
        let mut array = Writer::default();
        array.u32(7).u64(3).raw(&[0, 1, 2]);
        let bytes = file(3, &[kv("k", 9, &array.bytes)], &[], 8, &[]);
        assert_eq!(Gguf::parse(&bytes).unwrap_err(), GgufError::BadValue);
        let bytes = file(3, &[kv("k", 13, &[0; 8])], &[], 8, &[]);
        assert_eq!(Gguf::parse(&bytes).unwrap_err(), GgufError::BadValue);
        let mut array = Writer::default();
        array.u32(99).u64(0);
        let bytes = file(3, &[kv("k", 9, &array.bytes)], &[], 8, &[]);
        assert_eq!(Gguf::parse(&bytes).unwrap_err(), GgufError::BadValue);
        // Arrays nested without end.
        let mut array = Vec::new();
        for _ in 0..40 {
            array.extend_from_slice(&9u32.to_le_bytes());
            array.extend_from_slice(&1u64.to_le_bytes());
        }
        let bytes = file(3, &[kv("k", 9, &array)], &[], 8, &[]);
        assert!(Gguf::parse(&bytes).is_err());
        let mut array = Vec::new();
        for _ in 0..MAX_ARRAY_DEPTH + 1 {
            array.extend_from_slice(&9u32.to_le_bytes());
            array.extend_from_slice(&1u64.to_le_bytes());
        }
        array.extend_from_slice(&4u32.to_le_bytes());
        array.extend_from_slice(&0u64.to_le_bytes());
        let bytes = file(3, &[kv("k", 9, &array)], &[], 8, &[]);
        assert_eq!(Gguf::parse(&bytes).unwrap_err(), GgufError::BadValue);
    }

    #[test]
    fn duplicates_are_refused() {
        let kvs = [kv_u32("a", 1), kv_u32("b", 2), kv_u32("a", 3)];
        let bytes = file(3, &kvs, &[], 8, &[]);
        assert_eq!(
            Gguf::parse(&bytes).unwrap_err(),
            GgufError::DuplicateKey("a".into())
        );
        let tensors = [tensor_info("t", &[4], 0, 0), tensor_info("t", &[4], 0, 32)];
        let bytes = file(3, &[], &tensors, 32, &pattern(64));
        assert_eq!(
            Gguf::parse(&bytes).unwrap_err(),
            GgufError::DuplicateTensor("t".into())
        );
    }

    #[test]
    fn unsupported_tensor_types_name_the_type() {
        let tensors = [
            tensor_info("q4", &[32], 2, 0),
            tensor_info("q2k", &[256], 10, 32),
            tensor_info("odd", &[7], 99, 64),
            tensor_info("removed", &[7], 4, 96),
        ];
        let bytes = file(3, &[], &tensors, 32, &pattern(128));
        let gguf = Gguf::parse(&bytes).unwrap();
        let check = |name: &str, code: u32, text: &str| {
            let error = gguf
                .require_supported(gguf.tensor(name).unwrap())
                .unwrap_err();
            assert_eq!(
                error,
                GgufError::UnsupportedTensorType {
                    name: name.into(),
                    ggml_type: code
                }
            );
            let shown = error.to_string();
            assert!(shown.contains(text), "{shown}");
            assert!(shown.contains(name), "{shown}");
            assert_eq!(
                gguf.tensor_data(gguf.tensor(name).unwrap()).unwrap_err(),
                error
            );
        };
        check("q4", 2, "Q4_0");
        check("q2k", 10, "Q2_K");
        check("odd", 99, "unknown ggml type 99");
        check("removed", 4, "unknown ggml type 4");
    }

    #[test]
    fn typed_getters_report_what_is_wrong() {
        let bytes = sample();
        let gguf = Gguf::parse(&bytes).unwrap();
        assert_eq!(
            gguf.get_u32("nope").unwrap_err(),
            GgufError::MissingKey("nope".into())
        );
        assert_eq!(
            gguf.get_u64("llama.block_count").unwrap_err(),
            GgufError::WrongType {
                key: "llama.block_count".into(),
                expected: "u64"
            }
        );
        assert_eq!(
            gguf.get_u32("general.architecture").unwrap_err(),
            GgufError::WrongType {
                key: "general.architecture".into(),
                expected: "u32"
            }
        );
        assert!(matches!(
            gguf.get_f32("t.f64"),
            Err(GgufError::WrongType { .. })
        ));
        assert!(matches!(
            gguf.get_str("t.flag"),
            Err(GgufError::WrongType { .. })
        ));
        assert!(matches!(
            gguf.get_bool("t.u8"),
            Err(GgufError::WrongType { .. })
        ));
        assert!(matches!(
            gguf.get_bool("none"),
            Err(GgufError::MissingKey(_))
        ));
    }

    #[test]
    fn errors_have_readable_messages() {
        let errors = [
            (GgufError::BadMagic, "magic"),
            (GgufError::UnsupportedVersion(7), "version 7"),
            (GgufError::Truncated, "ends"),
            (GgufError::BadAlignment(12), "12"),
            (GgufError::TooLarge, "too large"),
            (
                GgufError::TensorOutOfBounds {
                    name: "wname".into(),
                },
                "wname",
            ),
            (
                GgufError::MisalignedTensor {
                    name: "wname".into(),
                },
                "wname",
            ),
            (
                GgufError::UnsupportedTensorType {
                    name: "wname".into(),
                    ggml_type: 2,
                },
                "wname",
            ),
            (GgufError::MissingKey("kname".into()), "kname"),
            (
                GgufError::WrongType {
                    key: "kname".into(),
                    expected: "u32",
                },
                "kname",
            ),
            (
                GgufError::BadTensorShape {
                    name: "wname".into(),
                },
                "wname",
            ),
            (GgufError::BadValue, "invalid"),
            (GgufError::DuplicateKey("kname".into()), "kname"),
            (GgufError::DuplicateTensor("wname".into()), "wname"),
        ];
        for (error, word) in errors {
            let shown = error.to_string();
            assert!(shown.contains(word), "{shown} lacks {word}");
            let _: &dyn std::error::Error = &error;
        }
        let shown = GgufError::WrongType {
            key: "kname".into(),
            expected: "u32",
        }
        .to_string();
        assert!(shown.contains("u32"), "{shown}");
        let shown = GgufError::UnsupportedTensorType {
            name: "wname".into(),
            ggml_type: 2,
        }
        .to_string();
        assert!(shown.contains("Q4_0"), "{shown}");
        let shown = GgufError::UnsupportedVersion(7).to_string();
        assert!(shown.contains('2') && shown.contains('3'), "{shown}");
        let shown = GgufError::BadAlignment(12).to_string();
        assert!(
            shown.contains("power of two") && shown.contains('8'),
            "{shown}"
        );
    }

    #[test]
    fn ggml_type_names_follow_the_enum() {
        assert_eq!(ggml_type_name(0), Some("F32"));
        assert_eq!(ggml_type_name(12), Some("Q4_K"));
        assert_eq!(ggml_type_name(30), Some("BF16"));
        assert_eq!(ggml_type_name(43), Some("F8_E4M3"));
        for removed in [4, 5, 31, 32, 33, 36, 37, 38, 44] {
            assert_eq!(ggml_type_name(removed), None);
        }
    }

    #[test]
    fn a_file_without_tensors_has_its_data_right_after_the_pairs() {
        let tail = b"tail!";
        let bytes = file(3, &[kv_u32("a", 1)], &[], 1, tail);
        let unpadded = bytes.len() - tail.len();
        assert_ne!(unpadded % 32, 0);
        assert_ne!(tail.len() % 32, 0);
        let gguf = Gguf::parse(&bytes).unwrap();
        assert_eq!(gguf.alignment(), 32);
        assert!(gguf.tensors().is_empty());
        assert_eq!(gguf.get_u32("a").unwrap(), 1);
        assert_eq!(gguf.data_offset(), unpadded);
    }
}
