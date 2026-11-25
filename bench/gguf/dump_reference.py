#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Reference dequantisation of GGUF tensors, from the gguf-py of llama.cpp.

Issue 30 reads quantized GGUF files in Rust and compares the result with a
published implementation. This script is that implementation's side of the
comparison for the dequantisation: it reads GGUF files with the `gguf` package
that ships in the llama.cpp checkout (pass its `gguf-py` directory with
--gguf-py, so that the version is the pinned one) and writes

* `types.json`: for each file, the number of tensors of each type and the
  names of the tensors of each type;
* `tensors-<file>.json`: for each tensor of a quantized type, its shape, type,
  byte size, the SHA-256 of its raw bytes and the SHA-256 of its dequantised
  values as little endian float32;
* `vectors.json`: a few whole blocks of each quantized type, the raw bytes in
  hexadecimal and the dequantised float32 values as 8 hexadecimal digits each
  (their bits), which are small enough to be committed and to be the test
  vectors of the Rust decoders.

Usage: dump_reference.py --gguf-py DIR --out DIR FILE.gguf [FILE.gguf ...]

It needs numpy and nothing else. It does not write inside the repository unless
--out says so.
"""

import argparse
import collections
import hashlib
import json
import os
import struct
import sys

# How many blocks of each type go to the test vectors, and where they come from.
VECTOR_BLOCKS = 6


def load_gguf(gguf_py):
    sys.path.insert(0, gguf_py)
    import gguf  # noqa: E402  (the path above is why the import is here)

    return gguf


def float_bits(values):
    """The float32 values as a list of 8 digit hexadecimal strings."""
    return ["%08x" % struct.unpack("<I", struct.pack("<f", float(v)))[0] for v in values]


def block_geometry(gguf, tensor_type):
    block, size = gguf.GGML_QUANT_SIZES[tensor_type]
    return block, size


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gguf-py", required=True, help="the gguf-py directory of llama.cpp")
    parser.add_argument("--out", required=True, help="where the JSON files are written")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    gguf = load_gguf(args.gguf_py)
    import numpy as np  # noqa: E402

    os.makedirs(args.out, exist_ok=True)
    types = {}
    vectors = {}
    quantized = ("Q8_0", "Q5_0", "Q4_K", "Q6_K")

    for path in args.files:
        reader = gguf.GGUFReader(path)
        name = os.path.basename(path)
        counts = collections.Counter(t.tensor_type.name for t in reader.tensors)
        by_type = collections.defaultdict(list)
        tensors = []
        for tensor in reader.tensors:
            tname = tensor.tensor_type.name
            by_type[tname].append(tensor.name)
            if tname not in quantized:
                continue
            raw = np.asarray(tensor.data).reshape(-1).view(np.uint8)
            values = gguf.quants.dequantize(np.asarray(tensor.data), tensor.tensor_type)
            values = np.ascontiguousarray(values, dtype="<f4")
            tensors.append(
                {
                    "name": tensor.name,
                    "type": tname,
                    "shape": [int(x) for x in tensor.shape],
                    "bytes": int(raw.size),
                    "sha256_raw": hashlib.sha256(raw.tobytes()).hexdigest(),
                    "sha256_f32": hashlib.sha256(values.tobytes()).hexdigest(),
                }
            )
            # Whole blocks for the test vectors, taken from the first tensor
            # of the type that has enough of them, at the start, one third of
            # the way, two thirds of the way and the last blocks.
            if tname in vectors:
                continue
            elements, size = block_geometry(gguf, tensor.tensor_type)
            blocks = raw.size // size
            if blocks < VECTOR_BLOCKS * 3:
                continue
            picks = sorted(
                {0, 1, blocks // 3, blocks // 3 + 1, 2 * blocks // 3, blocks - 1}
            )
            chosen = []
            for index in picks[:VECTOR_BLOCKS]:
                block_bytes = raw[index * size : (index + 1) * size]
                block_values = values.reshape(-1)[index * elements : (index + 1) * elements]
                chosen.append(
                    {
                        "tensor": tensor.name,
                        "block": int(index),
                        "bytes_hex": block_bytes.tobytes().hex(),
                        "f32_bits": float_bits(block_values),
                    }
                )
            vectors[tname] = {
                "block_elements": int(elements),
                "block_bytes": int(size),
                "blocks": chosen,
            }
        types[name] = {"counts": dict(counts), "tensors": dict(by_type)}
        with open(os.path.join(args.out, "tensors-%s.json" % name), "w") as handle:
            json.dump(tensors, handle, indent=1)
            handle.write("\n")

    with open(os.path.join(args.out, "types.json"), "w") as handle:
        json.dump(types, handle, indent=1)
        handle.write("\n")
    with open(os.path.join(args.out, "vectors.json"), "w") as handle:
        json.dump(vectors, handle, indent=1)
        handle.write("\n")


if __name__ == "__main__":
    main()
