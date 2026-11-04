"""A minimal GGUF version 3 writer for the test fixtures, numpy only.

The layout is the one src/gguf/gguf.h reads: the magic "GGUF", u32 version 3, u64 tensor count, u64 metadata count;
the metadata as (string key, u32 value type, value); the tensor infos as (string name, u32 dimensions, u64 extent per
dimension with the fastest-varying first, u32 tensor type, u64 offset into the data); then the tensor data, each
tensor starting at a multiple of the alignment (32) and the file padded to one. Strings are a u64 byte count and the
bytes; arrays are a u32 element type, a u64 count and the elements. Everything is little-endian.
"""

import struct
from pathlib import Path

import numpy as np

ALIGNMENT = 32
U16, I32, U32, F32, STRING, ARRAY = 2, 5, 4, 6, 8, 9  # the value types this writer uses
SCALAR = {U16: "<H", I32: "<i", U32: "<I", F32: "<f"}


def string(s: str) -> bytes:
    data = s.encode()
    return struct.pack("<Q", len(data)) + data


def value(vtype: int, v) -> bytes:
    if vtype == STRING:
        return string(v)
    if vtype == ARRAY:  # v = (element type, list of elements)
        element, items = v
        return struct.pack("<IQ", element, len(items)) + b"".join(value(element, item) for item in items)
    return struct.pack(SCALAR[vtype], v)


def pad(n: int) -> int:
    return -n % ALIGNMENT


def write(path: Path, metadata: list, tensors: list) -> None:
    """metadata: [(key, value type, value)] in file order; tensors: [(name, shape fastest first, type, raw bytes)]."""
    out = [b"GGUF", struct.pack("<IQQ", 3, len(tensors), len(metadata))]
    out += [string(key) + struct.pack("<I", vtype) + value(vtype, v) for key, vtype, v in metadata]
    offset, data = 0, []
    for name, shape, ttype, raw in tensors:
        out.append(string(name) + struct.pack(f"<I{len(shape)}QIQ", len(shape), *shape, ttype, offset))
        raw = np.asarray(raw, dtype=np.uint8).tobytes()
        data.append(raw + bytes(pad(len(raw))))
        offset += len(data[-1])
    header = b"".join(out)
    path.write_bytes(header + bytes(pad(len(header))) + b"".join(data))
