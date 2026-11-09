"""A minimal GGUF version 3 writer for the test fixtures and tools/make_draft.py, numpy only.

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


def layout(metadata: list, infos: list) -> tuple[bytes, list]:
    """The header padded to the alignment, and the file offset of each tensor's data followed by the file size.

    metadata: [(key, value type, value)] in file order; infos: [(name, shape fastest first, type, byte count)]."""
    out = [b"GGUF", struct.pack("<IQQ", 3, len(infos), len(metadata))]
    out += [string(key) + struct.pack("<I", vtype) + value(vtype, v) for key, vtype, v in metadata]
    offset, offsets = 0, []
    for name, shape, ttype, nbytes in infos:
        out.append(string(name) + struct.pack(f"<I{len(shape)}QIQ", len(shape), *shape, ttype, offset))
        offsets.append(offset)
        offset += nbytes + pad(nbytes)
    header = b"".join(out)
    header += bytes(pad(len(header)))
    return header, [len(header) + o for o in offsets + [offset]]


def write(path: Path, metadata: list, tensors: list) -> None:
    """metadata: [(key, value type, value)] in file order; tensors: [(name, shape fastest first, type, raw bytes)]."""
    raws = [np.asarray(raw, dtype=np.uint8).tobytes() for *_, raw in tensors]
    infos = [(name, shape, ttype, len(raw)) for (name, shape, ttype, _), raw in zip(tensors, raws)]
    header, _ = layout(metadata, infos)
    path.write_bytes(header + b"".join(raw + bytes(pad(len(raw))) for raw in raws))
