"""Deterministic NIfTI-1 .nii sample generator for tests and smoke checks.

Only used on the client side (unit tests and the one-shot ``verify``
service); the server never imports this module.
"""
from __future__ import annotations

import struct
import sys
from array import array

_DTYPES = {
    "int16": (4, "h", 16),
    "float32": (16, "f", 32),
}


def build_nifti(*, endian="<", datatype="float32", dims=(4, 5, 6),
                data_fn=lambda i, j, k: 0, transform="sform",
                slope=0.0, inter=0.0,
                srow_x=(1.0, 0.0, 0.0, 0.0),
                srow_y=(0.0, 1.0, 0.0, 0.0),
                srow_z=(0.0, 0.0, 1.0, 0.0),
                quatern=(0.0, 0.0, 0.0),
                qoffset=(0.0, 0.0, 0.0),
                pixdim=(1.0, 1.0, 1.0, 1.0),
                qform_code=None, sform_code=None,
                vox_offset=352.0, magic=b"n+1\x00",
                dim0=3, tail_dims=(1, 1, 1, 1),
                bitpix=None, datatype_code=None, sizeof_hdr=348,
                extra=b"", truncate=0):
    """Build a .nii document.  ``data_fn(i, j, k)`` yields raw voxel values
    (i fastest).  Keyword overrides allow crafting malformed headers for
    negative tests.  ``extra`` appends trailing bytes, ``truncate`` drops
    bytes from the end."""
    code = datatype_code if datatype_code is not None else _DTYPES[datatype][0]
    typecode = _DTYPES[datatype][1]
    bp = bitpix if bitpix is not None else _DTYPES[datatype][2]

    nx, ny, nz = dims
    values = [data_fn(i, j, k) for k in range(nz) for j in range(ny) for i in range(nx)]
    if typecode == "h":
        values = [int(v) for v in values]
    arr = array(typecode, values)
    if (sys.byteorder == "little") != (endian == "<"):
        arr.byteswap()
    payload = arr.tobytes()

    if qform_code is None:
        qform_code = 1 if transform == "qform" else 0
    if sform_code is None:
        sform_code = 1 if transform == "sform" else 0

    hdr = bytearray(348)
    struct.pack_into(endian + "i", hdr, 0, sizeof_hdr)
    struct.pack_into(endian + "8h", hdr, 40, dim0, nx, ny, nz, *tail_dims)
    struct.pack_into(endian + "h", hdr, 70, code)
    struct.pack_into(endian + "h", hdr, 72, bp)
    struct.pack_into(endian + "8f", hdr, 76, *pixdim, 0.0, 0.0, 0.0, 0.0)
    struct.pack_into(endian + "f", hdr, 108, float(vox_offset))
    struct.pack_into(endian + "f", hdr, 112, float(slope))
    struct.pack_into(endian + "f", hdr, 116, float(inter))
    struct.pack_into(endian + "h", hdr, 252, qform_code)
    struct.pack_into(endian + "h", hdr, 254, sform_code)
    struct.pack_into(endian + "3f", hdr, 256, *quatern)
    struct.pack_into(endian + "3f", hdr, 268, *qoffset)
    struct.pack_into(endian + "4f", hdr, 280, *srow_x)
    struct.pack_into(endian + "4f", hdr, 296, *srow_y)
    struct.pack_into(endian + "4f", hdr, 312, *srow_z)
    hdr[344:348] = magic

    gap = int(vox_offset) - 348
    body = bytes(hdr) + (b"\x00" * gap if gap > 0 else b"") + payload + extra
    if truncate:
        body = body[:-truncate]
    return body
