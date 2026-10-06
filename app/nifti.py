"""Strict NIfTI-1 single-file (.nii) parsing for the QC sampling API.

Only a deliberately small, QC-safe subset is accepted; anything else is
rejected with a stable error code naming the offending header field:

* magic ``n+1`` (single .nii file, not a .hdr/.img pair)
* ``dim[0] == 3`` and ``dim[4..7] == 1`` (complete 3D volume)
* datatype int16 (4) or float32 (16) with matching bitpix
* little- or big-endian header and voxel data
* finite, integral ``vox_offset >= 352``
* payload length exactly ``nx*ny*nz*(bitpix/8)`` with no trailing bytes
* finite ``scl_slope`` / ``scl_inter``
* a usable spatial affine: a valid sform (``sform_code > 0``) is preferred,
  otherwise the qform (``qform_code > 0``); the chosen affine must be
  invertible.  Missing both, or a non-invertible chosen transform, is
  rejected.
"""
from __future__ import annotations

import math
import struct
import sys
from array import array

from .errors import ApiError

HEADER_SIZE = 348
MIN_VOX_OFFSET = 352  # 348-byte header + 4-byte extender in "n+1" files
MAGIC_NII = b"n+1\x00"

DT_INT16 = 4
DT_FLOAT32 = 16
# datatype -> (array typecode, itemsize, expected bitpix)
_DTYPES = {
    DT_INT16: ("h", 2, 16),
    DT_FLOAT32: ("f", 4, 32),
}


class NiftiVolume:
    """Parsed, validated volume ready for sampling."""

    __slots__ = (
        "dims", "datatype", "endian", "vox_offset", "scl_slope", "scl_inter",
        "transform", "affine", "inverse", "data",
    )

    def __init__(self, *, dims, datatype, endian, vox_offset, scl_slope,
                 scl_inter, transform, affine, inverse, data):
        self.dims = dims                # (nx, ny, nz)
        self.datatype = datatype        # 4 (int16) or 16 (float32)
        self.endian = endian            # "<" or ">"
        self.vox_offset = vox_offset
        self.scl_slope = scl_slope
        self.scl_inter = scl_inter
        self.transform = transform      # "sform" or "qform"
        self.affine = affine            # voxel (i,j,k) -> RAS (x,y,z), 4x4
        self.inverse = inverse          # RAS -> voxel, 4x4
        self.data = data                # array('h') or array('f'), i fastest

    def value_at(self, i, j, k):
        """Scaled voxel value at integer indices (NIfTI: first dim fastest)."""
        nx, ny, _ = self.dims
        raw = self.data[i + nx * (j + ny * k)]
        if self.scl_slope != 0.0:
            return raw * self.scl_slope + self.scl_inter
        return float(raw)


def quatern_to_affine(b, c, d, qx, qy, qz, dx, dy, dz, qfac):
    """Quaternion (b,c,d) + offset + zooms -> 4x4 affine.

    Follows nifti1_io.c ``nifti_quatern_to_mat44``: ``a`` is recovered as
    ``sqrt(1 - (b^2+c^2+d^2))``; if that falls below 1e-7 the (b,c,d)
    vector is normalized and ``a = 0`` (180-degree rotation).
    """
    ss = b * b + c * c + d * d
    a2 = 1.0 - ss
    if a2 < 1.0e-7:  # special case: normalize (b,c,d), a = 0
        norm = math.sqrt(ss)
        b, c, d = b / norm, c / norm, d / norm
        a = 0.0
    else:
        a = math.sqrt(a2)
    rot = (
        (a * a + b * b - c * c - d * d, 2.0 * (b * c - a * d), 2.0 * (b * d + a * c)),
        (2.0 * (b * c + a * d), a * a + c * c - b * b - d * d, 2.0 * (c * d - a * b)),
        (2.0 * (b * d - a * c), 2.0 * (c * d + a * b), a * a + d * d - c * c - b * b),
    )
    z = dz * qfac
    return [
        [rot[0][0] * dx, rot[0][1] * dy, rot[0][2] * z, qx],
        [rot[1][0] * dx, rot[1][1] * dy, rot[1][2] * z, qy],
        [rot[2][0] * dx, rot[2][1] * dy, rot[2][2] * z, qz],
        [0.0, 0.0, 0.0, 1.0],
    ]


def invert_affine(m, *, field):
    """Invert a 4x4 affine ``[R t; 0 1]``.

    Raises :class:`ApiError` ``singular_affine`` when the linear part is
    singular or the inverse is not finite.
    """
    (a, b, c, tx), (d, e, f, ty), (g, h, i, tz) = m[0], m[1], m[2]
    ca = e * i - f * h
    cb = -(d * i - f * g)
    cc = d * h - e * g
    det = a * ca + b * cb + c * cc
    if not math.isfinite(det) or det == 0.0:
        raise ApiError("singular_affine", "spatial affine is not invertible", field=field)
    inv = 1.0 / det
    r = (
        (ca * inv, -(b * i - c * h) * inv, (b * f - c * e) * inv),
        (cb * inv, (a * i - c * g) * inv, -(a * f - c * d) * inv),
        (cc * inv, -(a * h - b * g) * inv, (a * e - b * d) * inv),
    )
    out = [
        [r[0][0], r[0][1], r[0][2], -(r[0][0] * tx + r[0][1] * ty + r[0][2] * tz)],
        [r[1][0], r[1][1], r[1][2], -(r[1][0] * tx + r[1][1] * ty + r[1][2] * tz)],
        [r[2][0], r[2][1], r[2][2], -(r[2][0] * tx + r[2][1] * ty + r[2][2] * tz)],
        [0.0, 0.0, 0.0, 1.0],
    ]
    if not all(math.isfinite(v) for row in out for v in row):
        raise ApiError("singular_affine", "spatial affine inverse is not finite", field=field)
    return out


def parse_nifti(raw: bytes) -> NiftiVolume:
    """Parse and strictly validate a single-file NIfTI-1 .nii document."""
    if len(raw) < HEADER_SIZE:
        raise ApiError("header_too_short",
                       f"file has {len(raw)} bytes, less than the 348-byte NIfTI-1 header",
                       field="file")
    (sizeof_hdr_le,) = struct.unpack_from("<i", raw, 0)
    if sizeof_hdr_le == HEADER_SIZE:
        en = "<"
    else:
        (sizeof_hdr_be,) = struct.unpack_from(">i", raw, 0)
        if sizeof_hdr_be == HEADER_SIZE:
            en = ">"
        else:
            raise ApiError("bad_sizeof_hdr",
                           "sizeof_hdr is not 348 in either byte order",
                           field="sizeof_hdr")

    magic = raw[344:348]
    if magic != MAGIC_NII:
        raise ApiError("unsupported_magic",
                       f"magic={magic!r}; only single-file .nii (magic 'n+1') is accepted",
                       field="magic")

    dim = struct.unpack_from(en + "8h", raw, 40)
    nx, ny, nz = dim[1], dim[2], dim[3]
    if dim[0] != 3:
        raise ApiError("invalid_dimensions",
                       f"dim[0]={dim[0]}; only complete 3D volumes (dim[0]=3) are accepted",
                       field="dim")
    if nx < 1 or ny < 1 or nz < 1:
        raise ApiError("invalid_dimensions",
                       f"dim[1:4]=({nx},{ny},{nz}); dimensions must be positive",
                       field="dim")
    if tuple(dim[4:]) != (1, 1, 1, 1):
        raise ApiError("invalid_dimensions",
                       f"dim[4:8]={tuple(dim[4:])}; expected (1, 1, 1, 1) for a complete 3D volume",
                       field="dim")

    (datatype,) = struct.unpack_from(en + "h", raw, 70)
    (bitpix,) = struct.unpack_from(en + "h", raw, 72)
    if datatype not in _DTYPES:
        raise ApiError("unsupported_datatype",
                       f"datatype={datatype}; only int16 (4) and float32 (16) are accepted",
                       field="datatype")
    typecode, itemsize, expected_bitpix = _DTYPES[datatype]
    if bitpix != expected_bitpix:
        raise ApiError("bitpix_mismatch",
                       f"bitpix={bitpix} inconsistent with datatype={datatype} "
                       f"(expected {expected_bitpix})",
                       field="bitpix")

    (vox_offset_f,) = struct.unpack_from(en + "f", raw, 108)
    if not math.isfinite(vox_offset_f) or vox_offset_f != math.floor(vox_offset_f):
        raise ApiError("invalid_vox_offset",
                       f"vox_offset={vox_offset_f} is not a finite integer",
                       field="vox_offset")
    vox_offset = int(vox_offset_f)
    if vox_offset < MIN_VOX_OFFSET:
        raise ApiError("invalid_vox_offset",
                       f"vox_offset={vox_offset} is below the {MIN_VOX_OFFSET}-byte "
                       f"minimum for single-file .nii",
                       field="vox_offset")

    (scl_slope,) = struct.unpack_from(en + "f", raw, 112)
    (scl_inter,) = struct.unpack_from(en + "f", raw, 116)
    if not math.isfinite(scl_slope):
        raise ApiError("non_finite_scaling", f"scl_slope={scl_slope} is not finite",
                       field="scl_slope")
    if not math.isfinite(scl_inter):
        raise ApiError("non_finite_scaling", f"scl_inter={scl_inter} is not finite",
                       field="scl_inter")

    nvox = nx * ny * nz
    payload_len = nvox * itemsize
    expected_total = vox_offset + payload_len
    if len(raw) < expected_total:
        raise ApiError("payload_length_mismatch",
                       f"file is {len(raw)} bytes but the header implies {expected_total} "
                       f"(vox_offset {vox_offset} + {nvox} voxels x {itemsize} bytes)",
                       field="file")
    if len(raw) > expected_total:
        raise ApiError("trailing_bytes",
                       f"file has {len(raw) - expected_total} trailing byte(s) beyond "
                       f"the voxel payload",
                       field="file")

    qform_code, sform_code = struct.unpack_from(en + "2h", raw, 252)
    if sform_code > 0:
        rows = []
        for off, name in ((280, "srow_x"), (296, "srow_y"), (312, "srow_z")):
            row = list(struct.unpack_from(en + "4f", raw, off))
            if not all(math.isfinite(v) for v in row):
                raise ApiError("non_finite_affine", f"{name} contains non-finite values",
                               field=name)
            rows.append(row)
        affine = rows + [[0.0, 0.0, 0.0, 1.0]]
        transform = "sform"
        affine_field = "srow"
    elif qform_code > 0:
        qb, qc, qd = struct.unpack_from(en + "3f", raw, 256)
        qx, qy, qz = struct.unpack_from(en + "3f", raw, 268)
        for name, val in (("quatern_b", qb), ("quatern_c", qc), ("quatern_d", qd),
                          ("qoffset_x", qx), ("qoffset_y", qy), ("qoffset_z", qz)):
            if not math.isfinite(val):
                raise ApiError("non_finite_affine", f"{name} is not finite", field=name)
        pixdim = struct.unpack_from(en + "8f", raw, 76)
        dx, dy, dz = pixdim[1], pixdim[2], pixdim[3]
        if not (math.isfinite(dx) and math.isfinite(dy) and math.isfinite(dz)) \
                or dx <= 0.0 or dy <= 0.0 or dz <= 0.0:
            raise ApiError("invalid_pixdim",
                           f"pixdim[1:4]=({dx},{dy},{dz}); voxel sizes must be positive "
                           f"and finite when the qform is used",
                           field="pixdim")
        qfac = -1.0 if pixdim[0] < 0.0 else 1.0
        affine = quatern_to_affine(qb, qc, qd, qx, qy, qz, dx, dy, dz, qfac)
        transform = "qform"
        affine_field = "qform"
    else:
        raise ApiError("missing_affine",
                       "qform_code and sform_code are both zero; no spatial transform "
                       "available",
                       field="qform_code,sform_code")

    inverse = invert_affine(affine, field=affine_field)

    payload = raw[vox_offset:expected_total]
    data = array(typecode)
    data.frombytes(payload)
    if data.itemsize != itemsize:  # pragma: no cover - platform guard
        raise ApiError("internal_error", "platform array itemsize mismatch", status=500)
    if (sys.byteorder == "little") != (en == "<"):
        data.byteswap()

    return NiftiVolume(
        dims=(nx, ny, nz), datatype=datatype, endian=en, vox_offset=vox_offset,
        scl_slope=scl_slope, scl_inter=scl_inter, transform=transform,
        affine=affine, inverse=inverse, data=data,
    )
