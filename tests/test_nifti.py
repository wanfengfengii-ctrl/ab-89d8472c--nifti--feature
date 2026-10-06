"""Header parsing, validation and affine tests for app.nifti."""
import math
import unittest

from app.errors import ApiError
from app.nifti import invert_affine, parse_nifti, quatern_to_affine
from verify.nifti_gen import build_nifti


def data_fn(i, j, k):
    return i + 10 * j + 100 * k


class ParseTests(unittest.TestCase):
    def test_little_endian_int16_sform(self):
        raw = build_nifti(endian="<", datatype="int16", dims=(4, 5, 6),
                          data_fn=data_fn, transform="sform",
                          srow_x=(2, 0, 0, 10), srow_y=(0, 3, 0, 20),
                          srow_z=(0, 0, 4, 30), slope=2.0, inter=5.0)
        vol = parse_nifti(raw)
        self.assertEqual(vol.dims, (4, 5, 6))
        self.assertEqual(vol.transform, "sform")
        self.assertEqual(vol.endian, "<")
        self.assertEqual(vol.affine[0], [2.0, 0.0, 0.0, 10.0])
        inv = vol.inverse
        self.assertAlmostEqual(inv[0][0], 0.5)
        self.assertAlmostEqual(inv[0][3], -5.0)
        self.assertAlmostEqual(inv[1][1], 1.0 / 3.0)
        self.assertAlmostEqual(inv[1][3], -20.0 / 3.0)
        self.assertAlmostEqual(inv[2][2], 0.25)
        self.assertAlmostEqual(inv[2][3], -7.5)
        self.assertEqual(vol.value_at(1, 2, 3), (1 + 20 + 300) * 2 + 5)

    def test_big_endian_float32_qform(self):
        raw = build_nifti(endian=">", datatype="float32", dims=(2, 3, 4),
                          data_fn=data_fn, transform="qform",
                          quatern=(0, 0, 0), qoffset=(10, 20, 30),
                          pixdim=(1, 2, 3, 4))
        vol = parse_nifti(raw)
        self.assertEqual(vol.transform, "qform")
        self.assertEqual(vol.endian, ">")
        self.assertEqual(vol.affine[0][0], 2.0)
        self.assertEqual(vol.affine[1][1], 3.0)
        self.assertEqual(vol.affine[2][2], 4.0)
        self.assertEqual(vol.affine[0][3], 10.0)
        self.assertEqual(vol.value_at(1, 1, 1), 111.0)

    def test_qform_rotation_matrix(self):
        s2 = math.sqrt(0.5)
        raw = build_nifti(transform="qform", datatype="float32", dims=(2, 2, 2),
                          quatern=(0, 0, s2), qoffset=(1, 2, 3),
                          pixdim=(1, 1, 1, 1))
        vol = parse_nifti(raw)
        m = vol.affine
        self.assertAlmostEqual(m[0][0], 0.0, places=6)
        self.assertAlmostEqual(m[0][1], -1.0, places=6)
        self.assertAlmostEqual(m[1][0], 1.0, places=6)
        self.assertAlmostEqual(m[1][1], 0.0, places=6)
        self.assertAlmostEqual(m[2][2], 1.0, places=6)
        self.assertEqual(m[0][3], 1.0)
        self.assertEqual(m[1][3], 2.0)
        self.assertEqual(m[2][3], 3.0)

    def test_qfac_negative_flips_third_column(self):
        raw = build_nifti(transform="qform", datatype="float32", dims=(2, 2, 2),
                          quatern=(0, 0, 0), qoffset=(0, 0, 0),
                          pixdim=(-1, 2, 3, 4))
        vol = parse_nifti(raw)
        self.assertEqual(vol.affine[2][2], -4.0)

    def test_sform_wins_over_qform(self):
        raw = build_nifti(transform="sform", datatype="float32", dims=(2, 2, 2),
                          srow_x=(2, 0, 0, 0), srow_y=(0, 2, 0, 0),
                          srow_z=(0, 0, 2, 0),
                          quatern=(0, 0, 0), qoffset=(9, 9, 9),
                          pixdim=(1, 9, 9, 9), qform_code=1, sform_code=1)
        vol = parse_nifti(raw)
        self.assertEqual(vol.transform, "sform")
        self.assertEqual(vol.affine[0][0], 2.0)

    def test_vox_offset_with_extension_gap(self):
        raw = build_nifti(datatype="int16", dims=(2, 2, 2), data_fn=data_fn,
                          vox_offset=400.0)
        vol = parse_nifti(raw)
        self.assertEqual(vol.vox_offset, 400)
        self.assertEqual(vol.value_at(1, 1, 1), 111.0)

    def test_slope_zero_means_unscaled(self):
        raw = build_nifti(datatype="int16", dims=(2, 2, 2), data_fn=data_fn,
                          slope=0.0, inter=999.0)
        vol = parse_nifti(raw)
        self.assertEqual(vol.value_at(1, 1, 1), 111.0)


class HeaderErrorTests(unittest.TestCase):
    def assert_code(self, code, raw, field=None):
        with self.assertRaises(ApiError) as ctx:
            parse_nifti(raw)
        self.assertEqual(ctx.exception.code, code)
        if field is not None:
            self.assertEqual(ctx.exception.field, field)

    def test_header_too_short(self):
        self.assert_code("header_too_short", b"\x00" * 100, "file")

    def test_bad_sizeof_hdr(self):
        self.assert_code("bad_sizeof_hdr", build_nifti(sizeof_hdr=347), "sizeof_hdr")
        self.assert_code("bad_sizeof_hdr",
                         build_nifti(endian=">", sizeof_hdr=347), "sizeof_hdr")

    def test_unsupported_magic(self):
        self.assert_code("unsupported_magic", build_nifti(magic=b"ni1\x00"), "magic")
        self.assert_code("unsupported_magic", build_nifti(magic=b"n+2\x00"), "magic")

    def test_invalid_dimensions(self):
        self.assert_code("invalid_dimensions", build_nifti(dim0=4), "dim")
        self.assert_code("invalid_dimensions", build_nifti(dim0=2), "dim")
        self.assert_code("invalid_dimensions",
                         build_nifti(tail_dims=(1, 1, 1, 0)), "dim")
        self.assert_code("invalid_dimensions", build_nifti(dims=(0, 2, 2)), "dim")

    def test_unsupported_datatype(self):
        self.assert_code("unsupported_datatype",
                         build_nifti(datatype_code=2), "datatype")
        self.assert_code("unsupported_datatype",
                         build_nifti(datatype_code=64), "datatype")

    def test_bitpix_mismatch(self):
        self.assert_code("bitpix_mismatch", build_nifti(bitpix=8), "bitpix")
        self.assert_code("bitpix_mismatch",
                         build_nifti(datatype="int16", bitpix=32), "bitpix")

    def test_invalid_vox_offset(self):
        self.assert_code("invalid_vox_offset",
                         build_nifti(vox_offset=100.0), "vox_offset")
        self.assert_code("invalid_vox_offset",
                         build_nifti(vox_offset=352.5), "vox_offset")

    def test_payload_length_mismatch(self):
        self.assert_code("payload_length_mismatch",
                         build_nifti(truncate=2), "file")
        self.assert_code("payload_length_mismatch",
                         build_nifti(endian=">", datatype="int16", truncate=1),
                         "file")

    def test_trailing_bytes(self):
        self.assert_code("trailing_bytes", build_nifti(extra=b"\x00"), "file")
        self.assert_code("trailing_bytes",
                         build_nifti(endian=">", extra=b"abc"), "file")

    def test_non_finite_scaling(self):
        self.assert_code("non_finite_scaling",
                         build_nifti(slope=float("nan")), "scl_slope")
        self.assert_code("non_finite_scaling",
                         build_nifti(inter=float("inf")), "scl_inter")

    def test_missing_affine(self):
        self.assert_code("missing_affine",
                         build_nifti(qform_code=0, sform_code=0),
                         "qform_code,sform_code")

    def test_non_finite_affine(self):
        self.assert_code("non_finite_affine",
                         build_nifti(srow_x=(float("nan"), 0, 0, 0)), "srow_x")
        self.assert_code("non_finite_affine",
                         build_nifti(transform="qform",
                                     quatern=(0, 0, float("inf"))), "quatern_d")

    def test_singular_affine(self):
        self.assert_code("singular_affine",
                         build_nifti(srow_x=(0, 0, 0, 0)), "srow")
        self.assert_code("singular_affine",
                         build_nifti(srow_z=(0, 0, 0, 0)), "srow")

    def test_invalid_pixdim_for_qform(self):
        self.assert_code("invalid_pixdim",
                         build_nifti(transform="qform", pixdim=(1, 2, 0, 4)),
                         "pixdim")
        self.assert_code("invalid_pixdim",
                         build_nifti(transform="qform",
                                     pixdim=(1, 2, float("nan"), 4)), "pixdim")

    def test_singular_sform_rejected_even_with_valid_qform(self):
        raw = build_nifti(transform="sform", qform_code=1, sform_code=1,
                          srow_x=(0, 0, 0, 0),
                          quatern=(0, 0, 0), qoffset=(0, 0, 0),
                          pixdim=(1, 1, 1, 1))
        self.assert_code("singular_affine", raw, "srow")


class AffineTests(unittest.TestCase):
    def test_quatern_to_affine_known_rotation(self):
        m = quatern_to_affine(0.0, 0.0, math.sqrt(0.5), 1, 2, 3, 2.0, 3.0, 4.0, 1.0)
        self.assertAlmostEqual(m[0][0], 0.0, places=6)
        self.assertAlmostEqual(m[0][1], -3.0, places=6)
        self.assertAlmostEqual(m[1][0], 2.0, places=6)
        self.assertAlmostEqual(m[2][2], 4.0, places=6)
        self.assertEqual(m[0][3], 1)
        self.assertEqual(m[2][3], 3)

    def test_invert_affine_roundtrip(self):
        m = quatern_to_affine(0.1, -0.2, 0.3, 5, -7, 11, 2.0, 3.0, 4.0, -1.0)
        inv = invert_affine(m, field="qform")
        for r in range(3):
            for c in range(4):
                val = sum(m[r][k] * inv[k][c] for k in range(3))
                if c == 3:
                    val += m[r][3]
                self.assertAlmostEqual(val, 1.0 if r == c else 0.0, places=9)

    def test_invert_affine_singular(self):
        m = [[1, 2, 3, 0], [2, 4, 6, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        with self.assertRaises(ApiError) as ctx:
            invert_affine(m, field="srow")
        self.assertEqual(ctx.exception.code, "singular_affine")


if __name__ == "__main__":
    unittest.main()
