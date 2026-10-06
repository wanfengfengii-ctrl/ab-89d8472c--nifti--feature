"""End-to-end tests for ``POST /api/nifti/compare``.

The two uploads deliberately mix byte order, datatype, affine source and
grid geometry; every world coordinate must be mapped into each grid
independently.
"""
import json
import logging
import threading
import unittest
from http.server import ThreadingHTTPServer

from app.server import Handler, MAX_FILE_BYTES
from verify.httpclient import (BOUNDARY, build_compare_multipart, get_json,
                               post_compare)
from verify.nifti_gen import build_nifti

# baseline: little-endian int16, anisotropic sform on a 4x5x6 grid
B_AFFINE = ((2.0, 0.0, 0.0, 10.0),
            (0.0, 3.0, 0.0, 20.0),
            (0.0, 0.0, 4.0, 30.0))
# followup: big-endian float32, identity qform on a different-sized grid.
# Chosen so world (12, 26, 42) is inside while (16, 32, 50) is out on z.
F_DIMS = (20, 33, 43)


def b_data(i, j, k):
    return i + 10 * j + 100 * k


def f_data(i, j, k):
    return 1000.0 + 2.0 * i + 3.0 * j + 5.0 * k


def baseline_file(**overrides):
    kw = dict(endian="<", datatype="int16", dims=(4, 5, 6), data_fn=b_data,
              transform="sform",
              srow_x=B_AFFINE[0], srow_y=B_AFFINE[1], srow_z=B_AFFINE[2])
    kw.update(overrides)
    return build_nifti(**kw)


def followup_file(**overrides):
    kw = dict(endian=">", datatype="float32", dims=F_DIMS, data_fn=f_data,
              transform="qform", quatern=(0.0, 0.0, 0.0),
              qoffset=(0.0, 0.0, 0.0), pixdim=(1.0, 1.0, 1.0, 1.0))
    kw.update(overrides)
    return build_nifti(**kw)


def baseline_world(voxel):
    i, j, k = voxel
    return [2.0 * i + 10.0, 3.0 * j + 20.0, 4.0 * k + 30.0]


class CompareApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.daemon_threads = True
        cls.server.ready = True
        cls.port = cls.server.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        logging.disable(logging.NOTSET)

    def post_raw(self, body, content_type):
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            conn.request("POST", "/api/nifti/compare", body=body,
                         headers={"Content-Type": content_type})
            resp = conn.getresponse()
            raw = resp.read()
            status = resp.status
        finally:
            conn.close()
        try:
            return status, json.loads(raw)
        except (UnicodeDecodeError, ValueError):
            return status, None

    # -- happy paths ----------------------------------------------------------
    def test_independent_mapping_and_difference(self):
        points = [
            {"id": 7, "point": baseline_world((1.0, 2.0, 3.0))},
            {"id": 3, "point": baseline_world((0.5, 0.5, 0.5))},
        ]
        status, payload = post_compare(self.base, baseline_file(),
                                       followup_file(), points)
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["transform"],
                         {"baseline": "sform", "followup": "qform"})
        self.assertEqual([r["id"] for r in payload["results"]], [7, 3])

        r7 = payload["results"][0]
        self.assertEqual(r7["status"], "ok")
        # baseline maps to (1,2,3); the same world point maps to (12,26,42)
        # in the followup identity grid.
        self.assertEqual(r7["baseline"]["voxel"], [1.0, 2.0, 3.0])
        for got, want in zip(r7["followup"]["voxel"], (12.0, 26.0, 42.0)):
            self.assertAlmostEqual(got, want, places=5)
        self.assertEqual(r7["baseline"]["intensity"], float(b_data(1, 2, 3)))
        self.assertAlmostEqual(r7["followup"]["intensity"], f_data(12, 26, 42),
                               places=3)
        self.assertEqual(r7["baseline"]["transform"], "sform")
        self.assertEqual(r7["followup"]["transform"], "qform")
        self.assertAlmostEqual(
            r7["difference"],
            f_data(12, 26, 42) - b_data(1, 2, 3), places=3)

        # fractional voxel coordinates on both sides, interpolated, and the
        # difference is followup minus baseline.
        r3 = payload["results"][1]
        self.assertEqual(r3["status"], "ok")
        for got, want in zip(r3["baseline"]["voxel"], (0.5, 0.5, 0.5)):
            self.assertAlmostEqual(got, want, places=5)
        # followup voxel for world (11,21.5,32): same numbers under identity
        for got, want in zip(r3["followup"]["voxel"], (11.0, 21.5, 32.0)):
            self.assertAlmostEqual(got, want, places=4)
        b_avg = sum(b_data(i, j, k)
                    for i in (0, 1) for j in (0, 1) for k in (0, 1)) / 8.0
        # followup data is linear in voxel coords, so trilinear interpolation
        # equals the function evaluated at the continuous coordinate.
        f_exact = 1000.0 + 2.0 * 11.0 + 3.0 * 21.5 + 5.0 * 32.0
        self.assertAlmostEqual(r3["baseline"]["intensity"], b_avg, places=3)
        self.assertAlmostEqual(r3["followup"]["intensity"], f_exact, places=3)
        self.assertAlmostEqual(r3["difference"], f_exact - b_avg, places=3)

    def test_scaling_is_per_side(self):
        # Same grid/affine geometry, but each side applies its own slope/inter.
        raw_b = baseline_file(slope=2.0, inter=5.0)
        raw_f = build_nifti(endian=">", datatype="float32", dims=(4, 5, 6),
                            data_fn=b_data, transform="sform", slope=0.25,
                            inter=1.0,
                            srow_x=B_AFFINE[0], srow_y=B_AFFINE[1],
                            srow_z=B_AFFINE[2])
        points = [{"id": 1, "point": baseline_world((1.0, 2.0, 3.0))}]
        status, payload = post_compare(self.base, raw_b, raw_f, points)
        self.assertEqual(status, 200, payload)
        (r,) = payload["results"]
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["baseline"]["intensity"], b_data(1, 2, 3) * 2.0 + 5.0)
        self.assertAlmostEqual(r["followup"]["intensity"],
                               b_data(1, 2, 3) * 0.25 + 1.0, places=3)

    # -- per-point, per-side failures ----------------------------------------
    def test_one_side_out_of_bounds_is_isolated(self):
        points = [
            # world inside both grids
            {"id": 1, "point": baseline_world((1.0, 1.0, 1.0))},
            # outside baseline (negative inverse voxel), inside followup
            {"id": 2, "point": [0.5, 0.5, 0.5]},
            # inside baseline, outside the 40-cube followup grid (z=50)
            {"id": 3, "point": baseline_world((3.0, 4.0, 5.0))},
        ]
        status, payload = post_compare(self.base, baseline_file(),
                                       followup_file(), points)
        self.assertEqual(status, 200, payload)
        by_id = {r["id"]: r for r in payload["results"]}

        self.assertEqual(by_id[1]["status"], "ok")
        r2 = by_id[2]
        self.assertEqual(r2["status"], "error")
        self.assertEqual(set(r2["errors"]), {"baseline"})
        self.assertEqual(r2["errors"]["baseline"]["code"], "out_of_bounds")
        self.assertIn("voxel", r2["errors"]["baseline"])
        r3 = by_id[3]
        self.assertEqual(r3["status"], "error")
        self.assertEqual(set(r3["errors"]), {"followup"})
        self.assertEqual(r3["errors"]["followup"]["code"], "out_of_bounds")
        self.assertIn("voxel", r3["errors"]["followup"])

    def test_non_finite_data_marked_on_failing_side(self):
        raw_f = followup_file(
            data_fn=lambda i, j, k: float("nan")
            if (i, j, k) == (12, 26, 42) else f_data(i, j, k))
        points = [{"id": 1, "point": baseline_world((1.0, 2.0, 3.0))}]
        status, payload = post_compare(self.base, baseline_file(), raw_f, points)
        self.assertEqual(status, 200, payload)
        (r,) = payload["results"]
        self.assertEqual(r["status"], "error")
        self.assertEqual(set(r["errors"]), {"followup"})
        self.assertEqual(r["errors"]["followup"]["code"], "non_finite_data")

    def test_both_sides_failing_listed_together(self):
        # world point outside both grids
        points = [{"id": 1, "point": [-100.0, -100.0, -100.0]}]
        status, payload = post_compare(self.base, baseline_file(),
                                       followup_file(), points)
        self.assertEqual(status, 200, payload)
        (r,) = payload["results"]
        self.assertEqual(r["status"], "error")
        self.assertEqual(set(r["errors"]), {"baseline", "followup"})
        for side in ("baseline", "followup"):
            self.assertEqual(r["errors"][side]["code"], "out_of_bounds")

    # -- file-level errors ------------------------------------------------------
    def assert_error(self, status, payload, want_status, code, file=None, field=None):
        self.assertEqual(status, want_status, payload)
        err = payload["error"]
        self.assertEqual(err["code"], code)
        if file is not None:
            self.assertEqual(err["file"], file)
        if field is not None:
            self.assertEqual(err["field"], field)

    def test_bad_baseline_header_is_tagged(self):
        raw_b = baseline_file(extra=b"\x00")
        status, payload = post_compare(
            self.base, raw_b, followup_file(),
            [{"id": 1, "point": baseline_world((1.0, 1.0, 1.0))}])
        self.assert_error(status, payload, 400, "trailing_bytes",
                          file="baseline", field="file")

    def test_bad_followup_header_is_tagged(self):
        raw_f = followup_file(qform_code=0, sform_code=0)
        status, payload = post_compare(
            self.base, baseline_file(), raw_f,
            [{"id": 1, "point": baseline_world((1.0, 1.0, 1.0))}])
        self.assert_error(status, payload, 400, "missing_affine",
                          file="followup", field="qform_code,sform_code")

    def test_big_endian_baseline_singular_affine_tagged(self):
        raw_b = build_nifti(endian=">", datatype="int16", dims=(2, 2, 2),
                            data_fn=b_data, transform="sform",
                            srow_x=(0.0, 0.0, 0.0, 0.0))
        status, payload = post_compare(
            self.base, raw_b, followup_file(),
            [{"id": 1, "point": [0.0, 0.0, 0.0]}])
        self.assert_error(status, payload, 400, "singular_affine",
                          file="baseline", field="srow")

    # -- form-level errors -----------------------------------------------------
    def test_missing_side_parts(self):
        good = baseline_file()
        boundary = BOUNDARY.encode()
        body = b"\r\n".join([
            b"--" + boundary,
            b'Content-Disposition: form-data; name="baseline"; filename="b.nii"',
            b"Content-Type: application/octet-stream",
            b"",
            good,
            b"--" + boundary,
            b'Content-Disposition: form-data; name="points"',
            b"Content-Type: application/json",
            b"",
            b'[{"id": 1, "point": [0, 0, 0]}]',
            b"--" + boundary + b"--",
            b"",
        ])
        status, payload = self.post_raw(
            body, f"multipart/form-data; boundary={BOUNDARY}")
        self.assert_error(status, payload, 400, "missing_file", field="followup")

    def test_duplicate_side_part(self):
        good = followup_file()
        boundary = BOUNDARY.encode()
        base_part = b"\r\n".join([
            b"--" + boundary,
            b'Content-Disposition: form-data; name="baseline"; filename="b.nii"',
            b"Content-Type: application/octet-stream",
            b"",
            baseline_file(),
        ])
        second = b"\r\n".join([
            b"--" + boundary,
            b'Content-Disposition: form-data; name="baseline"; filename="b2.nii"',
            b"Content-Type: application/octet-stream",
            b"",
            baseline_file(),
        ])
        tail = b"\r\n".join([
            b"--" + boundary,
            b'Content-Disposition: form-data; name="followup"; filename="f.nii"',
            b"Content-Type: application/octet-stream",
            b"",
            good,
            b"--" + boundary,
            b'Content-Disposition: form-data; name="points"',
            b"Content-Type: application/json",
            b"",
            b'[{"id": 1, "point": [0, 0, 0]}]',
            b"--" + boundary + b"--",
            b"",
        ])
        body = base_part + b"\r\n" + second + b"\r\n" + tail
        status, payload = self.post_raw(
            body, f"multipart/form-data; boundary={BOUNDARY}")
        self.assert_error(status, payload, 400, "multiple_files", field="baseline")

    def test_unexpected_file_part(self):
        good = baseline_file()
        body = build_compare_multipart(good, followup_file(),
                                       [{"id": 1, "point": [0, 0, 0]}])
        extra = (b"--" + BOUNDARY.encode() + b"\r\n"
                 b'Content-Disposition: form-data; name="other"; filename="x.nii"\r\n'
                 b"Content-Type: application/octet-stream\r\n\r\n" + good + b"\r\n")
        body = body.replace(b"--" + BOUNDARY.encode() + b"--\r\n",
                            extra + b"--" + BOUNDARY.encode() + b"--\r\n")
        status, payload = self.post_raw(
            body, f"multipart/form-data; boundary={BOUNDARY}")
        self.assert_error(status, payload, 400, "unexpected_file", field="other")

    def test_side_file_too_large(self):
        huge = b"\x00" * (MAX_FILE_BYTES + 1)
        status, payload = post_compare(
            self.base, baseline_file(), huge,
            [{"id": 1, "point": [0, 0, 0]}])
        self.assert_error(status, payload, 413, "file_too_large",
                          file="followup", field="followup")

    def test_invalid_points_still_rejected(self):
        status, payload = post_compare(
            self.base, baseline_file(), followup_file(), [])
        self.assert_error(status, payload, 400, "invalid_points", field="points")

    # -- routing ---------------------------------------------------------------
    def test_get_on_compare_path_is_405(self):
        status, payload = get_json(self.base, "/api/nifti/compare")
        self.assertEqual(status, 405)
        self.assertEqual(payload["error"]["code"], "method_not_allowed")

    def test_sample_endpoint_still_works(self):
        from verify.httpclient import post_sample
        status, payload = post_sample(
            self.base, baseline_file(),
            [{"id": 1, "point": baseline_world((1.0, 1.0, 1.0))}])
        self.assertEqual(status, 200)
        self.assertEqual(payload["transform"], "sform")
        self.assertEqual(payload["results"][0]["intensity"], 111.0)


if __name__ == "__main__":
    unittest.main()
