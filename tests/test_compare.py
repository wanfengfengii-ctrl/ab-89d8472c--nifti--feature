"""End-to-end tests for POST /api/nifti/compare against a live server."""
import http.client
import json
import logging
import math
import threading
import unittest
from http.server import ThreadingHTTPServer

from app.server import Handler, MAX_COMPARE_BODY_BYTES, MAX_FILE_BYTES
from verify.httpclient import get_json, post_compare
from verify.nifti_gen import build_nifti

# Baseline grid: diag(2,3,4) + t(10,20,30) on dims (4,5,6), linear data.
SFORM_A = ((2.0, 0.0, 0.0, 10.0), (0.0, 3.0, 0.0, 20.0), (0.0, 0.0, 4.0, 30.0))
# Followup grid: rotz(+90 deg) @ diag(1,2,5) + t(14,20,30) on dims (6,5,4),
# constant data.  voxel (i,j,k) -> world (14-2j, 20+i, 30+5k).
F_DIMS = (6, 5, 4)
QUAT_90Z = (0.0, 0.0, math.sqrt(0.5))


def data_fn(i, j, k):
    return i + 10 * j + 100 * k


def baseline_file(**overrides):
    kw = dict(endian="<", datatype="int16", dims=(4, 5, 6), data_fn=data_fn,
              transform="sform", slope=2.0, inter=5.0,
              srow_x=SFORM_A[0], srow_y=SFORM_A[1], srow_z=SFORM_A[2])
    kw.update(overrides)
    return build_nifti(**kw)


def followup_file(**overrides):
    kw = dict(endian=">", datatype="float32", dims=F_DIMS,
              data_fn=lambda i, j, k: 7.0, transform="qform",
              slope=2.0, inter=1.0,
              quatern=QUAT_90Z, qoffset=(14.0, 20.0, 30.0),
              pixdim=(1.0, 1.0, 2.0, 5.0))
    kw.update(overrides)
    return build_nifti(**kw)


class CompareTests(unittest.TestCase):
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

    # -- happy paths -----------------------------------------------------------
    def test_mixed_grids_both_sides_and_delta(self):
        points = [
            {"id": 9, "point": [12.0, 22.0, 35.0]},
            {"id": 4, "point": [11.0, 24.0, 40.0]},
        ]
        status, payload = post_compare(self.base, baseline_file(), followup_file(),
                                       points)
        self.assertEqual(status, 200)
        self.assertEqual(payload["baseline"]["transform"], "sform")
        self.assertEqual(payload["followup"]["transform"], "qform")
        self.assertEqual([r["id"] for r in payload["results"]], [9, 4])

        r9, r4 = payload["results"]
        # world (12,22,35): baseline voxel (1, 2/3, 1.25); followup voxel (2,1,1)
        self.assertEqual(r9["status"], "ok")
        self.assertEqual(r9["baseline"]["transform"], "sform")
        for got, want in zip(r9["baseline"]["voxel"], (1.0, 2.0 / 3.0, 1.25)):
            self.assertAlmostEqual(got, want, places=6)
        self.assertAlmostEqual(r9["baseline"]["intensity"],
                               (1.0 + 20.0 / 3.0 + 125.0) * 2.0 + 5.0, places=3)
        self.assertEqual(r9["followup"]["transform"], "qform")
        for got, want in zip(r9["followup"]["voxel"], (2.0, 1.0, 1.0)):
            self.assertAlmostEqual(got, want, places=4)
        self.assertAlmostEqual(r9["followup"]["intensity"], 15.0, places=3)
        self.assertAlmostEqual(
            r9["delta"],
            r9["followup"]["intensity"] - r9["baseline"]["intensity"], places=9)

        # world (11,24,40): baseline voxel (0.5, 4/3, 2.5); followup (4, 1.5, 2)
        self.assertEqual(r4["status"], "ok")
        for got, want in zip(r4["baseline"]["voxel"], (0.5, 4.0 / 3.0, 2.5)):
            self.assertAlmostEqual(got, want, places=6)
        for got, want in zip(r4["followup"]["voxel"], (4.0, 1.5, 2.0)):
            self.assertAlmostEqual(got, want, places=4)
        self.assertAlmostEqual(r4["followup"]["intensity"], 15.0, places=3)

    def test_per_side_out_of_bounds(self):
        points = [
            {"id": 1, "point": [12.0, 24.0, 40.0]},   # in bounds on both
            {"id": 2, "point": [15.0, 22.0, 35.0]},   # outside followup only
            {"id": 3, "point": [8.0, 22.0, 35.0]},    # outside baseline only
            {"id": 4, "point": [20.0, 22.0, 35.0]},   # outside both
        ]
        status, payload = post_compare(self.base, baseline_file(), followup_file(),
                                       points)
        self.assertEqual(status, 200)
        r1, r2, r3, r4 = payload["results"]
        self.assertEqual(r1["status"], "ok")
        self.assertEqual(r2["status"], "error")
        self.assertEqual(r2["error"]["code"], "out_of_bounds")
        self.assertEqual(r2["error"]["side"], "followup")
        self.assertIn("voxel", r2["error"])
        self.assertEqual(r3["status"], "error")
        self.assertEqual(r3["error"]["code"], "out_of_bounds")
        self.assertEqual(r3["error"]["side"], "baseline")
        # both sides out of bounds: the baseline side is reported
        self.assertEqual(r4["status"], "error")
        self.assertEqual(r4["error"]["side"], "baseline")

    def test_non_finite_data_per_side(self):
        base = baseline_file(datatype="float32",
                             data_fn=lambda i, j, k: float("nan")
                             if (i, j, k) == (1, 1, 1) else data_fn(i, j, k))
        foll = followup_file(data_fn=lambda i, j, k: float("nan")
                             if (i, j, k) == (4, 2, 3) else 7.0)
        points = [
            {"id": 1, "point": [12.0, 23.0, 34.0]},  # baseline voxel (1,1,1) NaN
            {"id": 2, "point": [10.0, 24.0, 45.0]},  # followup voxel (4,2,3) NaN
            {"id": 3, "point": [12.0, 24.0, 40.0]},  # clean on both sides
        ]
        status, payload = post_compare(self.base, base, foll, points)
        self.assertEqual(status, 200)
        r1, r2, r3 = payload["results"]
        self.assertEqual(r1["status"], "error")
        self.assertEqual(r1["error"]["code"], "non_finite_data")
        self.assertEqual(r1["error"]["side"], "baseline")
        self.assertEqual(r2["status"], "error")
        self.assertEqual(r2["error"]["code"], "non_finite_data")
        self.assertEqual(r2["error"]["side"], "followup")
        self.assertEqual(r3["status"], "ok")

    # -- file-level errors locate the failing upload -----------------------------
    def assert_error(self, status, payload, want_status, code, field=None,
                     side=None):
        self.assertEqual(status, want_status, payload)
        err = payload["error"]
        self.assertEqual(err["code"], code)
        if field is not None:
            self.assertEqual(err["field"], field)
        if side is not None:
            self.assertEqual(err["side"], side)

    def test_parse_errors_carry_side_and_header_field(self):
        one = [{"id": 1, "point": [12.0, 22.0, 35.0]}]
        status, payload = post_compare(self.base, baseline_file(extra=b"\x00"),
                                       followup_file(), one)
        self.assert_error(status, payload, 400, "trailing_bytes", "file",
                          "baseline")
        status, payload = post_compare(self.base, baseline_file(),
                                       followup_file(magic=b"ni1\x00"), one)
        self.assert_error(status, payload, 400, "unsupported_magic", "magic",
                          "followup")
        status, payload = post_compare(self.base, baseline_file(),
                                       followup_file(slope=float("nan")), one)
        self.assert_error(status, payload, 400, "non_finite_scaling",
                          "scl_slope", "followup")

    def test_missing_and_duplicate_parts(self):
        one = [{"id": 1, "point": [12.0, 22.0, 35.0]}]
        status, payload = post_compare(self.base, baseline_file(), followup_file(),
                                       one, followup_field="file")
        self.assert_error(status, payload, 400, "missing_file", "followup",
                          "followup")
        status, payload = post_compare(self.base, baseline_file(), followup_file(),
                                       one, baseline_field="file")
        self.assert_error(status, payload, 400, "missing_file", "baseline",
                          "baseline")
        status, payload = post_compare(self.base, baseline_file(), followup_file(),
                                       one, followup_field="baseline")
        self.assert_error(status, payload, 400, "multiple_files", "baseline",
                          "baseline")

    def test_oversized_side_rejected(self):
        big = b"\x00" * (MAX_FILE_BYTES + 1)
        one = [{"id": 1, "point": [0.0, 0.0, 0.0]}]
        status, payload = post_compare(self.base, big, followup_file(), one)
        self.assert_error(status, payload, 413, "file_too_large", "baseline",
                          "baseline")
        status, payload = post_compare(self.base, baseline_file(), big, one)
        self.assert_error(status, payload, 413, "file_too_large", "followup",
                          "followup")

    def test_body_limit_rejected_before_read(self):
        # Content-Length above the two-file compare limit is rejected during
        # Expect: 100-continue negotiation, before the body is streamed.
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            conn.putrequest("POST", "/api/nifti/compare")
            conn.putheader("Content-Type", "multipart/form-data; boundary=x")
            conn.putheader("Content-Length", str(MAX_COMPARE_BODY_BYTES + 1))
            conn.putheader("Expect", "100-continue")
            conn.endheaders()
            resp = conn.getresponse()
            raw = resp.read()
            self.assertEqual(resp.status, 413)
            self.assertEqual(json.loads(raw)["error"]["code"], "file_too_large")
        finally:
            conn.close()

    # -- shared validation is reused ---------------------------------------------
    def test_invalid_points(self):
        status, payload = post_compare(self.base, baseline_file(), followup_file(),
                                       [])
        self.assert_error(status, payload, 400, "invalid_points", "points")
        self.assertNotIn("side", payload["error"])

    def test_missing_points_field(self):
        status, payload = post_compare(self.base, baseline_file(), followup_file(),
                                       [], points_field="other")
        self.assert_error(status, payload, 400, "missing_points", "points")

    def test_wrong_content_type(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            conn.request("POST", "/api/nifti/compare", body=b"hello",
                         headers={"Content-Type": "text/plain"})
            resp = conn.getresponse()
            payload = json.loads(resp.read())
            self.assertEqual(resp.status, 415)
            self.assertEqual(payload["error"]["code"], "invalid_multipart")
        finally:
            conn.close()

    def test_get_on_compare_path_is_405(self):
        status, payload = get_json(self.base, "/api/nifti/compare")
        self.assertEqual(status, 405)
        self.assertEqual(payload["error"]["code"], "method_not_allowed")


if __name__ == "__main__":
    unittest.main()
