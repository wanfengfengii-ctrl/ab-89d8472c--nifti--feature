"""End-to-end API tests against a live in-process HTTP server."""
import http.client
import json
import logging
import math
import threading
import unittest
from http.server import ThreadingHTTPServer

from app.server import Handler, MAX_FILE_BYTES
from verify.httpclient import BOUNDARY, build_multipart, get_json, post_sample
from verify.nifti_gen import build_nifti

SFORM = ((2.0, 0.0, 0.0, 10.0), (0.0, 3.0, 0.0, 20.0), (0.0, 0.0, 4.0, 30.0))


def data_fn(i, j, k):
    return i + 10 * j + 100 * k


def world_of(voxel):
    i, j, k = voxel
    return [2.0 * i + 10.0, 3.0 * j + 20.0, 4.0 * k + 30.0]


def sform_file(**overrides):
    kw = dict(endian="<", datatype="int16", dims=(4, 5, 6), data_fn=data_fn,
              transform="sform",
              srow_x=SFORM[0], srow_y=SFORM[1], srow_z=SFORM[2])
    kw.update(overrides)
    return build_nifti(**kw)


class ApiTests(unittest.TestCase):
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
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            conn.request("POST", "/api/nifti/sample", body=body,
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

    # -- happy paths ---------------------------------------------------------
    def test_sform_sampling_and_order(self):
        points = [
            {"id": 7, "point": world_of((1.0, 1.0, 1.0))},
            {"id": 3, "point": world_of((0.5, 0.5, 0.5))},
            {"id": 9, "point": world_of((3.0, 4.0, 5.0))},
        ]
        status, payload = post_sample(self.base, sform_file(), points)
        self.assertEqual(status, 200)
        self.assertEqual(payload["transform"], "sform")
        self.assertEqual([r["id"] for r in payload["results"]], [7, 3, 9])
        r7, r3, r9 = payload["results"]
        self.assertEqual(r7["status"], "ok")
        self.assertEqual(r7["voxel"], [1.0, 1.0, 1.0])
        self.assertEqual(r7["intensity"], 111.0)
        self.assertEqual(r7["transform"], "sform")
        self.assertEqual(r3["status"], "ok")
        self.assertAlmostEqual(r3["intensity"], 55.5)
        self.assertEqual(r9["status"], "ok")
        self.assertEqual(r9["intensity"], 543.0)

    def test_big_endian_float32_qform(self):
        quat = (0.0, 0.0, math.sqrt(0.5))  # rotz(+90 deg)
        raw = build_nifti(endian=">", datatype="float32", dims=(4, 5, 6),
                          data_fn=data_fn, transform="qform",
                          quatern=quat, qoffset=(10.0, 20.0, 30.0),
                          pixdim=(1.0, 2.0, 3.0, 4.0))
        # intended affine: [[0,-3,0,10],[2,0,0,20],[0,0,4,30]]
        world = [0.0 * 1 - 3.0 * 2 + 10.0, 2.0 * 1 + 20.0, 4.0 * 3 + 30.0]
        status, payload = post_sample(self.base, raw,
                                      [{"id": 1, "point": world}])
        self.assertEqual(status, 200)
        self.assertEqual(payload["transform"], "qform")
        (result,) = payload["results"]
        self.assertEqual(result["status"], "ok")
        for got, want in zip(result["voxel"], (1.0, 2.0, 3.0)):
            self.assertAlmostEqual(got, want, places=4)
        self.assertAlmostEqual(result["intensity"], 321.0, places=3)

    def test_per_point_errors_do_not_fail_request(self):
        points = [
            {"id": 1, "point": world_of((1.0, 1.0, 1.0))},
            {"id": 2, "point": world_of((-1.0, 0.0, 0.0))},
            {"id": 3, "point": world_of((0.0, 0.0, 6.0))},
        ]
        status, payload = post_sample(self.base, sform_file(), points)
        self.assertEqual(status, 200)
        ok, oob1, oob2 = payload["results"]
        self.assertEqual(ok["status"], "ok")
        self.assertEqual(oob1["status"], "error")
        self.assertEqual(oob1["error"]["code"], "out_of_bounds")
        self.assertIn("voxel", oob1["error"])
        self.assertEqual(oob2["error"]["code"], "out_of_bounds")

    def test_non_finite_data_per_point(self):
        raw = sform_file(datatype="float32",
                         data_fn=lambda i, j, k: float("nan")
                         if (i, j, k) == (1, 1, 1) else data_fn(i, j, k))
        points = [{"id": 5, "point": world_of((0.5, 0.5, 0.5))}]
        status, payload = post_sample(self.base, raw, points)
        self.assertEqual(status, 200)
        (result,) = payload["results"]
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "non_finite_data")

    # -- structural errors -----------------------------------------------------
    def assert_error(self, status, payload, want_status, code, field=None):
        self.assertEqual(status, want_status, payload)
        err = payload["error"]
        self.assertEqual(err["code"], code)
        if field is not None:
            self.assertEqual(err["field"], field)

    def test_structural_errors(self):
        cases = [
            (sform_file(extra=b"\x00"), "trailing_bytes", "file"),
            (sform_file(truncate=4), "payload_length_mismatch", "file"),
            (sform_file(qform_code=0, sform_code=0), "missing_affine",
             "qform_code,sform_code"),
            (sform_file(srow_x=(0, 0, 0, 0)), "singular_affine", "srow"),
            (sform_file(datatype_code=8), "unsupported_datatype", "datatype"),
            (sform_file(slope=float("nan")), "non_finite_scaling", "scl_slope"),
            (sform_file(dim0=4), "invalid_dimensions", "dim"),
            (sform_file(magic=b"ni1\x00"), "unsupported_magic", "magic"),
            (sform_file(vox_offset=100.0), "invalid_vox_offset", "vox_offset"),
            (sform_file(bitpix=8), "bitpix_mismatch", "bitpix"),
            (b"\x00" * 10, "header_too_short", "file"),
        ]
        for raw, code, field in cases:
            with self.subTest(code=code):
                status, payload = post_sample(self.base, raw,
                                              [{"id": 1, "point": [0, 0, 0]}])
                self.assert_error(status, payload, 400, code, field)

    def test_big_endian_structural_error(self):
        raw = build_nifti(endian=">", datatype="float32", dims=(2, 2, 2),
                          data_fn=data_fn, extra=b"zz")
        status, payload = post_sample(self.base, raw,
                                      [{"id": 1, "point": [0, 0, 0]}])
        self.assert_error(status, payload, 400, "trailing_bytes", "file")

    # -- points validation -----------------------------------------------------
    def test_invalid_points(self):
        raw = sform_file()
        cases = [
            [],
            [{"id": i, "point": [0, 0, 0]} for i in range(257)],
            [{"id": 1, "point": [0, 0, 0]}, {"id": 1, "point": [1, 1, 1]}],
            [{"id": 1, "point": [float("nan"), 0, 0]}],
            [{"id": -1, "point": [0, 0, 0]}],
            [{"id": 1, "point": [0, 0]}],
            [{"id": 1}],
            b"[{",
            b"{}",
        ]
        for points in cases:
            with self.subTest(points=str(points)[:60]):
                status, payload = post_sample(self.base, raw, points)
                self.assert_error(status, payload, 400, "invalid_points", "points")

    # -- form-level errors -----------------------------------------------------
    def test_missing_file_part(self):
        body = (b"--" + BOUNDARY.encode() + b"\r\n"
                b'Content-Disposition: form-data; name="points"\r\n\r\n'
                b'[{"id": 1, "point": [0, 0, 0]}]\r\n'
                b"--" + BOUNDARY.encode() + b"--\r\n")
        status, payload = self.post_raw(
            body, f"multipart/form-data; boundary={BOUNDARY}")
        self.assert_error(status, payload, 400, "missing_file", "file")

    def test_multiple_file_parts(self):
        raw = sform_file()
        body = build_multipart(raw, [{"id": 1, "point": [0, 0, 0]}])
        # inject a second file part before the closing boundary
        extra = (b"--" + BOUNDARY.encode() + b"\r\n"
                 b'Content-Disposition: form-data; name="f2"; filename="b.nii"\r\n'
                 b"Content-Type: application/octet-stream\r\n\r\n" + raw + b"\r\n")
        body = body.replace(b"--" + BOUNDARY.encode() + b"--\r\n",
                            extra + b"--" + BOUNDARY.encode() + b"--\r\n")
        status, payload = self.post_raw(
            body, f"multipart/form-data; boundary={BOUNDARY}")
        self.assert_error(status, payload, 400, "multiple_files", "file")

    def test_missing_points_field(self):
        body = (b"--" + BOUNDARY.encode() + b"\r\n"
                b'Content-Disposition: form-data; name="file"; filename="v.nii"\r\n'
                b"Content-Type: application/octet-stream\r\n\r\n" + sform_file() +
                b"\r\n--" + BOUNDARY.encode() + b"--\r\n")
        status, payload = self.post_raw(
            body, f"multipart/form-data; boundary={BOUNDARY}")
        self.assert_error(status, payload, 400, "missing_points", "points")

    def test_wrong_content_type(self):
        status, payload = self.post_raw(b"hello", "text/plain")
        self.assert_error(status, payload, 415, "invalid_multipart")

    def test_file_too_large(self):
        raw = b"\x00" * (MAX_FILE_BYTES + 1)
        status, payload = post_sample(self.base, raw,
                                      [{"id": 1, "point": [0, 0, 0]}])
        self.assert_error(status, payload, 413, "file_too_large", "file")

    # -- routing ----------------------------------------------------------------
    def test_healthz(self):
        status, payload = get_json(self.base, "/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok", "ready": True})

    def test_unknown_routes(self):
        status, _ = get_json(self.base, "/nope")
        self.assertEqual(status, 404)
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request("POST", "/nope", body=b"{}")
            resp = conn.getresponse()
            resp.read()
            self.assertEqual(resp.status, 404)
        finally:
            conn.close()

    def test_get_on_sample_path_is_405(self):
        status, payload = get_json(self.base, "/api/nifti/sample")
        self.assertEqual(status, 405)
        self.assertEqual(payload["error"]["code"], "method_not_allowed")


if __name__ == "__main__":
    unittest.main()
