"""Minimal multipart/form-data client helpers (stdlib only)."""
from __future__ import annotations

import http.client
import json
from urllib.parse import urlsplit

BOUNDARY = "nifti-verify-7f3a9c51e2b44d08a1"


def _points_bytes(points):
    """JSON-encode ``points`` unless raw bytes/str were given (malformed-input
    tests send those as-is)."""
    if not isinstance(points, (bytes, str)):
        points = json.dumps(points)
    if isinstance(points, str):
        points = points.encode("utf-8")
    return points


def _file_segment(boundary, field, filename):
    return [b"--" + boundary,
            b'Content-Disposition: form-data; name="%s"; filename="%s"'
            % (field.encode("utf-8"), filename.encode("utf-8")),
            b"Content-Type: application/octet-stream",
            b""]


def build_multipart(file_bytes, points, *, file_field="file",
                    filename="vol.nii", points_field="points"):
    """Assemble a multipart/form-data body with one file part and one
    ``points`` field."""
    boundary = BOUNDARY.encode("ascii")
    return b"\r\n".join(
        _file_segment(boundary, file_field, filename) + [
            file_bytes,
            b"--" + boundary,
            b'Content-Disposition: form-data; name="%s"' % points_field.encode("utf-8"),
            b"Content-Type: application/json",
            b"",
            _points_bytes(points),
            b"--" + boundary + b"--",
            b"",
        ])


def build_multipart_compare(baseline_bytes, followup_bytes, points, *,
                            baseline_field="baseline", followup_field="followup",
                            points_field="points"):
    """Assemble a multipart/form-data body for ``POST /api/nifti/compare``:
    one ``baseline`` file part, one ``followup`` file part and one ``points``
    field.  Field-name overrides allow crafting malformed forms for negative
    tests."""
    boundary = BOUNDARY.encode("ascii")
    return b"\r\n".join(
        _file_segment(boundary, baseline_field, "baseline.nii") + [baseline_bytes]
        + _file_segment(boundary, followup_field, "followup.nii") + [followup_bytes]
        + [
            b"--" + boundary,
            b'Content-Disposition: form-data; name="%s"' % points_field.encode("utf-8"),
            b"Content-Type: application/json",
            b"",
            _points_bytes(points),
            b"--" + boundary + b"--",
            b"",
        ])


def _post_multipart(base_url, path, body, timeout):
    """POST ``body`` to ``path``; returns ``(status, parsed_json_or_None)``."""
    url = urlsplit(base_url)
    conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=timeout)
    try:
        conn.request(
            "POST", path,
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
        )
        resp = conn.getresponse()
        raw = resp.read()
        status = resp.status
    finally:
        conn.close()
    try:
        return status, json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        return status, None


def post_sample(base_url, file_bytes, points, *, timeout=30):
    """POST /api/nifti/sample; returns ``(status, parsed_json_or_None)``."""
    return _post_multipart(base_url, "/api/nifti/sample",
                           build_multipart(file_bytes, points), timeout)


def post_compare(base_url, baseline_bytes, followup_bytes, points, *,
                 timeout=30, **build_kw):
    """POST /api/nifti/compare; returns ``(status, parsed_json_or_None)``.
    Extra keyword arguments are forwarded to :func:`build_multipart_compare`."""
    return _post_multipart(
        base_url, "/api/nifti/compare",
        build_multipart_compare(baseline_bytes, followup_bytes, points, **build_kw),
        timeout)


def get_json(base_url, path, *, timeout=5):
    """GET a JSON document; returns ``(status, parsed_json_or_None)``."""
    url = urlsplit(base_url)
    conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        raw = resp.read()
        status = resp.status
    finally:
        conn.close()
    try:
        return status, json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        return status, None
