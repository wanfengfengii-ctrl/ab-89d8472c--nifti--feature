"""Minimal multipart/form-data client helpers (stdlib only)."""
from __future__ import annotations

import http.client
import json
from urllib.parse import urlsplit

BOUNDARY = "nifti-verify-7f3a9c51e2b44d08a1"


def build_multipart(file_bytes, points, *, file_field="file",
                    filename="vol.nii", points_field="points"):
    """Assemble a multipart/form-data body with one file part and one
    ``points`` field.  ``points`` may be a Python object (JSON-encoded) or
    raw bytes/str (sent as-is, for malformed-input tests)."""
    if not isinstance(points, (bytes, str)):
        points = json.dumps(points)
    if isinstance(points, str):
        points = points.encode("utf-8")
    boundary = BOUNDARY.encode("ascii")
    return b"\r\n".join([
        b"--" + boundary,
        b'Content-Disposition: form-data; name="%s"; filename="%s"'
        % (file_field.encode("utf-8"), filename.encode("utf-8")),
        b"Content-Type: application/octet-stream",
        b"",
        file_bytes,
        b"--" + boundary,
        b'Content-Disposition: form-data; name="%s"' % points_field.encode("utf-8"),
        b"Content-Type: application/json",
        b"",
        points,
        b"--" + boundary + b"--",
        b"",
    ])


def post_sample(base_url, file_bytes, points, *, timeout=30):
    """POST /api/nifti/sample; returns ``(status, parsed_json_or_None)``."""
    url = urlsplit(base_url)
    conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=timeout)
    try:
        conn.request(
            "POST", "/api/nifti/sample",
            body=build_multipart(file_bytes, points),
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
