"""Stable error types for the NIfTI sampling API.

Two error levels exist:

* :class:`ApiError` — request/file-level failure.  Rendered as a 4xx/5xx
  JSON body ``{"error": {"code", "message", "field"}}`` where ``field``
  locates the offending NIfTI header field or form field.
* :class:`PointError` — per-point sampling failure.  Reported inside a
  200 response next to the point id so one bad point never hides the
  remaining results.
"""


class ApiError(Exception):
    """Request/file-level error rendered as a stable JSON error object."""

    def __init__(self, code, message, field=None, status=400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field
        self.status = status

    def to_dict(self):
        err = {"code": self.code, "message": self.message}
        if self.field is not None:
            err["field"] = self.field
        return {"error": err}


class PointError(Exception):
    """Per-point sampling failure (reported inside a 200 response)."""

    def __init__(self, code, message, voxel=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.voxel = voxel
