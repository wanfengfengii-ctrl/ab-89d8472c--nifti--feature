"""Trilinear interpolation of scaled voxel data in continuous voxel space."""
from __future__ import annotations

import math

from .errors import PointError

# Tolerance (in voxels) absorbing float64 round-off when a world coordinate
# that sits exactly on the voxel-center boundary is mapped back through the
# inverse affine.  Points further outside than this are rejected.
BOUND_TOL = 1e-6


def _axis_stencil(fc, n):
    """Return ``((index, weight), ...)`` for one axis of the trilinear stencil.

    ``fc`` must lie in the closed voxel-center domain ``[0, n-1]`` (plus the
    round-off tolerance).  On the boundary the axis collapses to the unique
    endpoint with weight 1.
    """
    hi = n - 1
    if not math.isfinite(fc) or fc < -BOUND_TOL or fc > hi + BOUND_TOL:
        raise PointError(
            "out_of_bounds",
            f"continuous voxel coordinate {fc!r} is outside the closed "
            f"voxel-center domain [0, {hi}]",
        )
    fc = min(max(fc, 0.0), float(hi))
    i0 = math.floor(fc)
    t = fc - i0
    if i0 >= hi:  # boundary: axis fixed to the unique endpoint
        return ((hi, 1.0),)
    return ((i0, 1.0 - t), (i0 + 1, t))


def sample_point(volume, point):
    """Sample ``volume`` at RAS world point ``(x, y, z)``.

    Returns ``((fi, fj, fk), intensity)``: the continuous voxel coordinate
    produced by the inverse affine and the trilinearly interpolated,
    scaling-applied intensity.  Raises :class:`PointError` for out-of-bounds
    points and for non-finite scaled data inside the stencil.
    """
    x, y, z = point
    inv = volume.inverse
    fi = inv[0][0] * x + inv[0][1] * y + inv[0][2] * z + inv[0][3]
    fj = inv[1][0] * x + inv[1][1] * y + inv[1][2] * z + inv[1][3]
    fk = inv[2][0] * x + inv[2][1] * y + inv[2][2] * z + inv[2][3]
    voxel = (fi, fj, fk)

    nx, ny, nz = volume.dims
    try:
        xs = _axis_stencil(fi, nx)
        ys = _axis_stencil(fj, ny)
        zs = _axis_stencil(fk, nz)
    except PointError as exc:
        exc.voxel = voxel
        raise

    total = 0.0
    for ix, wx in xs:
        if wx == 0.0:
            continue
        for iy, wy in ys:
            if wy == 0.0:
                continue
            for iz, wz in zs:
                w = wx * wy * wz
                if w == 0.0:
                    continue  # zero-weight neighbours do not participate
                value = volume.value_at(ix, iy, iz)
                if not math.isfinite(value):
                    raise PointError(
                        "non_finite_data",
                        f"non-finite scaled data at voxel ({ix}, {iy}, {iz})",
                        voxel=voxel,
                    )
                total += w * value
    return voxel, total
