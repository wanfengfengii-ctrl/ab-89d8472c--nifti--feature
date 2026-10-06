"""One-shot verification service.

Aggregates three stages into the process exit code (bit flags):

* bit 0 (1): unit/integration tests (``tests/``) failed
* bit 1 (2): image build validation failed (manifest, runtime, imports)
* bit 2 (4): API smoke failed (big/little endian x sform/qform x
  int16/float32 sample matrix plus negative cases, and mixed
  endianness/datatype/affine/grid compare checks)

Exit code 0 means every stage passed.  The smoke stage waits for the API
service to report readiness on ``/healthz`` before sending traffic.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verify.httpclient import post_compare, post_sample  # noqa: E402
from verify.nifti_gen import build_nifti  # noqa: E402

EXIT_TESTS = 1
EXIT_IMAGE = 2
EXIT_SMOKE = 4

# ---------------------------------------------------------------------------
# stage 1: code tests
# ---------------------------------------------------------------------------

def stage_tests():
    print("== stage 1/3: code tests ==", flush=True)
    suite = unittest.TestLoader().discover(start_dir=str(ROOT / "tests"),
                                           top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    ok = result.wasSuccessful()
    print(f"stage tests: {'PASS' if ok else 'FAIL'} "
          f"({result.testsRun} tests, {len(result.failures)} failures, "
          f"{len(result.errors)} errors)", flush=True)
    return ok


# ---------------------------------------------------------------------------
# stage 2: image build validation
# ---------------------------------------------------------------------------

def stage_image():
    print("== stage 2/3: image build validation ==", flush=True)
    checks = []

    def check(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    manifest_path = ROOT / "image-manifest.json"
    manifest = None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        check("image manifest present", True, str(manifest_path))
    except Exception as exc:
        check("image manifest present", False, f"{manifest_path}: {exc}")
    if manifest is not None:
        check("manifest name", manifest.get("name") == "nifti-sampler",
              repr(manifest.get("name")))
        check("manifest version", bool(manifest.get("version")),
              repr(manifest.get("version")))
        env_version = os.environ.get("APP_VERSION")
        if env_version is not None:
            check("manifest version matches APP_VERSION env",
                  manifest.get("version") == env_version,
                  f"manifest={manifest.get('version')!r} env={env_version!r}")
    check("python >= 3.11", sys.version_info >= (3, 11), sys.version.split()[0])
    try:
        import app.nifti  # noqa: F401
        import app.sampling  # noqa: F401
        import app.server  # noqa: F401
        check("app modules importable", True)
    except Exception as exc:
        check("app modules importable", False, repr(exc))
    try:
        app.server.self_check()
        check("sampler self-check", True)
    except Exception as exc:
        check("sampler self-check", False, repr(exc))

    ok = True
    for name, passed, detail in checks:
        ok = ok and passed
        suffix = f" — {detail}" if detail and not passed else ""
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}{suffix}", flush=True)
    print(f"stage image: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


# ---------------------------------------------------------------------------
# stage 3: API smoke
# ---------------------------------------------------------------------------

SFORM_AFFINE = ((2.0, 0.0, 0.0, 10.0),
                (0.0, 3.0, 0.0, 20.0),
                (0.0, 0.0, 4.0, 30.0))
# rotz(+90 deg) @ diag(2,3,4) + translation, written via the qform fields.
QFORM_AFFINE = ((0.0, -3.0, 0.0, 10.0),
                (2.0, 0.0, 0.0, 20.0),
                (0.0, 0.0, 4.0, 30.0))
QUATERN_90Z = (0.0, 0.0, math.sqrt(0.5))
DIMS = (4, 5, 6)
SLOPE, INTER = 2.0, 5.0


def data_fn(i, j, k):
    return i + 10 * j + 100 * k


def world_of(affine, voxel):
    i, j, k = voxel
    return tuple(affine[r][0] * i + affine[r][1] * j + affine[r][2] * k + affine[r][3]
                 for r in range(3))


def avg8():
    return sum(data_fn(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)) / 8.0


def wait_healthy(base_url, timeout_s=90.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/healthz", timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def stage_smoke(base_url):
    print("== stage 3/3: API smoke ==", flush=True)
    failures = []
    total = 0

    def check(name, cond, detail=""):
        nonlocal total
        total += 1
        if not cond:
            failures.append(name)
        suffix = f" — {detail}" if detail and not cond else ""
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{suffix}", flush=True)
        return bool(cond)

    check("service becomes healthy", wait_healthy(base_url), base_url)
    if failures:
        return False

    # -- sample matrix: endianness x transform x datatype -------------------
    combos = [(e, t, d) for e in ("<", ">") for t in ("sform", "qform")
              for d in ("int16", "float32")]
    expected_raw = {1: data_fn(1, 2, 3), 2: avg8(), 3: data_fn(3, 4, 5)}
    expected_vox = {1: (1.0, 2.0, 3.0), 2: (0.5, 0.5, 0.5), 3: (3.0, 4.0, 5.0)}
    for endian, transform, datatype in combos:
        tag = f"{'LE' if endian == '<' else 'BE'}/{transform}/{datatype}"
        affine = SFORM_AFFINE if transform == "sform" else QFORM_AFFINE
        raw = build_nifti(
            endian=endian, datatype=datatype, dims=DIMS, data_fn=data_fn,
            transform=transform, slope=SLOPE, inter=INTER,
            srow_x=SFORM_AFFINE[0], srow_y=SFORM_AFFINE[1], srow_z=SFORM_AFFINE[2],
            quatern=QUATERN_90Z, qoffset=(10.0, 20.0, 30.0),
            pixdim=(1.0, 2.0, 3.0, 4.0),
        )
        voxels = {3: (3.0, 4.0, 5.0), 1: (1.0, 2.0, 3.0), 4: (-1.0, 0.0, 0.0),
                  2: (0.5, 0.5, 0.5), 5: (0.0, 0.0, 6.0)}
        points = [{"id": pid, "point": list(world_of(affine, v))}
                  for pid, v in voxels.items()]
        status, payload = post_sample(base_url, raw, points)
        if not check(f"{tag}: HTTP 200", status == 200, f"got {status}: {payload}"):
            continue
        check(f"{tag}: transform source", isinstance(payload, dict)
              and payload.get("transform") == transform,
              repr(payload and payload.get("transform")))
        results = payload.get("results") if isinstance(payload, dict) else None
        if not check(f"{tag}: result count", isinstance(results, list)
                     and len(results) == 5, repr(results)):
            continue
        check(f"{tag}: request order preserved",
              [r.get("id") for r in results] == [3, 1, 4, 2, 5],
              repr([r.get("id") for r in results]))
        by_id = {r.get("id"): r for r in results}
        for pid in (1, 2, 3):
            r = by_id.get(pid, {})
            got_vox = r.get("voxel") or []
            vox_ok = r.get("status") == "ok" and len(got_vox) == 3 and all(
                abs(g - e) <= 1e-4 for g, e in zip(got_vox, expected_vox[pid]))
            check(f"{tag}: id {pid} voxel coords", vox_ok, repr(r))
            want = expected_raw[pid] * SLOPE + INTER
            check(f"{tag}: id {pid} intensity",
                  r.get("status") == "ok"
                  and abs((r.get("intensity") or 0.0) - want) <= 1e-3,
                  f"want {want}, got {r.get('intensity')!r}")
            check(f"{tag}: id {pid} transform field",
                  r.get("transform") == transform, repr(r.get("transform")))
        for pid in (4, 5):
            r = by_id.get(pid, {})
            check(f"{tag}: id {pid} out_of_bounds",
                  r.get("status") == "error"
                  and r.get("error", {}).get("code") == "out_of_bounds", repr(r))

    # -- negative file-structure cases --------------------------------------
    base = dict(endian="<", datatype="float32", dims=DIMS, data_fn=data_fn,
                transform="sform",
                srow_x=SFORM_AFFINE[0], srow_y=SFORM_AFFINE[1], srow_z=SFORM_AFFINE[2])
    one_point = [{"id": 1, "point": list(world_of(SFORM_AFFINE, (1.0, 1.0, 1.0)))}]
    bad_files = [
        ("truncated payload", dict(truncate=2), "payload_length_mismatch", "file"),
        ("trailing bytes", dict(extra=b"\x00"), "trailing_bytes", "file"),
        ("no affine", dict(qform_code=0, sform_code=0), "missing_affine",
         "qform_code,sform_code"),
        ("singular sform", dict(srow_x=(0.0, 0.0, 0.0, 0.0)), "singular_affine", "srow"),
        ("unsupported datatype", dict(datatype_code=2), "unsupported_datatype", "datatype"),
        ("non-finite scl_slope", dict(slope=float("nan")), "non_finite_scaling", "scl_slope"),
        ("non-3D dims", dict(dim0=4, tail_dims=(2, 1, 1, 1)), "invalid_dimensions", "dim"),
        ("Analyze magic", dict(magic=b"ni1\x00"), "unsupported_magic", "magic"),
        ("bad vox_offset", dict(vox_offset=100.0), "invalid_vox_offset", "vox_offset"),
        ("bitpix mismatch", dict(bitpix=8), "bitpix_mismatch", "bitpix"),
    ]
    for name, overrides, code, field in bad_files:
        raw = build_nifti(**{**base, **overrides})
        status, payload = post_sample(base_url, raw, one_point)
        err = payload.get("error", {}) if isinstance(payload, dict) else {}
        check(f"negative[{name}]: 400/{code}",
              status == 400 and err.get("code") == code and err.get("field") == field,
              f"got {status}: {payload}")

    # big-endian structural error is detected identically
    raw = build_nifti(**{**base, "endian": ">", "datatype": "int16", "extra": b"\x00"})
    status, payload = post_sample(base_url, raw, one_point)
    err = payload.get("error", {}) if isinstance(payload, dict) else {}
    check("negative[BE trailing]: 400/trailing_bytes",
          status == 400 and err.get("code") == "trailing_bytes",
          f"got {status}: {payload}")

    # -- non-finite voxel data -> per-point error ----------------------------
    raw = build_nifti(**{**base,
                         "data_fn": lambda i, j, k: float("nan") if (i, j, k) == (1, 1, 1)
                         else data_fn(i, j, k)})
    points = [
        {"id": 10, "point": list(world_of(SFORM_AFFINE, (0.5, 0.5, 0.5)))},
        {"id": 11, "point": list(world_of(SFORM_AFFINE, (3.0, 4.0, 5.0)))},
    ]
    status, payload = post_sample(base_url, raw, points)
    results = {r.get("id"): r for r in payload.get("results", [])} \
        if isinstance(payload, dict) else {}
    r10, r11 = results.get(10, {}), results.get(11, {})
    check("non-finite data: point 10 flagged",
          status == 200 and r10.get("status") == "error"
          and r10.get("error", {}).get("code") == "non_finite_data",
          f"got {status}: {payload}")
    check("non-finite data: point 11 unaffected",
          r11.get("status") == "ok"
          and abs((r11.get("intensity") or 0.0) - data_fn(3, 4, 5)) <= 1e-3,
          repr(r11))

    # -- invalid points payloads ---------------------------------------------
    raw = build_nifti(**base)
    bad_points = [
        ("duplicate ids", [{"id": 1, "point": [0, 0, 0]}, {"id": 1, "point": [1, 1, 1]}]),
        ("zero points", []),
        ("257 points", [{"id": i, "point": [0, 0, 0]} for i in range(257)]),
        ("non-finite coord", [{"id": 1, "point": [float("inf"), 0, 0]}]),
        ("malformed JSON", b"[{"),
    ]
    for name, points_payload in bad_points:
        status, payload = post_sample(base_url, raw, points_payload)
        err = payload.get("error", {}) if isinstance(payload, dict) else {}
        check(f"negative[points {name}]: 400/invalid_points",
              status == 400 and err.get("code") == "invalid_points"
              and err.get("field") == "points",
              f"got {status}: {payload}")

    # -- compare endpoint: mixed endianness/datatype/affine/grid --------------
    try:
        from app.nifti import invert_affine
    except Exception as exc:  # pragma: no cover - defensive
        check("compare: app.nifti importable", False, repr(exc))
        invert_affine = None

    if invert_affine is not None:
        # Followup grid: different dims/affine from the baseline grid DIMS.
        # F_SFORM = diag(1,2,5) + t(9,18,25);
        # F_QFORM = rotz(+90 deg) @ diag(1,2,5) + t(14,20,30).
        F_DIMS = (6, 5, 4)
        F_SFORM = ((1.0, 0.0, 0.0, 9.0),
                   (0.0, 2.0, 0.0, 18.0),
                   (0.0, 0.0, 5.0, 25.0))
        F_QFORM = ((0.0, -2.0, 0.0, 14.0),
                   (1.0, 0.0, 0.0, 20.0),
                   (0.0, 0.0, 5.0, 30.0))
        F_SLOPE, F_INTER, F_RAW = 2.0, 1.0, 7.0
        # constant followup data -> scaled intensity is 15.0 anywhere in-bounds
        F_INTENSITY = F_RAW * F_SLOPE + F_INTER

        def exp_voxel(affine, world):
            inv = invert_affine([list(r) for r in affine] + [[0.0, 0.0, 0.0, 1.0]],
                                field="smoke")
            x, y, z = world
            return tuple(inv[r][0] * x + inv[r][1] * y + inv[r][2] * z + inv[r][3]
                         for r in range(3))

        def close3(got, want, tol=1e-4):
            return isinstance(got, list) and len(got) == 3 and all(
                abs(g - e) <= tol for g, e in zip(got, want))

        def build_side(endian, datatype, transform, dims, side_data_fn,
                       slope, inter, sform_affine, qoffset, pixdim):
            return build_nifti(
                endian=endian, datatype=datatype, dims=dims, data_fn=side_data_fn,
                transform=transform, slope=slope, inter=inter,
                srow_x=sform_affine[0], srow_y=sform_affine[1],
                srow_z=sform_affine[2],
                quatern=QUATERN_90Z, qoffset=qoffset, pixdim=pixdim)

        def baseline_file(endian, datatype, transform):
            return build_side(endian, datatype, transform, DIMS, data_fn,
                              SLOPE, INTER, SFORM_AFFINE, (10.0, 20.0, 30.0),
                              (1.0, 2.0, 3.0, 4.0))

        def followup_file(endian, datatype, transform):
            return build_side(endian, datatype, transform, F_DIMS,
                              lambda i, j, k: F_RAW, F_SLOPE, F_INTER,
                              F_SFORM, (14.0, 20.0, 30.0), (1.0, 1.0, 2.0, 5.0))

        # (tag, baseline cfg, followup cfg, world points inside both footprints)
        pairs = [
            ("LE/int16/sform x BE/float32/qform", ("<", "int16", "sform"),
             (">", "float32", "qform"),
             [(12.0, 22.0, 35.0), (11.0, 24.0, 40.0), (13.5, 20.5, 44.0)]),
            ("BE/float32/qform x LE/int16/sform", (">", "float32", "qform"),
             ("<", "int16", "sform"),
             [(9.5, 22.0, 35.0), (9.7, 25.0, 38.0)]),
            ("BE/int16/sform x LE/float32/sform", (">", "int16", "sform"),
             ("<", "float32", "sform"),
             [(12.0, 22.0, 35.0), (13.0, 24.0, 38.0)]),
            ("LE/int16/qform x BE/float32/qform", ("<", "int16", "qform"),
             (">", "float32", "qform"),
             [(8.0, 22.0, 35.0), (9.5, 24.0, 40.0)]),
        ]
        for tag, bcfg, fcfg, worlds in pairs:
            b_aff = SFORM_AFFINE if bcfg[2] == "sform" else QFORM_AFFINE
            f_aff = F_SFORM if fcfg[2] == "sform" else F_QFORM
            raw_b = baseline_file(*bcfg)
            raw_f = followup_file(*fcfg)
            ids = list(range(len(worlds), 0, -1))  # ids reversed vs point order
            points = [{"id": pid, "point": list(w)} for pid, w in zip(ids, worlds)]
            status, payload = post_compare(base_url, raw_b, raw_f, points)
            if not check(f"compare[{tag}]: HTTP 200", status == 200,
                         f"got {status}: {payload}"):
                continue
            check(f"compare[{tag}]: top-level transforms",
                  isinstance(payload, dict)
                  and payload.get("baseline", {}).get("transform") == bcfg[2]
                  and payload.get("followup", {}).get("transform") == fcfg[2],
                  repr(payload))
            results = payload.get("results") if isinstance(payload, dict) else None
            if not check(f"compare[{tag}]: request order preserved",
                         isinstance(results, list)
                         and [r.get("id") for r in results] == ids,
                         repr(results)):
                continue
            for res, world in zip(results, worlds):
                pid = res.get("id")
                eb = exp_voxel(b_aff, world)
                ef = exp_voxel(f_aff, world)
                # baseline data is linear, so trilinear interpolation is exact
                b_int = (eb[0] + 10.0 * eb[1] + 100.0 * eb[2]) * SLOPE + INTER
                bs = res.get("baseline", {})
                fs = res.get("followup", {})
                ok = res.get("status") == "ok"
                ok = ok and close3(bs.get("voxel"), eb) \
                    and abs((bs.get("intensity") or 0.0) - b_int) <= 1e-3 \
                    and bs.get("transform") == bcfg[2]
                ok = ok and close3(fs.get("voxel"), ef) \
                    and abs((fs.get("intensity") or 0.0) - F_INTENSITY) <= 1e-3 \
                    and fs.get("transform") == fcfg[2]
                ok = ok and abs((res.get("delta") or 0.0)
                                - (F_INTENSITY - b_int)) <= 1e-3
                check(f"compare[{tag}]: id {pid} both sides + delta", ok, repr(res))

        # -- compare: per-side point errors on mismatched grids ----------------
        nan_base = build_nifti(
            endian="<", datatype="float32", dims=DIMS,
            data_fn=lambda i, j, k: float("nan") if (i, j, k) == (1, 1, 1)
            else data_fn(i, j, k),
            transform="sform", slope=SLOPE, inter=INTER,
            srow_x=SFORM_AFFINE[0], srow_y=SFORM_AFFINE[1], srow_z=SFORM_AFFINE[2])
        nan_foll = build_nifti(
            endian=">", datatype="float32", dims=F_DIMS,
            data_fn=lambda i, j, k: float("nan") if (i, j, k) == (4, 2, 3)
            else F_RAW,
            transform="qform", slope=F_SLOPE, inter=F_INTER,
            quatern=QUATERN_90Z, qoffset=(14.0, 20.0, 30.0),
            pixdim=(1.0, 1.0, 2.0, 5.0))
        side_points = [
            (1, (12.0, 24.0, 40.0), "ok", None, None),
            (2, (15.0, 22.0, 35.0), "error", "followup", "out_of_bounds"),
            (3, (8.0, 22.0, 35.0), "error", "baseline", "out_of_bounds"),
            (4, (20.0, 22.0, 35.0), "error", "baseline", "out_of_bounds"),
            (5, (12.0, 23.0, 34.0), "error", "baseline", "non_finite_data"),
            (6, (10.0, 24.0, 45.0), "error", "followup", "non_finite_data"),
        ]
        points = [{"id": pid, "point": list(w)} for pid, w, _, _, _ in side_points]
        status, payload = post_compare(base_url, nan_base, nan_foll, points)
        results = {r.get("id"): r for r in payload.get("results", [])} \
            if status == 200 and isinstance(payload, dict) else {}
        check("compare sides: HTTP 200", status == 200, f"got {status}: {payload}")
        for pid, world, want_status, want_side, want_code in side_points:
            r = results.get(pid, {})
            if want_status == "ok":
                check(f"compare sides: id {pid} ok",
                      r.get("status") == "ok"
                      and abs((r.get("delta") or 0.0)
                              - (r.get("followup", {}).get("intensity", 0.0)
                                 - r.get("baseline", {}).get("intensity", 0.0)))
                      <= 1e-9,
                      repr(r))
            else:
                err = r.get("error", {})
                check(f"compare sides: id {pid} {want_side}/{want_code}",
                      r.get("status") == "error"
                      and err.get("code") == want_code
                      and err.get("side") == want_side,
                      repr(r))

        # -- compare: file-level errors locate the failing upload --------------
        good_base = baseline_file("<", "int16", "sform")
        good_foll = followup_file("<", "int16", "sform")
        one = [{"id": 1, "point": [12.0, 22.0, 35.0]}]
        bad_base = build_nifti(endian="<", datatype="int16", dims=DIMS,
                               data_fn=data_fn, transform="sform",
                               srow_x=SFORM_AFFINE[0], srow_y=SFORM_AFFINE[1],
                               srow_z=SFORM_AFFINE[2], extra=b"\x00")
        bad_foll = build_nifti(endian=">", datatype="int16", dims=F_DIMS,
                               data_fn=lambda i, j, k: 7, transform="sform",
                               srow_x=F_SFORM[0], srow_y=F_SFORM[1],
                               srow_z=F_SFORM[2], magic=b"ni1\x00")
        file_cases = [
            ("baseline trailing bytes", bad_base, good_foll, one, {},
             "trailing_bytes", "file", "baseline"),
            ("followup bad magic", good_base, bad_foll, one, {},
             "unsupported_magic", "magic", "followup"),
            ("missing followup", good_base, good_foll, one,
             {"followup_field": "file"}, "missing_file", "followup", "followup"),
            ("duplicate baseline", good_base, good_foll, one,
             {"followup_field": "baseline"}, "multiple_files", "baseline", "baseline"),
            ("invalid points", good_base, good_foll, [], {},
             "invalid_points", "points", None),
        ]
        for name, rb, rf, pts, kw, code, field, side in file_cases:
            status, payload = post_compare(base_url, rb, rf, pts, **kw)
            err = payload.get("error", {}) if isinstance(payload, dict) else {}
            ok = status == 400 and err.get("code") == code \
                and err.get("field") == field
            if side is not None:
                ok = ok and err.get("side") == side
            check(f"compare negative[{name}]: 400/{code}", ok,
                  f"got {status}: {payload}")

    ok = not failures
    print(f"stage smoke: {'PASS' if ok else 'FAIL'} "
          f"({total - len(failures)}/{total} checks passed)", flush=True)
    return ok


# ---------------------------------------------------------------------------

def main():
    base_url = os.environ.get("VERIFY_BASE_URL", "http://127.0.0.1:8000")
    print(f"verify: base_url={base_url}", flush=True)
    code = 0
    if not stage_tests():
        code |= EXIT_TESTS
    if not stage_image():
        code |= EXIT_IMAGE
    if not stage_smoke(base_url):
        code |= EXIT_SMOKE
    print(f"verify: exit code {code} (1=tests, 2=image, 4=smoke)", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
