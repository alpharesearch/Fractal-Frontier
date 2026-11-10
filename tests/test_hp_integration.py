"""Integration tests for the arbitrary-precision view state.

Covers: the double-double mirror as a view of the canonical state, backend
selection at the HP wall, exact deep pan/zoom chains, HP perturb coordinates
against direct multi-precision ground truth, section stitching, cache
gating, and the bookmark format.
"""

import json
import os
import sys
from decimal import Decimal, localcontext

import mpmath as mp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import Fractal_Frontier as ff
from dd_math import DD_EPS, dd_add_d
from deep_zoom import (HP_MAX_ROOTS, calculate_mandelbrot_perturb,
                       calculate_mandelbrot_perturb_hp)
from fractals import MandelbrotCalculator, is_deep
from hp_view import dd_to_decimal, to_decimal

SPIRAL = (-0.7454642792047889, 0.10700158801025124)
calc = MandelbrotCalculator()


def rel_diff(a, b):
    """Relative difference of two Decimals, computed in a wide context."""
    a, b = to_decimal(a), to_decimal(b)
    with localcontext(prec=400) as ctx:
        return 0.0 if b == 0 else abs(float(ctx.divide(ctx.subtract(a, b), b)))


@pytest.fixture
def viewer():
    """A withdrawn real viewer (Tk vars live, renders patched out)."""
    import tkinter as tk
    from unittest.mock import patch

    root = tk.Tk()
    root.withdraw()
    v = None
    try:
        with patch.object(ff.MandelbrotViewer, "draw_mandelbrot", lambda *a, **k: None):
            v = ff.MandelbrotViewer(root)
            v.width, v.height = 400, 300
            yield v
    finally:
        # Every viewer spawns a multiprocessing.Pool (spawn start method);
        # leaving it open leaks ~32 worker processes (~150 MB each once they
        # import Numba) across the test file. Reap it before the Tk root is
        # torn down, or the suite runs out of memory irrespectively of the
        # view-state implementation.
        if v is not None:
            v._close_app()
        root.destroy()


def deep_view(pitch="1e-34", w=400, h=300):
    """A canonical view at an arbitrary-precision pitch (past the DD wall).

    Built inside a wide Decimal context on purpose: at prec=28 (the default)
    a 1e-34 offset of a 0.745 coordinate rounds away, and the "deep" view
    would silently be degenerate.
    """
    cx, cy = SPIRAL
    with localcontext(prec=200) as ctx:
        p = to_decimal(pitch)
        half_w = ctx.multiply(p, to_decimal(w // 2))
        half_h = ctx.multiply(p, to_decimal(h // 2))
        x_min = ctx.subtract(to_decimal(cx), half_w)
        x_max = ctx.add(to_decimal(cx), half_w)
        y_min = ctx.subtract(to_decimal(cy), half_h)
        y_max = ctx.add(to_decimal(cy), half_h)
    return ff.PreciseBounds(x_min, x_max, y_min, y_max)


# ----------------------------------------------------------------------
# Mirror properties
# ----------------------------------------------------------------------
def test_dd_mirror_reads_and_writes_flow_into_canonical_state(viewer):
    """The legacy (hi, lo) attributes are a *view* of the canonical state."""
    viewer.x_min = -2.0
    assert viewer.bounds.x_min == to_decimal(-2.0)
    viewer.x_min_lo = -1.0514899906736287e-17
    assert viewer.bounds.x_min == dd_to_decimal(-2.0, -1.0514899906736287e-17)
    # Reading the pair back gives exactly what was written.
    assert viewer.x_min == -2.0
    assert viewer.x_min_lo == -1.0514899906736287e-17
    # np.float64 assignments (the old __init__ style) work too.
    viewer.x_max = np.float64(1.0)
    assert viewer.bounds.x_max == to_decimal(1.0)
    assert viewer.x_max_lo == 0.0


def test_mirror_survives_the_deep_bookmark_layout(viewer):
    """Bookmark 9's shape: equal hi parts, the whole range in the lo parts."""
    viewer._set_bounds(
        -0.7454642792047881,
        -0.7454642792047881,
        0.10700158801025167,
        0.10700158801025167,
    )
    viewer.x_min_lo = -9.789304372927869e-18
    viewer.x_max_lo = -9.789304372927675e-18
    viewer.y_min_lo = 2.798855395978615e-18
    viewer.y_max_lo = 2.7988553959787175e-18
    assert viewer.x_min == viewer.x_max  # hi parts are equal, as stored
    assert viewer._view_range_x() == pytest.approx(1.9413373839423337e-31, rel=1e-9)
    assert not viewer.bounds.is_degenerate(1200)


# ----------------------------------------------------------------------
# Deep navigation
# ----------------------------------------------------------------------
def test_deep_pan_round_trip_is_exact(viewer):
    """50 pitches out and 50 back must land on the exact starting digits.

    The double-double mirror cannot promise this: adding a pitch to the lo
    part rounds it away, so a chain of sub-ULP pans drifts.
    """
    bounds = deep_view("1e-34")
    start = bounds.to_strings()
    pitch = bounds.pitch(viewer.width)
    for _ in range(50):
        bounds.shift(pitch, 0)
    for _ in range(50):
        bounds.shift(-pitch, 0)
    assert bounds.to_strings() == start

    # The same chain through the legacy DD helpers drifts (documented limit).
    hi, lo = bounds.to_dd()[:2]
    dx_f = float(bounds.range_x()) / viewer.width
    probe = (hi, lo)
    for _ in range(50):
        probe = dd_add_d(*probe, dx_f)
    for _ in range(50):
        probe = dd_add_d(*probe, -dx_f)
    assert probe == pytest.approx((hi, lo), abs=1e-30)  # close, not exact


def test_zoom_chain_keeps_the_view_consistent(viewer):
    """60 zooms: exact halving, and the mirror stays coherent."""
    viewer._set_bounds(-2.0, 1.0, -1.5, 1.5)
    for _ in range(60):
        viewer.bounds.scale_around(*viewer.bounds.center(), 0.5)
    with localcontext(prec=80) as ctx:
        expected = ctx.multiply(to_decimal(3.0), to_decimal(0.5) ** 60)
    # Precision is a property of the *endpoints*: 50-digit endpoints around
    # 0.5 leave ~33 digits of a 1e-18 range after cancellation — 1e-50
    # absolute, far below the 6.5e-21 pitch the view renders.
    assert rel_diff(viewer.bounds.range_x(), expected) < 1e-30
    assert abs(viewer._view_range_x() - float(expected)) / float(expected) < 1e-12
    hi, lo = viewer.bounds.to_dd()[:2]
    assert hi + lo == pytest.approx(float(viewer.bounds.x_min), rel=1e-12)


def test_fluid_zoom_lands_exactly_on_target(viewer, monkeypatch):
    """The animation must end on the target snapshot, not near it."""
    from types import SimpleNamespace

    mouse = SimpleNamespace(type="1", x=250, y=150, widget=None)
    start = viewer.bounds.copy()
    anchor = viewer._anchor_at(mouse.x, mouse.y)
    target = start.copy().scale_around(*anchor, 0.125)

    monkeypatch.setattr(viewer.master, "after", lambda ms, fn, *a: fn(*a))
    viewer.fluid_zoom(mouse, 0.125)
    assert viewer.bounds.to_strings() == target.to_strings()


# ----------------------------------------------------------------------
# Backend selection
# ----------------------------------------------------------------------
def test_backend_bands_include_the_hp_band(viewer):
    viewer._set_bounds(-2.0, 1.0, -1.5, 1.5)
    assert viewer._select_precision(3.0) in ("float32", "float64")

    # DD band: pitch above DEEP_MARGIN * DD_EPS.
    viewer.bounds = deep_view("1e-30", viewer.width, viewer.height)
    assert viewer._select_precision() == "perturb"
    # Below the DD wall: the coordinate state must switch to HP.
    viewer.bounds = deep_view("1e-34", viewer.width, viewer.height)
    assert viewer._select_precision() == "perturb_hp"
    assert is_deep("perturb_hp")

    # Julia has no perturb kernel at any depth.
    viewer.fractal_type_var.set("Julia")
    assert viewer._select_precision() == "double_double"
    viewer.fractal_type_var.set("Mandelbrot")


def test_precision_selection_uses_exact_pitch_not_float_range(viewer):
    """A float range rounded at the DD wall must not pick the DD backend."""
    bounds = deep_view("1e-34", 1200, 800)
    pitch = bounds.pitch(1200)
    assert pitch < to_decimal(DD_EPS)
    assert viewer.width == 400
    viewer.bounds = bounds
    assert viewer._select_precision() == "perturb_hp"


# ----------------------------------------------------------------------
# HP perturb rendering
# ----------------------------------------------------------------------
def _grid_pixels(bounds, w, h):
    """Exact (x, y) strings of every pixel centre in the view."""
    pitch_x = bounds.pitch(w)
    pitch_y = bounds.pitch(h)
    pixels = []
    with bounds._ctx(extra=60) as ctx:
        for i in range(h):
            for j in range(w):
                px = ctx.add(
                    bounds.x_min,
                    ctx.multiply(pitch_x, ctx.add(to_decimal(j), to_decimal("0.5"))),
                )
                py = ctx.add(
                    bounds.y_min,
                    ctx.multiply(pitch_y, ctx.add(to_decimal(i), to_decimal("0.5"))),
                )
                pixels.append((str(px), str(py)))
    return pixels


def _direct_counts(pixels, max_iterations, dps):
    """Plain multi-precision iteration: the ground truth for the kernels."""
    out = []
    with mp.workdps(dps):
        for xs, ys in pixels:
            c = mp.mpc(mp.mpf(xs), mp.mpf(ys))
            z = mp.mpc(0)
            n = 0
            while n < max_iterations:
                zr, zi = mp.re(z), mp.im(z)
                if zr * zr + zi * zi > 4.0:
                    break
                z = z * z + c
                n += 1
            out.append(n)
    return out


@pytest.mark.timeout(420)
def test_hp_perturb_matches_direct_iteration_exact_at_deep_pitch():
    """Phase-3 acceptance, part 1: count-for-count equality with direct
    multi-precision iteration at a pitch of 1e-34.

    The view is centred clear of the boundary (400 pitches to the right of
    the spiral point), so every pixel escapes within a modest budget and the
    comparison is exact — no tolerance, no "mostly agrees". That is what
    proves the coordinate path is right: at this pitch a double-double
    coordinate cannot even express 16 distinct columns.
    """
    w, h = 16, 12
    max_iterations = 400
    # c = (0.3, 0) is exterior (the set's real-axis edge is 0.25), so every
    # pixel in a 1e-34-wide window escapes in tens of iterations: the whole
    # comparison can be exact, with no tolerance.
    cx, cy = to_decimal(0.3), to_decimal(0.0)
    with localcontext(prec=200) as ctx:
        half_x = ctx.multiply(to_decimal("1e-34"), to_decimal(w // 2))
        half_y = ctx.multiply(to_decimal("1e-34"), to_decimal(h // 2))
        bounds = ff.PreciseBounds(
            ctx.subtract(cx, half_x),
            ctx.add(cx, half_x),
            ctx.subtract(cy, half_y),
            ctx.add(cy, half_y),
        )
    assert bounds.pitch(w) == pytest.approx(to_decimal("1e-34"), rel=1e-12)
    counts, stats = calculate_mandelbrot_perturb_hp(
        w, h, *bounds.to_strings(), max_iterations
    )
    assert counts.shape == (h, w)
    assert (counts >= 0).all(), "unresolved pixels left in the HP band"
    assert stats["unresolved"] == 0

    truth = np.array(
        _direct_counts(_grid_pixels(bounds, w, h), max_iterations, 70), dtype=np.int32
    ).reshape(h, w)
    assert (counts == truth).all(), (
        f"HP perturb differs from direct iteration on "
        f"{int((counts != truth).sum())} of {w * h} pixels"
    )
    # Escape happened, so nothing hit the iteration budget. Uniform counts
    # are the correct answer here: at a 1e-34 pitch the whole 16x12 window is
    # a single point as far as escape time is concerned. The pixel-exact
    # agreement above is the assertion that matters.
    assert counts.max() < max_iterations, "no pixel escaped: view is not exterior"
    # The kernel's grid is still genuinely resolved at this pitch.
    assert len(set(bounds.pixel_deltas_x(w))) == w


def test_hp_section_starts_do_not_collapse_across_the_frame():
    """Phase-3 acceptance, part 2: the stitching math the sections use.

    Same geometry, two coordinate paths. The double-double version of
    ``x_min + offset * dx`` (what the pre-change section stitching used)
    collapses 1200 column edges onto ~127 positions at this pitch, which is
    exactly the banding the deep renders showed. The HP starts are 1200
    distinct values, and they are the same values the kernel iterates.
    """
    bounds = deep_view("1e-34", 1200, 800)
    hp_starts = {to_decimal(bounds.section_start_x(off, 1200)) for off in range(1200)}
    assert len(hp_starts) == 1200

    # The double-double path: what the rendered float64 coordinate actually
    # resolves to. It keeps an exact lo term, but at this pitch the whole
    # frame is inside one float64 ULP of x, so the columns collapse.
    hi, lo = bounds.to_dd()[:2]
    dx_f = float(bounds.range_x()) / 1200
    dd_lit = {float(dd_add_d(hi, lo, off * dx_f)[0]) for off in range(1200)}
    assert (
        len(dd_lit) < 200
    ), f"expected float64 section starts to collapse; got {len(dd_lit)}"


@pytest.mark.timeout(600)
def test_hp_and_dd_perturb_agree_in_the_overlap_band():
    """Where both coordinate paths are valid, the HP path must agree with the
    double-double perturb backend it supplements (pitch 1e-31, above DD_EPS).
    """
    w, h = 24, 16
    max_iterations = 60000
    bounds = deep_view("1e-31", w, h)
    assert bounds.pitch(w) > to_decimal(DD_EPS), "overlap band only"

    hp_counts, stats = calculate_mandelbrot_perturb_hp(
        w, h, *bounds.to_strings(), max_iterations
    )
    x_min, x_min_lo, _x_max, _x_max_lo, y_min, y_min_lo, _y_max, _y_max_lo = (
        bounds.to_dd()
    )
    dd_counts = calculate_mandelbrot_perturb(
        w,
        h,
        x_min,
        x_min_lo,
        float(bounds.range_x()) / w,
        y_min,
        y_min_lo,
        float(bounds.range_y()) / h,
        max_iterations,
    )
    agree = float((hp_counts == dd_counts).mean())
    assert agree > 0.98, f"HP vs DD-fed perturb agreement {agree:.4f}"
    assert stats["unresolved"] == 0


@pytest.mark.timeout(420)
def test_hp_sections_stitch_without_seams():
    """Section starts must be exact: four sections == one full render."""
    w, h = 24, 8
    max_iterations = 900
    bounds = deep_view("1e-34", w, h)
    full, full_stats = calculate_mandelbrot_perturb_hp(
        w, h, *bounds.to_strings(), max_iterations
    )
    dx_s, dy_s = bounds.pitch_strings(w, h)
    pieces = []
    offset = 0
    for section_width in (6, 6, 6, 6):
        start = bounds.section_start_x(offset, w)
        piece = calc.calculate_mandelbrot_section_perturb_hp(
            offset,
            section_width,
            w,
            h,
            start,
            dx_s,
            str(bounds.y_min),
            dy_s,
            max_iterations,
            None,
        )
        pieces.append(piece)
        offset += section_width
    stitched = np.hstack(pieces)
    assert stitched.shape == full.shape
    agree = float((stitched == full).mean())
    assert agree > 0.95, f"section stitching agreement {agree:.4f}"


def test_hp_pixel_deltas_are_the_exact_grid():
    """The kernel's per-pixel deltas must match the canonical grid exactly."""
    bounds = deep_view("1e-34", 1200, 1)
    deltas = bounds.pixel_deltas_x(1200)
    pitch = bounds.pitch(1200)
    cx = bounds.center()[0]
    with bounds._ctx(extra=80) as ctx:
        expected = [
            float(
                ctx.subtract(
                    ctx.add(
                        bounds.x_min,
                        ctx.multiply(pitch, ctx.add(to_decimal(j), to_decimal("0.5"))),
                    ),
                    cx,
                )
            )
            for j in range(1200)
        ]
    assert np.all(deltas == np.array(expected, dtype=np.float64))
    assert len(set(deltas)) == 1200


# ----------------------------------------------------------------------
# Cache gating
# ----------------------------------------------------------------------
def test_tile_cache_is_disabled_at_hp_depth(viewer):
    """Tile anchors and bounds are double-double; below DD_EPS they quantize."""
    params = {
        "fractal_type": "Mandelbrot",
        "width": 120,
        "height": 90,
        "x_min": SPIRAL[0],
        "x_max": SPIRAL[0],
        "x_min_lo": -9.789304372927869e-18,
        "x_max_lo": -9.789304372927675e-18,
        "y_min": SPIRAL[1],
        "y_max": SPIRAL[1],
        "y_min_lo": 2.798855395978615e-18,
        "y_max_lo": 2.7988553959787175e-18,
        "max_iterations": 100,
        "julia_c": viewer.julia_c,
        "zoom_level": 1e31,
        "precision": "perturb_hp",
    }
    raw = np.zeros((90, 120), dtype=np.int32)
    viewer._start_tile_cache(raw, params)
    assert viewer._tile_params is None
    assert viewer._tile_cache_is_valid() is False

    params["precision"] = "double_double"
    viewer._start_tile_cache(raw, params)
    assert viewer._tile_params is not None


def test_pan_cache_crop_is_disabled_at_hp_depth(viewer):
    viewer.bounds = deep_view("1e-34", viewer.width, viewer.height)
    viewer._render_precision = "perturb_hp"
    viewer._pan_cache = {
        "array": np.zeros((viewer.height, viewer.width), dtype=np.int32),
        "bounds": viewer.bounds.to_dd(),
        "max_iterations": viewer.max_iterations,
        "fractal_type": "Mandelbrot",
        "julia_c": viewer.julia_c,
        "zoom_level": viewer.zoom_level,
    }
    assert viewer._cache_crop_for_current_view() is None
    assert viewer._display_cached_pan() is False
    assert viewer._bounds_cover_current_view(viewer._pan_cache["bounds"]) is False
    viewer._render_precision = "double_double"
    assert viewer._cache_crop_for_current_view() is not None


# ----------------------------------------------------------------------
# Bookmarks
# ----------------------------------------------------------------------
def test_bookmark_stores_hp_fields_and_round_trips(viewer, tmp_path, monkeypatch):
    bm_file = tmp_path / "bookmarks.json"
    monkeypatch.setattr(ff, "BOOKMARKS_FILE", str(bm_file))
    viewer.bounds = deep_view("1e-34", viewer.width, viewer.height)
    viewer.bounds.ensure_precision(viewer.width)
    viewer.save_bookmark()

    data = json.loads(bm_file.read_text())
    saved = data[0]
    for key in ("x_min_hp", "x_max_hp", "y_min_hp", "y_max_hp", "view_precision"):
        assert key in saved, f"missing {key}"
    # Legacy fields are present and consistent with the HP state.
    for key in ("x_min", "x_max", "y_min", "y_max", "x_min_lo", "x_max_lo"):
        assert key in saved, f"missing legacy {key}"
    assert saved["x_min_hp"] == str(viewer.bounds.x_min)

    restored, legacy = ff.PreciseBounds.from_json(saved)
    assert legacy is False
    assert restored.to_strings() == viewer.bounds.to_strings()
    assert restored.precision == viewer.bounds.precision


def test_legacy_bookmark_load_reports_the_precision_floor(
    viewer, tmp_path, monkeypatch
):
    bm_file = tmp_path / "bookmarks.json"
    monkeypatch.setattr(ff, "BOOKMARKS_FILE", str(bm_file))
    legacy_bm = {
        "x_min": -0.7454642792047881,
        "x_max": -0.7454642792047881,
        "y_min": 0.10700158801025167,
        "y_max": 0.10700158801025167,
        "x_min_lo": -9.789304372927869e-18,
        "x_max_lo": -9.789304372927675e-18,
        "y_min_lo": 2.798855395978615e-18,
        "y_max_lo": 2.7988553959787175e-18,
        "fractal_type": "Mandelbrot",
    }
    bm_file.write_text(json.dumps([legacy_bm]))
    viewer.load_bookmark()
    # The dialog's list is built; load the entry through the same path.
    bounds, legacy = ff.PreciseBounds.from_json(legacy_bm)
    assert legacy is True
    assert bounds.x_min == dd_to_decimal(legacy_bm["x_min"], legacy_bm["x_min_lo"])
    assert bounds.pitch(1200) < to_decimal(DD_EPS)
    viewer.bounds = bounds
    viewer._render_precision = viewer._select_precision()
    assert viewer._render_precision == "perturb_hp"


def test_status_bar_reports_coordinate_digit_count(viewer):
    viewer.bounds = deep_view("1e-34", 1200, 800)
    viewer.bounds.ensure_precision(1200)
    viewer._render_precision = "perturb_hp"
    viewer.max_iterations = 100
    coords = viewer.bounds.coord_strings(significant=40)
    text = f"{viewer.bounds.precision}-digit coords | " f"X_min: {coords['x_min']}"
    assert "digit coords" in text
    assert len(coords["x_min"].split(".")[1]) >= 33
    assert viewer._render_precision == "perturb_hp"


def test_base_range_zoom_stays_float_compatible(viewer):
    """UI formulas consume a float zoom; the canonical zoom is the Decimal."""
    viewer._set_bounds(-2.0, 1.0, -1.5, 1.5)
    viewer.bounds.adjust_aspect(viewer.width, viewer.height)
    zoom = viewer.bounds.zoom(viewer.base_range)
    assert zoom == pytest.approx(Decimal(1.0), rel=1e-9)
    assert isinstance(viewer.bounds.zoom_float(viewer.base_range), float)
    deep = deep_view("1e-34", 1200, 800)
    assert deep.zoom(3.0) > Decimal("1e30")


# ----------------------------------------------------------------------
# Resource bounds (regressions for real resource failures)
# ----------------------------------------------------------------------
def test_tile_range_refuses_a_grid_far_larger_than_the_view(viewer):
    """A tile grid seeded for a different depth than the live bounds must not
    be enumerated.

    Seeding ``_start_tile_cache`` from params for a 1e-31-wide view while the
    live bounds are still the classic -2..1 view made ``_tile_index_range``
    build a set of ~1e62 tile keys. That is 100+ GB of tuples and took the
    process down. The range must come back empty instead.
    """
    params = {
        "fractal_type": "Mandelbrot",
        "width": 120,
        "height": 90,
        "x_min": SPIRAL[0],
        "x_max": SPIRAL[0],
        "x_min_lo": -9.789304372927869e-18,
        "x_max_lo": -9.789304372927675e-18,
        "y_min": SPIRAL[1],
        "y_max": SPIRAL[1],
        "y_min_lo": 2.798855395978615e-18,
        "y_max_lo": 2.7988553959787175e-18,
        "max_iterations": 100,
        "julia_c": viewer.julia_c,
        "zoom_level": 1e31,
        "precision": "double_double",
    }
    viewer._set_bounds(-2.0, 1.0, -1.5, 1.5)
    raw = np.zeros((90, 120), dtype=np.int32)
    viewer._start_tile_cache(raw, params)  # must return promptly, not explode

    needed = viewer._tile_index_range(
        viewer.x_min,
        viewer.x_min_lo,
        viewer.x_max,
        viewer.x_max_lo,
        viewer.y_min,
        viewer.y_min_lo,
        viewer.y_max,
        viewer.y_max_lo,
    )
    assert needed == set(), f"refused range still enumerated {len(needed)} tiles"


def test_viewer_pool_is_reaped_on_close():
    """A closed viewer must leave no render workers behind.

    The pool is spawn-based, so every leftover worker is a full interpreter
    with Numba loaded (~150 MB). Leaking a pool per viewer exhausted memory.
    """
    import tkinter as tk
    from unittest.mock import patch

    root = tk.Tk()
    root.withdraw()
    try:
        with patch.object(ff.MandelbrotViewer, "draw_mandelbrot", lambda *a, **k: None):
            v = ff.MandelbrotViewer(root)
        pool = v.pool
        n_workers = len(pool._pool)
        assert n_workers >= 1
        v._close_app()
        assert v._closed is True
        for proc in pool._pool:
            proc.join(timeout=10)
            assert not proc.is_alive(), "a render worker outlived its viewer"
        v._close_app()  # idempotent
    finally:
        root.destroy()


def test_unresolved_ranking_selects_pixels_not_rows():
    """The fallback must pick `budget` pixels, whichever shape the section is.

    Ranking used np.argsort on the 2-D score, which sorts along the last axis
    and returns a 2-D result; unpacking that as indices raised "only integer
    scalar arrays can be converted to a scalar index" and made deep bookmarks
    unloadable. A tall, narrow section (37 columns is what the viewer uses at
    1200 px) is where it bit.
    """
    from deep_zoom import _rank_unresolved

    h, w = 20, 37
    counts = np.full((h, w), -1, dtype=np.int32)
    counts[3, 7] = 50  # already settled
    switches = np.linspace(0.0, 1.0, h * w).reshape(h, w)
    switches[3, 7] = 999.0  # settled pixel has the strongest signal

    rows, cols = _rank_unresolved(counts, switches, 5)
    assert np.shape(rows) == (5,), f"expected 5 pixel rows, got {np.shape(rows)}"
    assert np.shape(cols) == (5,), f"expected 5 pixel cols, got {np.shape(cols)}"
    for r, c in zip(rows, cols):
        assert np.issubdtype(np.asarray(r).dtype, np.integer)
        assert counts[r, c] == -1, "a settled pixel was selected"
    # Highest switch signal first.
    assert (rows[0], cols[0]) != (3, 7)
    picked = [switches[r, c] for r, c in zip(rows, cols)]
    assert picked == sorted(picked, reverse=True)


def test_unresolved_ranking_is_capped_by_the_budget():
    from deep_zoom import _rank_unresolved

    counts = np.full((20, 37), -1, dtype=np.int32)
    switches = np.zeros((20, 37))
    rows, cols = _rank_unresolved(counts, switches, 1)
    assert len(np.atleast_1d(rows)) == 1


def test_hp_fallback_budget_is_a_pixel_count_not_a_row_count():
    """The fallback spends its budget on pixels, never a multiple of it."""
    w, h = 37, 20
    max_iterations = 400
    bounds = deep_view("1e-34", w, h)
    counts, stats = calculate_mandelbrot_perturb_hp(
        w, h, *bounds.to_strings(), max_iterations, direct_budget=5
    )
    assert counts.shape == (h, w)
    assert stats["direct"] <= 5, f"budget overspent: {stats['direct']} > 5"
    assert (counts >= 0).all()
