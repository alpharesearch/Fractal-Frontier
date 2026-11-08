"""
Tests for the deep-zoom backends: double-double kernels and the
perturbation (Deep Zoom Algorithm) module.

Fixtures use a deep self-spiral discovered by marching toward the deepest
pixel of a seahorse-valley window, where escape-count structure exists at
every scale.
"""

import math
import os
import sys
from fractions import Fraction

import numpy as np
import pytest

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from dd_math import (DD_EPS, dd_add_d, dd_center, dd_format, dd_sub,
                     dd_to_double, two_prod, two_sum)
from deep_zoom import calculate_mandelbrot_perturb, digits_for_pitch
from fractals import (MandelbrotCalculator, julia_section_dd_jit,
                      julia_section_jit, mandelbrot_section_dd_jit)

# Deep spiral fixture: rich escape-count structure from ~1e-14 down to 1e-22.
# repr-exact doubles (16-digit %g printouts land a thousand pixels away at
# deep pitch!).
SPIRAL = (-0.7454642792047889, 0.10700158801025124)

calc = MandelbrotCalculator()


# ----------------------------------------------------------------------
# dd_math primitives
# ----------------------------------------------------------------------
def test_two_sum_is_exact():
    rng = np.random.default_rng(11)
    for a, b in zip(rng.uniform(-1e6, 1e6, 200), rng.uniform(-1e6, 1e6, 200)):
        s, e = two_sum(float(a), float(b))
        assert Fraction(s) + Fraction(e) == Fraction(float(a)) + Fraction(float(b))


def test_two_prod_is_exact():
    rng = np.random.default_rng(12)
    for a, b in zip(rng.uniform(-1e3, 1e3, 200), rng.uniform(-1e3, 1e3, 200)):
        p, e = two_prod(float(a), float(b))
        assert Fraction(p) + Fraction(e) == Fraction(float(a)) * Fraction(float(b))


def test_dd_recovers_sub_ulp_increments():
    hi, lo = dd_add_d(1.0, 0.0, 1e-30)
    assert hi == 1.0
    assert abs(lo - 1e-30) < 1e-45
    hi2, lo2 = dd_add_d(hi, lo, 1e-30)
    assert abs(dd_to_double(hi2, lo2) - (1.0 + 2e-30)) < 1e-45


def test_dd_format_significant_digits():
    assert dd_format(0.0, 0.0, 30) == "0"
    assert dd_format(3.0, 0.0, 30) == "3"
    out = dd_format(2456508887656634.0, -0.25, 30)
    assert out == "2456508887656633.75"
    out = dd_format(-0.5, 1e-25, 30)
    assert out.startswith("-0.4999")


# ----------------------------------------------------------------------
# Double-double kernels
# ----------------------------------------------------------------------
def test_dd_matches_float64_at_shallow_pitch():
    """At coarse pitch the DD kernel must reproduce the float64 kernel.

    Sampling alignment: the DD kernel samples pixel centres
    (x_min + (col+0.5)*dx); the float64 kernel samples x_min + col*dx,
    so the float64 view is shifted by half a pixel to compare identical
    coordinates. Boundary-sensitivity (chaotic amplification of 1-ulp
    coordinate differences) is kept mild by using a coarse pitch and a
    modest iteration budget.
    """
    cx, cy = -0.748, 0.106
    pitch = 1e-6
    w, h = 48, 32
    it = 500
    dd = mandelbrot_section_dd_jit(
        0, w, w, h, cx, 0.0, pitch, cy, 0.0, pitch * 2 / 3, it
    )
    f64 = calc._compute_iterations(
        "mandelbrot",
        w,
        h,
        cx + 0.5 * pitch,
        cx + (w + 0.5) * pitch,
        cy + 0.5 * pitch * 2 / 3,
        cy + (h + 0.5) * pitch * 2 / 3,
        it,
        None,
        "float64",
    )
    agree = (dd == f64).mean()
    assert agree > 0.99, f"DD vs float64 agreement {agree:.4f}"


def test_dd_resolves_breakdown_pitch():
    """In the band where float64 quantizes, DD keeps the structure."""
    cx, cy = SPIRAL
    pitch = 2.19e-17
    w, h = 48, 32
    it = 100000
    dd = mandelbrot_section_dd_jit(
        0, w, w, h, cx, 0.0, pitch, cy, 0.0, pitch * 2 / 3, it
    )
    f64 = calc._compute_iterations(
        "mandelbrot",
        w,
        h,
        cx,
        cx + w * pitch,
        cy,
        cy + h * pitch * 2 / 3,
        it,
        None,
        "float64",
    )
    dd_uniq = len(np.unique(dd))
    f64_uniq = len(np.unique(f64))
    assert dd_uniq > 100, f"DD lost structure ({dd_uniq} unique counts)"
    assert f64_uniq < dd_uniq / 2, f"float64 not degraded ({f64_uniq} vs {dd_uniq})"


def test_julia_dd_matches_float64_at_shallow_pitch():
    c = complex(-0.7, 0.27015)
    x0, y0 = -1.5, -1.125
    pitch = 3.0 / 64
    w, h = 64, 48
    it = 500
    dd = julia_section_dd_jit(0, w, w, h, x0, 0.0, pitch, y0, 0.0, pitch * 3 / 4, it, c)
    f64 = julia_section_jit(
        0,
        w,
        w,
        h,
        x0 + 0.5 * pitch,
        x0 + (w + 0.5) * pitch,
        y0 + 0.5 * pitch * 3 / 4,
        y0 + (h + 0.5) * pitch * 3 / 4,
        it,
        c,
    )
    agree = (dd == f64).mean()
    assert agree > 0.95, f"Julia DD vs float64 agreement {agree:.4f}"


# ----------------------------------------------------------------------
# Perturbation backend
# ----------------------------------------------------------------------
def test_digits_for_pitch_grows_with_depth():
    assert digits_for_pitch(1e-10) < digits_for_pitch(1e-20) < digits_for_pitch(1e-30)


@pytest.mark.timeout(120)
def test_perturb_matches_dd_in_overlap_band():
    """In the band where both backends are valid they must agree."""
    cx, cy = SPIRAL
    pitch = 2.19e-17
    w, h = 48, 32
    it = 100000
    dd = mandelbrot_section_dd_jit(
        0, w, w, h, cx, 0.0, pitch, cy, 0.0, pitch * 2 / 3, it
    )
    pert = calculate_mandelbrot_perturb(
        w, h, cx, 0.0, pitch, cy, 0.0, pitch * 2 / 3, it
    )
    assert (pert >= 0).all(), "unresolved pixels after re-rooting"
    agree = (pert == dd).mean()
    assert agree > 0.98, f"perturb vs DD agreement {agree:.4f}"


@pytest.mark.timeout(240)
def test_perturb_deep_structure_pitch():
    """A pitch ~1e-20 (zoom ~1e20): only perturbation resolves this band
    reliably; structure must show as a rich count distribution."""
    cx, cy = SPIRAL
    pitch = 1e-20
    w, h = 48, 32
    it = 200000
    pert = calculate_mandelbrot_perturb(
        w, h, cx, 0.0, pitch, cy, 0.0, pitch * 2 / 3, it
    )
    assert (pert >= 0).all()
    assert len(np.unique(pert)) > 50, "no structure at deep pitch"


@pytest.mark.timeout(180)
def test_perturb_extreme_pitch():
    """1e-30 pitch: far past double-double, structure from the DD fallback
    and re-rooted references must complete without unresolved pixels."""
    cx, cy = SPIRAL
    pitch = 1e-30
    w, h = 24, 16
    it = 30000
    pert = calculate_mandelbrot_perturb(
        w, h, cx, 0.0, pitch, cy, 0.0, pitch * 2 / 3, it
    )
    assert (pert >= 0).all()
    assert len(np.unique(pert)) >= 1


# ----------------------------------------------------------------------
# Backend selection
# ----------------------------------------------------------------------
class _FakeVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def _fake_state(x_min, x_max, width, zoom_level=1.0, fractal="Mandelbrot"):
    import types

    return types.SimpleNamespace(
        x_min=np.float64(x_min),
        x_max=np.float64(x_max),
        y_min=np.float64(-1.0),
        y_max=np.float64(1.0),
        width=width,
        zoom_level=zoom_level,
        fractal_type_var=_FakeVar(fractal),
    )


def test_backend_selection_bands():
    from Fractal_Frontier import MandelbrotViewer

    sel = MandelbrotViewer._select_precision

    # Shallow view: float band (GPU-capable float32 at low zoom).
    st = _fake_state(-2.0, 1.0, 1000)
    assert sel(st, 3.0) == "float32"

    # High zoom: float64.
    st = _fake_state(-2.0, 1.0, 1000, zoom_level=1e12)
    assert sel(st, 3.0) == "float64"

    # Pitch in the float64 breakdown band -> double-double.
    cx = SPIRAL[0]
    st = _fake_state(cx, cx + 1000 * 1e-17, 1000)
    assert sel(st, 1000 * 1e-17) == "double_double"

    # Pitch past the double-double band -> perturbation.
    st = _fake_state(cx, cx + 1000 * 1e-31, 1000)
    assert sel(st, 1000 * 1e-31) == "perturb"

    # Julia has no perturbation backend: deep Julia uses double-double.
    st = _fake_state(cx, cx + 1000 * 1e-31, 1000, fractal="Julia")
    assert sel(st, 1000 * 1e-31) == "double_double"


def test_select_precision_threshold_consistency():
    """The DD band boundary sits at DEEP_MARGIN * DD_EPS."""
    from Fractal_Frontier import DEEP_MARGIN, MandelbrotViewer

    sel = MandelbrotViewer._select_precision
    cx = SPIRAL[0]
    pitch = 2 * DEEP_MARGIN * DD_EPS
    st = _fake_state(cx, cx + 1000 * pitch, 1000)
    assert sel(st, 1000 * pitch) == "double_double"
    pitch = DEEP_MARGIN * DD_EPS * 0.5
    st = _fake_state(cx, cx + 1000 * pitch, 1000)
    assert sel(st, 1000 * pitch) == "perturb"


# ----------------------------------------------------------------------
# Viewer double-double state
# ----------------------------------------------------------------------
@pytest.mark.timeout(30)
def test_viewer_deep_zoom_keeps_exact_range():
    """A long zoom chain must shrink the window exactly (DD state).

    60 halvings of a width-3 view: plain-float bound arithmetic drifts
    measurably; the double-double state must not.
    """
    import tkinter as tk
    from unittest.mock import patch

    root = tk.Tk()
    root.withdraw()
    try:
        from Fractal_Frontier import MandelbrotViewer

        viewer = MandelbrotViewer(root)
        with patch.object(viewer, "draw_mandelbrot", lambda *a, **k: None):

            class KbdEvent:
                type = "2"
                x = 0
                y = 0
                widget = None

            for _ in range(60):
                viewer.zoom(KbdEvent(), 0.5)

            expected = 3.0 * 0.5**60
            actual = viewer._view_range_x()
            assert abs(actual - expected) / expected < 1e-9
            # 60 halvings put the pitch deep into the double-double band.
            pitch = actual / viewer.width
            assert pitch < 64.0 * 2.22e-16, "expected deep band"
            prec = viewer._select_precision(actual)
            assert prec in ("double_double", "perturb")
            # The low parts must carry real information after deep pans.
            viewer._shift_bounds(0.7 * pitch, 0.0)
            assert viewer.x_min_lo != 0.0
    finally:
        root.destroy()


@pytest.mark.timeout(30)
def test_viewer_bookmark_roundtrip_keeps_low_parts(tmp_path, monkeypatch):
    import tkinter as tk
    from unittest.mock import patch

    import Fractal_Frontier as ff

    bm_file = tmp_path / "bookmarks.json"
    monkeypatch.setattr(ff, "BOOKMARKS_FILE", str(bm_file))

    root = tk.Tk()
    root.withdraw()
    try:
        from Fractal_Frontier import MandelbrotViewer

        viewer = MandelbrotViewer(root)
        with patch.object(viewer, "draw_mandelbrot", lambda *a, **k: None):
            # Craft a deep state: at 1e-17 pitch the hi parts round to the
            # same double (ulp is coarser), so the range lives in the lo
            # parts — exactly what the DD state represents.
            cx, cy = SPIRAL
            viewer._set_bounds(cx, cx, cy, cy)
            viewer.x_max_lo = 1e-17
            viewer.y_max_lo = 6.7e-18
            viewer._shift_bounds(0.3e-17, 0.2e-17)
            assert viewer.x_min_lo != 0.0
            viewer.save_bookmark()

            import json

            data = json.loads(bm_file.read_text())
            assert data[0]["x_min_lo"] == viewer.x_min_lo
            assert data[0]["x_max_lo"] == viewer.x_max_lo

            # Restore a fresh (shallow) state, then emulate the loader body
            # the dialog applies on double-click.
            viewer._set_bounds(-2.0, 1.0, -1.5, 1.5)
            bm = data[0]
            viewer.x_min = float(bm["x_min"])
            viewer.x_min_lo = float(bm["x_min_lo"])
            viewer.x_max = float(bm["x_max"])
            viewer.x_max_lo = float(bm["x_max_lo"])
            viewer.y_min = float(bm["y_min"])
            viewer.y_min_lo = float(bm["y_min_lo"])
            viewer.y_max = float(bm["y_max"])
            viewer.y_max_lo = float(bm["y_max_lo"])
            assert abs(viewer._view_range_x() - 1e-17) < 1e-24
    finally:
        root.destroy()
