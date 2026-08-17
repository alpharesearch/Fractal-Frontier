"""
Regression tests for previously fixed bugs.

Each test here guards a specific bug that was found during review:

Calculator / theme level (no display needed):
  - GPU availability must reflect a real CUDA device, not just an importable
    numba-cuda (app used to crash at startup on GPU-less machines).
  - A GPU failure at runtime must fall back to CPU instead of raising.
  - apply_color_theme() must tolerate zero/negative max_iterations.
  - The high-resolution rainbow palettes must not be collapsed by clipping.
  - Sectioned CPU rendering matches a single full-width computation pixel
    for pixel (section offset bug: non-uniform section widths shifted the
    last section's columns to wrong x coordinates).
  - The auto iteration count resolves bulb boundaries at the classic view
    and keeps growing with zoom depth (the old 500 cap rendered deep-zoom
    filaments and bulb halos as solid black).

Viewer level (needs a display; skipped otherwise):
  - Pixels stay square after resizes (a 3x3 range on a non-square canvas
    used to stretch the fractal vertically).
  - Exactly one canvas image item after repeated redraws (image leak).
  - Rendered image exactly covers the canvas width (section truncation).
  - Negative iteration offsets clamp max_iterations to >= 1.
  - fractal_type_changed() keeps the dropdown variable in sync.
  - Pan/zoom keys are ignored while typing in an Entry widget.
  - fluid_zoom() always releases the is_zooming lock.
  - Invalid Julia C input does not raise.
  - Degenerate view bounds reset instead of crashing the render request.
  - Bookmarks are written to BOOKMARKS_FILE and old formats load safely.
  - on_close() shuts down the render thread cleanly.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pytest

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from fractals import GPU_AVAILABLE, MandelbrotCalculator  # noqa: E402
from themes import apply_color_theme  # noqa: E402


# ---------------------------------------------------------------------------
# Calculator / theme level
# ---------------------------------------------------------------------------

def test_gpu_available_reflects_real_device():
    """GPU_AVAILABLE must mean 'a CUDA device exists', not 'numba-cuda imports'."""
    try:
        from numba import cuda
    except ImportError:
        assert GPU_AVAILABLE is False
        return
    expected = bool(cuda.is_available()) and len(cuda.gpus) > 0
    assert GPU_AVAILABLE == expected


def test_gpu_failure_falls_back_to_cpu():
    """Forcing GPU on without a device must not raise; CPU result is returned."""
    calc = MandelbrotCalculator()
    original = MandelbrotCalculator._use_gpu
    try:
        MandelbrotCalculator.set_use_gpu(True)
        iterations = calc._compute_iterations(
            "mandelbrot", 32, 16, -2.0, 1.0, -1.5, 1.5, 20
        )
        assert iterations.shape == (16, 32)
        assert iterations.dtype == np.int32
        if not GPU_AVAILABLE:
            # The calculator must have disabled itself after the fallback.
            assert MandelbrotCalculator._use_gpu is False
    finally:
        MandelbrotCalculator.set_use_gpu(original)


def test_apply_color_theme_zero_and_negative_max_iterations():
    """Zero/negative iteration counts (reachable via negative slider offset)
    must not divide by zero or emit warnings."""
    iterations = np.arange(25).reshape(5, 5).astype(np.int32)
    for max_iterations in (0, -5):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            colors = apply_color_theme(iterations, max_iterations, "Default")
        assert colors.shape == (5, 5, 3)
        assert colors.dtype == np.uint8


def test_high_res_rainbow_palettes_not_collapsed():
    """The old '*1024 then clip to 255' scaling collapsed the high-resolution
    palettes (Rainbow4 had only ~258 distinct colors). They must now span the
    full hue wheel."""
    iterations = np.arange(8192).astype(np.int32).reshape(1, 8192)
    for theme, min_distinct in (
        ("Rainbow", 200),
        ("Rainbow3", 900),
        ("Rainbow4", 1200),
    ):
        colors = apply_color_theme(iterations, 100000, theme)
        distinct = len(np.unique(colors.reshape(-1, 3), axis=0))
        assert distinct >= min_distinct, (
            f"{theme}: only {distinct} distinct colors (palette collapsed?)"
        )


def test_sectioned_render_matches_full_width():
    """The per-section CPU path must produce exactly the same iteration
    counts as a single full-width computation. Sections carry an explicit
    column offset and may have different widths (the last absorbs the
    remainder), so every column must land at its correct x coordinate."""
    from fractals import mandelbrot_section_jit

    width, height = 853, 40  # 853 is prime: no uniform sectioning covers it
    x_min, x_max, y_min, y_max, mi = -2.0, 1.0, -1.5, 1.5, 30

    full = mandelbrot_section_jit(
        0, width, width, height, x_min, x_max, y_min, y_max, mi
    )

    for num_sections in (7, 8, 32):
        base_w = max(1, width // num_sections)
        section_widths = [base_w] * num_sections
        section_widths[-1] += width - base_w * num_sections  # last absorbs remainder
        parts = []
        offset = 0
        for i in range(num_sections):
            parts.append(
                mandelbrot_section_jit(
                    offset, section_widths[i], width, height,
                    x_min, x_max, y_min, y_max, mi,
                )
            )
            offset += section_widths[i]
        sectioned = np.hstack(parts)
        assert sectioned.shape == full.shape, (
            f"{num_sections} sections: shape {sectioned.shape} != {full.shape}"
        )
        mismatch = (sectioned != full).sum()
        assert mismatch == 0, (
            f"{num_sections} sections: {mismatch} pixels differ from the "
            "full-width reference (section offset bug)"
        )


def test_auto_iterations_scale_with_zoom():
    """The auto iteration count must resolve the classic view (slow-escaping
    boundary points need >= ~256 iterations or bulb boundaries blur into
    black halos), grow with zoom depth, and respect the floor/ceiling. The
    old piecewise formula capped at 500 by zoom ~100, rendering deep-zoom
    filaments as solid black."""
    from Fractal_Frontier import MandelbrotViewer

    f = MandelbrotViewer._auto_iterations_for
    # Bare instance: the method does not use any instance state.
    viewer = object.__new__(MandelbrotViewer)

    assert f(viewer, 1.0) >= 256, "classic view needs enough iterations to resolve the neck"
    assert f(viewer, 100.0) > f(viewer, 10.0) > f(viewer, 1.0), "must grow with zoom depth"
    assert f(viewer, 1e6) == 8192, "ceiling not respected"
    assert f(viewer, 0.26) >= 32, "floor not respected at extreme zoom-out"


# ---------------------------------------------------------------------------
# Viewer level (requires a display)
# ---------------------------------------------------------------------------

def _pump(root, ms):
    """Run the Tk event loop for roughly `ms` milliseconds."""
    end = time.time() + ms / 1000.0
    while time.time() < end:
        root.update()
        time.sleep(0.02)


def _wait_for_render(viewer, root, timeout=20.0):
    """Pump until the most recent render has been applied (or timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        root.update()
        if not str(viewer.status_bar.cget("text")).startswith("Computing"):
            return True
        time.sleep(0.02)
    return False


@pytest.fixture(scope="module")
def app():
    """A live (withdrawn) viewer shared by the viewer-level tests."""
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        pytest.skip("no display available for Tkinter")
    from Fractal_Frontier import MandelbrotViewer

    root.withdraw()
    viewer = MandelbrotViewer(root)
    yield viewer, root
    try:
        viewer.on_close()
    except Exception:
        pass


class _MouseEv:
    """Fake mouse event (type != 2 => mouse path in zoom/fluid_zoom)."""

    type = "4"

    def __init__(self, x=80, y=60):
        self.x = x
        self.y = y


def test_pixels_stay_square_after_resize(app):
    """A 3x3 range on a non-square canvas used to stretch the fractal
    vertically. draw_mandelbrot must re-derive the y range from the canvas
    aspect ratio so pixels are square."""
    viewer, _ = app
    for w, h in ((800, 455), (640, 480), (1024, 768)):
        viewer.width, viewer.height = w, h
        x_range_before = viewer.x_max - viewer.x_min
        viewer.draw_mandelbrot()
        assert abs((viewer.x_max - viewer.x_min) - x_range_before) < 1e-9, (
            "x range must be preserved by the aspect adjustment"
        )
        dx = (viewer.x_max - viewer.x_min) / w
        dy = (viewer.y_max - viewer.y_min) / h
        assert abs(dx - dy) / dx < 1e-9, f"pixels not square at {w}x{h}"
    # Restore the initial view for the other tests.
    viewer.width, viewer.height = 160, 120


def test_single_canvas_image_item_after_redraws(app):
    """Every redraw used to add a new canvas image item that was never
    removed (unbounded leak). Exactly one item must exist after many draws."""
    viewer, root = app
    assert _wait_for_render(viewer, root), "initial render did not complete"
    for _ in range(3):
        viewer.draw_mandelbrot()
    assert _wait_for_render(viewer, root)
    items = [i for i in viewer.canvas.find_all() if viewer.canvas.type(i) == "image"]
    assert len(items) == 1, f"expected 1 canvas image item, found {len(items)}"


def test_rendered_image_covers_canvas_width(app):
    """With width % num_sections != 0 the hstacked sections used to leave a
    black strip on the right. The rendered photo must match the canvas."""
    viewer, root = app
    viewer.width, viewer.height = 853, 600  # 853 is not divisible by core count
    try:
        viewer.draw_mandelbrot()
        assert _wait_for_render(viewer, root)
        assert viewer.photo.width() == 853
    finally:
        viewer.width, viewer.height = 160, 120


def test_negative_offset_clamps_max_iterations(app):
    """The offset slider allows large negative values; max_iterations must be
    clamped to >= 1 instead of dividing by zero in the themes."""
    viewer, root = app
    saved = viewer.iteration_offset
    try:
        viewer.iteration_offset = -10**6
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            viewer.draw_mandelbrot()
        assert viewer.max_iterations == 1
        assert _wait_for_render(viewer, root)
    finally:
        viewer.iteration_offset = saved


def test_fractal_type_changed_syncs_dropdown(app):
    """fractal_type_changed() must update fractal_type_var even when called
    programmatically (bookmarks used to record the wrong fractal type)."""
    viewer, _ = app
    viewer.fractal_type_changed("Julia")
    assert viewer.fractal_type_var.get() == "Julia"
    viewer.fractal_type_changed("Mandelbrot")
    assert viewer.fractal_type_var.get() == "Mandelbrot"


def test_entry_guard_ignores_pan_and_zoom_keys(app):
    """w/a/s/d/q/e and arrows are bound on the root window; they must not pan,
    zoom, or change the theme while the event comes from an Entry."""
    viewer, _ = app
    x0, y0 = viewer.x_min, viewer.y_min
    theme0 = viewer.color_theme

    class EntryEv:
        type = "2"

        def __init__(self):
            self.widget = viewer.julia_c_real_entry

    ev = EntryEv()
    viewer.move_right(ev)
    viewer.move_up(ev)
    viewer.fluid_zoom(ev, 2.0)
    viewer.advance_theme(ev)

    assert (viewer.x_min, viewer.y_min) == (x0, y0), "pan fired inside an entry"
    assert viewer.color_theme == theme0, "theme changed inside an entry"
    assert not viewer.is_zooming


def test_fluid_zoom_releases_lock(app):
    """A failure mid-animation used to leave is_zooming stuck True forever;
    the lock must be released when the animation finishes."""
    viewer, root = app
    viewer.fluid_zoom(_MouseEv(), 2.0)
    _pump(root, 2500)
    assert not viewer.is_zooming


def test_invalid_julia_c_does_not_raise(app):
    """Non-numeric Julia C input used to raise ValueError in the callback."""
    viewer, _ = app

    class BadVar:
        def get(self):
            return "abc"

    saved_real = viewer.julia_c_real_var
    viewer.julia_c_real_var = BadVar()
    try:
        viewer.update_julia_c()  # must not raise
    finally:
        viewer.julia_c_real_var = saved_real
    assert "Invalid" in str(viewer.status_bar.cget("text"))


def test_degenerate_bounds_reset(app):
    """Zooming out past float limits used to crash the render request with a
    math domain error; degenerate bounds must reset to the default view."""
    viewer, root = app
    viewer.x_min, viewer.x_max = -1e308, 1e308
    viewer.y_min, viewer.y_max = -1e308, 1e308
    viewer.draw_mandelbrot()
    assert (viewer.x_min, viewer.x_max) == (-2.0, 1.0)
    # The y range is re-derived for square pixels around center 0.
    expected_y_range = 3.0 * viewer.height / viewer.width
    assert abs((viewer.y_max - viewer.y_min) - expected_y_range) < 1e-9
    assert abs(viewer.y_min + viewer.y_max) < 1e-9
    assert _wait_for_render(viewer, root)


def test_bookmark_roundtrip_and_old_format(app, monkeypatch, tmp_path):
    """Bookmarks are written to BOOKMARKS_FILE (script dir, not CWD), and an
    old-format bookmark with missing keys must load without KeyError."""
    import Fractal_Frontier as FF

    viewer, root = app
    target = tmp_path / "bookmarks.json"
    monkeypatch.setattr(FF, "BOOKMARKS_FILE", str(target))

    viewer.save_bookmark()
    data = json.loads(target.read_text())
    assert data and "fractal_type" in data[-1]

    # Old-format bookmark: only bounds present.
    target.write_text(json.dumps([{"x_min": -1.0, "x_max": 0.0}]))
    viewer.load_bookmark()
    top = [w for w in root.winfo_children() if w.winfo_class() == "Toplevel"][-1]
    lb = [w for w in top.winfo_children() if w.winfo_class() == "Listbox"][-1]
    lb.selection_set(0)
    invoked = False
    for frame in [w for w in top.winfo_children() if w.winfo_class() == "TFrame"]:
        for b in list(frame.winfo_children()):
            if str(b.cget("text")) == "Load Selected":
                b.invoke()  # destroys the dialog, so stop iterating
                invoked = True
                break
        if invoked:
            break
    assert invoked, "Load Selected button not found"
    _pump(root, 1500)
    assert (viewer.x_min, viewer.x_max) == (-1.0, 0.0)


def test_clean_shutdown(app):
    """on_close() must stop the render thread and close the pool."""
    viewer, root = app
    t0 = time.time()
    viewer.on_close()
    assert not viewer._render_thread.is_alive()
    assert time.time() - t0 < 10
