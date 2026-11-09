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
  - Keyboard zoom anchors the view CENTER (Tk event.type is the string
    '2'; comparing it with int 2 made the keyboard branch dead code and
    keyboard zoom anchored at the mouse pointer instead).
  - Mouse zoom scales AROUND the pointer: the world point under the
    cursor is preserved (the old formula centered the final view on the
    cursor point - overshooting the shift everywhere off-center).
  - Panning preserves the view size (delta recomputed after moving the
    first bound used to shrink the viewport 1% per pan keystroke).
  - Wheel notches coalesce into ONE animated, pointer-anchored zoom;
    Windows/macOS deltas map proportionally (trackpads scale smoothly).
  - The zoom-out floor reports "Maximum zoom-out reached" in the status bar.
  - Left button is dual-purpose: plain click = animated zoom-in; drag past
    a 4px threshold = grab-pan (view slides opposite the cursor, size kept);
    sub-threshold jitter stays a click.
  - Invalid Julia C input does not raise.
  - Degenerate view bounds reset instead of crashing the render request.
  - Bookmarks are written to BOOKMARKS_FILE and old formats load safely;
    the write is atomic (temp file + rename).
  - The "CPU Cores" randomizer excludes the meta-theme itself.
  - The worker pool never spawns more processes than sections per render.
  - on_close() shuts down the render thread cleanly.

Calculator level (Julia):
  - The Julia constant c = 0j must be respected, not replaced by the
    default ("c or default" treats 0j as falsy).
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
        assert (
            distinct >= min_distinct
        ), f"{theme}: only {distinct} distinct colors (palette collapsed?)"


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
                    offset,
                    section_widths[i],
                    width,
                    height,
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    mi,
                )
            )
            offset += section_widths[i]
        sectioned = np.hstack(parts)
        assert (
            sectioned.shape == full.shape
        ), f"{num_sections} sections: shape {sectioned.shape} != {full.shape}"
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

    assert (
        f(viewer, 1.0) >= 256
    ), "classic view needs enough iterations to resolve the neck"
    assert (
        f(viewer, 100.0) > f(viewer, 10.0) > f(viewer, 1.0)
    ), "must grow with zoom depth"
    assert f(viewer, 1e6) == 8192, "ceiling not respected"
    assert f(viewer, 0.26) >= 32, "floor not respected at extreme zoom-out"


def test_julia_zero_constant_is_respected():
    """c = 0j is a valid Julia constant (the disk |z| <= 2), but it is
    falsy: the old "c = c or complex(default)" silently replaced it with
    the default -0.7+0.27015j constant. Only None may trigger the default."""
    calc = MandelbrotCalculator()
    original = MandelbrotCalculator._use_gpu
    try:
        MandelbrotCalculator.set_use_gpu(False)
        common = (
            "julia",
            32,
            24,
            -2.0,
            2.0,
            -2.0,
            2.0,
            40,
        )
        zero = calc._compute_iterations(*common, complex(0, 0))
        default = calc._compute_iterations(*common, complex(-0.7, 0.27015))
        none = calc._compute_iterations(*common)  # c defaults

        # c = 0j: center never escapes (disk), the corner escapes immediately.
        assert zero[12, 16] == 40, "z=0 must not escape with c=0j"
        assert zero[0, 0] == 0, "z=-2-2j escapes at once with c=0j"
        # It must be the 0j set, not the default rabbit set.
        assert not np.array_equal(
            zero, default
        ), "c=0j was replaced by the default Julia constant"
        assert np.array_equal(none, default), "c=None must use the default"
    finally:
        MandelbrotCalculator.set_use_gpu(original)


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
    """Fake mouse event: Tk reports ButtonPress with the STRING type '4'
    (the string '4' never equals '2' => mouse path in zoom/fluid_zoom)."""

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
        assert (
            abs((viewer.x_max - viewer.x_min) - x_range_before) < 1e-9
        ), "x range must be preserved by the aspect adjustment"
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
        viewer.iteration_offset = -(10**6)
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


def test_keyboard_zoom_anchors_view_center(app):
    """Keyboard zoom (Next/Prior/q/e) must anchor at the CENTER of the view.

    Tkinter reports event.type as the STRING '2' for key events; the
    comparison "event.type == 2" (int) was always False, so keyboard zoom
    silently used event.x/event.y — the mouse pointer position — as anchor.
    This test fires a REAL Tk key event with the pointer parked at the
    canvas top-left, where pointer-anchoring and center-anchoring differ.
    """
    viewer, root = app
    viewer.x_min, viewer.x_max, viewer.y_min, viewer.y_max = -2.0, 1.0, -1.5, 1.5
    viewer.width, viewer.height = 160, 120
    center0_x, center0_y = -0.5, 0.0  # center of the classic view

    # Key events are only delivered to a mapped window: deiconify off-screen.
    root.geometry("+3000+3000")
    root.deiconify()
    try:
        for _ in range(20):
            root.update()
        root.event_generate("<Next>", x=0, y=0, warp=True)  # zoom in, factor 0.5
        for _ in range(40):
            root.update()
    finally:
        root.withdraw()

    center_x = (viewer.x_min + viewer.x_max) / 2
    center_y = (viewer.y_min + viewer.y_max) / 2
    width = viewer.x_max - viewer.x_min
    assert center_x == pytest.approx(center0_x, abs=1e-12), (
        f"keyboard zoom moved the view center to ({center_x}, {center_y}) "
        "(pointer-anchored — event.type string/int bug)"
    )
    assert center_y == pytest.approx(center0_y, abs=1e-12), (
        f"keyboard zoom moved the view center to ({center_x}, {center_y}) "
        "(pointer-anchored — event.type string/int bug)"
    )
    assert width == pytest.approx(1.5, rel=1e-9), "zoom factor 0.5 must halve the width"

    # fluid_zoom (q/e keys) must anchor at the center as well.
    viewer.x_min, viewer.x_max, viewer.y_min, viewer.y_max = -2.0, 1.0, -1.5, 1.5
    root.deiconify()
    try:
        for _ in range(20):
            root.update()
        root.event_generate("<e>", x=0, y=0, warp=True)  # fluid zoom out, factor 2
        _pump(root, 1500)
    finally:
        root.withdraw()
    assert viewer.x_max - viewer.x_min == pytest.approx(6.0, rel=1e-9)
    assert (viewer.x_min + viewer.x_max) / 2 == pytest.approx(center0_x, abs=1e-12)
    assert not viewer.is_zooming

    viewer.x_min, viewer.x_max, viewer.y_min, viewer.y_max = -2.0, 1.0, -1.5, 1.5
    viewer.draw_mandelbrot()


def test_mouse_zoom_anchors_pointer(app):
    """Mouse zoom must SCALE AROUND the cursor: the world point under the
    pointer lands exactly where it started. The old formula centered the
    final view ON the cursor point - identical only when the cursor is the
    center, overshooting the shift toward the cursor everywhere else."""
    viewer, root = app

    def run(x, y, factor):
        # Aspect-consistent canvas (160x120 => y range = x range * 120/160)
        # so the square-pixel adjustment cannot legitimately alter the y
        # anchor during the zoom.
        viewer.width, viewer.height = 160, 120
        viewer.x_min, viewer.x_max = -2.0, 1.0
        viewer.y_min, viewer.y_max = -1.125, 1.125
        fx, fy = x / 160.0, y / 120.0
        wx = viewer.x_min + fx * (viewer.x_max - viewer.x_min)
        wy = viewer.y_min + fy * (viewer.y_max - viewer.y_min)
        viewer.zoom(_MouseEv(x=x, y=y), factor)
        w = viewer.x_max - viewer.x_min
        h = viewer.y_max - viewer.y_min
        assert w == pytest.approx(3.0 * factor, rel=1e-9)
        assert viewer.x_min + fx * w == pytest.approx(wx, abs=1e-12), (
            "zoom moved the world point under the cursor horizontally "
            "(anchor overshoot)"
        )
        assert viewer.y_min + fy * h == pytest.approx(wy, abs=1e-12), (
            "zoom moved the world point under the cursor vertically "
            "(anchor overshoot)"
        )
        return w

    # Center: the trivial case that always worked.
    assert run(80, 60, 0.5) == pytest.approx(1.5, rel=1e-9)
    # Corner: x_min/y_min must stay put under zoom-in.
    assert run(0, 0, 0.5) == pytest.approx(1.5, rel=1e-9)
    assert viewer.x_min == pytest.approx(-2.0, abs=1e-12)
    assert viewer.y_min == pytest.approx(-1.125, abs=1e-12)
    # Mid off-center: the classic overshoot case, in and out.
    assert run(120, 90, 0.5) == pytest.approx(1.5, rel=1e-9)
    assert run(120, 90, 2.0) == pytest.approx(6.0, rel=1e-9)
    # Extreme off-center (95%).
    assert run(152, 114, 0.8) == pytest.approx(2.4, rel=1e-9)

    viewer.x_min, viewer.x_max, viewer.y_min, viewer.y_max = -2.0, 1.0, -1.5, 1.5
    viewer.draw_mandelbrot()


def test_pan_preserves_view_size(app):
    """Panning must move the view without changing its size. Each pan
    recomputed (max - min) after moving the first bound, shrinking the
    viewport by 1% per keystroke (10 pans => ~10% zoom-in for free)."""
    viewer, root = app
    # Square canvas so the aspect adjustment is a no-op for the size check.
    viewer.width, viewer.height = 160, 160
    viewer.x_min, viewer.x_max, viewer.y_min, viewer.y_max = -2.0, 1.0, -1.5, 1.5

    class PanEv:
        type = "2"

        def __init__(self, root):
            self.widget = root

    ev = PanEv(root)
    w0 = viewer.x_max - viewer.x_min
    h0 = viewer.y_max - viewer.y_min
    for _ in range(10):
        viewer.move_right(ev)
        viewer.move_up(ev)
    assert viewer.x_max - viewer.x_min == pytest.approx(
        w0, rel=1e-12
    ), "horizontal pan changed the view width (1%-shrink bug)"
    assert viewer.y_max - viewer.y_min == pytest.approx(
        h0, rel=1e-12
    ), "vertical pan changed the view height (1%-shrink bug)"
    # The pans must have actually moved the view.
    assert viewer.x_max > 1.0 and viewer.y_max < 1.5

    viewer.x_min, viewer.x_max, viewer.y_min, viewer.y_max = -2.0, 1.0, -1.5, 1.5
    viewer.width, viewer.height = 160, 120
    viewer.draw_mandelbrot()


def test_wheel_factor_from_delta():
    """Windows/macOS wheel deltas must map to proportional zoom factors:
    one 120-unit notch = the notch factor; trackpad micro-deltas scale
    smoothly; direction: scroll up (positive delta) zooms in."""
    from Fractal_Frontier import wheel_factor_from_delta

    assert wheel_factor_from_delta(120) == pytest.approx(1.0 / 1.25, rel=1e-9)
    assert wheel_factor_from_delta(-120) == pytest.approx(1.25, rel=1e-9)
    assert wheel_factor_from_delta(0) == 1.0
    # Small trackpad deltas stay gentle and in the same direction.
    small_in = wheel_factor_from_delta(10)
    small_out = wheel_factor_from_delta(-10)
    assert small_in < 1.0 < small_out
    assert small_in > 0.97 and small_out < 1.04


def test_wheel_notches_coalesce_into_one_animated_zoom(app):
    """Wheel gestures must accumulate for WHEEL_COALESCE_MS and apply as a
    single animated, pointer-anchored zoom (previously each notch was an
    instant jump chained onto a stale, lagging frame)."""
    viewer, root = app
    viewer.x_min, viewer.x_max, viewer.y_min, viewer.y_max = -2.0, 1.0, -1.5, 1.5
    viewer.width, viewer.height = 160, 120

    class WheelEv:
        type = "4"  # ButtonPress-type wheel event (Tk string type)

        def __init__(self, x, y, widget):
            self.x, self.y, self.widget = x, y, widget

    # Three zoom-in notches anchored at the canvas CENTER: they must
    # combine into 0.8^3 and keep the view center where it was.
    factor = 1.0 / 1.25
    for _ in range(3):
        viewer._queue_wheel(WheelEv(80, 60, root), factor)
    _pump(root, 2500)

    assert viewer.x_max - viewer.x_min == pytest.approx(
        3.0 * factor**3, rel=1e-6
    ), "wheel notches did not coalesce into the accumulated factor"
    assert (viewer.x_min + viewer.x_max) / 2 == pytest.approx(-0.5, abs=1e-9)
    assert (viewer.y_min + viewer.y_max) / 2 == pytest.approx(0.0, abs=1e-9)
    assert viewer._wheel_accum == 1.0, "accumulator not reset after flush"
    assert not viewer.is_zooming
    assert _wait_for_render(viewer, root)

    viewer.x_min, viewer.x_max, viewer.y_min, viewer.y_max = -2.0, 1.0, -1.5, 1.5
    viewer.draw_mandelbrot()


def test_zoom_out_floor_shows_status_hint(app):
    """The zoom-out guard (zoom_level <= 0.26) must tell the user why the
    view stopped widening instead of swallowing the gesture silently."""
    viewer, root = app
    saved = viewer.zoom_level
    viewer.zoom_level = 0.2  # pretend we are at the floor
    x0, w0 = viewer.x_min, viewer.x_max - viewer.x_min
    viewer.zoom(_MouseEv(), 2.0)
    assert "Maximum zoom-out reached" in str(viewer.status_bar.cget("text"))
    assert viewer.x_min == x0 and viewer.x_max == w0 + x0, "floor guard moved the view"
    assert not viewer.is_zooming
    viewer.zoom_level = saved


def test_click_zooms_and_drag_pans(app):
    """Left button must do both jobs: a plain press-release (no movement
    past the threshold) is the animated zoom-in click; a drag grabs the
    canvas and pans - the view window slides opposite to the cursor and
    the view SIZE is untouched."""
    viewer, root = app

    # Map the root OFF-SCREEN at a fixed geometry: a withdrawn/mapped root
    # cycles Configure events from earlier tests and the debounced redraw
    # lands mid-animation, letting the square-pixel adjustment legitimately
    # re-derive the y range. With a stable mapped canvas, geometry is quiet.
    root.geometry("+3000+3000")
    root.deiconify()

    class PtEv:
        type = "4"

        def __init__(self, x, y, widget):
            self.x, self.y, self.widget = x, y, widget

    def settled():
        """Put the viewer in the classic view and wait until the canvas
        geometry is QUANTITATIVELY stable (unchanged for ~250 ms, then the
        debounced redraw is flushed), so no resize lands mid-animation.
        The fixture's mapped size is environment-dependent; the test reads
        it back and computes all expectations from it."""
        viewer.x_min, viewer.x_max = -2.0, 1.0
        viewer.y_min, viewer.y_max = -1.125, 1.125
        viewer.draw_mandelbrot()
        last, stable = None, 0
        for _ in range(400):  # max ~4s: window layout can settle slowly
            root.update()
            time.sleep(0.01)
            cur = (viewer.canvas.winfo_width(), viewer.canvas.winfo_height())
            if cur == last:
                stable += 1
                if stable >= 25:
                    break
            else:
                stable, last = 0, cur
        _pump(root, 500)  # flush the final debounced redraw
        W, H = viewer.width, viewer.height
        return (
            W,
            H,
            (
                viewer.x_min,
                viewer.x_max,
                viewer.y_min,
                viewer.y_max,
            ),
        )

    # --- plain click: animated zoom-in at the cursor ---
    W, H, (x0, x1, y0, y1) = settled()
    viewer._pan_press(PtEv(100, 100, root))
    viewer._pan_release(PtEv(100, 100, root))
    _pump(root, 2500)
    assert viewer.x_max - viewer.x_min == pytest.approx(
        (x1 - x0) * 0.125, rel=1e-6
    ), "plain click did not zoom"
    # the world point under the cursor is preserved by the animated zoom
    fx, fy = 100 / W, 100 / H
    wx = x0 + fx * (x1 - x0)
    wy = y0 + fy * (y1 - y0)
    w = viewer.x_max - viewer.x_min
    h = viewer.y_max - viewer.y_min
    assert viewer.x_min + fx * w == pytest.approx(wx, abs=1e-9)
    # y gets a 3% tolerance: render status text changes can grow the
    # statusbar, shrink the canvas height mid-animation, and the
    # square-pixel adjustment then legitimately re-derives the y range.
    # (The anchor-overshoot bug this guards moved the point by 40%+.)
    assert viewer.y_min + fy * h == pytest.approx(
        wy, abs=0.03 * (y1 - y0)
    ), "y anchor overshoot"

    # --- drag: pans, keeps size, slides opposite the cursor ---
    W, H, (x0, x1, y0, y1) = settled()
    viewer._pan_press(PtEv(100, 100, root))
    viewer._pan_motion(PtEv(120, 100, root))  # 20 px > 4 px threshold
    viewer._pan_motion(PtEv(140, 100, root))  # total dx = 40 px
    viewer._pan_motion(PtEv(140, 118, root))  # total dy = 18 px
    assert viewer._panning is True
    viewer._pan_release(PtEv(140, 118, root))
    _pump(root, 1500)
    assert viewer._panning is False, "pan state not cleared on release"
    assert str(viewer.canvas.cget("cursor")) in ("", "XCursor"), "grab cursor left on"
    assert viewer.x_max - viewer.x_min == pytest.approx(
        x1 - x0, rel=1e-12
    ), "drag changed the view width (click zoom fired on a pan)"
    # grabbed content follows the cursor: window moves the other way
    assert viewer.x_min == pytest.approx(x0 - 40 / W * (x1 - x0), rel=1e-12)
    # y tolerance: the control frame's auto-resize (slider range digits)
    # can widen the window mid-drag, and the square-pixel adjustment then
    # legitimately rescales the y range, damping the y shift. The x axis
    # is untouched by that and asserted exactly.
    assert viewer.y_min == pytest.approx(
        y0 - 18 / H * (y1 - y0), abs=0.2 * (y1 - y0)
    ), "drag did not pan vertically with the grabbed content"

    # --- sub-threshold jitter must stay a click ---
    W, H, (x0, x1, y0, y1) = settled()
    viewer._pan_press(PtEv(100, 100, root))
    viewer._pan_motion(PtEv(102, 100, root))  # 2 px < threshold
    assert viewer._panning is False, "jitter started a pan"
    viewer._pan_release(PtEv(102, 100, root))
    _pump(root, 2500)
    assert viewer.x_max - viewer.x_min == pytest.approx(
        (x1 - x0) * 0.125, rel=1e-6
    ), "jittery click did not zoom"

    settled()
    root.withdraw()


def test_cpu_cores_randomizer_excludes_meta_theme(app):
    """The "CPU Cores" theme assigns a random theme per section. Picking
    from the full themes list could select "CPU Cores" itself; it has no
    theme function and silently rendered that section as "Default"."""
    viewer, _ = app
    assert (
        viewer.CPU_CORES_THEME not in viewer._theme_choices
    ), "CPU Cores randomizer pool contains the meta-theme"
    assert set(viewer._theme_choices) == set(viewer.themes) - {viewer.CPU_CORES_THEME}


def test_pool_is_capped_at_section_limit(app):
    """A render submits at most 32 section tasks; the pool must not spawn
    more worker processes than can ever receive one."""
    viewer, _ = app
    assert viewer.pool._processes <= 32
    assert viewer.pool._processes == max(1, min(viewer.num_cores, 32))


def test_pan_preview_cache_is_cropped_without_recompute(app):
    """The pan preview renders an overscan region of raw iterations; a small
    drag must be served by cropping it instead of requesting a full render
    for every motion event."""
    viewer, root = app
    viewer.width, viewer.height = 160, 120
    viewer.x_min, viewer.x_max = -2.0, 1.0
    viewer.y_min, viewer.y_max = -1.125, 1.125
    viewer.zoom_level = viewer.base_range / 3.0
    viewer._pan_cache = None
    try:
        viewer.draw_mandelbrot(preview=True, overscan=True)
        deadline = time.time() + 5.0
        while time.time() < deadline and viewer._pan_cache is None:
            root.update()
            time.sleep(0.02)
        assert viewer._pan_cache is not None, "no pan cache was stored"
        crop = viewer._cache_crop_for_current_view()
        assert crop is not None and crop.ndim == 2 and crop.size > 0

        class PtEv:
            type = "4"

            def __init__(self, x, y, widget):
                self.x, self.y, self.widget = x, y, widget

        seq_before = viewer._render_seq
        viewer._pan_start = (80, 60)
        viewer._pan_last = (80, 60)
        viewer._panning = True
        viewer._pan_motion(PtEv(90, 60, root))
        # A cached crop is instant; no new render sequence is queued.
        assert viewer._render_seq == seq_before + 1
        assert viewer._pan_cache is not None
        viewer._pan_release(PtEv(90, 60, root))
        viewer.draw_mandelbrot()
        assert _wait_for_render(viewer, root)
    finally:
        viewer._pan_cache = None
        viewer.width, viewer.height = 160, 120
        viewer.x_min, viewer.x_max = -2.0, 1.0
        viewer.y_min, viewer.y_max = -1.125, 1.125
        viewer.draw_mandelbrot()


@pytest.fixture
def tile_view(app):
    """Set up a view large enough to seed complete tiles, then restore it."""
    viewer, root = app
    saved = (
        viewer.width,
        viewer.height,
        viewer.x_min,
        viewer.x_max,
        viewer.y_min,
        viewer.y_max,
        viewer.zoom_level,
        viewer.color_theme,
        viewer.max_iterations,
    )
    viewer.width, viewer.height = 300, 200
    viewer.x_min, viewer.x_max = -2.0, 1.0
    viewer.y_min, viewer.y_max = -1.0, 1.0
    viewer.zoom_level = 1.0
    viewer.draw_mandelbrot()
    assert _wait_for_render(viewer, root), "tile setup render did not complete"
    try:
        yield viewer, root
    finally:
        viewer._interacting = False
        viewer._interaction_mode = None
        viewer._panning = False
        viewer._pan_start = None
        viewer._pan_last = None
        (
            viewer.width,
            viewer.height,
            viewer.x_min,
            viewer.x_max,
            viewer.y_min,
            viewer.y_max,
            viewer.zoom_level,
            viewer.color_theme,
            viewer.max_iterations,
        ) = saved
        viewer.draw_mandelbrot()
        _wait_for_render(viewer, root)


def test_full_render_seeds_tile_cache(tile_view):
    """A settled render seeds full-resolution tiles and requests a ring."""
    viewer, _ = tile_view
    assert viewer._tile_cache_is_valid()
    assert viewer._tiles, "no complete tiles were seeded"
    assert viewer._tile_pending, "no surrounding ring was requested"
    crop = viewer._compose_tile_view()
    assert crop is not None and crop.ndim == 2 and crop.size > 0


def test_tile_ring_is_computed_asynchronously(tile_view):
    """The worker fills the adjacent ring while the main thread stays free."""
    viewer, root = tile_view
    seeded = len(viewer._tiles)
    deadline = time.time() + 6.0
    while time.time() < deadline and viewer._tile_pending:
        root.update()
        time.sleep(0.02)
    assert not viewer._tile_pending, "ring tiles never finished"
    assert len(viewer._tiles) > seeded, "ring tiles were not stored"
    assert viewer._display_tile_view()


def test_pan_uses_tile_cache_without_full_render(tile_view):
    """A drag served by cached tiles must not enqueue a frame render."""
    viewer, root = tile_view

    class PtEv:
        type = "4"

        def __init__(self, x, y, widget):
            self.x, self.y, self.widget = x, y, widget

    seq_before = viewer._render_seq
    viewer._pan_start = (150, 100)
    viewer._pan_last = (150, 100)
    viewer._panning = True
    viewer._pan_motion(PtEv(190, 100, root))
    assert viewer._render_seq == seq_before + 1
    assert viewer._render_queue.empty(), "pan requested a full frame render"
    assert viewer._display_tile_view()
    needed = viewer._tile_index_range(
        viewer.x_min,
        viewer.x_min_lo,
        viewer.x_max,
        viewer.x_max_lo,
        viewer.y_min,
        viewer.y_min_lo,
        viewer.y_max,
        viewer.y_max_lo,
        ring=1,
    )
    assert set(viewer._tiles) <= needed, "tiles outside the ring were kept"
    viewer._pan_release(PtEv(190, 100, root))


def test_tile_cache_retains_recent_tiles_for_reversal(tile_view):
    """A quick pan out and back reuses retained tiles instead of recomputing."""
    viewer, _ = tile_view
    assert viewer._tile_cache_is_valid()
    old_key = next(iter(viewer._tiles))
    saved = (viewer.x_min, viewer.x_max, viewer.y_min, viewer.y_max)

    shift = 5 * viewer._tile_world
    viewer.x_min -= shift
    viewer.x_max -= shift
    viewer._request_tiles_for_view()
    assert old_key in viewer._tiles, "recent tile was evicted during the pan"

    viewer.x_min, viewer.x_max, viewer.y_min, viewer.y_max = saved
    viewer._request_tiles_for_view()
    assert old_key in viewer._tiles
    assert old_key not in viewer._tile_pending


def test_tile_cache_is_theme_independent_but_zoom_dependent(tile_view):
    """Tiles hold raw iterations, so theme changes reuse them; zoom does not."""
    viewer, _ = tile_view
    assert viewer._tile_cache_is_valid()
    viewer.color_theme = "Grayscale"
    assert viewer._tile_cache_is_valid()
    assert viewer._display_tile_view()

    viewer.max_iterations += 1
    assert not viewer._tile_cache_is_valid()
    viewer.max_iterations -= 1
    assert viewer._tile_cache_is_valid()

    viewer.zoom_level *= 2.0
    assert not viewer._tile_cache_is_valid()


def test_tile_display_uses_full_resolution_when_frame_budget_is_met(tile_view):
    """Full-resolution tiles are not unconditionally downscaled for display."""
    viewer, _ = tile_view
    viewer._tile_display_scale = 1.0
    viewer._tile_slow_streak = 0
    viewer._tile_fast_streak = 0
    assert viewer._display_tile_view()
    assert viewer._tile_display_scale == 1.0


def test_title_reports_three_decimal_elapsed_and_fps(app):
    """The title bar shows sub-millisecond render times and live FPS."""
    viewer, _ = app
    viewer._fps = 59.94
    viewer._update_title(0.004)
    title = viewer.master.title()
    assert "0.004s" in title
    assert "FPS: 59.9" in title


def test_title_zoom_shows_full_digits_and_compact_form(app):
    """Deep zooms must show both the full integer expansion and a compact
    scientific form, so a truncated title still carries the readable value.

    The title reads the canonical Decimal zoom, so the full form has no
    float64 tail (a float 1.5453264459925081e31 expands to
    …5081369940833861632; the view's own value is exact)."""
    from decimal import Decimal

    from Fractal_Frontier import zoom_forms

    viewer, _ = app
    viewer._zoom_dec = Decimal("1.5453264459925081e31")
    viewer._update_title(0.004)
    title = viewer.master.title()
    assert "1.54533e+31x" in title
    assert "15453264459925081000000000000000" in title
    # The float form keeps working for callers that pass a plain zoom.
    full, compact = zoom_forms(1.5453264459925081e31)
    assert full == "15453264459925081369940833861632"
    assert compact == "1.54533e+31"


def test_preview_scale_degrades_only_when_frame_budget_is_missed(app):
    """The adaptive controller keeps full resolution unless a frame is slow."""
    from Fractal_Frontier import FRAME_BUDGET_MS

    viewer, _ = app

    scale, slow, fast = viewer._adapt_preview_scale(1.0, FRAME_BUDGET_MS * 0.5, 0, 0)
    assert scale == 1.0
    assert (slow, fast) == (0, 0)

    scale, slow, fast = viewer._adapt_preview_scale(1.0, FRAME_BUDGET_MS * 1.2, 0, 0)
    assert scale == 0.5

    # Several cheap frames are required before climbing back to full.
    scale = 0.5
    slow = fast = 0
    for _ in range(5):
        scale, slow, fast = viewer._adapt_preview_scale(
            scale, FRAME_BUDGET_MS * 0.2, slow, fast
        )
    assert scale == 1.0


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
    # Atomic write: the temp file must be renamed away, never left behind.
    assert not (tmp_path / "bookmarks.json.tmp").exists()

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


# The shared `app` fixture is module-scoped and test_clean_shutdown closes
# it, so the spinner tests build their own fresh viewer (the spinner state
# must start clean; a closed viewer's draw_mandelbrot is a no-op).
@pytest.mark.timeout(30)
def test_spinner_arms_with_inflight_render():
    """draw_mandelbrot() arms the busy indicator; the tick animates it."""
    import tkinter as tk

    from Fractal_Frontier import MandelbrotViewer

    root = tk.Tk()
    root.withdraw()
    viewer = MandelbrotViewer(root)
    try:
        viewer.draw_mandelbrot()
        assert viewer._pending_seq == viewer._render_seq, "request not armed"
        viewer._tick_spinner()
        first = viewer.status_bar.cget("text")
        assert "Computing" in first, first
        # Force the next tick: glyph must advance while the frame is in flight.
        viewer._spinner_next_at = 0.0
        viewer._tick_spinner()
        second = viewer.status_bar.cget("text")
        assert "Computing" in second, second
        assert second != first, "spinner glyph did not advance"
    finally:
        viewer.on_close()  # destroys the root itself
        try:
            root.destroy()
        except tk.TclError:
            pass


@pytest.mark.timeout(30)
def test_spinner_disarms_when_the_render_lands():
    """The settled frame replaces the spinner text; pending clears."""
    import tkinter as tk

    from Fractal_Frontier import MandelbrotViewer

    root = tk.Tk()
    root.withdraw()
    viewer = MandelbrotViewer(root)
    try:
        viewer.draw_mandelbrot()
        deadline = time.time() + 15
        while viewer._pending_seq is not None and time.time() < deadline:
            root.update()
            time.sleep(0.02)
        assert viewer._pending_seq is None, "render never settled"
        assert "Computing" not in viewer.status_bar.cget("text")
        # Tick with nothing pending must be inert.
        text = viewer.status_bar.cget("text")
        viewer._tick_spinner()
        assert viewer.status_bar.cget("text") == text
    finally:
        viewer.on_close()  # destroys the root itself
        try:
            root.destroy()
        except tk.TclError:
            pass
