"""
Fractal Frontier - An interactive Mandelbrot set explorer with multiprocessing support.

This application provides a GUI for exploring the Mandelbrot set fractal with various
color themes, zoom capabilities, and bookmark functionality.
"""

import json
import math
import multiprocessing
import os
import platform
import queue
import random
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from fractals import GPU_AVAILABLE, MandelbrotCalculator
from themes import apply_color_theme

# Bookmarks live next to the script (not in whatever CWD the app was
# launched from), so they are found consistently.
BOOKMARKS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "bookmarks.json"
)

# Per-notch wheel zoom factor (zoom-out; zoom-in is its reciprocal rounded
# the same way) used on every platform: a 120-unit wheel notch maps to 1.25x,
# smaller trackpad deltas scale proportionally.
WHEEL_NOTCH_FACTOR = 1.25
WHEEL_COALESCE_MS = 120  # wheel notches merge into one animated zoom

# Float32 GPU arithmetic is ~32-64x faster than float64 on consumer cards,
# but its ~1e-7 relative precision stops resolving deep-zoom detail.  Use it
# only up to this zoom level; deeper views keep the float64 kernel.
GPU_FLOAT32_MAX_ZOOM = 10.0

# Interactive preview settings.  The overscan fallback renders at quarter
# resolution over a 3x-viewport region when the tile cache is unavailable.
PAN_PREVIEW_SCALE = 0.25
PAN_PREVIEW_MARGIN = 1.0
POLL_MS = 16

# Preview quality is adaptive: start at full resolution and only step down
# when a measured preview frame misses the 60-fps budget.  Recovery requires
# several consecutive cheap frames, which avoids oscillating between levels.
PREVIEW_SCALES = (1.0, 0.5, 0.25)
FRAME_BUDGET_MS = 1000.0 / 60.0
FAST_FRAME_FRACTION = 0.4
FAST_FRAMES_BEFORE_RAISE = 5

# Pan tile cache.  Tiles are full-resolution iteration arrays, keyed by an
# integer grid anchored at the view when a full render settles.  A pan only
# needs the visible tiles plus one ring of adjacent tiles; the ring is
# computed asynchronously in the background so moving towards a missing tile
# does not freeze the canvas.
TILE_PX = 128
TILE_RING = 1


def wheel_factor_from_delta(delta):
    """Wheel factor for a Tk delta (Windows/macOS): scroll up (positive,
    zoom in) => 1/WHEEL_NOTCH_FACTOR per 120 units; scroll down (negative,
    zoom out) => WHEEL_NOTCH_FACTOR. Trackpad micro-deltas scale smoothly."""
    return WHEEL_NOTCH_FACTOR ** (-delta / 120.0)


# --------------------------------------------------------------------------
# MAIN VIEWER CLASS
# --------------------------------------------------------------------------
class MandelbrotViewer:
    """
    Main class for the Mandelbrot set viewer application.

    Provides a GUI for exploring the Mandelbrot set with interactive zooming,
    panning, color theme selection, and bookmark functionality.
    """

    def __init__(self, master):
        """
        Initialize the Mandelbrot viewer.

        Args:
            master: Tkinter root window
        """
        self.master = master
        self.app_name = "Fractal Frontier V1.2"
        self._fps = None
        self._last_preview_frame_at = None
        self._title_elapsed = 0.0
        self._last_title_update = 0.0
        self.width = 160
        self.height = 120
        self.max_iterations = 100
        self.x_min = np.float64(-2.0)
        self.x_max = np.float64(1.0)
        self.y_min = np.float64(-1.5)
        self.y_max = np.float64(1.5)
        self.base_range = self.x_max - self.x_min  # 3.0
        self.zoom_level = 1.0
        self.is_zooming = False
        self.iteration_offset = 0
        self.auto_iterations = 50
        self.color_theme = "Default"
        self.julia_c = complex(-0.7, 0.27015)

        self.auto_adjust = True

        # GPU mode - enabled by default if available
        self.use_gpu = GPU_AVAILABLE
        self._render_precision = "float64"

        self.canvas = tk.Canvas(
            master, width=self.width, height=self.height, bg="black"
        )
        self.canvas.pack(expand=True, fill="both")
        self.canvas.bind("<Configure>", self.on_resize)

        self.control_frame = ttk.Frame(master)
        self.control_frame.pack()

        self.offset_label = ttk.Label(self.control_frame, text="Iteration Offset:")
        self.offset_label.grid(row=0, column=0, padx=5, pady=5)
        self.offset_scale = tk.Scale(
            self.control_frame,
            from_=0,
            to=2000,
            orient=tk.HORIZONTAL,
            command=self.slider_update,
        )
        self.offset_scale.set(self.iteration_offset)
        self.offset_scale.grid(row=0, column=1, padx=5, pady=5)

        self.theme_label = ttk.Label(self.control_frame, text="Color Theme:")
        self.theme_label.grid(row=0, column=2, padx=5, pady=5)
        self.CPU_CORES_THEME = "CPU Cores"
        self.themes = [
            "Default",
            "Grayscale",
            "Blue",
            "Fire",
            "Rainbow",
            "Rainbow2",
            "Rainbow3",
            "Rainbow4",
            self.CPU_CORES_THEME,
        ]
        self.color_theme_var = tk.StringVar(value=self.color_theme)
        # Theme pool for the "CPU Cores" randomizer: it must exclude the
        # meta-theme itself. A section that drew "CPU Cores" had no matching
        # theme function and silently rendered as "Default".
        self._theme_choices = [
            theme for theme in self.themes if theme != self.CPU_CORES_THEME
        ]
        self.theme_menu = ttk.OptionMenu(
            self.control_frame,
            self.color_theme_var,
            self.color_theme,
            *self.themes,
            command=self.theme_changed,
        )
        self.theme_menu.grid(row=0, column=3, padx=5, pady=5)

        self.save_bookmark_btn = ttk.Button(
            self.control_frame, text="Save Bookmark", command=self.save_bookmark
        )
        self.save_bookmark_btn.grid(row=1, column=1, padx=5, pady=5)
        self.load_bookmark_btn = ttk.Button(
            self.control_frame, text="Load/Del Bookmark", command=self.load_bookmark
        )
        self.load_bookmark_btn.grid(row=1, column=2, padx=5, pady=5)

        self.reset_button = ttk.Button(
            self.control_frame, text="Reset", command=self.reset_view
        )
        self.reset_button.grid(row=1, column=3, padx=5, pady=5)

        self.auto_adjust_var = tk.BooleanVar(
            value=self.auto_adjust
        )  # Bind to self.auto_adjust
        self.auto_adjust_checkbox = ttk.Checkbutton(
            self.control_frame,
            text="Auto Adjust",
            variable=self.auto_adjust_var,
            command=self.toggle_auto_adjust,
        )
        self.auto_adjust_checkbox.grid(row=1, column=0, padx=5, pady=5)

        # GPU toggle checkbox
        if GPU_AVAILABLE:
            self.gpu_var = tk.BooleanVar(value=self.use_gpu)
            self.gpu_checkbox = ttk.Checkbutton(
                self.control_frame,
                text="GPU",
                variable=self.gpu_var,
                command=self.toggle_gpu,
            )
            self.gpu_checkbox.grid(row=1, column=4, padx=5, pady=5)
        else:
            self.gpu_var = None

        self.status_bar = ttk.Label(
            master, text="", relief=tk.SUNKEN, anchor="w", font=("Consolas", 10)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Add fractal type dropdown
        self.fractal_type_label = ttk.Label(self.control_frame, text="Fractal Type:")
        self.fractal_type_label.grid(row=0, column=4, padx=5, pady=5)
        self.fractal_types = ["Mandelbrot", "Julia", "Fatou"]
        self.fractal_type_var = tk.StringVar(value="Mandelbrot")
        self.fractal_type_menu = ttk.OptionMenu(
            self.control_frame,
            self.fractal_type_var,
            "Mandelbrot",
            *self.fractal_types,
            command=self.fractal_type_changed,
        )
        self.fractal_type_menu.grid(row=0, column=5, padx=5, pady=5)

        # Add sliders for julia_c in the control frame
        self.julia_c_real_label = ttk.Label(self.control_frame, text="Julia C Real:")
        self.julia_c_real_label.grid(
            row=3, column=0, columnspan=3, padx=5, pady=5, sticky="ew"
        )  # Label on top
        self.julia_c_real_var = tk.DoubleVar(value=-0.7)  # Default real part
        self.julia_c_real_entry = ttk.Entry(
            self.control_frame, textvariable=self.julia_c_real_var, width=10
        )
        self.julia_c_real_entry.grid(row=3, column=1, padx=5, pady=5)
        self.julia_c_real_entry.bind("<Return>", self.update_julia_c)

        self.julia_c_real_slider = tk.Scale(
            self.control_frame,
            from_=-2.0,
            to=2.0,
            resolution=0.0001,  # Higher resolution for finer adjustments
            orient=tk.HORIZONTAL,
            variable=self.julia_c_real_var,
            command=lambda value: self.update_julia_c_slider(),
        )
        self.julia_c_real_slider.grid(
            row=4, column=0, columnspan=8, padx=1, pady=1, sticky="ew"
        )  # Expand slider

        self.julia_c_imag_label = ttk.Label(self.control_frame, text="Julia C Imag:")
        self.julia_c_imag_label.grid(
            row=5, column=0, columnspan=3, padx=5, pady=5, sticky="ew"
        )  # Label on top
        self.julia_c_imag_var = tk.DoubleVar(value=0.27015)  # Default imaginary part
        self.julia_c_imag_entry = ttk.Entry(
            self.control_frame, textvariable=self.julia_c_imag_var, width=10
        )
        self.julia_c_imag_entry.grid(row=5, column=1, padx=5, pady=5)
        self.julia_c_imag_entry.bind("<Return>", self.update_julia_c)

        self.julia_c_imag_slider = tk.Scale(
            self.control_frame,
            from_=-2.0,
            to=2.0,
            resolution=0.0001,  # Higher resolution for finer adjustments
            orient=tk.HORIZONTAL,
            variable=self.julia_c_imag_var,
            command=lambda value: self.update_julia_c_slider(),
        )
        self.julia_c_imag_slider.grid(
            row=6, column=0, columnspan=8, padx=1, pady=1, sticky="ew"
        )  # Expand slider

        # Add slider for "e exp i n" in the control frame
        self.e_exp_i_n_label = ttk.Label(
            self.control_frame, text="0.7885 * e^(i * slider_value)"
        )
        self.e_exp_i_n_label.grid(
            row=7, column=0, columnspan=3, padx=5, pady=5, sticky="ew"
        )  # Label on top

        self.e_exp_i_n_var = tk.DoubleVar(value=0.0)  # Default value for the slider
        self.e_exp_i_n_slider = tk.Scale(
            self.control_frame,
            from_=0.0,
            to=2 * math.pi,  # Slider range from 0 to 2π
            resolution=0.01,  # Higher resolution for finer adjustments
            orient=tk.HORIZONTAL,
            variable=self.e_exp_i_n_var,
            command=lambda value: self.update_e_exp_i_n_slider(),
        )
        self.e_exp_i_n_slider.grid(
            row=8, column=0, columnspan=8, padx=5, pady=5, sticky="ew"
        )  # Expand slider

        # Hide Julia C input fields initially
        self.julia_c_real_entry.grid_remove()
        self.julia_c_imag_entry.grid_remove()
        # Hide Julia C sliders initially
        self.julia_c_real_label.grid_remove()
        self.julia_c_real_slider.grid_remove()
        self.julia_c_imag_label.grid_remove()
        self.julia_c_imag_slider.grid_remove()
        # Hide the new slider initially
        self.e_exp_i_n_label.grid_remove()
        self.e_exp_i_n_slider.grid_remove()

        # Left press = candidate zoom; a drag beyond the threshold turns it
        # into a pan (the canvas is "grabbed": the image follows the cursor,
        # the view window slides the other way). A release WITHOUT a drag is
        # the plain animated zoom-in click.
        self._pan_last = None
        self._pan_start = None
        self._panning = False
        self.canvas.bind("<Button-1>", self._pan_press)
        self.canvas.bind("<B1-Motion>", self._pan_motion)
        self.canvas.bind("<ButtonRelease-1>", self._pan_release)
        self.canvas.bind("<Button-2>", self.advance_theme)
        self.canvas.bind("<Button-3>", lambda event: self.fluid_zoom(event, 8))

        # Wheel zoom: notches are accumulated for WHEEL_COALESCE_MS and
        # applied as ONE animated, pointer-anchored zoom, so a wheel spin
        # feels like the smooth click zoom instead of a chain of hard jumps.
        self._wheel_accum = 1.0
        self._wheel_x = 0
        self._wheel_y = 0
        self._wheel_widget = self.canvas
        self._wheel_job = None
        system = platform.system()
        if system == "Linux":
            self.canvas.bind(
                "<Button-4>", lambda e: self._queue_wheel(e, 1.0 / WHEEL_NOTCH_FACTOR)
            )
            self.canvas.bind(
                "<Button-5>", lambda e: self._queue_wheel(e, WHEEL_NOTCH_FACTOR)
            )
        elif system == "Windows":
            # MouseWheelTiny/MouseWheelBig are the Precision-touchpad
            # variants; without them two-finger scrolling does nothing.
            for seq in ("<MouseWheel>", "<MouseWheelTiny>", "<MouseWheelBig>"):
                self.canvas.bind(seq, self._queue_delta_wheel)
        elif system == "Darwin":  # macOS
            for seq in ("<MouseWheel>", "<MacMouseWheel>"):
                self.canvas.bind(seq, self._queue_delta_wheel)

        self.master.bind("<Tab>", self.advance_theme)
        self.master.bind("<Next>", lambda event: self.zoom(event, 0.5))
        self.master.bind("<Prior>", lambda event: self.zoom(event, 2))
        self.master.bind("<q>", lambda event: self.fluid_zoom(event, 0.5))
        self.master.bind("<e>", lambda event: self.fluid_zoom(event, 2))

        self.master.bind("<w>", self.move_up)
        self.master.bind("<s>", self.move_down)
        self.master.bind("<a>", self.move_left)
        self.master.bind("<d>", self.move_right)
        self.master.bind("<Up>", self.move_up)
        self.master.bind("<Down>", self.move_down)
        self.master.bind("<Left>", self.move_left)
        self.master.bind("<Right>", self.move_right)

        self.num_cores = multiprocessing.cpu_count()
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
        # A render never submits more than 32 section tasks, so a pool
        # larger than 32 only spawns workers that can never get work.
        self.pool = multiprocessing.Pool(processes=max(1, min(self.num_cores, 32)))
        self.calculator = MandelbrotCalculator()  # Reintroduce the calculator
        self.resize_job = None

        # Rendering happens on a worker thread so the Tk main loop (and thus
        # the UI) never freezes while fractals are being computed. Only the
        # most recent request is ever rendered: pending requests are dropped.
        # The worker never touches Tk directly (Tk is not thread-safe); it
        # hands finished results back through a plain queue that the main
        # thread polls periodically.
        self._render_queue = queue.Queue()
        self._tile_queue = queue.Queue()
        self._result_queue = queue.Queue()
        self._render_seq = 0
        self._last_applied_seq = 0
        self._closed = False
        self._image_item = None
        # Interaction state: previews are shown while the user drags or
        # animates a zoom, then a full-resolution render is requested when
        # the interaction settles.
        self._interacting = False
        self._interaction_mode = None  # "pan" or "zoom"
        self._pan_cache = None

        # Raw iteration tile cache for panning.  It is seeded from every
        # settled full-resolution render, then extended in a ring by the
        # worker thread.  `_tile_epoch` invalidates results from old zooms.
        self._tiles = {}
        self._tile_epoch = 0
        self._tile_anchor_x = None
        self._tile_anchor_y = None
        self._tile_world = None
        self._tile_px = None
        self._tile_params = None
        self._tile_pending = set()
        self._tile_display_scale = 1.0
        self._tile_slow_streak = 0
        self._tile_fast_streak = 0
        self._zoom_preview_scale = 1.0
        self._zoom_slow_streak = 0
        self._zoom_fast_streak = 0
        self._render_thread = threading.Thread(target=self._render_worker, daemon=True)
        self._render_thread.start()
        self._poll_results()

        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        self.draw_mandelbrot()

    class _WheelEvent:
        """Pseudo mouse event for the coalesced wheel flush: wheel events
        are ButtonPress-type, anchored at the last cursor position seen."""

        type = "4"

        def __init__(self, x, y, widget):
            self.x = x
            self.y = y
            self.widget = widget

    def _queue_wheel(self, event, factor):
        """Accumulate one wheel notch (factor + anchor); flush as a single
        animated zoom after the gesture pauses."""
        if self._wheel_job is not None:
            self.master.after_cancel(self._wheel_job)
        self._wheel_accum *= factor
        self._wheel_x = event.x
        self._wheel_y = event.y
        self._wheel_widget = event.widget
        self._wheel_job = self.master.after(WHEEL_COALESCE_MS, self._flush_wheel)

    def _queue_delta_wheel(self, event):
        """Windows/macOS wheel entry: convert the Tk delta to a factor."""
        self._queue_wheel(event, wheel_factor_from_delta(event.delta))

    def _flush_wheel(self):
        """Apply the accumulated wheel factor as one animated zoom."""
        self._wheel_job = None
        if getattr(self, "_closed", False):
            return
        factor = self._wheel_accum
        self._wheel_accum = 1.0
        if abs(factor - 1.0) < 1e-9:
            return
        if self.is_zooming:
            # A click/fluid animation is running; retry shortly so the
            # gesture is applied smoothly instead of being swallowed.
            self._wheel_job = self.master.after(60, self._flush_wheel)
            return
        ev = self._WheelEvent(self._wheel_x, self._wheel_y, self._wheel_widget)
        self.fluid_zoom(
            ev, factor, frames=6, duration=WHEEL_COALESCE_MS / 1000.0 + 0.05
        )

    def slider_update(self, value):
        """
        Update the iteration offset when the slider value changes.

        Args:
            value (str): New slider value
        """
        try:
            self.iteration_offset = int(value)
            self.draw_mandelbrot()
        except ValueError:
            pass

    def theme_changed(self, value):
        """
        Update the color theme when the dropdown selection changes.

        Args:
            value (str): New color theme name
        """
        self.color_theme = value
        self.draw_mandelbrot()

    def on_resize(self, event):
        """
        Handle window resize events with debouncing.

        Args:
            event: Tkinter resize event
        """
        if self.resize_job is not None:
            self.master.after_cancel(self.resize_job)
        self.resize_job = self.master.after(300, self.redraw)

    def redraw(self):
        """
        Redraw the Mandelbrot set after a resize event.
        """
        self.width = self.canvas.winfo_width()
        self.height = self.canvas.winfo_height()
        # During window creation/resize the canvas may briefly report a
        # degenerate size; skip those instead of rendering 1px images.
        if self.width < 2 or self.height < 2:
            return
        self.draw_mandelbrot()

    def toggle_gpu(self):
        """Toggle GPU acceleration on/off, ensuring CPU is used when GPU
        is off and vice versa."""
        self.use_gpu = self.gpu_var.get()
        if self.use_gpu:
            # GPU enabled — ensure calculator uses GPU
            MandelbrotCalculator.set_use_gpu(True)
            self.status_bar.config(text="Switched to GPU backend")
        else:
            # GPU disabled — explicitly force CPU mode on the calculator
            MandelbrotCalculator.set_use_gpu(False)
            self.status_bar.config(
                text=f"Switched to CPU backend ({self.num_cores} cores)"
            )
        self.draw_mandelbrot()

    def _auto_iterations_for(self, zoom_level):
        """Base iteration count for a given zoom level.

        Points near the set boundary escape slowly, and the required
        iteration count grows with zoom depth. The previous piecewise formula
        capped at 500 (reached by zoom ~100), so deeper views rendered bulb
        boundaries as black halos and the filament regions between components
        as solid black. The count now scales smoothly with zoom:
        256 at the classic view, ~640 at zoom 10, ~2560 at zoom 100,
        ~4050 at zoom 1000, capped at 8192 for extreme depths.
        """
        iters = int(256 * (max(zoom_level, 1e-9) ** 0.4))
        return min(max(iters, 32), 8192)

    def _adjust_aspect(self):
        """Keep pixels square by matching the y range to the canvas aspect.

        The x range and both centers are preserved; the y range is re-derived
        as x_range * height / width around the current y center. Without this,
        a 3x3 range drawn on a non-square canvas stretches the fractal
        vertically (the cardioid looks elliptical). Zoom and pan keep the
        ranges scaled equally, so this invariant holds through all of them.
        """
        if self.width < 2 or self.height < 2:
            return
        x_range = float(self.x_max - self.x_min)
        if not (x_range > 0 and math.isfinite(x_range)):
            return
        y_center = (self.y_min + self.y_max) / 2
        y_range = x_range * self.height / self.width
        self.y_min = y_center - y_range / 2
        self.y_max = y_center + y_range / 2

    def draw_mandelbrot(self, preview=False, overscan=False):
        """
        Request a render of the current view (non-blocking).

        `preview=True` renders a lower-resolution frame for interaction.
        `overscan=True` expands a pan preview beyond the viewport, so the
        result can be cached and cropped while the user drags without waiting
        for another calculation.  A normal call requests the full-resolution
        image once an interaction settles.

        The actual computation runs on a worker thread; the previous pending
        request is dropped so rapid input (zooming, panning, slider drags)
        never queues up a backlog of stale frames.
        """
        if getattr(self, "_closed", False):
            return
        current_range = float(self.x_max - self.x_min)
        if not (current_range > 0 and math.isfinite(current_range)):
            # Degenerate bounds (e.g. zoomed out past float limits): reset.
            self.x_min, self.x_max = -2.0, 1.0
            self.y_min, self.y_max = -1.5, 1.5
            current_range = 3.0
        self.zoom_level = self.base_range / current_range
        # Square pixels: re-derive the y range from the canvas aspect ratio.
        self._adjust_aspect()

        if self.auto_adjust:
            self.auto_iterations = self._auto_iterations_for(self.zoom_level)

            # Adjust slider range based on current zoom level. Only reconfigure
            # when it actually changes: reconfiguring on every draw could clamp
            # the user's in-progress slider value.
            scale_from = 2 - self.auto_iterations
            scale_to = int(
                200 * (10**1.2) + 100 * math.log(max(self.zoom_level, 1e-9) / 10)
            )
            if scale_to <= scale_from:
                scale_to = scale_from + 1
            cur_from = int(self.offset_scale.cget("from"))
            cur_to = int(self.offset_scale.cget("to"))
            if cur_from != scale_from or cur_to != scale_to:
                self.offset_scale.config(from_=scale_from, to=scale_to)

        # The offset slider allows negative offsets; clamp so the themes never
        # divide by zero or normalize against a negative iteration count.
        self.max_iterations = max(1, int(self.auto_iterations + self.iteration_offset))

        if preview and overscan:
            scale = PAN_PREVIEW_SCALE
            margin_x = max(1, int(self.width * PAN_PREVIEW_MARGIN))
            margin_y = max(1, int(self.height * PAN_PREVIEW_MARGIN))
            x_range = float(self.x_max - self.x_min)
            y_range = float(self.y_max - self.y_min)
            cache_x_min = self.x_min - margin_x / self.width * x_range
            cache_x_max = self.x_max + margin_x / self.width * x_range
            cache_y_min = self.y_min - margin_y / self.height * y_range
            cache_y_max = self.y_max + margin_y / self.height * y_range
            render_width = max(2, int((self.width + 2 * margin_x) * scale))
            render_height = max(2, int((self.height + 2 * margin_y) * scale))
            bounds = (cache_x_min, cache_x_max, cache_y_min, cache_y_max)
        elif preview:
            scale = self._zoom_preview_scale
            margin_x = margin_y = 0
            render_width = max(2, int(self.width * scale))
            render_height = max(2, int(self.height * scale))
            bounds = None
        else:
            scale = 1.0
            margin_x = margin_y = 0
            render_width, render_height = self.width, self.height
            bounds = None

        num_sections = max(1, min(self.num_cores, 32, render_width))
        base_width = max(1, render_width // num_sections)
        # The last section absorbs the remainder so the sections exactly cover
        # the render width (integer division used to leave a black strip).
        section_widths = [base_width] * num_sections
        section_widths[-1] += render_width - base_width * num_sections

        precision = "float32" if self.zoom_level <= GPU_FLOAT32_MAX_ZOOM else "float64"
        self._render_precision = precision
        params = {
            "fractal_type": self.fractal_type_var.get(),
            "width": render_width,
            "height": render_height,
            "x_min": float(bounds[0]) if bounds else float(self.x_min),
            "x_max": float(bounds[1]) if bounds else float(self.x_max),
            "y_min": float(bounds[2]) if bounds else float(self.y_min),
            "y_max": float(bounds[3]) if bounds else float(self.y_max),
            "max_iterations": self.max_iterations,
            "color_theme": self.color_theme,
            "julia_c": self.julia_c,
            "use_gpu": self.use_gpu,
            "precision": precision,
            "num_sections": num_sections,
            "section_widths": section_widths,
            "preview": preview,
            "overscan": overscan,
            "preview_scale": scale,
            "zoom_level": self.zoom_level,
            "cache_bounds": bounds,
        }

        self._render_seq += 1
        seq = self._render_seq
        try:
            self._render_queue.get_nowait()  # drop any pending request
        except queue.Empty:
            pass
        self._render_queue.put((seq, params))
        self.status_bar.config(text="Computing…")

    def _render_worker(self):
        """Worker thread loop: compute full/preview frames and cache tiles.

        Frame jobs are checked first so a settled full render is not delayed
        by background ring tiles.  Tile jobs are computed one at a time and
        returned immediately, so the visible cache can fill progressively.

        Deliberately makes NO Tk calls: tkinter objects are not thread-safe,
        so results are queued for the main thread's poller instead.
        """
        while True:
            job = None
            try:
                job = self._render_queue.get_nowait()
            except queue.Empty:
                try:
                    job = self._tile_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
            if job is None:  # shutdown sentinel
                break

            if isinstance(job, tuple) and job[0] == "tile":
                _, epoch, key, bounds, meta = job
                if epoch != self._tile_epoch:
                    continue  # zoom/render epoch changed; stale tile
                # A long drag can leave tiles from an older window queued.
                # Skip them without doing fractal work.
                needed = self._tile_index_range(
                    self.x_min, self.x_max, self.y_min, self.y_max, ring=TILE_RING
                )
                if key not in needed:
                    continue
                try:
                    tile_array = self._compute_tile(meta, bounds)
                except Exception as exc:  # never kill the worker thread
                    print(f"Tile error (epoch {epoch}, key {key}): {exc!r}")
                    self._result_queue.put(("tile_error", (epoch, key)))
                    continue
                self._result_queue.put(("tile", (epoch, key, tile_array, bounds)))
                continue

            seq, params = job
            start_time = time.time()
            try:
                full_array = self._compute_full(params)
            except Exception as exc:  # never kill the worker thread
                print(f"Render error (seq {seq}): {exc!r}")
                self._result_queue.put(("error", (seq, repr(exc), params)))
                continue
            elapsed = time.time() - start_time
            self._result_queue.put(("done", (seq, full_array, elapsed, params)))

    def _compute_tile(self, meta, bounds):
        """Compute one raw iteration tile (worker thread)."""
        # If a previous GPU call already fell back to CPU, do not try the GPU
        # again for every queued tile.
        use_gpu = meta["use_gpu"] and bool(MandelbrotCalculator._use_gpu)
        MandelbrotCalculator.set_use_gpu(use_gpu)
        fractal_type = meta["fractal_type"].lower()
        c = meta["julia_c"] if meta["fractal_type"] == "Julia" else None
        x_min, x_max, y_min, y_max = bounds
        tile_px = meta["tile_px"]
        return self.calculator.calculate_full(
            fractal_type,
            tile_px,
            tile_px,
            x_min,
            x_max,
            y_min,
            y_max,
            meta["max_iterations"],
            None,
            c=c,
            precision=meta["precision"],
        )

    def _poll_results(self):
        """Apply finished renders on the Tk main thread (recurring)."""
        tile_dirty = False
        try:
            while True:
                kind, payload = self._result_queue.get_nowait()
                if kind == "done":
                    seq, full_array, elapsed, params = payload
                    self._apply_render(seq, full_array, elapsed, params)
                elif kind == "tile":
                    epoch, key, tile_array, bounds = payload
                    if self._apply_tile(epoch, key, tile_array):
                        tile_dirty = True
                elif kind == "tile_error":
                    epoch, key = payload
                    if epoch == self._tile_epoch:
                        self._tile_pending.discard(key)
                else:
                    seq, message, params = payload
                    self._apply_render_error(seq, message)
        except queue.Empty:
            pass
        # Re-composite at most once per poll, not once per tile result.
        if tile_dirty and self._interacting and self._interaction_mode == "pan":
            self._display_tile_view()
        if not self._closed:
            try:
                # Poll often enough to display preview frames during an
                # animation; 50 ms limited the visible update rate to 20 Hz.
                self.master.after(POLL_MS, self._poll_results)
            except tk.TclError:
                pass  # root window was destroyed

    def _compute_full(self, params):
        """Compute raw iteration counts for one render request (worker thread)."""
        fractal_type = params["fractal_type"]
        width, height = params["width"], params["height"]
        x_min, x_max = params["x_min"], params["x_max"]
        y_min, y_max = params["y_min"], params["y_max"]
        max_iterations = params["max_iterations"]
        num_sections = params["num_sections"]
        section_widths = params["section_widths"]
        use_gpu = params["use_gpu"]
        precision = params.get("precision", "float64")

        MandelbrotCalculator.set_use_gpu(use_gpu)
        julia_c = params["julia_c"] if fractal_type == "Julia" else None

        if use_gpu:
            # GPU mode: one kernel for the whole image (no multiprocessing).
            # Raw iterations are returned; colour mapping happens on the main
            # thread so the result is also usable as the source of pan tiles.
            return self.calculator.calculate_full(
                fractal_type.lower(),
                width,
                height,
                x_min,
                x_max,
                y_min,
                y_max,
                max_iterations,
                None,
                c=julia_c,
                precision=precision,
            )

        # CPU mode: use multiprocessing with sections. Each task carries its
        # first column (offset), so sections may have different widths; the
        # last one absorbs the width remainder, so the sections exactly cover
        # the canvas (integer division used to leave a black strip).
        #
        # Workers return raw iteration counts, not RGB.  Colour mapping is a
        # single pass over the whole image afterwards.
        tasks = []
        offset = 0
        for i in range(num_sections):
            section_width = section_widths[i]
            task = (
                offset,
                section_width,
                width,
                height,
                x_min,
                x_max,
                y_min,
                y_max,
                max_iterations,
                None,
            )
            if fractal_type == "Julia":
                task = task + (julia_c,)
            tasks.append(task)
            offset += section_width

        if fractal_type == "Mandelbrot":
            section_arrays = self.pool.starmap(
                self.calculator.calculate_mandelbrot_section, tasks
            )
        elif fractal_type == "Julia":
            section_arrays = self.pool.starmap(
                self.calculator.calculate_julia_section, tasks
            )
        elif fractal_type == "Fatou":
            section_arrays = self.pool.starmap(
                self.calculator.calculate_fatou_section, tasks
            )
        else:
            raise ValueError(f"Unknown fractal type: {fractal_type!r}")

        return np.hstack(section_arrays)

    def _set_image(self, array, preview=False):
        """Build one PhotoImage from an RGB array and update the canvas item.

        Preview arrays are smaller than the canvas and are upscaled here;
        resizing is a display concern, deliberately separated from the
        fractal calculation in the worker thread.
        """
        image = Image.fromarray(array, mode="RGB")
        if image.size != (self.width, self.height):
            # Preview frames are upscaled at interaction rate; nearest-neighbour
            # keeps that cheap. Settled frames are already canvas-sized.
            resample = (
                Image.Resampling.NEAREST if preview else Image.Resampling.BILINEAR
            )
            image = image.resize((self.width, self.height), resample)
        self.photo = ImageTk.PhotoImage(image)
        # Keep exactly one canvas image item; update it in place instead of
        # stacking a new item (and a new Tcl image) on every redraw.
        if self._image_item is None:
            self._image_item = self.canvas.create_image(
                0, 0, anchor=tk.NW, image=self.photo
            )
        else:
            self.canvas.itemconfigure(self._image_item, image=self.photo)
        # A drag may have shifted the item to follow the cursor; every new
        # frame is drawn at the canvas origin so the contents and the view
        # bounds stay aligned.
        self.canvas.coords(self._image_item, 0, 0)

    def _cache_matches_current_params(self):
        """True if the pan cache was rendered for the current fractal state."""
        cache = self._pan_cache
        if cache is None:
            return False
        # The cache holds raw iterations; the theme is applied at display
        # time, so a theme change does not invalidate it.
        if cache["max_iterations"] != self.max_iterations:
            return False
        if cache["fractal_type"] != self.fractal_type_var.get():
            return False
        if cache["julia_c"] != self.julia_c:
            return False
        zoom = self.zoom_level
        return abs(cache["zoom_level"] - zoom) <= 1e-9 * max(1.0, abs(zoom))

    def _bounds_cover_current_view(self, bounds):
        """True if the world-space bounds contain the current view."""
        cx0, cx1, cy0, cy1 = bounds
        vx0, vx1, vy0, vy1 = self.x_min, self.x_max, self.y_min, self.y_max
        eps = 1e-12 * max(1.0, abs(cx0), abs(cx1), abs(cy0), abs(cy1))
        return not (
            vx0 < cx0 - eps or vx1 > cx1 + eps or vy0 < cy0 - eps or vy1 > cy1 + eps
        )

    def _cache_crop_for_current_view(self):
        """Return the pan-cache crop for the current view, or None."""
        cache = self._pan_cache
        if cache is None or not self._cache_matches_current_params():
            return None
        if not self._bounds_cover_current_view(cache["bounds"]):
            return None

        cx0, cx1, cy0, cy1 = cache["bounds"]
        vx0, vx1, vy0, vy1 = self.x_min, self.x_max, self.y_min, self.y_max
        array = cache["array"]
        cache_h, cache_w = array.shape[:2]
        fx0 = (vx0 - cx0) / (cx1 - cx0) * cache_w
        fx1 = (vx1 - cx0) / (cx1 - cx0) * cache_w
        fy0 = (vy0 - cy0) / (cy1 - cy0) * cache_h
        fy1 = (vy1 - cy0) / (cy1 - cy0) * cache_h
        j0 = max(0, min(cache_w - 1, int(math.floor(fx0))))
        j1 = max(j0 + 1, min(cache_w, int(math.ceil(fx1))))
        i0 = max(0, min(cache_h - 1, int(math.floor(fy0))))
        i1 = max(i0 + 1, min(cache_h, int(math.ceil(fy1))))
        crop = array[i0:i1, j0:j1]
        return crop if crop.size else None

    def _display_cached_pan(self):
        """Display the cached preview for the current view without computing.

        The cache stores raw iterations, so the current theme is applied at
        display time rather than baked into the cached pixels.
        """
        crop = self._cache_crop_for_current_view()
        if crop is None:
            return False
        rgb = self._colorize(crop, self.max_iterations, self.color_theme)
        self._set_image(rgb, preview=True)
        return True

    def _colorize(self, iterations, max_iterations, theme):
        """Apply a theme on the main thread (also used for cached tiles)."""
        if theme == self.CPU_CORES_THEME:
            width = iterations.shape[1]
            num_sections = max(1, min(self.num_cores, 32, width))
            base = width // num_sections
            widths = [base] * num_sections
            widths[-1] += width - base * num_sections
            pieces = []
            offset = 0
            for section_width in widths:
                piece = iterations[:, offset : offset + section_width]
                pieces.append(
                    apply_color_theme(
                        piece, max_iterations, random.choice(self._theme_choices)
                    )
                )
                offset += section_width
            return np.hstack(pieces)
        return apply_color_theme(iterations, max_iterations, theme)

    def _adapt_preview_scale(self, current, frame_ms, slow_streak, fast_streak):
        """Choose the next preview scale from the measured frame time.

        Lower resolution is used only when a frame misses the 60-fps budget.
        Raising the resolution requires several consecutive cheap frames so
        the controller does not oscillate between full and half resolution.
        """
        scales = PREVIEW_SCALES
        try:
            index = scales.index(current)
        except ValueError:
            index = 0

        if frame_ms > FRAME_BUDGET_MS:
            slow_streak += 1
            fast_streak = 0
            if slow_streak >= 1:
                index = min(len(scales) - 1, index + 1)
                slow_streak = 0
        elif frame_ms < FRAME_BUDGET_MS * FAST_FRAME_FRACTION:
            fast_streak += 1
            slow_streak = 0
            if fast_streak >= FAST_FRAMES_BEFORE_RAISE:
                index = max(0, index - 1)
                fast_streak = 0
        else:
            slow_streak = 0
            fast_streak = 0

        return scales[index], slow_streak, fast_streak

    def _drain_tile_queue(self):
        """Remove queued tile jobs that belong to an invalidated epoch."""
        while True:
            try:
                self._tile_queue.get_nowait()
            except queue.Empty:
                return

    def _invalidate_tile_cache(self):
        """Discard the tile cache and any queued tile jobs."""
        self._tile_epoch += 1
        self._tiles = {}
        self._tile_pending = set()
        self._tile_anchor_x = None
        self._tile_anchor_y = None
        self._tile_world = None
        self._tile_px = None
        self._tile_params = None
        self._drain_tile_queue()

    def _start_tile_cache(self, raw, params):
        """Seed a pan tile cache from a settled full-resolution render."""
        view_w, view_h = params["width"], params["height"]
        x_range = params["x_max"] - params["x_min"]
        if view_w <= 0 or view_h <= 0 or x_range <= 0:
            return

        self._tile_epoch += 1
        self._tiles = {}
        self._tile_pending = set()
        self._drain_tile_queue()
        self._pan_cache = None

        tile_px = min(TILE_PX, view_w, view_h)
        self._tile_px = tile_px
        self._tile_anchor_x = params["x_min"]
        self._tile_anchor_y = params["y_min"]
        self._tile_world = tile_px / view_w * x_range
        self._tile_params = {
            "fractal_type": params["fractal_type"],
            "max_iterations": params["max_iterations"],
            "julia_c": params["julia_c"],
            "zoom_level": params["zoom_level"],
            "precision": params["precision"],
            "view_width": view_w,
            "view_height": view_h,
            "tile_px": tile_px,
        }

        # Complete tiles are exact tile_px x tile_px slices of the full render.
        # Partial tiles at the right/bottom edge are requested from the worker.
        max_i = view_w // tile_px
        max_j = view_h // tile_px
        for j in range(max_j):
            row0 = j * tile_px
            row1 = row0 + tile_px
            for i in range(max_i):
                col0 = i * tile_px
                col1 = col0 + tile_px
                self._tiles[(i, j)] = raw[row0:row1, col0:col1].copy()

        # Precompute one ring around the viewport in the background.
        self._request_tiles_for_view()

    def _tile_cache_is_valid(self):
        """True if the tile grid matches the current fractal/zoom/view."""
        params = self._tile_params
        if params is None or self._tile_world is None or self._tile_px is None:
            return False
        if params["fractal_type"] != self.fractal_type_var.get():
            return False
        if params["max_iterations"] != self.max_iterations:
            return False
        if params["julia_c"] != self.julia_c:
            return False
        if params["view_width"] != self.width or params["view_height"] != self.height:
            return False
        if params["tile_px"] != self._tile_px:
            return False
        zoom = self.zoom_level
        return abs(params["zoom_level"] - zoom) <= 1e-9 * max(1.0, abs(zoom))

    def _tile_index_range(self, x0, x1, y0, y1, ring=0):
        """Return tile keys intersecting a world-space rectangle."""
        tw = self._tile_world
        if tw is None or tw <= 0:
            return set()
        ax, ay = self._tile_anchor_x, self._tile_anchor_y
        i0 = int(math.floor((x0 - ax) / tw))
        i1 = int(math.ceil((x1 - ax) / tw)) - 1
        j0 = int(math.floor((y0 - ay) / tw))
        j1 = int(math.ceil((y1 - ay) / tw)) - 1
        if i1 < i0:
            i1 = i0
        if j1 < j0:
            j1 = j0
        i0 -= ring
        i1 += ring
        j0 -= ring
        j1 += ring
        return {(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)}

    def _tile_bounds(self, key):
        """Full world bounds of one tile."""
        i, j = key
        x0 = self._tile_anchor_x + i * self._tile_world
        y0 = self._tile_anchor_y + j * self._tile_world
        return (x0, x0 + self._tile_world, y0, y0 + self._tile_world)

    def _request_tile(self, key):
        """Queue one missing tile for the worker."""
        if key in self._tiles or key in self._tile_pending:
            return
        self._tile_pending.add(key)
        meta = dict(self._tile_params)
        meta["use_gpu"] = self.use_gpu
        self._tile_queue.put(
            ("tile", self._tile_epoch, key, self._tile_bounds(key), meta)
        )

    def _request_tiles_for_view(self):
        """Queue visible missing tiles first, then the surrounding ring."""
        if self._tile_params is None:
            return
        visible = self._tile_index_range(self.x_min, self.x_max, self.y_min, self.y_max)
        ring = self._tile_index_range(
            self.x_min, self.x_max, self.y_min, self.y_max, ring=TILE_RING
        )
        # Keep only the current window; a long drag must not accumulate
        # every tile it has ever visited.
        keep = visible | ring
        for key in list(self._tiles):
            if key not in keep:
                del self._tiles[key]
        self._tile_pending &= keep

        cx = (self.x_min + self.x_max) / 2
        cy = (self.y_min + self.y_max) / 2
        tw = self._tile_world

        def distance(key):
            i, j = key
            tx = self._tile_anchor_x + (i + 0.5) * tw
            ty = self._tile_anchor_y + (j + 0.5) * tw
            return (tx - cx) ** 2 + (ty - cy) ** 2

        for key in sorted(visible, key=distance):
            self._request_tile(key)
        for key in sorted(ring - visible, key=distance):
            self._request_tile(key)

    def _tile_intersects_view(self, key):
        x0, x1, y0, y1 = self._tile_bounds(key)
        return not (
            x1 <= self.x_min or x0 >= self.x_max or y1 <= self.y_min or y0 >= self.y_max
        )

    def _apply_tile(self, epoch, key, tile_array):
        """Store a worker tile; return True if it touches the current view."""
        if epoch != self._tile_epoch:
            return False
        self._tile_pending.discard(key)
        if key in self._tiles:
            return False
        needed = self._tile_index_range(
            self.x_min, self.x_max, self.y_min, self.y_max, ring=TILE_RING
        )
        if key not in needed:
            return False
        self._tiles[key] = tile_array
        # A tile failure can also be the first place the GPU fallback is
        # observed; keep the checkbox in sync.
        if (
            self.gpu_var is not None
            and bool(MandelbrotCalculator._use_gpu) != self.use_gpu
        ):
            self.use_gpu = bool(MandelbrotCalculator._use_gpu)
            self.gpu_var.set(self.use_gpu)
        return self._tile_intersects_view(key)

    def _compose_tile_view(self):
        """Composite available tiles into a raw array covering the view."""
        if not self._tile_cache_is_valid():
            return None
        visible = self._tile_index_range(self.x_min, self.x_max, self.y_min, self.y_max)
        if not visible:
            return None

        tile_px = self._tile_px
        i0 = min(i for i, _ in visible)
        i1 = max(i for i, _ in visible)
        j0 = min(j for _, j in visible)
        j1 = max(j for _, j in visible)
        comp_w = (i1 - i0 + 1) * tile_px
        comp_h = (j1 - j0 + 1) * tile_px
        missing = self.max_iterations
        comp = np.full((comp_h, comp_w), missing, dtype=np.int32)

        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                tile = self._tiles.get((i, j))
                if tile is None or tile.shape != (tile_px, tile_px):
                    continue
                row = (j - j0) * tile_px
                col = (i - i0) * tile_px
                comp[row : row + tile_px, col : col + tile_px] = tile

        tw = self._tile_world
        comp_x0 = self._tile_anchor_x + i0 * tw
        comp_x1 = self._tile_anchor_x + (i1 + 1) * tw
        comp_y0 = self._tile_anchor_y + j0 * tw
        comp_y1 = self._tile_anchor_y + (j1 + 1) * tw
        fx0 = (self.x_min - comp_x0) / (comp_x1 - comp_x0) * comp_w
        fx1 = (self.x_max - comp_x0) / (comp_x1 - comp_x0) * comp_w
        fy0 = (self.y_min - comp_y0) / (comp_y1 - comp_y0) * comp_h
        fy1 = (self.y_max - comp_y0) / (comp_y1 - comp_y0) * comp_h
        col0 = max(0, min(comp_w - 1, int(math.floor(fx0))))
        col1 = max(col0 + 1, min(comp_w, int(math.ceil(fx1))))
        row0 = max(0, min(comp_h - 1, int(math.floor(fy0))))
        row1 = max(row0 + 1, min(comp_h, int(math.ceil(fy1))))
        crop = comp[row0:row1, col0:col1]
        return crop if crop.size else None

    def _display_tile_view(self):
        """Display the cached-tile composite for the current view."""
        if not self._tile_cache_is_valid():
            return False
        visible = self._tile_index_range(self.x_min, self.x_max, self.y_min, self.y_max)
        if not any(key in self._tiles for key in visible):
            return False
        start = time.perf_counter()
        crop = self._compose_tile_view()
        if crop is None:
            return False
        scale = self._tile_display_scale
        if scale < 1.0:
            step = max(1, int(round(1.0 / scale)))
            crop = crop[::step, ::step]
        rgb = self._colorize(crop, self.max_iterations, self.color_theme)
        self._set_image(rgb, preview=True)
        self._record_preview_frame()
        frame_ms = (time.perf_counter() - start) * 1000.0
        (
            self._tile_display_scale,
            self._tile_slow_streak,
            self._tile_fast_streak,
        ) = self._adapt_preview_scale(
            scale, frame_ms, self._tile_slow_streak, self._tile_fast_streak
        )
        return True

    def _record_preview_frame(self):
        """Track the interval between displayed preview frames.

        This measures the actual presentation rate, not just compute time:
        if the worker or main thread misses frames, the interval grows and
        the reported FPS falls. Title updates are throttled to ~10 Hz.
        """
        now = time.perf_counter()
        if self._last_preview_frame_at is not None:
            interval = now - self._last_preview_frame_at
            if interval > 0.0005:
                instant = 1.0 / interval
                if self._fps is None:
                    self._fps = instant
                else:
                    self._fps = 0.8 * self._fps + 0.2 * instant
        self._last_preview_frame_at = now
        if now - self._last_title_update >= 0.1:
            self._update_title()

    def _update_title(self, elapsed=None):
        """Refresh the title bar with render time and live preview FPS."""
        if self._closed:
            return
        if elapsed is not None:
            self._title_elapsed = elapsed
        fps = "--" if self._fps is None else f"{self._fps:.1f}"
        title_str = (
            f"{self.app_name} | {self._title_elapsed:.3f}s | FPS: {fps} | "
            + f"{self.zoom_level:.2f}x | Iter: {self.max_iterations} "
            + f"(Base: {self.auto_iterations} + Off: {self.iteration_offset}) "
            + f"| Theme: {self.color_theme}"
        )
        self.master.title(title_str)
        self._last_title_update = time.perf_counter()

    def _apply_render(self, seq, full_array, elapsed, params=None):
        """Display a finished render (main thread).

        Current results always win. Stale frames are still useful during a
        zoom animation, and a stale overscan pan frame can be reused as a
        cache if its expanded bounds still contain the current view. Other
        stale pan frames would jump backwards and are dropped.
        """
        params = params or {}
        if self._closed:
            return
        current = seq == self._render_seq
        preview = bool(params.get("preview"))
        overscan = bool(params.get("overscan"))
        if not current:
            zoom_preview = (
                self._interacting
                and self._interaction_mode == "zoom"
                and preview
                and not overscan
            )
            pan_overscan = (
                self._interacting
                and self._interaction_mode == "pan"
                and preview
                and overscan
                and self._bounds_cover_current_view(params["cache_bounds"])
            )
            if not (zoom_preview or pan_overscan):
                return
        if seq <= self._last_applied_seq:
            return
        self._last_applied_seq = seq

        display_start = time.perf_counter()
        if preview and overscan:
            self._pan_cache = {
                "array": full_array,
                "bounds": params["cache_bounds"],
                "max_iterations": params["max_iterations"],
                "fractal_type": params["fractal_type"],
                "julia_c": params["julia_c"],
                "zoom_level": params["zoom_level"],
            }
            crop = self._cache_crop_for_current_view()
            if crop is not None:
                rgb = self._colorize(
                    crop, params["max_iterations"], params["color_theme"]
                )
                self._set_image(rgb, preview=True)
            elif current:
                rgb = self._colorize(
                    full_array, params["max_iterations"], params["color_theme"]
                )
                self._set_image(rgb, preview=True)
            else:
                return
        else:
            rgb = self._colorize(
                full_array, params["max_iterations"], params["color_theme"]
            )
            self._set_image(rgb, preview=preview)

        display_ms = (time.perf_counter() - display_start) * 1000.0
        if preview:
            self._record_preview_frame()
        if preview and not overscan:
            # Zoom previews pay both worker compute time and main-thread
            # colour/display time. Drop resolution only when that frame
            # misses the target frame budget.
            frame_ms = elapsed * 1000.0 + display_ms
            (
                self._zoom_preview_scale,
                self._zoom_slow_streak,
                self._zoom_fast_streak,
            ) = self._adapt_preview_scale(
                self._zoom_preview_scale,
                frame_ms,
                self._zoom_slow_streak,
                self._zoom_fast_streak,
            )

        if not current:
            return

        # If the GPU failed at runtime the calculator fell back to CPU; sync
        # the viewer state and checkbox so the user can see it. The flag is a
        # plain bool (GIL-atomic reads/writes), so the only cost of the
        # worker/main-thread interleaving is a checkbox that lags by at most
        # one rendered frame.
        if (
            self.gpu_var is not None
            and bool(MandelbrotCalculator._use_gpu) != self.use_gpu
        ):
            self.use_gpu = bool(MandelbrotCalculator._use_gpu)
            self.gpu_var.set(self.use_gpu)

        # A settled full render covers exactly the viewport.  Seed the pan
        # tile cache from its raw iterations, then request the surrounding
        # ring in the background.
        if not preview:
            self._start_tile_cache(full_array, params)

        # Preview frames leave the status bar on "Computing…"; updating the
        # layout at interaction rate can resize the canvas and make the drag
        # jitter. Only the settled full-resolution frame updates it.
        if preview:
            return

        num_sections = max(1, min(self.num_cores, 32, self.width))
        backend = (
            f"GPU ({self._render_precision})"
            if self.use_gpu
            else f"Cores: {num_sections}"
        )
        self._update_title(elapsed)

        coord_str = (
            f"X_min: {self.x_min:.16g}, X_max: {self.x_max:.16g} | "
            f"Y_min: {self.y_min:.16g}, Y_max: {self.y_max:.16g}"
        )
        self.status_bar.config(text=f"{backend} | " + coord_str)

    def _apply_render_error(self, seq, message):
        """Show a render failure in the status bar (main thread)."""
        if self._closed or seq != self._render_seq:
            return
        self.status_bar.config(text=f"Render error: {message}")

    def on_close(self):
        """Window close handler: stop rendering, shut down the pool, destroy."""
        self._close_app()
        self.master.destroy()

    def _close_app(self):
        """Idempotent cleanup of the render thread and process pool."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        render_thread = getattr(self, "_render_thread", None)
        if render_thread is not None:
            try:
                self._render_queue.put(None)  # shutdown sentinel
                self._tile_queue.put(None)
                render_thread.join(timeout=5)
            except Exception:
                pass
        pool = getattr(self, "pool", None)
        if pool is not None:
            try:
                pool.close()
                # pool.join() blocks until an in-flight deep-zoom starmap
                # finishes; cap the wait so closing the window never hangs,
                # and terminate the workers if they are still computing.
                joiner = threading.Thread(target=pool.join, daemon=True)
                joiner.start()
                joiner.join(timeout=5)
                if joiner.is_alive():
                    pool.terminate()
                    joiner.join(timeout=5)
            except Exception:
                pass

    def save_bookmark(self):
        """
        Save the current view parameters as a bookmark to a JSON file.
        """
        bookmark = {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "iteration_offset": self.iteration_offset,
            "color_theme": self.color_theme,
            "auto_adjust": self.auto_adjust,
            "fractal_type": self.fractal_type_var.get(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "zoom_level": self.base_range / (self.x_max - self.x_min),
            "max_iterations": self.max_iterations,
            "auto_iterations": self.auto_iterations,
        }

        # Add julia_c to the bookmark if the fractal type is Julia
        if self.fractal_type_var.get() == "Julia":
            bookmark["julia_c"] = {"real": self.julia_c.real, "imag": self.julia_c.imag}

        file_path = BOOKMARKS_FILE
        bookmarks = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    bookmarks = json.load(f)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load bookmarks: {e}")
        bookmarks.append(bookmark)
        try:
            # Atomic write: dump to a temp file and rename, so a crash
            # mid-write can never leave a half-written bookmarks.json.
            tmp_path = file_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(bookmarks, f, indent=2)
            os.replace(tmp_path, file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save bookmark: {e}")

    def load_bookmark(self):
        """
        Open a dialog to load or delete saved bookmarks.
        """
        file_path = BOOKMARKS_FILE
        if not os.path.exists(file_path):
            messagebox.showinfo("No Bookmarks", "No bookmarks are saved yet.")
            return
        try:
            with open(file_path, "r") as f:
                bookmarks = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load bookmarks: {e}")
            return

        lb_window = tk.Toplevel(self.master)
        lb_window.title("Load Bookmark")
        lb_window.geometry("1024x768")

        # Vertical scrollbar
        v_scrollbar = tk.Scrollbar(lb_window, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Horizontal scrollbar
        h_scrollbar = tk.Scrollbar(lb_window, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Listbox with both scrollbars
        lb = tk.Listbox(
            lb_window,
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set,
            font=("Consolas", 10),
        )
        for index, bm in enumerate(bookmarks):
            # Old-format bookmarks may lack some fields; use safe defaults so
            # a stale file can never crash the dialog.
            summary = (
                f"{index+1}: {bm.get('timestamp', 'no date')} | "
                f"Fractal: {bm.get('fractal_type', 'Mandelbrot')} | "
                f"Theme: {bm.get('color_theme', 'Default')} | "
                f"Zoom: {float(bm.get('zoom_level', 1.0)):.2f}x | "
                f"Auto Adj: {bm.get('auto_adjust', True)} | "
                f"Iter_Offset: {bm.get('iteration_offset', 0)} | "
                f"Auto Iter: {bm.get('auto_iterations', 0)} | "
                f"X: {float(bm.get('x_min', -2.0)):.16g} "
                f"to {float(bm.get('x_max', 1.0)):.16g} | "
                f"Y: {float(bm.get('y_min', -1.5)):.16g} "
                f"to {float(bm.get('y_max', 1.5)):.16g}"
            )
            lb.insert(tk.END, summary)
        lb.pack(fill=tk.BOTH, expand=True)

        # Configure scrollbars
        v_scrollbar.config(command=lb.yview)
        h_scrollbar.config(command=lb.xview)

        def load_selected():
            """Load the selected bookmark."""
            selection = lb.curselection()
            if not selection:
                messagebox.showinfo("No Selection", "Please select a bookmark to load.")
                return
            index = selection[0]
            bm = bookmarks[index]
            # Safe defaults so old-format bookmarks cannot crash the loader.
            self.x_min = float(bm.get("x_min", -2.0))
            self.x_max = float(bm.get("x_max", 1.0))
            self.y_min = float(bm.get("y_min", -1.5))
            self.y_max = float(bm.get("y_max", 1.5))
            if not (self.x_min < self.x_max and self.y_min < self.y_max):
                messagebox.showerror(
                    "Invalid Bookmark", "This bookmark has invalid view bounds."
                )
                return
            self.iteration_offset = int(bm.get("iteration_offset", 0))
            current_max = self.offset_scale.cget("to")
            if int(current_max) < self.iteration_offset:
                self.offset_scale.config(to=max(1, self.iteration_offset * 2))
            self.offset_scale.set(self.iteration_offset)
            self.color_theme = bm.get("color_theme", "Default")
            self.color_theme_var.set(self.color_theme)
            self.auto_adjust = bm.get("auto_adjust", True)
            self.auto_adjust_var.set(self.auto_adjust)  # Update the checkbox state
            self.fractal_type_var.set(
                bm.get("fractal_type", "Mandelbrot")
            )  # Load the fractal type
            if self.fractal_type_var.get() == "Julia":
                julia_c_data = bm.get("julia_c", {"real": -0.7, "imag": 0.27015})
                self.julia_c = complex(
                    julia_c_data["real"], julia_c_data["imag"]
                )  # Reconstruct the complex number
                self.julia_c_real_var.set(
                    self.julia_c.real
                )  # Update the real part input field
                self.julia_c_imag_var.set(
                    self.julia_c.imag
                )  # Update the imaginary part input field
                self.julia_c_real_entry.grid()
                self.julia_c_imag_entry.grid()
                self.julia_c_real_label.grid()
                self.julia_c_real_slider.grid()
                self.julia_c_imag_label.grid()
                self.julia_c_imag_slider.grid()
            else:
                self.julia_c_real_entry.grid_remove()
                self.julia_c_imag_entry.grid_remove()
                self.julia_c_real_label.grid_remove()
                self.julia_c_real_slider.grid_remove()
                self.julia_c_imag_label.grid_remove()
                self.julia_c_imag_slider.grid_remove()
            self.draw_mandelbrot()
            lb_window.destroy()

        def delete_selected():
            """Delete the selected bookmark."""
            selection = lb.curselection()
            if not selection:
                messagebox.showinfo(
                    "No Selection", "Please select a bookmark to delete."
                )
                return
            index = selection[0]
            del bookmarks[index]
            try:
                tmp_path = file_path + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump(bookmarks, f, indent=2)
                os.replace(tmp_path, file_path)
                lb.delete(index)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete bookmark: {e}")

        lb.bind("<Double-Button-1>", lambda event: load_selected())

        button_frame = ttk.Frame(lb_window)
        button_frame.pack(pady=5)

        load_btn = ttk.Button(button_frame, text="Load Selected", command=load_selected)
        load_btn.grid(row=0, column=0, padx=5)

        delete_btn = ttk.Button(
            button_frame, text="Delete Selected", command=delete_selected
        )
        delete_btn.grid(row=0, column=1, padx=5)

    def reset_view(self):
        """
        Reset the view to the default parameters
        """
        self.fractal_type_changed(self.fractal_type_var.get())
        self.offset_scale.set(0)
        self.color_theme = "Default"
        self.color_theme_var.set(self.color_theme)
        self.auto_adjust = True
        self.auto_adjust_var.set(self.auto_adjust)
        self.julia_c = complex(-0.7, 0.27015)
        self.julia_c_real_var.set(self.julia_c.real)
        self.julia_c_imag_var.set(self.julia_c.imag)
        self.draw_mandelbrot()

    def zoom(self, event, zoom_factor):
        """
        Zoom in or out at the specified location.

        Args:
            event: Mouse or keyboard event containing position information
            zoom_factor (float): Factor to zoom by (>1 zooms out, <1 zooms in)
        """
        if self._typing_in_entry(event):
            return
        if self.zoom_level <= 0.26 and zoom_factor >= 2.0:
            self.status_bar.config(text="Maximum zoom-out reached")
            return
        if self.is_zooming:
            return

        self.is_zooming = True
        self._invalidate_tile_cache()
        try:
            width = self.x_max - self.x_min
            height = self.y_max - self.y_min
            # Tkinter reports event.type as a STRING ('2' = KeyPress,
            # '4'/'5' = ButtonPress); comparing with int 2 was always False,
            # so keyboard zoom silently fell into the pointer-anchored path.
            if str(event.type) == "2":
                center_x = (self.x_min + self.x_max) / 2
                center_y = (self.y_min + self.y_max) / 2
            else:
                center_x = self.x_min + (event.x / self.width) * width
                center_y = self.y_min + (event.y / self.height) * height

            # Mouse-anchored zoom: SCALE AROUND the cursor point so the
            # world point under the cursor lands exactly where it started.
            # (The old "center on the cursor point" formula only agrees with
            # this when the cursor IS the center, and overshoots the shift
            # toward the cursor by a factor of ~1/(1-factor) everywhere else.)
            self.x_min = center_x - (center_x - self.x_min) * zoom_factor
            self.x_max = center_x + (self.x_max - center_x) * zoom_factor
            self.y_min = center_y - (center_y - self.y_min) * zoom_factor
            self.y_max = center_y + (self.y_max - center_y) * zoom_factor
            self.draw_mandelbrot()
        finally:
            self.is_zooming = False  # Reset zooming flag

    def _typing_in_entry(self, event):
        """True if the key event originated in a text Entry widget.

        The pan/zoom/theme keys are bound on the root window and would
        otherwise also fire while the user types in the Julia C entries.
        """
        if event is None:
            return False
        try:
            return event.widget.winfo_class() in ("TEntry", "Entry")
        except Exception:
            return False

    def fluid_zoom(self, event, zoom_factor, frames=10, duration=0.09):
        """
        Perform a smooth zoom animation.

        Args:
            event: Mouse or keyboard event containing position information
            zoom_factor (float): Factor to zoom by (>1 zooms out, <1 zooms in)
            frames (int, optional): Number of animation frames. Defaults to 10.
            duration (float, optional): Total animation duration in seconds.
            Defaults to 0.09.
        """
        if self._typing_in_entry(event):
            return
        if self.zoom_level <= 0.26 and zoom_factor >= 2.0:
            self.status_bar.config(text="Maximum zoom-out reached")
            return
        if self.is_zooming:
            return

        self.is_zooming = True
        self._fps = None
        self._last_preview_frame_at = None
        self._interacting = True
        self._interaction_mode = "zoom"
        # A scale change invalidates both the pan cache and the tile grid:
        # their pixels correspond to the old zoom level.
        self._pan_cache = None
        self._invalidate_tile_cache()
        try:
            width = self.x_max - self.x_min
            height = self.y_max - self.y_min

            if str(event.type) == "2":  # Keyboard event (Tk type is a string)
                center_x = (self.x_min + self.x_max) / 2
                center_y = (self.y_min + self.y_max) / 2
            else:  # Mouse event
                center_x = self.x_min + (event.x / self.width) * width
                center_y = self.y_min + (event.y / self.height) * height

            # Mouse-anchored target (see zoom()): scale around the cursor.
            target_x_min = center_x - (center_x - self.x_min) * zoom_factor
            target_x_max = center_x + (self.x_max - center_x) * zoom_factor
            target_y_min = center_y - (center_y - self.y_min) * zoom_factor
            target_y_max = center_y + (self.y_max - center_y) * zoom_factor

            step_x_min = (target_x_min - self.x_min) / frames
            step_x_max = (target_x_max - self.x_max) / frames
            step_y_min = (target_y_min - self.y_min) / frames
            step_y_max = (target_y_max - self.y_max) / frames

            frame_duration = int(
                duration * 1000 / frames
            )  # Convert duration to milliseconds

            def animate_zoom_step(frame=0):
                """
                Recursive function to animate each step of the zoom.

                Args:
                    frame (int, optional): Current animation frame. Defaults to 0.
                """
                if frame >= frames:
                    self.is_zooming = False  # Reset zooming flag after animation
                    self._interacting = False
                    self._interaction_mode = None
                    # The animation used previews; settle at full resolution.
                    self.draw_mandelbrot()
                    return
                try:
                    self.x_min += step_x_min
                    self.x_max += step_x_max
                    self.y_min += step_y_min
                    self.y_max += step_y_max
                    self.draw_mandelbrot(preview=True, overscan=False)
                    self.master.after(frame_duration, animate_zoom_step, frame + 1)
                except Exception as exc:
                    # Runs in a later event-loop tick, so the outer try/except
                    # cannot catch this; always release the zoom lock.
                    print(f"Zoom animation error: {exc!r}")
                    self.is_zooming = False
                    self._interacting = False
                    self._interaction_mode = None

            animate_zoom_step()
        except Exception:
            self.is_zooming = False  # Ensure flag is reset in case of an error
            self._interacting = False
            self._interaction_mode = None

    _PAN_THRESHOLD_PX = 4

    def _pan_press(self, event):
        """Remember the press point; the release decides click vs pan."""
        self._pan_start = (event.x, event.y)
        self._pan_last = (event.x, event.y)
        self._panning = False
        self._fps = None
        self._last_preview_frame_at = None

    def _pan_motion(self, event):
        """Drag = grab-pan: move the view window opposite to the cursor
        delta, scaled to world units, so the grabbed point follows the
        cursor. Activates only past the threshold, so tiny hand jitters
        on a click still release as a zoom."""
        if self._pan_last is None:
            return
        lx, ly = self._pan_last
        if not self._panning:
            if (
                abs(event.x - self._pan_start[0]) + abs(event.y - self._pan_start[1])
                < self._PAN_THRESHOLD_PX
            ):
                return
            self._panning = True
            self._fps = None
            self._last_preview_frame_at = None
            self.canvas.config(cursor="fleur")
        dx = event.x - lx
        dy = event.y - ly
        self._pan_last = (event.x, event.y)
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min
        self.x_min -= dx / self.width * width
        self.x_max -= dx / self.width * width
        # Canvas row 0 (top) is y_min, so the same sign rule applies on both
        # axes: dragging the image down-left slides the view window the
        # opposite way in world coordinates.
        self.y_min -= dy / self.height * height
        self.y_max -= dy / self.height * height

        self._interacting = True
        self._interaction_mode = "pan"
        # Move the existing image immediately with the cursor. This is the
        # "grab" part of grab-pan: the fractal data need not be recalculated
        # before the canvas starts following the drag.
        if self._image_item is not None:
            self.canvas.move(self._image_item, dx, dy)

        if self._tile_cache_is_valid():
            # Full-resolution tiles are available for the visible area.
            # Request missing visible tiles first and the ring behind them;
            # the composite is refreshed when tile results arrive.
            self._request_tiles_for_view()
            self._display_tile_view()
            self._render_seq += 1
            try:
                self._render_queue.get_nowait()
            except queue.Empty:
                pass
        elif self._display_cached_pan():
            # Fallback for a missing/invalid tile cache: the overscan
            # preview already covers this view.
            self._render_seq += 1
            try:
                self._render_queue.get_nowait()
            except queue.Empty:
                pass
        else:
            # Cache absent, invalid, or too small for this view: request a
            # new overscan cache. A stale result may still cover the current
            # view (the margin is one viewport), in which case it is reused.
            self._pan_cache = None
            self.draw_mandelbrot(preview=True, overscan=True)

    def _pan_release(self, event):
        """Finish a pan, or perform the plain click zoom when no drag
        happened."""
        panned = self._panning
        self._panning = False
        self._pan_start = None
        self._pan_last = None
        if panned:
            self.canvas.config(cursor="")
            self._interacting = False
            self._interaction_mode = None
            # The drag used a preview; settle at full resolution now.
            self.draw_mandelbrot()
            return
        if getattr(self, "_closed", False):
            return
        self.fluid_zoom(event, 0.125)

    def move_down(self, event):
        """
        Move the view down by 10% of the current height.

        Args:
            event: Keyboard event
        """
        if self._typing_in_entry(event):
            return
        # Delta must be computed BEFORE moving the first bound; recomputing
        # (y_max - y_min) after the update shrank the viewport 1% per pan.
        dy = 0.1 * (self.y_max - self.y_min)
        self.y_min += dy
        self.y_max += dy
        self.draw_mandelbrot()

    def move_up(self, event):
        """
        Move the view up by 10% of the current height.

        Args:
            event: Keyboard event
        """
        if self._typing_in_entry(event):
            return
        dy = 0.1 * (self.y_max - self.y_min)
        self.y_min -= dy
        self.y_max -= dy
        self.draw_mandelbrot()

    def move_left(self, event):
        """
        Move the view left by 10% of the current width.

        Args:
            event: Keyboard event
        """
        if self._typing_in_entry(event):
            return
        # Delta computed once, before any bound is moved (see move_down).
        dx = 0.1 * (self.x_max - self.x_min)
        self.x_min -= dx
        self.x_max -= dx
        self.draw_mandelbrot()

    def move_right(self, event):
        """
        Move the view right by 10% of the current width.

        Args:
            event: Keyboard event
        """
        if self._typing_in_entry(event):
            return
        dx = 0.1 * (self.x_max - self.x_min)
        self.x_min += dx
        self.x_max += dx
        self.draw_mandelbrot()

    def toggle_auto_adjust(self):
        """
        Toggle automatic iteration adjustment based on zoom level.

        When enabled, the number of iterations is automatically adjusted
        based on zoom level. When disabled, the iteration slider range is
        expanded to allow manual control.
        """
        self.auto_adjust = self.auto_adjust_var.get()
        if not self.auto_adjust:
            self.offset_scale.config(from_=(2 + self.auto_iterations * -1))
            self.offset_scale.config(to=16384)
        self.draw_mandelbrot()

    def advance_theme(self, event=None):
        """
        Cycle to the next color theme in the list.

        Args:
            event: Event object, optional
        """
        if self._typing_in_entry(event):
            return  # let Tab move focus normally while editing an entry
        current_theme = self.color_theme_var.get()
        themes = self.themes
        current_index = themes.index(current_theme)
        next_index = (current_index + 1) % len(
            themes
        )  # Wrap around to the start if at the end
        next_theme = themes[next_index]
        self.color_theme_var.set(next_theme)
        self.color_theme = next_theme
        self.draw_mandelbrot()  # Redraw with the new theme

    def fractal_type_changed(self, value):
        """
        Update the fractal type and reset view parameters.

        Args:
            value (str): Selected fractal type
        """
        # Keep the dropdown variable in sync even when this method is called
        # programmatically (e.g. from reset_view), not just from the menu.
        self.fractal_type_var.set(value)
        if value == "Mandelbrot":
            self.x_min, self.x_max = -2.0, 1.0
            self.y_min, self.y_max = -1.5, 1.5
            # Hide Julia C input fields and the new slider
            self.julia_c_real_entry.grid_remove()
            self.julia_c_imag_entry.grid_remove()
            self.julia_c_real_label.grid_remove()
            self.julia_c_real_slider.grid_remove()
            self.julia_c_imag_label.grid_remove()
            self.julia_c_imag_slider.grid_remove()
            self.e_exp_i_n_label.grid_remove()
            self.e_exp_i_n_slider.grid_remove()
        elif value == "Julia":
            self.x_min, self.x_max = -1.5, 1.5
            self.y_min, self.y_max = -1.5, 1.5
            self.julia_c = complex(-0.7, 0.27015)  # Example Julia constant
            # Show Julia C input fields and the new slider
            self.julia_c_real_entry.grid()
            self.julia_c_imag_entry.grid()
            self.julia_c_real_label.grid()
            self.julia_c_real_slider.grid()
            self.julia_c_imag_label.grid()
            self.julia_c_imag_slider.grid()
            self.e_exp_i_n_label.grid()
            self.e_exp_i_n_slider.grid()
        elif value == "Fatou":
            self.x_min, self.x_max = -2.0, 2.0
            self.y_min, self.y_max = -2.0, 2.0
            # Hide Julia C input fields and the new slider
            self.julia_c_real_entry.grid_remove()
            self.julia_c_imag_entry.grid_remove()
            self.julia_c_real_label.grid_remove()
            self.julia_c_real_slider.grid_remove()
            self.julia_c_imag_label.grid_remove()
            self.julia_c_imag_slider.grid_remove()
            self.e_exp_i_n_label.grid_remove()
            self.e_exp_i_n_slider.grid_remove()
        self.draw_mandelbrot()

    def _parse_julia_c(self):
        """Parse the Julia C entry fields; return None (with a status hint)
        if either field does not contain a number."""
        try:
            real_part = float(self.julia_c_real_var.get())
            imag_part = float(self.julia_c_imag_var.get())
        except (TypeError, ValueError):
            self.status_bar.config(text="Invalid Julia C value – please enter numbers")
            return None
        return complex(real_part, imag_part)

    def update_julia_c(self, event=None):
        """
        Update the Julia set constant based on input fields.

        Args:
            event: Event object, optional
        """
        c = self._parse_julia_c()
        if c is None:
            return
        self.julia_c = c
        self.draw_mandelbrot()

    def update_julia_c_slider(self):
        """
        Update the Julia set constant based on slider values.
        """
        c = self._parse_julia_c()
        if c is None:
            return
        self.julia_c = c
        self.draw_mandelbrot()

    def update_e_exp_i_n_slider(self):
        """
        Update the Julia set constant (c) based on the "e exp i n" slider value.
        """
        slider_value = self.e_exp_i_n_var.get()
        self.julia_c = 0.7885 * complex(
            math.cos(slider_value), math.sin(slider_value)
        )  # 0.7885 * e^(i * slider_value)
        self.julia_c_real_var.set(self.julia_c.real)  # Update the real part input field
        self.julia_c_imag_var.set(
            self.julia_c.imag
        )  # Update the imaginary part input field
        self.draw_mandelbrot()

    def __del__(self):
        """
        Clean up resources when the object is deleted.

        Ensures the render thread and multiprocessing pool are closed.
        Never raises (destructor context) and tolerates partially
        initialized instances.
        """
        try:
            self._close_app()
        except Exception:
            pass


if __name__ == "__main__":
    """
    Main entry point for the application.

    Initializes the Tkinter root window and creates the Mandelbrot viewer.
    """
    multiprocessing.freeze_support()
    root = tk.Tk()
    # (The previous "320x240" geometry was silently overridden by minsize.)
    root.geometry("800x680")
    root.minsize(800, 680)
    viewer = MandelbrotViewer(root)
    root.mainloop()
