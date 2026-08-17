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

from fractals import MandelbrotCalculator, GPU_AVAILABLE
from themes import apply_color_theme

# Bookmarks live next to the script (not in whatever CWD the app was
# launched from), so they are found consistently.
BOOKMARKS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "bookmarks.json"
)


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
        self.app_name = "Fractal Frontier V1.1"
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

        self.canvas.bind("<Button-1>", lambda event: self.fluid_zoom(event, 0.125))
        self.canvas.bind("<Button-2>", self.advance_theme)
        self.canvas.bind("<Button-3>", lambda event: self.fluid_zoom(event, 8))

        system = platform.system()
        if system == "Linux":
            self.canvas.bind("<Button-4>", lambda event: self.zoom(event, 0.75))
            self.canvas.bind("<Button-5>", lambda event: self.zoom(event, 1.5))
        elif system == "Windows":
            self.canvas.bind("<MouseWheel>", self.on_windows_mousewheel)
        elif system == "Darwin":  # macOS
            self.canvas.bind("<MouseWheel>", self.on_macos_mousewheel)

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
        self.pool = multiprocessing.Pool(processes=self.num_cores)
        self.calculator = MandelbrotCalculator()  # Reintroduce the calculator
        self.resize_job = None

        # Rendering happens on a worker thread so the Tk main loop (and thus
        # the UI) never freezes while fractals are being computed. Only the
        # most recent request is ever rendered: pending requests are dropped.
        # The worker never touches Tk directly (Tk is not thread-safe); it
        # hands finished results back through a plain queue that the main
        # thread polls periodically.
        self._render_queue = queue.Queue()
        self._result_queue = queue.Queue()
        self._render_seq = 0
        self._closed = False
        self._image_item = None
        self._render_thread = threading.Thread(
            target=self._render_worker, daemon=True
        )
        self._render_thread.start()
        self._poll_results()

        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        self.draw_mandelbrot()

    def on_windows_mousewheel(self, event):
        """Handle mouse wheel events on Windows."""
        # In Windows,
        # event.delta is positive when scrolling up and negative when scrolling down
        if event.delta > 0:
            self.zoom(event, 0.75)  # Zoom in
        else:
            self.zoom(event, 1.5)  # Zoom out

    def on_macos_mousewheel(self, event):
        """Handle mouse wheel events on macOS."""
        # In macOS,
        # event.delta works similarly to Windows but may have different scaling
        if event.delta > 0:
            self.zoom(event, 0.75)  # Zoom in
        else:
            self.zoom(event, 1.5)  # Zoom out

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
        """Toggle GPU acceleration on/off, ensuring CPU is used when GPU is off and vice versa."""
        self.use_gpu = self.gpu_var.get()
        if self.use_gpu:
            # GPU enabled — ensure calculator uses GPU
            MandelbrotCalculator.set_use_gpu(True)
            self.status_bar.config(text="Switched to GPU backend")
        else:
            # GPU disabled — explicitly force CPU mode on the calculator
            MandelbrotCalculator.set_use_gpu(False)
            self.status_bar.config(text=f"Switched to CPU backend ({self.num_cores} cores)")
        self.draw_mandelbrot()

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

    def draw_mandelbrot(self):
        """
        Request a render of the current view (non-blocking).

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
            # Use a piecewise function for better balance between performance
            # and quality
            # At low zoom levels, use a shallow curve for good performance
            # At high zoom levels, use a steeper curve for better detail
            if self.zoom_level < 1.0:
                # Low zoom: shallow curve for good performance
                self.auto_iterations = int(50 * (self.zoom_level**0.15))
            elif self.zoom_level < 10.0:
                # Medium zoom: moderate curve for balanced performance and quality
                self.auto_iterations = int(100 * (self.zoom_level**0.25))
            else:
                # High zoom: steep curve for maximum detail
                self.auto_iterations = int(200 * (self.zoom_level**0.4))

            # Add a reasonable ceiling to prevent excessive computation
            self.auto_iterations = min(self.auto_iterations, 500)

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

        num_sections = max(1, min(self.num_cores, 32, self.width))
        base_width = max(1, self.width // num_sections)
        # The last section absorbs the remainder so the sections exactly cover
        # the canvas width (integer division used to leave a black strip).
        section_widths = [base_width] * num_sections
        section_widths[-1] += self.width - base_width * num_sections

        params = {
            "fractal_type": self.fractal_type_var.get(),
            "width": self.width,
            "height": self.height,
            "x_min": float(self.x_min),
            "x_max": float(self.x_max),
            "y_min": float(self.y_min),
            "y_max": float(self.y_max),
            "max_iterations": self.max_iterations,
            "color_theme": self.color_theme,
            "julia_c": self.julia_c,
            "use_gpu": self.use_gpu,
            "num_sections": num_sections,
            "section_widths": section_widths,
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
        """Worker thread loop: compute renders and hand results back.

        Deliberately makes NO Tk calls: tkinter objects are not thread-safe,
        so results are queued for the main thread's poller instead.
        """
        while True:
            job = self._render_queue.get()
            if job is None:  # shutdown sentinel
                break
            seq, params = job
            start_time = time.time()
            try:
                full_array = self._compute_full(params)
            except Exception as exc:  # never kill the worker thread
                print(f"Render error (seq {seq}): {exc!r}")
                self._result_queue.put(("error", (seq, repr(exc))))
                continue
            elapsed = time.time() - start_time
            self._result_queue.put(("done", (seq, full_array, elapsed)))

    def _poll_results(self):
        """Apply finished renders on the Tk main thread (recurring)."""
        try:
            while True:
                kind, payload = self._result_queue.get_nowait()
                if kind == "done":
                    seq, full_array, elapsed = payload
                    self._apply_render(seq, full_array, elapsed)
                else:
                    seq, message = payload
                    self._apply_render_error(seq, message)
        except queue.Empty:
            pass
        if not self._closed:
            try:
                self.master.after(50, self._poll_results)
            except tk.TclError:
                pass  # root window was destroyed

    def _compute_full(self, params):
        """Compute the full RGB image for one render request (worker thread)."""
        fractal_type = params["fractal_type"]
        width, height = params["width"], params["height"]
        x_min, x_max = params["x_min"], params["x_max"]
        y_min, y_max = params["y_min"], params["y_max"]
        max_iterations = params["max_iterations"]
        theme = params["color_theme"]
        num_sections = params["num_sections"]
        section_widths = params["section_widths"]
        use_gpu = params["use_gpu"]

        MandelbrotCalculator.set_use_gpu(use_gpu)
        is_cpu_cores_theme = theme == self.CPU_CORES_THEME
        julia_c = params["julia_c"] if fractal_type == "Julia" else None

        if use_gpu:
            # GPU mode: one kernel for the whole image (no multiprocessing).
            if not is_cpu_cores_theme:
                return self.calculator.calculate_full(
                    fractal_type.lower(), width, height,
                    x_min, x_max, y_min, y_max,
                    max_iterations, theme, c=julia_c,
                )
            # "CPU Cores" theme on GPU: compute iterations once, then apply a
            # random theme per vertical section (mirrors the CPU behaviour).
            iterations = self.calculator.calculate_full(
                fractal_type.lower(), width, height,
                x_min, x_max, y_min, y_max,
                max_iterations, None, c=julia_c,
            )
            pieces = []
            offset = 0
            for i in range(num_sections):
                piece = iterations[:, offset:offset + section_widths[i]]
                pieces.append(
                    apply_color_theme(piece, max_iterations, random.choice(self.themes))
                )
                offset += section_widths[i]
            return np.hstack(pieces)

        # CPU mode: use multiprocessing with sections. Each task carries its
        # first column (offset), so sections may have different widths; the
        # last one absorbs the width remainder, so the sections exactly cover
        # the canvas (integer division used to leave a black strip).
        tasks = []
        offset = 0
        for i in range(num_sections):
            section_width = section_widths[i]
            task_theme = random.choice(self.themes) if is_cpu_cores_theme else theme
            task = (
                offset, section_width, width, height,
                x_min, x_max, y_min, y_max, max_iterations, task_theme,
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

    def _apply_render(self, seq, full_array, elapsed):
        """Display a finished render (main thread). Stale results are dropped."""
        if self._closed or seq != self._render_seq:
            return
        image = Image.fromarray(full_array, mode="RGB")
        self.photo = ImageTk.PhotoImage(image)
        # Keep exactly one canvas image item; update it in place instead of
        # stacking a new item (and a new Tcl image) on every redraw.
        if self._image_item is None:
            self._image_item = self.canvas.create_image(
                0, 0, anchor=tk.NW, image=self.photo
            )
        else:
            self.canvas.itemconfigure(self._image_item, image=self.photo)

        # If the GPU failed at runtime the calculator fell back to CPU; sync
        # the viewer state and checkbox so the user can see it.
        if (
            self.gpu_var is not None
            and bool(MandelbrotCalculator._use_gpu) != self.use_gpu
        ):
            self.use_gpu = bool(MandelbrotCalculator._use_gpu)
            self.gpu_var.set(self.use_gpu)

        num_sections = max(1, min(self.num_cores, 32, self.width))
        backend = "GPU" if self.use_gpu else f"Cores: {num_sections}"
        title_str = (
            f"{self.app_name} | {elapsed:.2f}s | "
            + f"{self.zoom_level:.2f}x | Iter: {self.max_iterations} "
            + f"(Base: {self.auto_iterations} + Off: {self.iteration_offset}) "
            + f"| Theme: {self.color_theme}"
        )
        self.master.title(title_str)

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
                render_thread.join(timeout=5)
            except Exception:
                pass
        pool = getattr(self, "pool", None)
        if pool is not None:
            try:
                pool.close()
                pool.join()
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
            with open(file_path, "w") as f:
                json.dump(bookmarks, f, indent=2)
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
                f"X: {float(bm.get('x_min', -2.0)):.16g} to {float(bm.get('x_max', 1.0)):.16g} | "
                f"Y: {float(bm.get('y_min', -1.5)):.16g} to {float(bm.get('y_max', 1.5)):.16g}"
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
                with open(file_path, "w") as f:
                    json.dump(bookmarks, f, indent=2)
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
            return
        if self.is_zooming:
            return

        self.is_zooming = True
        try:
            width = self.x_max - self.x_min
            height = self.y_max - self.y_min
            if event.type == 2:
                center_x = (self.x_min + self.x_max) / 2
                center_y = (self.y_min + self.y_max) / 2
            else:
                center_x = self.x_min + (event.x / self.width) * width
                center_y = self.y_min + (event.y / self.height) * height

            new_width = width * zoom_factor
            new_height = height * zoom_factor

            self.x_min = center_x - new_width / 2
            self.x_max = center_x + new_width / 2
            self.y_min = center_y - new_height / 2
            self.y_max = center_y + new_height / 2
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
            return
        if self.is_zooming:
            return

        self.is_zooming = True
        try:
            width = self.x_max - self.x_min
            height = self.y_max - self.y_min

            if event.type == 2:  # Keyboard event
                center_x = (self.x_min + self.x_max) / 2
                center_y = (self.y_min + self.y_max) / 2
            else:  # Mouse event
                center_x = self.x_min + (event.x / self.width) * width
                center_y = self.y_min + (event.y / self.height) * height

            new_width = width * zoom_factor
            new_height = height * zoom_factor

            target_x_min = center_x - new_width / 2
            target_x_max = center_x + new_width / 2
            target_y_min = center_y - new_height / 2
            target_y_max = center_y + new_height / 2

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
                    return
                try:
                    self.x_min += step_x_min
                    self.x_max += step_x_max
                    self.y_min += step_y_min
                    self.y_max += step_y_max
                    self.draw_mandelbrot()
                    self.master.after(frame_duration, animate_zoom_step, frame + 1)
                except Exception as exc:
                    # Runs in a later event-loop tick, so the outer try/except
                    # cannot catch this; always release the zoom lock.
                    print(f"Zoom animation error: {exc!r}")
                    self.is_zooming = False

            animate_zoom_step()
        except Exception:
            self.is_zooming = False  # Ensure flag is reset in case of an error

    def move_down(self, event):
        """
        Move the view down by 10% of the current height.

        Args:
            event: Keyboard event
        """
        if self._typing_in_entry(event):
            return
        self.y_min += 0.1 * (self.y_max - self.y_min)
        self.y_max += 0.1 * (self.y_max - self.y_min)
        self.draw_mandelbrot()

    def move_up(self, event):
        """
        Move the view up by 10% of the current height.

        Args:
            event: Keyboard event
        """
        if self._typing_in_entry(event):
            return
        self.y_min -= 0.1 * (self.y_max - self.y_min)
        self.y_max -= 0.1 * (self.y_max - self.y_min)
        self.draw_mandelbrot()

    def move_left(self, event):
        """
        Move the view left by 10% of the current width.

        Args:
            event: Keyboard event
        """
        if self._typing_in_entry(event):
            return
        self.x_min -= 0.1 * (self.x_max - self.x_min)
        self.x_max -= 0.1 * (self.x_max - self.x_min)
        self.draw_mandelbrot()

    def move_right(self, event):
        """
        Move the view right by 10% of the current width.

        Args:
            event: Keyboard event
        """
        if self._typing_in_entry(event):
            return
        self.x_min += 0.1 * (self.x_max - self.x_min)
        self.x_max += 0.1 * (self.x_max - self.x_min)
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
            self.status_bar.config(
                text="Invalid Julia C value – please enter numbers"
            )
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
