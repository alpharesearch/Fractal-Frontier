"""
Fractal Frontier - An interactive Mandelbrot set explorer with multiprocessing support.

This application provides a GUI for exploring the Mandelbrot set fractal with various
color themes, zoom capabilities, and bookmark functionality.
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import numpy as np
import multiprocessing
import time
import math
import json
import os
from numba import njit, prange
import colorsys
import platform
import random

# --------------------------------------------------------------------------
# COLOR THEME MAPPING (Vectorized)
# --------------------------------------------------------------------------
def apply_color_theme(iterations, max_iterations, theme="Default"):
    """
    Apply a color theme to the iteration counts of the Mandelbrot set.
    
    Args:
        iterations (numpy.ndarray): 2D array of iteration counts
        max_iterations (int): Maximum number of iterations used in calculation
        theme (str, optional): Color theme name. Defaults to "Default"
        
    Returns:
        numpy.ndarray: RGB color array with shape (height, width, 3)
    """
    def default_theme(iterations, max_iterations, colors):
        """Default color theme with blue-green gradient."""
        mask = iterations == max_iterations
        colors[mask] = (0, 0, 0)
        norm_vals = (iterations.astype(np.float32) * 255.0 / max_iterations).astype(np.uint8)
        non_mask = ~mask
        colors[non_mask, 0] = norm_vals[non_mask]
        colors[non_mask, 1] = 255 - norm_vals[non_mask]
        colors[non_mask, 2] = 100

    def grayscale_theme(iterations, max_iterations, colors):
        """Simple grayscale theme."""
        norm_vals = (iterations.astype(np.float32) * 255.0 / max_iterations).astype(np.uint8)
        colors[..., 0] = norm_vals
        colors[..., 1] = norm_vals
        colors[..., 2] = norm_vals
        mask = iterations == max_iterations
        colors[mask] = (0, 0, 0)

    def blue_theme(iterations, max_iterations, colors):
        """Blue-dominant color theme."""
        norm_vals = (iterations.astype(np.float32) * 255.0 / max_iterations).astype(np.uint8)
        colors[..., 0] = 255 - norm_vals  
        colors[..., 1] = 255 - norm_vals  
        colors[..., 2] = norm_vals  
        mask = iterations == max_iterations
        colors[mask] = (0, 0, 0)

    def fire_theme(iterations, max_iterations, colors):
        """Fire-like color theme with red and orange tones."""
        norm_vals = (iterations.astype(np.float32) * 255.0 / max_iterations).astype(np.uint8)
        colors[..., 0] = np.clip(norm_vals * 2, 0, 255)
        colors[..., 1] = np.clip(norm_vals + 50, 0, 255)
        colors[..., 2] = np.clip(norm_vals - 100, 0, 255)
        mask = iterations == max_iterations
        colors[mask] = (0, 0, 0)

    def rainbow_theme(iterations, max_iterations, colors):
        """Rainbow color theme using HSV color space with 256 colors."""
        palette = np.empty((256, 3), dtype=np.uint8)
        for i in range(256):
            h = i / 256.0
            r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            palette[i] = (int(r * 255), int(g * 255), int(b * 255))
        mod_vals = np.mod(iterations, 256)
        colors[:] = palette[mod_vals]
        mask = iterations == max_iterations
        colors[mask] = (0, 0, 0)

    def rainbow2_theme(iterations, max_iterations, colors):
        """Enhanced rainbow theme with higher color values."""
        palette = np.empty((256, 3), dtype=np.uint8)
        for i in range(256):
            h = i / 256.0
            r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            palette[i] = (int(r * 1024), int(g * 1024), int(b * 1024))
        mod_vals = np.mod(iterations, 256)
        colors[:] = palette[mod_vals]
        mask = iterations == max_iterations
        colors[mask] = (0, 0, 0)

    def rainbow3_theme(iterations, max_iterations, colors):
        """Rainbow theme with 1024 colors for smoother gradients."""
        palette = np.empty((1024, 3), dtype=np.uint8)
        for i in range(1024):
            h = i / 1024.0
            r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            palette[i] = (int(r * 1024), int(g * 1024), int(b * 1024))
        mod_vals = np.mod(iterations, 1024)
        colors[:] = palette[mod_vals]
        mask = iterations == max_iterations
        colors[mask] = (0, 0, 0)
    
    def rainbow4_theme(iterations, max_iterations, colors):
        """High-resolution rainbow theme with 8192 colors."""
        palette = np.empty((8192, 3), dtype=np.uint8)
        for i in range(8192):
            h = i / 8192.0
            r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            palette[i] = (int(r * 8192), int(g * 8192), int(b * 8192))
        mod_vals = np.mod(iterations, 8192)
        colors[:] = palette[mod_vals]
        mask = iterations == max_iterations
        colors[mask] = (0, 0, 0)
    
    theme_functions = {
        "Default": default_theme,
        "Grayscale": grayscale_theme,
        "Blue": blue_theme,
        "Fire": fire_theme,
        "Rainbow": rainbow_theme,
        "Rainbow2": rainbow2_theme,
        "Rainbow3": rainbow3_theme,
        "Rainbow4": rainbow4_theme
    }

    height, width = iterations.shape
    colors = np.empty((height, width, 3), dtype=np.uint8)
    theme_function = theme_functions.get(theme, default_theme)
    theme_function(iterations, max_iterations, colors)
    return colors

# --------------------------------------------------------------------------
# JIT-COMPILED MANDELBROT ITERATION CALCULATOR
# --------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def mandelbrot_section_jit(section_index, section_width, width, height,
                           x_min, x_max, y_min, y_max, max_iterations):
    """
    Calculate Mandelbrot set iterations for a section of the image using Numba JIT compilation.
    
    Args:
        section_index (int): Index of the section to calculate
        section_width (int): Width of the section in pixels
        width (int): Total width of the image in pixels
        height (int): Total height of the image in pixels
        x_min (float): Minimum x-coordinate in the complex plane
        x_max (float): Maximum x-coordinate in the complex plane
        y_min (float): Minimum y-coordinate in the complex plane
        y_max (float): Maximum y-coordinate in the complex plane
        max_iterations (int): Maximum number of iterations to perform
        
    Returns:
        numpy.ndarray: 2D array of iteration counts for the section
    """
    section_counts = np.empty((height, section_width), dtype=np.int32)
    for i in prange(height):
        for j in range(section_width):
            global_x = section_index * section_width + j
            x = x_min + global_x * (x_max - x_min) / width
            y = y_min + i * (y_max - y_min) / height
            c = complex(x, y)
            z = 0j
            count = 0
            while (z.real * z.real + z.imag * z.imag < 4.0) and (count < max_iterations):
                z = z * z + c
                count += 1
            section_counts[i, j] = count
    return section_counts

@njit(parallel=True, cache=True)
def julia_section_jit(section_index, section_width, width, height,
                      x_min, x_max, y_min, y_max, max_iterations, c):
    """
    Calculate Julia set iterations for a section of the image using Numba JIT compilation.
    
    Args:
        section_index (int): Index of the section to calculate
        section_width (int): Width of the section in pixels
        width (int): Total width of the image in pixels
        height (int): Total height of the image in pixels
        x_min (float): Minimum x-coordinate in the complex plane
        x_max (float): Maximum x-coordinate in the complex plane
        y_min (float): Minimum y-coordinate in the complex plane
        y_max (float): Maximum y-coordinate in the complex plane
        max_iterations (int): Maximum number of iterations to perform
        c (complex): Constant for the Julia set
        
    Returns:
        numpy.ndarray: 2D array of iteration counts for the section
    """
    section_counts = np.empty((height, section_width), dtype=np.int32)
    for i in prange(height):
        for j in range(section_width):
            global_x = section_index * section_width + j
            x = x_min + global_x * (x_max - x_min) / width
            y = y_min + i * (y_max - y_min) / height
            z = complex(x, y)
            count = 0
            while (z.real * z.real + z.imag * z.imag < 4.0) and (count < max_iterations):
                z = z * z + c
                count += 1
            section_counts[i, j] = count
    return section_counts

@njit(parallel=True, cache=True)
def fatou_section_jit(section_index, section_width, width, height,
                      x_min, x_max, y_min, y_max, max_iterations):
    """
    Calculate Fatou set iterations for a section of the image using Numba JIT compilation.
    
    Args:
        section_index (int): Index of the section to calculate
        section_width (int): Width of the section in pixels
        width (int): Total width of the image in pixels
        height (int): Total height of the image in pixels
        x_min (float): Minimum x-coordinate in the complex plane
        x_max (float): Maximum x-coordinate in the complex plane
        y_min (float): Minimum y-coordinate in the complex plane
        y_max (float): Maximum y-coordinate in the complex plane
        max_iterations (int): Maximum number of iterations to perform
        
    Returns:
        numpy.ndarray: 2D array of iteration counts for the section
    """
    section_counts = np.empty((height, section_width), dtype=np.int32)
    for i in prange(height):
        for j in range(section_width):
            global_x = section_index * section_width + j
            x = x_min + global_x * (x_max - x_min) / width
            y = y_min + i * (y_max - y_min) / height
            z = complex(x, y)
            count = 0
            while (z.real * z.real + z.imag * z.imag < 4.0) and (count < max_iterations):
                z = z - (z ** 3 - 1) / (3 * z ** 2)
                count += 1
            section_counts[i, j] = count
    return section_counts

# --------------------------------------------------------------------------
# MANDELBROT CALCULATOR CLASS
# --------------------------------------------------------------------------
class MandelbrotCalculator:
    """
    Class for calculating Mandelbrot, Julia, and Fatou set sections with color mapping.
    """

    def calculate_mandelbrot_section(self, section_index, section_width, width, height,
                                     x_min, x_max, y_min, y_max, max_iterations, theme):
        """
        Calculate and color a section of the Mandelbrot set.
        """
        iterations = mandelbrot_section_jit(section_index, section_width, width, height,
                                            x_min, x_max, y_min, y_max, max_iterations)
        colors = apply_color_theme(iterations, max_iterations, theme)
        return colors

    def calculate_julia_section(self, section_index, section_width, width, height,
                                 x_min, x_max, y_min, y_max, max_iterations, theme, c):
        """
        Calculate and color a section of the Julia set.
        """
        iterations = julia_section_jit(section_index, section_width, width, height,
                                       x_min, x_max, y_min, y_max, max_iterations, c)
        colors = apply_color_theme(iterations, max_iterations, theme)
        return colors

    def calculate_fatou_section(self, section_index, section_width, width, height,
                                 x_min, x_max, y_min, y_max, max_iterations, theme):
        """
        Calculate and color a section of the Fatou set.
        """
        iterations = fatou_section_jit(section_index, section_width, width, height,
                                       x_min, x_max, y_min, y_max, max_iterations)
        colors = apply_color_theme(iterations, max_iterations, theme)
        return colors

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
        self.app_name = "Fractal Frontier V1.0"
        self.width = 160
        self.height = 120
        self.max_iterations = 100
        self.x_min = np.float64(-2.0)
        self.x_max = np.float64(1.0)
        self.y_min = np.float64(-1.5)
        self.y_max = np.float64(1.5)
        self.base_range = self.x_max - self.x_min  # 3.0
        self.iteration_offset = 0
        self.auto_iterations = 50
        self.color_theme = "Default"
        
        self.auto_adjust = True
        
        self.canvas = tk.Canvas(master, width=self.width, height=self.height, bg="black")
        self.canvas.pack(expand=True, fill="both")
        self.canvas.bind("<Configure>", self.on_resize)
        
        self.control_frame = ttk.Frame(master)
        self.control_frame.pack()
        
        self.offset_label = ttk.Label(self.control_frame, text="Iteration Offset:")
        self.offset_label.grid(row=0, column=0, padx=5, pady=5)
        self.offset_scale = tk.Scale(self.control_frame, from_=0, to=2000,
                                     orient=tk.HORIZONTAL, command=self.slider_update)
        self.offset_scale.set(self.iteration_offset)
        self.offset_scale.grid(row=0, column=1, padx=5, pady=5)
        
        self.theme_label = ttk.Label(self.control_frame, text="Color Theme:")
        self.theme_label.grid(row=0, column=2, padx=5, pady=5)
        self.CPU_CORES_THEME = "CPU Cores"
        self.themes = ["Default", "Grayscale", "Blue", "Fire", "Rainbow", "Rainbow2", "Rainbow3", "Rainbow4", self.CPU_CORES_THEME]
        self.color_theme_var = tk.StringVar(value=self.color_theme)
        self.theme_menu = ttk.OptionMenu(self.control_frame, self.color_theme_var, self.color_theme,
                                         *self.themes, command=self.theme_changed)
        self.theme_menu.grid(row=0, column=3, padx=5, pady=5)
        
        self.save_bookmark_btn = ttk.Button(self.control_frame, text="Save Bookmark", command=self.save_bookmark)
        self.save_bookmark_btn.grid(row=1, column=1, padx=5, pady=5)
        self.load_bookmark_btn = ttk.Button(self.control_frame, text="Load/Del Bookmark", command=self.load_bookmark)
        self.load_bookmark_btn.grid(row=1, column=2, padx=5, pady=5)
        
        self.reset_button = ttk.Button(self.control_frame, text="Reset", command=self.reset_view)
        self.reset_button.grid(row=1, column=3, padx=5, pady=5)

        self.auto_adjust_var = tk.BooleanVar(value=self.auto_adjust)  # Bind to self.auto_adjust
        self.auto_adjust_checkbox = ttk.Checkbutton(
            self.control_frame,
            text="Auto Adjust",
            variable=self.auto_adjust_var,
            command=self.toggle_auto_adjust
        )
        self.auto_adjust_checkbox.grid(row=1, column=0, padx=5, pady=5)
        
        self.status_bar = ttk.Label(master, text="", relief=tk.SUNKEN, anchor="w", font=("Consolas", 10))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add fractal type dropdown
        self.fractal_type_label = ttk.Label(self.control_frame, text="Fractal Type:")
        self.fractal_type_label.grid(row=0, column=4, padx=5, pady=5)
        self.fractal_types = ["Mandelbrot", "Julia", "Fatou"]
        self.fractal_type_var = tk.StringVar(value="Mandelbrot")
        self.fractal_type_menu = ttk.OptionMenu(self.control_frame, self.fractal_type_var, "Mandelbrot",
                                                *self.fractal_types, command=self.fractal_type_changed)
        self.fractal_type_menu.grid(row=0, column=5, padx=5, pady=5)

        # Add sliders for julia_c in the control frame
        self.julia_c_real_label = ttk.Label(self.control_frame, text="Julia C Real:")
        self.julia_c_real_label.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="ew")  # Label on top
        self.julia_c_real_var = tk.DoubleVar(value=-0.7)  # Default real part
        self.julia_c_real_entry = ttk.Entry(self.control_frame, textvariable=self.julia_c_real_var, width=10)
        self.julia_c_real_entry.grid(row=3, column=1, padx=5, pady=5)
        self.julia_c_real_entry.bind("<Return>", self.update_julia_c)

        self.julia_c_real_slider = tk.Scale(
            self.control_frame,
            from_=-2.0,
            to=2.0,
            resolution=0.0001,  # Higher resolution for finer adjustments
            orient=tk.HORIZONTAL,
            variable=self.julia_c_real_var,
            command=lambda value: self.update_julia_c_slider()
        )
        self.julia_c_real_slider.grid(row=4, column=0, columnspan=8, padx=1, pady=1, sticky="ew")  # Expand slider

        self.julia_c_imag_label = ttk.Label(self.control_frame, text="Julia C Imag:")
        self.julia_c_imag_label.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky="ew")  # Label on top
        self.julia_c_imag_var = tk.DoubleVar(value=0.27015)  # Default imaginary part
        self.julia_c_imag_entry = ttk.Entry(self.control_frame, textvariable=self.julia_c_imag_var, width=10)
        self.julia_c_imag_entry.grid(row=5, column=1, padx=5, pady=5)
        self.julia_c_imag_entry.bind("<Return>", self.update_julia_c)

        self.julia_c_imag_slider = tk.Scale(
            self.control_frame,
            from_=-2.0,
            to=2.0,
            resolution=0.0001,  # Higher resolution for finer adjustments
            orient=tk.HORIZONTAL,
            variable=self.julia_c_imag_var,
            command=lambda value: self.update_julia_c_slider()
        )
        self.julia_c_imag_slider.grid(row=6, column=0, columnspan=8, padx=1, pady=1, sticky="ew")  # Expand slider

        # Add slider for "e exp i n" in the control frame
        self.e_exp_i_n_label = ttk.Label(self.control_frame, text="0.7885 * e^(i * slider_value)")
        self.e_exp_i_n_label.grid(row=7, column=0, columnspan=3, padx=5, pady=5, sticky="ew")  # Label on top

        self.e_exp_i_n_var = tk.DoubleVar(value=0.0)  # Default value for the slider
        self.e_exp_i_n_slider = tk.Scale(
            self.control_frame,
            from_=0.0,
            to=2 * math.pi,  # Slider range from 0 to 2π
            resolution=0.01,  # Higher resolution for finer adjustments
            orient=tk.HORIZONTAL,
            variable=self.e_exp_i_n_var,
            command=lambda value: self.update_e_exp_i_n_slider()
        )
        self.e_exp_i_n_slider.grid(row=8, column=0, columnspan=8, padx=5, pady=5, sticky="ew")  # Expand slider

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
        self.draw_mandelbrot()

    def on_windows_mousewheel(self, event):
        """Handle mouse wheel events on Windows."""
        # In Windows, event.delta is positive when scrolling up and negative when scrolling down
        if event.delta > 0:
            self.zoom(event, 0.75)  # Zoom in
        else:
            self.zoom(event, 1.5)    # Zoom out

    def on_macos_mousewheel(self, event):
        """Handle mouse wheel events on macOS."""
        # In macOS, event.delta works similarly to Windows but may have different scaling
        if event.delta > 0:
            self.zoom(event, 0.75)  # Zoom in
        else:
            self.zoom(event, 1.5)    # Zoom out

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
        self.draw_mandelbrot()
    
    def draw_mandelbrot(self):
        """
        Calculate and draw the fractal set using multiprocessing.
        
        Updates the display with the current view parameters and color theme.
        """
        current_range = self.x_max - self.x_min
        self.zoom_level = self.base_range / current_range

        if self.auto_adjust:
            self.auto_iterations = int(100 * (self.zoom_level ** 0.11))
            self.offset_scale.config(from_=(2 + self.auto_iterations * -1))
            self.offset_scale.config(to=int(200 * (10 ** 1.2) + 100 * math.log(self.zoom_level / 10)))

        self.max_iterations = self.auto_iterations + self.iteration_offset

        start_time = time.time()
        num_sections = 2 ** int(np.floor(np.log2(self.num_cores)))
        section_width = self.width // num_sections

        fractal_type = self.fractal_type_var.get()
        tasks = []
        if (fractal_type == "Mandelbrot"):
            if self.color_theme == self.CPU_CORES_THEME:
                tasks = [
                    (i, section_width, self.width, self.height,
                    self.x_min, self.x_max, self.y_min, self.y_max, self.max_iterations, random.choice(self.themes))
                    for i in range(num_sections)
                ]
            else:
                tasks = [
                    (i, section_width, self.width, self.height,
                    self.x_min, self.x_max, self.y_min, self.y_max, self.max_iterations, self.color_theme)
                    for i in range(num_sections)
                ]
            section_arrays = self.pool.starmap(self.calculator.calculate_mandelbrot_section, tasks)
        elif fractal_type == "Julia":
            if self.color_theme == self.CPU_CORES_THEME:
                tasks = [
                    (i, section_width, self.width, self.height,
                    self.x_min, self.x_max, self.y_min, self.y_max, self.max_iterations, random.choice(self.themes), self.julia_c)
                    for i in range(num_sections)
                ]
            else:
                tasks = [
                    (i, section_width, self.width, self.height,
                    self.x_min, self.x_max, self.y_min, self.y_max, self.max_iterations, self.color_theme, self.julia_c)
                    for i in range(num_sections)
                ]
            section_arrays = self.pool.starmap(self.calculator.calculate_julia_section, tasks)
        elif fractal_type == "Fatou":
            if self.color_theme == self.CPU_CORES_THEME:
                tasks = [
                    (i, section_width, self.width, self.height,
                    self.x_min, self.x_max, self.y_min, self.y_max, self.max_iterations, random.choice(self.themes))
                    for i in range(num_sections)
                ]
            else:
                tasks = [
                    (i, section_width, self.width, self.height,
                    self.x_min, self.x_max, self.y_min, self.y_max, self.max_iterations, self.color_theme)
                    for i in range(num_sections)
                ]
            section_arrays = self.pool.starmap(self.calculator.calculate_fatou_section, tasks)

        full_array = np.hstack(section_arrays)
        image = Image.fromarray(full_array, mode="RGB")
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        elapsed = time.time() - start_time

        title_str = (f"{self.app_name} | {elapsed:.2f}s | {self.zoom_level:.2f}x | Iter: {self.max_iterations} "
                     f"(Base: {self.auto_iterations} + Off: {self.iteration_offset}) "
                     f"| Theme: {self.color_theme}")
        self.master.title(title_str)

        coord_str = (f"X_min: {self.x_min:.16g}, X_max: {self.x_max:.16g} | "
                     f"Y_min: {self.y_min:.16g}, Y_max: {self.y_max:.16g}")
        self.status_bar.config(text=f"Cores: {num_sections} | " + coord_str)
    
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
            "fractal_type": self.fractal_type_var.get(),  # Save the current fractal type
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "zoom_level": self.base_range / (self.x_max - self.x_min),
            "max_iterations": self.max_iterations,
            "auto_iterations": self.auto_iterations,
        }

        # Add julia_c to the bookmark if the fractal type is Julia
        if self.fractal_type_var.get() == "Julia":
            bookmark["julia_c"] = {
                "real": self.julia_c.real,
                "imag": self.julia_c.imag
            }

        file_path = "bookmarks.json"
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
        file_path = "bookmarks.json"
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
        lb = tk.Listbox(lb_window, yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set, font=("Consolas", 10))
        for index, bm in enumerate(bookmarks):
            summary = (
                f"{index+1}: {bm['timestamp']} | "
                f"Fractal: {bm.get('fractal_type', 'Mandelbrot')} | "
                f"Theme: {bm['color_theme']} | "
                f"Zoom: {bm['zoom_level']:.2f}x | "
                f"Auto Adj: {bm['auto_adjust']} | "
                f"Iter_Offset: {bm['iteration_offset']} | "
                f"Auto Iter: {bm['auto_iterations']} | "
                f"X: {bm['x_min']:.16g} to {bm['x_max']:.16g} | "
                f"Y: {bm['y_min']:.16g} to {bm['y_max']:.16g}"
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
            self.x_min = bm["x_min"]
            self.x_max = bm["x_max"]
            self.y_min = bm["y_min"]
            self.y_max = bm["y_max"]
            self.iteration_offset = bm["iteration_offset"]
            self.offset_scale.config(to=self.iteration_offset)  # Force slider maximum to the loaded value.
            self.offset_scale.set(self.iteration_offset)
            self.color_theme = bm["color_theme"]
            self.color_theme_var.set(self.color_theme)
            self.auto_adjust = bm.get("auto_adjust", True)
            self.auto_adjust_var.set(self.auto_adjust)  # Update the checkbox state
            self.fractal_type_var.set(bm.get("fractal_type", "Mandelbrot"))  # Load the fractal type
            if self.fractal_type_var.get() == "Julia":
                julia_c_data = bm.get("julia_c", {"real": -0.7, "imag": 0.27015})
                self.julia_c = complex(julia_c_data["real"], julia_c_data["imag"])  # Reconstruct the complex number
                self.julia_c_real_var.set(self.julia_c.real)  # Update the real part input field
                self.julia_c_imag_var.set(self.julia_c.imag)  # Update the imaginary part input field
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
                messagebox.showinfo("No Selection", "Please select a bookmark to delete.")
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

        delete_btn = ttk.Button(button_frame, text="Delete Selected", command=delete_selected)
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
        if self.zoom_level <= 0.26 and zoom_factor >= 2.0:
            self.is_zooming = False
            return
        if getattr(self, 'is_zooming', False):  # Ignore if already zooming
            return

        self.is_zooming = True  # Set zooming flag
        try:
            width = self.x_max - self.x_min
            height = self.y_max - self.y_min
            if event.type == "2":
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

    def fluid_zoom(self, event, zoom_factor, frames=10, duration=0.09):
        """
        Perform a smooth zoom animation.
        
        Args:
            event: Mouse or keyboard event containing position information
            zoom_factor (float): Factor to zoom by (>1 zooms out, <1 zooms in)
            frames (int, optional): Number of animation frames. Defaults to 10.
            duration (float, optional): Total animation duration in seconds. Defaults to 0.09.
        """
        if self.zoom_level <= 0.26 and zoom_factor >= 2.0:
            self.is_zooming = False
            return
        if getattr(self, 'is_zooming', False):  # Ignore if already zooming
            return

        self.is_zooming = True  # Set zooming flag
        try:
            
            width = self.x_max - self.x_min
            height = self.y_max - self.y_min

            if event.type == "2":  # Keyboard event
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

            frame_duration = int(duration * 1000 / frames)  # Convert duration to milliseconds

            def animate_zoom_step(frame=0):
                """
                Recursive function to animate each step of the zoom.
                
                Args:
                    frame (int, optional): Current animation frame. Defaults to 0.
                """
                if frame < frames:
                    self.x_min += step_x_min
                    self.x_max += step_x_max
                    self.y_min += step_y_min
                    self.y_max += step_y_max
                    self.draw_mandelbrot()
                    self.master.after(frame_duration, animate_zoom_step, frame + 1)
                else:
                    self.is_zooming = False  # Reset zooming flag after animation

            animate_zoom_step()
        except Exception:
            self.is_zooming = False  # Ensure flag is reset in case of an error

    def move_down(self, event):
        """
        Move the view down by 10% of the current height.
        
        Args:
            event: Keyboard event
        """
        self.y_min += 0.1 * (self.y_max - self.y_min)
        self.y_max += 0.1 * (self.y_max - self.y_min)
        self.draw_mandelbrot()
    
    def move_up(self, event):
        """
        Move the view up by 10% of the current height.
        
        Args:
            event: Keyboard event
        """
        self.y_min -= 0.1 * (self.y_max - self.y_min)
        self.y_max -= 0.1 * (self.y_max - self.y_min)
        self.draw_mandelbrot()
    
    def move_left(self, event):
        """
        Move the view left by 10% of the current width.
        
        Args:
            event: Keyboard event
        """
        self.x_min -= 0.1 * (self.x_max - self.x_min)
        self.x_max -= 0.1 * (self.x_max - self.x_min)
        self.draw_mandelbrot()
    
    def move_right(self, event):
        """
        Move the view right by 10% of the current width.
        
        Args:
            event: Keyboard event
        """
        self.x_min += 0.1 * (self.x_max - self.x_min)
        self.x_max += 0.1 * (self.x_max - self.x_min)
        self.draw_mandelbrot()
    
    def toggle_auto_adjust(self):
        """
        Toggle automatic iteration adjustment based on zoom level.
        
        When enabled, the number of iterations is automatically adjusted based on zoom level.
        When disabled, the iteration slider range is expanded to allow manual control.
        """
        self.auto_adjust = self.auto_adjust_var.get()
        if not self.auto_adjust:
            self.offset_scale.config(from_= (2+self.auto_iterations * -1))
            self.offset_scale.config(to= 16384)
        self.draw_mandelbrot()
    
    def advance_theme(self, event=None):
        """
        Cycle to the next color theme in the list.
        
        Args:
            event: Event object, optional
        """
        current_theme = self.color_theme_var.get()
        themes = self.themes
        current_index = themes.index(current_theme)
        next_index = (current_index + 1) % len(themes)  # Wrap around to the start if at the end
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

    def update_julia_c(self, event=None):
        """
        Update the Julia set constant based on input fields.
        
        Args:
            event: Event object, optional
        """
        real_part = self.julia_c_real_var.get()
        imag_part = self.julia_c_imag_var.get()
        self.julia_c = complex(real_part, imag_part)
        self.draw_mandelbrot()

    def update_julia_c_slider(self):
        """
        Update the Julia set constant based on slider values.
        """
        real_part = self.julia_c_real_var.get()
        imag_part = self.julia_c_imag_var.get()
        self.julia_c = complex(real_part, imag_part)
        self.draw_mandelbrot()

    def update_e_exp_i_n_slider(self):
        """
        Update the Julia set constant (c) based on the "e exp i n" slider value.
        """
        slider_value = self.e_exp_i_n_var.get()
        self.julia_c = 0.7885 * complex(math.cos(slider_value), math.sin(slider_value))  # 0.7885 * e^(i * slider_value)
        self.julia_c_real_var.set(self.julia_c.real)  # Update the real part input field
        self.julia_c_imag_var.set(self.julia_c.imag)  # Update the imaginary part input field
        self.draw_mandelbrot()

    def __del__(self):
        """
        Clean up resources when the object is deleted.
        
        Ensures the multiprocessing pool is properly closed.
        """
        if hasattr(self, 'pool'):
            self.pool.close()
            self.pool.join()

if __name__ == "__main__":
    """
    Main entry point for the application.
    
    Initializes the Tkinter root window and creates the Mandelbrot viewer.
    """
    multiprocessing.freeze_support()
    root = tk.Tk()
    root.geometry("320x240")
    root.minsize(800,680)
    viewer = MandelbrotViewer(root)
    root.mainloop()
