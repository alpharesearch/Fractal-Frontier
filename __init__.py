"""
Fractal Frontier - An interactive Mandelbrot set explorer with multiprocessing support.

This application provides a GUI for exploring the Mandelbrot set fractal with various
color themes, zoom capabilities, and bookmark functionality.
"""

from fractals import (MandelbrotCalculator, fatou_section_jit,
                      julia_section_jit, mandelbrot_section_jit)
from themes import apply_color_theme

__all__ = [
    "MandelbrotCalculator",
    "mandelbrot_section_jit",
    "julia_section_jit",
    "fatou_section_jit",
    "apply_color_theme",
]
