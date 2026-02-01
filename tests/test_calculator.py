import os
import sys

import numpy as np

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from Fractal_Frontier import MandelbrotCalculator, apply_color_theme

# List of themes used in the application
# All themes defined in the application.  The high‑resolution ones now clip to avoid overflow.
THEMES = [
    "Default",
    "Grayscale",
    "Blue",
    "Fire",
    "Rainbow",
    "Rainbow2",
    "Rainbow3",
    "Rainbow4",
]


def test_apply_color_theme_shape_and_type():
    """Ensure each theme returns an array of shape (H,W,3) and dtype uint8."""
    h, w = 5, 7
    iterations = np.zeros((h, w), dtype=int)
    max_iterations = 100
    for theme in THEMES:
        colors = apply_color_theme(iterations, max_iterations, theme=theme)
        assert colors.shape == (h, w, 3), f"Theme {theme} produced wrong shape"
        assert colors.dtype == np.uint8, f"Theme {theme} produced wrong dtype"

    # Test the CPU Cores theme
    # This theme randomly selects a color theme for each section
    # and should still produce valid output
    colors = apply_color_theme(iterations, max_iterations, theme="CPU Cores")
    assert colors.shape == (h, w, 3), "CPU Cores theme produced wrong shape"
    assert colors.dtype == np.uint8, "CPU Cores theme produced wrong dtype"

    # Test clipping behavior in high-resolution themes
    # The themes Rainbow2, Rainbow3, and Rainbow4 now clip to avoid overflow
    # They should not produce values outside the uint8 range (0-255)
    for theme in ["Rainbow2", "Rainbow3", "Rainbow4"]:
        colors = apply_color_theme(iterations, max_iterations, theme=theme)
        assert np.all(colors >= 0) and np.all(
            colors <= 255
        ), f"Theme {theme} produced values outside uint8 range"


def test_mandelbrot_calculator_output_shape_and_black_for_max():
    """Calculate a small section and verify output shape and that max iterations map to black."""
    calc = MandelbrotCalculator()
    h = w = 8
    # Use section_index 0, width=8, height=8
    result = calc.calculate_mandelbrot_section(
        section_index=0,
        section_width=w,
        width=w,
        height=h,
        x_min=-2.0,
        x_max=1.0,
        y_min=-1.5,
        y_max=1.5,
        max_iterations=50,
        theme="Default",
    )
    assert result.shape == (h, w, 3), "Result shape incorrect"
    # When all iterations are max_iterations, the color should be black
    full_iter = np.full((h, w), 50)
    colors_for_max = apply_color_theme(full_iter, 50, theme="Default")
    assert (colors_for_max == 0).all(), "Max iteration pixels not black"

    # Test Julia set calculations
    # The Julia set should produce the same output shape and have max iterations mapped to black
    result = calc.calculate_julia_section(
        section_index=0,
        section_width=w,
        width=w,
        height=h,
        x_min=-2.0,
        x_max=1.0,
        y_min=-1.5,
        y_max=1.5,
        max_iterations=50,
        theme="Default",
        c=complex(-0.7, 0.27015),
    )
    assert result.shape == (h, w, 3), "Julia set result shape incorrect"
    full_iter = np.full((h, w), 50)
    colors_for_max = apply_color_theme(full_iter, 50, theme="Default")
    assert (colors_for_max == 0).all(), "Julia set max iteration pixels not black"

    # Test Fatou set calculations
    # The Fatou set should produce the same output shape and have max iterations mapped to black
    result = calc.calculate_fatou_section(
        section_index=0,
        section_width=w,
        width=w,
        height=h,
        x_min=-2.0,
        x_max=1.0,
        y_min=-1.5,
        y_max=1.5,
        max_iterations=50,
        theme="Default",
    )
    assert result.shape == (h, w, 3), "Fatou set result shape incorrect"
    full_iter = np.full((h, w), 50)
    colors_for_max = apply_color_theme(full_iter, 50, theme="Default")
    assert (colors_for_max == 0).all(), "Fatou set max iteration pixels not black"
