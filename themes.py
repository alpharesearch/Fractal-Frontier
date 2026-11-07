import colorsys

import numpy as np


# --------------------------------------------------------------------------
# COLOR THEME MAPPING (Vectorized)
# --------------------------------------------------------------------------
#
# The themes are pure functions of the iteration count.  Building the RGB
# array is a significant part of a render at low zoom levels (tens of ms for
# a 1920x1080 canvas), so the implementations below avoid per-channel fancy
# indexing of full-size boolean masks and cache the HSV palettes.  A palette
# for 8192 hues used to be rebuilt with a Python loop on every single render.

_HSV_PALETTE_CACHE = {}


def _build_hsv_palette(steps):
    """Return a cached RGB uint8 palette of `steps` evenly spaced hues."""
    palette = _HSV_PALETTE_CACHE.get(steps)
    if palette is None:
        palette = np.empty((steps, 3), dtype=np.uint8)
        for i in range(steps):
            h = i / steps
            r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            # hsv_to_rgb returns values in [0, 1]; scale to the uint8 range.
            palette[i] = (int(r * 255), int(g * 255), int(b * 255))
        _HSV_PALETTE_CACHE[steps] = palette
    return palette


def _normalize(iterations, max_iterations):
    """Map iteration counts to 0..255 without a full-size float temp array."""
    return (iterations * 255 // max_iterations).astype(np.uint8)


def default_theme(iterations, max_iterations):
    """Default color theme with blue-green gradient."""
    norm_vals = _normalize(iterations, max_iterations)
    colors = np.empty(iterations.shape + (3,), dtype=np.uint8)
    colors[..., 0] = norm_vals
    colors[..., 1] = 255 - norm_vals
    colors[..., 2] = 100
    colors[iterations == max_iterations] = (0, 0, 0)
    return colors


def grayscale_theme(iterations, max_iterations):
    """Simple grayscale theme."""
    norm_vals = _normalize(iterations, max_iterations)
    colors = np.empty(iterations.shape + (3,), dtype=np.uint8)
    colors[..., 0] = norm_vals
    colors[..., 1] = norm_vals
    colors[..., 2] = norm_vals
    colors[iterations == max_iterations] = (0, 0, 0)
    return colors


def blue_theme(iterations, max_iterations):
    """Blue-dominant color theme."""
    norm_vals = _normalize(iterations, max_iterations)
    colors = np.empty(iterations.shape + (3,), dtype=np.uint8)
    colors[..., 0] = 255 - norm_vals
    colors[..., 1] = 255 - norm_vals
    colors[..., 2] = norm_vals
    colors[iterations == max_iterations] = (0, 0, 0)
    return colors


def fire_theme(iterations, max_iterations):
    """Fire-like color theme with red and orange tones."""
    # Keep the existing (uint8) arithmetic to preserve the exact palette.
    norm_vals = _normalize(iterations, max_iterations)
    colors = np.empty(iterations.shape + (3,), dtype=np.uint8)
    colors[..., 0] = np.clip(norm_vals * 2, 0, 255)
    colors[..., 1] = np.clip(norm_vals + 50, 0, 255)
    colors[..., 2] = np.clip(norm_vals - 100, 0, 255)
    colors[iterations == max_iterations] = (0, 0, 0)
    return colors


def rainbow_theme(iterations, max_iterations):
    """Rainbow color theme using a 256-step HSV palette."""
    return _rainbow(iterations, max_iterations, 256)


def rainbow2_theme(iterations, max_iterations):
    """Rainbow theme with a 256-step hue palette."""
    return _rainbow(iterations, max_iterations, 256)


def rainbow3_theme(iterations, max_iterations):
    """Rainbow theme with a finer 1024-step hue palette."""
    return _rainbow(iterations, max_iterations, 1024)


def rainbow4_theme(iterations, max_iterations):
    """Rainbow theme with a very fine 8192-step hue palette."""
    return _rainbow(iterations, max_iterations, 8192)


def _rainbow(iterations, max_iterations, steps):
    """Shared palette lookup for all Rainbow variants.

    `steps` is always a power of two, so a bit mask replaces the slower
    `np.mod`; `np.take` is substantially faster than advanced indexing here.
    """
    palette = _build_hsv_palette(steps)
    indices = np.bitwise_and(iterations, steps - 1)
    colors = np.take(palette, indices, axis=0)
    colors[iterations == max_iterations] = (0, 0, 0)
    return colors


_THEME_FUNCTIONS = {
    "Default": default_theme,
    "Grayscale": grayscale_theme,
    "Blue": blue_theme,
    "Fire": fire_theme,
    "Rainbow": rainbow_theme,
    "Rainbow2": rainbow2_theme,
    "Rainbow3": rainbow3_theme,
    "Rainbow4": rainbow4_theme,
}


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
    # Guard against zero/negative iteration counts (the GUI slider allows a
    # negative offset); without this the normalization below divides by zero.
    max_iterations = max(1, int(max_iterations))

    theme_function = _THEME_FUNCTIONS.get(theme, default_theme)
    return theme_function(iterations, max_iterations)
