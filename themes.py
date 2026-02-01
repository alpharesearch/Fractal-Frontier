import colorsys

import numpy as np


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
        norm_vals = (iterations.astype(np.float32) * 255.0 / max_iterations).astype(
            np.uint8
        )
        non_mask = ~mask
        colors[non_mask, 0] = norm_vals[non_mask]
        colors[non_mask, 1] = 255 - norm_vals[non_mask]
        colors[non_mask, 2] = 100

    def grayscale_theme(iterations, max_iterations, colors):
        """Simple grayscale theme."""
        norm_vals = (iterations.astype(np.float32) * 255.0 / max_iterations).astype(
            np.uint8
        )
        colors[..., 0] = norm_vals
        colors[..., 1] = norm_vals
        colors[..., 2] = norm_vals
        mask = iterations == max_iterations
        colors[mask] = (0, 0, 0)

    def blue_theme(iterations, max_iterations, colors):
        """Blue-dominant color theme."""
        norm_vals = (iterations.astype(np.float32) * 255.0 / max_iterations).astype(
            np.uint8
        )
        colors[..., 0] = 255 - norm_vals
        colors[..., 1] = 255 - norm_vals
        colors[..., 2] = norm_vals
        mask = iterations == max_iterations
        colors[mask] = (0, 0, 0)

    def fire_theme(iterations, max_iterations, colors):
        """Fire-like color theme with red and orange tones."""
        norm_vals = (iterations.astype(np.float32) * 255.0 / max_iterations).astype(
            np.uint8
        )
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
        """Enhanced rainbow theme with higher color values –
        clipped to avoid overflow."""
        palette = np.empty((256, 3), dtype=np.uint8)
        for i in range(256):
            h = i / 256.0
            r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            # Scale up then clip to 255 before casting.
            palette[i] = np.clip(np.array([r, g, b]) * 1024, 0, 255).astype(np.uint8)
        mod_vals = np.mod(iterations, 256)
        colors[:] = palette[mod_vals]
        mask = iterations == max_iterations
        colors[mask] = (0, 0, 0)

    def rainbow3_theme(iterations, max_iterations, colors):
        """Rainbow theme with 1024 colors – clipped to avoid overflow."""
        palette = np.empty((1024, 3), dtype=np.uint8)
        for i in range(1024):
            h = i / 1024.0
            r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            palette[i] = np.clip(np.array([r, g, b]) * 1024, 0, 255).astype(np.uint8)
        mod_vals = np.mod(iterations, 1024)
        colors[:] = palette[mod_vals]
        mask = iterations == max_iterations
        colors[mask] = (0, 0, 0)

    def rainbow4_theme(iterations, max_iterations, colors):
        """Return the original 8192‑color palette but clip to 255 so
        NumPy doesn’t raise an error."""
        palette = np.empty((8192, 3), dtype=np.uint8)
        for i in range(8192):
            h = i / 8192.0
            r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            # Scale to the 8192‑step palette then clip to 255 before cast.
            palette[i] = np.clip(np.array([r, g, b]) * 8192, 0, 255).astype(np.uint8)
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
        "Rainbow4": rainbow4_theme,
    }

    height, width = iterations.shape
    colors = np.empty((height, width, 3), dtype=np.uint8)
    theme_function = theme_functions.get(theme, default_theme)
    theme_function(iterations, max_iterations, colors)
    return colors
