import numpy as np
from numba import njit, prange

from themes import apply_color_theme


# --------------------------------------------------------------------------
# JIT-COMPILED MANDELBROT ITERATION CALCULATOR
# --------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def mandelbrot_section_jit(
    section_index,
    section_width,
    width,
    height,
    x_min,
    x_max,
    y_min,
    y_max,
    max_iterations,
):
    """
    Calculate Mandelbrot set iterations for a section
    of the image using Numba JIT compilation.

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
            while (z.real * z.real + z.imag * z.imag < 4.0) and (
                count < max_iterations
            ):
                z = z * z + c
                count += 1
            section_counts[i, j] = count
    return section_counts


@njit(parallel=True, cache=True)
def julia_section_jit(
    section_index,
    section_width,
    width,
    height,
    x_min,
    x_max,
    y_min,
    y_max,
    max_iterations,
    c,
):
    """
    Calculate Julia set iterations for a section of the image using
    Numba JIT compilation.

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
            while (z.real * z.real + z.imag * z.imag < 4.0) and (
                count < max_iterations
            ):
                z = z * z + c
                count += 1
            section_counts[i, j] = count
    return section_counts


@njit(parallel=True, cache=True)
def fatou_section_jit(
    section_index,
    section_width,
    width,
    height,
    x_min,
    x_max,
    y_min,
    y_max,
    max_iterations,
):
    """
    Calculate Fatou set iterations for a section of the image using
    Numba JIT compilation.

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
            while (z.real * z.real + z.imag * z.imag < 4.0) and (
                count < max_iterations
            ):
                z = z - (z**3 - 1) / (3 * z**2)
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

    def calculate_mandelbrot_section(
        self,
        section_index,
        section_width,
        width,
        height,
        x_min,
        x_max,
        y_min,
        y_max,
        max_iterations,
        theme,
    ):
        """
        Calculate and color a section of the Mandelbrot set.
        """
        iterations = mandelbrot_section_jit(
            section_index,
            section_width,
            width,
            height,
            x_min,
            x_max,
            y_min,
            y_max,
            max_iterations,
        )
        colors = apply_color_theme(iterations, max_iterations, theme)
        return colors

    def calculate_julia_section(
        self,
        section_index,
        section_width,
        width,
        height,
        x_min,
        x_max,
        y_min,
        y_max,
        max_iterations,
        theme,
        c,
    ):
        """
        Calculate and color a section of the Julia set.
        """
        iterations = julia_section_jit(
            section_index,
            section_width,
            width,
            height,
            x_min,
            x_max,
            y_min,
            y_max,
            max_iterations,
            c,
        )
        colors = apply_color_theme(iterations, max_iterations, theme)
        return colors

    def calculate_fatou_section(
        self,
        section_index,
        section_width,
        width,
        height,
        x_min,
        x_max,
        y_min,
        y_max,
        max_iterations,
        theme,
    ):
        """
        Calculate and color a section of the Fatou set.
        """
        iterations = fatou_section_jit(
            section_index,
            section_width,
            width,
            height,
            x_min,
            x_max,
            y_min,
            y_max,
            max_iterations,
        )
        colors = apply_color_theme(iterations, max_iterations, theme)
        return colors
