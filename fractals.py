import numpy as np
from numba import njit, prange

from themes import apply_color_theme


# --------------------------------------------------------------------------
# JIT-COMPILED MANDELBROT ITERATION CALCULATOR
# --------------------------------------------------------------------------
@njit(parallel=True, cache=True, fastmath=True)
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
    of the image using Numba JIT compilation with multiple optimizations:
    - Fast math optimizations
    - Unrolled complex arithmetic (avoids complex object allocations)
    - Periodicity checking (main cardioid and period-2 bulb)

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
    
    # Precompute constants
    width_inv = 1.0 / width
    height_inv = 1.0 / height
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    for i in prange(height):
        for j in range(section_width):
            global_x = section_index * section_width + j
            
            # Map pixel coordinates to complex plane
            x = x_min + global_x * x_range * width_inv
            y = y_min + i * y_range * height_inv
            
            # --- Periodicity checking ---
            # Main cardioid check: (x-0.25)² + y² < 0.0625
            q = (x - 0.25) * (x - 0.25) + y * y
            if q * (q + (x - 0.25)) < 0.25 * y * y:
                section_counts[i, j] = max_iterations
                continue
            
            # Period-2 bulb check: (x+1)^2 + y^2 < 0.25
            if (x + 1.0) * (x + 1.0) + y * y < 0.25:
                section_counts[i, j] = max_iterations
                continue
            
            # --- Unrolled complex iteration ---
            # Avoid complex object creation; use separate real/imag variables
            zr = 0.0
            zi = 0.0
            count = 0
            
            while (zr * zr + zi * zi < 4.0) and (count < max_iterations):
                # z = z^2 + c
                # New real: zr^2 - zi^2 + x
                # New imag: 2 * zr * zi + y
                new_r = zr * zr - zi * zi + x
                new_i = 2.0 * zr * zi + y
                zr = new_r
                zi = new_i
                count += 1
            
            section_counts[i, j] = count
    return section_counts


@njit(parallel=True, cache=True, fastmath=True)
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
    Numba JIT compilation with optimizations:
    - Fast math optimizations
    - Unrolled complex arithmetic (avoids complex object allocations)

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
    
    # Precompute constants
    width_inv = 1.0 / width
    height_inv = 1.0 / height
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Unpack c into real/imag to avoid complex ops
    cr = c.real
    ci = c.imag
    
    for i in prange(height):
        for j in range(section_width):
            global_x = section_index * section_width + j
            
            x = x_min + global_x * x_range * width_inv
            y = y_min + i * y_range * height_inv
            
            # Unrolled complex iteration: z = z^2 + c
            zr = x
            zi = y
            count = 0
            
            while (zr * zr + zi * zi < 4.0) and (count < max_iterations):
                new_r = zr * zr - zi * zi + cr
                new_i = 2.0 * zr * zi + ci
                zr = new_r
                zi = new_i
                count += 1
            
            section_counts[i, j] = count
    return section_counts


@njit(parallel=True, cache=True, fastmath=True)
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
    Numba JIT compilation with optimizations:
    - Fast math optimizations
    - Unrolled complex arithmetic (avoids complex object allocations)

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
    
    # Precompute constants
    width_inv = 1.0 / width
    height_inv = 1.0 / height
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    for i in prange(height):
        for j in range(section_width):
            global_x = section_index * section_width + j
            
            x = x_min + global_x * x_range * width_inv
            y = y_min + i * y_range * height_inv
            
            # Fatou set: Newton's method for z^3 - 1 = 0
            # z = z - (z^3 - 1) / (3z^2)
            zr = x
            zi = y
            count = 0
            
            while (zr * zr + zi * zi < 4.0) and (count < max_iterations):
                # z^2
                sr = zr * zr - zi * zi
                si = 2.0 * zr * zi
                # 3*z^2
                tr = 3.0 * sr
                ti = 3.0 * si
                # z^3 = z * z^2
                zr3 = zr * sr - zi * si
                zi3 = zr * si + zi * sr
                # numerator: z^3 - 1
                nr = zr3 - 1.0
                ni = zi3
                # division: (nr + ni*i) / (tr + ti*i)
                denom = tr * tr + ti * ti
                if denom == 0.0:
                    break
                div_r = (nr * tr + ni * ti) / denom
                div_i = (ni * tr - nr * ti) / denom
                # z = z - division
                zr = zr - div_r
                zi = zi - div_i
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
