import numpy as np
from numba import njit, prange

from themes import apply_color_theme

# Try to import GPU calculator and verify a CUDA device is actually present.
# A successful import only proves numba-cuda is installed; without a real GPU
# the first kernel launch (or device query) raises CudaSupportError, which used
# to crash the app at startup.
try:
    from gpu_fractals import GPUCalculator
    from numba import cuda as _numba_cuda

    try:
        GPU_AVAILABLE = bool(_numba_cuda.is_available()) and len(_numba_cuda.gpus) > 0
    except Exception:
        GPU_AVAILABLE = False
except ImportError:
    GPUCalculator = None
    GPU_AVAILABLE = False


# --------------------------------------------------------------------------
# JIT-COMPILED MANDELBROT ITERATION CALCULATOR
# --------------------------------------------------------------------------
@njit(parallel=True, cache=True, fastmath=True)
def mandelbrot_section_jit(
    section_offset,
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
        section_offset (int): First global column of this section
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
            global_x = section_offset + j
            
            # Map pixel coordinates to complex plane
            x = x_min + global_x * x_range * width_inv
            y = y_min + i * y_range * height_inv
            
            # --- Periodicity checking ---
            # Main cardioid check: (x-0.25)² + y² < 0.0625
            q = (x - 0.25) * (x - 0.25) + y * y
            if q * (q + (x - 0.25)) < 0.25 * y * y:
                section_counts[i, j] = max_iterations
                continue
            
            # Period-2 bulb check: the period-2 component is the disk
            # |c+1| < 1/4, i.e. (x+1)^2 + y^2 < 0.0625. (A radius of 0.5
            # painted a disk twice too big, swallowing the neck between the
            # two main components and the antenna base.)
            if (x + 1.0) * (x + 1.0) + y * y < 0.0625:
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
    section_offset,
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
    Numba JIT compilation with optimizations.

    Args:
        section_offset (int): First global column of this section
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
            global_x = section_offset + j
            
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
    section_offset,
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
    Numba JIT compilation with optimizations.

    Args:
        section_offset (int): First global column of this section
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
            global_x = section_offset + j
            
            x = x_min + global_x * x_range * width_inv
            y = y_min + i * y_range * height_inv
            
            # Fatou set: Newton's method for z^3 - 1 = 0
            zr = x
            zi = y
            count = 0
            
            while (zr * zr + zi * zi < 4.0) and (count < max_iterations):
                sr = zr * zr - zi * zi
                si = 2.0 * zr * zi
                tr = 3.0 * sr
                ti = 3.0 * si
                zr3 = zr * sr - zi * si
                zi3 = zr * si + zi * sr
                nr = zr3 - 1.0
                ni = zi3
                denom = tr * tr + ti * ti
                if denom == 0.0:
                    break
                div_r = (nr * tr + ni * ti) / denom
                div_i = (ni * tr - nr * ti) / denom
                zr = zr - div_r
                zi = zi - div_i
                count += 1
            
            section_counts[i, j] = count
    return section_counts


# --------------------------------------------------------------------------
# GPU-AWARE MANDELBROT CALCULATOR CLASS
# --------------------------------------------------------------------------
class MandelbrotCalculator:
    """
    Class for calculating Mandelbrot, Julia, and Fatou sets with color mapping.
    
    Supports both CPU (multi-process) and GPU (CUDA) backends.
    Use set_use_gpu(True) to enable GPU acceleration.
    """
    
    _use_gpu = GPU_AVAILABLE
    _gpu_instance = None

    @classmethod
    def set_use_gpu(cls, enabled: bool):
        """Enable or disable GPU acceleration."""
        if enabled and not GPU_AVAILABLE:
            print("WARNING: GPU not available, using CPU fallback")
            cls._use_gpu = False
        else:
            cls._use_gpu = enabled

    @classmethod
    def get_gpu_calculator(cls):
        """Return a shared GPUCalculator (created once, reused across draws)."""
        if cls._gpu_instance is None:
            cls._gpu_instance = GPUCalculator()
        return cls._gpu_instance

    def _compute_iterations(
        self, fractal_type, width, height, x_min, x_max, y_min, y_max,
        max_iterations, c=None,
    ):
        """
        Compute the raw int32 iteration array for the full image.

        Uses the GPU when enabled; falls back to the CPU JIT path if the GPU
        fails at runtime (e.g. driver lost), disabling GPU mode afterwards.
        """
        max_iterations = max(1, int(max_iterations))
        if self._use_gpu:
            try:
                gpu_calc = self.get_gpu_calculator()
                if fractal_type == 'mandelbrot':
                    return gpu_calc.calculate_mandelbrot_gpu(
                        width, height, x_min, x_max, y_min, y_max, max_iterations
                    )
                elif fractal_type == 'julia':
                    c = c or complex(-0.7, 0.27015)
                    return gpu_calc.calculate_julia_gpu(
                        width, height, x_min, x_max, y_min, y_max, max_iterations, c
                    )
                elif fractal_type == 'fatou':
                    return gpu_calc.calculate_fatou_gpu(
                        width, height, x_min, x_max, y_min, y_max, max_iterations
                    )
            except Exception as exc:
                print(f"WARNING: GPU calculation failed ({exc!r}); falling back to CPU")
                self.set_use_gpu(False)

        # CPU path (full image as a single section)
        if fractal_type == 'mandelbrot':
            return mandelbrot_section_jit(
                0, width, width, height, x_min, x_max, y_min, y_max, max_iterations
            )
        elif fractal_type == 'julia':
            c = c or complex(-0.7, 0.27015)
            return julia_section_jit(
                0, width, width, height, x_min, x_max, y_min, y_max, max_iterations, c
            )
        elif fractal_type == 'fatou':
            return fatou_section_jit(
                0, width, width, height, x_min, x_max, y_min, y_max, max_iterations
            )
        raise ValueError(f"Unknown fractal type: {fractal_type!r}")

    def calculate_full(self, fractal_type, width, height, x_min, x_max, y_min, y_max, max_iterations, theme, c=None):
        """
        Calculate the full fractal image in one GPU/CPU call.
        More efficient than section-based calculation for GPU.

        Args:
            fractal_type: 'mandelbrot', 'julia', or 'fatou'
            theme: color theme name, or None to return raw iteration counts
            c: Julia constant (complex), used only for julia type
        """
        iterations = self._compute_iterations(
            fractal_type, width, height, x_min, x_max, y_min, y_max, max_iterations, c
        )
        if theme is None:
            return iterations
        return apply_color_theme(iterations, max_iterations, theme)

    def calculate_mandelbrot_section(
        self,
        section_offset,
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
        Calculate and color a section of the Mandelbrot set (CPU-only).
        Called from the multiprocessing pool path; GPU is handled by calculate_full().
        """
        iterations = mandelbrot_section_jit(
            section_offset,
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
        section_offset,
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
            section_offset,
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
        section_offset,
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
            section_offset,
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