"""
GPU-accelerated fractal calculations using Numba CUDA.

Provides JIT-compiled CUDA kernels for Mandelbrot, Julia, and Fatou sets
that run on the GPU for 10-100x speedup over CPU-only approaches.
"""

import numpy as np
from numba import cuda

# --------------------------------------------------------------------------
# CUDA KERNELS - Run on GPU
# --------------------------------------------------------------------------


@cuda.jit
def mandelbrot_kernel(
    result,
    x_min,
    x_max,
    y_min,
    y_max,
    width,
    height,
    max_iterations,
):
    """
    CUDA kernel to calculate the Mandelbrot set for all pixels in parallel.

    Each thread computes one pixel.
    """
    # Calculate thread position in the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if x >= width or y >= height:
        return

    # Map pixel coordinates to complex plane
    cx = x_min + x * (x_max - x_min) / width
    cy = y_min + y * (y_max - y_min) / height

    # Periodicity checking - skip if inside main cardioid
    q = (cx - 0.25) * (cx - 0.25) + cy * cy
    if q * (q + (cx - 0.25)) < 0.25 * cy * cy:
        result[y, x] = max_iterations
        return

    # Period-2 bulb check: the period-2 component is the disk |c+1| < 1/4,
    # i.e. (cx+1)^2 + cy^2 < 0.0625 (a radius of 0.5 was twice too big).
    if (cx + 1.0) * (cx + 1.0) + cy * cy < 0.0625:
        result[y, x] = max_iterations
        return

    # Iteration with unrolled complex arithmetic
    zr = 0.0
    zi = 0.0
    count = 0

    while (zr * zr + zi * zi < 4.0) and (count < max_iterations):
        new_r = zr * zr - zi * zi + cx
        new_i = 2.0 * zr * zi + cy
        zr = new_r
        zi = new_i
        count += 1

    result[y, x] = count


@cuda.jit
def mandelbrot_kernel_f32(
    result,
    x_min,
    x_max,
    y_min,
    y_max,
    width,
    height,
    max_iterations,
):
    """
    Float32 version of the Mandelbrot kernel.

    float64 arithmetic is ~1/32-1/64 as fast as float32 on consumer GPUs.
    This kernel is used for shallow zooms where the pixel spacing is still
    comfortably above float32's ~1e-7 relative precision; deep views use the
    float64 kernel above.
    """
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if x >= width or y >= height:
        return

    cx = x_min + np.float32(x) * (x_max - x_min) / np.float32(width)
    cy = y_min + np.float32(y) * (y_max - y_min) / np.float32(height)

    c25 = np.float32(0.25)
    q = (cx - c25) * (cx - c25) + cy * cy
    if q * (q + (cx - c25)) < c25 * cy * cy:
        result[y, x] = max_iterations
        return

    c0625 = np.float32(0.0625)
    if (cx + np.float32(1.0)) * (cx + np.float32(1.0)) + cy * cy < c0625:
        result[y, x] = max_iterations
        return

    zr = np.float32(0.0)
    zi = np.float32(0.0)
    count = 0
    c4 = np.float32(4.0)
    c2 = np.float32(2.0)

    while (zr * zr + zi * zi < c4) and (count < max_iterations):
        new_r = zr * zr - zi * zi + cx
        new_i = c2 * zr * zi + cy
        zr = new_r
        zi = new_i
        count += 1

    result[y, x] = count


@cuda.jit
def julia_kernel(
    result,
    x_min,
    x_max,
    y_min,
    y_max,
    width,
    height,
    max_iterations,
    cr,
    ci,
):
    """
    CUDA kernel to calculate the Julia set for all pixels in parallel.

    Each thread computes one pixel.
    """
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if x >= width or y >= height:
        return

    cx = x_min + x * (x_max - x_min) / width
    cy = y_min + y * (y_max - y_min) / height

    # Iteration: z = z^2 + c (Julia set)
    zr = cx
    zi = cy
    count = 0

    while (zr * zr + zi * zi < 4.0) and (count < max_iterations):
        new_r = zr * zr - zi * zi + cr
        new_i = 2.0 * zr * zi + ci
        zr = new_r
        zi = new_i
        count += 1

    result[y, x] = count


@cuda.jit
def julia_kernel_f32(
    result,
    x_min,
    x_max,
    y_min,
    y_max,
    width,
    height,
    max_iterations,
    cr,
    ci,
):
    """
    Float32 version of the Julia kernel (same precision trade-off as the
    Mandelbrot float32 kernel).
    """
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if x >= width or y >= height:
        return

    cx = x_min + np.float32(x) * (x_max - x_min) / np.float32(width)
    cy = y_min + np.float32(y) * (y_max - y_min) / np.float32(height)

    zr = cx
    zi = cy
    count = 0
    c4 = np.float32(4.0)
    c2 = np.float32(2.0)

    while (zr * zr + zi * zi < c4) and (count < max_iterations):
        new_r = zr * zr - zi * zi + cr
        new_i = c2 * zr * zi + ci
        zr = new_r
        zi = new_i
        count += 1

    result[y, x] = count


@cuda.jit
def fatou_kernel(
    result,
    x_min,
    x_max,
    y_min,
    y_max,
    width,
    height,
    max_iterations,
):
    """
    CUDA kernel to calculate the Fatou set for all pixels in parallel.

    Each thread computes one pixel using Newton's method for z^3 - 1 = 0.
    """
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if x >= width or y >= height:
        return

    cx = x_min + x * (x_max - x_min) / width
    cy = y_min + y * (y_max - y_min) / height

    zr = cx
    zi = cy
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

    result[y, x] = count


@cuda.jit
def fatou_kernel_f32(
    result,
    x_min,
    x_max,
    y_min,
    y_max,
    width,
    height,
    max_iterations,
):
    """
    Float32 version of the Fatou/Newton kernel.
    """
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if x >= width or y >= height:
        return

    cx = x_min + np.float32(x) * (x_max - x_min) / np.float32(width)
    cy = y_min + np.float32(y) * (y_max - y_min) / np.float32(height)

    zr = cx
    zi = cy
    count = 0
    c4 = np.float32(4.0)
    c2 = np.float32(2.0)
    c3 = np.float32(3.0)
    c1 = np.float32(1.0)

    while (zr * zr + zi * zi < c4) and (count < max_iterations):
        sr = zr * zr - zi * zi
        si = c2 * zr * zi
        tr = c3 * sr
        ti = c3 * si
        zr3 = zr * sr - zi * si
        zi3 = zr * si + zi * sr
        nr = zr3 - c1
        ni = zi3
        denom = tr * tr + ti * ti
        if denom == np.float32(0.0):
            break
        div_r = (nr * tr + ni * ti) / denom
        div_i = (ni * tr - nr * ti) / denom
        zr = zr - div_r
        zi = zi - div_i
        count += 1

    result[y, x] = count


# --------------------------------------------------------------------------
# GPU CALCULATOR CLASS
# --------------------------------------------------------------------------


class GPUCalculator:
    """
    Class for calculating fractal sets using GPU acceleration.
    """

    def __init__(self):
        """Initialize the GPU calculator."""
        self.device = cuda.get_current_device()
        print(f"GPU: {self.device.name}")
        print(f"  SMs: {self.device.MULTIPROCESSOR_COUNT}")
        try:
            max_threads = self.device.MAX_THREADS_PER_MULTIPROCESSOR
            print(f"  Threads/SM: {max_threads}")
        except AttributeError:
            # Newer Numba versions may not expose this attribute
            pass
        # One reusable device buffer per size.  A render is single-threaded,
        # and copy_to_host synchronises before returning, so reuse is safe.
        self._d_result = None
        self._d_shape = None

    @staticmethod
    def _validate_precision(precision):
        if precision not in ("float32", "float64"):
            raise ValueError(f"Unknown GPU precision: {precision!r}")

    def _get_device_result(self, height, width):
        """Return a reusable uninitialised int32 device buffer."""
        shape = (int(height), int(width))
        if self._d_shape != shape:
            # The kernel writes every valid pixel, so no zero-initialisation
            # or host->device copy is needed.
            self._d_result = cuda.device_array(shape, dtype=np.int32)
            self._d_shape = shape
        return self._d_result

    @staticmethod
    def _grid(width, height):
        """Threads per block and exact block counts for the image."""
        threads_per_block = (32, 8)
        # Launch exactly as many blocks as needed; the kernel already guards
        # against out-of-range thread indices.
        blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
        return (blocks_per_grid_x, blocks_per_grid_y), threads_per_block

    def calculate_mandelbrot_gpu(
        self,
        width,
        height,
        x_min,
        x_max,
        y_min,
        y_max,
        max_iterations,
        precision="float64",
    ):
        """
        Calculate Mandelbrot set using GPU.

        precision="float32" is much faster on consumer GPUs but only safe for
        shallow zooms; use "float64" for deep views.
        """
        self._validate_precision(precision)
        h_result = np.empty((height, width), dtype=np.int32)
        d_result = self._get_device_result(height, width)
        blocks_per_grid, threads_per_block = self._grid(width, height)

        if precision == "float32":
            mandelbrot_kernel_f32[blocks_per_grid, threads_per_block](
                d_result,
                np.float32(x_min),
                np.float32(x_max),
                np.float32(y_min),
                np.float32(y_max),
                width,
                height,
                max_iterations,
            )
        else:
            mandelbrot_kernel[blocks_per_grid, threads_per_block](
                d_result,
                np.float64(x_min),
                np.float64(x_max),
                np.float64(y_min),
                np.float64(y_max),
                width,
                height,
                max_iterations,
            )

        d_result.copy_to_host(h_result)
        return h_result

    def calculate_julia_gpu(
        self,
        width,
        height,
        x_min,
        x_max,
        y_min,
        y_max,
        max_iterations,
        c,
        precision="float64",
    ):
        """
        Calculate Julia set using GPU.
        """
        self._validate_precision(precision)
        h_result = np.empty((height, width), dtype=np.int32)
        d_result = self._get_device_result(height, width)
        blocks_per_grid, threads_per_block = self._grid(width, height)

        if precision == "float32":
            julia_kernel_f32[blocks_per_grid, threads_per_block](
                d_result,
                np.float32(x_min),
                np.float32(x_max),
                np.float32(y_min),
                np.float32(y_max),
                width,
                height,
                max_iterations,
                np.float32(c.real),
                np.float32(c.imag),
            )
        else:
            julia_kernel[blocks_per_grid, threads_per_block](
                d_result,
                np.float64(x_min),
                np.float64(x_max),
                np.float64(y_min),
                np.float64(y_max),
                width,
                height,
                max_iterations,
                c.real,
                c.imag,
            )

        d_result.copy_to_host(h_result)
        return h_result

    def calculate_fatou_gpu(
        self,
        width,
        height,
        x_min,
        x_max,
        y_min,
        y_max,
        max_iterations,
        precision="float64",
    ):
        """
        Calculate Fatou set using GPU.
        """
        self._validate_precision(precision)
        h_result = np.empty((height, width), dtype=np.int32)
        d_result = self._get_device_result(height, width)
        blocks_per_grid, threads_per_block = self._grid(width, height)

        if precision == "float32":
            fatou_kernel_f32[blocks_per_grid, threads_per_block](
                d_result,
                np.float32(x_min),
                np.float32(x_max),
                np.float32(y_min),
                np.float32(y_max),
                width,
                height,
                max_iterations,
            )
        else:
            fatou_kernel[blocks_per_grid, threads_per_block](
                d_result,
                np.float64(x_min),
                np.float64(x_max),
                np.float64(y_min),
                np.float64(y_max),
                width,
                height,
                max_iterations,
            )

        d_result.copy_to_host(h_result)
        return h_result
