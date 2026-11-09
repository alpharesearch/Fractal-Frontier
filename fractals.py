from decimal import localcontext

import numpy as np
from numba import njit, prange

from themes import apply_color_theme

# Try to import GPU calculator and verify a CUDA device is actually present.
# A successful import only proves numba-cuda is installed; without a real GPU
# the first kernel launch (or device query) raises CudaSupportError, which used
# to crash the app at startup.
try:
    from numba import cuda as _numba_cuda

    from gpu_fractals import GPUCalculator

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
    - Periodicity checking (main cardioid, period-2 bulb, and Brent cycle
      detection for interior points at high iteration counts)

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

            # --- Unrolled complex iteration with Brent cycle detection ---
            # Avoid complex object creation; use separate real/imag variables.
            # Interior points converge to an attracting cycle, so they would
            # otherwise run to max_iterations.  Detecting the cycle early and
            # returning max_iterations produces the same pixel value while
            # saving the bulk of the work at high iteration counts.
            zr = 0.0
            zi = 0.0
            count = 0
            tortoise_r = 0.0
            tortoise_i = 0.0
            power = 1
            lam = 0

            while (zr * zr + zi * zi < 4.0) and (count < max_iterations):
                # z = z^2 + c
                # New real: zr^2 - zi^2 + x
                # New imag: 2 * zr * zi + y
                new_r = zr * zr - zi * zi + x
                new_i = 2.0 * zr * zi + y
                zr = new_r
                zi = new_i
                count += 1

                if (zr - tortoise_r) ** 2 + (zi - tortoise_i) ** 2 < 1e-24:
                    count = max_iterations
                    break
                lam += 1
                if lam == power:
                    tortoise_r = zr
                    tortoise_i = zi
                    power *= 2
                    lam = 0

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
    Numba JIT compilation with optimizations:
    - Unrolled complex arithmetic
    - Brent cycle detection for points in the filled Julia set

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

            # Unrolled complex iteration: z = z^2 + c, with Brent cycle
            # detection.  Points in the filled Julia set converge to an
            # attracting cycle; return max_iterations early (same colour).
            zr = x
            zi = y
            count = 0
            tortoise_r = zr
            tortoise_i = zi
            power = 1
            lam = 0

            while (zr * zr + zi * zi < 4.0) and (count < max_iterations):
                new_r = zr * zr - zi * zi + cr
                new_i = 2.0 * zr * zi + ci
                zr = new_r
                zi = new_i
                count += 1

                if (zr - tortoise_r) ** 2 + (zi - tortoise_i) ** 2 < 1e-24:
                    count = max_iterations
                    break
                lam += 1
                if lam == power:
                    tortoise_r = zr
                    tortoise_i = zi
                    power *= 2
                    lam = 0

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
# DOUBLE-DOUBLE (QUAD-FLOAT) DEEP-ZOOM KERNELS
# --------------------------------------------------------------------------
# Coordinates are carried as (hi, lo) pairs — an unevaluated 106-bit sum of
# two doubles (~32 significant digits). This pushes the smooth-zoom ceiling
# from ~1e13 magnification (float64 ULP) to ~1e30 before the perturbation
# backend takes over. TwoSum/TwoProd rely on strict IEEE round-to-nearest,
# so these kernels are compiled WITHOUT fastmath (contraction destroys the
# error term) and with cache=True.
_DD_SPLIT = 134217857.0  # 2^27 + 1 (Dekker splitter)

# CPU-only deep backends. "perturb_hp" is the perturbation algorithm fed by
# the arbitrary-precision view state (hp_view.PreciseBounds) instead of
# double-double coordinates; it is what renders below DD_EPS.
DEEP_PRECISIONS = ("double_double", "perturb", "perturb_hp")


def is_deep(precision):
    """True for the CPU-only deep backends (never GPU, never a tile cache)."""
    return precision in DEEP_PRECISIONS


@njit(cache=True)
def _dd_two_sum(a, b):
    s = a + b
    bb = s - a
    aa = s - bb
    return s, (a - aa) + (b - bb)


@njit(cache=True)
def _dd_split(a):
    t = a * _DD_SPLIT
    ah = t - (t - a)
    return ah, a - ah


@njit(cache=True)
def _dd_two_prod(a, b):
    p = a * b
    ah, al = _dd_split(a)
    bh, bl = _dd_split(b)
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


@njit(cache=True)
def _dd_add(ah, al, bh, bl):
    hi, lo = _dd_two_sum(ah, bh)
    lo += al + bl
    return _dd_two_sum(hi, lo)


@njit(cache=True)
def _dd_add_d(ah, al, b):
    hi, lo = _dd_two_sum(ah, b)
    lo += al
    return _dd_two_sum(hi, lo)


@njit(cache=True)
def _dd_mul(ah, al, bh, bl):
    hi, lo = _dd_two_prod(ah, bh)
    lo += ah * bl + al * bh
    return _dd_two_sum(hi, lo)


@njit(parallel=True, cache=True)
def mandelbrot_section_dd_jit(
    section_offset,
    section_width,
    width,
    height,
    x_min,
    x_min_lo,
    dx,
    y_min,
    y_min_lo,
    dy,
    max_iterations,
):
    """Double-deep Mandelbrot section with double-double coordinates.

    Same output contract as ``mandelbrot_section_jit`` (int32 iteration
    counts), but pixel positions are 106-bit DD sums: ``x_min``/``y_min``
    arrive as (hi, lo) pairs and ``dx``/``dy`` are plain per-pixel spacings
    (window-scale, exactly representable as doubles).

    Cardioid/bulb checks are dropped: deep views never sit inside them.
    Brent cycle detection stays (hi-part distance, threshold 1e-24).
    """
    section_counts = np.empty((height, section_width), dtype=np.int32)

    for i in prange(height):
        yh, yl = _dd_add_d(y_min, y_min_lo, (i + 0.5) * dy)
        for j in range(section_width):
            xh, xl = _dd_add_d(x_min, x_min_lo, (section_offset + j + 0.5) * dx)

            zr, zlr = 0.0, 0.0
            zi, zli = 0.0, 0.0
            count = 0
            tort_r, tort_i = 0.0, 0.0
            power = 1
            lam = 0

            while (zr * zr + zi * zi < 4.0) and (count < max_iterations):
                # z = z^2 + c in double-double
                ar, al_ = _dd_mul(zr, zlr, zr, zlr)  # zr^2
                br, bl_ = _dd_mul(zi, zli, zi, zli)  # zi^2
                rr, rl = _dd_add(ar, al_, -br, -bl_)  # zr^2 - zi^2
                mr, ml = _dd_mul(zr, zlr, zi, zli)  # zr * zi
                nr, nl = _dd_add(rr, rl, xh, xl)
                ni, nil = _dd_add(mr * 2.0, ml * 2.0, yh, yl)
                zr, zlr, zi, zli = nr, nl, ni, nil
                count += 1

                dr = zr - tort_r
                di = zi - tort_i
                if dr * dr + di * di < 1e-24:
                    count = max_iterations
                    break
                lam += 1
                if lam == power:
                    tort_r, tort_i = zr, zi
                    power *= 2
                    lam = 0

            section_counts[i, j] = count
    return section_counts


@njit(parallel=True, cache=True)
def julia_section_dd_jit(
    section_offset,
    section_width,
    width,
    height,
    x_min,
    x_min_lo,
    dx,
    y_min,
    y_min_lo,
    dy,
    max_iterations,
    c,
):
    """Julia section with double-double starting points (c stays a double)."""
    section_counts = np.empty((height, section_width), dtype=np.int32)

    cr = c.real
    ci = c.imag

    for i in prange(height):
        yh, yl = _dd_add_d(y_min, y_min_lo, (i + 0.5) * dy)
        for j in range(section_width):
            xh, xl = _dd_add_d(x_min, x_min_lo, (section_offset + j + 0.5) * dx)

            zr, zlr = xh, xl
            zi, zli = yh, yl
            count = 0
            tort_r, tort_i = zr, zi
            power = 1
            lam = 0

            while (zr * zr + zi * zi < 4.0) and (count < max_iterations):
                ar, al_ = _dd_mul(zr, zlr, zr, zlr)
                br, bl_ = _dd_mul(zi, zli, zi, zli)
                rr, rl = _dd_add(ar, al_, -br, -bl_)
                mr, ml = _dd_mul(zr, zlr, zi, zli)
                nr, nl = _dd_add(rr, rl, cr, 0.0)
                ni, nil = _dd_add(mr * 2.0, ml * 2.0, ci, 0.0)
                zr, zlr, zi, zli = nr, nl, ni, nil
                count += 1

                dr = zr - tort_r
                di = zi - tort_i
                if dr * dr + di * di < 1e-24:
                    count = max_iterations
                    break
                lam += 1
                if lam == power:
                    tort_r, tort_i = zr, zi
                    power *= 2
                    lam = 0

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

    @staticmethod
    def _deep_spacings(x_min, x_max, y_min, y_max, dd, width, height):
        """Window-scale spacings for deep backends (DD subtraction is exact)."""
        from dd_math import dd_sub, dd_to_double

        if dd is None:
            dd = (0.0, 0.0, 0.0, 0.0)
        x_min_lo, x_max_lo, y_min_lo, y_max_lo = dd
        dx = dd_to_double(*dd_sub(x_max, x_max_lo, x_min, x_min_lo)) / width
        dy = dd_to_double(*dd_sub(y_max, y_max_lo, y_min, y_min_lo)) / height
        return x_min_lo, y_min_lo, dx, dy

    def _compute_deep(
        self,
        fractal_type,
        width,
        height,
        x_min,
        x_max,
        y_min,
        y_max,
        max_iterations,
        c,
        precision,
        dd,
        hp=None,
    ):
        """CPU deep-zoom dispatch (double-double / perturbation)."""
        x_min_lo, y_min_lo, dx, dy = self._deep_spacings(
            x_min, x_max, y_min, y_max, dd, width, height
        )
        if fractal_type == "mandelbrot":
            if precision == "perturb_hp":
                from deep_zoom import calculate_mandelbrot_perturb_hp

                if hp is None:
                    raise ValueError("perturb_hp needs the hp coordinate payload")
                counts, stats = calculate_mandelbrot_perturb_hp(
                    width,
                    height,
                    hp["x_min"],
                    hp["x_max"],
                    hp["y_min"],
                    hp["y_max"],
                    max_iterations,
                )
                if stats["unresolved"]:
                    print(
                        f"HP perturb: {stats['unresolved']} px unresolved "
                        f"({stats['direct']} direct-iterated, roots={stats['roots']}, "
                        f"dps={stats['dps']})"
                    )
                return counts
            if precision == "perturb":
                from deep_zoom import calculate_mandelbrot_perturb

                return calculate_mandelbrot_perturb(
                    width,
                    height,
                    x_min,
                    x_min_lo,
                    dx,
                    y_min,
                    y_min_lo,
                    dy,
                    max_iterations,
                )
            return mandelbrot_section_dd_jit(
                0,
                width,
                width,
                height,
                x_min,
                x_min_lo,
                dx,
                y_min,
                y_min_lo,
                dy,
                max_iterations,
            )
        # Deep Julia clamps to double-double (no perturbation kernel for it);
        # a perturb_hp view arrives here only if the caller bypassed the
        # viewer's backend selection.
        if precision == "perturb_hp":
            precision = "double_double"
        if fractal_type == "julia":
            if c is None:
                c = complex(-0.7, 0.27015)
            return julia_section_dd_jit(
                0,
                width,
                width,
                height,
                x_min,
                x_min_lo,
                dx,
                y_min,
                y_min_lo,
                dy,
                max_iterations,
                c,
            )
        # Fatou has no deep kernel yet: iterate in plain float64 (the image
        # degrades at deep zoom, which beats failing the render outright).
        return fatou_section_jit(
            0, width, width, height, x_min, x_max, y_min, y_max, max_iterations
        )

    def _compute_iterations(
        self,
        fractal_type,
        width,
        height,
        x_min,
        x_max,
        y_min,
        y_max,
        max_iterations,
        c=None,
        precision="float64",
        dd=None,
        hp=None,
    ):
        """
        Compute the raw int32 iteration array for the full image.

        Backend selection by `precision`:
          - "float32"/"float64": GPU when enabled (falls back to the CPU JIT
            path at runtime, disabling GPU mode afterwards).
          - "double_double": 106-bit coordinates, mandelbrot/julia on the CPU.
          - "perturb": Deep Zoom Algorithm (mpmath reference orbit + float64
            deltas) for mandelbrot; falls back to double-double for julia.
        Deep backends never use the GPU kernels.

        `dd` carries the low parts for deep precision:
        (x_min_lo, x_max_lo, y_min_lo, y_max_lo).
        """
        max_iterations = max(1, int(max_iterations))

        if is_deep(precision):
            return self._compute_deep(
                fractal_type,
                width,
                height,
                x_min,
                x_max,
                y_min,
                y_max,
                max_iterations,
                c,
                precision,
                dd,
                hp,
            )

        if self._use_gpu:
            try:
                gpu_calc = self.get_gpu_calculator()
                if fractal_type == "mandelbrot":
                    return gpu_calc.calculate_mandelbrot_gpu(
                        width,
                        height,
                        x_min,
                        x_max,
                        y_min,
                        y_max,
                        max_iterations,
                        precision=precision,
                    )
                elif fractal_type == "julia":
                    # "c or default" would replace the valid c = 0j (falsy!)
                    # with the default constant; only None means "use default".
                    if c is None:
                        c = complex(-0.7, 0.27015)
                    return gpu_calc.calculate_julia_gpu(
                        width,
                        height,
                        x_min,
                        x_max,
                        y_min,
                        y_max,
                        max_iterations,
                        c,
                        precision=precision,
                    )
                elif fractal_type == "fatou":
                    return gpu_calc.calculate_fatou_gpu(
                        width,
                        height,
                        x_min,
                        x_max,
                        y_min,
                        y_max,
                        max_iterations,
                        precision=precision,
                    )
            except Exception as exc:
                print(f"WARNING: GPU calculation failed ({exc!r}); falling back to CPU")
                self.set_use_gpu(False)

        # CPU path (full image as a single section)
        if fractal_type == "mandelbrot":
            return mandelbrot_section_jit(
                0, width, width, height, x_min, x_max, y_min, y_max, max_iterations
            )
        elif fractal_type == "julia":
            if c is None:
                c = complex(-0.7, 0.27015)
            return julia_section_jit(
                0, width, width, height, x_min, x_max, y_min, y_max, max_iterations, c
            )
        elif fractal_type == "fatou":
            return fatou_section_jit(
                0, width, width, height, x_min, x_max, y_min, y_max, max_iterations
            )
        raise ValueError(f"Unknown fractal type: {fractal_type!r}")

    def calculate_full(
        self,
        fractal_type,
        width,
        height,
        x_min,
        x_max,
        y_min,
        y_max,
        max_iterations,
        theme,
        c=None,
        precision="float64",
        dd=None,
        hp=None,
    ):
        """
        Calculate the full fractal image in one GPU/CPU call.
        More efficient than section-based calculation for GPU.

        Args:
            fractal_type: 'mandelbrot', 'julia', or 'fatou'
            theme: color theme name, or None to return raw iteration counts
            c: Julia constant (complex), used only for julia type
            precision: backend,
                "float32"/"float64"/"double_double"/"perturb"/"perturb_hp"
            dd: (x_min_lo, x_max_lo, y_min_lo, y_max_lo) low parts for deep
                precision backends
            hp: {"x_min", "x_max", "y_min", "y_max"} exact decimal strings for
                the arbitrary-precision perturb band
        """
        iterations = self._compute_iterations(
            fractal_type,
            width,
            height,
            x_min,
            x_max,
            y_min,
            y_max,
            max_iterations,
            c,
            precision=precision,
            dd=dd,
            hp=hp,
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
        if theme is None:
            return iterations
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
        if theme is None:
            return iterations
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
        if theme is None:
            return iterations
        colors = apply_color_theme(iterations, max_iterations, theme)
        return colors

    def calculate_mandelbrot_section_dd(
        self,
        section_offset,
        section_width,
        width,
        height,
        x_min,
        x_min_lo,
        dx,
        y_min,
        y_min_lo,
        dy,
        max_iterations,
        theme,
    ):
        """Double-double Mandelbrot section (pool worker, deep zoom).

        ``x_min``/``y_min`` are the view's base (hi, lo) pairs;
        ``section_offset`` is the section's global column offset, which the
        kernel applies. ``dx``/``dy`` are the plain per-pixel spacings.
        """
        iterations = mandelbrot_section_dd_jit(
            section_offset,
            section_width,
            width,
            height,
            x_min,
            x_min_lo,
            dx,
            y_min,
            y_min_lo,
            dy,
            max_iterations,
        )
        if theme is None:
            return iterations
        return apply_color_theme(iterations, max_iterations, theme)

    def calculate_julia_section_dd(
        self,
        section_offset,
        section_width,
        width,
        height,
        x_min,
        x_min_lo,
        dx,
        y_min,
        y_min_lo,
        dy,
        max_iterations,
        theme,
        c,
    ):
        """Double-double Julia section (pool worker, deep zoom).

        ``x_min``/``y_min`` are the view's base (hi, lo) pairs;
        ``section_offset`` is the section's global column offset, which the
        kernel applies.
        """
        iterations = julia_section_dd_jit(
            section_offset,
            section_width,
            width,
            height,
            x_min,
            x_min_lo,
            dx,
            y_min,
            y_min_lo,
            dy,
            max_iterations,
            c,
        )
        if theme is None:
            return iterations
        return apply_color_theme(iterations, max_iterations, theme)

    def calculate_mandelbrot_section_perturb(
        self,
        section_offset,
        section_width,
        width,
        height,
        x_min,
        x_min_lo,
        dx,
        y_min,
        y_min_lo,
        dy,
        max_iterations,
        theme,
    ):
        """Perturbation-theory Mandelbrot section (pool worker, deep zoom).

        ``section_offset`` is unused: the reference orbits are built for this
        section's own rectangle. Kept in the signature so the pool task
        tuples stay uniform across backends.
        """
        from deep_zoom import calculate_mandelbrot_perturb

        iterations = calculate_mandelbrot_perturb(
            section_width,
            height,
            x_min,
            x_min_lo,
            dx,
            y_min,
            y_min_lo,
            dy,
            max_iterations,
        )
        if theme is None:
            return iterations
        return apply_color_theme(iterations, max_iterations, theme)

    def calculate_mandelbrot_section_perturb_hp(
        self,
        section_offset,
        section_width,
        width,
        height,
        x_min_s,
        dx_s,
        y_min_s,
        dy_s,
        max_iterations,
        theme,
    ):
        """Perturbation section with arbitrary-precision coordinates (pool worker).

        ``section_offset`` is unused: each section builds reference orbits for
        its own rectangle. The section's left edge and pitch arrive as exact
        decimal strings from the canonical view state, and the right edge is
        ``x_min + section_width * dx`` in Decimal — no double-double addition,
        which is what quantized deep section stitching before.
        """
        from deep_zoom import calculate_mandelbrot_perturb_hp
        from hp_view import to_decimal

        with localcontext(prec=120) as ctx:
            x_min = to_decimal(x_min_s)
            dx = to_decimal(dx_s)
            x_max = ctx.add(x_min, ctx.multiply(dx, to_decimal(int(section_width))))
            y_min = to_decimal(y_min_s)
            dy = to_decimal(dy_s)
            y_max = ctx.add(y_min, ctx.multiply(dy, to_decimal(int(height))))
        iterations, stats = calculate_mandelbrot_perturb_hp(
            section_width,
            height,
            str(x_min),
            str(x_max),
            str(y_min),
            str(y_max),
            max_iterations,
        )
        if stats["unresolved"]:
            print(
                f"HP section {section_offset}: {stats['unresolved']} px unresolved "
                f"({stats['direct']} direct, roots={stats['roots']}, "
                f"dps={stats['dps']})"
            )
        if theme is None:
            return iterations
        return apply_color_theme(iterations, max_iterations, theme)
