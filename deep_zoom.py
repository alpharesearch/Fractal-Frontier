"""Perturbation-theory deep zoom for the Mandelbrot set.

At extreme magnifications the double-double backend also runs out of bits
(106-bit mantissa ≈ 32 decimal digits). The deep backend keeps float64 for
99% of the work:

1. One high-precision *reference orbit* ``z_n(c0)`` is computed with mpmath
   (50-150 digits) for the section centre ``c0``.
2. Every pixel is ``c0 + δ`` with δ in plain float64 and iterates the delta
   recurrence ``δ_{n+1} = δ_n² + 2·z_n·δ_n + δ_0`` — two complex multiplies
   per step, no big-number arithmetic.
3. When a pixel's δ grows past the switching test (|δ| > 1e-6·|z|) it is
   *re-rooted*: a new reference orbit is built at the worst offender's
   centre and surrounding pixels are iterated against it.
4. Anything still unresolved falls back to the double-double kernel.

This is the classic Deep Zoom Algorithm. It extends smooth rendering to
magnifications limited only by the iteration budget and the user's patience
(mpmath reference orbits are the bottleneck, not the pixels).
"""

import math

import mpmath as mp
import numpy as np
from numba import njit, prange

from dd_math import dd_add_d, dd_sub
from fractals import mandelbrot_section_dd_jit

# Reference orbits iterate past the escape radius so pixels lagging behind
# the centre stay covered; pixel counts use the classic |z| > 2 threshold.
_REFERENCE_BAILOUT2 = 1e16
_ESCAPE_RADIUS2 = 4.0
# Switching test: |δ|² > SWITCH_RATIO2 · |z|²  (|δ| > 1e-6·|z|).
SWITCH_RATIO2 = 1e-12
# Reference rebuilds per section before the DD fallback takes over.
MAX_ROOTS = 12


def _dd_to_mpf(hi, lo):
    """Exact mpmath value of a double-double (float→mpf conversion is exact)."""
    return mp.mpf(float(hi)) + mp.mpf(float(lo))


def digits_for_pitch(pitch, extra=5):
    """mpmath digits needed for a reference orbit at the given pixel pitch."""
    if pitch <= 0.0 or not math.isfinite(pitch):
        return 40
    depth = -math.log10(pitch)  # ≈ log10 of the magnification
    return int(min(150, max(40, 35 + depth * 0.55 + extra)))


def build_reference(c0_hi, c0_lo, dps, max_iterations):
    """High-precision reference orbit of c0, rounded to float64 arrays.

    Returns (zs_r, zs_i): arrays of z_0 … z_N. The rounded doubles are
    sufficient — the switching test keeps |δ| ≪ |z|, so the rounding of the
    reference perturbs δ far below the noise floor.
    """
    with mp.workdps(dps):
        c0 = _dd_to_mpf(c0_hi, c0_lo)
        z = mp.mpc(0)
        zr = []
        zi = []
        n = 0
        while n <= max_iterations:
            zr.append(float(mp.re(z)))
            zi.append(float(mp.im(z)))
            mag2 = mp.re(z) * mp.re(z) + mp.im(z) * mp.im(z)
            if mag2 > _REFERENCE_BAILOUT2:
                break
            z = z * z + c0
            n += 1
    return np.array(zr, dtype=np.float64), np.array(zi, dtype=np.float64)


@njit(cache=True, parallel=True)
def _perturb_counts_jit(
    width,
    height,
    d0r,
    d0i,
    zs_r,
    zs_i,
    zs_len,
    max_iterations,
):
    """Iterate every pixel's delta against the reference orbit.

    Returns (counts, switches):
      counts[i, j]  escape iteration, max_iterations for interior,
                    -1 for pixels that tripped the switching test
      switches[i, j]  |δ|² when the test tripped (0 elsewhere), used to pick
                      the worst pixel for the next re-root.
    """
    counts = np.full((height, width), -1, dtype=np.int32)
    switches = np.zeros((height, width), dtype=np.float64)

    for i in prange(height):
        for j in range(width):
            a0r = d0r[j]
            a0i = d0i[i]
            dr = a0r
            di = a0i
            n = 0
            while n < zs_len and n < max_iterations:
                zr = zs_r[n]
                zi = zs_i[n]
                tr = zr + dr
                ti = zi + di
                if tr * tr + ti * ti > _ESCAPE_RADIUS2:
                    counts[i, j] = n
                    break
                z2 = zr * zr + zi * zi
                d2 = dr * dr + di * di
                if d2 > SWITCH_RATIO2 * z2:
                    switches[i, j] = d2
                    break
                nr = dr * dr - di * di + 2.0 * (zr * dr - zi * di) + a0r
                ni = 2.0 * dr * di + 2.0 * (zr * di + zi * dr) + a0i
                dr = nr
                di = ni
                n += 1
            if n >= zs_len or n >= max_iterations:
                # Reference ended with |δ| still small: interior pixel.
                counts[i, j] = max_iterations
    return counts, switches


def calculate_mandelbrot_perturb(
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
    """Render one rectangle (full frame, section, or tile) with perturbation.

    ``x_min``/``y_min`` are double-double (hi, lo) pairs; ``dx``/``dy`` are
    the plain per-pixel spacings (window-scale doubles). Same int32 count
    contract as the other kernels.
    """
    # Centre of this rectangle as a double-double.
    cx_hi, cx_lo = dd_add_d(x_min, x_min_lo, (width * 0.5) * dx)
    cy_hi, cy_lo = dd_add_d(y_min, y_min_lo, (height * 0.5) * dy)

    # Exact per-pixel deltas: pixel − centre (DD subtraction → double).
    d0r = np.empty(width, dtype=np.float64)
    d0i = np.empty(height, dtype=np.float64)
    for j in range(width):
        ph, pl = dd_add_d(x_min, x_min_lo, (j + 0.5) * dx)
        dh, dl = dd_sub(ph, pl, cx_hi, cx_lo)
        d0r[j] = dh + dl
    for i in range(height):
        ph, pl = dd_add_d(y_min, y_min_lo, (i + 0.5) * dy)
        dh, dl = dd_sub(ph, pl, cy_hi, cy_lo)
        d0i[i] = dh + dl

    dps = digits_for_pitch(min(abs(dx), abs(dy)))
    counts = np.full((height, width), -1, dtype=np.int32)

    roots = 0
    while roots <= MAX_ROOTS:
        zs_r, zs_i = build_reference(cx_hi, cx_lo, dps, max_iterations)
        zs_len = min(len(zs_r) - 1, max_iterations)
        new_counts, switches = _perturb_counts_jit(
            width,
            height,
            d0r,
            d0i,
            zs_r,
            zs_i,
            zs_len,
            max_iterations,
        )
        # Settled pixels keep their counts; -1 stays -1.
        counts[counts == -1] = new_counts[counts == -1]
        if not np.any(counts == -1):
            return counts
        # Re-root at the pixel that tripped the switching test hardest.
        flat = int(np.argmax(switches))
        i_w, j_w = np.unravel_index(flat, switches.shape)
        if switches[i_w, j_w] <= 0.0:
            break  # no switch signal left: let the DD fallback resolve
        cx_hi, cx_lo = dd_add_d(cx_hi, cx_lo, d0r[j_w])
        cy_hi, cy_lo = dd_add_d(cy_hi, cy_lo, d0i[i_w])
        for j in range(width):
            ph, pl = dd_add_d(x_min, x_min_lo, (j + 0.5) * dx)
            dh, dl = dd_sub(ph, pl, cx_hi, cx_lo)
            d0r[j] = dh + dl
        for i in range(height):
            ph, pl = dd_add_d(y_min, y_min_lo, (i + 0.5) * dy)
            dh, dl = dd_sub(ph, pl, cy_hi, cy_lo)
            d0i[i] = dh + dl
        dps = min(150, dps + 5)
        roots += 1

    # Pixels that outlived the re-root budget: exact double-double iteration.
    unresolved = counts == -1
    dd_counts = mandelbrot_section_dd_jit(
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
    counts[unresolved] = dd_counts[unresolved]
    return counts


__all__ = [
    "build_reference",
    "calculate_mandelbrot_perturb",
    "digits_for_pitch",
]
