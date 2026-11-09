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
from decimal import localcontext

import mpmath as mp
import numpy as np
from numba import njit, prange

from dd_math import DD_EPS, dd_add_d, dd_sub
from fractals import mandelbrot_section_dd_jit
from hp_view import decimal_to_dd, exact_precision, to_decimal

# Reference orbits iterate past the escape radius so pixels lagging behind
# the centre stay covered; pixel counts use the classic |z| > 2 threshold.
_REFERENCE_BAILOUT2 = 1e16
_ESCAPE_RADIUS2 = 4.0
# Switching test: |δ|² > SWITCH_RATIO2 · |z|²  (|δ| > 1e-6·|z|).
SWITCH_RATIO2 = 1e-12
# Reference rebuilds per section before the fallback takes over.
MAX_ROOTS = 12
# Arbitrary-precision band: more re-roots (each is one reference orbit, cheap
# next to per-pixel iteration) and a wider dps budget, because the coordinates
# no longer stop at the double-double wall.
HP_MAX_ROOTS = 24
HP_MAX_DPS = 250
# mpmath direct-iteration budget for pixels no reference orbit resolved.
# Direct iteration costs ~max_iterations multi-precision steps per pixel, so
# this is deliberately small: it guarantees correctness for a bounded number
# of pixels and reports the rest instead of silently mis-colouring them.
HP_DIRECT_BUDGET = 256


def _dd_to_mpf(hi, lo):
    """Exact mpmath value of a double-double (float→mpf conversion is exact)."""
    return mp.mpf(float(hi)) + mp.mpf(float(lo))


def digits_for_pitch(pitch, extra=5, cap=150):
    """mpmath digits needed for a reference orbit at the given pixel pitch."""
    if pitch <= 0.0 or not math.isfinite(pitch):
        return 40
    depth = -math.log10(pitch)  # ≈ log10 of the magnification
    return int(min(cap, max(40, 35 + depth * 0.55 + extra)))


def build_reference(c0r_hi, c0r_lo, c0i_hi, c0i_lo, dps, max_iterations):
    """High-precision complex reference orbit of c0 = (c0r, c0i)."""
    with mp.workdps(dps):
        c0 = mp.mpc(
            _dd_to_mpf(c0r_hi, c0r_lo),
            _dd_to_mpf(c0i_hi, c0i_lo),
        )
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


def build_reference_hp(c0r_str, c0i_str, dps, max_iterations):
    """Reference orbit from exact decimal strings (arbitrary-precision band).

    The strings come from the canonical ``PreciseBounds`` state, so the
    reference centre carries every digit the view has — not the ~32 a
    double-double pair can hold. Parsing happens **inside** ``workdps`` so
    mpmath keeps all of them.
    """
    with mp.workdps(dps):
        c0 = mp.mpc(mp.mpf(str(c0r_str)), mp.mpf(str(c0i_str)))
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
                # n == 0 has z_0 == 0, so the relative test would flag every
                # non-zero delta; skip it and start comparing at z_1.
                if n > 0 and d2 > SWITCH_RATIO2 * z2:
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


def _hp_grid(x_min_s, x_max_s, count):
    """Exact per-pixel coordinates along one axis.

    Returns ``(pitch, centre_str, deltas, pixel_strs, pixel_vals)`` where
    ``deltas[k] = float(pixel_k − centre)``. Pixel positions and the centre
    are computed as Decimals and only the delta is rounded to float64 — the
    step that quantizes when it goes through double-double arithmetic.
    """
    a, b = to_decimal(x_min_s), to_decimal(x_max_s)
    n = max(1, int(count))
    half = to_decimal("0.5")
    two = to_decimal(2)
    with localcontext(prec=exact_precision(a, b, extra=60)) as ctx:
        pitch = ctx.divide(ctx.add(b, -a), to_decimal(n))
        centre = ctx.add(a, ctx.multiply(pitch, ctx.divide(to_decimal(n), two)))
        deltas = np.empty(n, dtype=np.float64)
        pixels = []
        for k in range(n):
            pixel = ctx.add(a, ctx.multiply(ctx.add(to_decimal(k), half), pitch))
            pixels.append(pixel)
            deltas[k] = float(ctx.subtract(pixel, centre))
    return pitch, str(centre), deltas, [str(p) for p in pixels], pixels


def _hp_deltas_vs(pixels, centre):
    """float64 per-pixel delta against an arbitrary re-root centre."""
    centre = to_decimal(centre)
    with localcontext(prec=exact_precision(centre, *pixels[:1], extra=40)) as ctx:
        out = np.empty(len(pixels), dtype=np.float64)
        for k, pixel in enumerate(pixels):
            out[k] = float(ctx.subtract(pixel, centre))
    return out


def _hp_direct_counts(pairs, dps, max_iterations):
    """Direct multi-precision iteration for pixels no reference orbit resolved.

    Same int32 contract as every other kernel (count = index of the escaped
    iterate, ``max_iterations`` for interior). Correct at any depth, and
    ~max_iterations multi-precision steps per pixel — hence the small budget.
    """
    out = []
    with mp.workdps(dps):
        for cx_s, cy_s in pairs:
            c = mp.mpc(mp.mpf(cx_s), mp.mpf(cy_s))
            z = mp.mpc(0)
            n = 0
            while n < max_iterations:
                zr = mp.re(z)
                zi = mp.im(z)
                if zr * zr + zi * zi > _ESCAPE_RADIUS2:
                    break
                z = z * z + c
                n += 1
            out.append(n)
    return out


def calculate_mandelbrot_perturb_hp(
    width,
    height,
    x_min_s,
    x_max_s,
    y_min_s,
    y_max_s,
    max_iterations,
    dps=None,
    root_budget=HP_MAX_ROOTS,
    direct_budget=HP_DIRECT_BUDGET,
):
    """Perturbation render with arbitrary-precision coordinates.

    Coordinates arrive as exact decimal strings straight from the canonical
    view state: pitch, centre, per-pixel positions and every re-root centre
    are Decimal arithmetic. Only the per-pixel delta entering the float64
    delta recurrence is rounded. Returns ``(counts, stats)``; ``stats``
    reports roots used, dps, and how many pixels needed a fallback, so the
    caller can tell the user when a frame was not fully resolved.
    """
    width, height = max(1, int(width)), max(1, int(height))
    dx, cx_str, d0r, px_str, px_vals = _hp_grid(x_min_s, x_max_s, width)
    dy, cy_str, d0i, py_str, py_vals = _hp_grid(y_min_s, y_max_s, height)

    pitch = min(abs(float(dx)), abs(float(dy)))
    if dps is None:
        dps = digits_for_pitch(pitch, cap=HP_MAX_DPS)

    counts = np.full((height, width), -1, dtype=np.int32)
    cx, cy = to_decimal(cx_str), to_decimal(cy_str)
    roots = 0
    switches = np.zeros((height, width), dtype=np.float64)

    while roots <= root_budget:
        zs_r, zs_i = build_reference_hp(str(cx), str(cy), dps, max_iterations)
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
        counts[counts == -1] = new_counts[counts == -1]
        if not np.any(counts == -1):
            return counts, {"roots": roots, "dps": dps, "unresolved": 0, "direct": 0}

        # Re-root at the worst *unresolved* pixel. argmax(switches) alone can
        # re-pick a pixel that already settled (settled pixels keep their
        # count, and a large switch signal is not cleared when they do).
        unresolved = counts == -1
        score = np.where(unresolved, switches, -np.inf)
        flat = int(np.argmax(score))
        if not np.isfinite(score.flat[flat]) or score.flat[flat] <= 0.0:
            break  # no switch signal left: go straight to the fallback
        i_w, j_w = np.unravel_index(flat, switches.shape)
        cx, cy = px_vals[j_w], py_vals[i_w]
        d0r = _hp_deltas_vs(px_vals, cx)
        d0i = _hp_deltas_vs(py_vals, cy)
        dps = min(HP_MAX_DPS, dps + 5)
        roots += 1

    n_unresolved = int(np.count_nonzero(counts == -1))
    direct = 0
    if pitch >= float(DD_EPS):
        # In this band the double-double kernel is exact, so it is the
        # cheapest correct fallback.
        x_hi, x_lo = decimal_to_dd(to_decimal(x_min_s))
        y_hi, y_lo = decimal_to_dd(to_decimal(y_min_s))
        dd_counts = mandelbrot_section_dd_jit(
            0,
            width,
            width,
            height,
            x_hi,
            x_lo,
            float(dx),
            y_hi,
            y_lo,
            float(dy),
            max_iterations,
        )
        counts[counts == -1] = dd_counts[counts == -1]
    elif direct_budget > 0 and n_unresolved:
        # Below the double-double wall a DD fallback would re-introduce
        # exactly the quantization this module removes, so the hardest
        # unresolved pixels are iterated directly in multi-precision.
        budget = min(n_unresolved, max(0, int(direct_budget)))
        order = np.argsort(-np.where(counts == -1, switches, -np.inf))[:budget]
        rows, cols = np.unravel_index(order, (height, width))
        values = _hp_direct_counts(
            [(px_str[j], py_str[i]) for i, j in zip(rows, cols)], dps, max_iterations
        )
        for k in range(len(rows)):
            counts[rows[k], cols[k]] = values[k]
        direct = len(rows)
        n_unresolved = int(np.count_nonzero(counts == -1))
        if n_unresolved:
            # Pixels outside the budget: report them as interior (the set's
            # colour) rather than iterating a float64/DD coordinate that
            # cannot resolve them. The stats flag makes that visible.
            counts[counts == -1] = max_iterations
    else:
        counts[counts == -1] = max_iterations

    return counts, {
        "roots": roots,
        "dps": dps,
        "unresolved": n_unresolved,
        "direct": direct,
    }


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
        zs_r, zs_i = build_reference(cx_hi, cx_lo, cy_hi, cy_lo, dps, max_iterations)
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
    "HP_DIRECT_BUDGET",
    "HP_MAX_DPS",
    "HP_MAX_ROOTS",
    "build_reference",
    "build_reference_hp",
    "calculate_mandelbrot_perturb",
    "calculate_mandelbrot_perturb_hp",
    "digits_for_pitch",
]
