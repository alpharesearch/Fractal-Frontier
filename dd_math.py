"""Double-double (quad-float) arithmetic helpers.

A double-double number is an unevaluated sum ``hi + lo`` of two IEEE-754
doubles, giving a 106-bit mantissa (~32 significant decimal digits) while
using ordinary float64 hardware. Dekker/Knuth TwoSum/TwoProd require strict
IEEE round-to-nearest semantics: the Numba kernels that mirror these routines
must NOT be compiled with ``fastmath=True`` (contraction and fast-math
reassociation silently destroy the error term).

This module is the Python side; the hot loops live in ``fractals.py`` as
Numba clones of the same formulas.
"""

import math

# Dekker splitting constant 2^27 + 1: splits a double into two 26-bit halves
# whose product is exact.
_SPLIT = 134217857.0

# Effective machine epsilon of a double-double (2^-105 ≈ 2.5e-32).
DD_EPS = 2.0**-105


def two_sum(a, b):
    """Exact sum of two doubles: (s + e) == a + b in exact arithmetic.

    Knuth's TwoSum; requires strict IEEE round-to-nearest (no fastmath).
    """
    s = a + b
    bb = s - a
    aa = s - bb
    e = (a - aa) + (b - bb)
    return s, e


def split(a):
    """Dekker split: a = ah + al with ah having 26 significant bits."""
    t = a * _SPLIT
    ah = t - (t - a)
    al = a - ah
    return ah, al


def two_prod(a, b):
    """Exact product of two doubles: a * b + e == a * b in exact arithmetic."""
    p = a * b
    ah, al = split(a)
    bh, bl = split(b)
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


def renorm(ah, al):
    """Compress a pair so |al| <= DD_EPS * |ah| (normalised DD number)."""
    hi, lo = two_sum(ah, al)
    return hi, lo


def dd_add(ah, al, bh, bl):
    """Double-double + double-double."""
    hi, lo = two_sum(ah, bh)
    lo += al + bl
    return renorm(hi, lo)


def dd_sub(ah, al, bh, bl):
    """Double-double - double-double."""
    hi, lo = two_sum(ah, -bh)
    lo += al - bl
    return renorm(hi, lo)


def dd_add_d(ah, al, b):
    """Double-double + double."""
    hi, lo = two_sum(ah, b)
    lo += al
    return renorm(hi, lo)


def dd_sub_d(ah, al, b):
    """Double-double - double."""
    hi, lo = two_sum(ah, -b)
    lo += al
    return renorm(hi, lo)


def dd_mul_d(ah, al, b):
    """Double-double * double."""
    hi, lo = two_prod(ah, b)
    lo += al * b
    return renorm(hi, lo)


def dd_half(ah, al):
    """Double-double / 2 (exact: exponent shift only)."""
    return ah * 0.5, al * 0.5


def dd_center(ah, al, bh, bl):
    """Midpoint of two double-double numbers."""
    return dd_half(*dd_add(ah, al, bh, bl))


def dd_to_double(ah, al):
    """Round a double-double down to a plain double."""
    return ah + al


def dd_to_decimal(ah, al):
    """Exact decimal representation of hi + lo (Decimal is exact for floats)."""
    from decimal import Decimal

    return Decimal(ah) + Decimal(al)


def dd_format(ah, al, significant=30):
    """Fixed-notation string of hi + lo with ``significant`` significant digits.

    Uses exact Decimal arithmetic on the binary-exact float values, so the
    output carries every bit the pair can represent.
    """
    from decimal import Decimal, localcontext

    if not (math.isfinite(ah) and math.isfinite(al)):
        return str(ah)
    value = dd_to_decimal(ah, al)
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    value = value.copy_abs()
    t = value.normalize().as_tuple()
    digits, exp = t.digits, t.exponent
    scale = exp + len(digits)  # digits in the integer part
    if len(digits) > significant:
        with localcontext() as ctx:
            ctx.prec = len(digits) + 1
            # quantize to 10^(scale - significant) = the `significant`-th digit
            value = value.quantize(
                Decimal(1).scaleb(scale - significant),
                rounding="ROUND_HALF_EVEN",
                context=ctx,
            )
        t = value.normalize().as_tuple()
        digits, exp = t.digits, t.exponent
        scale = exp + len(digits)
    s = "".join(map(str, digits))
    if scale > 0:
        s = s.ljust(scale, "0") if len(s) < scale else s
        intpart, frac = s[:scale], s[scale:].rstrip("0")
        return sign + intpart + ("." + frac if frac else "")
    return sign + "0." + ("0" * -scale) + s


__all__ = [
    "DD_EPS",
    "dd_add",
    "dd_add_d",
    "dd_center",
    "dd_format",
    "dd_half",
    "dd_mul_d",
    "dd_sub",
    "dd_sub_d",
    "dd_to_decimal",
    "dd_to_double",
    "renorm",
    "split",
    "two_prod",
    "two_sum",
]
