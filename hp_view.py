"""Arbitrary-precision view state (canonical deep-zoom coordinates).

The viewer's view bounds are the *limiting* precision of every deep render:
the reference orbit is already arbitrary precision (mpmath), and the
perturbation kernel is algorithmically unbounded, but the bounds themselves
were double-double sums (106-bit mantissa ≈ 32 decimal digits). Once the
pixel pitch drops below one ``lo``-ULP (~1.5e-33 at |x| ≈ 1) adjacent columns
collapse onto the same coordinate and the image quantizes no matter which
backend renders it (measured: bookmark 9 resolves only 127 distinct columns
of a 600 px canvas).

``PreciseBounds`` replaces double-double as the **canonical** representation:
four ``Decimal`` endpoints plus the working precision they need. ``Decimal``
(not mpmath) is the canonical type because

* decimal strings round-trip through JSON exactly (bookmarks),
* precision is explicit per operation via ``localcontext`` instead of
  mpmath's process-global ``workdps``,
* there is no binary/decimal conversion ambiguity at the storage boundary.

mpmath stays the render-side reference-orbit engine; float64 stays the
per-pixel delta arithmetic. Only the *coordinates* become arbitrary precision.

Two Decimal traps this module exists to absorb:

1. ``Decimal(float)`` is binary-exact; ``Decimal(str(float))`` is not (the
   shortest round-trip repr differs from the binary value by ~1e-17
   relative — ~17 orders of magnitude coarser than the pitches represented
   here). Floats always go in through :func:`to_decimal`.
2. Every Decimal *operation* rounds to the active context, and the default
   context has prec=28. Raw ``Decimal`` arithmetic in a caller's code (or
   ``.normalize()``) therefore destroys deep precision. All math lives here,
   in contexts sized from the operands (:meth:`PreciseBounds._ctx`); callers
   get exact results, float64 arrays, or strings.
"""

import math
from decimal import Decimal, localcontext

import numpy as np

# ---------------------------------------------------------------------------
# Precision policy
# ---------------------------------------------------------------------------
# Floor precision of a shallow view: 50 digits is far past float64 and leaves
# headroom for pan/zoom chains at no measurable cost.
HP_MIN_PRECISION = 50
# Digits of headroom beyond the pixel pitch: division by the canvas width,
# interpolation factors and aspect ratios all consume digits.
HP_PRECISION_MARGIN = 20
# Optional guard for memory/performance. None (the default) = unbounded:
# precision grows with depth and is never reduced. A configured cap is not a
# theoretical limit; renders past it report the shortfall instead of silently
# quantizing.
HP_PRECISION_CAP = None
# Context precision for exact Decimal reconstruction of a double-double pair.
# A pair's expansion needs ~107 significant digits plus the exponent span of
# the low part (measured 105–108 digits for real bookmarks).
DD_DECIMAL_PRECISION = 160


def to_decimal(value):
    """Exact ``Decimal`` for a float/int/Decimal/str (see module docstring 1)."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        return Decimal(int(value))
    if isinstance(value, float):
        return Decimal.from_float(value)
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, str):
        return Decimal(value)
    raise TypeError(f"cannot build an exact Decimal from {type(value).__name__}")


def digit_span(value):
    """Digits a value's exact expansion occupies (integer + fractional).

    Uses the raw tuple — ``normalize()`` rounds to the active context and
    would report the context's digit count, not the value's.
    """
    t = to_decimal(value).as_tuple()
    digits = len(t.digits)
    scale = t.exponent + digits  # digits in the integer part (≤0 when |v|<1)
    frac = max(0, -t.exponent)  # digits after the decimal point
    return max(digits, frac + max(0, scale))


def exact_precision(*values, extra=16):
    """Context precision that makes arithmetic on ``values`` exact.

    Addition/subtraction of two Decimals is exact when prec covers the
    integer span plus the fractional span of the operands; generous by
    ``extra`` digits so quotients keep their digits too.
    """
    need = HP_MIN_PRECISION
    for value in values:
        t = to_decimal(value).as_tuple()
        need = max(need, digit_span(value) + len(t.digits))
    return int(need + extra)


def required_precision(
    pitch, margin=HP_PRECISION_MARGIN, floor=HP_MIN_PRECISION, cap=HP_PRECISION_CAP
):
    """Digits needed to resolve a pixel pitch.

    ``depth = ceil(-log10(pitch))``, so pitch 3.2e-34 → 54 digits and
    1e-100 → 120. Uses the pitch's *magnitude* (integer-part digit count),
    not the raw exponent: a division under a wide context carries trailing
    digits that inflate the exponent (1.94e-31/600 must ask for ~54 digits,
    not ~170).
    """
    p = to_decimal(pitch)
    if p <= 0 or not p.is_finite():
        return int(floor)
    t = p.as_tuple()
    scale = t.exponent + len(t.digits)
    depth = 1 - scale  # ≈ ceil(-log10(pitch))
    value = max(int(floor), depth + int(margin))
    if cap is not None:
        value = min(int(cap), value)
    return int(value)


def dd_to_decimal(hi, lo):
    """Exact Decimal value of a double-double pair (hi + lo)."""
    with localcontext(prec=exact_precision(hi, lo)) as ctx:
        return ctx.add(to_decimal(hi), to_decimal(lo))


def decimal_to_dd(value, prec=HP_MIN_PRECISION):
    """Split a Decimal into a double-double ``(hi, lo)`` pair.

    ``hi`` is the correctly rounded double and ``lo`` the remainder, so
    ``hi + lo`` is the DD number nearest ``value``.
    """
    d = to_decimal(value)
    hi = float(d)
    if not math.isfinite(hi):
        return hi, 0.0
    with localcontext(prec=exact_precision(d, hi)) as ctx:
        lo = float(ctx.subtract(d, Decimal.from_float(hi)))
    return hi, lo


def format_decimal(value, significant=30):
    """Fixed-notation string with ``significant`` significant digits.

    Same display contract as ``dd_math.dd_format`` (which is capped at 28
    digits by the default context), driven by the canonical value.
    """
    d = to_decimal(value)
    if d == 0:
        return "0"
    sign = "-" if d < 0 else ""
    with localcontext(prec=exact_precision(d, extra=40)) as ctx:
        d = ctx.add(d, Decimal(0))  # copy into the wide context
        if significant is not None and significant > 0:
            t = d.as_tuple()
            if len(t.digits) > significant:
                scale = t.exponent + len(t.digits)
                d = d.quantize(
                    Decimal(1).scaleb(scale - significant),
                    rounding="ROUND_HALF_EVEN",
                    context=ctx,
                )
        digits, exponent = d.as_tuple().digits, d.as_tuple().exponent
    body = "".join(map(str, digits))
    scale = exponent + len(digits)
    if scale > 0:
        body = body.ljust(scale, "0") if len(body) < scale else body
        intpart, frac = body[:scale], body[scale:].rstrip("0")
        return sign + intpart + ("." + frac if frac else "")
    return sign + "0." + ("0" * -scale) + body


class PreciseBounds:
    """Arbitrary-precision view rectangle: four Decimal endpoints + precision.

    Canonical viewer state. Mutations are in place; ``precision`` is the
    policy depth and only ever grows (a deep view never gives digits back).
    Operations run in contexts sized from the operands, so results are
    exact, not context-rounded.
    """

    def __init__(self, x_min, x_max, y_min, y_max, precision=None):
        self.x_min = to_decimal(x_min)
        self.x_max = to_decimal(x_max)
        self.y_min = to_decimal(y_min)
        self.y_max = to_decimal(y_max)
        if precision is None:
            precision = self.implied_precision()
        self.precision = int(max(HP_MIN_PRECISION, precision))

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_floats(cls, x_min, x_max, y_min, y_max):
        """Bounds from plain doubles (lo parts implicitly zero)."""
        return cls(x_min, x_max, y_min, y_max)

    @classmethod
    def from_dd(
        cls, x_min, x_min_lo, x_max, x_max_lo, y_min, y_min_lo, y_max, y_max_lo
    ):
        """Bounds from double-double pairs: exactly the DD information, no more.

        Digits that were never stored cannot be recovered, so a DD-derived
        view is the precision floor for legacy bookmarks.
        """
        return cls(
            dd_to_decimal(x_min, x_min_lo),
            dd_to_decimal(x_max, x_max_lo),
            dd_to_decimal(y_min, y_min_lo),
            dd_to_decimal(y_max, y_max_lo),
            precision=DD_DECIMAL_PRECISION,
        )

    @classmethod
    def from_json(cls, data):
        """Bounds from a bookmark dict: ``(bounds, legacy)``.

        HP fields win when present; a legacy bookmark is reconstructed from
        its ``hi``/``lo`` pairs and flagged, because it carries fewer digits
        than the view it describes.
        """
        keys = ("x_min_hp", "x_max_hp", "y_min_hp", "y_max_hp")
        if all(key in data for key in keys):
            precision = data.get("view_precision")
            return (
                cls(
                    data["x_min_hp"],
                    data["x_max_hp"],
                    data["y_min_hp"],
                    data["y_max_hp"],
                    precision=int(precision) if precision else None,
                ),
                False,
            )
        return (
            cls.from_dd(
                float(data.get("x_min", -2.0)),
                float(data.get("x_min_lo", 0.0)),
                float(data.get("x_max", 1.0)),
                float(data.get("x_max_lo", 0.0)),
                float(data.get("y_min", -1.5)),
                float(data.get("y_min_lo", 0.0)),
                float(data.get("y_max", 1.5)),
                float(data.get("y_max_lo", 0.0)),
            ),
            True,
        )

    def copy(self):
        return PreciseBounds(
            self.x_min, self.x_max, self.y_min, self.y_max, precision=self.precision
        )

    # ------------------------------------------------------------------
    # Contexts and precision policy
    # ------------------------------------------------------------------
    def _ctx(self, *operands, extra=16):
        """Context wide enough that operations on ``operands`` are exact.

        For **reads** only (ranges, pitch, zoom, pixel grids): comparing two
        deep endpoints needs a context wider than the policy precision, and a
        read cannot grow the state. Mutations use :meth:`_mut_ctx` instead —
        running them here would let every operation widen the endpoints, and
        the precision policy would creep upward on each zoom.
        """
        values = list(operands) or [self.x_min, self.x_max, self.y_min, self.y_max]
        wanted = exact_precision(*values, extra=extra)
        return localcontext(prec=max(self.precision, wanted))

    def _mut_ctx(self):
        """Context for a mutation: exactly the policy precision.

        Results are rounded to the policy depth, so endpoints never carry
        more digits than ``precision`` and the policy grows only with zoom
        depth (see :meth:`_op_precision`), never by itself.
        """
        return localcontext(prec=self.precision)

    def _op_precision(self, *values):
        """Digits an operation's magnitudes demand (window-scale inputs).

        A pan delta or a zoom factor is the quantity whose magnitude sets
        the depth: shifting by 1e-31 or scaling by 1e-290 needs that many
        digits plus the margin.
        """
        need = self.precision
        for value in values:
            v = abs(to_decimal(value))
            if 0 < v < 1:
                need = max(need, required_precision(v))
        return int(need)

    def implied_precision(self):
        """Digits the four endpoints' exact expansions occupy."""
        need = HP_MIN_PRECISION
        for value in (self.x_min, self.x_max, self.y_min, self.y_max):
            need = max(need, digit_span(value) + 4)
        return int(need)

    def adopt_precision(self, precision=None):
        """Raise the working precision; never lowers it."""
        if precision is None:
            precision = self.implied_precision()
        precision = int(max(HP_MIN_PRECISION, precision))
        if precision > self.precision:
            self.precision = precision
        return self.precision

    def needed_precision(self, width):
        """Digits required to resolve this view at ``width`` columns."""
        with self._ctx(self.x_min, self.x_max) as ctx:
            span = ctx.add(self.x_max, -self.x_min)
            pitch = ctx.divide(span, to_decimal(max(1, int(width))))
        return required_precision(pitch)

    def ensure_precision(self, width):
        """Grow precision for the current pitch. Returns ``(prec, capped)``."""
        needed = self.needed_precision(width)
        self.adopt_precision(needed)
        capped = HP_PRECISION_CAP is not None and needed > HP_PRECISION_CAP
        return self.precision, bool(capped and self.precision < needed)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def ranges(self):
        with self._ctx() as ctx:
            return (ctx.add(self.x_max, -self.x_min), ctx.add(self.y_max, -self.y_min))

    def range_x(self):
        return self.ranges()[0]

    def range_y(self):
        return self.ranges()[1]

    def center(self):
        two = Decimal(2)
        with self._ctx(extra=32) as ctx:
            cx = ctx.divide(ctx.add(self.x_max, self.x_min), two)
            cy = ctx.divide(ctx.add(self.y_max, self.y_min), two)
        return cx, cy

    def pitch(self, width):
        """Exact per-pixel pitch for ``width`` columns."""
        with self._ctx(self.x_min, self.x_max) as ctx:
            return ctx.divide(
                ctx.add(self.x_max, -self.x_min), to_decimal(max(1, int(width)))
            )

    def pitch_y(self, height):
        with self._ctx(self.y_min, self.y_max) as ctx:
            return ctx.divide(
                ctx.add(self.y_max, -self.y_min), to_decimal(max(1, int(height)))
            )

    def zoom(self, base_range=3.0):
        """Zoom factor as a Decimal (arbitrary depth: no float overflow)."""
        with self._ctx(extra=32) as ctx:
            rx = ctx.add(self.x_max, -self.x_min)
            if rx <= 0:
                return Decimal("Infinity")
            return ctx.divide(to_decimal(base_range), rx)

    def zoom_float(self, base_range=3.0):
        """Zoom as a float for UI and iteration formulas; ``inf`` past floats."""
        try:
            return float(self.zoom(base_range))
        except (OverflowError, ValueError):
            return math.inf

    def is_degenerate(self, width=1):
        """True if the rectangle cannot render (non-positive/non-finite range)."""
        rx, ry = self.ranges()
        if not (rx.is_finite() and ry.is_finite()):
            return True
        if rx <= 0 or ry <= 0:
            return True
        return self.pitch(max(1, int(width))) <= 0

    def contains(self, other):
        """True if this rectangle covers ``other`` (exact comparison)."""
        return (
            self.x_min <= other.x_min
            and self.x_max >= other.x_max
            and self.y_min <= other.y_min
            and self.y_max >= other.y_max
        )

    def offsets(self, x, y):
        """Signed offsets ``(x − x_min, y − y_min)`` as Decimals."""
        with self._ctx(self.x_min, self.y_min, to_decimal(x), to_decimal(y)) as ctx:
            return (
                ctx.add(to_decimal(x), -self.x_min),
                ctx.add(to_decimal(y), -self.y_min),
            )

    def offset_point(self, fx, fy):
        """Canonical point at window-scale offsets ``(fx, fy)`` from x_min/y_min.

        Used for the zoom anchor: the offset is a canvas fraction times the
        range (exact), added to the canonical left/top edge.
        """
        fx, fy = to_decimal(fx), to_decimal(fy)
        with self._ctx(fx, fy, extra=24) as ctx:
            return ctx.add(self.x_min, fx), ctx.add(self.y_min, fy)

    # ------------------------------------------------------------------
    # Mutations (canonical, in place)
    # ------------------------------------------------------------------
    def set(self, x_min, x_max, y_min, y_max, precision=None):
        self.x_min = to_decimal(x_min)
        self.x_max = to_decimal(x_max)
        self.y_min = to_decimal(y_min)
        self.y_max = to_decimal(y_max)
        if precision is None:
            precision = max(self.precision, self.implied_precision())
        self.adopt_precision(precision)
        return self

    def set_bound(self, name, value):
        """Assign one endpoint (the DD ``hi`` mirror path: lo implicitly zero)."""
        setattr(self, name, to_decimal(value))
        self.adopt_precision(max(self.precision, self.implied_precision()))
        return self

    def set_bound_dd(self, name, hi, lo):
        """Assign one endpoint from a ``(hi, lo)`` double-double pair."""
        setattr(self, name, dd_to_decimal(hi, lo))
        self.adopt_precision(max(self.precision, self.implied_precision()))
        return self

    def shift(self, dx, dy):
        """Translate by window-scale deltas (exact at any zoom)."""
        dx, dy = to_decimal(dx), to_decimal(dy)
        self.adopt_precision(self._op_precision(dx, dy))
        with self._mut_ctx() as ctx:
            self.x_min = ctx.add(self.x_min, dx)
            self.x_max = ctx.add(self.x_max, dx)
            self.y_min = ctx.add(self.y_min, dy)
            self.y_max = ctx.add(self.y_max, dy)
        return self

    def scale_around(self, cx, cy, factor):
        """Scale every endpoint around an anchor (``factor > 1`` zooms out)."""
        f, cx, cy = to_decimal(factor), to_decimal(cx), to_decimal(cy)
        self.adopt_precision(self._op_precision(f))
        with self._mut_ctx() as ctx:
            self.x_min = ctx.subtract(cx, ctx.multiply(ctx.subtract(cx, self.x_min), f))
            self.x_max = ctx.add(cx, ctx.multiply(ctx.subtract(self.x_max, cx), f))
            self.y_min = ctx.subtract(cy, ctx.multiply(ctx.subtract(cy, self.y_min), f))
            self.y_max = ctx.add(cy, ctx.multiply(ctx.subtract(self.y_max, cy), f))
        return self

    def adjust_aspect(self, width, height):
        """Square pixels: the y range becomes ``x_range * height / width``.

        Preserves the x range and the y centre — same invariant as the
        double-double version it replaces.
        """
        if width < 2 or height < 2:
            return self
        two = Decimal(2)
        with self._mut_ctx() as ctx:
            x_range = ctx.add(self.x_max, -self.x_min)
            if x_range <= 0:
                return self
            cy = ctx.divide(ctx.add(self.y_max, self.y_min), two)
            y_range = ctx.multiply(
                x_range, ctx.divide(to_decimal(height), to_decimal(width))
            )
            half = ctx.divide(y_range, two)
            self.y_min = ctx.subtract(cy, half)
            self.y_max = ctx.add(cy, half)
        return self

    def lerp(self, target, t):
        """Interpolated snapshot: ``start + (target − start) * t`` per endpoint.

        Returns a new ``PreciseBounds`` (fluid zoom walks start → target with
        this instead of stepping double-double pairs).
        """
        t = to_decimal(t)
        self.adopt_precision(max(self.precision, target.precision))
        with self._mut_ctx() as ctx:

            def one(a, b):
                return ctx.add(a, ctx.multiply(ctx.subtract(b, a), t))

            return PreciseBounds(
                one(self.x_min, target.x_min),
                one(self.x_max, target.x_max),
                one(self.y_min, target.y_min),
                one(self.y_max, target.y_max),
                precision=max(self.precision, target.precision),
            )

    def expanded(self, margin_x, margin_y):
        """Overscan snapshot: grow by ``margin`` fractions of each range per side."""
        mx, my = to_decimal(margin_x), to_decimal(margin_y)
        with self._mut_ctx() as ctx:
            dx = ctx.multiply(ctx.add(self.x_max, -self.x_min), mx)
            dy = ctx.multiply(ctx.add(self.y_max, -self.y_min), my)
            return PreciseBounds(
                ctx.subtract(self.x_min, dx),
                ctx.add(self.x_max, dx),
                ctx.subtract(self.y_min, dy),
                ctx.add(self.y_max, dy),
                precision=self.precision,
            )

    # ------------------------------------------------------------------
    # Conversions and render-facing helpers
    # ------------------------------------------------------------------
    def to_dd(self):
        """Mirror layout ``(x_min, x_min_lo, x_max, x_max_lo, y_min, …)``."""
        pairs = []
        for value in (self.x_min, self.x_max, self.y_min, self.y_max):
            pairs.extend(decimal_to_dd(value, prec=self.precision))
        return tuple(pairs)

    def to_floats(self):
        """``(x_min, x_max, y_min, y_max)`` rounded to doubles."""
        return (
            float(self.x_min),
            float(self.x_max),
            float(self.y_min),
            float(self.y_max),
        )

    def to_strings(self):
        """Exact decimal strings (loss-free, picklable) for the kernels."""
        return (str(self.x_min), str(self.x_max), str(self.y_min), str(self.y_max))

    def center_strings(self):
        cx, cy = self.center()
        return str(cx), str(cy)

    def pitch_strings(self, width, height):
        """``(dx, dy)`` per-pixel spacings as exact strings."""
        with self._ctx(extra=40) as ctx:
            w_dec = to_decimal(max(1, width))
            h_dec = to_decimal(max(1, height))
            dx = ctx.divide(ctx.add(self.x_max, -self.x_min), w_dec)
            dy = ctx.divide(ctx.add(self.y_max, -self.y_min), h_dec)
        return str(dx), str(dy)

    def section_start_x(self, offset, width):
        """X coordinate of column ``offset`` in a ``width``-wide frame (string).

        Perturbation sections need their own left edge; with the DD state this
        was a ``dd_add_d`` that quantizes at deep pitch. Here it is exact.
        """
        with self._ctx(extra=40) as ctx:
            dx = ctx.divide(ctx.add(self.x_max, -self.x_min), to_decimal(max(1, width)))
            return str(ctx.add(self.x_min, ctx.multiply(dx, to_decimal(int(offset)))))

    def _pixel_deltas(self, base, span, count):
        """float64 ``(base + (k+0.5)·pitch) − (base + span/2·pitch)`` per pixel.

        Pixels and centre are computed exactly in one context and only the
        final delta is rounded to double — the step that must never go
        through double-double arithmetic.
        """
        half = Decimal("0.5")
        n = max(1, int(count))
        with self._ctx(base, span, extra=40) as ctx:
            pitch = ctx.divide(to_decimal(span), to_decimal(n))
            half_n = ctx.divide(to_decimal(n), Decimal(2))
            centre = ctx.add(base, ctx.multiply(pitch, half_n))
            out = np.empty(n, dtype=np.float64)
            for k in range(n):
                pixel = ctx.add(base, ctx.multiply(ctx.add(to_decimal(k), half), pitch))
                out[k] = float(ctx.subtract(pixel, centre))
        return out

    def pixel_deltas_x(self, width):
        """float64 per-column delta from the view centre."""
        return self._pixel_deltas(self.x_min, self.range_x(), width)

    def pixel_deltas_y(self, height):
        """float64 per-row delta from the view centre."""
        return self._pixel_deltas(self.y_min, self.range_y(), height)

    def to_json_fields(self):
        """Bookmark fields: HP strings + precision + legacy hi/lo parts."""
        fields = {
            "x_min_hp": str(self.x_min),
            "x_max_hp": str(self.x_max),
            "y_min_hp": str(self.y_min),
            "y_max_hp": str(self.y_max),
            "view_precision": int(self.precision),
        }
        dd = self.to_dd()
        x_min, x_min_lo, x_max, x_max_lo = dd[0], dd[1], dd[2], dd[3]
        y_min, y_min_lo, y_max, y_max_lo = dd[4], dd[5], dd[6], dd[7]
        fields.update(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            x_min_lo=x_min_lo,
            x_max_lo=x_max_lo,
            y_min_lo=y_min_lo,
            y_max_lo=y_max_lo,
        )
        return fields

    def coord_strings(self, significant=None):
        """Status-bar coordinates at dynamic precision."""
        return {
            "x_min": format_decimal(self.x_min, significant),
            "x_max": format_decimal(self.x_max, significant),
            "y_min": format_decimal(self.y_min, significant),
            "y_max": format_decimal(self.y_max, significant),
        }

    def __repr__(self):
        return (
            f"PreciseBounds(x_min={self.x_min}, x_max={self.x_max}, "
            f"y_min={self.y_min}, y_max={self.y_max}, prec={self.precision})"
        )


__all__ = [
    "DD_DECIMAL_PRECISION",
    "HP_MIN_PRECISION",
    "HP_PRECISION_CAP",
    "HP_PRECISION_MARGIN",
    "PreciseBounds",
    "dd_to_decimal",
    "decimal_to_dd",
    "digit_span",
    "exact_precision",
    "format_decimal",
    "required_precision",
    "to_decimal",
]
