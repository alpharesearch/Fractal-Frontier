"""Tests for the arbitrary-precision view state (hp_view.PreciseBounds)."""

import json
import math
from contextlib import contextmanager
from decimal import Decimal, localcontext

import pytest

from dd_math import DD_EPS, dd_add_d, dd_sub, dd_to_double
from hp_view import (HP_MIN_PRECISION, PreciseBounds, dd_to_decimal,
                     decimal_to_dd, digit_span, exact_precision,
                     format_decimal, required_precision, to_decimal)

# Bookmark 9 as stored (the deep-zoom limit case): the hi parts are equal
# doubles and the entire range lives in the double-double lo parts.
BM9 = {
    "x_min": -0.7454642792047881,
    "x_max": -0.7454642792047881,
    "y_min": 0.10700158801025167,
    "y_max": 0.10700158801025167,
    "x_min_lo": -9.789304372927869e-18,
    "x_max_lo": -9.789304372927675e-18,
    "y_min_lo": 2.798855395978615e-18,
    "y_max_lo": 2.7988553959787175e-18,
}


def bm9_bounds():
    return PreciseBounds.from_dd(
        BM9["x_min"],
        BM9["x_min_lo"],
        BM9["x_max"],
        BM9["x_max_lo"],
        BM9["y_min"],
        BM9["y_min_lo"],
        BM9["y_max"],
        BM9["y_max_lo"],
    )


@contextmanager
def wide(extra=40):
    """Context wide enough for exact Decimal math in the tests.

    Deliberate duplication of hp_view's rule: tests must not do deep Decimal
    arithmetic in the default prec=28 context, which silently rounds it.
    """
    operands = []
    yield_ctx = localcontext(prec=600)
    with yield_ctx as ctx:
        yield ctx


def rel_diff(a, b):
    """Relative difference of two Decimals, computed in a wide context."""
    a, b = to_decimal(a), to_decimal(b)
    with localcontext(prec=max(exact_precision(a, b, extra=40), 200)) as ctx:
        if b == 0:
            return 0.0 if a == 0 else math.inf
        return abs(float(ctx.divide(ctx.subtract(a, b), b)))


# ----------------------------------------------------------------------
# Precision policy
# ----------------------------------------------------------------------
def test_required_precision_matches_depth():
    assert required_precision(to_decimal("3.2e-34")) == 54
    assert required_precision(to_decimal("1e-100")) == 120
    assert required_precision(1.0) == HP_MIN_PRECISION
    assert required_precision(0.0) == HP_MIN_PRECISION
    # Magnitude, not the raw exponent of a wide-context division: bm9's pitch
    # at 600 px is 3.2e-34 → ~54 digits, not ~170.
    pitch = bm9_bounds().pitch(600)
    assert required_precision(pitch) == math.ceil(-math.log10(float(pitch))) + 20


def test_precision_is_monotonic():
    bounds = PreciseBounds.from_floats(-2.0, 1.0, -1.5, 1.5)
    first = bounds.ensure_precision(1000)[0]
    bounds.scale_around(*bounds.center(), 1e-40)
    second = bounds.ensure_precision(1000)[0]
    assert second > first
    bounds.scale_around(*bounds.center(), 1e40)  # zoom back out
    assert bounds.ensure_precision(1000)[0] >= second


def test_implied_precision_covers_the_exact_expansion():
    """A double-double pair expands to >100 exact digits; the state must carry them."""
    bounds = bm9_bounds()
    assert bounds.implied_precision() > 100
    assert bounds.precision >= bounds.implied_precision()
    assert digit_span(bounds.x_min) > 100


# ----------------------------------------------------------------------
# Arithmetic
# ----------------------------------------------------------------------
def test_shift_is_exact_at_deep_pitch():
    """600 sub-ULP pans accumulate to exactly the full width."""
    bounds = bm9_bounds()
    pitch = bounds.pitch(600)
    start = bounds.x_min
    for _ in range(600):
        bounds.shift(pitch, 0)
    with bounds._ctx(start, bounds.x_min) as ctx:
        moved = ctx.subtract(bounds.x_min, start)
    assert rel_diff(moved, ctx.multiply(pitch, 600)) < 1e-30
    assert rel_diff(bounds.range_x(), bm9_bounds().range_x()) < 1e-30


def test_scale_around_anchor_is_invariant():
    bounds = bm9_bounds()
    cx, cy = bounds.center()
    base = bm9_bounds()
    assert (cx, cy) == base.center()
    bounds.scale_around(cx, cy, 0.5)
    ncx, ncy = bounds.center()
    assert ncx == cx and ncy == cy
    with wide() as ctx:
        half_range = ctx.multiply(base.range_x(), to_decimal(0.5))
    assert rel_diff(bounds.range_x(), half_range) < 1e-30


def test_corner_anchor_stays_put():
    """Mouse-anchored zoom must pin the anchor exactly."""
    bounds = PreciseBounds.from_floats(-2.0, 1.0, -1.5, 1.5)
    bounds.scale_around(bounds.x_min, bounds.y_min, 0.5)
    assert bounds.x_min == to_decimal(-2.0) and bounds.y_min == to_decimal(-1.5)
    assert bounds.range_x() == to_decimal(1.5)


def test_adjust_aspect_squares_pixels():
    bounds = bm9_bounds()
    bounds.adjust_aspect(1200, 400)
    rx, ry = bounds.ranges()
    assert rel_diff(ry / rx, Decimal(400) / Decimal(1200)) < 1e-30
    cx, cy = bounds.center()
    bcx, bcy = bm9_bounds().center()
    assert cx == bcx and rel_diff(cy, bcy) < 1e-30


def test_lerp_endpoints_and_monotonicity():
    start = PreciseBounds.from_floats(-2.0, 1.0, -1.5, 1.5)
    target = start.copy().scale_around(to_decimal(-0.5), to_decimal(0.0), 0.5)
    mid = start.lerp(target, 0.5)
    assert (
        rel_diff(mid.range_x(), (start.range_x() + target.range_x()) / to_decimal(2))
        < 1e-12
    )
    assert mid.precision >= max(start.precision, target.precision)
    assert start.lerp(target, 0).x_min == start.x_min
    assert start.lerp(target, 1).x_min == target.x_min


def test_expanded_overscan_covers_view():
    bounds = bm9_bounds()
    overscan = bounds.expanded(1.0, 1.0)
    assert overscan.contains(bounds)
    assert rel_diff(overscan.range_x(), bounds.range_x() * 3) < 1e-25


def test_section_start_is_the_exact_column_edge():
    """Perturb section stitching must land exactly on the column boundary."""
    bounds = bm9_bounds()
    width = 600
    for offset in (0, 137, 300, 599):
        edge = to_decimal(bounds.section_start_x(offset, width))
        with wide() as ctx:
            expected = ctx.add(
                bounds.x_min, ctx.multiply(bounds.pitch(width), to_decimal(offset))
            )
        assert rel_diff(edge, expected) < 1e-25


# ----------------------------------------------------------------------
# DD interoperability
# ----------------------------------------------------------------------
def test_hp_to_dd_roundtrip_is_exact_for_dd_values():
    """HP → DD → HP reproduces the pair bit-for-bit (no drift at the boundary)."""
    bounds = bm9_bounds()
    mirror = bounds.to_dd()
    assert len(mirror) == 8
    back = PreciseBounds.from_dd(*mirror)
    assert back.to_dd() == mirror
    assert back.to_strings() == bounds.to_strings()


def test_dd_pair_conversion_matches_helpers():
    hi, lo = -0.19962502420084796, -1.0514899906736287e-17
    value = dd_to_decimal(hi, lo)
    assert decimal_to_dd(value) == (hi, lo)
    # The default Decimal context (prec=28) truncates a DD sum; the helper
    # must not, or the deep digits are silently lost.
    assert len(str(value).lstrip("-").replace(".", "")) > 28


def test_view_state_beats_dd_column_resolution():
    """The measurement that defines this change.

    At bookmark 9's pitch the double-double coordinate state resolves only
    ~127 of 600 columns; the HP state resolves all 600.
    """
    bounds = bm9_bounds()
    width = 600
    assert len(set(bounds.pixel_deltas_x(width))) == width

    # The same geometry through the double-double helpers the legacy kernels
    # use (dd_add_d): quantized, as the README's "practical ceiling" says.
    hi, lo = bounds.to_dd()[:2]
    dx_f = float(bounds.range_x()) / width
    dd_cols = {dd_add_d(hi, lo, (j + 0.5) * dx_f) for j in range(width)}
    assert len(dd_cols) < width // 4, f"DD resolved {len(dd_cols)} of {width}"


def test_dd_eps_stays_the_backend_threshold():
    """DD_EPS selects the backend band; it no longer bounds the stored view."""
    bounds = bm9_bounds()
    assert bounds.pitch(1200) < to_decimal(DD_EPS)
    # A view far past DD_EPS keeps a positive, resolving pitch and real digits.
    deep = PreciseBounds.from_floats(-0.745, -0.744999, 0.107, 0.108)
    deep.scale_around(*deep.center(), to_decimal("1e-60"))
    prec, capped = deep.ensure_precision(1200)
    assert prec > 80 and not capped
    assert deep.pitch(1200) > 0
    assert len(set(deep.pixel_deltas_x(1200))) == 1200


# ----------------------------------------------------------------------
# Serialization / bookmarks
# ----------------------------------------------------------------------
def test_json_roundtrip_is_lossless():
    bounds = bm9_bounds()
    fields = bounds.to_json_fields()
    restored, legacy = PreciseBounds.from_json(json.loads(json.dumps(fields)))
    assert not legacy
    assert restored.to_strings() == bounds.to_strings()
    assert restored.precision == bounds.precision
    assert restored.to_dd() == bounds.to_dd()  # legacy fields stay in sync


def test_legacy_bookmark_load_preserves_dd_information_only():
    bounds, legacy = PreciseBounds.from_json(BM9)
    assert legacy is True
    # Exactly the stored DD value: hi + lo, with no invented digits...
    assert bounds.x_min == dd_to_decimal(BM9["x_min"], BM9["x_min_lo"])
    with wide() as ctx:
        stored_range = ctx.subtract(
            dd_to_decimal(BM9["x_max"], BM9["x_max_lo"]),
            dd_to_decimal(BM9["x_min"], BM9["x_min_lo"]),
        )
    assert rel_diff(bounds.range_x(), stored_range) < 1e-25
    # ...and that stored range is coarser than a 600 px pitch, so the view
    # cannot resolve more columns than it recorded.
    with wide() as ctx:
        assert bounds.pitch(600) < ctx.divide(bounds.range_x(), to_decimal(100))
    fields = bounds.to_json_fields()
    assert fields["x_min"] == BM9["x_min"]
    assert fields["x_min_lo"] == pytest.approx(BM9["x_min_lo"])


def test_from_json_defaults_to_the_classic_view():
    bounds, legacy = PreciseBounds.from_json({})
    assert legacy is True
    assert bounds.to_floats() == (-2.0, 1.0, -1.5, 1.5)


# ----------------------------------------------------------------------
# Display and queries
# ----------------------------------------------------------------------
def test_zoom_survives_beyond_float_range():
    bounds = bm9_bounds()
    zoom = bounds.zoom(3.0)
    assert zoom > to_decimal("1e30")
    assert bounds.zoom_float(3.0) == pytest.approx(float(zoom), rel=1e-9)
    deep = bounds.copy()
    deep.scale_around(*deep.center(), to_decimal("1e-290"))
    assert deep.zoom(3.0) > to_decimal("1e320")
    assert deep.zoom_float(3.0) == math.inf


def test_format_decimal_shows_the_deep_digits():
    """Binary-exact display: the lo part's digits are real, past digit 17."""
    bounds = bm9_bounds()
    coords = bounds.coord_strings(significant=40)
    # Digits past 17 are the stored lo part's real contribution; the DD pair
    # carries meaningful digits to ~33, and the 40-digit form is the exact
    # sum rounded to 40 significant digits.
    assert coords["x_min"].startswith("-0.7454642792047880926114239499589857569719")
    assert len(format_decimal(bounds.x_min, 60)) > 50
    assert format_decimal(0) == "0"


def test_pixel_deltas_are_centred_and_ordered():
    bounds = bm9_bounds()
    deltas = bounds.pixel_deltas_x(600)
    assert deltas.shape == (600,)
    assert deltas[0] < 0 < deltas[-1]
    assert abs(deltas[-1] + deltas[0]) < 1e-40
    step = float(deltas[1] - deltas[0])
    assert step == pytest.approx(float(bounds.pitch(600)), rel=1e-12)


def test_is_degenerate():
    assert PreciseBounds.from_floats(1.0, 1.0, -1.0, 1.0).is_degenerate()
    assert PreciseBounds.from_floats(-1.0, 1.0, 0.0, 0.0).is_degenerate()
    assert not bm9_bounds().is_degenerate(600)


def test_offsets_match_dd_sub():
    """HP offsets agree with what dd_sub reports for the same geometry."""
    bounds = bm9_bounds()
    other = bounds.copy().shift(to_decimal("1e-32"), 0)
    hp, _ = other.offsets(bounds.x_min, bounds.y_min)
    mirror = bounds.to_dd()
    other_mirror = other.to_dd()
    dd = dd_to_double(*dd_sub(other_mirror[0], other_mirror[1], mirror[0], mirror[1]))
    assert float(hp) == pytest.approx(dd, rel=1e-6)
