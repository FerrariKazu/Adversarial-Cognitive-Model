"""Unit-aware epsilon tests, replacing the assertion that checked nothing.

The old guard read

    assert abs(eps_norm * MEAN_STD * 255 - round(eps_norm * MEAN_STD * 255)) > 1e-6 \
        or eps_norm in {0.0, 0.031, 0.062, 0.094}, "eps_norm looks like a pixel fraction"

which passes for every value in the allow-list *and* for almost every other value,
because the first disjunct is true whenever the product is not an integer.  It was
removed rather than repaired: a membership list of legal numbers cannot establish
which unit a number is in.

These tests check the four units the project actually uses and the conversion in
`training/curriculum_gen2.py`, which is the implementation the trainer and the
attacks call:

    pixel intensity, [0, 255]     n            e.g. 5.417 / 255
    pixel value, [0, 1]           e = n / 255  = eps_norm * mean(std)
    normalised model input        eps_norm     e.g. 0.094
    the /255 reporting convention n / 255 written as "n/255"

The failure mode this guards against is a radius expressed as a pixel fraction
(2/255, 4/255, 8/255) being handed to an interface that expects normalised units.
At 8/255 the mistake is a factor of 1.48 in the radius; at 2/255 it is a factor of
5.9 smaller and silently reduces the attack to almost nothing.
"""

from __future__ import annotations

import math

import pytest

from training.curriculum_gen2 import (IMAGENET_MEAN, IMAGENET_STD, MEAN_STD,
                                      norm_to_pixel_eps)

#: The radii the project registers as trained or evaluated conditions.
NORM_RADII = (0.0, 0.031, 0.062, 0.094)

#: Radii that exist only in the /255 reporting convention.
PIXEL_FRACTIONS = (2 / 255, 4 / 255, 8 / 255)


def test_mean_std_is_the_mean_of_the_channel_deviations() -> None:
    assert MEAN_STD == pytest.approx(sum(IMAGENET_STD) / 3)
    assert MEAN_STD == pytest.approx(0.226, abs=1e-3)


@pytest.mark.parametrize("eps_norm", NORM_RADII)
def test_normalised_radius_round_trips_through_the_implementation(eps_norm: float) -> None:
    """eps_norm -> [0,1] pixel -> eps_norm must be exact for a legitimate radius."""
    pixel = norm_to_pixel_eps(eps_norm)
    assert pixel == pytest.approx(eps_norm * MEAN_STD)
    # the inverse direction is a division, not a second call to the forward map
    assert pixel / MEAN_STD == pytest.approx(eps_norm)
    assert norm_to_pixel_eps(pixel / MEAN_STD) == pytest.approx(pixel)


def test_the_trained_radius_in_all_three_units() -> None:
    """0.094 normalised is 0.0212 in [0,1] and 5.42/255, not 8/255 or 0.094/255."""
    pixel = norm_to_pixel_eps(0.094)
    assert pixel == pytest.approx(0.02124, abs=1e-5)
    assert pixel * 255 == pytest.approx(5.42, abs=0.02)
    assert pixel * 255 != pytest.approx(8.0, abs=0.1)


@pytest.mark.parametrize("fraction", PIXEL_FRACTIONS)
def test_pixel_fractions_are_not_normalised_radii(fraction: float) -> None:
    """A pixel fraction is not in [0.03, 0.14]; conflating the two mis-sizes the attack.

    The comparison is made in the unit the implementation consumes: a legitimate
    normalised radius converts to at most 0.095 in [0,1] pixel value, while 8/255
    passed straight through the normalised interface is 0.0314 -- smaller than the
    trained radius by a factor of three, and smaller than the pixel-space radius it
    was meant to express by the factor 1 / MEAN_STD.
    """
    as_normalised = fraction  # the bug: a pixel fraction where eps_norm is expected
    correct_normalised = fraction / MEAN_STD
    assert correct_normalised == pytest.approx(fraction * 4.425, rel=1e-3)
    assert as_normalised < 0.094, (
        f"{fraction:.6f} used as a normalised radius is {as_normalised / 0.094:.1%} of "
        "the trained radius")
    assert not math.isclose(as_normalised, correct_normalised, rel_tol=0.1)


def test_the_valid_normalised_range_is_what_the_clamp_must_contain() -> None:
    """The [-4, 4] clamp is only safe because it contains the true valid range."""
    lows = [(0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
    highs = [(1 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
    assert min(lows) == pytest.approx(-2.118, abs=1e-3)
    assert max(highs) == pytest.approx(2.640, abs=1e-3)
    assert all(-4.0 < lo and hi < 4.0 for lo, hi in zip(lows, highs))
    # a radius of 0.094 is small next to the valid range, so the intersection is
    # the epsilon box for every channel
    assert max(abs(min(lows)), abs(max(highs))) > 20 * 0.094


def test_an_eight_255_radius_is_a_different_radius() -> None:
    """8/255 normalised is 0.1388, and the book must never print it as 0.094."""
    assert (8 / 255) / MEAN_STD == pytest.approx(0.1388, abs=5e-4)
    assert (8 / 255) / MEAN_STD != pytest.approx(0.094, abs=1e-3)
    assert norm_to_pixel_eps((8 / 255) / MEAN_STD) == pytest.approx(8 / 255)
