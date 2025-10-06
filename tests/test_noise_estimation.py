import numpy as np
import pytest
from unittest import patch

from sncast.noise_estimation import get_freq_range_from_centre, get_f0_octaves_from_f1f2


@pytest.parametrize(
    "f0, n, expected_f1, expected_f2",
    [
        (2.0, 2.0),
        (0.5, 2.0, 0.25, 1.0),
        (1.0, 1.0, np.power(2, -0.5), np.power(2, 0.5)),
        (100, 2.0, 50.0, 200.0),
        (100, 0.4, 100 * np.power(2, -0.2), 100 * np.power(2, 0.2)),
    ],
)
def test_get_freq_range_from_centre(f0, n, expected_f1, expected_f2):

    f1, f2 = get_freq_range_from_centre(f0, n)
    assert np.isclose(f1, expected_f1)
    assert np.isclose(f2, expected_f2)
    assert f2 > f1


@pytest.mark.parametrize(
    "f0, n",
    [(2.0, 2.0), (0.5, 2.0), (1.0, 0.2), (100, 3.2), (100, 0.4), (2000, 4)],
)
def test_get_freq_range_math(f0, n):
    """Test that the maths is consistent between the two functions"""
    f1, f2 = get_freq_range_from_centre(f0, n)
    f0_out = np.sqrt(f1 * f2)
    ratio = f2 / f1
    expected_n = np.log2(ratio)
    assert np.isclose(f0, f0_out)
    assert np.isclose(n, expected_n)


def test_get_freq_range_from_centre_invalid_n():
    with pytest.raises(ValueError, match="n must be a non-zero number"):
        get_freq_range_from_centre(1.0, 0)
    with pytest.raises(ValueError, match="n must be a non-zero number"):
        get_freq_range_from_centre(1.0, "two")
    with pytest.raises(ValueError, match="n must be a non-zero number"):
        get_freq_range_from_centre(1.0, -3)


def test_get_freq_range_from_centre_invalid_f0():
    with pytest.raises(ValueError, match="f0 must be a positive number"):
        get_freq_range_from_centre(0, 1.0)
    with pytest.raises(ValueError, match="f0 must be a positive number"):
        get_freq_range_from_centre("five", 1.0)
    with pytest.raises(ValueError, match="f0 must be a positive number"):
        get_freq_range_from_centre(-1.0, 1.0)
