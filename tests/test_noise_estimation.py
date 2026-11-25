import re

import numpy as np
import pytest

# from unittest import patch
from sncast.noise_estimation import (
    get_f0_octaves_from_f1f2,
    get_freq_range_from_centre,
    psd_db_convert,
    psd_db_to_displacement_amplitude,
    psd_db_to_velocity,
)

# from sncast.noise_estimation import make_noise_estimate_for_ppsds


@pytest.mark.parametrize(
    'f0, n, expected_f1, expected_f2',
    [
        (2.0, 2.0, 1.0, 4.0),
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
    'f0, n',
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
    with pytest.raises(ValueError, match=re.escape('n must be greater than 0')):
        get_freq_range_from_centre(1.0, 0)
    with pytest.raises(ValueError, match=re.escape('n must be greater than 0')):
        get_freq_range_from_centre(1.0, -3)
    with pytest.raises(
        ValueError, match=re.escape('n must be a positive int or float')
    ):
        get_freq_range_from_centre(1.0, 'two')


def test_get_freq_range_from_centre_invalid_f0():
    with pytest.raises(ValueError, match='f0 must be greater than 0'):
        get_freq_range_from_centre(0, 1.0)
    with pytest.raises(ValueError, match='f0 must be greater than 0'):
        get_freq_range_from_centre(-1.0, 1.0)
    with pytest.raises(ValueError, match='f0 must be a positive int or float'):
        get_freq_range_from_centre('five', 1.0)


# test get f0_octaves_from_f1f2
@pytest.mark.parametrize(
    'f1, f2, expected_f0, expected_n',
    [
        (1.0, 4.0, 2.0, 2.0),
        (10, 20, np.sqrt(200), 1.0),
        (0.25, 1.0, 0.5, 2.0),
        (50, 200, 100, 2.0),
    ],
)
def test_get_f0_octaves_from_f1f2(f1, f2, expected_f0, expected_n):
    f0, n = get_f0_octaves_from_f1f2(f1, f2)
    assert np.isclose(f0, expected_f0)
    assert np.isclose(n, expected_n)


# test conversion db to displacement amplitude


@pytest.mark.parametrize(
    'psd_in_db, expected_psd',
    [
        (-120, 1e-12),
        (-110, 1e-11),
        (-140, 1e-14),
        (0, 1.0),
        (20, 100.0),
    ],
)
def test_psd_db_convert(psd_in_db, expected_psd):
    psd = psd_db_convert(psd_in_db)
    assert np.isclose(psd, expected_psd)


# test psd_db_to_displacement_amplitude


@pytest.mark.parametrize(
    'psd_db, f1, f2',
    [
        (-120, 0.5, 2),
        (-110, 0.5, 2),
        (-140, 0.5, 2),
    ],
)
def test_psd_db_to_displacement_amplitude(psd_db, f1, f2):
    displacement = psd_db_to_displacement_amplitude(psd_db, f1=f1, f2=f2)
    assert isinstance(displacement, float)
    assert displacement > 0


def test_psd_db_to_displacement_amplitude_values():
    psd_db = -120
    # two octaves around 4 Hz
    f1 = 2
    f2 = 8
    psd_acc = psd_db_convert(psd_db)
    f0 = np.sqrt(f1 * f2)
    expected_displacement = (
        (3.75) / ((2 * np.pi * f0) ** 2) * np.sqrt(psd_acc * (f2 - f1))
    )
    displacement = psd_db_to_displacement_amplitude(psd_db, f1=f1, f2=f2)
    assert np.isclose(displacement, expected_displacement)


# test psd_db_to_velocity


@pytest.mark.parametrize(
    'psd_db, f1, f2',
    [
        (-120, 0.5, 2),
        (-110, 0.5, 2),
        (-140, 0.5, 2),
    ],
)
def test_psd_db_to_velocity(psd_db, f1, f2):
    velocity = psd_db_to_velocity(psd_db, f1=f1, f2=f2)
    assert isinstance(velocity, float)
    assert velocity > 0


# test make_noise_estimates_for_ppsds

# def test_make_noise_estimate_for_ppsds():
