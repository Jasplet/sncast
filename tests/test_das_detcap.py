"""
Test functions in model_detection_capability.py which
implement detection capability modelling for DAS.

Author: Joseph Asplet
Email: joseph.asplet@earth.ox.ac.uk
Affiliation: Department of Earth Sciences, University of Oxford

Copyright (C) 2025 Joseph Asplet, University of Oxford

"""

import numpy as np
import pandas as pd
import pygc

# import pytest
from unittest.mock import patch

# import functions to test
from sncast.model_detection_capability import get_das_noise_levels
from sncast.model_detection_capability import calc_min_ml_at_gridpoint_das


# test get_das_noise_levels


def test_get_das_noise_levels_normal():
    """Tests expected behaviour of get_das_noise_levels function."""

    simple_noise = np.ones(50) * 2.0
    # make a df with geometry and noise
    wind_len = 4
    test_noise = get_das_noise_levels(simple_noise, wind_len, model_stacking=True)
    # test shape is correct
    assert isinstance(test_noise, np.ndarray)
    assert test_noise.shape == simple_noise.shape
    # test noise levels away from edges are the same as input
    assert np.all(test_noise[wind_len // 2 : -wind_len // 2] == 2.0 / np.sqrt(wind_len))


def test_get_das_noise_levels_basic():
    """Tests basic functionality of get_das_noise_levels function."""
    simple_noise = np.ones(50) * 2.0
    wind_len = 10
    test_noise = get_das_noise_levels(simple_noise, wind_len, model_stacking=False)
    # test shape is correct
    assert isinstance(test_noise, np.ndarray)
    assert test_noise.shape == simple_noise.shape
    # test noise levels away from edges are the same as input
    assert np.all(test_noise[wind_len // 2 : -wind_len // 2] == 2.0)


def test_get_das_noise_levels_stacking_edges_even():
    """Tests edge handling of get_das_noise_levels function with stacking."""
    simple_noise = np.ones(50) * 4e-8
    wind_len = 10
    test_noise = get_das_noise_levels(simple_noise, wind_len, model_stacking=True)
    # test shape is correct
    assert isinstance(test_noise, np.ndarray)
    assert test_noise.shape == simple_noise.shape
    # test start edge
    expect_noise_start = simple_noise[: wind_len // 2] / np.sqrt(
        np.arange(wind_len // 2, wind_len)
    )
    expect_noise_end = simple_noise[-wind_len // 2 :] / np.sqrt(
        np.arange(wind_len, wind_len // 2, -1)
    )
    assert np.allclose(test_noise[: wind_len // 2], expect_noise_start)
    # test end edge
    assert np.allclose(test_noise[-wind_len // 2 :], expect_noise_end)


def test_get_das_noise_levels_stacking_edges_odd():
    """Tests edge handling of get_das_noise_levels function with stacking."""
    simple_noise = np.ones(10)
    wind_len = 3
    half_win = wind_len // 2
    test_noise = get_das_noise_levels(simple_noise, wind_len, model_stacking=True)
    # test shape is correct
    assert isinstance(test_noise, np.ndarray)
    assert test_noise.shape == simple_noise.shape
    # test start edge
    expect_noise = simple_noise[: half_win + 1] / np.sqrt(
        np.arange(half_win + 1, wind_len + 1)
    )
    assert np.allclose(test_noise[: half_win + 1], expect_noise)
    # test end edge
    assert np.allclose(test_noise[-(half_win + 1) :], expect_noise[::-1])


#   test calc_min_ml_at_gridpoint_das
@patch("sncast.model_detection_capability.get_das_noise_levels")
@patch("sncast.model_detection_capability.calc_local_magnitude")
def test_calc_min_ml_at_gridpoint_das(
    mock_calc_local_magnitude, mock_get_das_noise_levels
):
    """Tests the calc_min_ml_at_gridpoint_das function."""
    test_mags = np.array([0.5, 0.7, 1.2, 1.3])
    test_noise = np.array([6e-9, 6e-9, 6e-9, 6e-9])
    mock_calc_local_magnitude.return_value = test_mags
    mock_get_das_noise_levels.return_value = test_noise
    dummy_fibre = pd.read_csv("tests/data/das_dummy_data.csv")
    lat = 1
    lon = 50
    distances_km = (
        pygc.great_distance(
            start_latitude=dummy_fibre["latitude"].values,
            end_latitude=lat,
            start_longitude=dummy_fibre["longitude"].values,
            end_longitude=lon,
        )["distance"]
        * 1e-3
    )
    print(distances_km)
    result = calc_min_ml_at_gridpoint_das(
        lon=lon,
        lat=lat,
        fibre=dummy_fibre,
        detection_length_m=100,
        gauge_length_m=10,
        model_stacking=True,
        foc_depth=0,
        snr=1,
        region="UK",
        method="ML",
        mag_min=-2,
        mag_delta=0.1,
    )
    minmag = min(test_mags)
    assert result == minmag
    mock_calc_local_magnitude.assert_called_once()
    mock_get_das_noise_levels.assert_called_once()
