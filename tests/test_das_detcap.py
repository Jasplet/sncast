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
import pytest
from unittest.mock import patch

# import functions to test
from sncast.model_detection_capability import read_das_noise_data
from sncast.model_detection_capability import get_das_noise_levels
from sncast.model_detection_capability import calc_min_ml_at_gridpoint_das


def test_read_das_noise_data_from_str():
    """Tests reading DAS noise data from a CSV file path."""
    dummy_str = "tests/data/das_dummy_data.csv"
    # read dummy df
    dummy_df = pd.read_csv(dummy_str)
    # read using function
    out_df = read_das_noise_data(dummy_str)
    # check they are the same
    assert out_df.equals(dummy_df)
    # check columns
    assert all(
        col in out_df.columns
        for col in [
            "channel_index",
            "fiber_length_m",
            "longitude",
            "latitude",
            "noise_m",
            "elevation_km",
        ]
    )


def test_read_das_noise_data_from_df():
    """Tests reading DAS noise data from a DataFrame."""
    # create dummy df
    dummy_df = pd.read_csv("tests/data/das_dummy_data.csv")
    # read using function
    out_df = read_das_noise_data(dummy_df)
    # check they are the same
    assert out_df.equals(dummy_df)
    # check columns
    assert all(
        col in out_df.columns
        for col in [
            "channel_index",
            "fiber_length_m",
            "longitude",
            "latitude",
            "noise_m",
            "elevation_km",
        ]
    )


def test_read_das_noise_data_missing_columns():
    """Tests that an error is raised if required columns are missing."""
    # create dummy df with missing columns
    dummy_df = pd.DataFrame(
        {
            "channel_index": [10010, 10020],
            "fiber_length_m": [1000, 2000],
            "longitude": [0.01, 0.01],
            # 'latitude' column is missing
            "noise_m": [1e-8, 2e-9],
            "elevation_km": [0.0, 10.0],
        }
    )
    with pytest.raises(ValueError):
        read_das_noise_data(dummy_df)


def test_read_das_noise_data_no_elevation():
    """Tests that the function works if elevation_km column is missing."""
    # elev is optional, so test without it too
    dummy_no_elev_df = pd.DataFrame(
        {
            "channel_index": [10010, 10020],
            "fiber_length_m": [1000, 2000],
            "longitude": [0.01, 0.01],
            "latitude": [50.01, 50.02],
            "noise_m": [1e-8, 2e-9],
            # 'elevation_km' column is missing
        }
    )
    # This should not raise an error
    out_df = read_das_noise_data(dummy_no_elev_df)
    # function should add elevation_km column of 0s
    assert "elevation_km" in out_df.columns
    assert np.all(out_df["elevation_km"].values == 0.0)


def test_read_das_noise_data_empty_dataframe():
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        read_das_noise_data(empty_df)


## test get_das_noise_levels


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
@patch("sncast.model_detection_capability.calc_local_magnitude")
@patch("sncast.model_detection_capability.get_das_noise_levels")
def test_calc_min_ml_at_gridpoint_das(
    mock_calc_local_magnitude, mock_get_das_noise_levels
):
    """Tests the calc_min_ml_at_gridpoint_das function."""
    mock_calc_local_magnitude.return_value = np.array([0.5, 0.7, 1.2, 1.3])
    mock_get_das_noise_levels.return_value = np.array([0.5, 0.7, 1.2, 1.3]) * 0.2
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
    result = calc_min_ml_at_gridpoint_das(
        dummy_fibre,
        100,
        lon,
        lat,
        foc_depth=0,
        snr=1,
        region="UK",
        method="ML",
        mag_min=-2,
        mag_delta=0.1,
        gauge_len=10,
    )
    print(np.min(mock_calc_local_magnitude.return_value))
    assert result == np.min(mock_calc_local_magnitude.return_value)
    mock_calc_local_magnitude.assert_called_once_with(
        mock_get_das_noise_levels.return_value, distances_km, "UK", -2.0, 0.1
    )
    mock_get_das_noise_levels.assert_called_once_with(
        dummy_fibre["noise_m"].values, 10, model_stacking=True
    )
