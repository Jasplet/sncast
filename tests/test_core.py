# test core.py
import numpy as np
import pandas as pd
import pytest

from sncast.core import _read_station_data
from sncast.core import _read_das_noise_data
from sncast.core import ModelConfig


def test_ModelConfig_defaults():

    Config = ModelConfig()
    assert Config.snr == 3.0
    assert Config.foc_depth_km == 2.0
    assert Config.region == "CAL"
    assert Config.nproc == 1
    assert Config.method == "ML"
    assert Config.mag_min == -2.0
    assert Config.mag_delta == 0.1
    assert Config.model_stacking_das is True
    assert Config.gmpe is None
    assert Config.gmpe_model_type is None

    Config_GMPE_default = ModelConfig(method="GMPE")
    assert Config_GMPE_default.gmpe == "AK14"
    assert Config_GMPE_default.gmpe_model_type == "PGV"


@pytest.mark.parametrize("bad_region", ["USA", "", 10, "MARS"])
def test_ModelConfig_bad_regions(bad_region):
    with pytest.raises(ValueError):
        ModelConfig(region=bad_region)


def test_ModelConfig_bad_method():
    with pytest.raises(ValueError):
        ModelConfig(method="some junk")


def test_read_station_data():
    # Create a small test DataFrame
    df = pd.DataFrame(
        {
            "longitude": [0.0, 1.0],
            "latitude": [50.0, 51.0],
            "elevation_km": [0.0, 0.0],
            "noise [nm]": [1.0, 1.0],
            "station": ["STA1", "STA2"],
        }
    )
    output = _read_station_data(df)
    output_csv = _read_station_data("tests/data/station_data.csv")
    assert output.equals(df)
    assert output_csv.equals(df)


def test_read_station_data_from_str():
    dummy_str = "tests/data/station_data.csv"
    # read dummy df
    dummy_df = pd.read_csv(dummy_str)
    # read using function
    out_df = _read_station_data(dummy_str)
    # check they are the same
    assert out_df.equals(dummy_df)
    # check columns
    assert all(
        col in out_df.columns
        for col in ["longitude", "latitude", "elevation_km", "noise [nm]", "station"]
    )


def test_read_station_data_elev_conversion():
    """Tests that elevation is correctly converted to km if in meters."""
    # create dummy df with elevation in meters
    dummy_df = pd.DataFrame(
        {
            "longitude": [0.0, 1.0],
            "latitude": [50.0, 51.0],
            "elevation_m": [1000.0, 2000.0],
            "noise [nm]": [1.0, 1.0],
            "station": ["STA1", "STA2"],
        }
    )
    out_df = _read_station_data(dummy_df)
    # check elevation_km column exists and is correct
    assert "elevation_km" in out_df.columns
    assert np.allclose(out_df["elevation_km"].values, np.array([1.0, 2.0]))


def test_read_das_noise_data():
    """Tests reading DAS noise data from a DataFrame and CSV file."""
    # Create a DataFrame matching the dummy CSV file
    dummy_df = pd.DataFrame(
        {
            "channel_index": [
                10010,
                10020,
                10030,
                10040,
                10050,
                10060,
                10070,
                10080,
                10090,
                10100,
            ],
            "fiber_length_m": [
                1000,
                2000,
                3000,
                4000,
                5000,
                6000,
                7000,
                8000,
                9000,
                10000,
            ],
            "longitude": [0.01] * 10,
            "latitude": [
                50.01,
                50.02,
                50.03,
                50.04,
                50.05,
                50.06,
                50.07,
                50.08,
                50.09,
                50.10,
            ],
            "noise_m": [
                1e-8,
                2e-9,
                2.6e-8,
                1.2e-8,
                3e-9,
                4e-9,
                1.5e-8,
                2.2e-8,
                5e-9,
                3.1e-8,
            ],
            "elevation_km": [0.01] * 10,
        }
    )
    output = _read_das_noise_data(dummy_df)
    output_csv = _read_das_noise_data("tests/data/das_dummy_data.csv")
    assert output.equals(dummy_df)
    assert output_csv.equals(dummy_df)


def test_read_das_noise_data_from_str():
    """Tests reading DAS noise data from a CSV file path."""
    dummy_str = "tests/data/das_dummy_data.csv"
    # read dummy df
    dummy_df = pd.read_csv(dummy_str)
    # read using function
    out_df = _read_das_noise_data(dummy_str)
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
    out_df = _read_das_noise_data(dummy_df)
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
        _read_das_noise_data(dummy_df)


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
    out_df = _read_das_noise_data(dummy_no_elev_df)
    # function should add elevation_km column of 0s
    assert "elevation_km" in out_df.columns
    assert np.all(out_df["elevation_km"].values == 0.0)


def test_read_das_noise_data_empty_dataframe():
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        _read_das_noise_data(empty_df)
