# test core.py
import re

import numpy as np
import pandas as pd
import pytest

from sncast.core import (
    ModelConfig,
    SeismicArrayNetwork,
    SeismicNetwork,
    _read_das_noise_data,
    _read_station_data,
)

TEST_STATION_DATA = pd.read_csv("tests/data/station_data.csv")


def test_SeismicNetwork_initialization():
    net = SeismicNetwork(TEST_STATION_DATA)
    assert len(net.stations) == len(
        TEST_STATION_DATA
    )  # Test default params initialization
    assert net.network_code == "XX"
    assert net.required_detections == 5


def test_SeismicNetwork_csv_initialization():
    net = SeismicNetwork("tests/data/station_data.csv")
    assert len(net.stations) == len(TEST_STATION_DATA)
    assert net.stations.iloc[0]["station"] == TEST_STATION_DATA.iloc[0]["station"]
    assert net.stations.iloc[1]["station"] == TEST_STATION_DATA.iloc[1]["station"]
    # Test default params initialization
    assert net.network_code == "XX"
    assert net.required_detections == 5


def test_SeismicNetwork_custom_initialization():
    net = SeismicNetwork(TEST_STATION_DATA, network_code="AB", required_detections=1)
    assert net.network_code == "AB"
    assert net.required_detections == 1


def test_SeismicNetwork_empty_initialization():
    with pytest.raises(ValueError, match="No stations in the seismic network."):
        SeismicNetwork(
            pd.DataFrame(
                [
                    {
                        "station": [],
                        "longitude": [],
                        "latitude": [],
                        "elevation_km": [],
                        "noise [nm]": [],
                    }
                ]
            )
        )


def test_SeismicNetwork_too_few_stations():
    with pytest.raises(
        ValueError,
        match=f"Not enough stations in the seismic network. Required: 5, found: {len(stations)}",
    ):
        SeismicNetwork(TEST_STATION_DATA, required_detections=5)


def test_SeismicNetwork_add_stations():
    stations_initial = TEST_STATION_DATA.iloc[0]
    net = SeismicNetwork(stations_initial, required_detections=1)
    stations_to_add = TEST_STATION_DATA.iloc[1:]
    net.add_stations(stations_to_add)
    comb_df = pd.concat([stations_initial, stations_to_add], ignore_index=True)
    pd.testing.assert_frame_equal(net.stations, comb_df)
    assert net.num_stations == 3


def test_SeismicArrayNetwork_initialization():
    arrays = [
        {
            "station": "ARRAY1",
            "longitude": 0.0,
            "latitude": 50.0,
            "elevation_km": 0.0,
            "noise [nm]": 1.0,
        },
        {
            "station": "ARRAY2",
            "longitude": 1.0,
            "latitude": 51.0,
            "elevation_km": 0.1,
            "noise [nm]": 10.0,
        },
        {
            "station": "ARRAY3",
            "longitude": 2.0,
            "latitude": 52.0,
            "elevation_km": 0.2,
            "noise [nm]": 5.0,
        },
    ]
    array = SeismicArrayNetwork(arrays)
    assert len(array.stations) == 3
    # Test default params initialization
    assert array.network_code == "XX"
    assert array.required_detections == 2
    # Test inherited class
    assert isinstance(array, SeismicNetwork)


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


def test_ModelConfig_custom_initialization():
    """Test initialization with custom parameters"""
    config = ModelConfig(
        snr=5.0,
        foc_depth_km=10.0,
        region="UK",
        nproc=4,
        method="ML",
        mag_min=-3.0,
        mag_delta=0.05,
        model_stacking_das=False,
    )

    assert config.snr == 5.0
    assert config.foc_depth_km == 10.0
    assert config.region == "UK"
    assert config.nproc == 4
    assert config.mag_min == -3.0
    assert config.mag_delta == 0.05
    assert config.model_stacking_das is False


@pytest.mark.parametrize("bad_region", ["USA", "", 10, "MARS"])
def test_ModelConfig_bad_regions(bad_region):
    with pytest.raises(ValueError):
        ModelConfig(region=bad_region)


def test_ModelConfig_bad_method():
    with pytest.raises(ValueError):
        ModelConfig(method="some junk")


def test_ModelConfig_negative_snr():
    with pytest.raises(ValueError, match="SNR must be a positive value"):
        ModelConfig(snr=-1.0)


def test_ModelConfig_negative_foc_depth():
    with pytest.raises(ValueError, match="Focal depth must be a positive value"):
        ModelConfig(foc_depth_km=-10.0)


def test_ModelConfig_invalid_gmpe():
    with pytest.raises(ValueError, match="Invalid GMPE"):
        ModelConfig(method="GMPE", gmpe="INVALID")


@pytest.mark.parametrize(
    "lon0, lat0, lon1, lat1, dlon, dlat",
    [
        (0, 50, 1, 51, 0.1, 0.1),
        (-10, 30, -9, 31, 0.5, 0.5),
        (100, -20, 101, -19, 1.0, 1.0),
    ],
)
def test_ModelConfig_create_grid_working_case(lon0, lat0, lon1, lat1, dlon, dlat):
    config = ModelConfig()
    config.add_grid_params(lon0, lon1, lat0, lat1, dlon, dlat)
    assert config.lon0 == lon0
    assert config.lon1 == lon1
    assert config.lat0 == lat0
    assert config.lat1 == lat1
    assert config.dlon == dlon
    assert config.dlat == dlat


def test_ModelConfig_create_grid_reversed_coords():
    lon0 = 1
    lon1 = 0
    lat0 = 51
    lat1 = 50
    with pytest.raises(ValueError, match=f"lon0 {lon0} must be less than lon1 {lon1}"):
        config = ModelConfig()
        config.add_grid_params(lon0, lon1, lat0, lat1)
    lon0 = 0
    lon1 = 1
    lat0 = 51
    lat1 = 50
    with pytest.raises(ValueError, match=f"lat0 {lat0} must be less than lat1 {lat1}"):
        config = ModelConfig()
        config.add_grid_params(lon0, lon1, lat0, lat1)


def test_ModelConfig_create_grid_negative_dlon_dlat():
    dlon_neg = -0.1
    dlat_neg = -1
    with pytest.raises(
        ValueError,
        match=re.escape(f"dlon and dlat ({dlon_neg, 0.1}) must be positive values"),
    ):
        config = ModelConfig()
        config.add_grid_params(0, 1, 50, 51, dlon=dlon_neg, dlat=0.1)
    with pytest.raises(
        ValueError,
        match=re.escape(f"dlon and dlat ({0.1, dlat_neg}) must be positive values"),
    ):
        config = ModelConfig()
        config.add_grid_params(0, 1, 50, 51, dlat=dlat_neg, dlon=0.1)


def test_ModelConfig_create_grid_indivisible():
    lon0 = 0
    lon1 = 1
    lat0 = 50
    lat1 = 51
    dlon = 0.3
    dlat = 0.7
    with pytest.raises(
        ValueError,
        match=re.escape(f"lon1 {lon1} - lon0 {lon0} must be divisible by dlon {dlon}"),
    ):
        config = ModelConfig()
        config.add_grid_params(lon0, lon1, lat0, lat1, dlon=dlon)

    with pytest.raises(
        ValueError,
        match=re.escape(f"lat1 {lat1} - lat0 {lat0} must be divisible by dlat {dlat}"),
    ):
        config = ModelConfig()
        config.add_grid_params(lon0, lon1, lat0, lat1, dlat=dlat)


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
