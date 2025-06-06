import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from sncast.detcap_model import update_with_arrays
from sncast.detcap_model import minML
from sncast.detcap_model import read_station_data, read_das_noise_data
from sncast.detcap_model import _est_min_ML_at_station
from sncast.detcap_model import calc_ampl_from_magnitude


def test_minML_basic():
    # Create a small test DataFrame
    df = pd.DataFrame(
        {
            "longitude": [0.0, 1.0],
            "latitude": [50.0, 51.0],
            "elevation_km": [0.0, 0.0],
            "noise [nm]": [0.1, 0.1],
            "station": ["STA1", "STA2"],
        }
    )
    result = minML(
        df,
        lon0=0,
        lon1=1,
        lat0=50,
        lat1=51,
        dlon=1,
        dlat=1,
        stat_num=1,
        snr=1,
        method="ML",
        region="UK",
    )
    assert hasattr(result, "shape")
    assert result.shape == (2, 2)


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
    output = read_station_data(df)
    output_csv = read_station_data("tests/data/station_data.csv")
    assert output.equals(df)
    assert output_csv.equals(df)


def test_read_das_noise_data():
    # Make a test DataFrame
    df = pd.DataFrame(
        {
            "channel_index": [10010, 10020, 10030],
            "fiber_length_m": [1000, 2000, 3000],
            "longitude": [0.01, 0.01, 0.01],
            "latitude": [50.01, 50.02, 50.03],
            "noise_m": [1e-8, 2e-9, 2.6e-8]
    }
    )
    output = read_das_noise_data(df)
    output_csv = read_das_noise_data("tests/data/das_noise_data.csv")
    assert output.equals(df)
    assert output_csv.equals(df)


def test_read_das_noise_data_invalid():
    



def test_calc_amplitude_UK():

    a = 1.11
    b = 0.00189
    c = -2.09
    d = -1.16
    e = -0.2

    for local_mag, hypo_dist in zip(
        [-3, 0.0, 1.0, 2.0, 3.0, 4.0], [0.1, 10, 20, 30, 100]
    ):
        expect = np.power(
            10,
            (
                local_mag
                - a * np.log10(hypo_dist)
                - b * hypo_dist
                - c
                - d * np.exp(e * hypo_dist)
            ),
        )
        actual = calc_ampl_from_magnitude(local_mag, hypo_dist, region="UK")
        assert np.isclose(
            actual, expect
        ), f"Failed for local_mag={local_mag}, hypo_dist={hypo_dist}"


def test_calc_amplitude_CAL():

    a = 1.11
    b = 0.00189
    c = -2.09

    for local_mag, hypo_dist in zip(
        [-3, 0.0, 1.0, 2.0, 3.0, 4.0], [0.1, 10, 20, 30, 100]
    ):
        expect = np.power(10, (local_mag - a * np.log10(hypo_dist) - b * hypo_dist - c))
        actual = calc_ampl_from_magnitude(local_mag, hypo_dist, region="CAL")
        assert np.isclose(
            actual, expect
        ), f"Failed for local_mag={local_mag}, hypo_dist={hypo_dist}"


@pytest.mark.parametrize(
    "noise, distance, snr, mag_delta, mag_min",
    [
        (10.0, 1.0, 1, 0.1, -2.0),
        (0.5, 2.0, 2, 0.5, -4),
        (0.01, 100, 1, 0.2, 0.0),
        (1, 0.01, 10, 0.05, -5),
    ],
)
def test_est_min_ML_at_station(noise, distance, snr, mag_delta, mag_min):
    def ml_uk(a_s, r, mag_min):
        """UK Local Magnitude Scale"""
        a = 1.11
        b = 0.00189
        c = -2.09
        d = -1.16
        e = -0.2
        ml = np.log10(a_s) + a * np.log10(r) + b * r + c + d * np.exp(e * r)
        if ml < mag_min:
            return mag_min
        else:
            return ml

    expected = ml_uk(noise * snr, distance, mag_min)

    # Test with default parameters
    result = _est_min_ML_at_station(
        noise, mag_min, mag_delta, distance, snr, method="ML", region="UK"
    )
    # Difference between ML calculated for a given noise and distance
    # and the modelled min ML should be less than the mag_delta
    assert np.isclose(
        expected - result, 0, atol=mag_delta
    ), f"Expected {expected}, got {result}, mag_delta {mag_delta}"
    assert isinstance(
        result, float
    ), f"Result {result} is {type(result)}, expected float"


    # Test for CAL region if supported
    def ml_cal(a_s, r):
        a = 1.11
        b = 0.00189
        c = -2.09
        ml = np.log10(a_s) + a * np.log10(r) + b * r + c
        if ml < mag_min:
            return mag_min
        else:
            return ml

    expected = ml_cal(noise * snr, distance)
    result = _est_min_ML_at_station(
        noise, mag_min, mag_delta, distance, snr, method="ML", region="CAL"
    )
    assert np.isclose(
        expected - result, 0, atol=mag_delta
    ), f"Failed for CAL region: expected {expected}, got {result}"


def test_calc_min_ML_gridpoint():
    df = pd.DataFrame(
        {
            "longitude": [0.0, 0.5],
            "latitude": [49.0, 49.0],
            "elevation_km": [0.0, 0.0],
            "noise [nm]": [10.0, 10.0],
            "station": ["STA1", "STA2"],
        }
    )


def test_update_with_arrays_lower():
    df = pd.DataFrame(
        {
            "longitude": [0.0],
            "latitude": [50.0],
            "elevation_km": [0.0],
            "noise [nm]": [1.0],
            "station": ["STA1"],
        }
    )
    # Use a dummy kwargs dict for ML method
    kwargs = {"method": "ML", "gmpe": None, "gmpe_model_type": None, "region": "UK"}
    # mag_grid_val is higher than what the array will return
    result = update_with_arrays(5.0, df, 1, 0.0, 50.0, 0.0, 1, -2.0, 0.1, **kwargs)
    assert result < 5.0


def test_update_with_arrays_higher():
    df = pd.DataFrame(
        {
            "longitude": [0.0],
            "latitude": [50.0],
            "elevation_km": [0.0],
            "noise [nm]": [1000.0],  # High noise, so ML will be high
            "station": ["STA1"],
        }
    )
    kwargs = {"method": "ML", "gmpe": None, "gmpe_model_type": None, "region": "UK"}
    # mag_grid_val is lower than what the array will return
    result = update_with_arrays(1.0, df, 1, 1.0, 50.0, 0.0, 1, -2.0, 0.1, **kwargs)
    assert result == 1.0


def test_update_with_arrays_missing_column():
    df = pd.DataFrame(
        {
            "longitude": [0.0],
            "latitude": [50.0],
            # 'elevation_km' missing
            "noise [nm]": [1.0],
            "station": ["STA1"],
        }
    )
    kwargs = {"method": "ML", "gmpe": None, "gmpe_model_type": None, "region": "UK"}
    try:
        update_with_arrays(1.0, df, 1, 0.0, 50.0, 0.0, 1, -2.0, 0.1, **kwargs)
        assert False, "Should raise ValueError"
    except Exception:
        pass
