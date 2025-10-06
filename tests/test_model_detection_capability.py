import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from sncast.model_detection_capability import find_min_ml
from sncast.model_detection_capability import create_grid
from sncast.model_detection_capability import read_station_data
from sncast.model_detection_capability import _est_min_ml_at_station
from sncast.model_detection_capability import calc_ampl_from_magnitude
from sncast.model_detection_capability import calc_local_magnitude


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
    "amplitude, distance, region",
    [
        (10.0, 10, "UK"),
        (10, 10, "CAL"),
    ],
)
def test_calc_local_mag_scalar(
    amplitude,
    distance,
    region,
):
    ml = calc_local_magnitude(amplitude, distance, region, -2, 0.1)
    assert isinstance(ml, float) or isinstance(ml, np.floating)


@pytest.mark.parametrize(
    "region",
    [
        ("UK"),
        ("CAL"),
    ],
)
def test_ml_magnitude_array(region):
    amps = np.array([10, 20])
    dists = np.array([10, 20])
    ml = calc_local_magnitude(amps, dists, region, -2, 0.1)
    assert ml.shape == (2,)
    assert np.all(ml > -2)


@pytest.mark.parametrize(
    "ampl, distance, snr, mag_delta, mag_min",
    [
        (10.0, 1.0, 1, 0.1, -2.0),
        (0.5, 2.0, 2, 0.5, -4),
        (0.01, 100, 1, 0.2, 0.0),
        (1, 0.01, 10, 0.05, -2),
    ],
)
def test_calc_local_magnitude_UK(ampl, distance, snr, mag_delta, mag_min):
    def ml_uk(a_s, r, mag_min):
        """UK Local Magnitude Scale"""
        a = 1.11
        b = 0.00189
        c = -2.09
        d = -1.16
        e = -0.2
        ml = np.log10(a_s) + a * np.log10(r) + b * r + c + d * np.exp(e * r)
        ml = np.maximum(
            mag_min, np.ceil((ml - mag_min) / mag_delta) * mag_delta + mag_min
        )
        if ml < mag_min:
            return mag_min
        else:
            return ml

    expected = ml_uk(ampl * snr, distance, mag_min)

    # Test with default parameters
    result = calc_local_magnitude(
        ampl * snr, distance, region="UK", mag_min=mag_min, mag_delta=mag_delta
    )
    # Difference between ML calculated for a given noise and distance
    # and the modelled min ML should be less than the mag_delta
    assert np.isclose(
        expected - result, 0, atol=mag_delta
    ), f"Expected {expected}, got {result}, mag_delta {mag_delta}"
    assert isinstance(
        result, float
    ), f"Result {result} is {type(result)}, expected float"


@pytest.mark.parametrize(
    "ampl, distance, snr, mag_delta, mag_min",
    [
        (10.0, 1.0, 1, 0.1, -2.0),
        (0.5, 2.0, 2, 0.5, -4),
        (0.01, 100, 1, 0.2, 0.0),
        (1.0, 0.01, 2.0, 0.05, -2),
    ],
)
def test_calc_local_magnitude_CAL(ampl, distance, snr, mag_delta, mag_min):
    def ml_cal(a_s, r):
        a = 1.11
        b = 0.00189
        c = -2.09
        ml = np.log10(a_s) + a * np.log10(r) + b * r + c
        ml = np.maximum(
            mag_min, np.ceil((ml - mag_min) / mag_delta) * mag_delta + mag_min
        )
        if ml < mag_min:
            return mag_min
        else:
            return ml

    expected = ml_cal(ampl * snr, distance)
    result = calc_local_magnitude(
        ampl * snr, distance, region="CAL", mag_min=mag_min, mag_delta=mag_delta
    )
    assert np.isclose(
        expected - result, 0, atol=mag_delta
    ), f"Failed for CAL region: expected {expected}, got {result}"


def test_calc_local_magnitude_raises_on_unknown_region():
    with pytest.raises(ValueError):
        calc_local_magnitude(1.0, 10.0, "MARS", -2, 0.1)


@pytest.mark.parametrize("ampl", [0, -10])
def test_calc_local_magnitude_raise_on_bad_ampl(ampl):
    with pytest.raises(ValueError):
        calc_local_magnitude(ampl, 1, "CAL", -2, 0.1)


def test_est_min_ml_at_station_raises_unsupported():
    for mode in ["ML", "foo", "bar"]:
        with pytest.raises(ValueError):
            _est_min_ml_at_station(
                noise=10, mag_min=-2, mag_delta=0.1, distance=50, snr=3, method=mode
            )


# Test create_grid function #
@pytest.mark.parametrize(
    "inputs, expected",
    [
        ((-1, 1, 50, 52, 0.5, 0.5), (5, 5)),
    ],
)
def test_create_grid_expected(inputs, expected):
    lons, lats, nlon, nlat = create_grid(*inputs)
    assert nlon == expected[0]
    assert nlat == expected[1]
    assert lons.shape == (nlon,)
    assert lats.shape == (nlat,)
    assert np.isclose(lons[0], inputs[0])
    assert np.isclose(lons[-1], inputs[1])
    assert np.isclose(lats[0], inputs[2])
    assert np.isclose(lats[-1], inputs[3])


def test_create_grid_invalid_bounds():
    with pytest.raises(ValueError, match="lon0 2 must be less than lon1 -2"):
        create_grid(2, -2, 50, 52, 1, 1)

    with pytest.raises(ValueError, match="lat0 52 must be less than lat1 50"):
        create_grid(-2, 2, 52, 50, 1, 1)


@pytest.mark.parametrize("dlon, dlat", [(0, 1), (1, 0), (-1, -1), (0, 0)])
def test_create_grid_negative_increments(dlon, dlat):

    with pytest.raises(
        ValueError, match="dlon and dlat ({dlon, dlat}) must be positive values"
    ):
        create_grid(-2, 2, 50, 52, dlon, dlat)


def test_create_grid_non_divisible():
    with pytest.raises(ValueError, match="must be divisible by"):
        create_grid(-2, 2, 50, 52.3, 1, 1)  # Non-divisible range


# Test find_min_ml function #


def test_find_min_ml_basic():
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
    result = find_min_ml(
        networks=df,
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


def test_find_min_ml_no_networks_provided():
    with pytest.raises(ValueError, match="No seismic networks, arrays or DAS provided"):
        find_min_ml(-1, 1, 50, 52, 0.5, 0.5)


@pytest.mark.parametrize("bad_region", ["", None, "US", "MARS"])
def test_find_min_ml_unsupported_region(bad_region):
    with pytest.raises(ValueError, match=f"Region {bad_region} not supported"):
        find_min_ml(-1, 1, 50, 52, 0.5, 0.5, networks=["dummy.csv"], region=bad_region)


def test_find_min_ml_warns_unknown_ML():
    df = pd.DataFrame(
        {
            "longitude": [0.0, 1.0],
            "latitude": [50.0, 51.0],
            "elevation_km": [0.0, 0.0],
            "noise [nm]": [0.1, 0.1],
            "station": ["STA1", "STA2"],
        }
    )
    for method in ["foo", "bar", ""]:
        with pytest.warns(
            UserWarning, match="Method not recognised, using ML as default"
        ):
            find_min_ml(
                networks=df,
                lon0=0,
                lon1=1,
                lat0=50,
                lat1=51,
                dlon=1,
                dlat=1,
                stat_num=1,
                snr=1,
                method=method,
                region="UK",
            )


def test_find_min_ml_warns_no_region():
    with pytest.warns(UserWarning, match="Region not specified, using CAL as default"):
        find_min_ml(-1, 1, 50, 52, 0.5, 0.5, networks=["dummy.csv"])


@patch("sncast.model_detection_capability.create_grid")
@patch("sncast.model_detection_capability.read_station_data")
@patch("sncast.model_detection_capability._minml_worker")
def test_find_min_ml_bad_gmpe_inputs(mock_worker, mock_read_station_data, mock_grid):
    """
    Tests that find_min_ml raises errors when required kwargs for
    the GMPE method are not provided.
    """

    # Mock the internal functions to return dummy data
    mock_read_station_data.return_value = pd.DataFrame(
        {
            "longitude": [0.0],
            "latitude": [50.0],
            "elevation_km": [0.0],
            "noise [nm]": [1.0],
            "station": ["STA1"],
        }
    )
    mock_grid.return_value = ([0.0, 1.0], [50.0, 51.0], 2, 2)
    mock_worker.return_value = (0, 0, 1.5)  # Dummy result

    # should raise value error if method is GMPE but no gmpe provided
    with pytest.raises(ValueError):
        find_min_ml(
            networks="dummy_path.csv",
            lon0=0,
            lon1=1,
            lat0=50,
            lat1=51,
            dlon=1,
            dlat=1,
            stat_num=1,
            snr=1,
            method="GMPE",
            region="UK",
        )

    with pytest.raises(ValueError):
        find_min_ml(
            networks="dummy_path.csv",
            lon0=0,
            lon1=1,
            lat0=50,
            lat1=51,
            dlon=1,
            dlat=1,
            stat_num=1,
            snr=1,
            method="GMPE",
            region="UK",
            gmpe="AK14",
        )
