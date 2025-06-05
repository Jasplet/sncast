import numpy as np
import pytest
from unittest.mock import patch
from sncast.magnitude_conversions import convert_mw_to_ml


def test_convert_mw_to_ml_uk_small_mw():
    mw = np.array([2.0, 2.5, 2.9])
    expected_ml = (mw - 0.75) / 0.69
    np.testing.assert_array_almost_equal(convert_mw_to_ml(mw, region="UK"), expected_ml)


def test_convert_mw_to_ml_uk_large_mw():
    mw = np.array([3.1, 4.0, 5.0])
    expected_ml = (mw - 0.23) / 0.85
    np.testing.assert_array_almost_equal(convert_mw_to_ml(mw, region="UK"), expected_ml)


def test_convert_mw_to_ml_uk_boundary():
    mw = np.array([3.0])
    expected_ml = (mw - 0.75) / 0.69
    np.testing.assert_array_almost_equal(convert_mw_to_ml(mw, region="UK"), expected_ml)


def test_convert_mw_to_ml_unsupported_region():
    mw = np.array([2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="Unsupported region"):
        convert_mw_to_ml(mw, region="US")


@patch("tests.test_convert_mw_to_ml.convert_mw_to_ml")
def test_mock_convert_mw_to_ml(mock_convert):
    mw = np.array([2.0, 3.0, 4.0])
    mock_convert.return_value = np.array([1.0, 2.0, 3.0])
    result = convert_mw_to_ml(mw, region="UK")
    mock_convert.assert_called_once_with(mw, region="UK")
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))
