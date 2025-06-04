import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from sncast.detcap_model import minML

def test_minML_basic():
    # Create a small test DataFrame
    df = pd.DataFrame({
        'longitude': [0.0, 1.0],
        'latitude': [50.0, 51.0],
        'elevation_km': [0.0, 0.0],
        'noise [nm]': [1.0, 1.0],
        'station': ['STA1', 'STA2']
    })
    result = minML(df,
                   lon0=0,
                   lon1=1,
                   lat0=50,
                   lat1=51,
                   dlon=1,
                   dlat=1,
                   stat_num=1,
                   snr=1,
                   method='ML',
                   region='UK')
    assert hasattr(result, 'shape')
    assert result.shape == (2, 2)