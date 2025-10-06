#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Joseph Asplet, University of Oxford
# ------------------------------------------------------------------
"""
Filename: magnitude_conversions.py
Purpose:  Ultility functions for converting between
          moment magnitude and local magnitude
          using empirical scaling relationships.
Author:   Joseph Asplet, University of Oxford

   Copyright (C) 2025 Joseph Asplet

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.

   email:       joseph.asplet@earth.ox.ac.uk
   web:         www.jasplet.github.io

"""

import numpy as np


def convert_mw_to_ml(mw, region="UK"):
    """
    Converts Moment magntiude to local magnitude
    using an empirical scaling relationship

    Default region is the UK.
    For small M<3 event we use the conversion developed by Butcher et al., (2020)
    https://doi.org/10.1785/0120190032 using data from
    New Ollerton and is applicable for small magntiude events.

    For Mw > 3 the Ottemöller and Sargeant (2013) https://doi.org/10.1785/0120130085
    conversion is used instead.

    Parameters:
    ----------
    mw : numpy.Array
        array of moment magnitudes to convert
    region : str
        Region's scaling relationship to use. Defaults to UK

    Returns:
    ----------
    ml : numpy.Array
        empirically converted local magnitudes
    """

    if region == "UK":
        butcher_uk_small_mw = np.vectorize(lambda x: (x - 0.75) / 0.69)
        ottermoller_uk = np.vectorize(lambda x: (x - 0.23) / 0.85)

        ml = np.where(mw <= 3, butcher_uk_small_mw(mw), ottermoller_uk(mw))
    else:
        raise ValueError(f"Unsupported region {region}")

    return ml


def convert_ml_to_mw(ml, region="UK"):
    """
    Converts Local magnitude to moment magnitude
    using an empirical scaling relationship

    Default region is the UK.
    For small M<3 event we use the conversion developed by Butcher et al., (2020)
    https://doi.org/10.1785/0120190032 using data from
    New Ollerton and is applicable for small magntiude events.

    For Mw > 3 the Ottemöller and Sargeant (2013) https://doi.org/10.1785/0120130085
    conversion is used instead.

    Parameters:
    ----------
    ml : numpy.Array
        array of local magnitudes to convert
    region : str
        Region's scaling relationship to use. Defaults to UK

    Returns:
    ----------
    mw : numpy.Array
        empirically converted moment magnitudes
    """

    if region == "UK":
        butcher_uk_small_mw = np.vectorize(lambda x: 0.75 * x + 0.69)
        ottermoller_uk = np.vectorize(lambda x: 0.85 * x + 0.23)

        mw = np.where(ml <= 3, butcher_uk_small_mw(ml), ottermoller_uk(ml))
    else:
        raise ValueError(f"Unsupported region {region}")

    return mw
