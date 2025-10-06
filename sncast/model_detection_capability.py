#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2019 Martin Möllhoff
# SPDX-FileCopyrightText: 2024 Joseph Asplet, University of Oxford
# ------------------------------------------------------------------
"""
Filename: model_detection_capability.py
Purpose:  Seismic Network Capability Assessment Software Tool (SNCAST)
Author:   Martin Möllhoff, DIAS
Citation: Möllhoff, M., Bean, C.J. & Baptie, B.J.,
          SN-CAST: seismic network capability assessment software tool
          for regional networks - examples from Ireland.
          J Seismol 23, 493-504 (2019).
          https://doi.org/10.1007/s10950-019-09819-0

   Copyright (C) 2019 Martin Möllhoff, DIAS
   Copyright (C) 2024 Joseph Asplet, University of Oxford

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

--------------------------------------------------------------------
   Changes
    - Refactor and re-write of entire codebase, [Joseph Asplet, 2025]
    - Added support for DAS deployments [Joseph Asplet, 2025]
    - Implementation of GMPE based method (still in development [Joseph Asplet, 2025]
    - Implementation of BGS Local magnitude scale, [Joseph Asplet, 2024]
    - Functionality to calculate of a depth cross-section [Joseph Asplet, 2024]
    - Outputting of models as xarray.DataArray objects for easier plotting with
      PyGMT [Joseph Asplet, 2024]
    - Added support for seismic arrays and OBS with separate
      detection requirements [Joseph Asplet, 2024]

     Author: J Asplet
     email : joseph.asplet@earth.ox.ac.uk
"""

from decimal import Decimal
from multiprocessing import Pool
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d
import xarray

import pygc

from .gmpes import eval_gmpe
from .magnitude_conversions import convert_ml_to_mw, convert_mw_to_ml

ML_COEFFS = {
    "UK": {"a": 1.11, "b": 0.00189, "c": -2.09, "d": -1.16, "e": -0.2},
    "CAL": {"a": 1.11, "b": 0.00189, "c": -2.09},
}

SUPPORTED_ML_REGIONS = list(ML_COEFFS.keys())


def calc_ampl_from_magnitude(local_mag, hypo_dist, region):
    """
    Calculate the amplitude of a seismic signal given a local magnitude
    and hypocentral distance. Local magnitude scales for the UK and California
    (Hutton and Boore, 1987) are supported. The Hutton and Boore (1987) scale is
    the default ML scale reccomended by the IASPEI working group on earthquake magnitude
    determination and is consistent with the magnitude of Richter (1935).

    Parameters
    ----------
    local_mag : float, np.ndarray
        Local magnitude to calculate displacement amplitude for.
    hypo_dist : float, np.ndarray
        Hypocentral distance in km.
    region : str
        Seismic region. "UK" for Luckett et al. (2019) scale, "CAL" for
        Hutton and Boore (1987) scale.

    Returns
    -------
    ampl : float, np.ndarray
        Displacement amplitude in nm.
    """
    #   region specific ML = log(ampl) + a*log(hypo-dist) + b*hypo_dist + c
    if region == "UK":
        #   UK Scale uses new ML equation from Luckett et al., (2019)
        #   https://doi.org/10.1093/gji/ggy484
        #   Takes form local_mag = log(amp) + a*log(hypo-dist) + b*hypo-dist
        #                          + d*exp(e * hypo-dist) + c
        a = ML_COEFFS[region]["a"]
        b = ML_COEFFS[region]["b"]
        c = ML_COEFFS[region]["c"]
        d = ML_COEFFS[region]["d"]
        e = ML_COEFFS[region]["e"]
        ampl = np.power(
            10,
            (
                local_mag
                - a * np.log10(hypo_dist)
                - b * hypo_dist
                - c
                - d * np.exp(e * hypo_dist)
            ),
        )

    elif region == "CAL":
        # South. California scale, IASPEI (2005),
        # www.iaspei.org/commissions/CSOI/summary_of_WG_recommendations_2005.pdf
        a = ML_COEFFS[region]["a"]
        b = ML_COEFFS[region]["b"]
        c = ML_COEFFS[region]["c"]

        ampl = np.power(10, (local_mag - a * np.log10(hypo_dist) - b * hypo_dist - c))

    return ampl


def calc_local_magnitude(required_ampl, hypo_dist, region, mag_min, mag_delta):
    """
    Compute local magnitude (ML) for a given region for a set of amplitudes (in displacement)
    and hypocentral distances.

    Vectorized for numpy arrays. Magnitudes are snapped to the nearest interval <mag_delta>.

    Parameters
    ----------
    required_ampl : float or np.ndarray
        Displacement amplitude in nm.
    hypo_dist : float or np.ndarray
        Hypocentral distance in km.
    region : str
        Seismic region. "UK" for Luckett et al. (2019) scale, "CAL" for
        Hutton and Boore (1987) scale.
    mag_min : float
        Minimum magnitude to consider.
    mag_delta : float
        Magnitude bin width.

    Returns
    -------
    ml: np.ndarray
        Local magnitudes (ML) for the given amplitudes and distances.
    """
    if np.min(required_ampl <= 0):
        raise ValueError("At least one amplitude <=0!")

    if region == "UK":
        coeffs = ML_COEFFS[region]
        a = coeffs["a"]
        b = coeffs["b"]
        c = coeffs["c"]
        d = coeffs["d"]
        e = coeffs["e"]
        ml = (
            np.log10(required_ampl)
            + a * np.log10(hypo_dist)
            + b * hypo_dist
            + c
            + d * np.exp(e * hypo_dist)
        )
    elif region == "CAL":
        coeffs = ML_COEFFS[region]
        a = coeffs["a"]
        b = coeffs["b"]
        c = coeffs["c"]
        ml = np.log10(required_ampl) + a * np.log10(hypo_dist) + b * hypo_dist + c
    else:
        raise ValueError(f"Unknown region: {region}")

    # Snap to nearest mag_delta step above mag_min as
    # local magntiude is often only report to 1 decimal place
    # or some other fixed rounding level (the default is 0.1)
    ml = np.maximum(mag_min, np.ceil((ml - mag_min) / mag_delta) * mag_delta + mag_min)
    return ml


def _est_min_ml_at_station(noise, mag_min, mag_delta, distance, snr, **kwargs):
    """
    Estimates minimum detectable magnitude at a given station

    Function deprecated for noise displacement input, use calc_local_magnitude
    with vectorised numpy arrays instead. This function will be removed/replaced in future
    when work on GMPE method is complete.

    Parameters
    ----------
    noise : float
        Noise level at the station in nm.
    mag_min : float
        Minimum local magnitude.
    mag_delta : float
        Magnitude increment. Returned magnitude will be rounded up this increment.
    distance : float
        Hypocentral distance in km.
    snr : float
        Required signal-to-noise ratio for detection.
    **kwargs : dict
        Additional keyword arguments to control the method and parameters:
        - method: 'ML' or 'GMPE'. Default is 'ML'.
        - gmpe: GMPE model to use if method is 'GMPE'. Default is None.
        - gmpe_model_type: Type of GMPE model to use if method is 'GMPE'.
                           Default is None.
        - region: Locality for assumed ML scale parameters ('UK' or 'CAL').
                           Default is 'CAL'.
    """
    warnings.warn(
        "_est_min_ml_at_station is deprecated and only for GMPE dev use, use calc_local_magnitude",
        DeprecationWarning,
    )
    method = kwargs.get("method", "ML")
    region = kwargs.get("region", "CAL")
    gmpe = kwargs.get("gmpe", None)
    gmpe_model_type = kwargs.get("gmpe_model_type", None)
    if method == "GMPE":
        signal = 0
        ml = mag_min - mag_delta
        while signal < snr * noise:
            ml = ml + mag_delta
            mw = convert_ml_to_mw(ml, region)
            signal = eval_gmpe(mw, distance, gmpe, model_type=gmpe_model_type)
            ml = convert_mw_to_ml(mw, region)
            if ml > 3:
                break
        return ml
    elif method == "ML":
        raise ValueError("ML no longer supported, use vectorised function")
    else:
        raise ValueError(f"Unknown method: {method}")


def find_min_ml(
    lon0,
    lon1,
    lat0,
    lat1,
    dlon,
    dlat,
    foc_depth=0,
    networks=None,
    stat_num=[5],
    **kwargs,
):
    """
    This routine calculates the geographic distribution of the minimum
    detectable local magnitude ML for a given seismic network.

    Input requirements for seismic stations, arrays, and DAS:
    - networks: List of paths to CSV files or DataFrames containing station data for each network.
    - stat_num: List of required number of station detections for each network.
    - arrays: List of paths to CSV files or DataFrames containing seismic array data (optional).
    - array_num: List of required number of station detections for each array (default is 1).
    - das: List of paths to CSV files or DataFrames containing DAS noise data (optional).

    Example of the input file format for stations and arrays:
        longitude, latitude, elevation_km, noise [nm], station

    Example of the input file format for DAS:
        channel_index, fiber_length_m, longitude, latitude, noise_m, elevation_km

    Parameters
    ----------
    lon0 : float
        Minimum longitude of the region.
    lon1 : float
        Maximum longitude of the region.
    lat0 : float
        Minimum latitude of the region.
    lat1 : float
        Maximum latitude of the region.
    dlon : float
        Longitude increment for the grid.
    dlat : float
        Latitude increment for the grid.
    foc_depth : float, optional
        Assumed earthquake focal depth in km. Default is 0.
    networks : list or str or pd.DataFrame, optional
        List of paths to CSV files or DataFrames containing station data for each network.
    stat_num : list of int.
        List of required number of station detections for each network.
    arrays : list or str or pd.DataFrame, optional
        List of paths to CSV files or DataFrames containing seismic array data.
    das : list or str or pd.DataFrame, optional
        List of paths to CSV files or DataFrames containing DAS noise data.
    **kwargs : dict
        Additional keyword arguments to control the method and parameters:
        - method: 'ML' or 'GMPE'. Default is 'ML'.
        - gmpe: GMPE model to use if method is 'GMPE'. Default is None.
        - gmpe_model_type: Type of GMPE model to use if method is 'GMPE'. Default is None.
        - region: Locality for assumed ML scale parameters ('UK' or 'CAL'). Default is 'CAL'.
        - array_num: Number of stations required for a detection on an array. Default is 1.
        - obs_stat_num: Number of stations required for a detection on an OBS. Default is 3.
        - nproc: Number of processors to use for parallel processing. Default is 1.
        - detection_length: float, optional. Length of fiber (in meters) required for a detection.
          Default is 1000 m.
        - mag_min: float, optional. Minimum local magnitude to consider. Default is -2.0.
        - mag_delta: float, optional. Increment for local magnitude. Default is 0.1.
        - snr: float, optional. Required signal-to-noise ratio for detection. Default is 3.0.

    Returns
    -------
    mag_det : xarray.DataArray
        A 2D xarray DataArray with the following dimensions:
            - Latitude: latitude of the grid point in decimal degrees
            - Longitude: longitude of the grid point in decimal degrees
        The values in the DataArray are the minimum detectable local magnitude ML
        at that grid point.
    """
    # exit if no stations provided
    if (
        (networks is None)
        and (kwargs.get("arrays") is None)
        and (kwargs.get("das") is None)
    ):
        raise ValueError("No seismic networks, arrays or DAS provided!")
    # Make kwargs for worker function
    kwargs_worker = {}
    kwargs_worker["foc_depth"] = foc_depth
    kwargs_worker["snr"] = kwargs.get("snr", 3.0)
    kwargs_worker["mag_min"] = kwargs.get("mag_min", -2.0)
    kwargs_worker["mag_delta"] = kwargs.get("mag_delta", 0.1)

    if kwargs.get("method") == "GMPE":
        if "gmpe" not in kwargs:
            raise ValueError(
                "GMPE model must be specified if" + "GMPE method is selected"
            )
        if "gmpe_model_type" not in kwargs:
            raise ValueError(
                "GMPE model type must be specified if" + "GMPE method is selected"
            )
        kwargs_worker["gmpe"] = kwargs["gmpe"]
        kwargs_worker["gmpe_model_type"] = kwargs["gmpe_model_type"]
    elif kwargs.get("method") == "ML":
        kwargs_worker["method"] = "ML"
        kwargs_worker["gmpe"] = None
        kwargs_worker["gmpe_model_type"] = None

    else:
        kwargs_worker["method"] = "ML"
        warnings.warn("Method not recognised, using ML as default")

    if kwargs.get("region") is None:
        kwargs_worker["region"] = "CAL"
        warnings.warn("Region not specified, using CAL as default")
    elif kwargs["region"] not in SUPPORTED_ML_REGIONS:
        raise ValueError(
            f"Region {kwargs['region']} not supported, "
            + f"supported regions are {SUPPORTED_ML_REGIONS}"
        )
    else:
        kwargs_worker["region"] = kwargs["region"]

    print(f'Method : {kwargs_worker["method"]}')
    print(f'Region : {kwargs_worker["region"]}')

    if networks is not None:
        # read in data, file format: "LON, LAT, NOISE [nm], STATION"
        if isinstance(networks, (list, tuple)):
            network_noise_dfs = [read_station_data(n) for n in networks]
        else:
            network_noise_dfs = [read_station_data(networks)]

        if isinstance(stat_num, int):
            warnings.warn(
                "Single integer provided for stat_num, "
                + "assuming this applies to all networks"
            )
            stat_num = [stat_num]

        if len(stat_num) != len(network_noise_dfs):
            warnings.warn(
                f"Number of networks ({len(network_noise_dfs)}) does not match "
                + f"number of required stations ({len(stat_num)}), "
                + f"using first value, {stat_num[0]}, for all networks"
            )
            stat_num = [stat_num[0]] * len(network_noise_dfs)

        # check there are enough stations in each network
        for i, df in enumerate(network_noise_dfs):
            if len(df) < stat_num[i]:
                raise ValueError(
                    f"Not enough stations in network {i+1}: "
                    + f"have {len(df)}, need {stat_num[i]}"
                )
        kwargs_worker["network_noise_dfs"] = network_noise_dfs
        kwargs_worker["stat_num"] = stat_num
    else:
        print("No seismic networks provided")

    if kwargs.get("arrays") is not None:
        print(
            "Using seismic arrays in model. Arrays are modelled"
            + " as the central station with a required station number of 1."
        )
        if isinstance(kwargs["arrays"], (list, tuple)):
            array_dfs = [read_station_data(a) for a in kwargs["arrays"]]
        else:
            array_dfs = [read_station_data(kwargs["arrays"])]

        array_num = kwargs.get("array_num", 1)
        if isinstance(array_num, int):
            array_num = [array_num]

        if len(array_num) != len(array_dfs):
            warnings.warn(
                f"Number of arrays ({len(array_dfs)}) does not match "
                + f"number of required stations ({len(array_num)}), "
                + f"using first value, {array_num[0]}, for all arrays"
            )
            array_num = [array_num[0]] * len(array_dfs)

        kwargs_worker["array_num"] = array_num
        kwargs_worker["array_dfs"] = array_dfs

        # check there are enough stations in each array
        for i, df in enumerate(array_dfs):
            if len(df) < array_num[i]:
                raise ValueError(
                    f"Not enough stations in array {i+1}: "
                    + f"have {len(df)}, need {array_num[i]}"
                )

            array_num = [stat_num[0]] * len(network_noise_dfs)

        # check there are enough stations in each network
        for i, df in enumerate(network_noise_dfs):
            if len(df) < stat_num[i]:
                raise ValueError(
                    f"Not enough stations in network {i+1}: "
                    + f"have {len(df)}, need {stat_num[i]}"
                )

        kwargs_worker["array_dfs"] = array_dfs

    if "das" in kwargs:
        if "detection_length" not in kwargs:
            warnings.warn("Detection length not specified, using default of 1.0 km")
            kwargs["detection_length"] = 1e3  # Default to 1 km
        # Read in DAS noise data
        das_in = kwargs["das"]
        if isinstance(das_in, (list, tuple)):
            das_dfs = [read_das_noise_data(d) for d in das_in]
        else:
            das_dfs = [read_das_noise_data(das_in)]

        for df in das_dfs:
            if len(df) == 0:
                raise ValueError("No DAS data found in one of the input files")
        print(f'DAS detection length: {kwargs["detection_length"]} m')
        kwargs_worker["das_dfs"] = das_dfs

    lons, lats, nx, ny = create_grid(lon0, lon1, lat0, lat1, dlon, dlat)

    nproc = kwargs.get("nproc", 1)
    print(f"Using {nproc} cores")
    # Ensure fixed args are all in kwargs for worker function
    args_list = [
        ((ix, iy, lons[ix], lats[iy]), kwargs_worker)
        for ix in range(nx)
        for iy in range(ny)
    ]
    mag_grid = np.zeros((ny, nx))
    # Detection capability has to be calulated at each grid point,
    # Split this up using Pool and imap_unordered to multiple cores
    # maybe numba would be quicker here?

    with Pool(processes=nproc) as pool:
        for iy, ix, val in pool.imap_unordered(_wrapper_minml_worker, args_list):
            mag_grid[iy, ix] = val

    # Make xarray grid to output
    mag_det = xarray.DataArray(
        mag_grid, coords=[lats, lons], dims=["Latitude", "Longitude"]
    )
    return mag_det


def find_min_ml_x_section(
    lon0,
    lat0,
    azi,
    length_km,
    networks=None,
    min_depth=0,
    max_depth=20,
    ddist=5,
    ddepth=0.5,
    stat_num=[5],
    **kwargs,
):
    """
    Function to calculate a 2-D cross section of a SNCAST model.
    X-section line defined by start lat/lon and the azimuth and length (in km)
    of the line.

    Input should be a csv file (or Pandas DataFrame)
      longitude, latitude, noise [nm], station name
    e.g.: -7.5100, 55.0700, 0.53, IDGL

    Parameters
    ----------
        lon0 : float
            Longitude of the start of the cross-section line
        lat0 : float
            Latitude of the start of the cross-section line
        azi : float
            Azimuth of cross-section in degrees from north
        length_km : float
            Cross-section length in km
        networks : list or str or pd.DataFrame, optional
            List of paths to CSV files or DataFrames containing station data for each network.
        min_depth : float, optional
            Minimum depth of cross-section in km. Default is 0.
        max_depth : float, optional
            Maximum depth of cross-section in km. Default is 20.
        ddist : float, optional
            Distance increment along the cross-section in km. Default is 5.
        ddepth : float, optional
            Depth increment along the cross-section in km. Default is 0.5.
        stat_num : int, optional
            Number of stations required for a detection. Default is 4.
        snr : float, optional
            Required signal-to-noise ratio for detection. Default is 3.
        region : str, optional
            Locality for assumed ML scale parameters ('UK' or 'CAL').
            Default is 'CAL'.
        mag_min : float, optional
            Minimum local magnitude to consider when modelling detections.
            Default is -3.0.
        mag_delta : float, optional
            Increment for local magnitude. Default is 0.1.
        arrays : DataFrame or csv filename, optional
            Station information for seismic arrays including lat/lon and noise levels.
            If provided, the model will include detections from arrays.
            File is in the same format as stations_in
        obs : DataFrame or csv filename, optional
            Station information for OBS including lat/lon and noise levels.
            If provided, the model will include detections from OBS.
            File is in the same format as stations_in
        **kwargs : dict
            Additional keyword arguments to control the method and parameters:
            - method: 'ML' or 'GMPE'. Default is 'ML'.
            - gmpe: GMPE model to use if method is 'GMPE'. Default is None.
            - gmpe_model_type: Type of GMPE model to use if method is 'GMPE'.
                               Default is None.
            - region: Locality for assumed ML scale parameters ('UK' or 'CAL').
                               Default is 'CAL'.
            - array_num: Number of stations required for a detection on an array.
                            Default is 1.
            - obs_stat_num: Number of stations required for a detection on an OBS.
                            Default is 3.
    Returns
    -------
        array : xarray.DataArray
            A 2D xarray DataArray with the following dimensions:
                - depth_km: depth in km
                - distance_along_xsection_km: distance along the cross-section in km
            The values in the DataArray are the minimum detectable local magnitude ML
            at that grid point.
    """
    # exit if no stations provided
    if (
        (networks is None)
        and (kwargs.get("arrays") is None)
        and (kwargs.get("das") is None)
    ):
        raise ValueError("No seismic networks, arrays or DAS provided!")

    kwargs_worker = {}
    kwargs_worker["snr"] = kwargs.get("snr", 3.0)
    kwargs_worker["mag_min"] = kwargs.get("mag_min", -2.0)
    kwargs_worker["mag_delta"] = kwargs.get("mag_delta", 0.1)

    if kwargs.get("method") == "GMPE":
        if "gmpe" not in kwargs:
            raise ValueError(
                "GMPE model must be specified if" + "GMPE method is selected"
            )
        if "gmpe_model_type" not in kwargs:
            raise ValueError(
                "GMPE model type must be specified if" + "GMPE method is selected"
            )
        kwargs_worker["gmpe"] = None
        kwargs_worker["gmpe_model_type"] = None
    elif kwargs.get("method") == "ML":
        kwargs_worker["method"] = "ML"
        kwargs_worker["gmpe"] = None
        kwargs_worker["gmpe_model_type"] = None

    else:
        kwargs_worker["method"] = "ML"
        warnings.warn("Method not recognised, using ML as default")

    if kwargs.get("region") is None:
        kwargs_worker["region"] = "CAL"
        warnings.warn("Region not specified, using CAL as default")
    elif kwargs["region"] not in SUPPORTED_ML_REGIONS:
        raise ValueError(
            f"Region {kwargs['region']} not supported, "
            + f"supported regions are {SUPPORTED_ML_REGIONS}"
        )
    else:
        kwargs_worker["region"] = kwargs["region"]

    print(f'Method : {kwargs_worker["method"]}')
    print(f'Region : {kwargs_worker["region"]}')

    if networks is not None:
        # read in data, file format: "LON, LAT, NOISE [nm], STATION"
        if isinstance(networks, (list, tuple)):
            network_noise_dfs = [read_station_data(n) for n in networks]
        else:
            network_noise_dfs = [read_station_data(networks)]

        if isinstance(stat_num, int):
            warnings.warn(
                "Single integer provided for stat_num, "
                + "assuming this applies to all networks"
            )
            stat_num = [stat_num]

        if len(stat_num) != len(network_noise_dfs):
            warnings.warn(
                f"Number of networks ({len(network_noise_dfs)}) does not match "
                + f"number of required stations ({len(stat_num)}), "
                + f"using first value, {stat_num[0]}, for all networks"
            )
            stat_num = [stat_num[0]] * len(network_noise_dfs)

        # check there are enough stations in each network
        for i, df in enumerate(network_noise_dfs):
            if len(df) < stat_num[i]:
                raise ValueError(
                    f"Not enough stations in network {i+1}: "
                    + f"have {len(df)}, need {stat_num[i]}"
                )
        kwargs_worker["network_noise_dfs"] = network_noise_dfs
        kwargs_worker["stat_num"] = stat_num
    else:
        print("No seismic networks provided")

    if kwargs.get("arrays") is not None:
        print(
            "Using seismic arrays in model. Arrays are modelled"
            + " as the central station with a required station number of 1."
        )
        if isinstance(kwargs["arrays"], (list, tuple)):
            array_dfs = [read_station_data(a) for a in kwargs["arrays"]]
        else:
            array_dfs = [read_station_data(kwargs["arrays"])]

        array_num = kwargs.get("array_num", 1)
        if isinstance(array_num, int):
            array_num = [array_num]

        if len(array_num) != len(array_dfs):
            warnings.warn(
                f"Number of arrays ({len(array_dfs)}) does not match "
                + f"number of required stations ({len(array_num)}), "
                + f"using first value, {array_num[0]}, for all arrays"
            )
            array_num = [array_num[0]] * len(array_dfs)

        kwargs_worker["array_num"] = array_num
        kwargs_worker["array_dfs"] = array_dfs

        # check there are enough stations in each array
        for i, df in enumerate(array_dfs):
            if len(df) < array_num[i]:
                raise ValueError(
                    f"Not enough stations in array {i+1}: "
                    + f"have {len(df)}, need {array_num[i]}"
                )

            array_num = [stat_num[0]] * len(network_noise_dfs)

        # check there are enough stations in each network
        for i, df in enumerate(network_noise_dfs):
            if len(df) < stat_num[i]:
                raise ValueError(
                    f"Not enough stations in network {i+1}: "
                    + f"have {len(df)}, need {stat_num[i]}"
                )

        kwargs_worker["array_dfs"] = array_dfs

    if "das" in kwargs:
        if "detection_length" not in kwargs:
            warnings.warn("Detection length not specified, using default of 1.0 km")
            kwargs["detection_length"] = 1e3  # Default to 1 km
        # Read in DAS noise data
        das_in = kwargs["das"]
        if isinstance(das_in, (list, tuple)):
            das_dfs = [read_das_noise_data(d) for d in das_in]
        else:
            das_dfs = [read_das_noise_data(das_in)]

        for df in das_dfs:
            if len(df) == 0:
                raise ValueError("No DAS data found in one of the input files")
        print(f'DAS detection length: {kwargs["detection_length"]} m')
        kwargs_worker["das_dfs"] = das_dfs

    # Calculate lon/lat co-ordinates for X-section line
    ndists = int((length_km / ddist) + 1)
    distance_km = np.linspace(0, length_km, ndists)
    xsection = pygc.great_circle(
        latitude=lat0, longitude=lon0, azimuth=azi, distance=distance_km * 1e3
    )
    ndepths = int((max_depth - min_depth) / ddepth) + 1
    depths = np.linspace(min_depth, max_depth, ndepths)
    # Iterate along cross-section
    args_list = [
        (
            (ix, iz, xsection["latitude"][ix], xsection["longitude"][ix], depths[iz]),
            kwargs_worker,
        )
        for ix in range(ndists)
        for iz in range(ndepths)
    ]
    mag_grid = np.zeros((ndepths, ndists))

    nproc = kwargs.get("nproc", 1)
    print(f"Using {nproc} cores")
    # Make xarray grid to output
    with Pool(processes=nproc) as pool:
        for iz, ix, val in pool.imap_unordered(_wrapper_minml_worker, args_list):
            mag_grid[iz, ix] = val

    array = xarray.DataArray(
        mag_grid,
        coords=[depths, distance_km],
        dims=["depth_km", "distance_along_xsection_km"],
    )
    return array


def _wrapper_minml_worker(arg):
    """
    Function to act as a wrapper for the _minml_worker function to allow
    passing multiple arguments using multiprocessing.Pool.imap_unordered

    """
    args, kwargs = arg
    return _minml_worker(*args, **kwargs)


def _wrapper_minml_xsection_worker(arg):
    """
    Function to act as a wrapper for the _minml_x_section_worker function to allow
    passing multiple arguments using multiprocessing.Pool.imap_unordered

    """
    args, kwargs = arg
    return _minml_x_section_worker(*args, **kwargs)


def _minml_worker(ix, iy, lon, lat, **kwargs):
    """
    Worker function for minML which allows the magnitude grid to be parallelised
    over multiple processors using multiprocessing.Pool

    Parameters
    ----------
    ix : int
        x-index of the grid point.
    iy : int
        y-index of the grid point.
    lon : float
        Longitude of the grid point.
    lat : float
        Latitude of the grid point.
    **kwargs : dict
        Additional keyword arguments to pass to the worker function.

    Returns
    -------
    tuple
        Tuple containing the y-index, x-index, and minimum magnitude for the grid point.
    """
    # Initialize min_mag to absurdly high value
    min_mag = 100.0
    if "network_noise_dfs" in kwargs:
        for n, net_df in enumerate(kwargs["network_noise_dfs"]):
            # spell out kwargs here for clarify and to avoid passing
            # unnecessary data to worker processes
            min_mag_net = calc_min_ml_at_gridpoint(
                net_df,
                lon,
                lat,
                kwargs["stat_num"][n],
                kwargs["foc_depth"],
                kwargs["snr"],
                mag_min=kwargs["mag_min"],
                mag_delta=kwargs["mag_delta"],
                method=kwargs["method"],
                region=kwargs["region"],
                gmpe=kwargs.get("gmpe", None),
                gmpe_model_type=kwargs.get("gmpe_model_type", None),
            )
            min_mag = min(min_mag, min_mag_net)
    # Add arrays if provided
    if kwargs.get("array_dfs") is not None and not kwargs["array_dfs"].empty:
        for a, array_df in enumerate(kwargs["array_dfs"]):

            min_mag_arrays = calc_min_ml_at_gridpoint(
                array_df,
                lon,
                lat,
                stat_num=kwargs["array_num"][a],
                foc_depth=kwargs["foc_depth"],
                snr=kwargs["snr"],
                mag_min=kwargs["mag_min"],
                mag_delta=kwargs["mag_delta"],
                method=kwargs["method"],
                region=kwargs["region"],
                gmpe=kwargs["gmpe"],
                gmpe_model_type=kwargs["gmpe_model_type"],
            )
            min_mag = min(min_mag, min_mag_arrays)

    if kwargs.get("das_dfs") is not None:
        for das_df in kwargs["das_dfs"]:
            if das_df is not None and not das_df.empty:
                mag_min_das = calc_min_ml_at_gridpoint_das(
                    das_df,
                    lon,
                    lat,
                    foc_depth=kwargs["foc_depth"],
                    snr=kwargs["snr"],
                    mag_min=kwargs["mag_min"],
                    mag_delta=kwargs["mag_delta"],
                    detection_length=kwargs.get("detection_length", 1000),
                    gmpe=kwargs.get("gmpe", None),
                    gmpe_model_type=kwargs.get("gmpe_model_type", None),
                    region=kwargs.get("region", "CAL"),
                    method=kwargs.get("method", "ML"),
                )
                min_mag = min(min_mag, mag_min_das)
    return (iy, ix, min_mag)


def _minml_x_section_worker(ix, iz, lon, lat, depth, **kwargs):
    """
    Worker function for minML x-section which allows the 2-D grid of
    lat/lon along a cross-section and depth to be parallelised.

    Parameters
    ----------
    ix : int
        x-index along cross-section line.
    iz : int
        z-index (depth) of the grid point.
    lon : float
        Longitude along cross-section line of grid point
    lat : float
        Latitude along cross-section line of grid point
    depth : float
        Depth of the grid point in km.
    **kwargs : dict
        Additional keyword arguments to pass to the worker function.

    Returns
    -------
    tuple
        Tuple containing the z-index, x-index, and minimum magnitude for the grid point.
    """
    # Initialize min_mag to absurdly high value
    min_mag = 100.0
    if "network_noise_dfs" in kwargs:
        for n, net_df in enumerate(kwargs["network_noise_dfs"]):
            # spell out kwargs here for clarify and to avoid passing
            # unnecessary data to worker processes
            min_mag_net = calc_min_ml_at_gridpoint(
                net_df,
                lon,
                lat,
                kwargs["stat_num"][n],
                depth,
                kwargs["snr"],
                mag_min=kwargs["mag_min"],
                mag_delta=kwargs["mag_delta"],
                method=kwargs["method"],
                region=kwargs["region"],
                gmpe=kwargs["gmpe"],
                gmpe_model_type=kwargs["gmpe_model_type"],
            )
            min_mag = min(min_mag, min_mag_net)
    # Add arrays if provided
    if kwargs.get("array_dfs") is not None and not kwargs["array_dfs"].empty:
        for a, array_df in enumerate(kwargs["array_dfs"]):

            min_mag_arrays = calc_min_ml_at_gridpoint(
                array_df,
                lon,
                lat,
                foc_depth=depth,
                stat_num=kwargs["array_num"][a],
                snr=kwargs["snr"],
                mag_min=kwargs["mag_min"],
                mag_delta=kwargs["mag_delta"],
                method=kwargs["method"],
                region=kwargs["region"],
                gmpe=kwargs.get("gmpe", None),
                gmpe_model_type=kwargs.get("gmpe_model_type", None),
            )
            min_mag = min(min_mag, min_mag_arrays)

    if kwargs.get("das_dfs") is not None:
        for das_df in kwargs["das_dfs"]:
            if das_df is not None and not das_df.empty:
                mag_min_das = calc_min_ml_at_gridpoint_das(
                    das_df,
                    lon,
                    lat,
                    foc_depth=depth,
                    snr=kwargs["snr"],
                    mag_min=kwargs["mag_min"],
                    mag_delta=kwargs["mag_delta"],
                    detection_length=kwargs.get("detection_length", 1000),
                    gmpe=kwargs.get("gmpe", None),
                    gmpe_model_type=kwargs.get("gmpe_model_type", None),
                    region=kwargs.get("region", "CAL"),
                    method=kwargs.get("method", "ML"),
                )
                min_mag = min(min_mag, mag_min_das)

    return (iz, ix, min_mag)


def create_grid(lon0, lon1, lat0, lat1, dlon, dlat):
    """
    Initialize lat/lon grid for SNCAST model.

    Parameters
    ----------
    lon0 : float
        Minimum longitude of the grid.
    lon1 : float
        Maximum longitude of the grid.
    lat0 : float
        Minimum latitude of the grid.
    lat1 : float
        Maximum latitude of the grid.
    dlon : float
        Longitude increment for the grid.
    dlat : float
        Latitude increment for the grid.
    Returns
    -------
    lats : np.ndarray
        Array of latitudes for the grid.
    lons : np.ndarray
        Array of longitudes for the grid.
    nx : int
        Number of grid points in the x-direction (longitude).
    ny : int
        Number of grid points in the y-direction (latitude).
    """
    if lon0 > lon1:
        raise ValueError(f"lon0 {lon0} must be less than lon1 {lon1}")
    if lat0 > lat1:
        raise ValueError(f"lat0 {lat0} must be less than lat1 {lat1}")
    if dlon <= 0 or dlat <= 0:
        raise ValueError(f"dlon and dlat ({dlon, dlat}) must be positive values")
    if (Decimal(str(lat1)) - Decimal(str(lat0))) % Decimal(str(dlat)) != 0:
        raise ValueError(f"lat1 {lat1} - lat0 {lat0} must be divisible by dlat {dlat}")
    if (Decimal(str(lon1)) - Decimal(str(lon0))) % Decimal(str(dlon)) != 0:
        raise ValueError(f"lon1 {lon1} - lon0 {lon0} must be divisible by dlon {dlon}")

    nx = int((lon1 - lon0) / dlon) + 1
    ny = int((lat1 - lat0) / dlat) + 1
    lats = np.linspace(lat1, lat0, ny)
    lons = np.linspace(lon0, lon1, nx)
    return lons, lats, nx, ny


def calc_min_ml_at_gridpoint(
    stations_df,
    lon,
    lat,
    stat_num,
    foc_depth,
    snr,
    **kwargs,
):
    """
    Calculates the minimum local magnitude which can be detected by a
    set of seismic stations at a given grid point.

    Parameters
    ----------
    stations_df : pd.DataFrame
        DataFrame containing station data with columns:
        - 'longitude': longitude of the station in decimal degrees
        - 'latitude': latitude of the station in decimal degrees
        - 'elevation_km': elevation of the station in km
        - 'noise [nm]': noise level at the station in nanometres
        - 'station': station name
    lon : float
        Longitude of the grid point in decimal degrees.
    lat : float
        Latitude of the grid point in decimal degrees.
    foc_depth : float
        Focal depth of the event in kilometres.
    stat_num : int
        Required number of station detections to calculate minimum ML.
    snr : float
        Signal-to-noise ratio required for detection.

    mag_min : float
        - Minimum local magnitude to consider when modelling detections.
    mag_delta : float
        - Increment for local magnitude.
    method: str
        Method to use for 'ML'  (local mag scale) or 'GMPE'. Default is 'ML'.
    region: str
        Locality for assumed ML scale parameters ('UK' or 'CAL').
        Default is 'CAL'.
    gmpe: str
        GMPE model to use if method is 'GMPE'. Default is None.
    gmpe_model_type: str
        Type of GMPE model to use if method is 'GMPE'. Default is None.
    Returns
    -------
    float
        Minimum local magnitude that can be detected at the grid point.
    """
    method = kwargs.get("method", "ML")
    region = kwargs.get("region", "CAL")
    mag_min = kwargs.get("mag_min", -2.0)
    mag_delta = kwargs.get("mag_delta", 0.1)
    if method == "ML":

        noise = stations_df["noise [nm]"].values
        distances_km = (
            pygc.great_distance(
                start_latitude=lat,
                end_latitude=stations_df["latitude"].values,
                start_longitude=lon,
                end_longitude=stations_df["longitude"].values,
            )["distance"]
            * 1e-3
        )
        dz = np.abs(foc_depth - stations_df["elevation_km"].values)
        # calculate hypcocentral distance
        hypo_dist = np.sqrt(distances_km**2 + dz**2)
        required_ampls = snr * noise
        mags = calc_local_magnitude(
            required_ampls,
            hypo_dist,
            region=region,
            mag_min=mag_min,
            mag_delta=mag_delta,
        )
        sorted_mags = np.sort(mags)
        return sorted_mags[stat_num - 1]

    elif method == "GMPE":
        noise = stations_df["noise [cm/s]"].values
        # Use pygc to compute great-circle (epicentral) distance
        # pygc returns this in meters, then we convert to km
        distances_km = (
            pygc.great_distance(
                start_latitude=lat,
                end_latitude=stations_df["latitude"].values,
                start_longitude=lon,
                end_longitude=stations_df["longitude"].values,
            )["distance"]
            * 1e-3
        )
        dz = np.abs(foc_depth - stations_df["elevation_km"].values)
        # calculate hypcocentral distance
        hypo_dist = np.sqrt(distances_km**2 + dz**2)
        mag = [
            _est_min_ml_at_station(
                noise[s],
                mag_min,
                mag_delta,
                hypo_dist[s],
                snr,
                method=kwargs["method"],
                gmpe=kwargs.get("gmpe", None),
                gmpe_model_type=kwargs.get("gmpe_model_type", None),
                region=kwargs["region"],
            )
            for s in range(len(stations_df))
        ]
        # sort magnitudes in ascending order so we can
        # then find the minumum magnitude detected at
        # the required number of stations
        mag = sorted(mag)
        return mag[stat_num - 1]
    else:
        raise ValueError(f"Unsupported Method {method}")


def read_station_data(stations_in):
    """
    Read and validate station data from a DataFrame or CSV file.

    Parameters
    ----------
    stations_in : str or pd.DataFrame
        Path to a CSV file or a DataFrame containing station data.
    Returns
    -------
    pd.DataFrame
        DataFrame containing the station data with required columns.
    Raises
    ------
    ValueError
        If required columns are missing from the DataFrame.

    """
    if isinstance(stations_in, str):
        stations_df = pd.read_csv(stations_in)
    else:
        stations_df = stations_in.copy()
    if "elevation_m" in stations_df.columns:
        stations_df["elevation_m"] *= 1e-3
        stations_df.rename(columns={"elevation_m": "elevation_km"}, inplace=True)
    required_cols = {
        "longitude",
        "latitude",
        "elevation_km",
        "noise [nm]",
        "station",
    }
    if not required_cols.issubset(stations_df.columns):
        raise ValueError(f"Missing columns: {required_cols - set(stations_df.columns)}")
    return stations_df


def read_das_noise_data(das_in):
    """
    Read and validate DAS noise data from a DataFrame or CSV file.

    Parameters
    ----------
    das_in : str or pd.DataFrame
        Path to a CSV file or a DataFrame containing DAS noise data.
        Expected columns are:
            - channel_index: Index of the DAS channel
            - fiber_length_m: Length of the fiber in meters
            - longitude: Longitude of the channel in decimal degrees
            - latitude: Latitude of the channel in decimal degrees
            - noise_m: Noise level at the channel in meters
            - elevation_km: (optional) Elevation of the channel in kilometers
    Returns
    -------
    pd.DataFrame
        DataFrame containing the DAS noise data with required columns.
    Raises
    ------
    ValueError
        If required columns are missing from the DataFrame.
    """
    if isinstance(das_in, str):
        das_df = pd.read_csv(das_in)
    else:
        das_df = das_in.copy()
    required_cols = {
        "channel_index",
        "fiber_length_m",
        "longitude",
        "latitude",
        "noise_m",
    }
    if not required_cols.issubset(das_df.columns):
        raise ValueError(f"Missing columns: {required_cols - set(das_df.columns)}")
    if "elevation_km" not in das_df.columns:
        # If elevation is not provided, set it to zero
        das_df["elevation_km"] = 0.0

    return das_df


def get_das_noise_levels(das_noise, wind_len_idx, model_stacking=True):
    """
    Gets the maximum seismic noise level (in displacement) along
    a given continuous fibre length.

    Parameters
    ----------
    das_noise : np.ndarray
        Array of DAS noise levels.
    wind_len_idx : int
        Length of the sliding window in samples.
    model_stacking : bool, optional
        Whether to apply a sqrt(N) improvement to SNR assuming
        That the DAS data will be stacked. Default is True.

    Returns
    -------
    np.ndarray
        Array of noise levels for each section of the fibre.
    """
    max_filtered_noise = maximum_filter1d(das_noise, size=wind_len_idx, mode="nearest")
    # model includes a assumed improvment in signal/noise from stacking
    # over the length of the fibre section
    if model_stacking:
        # calc root(n) improvment in noise level
        noise_rootn = np.ones(das_noise.shape) * np.sqrt(wind_len_idx)
        # at start/end of fiber the stacking length is reduced
        # here we do this in samples/array indexes and assume evenly spaced channels
        # sliding window is centered on each channel so even at the start
        # we have wind_len_idx//2 channels stacked
        half_win = wind_len_idx // 2
        # Apply window for either and odd or even window length
        if wind_len_idx % 2 == 0:
            # Even window: left and right edges have different lengths
            noise_start = np.sqrt(np.arange(half_win, wind_len_idx))
            noise_end = np.sqrt(np.arange(wind_len_idx, half_win, -1))
            noise_rootn[:half_win] = noise_start
            noise_rootn[-half_win:] = noise_end
        else:
            # Odd window is symmetric about center
            noise_start = np.sqrt(np.arange(half_win + 1, wind_len_idx + 1))
            noise_rootn[: half_win + 1] = noise_start
            noise_rootn[-(half_win + 1) :] = noise_start[::-1]

        max_filtered_noise /= noise_rootn

        # raise value error if any values in max_filtered_noise are zero or negative
        if np.any(max_filtered_noise <= 0):
            raise ValueError("Filtered noise levels contain zero or negative values.")
        elif np.any(np.isnan(max_filtered_noise)):
            raise ValueError("Filtered noise levels contain NaN values.")

    return max_filtered_noise


def calc_min_ml_at_gridpoint_das(
    fibre, lon, lat, detection_length, foc_depth, snr, **kwargs
):
    """
    Calculates the minimum local magnitude which can
    de detected along a given detection length of the fibre.
    Parameters
    ----------
    fibre : pd.DataFrame
        DataFrame containing the fibre data with columns:
        - 'channel_index': index of the channel
        - 'fiber_length_m': length of the fibre in metres
        - 'longitude': longitude of the fibre in decimal degrees
        - 'latitude': latitude of the fibre in decimal degrees
        - 'noise_m': noise level of the fibre in metres
    lon : float
        Grid point longitude in decimal degrees.
    lat : float
        Grid point latitude in decimal degrees.
    detection_length : float
        Length of the fibre over which to calculate the noise level in metres.
    foc_depth : float
        Focal depth of the event in kilometres.
    snr : float
        Signal-to-noise ratio required for detection.
    kwargs : dict
        Additional keyword arguments, including:
        - 'mag_min': minimum local magnitude to consider
        - 'mag_delta': increment for local magnitude
        - 'method': method to use for calculation ('ML' or 'GMPE')
        - 'gmpe': GMPE model to use if method is 'GMPE'
        - 'gmpe_model_type': type of GMPE model to use if method is 'GMPE'
        - 'region': region for the local magnitude scale ('UK' or 'CAL')
    Returns
    -------
    float
        Minimum local magnitude that can be detected by a continuous section of fibre
        of the input detection length.
    """
    method = kwargs.get("method", "ML")
    region = kwargs.get("region", "CAL")
    mag_min = kwargs.get("mag_min", -2)
    mag_delta = kwargs.get("mag_delta", 0.1)
    # Set model_stacking to True by default
    model_stacking = kwargs.get("model_stacking", True)

    if method != "ML":
        raise ValueError(f"Method: {method} not supported for DAS at this time")

    gauge_len = kwargs.get("gauge_length", 20)
    window_size = int(np.ceil((detection_length / gauge_len)))
    # print("~" * 50)
    # print(f"There are {window_size} channels in the sliding window.")
    # Covert noise from metres to nanometres
    noise_nm = fibre["noise_m"].values * 1e9
    # Apply moving maximum filter to get max noise level along fibre section
    # of length detection_length
    windowed_noise_nm = get_das_noise_levels(noise_nm, window_size, model_stacking)

    # Precompute the hypocoentral distances using pygc
    # pygc gives distances in metres so convert to km
    distances_km = (
        pygc.great_distance(
            start_latitude=fibre["latitude"].values,
            end_latitude=lat,
            start_longitude=fibre["longitude"].values,
            end_longitude=lon,
        )["distance"]
        * 1e-3
    )
    dz = np.abs(foc_depth - fibre["elevation_km"].values)
    hypo_distances = np.sqrt(distances_km**2 + dz**2)
    # Calculate the minimum local magnitude for each section of the fibre
    required_ampls = snr * windowed_noise_nm
    mags = calc_local_magnitude(
        required_ampls,
        hypo_distances,
        region=region,
        mag_min=mag_min,
        mag_delta=mag_delta,
    )
    # Get smallest ML detected at any one window along the fiber
    return np.min(mags)


# Deprecated function - not currently used

# def get_min_ml_for_das_section(channel_pos, mags, detection_length, slide_length=1):
#     """
#     Gets the earthquake magntiude which is detectable across a given
#     continuous fibre length (detection_length).
#     Parameters
#     ----------
#     channel_pos : np.ndarray
#         position of channels along the fibre in metres.
#     mags : array
#         Noise level of the fibre in metres.
#     detection_length : float
#         Length of the fibre over which to calculate the noise level in metres.
#     slide_length : float, optional
#         Length to slide the detection window along the fibre in metres.
#         Default is 1 metre.
#     Returns
#     -------
#     np.ndarray
#         Array of noise levels for each section of the fibre.
#     """
#     fibre_length = channel_pos[-1] - channel_pos[0]
#     if detection_length > fibre_length:
#         raise ValueError(
#             f"detection_length {detection_length} must be less"
#             + f" than fibre_length {fibre_length}"
#         )
#     if detection_length <= 0:
#         raise ValueError(f"detection_length {detection_length} must be positive")

#     # Calculate the number of windows along the fibre
#     n_windows = (
#         int(
#             np.floor(
#                 (channel_pos[-1] - channel_pos[0] - detection_length) / slide_length
#             )
#         )
#         + 1
#     )
#     ml_at_windows = np.zeros(n_windows)
#     mags = np.array(mags)  # Convert to NumPy array for advanced indexing

#     return np.min(ml_at_windows)
