#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2019 Martin Möllhoff
# SPDX-FileCopyrightText: 2024–2025 Joseph Asplet, University of Oxford
# ------------------------------------------------------------------
"""
Filename: detcap_model.py
Purpose:  Seismic Network Capability Assessment Software Tool (SNCAST)
Author:   Martin Möllhoff, DIAS
Citation: Möllhoff, M., Bean, C.J. & Baptie, B.J.,
          SN-CAST: seismic network capability assessment software tool
          for regional networks - examples from Ireland.
          J Seismol 23, 493-504 (2019).
          https://doi.org/10.1007/s10950-019-09819-0

   Copyright (C) 2019 Martin Möllhoff
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
    - Refactor and re-write of enitre codebase, [Joseph Asplet, 2025]
    - Added support for DAS deployments [Joseph Asplet, 2025]
    - Implementation of GMPE based method (still in development [Jospeh Asplet, 2025]
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
import warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool
from obspy.signal.util import util_geo_km
from scipy.ndimage import maximum_filter1d

import pygc
import xarray


from .gmpes import eval_gmpe
from .magnitude_conversions import convert_ml_to_mw, convert_mw_to_ml

ML_COEFFS = {
    "UK": {"a": 1.11, "b": 0.00189, "c": -2.09, "d": -1.16, "e": -0.2},
    "CAL": {"a": 1.11, "b": 0.00189, "c": -2.09},
}


def calc_ampl_from_magnitude(local_mag, hypo_dist, region):
    """
    Calculate the amplitude of a seismic signal given a local magnitude
    and hypocentral distance. The empirical local magnitude
    scales from the UK and California regions are supported.
    """
    #   region specific ML = log(ampl) + a*log(hypo-dist) + b*hypo_dist + c
    if region == "UK":
        #   UK Scale uses new ML equation from Luckett et al., (2019)
        #   https://doi.org/10.1093/gji/ggy484
        #   Takes form local_mag = log(amp) + a*log(hypo-dist) + b*hypo-dist
        #                          + d*exp(e * hypo-dist) + c
        a = 1.11
        b = 0.00189
        c = -2.09
        d = -1.16
        e = -0.2
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
        a = 1.11
        b = 0.00189
        c = -2.09
        ampl = np.power(10, (local_mag - a * np.log10(hypo_dist) - b * hypo_dist - c))

    return ampl


def calc_local_magnitude(required_ampl, hypo_dist, region, mag_min, mag_delta):
    """
    Compute local magnitude (ML) for a given region.
    Vectorized for numpy arrays.
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


def _est_min_ML_at_station(noise, mag_min, mag_delta, distance, snr, **kwargs):
    """
    Estimates minimum detectable magntiude at a given station

    For using
    """
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


def minML(
    stations_in,
    lon0=-12,
    lon1=-4,
    lat0=50.5,
    lat1=56.6,
    dlon=0.2,
    dlat=0.2,
    stat_num=4,
    snr=3,
    foc_depth=0,
    mag_min=-2.0,
    mag_delta=0.1,
    arrays=None,
    obs=None,
    obs_stat_num=3,
    **kwargs,
):
    """
    This routine calculates the geographic distribution of the minimum
    detectable local magnitude ML for a given seismic network.

    Inputs stations_in is a Pandas DataFrame or path to a csv file which contains
    the following columns:
        - longitude: longitude of the station in decimal degrees
        - latitude: latitude of the station in decimal degrees
        - elevation_km: elevation of the station in km
        - station: station name
        - noise [nm]: noise level at the station in nanometres

    Example of the input file format:
        longitude, latitude, elevation, noise [nm], station name
        -7.5100, 55.0700, 0, 0.53, IDGL

    Parameters
    ----------
    stations_in : str or pd.DataFrame
        Path to a CSV file or a DataFrame containing station data.
    lon0 : float, optional
        Minimum longitude of the region. Default is -12.
    lon1 : float, optional
        Maximum longitude of the region. Default is -4.
    lat0 : float, optional
        Minimum latitude of the region. Default is 50.5.
    lat1 : float, optional
        Maximum latitude of the region. Default is 56.6.
    dlon : float, optional
        Longitude increment for the grid. Default is 0.2.
    dlat : float, optional
        Latitude increment for the grid. Default is 0.2.
    stat_num : int, optional
        Required number of station detections to calculate minimum ML. Default is 4.
    snr : float, optional
        Required signal-to-noise ratio for detection. Default is 3.
    foc_depth : float, optional
        Assumed earthquake focal depth in km. Default is 0.
    mag_min : float, optional
        Minimum local magnitude to consider when modelling detections. Default is -2.0.
    mag_delta : float, optional
        Increment for local magnitude. Default is 0.1.
    arrays : str or pd.DataFrame, optional
        Path to a CSV file or a DataFrame containing seismic array data.
        If provided, the model will include detections from arrays.
        File is in the same format as stations_in
    array_num : int, optional
        Number of detections required at an array.
        Default is 1 if arrays are provided.
    obs : str or pd.DataFrame, optional
        Path to a CSV file or a DataFrame containing OBS data.
        If provided, the model will include detections from OBS.
        File is in the same format as stations_in
    obs_stat_num : int, optional
        Required number of station detections from OBS to calculate minimum ML.
        Default is 3.
    **kwargs : dict, optional
        Additional keyword arguments to control the method and parameters:
        - method: 'ML' or 'GMPE'. Default is 'ML'.
        - gmpe: GMPE model to use if method is 'GMPE'. Default is None.
        - gmpe_model_type: Type of GMPE model to use if method is 'GMPE'.
                           Default is None.
        - region: Locality for assumed ML scale parameters ('UK' or 'CAL').
                           Default is 'CAL'.
        - das: Path to a CSV file or a DataFrame containing DAS noise data.
                           Can also be a list or tuple of DataFrames
        - detection_length: Length of the fibre over which to calculate the noise
                            level in metres.
                           Default is 1 km.
        - slide_length: Length to slide the detection window along the fibre in metres.
                        Default is 1 m.

    Returns
    -------
    mag_det : xarray.DataArray
        A 2D xarray DataArray with the following dimensions:
            - Latitude: latitude of the grid point in decimal degrees
            - Longitude: longitude of the grid point in decimal degrees
        The values in the DataArray are the minimum detectable local magnitude ML
        at that grid point.

    """

    if kwargs["method"] == "ML":
        kwargs["gmpe"] = None
        kwargs["gmpe_model_type"] = None
    elif kwargs["method"] == "GMPE":
        if not kwargs["gmpe"]:
            raise ValueError(
                "GMPE model must be specified if" + "GMPE method is selected"
            )
        if not kwargs["gmpe_model_type"]:
            raise ValueError(
                "GMPE model type must be specified if" + "GMPE method is selected"
            )
    else:
        kwargs["method"] = "ML"
        warnings.warn("Method not recognised, using ML as default")

    if not kwargs["region"]:
        kwargs["region"] = "CAL"
        warnings.warn("Region not specified, using CAL as default")

    nproc = kwargs.get("nproc", 1)

    print(f'Method : {kwargs["method"]}')
    print(f'Region : {kwargs["region"]}')
    print(f"Using {nproc} cores")
    # read in data, file format: "LON, LAT, NOISE [nm], STATION"
    stations_df = read_station_data(stations_in)
    # Read in arrays and obs data if provided
    arrays_df = read_station_data(arrays) if arrays is not None else None
    obs_df = read_station_data(obs) if obs is not None else None
    if len(stations_df) < stat_num:
        raise ValueError(
            f"Not enough stations ({len(stations_df)}) "
            + f"to calculate minimum ML at {stat_num} stations"
        )

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
    else:
        das_dfs = []

    lons, lats, nx, ny = create_grid(lon0, lon1, lat0, lat1, dlon, dlat)
    args_list = [
        (
            ix,
            iy,
            lons,
            lats,
            stations_df,
            foc_depth,
            stat_num,
            snr,
            mag_min,
            mag_delta,
            arrays_df,
            obs_df,
            das_dfs,
            kwargs,
        )
        for ix in range(nx)
        for iy in range(ny)
    ]
    mag_grid = np.zeros((ny, nx))
    with Pool(processes=nproc) as pool:
        for iy, ix, val in pool.imap_unordered(_minML_worker, args_list):
            mag_grid[iy, ix] = val

    # # Make xarray grid to output
    mag_det = xarray.DataArray(
        mag_grid, coords=[lats, lons], dims=["Latitude", "Longitude"]
    )
    return mag_det


def _minML_worker(args):
    """
    Worker function for minML which allows the magntiude grid to be parallelised
    """
    # Unpack args
    (
        ix,
        iy,
        lons,
        lats,
        stations_df,
        foc_depth,
        stat_num,
        snr,
        mag_min,
        mag_delta,
        arrays_df,
        obs_df,
        das_dfs,
        kwargs,
    ) = args
    min_mag = calc_min_ML_at_gridpoint(
        stations_df,
        lons[ix],
        lats[iy],
        foc_depth,
        stat_num,
        snr,
        mag_min,
        mag_delta,
        **kwargs,
    )
    # Add arrays/obs/das as in your main loop if needed
    if arrays_df is not None and not arrays_df.empty:
        if "array_num" not in kwargs:
            kwargs["array_num"] = 1
            warnings.warn("array_num not specified, using 1 as default")

        min_mag = update_with_arrays(
            min_mag,
            arrays_df,
            lons[ix],
            lats[iy],
            foc_depth,
            snr,
            mag_min,
            mag_delta,
            **kwargs,
        )
    if obs_df is not None and not obs_df.empty:
        if "obs_stat_num" not in kwargs:
            kwargs["obs_stat_num"] = 3
            warnings.warn("obs_stat_num not specified, using 3 as default")

        min_mag = update_with_obs(
            min_mag,
            obs_df,
            lons[ix],
            lats[iy],
            foc_depth,
            snr,
            mag_min,
            mag_delta,
            **kwargs,
        )

    for das_df in das_dfs:
        if das_df is not None and not das_df.empty:
            min_mag = update_with_das(
                min_mag,
                das_df,
                detection_length=kwargs.get("detection_length", 1000),
                lon=lons[ix],
                lat=lats[iy],
                foc_depth=foc_depth,
                snr=snr,
                mag_min=mag_min,
                mag_delta=mag_delta,
                gmpe=kwargs.get("gmpe", None),
                gmpe_model_type=kwargs.get("gmpe_model_type", None),
                region=kwargs.get("region", "CAL"),
                method=kwargs.get("method", "ML"),
            )
    return (iy, ix, min_mag)


def minML_x_section(
    stations_in,
    lon0,
    lat0,
    azi,
    length_km,
    min_depth=0,
    max_depth=20,
    ddist=5,
    ddepth=0.5,
    stat_num=4,
    snr=3,
    region="CAL",
    mag_min=-3.0,
    mag_delta=0.1,
    arrays=None,
    obs=None,
    obs_stat_num=3,
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
        stations : DataFrame or csv filename
    """
    stations_df = read_station_data(stations_in)
    # Calculate lon/lat co-ordinates for X-section line
    ndists = int((length_km / ddist) + 1)
    distance_km = np.linspace(0, length_km, ndists)
    xsection = pygc.great_circle(
        latitude=lat0, longitude=lon0, azimuth=azi, distance=distance_km * 1e3
    )
    ndepths = int((max_depth - min_depth) / ddepth) + 1
    depths = np.linspace(min_depth, max_depth, ndepths)
    # Iterate along cross-section
    mag = []
    array_mag = []
    obs_mag = []
    # dets = {'Distance_km': [], 'Depth_km': [], 'ML_min':[]}
    mag_grid = np.zeros((ndepths, ndists))
    for i in range(0, ndists):
        # get lat/lon of each point on line
        ilat = xsection["latitude"][i]
        ilon = xsection["longitude"][i]
        # Iterate over depth
        for d in range(ndepths):
            mag_grid[d, i] = calc_min_ML_at_gridpoint(
                stations_df,
                ilon,
                ilat,
                depths[d],
                stat_num,
                snr,
                mag_min,
                mag_delta,
                **kwargs,
            )
            mag_grid[d, i] = mag[stat_num - 1]
            # add array bit
            if arrays:
                for a in range(0, len(arrays["lon"])):
                    dx, dy = util_geo_km(ilon, ilat, arrays["lon"][a], arrays["lat"][a])
                    dz = np.abs(arrays["elevation_km"][a] - depths[d])
                    hypo_dist = np.sqrt(dx**2 + dy**2 + dz**2)
                    m = _est_min_ML_at_station(
                        arrays["noise"][a],
                        mag_min,
                        mag_delta,
                        hypo_dist,
                        snr,
                        method=kwargs["method"],
                        gmpe=kwargs["gmpe"],
                        gmpe_model_type=kwargs["gmpe_model_type"],
                        region=kwargs["region"],
                    )
                    array_mag.append(m)
                if np.min(array_mag) < mag_grid[d, i]:
                    mag_grid[d, i] = np.min(array_mag)

            if obs:
                for o in range(0, len(obs["longitude"])):
                    dz = np.abs(obs["elevation_km"][o] - depths[d])
                    dx, dy = util_geo_km(
                        ilon, ilat, obs["longitude"][o], obs["latitude"][o]
                    )
                    hypo_dist = np.sqrt(dx**2 + dy**2 + dz**2)
                    # estimated noise level on array
                    # rootn or another cleverer method
                    # to get a displaement number)
                    m = _est_min_ML_at_station(
                        obs["noise [nm]"][o],
                        mag_min,
                        mag_delta,
                        hypo_dist,
                        snr,
                        method=kwargs["method"],
                        gmpe=kwargs["gmpe"],
                        gmpe_model_type=kwargs["gmpe_model_type"],
                        region=kwargs["region"],
                    )
                    obs_mag.append(m)
                if obs_mag[obs_stat_num - 1] < mag_grid[d, i]:
                    mag_grid[d, i] = obs_mag[obs_stat_num - 1]

            del array_mag[:]
            del mag[:]
            del obs_mag[:]

    # Make xarray grid to output

    array = xarray.DataArray(
        mag_grid,
        coords=[depths, distance_km],
        dims=["depth_km", "distance_along_xsection_km"],
    )
    return array


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


def get_das_noise_levels(channel_pos, noise, detection_length, slide_length=1):
    """
    Gets the maximum seismic noise level (in displacement) along
    a given continuous fibre length.
    Parameters
    ----------
    channel_pos : np.ndarray
        position of channels along the fibre in metres.
    noise : float
        Noise level of the fibre in metres.
    detection_length : float
        Length of the fibre over which to calculate the noise level in metres.
    slide_length : float, optional
        Length to slide the detection window along the fibre in metres.
        Default is 1 metre.
    Returns
    -------
    np.ndarray
        Array of noise levels for each section of the fibre.
    """
    fibre_length = channel_pos[-1] - channel_pos[0]
    if detection_length > fibre_length:
        raise ValueError(
            f"detection_length {detection_length:4.2f} must be less "
            + f"than fibre_length {fibre_length:4.2f}"
        )
    if detection_length <= 0:
        raise ValueError(f"detection_length {detection_length} must be positive")

    start_length = channel_pos[0]
    # Calculate the number of sections along the fibre
    noise_at_sections = np.zeros(channel_pos.shape)
    for i in range(len(channel_pos)):
        idx = np.argwhere(
            (channel_pos >= start_length + i * slide_length)
            & (channel_pos < start_length + detection_length + i * slide_length)
        ).flatten()
        noise_at_sections[i] = np.max(noise[idx])

    return noise_at_sections


def get_min_ML_for_das_section(channel_pos, mags, detection_length, slide_length=1):
    """
    Gets the earthquake magntiude which is detectable across a given
    continuous fibre length (detection_length).
    Parameters
    ----------
    channel_pos : np.ndarray
        position of channels along the fibre in metres.
    mags : array
        Noise level of the fibre in metres.
    detection_length : float
        Length of the fibre over which to calculate the noise level in metres.
    slide_length : float, optional
        Length to slide the detection window along the fibre in metres.
        Default is 1 metre.
    Returns
    -------
    np.ndarray
        Array of noise levels for each section of the fibre.
    """
    fibre_length = channel_pos[-1] - channel_pos[0]
    if detection_length > fibre_length:
        raise ValueError(
            f"detection_length {detection_length} must be less"
            + f" than fibre_length {fibre_length}"
        )
    if detection_length <= 0:
        raise ValueError(f"detection_length {detection_length} must be positive")

    # Calculate the number of windows along the fibre
    n_windows = (
        int(
            np.floor(
                (channel_pos[-1] - channel_pos[0] - detection_length) / slide_length
            )
        )
        + 1
    )
    ml_at_windows = np.zeros(n_windows)
    mags = np.array(mags)  # Convert to NumPy array for advanced indexing

    return np.min(ml_at_windows)


def calc_min_ML_at_gridpoint(
    stations_df,
    lon,
    lat,
    foc_depth,
    stat_num,
    snr,
    mag_min,
    mag_delta,
    **kwargs,
):
    """

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
        Minimum local magnitude to consider when modelling detections.
    mag_delta : float
        Increment for local magnitude.
    kwargs : dict, optional
        Additional keyword arguments to control the method and parameters:
        - method: 'ML' or 'GMPE'. Default is 'ML'.
        - gmpe: GMPE model to use if method is 'GMPE'. Default is None.
        - gmpe_model_type: Type of GMPE model to use if method is 'GMPE'.
                           Default is None.
        - region: Locality for assumed ML scale parameters ('UK' or 'CAL').
                  Default is 'CAL'.
    Returns
    -------
    float
        Minimum local magnitude that can be detected at the grid point.
    """
    method = kwargs.get("method", "ML")
    region = kwargs.get("region", "CAL")

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
            _est_min_ML_at_station(
                noise[s],
                mag_min,
                mag_delta,
                hypo_dist[s],
                snr,
                method=kwargs["method"],
                gmpe=kwargs["gmpe"],
                gmpe_model_type=kwargs["gmpe_model_type"],
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


def calc_min_ML_at_gridpoint_das(
    fibre, detection_length, lon, lat, foc_depth, snr, **kwargs
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
    detection_length : float
        Length of the fibre over which to calculate the noise level in metres.
    lon : float
        Longitude of the grid point in decimal degrees.
    lat : float
        Latitude of the grid point in decimal degrees.
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

    if method != "ML":
        raise ValueError(f"Method: {method} not supported for DAS at this time")

    gauge_len = kwargs.get("gauge_length", 20)
    window_size = int(np.ceil((detection_length / gauge_len)))
    # print("~" * 50)
    # print(f"There are {window_size} channels in the sliding window.")
    # Covert noise from metres to nanometres
    noise_nm = fibre["noise_m"].values * 1e9
    # print(f"This improves mean noise level from {noise_nm.mean():4.2f}")
    noise_nm = noise_nm / np.sqrt(window_size)
    # print(f"To a mean noise level of {noise_nm.mean():4.2f} ")
    # print("~" * 50)
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
    # Vectorize _est_min_ML_at_station if possible
    # Otherwise, use a generator expression for min
    required_ampls = snr * noise_nm
    mags = calc_local_magnitude(
        required_ampls,
        hypo_distances,
        region=region,
        mag_min=mag_min,
        mag_delta=mag_delta,
    )
    # Take a rolling maximum filter along fiber
    min_windowed_mag = maximum_filter1d(mags, size=window_size, mode="nearest")
    # Get smallest ML detected at any one window along the fiber
    return np.min(min_windowed_mag)


def update_with_arrays(
    mag_grid_val,
    arrays_df,
    lon,
    lat,
    foc_depth,
    snr,
    mag_min,
    mag_delta,
    **kwargs,
):
    """
    Update the grid value with the minimum ML from arrays, if lower.
    """

    mag_arrays = calc_min_ML_at_gridpoint(
        arrays_df,
        lon,
        lat,
        foc_depth,
        kwargs["array_num"],
        snr,
        mag_min,
        mag_delta,
        method=kwargs["method"],
        gmpe=kwargs["gmpe"],
        gmpe_model_type=kwargs["gmpe_model_type"],
        region=kwargs["region"],
    )
    return min(mag_grid_val, mag_arrays)


def update_with_obs(
    mag_grid_val,
    obs_df,
    lon,
    lat,
    foc_depth,
    snr,
    mag_min,
    mag_delta,
    **kwargs,
):
    """
    Update the grid value with the minimum ML from OBS, if lower.
    """
    mag_obs = calc_min_ML_at_gridpoint(
        obs_df,
        lon,
        lat,
        foc_depth,
        kwargs["obs_stat_num"],
        snr,
        mag_min,
        mag_delta,
        **kwargs,
    )
    return min(mag_grid_val, mag_obs)


def update_with_das(
    mag_grid_val,
    das_df,
    detection_length,
    lon,
    lat,
    foc_depth,
    snr,
    mag_min,
    mag_delta,
    **kwargs,
):
    """
    Update the grid value with the minimum ML from DAS, if lower.
    """
    mag_das = calc_min_ML_at_gridpoint_das(
        das_df,
        detection_length,
        lon,
        lat,
        foc_depth,
        snr,
        mag_min=mag_min,
        mag_delta=mag_delta,
        **kwargs,
    )
    return min(mag_grid_val, mag_das)
