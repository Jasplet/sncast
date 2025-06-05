#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: sncast.py
# Purpose:  Seismic Network Capability Assessment Software Tool (SNCAST)
# Author:   Martin Möllhoff, DIAS
# Citation: Möllhoff, M., Bean, C.J. & Baptie, B.J.,
#           SN-CAST: seismic network capability assessment software tool
#           for regional networks - examples from Ireland.
#           J Seismol 23, 493-504 (2019).
#           https://doi.org/10.1007/s10950-019-09819-0
#
#    Copyright (C) 2019 Martin Möllhoff
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#    email:       martin@dias.ie
#    snail mail:  Martin Möllhoff, DIAS, 5 Merrion Square, Dublin 2, Ireland
#    web:         www.dias.ie/martin  www.insn.ie
#
# --------------------------------------------------------------------
# This version of SNCAST has been refactored and extended
# to add various additional functionality such as:
#   - Implementation of BGS Local magnitude
#   - Implementation of GMPE based method
#   - Functionality to calculate of a depth cross-section
#   - Outputs model as xarray DataArray, for easier plotting
#   - Added support for seismic arrays
# Author: J Asplet
# Copyright (C) 2024 Joseph Asplet, University of Oxford
# email : joseph.asplet@earth.ox.ac.uk

import numpy as np
import pandas as pd
from obspy.signal.util import util_geo_km
from math import sqrt
import pygc
import xarray
import warnings

from .gmpes import eval_gmpe
from .magnitude_conversions import convert_ml_to_mw, convert_mw_to_ml


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
        #   Takes form local_mag = log(amp) + a*log(hypo-dist) + b*hypo-dist + d*exp(e * hypo-dist) + c
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


def _est_min_ML_at_station(
    noise,
    mag_min,
    mag_delta,
    distance,
    snr,
    method="ML",
    region="UK",
    gmpe="RE19",
    gmpe_model_type="PGV",
):

    signal = 0
    ml = mag_min - mag_delta
    while signal < snr * noise:
        ml = ml + mag_delta
        if method == "ML":
            signal = calc_ampl_from_magnitude(ml, distance, region)
        elif method == "GMPE":
            mw = convert_ml_to_mw(ml, region)
            signal = eval_gmpe(mw, distance, gmpe, model_type=gmpe_model_type)
            ml = convert_mw_to_ml(mw, region)
            if ml > 3:
                # print('Warning: ML > 3, check your input parameters')
                # print(f'{signal} {noise} {snr}')
                break
    return ml


def minML(
    stations_in,
    lon0=-12,
    lon1=-4,
    lat0=50.5,
    lat1=56.6,
    dlon=0.33,
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

    Inputs stations_in is a Pandas DataFrame which contains the following
    columns:
    - longitude: longitude of the station in decimal degrees
    - latitude: latitude of the station in decimal degrees
    - elevation_km: elevation of the station in km
    - station: station name
    - noise [nm]: noise level at the station in nanometres

    Example of the input file format:
        longitude, latitude, elevation, noise [nm], station name
        -7.5100, 55.0700, 0, 0.53, IDGL

    Model output is a 2D xarray DataArray with the following dimensions:
    - Latitude: latitude of the grid point in decimal degrees
    - Longitude: longitude of the grid point in decimal degrees
    The values in the DataArray are the minimum detectable local magnitude ML
    at that grid point.

    Optional parameters are:

    :param  lon0:	minimum longitude of search grid
    :param  lon1:	maximum longitude of search grid
    :param  lat0:	minimum latitude of search grid
    :param  lat1:	maximum latitude of search grid
    :param  dlon:	longitude increment of search grid
    :param  dlat:	latitude increment of search grid
    :param  stat_num:	required number of station detections
    :param  snr:	required signal-to-noise ratio for detection
    :param  foc_depth:  assumed focal event depth
    :param  region:	locality for assumed ML scale parameters ('UK' or 'CAL')
    :param  mag_min:	minimum ML value for grid search
    :param  mag_delta:  ML increment used in grid search
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

    print(f'Method : {kwargs["method"]}')
    print(f'Region : {kwargs["region"]}')
    # read in data, file format: "LON, LAT, NOISE [nm], STATION"
    stations_df = read_station_data(stations_in)
    # Read in arrays and obs data if provided
    arrays_df = read_station_data(arrays) if arrays is not None else None
    obs_df = read_station_data(obs) if obs is not None else None
    if len(stations_df) < stat_num:
        raise ValueError(
            f"Not enough stations ({len(stations_df)}) to calculate minimum ML at {stat_num} stations"
        )

    lons, lats, nx, ny = create_grid(lon0, lon1, lat0, lat1, dlon, dlat)
    mag_grid = np.zeros((ny, nx))
    for ix in range(nx):  # loop through longitude increments
        for iy in range(ny):  # loop through latitude increments
            # add array bit
            mag_grid[iy, ix] = calc_min_ML_at_gridpoint(
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
            if arrays is not None:
                # Assume an array will always make a detection
                if "array_num" not in kwargs:
                    kwargs["array_num"] = 1
                mag_grid[iy, ix] = update_with_arrays(
                    mag_grid[iy, ix],
                    arrays_df,
                    kwargs["array_num"],
                    lons[ix],
                    lats[iy],
                    foc_depth,
                    snr,
                    mag_min,
                    mag_delta,
                    **kwargs,
                )
            if obs is not None:
                mag_grid[iy, ix] = update_with_obs(
                    mag_grid[iy, ix],
                    obs_df,
                    lons[ix],
                    lats[iy],
                    foc_depth,
                    obs_stat_num,
                    snr,
                    mag_min,
                    mag_delta,
                    **kwargs,
                )

    # Make xarray grid to output
    array = xarray.DataArray(
        mag_grid, coords=[lats, lons], dims=["Latitude", "Longitude"]
    )
    return array


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
    X-section line defined by start lat/lon and the azimuth and length (in km) of the line

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
                    hypo_dist = sqrt(dx**2 + dy**2 + dz**2)
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
                    hypo_dist = sqrt(dx**2 + dy**2 + dz**2)
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
    required_cols = {"longitude", "latitude", "elevation_km", "noise [nm]", "station"}
    if not required_cols.issubset(stations_df.columns):
        raise ValueError(f"Missing columns: {required_cols - set(stations_df.columns)}")
    return stations_df


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
    if (lon1 - lon0) % dlon != 0:
        raise ValueError("lon1 - lon0 must be a multiple of dlon")
    if (lat1 - lat0) % dlat != 0:
        raise ValueError("lat1 - lat0 must be a multiple of dlat")

    nx = int((lon1 - lon0) / dlon) + 1
    ny = int((lat1 - lat0) / dlat) + 1
    lats = np.linspace(lat1, lat0, ny)
    lons = np.linspace(lon0, lon1, nx)
    return lons, lats, nx, ny


def calc_min_ML_at_gridpoint(
    stations_df, lon, lat, foc_depth, stat_num, snr, mag_min, mag_delta, **kwargs
):
    try:
        noise = stations_df["noise [nm]"].values
    except KeyError if kwargs["method"] == "GMPE" else KeyError:
        noise = stations_df["noise [cm/s]"].values

    mag = []
    for s in range(len(stations_df)):
        # loop through stations
        # calculate hypcocentral distance in km
        dx, dy = util_geo_km(
            lon, lat, stations_df["longitude"].iloc[s], stations_df["latitude"][s]
        )
        dz = np.abs(foc_depth - stations_df["elevation_km"][s])
        # calculate hypcocentral distance
        hypo_dist = sqrt(dx**2 + dy**2 + dz**2)
        m = _est_min_ML_at_station(
            noise[s],
            mag_min,
            mag_delta,
            hypo_dist,
            snr,
            method=kwargs["method"],
            gmpe=kwargs["gmpe"],
            gmpe_model_type=kwargs["gmpe_model_type"],
            region=kwargs["region"],
        )
        mag.append(m)
    # sort magnitudes in ascending order
    mag = sorted(mag)
    return mag[stat_num - 1]


def update_with_arrays(
    mag_grid_val,
    arrays_df,
    array_num,
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
        array_num,
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
    obs_stat_num,
    snr,
    mag_min,
    mag_delta,
    **kwargs,
):
    """
    Update the grid value with the minimum ML from OBS, if lower.
    """
    mag_obs = calc_min_ML_at_gridpoint(
        obs_df, lon, lat, foc_depth, obs_stat_num, snr, mag_min, mag_delta, **kwargs
    )
    return min(mag_grid_val, mag_obs)
