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
from decimal import Decimal
import warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool
from obspy.signal.util import util_geo_km

import pygc
import xarray


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


def _est_min_ML_at_station(noise, mag_min, mag_delta, distance, snr, **kwargs):
    method = kwargs.get("method", "ML")
    region = kwargs.get("region", "CAL")
    gmpe = kwargs.get("gmpe", None)
    gmpe_model_type = kwargs.get("gmpe_model_type", None)
    required_ampl = snr * noise
    if method == "ML":
        if region == "UK":
            a = 1.11
            b = 0.00189
            c = -2.09
            d = -1.16
            e = -0.2
            ml = (
                np.log10(required_ampl)
                + a * np.log10(distance)
                + b * distance
                + c
                + d * np.exp(e * distance)
            )
        elif region == "CAL":
            a = 1.11
            b = 0.00189
            c = -2.09
            ml = np.log10(required_ampl) + a * np.log10(distance) + b * distance + c
        else:
            raise ValueError(f"Unknown region: {region}")
        # Snap to nearest mag_delta step above mag_min
        ml = max(mag_min, np.ceil((ml - mag_min) / mag_delta) * mag_delta + mag_min)
        return ml
    elif method == "GMPE":
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
        - gmpe_model_type: Type of GMPE model to use if method is 'GMPE'. Default is None.
        - region: Locality for assumed ML scale parameters ('UK' or 'CAL'). Default is 'CAL'.
        - das: Path to a CSV file or a DataFrame containing DAS noise data.
        - detection_length: Length of the fibre over which to calculate the noise level in metres.
                           Default is 1 km.
        - slide_length: Length to slide the detection window along the fibre in metres. Default is 1 m.

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
    # read in data, file format: "LON, LAT, NOISE [nm], STATION"
    stations_df = read_station_data(stations_in)
    # Read in arrays and obs data if provided
    arrays_df = read_station_data(arrays) if arrays is not None else None
    obs_df = read_station_data(obs) if obs is not None else None
    if len(stations_df) < stat_num:
        raise ValueError(
            f"Not enough stations ({len(stations_df)}) to calculate minimum ML at {stat_num} stations"
        )

    if "das" in kwargs:
        if "detection_length" not in kwargs:
            warnings.warn("Detection length not specified, using default of 1.0 km")
            kwargs["detection_length"] = 1e3  # Default to 1 km
        if "slide_length" not in kwargs:
            warnings.warn("Slide length not specified, using default of 1.0 m")
            kwargs["slide_length"] = 1.0
        # Read in DAS noise data
        das_in = kwargs["das"]
        das_df = read_das_noise_data(das_in)
        if len(das_df) == 0:
            raise ValueError("No DAS data found in the input file")
        print(f'DAS detection length: {kwargs["detection_length"]} m')
        print(f'DAS slide length: {kwargs["slide_length"]} m')
    else:
        das_df = None

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
            obs_stat_num,
            das_df,
            kwargs,
        )
        for ix in range(nx)
        for iy in range(ny)
    ]
    mag_grid = np.zeros((ny, nx))

    with Pool(processes=nproc) as pool:
        for iy, ix, val in pool.imap_unordered(_minML_worker, args_list):
            mag_grid[iy, ix] = val
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
            # Add array detection if arrays_df is provided and not empty
            if arrays_df is not None and not arrays_df.empty:
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
            # Add OBS detection if obs_df is provided and not empty
            if obs_df is not None and not obs_df.empty:
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
            # Add DAS detection if 'das' is in kwargs and das_df is not empty
            if das_df is not None and not das_df.empty:
                # Remove 'detection_length' from kwargs to avoid multiple values error
                # kwargs_no_dl = dict(kwargs)
                # if "detection_length" in kwargs_no_dl:
                #     del kwargs_no_dl["detection_length"]
                mag_grid[iy, ix] = update_with_das(
                    mag_grid[iy, ix],
                    das_df,
                    detection_length=kwargs["detection_length"],
                    lon=lons[ix],
                    lat=lats[iy],
                    foc_depth=foc_depth,
                    snr=snr,
                    mag_min=mag_min,
                    mag_delta=mag_delta,
                    gmpe=kwargs["gmpe"],
                    gmpe_model_type=kwargs["gmpe_model_type"],
                    region=kwargs["region"],
                    method=kwargs["method"],
                )

    # Make xarray grid to output
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
        obs_stat_num,
        das_df,
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
        min_mag = update_with_arrays(
            min_mag,
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
    if obs_df is not None and not obs_df.empty:
        min_mag = update_with_obs(
            min_mag,
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
    if das_df is not None and not das_df.empty:
        min_mag = update_with_das(
            min_mag,
            das_df,
            detection_length=kwargs["detection_length"],
            lon=lons[ix],
            lat=lats[iy],
            foc_depth=foc_depth,
            snr=snr,
            mag_min=mag_min,
            mag_delta=mag_delta,
            gmpe=kwargs["gmpe"],
            gmpe_model_type=kwargs["gmpe_model_type"],
            region=kwargs["region"],
            method=kwargs["method"],
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
    required_cols = {"longitude", "latitude", "elevation_km", "noise [nm]", "station"}
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


def calc_min_ML_at_gridpoint(
    stations_df, lon, lat, foc_depth, stat_num, snr, mag_min, mag_delta, **kwargs
):
    try:
        noise = stations_df["noise [nm]"].values
    except KeyError if kwargs["method"] == "GMPE" else KeyError:
        noise = stations_df["noise [cm/s]"].values

    mag = []

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

    for s in range(len(stations_df)):
        # loop through stations
        # calculate hypcocentral distance in km
        # Use pygc to compute great-circle distance in meters, then convert to km

        m = _est_min_ML_at_station(
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
        mag.append(m)
    # sort magnitudes in ascending order
    mag = sorted(mag)
    return mag[stat_num - 1]


def get_das_noise_levels(channel_pos, noise, detection_length, slide_length=1):
    """
    Gets the maximum seismic noise level (in displacement) along
    a given continuous fibre length.
    Ppos)):arameters
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
            f"detection_length {detection_length} must be less than fibre_legth {fibre_legth}"
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
            f"detection_length {detection_length} must be less than fibre_legth {fibre_legth}"
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
    for i in range(n_windows):
        window_start = channel_pos[0] + i * slide_length
        window_end = window_start + detection_length
        idx = np.where((channel_pos >= window_start) & (channel_pos < window_end))[0]
        if len(idx) > 0:
            ml_at_windows[i] = np.max(mags[idx])
        else:
            ml_at_windows[i] = np.nan  # or another fill value if no channels in window

    return np.min(ml_at_windows)


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
    if "slide_length" not in kwargs:
        slide_length = 1.0
    else:
        slide_length = kwargs["slide_length"]
    # Get the noise levels for the given lengths of fibre
    # noise_at_sections = get_das_noise_levels(
    #     fibre["fiber_length_m"].values,
    #     fibre["noise_m"].values,
    #     detection_length,
    #     slide_length,
    # )
    # Covert noise from metres to nanometres
    noise_nm = fibre["noise_m"].values * 1e9
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
    mags = [
        _est_min_ML_at_station(
            noise_nm[d],
            kwargs["mag_min"],
            kwargs["mag_delta"],
            hypo_distances[d],
            snr,
            method=kwargs["method"],
            gmpe=kwargs["gmpe"],
            gmpe_model_type=kwargs["gmpe_model_type"],
            region=kwargs["region"],
        )
        for d in range(len(hypo_distances))
    ]
    min_windowed_mag = get_min_ML_for_das_section(
        fibre["fiber_length_m"].values, mags, detection_length, slide_length
    )
    return min_windowed_mag


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
