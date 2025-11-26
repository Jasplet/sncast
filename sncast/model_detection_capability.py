#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2019 Martin Möllhoff
# SPDX-FileCopyrightText: 2024 Joseph Asplet, University of Oxford
# ------------------------------------------------------------------
"""
Filename: model_detection_capability.py

Purpose:  Calculate the minimum earthquake detection capability of seismic networks

Author:   Joseph Asplet (refactor of original code by Martin Möllhoff)

Changelog:
    - Implementation of GMPE based method (still in development [Joseph Asplet, 2025]
    - Implementation of BGS Local magnitude scale, [Joseph Asplet, 2024]
    - Functionality to calculate of a depth cross-section [Joseph Asplet, 2024]
    - Outputting of models as xarray.DataArray objects for easier plotting with
      PyGMT [Joseph Asplet, 2024]
    - Added support for seismic arrays and OBS with separate
      detection requirements [Joseph Asplet, 2024]
    - Refactor and re-write of entire codebase, [Joseph Asplet, 2025]
    - Added support for DAS deployments [Joseph Asplet, 2025]

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
"""

from decimal import Decimal
from functools import partial
from multiprocessing import Pool
import warnings

import numpy as np
from scipy.ndimage import maximum_filter1d
import xarray

import pygc

from .gmpes import eval_gmpe
from .magnitude_conversions import convert_ml_to_mw, convert_mw_to_ml
from .core import SeismicNetwork, SeismicArrayNetwork, ModelConfig, DASFibre

ML_COEFFS = {
    "UK": {"a": 1.11, "b": 0.00189, "c": -2.09, "d": -1.16, "e": -0.2},
    "CAL": {"a": 1.11, "b": 0.00189, "c": -2.09},
}

SUPPORTED_ML_REGIONS = list(ML_COEFFS.keys())


class DetectionCapabilityModel:
    """
    A class to represent and model the earthqauake detection capability
    of seismic networks based on station noise levels and network geometry.
    """

    def __init__(self, **kwargs):
        """
        Initialize a DetectionCapabilityModel instance.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to control the method and parameters:
            - method: 'ML' or 'GMPE'. Default is "ML".
            - region: Locality for assumed ML scale parameters ("UK" or "CAL"). Default is "CAL".
        """
        self.config = ModelConfig(**kwargs)
        self.networks = []
        self.n_networks = 0
        self.arrays = []
        self.n_arrays = 0
        self.das_fibres = []
        self.n_das_fibres = 0
        print("Detection Capability Model initialized.")

    def __repr__(self):
        return f"<DetectionCapabilityModel with {len(self.networks)} networks, {len(self.arrays)} arrays, and {len(self.das_fibres)} DAS fibres>"

    def add_network(self, network):
        """
        Add a seismic network to the model.
        """
        if isinstance(network, SeismicNetwork):
            self.networks.append(network)
            self.n_networks += 1
            print(f"Seismic network {network.network_code} added to model.")
        else:
            net_to_add = SeismicNetwork(stations=network)
            self.networks.append(net_to_add)
            self.n_networks += 1
            print(
                f"Seismic network {net_to_add.network_code} created and added to model."
            )

    def add_array(self, array):
        """
        Add a seismic array to the model.
        """
        if isinstance(array, SeismicArrayNetwork):
            self.arrays.append(array)
            self.n_arrays += 1
            print(f"Seismic array {array.array_code} added to model.")
        else:
            arr_to_add = SeismicArrayNetwork(arrays=array)
            self.arrays.append(arr_to_add)
            self.n_arrays += 1
            print(f"Seismic array {arr_to_add.array_code} created and added to model.")

    def add_das_fibre(self, das_fibre):
        """
        Add a DAS fibre to the model.
        """
        if isinstance(das_fibre, DASFibre):
            self.das_fibres.append(das_fibre)
            self.n_das_fibres += 1
            print(f"DAS fibre {das_fibre.fibre_code} added to model.")
        else:
            das_to_add = DASFibre(fibres=das_fibre)
            self.das_fibres.append(das_to_add)
            self.n_das_fibres += 1
            print(f"DAS fibre {das_to_add.fibre_code} created and added to model.")

    def setup_grid(self, lon0, lon1, lat0, lat1, dlon=0.1, dlat=0.1):
        """
        Setup the geographic grid for the detection capability model.

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
            Longitude increment for the grid. Default is 0.1
        dlat : float
            Latitude increment for the grid. Default is 0.1
        """
        self.config.add_grid_params(lon0, lon1, lat0, lat1, dlon, dlat)

    def setup_xsection(
        self,
        lon0,
        lat0,
        azi,
        length_km,
        ddist_km,
        min_depth_km,
        max_depth_km,
        ddepth_km,
    ):
        """
        Setup the cross-section grid for the detection capability model.

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
        ddist_km : float
            Distance increment along the cross-section in km.
        min_depth_km : float
            Minimum depth of cross-section in km.
        max_depth_km : float
            Maximum depth of cross-section in km.
        ddepth_km : float
            Depth increment along the cross-section in km.
        """
        self.config.add_xsection_params(
            lon0,
            lat0,
            azi,
            length_km,
            ddist_km,
            min_depth_km,
            max_depth_km,
            ddepth_km,
        )

    def _make_model_kwargs(self):
        """
        Makes kwargs dict from Config for passing to model functions.
        """
        model_kwargs = {
            "lon0": self.config.lon0,
            "lon1": self.config.lon1,
            "lat0": self.config.lat0,
            "lat1": self.config.lat1,
            "dlon": self.config.dlon,
            "dlat": self.config.dlat,
            "foc_depth": self.config.foc_depth_km,
            "snr": self.config.snr,
            "mag_min": self.config.mag_min,
            "mag_delta": self.config.mag_delta,
            "method": self.config.method,
            "region": self.config.region,
            "nproc": getattr(self.config, "nproc", 1),
            "model_stacking_das": getattr(self.config, "model_stacking_das", True),
        }
        if self.config.method == "GMPE":
            model_kwargs["gmpe"] = self.config.gmpe
            model_kwargs["gmpe_model_type"] = self.config.gmpe_model_type
        if self.n_networks > 0:
            model_kwargs["networks"] = self.networks
        else:
            print("No seismic networks provided to model.")
        if self.n_arrays > 0:
            model_kwargs["arrays"] = self.arrays
        else:
            print("No seismic arrays provided to model.")
        if self.n_das_fibres > 0:
            model_kwargs["das_fibres"] = self.das_fibres
        else:
            print("No DAS fibres provided to model.")

        return model_kwargs

    def run_model(self):
        """
        Run the detection capability model over a specified geographic region.

        Returns
        -------
        mag_det : xarray.DataArray
            A 2D xarray DataArray with the dimensions in Latitude and Longitude.
            The values in the DataArray are the minimum detectable local magnitude ML
            at that grid point.
        """
        # exit if no stations provided
        if (self.n_networks == 0) and (self.n_arrays == 0) and (self.n_das_fibres == 0):
            raise ValueError("No seismic networks, arrays or DAS provided!")

        model_kwargs = self._make_model_kwargs()
        mag_det = find_min_ml(
            **model_kwargs,
        )
        return mag_det


def find_min_ml(**model_kwargs):
    """
    This routine calculates the geographic distribution of the minimum
    detectable local magnitude ML for a given seismic network.


    Example of the input file format for stations and arrays:
        longitude, latitude, elevation_km, noise [nm], station

    Example of the input file format for DAS:
        channel_index, fiber_length_m, longitude, latitude, noise_m, elevation_km

    Parameters
    ----------
    model_kwargs : dict
        A dictionary of keyword arguments for the model including
        longitude and latitude bounds, grid increments, noise dataframes, and other model parameters.

    Returns
    -------
    mag_det : xarray.DataArray
        A 2D xarray DataArray with the dimensions in Latitude and Longitude.
        The values in the DataArray are the minimum detectable local magnitude ML
        at that grid point.
    """

    # Initialize grid
    lons, lats, nx, ny = create_grid(
        model_kwargs["lon0"],
        model_kwargs["lon1"],
        model_kwargs["lat0"],
        model_kwargs["lat1"],
        model_kwargs["dlon"],
        model_kwargs["dlat"],
    )

    # Ensure fixed args are all in kwargs for worker function
    args_list = [
        (ilat, ilon, lats[ilat], lons[ilon]) for ilat in range(ny) for ilon in range(nx)
    ]

    mag_grid = np.zeros((ny, nx))
    # Detection capability has to be calculated at each grid point,
    # Split this up using Pool and imap_unordered to multiple cores
    # maybe numba would be quicker here?
    print(f"Using {model_kwargs['nproc']} cores")
    worker_func = partial(_minml_worker, **model_kwargs)
    with Pool(processes=model_kwargs["nproc"]) as pool:
        for iy, ix, val in pool.imap_unordered(worker_func, args_list):
            mag_grid[iy, ix] = val

    # Make xarray grid to output
    mag_det = xarray.DataArray(
        mag_grid, coords=[lats, lons], dims=["Latitude", "Longitude"]
    )
    return mag_det


def _minml_worker(grid_point, **kwargs):
    """
    Worker function for minML which allows the magnitude grid to be parallelised
    over multiple processors using multiprocessing.Pool

    Parameters
    ----------
    grid_point : tuple
        Tuple containing x,y index of grid point and the latitude/longitude

    **kwargs : dict
        Additional keyword arguments to pass to the worker function.

    Returns
    -------
    tuple
        Tuple containing the y-index, x-index, and minimum magnitude for the grid point.
    """
    # Initialize min_mag to absurdly high value
    min_mag = 100.0
    ilat = grid_point[0]
    ilon = grid_point[1]
    lat = grid_point[2]
    lon = grid_point[3]

    if "networks" in kwargs:
        for Network in kwargs["networks"]:
            print(
                f"Calculating min ML at grid point for Seismic Network {Network.network_code}"
            )
            # spell out kwargs here for clarify and to avoid passing
            # unnecessary data to worker processes
            min_mag_net = calc_min_ml_at_gridpoint(
                lon=lon,
                lat=lat,
                stations_df=Network.stations,
                stat_num=Network.required_detections,
                foc_depth=kwargs["foc_depth"],
                snr=kwargs["snr"],
                mag_min=kwargs["mag_min"],
                mag_delta=kwargs["mag_delta"],
                method=kwargs["method"],
                region=kwargs["region"],
                gmpe=kwargs["gmpe"],
                gmpe_model_type=kwargs["gmpe_model_type"],
            )
            min_mag = min(min_mag, min_mag_net)
    # Add arrays if provided
    if "arrays" in kwargs:
        for Array in kwargs["arrays"]:
            min_mag_arrays = calc_min_ml_at_gridpoint(
                lon=lon,
                lat=lat,
                stations_df=Array.stations,
                stat_num=Array.required_detections,
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

    if "das_fibres" in kwargs:
        for Fibre in kwargs["das_fibres"]:
            mag_min_das = calc_min_ml_at_gridpoint_das(
                lon=lon,
                lat=lat,
                fibre=Fibre.das_channels,
                detection_length_m=Fibre.detection_length_m,
                gauge_length_m=Fibre.gauge_length_m,
                foc_depth=kwargs["foc_depth"],
                snr=kwargs["snr"],
                mag_min=kwargs["mag_min"],
                mag_delta=kwargs["mag_delta"],
                method=kwargs["method"],
                region=kwargs["region"],
                gmpe=kwargs["gmpe"],
                gmpe_model_type=kwargs["gmpe_model_type"],
                model_stacking=kwargs["model_stacking_das"],
            )

            min_mag = min(min_mag, mag_min_das)
    return (ilat, ilon, min_mag)


def find_min_ml_x_section(
    model_kwargs,
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
            A 2D xarray DataArray with dimenstions in depth (km) and distance (km) along the cross-section
            from the start point.
            The values in the DataArray are the minimum detectable local magnitude ML
            at that grid point.
    """

    xsection, depths, distance_km = create_xsection_grid(
        model_kwargs["lon0"],
        model_kwargs["lat0"],
        model_kwargs["azi"],
        model_kwargs["length_km"],
        model_kwargs["ddist"],
        model_kwargs["min_depth"],
        model_kwargs["max_depth"],
        model_kwargs["ddepth"],
    )
    ndists = len(xsection["longitude"])
    ndepths = len(depths)
    # Iterate along cross-section
    args_list = [
        (ix, iz, xsection["latitude"][ix], xsection["longitude"][ix], depths[iz])
        for ix in range(ndists)
        for iz in range(ndepths)
    ]
    mag_grid = np.zeros((ndepths, ndists))

    print(f"Using {model_kwargs['nproc']} cores")
    worker_func = partial(_minml_x_section_worker, **model_kwargs)
    with Pool(processes=model_kwargs["nproc"]) as pool:
        for iz, ix, val in pool.imap_unordered(worker_func, args_list):
            mag_grid[iz, ix] = val

    array = xarray.DataArray(
        mag_grid,
        coords=[depths, distance_km],
        dims=["depth_km", "distance_along_xsection_km"],
    )
    return array


def _minml_x_section_worker(
    grid_point,
    **kwargs,
):
    """
    Parallel worker for cross-section calculations.

    This version is designed to be used with functools.partial so that
    shared configuration parameters are bound once, and the worker only
    receives the per-point tuple.

    Parameters
    ----------
    grid_point : tuple
        (ix, iz, lon, lat, depth_km)
    **kwargs : dict
        Additional keyword arguments to pass to the worker function.
    Returns
    -------
    tuple
        (iz, ix, min_mag)
    """
    ix, iz, lon, lat, depth = grid_point

    # Initialize min_mag to absurdly high value
    min_mag = 100.0

    # Handle networks
    if "networks" in kwargs:
        # Normalize inputs to bare DataFrames + required detections
        for Network in kwargs["networks"]:
            min_mag_net = calc_min_ml_at_gridpoint(
                lon=lon,
                lat=lat,
                stations_df=Network.stations,
                stat_num=Network.required_detections,
                foc_depth=depth,
                snr=kwargs["snr"],
                mag_min=kwargs["mag_min"],
                mag_delta=kwargs["mag_delta"],
                method=kwargs["method"],
                region=kwargs["region"],
                gmpe=kwargs["gmpe"],
                gmpe_model_type=kwargs["gmpe_model_type"],
            )
            min_mag = min(min_mag, min_mag_net)

    # Handle arrays
    if "arrays" in kwargs:
        for Array in kwargs["arrays"]:
            min_mag_arrays = calc_min_ml_at_gridpoint(
                lon=lon,
                lat=lat,
                stations_df=Array.stations,
                stat_num=Array.required_detections,
                foc_depth=depth,
                snr=kwargs["snr"],
                mag_min=kwargs["mag_min"],
                mag_delta=kwargs["mag_delta"],
                method=kwargs["method"],
                region=kwargs["region"],
                gmpe=kwargs["gmpe"],
                gmpe_model_type=kwargs["gmpe_model_type"],
            )
            min_mag = min(min_mag, min_mag_arrays)

    # Handle DAS fibres
    if kwargs["das_fibres"] is not None:
        for Fibre in kwargs["das_fibres"]:
            mag_min_das = calc_min_ml_at_gridpoint_das(
                lon=lon,
                lat=lat,
                fibre=Fibre.das_channels,
                detection_length_m=Fibre.detection_length_m,
                gauge_length_m=Fibre.gauge_length_m,
                model_stacking=kwargs["model_stacking_das"],
                foc_depth=depth,
                snr=kwargs["snr"],
                mag_min=kwargs["mag_min"],
                mag_delta=kwargs["mag_delta"],
                method=kwargs["method"],
                region=kwargs["region"],
                gmpe=kwargs["gmpe"],
                gmpe_model_type=kwargs["gmpe_model_type"],
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
    lons : np.ndarray
        Array of longitudes for the grid.
    lats : np.ndarray
        Array of latitudes for the grid.
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
    print(f"Grid created with {nx} x {ny} points.")
    return lons, lats, nx, ny


def create_xsection_grid(
    lon0, lat0, azi, length_km, ddist, min_depth, max_depth, ddepth
):
    """
    Create a 2-D grid for a cross-section defined by a start lat/lon,
    azimuth and length (in km) of the line.

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
        ddist : float
            Distance increment along the cross-section in km.
        min_depth : float
            Minimum depth of cross-section in km.
        max_depth : float
            Maximum depth of cross-section in km.
        ddepth : float
            Depth increment along the cross-section in km.
    """
    # Calculate lon/lat co-ordinates for X-section line
    ndists = int((length_km / ddist) + 1)
    distance_km = np.linspace(0, length_km, ndists)
    xsection = pygc.great_circle(
        latitude=lat0, longitude=lon0, azimuth=azi, distance=distance_km * 1e3
    )
    ndepths = int((max_depth - min_depth) / ddepth) + 1
    depths = np.linspace(min_depth, max_depth, ndepths)
    return xsection, depths, distance_km


def calc_min_ml_at_gridpoint(
    lon,
    lat,
    stations_df,
    stat_num,
    foc_depth,
    snr,
    mag_min,
    mag_delta,
    method,
    region,
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
        if "gmpe" not in kwargs or kwargs["gmpe"] is None:
            raise ValueError("GMPE model must be specified for GMPE method")
        if "gmpe_model_type" not in kwargs or kwargs["gmpe_model_type"] is None:
            raise ValueError("GMPE model type must be specified for GMPE method")

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
    lon,
    lat,
    fibre,
    detection_length_m,
    gauge_length_m,
    model_stacking,
    foc_depth,
    snr,
    mag_min,
    mag_delta,
    method,
    region,
    **kwargs,
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

    if method != "ML":
        raise ValueError(f"Method: {method} not supported for DAS at this time")

    window_size = int(np.ceil((detection_length_m / gauge_length_m)))
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


def calc_ampl_from_magnitude(local_mag, hypo_dist, region):
    """
    Calculate the amplitude of a seismic signal given a local magnitude
    and hypocentral distance. Local magnitude scales for the UK and California
    [Hutton1987]_ are supported. The [Hutton1987]_ scale is
    the default ML scale reccomended by the IASPEI working group on earthquake magnitude
    determination and is consistent with the magnitude of [Richter1935]_.

    Parameters
    ----------
    local_mag : float, np.ndarray
        Local magnitude to calculate displacement amplitude for.
    hypo_dist : float, np.ndarray
        Hypocentral distance in km.
    region : str
        Regional ML scale to use. "UK" for [Luckett2019]_ UK scale, "CAL" for
        [Hutton1987]_ California scale.

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
        Regional ML scale to use. "UK" for [Luckett2019]_ UK scale, "CAL" for
        [Hutton1987]_ California scale.
    mag_min : float
        Minimum magnitude to consider.
    mag_delta : float
        Magnitude bin width.

    Returns
    -------
    ml: np.ndarray
        Local magnitudes (ML) for the given amplitudes and distances.
    """
    if np.any(required_ampl <= 0):
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
        - region:         Regional ML scale to use. "UK" for [Luckett2019]_ UK scale, "CAL" for
                          [Hutton1987]_ California scale. Default is "CAL".
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
