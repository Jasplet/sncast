#! /usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Joseph Asplet, University of Oxford
# ------------------------------------------------------------------
"""
Filename : core.py

Purpose : High-level core Classes for SNACT

Author : Joseph Asplet, University of Oxford

Email : joseph.asplet@earth.ox.ac.uk
"""

import pandas as pd

SUPPORTED_METHODS = ["ML", "GMPE"]
SUPPORTED_REGIONS = ["UK", "CAL"]
SUPPORTED_GMPES = ["RE19", "AK14"]
SUPPORTED_GMPE_MODEL_TYPES = ["PGV", "PGA"]


class SeismicNetwork:
    """
    A class representing a seismic network with noise estimates for
    each station (this must be done in advance).
    """

    def __init__(
        self,
        stations: str | pd.DataFrame,
        network_code: str = "XX",
        required_detections: int = 5,
    ):
        """
        Initialize a SeismicNetwork instance.

        Parameters
        ----------
        stations : str or pd.DataFrame
            Path to a CSV file or a DataFrame containing station data with noise.
        network_code : str, optional
            The FDSN (or internal) network code for the seismic network, by default "XX".
        required_detections : int, optional
        """
        self.network_code = network_code
        self.required_detections = required_detections
        self.stations = _read_station_data(stations)
        self._validate()

    def add_stations(self, stations: str | pd.DataFrame):
        """
        Add more stations to the seismic network.

        Parameters
        ----------
        station : Station
            The station to add to the network.
        """
        new_stations = _read_station_data(stations)
        self.stations.append(new_stations)

    def __repr__(self):
        return (
            f"<SeismicNetwork {self.network_code} with {len(self.stations)} stations>"
        )

    def _validate(self):
        """
        Validate the seismic network data.
        """
        if len(self.stations) == 0:
            raise ValueError("No stations in the seismic network.")
        if len(self.stations) < self.required_detections:
            raise ValueError(
                f"Not enough stations in the seismic network. Required: {self.required_detections}, found: {len(self.stations)}"
            )

    @property
    def num_stations(self) -> int:
        """
        Get the number of stations in the seismic network.

        Returns
        -------
        int
            Number of stations in the seismic network.
        """
        return len(self.stations)


class SeismicArrayNetwork(SeismicNetwork):
    """
    A class representing a network of seismic arrays, which we will handle differently
    to traditional seismic networks. The noise is calcualted differently (in advance) and we
    want to set different detection thresholds.
    Here we treat each array as a single station, even though they are a collection
    of several closely space stations.
    """

    def __init__(
        self,
        arrays: str | pd.DataFrame,
        array_code: str = "XX",
        required_detections: int = 1,
    ):
        super().__init__(
            stations=arrays,
            network_code=array_code,
            required_detections=required_detections,
        )

    def __repr__(self):
        return f"<SeismicArray {self.array_code} with {len(self.arrays)} DAS channels>"


class DASFibre:
    """
    Class representing a DAS fibre with noise estimates for each channel.
    """

    def __init__(self, das_data: str | pd.DataFrame, detection_length_m: int = 1000):
        """
        Initialize a DASFibre instance.

        Parameters
        ----------
        das_data : str or pd.DataFrame
            Path to a CSV file or a DataFrame containing DAS noise data.
        """
        self.das_data = _read_das_noise_data(das_data)

        self.detection_length_m = detection_length_m
        print(f"DAS Fibre initialized with {len(self.das_data)} channels.")

    def __repr__(self):
        return f"<DASFibre with {len(self.das_data)} channels>"


class ModelConfig:
    """
    Class representing the model configuration for SNCAST.

    Parameters
    ----------
    snr : float, optional
        Signal-to-noise ratio threshold for detection, by default 3.0
    foc_depth_km : float, optional
        Focal depth in kilometers, by default 2.0
    region : str, optional
        Geographical region for the model, by default "CAL"
    nproc : int, optional
        Number of processors to use for computation, by default 1
    method : str, optional
        Method for ground motion prediction, by default "ML"
    gmpe : str, optional
        GMPE model to use if method is "GMPE", by default "AK14"
    gmpe_model_type : str, optional
        Type of GMPE model, by default "PGV"
    """

    def __init__(self, **kwargs):
        self.snr: float = kwargs.get("snr", 3.0)
        self.foc_depth_km: float = kwargs.get("foc_depth_km", 2.0)
        self.region: str = kwargs.get("region", "CAL")
        self.nproc: int = kwargs.get("nproc", 1)
        self.method: str = kwargs.get("method", "ML")
        if self.method == "GMPE":
            self.gmpe: str = kwargs.get("gmpe", "AK14")
            self.gmpe_model_type: str = kwargs.get("gmpe_model_type", "PGV")

        self._validate()

    def _validate(self):
        """
        Validate the model configuration by checking the method and region are supported.
        """

        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Invalid method: {self.method}. Valid methods are {SUPPORTED_METHODS}."
            )
        if self.method == "GMPE":
            if self.gmpe not in SUPPORTED_GMPES:
                raise ValueError(
                    f"Invalid GMPE: {self.gmpe}. Valid GMPEs are {SUPPORTED_GMPES}."
                )
            if self.gmpe_model_type not in SUPPORTED_GMPE_MODEL_TYPES:
                raise ValueError(
                    f"Invalid GMPE model type: {self.gmpe_model_type}. Valid types are {SUPPORTED_GMPE_MODEL_TYPES}."
                )

        if self.region not in SUPPORTED_REGIONS:
            raise ValueError(
                f"Invalid region: {self.region}. Valid regions are {SUPPORTED_REGIONS}."
            )
        if self.snr <= 0:
            raise ValueError("SNR must be a positive value.")
        if self.foc_depth_km < 0:
            raise ValueError("Focal depth must be a positive value.")

    def __repr__(self):
        return f"<ModelConfig with method={self.method}, region={self.region}, snr={self.snr}>"


def _read_station_data(stations_in):
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


def _read_das_noise_data(das_in):
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
