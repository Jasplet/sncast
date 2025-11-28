# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Joseph Asplet, University of Oxford
# ------------------------------------------------------------------"""
"""
Filename:   noise_estimation.py

Purpose:    Functions to estimate noise displacement and velocity
            from probabilistic power spectral densities (PPSDs)
            for stations in a given Inventory. Implements equations given in
            [Molhoff2019]_ and [Haskov2016]_.

Author:     Joseph Asplet, University of Oxford

Email:      joseph.asplet@earth.ox.ac.uk

Web:        www.jasplet.github.io

Github:     www.github.com/jasplet

Address:    Department of Earth Sciences, University of Oxford,
            South Parks Road, Oxford, OX1 3AN, UK

orcidID:    https://orcid.org/0000-0002-0375-011X

Citation:   Möllhoff, M., Bean, C.J. & Baptie, B.J.,
            SN-CAST: seismic network capability assessment software tool
            for regional networks - examples from Ireland.
            J Seismol 23, 493-504 (2019).
            https://doi.org/10.1007/s10950-019-09819-0

            Havskov and Alguacil (2016), Instrumentation in Earthquake
            Seismology (2nd Edition), https://doi.org/10.1007/978-3-319-21314-9

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
"""

from pathlib import Path

import numpy as np
import pandas as pd
from obspy.signal import PPSD


def estimate_noise_displacement(station_ppsd, f0=5, n=0.5, case="worst", verbose=False):
    """
    Implements equation 5 from Mölhoff et al., (2019) to estimate noise displacement
    within a narrow frequency band of interest.

    Adapted from equations in Haskov and Alguacil (2016)

    See:
    Havskov and Alguacil (2016), Instrumentation in Earthquake
    Seismology (2nd Edition), https://doi.org/10.1007/978-3-319-21314-9
    Möllhoff, M., Bean, C. J., & Baptie, B. J. (2019).
    SN-CAST: seismic network capability assessment software tool for
    regional networks - examples from Ireland. Journal of Seismology,
    23, 493-504.

    Parameters
    ----------
    station_ppsd : obspy.signal.PPSD station_ppsd : obspy.signal.PPSD
        PPSD object for a seismic station.
    f0 : float
        Centre frequency of the band of interest in Hz. Default is 5 Hz.
    n : float
        Width of the frequency band in octaves. Default is 0.5 octaves, which
        gives an octave range cnetered on f0.
    case : str or int
        Case for the noise estimate. Can be 'worst' (95th percentile),
        'mode' (mode of the distribution) or an integer giving the desired
        percentile (e.g. 50 for median). Default is 'worst'.
    verbose : bool
        If True, prints information about the noise estimate being made.
        Default is False.

    Returns
    -------
    displacement : float
        Estimated noise in displacement [m].
    """

    f1, f2 = get_freq_range_from_centre(f0, n)
    if verbose:
        print(f"{case} noise estimate. f0={f0:4.2f}, n={n:2.1f}")
    if case == "worst":
        period, psd_accl = station_ppsd.get_percentile(95)
        freqs_psd = 1 / period
    elif case == "mode":
        period, psd_accl = station_ppsd.get_mode()
        freqs_psd = 1 / period
    elif type(case) is int:
        period, psd_accl = station_ppsd.get_percentile(case)
        freqs_psd = 1 / period

    mean_psd_db = psd_accl[(freqs_psd >= f1) & (freqs_psd <= f2)].mean()

    displacement = psd_db_to_displacement_amplitude(mean_psd_db, f0=f0, f1=f1, f2=f2)

    return displacement


def estimate_noise_velocity(station_ppsd, f0=5, n=0.5, case="worst", verbose=False):
    """
    Estimates 95-percentile noise values in velocity [m/s] in a frequency band
    centered on f0. Similar to estimate_noise_displacement but we are only integrating
    the PPSD (which is in acceleration) once.

    Parameters
    ----------
    station_ppsd : obspy.signal.PPSD
        PPSD object for a seismic station.
    f0 : float
        Centre frequency of the band of interest in Hz. Default is 5 Hz.
    n : float
        Width of the frequency band in octaves. Default is 0.5 octaves, which
        gives an octave range cnetered on f0.
    case : str or int
        Case for the noise estimate. Can be 'worst' (95th percentile),
        'mode' (mode of the distribution) or an integer giving the desired
        percentile (e.g. 50 for median). Default is 'worst'.
    verbose : bool
        If True, prints information about the noise estimate being made.
        Default is False.

    Returns
    -------
    velocity : float
        Estimated noise in velocity [m/s].
    """
    f1, f2 = get_freq_range_from_centre(f0, n)
    if verbose:
        print(f"{case} noise estimate. f0={f0:4.2f}, n={n:2.1f}")
    if case == "worst":
        period, psd_accl = station_ppsd.get_percentile(95)
        freqs_psd = 1 / period
    elif case == "mode":
        period, psd_accl = station_ppsd.get_mode()
        freqs_psd = 1 / period
    elif type(case) is int:
        period, psd_accl = station_ppsd.get_percentile(case)
        freqs_psd = 1 / period

    mean_psd_db = psd_accl[(freqs_psd >= f1) & (freqs_psd <= f2)].mean()

    velocity = psd_db_to_velocity(mean_psd_db, f0=f0, f1=f1, f2=f2)

    return velocity


def get_f0_octaves_from_f1f2(f1, f2):
    """
    Calculates the centre frequency and width in octaves
    from a given frequency range.

    Parameters
    ----------
    f1 : float
        Lower bound of the frequency range in Hz
    f2 : float
        Upper bound of the frequency range in Hz

    Returns
    -------
    f0 : float
        Centre frequency in Hz
    n : float
        Width of the frequency band in octaves

    """
    n = np.log2(f2 / f1)  # width of the frequency band in octaves
    f0 = np.sqrt(f1 * f2)  # centre frequency
    return f0, n


def get_freq_range_from_centre(f0, n=0.5):
    """
    Calculates a 2n octave frequency range centred on f0

    Parameters
    ----------
    f0 : float
        Centre frequency in Hz
    n : float
        Width of the frequency band in octaves. Default is 0.5 octaves, which
        gives an octave range centered on f0.

    Returns
    -------
    f1 : float
        Lower bound of the frequency range in Hz
    f2 : float
        Upper bound of the frequency range in Hz

    """
    if isinstance(n, (int, float)):
        if n <= 0:
            raise ValueError("n must be greater than 0")
    else:
        raise ValueError("n must be a positive int or float")
    if isinstance(f0, (int, float)):
        if f0 <= 0:
            raise ValueError("f0 must be greater than 0")
    else:
        raise ValueError("f0 must be a positive int or float")
    f1 = f0 * 2 ** (-n / 2)  # lower bound of frequency span
    f2 = f0 * 2 ** (n / 2)  # upper bound of frequency span
    return f1, f2


def psd_db_to_displacement_amplitude(psd_in_db, f1, f2, f0=None):
    """
    Take acceleration power density (in dB relative to 1 ((m/s)^2)^2 / Hz)
    and calculates displacement in meters.

    Intended use is to estimate seismic station noise in displacement from
    probabilistic power spectral densities (PPSDs).

    Implentation is based on equation 5 from Mölhoff et al., (2019).

    Parameters
    ----------
    psd_in_db : float
        Acceleration power spectral density in dB relative to 1 ((m/s)^2)^2 / Hz
    f1 : float
        Lower bound of the frequency range in Hz
    f2 : float
        Upper bound of the frequency range in Hz

    Returns
    -------
    displ : float
        Estimated PSD in frequency range converted to displacement [m].
    """

    f0 = np.sqrt(f1 * f2)  # find centre frequency, geometric mean of f1 and f2
    psd = psd_db_convert(psd_in_db)
    displ = (3.75) / ((2 * np.pi * f0) ** 2) * np.sqrt(psd * (f2 - f1))

    return displ


def psd_db_to_velocity(psd_in_db, f1, f2):
    """
    Take acceleration power density (in dB relative to 1 ((m/s)^2)^2 / Hz)
    and calculates velocity in m/s.

    Intended use is to estimate seismic station noise in velocity from
    probabilistic power spectral densities (PPSDs).

    Implementation inspired by equation 5 from Mölhoff et al., (2019), but instead of converting
    to displacement, we only integrate once and convert to velocity.

    Parameters
    ----------
    psd_in_db : float
        Acceleration power spectral density in dB relative to 1 ((m/s)^2)^2 / Hz
    f1 : float
        Lower bound of the frequency range in Hz
    f2 : float
        Upper bound of the frequency range in Hz
    """
    f0 = np.sqrt(f1 * f2)  # find centre frequency

    power_acc = psd_db_convert(psd_in_db)
    power_vel = power_acc / (2 * np.pi * f0) ** 2
    velocity = 1.25 * np.sqrt(power_vel * (f2 - f1))
    # N.B for displacement this equation under estimates by a facotr
    # of 3 (Haskov and Agluacil, 2006) so maybe we need to
    # dig into how well it works for velocity.

    return velocity


def psd_db_convert(psd_in_db):
    """
    Convert probsabilistic PSD values from decibels to linear units.

    Parameters
    ----------
    psd_in_db : float
        Power spectral density in dB relative to 1 ((m/s)^2)^2 / Hz
    Returns
    -------
    psd : float
        Power spectral density in linear units [((m/s)^2)^2 / Hz]
    """
    return np.power(10, psd_in_db / 10)


def make_noise_estimate_for_ppsds(
    inventory,
    case,
    file_ext,
    unit,
    ppsd_path=None,
    **kwargs,
):
    """
    Function to make noise estimates from previously
    calculated probabilistic power spectral densities (PPSDs)
    for stations in a given Inventory.

    Can return estimated noise displacement amplitude or
    estimated noise velocity amplitude. Velocity estimates
    are still a bit provisional.

    Parameters
    ----------
    inventory : obspy.core.inventory.Inventory
        Inventory object containing stations to make noise estimates for.
    case : str or int
        Case for the noise estimate. Can be 'worst' (95th percentile),
        'mode' (mode of the distribution) or an integer giving the desired
        percentile (e.g. 50 for median). Default is 'worst'.
    file_ext : str
        Suffix for the PPSD files. Function assumes files are named as
        '{network}_{station}_{channel}_{file_ext}.npz' and (implicity) are for
        the same date range.
    unit : str
        Type of noise estimate to make. Can be 'displ' for displacement
        estimates in nm, or 'vel' for velocity estimates in cm/s. Default is 'displ'.
    ppsd_path : str or pathlib.Path, optional
        Path to the directory containing the PPSD files. Default is Path.cwd() / "ppsd".
    Returns
    -------
    station_noise_df : pandas.DataFrame
        DataFrame containing station metadata and estimated noise values.
        Columns are: ['longitude', 'latitude', 'elevation_km', 'noise [nm]', 'station']
        or ['longitude', 'latitude', 'elevation_km', 'noise [cm/s]', 'station']
        depending on the value of `kind`.
    -----------

    Notes:
    - This function assumes that the PPSD files are stored in a specific directory structure.
      This needs to be fixed before release.
    """
    if unit not in ["displ", "vel"]:
        raise ValueError(f'unit must be one of ["displ", "vel"] not {unit}')
    elif unit == "displ":
        noise_key = "noise [nm]"
    elif unit == "vel":
        noise_key = "noise [cm/s]"
    station_noise_dict = {
        "longitude": [],
        "latitude": [],
        "elevation_km": [],
        noise_key: [],
        "station": [],
    }
    if ppsd_path is not None:
        ppsd_path = Path(kwargs["ppsd_path"])
    else:
        ppsd_path = Path.cwd() / "ppsd"
    for network in inventory:
        network_ppsd_path = ppsd_path / network.code
        for station in network:
            for channel in station:
                ppsd_file = (
                    f"{network.code}_{station.code}_{channel.code}_{file_ext}.npz"
                )
                try:
                    ppsd = PPSD.load_npz(network_ppsd_path / ppsd_file)

                    if unit == "displ":
                        if "f0" in kwargs:
                            displ_m = estimate_noise_displacement(
                                ppsd, case=case, f0=kwargs["f0"]
                            )
                        else:
                            displ_m = estimate_noise_displacement(ppsd, case=case)
                        noise = displ_m * 1e9  # disp in nm
                    elif unit == "vel":
                        if "f0" in kwargs:
                            vel_ms = estimate_noise_velocity(
                                ppsd, case=case, f0=kwargs["f0"]
                            )
                        else:
                            vel_ms = estimate_noise_velocity(ppsd, case=case)
                        noise = vel_ms * 1e2  # vel in cm/s

                except FileNotFoundError:
                    print(ppsd_file)
                    if unit == "displ":
                        noise = 10  # nm
                    elif unit == "vel":
                        noise = 2e-05  # cm/s

                station_noise_dict["longitude"].append(station.longitude)
                station_noise_dict["latitude"].append(station.latitude)
                station_noise_dict["elevation_km"].append(station.elevation * 1e-3)
                station_noise_dict[noise_key].append(noise)
                station_noise_dict["station"].append(station.code)
                station_noise_dict["channel"].append(channel.code)

    return pd.DataFrame(station_noise_dict)
