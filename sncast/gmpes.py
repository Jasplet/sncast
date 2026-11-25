#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Joseph Asplet, University of Oxford
# ------------------------------------------------------------------
"""
Filename: gmpes.py
Purpose:  Augmenting SNCAST by adding functionality to
          use ground motion prediction equations (GMPEs)
Author:   Joseph Asplet, University of Oxford
Email:       joseph.asplet@earth.ox.ac.uk
Web:         www.jasplet.github.io
Github:      www.github.com/jasplet
Address:     Department of Earth Sciences, University of Oxford,
            South Parks Road, Oxford, OX1 3AN, UK
orcidID:       https://orcid.org/0000-0002-0375-011X

Supported GMPES:

RE19: Edwards, B (2019) Update of the UK stochastic ground motion model using
    a decade of broadband data. In: SECED 2019 Conference, 2019-9-9 - 2019-9-10,
    Greenwich, London.
    https://livrepository.liverpool.ac.uk/3060529/1/SECED_2019_paper_Rietbrock_Edwards_FINAL.pdf
Notes: Using the 10Mpa model for RE19.

AK14: Akkar, S., Sandıkkaya, M.A., Bommer, J.J., 2014.,
    Empirical ground-motion models for point- and extended-source crustal
    earthquake scenarios in Europe and the Middle East. Bull Earthquake Eng 12, 359–387.
    https://doi.org/10.1007/s10518-013-9461-4

Note: This module is still a work in progress, use with caution.


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

import json
from pathlib import Path

import numpy as np

base_path = Path(__file__).resolve().parent.parent
data_path = base_path / 'data' / 'gmpe_coeffs.json'

SUPPORT_GMPES = ['RE19', 'AK14']


def eval_gmpe(mw, epic_dist, gmpe, model_type='PGV', debug=False):
    """
    Evaluates a chosen ground motion prediction equation (GMPE) for a given
    moment magnitude and epicentral distance.

    Currently supported GMPEs are:
        - RE19. Rietbrock and Edwards (2019)
        - AK14. Aker et al., (2014)

    Parameters
    ----------
    mw : float
        Moment magnitude of the earthquake.
    epic_dist : float or array-like
        Epicentral distance in km.
    gmpe : str
        Author of the GMPE. Supported values are "RE19" and "AK14".
        Author of the GMPE. Supported values are "RE19" and "AK14".
    model_type : str, optional
        Type of ground motion to evaluate. Supported values are
        "PGA", "PGV". Default is "PGV".
    debug : bool, optional
        If True, returns intermediate calculation steps for debugging.
        Default is False.

    Returns
    -------
    y : float or array-like
        Evaluated ground motion (PGA in g (m/s2) or PGV in m/s).
    """
    if gmpe not in SUPPORT_GMPES:
        raise ValueError(f'GMPE {gmpe} not supported')

    with open(data_path) as f:
        GMPES = json.load(f)

    coeffs = GMPES[gmpe][model_type]
    if gmpe == 'RE19':
        y = _eval_re19(coeffs, mw, epic_dist, debug=debug)

    elif gmpe == 'AK14':
        coeffs_pga = GMPES['AK14']['PGA']
        y = _eval_ak14(coeffs, coeffs_pga, mw, epic_dist)

    return y


def _eval_re19(coeffs, mw, epic_dist, debug=False):
    """
    Implementations of the Rietbrock and Edwards (2019) ground motion prediction
    equation.

    Parameters
    ----------
    coeffs : dict
        Coefficients for the RE19 GMPE, selected from JSON file of coefficients.
    mw : float
        Moment magnitude of the earthquake to model
    epic_dist : float or array-like
        Epicentral distance in km. Strictly speaking this should be Joyner-Boore distance
        but epicentral distance is used here as a proxy. This is valid for strike-slip
        earthquakes and larger distances.
    debug : bool, optional
        If True, returns intermediate calculation steps for debugging.
        Default is False.
    Returns
    -------
    y : float or array-like
        Evaluated ground motion (PGA in g (m/s2) or PGV in m/s).
    or
    (y1, y2, y3, y4, y5) if debug is True
        Intermediate calculation steps for debugging.

    """
    dist = np.sqrt(coeffs['c11'] ** 2 + epic_dist)
    f0, f1, f2 = _re19_f_terms(dist)
    #
    # y can be either PGA, PGV or one of the periods if those
    # coefficiants are added.
    y1 = coeffs['c1'] + coeffs['c2'] * mw + coeffs['c3'] * (mw**2)
    y2 = f0 * (coeffs['c4'] + coeffs['c5'] * mw)
    y3 = f1 * (coeffs['c6'] + coeffs['c7'] * mw)
    y4 = f2 * (coeffs['c8'] + coeffs['c9'] * mw)
    y5 = coeffs['c10'] * dist
    log10y = y1 + y2 + y3 + y4 + y5
    y = np.power(10, log10y)
    if debug:
        return (y1, y2, y3, y4, y5)
    else:
        return y


def _re19_f_terms(dist, r0=10, r1=50, r2=100):
    """
    Calculates the f0, f1, f2 terms from Rietbrock and Edwards (2019)
    based on distance.

    Parameters
    ----------
    dist : float or array-like
        Distance in km. Corresponds to R in Rietbrock and Edwards (2019) equation.
    r0 : float, optional
        Distance threshold for f0 term. Default is 10 km.
    r1 : float, optional
        Distance threshold for f1 term. Default is 50 km.
    r2 : float, optional
        Distance threshold for f2 term. Default is 100 km.
    Returns
    -------
    f0, f1, f2 : tuple of float or array-like
        Calculated f0, f1, f2 terms.
    """
    dist = np.asarray(dist)

    f0 = np.where(dist <= r0, np.log10(r0 / dist), 0)
    f1 = np.where(dist <= r1, np.log10(dist / 1.0), np.log10(r1 / 1.0))
    f2 = np.where(dist <= r2, 0, np.log10(dist / r2))

    return f0, f1, f2


def _eval_ak14(coeffs, pga_coeffs, mw, dist, vs30=750, nstd=1):
    """
    Evaluates Aker et al., (2014) ground motion prediction equation

    Epicentral distance is the only implementation here for now.

    Parameters
    ----------
    coeffs : dict
        Coefficients for the AK14 GMPE, selected from JSON file of coefficients.
    pga_coeffs : dict
        Coefficients for the AK14 PGA GMPE, selected from JSON file of coefficients.
        This is used to calculate the reference PGA for site amplification.
    mw : float
        Moment magnitude of the earthquake to model
    dist : float or array-like
        Epicentral distance in km.
    vs30 : float, optional
        Average shear-wave velocity in the top 30m of the site in m/s.
        Default is 750 m/s, which is the reference Vs30 for AK14.
    nstd : int, optional
        Number of standard deviations to apply to the mean prediction.
        Default is 1.

    Returns
    -------
    float or array-like
        Evaluated ground motion (PGA in g (m/s2) or PGV in m/s).
    """
    # First calculate PGA_REF, which is done for a refenrece Vs30=750m/s
    ln_pga_750 = _ak14_yref(pga_coeffs, mw, dist, sof='SS')
    pga_750 = np.exp(ln_pga_750)
    # Calculate the site amplification
    site_ampl = _ak14_site_ampl(coeffs, pga_750, vs30)
    # Calculate total aleatory variability/ standard deviation
    sigma = np.sqrt(coeffs['tau'] ** 2 + coeffs['phi'] ** 2)
    # Calculate the final ground motion
    ln_yref = _ak14_yref(coeffs, mw, dist, sof='SS')
    ln_y = ln_yref + site_ampl + nstd * sigma
    return np.exp(ln_y)


def _ak14_yref(coeffs, mw, epic_dist, sof='SS'):
    """
    Evaluates equation 2 in Aker et al., (2014) for ln(Y_REF)

    Parameters
    ----------
    coeffs : dict
        Coefficients for the AK14 GMPE, pre-selected from JSON file of coefficients.
    mw : float
        Moment magnitude of the earthquake to model
    epic_dist : float or array-like
        Epicentral distance in km.
    sof : str, optional
        Style of faulting. Supported values are "Normal" (or "N"),
        "Reverse" (or "R") and "Strike-Slip" (or "SS"). Default is "SS".

    Returns
    -------
    y : float or array-like
        Evaluated ln(Y_REF)

    Notes
    -----
    Work in progress. Check maths to ensure we are correctly adding
    the S term in AK14 for Y_REF.
    """
    # if mw <= c1 we use coefficat a2, if mw > c1 we use a7

    if sof in ['Normal', 'normal', 'N']:
        sof_N = 1
    elif sof in ['Reverse', 'reverse', 'R']:
        sof_R = 1
    else:
        sof_N = 0
        sof_R = 0

    if mw <= coeffs['c1']:
        y1 = coeffs['a1'] + coeffs['a2'] * (mw - coeffs['c1'])
    else:
        y1 = coeffs['a1'] + coeffs['a7'] * (mw - coeffs['c1'])

    y2 = coeffs['a3'] * (8.5 - mw) ** 2
    y3 = (coeffs['a4'] + coeffs['a5'] * (mw - coeffs['c1'])) * np.log(
        np.sqrt(epic_dist**2 + coeffs['a6'] ** 2)
    )
    y4 = coeffs['a8'] * sof_N + coeffs['a9'] * sof_R
    y = y1 + y2 + y3 + y4

    return y


def _ak14_site_ampl(coeffs, pga_ref, Vs30, Vsref=750):
    """
    Equation 3 in Aker et al., (2014)

    Reference Vs30 if 750 m/s
    V_con is 1000 m/s following Aker et al., (2014)

    Parameters
    ----------
    coeffs : dict
        Coefficients for the AK14 GMPE, pre-selected from JSON file of coefficients.
    pga_ref : float or array-like
        Reference PGA in g (m/s2) for Vs30=750m/s
    Vs30 : float or array-like
        Average shear-wave velocity in the top 30m of the site in m/s.
    Vsref : float, optional
        Reference Vs30 in m/s. Default is 750 m/s.

    Returns
    -------
    s : float or array-like
        s term from equation 3 in Aker et al., (2014)
    """
    vcon = 1000  # m/s. Limiting Vs30 after which site amplification is constant
    if Vs30 > Vsref:
        s = coeffs['b1'] * np.log(np.min([Vs30, vcon]) / Vsref)
    else:
        s1 = coeffs['b1'] * np.log(Vs30 / Vsref)
        numer = pga_ref + coeffs['c'] * (Vs30 / Vsref) ** coeffs['n']
        denom = (pga_ref + coeffs['c']) * (Vsref / Vsref) ** coeffs['n']
        s2 = coeffs['b2'] * np.log(numer / denom)
        s = s1 + s2

    return s
