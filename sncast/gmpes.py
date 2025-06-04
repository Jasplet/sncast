# ------------------------------------------------------------------
# Filename: gmpes.py
# Purpose:  Augementing SNCAST by adding functionality to
#           use ground motion prediction equations (GMPEs)
# Author:   Joseph Asplet, University of Oxford
#
#    Copyright (C) 2025 Joseph Asplet
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
#    email:       joseph.asplet@earth.ox.ac.uk
#    web:         www.jasplet.github.io
#
# --------------------------------------------------------------------

import numpy as np
import json
from pathlib import Path

base_path = Path(__file__).resolve().parent.parent
data_path = base_path / 'data' / 'gmpe_coeffs.json'

SUPPORT_GMPES = ['RE19', 'AK14']
# Citations for GMPES
# RE19: Edwards, B (2019) Update of the UK stochastic ground motion model using 
#       a decade of broadband data. In: SECED 2019 Conference, 2019-9-9 - 2019-9-10,
#       Greenwich, London.
#       https://livrepository.liverpool.ac.uk/3060529/1/SECED_2019_paper_Rietbrock_Edwards_FINAL.pdf
# Using the 10Mpa model for RE19.

# AK14: Akkar, S., Sandıkkaya, M.A., Bommer, J.J., 2014.,
#       Empirical ground-motion models for point- and extended-source crustal
#       earthquake scenarios in Europe and the Middle East. Bull Earthquake Eng 12, 359–387.
#       https://doi.org/10.1007/s10518-013-9461-4


def eval_gmpe(mw, epic_dist, author, model_type='PGV', debug=False):

    if author not in SUPPORT_GMPES:
        raise ValueError(f'Author {author} not supported')

    with open(data_path) as f:
        GMPES = json.load(f)

    coeffs = GMPES[author][model_type]
    if author == 'RE19':
        y = _eval_re19(coeffs, mw, epic_dist, debug=debug)

    elif author == 'AK14':
        coeffs_pga = GMPES['AK14']['PGA']
        y = _eval_ak14(coeffs, coeffs_pga, mw, epic_dist)

    return y


def _eval_re19(coeffs, mw, epic_dist, debug=False):
    '''
    Implementations of the Rietbrock and Edwards (2019) ground motion
    '''
    dist = np.sqrt(coeffs['c11']**2 + epic_dist)
    f0, f1, f2 = _re19_f_terms(dist)
    #
    # y can be either PGA, PGV or one of the periods if those
    # coefficiants are added.
    y1 = coeffs['c1'] + coeffs['c2']*mw + coeffs['c3']*(mw**2)
    y2 = f0*(coeffs['c4'] + coeffs['c5']*mw)
    y3 = f1*(coeffs['c6'] + coeffs['c7']*mw)
    y4 = f2*(coeffs['c8'] + coeffs['c9']*mw)
    y5 = coeffs['c10'] * dist
    log10y = y1 + y2 + y3 + y4 + y5
    y = np.power(10, log10y)
    if debug:
        return (y1, y2, y3, y4, y5)
    else:
        return y


def _re19_f_terms(dist, r0=10, r1=50, r2=100):

    dist = np.asarray(dist)

    f0 = np.where(dist <= r0, np.log10(r0 / dist), 0)
    f1 = np.where(dist <= r1, np.log10(dist/1.0), np.log10(r1/1.0))
    f2 = np.where(dist <= r2, 0, np.log10(dist / r2))

    return f0, f1, f2


def _eval_ak14(coeffs, pga_coeffs, mw, dist, vs30=750, nstd=1):
    '''
    Evaluates Aker et al., (2014) ground motion prediction equation

    Epicentral distance is the only implementation here for now
    '''
    # First calculate PGA_REF, which is done for a refenrece Vs30=750m/s    
    ln_pga_750 = _ak14_yref(pga_coeffs, mw, dist, sof='SS')
    pga_750 = np.exp(ln_pga_750)
    # Calculate the site amplification
    site_ampl = _ak14_site_ampl(coeffs, pga_750, vs30)
    # Calculate total aleatory variability/ standard deviation
    sigma = np.sqrt(coeffs['tau']**2 + coeffs['phi']**2)
    # Calculate the final ground motion
    ln_yref = _ak14_yref(coeffs, mw, dist, sof='SS')
    ln_y = ln_yref + site_ampl + nstd*sigma
    return np.exp(ln_y)


def _ak14_yref(coeffs,
               mw,
               epic_dist,
               sof='SS'):
    # if mw <= c1 we use coefficat a2, if mw > c1 we use a7

    if sof in ['Normal', 'normal', 'N']:
        sof_N = 1
    elif sof in ['Reverse', 'reverse', 'R']:
        sof_R = 1
    else:
        sof_N = 0
        sof_R = 0

    if mw <= coeffs['c1']:
        y1 = coeffs['a1'] + coeffs['a2']*(mw - coeffs['c1'])
    else:
        y1 = coeffs['a1'] + coeffs['a7']*(mw - coeffs['c1'])

    y2 = coeffs['a3']*(8.5 - mw)**2
    y3 = (coeffs['a4'] + coeffs['a5']*(mw - coeffs['c1']))*np.log(np.sqrt(epic_dist**2 + coeffs['a6']**2))
    y4 = coeffs['a8']*sof_N + coeffs['a9']*sof_R
    y = y1 + y2 + y3 + y4

    return y


def _ak14_site_ampl(coeffs, pga_ref, Vs30, Vsref=750):
    '''
    Equation 3 in Aker et al., (2014)

    Reference Vs30 if 750 m/s
    V_con is 1000 m/s following Aker et al., (2014)
    '''
    vcon = 1000  # m/. Limiting Vs30 after which site amplification is constant
    if Vs30 > Vsref:
        s = coeffs['b1']*np.log(np.min([Vs30, vcon])/Vsref)
    else:
        s1 = coeffs['b1']*np.log(Vs30 / Vsref)
        numer = pga_ref + coeffs['c']*(Vs30 / Vsref)**coeffs['n']
        denom = (pga_ref + coeffs['c'])*(Vsref / Vsref)**coeffs['n']
        s2 = coeffs['b2'] * np.log(numer/denom)
        s = s1 + s2

    return s
