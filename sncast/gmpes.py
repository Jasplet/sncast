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

# AK14: Akkar, S., Sandıkkaya, M.A., Bommer, J.J., 2014.,
#       Empirical ground-motion models for point- and extended-source crustal
#       earthquake scenarios in Europe and the Middle East. Bull Earthquake Eng 12, 359–387.
#       https://doi.org/10.1007/s10518-013-9461-4


def eval_gmpe(mw, epic_dist, author, model_type='PGV'):

    if author not in SUPPORT_GMPES:
        raise ValueError(f'Author {author} not supported')

    with open(data_path) as f:
        GMPES = json.load(f)

    coeffs = GMPES[author][model_type]
    if author == 'RE19':
        y = _eval_re19(coeffs, mw, epic_dist)

    elif author == 'AK14':
        coeffs_pga = GMPES['AK14']['PGA']
        y = _eval_ak14(coeffs, coeffs_pga, mw, epic_dist)

    return y


def _eval_re19(coeffs, mw, epic_dist):
    '''
    Implementations of the Rietbrock and Edwards (2019) ground motion
    '''
    dist = np.sqrt(coeffs['c11']**2 + epic_dist)
    f0, f1, f2 = _re19_f_terms(dist)
    #
    # y can be either PGA, PGV or one of the periods if those
    # coefficiants are added.
    y1 = coeffs['c1'] + coeffs['c2']*mw + coeffs['c3']*mw**2
    y2 = f0*(coeffs['c4'] + coeffs['c5']*mw)
    y3 = f1*(coeffs['c6'] + coeffs['c7']*mw)
    y4 = f2*(coeffs['c8'] + coeffs['c9']*mw)
    y5 = coeffs['c10'] * dist
    y = np.power(y1 + y2 + y3 + y4 + y5, 10)
    return y


def _re19_f_terms(dist, r0=10, r1=50, r2=100):

    if dist <= r0:
        f0 = np.log10(r0/dist)
    else:
        f0 = 0

    if dist <= r1:
        f1 = np.log10(dist/1.0)
    elif dist > r1:
        f1 = np.log10(dist/1.0)

    if dist <= r2:
        f2 = 0
    elif dist > r2:
        f2 = np.log10(dist/r2)

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
