import numpy as np
import json
import os
from pathlib import Path

base_path = Path(__file__).resolve().parent.parent
data_path = base_path / 'data' / 'gmpe_coeffs.json'

SUPPORT_GMPES = ['RE19', 'AK14']


def calc_pgv(mw, epic_dist, author, model_type='PGV'):

    if author not in SUPPORT_GMPES:
        raise ValueError(f'Author {author} not supported')
    
    with open(data_path) as f:
        GMPES = json.load(f)

    coeffs = GMPES[author][model_type]
    if author == 'RE19':
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

    elif author == 'AK14':
        
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