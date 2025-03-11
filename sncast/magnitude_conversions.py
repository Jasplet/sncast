import numpy as np

def convert_mw_to_ml(mw, region='UK'):
    '''
    Converts Moment magntiude to local magnitude
    using an empirical scaling relationship

    Default region is the UK.
    For small M<3 event we use the conversion developed by Butcher et al., (2020)
    https://doi.org/10.1785/0120190032 using data from
    New Ollerton and is applicable for small magntiude events.

    For Mw > 3 the Ottemöller and Sargeant (2013) https://doi.org/10.1785/0120130085
    conversion is used instead.

    Parameters:
    ----------
    mw : numpy.Array
        array of moment magnitudes to convert
    region : str
        Region's scaling relationship to use. Defaults to UK
    '''

    if region == 'UK':
        butcher_uk_small_mw = np.vectorize(lambda x: (x - 0.75) / 0.69)
        ottermoller_uk = np.vectorize(lambda x: (x - 0.23) / 0.85)

        ml = np.where(mw <= 3, butcher_uk_small_mw(mw), ottermoller_uk(mw))
    else:
        raise ValueError(f'Unsupported region {region}')

    return ml


def convert_ml_to_mw(ml, region='UK'):
    '''
    Converts Local magnitude to moment magnitude
    using an empirical scaling relationship

    Default region is the UK.
    For small M<3 event we use the conversion developed by Butcher et al., (2020)
    https://doi.org/10.1785/0120190032 using data from
    New Ollerton and is applicable for small magntiude events.

    For Mw > 3 the Ottemöller and Sargeant (2013) https://doi.org/10.1785/0120130085
    conversion is used instead.

    Parameters:
    ----------
    mw : numpy.Array
        array of moment magnitudes to convert
    region : str
        Region's scaling relationship to use. Defaults to UK
    '''

    if region == 'UK':
        butcher_uk_small_mw = np.vectorize(lambda x: 0.75*x + 0.69)
        ottermoller_uk = np.vectorize(lambda x: 0.85*x + 0.23)

        ml = np.where(ml <= 3, butcher_uk_small_mw(ml), ottermoller_uk(ml))
    else:
        raise ValueError(f'Unsupported region {region}')

    return mw