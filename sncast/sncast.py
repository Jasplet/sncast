#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: sncast.py
# Purpose:  Seismic Network Capability Assessment Software Tool (SNCAST)
# Author:   Martin Möllhoff, DIAS
# Citation: Möllhoff, M., Bean, C.J. & Baptie, B.J.,
#           SN-CAST: seismic network capability assessment software tool
#           for regional networks - examples from Ireland. 
#           J Seismol 23, 493-504 (2019). https://doi.org/10.1007/s10950-019-09819-0
#
# You can run SNCAST in a browser, without a python installation on your computer:
#
#        - browse to https://github.com/moellhoff/Jupyter-Notebooks
#        - click on the "launch binder" icon
#        - wait a few minutes until the repository finished the starting process
#        - in the folder 'SNCAST' click "sncast-getting-started.ipynb" and follow the instructions
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

import numpy as np
import pandas as pd
from obspy.signal.util import util_geo_km
from math import sqrt
import pygc
import xarray

# Citations for GMPES
# RE19: Edwards, B (2019) Update of the UK stochastic ground motion model using 
#       a decade of broadband data. In: SECED 2019 Conference, 2019-9-9 - 2019-9-10,
#       Greenwich, London. 
#       https://livrepository.liverpool.ac.uk/3060529/1/SECED_2019_paper_Rietbrock_Edwards_FINAL.pdf

# AK14: Akkar, S., Sandıkkaya, M.A., Bommer, J.J., 2014.,
#       Empirical ground-motion models for point- and extended-source crustal
#       earthquake scenarios in Europe and the Middle East. Bull Earthquake Eng 12, 359–387.
#       https://doi.org/10.1007/s10518-013-9461-4


GMPES = {'RE19': {'PGV': {'c1': -4.4578, 'c2': 1.6540,
                          'c3': -0.1044, 'c4': -1.6308,
                          'c5': 0.2082, 'c6': -1.6465,
                          'c7': 0.1568, 'c8': -2.3547,
                          'c9': 0.0676, 'c10': -0.000991,
                          'c11': 2.8899},
                  'PGA': {'c1': -2.1780, 'c2': 1.6355,
                          'c3': -0.1241, 'c4': -1.8303,
                          'c5': 0.2165, 'c6': -1.8318,
                          'c7': 0.1622, 'c8': -1.9899,
                          'c9': 0.0678, 'c10': -0.002073,
                          'c11': 2.3460}},
         'AK14': {'PGA': {'a1':2.52977, 'a2': 0.0029, 'a3': -0.05496,
                          'a4':-1.31001, 'a5': 0.2529, 'a6':	7.5,
                          'a7':-0.5096, 'a8': -0.1091,'a9':0.0937,
                          'c1':6.75, 'V_con':1000, 'V_ref': 750, 
                          'c':2.5, 'n':3.2, 'b1':-0.41997,
                          'b2':-0.28846, 'phi':	0.6375,
                          'tau':	0.3581},
                 'PGV': {'a1':6.13498, 'a2':0.0029, 'a3': -0.12091,
                        'a4':-1.04013, 'a5': 0.2529, 'a6':7.5,
                        'a7':-0.5096, 'a8':-0.0616, 'a9':0.063,
                        'c1':6.75, 'V_con':	1000, 'V_ref':	750,
                        'c':2.5, 'n':3.2, 'b1': -0.72057,
                        'b2':-0.19688, 'phi':0.6143, 'tau':0.3485
                        }
                }
        }


def calc_pgv(mw, epic_dist, author, model_type='PGV'):

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
        y5 = coeffs['c10']* dist
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


def calc_ampl_from_magnitude(local_mag, hypo_dist, region):

    #   region specific ML = log(ampl) + a*log(hypo-dist) + b*hypo_dist + c
    if region == 'UK':
        #   UK Scale uses new ML equation from Luckett et al., (2019)
        #   https://doi.org/10.1093/gji/ggy484
        #   Takes form local_mag = log(amp) + a*log(hypo-dist) + b*hypo-dist + d*exp(e * hypo-dist) + c
        a = 1.11
        b = 0.00189
        c = -2.09
        d = -1.16
        e = -0.2
        ampl = np.power(10, (local_mag - a*np.log10(hypo_dist) - b*hypo_dist - c - d*np.exp(e*hypo_dist)))

    elif region == 'CAL':
        # South. California scale, IASPEI (2005),
        # www.iaspei.org/commissions/CSOI/summary_of_WG_recommendations_2005.pdf
        a = 1.11
        b = 0.00189
        c = -2.09
        ampl = np.power(10, (local_mag - a*np.log10(hypo_dist) - b*hypo_dist - c))

    return ampl


def minML(stations_in, lon0=-12, lon1=-4, lat0=50.5, lat1=56.6, dlon=0.33,
          dlat=0.2, stat_num=4, snr=3, foc_depth=0, region='CAL', mag_min=-2.0, mag_delta=0.1,
          arrays=None, obs=None, obs_stat_num=3):
    """
    This routine calculates the geographic distribution of the minimum 
    detectable local magnitude ML for a given seismic network. Required 
#### 9.10.2020    input is a file (or a pandas DataFrame) containing four comma separated
    columns containing for each seismic station:

         longitude, latitude, elevation, noise [nm], station name
    e.g.: -7.5100, 55.0700, 0, 0.53, IDGL

    The output file *.grd lists in ASCII xyz format: longitud, latitude, ML
  
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
    # read in data, file format: "LON, LAT, NOISE [nm], STATION"
    if stations_in is str:
        stations_df = pd.read_csv(stations_in)
    else:
        stations_df = stations_in.copy()

    stat_lon = stations_df['longitude'].values
    stat_lat = stations_df['latitude'].values
    stat_elev = stations_df['elevation_km'].values
    stat = stations_df['station'].values
    noise = stations_df['noise [nm]'].values
    # grid size
    nx = int( (lon1 - lon0) / dlon) + 1
    ny = int( (lat1 - lat0) / dlat) + 1
    lats = np.linspace(lat1, lat0, ny)
    lons = np.linspace(lon0, lon1, nx)
    # open output file:
# ## 9.10.2020    f = open('%s/%s-stat%s-foc%s-snr%s-%s.grd' %(dir_in, filename, stat_num, foc_depth, snr, region), 'wb')
    mag = []
    array_mag = []
    obs_mag = []
    mag_grid = np.zeros((ny, nx))
    for ix in range(nx): # loop through longitude increments
        for iy in range(ny): # loop through latitude increments
            for j, jstat in enumerate(stat): # loop through stations
                # calculate hypcocentral distance in km
                dx, dy = util_geo_km(lons[ix], lats[iy], stat_lon[j], stat_lat[j])
                dz = np.abs(foc_depth - stat_elev[j])
                hypo_dist = sqrt(dx**2 + dy**2 + dz**2)
                # find smallest detectable magnitude
                ampl = 0.0
                m = mag_min - mag_delta
                while ampl < snr*noise[j]: 
                    m = m + mag_delta
                    ampl = calc_ampl_from_magnitude(m, hypo_dist, region)
                mag.append(m)
            # sort magnitudes in ascending order
            mag = sorted(mag)
            # add array bit
            mag_grid[iy, ix] = mag[stat_num-1]

            if arrays:
                for a in range(0,len(arrays['lon'])):
                    dx, dy = util_geo_km(lons[ix],
                                         lats[iy],
                                         arrays['lon'][a],
                                         arrays['lat'][a])
                    dz = np.abs(foc_depth - arrays['elev'][a])
                    hypo_dist = sqrt(dx**2 + dy**2 + dz**2)
                    # estimated noise level on array (rootn or another cleverer
                    # method to get a displaement number)
                    m = mag_min - mag_delta
                    ampl = 0
                    while ampl < snr*arrays['noise'][a]:
                        m = m + mag_delta
                        ampl = calc_ampl(m, hypo_dist, region)
                    array_mag.append(m)
                if np.min(array_mag) < mag_grid[iy,ix]:
                    mag_grid[iy, ix] = np.min(array_mag)
            
            if obs:
                for o in range(0, len(obs['longitude'])):
                    dx, dy = util_geo_km(lons[ix], lats[iy],
                                         obs['longitude'][o],
                                         obs['latitude'][o])
                    dz = np.abs(foc_depth - obs['elevation_km'][o])
                    hypo_dist = sqrt(dx**2 + dy**2 + dz**2)
                    #estimated noise level on array (rootn or another cleverer method to get a displaement number)
                    m = mag_min - mag_delta
                    ampl = 0
                    while ampl < snr*obs['noise [nm]'][o]:
                        m = m + mag_delta
                        ampl = calc_ampl(m, hypo_dist, region)
                    obs_mag.append(m)
                if obs_mag[obs_stat_num-1] < mag_grid[iy,ix]:
                    mag_grid[iy, ix] = obs_mag[obs_stat_num-1]
        
            del array_mag[:]
            del mag[:]
            del obs_mag[:]

    array = xarray.DataArray(mag_grid, coords=[lats,lons], dims=['Latitude','Longitude'])

    return array


def minML_x_section(stations_in, lon0, lat0, azi, length_km, min_depth=0, max_depth=20, ddist=5, ddepth=0.5,
                    stat_num=4, snr=3, region='CAL', mag_min=-3.0, mag_delta=0.1,
                    arrays=None, obs=None, obs_stat_num=3):

    '''
    Function to calculate a 2-D cross section of a SNCAST model. 
    X-section line defined by start lat/lon and the azimuth and length (in km) of the line

    Input should be a csv file (or Pandas DataFrame)
      longitude, latitude, noise [nm], station name
    e.g.: -7.5100, 55.0700, 0.53, IDGL

    Parameters
    ----------
        stations : DataFrame or csv filename
    '''

    if type(stations_in) == str:
        stations_df = pd.read_csv(stations_in)
    else:
        stations_df = stations_in.copy()

    lon = stations_df['longitude'].values
    lat = stations_df['latitude'].values
    elev = stations_df['elevation_km'].values
    stat = stations_df['station'].values
    noise = stations_df['noise [nm]'].values

    ## Calculate lon/lat co-ordinates for X-section line 
    ndists = int((length_km / ddist)+1)
    distance_km = np.linspace(0, length_km, ndists)
    xsection = pygc.great_circle(latitude=lat0, longitude=lon0,
                                 azimuth=azi, distance=distance_km*1e3)
    ndepths = int( (max_depth - min_depth) / ddepth) + 1
    depths = np.linspace(min_depth, max_depth, ndepths)
    # Iterate along cross-section
    mag=[]
    array_mag = []
    obs_mag = [] 
    # dets = {'Distance_km': [], 'Depth_km': [], 'ML_min':[]}
    mag_grid = np.zeros((ndepths, ndists))
    for i in range(0, ndists):
        # get lat/lon of each point on line
        ilat = xsection['latitude'][i]
        ilon = xsection['longitude'][i]
        # Iterate over depth 
        for d in range(ndepths):
            for s, _ in enumerate(stat):
                # loop through stations
                # calculate hypcocentral distance in km
                dz = elev[s] - depths[d]
                dx, dy = util_geo_km(ilon, ilat, lon[s], lat[s])
                hypo_dist = sqrt(dx**2 + dy**2 + dz**2)
                # find smallest detectable magnitude
                ampl = 0.0
                m = mag_min - mag_delta
                while ampl < snr*noise[s]:
                    m = m + mag_delta
                    ampl = calc_ampl(m, hypo_dist, region)
                mag.append(m)
            # sort magnitudes in ascending order
            mag = sorted(mag)
            mag_grid[d,i] = mag[stat_num-1]
            # add array bit
            if arrays:
                for a in range(0,len(arrays['lon'])):
                    dx, dy = util_geo_km(ilon, ilat, arrays['lon'][a], arrays['lat'][a])
                    hypo_dist = sqrt(dx**2 + dy**2 + dz**2)
                    m = mag_min - mag_delta
                    ampl = 0
                    while ampl < snr*arrays['noise'][a]:
                        m = m + mag_delta
                        ampl = calc_ampl(m, hypo_dist, region)
                    array_mag.append(m)
                if np.min(array_mag) < mag_grid[d, i]:
                    mag_grid[d, i] = np.min(array_mag)

            if obs:
                for o in range(0, len(obs['longitude'])):              
                    dz = np.abs(obs['elevation_km'][o] - depths[d])
                    dx, dy = util_geo_km(ilon, ilat,
                                         obs['longitude'][o],
                                         obs['latitude'][o])
                    hypo_dist = sqrt(dx**2 + dy**2 + dz**2)
                    hypo_dist = sqrt(dx**2 + dy**2 + dz**2)
                    # estimated noise level on array
                    # rootn or another cleverer method
                    # to get a displaement number)
                    m = mag_min - mag_delta
                    ampl = 0
                    while ampl < snr*obs['noise [nm]'][o]:
                        m = m + mag_delta
                        ampl = calc_ampl(m, hypo_dist, region)
                    obs_mag.append(m)
                if obs_mag[obs_stat_num-1] < mag_grid[d,i]:
                    mag_grid[d, i] = obs_mag[obs_stat_num-1]

            del array_mag[:]
            del mag[:]
            del obs_mag[:]

    # Make xarray grid to output

    array = xarray.DataArray(mag_grid,
                             coords=[depths, distance_km],
                             dims=['depth_km',
                                   'distance_along_xsection_km'])
    return array
