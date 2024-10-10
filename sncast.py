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

import numpy as np
import pandas as pd
from pathlib import Path
from obspy.signal.util import util_geo_km
from math import pow, log10, sqrt
import pygc 
import xarray

def calc_ampl(local_mag, hypo_dist, region):

        # region specific ML = log(ampl) + a*log(hypo-dist) + b*hypo_dist + c
    if region == 'UK':
        #UK Scale uses new ML equation from Luckett et al., (2019) https://doi.org/10.1093/gji/ggy484
        # Takes form local_mag = log(amp) + a*log(hypo-dist) + b*hypo-dist + d*exp(e * hypo-dist) + c
        a = 1.11
        b = 0.00189
        c = -2.09
        d = -1.16
        e = -0.2
        ampl = np.power(10, (local_mag - a*np.log10(hypo_dist) - b*hypo_dist - c - d*np.exp(e*hypo_dist)))

    elif region == 'CAL': # South. California scale, IASPEI (2005), 
                          # www.iaspei.org/commissions/CSOI/summary_of_WG_recommendations_2005.pdf
        a = 1.11
        b = 0.00189
        c = -2.09
        ampl = np.power(10, (local_mag - a*np.log10(hypo_dist) - b*hypo_dist - c))

    return ampl

def minML(stations_in, lon0=-12, lon1=-4, lat0=50.5, lat1=56.6, dlon=0.33,
          dlat=0.2,stat_num=4, snr=3, foc_depth=0, region='CAL', mag_min=-2.0, mag_delta=0.1,
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
    if type(stations_in) == str:
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
### 9.10.2020    f = open('%s/%s-stat%s-foc%s-snr%s-%s.grd' %(dir_in, filename, stat_num, foc_depth, snr, region), 'wb')
    mag=[]
    array_mag = []
    mag_grid = np.zeros((ny, nx))
    for ix in range(nx): # loop through longitude increments
        for iy in range(ny): # loop through latitude increments
            for j,jstat in enumerate(stat): # loop through stations 
                # calculate hypcocentral distance in km
                dx, dy = util_geo_km(lons[ix], lats[iy], stat_lon[j], stat_lat[j])
                dz = np.abs(foc_depth - stat_elev[j])
                hypo_dist = sqrt(dx**2 + dy**2 + dz**2)
                # find smallest detectable magnitude
                ampl = 0.0
                m = mag_min - mag_delta
                while ampl < snr*noise[j]: 
                    m = m + mag_delta
                    ampl = calc_ampl(m, hypo_dist, region)
                mag.append(m)
            # sort magnitudes in ascending order
            mag = sorted(mag)
            # add array bit
            mag_grid[iy, ix] = mag[stat_num-1]

            if arrays:
                for a in range(0,len(arrays['lon'])):
                    dx, dy = util_geo_km(ilon, ilat, arrays['lon'][a], arrays['lat'][a])
                    dz = np.abs(foc_depth - arrays['elev'][a])
                    hypo_dist = sqrt(dx**2 + dy**2 + dz**2)
                    #estimated noise level on array (rootn or another cleverer method to get a displaement number)
                    m = mag_min - mag_delta
                    ampl = 0
                    while ampl < snr*arrays['noise'][a]:
                        m = m + mag_delta
                        ampl = calc_ampl(m, hypo_dist, region)
                    array_mag.append(m)
                if np.min(array_mag) < mag_grid[iy,ix]:
                    mag_grid[iy, ix] = np.min(array_mag)
            
            if obs:
                for o in range(0, len(obs['lon'])):
                    dx, dy = util_geo_km(ilon, ilat, arrays['lon'][a], arrays['lat'][a])
                    dz = np.abs(foc_depth - arrays['elev'][a])
                    hypo_dist = sqrt(dx**2 + dy**2 + dz**2)
                    #estimated noise level on array (rootn or another cleverer method to get a displaement number)
                    m = mag_min - mag_delta
                    ampl = 0
                    while ampl < snr*arrays['noise'][a]:
                        m = m + mag_delta
                        ampl = calc_ampl(m, hypo_dist, region)
                    obs_mag.append(m)
                if np.min(obs_mag) < mag_grid[iy,ix]:
                    mag_grid[iy, ix] = mag[obs_stat_num-1]
        
            del array_mag[:]
            del mag[:]
            del obs_mag[:]

    array = xarray.DataArray(mag_grid, coords=[lats,lons], dims=['Latitude','Longitude'])

    return array


def minML_x_section(stations_in, lon0, lat0, azi, length_km, min_depth=0, max_depth=20, ddist=5, ddepth=0.5,
                    stat_num=4, snr=3, region='CAL', mag_min=-3.0, mag_delta=0.1, arrays=None):
    
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
    # dets = {'Distance_km': [], 'Depth_km': [], 'ML_min':[]}
    mag_grid = np.zeros((ndepths, ndists))
    for i in range(0, ndists):
        # get lat/lon of each point on line
        ilat = xsection['latitude'][i]
        ilon = xsection['longitude'][i]
        dist_km = distance_km[i]
        # Iterate over depth 
        for d in range(ndepths):
            for s, _ in enumerate(stat) : # loop through stations 
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
            # add array bit    
            if arrays:
                for a in range(0,len(arrays['lon'])):
                    dx, dy = util_geo_km(ilon, ilat, arrays['lon'][a], arrays['lat'][a])
                    hypo_dist = sqrt(dx**2 + dy**2 + dz**2)
                    #estimated noise level on array (rootn or another cleverer method to get a displaement number)
                    m = mag_min - mag_delta
                    ampl = 0
                    while ampl < snr*arrays['noise'][a]:
                        m = m + mag_delta
                        ampl = calc_ampl(m, hypo_dist, region)
                    array_mag.append(m)
                if np.min(array_mag) < mag[stat_num-1]:
                    mag_grid[d, i] = np.min(array_mag)
                else:
                    mag_grid[d, i] = mag[stat_num-1]
            else:
                mag_grid[d, i] = mag[stat_num-1]

            del array_mag[:]
            del mag[:]

    # Make xarray grid to output

    array = xarray.DataArray(mag_grid, coords=[depths,distance_km ], dims=['depth_km','distance_along_xsection_km'])
    return array