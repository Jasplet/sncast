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
import sys
from obspy.signal.util import util_geo_km
from math import pow, log10, sqrt

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

def minML(filename, dir_in='./', lon0=-12, lon1=-4, lat0=50.5, lat1=56.6, dlon=0.33,
          dlat=0.2,stat_num=4, snr=3, foc_depth=0, region='CAL', mag_min=-3.0, mag_delta=0.1):
    """
    This routine calculates the geographic distribution of the minimum 
    detectable local magnitude ML for a given seismic network. Required 
#### 9.10.2020    input is a file containg four comma separated
    columns containing for each seismic station:

         longitude, latitude, noise [nm], station name
    e.g.: -7.5100, 55.0700, 0.53, IDGL

    The output file *.grd lists in ASCII xyz format: longitud, latitude, ML
  
    Optional parameters are:

    :param  dir_in:	full path to input and output file
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
#### 9.10.2020    array_in = np.genfromtxt('%s/%s.dat' %(dir_in, filename), dtype=None, delimiter=",")
#### 9.10.2020    array_in = np.genfromtxt('%s/%s.dat' %(dir_in, filename), encoding='ASCII', dtype=None, delimiter=",")
    array_in = np.genfromtxt('%s/%s' %(dir_in, filename), encoding='ASCII', dtype=None, delimiter=",", skip_header=1)
    lon = ([t[0] for t in array_in])

    lat = [t[1] for t in array_in]
    noise = [t[2] for t in array_in]
    stat = [t[3] for t in array_in]
    # grid size
    nx = int( (lon1 - lon0) / dlon) + 1
    ny = int( (lat1 - lat0) / dlat) + 1
    # open output file:
### 9.10.2020    f = open('%s/%s-stat%s-foc%s-snr%s-%s.grd' %(dir_in, filename, stat_num, foc_depth, snr, region), 'wb')
    mag=[]
    dets = {'lat':[], 'lon':[],'mag':[]}
    for ix in range(nx): # loop through longitude increments
        ilon = lon0 + ix*dlon
        for iy in range(ny): # loop through latitude increments
            ilat = lat0 + iy*dlat 
            for j,jstat in enumerate(stat): # loop through stations 
                # calculate hypcocentral distance in km
                dx, dy = util_geo_km(ilon, ilat, lon[j], lat[j])
                hypo_dist = sqrt(dx**2 + dy**2 + foc_depth**2)
                # find smallest detectable magnitude
                ampl = 0.0
                m = mag_min - mag_delta
                while ampl < snr*noise[j]: 
                    m = m + mag_delta
                    ampl = calc_ampl(m, hypo_dist, region)
                mag.append(m)
            # sort magnitudes in ascending order
            mag = sorted(mag)
            # write out lonngitude, latitude and smallest detectable magnitude
            dets['lon'].append(ilon)
            dets['lat'].append(ilat)
            dets['mag'].append(mag[stat_num-1])
            del mag[:]
    return dets

# utility functions added by J Asplet (2024)

def estimate_noise_displacement(station_ppsd, f0=5, n=0.5, case='worst'):
    '''
    Implements equation 5 from Mölhoff et al., (2019) to estiamte noise displacement within a narrow frequency band of interest.

    Adapted from equations in Haskov and Alguacil (2006)

    See: 
    
    Havskov and Alguacil (2016), Instrumentation in Earthquake Seismology (2nd Edition), https://doi.org/10.1007/978-3-319-21314-9
    
    Möllhoff, M., Bean, C. J., & Baptie, B. J. (2019).
    SN-CAST: seismic network capability assessment software tool for regional networks-examples from Ireland. Journal of Seismology, 23, 493-504.

    
    '''

    
    f1, f2 = get_freq_range_from_centre(f0,n)
    
    if case == 'worst':
        period, psd_accl = station_ppsd.get_percentile(95)
        freqs_psd = 1/period
    elif case == 'mode':
        period, psd_accl = station_ppsd.get_mode()
        freqs_psd = 1/period
        
    mean_psd_db = psd_accl[(freqs_psd >= f1) & (freqs_psd <= f2)].mean()
    
    displacement = psd_db_to_displacement_amplitude(mean_psd_db, f0=f0, f1=f1, f2=f2)
    
    return displacement

def get_freq_range_from_centre(f0, n=0.5):
    f1 = f0*2**(-n/2) # lower bound of frequency span
    f2 = f0*2**(n/2) # upper bound of frequency span
    return f1, f2

def psd_db_to_displacement_amplitude(psd_in_db, f1, f2, f0=None):
    '''
    Take acceleration power density (in dB relative to 1 ((m/s)^2)^2 / Hz) and calculates
    Displacement in meters
    '''
    if f0 is None:
        f0 = np.sqrt(f1*f2) # find centre frequency 
        
    psd = psd_db_convert(psd_in_db)
    displ = (3.75)/((2*np.pi*f0)**2) * np.sqrt(psd * (f2 - f1))

    return displ
    
def psd_db_convert(psd_in_db):
    '''
    Convert probsabilistic PSD values from db relative to 1 ((m/s)^2)^2 / Hz
    '''
    return np.power(10, psd_in_db/10)