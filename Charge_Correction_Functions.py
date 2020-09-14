#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import the module
from __future__ import division, absolute_import
from __future__ import print_function
import itertools

from tshirt.pipeline import spec_pipeline

import matplotlib.pyplot as plt
from matplotlib import colors

#get_ipython().run_line_magic('matplotlib', 'inline')

#import bokeh to enable interactive plots
from bokeh.plotting import figure
from bokeh.io import output_notebook, push_notebook, show

output_notebook()

#import yaml to read in the parameter file
import yaml

#imports to use RECTE
import os
from astropy.table import QTable
import astropy.units as u
import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table, join
import pandas as pd
from astropy.time import Time


#import to copy
from copy import deepcopy

#modeling light curves
from scipy.optimize import curve_fit
import batman

#to fix errors
import pdb

#to correct for time differences
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation


# # Applying RECTE

# In[305]:


#! /usr/bin/env python
"""calculate RECTE model using a a template grism Image
"""

def RECTE(
        cRates,
        tExp,
        exptime=100.651947,
        trap_pop_s=200,
        trap_pop_f=0,
        dTrap_s=0,
        dTrap_f=0,
        dt0=0,
        lost=0,
        mode='staring'
):
    """Hubble Space Telescope ramp effet model

    Parameters:
    cRates -- intrinsic count rate of each exposures, unit e/s
    tExp -- start time of every exposures
    expTime -- (default 180 seconds) exposure time of the time series
    trap_pop -- (default 0) number of occupied traps at the beginning of the observations
    dTrap -- (default [0])number of extra trap added in the gap
    between two orbits
    dt0 -- (default 0) possible exposures before very beginning, e.g.,
    possible guiding adjustment
    lost -- (default 0, no lost) proportion of trapped electrons that are not eventually detected
    (mode) -- (default scanning, scanning or staring, or others), for scanning mode
      observation , the pixel no longer receive photons during the overhead
      time, in staring mode, the pixel keps receiving elctrons
    """
    nTrap_s = 1525.38 
    eta_trap_s = 0.013318 
    tau_trap_s = 1.63e4  # = 1.63e4
    nTrap_f = 162.38
    eta_trap_f = 0.008407
    tau_trap_f = 281.463
    
    #nTrap_s = 2192  # = 1525.38  # 1320.0
    #eta_trap_s = 0.02075  # = 0.013318  # 0.01311
    #tau_trap_s = 1.63e4  # = 1.63e4
    #nTrap_f = 225.7  # = 162.38
    #eta_trap_f = 0.0116  # = 0.008407
    #tau_trap_f = 3344  # = 281.463
    
    # nTrap_s = 1525.38  # 1320.0
    # eta_trap_s = 0.013318  # 0.01311
    # tau_trap_s = 1.63e4
    # nTrap_f = 162.38
    # eta_trap_f = 0.008407
    # tau_trap_f = 281.463

    try:
        dTrap_f = itertools.cycle(dTrap_f)
        dTrap_s = itertools.cycle(dTrap_s)
        dt0 = itertools.cycle(dt0)
    except TypeError:
        dTrap_f = itertools.cycle([dTrap_f])
        dTrap_s = itertools.cycle([dTrap_s])
        dt0 = itertools.cycle([dt0])
    obsCounts = np.zeros(len(tExp))
    trap_pop_s = min(trap_pop_s, nTrap_s)
    trap_pop_f = min(trap_pop_f, nTrap_f)
    dEsList = np.zeros(len(tExp))
    dEfList = np.zeros(len(tExp))
    dt0_i = next(dt0)
    f0 = cRates[0]
    c1_s = eta_trap_s * f0 / nTrap_s + 1 / tau_trap_s  # a key factor
    c1_f = eta_trap_f * f0 / nTrap_f + 1 / tau_trap_f
    dE0_s = (eta_trap_s * f0 / c1_s - trap_pop_s) * (1 - np.exp(-c1_s * dt0_i))
    dE0_f = (eta_trap_f * f0 / c1_f - trap_pop_f) * (1 - np.exp(-c1_f * dt0_i))
    dE0_s = min(trap_pop_s + dE0_s, nTrap_s) - trap_pop_s
    dE0_f = min(trap_pop_f + dE0_f, nTrap_f) - trap_pop_f
    trap_pop_s = min(trap_pop_s + dE0_s, nTrap_s)
    trap_pop_f = min(trap_pop_f + dE0_f, nTrap_f)
    for i in range(len(tExp)):
        try:
            dt = tExp[i+1] - tExp[i]
        except IndexError:
            dt = exptime
        f_i = cRates[i]
        c1_s = eta_trap_s * f_i / nTrap_s + 1 / tau_trap_s  # a key factor
        c1_f = eta_trap_f * f_i / nTrap_f + 1 / tau_trap_f
        # number of trapped electron during one exposure
        dE1_s = (eta_trap_s * f_i / c1_s - trap_pop_s) * (1 - np.exp(-c1_s * exptime))
        dE1_f = (eta_trap_f * f_i / c1_f - trap_pop_f) * (1 - np.exp(-c1_f * exptime))
        dE1_s = min(trap_pop_s + dE1_s, nTrap_s) - trap_pop_s
        dE1_f = min(trap_pop_f + dE1_f, nTrap_f) - trap_pop_f
        trap_pop_s = min(trap_pop_s + dE1_s, nTrap_s)
        trap_pop_f = min(trap_pop_f + dE1_f, nTrap_f)
        obsCounts[i] = f_i * exptime - dE1_s - dE1_f
        if dt < 5 * exptime:  # whether next exposure is in next batch of exposures
            # same orbits
            if mode == 'scanning':
                # scanning mode, no incoming flux between exposures
                dE2_s = - trap_pop_s * (1 - np.exp(-(dt - exptime)/tau_trap_s))
                dE2_f = - trap_pop_f * (1 - np.exp(-(dt - exptime)/tau_trap_f))
                dEsList[i] = dE1_s + dE2_s
                dEfList[i] = dE1_f + dE2_f
            elif mode == 'staring':
                # for staring mode, there is flux between exposures
                dE2_s = (eta_trap_s * f_i / c1_s - trap_pop_s) * (1 - np.exp(-c1_s * (dt - exptime)))
                dE2_f = (eta_trap_f * f_i / c1_f - trap_pop_f) * (1 - np.exp(-c1_f * (dt - exptime)))
            else:
                # others, same as scanning
                dE2_s = - trap_pop_s * (1 - np.exp(-(dt - exptime)/tau_trap_s))
                dE2_f = - trap_pop_f * (1 - np.exp(-(dt - exptime)/tau_trap_f))
            trap_pop_s = min(trap_pop_s + dE2_s, nTrap_s)
            trap_pop_f = min(trap_pop_f + dE2_f, nTrap_f)
        elif dt < 1200:
            trap_pop_s = min(trap_pop_s * np.exp(-(dt-exptime)/tau_trap_s), nTrap_s)
            trap_pop_f = min(trap_pop_f * np.exp(-(dt-exptime)/tau_trap_f), nTrap_f)
        else:
            # switch orbit
            dt0_i = next(dt0)
            trap_pop_s = min(trap_pop_s * np.exp(-(dt-exptime-dt0_i)/tau_trap_s) + next(dTrap_s), nTrap_s)
            trap_pop_f = min(trap_pop_f * np.exp(-(dt-exptime-dt0_i)/tau_trap_f) + next(dTrap_f), nTrap_f)
            f_i = cRates[i + 1]
            c1_s = eta_trap_s * f_i / nTrap_s + 1 / tau_trap_s  # a key factor
            c1_f = eta_trap_f * f_i / nTrap_f + 1 / tau_trap_f
            dE3_s = (eta_trap_s * f_i / c1_s - trap_pop_s) * (1 - np.exp(-c1_s * dt0_i))
            dE3_f = (eta_trap_f * f_i / c1_f - trap_pop_f) * (1 - np.exp(-c1_f * dt0_i))
            dE3_s = min(trap_pop_s + dE3_s, nTrap_s) - trap_pop_s
            dE3_f = min(trap_pop_f + dE3_f, nTrap_f) - trap_pop_f
            trap_pop_s = min(trap_pop_s + dE3_s, nTrap_s)
            trap_pop_f = min(trap_pop_f + dE3_f, nTrap_f)
        trap_pop_s = max(trap_pop_s, 0)
        trap_pop_f = max(trap_pop_f, 0)

    return obsCounts


# In[306]:


def RECTEMulti(template,
                 variability,
                 tExp,
                 exptime,
                 trap_pop_s=200,
                 trap_pop_f=0,
                 dTrap_s=0,
                 dTrap_f=0,
                 dt0=0,
                 mode='staring'):
    """loop through every pixel in the template
    calculate for 6 orbit
    return
    model light curves

    template -- a template image of the input sereis
    variablities -- normalized model light curves
    tExp -- starting times of each exposure of the time resolved observations
    trap_pop_s -- (default=0)number of initially occupied traps -- slow poplulation
    trap_pop_f -- number of initially occupied traps -- fast poplulation
    dTrap_s -- (default=0, can be either number or list) number of extra
        trapped charge carriers added in the middle of two orbits
        -- slow population. If it is a number, it assumes that all
        the extra added trap charge carriers are the same
    dTrap_f -- (default=0, can be either number or list) number of extra
         trapped charge carriers added in the middle of two orbits
        -- fast population. If it is a number, it assumes that all
        the extra added trap charge carriers are the same
    """
    specShape = template.shape #shape of template image of the input sereis. it gives the dimensions of an array 
    outSpec = np.zeros((specShape[0], len(tExp))) #specShape[0] gives the number of rows, this is the shape of this new array. this would return something like : array([0, 0, 0, 0, 0]) but with a length desired. 
    
    for i in range(specShape[0]):
        outSpec[i, :] = RECTE(
            variability * template[i],
            tExp,
            exptime,
            trap_pop_s,
            trap_pop_f,
            dTrap_s=dTrap_s,
            dTrap_f=dTrap_f,
            dt0=dt0,
            lost=0,
            mode=mode)
    return np.sum(outSpec, axis=(0))


# In[307]:


def calculate_correction(csv_file,median_image):
    '''
    Calculate the RECTE ramp correction 
    
    Parameters
    ----------
    csv_file: file
        Read in a csv file with all the required data. 
        
    median_image: fits file
        Read in a fits file of the median image         
        '''
    info = pd.read_csv(
        csv_file,
        parse_dates=True,
        index_col='Time (UTC)')
    info['Time'] = np.float32(info.index - info.index.values[0]) / 1e9
    grismInfo = info[info['Filter'] == 'G141']
    exptime = grismInfo['Exp Time'].values[0]
    tExp = grismInfo['Time'].values
    tExp = tExp - tExp[0]
    # cRates = np.ones(len(LC)) * LC.mean() * 1.002
    cRates = np.ones(len(tExp))
    variability = cRates / cRates.mean()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = fits.getdata(median_image)
   # bbox = [0, 128, 59, 89]  # define the bounding box of the area of interest
    bbox = [0, 128, 69, 79]
    xList = np.arange(bbox[0], bbox[1])
    ramps = np.zeros((len(xList), len(tExp)))
    dTrap_fList = [0]
    dTrap_sList = [0]
    dtList = [0]
    full_well = 8e4
    for i, x in enumerate(xList):
        template = im[bbox[2]:bbox[3], x] 
        for j, flux in enumerate(template):
            if flux * exptime > full_well:
                template[j] = full_well / exptime

        obs = RECTEMulti(template, variability, tExp, exptime,
                         dTrap_f=dTrap_fList,
                         dTrap_s=dTrap_sList,
                         trap_pop_f=0,
                         trap_pop_s=200,
                         dt0=dtList,
                         mode='staring')
        obs = obs / exptime / np.nansum(template)
        # ax.plot(tExp, obs, '.', color='0.8', ms=1)
        ramps[i, :] = obs
    ax.plot(tExp, ramps[30, :], '.')
    #plt.show()
    return ramps


# In[308]:


def calculate_correction_fast(x,exptime,median_image,dtrap_s=[0],trap_pop_s=200,xList=np.arange(0,13)):
    '''
    Calculate the RECTE ramp correction: fast-version 
    
    Parameters
    ----------
    x:  
       Time in JD
         
    exptime: int
        Defines the exposure time for the observation
        
     median_image: fits file
        Read in a fits file of the median image
    
    trap_pop_s: int
        (default=0)number of initially occupied traps -- slow poplulation
   
    dTrap_s: int
        (default=0, can be either number or list) number of extra
        trapped charge carriers added in the middle of two orbits
        -- slow population. If it is a number, it assumes that all
        the extra added trap charge carriers are the same
        
    xList: list 
        A list on the range of Dispersion
    
    '''
    tExp=(x-x[0])*3600*24
    # cRates = np.ones(len(LC)) * LC.mean() * 1.002
    cRates = np.ones(len(tExp))
    variability = cRates / cRates.mean()
    im = median_image
   # bbox = [0, 128, 59, 89]  # define the bounding box of the area of interest
    bbox = [0, 128, 69, 79]
    xList = xList
    ramps = np.zeros((len(xList), len(tExp)))
    dTrap_fList = [0]
    dTrap_sList = [0]
    dtList = [0]
    full_well = 8e4
    for i, x in enumerate(xList):
        template = im[bbox[2]:bbox[3], x]
        for j, flux in enumerate(template):
            if flux * exptime > full_well:
                template[j] = full_well / exptime

        obs = RECTEMulti(template, variability, tExp, exptime,
                         dTrap_f=dTrap_fList,
                         dTrap_s=dtrap_s,
                         trap_pop_f=0,
                         trap_pop_s=trap_pop_s,
                         dt0=dtList,
                         mode='staring')
        obs = obs / exptime / np.nansum(template)
        # ax.plot(tExp, obs, '.', color='0.8', ms=1)
        ramps[i, :] = obs
    return ramps


# In[309]:


def charge_correction(self,ramps):
    '''
    Returns the ramp corrected flux data 
    
    Parameters
    ----------
    
    ramps:
        Returns the data for correcting ramp effect. 
        
    '''
    HDUList = fits.open(self.specFile)
    origData = HDUList['OPTIMAL SPEC'].data
    
    newData = deepcopy(origData)
    newData[0,:,:] = origData[0,:,:] / ramps.transpose()
    
    HDUList['OPTIMAL SPEC'].data = newData
    correctedSpecFile = os.path.splitext(self.specFile)[0]+'_corrected.fits'
    HDUList.writeto(correctedSpecFile,overwrite=True)
    
    new_param = deepcopy(self.param)
    new_param['srcNameShort'] = 'corot1_corrected'
    new_spec = spec_pipeline.spec(directParam=new_param)
    new_spec.specFile = correctedSpecFile

    
    return newData,new_spec


# In[ ]:




