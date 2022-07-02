#IMPORT LIBRARY:

from tshirt.pipeline import spec_pipeline #import the spectroscopic module from the tshirt pipeline


#astropy imports
import astropy.units as u
from astropy.io import fits, ascii
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation

#General Imports
import yaml 
import numpy as np
import pathlib
import os
from multiprocessing import Pool
from functools import partial
import tqdm
import pdb
import time

#Plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
from bokeh.plotting import figure
from bokeh.io import output_notebook, push_notebook, show
output_notebook()

#Light-Curve Modeling Imports
import batman
from scipy.optimize import curve_fit
from scipy.optimize import minimize

#Corner & emceee 
import corner
import emcee

#RECTE Imports 
import itertools
#----------------------------------------------------------------------------------------
class lightcurve_model(object):
    
    def __init__(self, tshirt_obj, planet_file):
        """
        Called when an object is created from the class to initialize (assign values) to the attributes of the class.
            
        Parameters:
        -----------
        self: keyword
            Class object
        tshirt_obj: keyword
            Time Series Helper & Integration Reduction Tool (tshirt) Spectrometry Object
        planet_file: YAML file
            The YAML file containing desired planet parameters
        """
        self.planet_file = planet_file  #Assign the object a planet parameter YAML file
        with open(planet_file, "r") as stream:  #Open the planet parameter YAML file; call it planet_data
            self.planet_data = yaml.safe_load(stream)
            
        results = tshirt_obj.get_wavebin_series()  #Obtain a table of the the wavelength-binned time series from tshirt.
        raw_results = results[0].to_pandas() #Put the flux values and time in a pandas table
        
        time_correction = self.barycenter_correction(tshirt_obj) #Call the barycenter time correction function. Returns correction in days. 
        self.x = raw_results['Time'].values+time_correction #Assign the object's time data accounting for barycenter correction in days.
        
        self.im = self.median_image(tshirt_obj)  #Define the median image data (im) by calling the median_image function
        
        head = fits.getheader(tshirt_obj.fileL[0], extname="PRIMARY")  #Grab the header information from one of the fits files
        self.exptime=head['EXPTIME'] #Define the exposure time from the header information
        self.obsdate = head['DATE-OBS'] #Define observation date for plotting purposes
        
        #Obtain a table of wavelength bins (default nbins=1), with theoretical noise and measured standard deviation across time.
        table_noise = tshirt_obj.print_noise_wavebin(nbins=1)
        table_noise = table_noise.to_pandas() #convert to a pandas table
        #Defining the global xList parameter needed in the model functions with RECTE
        Disp_st = table_noise['Disp St'][0] #Start of the dispersion range
        Disp_end = table_noise['Disp End'][0] #End of the dispersion range
        Disp_xList = np.arange(Disp_st, Disp_end,1) #Return Numpy array of evenly spaced values within a given interval. 
        self.xList = Disp_xList
        
        #Define Bounding Box around source based on apWidth, starPosition, and dispersion pixels
        bbox = []
        bbox.append(self.planet_data['dispPixels'][0])
        bbox.append(self.planet_data['dispPixels'][1])
        bbox.append(int(self.planet_data['starPositions'][0]) - int(self.planet_data['apWidth']/2))
        bbox.append(int(self.planet_data['starPositions'][0]) + int(self.planet_data['apWidth']/2))
        self.bbox = bbox
        
    def barycenter_correction(self,tshirt_obj):
        '''
        Calculates the barycentric time difference 
        
        Parameters
        ----------
        tshirt_obj: keyword
            Time Series Helper & Integration Reduction Tool (tshirt) Spectrometry Object
        
        Retruns
        ---------
        The barycentric time difference in days 
        '''
        results = tshirt_obj.get_wavebin_series()  #Obtain a table of the the wavelength-binned time series from tshirt.
        raw_results = results[0].to_pandas()       #Put the flux values and time in a pandas table

        head = fits.getheader(tshirt_obj.fileL[0]) #Grab the header information from one of the fits files
            
        expStartJD = head['EXPSTART'] + 2400000.5  #Define the exposure start time in Julian Days
         
        raw_results = Time(raw_results['Time'][0],format='jd') #Re-define the raw results table making the time column is in Julian Days
            
        coord = SkyCoord('06 48 19.1724141241 -03 06 07.710423478',unit=(u.hourangle,u.deg)) #Define Coordinates and Location
        loc = EarthLocation.of_site('keck')
            
        diff = raw_results.light_travel_time(coord,location=loc) #Find the time difference 
        
        #print("Barycenter correction (days) = "+str((diff / u.day).si))
            
        return (diff / u.day).si #Return the barycenter time correction in days. 
    
    def median_image(self, tshirt_obj, showPlot=False, recalculation = False):
        '''
        Generates a median image of the fits files 
        
        Parameters
        ----------
        tshirt_obj: keyword
            Time Series Helper & Integration Reduction Tool (tshirt) Spectrometry Object
        showPlot: bool
            Make the plot visible? The Default is "False"
        
        Returns
        --------
        medianImage: array
            Array data of the median image fluxes
        '''
        planet_data = self.planet_data  #Call the planet file data
        
        new_dir = pathlib.Path(planet_data['BaseDir'], 'Median_Images') #Establish a path to the median images based on base directory
        new_dir.mkdir(parents=True, exist_ok=True) #Make the new directory
        
        #Generate a name for the median image file based on planet parameters
        filename = os.path.join(new_dir,"{}_MedianImage_{}".format(planet_data['srcName'],tshirt_obj.param['nightName'])) 
        
        #If the previously defined lightcurve_file exists and the recalculation parameter is set to False, read it in. 
        if (os.path.exists(filename+".fits") == True) and (recalculation != True):
            medianImage = fits.getdata(filename+".fits")
        else:
            head = fits.getheader(tshirt_obj.fileL[0], extname='SCI')  #Grab the SCI header information from one of the fits files

            #Generate a 3D array of zeros; the size is based on header information
            cube3d = np.zeros([len(tshirt_obj.fileL),head['NAXIS2'],head['NAXIS1']]) 
            
            for ind,oneFile in enumerate(tshirt_obj.fileL): #Loop through all the fits files and append the image data to the 3D array 
                cube3d[ind,:,:] = fits.getdata(oneFile,extname='SCI')
    
            medianImage = np.median(cube3d,axis=0)  #Find the median of the data
                    
            outHDU = fits.PrimaryHDU(medianImage,head) #Write the median image data to a file
            outHDU.writeto(filename+".fits", overwrite=True)
            plt.imsave(filename+".pdf",medianImage)  #Save the median image as a pdf                                           
            
        if showPlot==True:
            medianImage_plot = plt.imshow(medianImage) #If True, plot the median image 
            plt.title("{} Median Image {}".format(planet_data['srcName'],tshirt_obj.param['nightName']))
            plt.xlabel("x-pixels")
            plt.ylabel("y-pixels")            
            plt.show()
        else:
            None
        #print("Median Iamge Found at: " + str(filename))
        return medianImage
    
    def RECTE(
        self,
        cRates,
        tExp,
        exptime=180,
        trap_pop_s=200,
        trap_pop_f=0,
        dTrap_s=0,
        dTrap_f=0,
        dt0=0,
        lost=0,
        mode='staring'):
        """
        Hubble Space Telescope ramp effet model
        
        Parameters
        ----------
        cRates: array
            Intrinsic count rate of each exposures, unit e/s. Is now a 2D array
        tExp: float
            Start time of every exposures
        expTime: float
            Exposure time of the time series (default 180 seconds)
        trap_pop: int
            Number of occupied traps at the beginning of the observations (default 0)
        dTrap: int 
            Number of extra trap added in the gap between two orbits (default [0])
        dt0: int 
           Possible exposures before very beginning (default 0), e.g., possible guiding adjustment 
        lost: int 
            Proportion of trapped electrons that are not eventually detected (default 0, no lost)
        mode: str
           For scanning mode observation , the pixel no longer receive photons during the overhead time, in staring mode, the pixel
           keps receiving elctrons (default scanning, scanning or staring, or others)
        
        Returns
        -------
        obsCounts: array
            The modeled ramp effect
            
    """
        exptime=self.exptime #Exposure time
        
        #Obtained from Hubble Values (Zhou et al.)
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
            
        #Create an obsCounts array the same size as the cRates array
        obsCounts = np.zeros_like(cRates)
        trap_pop_s = min(trap_pop_s, nTrap_s)
        trap_pop_f = min(trap_pop_f, nTrap_f)
        dEsList = np.zeros(len(tExp))
        dEfList = np.zeros(len(tExp))
        dt0_i = next(dt0)
        #cRates has the time element along the y-direction (the rows) and the pixels data along the x-direction (the columns)
        f0 = cRates[0] #grabs the first element(a 1D array) of the 2D array
        
        c1_s = eta_trap_s * f0 / nTrap_s + 1 / tau_trap_s  # a key factor
        c1_f = eta_trap_f * f0 / nTrap_f + 1 / tau_trap_f
    
        dE0_s = (eta_trap_s * f0 / c1_s - trap_pop_s) * (1 - np.exp(-c1_s * dt0_i))
        dE0_f = (eta_trap_f * f0 / c1_f - trap_pop_f) * (1 - np.exp(-c1_f * dt0_i))
        
        #np.minimum will compare each element of the array to the constant value of nTrap_s returning a minimum of the array element-wise
        dE0_s = np.minimum(trap_pop_s + dE0_s, nTrap_s) - trap_pop_s
        dE0_f = np.minimum(trap_pop_f + dE0_f, nTrap_f) - trap_pop_f
        trap_pop_s = np.minimum(trap_pop_s + dE0_s, nTrap_s)
        trap_pop_f = np.minimum(trap_pop_f + dE0_f, nTrap_f)
    
        #for loop over the time element
        for i in range(len(tExp)):
            try:
                dt = tExp[i+1] - tExp[i]
            except IndexError:
                dt = exptime
            # cRates[i] will sequently grab each element(a 1D array)in the 2D array. 
            f_i = cRates[i]
            c1_s = eta_trap_s * f_i / nTrap_s + 1 / tau_trap_s  # a key factor
            c1_f = eta_trap_f * f_i / nTrap_f + 1 / tau_trap_f
            # number of trapped electron during one exposure
            dE1_s = (eta_trap_s * f_i / c1_s - trap_pop_s) * (1 - np.exp(-c1_s * exptime))
            dE1_f = (eta_trap_f * f_i / c1_f - trap_pop_f) * (1 - np.exp(-c1_f * exptime))
            dE1_s = np.minimum(trap_pop_s + dE1_s, nTrap_s)- trap_pop_s
            dE1_f = np.minimum(trap_pop_f + dE1_f, nTrap_f)- trap_pop_f
            trap_pop_s = np.minimum(trap_pop_s + dE1_s, nTrap_s)
            trap_pop_f = np.minimum(trap_pop_f + dE1_f, nTrap_f)
            
            #obsCount for each 1D array element from the 2D array
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
                trap_pop_s = np.minimum(trap_pop_s + dE2_s, nTrap_s)
                trap_pop_f = np.minimum(trap_pop_f + dE2_f, nTrap_f)
            elif dt < 1200:
                trap_pop_s = np.minimum(trap_pop_s * np.exp(-(dt-exptime)/tau_trap_s),nTrap_s)
                trap_pop_f = np.minimum(trap_pop_f * np.exp(-(dt-exptime)/tau_trap_f),nTrap_f)
            else:
                # switch orbit
                dt0_i = next(dt0)
                trap_pop_s = np.minimum(trap_pop_s * np.exp(-(dt-exptime-dt0_i)/tau_trap_s) + next(dTrap_s), nTrap_s)
                trap_pop_f = np.minimum(trap_pop_f * np.exp(-(dt-exptime-dt0_i)/tau_trap_f) + next(dTrap_f), nTrap_f)
                f_i = cRates[i+1]
                c1_s = eta_trap_s * f_i / nTrap_s + 1 / tau_trap_s  # a key factor
                c1_f = eta_trap_f * f_i / nTrap_f + 1 / tau_trap_f
                dE3_s = (eta_trap_s * f_i / c1_s - trap_pop_s) * (1 - np.exp(-c1_s * dt0_i))
                dE3_f = (eta_trap_f * f_i / c1_f - trap_pop_f) * (1 - np.exp(-c1_f * dt0_i))
                dE3_s = np.minimum(trap_pop_s + dE3_s, nTrap_s) - trap_pop_s
                dE3_f = np.minimum(trap_pop_f + dE3_f, nTrap_f) - trap_pop_f
                trap_pop_s = np.minimum(trap_pop_s + dE3_s, nTrap_s)
                trap_pop_f = np.minimum(trap_pop_f + dE3_f, nTrap_f)
            trap_pop_s = np.maximum(trap_pop_s, 0)
            trap_pop_f = np.maximum(trap_pop_f, 0)
    
        return obsCounts

    def RECTEMulti(self,
                   template,
                   variability,
                   tExp,
                   exptime,
                   trap_pop_s=200,
                   trap_pop_f=0,
                   dTrap_s=0,
                   dTrap_f=0,
                   dt0=0,
                   mode='staring',doSum=False):
        """
        Loop through every pixel in the template, calculate for 6 orbit
        
        Parameters
        ----------
        template:
            A template image of the input sereis
        variablities:
            Normalized model light curves
        tExp: float
            Starting times of each exposure of the time resolved observations
        trap_pop_s: int
            Number of initially occupied traps -- slow poplulation (default=200)
        trap_pop_f: int
            Number of initially occupied traps -- fast poplulation (default = 0)
        dTrap_s: int, list
           Number of extra trapped charge carriers added in the middle of two orbits -- slow population. 
           If it is a number, it assumes that all the extra added trap charge carriers are the same (default=0, 
           can be either number or list)
        dTrap_f: int, list
           Number of extra trapped charge carriers added in the middle of two orbits -- fast population. 
           If it is a number, it assumes that all the extra added trap charge carriers are the same (default=0, 
           can be either number or list)
        doSum: bool
            Sum the effect of all pixels?
    
        Returns 
        -------
        Return model light curves
        """
        #multiplies outter product of two vectors out[i, j] = variability[i] * template[j]
        rates2D = np.outer(variability,template)
        outSpec = self.RECTE(
                rates2D,
                tExp,
                exptime,
                trap_pop_s,
                trap_pop_f,
                dTrap_s=dTrap_s,
                dTrap_f=dTrap_f,
                dt0=dt0,
                lost=0,
                mode=mode)
        #transpose the array in order to sum along the zero axis. 
        if doSum == True:
            return np.sum(outSpec.transpose(),axis=0)
        else:
            return outSpec.transpose()
    
    def calculate_correction_fast(self,x,exptime,median_image,dTrap_s=[0],trap_pop_s=200,dTrap_f=[0],trap_pop_f=0,xList=np.arange(0,13)):
        '''
        Calculate the RECTE ramp correction: fast-version 
        
        Parameters
        ----------
        x: array 
            Time Array in JD   
        exptime: int
            Defines the exposure time for the observation
        median_image: fits file
            Read in a fits file of the median image
        trap_pop_s: int
            Number of initially occupied traps -- slow poplulation (default=200)   
        trap_pop_f: int
            Number of initially occupied traps -- fast poplulation (default=0)
        dTrap_s: int
            Number of extra trapped charge carriers added in the middle of two orbits -- slow population. 
            If it is a number, it assumes that all the extra added trap charge carriers are the same (default=0, 
            can be either number or list).
        dTrap_f: int
            Number of extra trapped charge carriers added in the middle of two orbits -- fast population. 
            If it is a number, it assumes that all the extra added trap charge carriers are the same (default=0, 
            can be either number or list).
        xList: list 
            A list on the range of Dispersion (default 0-13 pixels)
        
        Retruns
        -------
        Return the ramp light curve profile
        '''
        
        tExp=(x-x[0])*3600*24 #Convert start times of each exposure from days to seconds
        # cRates = np.ones(len(LC)) * LC.mean() * 1.002
        cRates = np.ones(len(tExp)) #Create array of ones based on the length of the data
        variability = cRates / cRates.mean()
        
        nTime = len(variability)
        
        im = median_image #Median image
        bbox = self.bbox #[0, 128, 59, 89]  #Define the bounding box of the area of interest
        xList = xList
        
        #dTrap_fList = [0]
        #dTrap_sList = [0]
        dtList = [0]
        full_well = 8e4
        
        ## Make a 2D cutout of pixels of interest
        template2D = im[bbox[2]:bbox[3], xList[0]:xList[-1]+1]
        nY = (bbox[3] - bbox[2])
        nX = (xList[-1] - xList[0]) + 1
        
        ## Cap all values at the full well value, which physically shouldn't be exceeded
        high_points = (template2D > full_well / exptime) ## find points
        template2D[high_points] = full_well / exptime ## set those points to full well
        
        template1D = template2D.ravel()
        
        ## Calculate RECTE corection for all pixels
        obs1D_time = self.RECTEMulti(template1D, variability, tExp, exptime,
                                dTrap_f=dTrap_f,
                                dTrap_s=dTrap_s,
                                trap_pop_f=trap_pop_f,
                                trap_pop_s=trap_pop_s,
                                dt0=dtList,
                                mode='staring',doSum=False)
        
        
        ## Reshape to NY, NX, NTime
        obs2D = np.reshape(obs1D_time,[nY,nX,nTime])
        ## Do aperture sum along Y direction
        obs2D_sum = np.sum(obs2D,axis=0)
        
        ## Divide by aperture sum and exptime to normalize
        template_sum = np.nansum(template2D,axis=0)
        ## Make the template_sum an array that is NY, NTime
        template_sum_time = np.tile(template_sum,[nTime,1]).transpose()
        
        ramps = obs2D_sum / exptime / template_sum_time
        
        return ramps
    
    def transit_model(self, x, rp, a, b):
        '''
        Models transit light curve using Python package `batman` based on initial parameters stored in params_transit.
        
        Parameters
        ----------
        x: array
            Time in Julian days    
        rp: int
            Planet-to-star radius ratio
        a: int
            Baseline linear regression y-intercept applied to the modeled normalized flux
        b: int
            Baseline linear regression slope applied to the modeled normalized flux
            
        Returns
        ---------
        flux: list
            A list of the modeled transit flux data
        '''
        planet_data = self.planet_data                            #Call the planet file data
        
        params_transit = batman.TransitParams()                   #Object to store transit parameters
        
        params_transit.t0 = planet_data['t0']                     #Time of inferior conjunction (days)
        params_transit.per = planet_data['per']                   #Orbital period (days)
        params_transit.a = planet_data['ax']                      #Semi-major axis (in units of stellar radii)
        params_transit.inc = planet_data['inc']                   #Orbital inclination (in degrees)
        params_transit.ecc = planet_data['ecc']                   #Eccentricity
        params_transit.w = planet_data['w']                       #Longitude of periastron (in degrees)
        params_transit.limb_dark = planet_data['limb_dark']       #Limb darkening model
        params_transit.u = planet_data['u']                       #Limb darkening coefficients [u1, u2, u3, u4]
        
        params_transit.rp = rp                                    #Planet-to-star radius ratio - Will depend on function input
        m = batman.TransitModel(params_transit, x)                #Initializes model
        
        #Modifying the linear regression slope: Julian Date(x) - Initial Julian Date(x0) 
        x0 = np.min(x)
        flux = m.light_curve(params_transit)*(a+b*(x-x0))         #Calculate the light curve modeled flux
        return flux
    
    def transit_model_RECTE(self, x, rp, a, b, trap_pop_s, dtrap_s, trap_pop_f, dtrap_f):
        '''
        Models transit light curve using Python package `batman` based on initial parameters stored in params_transit. 
        These transit models account for charge trapping systematics using Python package `RECTE`. 

        Parameters
        ----------
        x: array
            Time in Julian days
        rp: int
            Planet-to-star radius ratio
        a: int
            Baseline linear regression y-intercept applied to the modeled normalized flux
        b: int
            Baseline linear regression slope applied to the modeled normalized flux
        trap_pop_s: int
            (default=0) number of initially occupied traps -- slow poplulation
        trap_pop_f: int
            (default=0) number of initially occupied traps -- fast poplulation
        dtrap_s: int
            (default=0, can be either number or list) number of extra
            trapped charge carriers added in the middle of two orbits
            -- slow population. If it is a number, it assumes that all
            the extra added trap charge carriers are the same
        dtrap_f: int
            (default=0, can be either number or list) number of extra
             trapped charge carriers added in the middle of two orbits
            -- fast population. If it is a number, it assumes that all
            the extra added trap charge carriers are the same
        
        Returns
        ---------
        flux_modified: list
            A list of the RECTE modified modeled transit flux data. The modified flux accounts for the ramp effect in the real data. 
        ''' 
        planet_data = self.planet_data #Call the planet file data
        
        im = self.im  #Call the median image

        exptime = self.exptime #Call the exposure time
        
        xList = self.xList  #Call the dispersion range

        #Define the initial flux for the model based on the regular transit_model function
        flux = self.transit_model(x,rp,a,b)

        #Calculate the ramp profile in the initial flux data
        ramp=self.calculate_correction_fast(x,exptime,im,xList=xList,trap_pop_s=trap_pop_s, dTrap_s=[dtrap_s], trap_pop_f=trap_pop_f, dTrap_f=[dtrap_f])

        #Return the modified flux based on the ramp profile in the data
        flux_modified = flux*np.mean(ramp,axis=0)         #Calculate the light curve
        return flux_modified
    
    def eclipse_model(self, x, fp, a, b):
        '''
        Models eclipse light curve using Python package `batman` based on initial parameters stored in params_eclipse.

        Parameters
        ----------
        x: array
            Time in Julian days    
        fp: int
            Planet-to-star flux ratio
        a: int
            Baseline linear regression y-intercept applied to the modeled normalized flux
        b: int
            Baseline linear regression slope applied to the modeled normalized flux
            
        Returns
        ---------
        flux: list
            A list of the modeled secondary eclipse flux data
        '''
        planet_data = self.planet_data                            #Call the planet file data

        params_eclipse = batman.TransitParams()                   #Object to store secondary eclipse parameters

        params_eclipse.t0 = planet_data['t0']                     #Time of inferior conjunction (days)
        params_eclipse.per = planet_data['per']                   #Orbital period (days)
        params_eclipse.a = planet_data['ax']                      #Semi-major axis (in units of stellar radii)
        params_eclipse.inc = planet_data['inc']                   #Orbital inclination (in degrees)
        params_eclipse.ecc = planet_data['ecc']                   #Eccentricity
        params_eclipse.w = planet_data['w']                       #Longitude of periastron (in degrees)
        params_eclipse.limb_dark = planet_data['limb_dark']       #Limb darkening model
        params_eclipse.u = planet_data['u']                       #Limb darkening coefficients [u1, u2, u3, u4]
        params_eclipse.rp =  planet_data['rp']                    #Planet-to-star radius ratio
        params_eclipse.t_secondary = planet_data['t_secondary']   #The central eclipse time

        params_eclipse.fp = fp/1000000                            #Planet-to-star flux ratio (fp) is in ppm - Will depend on function input

        m = batman.TransitModel(params_eclipse, x, transittype="secondary") #Initializes model

        #Modifying the linear regression slope: Julian Date(x) - Initial Julian Date(x0) 
        x0 = np.min(x)
        flux = m.light_curve(params_eclipse)*(a+b*(x-x0))         #Calculate the light curve
        return flux
    
    def eclipse_model_RECTE(self, x, fp, a, b, trap_pop_s, dtrap_s, trap_pop_f, dtrap_f):
        '''
        Models eclipse light curve using Python package `batman` based on initial parameters stored in params_eclipse. These transit models         account for charge trapping systematics using Python package `RECTE`. 

        Parameters
        ----------
        x: array
            Time in Julian days 
        fp: int
            Planet-to-star flux ratio
        a: int
            Baseline linear regression y-intercept applied to the modeled normalized flux
        b: int
            Baseline linear regression slope applied to the modeled normalized flux
        trap_pop_s: int
            (default=0) number of initially occupied traps -- slow poplulation
        trap_pop_f: int
            (default=0) number of initially occupied traps -- fast poplulation
        dTrap_s: int
            (default=0, can be either number or list) number of extra
            trapped charge carriers added in the middle of two orbits
            -- slow population. If it is a number, it assumes that all
            the extra added trap charge carriers are the same
         dtrap_f: int
            (default=0, can be either number or list) number of extra
             trapped charge carriers added in the middle of two orbits
            -- fast population. If it is a number, it assumes that all
            the extra added trap charge carriers are the same 
            
        Returns
        ---------
        flux_modified: list
            A list of the RECTE modified modeled secondary eclipse flux data. 
            The modified flux accounts for the ramp effect in the real data.
        '''
        planet_data = self.planet_data #Call the planet file data
        
        im = self.im  #Call the median image

        exptime = self.exptime #Call the exposure time
        
        xList = self.xList #Call the dispersion range

        #Define the initial flux for the model based on the regular eclipse_model function
        flux = self.eclipse_model(x,fp,a,b)

        #Define the ramp profile in the initial flux data
        ramp=self.calculate_correction_fast(x,exptime,im,xList=xList,trap_pop_s=trap_pop_s, dTrap_s=[dtrap_s], trap_pop_f=trap_pop_f, dTrap_f=[dtrap_f])

        #return the modified flux based on the ramp profile in the data
        flux_modified = flux*np.mean(ramp,axis=0)  #Calculate the light curve
        return flux_modified
    
    def curve_fit_function(self, model, x, y, yerr, p0, bounds=(-np.inf, np.inf)):
        """
        Calling the the scipy.optimize.curve_fit function
        
        Parameters
        ----------
        model: function
            A function that models either transits or secondary eclipses. 
            Must be previously defined (transit_model/eclipse_model or transit_model_RECTE/eclipse_model_RECTE).
        x: array
            Time in Julian days 
        y: array
            Flux data
        yerr: list
            Flux data errors
        p0: list
            A list of initial guess values for the model parameters.
        bounds: tuple list
            Lower and upper bounds on parameters. Defaults to no bounds.

        Retruns
        ---------
        popt: array
            Optimal values for the parameters
        pcov: 2D array
            The estimated covariance of popt. The diagonals provide the variance
            of the parameter estimate. 
        """
        
        popt, pcov = curve_fit(model,x,y,sigma=yerr,p0=p0,bounds=bounds)
        return popt, pcov
    
    def optimize_batman_model(self, tshirt_obj, model, nbins=10, int_guess=None, showPlot=False, recalculate=False, useMultiprocessing=False):
        """
        Optimizes batman model light curves (for transits and/or secondary eclipses) based on initial parameters. 
        This function does NOT consider `RECTE` charge trapping parameters. 
        This function utilizies the scipy.optimize.curve_fit model fitting approach. 
        
        Parameters
        ----------
        tshirt_obj: keyword
            Time Series Helper & Integration Reduction Tool (tshirt) Spectrometry Object
        model: function
            A function that models either transits or secondary eclipses. Must be previously defined (transit_model or eclipse_model).
        nbins: int
            The number of wavelength bins. The Default is "nbins=10".
        int_guess: list
            A list of initial guess values for the model parameters. Default == None and will use preset numbers. 
        showPlot: bool
            Make the plot visible? The Default is "False"
        recalculate: bool
            Recalculate the model optimizations? The Default is "False"
        useMultiprocessing: bool
            Use multiprocessing for faster computations?
        
        Retruns
        ---------
        popt_list: list
            List of arrays containing the optimal values for the parameters for each wavelength channel.
        pcov: list
            List of arrays containing the variance of the parameter estimates for each wavelength channel.
        """
        planet_data = self.planet_data #Call the planet file data
            
        #Obtain a table of the the wavelength-binned time series (with `tshirt`). 
        #Seperate out the raw flux data (raw_results) and the raw flux error data (raw_results_errors) into two different pandas tables.
        results = tshirt_obj.get_wavebin_series(nbins=nbins)
        raw_results = results[0].to_pandas()
        raw_results_errors = results[1].to_pandas()
        
        #Define the axis columns as well as the corresponding error columns.
        ydata_columns = raw_results.columns[1:].values #Skip over the time column
        ydata_errors_columns = raw_results_errors.columns[1:].values #Skip over the time column
        xdata = self.x #Time column data in terms of days accounting for Solar barycenter correction
    
        #Obtain a table of wavelength bins, with theoretical noise and measured standard deviation across time.
        table_noise = tshirt_obj.print_noise_wavebin(nbins=nbins)
        table_noise=table_noise.to_pandas() #convert to a pandas table
        
        #Wavelength calibration to turn the dispersion pixels into wavelengths. 
        wavelength_list = tshirt_obj.wavecal(table_noise['Disp Mid'],waveCalMethod = planet_data['WaveCalMethod'])
        
        #Define empty lists to store scipy.optimize.curve_fit results.
        popt_list=[] #Optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized.
        pcov_list=[] #List of one standard deviation errors on the parameters.
        
        #Plotting options
        if(showPlot==True):
            
            fig, (ax, ax2) = plt.subplots(1,2,figsize=(20,10),sharey=False) #Set up the figure space
            fig.subplots_adjust(wspace=0.1)

        
        #Establish the model in use based on initial function input. Establish the parameters and the initial parameter guess values (p0) for each model. 
        if(model==self.transit_model):
            text = 'fit: rp=%5.3f, a=%5.3f, b=%5.3f'
            if int_guess == None:
                print('\033[91m'+"Hint: The closer your guess, the better the model fit"+'\033[91m')
                rp_guess = input("Pick a guess value for the planet-to-star radius ratio (rp):")
                a_guess = input("Pick a guess value for the Baseline linear regression y-intercept applied to the normalized modeled flux (a):") #Prompt User input for the guess values
                b_guess = input("Pick a guess value for the Baseline linear regression slope applied to the normalized modeled flux (b):")
                print('\033[91m'+"Save these values and use as input for int_guess next time you run this function:"+'\033[91m'+str([rp_guess, a_guess, b_guess]))
                p0 = [float(rp_guess),float(a_guess),float(b_guess)] #Each guess value in the list corresponds to the parameter order in text
            else: 
                p0 = int_guess
        
        elif(model==self.eclipse_model):
            text = 'fit: fp=%5.3f, a=%5.3f, b=%5.3f'
            if int_guess == None:
                print('\033[91m'+"Hint: The closer your guess, the better the model fit"+'\033[91m')
                fp_guess = input("Pick a guess value for the planet-to-star flux ratio (fp in ppm):")
                a_guess = input("Pick a guess value for the Baseline linear regression y-intercept applied to the normalized modeled flux (a):") #Prompt User input for the guess values
                b_guess = input("Pick a guess value for the  Baseline linear regression slope applied to the normalized modeled flux (b):")
                print('\033[91m'+"Save these values and use as input for int_guess next time you run this function:"+'\033[91m'+str([fp_guess, a_guess, b_guess]))
                p0 = [float(fp_guess),float(a_guess),float(b_guess)] #each guess value in the list corresponds to the parameter order in text
            else:
                p0 = int_guess
                
        else: 
            print("Invalid Model Input") #This function only works on the above previously defined models!
        
        #Establsih a color map index, to be iterated over, based on the number of wavebins defined. 
        color_idx_range = np.linspace(0.3, 0.8, nbins)
        
        #Trim the data used in the model. Exclude the first orbit in each visit since RECTE is not optimized here and the ramp profile is still prevalent.
        tshirt_obj.plot_wavebin_series(interactive=True) #Use the interactive plot to find where the first visit ends
        
        data_trim = int(input('\033[91m'+"This model excludes the first orbit in each visit since the ramp profile is not considered. Use the interactive plot to determine at what index to trim the data:"+'\033[91m'))

        if(useMultiprocessing==True):
            start = time.time()
            curve_fit_function_data =[] #Empty list to store the curve_fit_function data
            for columns,columns_errors in zip(ydata_columns,ydata_errors_columns):
            #Trim the data used in the model. Exclude the first orbit in each visit since RECTE is not optimized here and the ramp profile is still prevalent.
                xdata_trimmed = xdata[data_trim:]  
                ydata_trimmed = raw_results[columns][data_trim:]
                ydata_error_trimmed = raw_results_errors[columns_errors][data_trim:].tolist() #Convert error data to a list in order to use in scipy.optimize.curve_fit 
            
                curve_fit_function_params = (model,xdata_trimmed,ydata_trimmed,ydata_error_trimmed,p0) #Gather all parameters required for curve_fit_function to run
                curve_fit_function_data.append(curve_fit_function_params)
            
            maxCPUs = 10 #How many CPU's? 
            p = Pool(maxCPUs)
            
            curve_fit_function_results= list(tqdm.tqdm(p.starmap(self.curve_fit_function,curve_fit_function_data),total=len(list(curve_fit_function_data))))

            end = time.time()
            print("optimize_batman_model: Multiprocessing Took "+str(end-start)+" seconds")
        else:
            None

        #Loop over the flux data and their respective flux data error columns simultaneously for each wavelength. will loop over a color index and bin number for plotting purposes.  
        for columns,columns_errors,bin_number,color_idx,wavelength in zip(ydata_columns,ydata_errors_columns,np.arange(nbins),color_idx_range,wavelength_list):
            
            #Trim the data used in the model. Exclude the first orbit in each visit since RECTE is not optimized here and the ramp profile is still prevalent.
            xdata_trimmed = xdata[data_trim:]  
            ydata_trimmed = raw_results[columns][data_trim:]
            ydata_error_trimmed = raw_results_errors[columns_errors][data_trim:].tolist() #Convert error data to a list in order to use in scipy.optimize.curve_fit 
            
            new_dir = pathlib.Path(planet_data['BaseDir'], 'optimize_batman_model')  #Establish a path to the saved data based on the given base directory
            OptimalParams_file = os.path.join(new_dir,"{}_OptimalParams_{}_wavelength_ind_{}.csv".format(planet_data['srcName'],tshirt_obj.param['nightName'],columns)) #Generate a name for the light curve file based on YAML files
    
            if (os.path.exists(OptimalParams_file) == True) and (recalculate == False):
                dat_OptimalParams = ascii.read(OptimalParams_file)
                popt = np.array(dat_OptimalParams['Optimal Values']) #read in the optimized parameter values
                pcov_diag = np.array(dat_OptimalParams['Estimated Covariance']) #read in the one standard deviation errors on the parameters.
                #Append these returned arrays into the previously defined empty lists.
                popt_list.append(popt)
                pcov_list.append(pcov_diag)
                
            #If useMultiprocessing==True, call the results for each channel and save them to seperate files to match the non useMultiprocessing method of saving and plotting
            elif(useMultiprocessing == True): 
                popt_multi = [i[0] for i in curve_fit_function_results] #optimal values for the parameters (popt)
                pcov_multi = [i[1] for i in curve_fit_function_results] 
                pcov_diag_multi = [np.sqrt(np.diag(i)) for i in pcov_multi] #To compute one standard deviation errors on the parameters
                popt_list.append(popt_multi)
                pcov_list.append(pcov_diag_multi)
                
                for i,j in zip(popt_multi,pcov_diag_multi):
                    dat_OptimalParams = Table()
                    dat_OptimalParams['Optimal Values'] = i
                    dat_OptimalParams['Estimated Covariance'] = j
                    dat_OptimalParams.write(OptimalParams_file,overwrite=True)  #Write the light curve data to a file
            else: 
                #Call and run scipy.optimize.curve_fit. 
                #Returns an array of optimal values for the parameters (popt) and an array for the the estimated covariance of popt (pcov).
                popt, pcov = curve_fit(model,xdata_trimmed,ydata_trimmed,sigma=ydata_error_trimmed,p0=p0)    
                pcov_diag = np.sqrt(np.diag(pcov)) #To compute one standard deviation errors on the parameters
                       
                new_dir.mkdir(parents=True, exist_ok=True)  #Make the new directory
                dat_OptimalParams = Table()
                dat_OptimalParams['Optimal Values'] = popt
                dat_OptimalParams['Estimated Covariance'] = pcov_diag
                dat_OptimalParams.write(OptimalParams_file,overwrite=True)  #Write the light curve data to a file
            
                #Append these returned arrays into the previously defined empty lists.
                popt_list.append(popt)
                pcov_list.append(pcov_diag)
            
            #Light Curve Plotting Options
            if(showPlot==True):
                
                #This line is used to save light curve modeling results to a specific folder in order to streamline previously run data. (Can be altered)
                lightcurve_file = os.path.join(new_dir,"{}_LightCurve_{}_wavelength_ind_{}.csv".format(planet_data['srcName'],tshirt_obj.param['nightName'],columns)) #Generate a name for the light curve file based on YAML files
    
                #If the previously defined lightcurve_file exists and the recalculation parameter is set to False, read it in. 
                if (os.path.exists(lightcurve_file) == True ) and (recalculate == False):
                    dat_lightcurve = ascii.read(lightcurve_file)
                    Time = np.array(dat_lightcurve['Time'])
                    
                    #Saftey check if the saved time and user input trimmed data is equal in length. 
                    if(len(Time)!=len(xdata_trimmed)):
                        raise Exception("The length of the trimmed data (based on index input) does not match the saved results. Re-input the index or set recalculate=True.")
                    #Saftey check If the Time read does not match the xdata (time) within a tolerance raise an error. 
                    elif np.allclose(Time,xdata_trimmed,rtol=1e-15) == False:
                        raise Exception("Times don't match")
                    
                    ymodel = np.array(dat_lightcurve['ymodel'])
                    
                elif(useMultiprocessing==True):
                    for popt in popt_list[0]:
                        
                        ymodel = model(xdata_trimmed, *popt) #define the model based on optimized parameters (popt)
                        #Create a table for these results and save them to the lightcurve_file defined previously.
                        dat_lightcurve = Table()
                        dat_lightcurve['Time'] = xdata_trimmed
                        dat_lightcurve['ymodel'] = ymodel
                        dat_lightcurve.write(lightcurve_file,overwrite=True)
                    
                #If the previously defined lightcurve_file does not exsit or if the recalculation parameter is set to True
                else: 
                    ymodel = model(xdata_trimmed, *popt) #define the model based on optimized parameters (*popt)
                    #Create a table for these results and save them to the lightcurve_file defined previously. 
                    dat_lightcurve = Table()
                    dat_lightcurve['Time'] = xdata_trimmed
                    dat_lightcurve['ymodel'] = ymodel
                    dat_lightcurve.write(lightcurve_file,overwrite=True)
                
                offset = 0.007 #set an offset between each wavelegnth's light curve.
                
                #The first plot (ax) is the original light curves with the models overlaid.
                ax.plot(xdata_trimmed, ymodel-bin_number*offset, 'r-',
                        label=text % tuple(popt))  #The *popt will grab the optimized parameters required for the model. 
                ax.plot(xdata, raw_results[columns]-bin_number*offset,'o',color=plt.cm.gist_heat(color_idx),alpha=0.8) #Plot the time data vs. initial flux data - no model
                
                #plot labels
                ax.set_xlabel('Time (BJD)')
                ax.set_ylabel('Normalized Flux')
                ax.set_title(str(planet_data['srcName'])+' Light Curves')
                
                #Plot the residuals
                ax2.plot(xdata_trimmed, (ydata_trimmed-ymodel)-bin_number*offset,'o',color=plt.cm.gist_heat(color_idx),label="{:.2f}$\mu m$".format(wavelength))
                ax2.set_ylabel('Residuals')
                ax2.set_xlabel('Time (BJD)')
                ax2.set_title(str(planet_data['srcName'])+' Model Residuals')
                            
                box = ax.get_position()
                box2= ax2.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.99, box.height])
                ax2.set_position([box2.x0, box2.y0, box2.width * 0.99, box2.height])
                ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                #Annotation Settings
                #if(model==self.transit_model):
                #    ax.annotate("{:.2f}$\mu m$".format(wavelength), xy =(np.mean(xdata)-0.085, np.mean(raw_results[columns])-bin_number*offset+0.0085),fontsize=15,weight='bold',color=plt.cm.gist_heat(color_idx)) #annotation for wavelength values
                 #   ax2.annotate("{:.2f}$\mu m$".format(wavelength), xy =(np.mean(xdata_trimmed)-0.045, np.mean(ydata_trimmed-ymodel)-bin_number*offset+0.0015),fontsize=15,weight='bold',color=plt.cm.gist_heat(color_idx)) #Annotate the wavelengths on the plot

                #elif(model==self.eclipse_model):
                #    ax.annotate("{:.2f}$\mu m$".format(wavelength), xy =(np.mean(xdata)-0.085, np.mean(raw_results[columns])-bin_number*offset+0.002),fontsize=15,weight='bold',color=plt.cm.gist_heat(color_idx)) #annotation for wavelength values
                #    ax2.annotate("{:.2f}$\mu m$".format(wavelength), xy =(np.mean(xdata_trimmed)-0.05, np.mean(ydata_trimmed-ymodel)-bin_number*offset+0.0035),fontsize=15,weight='bold',color=plt.cm.gist_heat(color_idx)) #Annotate the wavelengths on the plot

                #else: 
                  #  None
                    
            figure_file = os.path.join(new_dir,"{}_LightCurve_{}.pdf".format(planet_data['srcName'],tshirt_obj.param['nightName'])) #Generate a name for the light curve file based on YAML files
            plt.savefig(figure_file)  #Save the light curve image as a pdf
        print("Figure and Data found in: " +str(new_dir))
        return #popt_list,pcov_list
    
    def optimize_batman_model_RECTE(self, tshirt_obj, model, nbins=10, int_guess=None, showPlot=False, recalculate=False, useMultiprocessing=False):
        """
        Optimizes batman model light curves (for transits and/or secondary eclipses) based on initial parameters. 
        This function does consider RECTE charge trapping parameters. 
        This function utilizies the scipy.optimize.curve_fit model fitting approach. 
        
        Parameters
        ----------
        tshirt_obj: keyword
            Time Series Helper & Integration Reduction Tool (tshirt) Spectrometry Object
        model: function
            A function that models either transits or secondary eclipses that encorporate RECTE charge trapping parameters . 
            Must be previously defined.
        nbins: int
            The number of wavelength bins. The Default is "nbins=10".
        int_guess: list
            A list of initial guess values for the model parameters. Default == None and will use preset numbers. 
        showPlot: bool
            Make the plot visible? The Default is "False"
        recalculate: bool
            Recalculate the model optimizations? The Default is "False"
        useMultiprocessing: bool
            Use multiprocessing for faster computations?
            
        Retruns
        ---------
        popt_list: list
            List of arrays containing the optimal values for the parameters for each wavelength channel.
        pcov: list
            List of arrays containing the variance of the parameter estimates for each wavelength channel.
        """
        
        planet_data = self.planet_data #Call the planet file data
        
        im = self.im  #Call the median image

        exptime = self.exptime #Call the exposure time
            
        #Obtain a table of the the wavelength-binned time series (with `tshirt`). 
        #Seperate out the raw flux data (raw_results) and the raw flux error data (raw_results_errors) into two different pandas tables.
        results = tshirt_obj.get_wavebin_series(nbins=nbins)
        raw_results = results[0].to_pandas()
        raw_results_errors = results[1].to_pandas()
        
        #Define the axis columns as well as the corresponding error columns.
        ydata_columns = raw_results.columns[1:].values #Skip over the time column
        ydata_errors_columns = raw_results_errors.columns[1:].values #Skip over the time column
        xdata = self.x #Time column data in terms of days accounting for Solar barycenter correction
    
        #Obtain a table of wavelength bins, with theoretical noise and measured standard deviation across time.
        table_noise = tshirt_obj.print_noise_wavebin(nbins=nbins)
        table_noise=table_noise.to_pandas() #convert to a pandas table
        
        #Wavelength calibration to turn the dispersion pixels into wavelengths. 
        wavelength_list = tshirt_obj.wavecal(table_noise['Disp Mid'],waveCalMethod = planet_data['WaveCalMethod'])
        
        #Defining the global xList parameter needed in the model functions with RECTE
        xList_all = [] #empty list to store each wavlengths dispersion indices.
        #Loop over each wavelength's bin index. The dispersion indices will be different for each wavelength bin. 
        for ind in table_noise.index: 
            Disp_st = table_noise['Disp St'][ind] #Start of the dispersion range
            Disp_end = table_noise['Disp End'][ind] #End of the dispersion range
            Disp_xList = np.arange(Disp_st, Disp_end,1) #Return Numpy array of evenly spaced values within a given interval. 
            xList_all.append(Disp_xList) #Append Numpy array to the empty list to late be iterated over.
            
        #Define empty lists to store scipy.optimize.curve_fit results.
        popt_list=[] #Optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized.
        pcov_list=[] #List of one standard deviation errors on the parameters. 
        
        #Plotting options
        if(showPlot==True):
            
            fig, (ax, ax2, ax3) = plt.subplots(1,3,figsize=(20,10),sharey=False) #Set up the figure space        
            
        #Establish the model in use based on initial function input. Establish the parameters and the initial parameter guess values (p0) for each model. 
        if(model==self.transit_model_RECTE):
            text = 'fit: rp=%5.3f, a=%5.3f,b=%5.3f,trap_pop_s=%5.3f,dtrap_s=%5.3f,trap_pop_f=%5.3f,dtrap_f=%5.3f'
            if int_guess == None:
                print('\033[91m'+"Hint: The closer your guess, the better the model fit"+'\033[91m')
                rp_guess = input("Pick a guess value for the planet-to-star radius ratio (rp) [limits:(-\u221e,\u221e)]:")
                a_guess = input("Pick a guess value for the Baseline linear regression y-intercept applied to the normalized modeled flux (a) [limits:(0,\u221e)]:") #Prompt User input for the guess values
                b_guess = input("Pick a guess value for the Baseline linear regression slope applied to the normalized modeled flux (b) [limits:(-\u221e,\u221e)]:")
                trap_pop_s = input("Pick a guess value for the number of initially occupied traps -- slow poplulation (trap_pop_s) [limits:(0,500)]:")
                dtrap_s = input("Pick a guess value for the number of extra trapped charge carriers added in the middle of two orbits -- slow population. (dtrap_s) [limits:(0,500)]:")
                trap_pop_f = input("Pick a guess value for the number of initially occupied traps -- fast poplulation (trap_pop_f) [limits:(0,200)]:")
                dtrap_f = input("Pick a guess value for the number of extra trapped charge carriers added in the middle of two orbits -- fast population. (dtrap_f) [limits:(0,200)]:")
                print('\033[91m'+"Save these values and use as input for int_guess next time you run this function:"+'\033[91m'+str([rp_guess, a_guess, b_guess,trap_pop_s,dtrap_s,trap_pop_f,dtrap_f]))
                p0 = [float(rp_guess),float(a_guess),float(b_guess),int(trap_pop_s),int(dtrap_s),int(trap_pop_f),int(dtrap_f)] #each guess value in the list corresponds to the parameter order in text
            else:
                p0 = int_guess
                
        elif(model==self.eclipse_model_RECTE):
            text = 'fit: fp=%5.3f, a=%5.3f,b=%5.3f,trap_pop_s=%5.3f,dtrap_s=%5.3f,trap_pop_f=%5.3f,dtrap_f=%5.3f'
            if int_guess == None:
                print('\033[91m'+"Hint: The closer your guess, the better the model fit"+'\033[91m')
                fp_guess = input("Pick a guess value for the planet-to-star flux ratio (fp):")
                a_guess = input("Pick a guess value for the Baseline linear regression y-intercept applied to the normalized modeled flux (a):") #Prompt User input for the guess values
                b_guess = input("Pick a guess value for the Baseline linear regression slope applied to the normalized modeled flux (b):")
                trap_pop_s = input("Pick a guess value for the number of initially occupied traps -- slow poplulation (trap_pop_s):")
                dtrap_s = input("Pick a guess value for the number of extra trapped charge carriers added in the middle of two orbits -- slow population. (dtrap_s):")
                trap_pop_f = input("Pick a guess value for the number of initially occupied traps -- fast poplulation (trap_pop_f):")
                dtrap_f = input("Pick a guess value for the number of extra trapped charge carriers added in the middle of two orbits -- fast population. (dtrap_f):")
                print('\033[91m'+"Save these values and use as input for int_guess next time you run this function:"+'\033[91m'+str([fp_guess, a_guess, b_guess,trap_pop_s,dtrap_s,trap_pop_f,dtrap_f]))
                p0 = [float(fp_guess),float(a_guess),float(b_guess),int(trap_pop_s),int(dtrap_s),int(trap_pop_f),int(dtrap_f)] #each guess value in the list corresponds to the parameter order in text
            else:
                p0=int_guess
        else: 
            print("Invalid Model Input") #This function only works on the above previously defined models!
        
        #Establsih a color map index, to be iterated over, based on the number of wavebins defined. 
        color_idx_range = np.linspace(0.3, 0.8, nbins)
        
        if(useMultiprocessing==True):
            start = time.time()
            curve_fit_function_data =[] #Empty list to store the curve_fit_function data
            for columns,columns_errors in zip(ydata_columns,ydata_errors_columns):
                ydata = raw_results[columns] #Return as a Numpy representation of the data.
                yerr = raw_results_errors[columns_errors]
                ydata_error_list = yerr.tolist() #Convert error data to a list in order to use in scipy.optimize.curve_fit
                bounds =([-np.inf,0,-np.inf,0,0,0,0],[np.inf,np.inf,np.inf,500,500,200,200]) #set upper and lower limits for all parameters. Most importantly set the RECTE parameters limits to be within reason. 

                curve_fit_function_params = (model,xdata,ydata,ydata_error_list,p0,bounds) #Gather all parameters required for curve_fit_function to run
                curve_fit_function_data.append(curve_fit_function_params)
                
            maxCPUs = 10 #How many CPU's? 
            p = Pool(maxCPUs)
            
            curve_fit_function_results= list(tqdm.tqdm(p.starmap(self.curve_fit_function,curve_fit_function_data),total=len(list(curve_fit_function_data))))
            end = time.time()
            print("optimize_batman_model_RECTE: Multiprocessing Took "+str(end-start)+" seconds")
        else:
            None
    
        #Loop over the flux data and their respective flux data error columns simultaneously for each wavelength. 
        #Each wavelength will have an associated color (determined by the color index), bin number, and dispersion range.  
        for columns,columns_errors,disp_range,bin_number,color_idx,wavelength in zip(ydata_columns,ydata_errors_columns,xList_all,np.arange(nbins),color_idx_range,wavelength_list):
            
            xList = disp_range #The xList will correspond to the dispersion range. 
    
            ydata = raw_results[columns] #Return as a Numpy representation of the data.
            yerr = raw_results_errors[columns_errors]
            ydata_error_list = yerr.tolist() #Convert error data to a list in order to use in scipy.optimize.curve_fit 
            
            new_dir = pathlib.Path(planet_data['BaseDir'], 'optimize_batman_model_RECTE')  #Establish a path to the saved data based on the given base directory
            OptimalParams_file = os.path.join(new_dir,"{}_OptimalParams_{}_wavelength_ind_{}.csv".format(planet_data['srcName'],tshirt_obj.param['nightName'],columns)) #Generate a name for the light curve file based on YAML files
            
            if (os.path.exists(OptimalParams_file) == True) and (recalculate == False):
                dat = ascii.read(OptimalParams_file)
                popt = np.array(dat['Optimal Values']) #read in the optimized parameter values
                pcov_diag = np.array(dat['Estimated Covariance']) #read in the one standard deviation errors on the parameters.
                
                #Append these returned arrays into the previously defined empty lists.
                popt_list.append(popt)
                pcov_list.append(pcov_diag)
                
            #If useMultiprocessing==True, call the results for each channel and save them to seperate files to match the non useMultiprocessing method of saving and plotting
            elif(useMultiprocessing == True):
                popt_multi = [i[0] for i in curve_fit_function_results] #optimal values for the parameters (popt)
                pcov_multi = [i[1] for i in curve_fit_function_results] 
                pcov_diag_multi = [np.sqrt(np.diag(i)) for i in pcov_multi] #To compute one standard deviation errors on the parameters
                popt_list.append(popt_multi)
                pcov_list.append(pcov_diag_multi)
                
                for i,j in zip(popt_multi,pcov_diag_multi):
                    dat_OptimalParams = Table()
                    dat_OptimalParams['Optimal Values'] = i
                    dat_OptimalParams['Estimated Covariance'] = j
                    dat_OptimalParams.write(OptimalParams_file,overwrite=True)  #Write the light curve data to a file
         
            #If the previously defined results_file does not exsit or if the recalculation parameter is set to True, call and run scipy.optimize.curve_fit 
            else: 
                #Call and run scipy.optimize.curve_fit.
                bounds =([-np.inf,0,-np.inf,0,0,0,0],[np.inf,np.inf,np.inf,500,500,200,200]) #set upper and lower limits for all parameters. Most importantly set the RECTE parameters limits to be within reason. 
                #Returns an array of optimal values for the parameters (popt) and an array for the the estimated covariance of popt (pcov).
                popt, pcov = curve_fit(model,xdata,ydata,sigma=ydata_error_list,p0=p0,bounds=bounds)    
                pcov_diag = np.sqrt(np.diag(pcov)) #To compute one standard deviation errors on the parameters
                
                new_dir.mkdir(parents=True, exist_ok=True)  #Make the new directory
                dat = Table()
                dat['Optimal Values'] = popt
                dat['Estimated Covariance'] = pcov_diag
                dat.write(OptimalParams_file,overwrite=True)  #Write the light curve data to a file
    
                #Append these returned arrays into the previously defined empty lists.
                popt_list.append(popt)
                pcov_list.append(pcov_diag)
            
            #Plotting Options
            if(showPlot==True):
                #This line is used to save light curve modeling results to a specific folder in order to streamline previously run data. (Can be altered)
                lightcurve_file = os.path.join(new_dir,"{}_LightCurve_{}_wavelength_ind_{}.csv".format(planet_data['srcName'],tshirt_obj.param['nightName'],columns)) #Generate a name for the light curve file based on YAML files
    
                #If the previously defined lightcurve_file exists and the recalculation parameter is set to False, read it in. 
                if (os.path.exists(lightcurve_file) == True) and (recalculate == False):
                    dat = ascii.read(lightcurve_file)
                    Time = np.array(dat['Time'])
                    
                    #Saftey check If the Time read does not match the xdata (time) within a tolerance raise an error. 
                    if np.allclose(Time,xdata,rtol=1e-15) == False:
                        raise Exception("Times don't match")
                        
                    ymodel = np.array(dat['ymodel'])
                    ramp_model = np.array(dat['ramp_model'])
                
                elif(useMultiprocessing==True):
                    for popt in popt_list[0]:
                        ymodel = model(xdata, *popt) #define the model based on optimized parameters (popt)
                        ramp=self.calculate_correction_fast(xdata,exptime,im,xList=xList,trap_pop_s=popt[3], dTrap_s=[popt[4]], trap_pop_f=popt[5], dTrap_f=[popt[6]]) #calculate the ramp in the data based on optimized parameters (popt)
                        ramp_model = np.mean(ramp,axis=0) #Defing the ramp model; the mean along the rows(axis=0)
                        #Create a table for these results and save them to the lightcurve_file defined previously.
                        dat_lightcurve = Table()
                        dat_lightcurve['Time'] = xdata
                        dat_lightcurve['ymodel'] = ymodel
                        dat_lightcurve['ramp_model'] = ramp_model
                        dat_lightcurve.write(lightcurve_file,overwrite=True)
                
                #If the previously defined lightcurve_file does not exsit or if the recalculation parameter is set to True
                else: 
                    ymodel = model(xdata, *popt) #define the model based on optimized parameters (*popt)
                    ramp=self.calculate_correction_fast(xdata,exptime,im,xList=xList,trap_pop_s=popt[3], dTrap_s=[popt[4]], trap_pop_f=popt[5], dTrap_f=[popt[6]]) #calculate the ramp in the data based on optimized parameters (popt)
                    ramp_model = np.mean(ramp,axis=0) #Defing the ramp model; the mean along the rows(axis=0)
                    #Create a table for these results and save them to the lightcurve_file defined previously. 
                    dat = Table()
                    dat['Time'] = xdata
                    dat['ymodel'] = ymodel
                    dat['ramp_model'] = ramp_model
                    dat.write(lightcurve_file,overwrite=True)                
        
                #Plots
                offset = 0.007 #set an offset between each wavelegnth's light curve.
                
                #The first plot (ax) is the original light curves with the models accounting for RECTE charge trapping parameters overlaid. 
                ax.plot(xdata, ymodel-bin_number*offset, 'r-',
                        label=text % tuple(popt)) #models accounting for RECTE parameters
                ax.plot(xdata, ydata-bin_number*offset,'o',color=plt.cm.gist_heat(color_idx),alpha=0.8) #original data
                ax.set_xlabel('Time (BJD)')
                ax.set_ylabel('Normalized Flux')
                ax.set_title(str(planet_data['srcName'])+' Light Curves')                
                
                #The second plot (ax2) is the original light curves with the ramp_model systematics divided out along with the models accounting for RECTE charge trapping parameters with the ramp_model divided out divided and overlaid. 
                ax2.plot(xdata, (ymodel/ramp_model)-bin_number*offset, 'r-',
                        label=text % tuple(popt)) #models with the ramp_model divided out accouting for the RECTE parameters 
                ax2.plot(xdata,(ydata/ramp_model)-bin_number*offset,'o',color=plt.cm.gist_heat(color_idx),alpha=0.8) #original data with the ramp model divided out. 
                ax2.set_xlabel('Time (BJD)')
                ax2.set_title(str(planet_data['srcName'])+' Light Curves - Ramp Systematic Removed')            
    
                #The third plot (ax3) is the residuals of the original data from the model
                ax3.plot(xdata, (ydata-ymodel)-bin_number*offset,'o',color=plt.cm.gist_heat(color_idx), label = "{:.2f}$\mu m$".format(wavelength))
                ax3.set_ylabel('Residuals')
                ax3.set_xlabel('Time (BJD)')
                ax3.set_title(str(planet_data['srcName'])+' Systematics')
                
                box = ax.get_position()
                box2= ax2.get_position()
                box3= ax3.get_position()

                ax.set_position([box.x0, box.y0, box.width * 0.99, box.height])
                ax2.set_position([box2.x0, box2.y0, box2.width * 0.99, box2.height])
                ax3.set_position([box3.x0, box3.y0, box3.width * 0.99, box3.height])

                ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                
                #Annotation Settings
                #if(model==self.transit_model_RECTE):
                #    ax.annotate("{:.2f}$\mu m$".format(wavelength), xy =(np.mean(xdata)-0.085, np.mean(ydata)-bin_number*offset+0.0085),fontsize=10,weight='bold',color=plt.cm.gist_heat(color_idx)) #annotation for wavelength values
                #    ax2.annotate("{:.2f}$\mu m$".format(wavelength), xy =(np.mean(xdata)-0.085, np.mean(ydata/ramp_model)-bin_number*offset+0.0085),fontsize=10,weight='bold',color=plt.cm.gist_heat(color_idx)) #Annotate the wavelengths on the plot
                #    ax3.annotate("{:.2f}$\mu m$".format(wavelength), xy =(np.mean(xdata)-0.080, np.mean(ydata-ymodel)-bin_number*offset+0.0015),fontsize=10,weight='bold',color=plt.cm.gist_heat(color_idx)) #Annotate the wavelengths on the plot

                #elif(model==self.eclipse_model_RECTE):
                #    ax.annotate("{:.2f}$\mu m$".format(wavelength), xy =(np.mean(xdata)-0.085, np.mean(raw_results[columns])-bin_number*offset+0.002),fontsize=10,weight='bold',color=plt.cm.gist_heat(color_idx)) #annotation for wavelength values
                 #   ax2.annotate("{:.2f}$\mu m$".format(wavelength), xy =(np.mean(xdata)-0.09, np.mean(ydata/ramp_model)-bin_number*offset+0.0035),fontsize=10,weight='bold',color=plt.cm.gist_heat(color_idx)) #Annotate the wavelengths on the plot
                #    ax3.annotate("{:.2f}$\mu m$".format(wavelength), xy =(np.mean(xdata)-0.085, np.mean(ydata-ymodel)-bin_number*offset+0.0015),fontsize=10,weight='bold',color=plt.cm.gist_heat(color_idx)) #Annotate the wavelengths on the plot
               # else: 
                 #   None
                    
            figure_file = os.path.join(new_dir,"{}_LightCurve_{}.pdf".format(planet_data['srcName'],tshirt_obj.param['nightName'])) #Generate a name for the light curve file based on YAML files
            plt.savefig(figure_file)  #Save the light curve image as a pdf
        print("Figure and Data found in: " +str(new_dir))
        return #popt_list,pcov_list
    
    def log_likelihood(self, theta, model, x, y, yerr):
        """
        A supplemental function needed to run the function `MCMC`
        
        Defines the log liklihood function; a natural logarithm of the liklihood. 
        It Measures the goodness of fit of a statistical model to a sample of data for given values of the unknown parameters.
        Procedure for obtaining maximum likelihood estimation (the parameter values for the model such that they maximize the 
        likelihood of this model actually being observed) is done in the function `MCMC`. 
        
        Parameters
        ----------
        theta: vector
            A vector that contains the seven free parameters used by our model (transit_RECTE/eclipse_RECTE) function.
        model: function
            A function that models either transits or secondary eclipses. Must be previously defined 
            (transit_model_RECTE or eclipse_model_RECTE).
        x: array
            Time in Julian days 
        y: array
            Flux data
        yerr: array
            Error in the flux data 
            
        Returns
        ---------
        The estimated likelihood
        """
        if(model==self.transit_model_RECTE):
            #The unknown parameters, the blueprint for the model. 
            rp,a,b,trap_pop_s,dtrap_s,trap_pop_f,dtrap_f = theta
            model_data = self.transit_model_RECTE(x, rp, a, b, trap_pop_s, dtrap_s, trap_pop_f, dtrap_f) #the model 
            return -0.5 * np.sum((y - model_data) ** 2 / yerr ** 2)
        elif(model==self.eclipse_model_RECTE):
            #The unknown parameters, the blueprint for the model. 
            fp,a,b,trap_pop_s,dtrap_s,trap_pop_f,dtrap_f  = theta 
            model_data = self.eclipse_model_RECTE(x, fp, a, b, trap_pop_s, dtrap_s, trap_pop_f, dtrap_f) #the model 
            return -0.5 * np.sum((y - model_data) ** 2 / yerr ** 2)
        else:
            None
            
    def log_prior(self, theta, model):
        """
        A supplemental function needed to run the function `MCMC`
        
        Applies the prior knowledge of the parameters for the modeling functions by setting restriction bounds. 
        
        Parameters
        ----------
        theta: vector
            A vector that contains the seven free parameters used by our model (transit_RECTE/eclipse_RECTE) function.
        model: function
            A function that models either transits or secondary eclipses. Must be previously defined 
            (transit_model_RECTE or eclipse_model_RECTE).
        """
        if(model==self.transit_model_RECTE):
            #The unknown parameters, the blueprint for the model. 
            rp,a,b,trap_pop_s,dtrap_s,trap_pop_f,dtrap_f = theta
            #set bounds for the RECTE charge trapping parameters
            if 0.0 < trap_pop_s < 500 and 0.0 < dtrap_s < 500 and 0.0 < trap_pop_f < 200 and 0.0 < dtrap_f < 200 :
                return 0.0
            return -np.inf
        
        elif(model==self.eclipse_model_RECTE):
            #The unknown parameters, the blueprint for the model. 
            fp,a,b,trap_pop_s,dtrap_s,trap_pop_f,dtrap_f = theta 
            #set bounds for the RECTE charge trapping parameters
            if 0.0 < trap_pop_s < 500 and 0.0 < dtrap_s < 500 and 0.0 < trap_pop_f < 200 and 0.0 < dtrap_f < 200 :
                return 0.0
            return -np.inf
        else:
            None
    
    def log_probability(self, theta, model, x, y, yerr):
        """
        A supplemental function needed to run the function `MCMC`
        
        Sets up the full log-probability function. Combines the log_prior and the log_likelihood functions. 
        
        Parameters
        ----------
        theta: vector
            A vector that contains the seven free parameters used by our model (transit_RECTE/eclipse_RECTE) function.
        model: function
            A function that models either transits or secondary eclipses. Must be previously defined 
            (transit_model_RECTE or eclipse_model_RECTE).
        x: array
            Time in Julian days 
        y: array
            Flux data
        yerr: array
            Error in the flux data 
        """
        lp = self.log_prior(theta, model) #Name the log_prior
        if np.isfinite(lp): #Add the log prior and likelihood
            return lp + self.log_likelihood(theta, model, x, y, yerr)
        return -np.inf
    
    def MCMC(self, tshirt_obj, model, iterations, int_guess=None, nbins=10, recalculate=False, showPlot=False, LCPlot=False):
        """
        Optimizes batman model light curves (for transits and/or secondary eclipses) based on initial parameters. 
        This function does consider `RECTE` charge trapping parameters. 
        This function utilizies the `emcee` model fitting approach. 
        
        Parameters
        ----------
        tshirt_obj: keyword
            Time Series Helper & Integration Reduction Tool (tshirt) Spectrometry Object
        model: function
            A function that models either transits or secondary eclipses that encorporate RECTE charge trapping parameters . 
            Must be previously defined (transit_model_RECTE or eclipse_model_RECTE).
        iterations: int
            The number of steps to run the emcee sampler chain.
        int_guess: list
            A list of initial guess values for the model parameters. Default == None and will use preset numbers. 
        nbins: int
            The number of wavelength bins. The Default is "nbins=10".
        recalculate: bool
            Recalculate the model optimizations? The Default is "False"
        showPlot: bool
            Make the convercengece plots visible? The Default is "False"
        LCPlot: bool
            Make the light curve plots visible? The Default is "False"
        """
    
        planet_data = self.planet_data #Call the planet file data
        
        im = self.im  #Call the median image

        exptime = self.exptime #Call the exposure time
            
        #Obtain a table of the the wavelength-binned time series (with `tshirt`). 
        #Seperate out the raw flux data (raw_results) and the raw flux error data (raw_results_errors) into two different pandas tables.
        results = tshirt_obj.get_wavebin_series(nbins=nbins)
        raw_results = results[0].to_pandas()
        raw_results_errors = results[1].to_pandas()
        
        #Define the axis columns as well as the corresponding error columns.
        ydata_columns = raw_results.columns[1:].values #Skip over the time column
        ydata_errors_columns = raw_results_errors.columns[1:].values #Skip over the time column
        xdata = self.x #Time column data in terms of days accounting for Solar barycenter correction
    
        #Obtain a table of wavelength bins, with theoretical noise and measured standard deviation across time.
        table_noise = tshirt_obj.print_noise_wavebin(nbins=nbins)
        table_noise=table_noise.to_pandas() #convert to a pandas table
        
        #Wavelength calibration to turn the dispersion pixels into wavelengths. 
        wavelength_list = tshirt_obj.wavecal(table_noise['Disp Mid'],waveCalMethod = planet_data['WaveCalMethod'])
        
        #Defining the global xList parameter needed in the model functions with RECTE
        xList_all = [] #empty list to store each wavlengths dispersion indices.
        #Loop over each wavelength's bin index. The dispersion indices will be different for each wavelength bin. 
        for ind in table_noise.index: 
            Disp_st = table_noise['Disp St'][ind] #Start of the dispersion range
            Disp_end = table_noise['Disp End'][ind] #End of the dispersion range
            Disp_xList = np.arange(Disp_st, Disp_end,1) #Return Numpy array of evenly spaced values within a given interval. 
            xList_all.append(Disp_xList) #Append Numpy array to the empty list to late be iterated over.   
       
        #Define a LIST of initial guess error values for each parameter by calling the `optimize_batman_model_RECTE` function. This can also be a list of arrays, each array dedicated to a wavebin. 
        initial_error_list = self.optimize_batman_model_RECTE(tshirt_obj,model,nbins=nbins,int_guess=int_guess,useMultiprocessing = True)[1]
        error_2D = np.array(initial_error_list) #Make the error list into an array
        bad_points = np.isfinite(error_2D) == False #Define any non-finite values in the array as bad points. 
        error_2D[bad_points] = np.nan #Replace bad points as nan values
        avg_error=np.nanmean(error_2D, axis=0) #average over each row. If only one list in the array,average will return the initial list.   
        
        #Set up plotting options for the Light Curve
        if (LCPlot==True):
            fig, (ax2, ax3, ax4) = plt.subplots(1,3,figsize=(20,10),sharey=False)
        
        #Establish the model in use based on initial function input. Establish the parameters and the initial parameter guess values (p0) for each model. 
        if(model==self.transit_model_RECTE):
            labels = ["rp", "a", "b","trap_pop_s","dtrap_s", "trap_pop_f", "dtrap_f"]
            text = 'fit: rp=%5.3f, a=%5.3f,b=%5.3f,trap_pop_s=%5.3f,dtrap_s=%5.3f,trap_pop_f=%5.3f,dtrap_f=%5.3f'
            if int_guess == None:
                print('\033[91m'+"Hint: The closer your guess, the better the model fit"+'\033[91m')
                rp_guess = input("Pick a guess value for the planet-to-star radius ratio (rp) [limits:(-\u221e,\u221e)]:")
                a_guess = input("Pick a guess value for the Baseline linear regression y-intercept applied to the normalized modeled flux (a) [limits:(0,\u221e)]:") #Prompt User input for the guess values
                b_guess = input("Pick a guess value for the Baseline linear regression slope applied to the normalized modeled flux (b) [limits:(-\u221e,\u221e)]:")
                trap_pop_s = input("Pick a guess value for the number of initially occupied traps -- slow poplulation (trap_pop_s) [limits:(0,500)]:")
                dtrap_s = input("Pick a guess value for the number of extra trapped charge carriers added in the middle of two orbits -- slow population. (dtrap_s) [limits:(0,500)]:")
                trap_pop_f = input("Pick a guess value for the number of initially occupied traps -- fast poplulation (trap_pop_f) [limits:(0,200)]:")
                dtrap_f = input("Pick a guess value for the number of extra trapped charge carriers added in the middle of two orbits -- fast population. (dtrap_f) [limits:(0,200)]:")
                print('\033[91m'+"Save these values and use as input for int_guess next time you run this function:"+'\033[91m'+str([rp_guess, a_guess, b_guess,trap_pop_s,dtrap_s,trap_pop_f,dtrap_f]))
                p0 = [float(rp_guess),float(a_guess),float(b_guess),int(trap_pop_s),int(dtrap_s),int(trap_pop_f),int(dtrap_f)] #each guess value in the list corresponds to the parameter order in text
            else:
                p0 = int_guess
                
        elif(model==self.eclipse_model_RECTE):
            labels = ["fp", "a", "b","trap_pop_s","dtrap_s", "trap_pop_f", "dtrap_f"]
            text = 'fit: fp=%5.3f, a=%5.3f,b=%5.3f,trap_pop_s=%5.3f,dtrap_s=%5.3f,trap_pop_f=%5.3f,dtrap_f=%5.3f'
            if int_guess == None:
                print('\033[91m'+"Hint: The closer your guess, the better the model fit"+'\033[91m')
                fp_guess = input("Pick a guess value for the planet-to-star flux ratio (fp):")
                a_guess = input("Pick a guess value for the Baseline linear regression y-intercept applied to the normalized modeled flux (a):") #Prompt User input for the guess values
                b_guess = input("Pick a guess value for the Baseline linear regression slope applied to the normalized modeled flux (b):")
                trap_pop_s = input("Pick a guess value for the number of initially occupied traps -- slow poplulation (trap_pop_s):")
                dtrap_s = input("Pick a guess value for the number of extra trapped charge carriers added in the middle of two orbits -- slow population. (dtrap_s):")
                trap_pop_f = input("Pick a guess value for the number of initially occupied traps -- fast poplulation (trap_pop_f):")
                dtrap_f = input("Pick a guess value for the number of extra trapped charge carriers added in the middle of two orbits -- fast population. (dtrap_f):")
                print('\033[91m'+"Save these values and use as input for int_guess next time you run this function:"+'\033[91m'+str([fp_guess, a_guess, b_guess,trap_pop_s,dtrap_s,trap_pop_f,dtrap_f]))
                p0 = [float(fp_guess),float(a_guess),float(b_guess),int(trap_pop_s),int(dtrap_s),int(trap_pop_f),int(dtrap_f)] #each guess value in the list corresponds to the parameter order in text
            else:
                p0=int_guess
        else: 
            print("Invalid Model Input") #This function only works on the above previously defined models!
            
        #Define empty lists to store percentile computations
        q50_array =[] #50th, median
        q16_array =[] #16th, -1 std
        q84_array =[] #84th, +1 std
        
        #Establsih a color map index, to be iterated over, based on the number of wavebins defined. 
        color_idx_range = np.linspace(0.3, 0.8, nbins)
        
        #Loop over the flux data and their respective flux data error columns simultaneously for each wavelength. 
        #Each wavelength will have an associated color (determined by the color index) and bin number.  
        for columns,columns_errors,bin_number,color_idx, wavelength,disp_range in zip(ydata_columns,ydata_errors_columns,np.arange(nbins),color_idx_range,wavelength_list,xList_all):
            
            xList = disp_range #The xList will correspond to the dispersion range. 

            ydata = raw_results[columns].values # Return as a Numpy representation of the data.
            ydata_errors = raw_results_errors[columns_errors].values
            
            start_MLE = time.time() #Start of the internal timer for Maximum likelihood estimate
            
            #This line is used to save results to a specific folder in order to streamline previously run data. (Can be altered)
            new_dir = pathlib.Path(planet_data['BaseDir'], 'MCMC')  #Establish a path to the saved data based on the given base directory
            result_file = os.path.join(new_dir,"{}_MLE_{}_wavelength_ind_{}_nbins{}.csv".format(planet_data['srcName'],tshirt_obj.param['nightName'],columns,nbins)) #Generate a name for the light curve file based on YAML files
            #result_file = 'opt_result_tables/MCMC20000_soln.x_visit_{}_wavelength_ind_{}_nbins{}.csv'.format(self.param['nightName'],columns,nbins)
            new_dir.mkdir(parents=True, exist_ok=True)  #Make the new directory

            #If the previously defined results_file exists and the recalculation parameter is set to False, read it in. 
            if (os.path.exists(result_file) == True) and (recalculate == False):
                dat = ascii.read(result_file)
                soln_xarray = dat['MLE'] #read in the solution array,the numerical optimums of this likelihood function, the maximum liklihood estimates
                
            #If the previously defined results_file does not exsit or if the recalculation parameter is set to True, call and run the scipy.optimize.minimize function. 
            else:
                #Create a table for these solution array and save them to the results_file defined previously. 
                dat = Table() 
                #run MCMC
                nll = lambda *args: -self.log_probability(*args) #Define a small anonymous function (nll using lambda) that take all arguments required of the log_probability function. Define the expession to be exectured as the log_probability function. 
                initial = np.array([p0[0],p0[1],p0[2],p0[3],p0[4],p0[5],p0[6]]) #Define the initial guess values
                soln = minimize(nll, initial, args=(model,xdata, ydata, ydata_errors)) #Run the scipy.optimize.minimize function to return optimization results.
                soln_xarray = soln.x #Grab and save the solution array.
                print(soln.x)
                dat['MLE'] = soln['x']
                dat.write(result_file)
                
            end_MLE = time.time() #End of the internal timer for Maximum likelihood estimate
            MLE_time= end_MLE-start_MLE
            print("Maximum Likelihood Estimation Took {0:.1f} Seconds".format(MLE_time))
    
            nwalkers=14 #Define the number of walkers in the ensemble (can vary)
            ndim = 7 #Define the number of parameters in model
    
            pos = np.empty([nwalkers,ndim]) # Define the shape of the initial state or position vector.
            
            #Loop through each element of each array simultaneously (The initial guess error value of each parameter, the MLE value for each parameter, and a range where the index of the returned array pertains to each parameter.  
            for i, j, k in zip(avg_error,soln_xarray,np.arange(ndim)):
                if k == 3 or k == 4:
                    pos[:,k] = np.random.rand(nwalkers) * 500 #Confine the trap_pop_s and dtrap_s populations to a non-negative space with a limit at 500 (reasonable trap population bounds)
                elif k ==5 or k==6:
                    pos[:,k] = np.random.rand(nwalkers) * 200 #Confine the trap_pop_f and dtrap_f populations to a non-negative space with a limit at 200 (reasonable trap population bounds)
                else: 
                    pos[:,k] = j + i* np.random.randn(nwalkers)*10  #All other parameters initialized around the maximum likelihood results           
            
            #This line is used to save results to a specific folder in order to streamline previously run MCMC analysis. (Can be altered)
            MCMC_file = os.path.join(new_dir,"{}_MCMC_{}_wavelength_ind_{}_nbins{}.h5".format(planet_data['srcName'],tshirt_obj.param['nightName'],iterations,columns,nbins)) 
            #MCMC_file = '/fenrirdata1/kg_data/sample_chains20000/MCMC20000_visits_{}_wavelength_ind_{}_nbins{}.h5'.format(self.param['nightName'],columns,nbins)
            
            #If the previously defined MCMC_file exists and the recalculation parameter is set to False, read it in. 
            if (os.path.exists(MCMC_file) == True) and (recalculate == False):
                sampler = emcee.backends.HDFBackend(MCMC_file, read_only=True) #A reader for existing samplings
                check_step_size = sampler.get_chain() #Check the MCMC step size
                print("The Number of Steps = "+ str(check_step_size.shape[0]))
    
                #If the step size is less than the defined iterations in the function continue MCMC analysis
                if (check_step_size.shape[0] < iterations):
                    print("Found "+str(check_step_size.shape[0])+" steps, running "+ str(iterations - check_step_size.shape[0])+ " more steps.")
                    
                    with Pool(16) as pool: #preform with multiprocessing
                        new_backend = emcee.backends.HDFBackend(MCMC_file) #In order to save additional emcee runs, rename the backend object
                        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, args=(model,xdata, ydata, ydata_errors), backend=new_backend, pool=pool) #instantiating an EnsembleSampler for emcee
                        
                        #Iterate sampler for nsteps iterations and return the result. 
                        #Set initial state or position vector to NONE in order to resume where run_mcmc last was executed from. Set store=True to save runs. 
                        sampler.run_mcmc(None, iterations-check_step_size.shape[0], progress=True, store=True) 
    
            #If the previously defined MCMC_file does not exsit or if the recalculation parameter is set to True, call and run the scipy.optimize.minimize function. 
            else:
                with Pool(16) as pool: #preform with multiprocessing
                    
                    start_MCMC = time.time() #Start of the internal timer for MCMC analysis
    
                    backend = emcee.backends.HDFBackend(MCMC_file) #Create a backend that stores the chain in memory
                    backend.reset(nwalkers, ndim) #clear in case file exists
    
                    sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, args=(model,xdata, ydata, ydata_errors),backend=backend,pool=pool) #instantiating an EnsembleSampler for emcee
                    
                    #Iterate sampler for nsteps iterations and return the result.Define initial state or position vector and set store=True to save runs. 
                    sampler.run_mcmc(pos, iterations, progress=True,store=True);
                    
                    end_MCMC = time.time() #End of the internal timer for MCMC analysis
                    MCMC_time = end_MCMC - start_MCMC
                print("MCMC Multiprocessing Analysis Took {0:.1f} Seconds".format(MCMC_time))
            
            tau = sampler.get_autocorr_time(quiet=True) #Compute an estimate of the autocorrelation time for each parameter. The parameter quiet=True will return a warning rather than stop the code. 
    
            burnin = int(2 * np.max(tau)) #Define the "burn-in" steps for each parameter based on its autocorrelation time. To be discarded.
            thin = int(0.5 * np.min(tau)) #Define how to thin the sampler chain. Take only every "thin" steps from the chain. 
            #print("The Autocorrelation Time is: {0}".format(tau))
            #print("The Burn-In Steps: {0}".format(burnin))
            #print("Thin the Sampler Chain by: {0}".format(thin))
            
            flat_samples = sampler.get_chain(discard=0,thin=1,flat=True) # Flatten the chain so that we have a flat list of samples
            
            #Define the 50th Percentile for each parameter; the mean 
            q50 = np.percentile(flat_samples,50,axis=0)
            q50_array.append(q50)
            
            #Define the 16th Percentile for each parameter; -1 std
            q16 = np.percentile(flat_samples,16,axis=0)
            q16_array.append(q50-q16) #Define the lower unccertainty limit
            
            #Define the 84th Percentile for each parameter; +1 std
            q84 = np.percentile(flat_samples,84,axis=0)
            q84_array.append(q84-q50) #Define the upper unccertainty limit
                
    
            #Plotting options for the parameter distributions
            if(showPlot==True):
                fig, axes = plt.subplots(7, figsize=(10, 7), sharex=True) #Set up the figure space
                
                samples = sampler.get_chain() #Get the stored chain of MCMC samples
    
                #Loop through each parameter
                for k in range(ndim):
                    ax = axes[k]
                    ax.plot(samples[:, :, k], "k", alpha=0.3) #Plot each parameter distribution 
                    ax.set_xlim(0, len(samples))
                    ax.set_ylabel(labels[k])
                    #ax.yaxis.set_label_coords(-0.1, 0.5)
                axes[-1].set_xlabel("Step Number");
                
                pltcorner=corner.corner(flat_samples, labels=labels, quantiles=[0.16, 0.5, 0.84],show_titles=True); #generate a corner plot for each wavebin
                Corner_Plot_Path = os.path.join(new_dir,"{}_MCMC_corner_plot_{}_wavelength_ind_{}_nbins{}.pdf".format(planet_data['srcName'],tshirt_obj.param['nightName'],columns,nbins)) 
                pltcorner.savefig(Corner_Plot_Path, bbox_inches='tight')

            #Light Curve Plotting
            if(LCPlot==True):
                
                offset = 0.007 #Define an offset between wavebins
            
                inds = np.random.randint(len(flat_samples), size=100) #Define, at random number, indices of the flat sample to generate models 
                
                family_of_models=[]  #Define an empty array to store the family of potential light curve models that MCMC generated
                family_of_models_params=[] #Define an empty array to store the family of models parameters.
                
                #Loop through these indices
                for ind in inds:
                    sample = flat_samples[ind] #Pull the parameter values at this index as a sample
                    ymodel=model(xdata, *sample) #Plug in these sample values into the model function
                    family_of_models.append(ymodel)
                    family_of_models_params.append(sample)
                    
                    ramp_family=self.calculate_correction_fast(xdata,exptime,im,xList=xList,trap_pop_s=sample[3], dTrap_s=[sample[4]], trap_pop_f=sample[5], dTrap_f=[sample[6]]) #calculate the ramp in the data based on optimized parameters (sample)
                    ramp_model_family = np.mean(ramp_family,axis=0) #Defing the ramp model; the mean along the rows(axis=0)
                    baseline_model_family=self.eclipse_model(xdata, 0, sample[1], sample[2]) #define the basline, to be divided out to see only the eclipse later (fp=0, no eclipse)
                    
                    ax2.plot(xdata, ymodel-bin_number*offset, color=plt.cm.gist_heat(color_idx), alpha=0.1) #Plot the family of models
                    ax3.plot(xdata, (ymodel/ramp_model_family/baseline_model_family)-bin_number*offset, color=plt.cm.gist_heat(color_idx), alpha=0.1) #Plot the family of models
    
                ax2.errorbar(xdata, ydata-bin_number*offset, yerr=ydata_errors, color=plt.cm.gist_heat(color_idx), fmt="o", capsize=3,markersize=3) #Plot the light curves with error bars for each wavebin
    
                #log_pb = sampler.get_log_prob(discard=burnin,thin=thin,flat=True) #Get the chain of log probabilities evaluated at the MCMC samples
                #maximum_index = np.argmax(log_pb) #Define the maximum log probability
                #sample_max = flat_samples[maximum_index] #Pull the parameter values at this index of maximum log probability as sample_max
                #ymodel_max = model(xdata, *sample_max) #Plug in these sample_max values into the model function
                #ax2.plot(xdata, ymodel_max-bin_number*offset, color="black",linewidth=3) #Plot the Maximum Likelihood model
                median_model = np.median(family_of_models, axis=0)#Define the median in the family of models along rows (axis=0).
                median_model_params=np.median(family_of_models_params, axis=0) #Define the median in the family of models parameters along rows (axis=0)
                
                ax2.plot(xdata, median_model-bin_number*offset, color="black",linewidth=2) #Plot the mean model
                
                #Axis title specific to the overall figure
                fig.suptitle("{} {} Date:{}".format(planet_data['srcName'], tshirt_obj.param['nightName'], self.obsdate), fontsize=30)

                #Light Curve Plotting Labels/Legend
                ax2.set_ylabel("Normalized Flux + Offset", fontsize=20)
                ax3.set_xlabel("Time (BJD)", fontsize = 20)
                #ax2.tick_params(axis='x', labelsize=20)
                #ax2.tick_params(axis='y', labelsize=20)
                #ax2.xaxis.offsetText.set_fontsize(20)
    
                legend_elements = [Line2D([0], [0], color='black', lw=2, label='Median Model')] #Legend Plot
                ax2.legend(handles=legend_elements, fontsize=10)
                ax3.legend(handles=legend_elements, fontsize=10)
                
                ramp=self.calculate_correction_fast(xdata,self.exptime,self.im,xList=xList,trap_pop_s=median_model_params[3], dTrap_s=[median_model_params[4]], trap_pop_f=median_model_params[5], dTrap_f=[median_model_params[6]]) #calculate the ramp in the data based on optimized parameters (median_model_params)
                ramp_model = np.mean(ramp,axis=0) #Defing the ramp model; the mean along the rows(axis=0)
                baseline_model=self.eclipse_model(xdata, 0, median_model_params[1], median_model_params[2]) #define the basline, to be divided out to see only the eclipse later
                
                ax3.plot(xdata, (ydata/ramp_model/baseline_model)-bin_number*offset,'o',color=plt.cm.gist_heat(color_idx),markersize=5) 
                ax3.plot(xdata, (median_model/ramp_model/baseline_model)-bin_number*offset, color="black",linewidth=2) #Plot the mean model astrophysical data only
    
                #The third plot (ax4) is the residuals of the original data from the median model
                ax4.plot(xdata, (ydata-median_model)-bin_number*offset,'o',color=plt.cm.gist_heat(color_idx),markersize=5)
                ax4.yaxis.set_label_position("right")
                ax4.yaxis.tick_right()
                ax4.set_ylabel("Residuals + Offset", fontsize=20)
                
                #Annotation Settings
                if(model==self.transit_model_RECTE):
                    ax2.annotate("{:.2f}$\mu m$".format(wavelength), xy =(np.mean(xdata)-0.085, np.mean(ydata)-bin_number*offset+0.0085),fontsize=10,weight='bold',color=plt.cm.gist_heat(color_idx)) #annotation for wavelength values
                    ax3.annotate("{:.2f}$\mu m$".format(wavelength), xy =(np.mean(xdata)-0.085, np.mean(ydata/ramp_model)-bin_number*offset+0.0085),fontsize=10,weight='bold',color=plt.cm.gist_heat(color_idx)) #Annotate the wavelengths on the plot
                    ax4.annotate("{:.2f}$\mu m$".format(wavelength), xy =(np.mean(xdata)-0.080, np.mean(ydata-ymodel)-bin_number*offset+0.0015),fontsize=10,weight='bold',color=plt.cm.gist_heat(color_idx)) #Annotate the wavelengths on the plot

                elif(model==self.eclipse_model_RECTE):
                    ax2.annotate("{:.2f}$\mu m$".format(wavelength), xy =(np.mean(xdata)-0.085, np.mean(raw_results[columns])-bin_number*offset+0.002),fontsize=10,weight='bold',color=plt.cm.gist_heat(color_idx)) #annotation for wavelength values
                    ax3.annotate("{:.2f}$\mu m$".format(wavelength), xy =(np.mean(xdata)-0.09, np.mean(ydata/ramp_model)-bin_number*offset+0.0035),fontsize=10,weight='bold',color=plt.cm.gist_heat(color_idx)) #Annotate the wavelengths on the plot
                    ax4.annotate("{:.2f}$\mu m$".format(wavelength), xy =(np.mean(xdata)-0.085, np.mean(ydata-ymodel)-bin_number*offset+0.0015),fontsize=10,weight='bold',color=plt.cm.gist_heat(color_idx)) #Annotate the wavelengths on the plot

                else: 
                    None
                
                LightCurve_Plot_Path = os.path.join(new_dir,"{}_MCMC_{}_LightCurve.pdf".format(planet_data['srcName'],tshirt_obj.param['nightName'])) 
                fig.savefig(LightCurve_Plot_Path)
            
        return q50_array,q16_array,q84_array