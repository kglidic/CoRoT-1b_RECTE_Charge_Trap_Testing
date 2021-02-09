#import the spectroscopic module from the tshirt pipeline
from tshirt.pipeline import spec_pipeline

#import the RECTE charge correction functions
import Charge_Correction_Functions
from Charge_Correction_Functions import RECTE,RECTEMulti, calculate_correction_fast, charge_correction

#import the Modeling functions
import Transit_Eclipse_Modeling_Functions
from Transit_Eclipse_Modeling_Functions import transit_model,transit_model_RECTE,eclipse_model,eclipse_model_RECTE,barycenter_correction,optimize_batman_model,optimize_batman_model_RECTE

#import basic plotting libraries/set plot settings
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib import rcParams
rcParams['font.family'] = 'serif'

#import bokeh to enable interactive plots
from bokeh.plotting import figure
from bokeh.io import output_notebook, push_notebook, show
output_notebook()

#import yaml to read in the parameter file
import yaml

#Basic imports
import os
from astropy.table import QTable
import astropy.units as u
import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table, join
import pandas as pd
from astropy.time import Time
from copy import deepcopy
import time
from multiprocessing import Pool

#modeling transit/eclipse light curves
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import batman
import corner
import emcee


#to fix errors
import pdb

#to correct for time differences
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation

#-------------------------------------------------------------------------
#Define the log liklihood function; a natural logarithm of the liklihood
#Measures the goodness of fit of a statistical model to a sample of data for given values of the unknown parameters.
#Procedure for obtaining maximum likelihood estimation (the parameter values for the model such that they maximize the likelihood of this model actually being observed) is done in the function `MCMC`. 
def log_likelihood(theta, x, y, yerr):
    fp,a,b,trap_pop_s,dtrap_s,trap_pop_f,dtrap_f  = theta #the unknown parameters, the blueprint for the model
    model = eclipse_model_RECTE(x, fp, a, b, trap_pop_s, dtrap_s, trap_pop_f, dtrap_f) #the model 
    return -0.5 * np.sum((y - model) ** 2 / yerr ** 2)
#---------------------------------------------------------------
#Define prior knowledge of the parameters by setting bounds to restrict some parameters
def log_prior(theta):
    fp,a,b,trap_pop_s,dtrap_s,trap_pop_f,dtrap_f  = theta #parameters
    
    #set bounds for the RECTE charge trapping parameters
    if 0.0 < trap_pop_s < 500 and 0.0 < dtrap_s < 500 and 0.0 < trap_pop_f < 200 and 0.0 < dtrap_f < 200 :
        return 0.0
    return -np.inf
#-------------------------------------------------------------------
#Set up the full log -probability function; combine the log prior and the log likelyhood 
def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if np.isfinite(lp):
        return lp + log_likelihood(theta, x, y, yerr)
    return -np.inf
#-------------------------------------------------------------------
def MCMC(self,model,iterations,nbins=10,recalculate=False,showPlot=False,LCPlot=False,Co_add_visit_check=False):
    
    #Obtain a table of the the wavelength-binned time series. 
    #Seperate out the raw flux data (raw_results) and the raw flux error data (raw_results_errors) into two different pandas tables.
    results = self.get_wavebin_series(nbins=nbins)
    raw_results = results[0].to_pandas()
    raw_results_errors = results[1].to_pandas()
    
    #Call the barycenter time correction function. Will return correction in days. 
    time_correction = barycenter_correction(self)
    
    #Define the axis data as well as the corresponding errors.
    ydata_columns = raw_results.columns[1:].values #Skip over the time column
    ydata_errors_columns = raw_results_errors.columns[1:].values #Skip over the time column
    xdata = raw_results['Time'].values+time_correction #Time column data in terms of days accounting for Solar barycenter correction

    #Obtain a table of wavelength bins, with theoretical noise and measured standard deviation across time.
    table_noise = self.print_noise_wavebin(nbins=nbins)
    table_noise=table_noise.to_pandas() #convert to a pandas table
    
    #Wavelength calibration to turn the dispersion pixels into wavelengths. 
    #CoRoT-1 b used wavecalMethod='wfc3Dispersion' for the HST WFC3 grism 
    wavelength_list = self.wavecal(table_noise['Disp Mid'],waveCalMethod = 'wfc3Dispersion')    
    
    #Define a LIST of initial guess error values for each parameter by calling the `optimize_batman_model_RECTE` function. This can also be a list of arrays, each array dedicated to a wavebin. 
    initial_error_list = optimize_batman_model_RECTE(self,model,nbins=nbins)[1]
    error_2D = np.array(initial_error_list) #Make the error list into an array
    bad_points = np.isfinite(error_2D) == False #Define any non-finite values in the array as bad points. 
    error_2D[bad_points] = np.nan #Replace bad points as nan values
    avg_error=np.nanmean(error_2D, axis=0) #average over each row. If only one list in the array,average will return the initial list.   
    
    #Establish the model in use based on initial function input. Establish the parameters and the initial parameter guess values (p0) for each model. 
    if(model==transit_model_RECTE):
        labels = ["pr", "a", "b","trap_pop_s","dtrap_s", "trap_pop_f", "dtrap_f"]
        p0 = [0.13,1.0,0.0,200,100,20,1] #each guess value in the list corresponds to the parameter order in text
        
    elif(model==eclipse_model_RECTE):
        labels = ["fp", "a", "b","trap_pop_s","dtrap_s","trap_pop_f", "dtrap_f"]
        p0 = [500,1.0,0.0,200,100,20,1] #each guess value in the list corresponds to the parameter order in text
    
    else: 
        print("Invalid Model Input") #This function only works on the above previously defined models!
    
    #Define empty lists to store percentile computations
    q50_array =[] #50th, median
    q16_array =[] #16th, -1 std
    q84_array =[] #84th, +1 std
    
    #Establsih a color map index, to be iterated over, based on the number of wavebins defined. 
    color_idx_range = np.linspace(0.3, 0.8, nbins)
    
    #Set up plotting options for the Light Curve
    if (LCPlot==True):
        fig, (ax2) = plt.subplots(figsize=(20,20)) #Set up the figure space
    
    #Loop over the flux data and their respective flux data error columns simultaneously for each wavelength. 
    #Each wavelength will have an associated color (determined by the color index) and bin number.  
    for columns,columns_errors,bin_number,color_idx, wavelength in zip(ydata_columns,ydata_errors_columns,np.arange(nbins),color_idx_range,wavelength_list):
        
        ydata = raw_results[columns].values # Return as a Numpy representation of the data.
        ydata_errors = raw_results_errors[columns_errors].values
        
        start_MLE = time.time() #Start of the internal timer for Maximum likelihood estimate

        #This line is used to save results to a specific folder in order to streamline previously run data. (Can be altered)
        result_file = 'opt_result_tables/MCMC20000_soln.x_visit_{}_wavelength_ind_{}_nbins{}.csv'.format(self.param['nightName'],columns,nbins)
        
        #If the previously defined results_file exists and the recalculation parameter is set to False, read it in. 
        if (os.path.exists(result_file) == True) and (recalculate == False):
            dat = ascii.read(result_file)
            soln_xarray = dat['soln'] #read in the solution array,the numerical optimums of this likelihood function, the maximum liklihood estimates
            
        #If the previously defined results_file does not exsit or if the recalculation parameter is set to True, call and run the scipy.optimize.minimize function. 
        else:
            #Create a table for these solution array and save them to the results_file defined previously. 
            dat = Table() 
            #run MCMC
            nll = lambda *args: -log_probability(*args) #Define a small anonymous function (nll using lambda) that take all arguments required of the log_probability function. Define the expession to be exectured as the log_probability function. 
            initial = np.array([p0[0],p0[1], p0[2], p0[3],p0[4],p0[5],p0[6]]) #Define the initial guess values
            soln = minimize(nll, initial, args=(xdata, ydata, ydata_errors)) #Run the scipy.optimize.minimize function to return optimization results.
            soln_xarray = soln.x #Grab and save the solution array. 
            dat['soln'] = soln['x']
            dat.write(result_file)
            
        end_MLE = time.time() #End of the internal timer for Maximum likelihood estimate
        MLE_time= end_MLE-start_MLE
        print("Maximum Likelihood Estimation Took {0:.1f} Seconds".format(MLE_time))


        nwalkers = 14 #Define the number of walkers in the ensemble (can vary)
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
        MCMC_file = '/fenrirdata1/kg_data/sample_chains20000/MCMC20000_visits_{}_wavelength_ind_{}_nbins{}.h5'.format(self.param['nightName'],columns,nbins)
        
        #If the previously defined MCMC_file exists and the recalculation parameter is set to False, read it in. 
        if (os.path.exists(MCMC_file) == True) and (recalculate == False):
            sampler = emcee.backends.HDFBackend(MCMC_file, read_only=True) #A reader for existing samplings
            check_step_size = sampler.get_chain() #Check the MCMC step size

            #If the step size is less than the defined iterations in the function continue MCMC analysis
            if (check_step_size.shape[0] < iterations):
                print("Found "+str(check_step_size.shape[0])+" steps, running "+ str(iterations - check_step_size.shape[0])+ " more steps.")
                
                with Pool(16) as pool: #preform with multiprocessing
                    new_backend = emcee.backends.HDFBackend(MCMC_file) #In order to save additional emcee runs, rename the backend object
                    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(xdata, ydata, ydata_errors), backend=new_backend, pool=pool) #instantiating an EnsembleSampler for emcee
                    
                    #Iterate sampler for nsteps iterations and return the result. 
                    #Set initial state or position vector to NONE in order to resume where run_mcmc last was executed from. Set store=True to save runs. 
                    sampler.run_mcmc(None, iterations-check_step_size.shape[0], progress=True, store =True) 

        #If the previously defined MCMC_file does not exsit or if the recalculation parameter is set to True, call and run the scipy.optimize.minimize function. 
        else:
            with Pool(16) as pool: #preform with multiprocessing
                
                start_MCMC = time.time() #Start of the internal timer for MCMC analysis

                backend = emcee.backends.HDFBackend(MCMC_file) #Create a backend that stores the chain in memory

                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(xdata, ydata, ydata_errors),backend=backend,pool=pool) #instantiating an EnsembleSampler for emcee
                
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
        
        flat_samples = sampler.get_chain(discard=burnin,thin=thin,flat=True) # Flatten the chain so that we have a flat list of samples
        
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
            
            corner.corner(flat_samples, labels=labels, quantiles=[0.16, 0.5, 0.84],show_titles=True); #generate a corner plot for each wavebin
        
        #Light Curve Plotting
        if(LCPlot==True):
            
            offset = 0.007 #Define an offset between wavebins
        
            inds = np.random.randint(len(flat_samples), size=10) #Define, at random number, indices of the flat sample to generate models 
            
            #Loop through these indices
            for ind in inds:
                sample = flat_samples[ind] #Pull the parameter values at this index as a sample
                ymodel=model(xdata, *sample) #Plug in these sample values into the model function
                
                ax2.plot(xdata, ymodel-bin_number*offset, color=plt.cm.gist_heat(color_idx), alpha=0.1) #Plot the family of models
            
            ax2.errorbar(xdata, ydata-bin_number*offset, yerr=ydata_errors, color=plt.cm.gist_heat(color_idx), fmt="o", capsize=5) #Plot the light curves with error bars for each wavebin

            log_pb = sampler.get_log_prob(discard=burnin,thin=thin,flat=True) #Get the chain of log probabilities evaluated at the MCMC samples
            maximum_index = np.argmax(log_pb) #Define the maximum log probability
            sample_max = flat_samples[maximum_index] #Pull the parameter values at this index of maximum log probability as sample_max
            ymodel_max = model(xdata, *sample_max) #Plug in these sample_max values into the model function
            ax2.plot(xdata, ymodel_max-bin_number*offset, color="black",linewidth=3) #Plot the Maximum Likelihood model

            ax2.annotate("{:.2f}$\mu m$".format(wavelength), xy =(np.mean(xdata)-0.075, np.mean(ydata)-bin_number*offset+0.001),fontsize=25,weight='bold',color=plt.cm.gist_heat(color_idx)) #Annotate the wavelengths on the plot
            
            #Axis title specific to CoRoT-1b
            if self.param['nightName']=='visit1':
                ax2.set_title("CoRoT-1b Primary Transit \n Visit 1: $23^{rd}$ January 2012", fontsize=30)
            elif self.param['nightName']=='visit2':
                ax2.set_title("CoRoT-1b Secondary Eclipse \n Visit 2: $17^{th}$ January 2012", fontsize=30)
            elif self.param['nightName']=='visit3':
                ax2.set_title("CoRoT-1b Secondary Eclipse \n Visit 3: $27^{th}$ January 2012", fontsize=30)
            elif self.param['nightName']=='visit4':
                ax2.set_title("CoRoT-1b Secondary Eclipse \n Visit 4: $5^{th}$ February 2012", fontsize=30)
    
            #Light Curve Plotting Labels/Legend
            ax2.set_ylabel("Normalized Flux + Offset", fontsize=30)
            ax2.set_xlabel("Time (BJD)", fontsize = 30)
            ax2.tick_params(axis='x', labelsize=20)
            ax2.tick_params(axis='y', labelsize=20)
            ax2.xaxis.offsetText.set_fontsize(20)

            legend_elements = [Line2D([0], [0], color='black', lw=3, label='Maximum Likelihood')]
            ax2.legend(handles=legend_elements, fontsize=20)
            
            #figure_name='saved_figures/CoRoT-1b_MCMCLightcurve_{}.jpeg'.format(self.param['nightName'])   
            #fig.savefig(figure_name)
        if (Co_add_visit_check==True):
            
            Orbital_phase = np.mod((xdata- 2454138.32807)/1.5089682,1.0) #convert the time in days into orbital phase. This is specifically for CoRoT-1 b. 

            #This line is used to save results to a specific folder in order to streamline previously run MCMC generated models. (Can be altered)
            Family_model_file  = 'Co_add_visit_check/Family_models_visit_{}_wavelength_ind_{}_nbins{}.txt'.format(self.param['nightName'],columns,nbins)
        
            #If the previously defined Family_model_file exists and the recalculation parameter is set to False, read it in. 
            if (os.path.exists(Family_model_file) == True ) and (recalculate == False):
                family_of_models=np.loadtxt(Family_model_file) #read in the txt file. This file contains a number of possible models for the respective wavelength bins. 
            
            #If the previously defined Family_model_file does not exsit or if the recalculation parameter is set to True, generate the possible light curve models and save.  
            else:
                family_of_models = [] #an empty list to append the models
                family_of_models.append(xdata) #apped the time data to these files for reference. 
                
                inds = np.random.randint(len(flat_samples), size=100) #Define, at random number, indices of the flat sample to generate models. Size (number of models) can vary 
                
                #Loop through these indices
                for ind in inds:
                    sample = flat_samples[ind] #Pull the parameter values at this index as a sample
                    ymodel=model(xdata, *sample) #Plug in these sample values into the model function
                    family_of_models.append(ymodel) #append the models for this bin into the empty list
                np.savetxt(Family_model_file,family_of_models) #save the family_of_models as a a text file under the destination Family_model_file
                
            #This line is used to save results to a specific folder in order to streamline previously run model modifications on the data. (Can be altered)
            modified_flux_file  = 'Co_add_visit_check/modified_flux_visit_{}_wavelength_ind_{}_nbins{}.csv'.format(self.param['nightName'],columns,nbins)
            
            #If the previously defined modified_flux_file exists and the recalculation parameter is set to False, read it in. 
            if (os.path.exists(modified_flux_file) == True ) and (recalculate == False):
                dat = ascii.read(modified_flux_file)
                orbital_phase=dat['Orbital Phase']
                mean_model_divided_out = dat['Modified Flux']
                
            #If the previously defined modified_flux_file does not exsit or if the recalculation parameter is set to True, modify the flux based on the mean model.  
            else:
                dat_table = Table()
                mean_model = np.mean(family_of_models[1:], axis=0) #Define the mean in the family of models along rows (axis=0). Ignore the first array (time). 
                mean_model_divided_out = ydata/mean_model #Divide the raw data by the mean model. 
                dat_table['Orbital Phase'] = Orbital_phase #save the orbital phase rather than time in days.
                dat_table['Modified Flux'] = mean_model_divided_out #save the modified flux values
                dat_table.write(modified_flux_file) #save the table  
                

    return q50_array,q16_array,q84_array