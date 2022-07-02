#IMPORT LIBRARY:

#Basic imports
import numpy as np

#import yaml to read in the parameter file
import yaml

#modeling transit/eclipse light curves
import batman
#-------------------------------------------------------------------------------------------------------------------------------------------
def median_image(showPlot=False):
    '''
    Generates a median image from the fits files 
    
    Parameters
    ----------
    showPlot: bool
        Make the plot visible? The Default is "False"
    '''
    global planet_file                                                                       #Define the planet_file outside the function
    
    with open(planet_file, "r") as stream:                                                   #Open the planet parameter YAML file
        planet_data = yaml.safe_load(stream)
        
    for visit in range(len(planet_data['procFiles'])):                                       #Loop through each visit
        
        files = [fn for fn in glob.glob(planet_data['procFiles'][visit]) 
                 if not os.path.basename(fn).startswith(planet_data['excludeList'][visit])]  #Grab all files excluding ones listed in 'excludeList'
        
        head = fits.getheader(random.choice(files), extname="SCI")                           #Grab header information from a random file
        cube3d = np.zeros([len(files),head['NAXIS2'],head['NAXIS1']])                        #Generate a 3D array of zeros; the size is based on header information
        
        for ind,oneFile in enumerate(files):                                                 #Loop through all the fits files and append the image data to the 3D array 
            cube3d[ind,:,:] = fits.getdata(oneFile,extname='SCI')

        medianImage = np.median(cube3d,axis=0)                                              #Find the median of the data
        
        if showPlot==True:
            medianImage_plot = plt.imshow(medianImage)                                      #If True, Plot the median image 
            plt.title(str(planet_data['srcName'])+" Median Image "+str(planet_data['nightName'][visit]))
            plt.xlabel("x-pixels")
            plt.ylabel("y-pixels")
            plt.show()
        else:
            None
            
        new_dir = pathlib.Path(planet_data['BaseDir'], 'Median_Images')                     #Establish a path to the saved data based on the given base directory
        new_dir.mkdir(parents=True, exist_ok=True)                                          #Make the new directory
        
        filename = os.path.join(new_dir,str(planet_data['srcName']+"_MedianImage_"+planet_data['nightName'][visit])) #Generate a name for the median image file based on planet parameters

        outHDU = fits.PrimaryHDU(medianImage,head)                                          #Write the median image data to a file
        outHDU.writeto(filename, overwrite=True)
        plt.imsave(filename+".pdf",medianImage)                                             #Save the median image as a pdf

    return 
#-------------------------------------------------------------------------------------------------------------------------
def transit_model(x, rp, a, b):
    '''
    Models transit light curve using Python package `batman` based on initial parameters stored in params_transit.
    
    Parameters
    ----------
    
    x: array
        Time in Julian days 
    rp: int
        Planet-to-star radius ratio
    a: int
        Linear regression y-intercept applied to the modeled normalized flux
    b: int
        Linear regression slope applied to the modeled normalized flux
    '''
    global planet_file                                        #Define the planet_file outside the function
    
    with open(planet_file, "r") as stream:                    #Open the planet parameter YAML file
        planet_data = yaml.safe_load(stream)

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
#----------------------------------------------------------------------------------------------------------------------------------------
def transit_model_RECTE(x, rp, a, b, trap_pop_s, dtrap_s, trap_pop_f, dtrap_f):
    '''
    Models transit light curve using Python package `batman` based on initial parameters stored in params_transit. These transit models         account for charge trapping systematics using Python package `RECTE`. 
    
    Parameters
    ----------
    
    x: array
        Time in Julian days
    rp: int
        Planet-to-star radius ratio
    a: int
        Linear regression y-intercept applied to the modeled normalized flux
    b: int
        Linear regression slope applied to the modeled normalized flux
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
    '''
    global planet_file                                          #Define the planet_file outside the function
    global xList                                                #Dispersion Range

    with open(planet_file, "r") as stream:                      #Open the planet parameter YAML file
        planet_data = yaml.safe_load(stream)
    
    #Gather Median Image

    MedianImage_file = planet_data['BaseDir']+'Median_Images/'  #Median Image Base Directory

    options=(sorted([f for f in os.listdir(planet_data['BaseDir']+'Median_Images') if not f.endswith('.pdf') and not f.startswith('.')]))  #List out the files to chose from in the Base Directory
    
    print("Median Image Options:\n")                            #Print out the Median Image File Options
    print('\n'.join(map(str, options))) 

    im_input = input("Select a Median Image from the list above:")  #Ask for user Input
    MedianImage_path = str(MedianImage_file)+str(im_input)          #Median Image path based on user input

    if (os.path.exists(MedianImage_path) == True):                  #If path exists, print the path and define the median image data 'im'
        print("Median Image Path:"+str(MedianImage_path))
        im = fits.getdata(MedianImage_path)
    else:                                                           #If the path does not exist, redirect the user
        print('\033[91m'+"Median Image Path Does Not Exist: Please retry or calculate one using the median_image function"+'\033[91m')
    
    #Gather Exposure Time Information
    exptime_list = planet_data['exptime']                           #Grab the exposure time data from the planet_data
    print("Exposure Time List:\n")                                  #Print out the exposure time options
    print('\n'.join(map(str, exptime_list)))

    exptime = int(input("Select a Exposure Time based on the List above:"))  #Define the exposure time

    
    #Define the initial flux for the model based on the regular transit_model function
    flux = transit_model(x,rp,a,b)
    
    #Calculate the ramp profile in the initial flux data
    ramp=calculate_correction_fast(x,exptime,im,xList=xList,trap_pop_s=trap_pop_s, dTrap_s=[dtrap_s], trap_pop_f=trap_pop_f, dTrap_f=[dtrap_f])
    
    #Return the modified flux based on the ramp profile in the data
    flux_modified = flux*np.mean(ramp,axis=0)         #Calculate the light curve
    return flux_modified
#-------------------------------------------------------------------------------------------------------------------------------------------
def eclipse_model(x, fp, a, b):
    '''
    Models eclipse light curve using Python package `batman` based on initial parameters stored in params_eclipse.
    
    Parameters
    ----------
    
    x: array
        Time in Julian days 
    fp: int
        Planet-to-star flux ratio
    a: int
        Linear regression y-intercept applied to the normalized flux
    b: int
        Linear regression slope applied to the normalized flux
    '''
    global planet_file                                        #Define the planet_file outside the function
    
    with open(planet_file, "r") as stream:                    #Open the planet parameter YAML file
        planet_data = yaml.safe_load(stream)
        
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
    flux = m.light_curve(params_eclipse)*(a+b*(x-x0))                   #Calculate the light curve
    return flux
#-------------------------------------------------------------------------------------------------------------------------------------------
def eclipse_model_RECTE(x, fp, a, b, trap_pop_s, dtrap_s, trap_pop_f, dtrap_f):
    '''
    Models eclipse light curve using Python package `batman` based on initial parameters stored in params_eclipse. These transit models         account for charge trapping systematics using Python package `RECTE`. 
    
    Parameters
    ----------
    
    x: array
        Time in Julian days 
    fp: int
        Planet-to-star flux ratio
    a: int
        Linear regression y-intercept applied to the normalized modeled flux
    b: int
        Linear regression slope applied to the normalized modeled flux
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
    '''
    global planet_file                                          #Define the planet_file outside the function
    global xList                                                #Dispersion Range

    with open(planet_file, "r") as stream:                      #Open the planet parameter YAML file
        planet_data = yaml.safe_load(stream)
    
    #Gather Median Image

    MedianImage_file = planet_data['BaseDir']+'Median_Images/'  #Median Image Base Directory

    options=(sorted([f for f in os.listdir(planet_data['BaseDir']+'Median_Images') if not f.endswith('.pdf') and not f.startswith('.')]))  #List out the files to chose from in the Base Directory
    
    print("Median Image Options:\n")                            #Print out the Median Image File Options
    print('\n'.join(map(str, options))) 

    im_input = input("Select a Median Image from the list above:")  #Ask for user Input
    MedianImage_path = str(MedianImage_file)+str(im_input)          #Median Image path based on user input

    if (os.path.exists(MedianImage_path) == True):                  #If path exists, print the path and define the median image data 'im'
        print("Median Image Path:"+str(MedianImage_path))
        im = fits.getdata(MedianImage_path)
    else:                                                           #If the path does not exist, redirect the user
        print('\033[91m'+"Median Image Path Does Not Exist: Please retry or calculate one using the median_image function"+'\033[91m')
    
    #Gather Exposure Time Information
    exptime_list = planet_data['exptime']                           #Grab the exposure time data from the planet_data
    print("Exposure Time List:\n")                                  #Print out the exposure time options
    print('\n'.join(map(str, exptime_list)))

    exptime = int(input("Select a Exposure Time based on the List above:"))  #Define the exposure time
    
    #Define the initial flux for the model based on the regular eclipse_model function
    flux = eclipse_model(x,fp,a,b)
    
    #Define the ramp profile in the initial flux data
    ramp=calculate_correction_fast(x,exptime,im,xList=xList,trap_pop_s=trap_pop_s, dTrap_s=[dtrap_s], trap_pop_f=trap_pop_f, dTrap_f=[dtrap_f])
    
    #return the modified flux based on the ramp profile in the data
    flux_modified = flux*np.mean(ramp,axis=0)      #Calculate the light curve
    return flux_modified
#-------------------------------------------------------------------------------------------------------------------------------------------
def barycenter_correction(self):
    """
    Barycenter correction for the time.  
    
    Parameters
    ----------
    
    self: keyword
        Spectrometry Object
    """
    
    t1, t2 = self.get_wavebin_series()               #Get a table of the the wavelength-binned time series (`tshirt`)
    head = fits.getheader(self.fileL[0])             #Open header information
    #print("Time from tshirt: {}".format(t1['Time'][0]))
    
    expStartJD = head['EXPSTART'] + 2400000.5        #Start time in JD
    #print("Time from EXPSTART keyword {}".format(expStartJD))
    
    t1 = Time(t1['Time'][0],format='jd')             #Time data
    coord = SkyCoord('06 48 19.1724141241 -03 06 07.710423478',unit=(u.hourangle,u.deg))
    loc = EarthLocation.of_site('keck')
    diff = t1.light_travel_time(coord,location=loc) #Time difference
    #print('Travel Time from Keck to Barycenter= {} min'.format((diff / u.min).si))
    
    
    return (diff / u.day).si #Barycenter corrected time.
#-------------------------------------------------------------------------------------------------------------------------------------------
