import numpy as np
from scipy import signal
c = 299792458 # speed of light m/s
import cosmo
import HItools
from scipy.ndimage import gaussian_filter

def P_noise(A_sky,theta_FWHM,t_tot,N_dish,nu,lz,T_sys=None,deltav=1,epsilon=1,hitmap=None,return_sigma_N=False,verbose=False):
    ### Return a scale invariant level for the thermal noise power spectrum
    # To ensure a correct forecast, the pixel volumes used to go from sigma_N to
    #   P_N are based on the given beam size and freq resolution
    '''
    Based on Santos+15 (https://arxiv.org/pdf/1501.03989.pdf) eq 5.1
     - theta_FWHM beam size to base pixel size on (use minimum beam size) should be the same
        for all frequencies since angular pixel size will be the same at all frequencies
    '''
    if T_sys is None: # Calculate based on SKA red book eq1: https://arxiv.org/pdf/1811.02743.pdf
        Tspl = 3e3 #mK
        TCMB = 2.73e3 #mk
        T_sys = np.zeros(len(nu))
        for i in range(len(nu)):
            Tgal = 25e3*(408/nu[i])**2.75
            #Trx = 15e3 + 30e3*(nu[i]/1e3 - 0.75)**2 # From Red Book
            Trx = 7.5e3 + 10e3*(nu[i]/1e3 - 0.75)**2 # Amended from above to better fit Wang+20 MK Pilot Survey
            T_sys[i] = Trx + Tspl + TCMB + Tgal
        if verbose==True: print('\nCalculated System Temp [K]: %s'%np.round(np.min(T_sys)/1e3,2),'< T_sys < %s'%np.round(np.max(T_sys)/1e3,2) )
    else: T_sys = np.repeat(T_sys,len(nu)) # For freq independendent given T_sys
    deltav = deltav * 1e6 # Convert MHz to Hz
    t_tot = t_tot * 60 * 60 # Convert observing hours to seconds
    pix_size = theta_FWHM / 3 # [deg] based on MeerKAT pilot survey approach
    A_p = pix_size**2 # Area covered in each pointing (related to beam size - equation formed by Steve)
    N_p = A_sky / A_p # Number of pointings
    t_p = N_dish * t_tot / N_p  # time per pointing
    sigma_N = T_sys / (epsilon * np.sqrt(2 * deltav * t_p) ) # Santos+15 eq 5.1
    if return_sigma_N==True: return sigma_N
    nchannels = (np.max(nu) - np.min(nu))*1e6 / deltav # Effective number of channels given freq resolution
    deltalz = lz/nchannels # [Mpc/h] depth of each voxel on grid
    P_N = np.zeros(len(nu))
    for i in range(len(nu)):
        z = HItools.Freq2Red(nu[i])
        d_c = cosmotools.D_com(z)
        pix_area = (d_c * np.radians(pix_size) )**2 # [Mpc/h]^2 based on fixed pixels size in deg
        V_cell = pix_area * deltalz
        P_N[i] = V_cell * sigma_N[i]**2
    return P_N

def gen_noise_map(hitmap,nu,T_sys,dims):
    ''' Based on the counts/hitmap of data, this will generate the expected thermal noise
    Using eq5.1 in https://arxiv.org/pdf/1501.03989.pdf
    '''
    lx,ly,lz,nx,ny,nz = dims
    deltav = nu[1] - nu[0]
    deltav *= 1e6 # Convert MHz to Hz
    t_p = hitmap * 2 # time per pointing [secs] based on MeerKAT 2 seconds per time stamp
    sigma_N = np.zeros((nx,ny,nz))
    sigma_N = T_sys / np.sqrt( 2 * deltav * t_p)
    noise = np.random.normal(0,sigma_N,(nx,ny,nz))
    noise[hitmap<1] = 0
    return noise

def getbeampars(D_dish,nu,gamma=None,verbose=False):
    # Return beam size for given dish-size and frequency in MHz
    d_c = cosmo.D_com( HItools.Freq2Red(nu)) # Comoving distance to frequency bin
    theta_FWHM = np.degrees(c / (nu*1e6 * D_dish)) # freq-dependent beam size
    sig_beam = theta_FWHM/(2*np.sqrt(2*np.log(2)))
    R_beam = d_c * np.radians(sig_beam) #Beam sigma
    if gamma is not None and gamma!=0: R_beam = gamma*R_beam
    if verbose==True: print('\nTelescope Params: Dish size =',D_dish,'m, R_beam =',np.round(R_beam,1),'Mpc/h, theta_FWHM =',np.round(theta_FWHM,2),'deg')
    return theta_FWHM,R_beam

def ConvolveCube(dT,dims,R_beam=None,BeamType='Gaussian',ReConvolve=False,W=None,nu=None,D_dish=None,gamma=1,verbose=False):
    '''
    Function to smooth entire data cube one slice at a time with smoothimage
    if R_beam==None, it will run a frequency-dependent beam based on a given D_dish size.
    **** Code contributions by Paula Soares in ReConvole option ****
    '''
    if verbose==True: print('\nConvolving map with beam ...')
    lx,ly,lz,nx,ny,nz = dims
    dpix = np.mean([lx/nx,ly/ny]) # pixel size
    if lx!=ly or nx!=ny:
        p = (lx/nx - ly/ny)/(lx/nx)*100
        if verbose==True: print('\nWARNING: Angular dimensions do not match' + '\nPixel percentage difference is %s'%p )
    dT_smooth = np.zeros((nx,ny,nz))
    if BeamType=='Gaussian' and ReConvolve==False:
        for i in range(nz):
            if R_beam is None: R_beam_nu = getbeampars(D_dish,nu[i],verbose=False)[1]
            else: R_beam_nu = R_beam
            dT_smooth[:,:,i] = gaussian_filter(dT[:,:,i], sigma=R_beam_nu/dpix, mode='wrap')
    if BeamType=='Cosine' and ReConvolve==False:
        # Follow approach and default parameters in Matshawule+20 [https://arxiv.org/pdf/2011.10815.pdf]
        x0,y0 = lx/2,ly/2 #central x and y positions
        xbins,ybins = np.linspace(0,lx,nx+1),np.linspace(0,ly,ny+1)
        x,y = (xbins[1:] + xbins[:-1])/2,(ybins[1:] + ybins[:-1])/2  #centre of pixel bins
        y = y[:,np.newaxis]
        A = np.radians(0.1 / 60) #0.1 arcmins coverted to radians
        T = 20 #MHz
        theta_FWHM = c / (nu*1e6 * D_dish) # freq-dependent beam size
        theta_FWHM = theta_FWHM + A*np.sin(2*np.pi*nu/T) # include ripple
        for i in range(len(nu)):
            plottools.ProgressBar(i,len(nu),'\nConvolving with Cosine beam...')
            r = cosmotools.D_com( HItools.Freq2Red(nu[i]) )
            thetax = (x-x0)/r
            thetay = (y-y0)/r
            theta = np.sqrt(thetax**2 + thetay**2)
            kern =  ( np.cos(1.189*theta*np.pi/theta_FWHM[i])/(1-4*(1.189*theta/theta_FWHM[i])**2) )**2
            A = np.sum(kern)
            kern = kern/A #normalise gaussian so that all pixels sum to 1
            R_beam_nu = getbeampars(D_dish,nu[i],verbose=False)[1]
            dT_smooth[:,:,i] = gaussian_filter(dT[:,:,i], sigma=R_beam_nu/dpix, mode='wrap')
            dT_smooth[:,:,i] = signal.convolve2d(dT[:,:,i], kern, mode='same', boundary='wrap')
            #dT_smooth[:,:,i] = ndimage.convolve(dT[:,:,i], kern, mode='wrap') # Different function similiar speed and results
    if ReConvolve==True:
    # Reconvolve cube with frequency dependent beam to one with common effective Gaussian beamsize
        d_max = cosmotools.D_com( HItools.Freq2Red(np.min(nu)) ) # Comoving distance to maximum redshift
        theta_FWHMmax = np.degrees( c / (np.min(nu)*1e6 * D_dish) ) # maximum beam size
        sig_max = theta_FWHMmax/(2*np.sqrt(2*np.log(2)))
        R_max = d_max * np.radians(sig_max)
        for i in range(nz):
            theta_FWHM,R_z = getbeampars(D_dish,nu[i],verbose=False)
            R_beam = np.sqrt( gamma*R_max**2 - R_z**2 )
            if W is None: dT_smooth[:,:,i] = gaussian_filter(dT[:,:,i], sigma=R_beam/dpix, mode='wrap')
            else: # Apply normalised weighted convolution
                dT_weighted = gaussian_filter(dT[:,:,i] * W[:,:,i], sigma=R_beam/dpix, mode='wrap')
                norm = gaussian_filter(W[:,:,i], sigma=R_beam/dpix, mode='wrap')
                dT_smooth[:,:,i] = dT_weighted / norm
    return dT_smooth

def ReConvolve(dT,dims,W=None,nu=None,D_dish=None,gamma=1):
    lx,ly,lz,nx,ny,nz = dims
    dpix = np.mean([lx/nx,ly/ny]) # pixel size
    #if lx!=ly or nx!=ny:
        #p = (lx/nx - ly/ny)/(lx/nx)*100

    dT_smooth = np.zeros((nx,ny,nz))
    # Reconvolve cube with frequency dependent beam to one with common effective Gaussian beamsize
    d_max = cosmo.D_com( HItools.Freq2Red(np.min(nu)) ) # Comoving distance to maximum redshift
    theta_FWHMmax = np.degrees( c / (np.min(nu)*1e6 * D_dish) ) # maximum beam size
    sig_max = theta_FWHMmax/(2*np.sqrt(2*np.log(2)))
    R_max = d_max * np.radians(sig_max)
    for i in range(nz):
        theta_FWHM,R_z = getbeampars(D_dish,nu[i],verbose=False)
        R_beam = np.sqrt( gamma*R_max**2 - R_z**2 )
        if W is None: dT_smooth[:,:,i] = gaussian_filter(dT[:,:,i], sigma=R_beam/dpix, mode='wrap')
        else: # Apply normalised weighted convolution
            dT_weighted = gaussian_filter(dT[:,:,i] * W[:,:,i], sigma=R_beam/dpix, mode='wrap')
            norm = gaussian_filter(W[:,:,i], sigma=R_beam/dpix, mode='wrap')
            norm[norm==0] = 1e-30
            dT_smooth[:,:,i] = dT_weighted / norm
    return dT_smooth

def smooth(dT,map_ra,map_dec,nu, D_dish,gamma=1,freqdep=False):
    ''' Gaussian smooth with constant beam size
    '''
    '''
    if ra[0]>ra[1]: ra = np.flip(ra) # Ensure ra is ascending array
    if dec[0]>dec[1]: dec = np.flip(dec) # Ensure dec is ascending array
    dra = ra[1] - ra[0]
    ddec = dec[1] - dec[0]
    rawidth = ra[-1] - ra[0]
    decwidth = dec[-1] - dec[0]
    rabincenters = ra + dra/2
    decbincenters = dec + ddec/2
    r = rabincenters[:,np.newaxis]
    d = decbincenters[np.newaxis,:]
    r0 = np.min(ra) + rawidth/2 # central ra coordinate - Gaussian peaks here
    d0 = np.min(dec) + decwidth/2 # central dec coordinate - Gaussian peaks here
    '''
    r,d = map_ra,map_dec
    r[r>180] = r[r>180]-360
    r0 = np.median(r)
    d0 = np.median(d)
    if freqdep==True: # do frequency dependent  beam size
        theta_FWHM = np.degrees( c / (nu*1e6 * D_dish) )
    else: theta_FWHM = np.degrees( c / (np.min(nu)*1e6 * D_dish) )
    sigma = gamma * theta_FWHM/(2*np.sqrt(2*np.log(2)))
    for j in range(np.shape(dT)[2]):
        #Create Gaussian kernenls to convole with:
        if freqdep==True: gaussian = np.exp(-0.5 * (((r - r0)/sigma[j])**2 + ((d - d0)/sigma[j])**2))
        else: gaussian = np.exp(-0.5 * (((r - r0)/sigma)**2 + ((d - d0)/sigma)**2))
        gaussian = gaussian/np.sum(gaussian) #normalise gaussian so that all pixels sum to 1
        dT[:,:,j] = signal.fftconvolve(dT[:,:,j], gaussian, mode='same')
    return dT

def weighted_resmooth(dT, w, ra,dec,nu, D_dish, gamma=1):
    print('\nTODO: resmoothing not currently accounting for different pixels sizes across map')
    '''
    Steve's original Gaussian smoothing function rewritten by Paula for weighted
    resmoothing to common resolution purpose. Using Mario's equations in MeerKLASS
    notes overleaf.
    ____
    Smooth entire data cube one slice at a time, using weights
    INPUTS:
    dT: field to be smoothed, in format [nx,ny,nz] where nz is frequency direction
    w: weights for resmoothing
    gamma: padding variable to increase your beam size
    '''
    if ra[0]>ra[1]: ra = np.flip(ra) # Ensure ra is ascending array
    if dec[0]>dec[1]: dec = np.flip(dec) # Ensure dec is ascending array
    dra = ra[1] - ra[0]
    ddec = dec[1] - dec[0]
    rawidth = ra[-1] - ra[0]
    decwidth = dec[-1] - dec[0]
    r0 = np.min(ra) + rawidth/2 # central ra coordinate - Gaussian peaks here
    d0 = np.min(dec) + decwidth/2 # central dec coordinate - Gaussian peaks here
    rabincenters = ra + dra/2
    decbincenters = dec + ddec/2
    r = rabincenters[:,np.newaxis]
    d = decbincenters[np.newaxis,:]
    theta_FWHM = np.degrees( c / (np.min(nu)*1e6 * D_dish) )
    sigma_max = gamma * theta_FWHM/(2*np.sqrt(2*np.log(2)))
    var = np.zeros(np.shape(dT)) # New variance for smoothed map
    for j in range(np.shape(dT)[2]):
        #Create Gaussian kernenls to convole with:
        theta_FWHM = np.degrees( c / (nu[j]*1e6 * D_dish) )
        sigma_z = theta_FWHM/(2*np.sqrt(2*np.log(2)))
        sig = np.sqrt(sigma_max**2 - sigma_z**2)
        gaussian = np.exp(-0.5 * (((r - r0)/sig)**2 + ((d - d0)/sig)**2))
        gaussian2 = gaussian**2
        gaussian = gaussian/np.sum(gaussian) #normalise gaussian so that all pixels sum to 1
        gaussian2 = gaussian2/np.sum(gaussian2) #normalise gaussian2 so that all pixels sum to 1
        denom = signal.fftconvolve(w[:,:,j], gaussian, mode='same') # Normalising denominator factor
        denom[denom==0] = 1e30 # avoid divide by zero error and make zero weight infinitely high
        dT[:,:,j] = signal.fftconvolve(dT[:,:,j]*w[:,:,j], gaussian, mode='same') / denom
        var[:,:,j] = signal.fftconvolve(w[:,:,j], gaussian2, mode='same') / denom**2
    return dT,var

def weighted_resmooth_NEW(dT, w, ra,dec,nu, D_dish, gamma=1,weighted=True):
    '''
    Steve's original Gaussian smoothing function rewritten by Paula for weighted
    resmoothing to common resolution purpose. Using Mario's equations in MeerKLASS
    notes overleaf.
    ____
    Smooth entire data cube one slice at a time, using weights
    INPUTS:
    dT: field to be smoothed, in format [nx,ny,nz] where nz is frequency direction
    w: weights for resmoothing
    gamma: padding variable to increase your beam size
    '''
    if ra[0]>ra[1]: ra = np.flip(ra) # Ensure ra is ascending array
    if dec[0]>dec[1]: dec = np.flip(dec) # Ensure dec is ascending array
    rawidth = ra[-1] - ra[0]
    decwidth = dec[-1] - dec[0]
    if rawidth<0: # This means ra coords run over 0 deg line i.e. 360->0
        ra[ra>=ra[0]] = (ra[ra>=ra[0]] * 1) - 360 # Make continuous RA i.e. 359,360,1 -> -1,0,1
    if decwidth<0:
        print('\nreview decwidth in teletools!') # For case where ra runs over 360deg
        exit()
    # Define central ra/dec coordinate - Gaussian peaks here
    r0 = ra[int(len(ra)/2)] # RA centre approximated like this to avoid off centring it caused by varying pixel size
    d0 = np.min(dec) + decwidth/2 # central dec coordinate - Gaussian peaks here
    r = ra[:,np.newaxis]
    d = dec[np.newaxis,:]
    theta_FWHM = np.degrees( c / (np.min(nu)*1e6 * D_dish) )
    sigma_max = gamma * theta_FWHM/(2*np.sqrt(2*np.log(2)))
    var = np.zeros(np.shape(dT)) # New variance for smoothed map
    for j in range(np.shape(dT)[2]):
        #Create Gaussian kernenls to convole with:
        theta_FWHM = np.degrees( c / (nu[j]*1e6 * D_dish) )
        sigma_z = theta_FWHM/(2*np.sqrt(2*np.log(2)))
        sig = np.sqrt(sigma_max**2 - sigma_z**2)
        gaussian = np.exp(-0.5 * (((r - r0)/sig)**2 + ((d - d0)/sig)**2))
        gaussian2 = gaussian**2
        gaussian = gaussian/np.sum(gaussian) #normalise gaussian so that all pixels sum to 1
        gaussian2 = gaussian2/np.sum(gaussian2) #normalise gaussian2 so that all pixels sum to 1
        if weighted==True:
            denom = signal.fftconvolve(w[:,:,j], gaussian, mode='same') # Normalising denominator factor
            denom[denom==0] = 1e30 # avoid divide by zero error and make zero weight infinitely high
            dT[:,:,j] = signal.fftconvolve(dT[:,:,j]*w[:,:,j], gaussian, mode='same') / denom
            var[:,:,j] = signal.fftconvolve(w[:,:,j], gaussian2, mode='same') / denom**2
        if weighted==False:
            dT[:,:,j] = signal.fftconvolve(dT[:,:,j], gaussian, mode='same')
    if weighted==True: return dT,var,gaussian
    if weighted==False: return dT,gaussian
