import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord

def ReadIn(map_file,counts_file,numin=971,numax=1023.8):
    ''' Read-in .fits file for level6 or level5 saved maps '''
    map = fits.open(map_file)[0].data
    counts = fits.open(counts_file)[0].data
    ### Define the intensity map coordinates:
    wproj = WCS(map_file).dropaxis(-1)
    wproj.wcs.ctype = ['RA---ZEA', 'DEC--ZEA'] #projection
    ra,dec = np.zeros_like(map[:,:,0]),np.zeros_like(map[:,:,0])
    for i in range(np.shape(ra)[0]):
        for j in range(np.shape(ra)[1]):
            radec=pixel_to_skycoord(i,j,wproj)
            ra[i,j]=radec.ra.deg
            dec[i,j]=radec.dec.deg
    nu = np.linspace(1,4096,4096)
    nu_orig = cal_freq(nu)/1e6 # original MeerKAT frequency range [MHz]
    nu = nu_orig[(nu_orig>numin) & (nu_orig<numax)] # cut frequency channels to specified range
    map = map[:,:,(nu_orig>numin) & (nu_orig<numax)]
    counts = counts[:,:,(nu_orig>numin) & (nu_orig<numax)]
    map[np.isnan(map)] = 0 # set nan values to zero as default
    map *= 1e3 # convert temp maps from Kelvin to mK
    W = np.ones(np.shape(map)) # binary mask: 1 where pixel filled, 0 otherwise
    W[map==0] = 0
    ### Upgrade to more sophisticated weighting scheme here:
    w = np.copy(W) # currently using unity weighting
    ####################################################################
    nx,ny,nz = np.shape(map)
    dims = [0,0,0,nx,ny,nz] # [lx,ly,lz] calculated later in pipeline after regridding
    return map,w,W,counts,dims,ra,dec,nu,wproj

def cal_freq(ch):
    # Function from Jingying Wang to get L-band channel frequencies
    v_min=856.0
    v_max=1712.0
    dv=0.208984375
    assert((v_max-v_min)/dv==4096)
    freq_MHz=ch*dv+v_min
    freq=freq_MHz*1e6
    return freq

def FilterIncompleteLoS(map,w,W,counts):
    W_fullLoS = np.sum(W,2)==np.max(np.sum(W,2))
    W[np.logical_not(W_fullLoS)] = 0
    w[np.logical_not(W_fullLoS)] = 0
    map[np.logical_not(W_fullLoS)] = 0
    counts[np.logical_not(W_fullLoS)] = 0
    return map,w,W,counts

def MapTrim(map,w,W,counts,ra,dec,ramin=334,ramax=357,decmin=-35,decmax=-26.5):
    trimcut = (ra<ramin) + (ra>ramax) + (dec<decmin) + (dec>decmax)
    map[trimcut],w[trimcut],W[trimcut],counts[trimcut] = 0,0,0,0
    return map,w,W,counts

def ReadInLevel62021(HI_filename,countsfile,numin=970.95,numax=1075.84,nights=None,dishes=None,returnmapcoords=False,level5path=None,level6maskpath=None,manualflagcalib=False,CleanLev5=False):
    ''' For initialising cross-correlations using Jingying's level6 2021 data.
    dishes: set None to read in all combined dishes from level6
            otherwise specify dish numbers to average a certain subset
    manualflagcalib: False for initial JY fasttracked calibration
                     True for revised manual flagging calibration
    CleanLev5: Set True to perform cleaning on level5 dish and time maps individually
    [TODO: include overlapping galaxy data]
    '''
    if level5path is None:
        if manualflagcalib==False: level5path = '/idia/projects/hi_im/raw_vis/MeerKLASS2021/old_version/level5/data/'
        if manualflagcalib==True: level5path = '/idia/projects/hi_im/raw_vis/MeerKLASS2021/level5/data/'
    firstfile = True # Use for loading some data on first file only if loading multiple maps
    ### MeerKAT intensity maps:
    if dishes is None: numberofnights = numberofdishes = 1
    else: numberofnights,numberofdishes = len(nights),len(dishes)
    for n in range(numberofnights):
        for d in range(numberofdishes):
            if dishes is None and nights is None: cube = fits.open(HI_filename)[0].data
            else:
                HI_filename = level5path + nights[n]+'_m0'+dishes[d]+'_Sum_Tsky_xy_p0.3d.fits'
                if os.path.isfile(HI_filename) is False: continue
                print(nights[n],dishes[d])
                cube = fits.open(HI_filename)[0].data
            if firstfile==True: # Only need to do the below once
                ### Define the intensity map coordinates:
                wproj = WCS(HI_filename).dropaxis(-1)
                wproj.wcs.ctype = ['RA---ZEA', 'DEC--ZEA'] #projection
                map_ra,map_dec = np.zeros_like(cube[:,:,0]),np.zeros_like(cube[:,:,0])
                for i in range(np.shape(map_ra)[0]):
                    for j in range(np.shape(map_ra)[1]):
                        radec=pixel_to_skycoord(i,j,wproj)
                        map_ra[i,j]=radec.ra.deg
                        map_dec[i,j]=radec.dec.deg
                ra = map_ra[:,0]
                dec = map_dec[0]
                nu = np.linspace(1,4096,4096) # Original MeerKAT frequency range
                nu_orig = cal_freq(nu)/1e6 # all channels in MHz
                ### Cut frequency channels to Isa's freq-range:
                nu = nu_orig[(nu_orig>numin) & (nu_orig<numax)]
                if CleanLev5==True: trimcut = (map_ra<334) + (map_ra>357) + (map_dec>-26.5)  + (map_dec<-35)
            if dishes is None and nights is None: counts = fits.open(countsfile)[0].data
            else:
                countsfile = level5path + nights[n]+'_m0'+dishes[d]+'_Npix_xy_count_p0.3d.fits'
                counts = fits.open(countsfile)[0].data
                if manualflagcalib==True or level6maskpath is not None:
                    ### Apply Jingying's level6 sigma=6 outlier masks to level5 maps:
                    if level6maskpath is None: filename = '/idia/projects/hi_im/raw_vis/MeerKLASS2021/level6/ALL966/sigma_6/mask/'+nights[n]+'_m0'+dishes[d]+'_level6_p0.3d_sigma6.0_iter0_mask'
                    else: filename = level6maskpath + nights[n]+'_m0'+dishes[d]+'_level6_p0.3d_sigma3.0_iter2_mask'
                    file = pickle.load(open(filename,'rb'))
                    ch_mask = file['ch_mask']
                    cube[:,:,ch_mask] = 0
                    counts[:,:,ch_mask] = 0
            dT_MK = cube[:,:,(nu_orig>numin) & (nu_orig<numax)]; del cube
            counts = counts[:,:,(nu_orig>numin) & (nu_orig<numax)]
            dT_MK[np.isnan(dT_MK)] = 0 # Set nan values to zero as default

            if CleanLev5==True: dT_MK,counts = FGtools.CleanLevel5Map(np.copy(dT_MK),np.copy(counts),nu,w=None,trimcut=trimcut)
            if firstfile==True:
                dT_MK_sum = 1e3*dT_MK # Convert all temp maps from Kelvin to mK
                counts_sum = counts
                firstfile = False # Don't call some code again now first file loop has run
            else: # Add to preloaded maps, then average at the end
                dT_MK_sum += 1e3*dT_MK # Convert all temp maps from Kelvin to mK
                counts_sum += counts
    if dishes is not None or nights is not None: dT_MK = dT_MK_sum/counts_sum # average all maps
    else: dT_MK = np.copy(dT_MK_sum)
    dT_MK[np.isnan(dT_MK)] = 0 # Set any new nan values to zero caused by divide by counts

    W_HI = np.ones(np.shape(dT_MK)) # Binary mask: 1 where pixel filled, 0 otherwise
    W_HI[dT_MK==0] = 0
    ### Old noise weights method:
    #counts_sum[counts_sum==0] = 1e-30 # avoid divide by zero
    #noise = 1/counts_sum
    #w_HI = yichaotools.make_noise_factorizable(noise) # use Yi-Chao's function for defining seperable weights
    ### Set weights as inverse LoS variance, consistent along each LoS so weights do not add to rank of the maps
    var = np.nanvar(dT_MK,2)
    var = np.repeat(var[:,:,np.newaxis], np.shape(dT_MK)[2], axis=2)
    w_HI = np.ones((np.shape(dT_MK)))
    w_HI[W_HI==1] = 1/var[W_HI==1]

    ### Run regridding to comoving space to get grid dimensions:
    #dims = grid.regrid_Steve(dT_MK,map_ra,map_dec,nu)[1]
    #lx,ly,lz,nx,ny,nz = dims
    nx,ny,nz = np.shape(dT_MK)
    dims = [np.nan,np.nan,np.nan,nx,ny,nz] # lx,ly,lz calculated later in pipeline after regridding

    if returnmapcoords==True: return dT_MK,w_HI,W_HI,dims,ra,dec,nu,counts_sum,map_ra,map_dec,wproj
    else: return dT_MK,w_HI,W_HI,dims,ra,dec,nu,counts_sum
