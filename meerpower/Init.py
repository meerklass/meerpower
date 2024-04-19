import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
import os
import plot
import matplotlib.pyplot as plt

def ReadIn(map_file,counts_file=None,numin=971,numax=1023.8,getcoord=True):
    ''' Read-in .fits file for level6 or level5 saved maps '''
    # getcoord: Set True to return map coordinates and dimensions of cube.
    # numin,numax: the frequency range to chose data between
    map = fits.open(map_file)[0].data
    if counts_file is None: counts_file = map_file.replace("Tsky", "Npix_count")
    counts = fits.open(counts_file)[0].data
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
    if getcoord==True: ### Define the intensity map coordinates:
        wproj = WCS(map_file).dropaxis(-1)
        wproj.wcs.ctype = ['RA---ZEA', 'DEC--ZEA'] #projection
        ra,dec = np.zeros_like(map[:,:,0]),np.zeros_like(map[:,:,0])
        for i in range(np.shape(ra)[0]):
            for j in range(np.shape(ra)[1]):
                radec=pixel_to_skycoord(i,j,wproj)
                ra[i,j]=radec.ra.deg
                dec[i,j]=radec.dec.deg
        nx,ny,nz = np.shape(map)
        dims = [0,0,0,nx,ny,nz] # [lx,ly,lz] calculated later in pipeline after regridding
        return map,w,W,counts,dims,ra,dec,nu,wproj
    else: return map,w,W,counts

def subsetmap(level5path,dish_indx=None,scan_indx=None,verbose=False,output_path=None):
    '''Combine chosen combination of level5 maps. Use to construct subset maps for
    cross-correlating and isolating time- and dish- dependent systematics.
    '''
    # - dish_indx,scan_indx: arrays of indices for the dish and scans from which to
    #     build subsets
    # - output_path: specify a path to save output. If None is specified, map is not saved.
    scan,dish = get2021IDs()
    if verbose==True: # Print number of available maps for chosen dish/scan indices:
        count = 0
        for n in scan_indx:
            for m in dish_indx:
                filename = level5path + scan[n]+'_m0'+dish[m]+'_Sum_Tsky_xy_p0.3d.fits'
                if os.path.isfile(filename) is True: count+=1 # check file exists, if so, count
        print('\n' + str(int(count)) + ' maps in subset')
    if count==0:
        return map_ave,w,W,counts_sum
    map_sum,counts_sum = None,None
    i = 0
    for n in scan_indx:
        for m in dish_indx:
            plot.ProgressBar(i,N=len(scan_indx)*len(dish_indx),header='Building subset map:')
            i+=1
            map_file = level5path + scan[n]+'_m0'+dish[m]+'_Sum_Tsky_xy_p0.3d.fits'
            counts_file = level5path + scan[n]+'_m0'+dish[m]+'_Npix_xy_count_p0.3d.fits'
            if os.path.isfile(map_file) is False: continue # check file exists, if not, skip
            map,w,W,counts = ReadIn(map_file,counts_file,getcoord=False)
            # iteratively sum intensity maps and hit counts for all level5 maps:
            if map_sum is None: map_sum = map # for first map in loop
            else: map_sum += map
            if counts_sum is None: counts_sum = counts # for first map in loop
            else: counts_sum += counts
    map_ave = np.zeros(np.shape(map_sum))
    map_ave[counts_sum!=0] = map_sum[counts_sum!=0]/counts_sum[counts_sum!=0]

    W = np.ones(np.shape(map_ave)) # binary mask: 1 where pixel filled, 0 otherwise
    W[map_ave==0] = 0
    ### Upgrade to more sophisticated weighting scheme here:
    w = np.copy(W) # currently using unity weighting
    ####################################################################
    if output_path is not None: # save subset maps to output path
        np.save(output_path+'dish%s-%s_scan%s-%s'%(dish_indx[0],dish_indx[-1],scan_indx[0],scan_indx[-1]),[map_ave,w,W,counts_sum])
    return map_ave,w,W,counts_sum

def get2021IDs():
    # return UNIX IDs for scans, and indices for dish antenas for 2021 L-band observations
    scan = ['1630519596','1631379874','1631387336','1631552188','1631559762','1631659886',
            '1631667564','1631724508','1631732038','1631810671','1631818149','1631982988',
            '1631990463','1632069690','1632077222','1632184922','1632505883','1632760885',
            '1633365980','1633970780','1634252028','1634402485','1634748682','1634835083',
            '1637346562','1637354605','1637691677','1637699408','1638130295','1638294319',
            '1638301944','1638386189','1638639082','1638647186','1638898468','1639157507',
            '1639331184','1639935088','1640540184','1640712986','1640799689']
    dish = []
    for i in range(64):
        dish.append("%02d" %i)
    return scan,dish

def pre_process_2019Lband_CMASS_galaxies(ra_g,dec_g,z_g,ra,dec,zmin,zmax,W_HI):
    ramin_CMASS,ramax_CMASS = np.min(ra[np.mean(W_HI,2)>0]),np.max(ra[np.mean(W_HI,2)>0])
    decmin_CMASS,decmax_CMASS = np.min(dec[np.mean(W_HI,2)>0]),np.max(dec[np.mean(W_HI,2)>0])
    MKcut = (ra_g>ramin_CMASS) & (ra_g<ramax_CMASS) & (dec_g>decmin_CMASS) & (dec_g<decmax_CMASS) & (z_g>zmin) & (z_g<zmax)
    cornercut_lim1 = 146 # set low to turn off
    cornercut_lim2 = 172.5 # set high to turn off
    cornercut = (ra_g - dec_g > cornercut_lim1) & (ra_g - dec_g < cornercut_lim2)
    CMASSgalmask = MKcut & cornercut
    return ra_g[CMASSgalmask],dec_g[CMASSgalmask],z_g[CMASSgalmask]

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

def MapTrim(ra,dec,map1,map2=None,map3=None,map4=None,ramin=334,ramax=357,decmin=-35,decmax=-26.5):
    trimcut = (ra<ramin) + (ra>ramax) + (dec<decmin) + (dec>decmax)
    map1[trimcut] = 0
    if map2 is None: return map1
    else: map2[trimcut] = 0
    if map3 is None: return map1,map2
    else: map3[trimcut] = 0
    if map4 is None: return map1,map2,map3
    else: map4[trimcut] = 0
    return map1,map2,map3,map4

def GridTrim(f1,f2=None,f3=None,f4=None,x0=0,x1=None,y0=0,y1=None):
    nx,ny,nz = np.shape(f1)
    if x1==None: x1 = nx
    if y1==None: y1 = ny
    if x1<0: x1 = nx+(x1-1)
    if y1<0: y1 = ny+(y1-1)
    x = np.arange(0,nx)
    y = np.arange(0,ny)
    x,y = np.tile(x[:,np.newaxis],(1,ny)),np.tile(y[np.newaxis,:],(nx,1))
    trimcut = (x<x0) + (x>x1) + (y<y0) + (y>y1)
    f1[trimcut] = 0
    if f2 is None: return f1
    else: f2[trimcut] = 0
    if f3 is None: return f1,f2
    else: f3[trimcut] = 0
    if f4 is None: return f1,f2,f3
    else: f4[trimcut] = 0
    return f1,f2,f3,f4

#def ReadInLevel62021_Sims(dT_MK,W,counts,dims,ra,dec,nu,PerturbFGs=False,SecondPatch=False,doBeam=False,T_sys=16e3,doRSD=False):
def ReadIn_Sim(dT_MK,W,counts,dims,ra,dec,nu,PerturbFGs=False,SecondPatch=False,doBeam=False,T_sys=16e3,doRSD=False):
    import telescope
    import foreground
    # SecondPatch: Set True to load different foreground patch to test dish cross-correlations
    # T_sys: System tempertaure for noise caluclation. Set Zero for no noise. Default 16K from Jingying's paper
    if doRSD==False: T_HI = np.load('/idia/projects/hi_im/crosspower/2021/sims/T_HI_MultiDark_noRSD.npy')
    if doRSD==True: T_HI = np.load('/idia/projects/hi_im/crosspower/2021/sims/T_HI_MultiDark_wRSD.npy')
    # Remove a redshift bin to match data:
    T_HI = T_HI[:,:,:-1]
    dT_HI = T_HI - np.mean(T_HI)
    T_FG = np.load('/idia/projects/hi_im/crosspower/2021/sims/T_FG.npy')[:,:,:-1]
    if SecondPatch==True: T_FG2 = np.load('/idia/projects/hi_im/crosspower/2021/sims/T_FG_patch2.npy')[:,:,:-1]
    if PerturbFGs==False:
        if SecondPatch==False: T_obs = T_HI + T_FG
        if SecondPatch==True: T_obs = T_HI + T_FG2
    if PerturbFGs==True:
        if SecondPatch==False: T_obs = T_HI + T_FG * foreground.FGPeturbations(dT_MK,W,nu)
        if SecondPatch==True: T_obs = T_HI + T_FG2 * foreground.FGPeturbations(dT_MK,W,nu)
    # Include Beam:
    if doBeam==True: dT_HI = telescope.smooth(dT_HI,ra,dec,nu,D_dish=13.5)
    if doBeam==True: T_obs = telescope.smooth(T_obs,ra,dec,nu,D_dish=13.5)
    # Include Noise:
    if T_sys!=0: T_obs += telescope.gen_noise_map(counts,W,nu,T_sys,dims)
    dT_HI[W==0],T_obs[W==0] = 0,0
    return dT_HI,T_obs

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

#import h5py as h5
import numpy.ma as ma
from numpy.lib.utils import safe_eval

def ReadInYiChaoMaps(HI_filename,Opt_filename):
    dT_MK, ax = load_map(HI_filename,map_name='clean_map')
    nu,ra,dec = ax
    vmin,vmax = nu[0], nu[-1]
    noise,ax = load_map(HI_filename,map_name='noise_diag')
    freqmask = load_map(HI_filename,map_name='mask')
    w_HI = make_noise_factorizable(noise) # use Yi-Chao's function for defining seperable weights
    dT_MK,noise = 1e3*dT_MK,1e3*noise # Convert all temp maps from Kelvin to mK
    W_HI = np.ones(np.shape(dT_MK)) # window for intenisty map to mark dead pixels
    W_HI[dT_MK==0] = 0
    '''
    dT_dirty,ax = load_map(HI_filename,map_name='dirty_map')
    plottools.PlotMap(dT_MK,W_HI)
    plottools.PlotMap(dT_dirty,W_HI)
    plt.show()
    exit()
    '''
    lx,ly,lz = 817.4,268.7,178.5 # Mpc/h [from Yi-Chao draft]
    nx,ny,nz = np.shape(dT_MK)
    dims = [lx,ly,lz,nx,ny,nz]
    delta_g, ax = load_map(Opt_filename,map_name='delta')
    n_g_exp, ax = load_map(Opt_filename,map_name='separable') # <n_g> Optical selection function - use in weighting
    n_g = n_g_exp*(delta_g + 1) # convert to n_g for correct input to Steve's Pk estimators
    #print(np.sum(n_g)) # = 3953.1815530201766
    # n_g not summing to an exact integer as expected
    #############################################################################
    ####### CHECK YI-CHAO'S CALULCATION OF delta_g and W_g due to above #######
    #############################################################################
    W_g01 = np.ones(np.shape(n_g_exp)) # Binary window function for galaxies to mark dead pixels
    W_g01[n_g_exp==0] = 0
    ### Calculate FKP weigts:
    W_g = n_g_exp/np.sum(n_g_exp) # normalised window function for FKP weight calculation
    P0 = 1000
    nbar = np.sum(n_g)/(lx*ly*lz) # Calculate number density inside survey footprint
    w_g = 1/(1 + W_g*(nx*ny*nz)*nbar*P0)
    w_g[W_g01==0] = 0 # zero weight for dead pixels
    return dT_MK,n_g,w_HI,W_HI,w_g,n_g_exp,W_g01,nbar,dims,ra,dec,nu,freqmask


def load_map(data_path, map_name='clean_map'):
#### Yi-Chao's adapted function for reading his optical map data
    with h5.File(data_path, 'r') as f:
        #print(f.keys())
        _map = f[map_name][:]
        if map_name=='mask': return _map
        info = {}
        for key, value in f[map_name].attrs.items():
            # Value errors from below safe_eval are caused for different users
            #   so including both possible working variants.
            try: info[key] = safe_eval(value)
            except ValueError:
                info[key] = safe_eval(value.decode("utf-8"))
        _l = _map.shape
        axes = []
        for ii, ax in enumerate(info['axes']):
            axes.append( info[ax + '_delta'] * (np.arange(_l[ii]) - _l[ii] // 2)\
                    + info[ax + '_centre'])
    _map = np.swapaxes(_map,0,2) # [z,ra,dec] -> [dec,ra,z]
    _map = np.swapaxes(_map,0,1) # [dec,ra,z] -> [ra,dec,z]
    return _map, axes

def make_noise_factorizable(noise, weight_prior=1.e3):
    r"""Convert noise diag such that the factor into a function a
    frequency times a function of pixel by taking means over the original
    weights.

    input noise_diag;
    output weight

    weight_prior used to be 10^-30 before prior applied

    ### SC: Provided by Yi-Chao via his GitHub:
    https://github.com/meerklass/meerKAT_sim/blob/3ed8ec5ffe11cbcc010a1fdf9f739930ec2138d4/meerKAT_sim/ps/fgrm.py#L259
    Details of it are discussed in Switzer+13: https://academic.oup.com/mnrasl/article/434/1/L46/1165809
    """

    ## Swith back axes order to Yi-Chao's convention:
    noise = np.swapaxes(noise,0,2) # [ra,dec,z] -> [z,dec,ra]
    noise = np.swapaxes(noise,1,2) # [z,dec,ra] -> [z,ra,dec]

    #noise[noise < weight_prior] = 1.e-30
    #noise = 1. / noise
    #noise[noise < 5.e-5] = 0.
    noise[noise > 1./weight_prior] = 1.e30
    noise = ma.array(noise)
    # Get the freqency averaged noise per pixel.  Propagate mask in any
    # frequency to all frequencies.
    for noise_index in range(ma.shape(noise)[0]):
        if np.all(noise[noise_index, ...] > 1.e20):
            noise[noise_index, ...] = ma.masked
    noise_fmean = ma.mean(noise, 0)
    noise_fmean[noise_fmean > 1.e20] = ma.masked
    # Get the pixel averaged noise in each frequency.
    noise[noise > 1.e20] = ma.masked
    noise /= noise_fmean
    noise_pmean = ma.mean(ma.mean(noise, 1), 1)
    # Combine.
    noise = noise_pmean[:, None, None] * noise_fmean[None, :, :]
    #noise[noise == 0] = np.inf
    #noise[noise==0] = ma.masked
    #noise[noise==0] = 1e-30

    #weight = (1 / noise)
    weight = np.zeros(np.shape(noise))
    weight[noise!=0] = 1/noise[noise!=0]
    #weight[noise==1e-30] = 0

    #weight = weight.filled(0)
    cut_l  = np.percentile(weight, 10)
    cut_h = np.percentile(weight, 80)
    weight[weight<cut_l] = cut_l
    weight[weight>cut_h] = cut_h

    ## Switch back axes order to Steves's convention:
    weight = np.swapaxes(weight,0,2) # [z,ra,dec] -> [dec,ra,z]
    weight = np.swapaxes(weight,0,1) # [dec,ra,z] -> [ra,dec,z]
    #weight = weight.filled(0)
    return weight
