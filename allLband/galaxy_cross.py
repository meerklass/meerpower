import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
import sys
import os
sys.path.insert(1, '/idia/projects/hi_im/meerpower/meerpower')
import Init
import plot

def RunPipeline(survey,gal_cat,N_fg,gamma=1.4,kcuts=None,do2DTF=False,doHIauto=False,tukey_alpha=0.1):
    '''
    Use for looping over full pipeline with different choices of inputs for purposes
    of transfer function building. Input choices from below:
    # survey = '2019' or '2021'
    # gal_cat = 'wigglez', 'cmass', or 'gama'
    # N_fg = int (the number of PCA components to remove)
    # gamma = float or None (resmoothing parameter)
    # kcuts = [kperpmin,kparamin,kperpmax,kparamax] or None (exclude areas of k-space from spherical average)]
    '''
    # Load data and run some pre-processing steps:
    doMock = False # Set True to load mock data for consistency checks

    if survey=='2019':
        filestem = '/idia/projects/hi_im/raw_vis/katcali_output/level6_output/p0.3d/p0.3d_sigma2.5_iter2/'
        map_file = filestem + 'Nscan366_Tsky_cube_p0.3d_sigma2.5_iter2.fits'
        numin,numax = 971,1023.2
    if survey=='2021':
        filestem = '/idia/users/jywang/MeerKLASS/calibration2021/level6/0.3/sigma4_count40/re_cali1_round5/'
        map_file = filestem + 'Nscan961_Tsky_cube_p0.3d_sigma4.0_iter2.fits'
        numin,numax = 971,1023.8 # default setting in Init.ReadIn()
    MKmap,w_HI,W_HI,counts_HI,dims,ra,dec,nu,wproj = Init.ReadIn(map_file,numin=numin,numax=numax)
    if doMock==True:
        mockindx = np.random.randint(100)
        mockindx = 1
        MKmap_mock = np.load('/idia/projects/hi_im/meerpower/'+survey+'Lband/mocks/dT_HI_p0.3d_wBeam_%s.npy'%mockindx)
    nx,ny,nz = np.shape(MKmap)

    ### Remove incomplete LoS pixels from maps:
    MKmap,w_HI,W_HI,counts_HI = Init.FilterIncompleteLoS(MKmap,w_HI,W_HI,counts_HI)

    ### IM weights (averaging of counts along LoS so not to increase rank of the map for FG cleaning):
    w_HI = np.repeat(np.mean(counts_HI,2)[:, :, np.newaxis], nz, axis=2)

    # Initialise some fiducial cosmology and survey parameters:
    import cosmo
    nu_21cm = 1420.405751 #MHz
    #from astropy.cosmology import Planck15 as cosmo_astropy
    zeff = (nu_21cm/np.median(nu)) - 1 # Effective redshift - defined as redshift of median frequency
    zmin = (nu_21cm/np.max(nu)) - 1 # Minimum redshift of band
    zmax = (nu_21cm/np.min(nu)) - 1 # Maximum redshift of band
    cosmo.SetCosmology(builtincosmo='Planck18',z=zeff,UseCLASS=True)
    Pmod = cosmo.GetModelPk(zeff,kmax=25,UseCLASS=True) # high-kmax needed for large k-modes in NGP alisasing correction
    f = cosmo.f(zeff)
    sig_v = 0
    b_HI = 1.5
    OmegaHIbHI = 0.85e-3 # MKxWiggleZ constraint
    OmegaHI = OmegaHIbHI/b_HI
    import HItools
    import telescope
    Tbar = HItools.Tbar(zeff,OmegaHI)
    r = 0.9 # cross-correlation coefficient
    D_dish = 13.5 # Dish-diameter [metres]
    theta_FWHM,R_beam = telescope.getbeampars(D_dish,np.median(nu))
    gamma = 1.4 # resmoothing factor - set = None to have no resmoothing
    #gamma = None

    ### Map resmoothing:
    MKmap_unsmoothed = np.copy(MKmap)
    if gamma is not None:
        w_HI_orig = np.copy(w_HI)
        MKmap,w_HI = telescope.weighted_reconvolve(MKmap,w_HI_orig,W_HI,ra,dec,nu,D_dish,gamma=gamma)
        if doMock==True: MKmap_mock,null = telescope.weighted_reconvolve(MKmap_mock,w_HI_orig,W_HI,ra,dec,nu,D_dish,gamma=gamma)

    ### Trim map edges:
    doTrim = True
    if doTrim==True:
        if survey=='2019':
            #raminMK,ramaxMK = 152,173
            #decminMK,decmaxMK = -1,8
            raminMK,ramaxMK = 149,190
            decminMK,decmaxMK = -5,20

        if survey=='2021':
            raminMK,ramaxMK = 334,357
            #decminMK,decmaxMK = -35,-26.5
            decminMK,decmaxMK = np.min(dec[np.mean(W_HI,2)>0]),np.max(dec[np.mean(W_HI,2)>0])
        ### Before trimming map, show contour of trimmed area:
        MKmap_untrim,W_HI_untrim = np.copy(MKmap),np.copy(W_HI)
        MKmap,w_HI,W_HI,counts_HI = Init.MapTrim(ra,dec,MKmap,w_HI,W_HI,counts_HI,ramin=raminMK,ramax=ramaxMK,decmin=decminMK,decmax=decmaxMK)

        if survey=='2019':
            cornercut_lim = 146 # set low to turn off
            cornercut = ra - dec < cornercut_lim
            MKmap[cornercut],w_HI[cornercut],W_HI[cornercut],counts_HI[cornercut] = 0,0,0,0

    # Spectral analysis for possible frequency channel flagging:
    #Also remove some corners/extreme temp values

    MKmap_flag,w_HI_flag,W_HI_flag = np.copy(MKmap),np.copy(w_HI),np.copy(W_HI)

    if survey=='2019':
        extreme_temp_LoS = np.zeros(np.shape(ra))
        extreme_temp_LoS[MKmap[:,:,0]>3530] = 1
        extreme_temp_LoS[MKmap[:,:,0]<3100] = 1
        MKmap_flag[extreme_temp_LoS==1] = 0
        w_HI_flag[extreme_temp_LoS==1] = 0
        W_HI_flag[extreme_temp_LoS==1] = 0

    import model
    nra,ndec = np.shape(ra)
    offsets = np.zeros((nra,ndec,len(nu)))
    for i in range(nra):
        for j in range(ndec):
            if W_HI_flag[i,j,0]==0: continue
            poly = model.FitPolynomial(nu,MKmap_flag[i,j,:],n=2)
            offsets[i,j,:] = np.abs((MKmap_flag[i,j,:] - poly)/MKmap_flag[i,j,:])
    offsets = 100*np.mean(offsets,axis=(0,1))

    if survey=='2019': offsetcut = 0.029 # Set to zero for no additional flagging
    #if survey=='2019': offsetcut = None # Set to None for no additional flagging
    if survey=='2021': offsetcut = None

    if offsetcut is None: flagindx = []
    else: flagindx = np.where(offsets>offsetcut)[0]

    flags = np.full(nz,False)
    flags[flagindx] = True

    MKmap_flag[:,:,flags] = 0
    w_HI_flag[:,:,flags] = 0
    W_HI_flag[:,:,flags] = 0

    # Principal component analysis:
    import foreground # PCA clean and transfer function calculations performed here
    import model

    # Foreground clean:
    MKmap,w_HI,W_HI = np.copy(MKmap_flag),np.copy(w_HI_flag),np.copy(W_HI_flag) # Propagate flagged maps for rest of analysis
    if doMock==False:
        MKmap_clean = foreground.PCAclean(MKmap,N_fg=N_fg,W=W_HI,w=w_HI,MeanCentre=True) # weights included in covariance calculation
    if doMock==True:
        MKmap_mock[W_HI==0] = 0 # apply same flags, trims, cuts as data
        #MKmap_clean = MKmap_mock
        MKmap_clean = foreground.PCAclean(MKmap + MKmap_mock ,N_fg=N_fg,W=W_HI,w=w_HI,MeanCentre=True) # weights included in covariance calculation

    W_HI_untrim,w_HI_untrim = np.copy(W_HI),np.copy(w_HI)
    if gal_cat=='wigglez': # obtained from min/max of wigglez catalogue
        ramin_gal,ramax_gal = 152.906631, 172.099625
        decmin_gal,decmax_gal = -1.527391, 8.094599
    if gal_cat=='gama':
        ramin_gal,ramax_gal = 339,351
        decmin_gal,decmax_gal = -35,-30
    '''
    if gal_cat=='wigglez' or gal_cat=='gama':
        MKmap_clean,w_HI,W_HI,counts_HI = Init.MapTrim(ra,dec,MKmap_clean,w_HI,W_HI,counts_HI,ramin=ramin_gal,ramax=ramax_gal,decmin=decmin_gal,decmax=decmax_gal)
    '''
    if gal_cat=='wigglez':
        MKmap_clean,w_HI,W_HI,counts_HI = Init.MapTrim(ra,dec,MKmap_clean,w_HI,W_HI,counts_HI,ramin=ramin_gal,ramax=ramax_gal,decmin=decmin_gal,decmax=decmax_gal)
    if gal_cat=='gama':
        w_HI,W_HI,counts_HI = Init.MapTrim(ra,dec,w_HI,W_HI,counts_HI,ramin=ramin_gal,ramax=ramax_gal,decmin=decmin_gal,decmax=decmax_gal)
        # Re-do clean with trimmed mask:
        MKmap_clean = foreground.PCAclean(MKmap,N_fg=N_fg,W=W_HI,w=w_HI,MeanCentre=True) # weights included in covariance calculation


    # Read-in overlapping galaxy survey:
    from astropy.io import fits
    if survey=='2019':
        if gal_cat=='wigglez':
            if doMock==False: # Read-in WiggleZ galaxies (provided by Laura):
                galcat = np.genfromtxt('/users/scunnington/MeerKAT/LauraShare/wigglez_reg11hrS_z0pt30_0pt50/reg11data.dat', skip_header=1)
                ra_g,dec_g,z_g = galcat[:,0],galcat[:,1],galcat[:,2]
            if doMock==True: ra_g,dec_g,z_g = np.load('/idia/projects/hi_im/meerpower/2019Lband/mocks/mockWiggleZcat_%s.npy'%mockindx)
            z_Lband = (z_g>zmin) & (z_g<zmax) # Cut redshift to MeerKAT IM range:
            ra_g,dec_g,z_g = ra_g[z_Lband],dec_g[z_Lband],z_g[z_Lband]
        if gal_cat=='cmass':
            if doMock==False: # Read-in BOSS-CMASS galaxies (in Yi-Chao's ilifu folder - also publically available from: https://data.sdss.org/sas/dr12/boss/lss):
                filename = '/idia/users/ycli/SDSS/dr12/galaxy_DR12v5_CMASSLOWZTOT_North.fits.gz'
                hdu = fits.open(filename)
                ra_g,dec_g,z_g = hdu[1].data['RA'],hdu[1].data['DEC'],hdu[1].data['Z']
            if doMock==True: ra_g,dec_g,z_g = np.load('/idia/projects/hi_im/meerpower/2019Lband/mocks/mockCMASScat_%s.npy'%mockindx)
            ra_g,dec_g,z_g = Init.pre_process_2019Lband_CMASS_galaxies(ra_g,dec_g,z_g,ra,dec,zmin,zmax,W_HI)

    if survey=='2021':
        if doMock==False: # Read-in GAMA galaxies:
            Fits = '/idia/projects/hi_im/GAMA_DR4/G23TilingCatv11.fits'
            hdu = fits.open(Fits)
            ra_g,dec_g,z_g = hdu[1].data['RA'],hdu[1].data['DEC'],hdu[1].data['Z']
        if doMock==True: ra_g,dec_g,z_g = np.load('/idia/projects/hi_im/meerpower/2021Lband/mocks/mockGAMAcat_%s.npy'%mockindx)
        # Remove galaxies outside bulk GAMA footprint so they don't bias the simple binary selection function
        GAMAcutmask = (ra_g>ramin_gal) & (ra_g<ramax_gal) & (dec_g>decmin_gal) & (dec_g<decmax_gal) & (z_g>zmin) & (z_g<zmax)
        ra_g,dec_g,z_g = ra_g[GAMAcutmask],dec_g[GAMAcutmask],z_g[GAMAcutmask]

    print('Number of overlapping ', gal_cat,' galaxies: ', str(len(ra_g)))

    # Assign galaxy bias:
    if gal_cat=='wigglez': b_g = np.sqrt(0.83) # for WiggleZ at z_eff=0.41 - from https://arxiv.org/pdf/1104.2948.pdf [pg.9 rhs second quantity]
    if gal_cat=='cmass':b_g = 1.85 # Mentioned in https://arxiv.org/pdf/1607.03155.pdf
    if gal_cat=='gama': b_g = 2.35 # tuned by eye in GAMA auto-corr

    # Gridding maps and galaxies to Cartesian field:
    import grid # use this for going from (ra,dec,freq)->(x,y,z) Cartesian-comoving grid
    cell2vox_factor = 1.5 # increase for lower resolution FFT Cartesian grid
    Np = 5 # number of Monte-Carlo sampling particles per map voxel used in regridding
    window = 'ngp'
    compensate = True
    interlace = False
    nxmap,nymap,nzmap = np.shape(MKmap)
    ndim_rg = int(nxmap/cell2vox_factor),int(nymap/cell2vox_factor),int(nzmap/cell2vox_factor)
    nzcell2vox = int(nzmap/cell2vox_factor)
    if nzcell2vox % 2 != 0: nzcell2vox += 1 # Ensure z-dimension is even for FFT purposes
    ndim_rg = int(nxmap/cell2vox_factor),int(nymap/cell2vox_factor),nzcell2vox
    dims_rg,dims0_rg = grid.comoving_dims(ra,dec,nu,wproj,ndim_rg,W=W_HI_untrim,dobuffer=True) # dimensions of Cartesian grid for FFT
    lx,ly,lz,nx_rg,ny_rg,nz_rg = dims_rg

    # Regrid cleaned map, IM mask and weights to Cartesian field:
    ra_p,dec_p,nu_p,pixvals = grid.SkyPixelParticles(ra,dec,nu,wproj,map=MKmap_clean,W=W_HI,Np=Np)
    xp,yp,zp = grid.SkyCoordtoCartesian(ra_p,dec_p,HItools.Freq2Red(nu_p),ramean_arr=ra,decmean_arr=dec,doTile=False)
    MKmap_clean_rg,null,null = grid.mesh(xp,yp,zp,pixvals,dims0_rg,window,compensate,interlace,verbose=False)
    ra_p,dec_p,nu_p,pixvals = grid.SkyPixelParticles(ra,dec,nu,wproj,map=W_HI,W=W_HI,Np=Np)
    xp,yp,zp = grid.SkyCoordtoCartesian(ra_p,dec_p,HItools.Freq2Red(nu_p),ramean_arr=ra,decmean_arr=dec,doTile=False)
    W_HI_rg,null,null = grid.mesh(xp,yp,zp,pixvals,dims0_rg,window='ngp',compensate=False,interlace=False,verbose=False)
    ra_p,dec_p,nu_p,pixvals = grid.SkyPixelParticles(ra,dec,nu,wproj,map=w_HI,W=W_HI,Np=Np)
    xp,yp,zp = grid.SkyCoordtoCartesian(ra_p,dec_p,HItools.Freq2Red(nu_p),ramean_arr=ra,decmean_arr=dec,doTile=False)
    w_HI_rg = grid.mesh(xp,yp,zp,pixvals,dims0_rg,window='ngp',compensate=False,interlace=False,verbose=False)[0]

    # Grid galaxies straight to Cartesian field:
    xp,yp,zp = grid.SkyCoordtoCartesian(ra_g,dec_g,z_g,ramean_arr=ra,decmean_arr=dec,doTile=False)
    n_g_rg = grid.mesh(xp,yp,zp,dims=dims0_rg,window=window,compensate=compensate,interlace=interlace,verbose=False)[0]

    # Construct galaxy selection function:
    if survey=='2019':
        if gal_cat=='wigglez': # grid WiggleZ randoms straight to Cartesian field for survey selection:
            BuildSelFunc = False
            if BuildSelFunc==True:
                nrand = 1000 # number of WiggleZ random catalogues to use in selection function (max is 1000)
                W_g_rg = np.zeros(np.shape(n_g_rg))
                for i in range(1,nrand):
                    plot.ProgressBar(i,nrand)
                    galcat = np.genfromtxt( '/users/scunnington/MeerKAT/LauraShare/wigglez_reg11hrS_z0pt30_0pt50/reg11rand%s.dat' %'{:04d}'.format(i), skip_header=1)
                    ra_g_rand,dec_g_rand,z_g_rand = galcat[:,0],galcat[:,1],galcat[:,2]
                    z_Lband = (z_g_rand>zmin) & (z_g_rand<zmax) # Cut redshift to MeerKAT IM range:
                    ra_g_rand,dec_g_rand,z_g_rand = ra_g_rand[z_Lband],dec_g_rand[z_Lband],z_g_rand[z_Lband]
                    xp,yp,zp = grid.SkyCoordtoCartesian(ra_g_rand,dec_g_rand,z_g_rand,ramean_arr=ra,decmean_arr=dec,doTile=False)
                    W_g_rg += grid.mesh(xp,yp,zp,dims=dims0_rg,window='ngp',compensate=False,interlace=False,verbose=False)[0]
                W_g_rg /= nrand
                np.save('/idia/projects/hi_im/meerpower/2019Lband/wigglez/data/wiggleZ_stackedrandoms_cell2voxfactor=%s.npy'%cell2vox_factor,W_g_rg)
            W_g_rg = np.load('/idia/projects/hi_im/meerpower/2019Lband/wigglez/data/wiggleZ_stackedrandoms_cell2voxfactor=%s.npy'%cell2vox_factor)

        if gal_cat=='cmass':
        # Data obtained from untarrting DR12 file at: https://data.sdss.org/sas/dr12/boss/lss/dr12_multidark_patchy_mocks/Patchy-Mocks-DR12NGC-COMPSAM_V6C.tar.gz
            BuildSelFunc = False
            if BuildSelFunc==True:
                nrand = 2048 # number of WiggleZ random catalogues to use in selection function (max is 1000)
                W_g_rg = np.zeros(np.shape(n_g_rg))
                for i in range(1,nrand):
                    plot.ProgressBar(i,nrand)
                    galcat = np.genfromtxt( '/idia/projects/hi_im/meerpower/2019Lband/cmass/sdss/Patchy-Mocks-DR12NGC-COMPSAM_V6C_%s.dat' %'{:04d}'.format(i+1), skip_header=1)
                    ra_g_rand,dec_g_rand,z_g_rand = galcat[:,0],galcat[:,1],galcat[:,2]
                    ra_g_rand,dec_g_rand,z_g_rand = Init.pre_process_2019Lband_CMASS_galaxies(ra_g_rand,dec_g_rand,z_g_rand,ra,dec,zmin,zmax,W_HI)
                    xp,yp,zp = grid.SkyCoordtoCartesian(ra_g_rand,dec_g_rand,z_g_rand,ramean_arr=ra,decmean_arr=dec,doTile=False)
                    W_g_rg += grid.mesh(xp,yp,zp,dims=dims0_rg,window='ngp',compensate=False,interlace=False,verbose=False)[0]
                W_g_rg /= nrand
                np.save('/idia/projects/hi_im/meerpower/2019Lband/cmass/data/cmass_stackedrandoms_cell2voxfactor=%s.npy'%cell2vox_factor,W_g_rg)
            W_g_rg = np.load('/idia/projects/hi_im/meerpower/2019Lband/cmass/data/cmass_stackedrandoms_cell2voxfactor=%s.npy'%cell2vox_factor)

    if survey=='2021': # grid uncut pixels to obtain binary mask in comoving space in absence of GAMA mocks for survey selection:
        ra_p,dec_p,nu_p = grid.SkyPixelParticles(ra,dec,nu,wproj,Np=Np)
        '''
        if doTrim==True:
            MKcutmask = (ra_p>ramin_gal) & (ra_p<ramax_gal) & (dec_p>decmin_gal) & (dec_p<decmax_gal)
            xp,yp,zp = grid.SkyCoordtoCartesian(ra_p[MKcutmask],dec_p[MKcutmask],HItools.Freq2Red(nu_p[MKcutmask]),ramean_arr=ra,decmean_arr=dec,doTile=False)
            null,W_HI_rg,counts = grid.mesh(xp,yp,zp,dims=dims0_rg,window='ngp',compensate=False,interlace=False,verbose=False)
        '''
        GAMAcutmask = (ra_p>ramin_gal) & (ra_p<ramax_gal) & (dec_p>decmin_gal) & (dec_p<decmax_gal)
        xp,yp,zp = grid.SkyCoordtoCartesian(ra_p[GAMAcutmask],dec_p[GAMAcutmask],HItools.Freq2Red(nu_p[GAMAcutmask]),ramean_arr=ra,decmean_arr=dec,doTile=False)
        null,W_g_rg,null = grid.mesh(xp,yp,zp,dims=dims0_rg,window='ngp',compensate=False,interlace=False,verbose=False)

    # Calculate FKP weigts:
    '''
    W_g01_rg = np.ones(np.shape(W_g_rg)) # Binary window function for galaxies to mark dead pixels
    W_g01_rg[W_g_rg==0] = 0
    W_g_rg = W_g_rg/np.sum(W_g_rg) # normalised window function for FKP weight calculation
    P0 = 5000 # at k~0.1
    nbar = np.sum(n_g_rg)/(lx*ly*lz) # Calculate number density inside survey footprint
    w_g_rg = 1/(1 + W_g_rg*(nx*ny*nz)*nbar*P0)
    w_g_rg[W_g01_rg==0] = 0 # zero weight for dead pixels
    '''
    #w_g_rg = np.ones(np.shape(W_g_rg))
    w_g_rg = np.copy(W_g_rg)

    MKmap_clean_rg_notaper,w_HI_rg_notaper,W_HI_rg_notaper = np.copy(MKmap_clean_rg),np.copy(w_HI_rg),np.copy(W_HI_rg)
    n_g_rg_notaper,w_g_rg_notaper,W_g_rg_notaper = np.copy(n_g_rg),np.copy(w_g_rg),np.copy(W_g_rg)

    # Footprint tapering/apodisation:
    ### Chose no taper:
    #taper_HI,taper_g = 1,1

    ### Chose to use Blackman window function along z direction as taper:
    blackman = np.reshape( np.tile(np.blackman(nz_rg), (nx_rg,ny_rg)) , (nx_rg,ny_rg,nz_rg) ) # Blackman function along every LoS
    tukey = np.reshape( np.tile(signal.windows.tukey(nz_rg, alpha=tukey_alpha), (nx_rg,ny_rg)) , (nx_rg,ny_rg,nz_rg) ) # Blackman function along every LoS


    #taper_HI = blackman
    #taper_g = blackman
    taper_HI = tukey
    #taper_g = tukey
    #taper_HI = 1
    taper_g = 1


    # Multiply tapering windows by all fields that undergo Fourier transforms:
    #MKmap_clean_rg,w_HI_rg,W_HI_rg = taper_HI*MKmap_clean_rg_notaper,taper_HI*w_HI_rg_notaper,taper_HI*W_HI_rg_notaper
    #n_g_rg,W_g_rg,w_g_rg = taper_g*n_g_rg_notaper,taper_g*W_g_rg_notaper,taper_g*w_g_rg_notaper
    w_HI_rg = taper_HI*w_HI_rg_notaper
    w_g_rg = taper_g*w_g_rg_notaper

    # Power spectrum measurement and modelling (without signal loss correction):
    import power
    import model
    nkbin = 16
    kmin,kmax = 0.07,0.3
    kbins = np.linspace(kmin,kmax,nkbin+1) # k-bin edges [using linear binning]

    import model
    sig_v = 0
    dpix = 0.3
    d_c = cosmo.d_com(HItools.Freq2Red(np.min(nu)))
    s_pix = d_c * np.radians(dpix)
    s_para = np.mean( cosmo.d_com(HItools.Freq2Red(nu[:-1])) - cosmo.d_com(HItools.Freq2Red(nu[1:])) )

    ### Galaxy Auto-power (can use to constrain bias and use for analytical errors):
    W_g_rg /= np.max(W_g_rg)


    # Calculate power specs (to get k's for TF):
    if doHIauto==False: Pk_gHI,k,nmodes = power.Pk(MKmap_clean_rg,n_g_rg,dims_rg,kbins,corrtype='Cross',w1=w_HI_rg,w2=w_g_rg,W1=W_HI_rg,W2=W_g_rg,kcuts=kcuts)
    if doHIauto==True: Pk_HI,k,nmodes = power.Pk(MKmap_clean_rg,MKmap_clean_rg,dims_rg,kbins,corrtype='HIauto',w1=w_HI_rg,w2=w_HI_rg,W1=W_HI_rg,W2=W_HI_rg)

    LoadTF = False
    Nmock = 500
    if gamma is None: gamma_label = 'None'
    else: gamma_label = str(gamma)
    if kcuts is None: kcuts_label = 'nokcuts'
    else: kcuts_label = 'withkcuts'
    mockfilepath_HI = '/idia/projects/hi_im/meerpower/'+survey+'Lband/mocks/dT_HI_p0.3d_wBeam'
    if gal_cat=='wigglez': mockfilepath_g = '/idia/projects/hi_im/meerpower/2019Lband/mocks/mockWiggleZcat'
    if gal_cat=='cmass': mockfilepath_g = '/idia/projects/hi_im/meerpower/2019Lband/mocks/mockCMASScat'
    if gal_cat=='gama': mockfilepath_g = '/idia/projects/hi_im/meerpower/2021Lband/mocks/mockGAMAcat'

    if do2DTF==False:
        if doHIauto==False:
            #TFfile = '/idia/projects/hi_im/meerpower/'+survey+'Lband/'+gal_cat+'/TFdata/T_Nfg=%s_gamma=%s_'%(N_fg,gamma_label)+kcuts_label
            TFfile = '/idia/projects/hi_im/meerpower/'+survey+'Lband/'+gal_cat+'/TFdata/T_Nfg=%s_gamma=%s_'%(N_fg,gamma_label)+kcuts_label+'_tukeyHI=%s'%tukey_alpha
            T_wsub_i, T_nosub_i,k  = foreground.TransferFunction(MKmap_unsmoothed,Nmock,N_fg,'Cross',kbins,k,TFfile,ra,dec,nu,wproj,dims0_rg,
                                                        Np,window,compensate,interlace,mockfilepath_HI,mockfilepath_g,gal_cat=gal_cat,
                                                        gamma=gamma,D_dish=D_dish,w_HI=w_HI,W_HI=W_HI,doWeightFGclean=True,PCAMeanCentre=True,
                                                        w_HI_rg=w_HI_rg,W_HI_rg=W_HI_rg,w_g_rg=w_g_rg,W_g_rg=W_g_rg,kcuts=kcuts,
                                                        taper_HI=taper_HI,taper_g=taper_g,LoadTF=LoadTF)
        if doHIauto==True:
            TFfile = '/idia/projects/hi_im/meerpower/'+survey+'Lband/'+gal_cat+'/TFdata/T_HIauto_Nfg=%s_gamma=%s_'%(N_fg,gamma_label)+kcuts_label
            T_wsub_i, T_nosub_i,k  = foreground.TransferFunction(MKmap_unsmoothed,Nmock,N_fg,'HIauto',kbins,k,TFfile,ra,dec,nu,wproj,dims0_rg,
                                                        Np,window,compensate,interlace,mockfilepath_HI,mockfilepath_g,gal_cat=gal_cat,
                                                        gamma=gamma,D_dish=D_dish,w_HI=w_HI,W_HI=W_HI,doWeightFGclean=True,PCAMeanCentre=True,
                                                        w_HI_rg=w_HI_rg,W_HI_rg=W_HI_rg,w_g_rg=w_g_rg,W_g_rg=W_g_rg,kcuts=kcuts,
                                                        taper_HI=taper_HI,taper_g=taper_g,LoadTF=LoadTF)
    if do2DTF==True:
        kperpbins = np.linspace(0.008,0.3,34)
        kparabins = np.linspace(0.003,0.6,22)
        if doHIauto==False:
            TFfile = '/idia/projects/hi_im/meerpower/'+survey+'Lband/'+gal_cat+'/TFdata/T2D_Nfg=%s_gamma=%s_'%(N_fg,gamma_label)
            T2d_wsub_i, T2d_nosub_i,k2d  = foreground.TransferFunction(MKmap_unsmoothed,Nmock,N_fg,'Cross',kbins,k,TFfile,ra,dec,nu,wproj,dims0_rg,
                                                        Np,window,compensate,interlace,mockfilepath_HI,mockfilepath_g,gal_cat=gal_cat,
                                                        gamma=gamma,D_dish=D_dish,w_HI=w_HI,W_HI=W_HI,doWeightFGclean=True,PCAMeanCentre=True,
                                                        w_HI_rg=w_HI_rg,W_HI_rg=W_HI_rg,w_g_rg=w_g_rg,W_g_rg=W_g_rg,kcuts=kcuts,
                                                        taper_HI=taper_HI,taper_g=taper_g,LoadTF=LoadTF,TF2D=True,kperpbins=kperpbins,kparabins=kparabins)
        if doHIauto==True:
            TFfile = '/idia/projects/hi_im/meerpower/'+survey+'Lband/'+gal_cat+'/TFdata/T2D_HIauto_Nfg=%s_gamma=%s_'%(N_fg,gamma_label)
            T2d_wsub_i, T2d_nosub_i,k2d  = foreground.TransferFunction(MKmap_unsmoothed,Nmock,N_fg,'HIauto',kbins,k,TFfile,ra,dec,nu,wproj,dims0_rg,
                                                        Np,window,compensate,interlace,mockfilepath_HI,mockfilepath_g,gal_cat=gal_cat,
                                                        gamma=gamma,D_dish=D_dish,w_HI=w_HI,W_HI=W_HI,doWeightFGclean=True,PCAMeanCentre=True,
                                                        w_HI_rg=w_HI_rg,W_HI_rg=W_HI_rg,w_g_rg=w_g_rg,W_g_rg=W_g_rg,kcuts=kcuts,
                                                        taper_HI=taper_HI,taper_g=taper_g,LoadTF=LoadTF,TF2D=True,kperpbins=kperpbins,kparabins=kparabins)



do2DTF = False
doHIauto = False

#survey = '2019'
#gal_cat = 'wigglez'
#gal_cat = 'cmass'

survey = '2021'
gal_cat = 'gama'

if gal_cat=='cmass' or gal_cat=='wigglez': N_fgs = [12,10,8,15,6]
if gal_cat=='gama': N_fgs = [8,12,6,10,15]

N_fgs = [10,8,12,6,5,7,9,11,13,14,15,16,17,18,19,20]

#if gal_cat=='cmass': kcuts = [0.052,0.031,0.15,None] #[kperpmin,kparamin,kperpmax,kparamax] (exclude areas of k-space from spherical average)
#if gal_cat=='gama': kcuts = [0.052,0.031,0.175,None] #[kperpmin,kparamin,kperpmax,kparamax] (exclude areas of k-space from spherical average)
kcuts = [0.052,0.031,0.175,None] #[kperpmin,kparamin,kperpmax,kparamax] (exclude areas of k-space from spherical average)

#kcuts = None

'''
tukey_alpha = 0.1
for i in range(len(N_fgs)):
    RunPipeline(survey,gal_cat,N_fgs[i],kcuts=kcuts,do2DTF=do2DTF,doHIauto=doHIauto)
'''

N_fg = 10
tukey_alphas = [0.5,0.1,0.2,0.8,1]
for i in range(len(tukey_alphas)):
    RunPipeline(survey,gal_cat,N_fg,kcuts=kcuts,do2DTF=do2DTF,doHIauto=doHIauto,tukey_alpha=tukey_alphas[i])
