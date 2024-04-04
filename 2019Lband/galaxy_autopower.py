import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
local = True
if local==True: sys.path.insert(1, '/Users/user/Documents/MeerKAT/meerpower/meerpower')
if local==False: sys.path.insert(1, '/idia/projects/hi_im/meerpower/meerpower')
import Init
import plot

doMock = False
survey = '2019'
#gal_cat = 'wigglez'
gal_cat = 'cmass'

if local==True: filestem = '/Users/user/Documents/MeerKAT/meerpower/localdata/2019/'
if local==False: filestem = '/idia/projects/hi_im/raw_vis/katcali_output/level6_output/p0.3d/p0.3d_sigma2.5_iter2/'
map_file = filestem + 'Nscan366_Tsky_cube_p0.3d_sigma2.5_iter2.fits'
numin,numax = 971,1023.2
MKmap,w_HI,W_HI,counts_HI,dims,ra,dec,nu,wproj = Init.ReadIn(map_file,numin=numin,numax=numax)
if doMock==True:
    if local==True: mockindx = 0
    if local==False: mockindx = np.random.randint(100)
    if local==True: MKmap_mock = MKmap
    if local==False: MKmap_mock = np.load('/idia/projects/hi_im/meerpower/'+survey+'Lband/mocks/dT_HI_p0.3d_wBeam_%s.npy'%mockindx)
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
Pmod = cosmo.GetModelPk(zeff,kmax=25,UseCLASS=True,NonLinear=True) # high-kmax needed for large k-modes in NGP alisasing correction
f = cosmo.f(zeff)
sig_v = 300
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

### Trim map edges:
doTrim = True
if doTrim==True:
    #raminMK,ramaxMK = 152,173
    #decminMK,decmaxMK = -1,8
    raminMK,ramaxMK = 149,190
    decminMK,decmaxMK = -5,20
    MKmap,w_HI,W_HI,counts_HI = Init.MapTrim(ra,dec,MKmap,w_HI,W_HI,counts_HI,ramin=raminMK,ramax=ramaxMK,decmin=decminMK,decmax=decmaxMK)
    cornercut_lim = 146 # set low to turn off
    cornercut = ra - dec < cornercut_lim
    MKmap[cornercut],w_HI[cornercut],W_HI[cornercut],counts_HI[cornercut] = 0,0,0,0

MKmap_flag,w_HI_flag,W_HI_flag = np.copy(MKmap),np.copy(w_HI),np.copy(W_HI)
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
offsetcut = 0.029 # Set to zero for no additional flagging
if offsetcut is None: flagindx = []
else: flagindx = np.where(offsets>offsetcut)[0]
flags = np.full(nz,False)
flags[flagindx] = True
MKmap_flag[:,:,flags] = 0
w_HI_flag[:,:,flags] = 0
W_HI_flag[:,:,flags] = 0
MKmap,w_HI,W_HI = np.copy(MKmap_flag),np.copy(w_HI_flag),np.copy(W_HI_flag) # Propagate flagged maps for rest of analysis

if doMock==False:
    if gal_cat=='wigglez':
        # Read-in WiggleZ galaxies (provided by Laura):
        if local==True: galcat = np.genfromtxt('/Users/user/Documents/MeerKAT/LauraShare/wigglez_reg11hrS_z0pt30_0pt50/reg11data.dat', skip_header=1)
        if local==False: galcat = np.genfromtxt('/users/scunnington/MeerKAT/LauraShare/wigglez_reg11hrS_z0pt30_0pt50/reg11data.dat', skip_header=1)
        ra_g,dec_g,z_g = galcat[:,0],galcat[:,1],galcat[:,2]
        ### Cut redshift to MeerKAT IM range:
        z_Lband = (z_g>zmin) & (z_g<zmax)
        ra_g = ra_g[z_Lband]
        dec_g = dec_g[z_Lband]
        z_g = z_g[z_Lband]
    if gal_cat=='cmass': # Read-in BOSS-CMASS galaxies:
        from astropy.io import fits
        if local==True: filename = '/Users/user/Documents/MeerKAT/data/SDSS/dr12/galaxy_DR12v5_CMASSLOWZTOT_North.fits.gz'
        if local==False: filename = '/idia/users/ycli/SDSS/dr12/galaxy_DR12v5_CMASSLOWZTOT_North.fits.gz'
        hdu = fits.open(filename)
        ra_g = hdu[1].data['RA']
        dec_g = hdu[1].data['DEC']
        z_g = hdu[1].data['Z']

if doMock==True:
    if local==True: ra_g,dec_g,z_g = np.load('/Users/user/Documents/MeerKAT/meerpower/2019Lband/mocks/mockWiggleZcat_%s.npy'%mockindx)
    if local==False: ra_g,dec_g,z_g = np.load('/idia/projects/hi_im/meerpower/2019Lband/mocks/mockWiggleZcat_%s.npy'%mockindx)

if gal_cat=='cmass': # perform cuts to field to match MeerKAT patch
    ramin_CMASS,ramax_CMASS = np.min(ra[np.mean(W_HI,2)>0]),np.max(ra[np.mean(W_HI,2)>0])
    decmin_CMASS,decmax_CMASS = np.min(dec[np.mean(W_HI,2)>0]),np.max(dec[np.mean(W_HI,2)>0])
    MKcut = (ra_g>ramin_CMASS) & (ra_g<ramax_CMASS) & (dec_g>decmin_CMASS) & (dec_g<decmax_CMASS) & (z_g>zmin) & (z_g<zmax)
    cornercut_lim1 = 146 # set low to turn off
    cornercut_lim2 = 172.5 # set high to turn off
    cornercut = (ra_g - dec_g > cornercut_lim1) & (ra_g - dec_g < cornercut_lim2)
    CMASSgalmask = MKcut & cornercut
    ra_g = ra_g[CMASSgalmask]
    dec_g = dec_g[CMASSgalmask]
    z_g = z_g[CMASSgalmask]

print('Number of overlapping ', gal_cat,' galaxies: ', str(len(ra_g)))

# Assign galaxy bias:
if gal_cat=='wigglez':b_g = np.sqrt(0.83) # for WiggleZ at z_eff=0.41 - from https://arxiv.org/pdf/1104.2948.pdf [pg.9 rhs second quantity]
if gal_cat=='cmass':b_g = 1.85 # Mentioned in https://arxiv.org/pdf/1607.03155.pdf

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
dims_rg,dims0_rg = grid.comoving_dims(ra,dec,nu,wproj,ndim_rg,W=W_HI,dobuffer=True) # dimensions of Cartesian grid for FFT
lx,ly,lz,nx_rg,ny_rg,nz_rg = dims_rg

# Grid galaxies straight to Cartesian field:
xp,yp,zp = grid.SkyCoordtoCartesian(ra_g,dec_g,z_g,ramean_arr=ra,decmean_arr=dec,doTile=False)
n_g_rg = grid.mesh(xp,yp,zp,dims=dims0_rg,window=window,compensate=compensate,interlace=interlace,verbose=False)[0]

# Construct galaxy selection function:
if gal_cat=='wigglez': # grid WiggleZ randoms straight to Cartesian field for survey selection:
    BuildSelFunc = False
    if BuildSelFunc==True:
        nrand = 1000 # number of WiggleZ random catalogues to use in selection function (max is 1000)
        W_g_rg = np.zeros(np.shape(n_g_rg))
        for i in range(1,nrand):
            plot.ProgressBar(i,nrand)
            if local==True: galcat = np.genfromtxt( '/Users/user/Documents/MeerKAT/LauraShare/wigglez_reg11hrS_z0pt30_0pt50/reg11rand%s.dat' %'{:04d}'.format(i), skip_header=1)
            if local==False: galcat = np.genfromtxt( '/users/scunnington/MeerKAT/LauraShare/wigglez_reg11hrS_z0pt30_0pt50/reg11rand%s.dat' %'{:04d}'.format(i), skip_header=1)
            ra_g_rand,dec_g_rand,z_g_rand = galcat[:,0],galcat[:,1],galcat[:,2]
            z_Lband = (z_g_rand>zmin) & (z_g_rand<zmax)
            ra_g_rand = ra_g_rand[z_Lband]
            dec_g_rand = dec_g_rand[z_Lband]
            z_g_rand = z_g_rand[z_Lband]
            xp,yp,zp = grid.SkyCoordtoCartesian(ra_g_rand,dec_g_rand,z_g_rand,ramean_arr=ra,decmean_arr=dec,doTile=False)
            W_g_rg += grid.mesh(xp,yp,zp,dims=dims0_rg,window=window,compensate=compensate,interlace=interlace,verbose=False)[0]
        W_g_rg /= nrand
        if local==True: np.save('/Users/user/Documents/MeerKAT/meerpower/2019Lband/wigglez/data/wiggleZ_stackedrandoms_cell2voxfactor=%s.npy'%cell2vox_factor,W_g_rg)
        if local==False: np.save('/idia/projects/hi_im/meerpower/2019Lband/wigglez/data/wiggleZ_stackedrandoms_cell2voxfactor=%s.npy'%cell2vox_factor,W_g_rg)
    if local==True: W_g_rg = np.load('/Users/user/Documents/MeerKAT/meerpower/2019Lband/wigglez/data/wiggleZ_stackedrandoms_cell2voxfactor=%s.npy'%cell2vox_factor)
    if local==False: W_g_rg = np.load('/idia/projects/hi_im/meerpower/2019Lband/wigglez/data/wiggleZ_stackedrandoms_cell2voxfactor=%s.npy'%cell2vox_factor)
if gal_cat=='cmass':
    # Data obtained from untarrting DR12 file at: https://data.sdss.org/sas/dr12/boss/lss/dr12_multidark_patchy_mocks/Patchy-Mocks-DR12NGC-COMPSAM_V6C.tar.gz
    BuildSelFunc = False
    if BuildSelFunc==True:
        nrand = 2048 # number of WiggleZ random catalogues to use in selection function (max is 1000)
        ramin_CMASS,ramax_CMASS = np.min(ra[np.mean(W_HI,2)>0]),np.max(ra[np.mean(W_HI,2)>0])
        decmin_CMASS,decmax_CMASS = np.min(dec[np.mean(W_HI,2)>0]),np.max(dec[np.mean(W_HI,2)>0])
        W_g_rg = np.zeros(np.shape(n_g_rg))
        for i in range(1,nrand):
            plot.ProgressBar(i,nrand)
            galcat = np.genfromtxt( '/idia/projects/hi_im/meerpower/2019Lband/boss/sdss/Patchy-Mocks-DR12NGC-COMPSAM_V6C_%s.dat' %'{:04d}'.format(i+1), skip_header=1)
            ra_g_rand,dec_g_rand,z_g_rand = galcat[:,0],galcat[:,1],galcat[:,2]

            MKcut = (ra_g_rand>ramin_CMASS) & (ra_g_rand<ramax_CMASS) & (dec_g_rand>decmin_CMASS) & (dec_g_rand<decmax_CMASS) & (z_g_rand>zmin) & (z_g_rand<zmax)
            cornercut = (ra_g_rand - dec_g_rand > cornercut_lim1) & (ra_g_rand - dec_g_rand < cornercut_lim2)
            CMASSgalmask = MKcut & cornercut
            ra_g_rand = ra_g_rand[CMASSgalmask]
            dec_g_rand = dec_g_rand[CMASSgalmask]
            z_g_rand = z_g_rand[CMASSgalmask]

            xp,yp,zp = grid.SkyCoordtoCartesian(ra_g_rand,dec_g_rand,z_g_rand,ramean_arr=ra,decmean_arr=dec,doTile=False)
            W_g_rg += grid.mesh(xp,yp,zp,dims=dims0_rg,window='ngp',compensate=False,interlace=False,verbose=False)[0]
        W_g_rg /= nrand
        if local==True: np.save('/Users/user/Documents/MeerKAT/meerpower/2019Lband/boss/data/cmass_stackedrandoms_cell2voxfactor=%s.npy'%cell2vox_factor,W_g_rg)
        if local==False: np.save('/idia/projects/hi_im/meerpower/2019Lband/boss/data/cmass_stackedrandoms_cell2voxfactor=%s.npy'%cell2vox_factor,W_g_rg)
    if local==True: W_g_rg = np.load('/Users/user/Documents/MeerKAT/meerpower/2019Lband/boss/data/cmass_stackedrandoms_cell2voxfactor=%s.npy'%cell2vox_factor)
    if local==False: W_g_rg = np.load('/idia/projects/hi_im/meerpower/2019Lband/boss/data/cmass_stackedrandoms_cell2voxfactor=%s.npy'%cell2vox_factor)
#w_g_rg = np.ones(np.shape(W_g_rg))
w_g_rg = np.copy(W_g_rg)

n_g_rg_notaper,w_g_rg_notaper,W_g_rg_notaper = np.copy(n_g_rg),np.copy(w_g_rg),np.copy(W_g_rg)

### Chose to use Blackman window function along z direction as taper:
blackman = np.reshape( np.tile(np.blackman(nz_rg), (nx_rg,ny_rg)) , (nx_rg,ny_rg,nz_rg) ) # Blackman function along every LoS
taper_g = blackman

# Multiply tapering windows by all fields that undergo Fourier transforms:
#n_g_rg = taper_g*n_g_rg_notaper
#W_g_rg = taper_g*W_g_rg_notaper
w_g_rg = taper_g*w_g_rg_notaper

import power
import model
nkbin = 20
kmin,kmax = 0.02,0.28
kbins = np.linspace(kmin,kmax,nkbin+1) # k-bin edges [using linear binning]

import model
sig_v = 300

### Galaxy Auto-power (can use to constrain bias and use for analytical errors):
Pk_g,k,nmodes = power.Pk(n_g_rg,n_g_rg,dims_rg,kbins,corrtype='Galauto',w1=w_g_rg,w2=w_g_rg,W1=W_g_rg,W2=W_g_rg)
W_g_rg /= np.max(W_g_rg)
Vfrac = np.sum(W_g_rg)/(nx_rg*ny_rg*nz_rg)
nbar = np.sum(n_g_rg)/(lx*ly*lz*Vfrac) # Calculate number density inside survey footprint
P_SN = np.ones(len(k))*1/nbar # approximate shot-noise for errors (already subtracted in Pk estimator)
#pkmod,k = model.PkMod(Pmod,dims_rg,kbins,b_g,b_g,f,sig_v,Tbar1=1,Tbar2=1,r=1,R_beam1=0,R_beam2=0,w1=w_g_rg,w2=w_g_rg,W1=W_g_rg,W2=W_g_rg,interpkbins=True,MatterRSDs=False,gridinterp=True)[0:2]
pkmod,k = model.PkMod(Pmod,dims_rg,kbins,b_g,b_g,f,sig_v,Tbar1=1,Tbar2=1,r=r,R_beam1=0,R_beam2=0,w1=w_g_rg,w2=w_g_rg,W1=W_g_rg,W2=W_g_rg,s_pix=0,s_para=0,interpkbins=True,MatterRSDs=False,gridinterp=True)[0:2]
sig_g = 1/np.sqrt(nmodes)*(Pk_g+P_SN)
plt.errorbar(k,Pk_g,sig_g,ls='none',marker='o')
plt.plot(k,pkmod,color='black',ls='--')
plt.yscale('log')
plt.title(gal_cat+' auto-correlation')
plt.xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
plt.ylabel(r'$P_{\rm g}(k)\,[h^{-3}{\rm Mpc}^{3}]$')
plt.axhline(0,lw=0.8,color='black')
plt.show()
