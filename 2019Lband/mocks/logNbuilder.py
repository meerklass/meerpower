import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
local = False
if local==True: sys.path.insert(1, '/Users/user/Documents/MeerKAT/meerpower/meerpower')
if local==False: sys.path.insert(1, '/idia/projects/hi_im/meerpower/meerpower')
import plot
import Init

# Read-in 2019 MeerKAT data to obtain coordinates:
if local==True: filestem = '/Users/user/Documents/MeerKAT/meerpower/localdata/2019/'
if local==False: filestem = '/idia/projects/hi_im/raw_vis/katcali_output/level6_output/p0.3d/p0.3d_sigma2.5_iter2/'
map_file = filestem + 'Nscan366_Tsky_cube_p0.3d_sigma2.5_iter2.fits'
numin,numax = 971,1023.2

MKmap,w_HI,W_HI,counts_HI,dims,ra,dec,nu,wproj = Init.ReadIn(map_file,numin=numin,numax=numax)

### Initialise some fiducial cosmology and survey parameters:
import cosmo
nu_21cm = 1420.405751 #MHz
zeff = (nu_21cm/np.median(nu)) - 1 # Effective redshift (redshift of median frequency)
cosmo.SetCosmology(builtincosmo='Planck18',z=zeff,UseCLASS=True)
Pmod = cosmo.GetModelPk(zeff,kmax=25,UseCLASS=True,NonLinear=False) # high-kmax needed for large k-modes in NGP alisasing correction
f = cosmo.f(zeff)
sig_v = 300
b_HI = 1.5
b_g = np.sqrt(0.83) # for WiggleZ at z_eff=0.41 - from https://arxiv.org/pdf/1104.2948.pdf [pg.9 rhs second quantity]
b_g_CMASS = 1.85 # Mentioned in https://arxiv.org/pdf/1607.03155.pdf
OmegaHIbHI = 0.85e-3 # MKxWiggleZ constraint
OmegaHI = OmegaHIbHI/b_HI
import HItools
Tbar = HItools.Tbar(zeff,OmegaHI)
import telescope
D_dish = 13.5 # metres
theta_FWHM,R_beam = telescope.getbeampars(D_dish,nu)

# Read-in WiggleZ galaxies (provided by Laura):
if local==True: galcat = np.genfromtxt('/Users/user/Documents/MeerKAT/LauraShare/wigglez_reg11hrS_z0pt30_0pt50/reg11data.dat', skip_header=1)
if local==False: galcat = np.genfromtxt('/users/scunnington/MeerKAT/LauraShare/wigglez_reg11hrS_z0pt30_0pt50/reg11data.dat', skip_header=1)
ra_g,dec_g,z_g = galcat[:,0],galcat[:,1],galcat[:,2]
### Cut redshift to MeerKAT IM range:
zmin,zmax = HItools.Freq2Red(np.max(nu)),HItools.Freq2Red(np.min(nu))
z_Lband = (z_g>zmin) & (z_g<zmax)
ra_g = ra_g[z_Lband]
dec_g = dec_g[z_Lband]
z_g = z_g[z_Lband]
Ngal = len(ra_g)
print(Ngal)
# Read-in BOSS-CMASS galaxies:
from astropy.io import fits
if local==True: filename = '/Users/user/Documents/MeerKAT/data/SDSS/dr12/galaxy_DR12v5_CMASSLOWZTOT_North.fits.gz'
if local==False: filename = '/idia/users/ycli/SDSS/dr12/galaxy_DR12v5_CMASSLOWZTOT_North.fits.gz'
hdu = fits.open(filename)
ra_g_CMASS = hdu[1].data['RA']
dec_g_CMASS = hdu[1].data['DEC']
z_g_CMASS = hdu[1].data['Z']
ramin_CMASS,ramax_CMASS = np.min(ra[np.mean(W_HI,2)>0]),np.max(ra[np.mean(W_HI,2)>0])
decmin_CMASS,decmax_CMASS = np.min(dec[np.mean(W_HI,2)>0]),np.max(dec[np.mean(W_HI,2)>0])
MKcut = (ra_g_CMASS>ramin_CMASS) & (ra_g_CMASS<ramax_CMASS) & (dec_g_CMASS>decmin_CMASS) & (dec_g_CMASS<decmax_CMASS) & (z_g_CMASS>zmin) & (z_g_CMASS<zmax)
ra_g_CMASS = ra_g_CMASS[MKcut]
dec_g_CMASS = dec_g_CMASS[MKcut]
z_g_CMASS = z_g_CMASS[MKcut]
Ngal_CMASS = len(ra_g_CMASS)
print(Ngal_CMASS)

### Create high-res grids to generate lognormal onto which then gets transformed
###  to sky voxels
import grid
import mock
ndim = [512,512,512]
#ndim = [256,256,256]
#ndim = [128,128,128]
W_g = grid.AstropyGridding(ra_g_CMASS,dec_g_CMASS,z_g_CMASS,nu,particleweights=np.ones(len(ra_g_CMASS)),obsdata='2019')[0]
W_comb = W_HI + W_g
W_comb[W_comb!=0] = 1
dims,dims0 = grid.comoving_dims(ra,dec,nu,wproj,ndim,W=W_comb)

'''
### Galaxy selection functions: use galaxy mocks and stack to lower resolution
###   grid to avoid under-sampled masks - then transform this to high-res grid
###   for application to generated lognormal field and galaxy catalogues that are
###   consistent with the real survey selections.
'''
import grid
ndim_lr = [128,128,128] # create lower resolution grid to create survey mask then extrapolate to high-res field
dims_lr,dims0_lr = grid.comoving_dims(ra,dec,nu,wproj,ndim_lr,W=W_comb)
### WiggleZ selection function:
nrand = 100 # number of WiggleZ random catalogues to use in selection function (max is 1000)
W_g_rg = np.zeros(ndim_lr)
for i in range(1,nrand):
    plot.ProgressBar(i,nrand)
    if local==True: galcat = np.genfromtxt('/Users/user/Documents/MeerKAT/LauraShare/wigglez_reg11hrS_z0pt30_0pt50/reg11rand%s.dat' %'{:04d}'.format(i), skip_header=1)
    if local==False: galcat = np.genfromtxt( '/users/scunnington/MeerKAT/LauraShare/wigglez_reg11hrS_z0pt30_0pt50/reg11rand%s.dat' %'{:04d}'.format(i), skip_header=1)
    ra_g_rand,dec_g_rand,z_g_rand = galcat[:,0],galcat[:,1],galcat[:,2]
    z_Lband = (z_g_rand>zmin) & (z_g_rand<zmax)
    ra_g_rand = ra_g_rand[z_Lband]
    dec_g_rand = dec_g_rand[z_Lband]
    z_g_rand = z_g_rand[z_Lband]
    xp,yp,zp = grid.SkyCoordtoCartesian(ra_g_rand,dec_g_rand,z_g_rand,ramean_arr=ra,decmean_arr=dec,doTile=False)
    W_g_rg += grid.mesh(xp,yp,zp,dims=dims0_lr,window='ngp',compensate=False,interlace=False,verbose=False)[0]
W_g_rg /= nrand
### CMASS selection function:
# Data obtained from untarrting DR12 file at: https://data.sdss.org/sas/dr12/boss/lss/dr12_multidark_patchy_mocks/Patchy-Mocks-DR12NGC-COMPSAM_V6C.tar.gz
nrand = 2048 # number of MD-Patch random catalogues to use in selection function (max is 2048)
ramin_CMASS,ramax_CMASS = np.min(ra[np.mean(W_HI,2)>0]),np.max(ra[np.mean(W_HI,2)>0])
decmin_CMASS,decmax_CMASS = np.min(dec[np.mean(W_HI,2)>0]),np.max(dec[np.mean(W_HI,2)>0])
W_g_rg_CMASS = np.zeros(ndim_lr)
for i in range(1,nrand):
    plot.ProgressBar(i,nrand)
    galcat = np.genfromtxt( '/idia/projects/hi_im/meerpower/2019Lband/boss/sdss/Patchy-Mocks-DR12NGC-COMPSAM_V6C_%s.dat' %'{:04d}'.format(i+1), skip_header=1)
    ra_g_rand,dec_g_rand,z_g_rand = galcat[:,0],galcat[:,1],galcat[:,2]
    MKcut = (ra_g_rand>ramin_CMASS) & (ra_g_rand<ramax_CMASS) & (dec_g_rand>decmin_CMASS) & (dec_g_rand<decmax_CMASS) & (z_g_rand>zmin) & (z_g_rand<zmax)
    ra_g_rand = ra_g_rand[MKcut]
    dec_g_rand = dec_g_rand[MKcut]
    z_g_rand = z_g_rand[MKcut]
    xp,yp,zp = grid.SkyCoordtoCartesian(ra_g_rand,dec_g_rand,z_g_rand,ramean_arr=ra,decmean_arr=dec,doTile=False)
    W_g_rg_CMASS += grid.mesh(xp,yp,zp,dims=dims0_lr,window='ngp',compensate=False,interlace=False,verbose=False)[0]
W_g_rg_CMASS /= nrand
# Transform low-res selection functions to high-res field:
xp,yp,zp,cellvals = grid.ParticleSampling(W_g_rg,dims0_lr,dims0,Np=1,sample_ingrid=False)
W_g_rg = grid.mesh(xp,yp,zp,T=cellvals,dims=dims0,window='ngp',compensate=False,interlace=False,verbose=False)[0]
xp,yp,zp,cellvals = grid.ParticleSampling(W_g_rg_CMASS,dims0_lr,dims0,Np=1,sample_ingrid=False)
W_g_rg_CMASS = grid.mesh(xp,yp,zp,T=cellvals,dims=dims0,window='ngp',compensate=False,interlace=False,verbose=False)[0]

'''
### Create Nmock lognormal realisations with correalted HI map and galaxy catalogue
###   counterpart.
'''
Np = 10 # number of sampling particles to use for Cartesian grid-> Sky map
Nmock = 500
for i in range(Nmock):
    print(i)
    seed = np.random.randint(0,1e6) # Use to generate consistent HI IM and galaxies from same random seed

    ### HI intensity map:
    delta_HI = mock.Generate(Pmod,dims=dims,b=b_HI,f=f,sig_v=sig_v,Tbar=Tbar,doRSD=True,seed=seed,W=None,LightCone=False)
    map_HI = grid.lightcone(delta_HI,dims0,ra,dec,nu,wproj,W=None,Np=Np,obsdata='2019',verbose=True)
    # Include beam at map level:
    map_HI = telescope.ConvolveMap(map_HI,theta_FWHM,ra,dec)
    map_HI[W_HI==0] = 0
    np.save('/idia/projects/hi_im/meerpower/2019Lband/mocks/dT_HI_p0.3d_wBeam_%s'%i,map_HI)

    ### WiggleZ galaxy field and catalogue:
    delta_g = mock.Generate(Pmod,dims=dims,b=b_g,f=f,sig_v=sig_v,Tbar=1,doRSD=True,seed=seed,W=None,LightCone=False)
    n_g_mock = mock.PoissonSampleGalaxies(delta_g,dims,Ngal,W=W_g_rg,ObtainExactNgal=True)
    ra_mockg,dec_mockg,z_mockg = mock.LogGalaxyCatalogue(n_g_mock,dims0,ra,dec)
    print(Ngal)
    print(len(ra_mockg))
    np.save('/idia/projects/hi_im/meerpower/2019Lband/mocks/mockWiggleZcat_%s'%i,[ra_mockg,dec_mockg,z_mockg])

    ### CMASS galaxy field and catalogue:
    delta_g = mock.Generate(Pmod,dims=dims,b=b_g_CMASS,f=f,Tbar=1,doRSD=True,seed=seed,W=None,LightCone=False)
    n_g_mock = mock.PoissonSampleGalaxies(delta_g,dims,Ngal_CMASS,W=W_g_rg_CMASS,ObtainExactNgal=True)
    ra_mockg,dec_mockg,z_mockg = mock.LogGalaxyCatalogue(n_g_mock,dims0,ra,dec)
    print(Ngal_CMASS)
    print(len(ra_mockg))
    np.save('/idia/projects/hi_im/meerpower/2019Lband/mocks/mockCMASScat_%s'%i,[ra_mockg,dec_mockg,z_mockg])
