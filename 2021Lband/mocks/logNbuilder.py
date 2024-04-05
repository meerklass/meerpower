import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
local = False
if local==True: sys.path.insert(1, '/Users/user/Documents/MeerKAT/meerpower/meerpower')
if local==False: sys.path.insert(1, '/idia/projects/hi_im/meerpower/meerpower')
import plot
import Init

# Read-in 2021 MeerKAT data to obtain coordinates:
if local==True: filestem = '/Users/user/Documents/MeerKAT/meerpower/localdata/'
if local==False: filestem = '/idia/projects/hi_im/raw_vis/MeerKLASS2021/level6/0.3/sigma_3/data/'
map_file = filestem + 'Nscan966_Tsky_cube_p0.3d_sigma3.0_iter2.fits'
numin,numax = 971,1023.8 # default setting in Init.ReadIn()
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
b_g = 2.35 # tuned by eye in GAMA auto-corr
OmegaHIbHI = 0.85e-3 # MKxWiggleZ constraint
OmegaHI = OmegaHIbHI/b_HI
import HItools
Tbar = HItools.Tbar(zeff,OmegaHI)
import telescope
D_dish = 13.5 # metres
theta_FWHM,R_beam = telescope.getbeampars(D_dish,nu)

# Read-in GAMA G-23 field galaxies:
from astropy.io import fits
if local==True: Fits = '/Users/user/Documents/MeerKAT/GAMA_DR4/G23TilingCatv11.fits'
if local==False: Fits = '/idia/projects/hi_im/GAMA_DR4/G23TilingCatv11.fits'
hdu = fits.open(Fits)
hdr = hdu[1].header
#print(hdr)
ra_g = hdu[1].data['RA'] # Right ascension (J2000) [deg]
dec_g = hdu[1].data['DEC'] # Declination (J2000) [deg]
z_g = hdu[1].data['Z'] # Spectroscopic redshift, -1 for none attempted
### Remove galaxies outside bulk GAMA footprint so they don't ruin the binary mask
zmin,zmax = HItools.Freq2Red(np.max(nu)),HItools.Freq2Red(np.min(nu))
raminGAMA,ramaxGAMA = 339,351
decminGAMA,decmaxGAMA = -35,-30
GAMAcutmask = (ra_g>raminGAMA) & (ra_g<ramaxGAMA) & (dec_g>decminGAMA) & (dec_g<decmaxGAMA) & (z_g>zmin) & (z_g<zmax)
ra_g,dec_g,z_g = ra_g[GAMAcutmask],dec_g[GAMAcutmask],z_g[GAMAcutmask]
Ngal = len(ra_g)
print(Ngal)

### Create high-res grids to generate lognormal onto which then gets transformed
###  to sky voxels
import grid
import mock
ndim = [512,512,512]
#ndim = [256,256,256]
#ndim = [128,128,128]
dims,dims0 = grid.comoving_dims(ra,dec,nu,wproj,ndim,W=W_HI)

'''
# For GAMA selection function, obtain GAMA footprint on 512^3 grid by creating
#   particle for every cell, transforming to get Sky coord, then cut based on footprint:
'''
xp,yp,zp = grid.GridParticles(dims0,delta=None,W=None,Np=1)
ra_p,dec_p,z_p = grid.Cart2SphericalCoords(xp,yp,zp,ramean_arr=ra,decmean_arr=dec)
GAMAcutmask = (ra_p>raminGAMA) & (ra_p<ramaxGAMA) & (dec_p>decminGAMA) & (dec_p<decmaxGAMA) & (z_p>zmin) & (z_p<zmax)
xp,yp,zp = grid.SkyCoordtoCartesian(ra_p[GAMAcutmask],dec_p[GAMAcutmask],z_p[GAMAcutmask],ramean_arr=ra,decmean_arr=dec,doTile=False)
null,W_g_rg,counts = grid.mesh(xp,yp,zp,dims=dims0,window='ngp',compensate=False,interlace=False,verbose=True)

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
    map_HI = grid.lightcone(delta_HI,dims0,ra,dec,nu,wproj,W=None,Np=Np,obsdata='2021',verbose=True)
    # Include beam at map level:
    map_HI = telescope.ConvolveMap(map_HI,theta_FWHM,ra,dec)
    map_HI[W_HI==0] = 0
    np.save('/idia/projects/hi_im/meerpower/2021Lband/mocks/dT_HI_p0.3d_wBeam_%s'%i,map_HI)

    ### GAMA galaxy field and catalogue:
    delta_g = mock.Generate(Pmod,dims=dims,b=b_g,f=f,sig_v=sig_v,Tbar=1,doRSD=True,seed=seed,W=None,LightCone=False)
    n_g_mock = mock.PoissonSampleGalaxies(delta_g,dims,Ngal,W=W_g_rg,ObtainExactNgal=True)
    ra_mockg,dec_mockg,z_mockg = mock.LogGalaxyCatalogue(n_g_mock,dims0,ra,dec)
    print(Ngal)
    print(len(ra_mockg))
    np.save('/idia/projects/hi_im/meerpower/2021Lband/mocks/mockGAMAcat_%s'%i,[ra_mockg,dec_mockg,z_mockg])
