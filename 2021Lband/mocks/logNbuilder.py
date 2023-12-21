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
counts_file = filestem + 'Nscan966_Npix_count_cube_p0.3d_sigma3.0_iter2.fits'
MKmap,w_HI,W_HI,counts_HI,dims,ra,dec,nu,wproj = Init.ReadIn(map_file,counts_file)

### Initialise some fiducial cosmology and survey parameters:
import cosmo
nu_21cm = 1420.405751 #MHz
zeff = (nu_21cm/np.median(nu)) - 1 # Effective redshift (redshift of median frequency)
cosmo.SetCosmology(builtincosmo='Planck18',z=zeff,UseCLASS=True)
Pmod = cosmo.GetModelPk(zeff,kmax=25,UseCLASS=True) # high-kmax needed for large k-modes in NGP alisasing correction
f = cosmo.f(zeff)
sig_v = 0
b_HI = 1.5
b_g = 1.6 # tuned by eye in GAMA auto-corr
OmegaHIbHI = 0.85e-3 # MKxWiggleZ constraint
OmegaHI = OmegaHIbHI/b_HI
import HItools
Tbar = HItools.Tbar(zeff,OmegaHI)
import telescope
D_dish = 13.5 # metres
theta_FWHM,R_beam = telescope.getbeampars(D_dish,nu)

### Remove incomplete LoS pixels from maps:
MKmap,w_HI,W_HI,counts_HI = Init.FilterIncompleteLoS(MKmap,w_HI,W_HI,counts_HI)

### Trim map edges (can fine-tune boarders if needed):
raminMK,ramaxMK = 334,357
decminMK,decmaxMK = -35,-26.5
MKmap,w_HI,W_HI,counts_HI = Init.MapTrim(ra,dec,MKmap,w_HI,W_HI,counts_HI,ramin=raminMK,ramax=ramaxMK,decmin=decminMK,decmax=decmaxMK)

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
### Cut redshift to MeerKAT IM range:
zmin,zmax = HItools.Freq2Red(np.max(nu)),HItools.Freq2Red(np.min(nu))
z_Lband = (z_g>zmin) & (z_g<zmax)
ra_g = ra_g[z_Lband]
dec_g = dec_g[z_Lband]
z_g = z_g[z_Lband]
### Remove galaxies outside bulk GAMA footprint so they don't ruin the binary mask
#plt.scatter(ra_g,dec_g,s=0.5)
raminGAMA,ramaxGAMA = 339,351
decminGAMA,decmaxGAMA = -35,-30
GAMAcutmask = (ra_g>raminGAMA) & (ra_g<ramaxGAMA) & (dec_g>decminGAMA) & (dec_g<decmaxGAMA)
ra_g,dec_g,z_g = ra_g[GAMAcutmask],dec_g[GAMAcutmask],z_g[GAMAcutmask]
#plt.scatter(ra_g,dec_g,s=0.5)
#plt.show()
#exit()

import grid
import mock
import power
import model
nkbin = 15
kmin,kmax = 0.05,0.28
kbins = np.linspace(kmin,kmax,nkbin+1) # k-bin edges [using linear binning]

ndim = [512,512,512]
#ndim = [256,256,256]
#ndim = [128,128,128]
dims,dims0 = grid.comoving_dims(ra,dec,nu,wproj,ndim,W=W_HI)
nxmap,nymap,nzmap = np.shape(MKmap)
ndim_rg = int(nxmap/1.5),int(nymap/1.5),int(nzmap/1.5)
print(ndim_rg)
nx_fft,ny_fft,nz_fft = ndim_rg
dims_rg,dims0_rg = grid.comoving_dims(ra,dec,nu,wproj,ndim_rg,W=W_HI)

### Grid GAMA galaxies to obtain target number density:
xp,yp,zp = grid.SkyCoordtoCartesian(ra_g,dec_g,z_g,ramean_arr=ra,decmean_arr=dec,doTile=False)
n_g_rg,W_fft,counts = grid.mesh(xp,yp,zp,dims=dims0_rg,window='ngp',compensate=False,interlace=False,verbose=True)

# Grid uncut pixels to obtain binary mask in comoving space:
ra_p,dec_p,nu_p = grid.SkyPixelParticles(ra,dec,nu,wproj,Np=5)
MKcutmask = (ra_p>raminMK) & (ra_p<ramaxMK) & (dec_p>decminMK) & (dec_p<decmaxMK)
xp,yp,zp = grid.SkyCoordtoCartesian(ra_p[MKcutmask],dec_p[MKcutmask],HItools.Freq2Red(nu_p[MKcutmask]),ramean_arr=ra,decmean_arr=dec,doTile=False)
null,W01_HI_rg,counts = grid.mesh(xp,yp,zp,dims=dims0_rg,window='ngp',compensate=False,interlace=False,verbose=True)
GAMAcutmask = (ra_p>raminGAMA) & (ra_p<ramaxGAMA) & (dec_p>decminGAMA) & (dec_p<decmaxGAMA)
xp,yp,zp = grid.SkyCoordtoCartesian(ra_p[GAMAcutmask],dec_p[GAMAcutmask],HItools.Freq2Red(nu_p[GAMAcutmask]),ramean_arr=ra,decmean_arr=dec,doTile=False)
null,W01_g_rg,counts = grid.mesh(xp,yp,zp,dims=dims0_rg,window='ngp',compensate=False,interlace=False,verbose=True)

Ngal = np.sum(n_g_rg)
Wfrac = np.sum(W01_g_rg)/(nx_fft*ny_fft*nz_fft)

Np = 10
window = 'ngp'
compensate = True
interlace = False

Nmock = 100
#'''
Pk = np.zeros((Nmock,nkbin))
for i in range(Nmock):
    print(i)
    seed = np.random.randint(0,1e6) # Use to generate consistent HI IM and galaxies from same random seed
    delta_HI = mock.Generate(Pmod,dims=dims,b=b_HI,f=f,Tbar=Tbar,doRSD=True,seed=seed,W=None,LightCone=False)

    #map_HI = grid.lightcone(delta_HI,dims0,ra,dec,nu,wproj,W=W_HI,Np=Np,verbose=True)
    map_HI = grid.lightcone(delta_HI,dims0,ra,dec,nu,wproj,W=None,Np=Np,verbose=True)
    # Include beam at map level:
    map_HI = telescope.ConvolveMap(map_HI,theta_FWHM,ra,dec,mode='reflect')
    map_HI[W_HI==0] = 0
    np.save('/idia/projects/hi_im/meerpower/2021Lband/mocks/dT_HI_p0.3d_wBeam_%s'%i,map_HI)

    delta_g = mock.Generate(Pmod,dims=dims,b=b_g,f=f,Tbar=1,doRSD=True,seed=seed,W=None,LightCone=False)
    n_g_mock = mock.PoissonSampleGalaxies(delta_g,dims,int(Ngal/Wfrac),ObtainExactNgal=True)
    ra_mockg,dec_mockg,z_mockg = mock.LogGalaxyCatalogue(n_g_mock,dims0,ra,dec)

    GAMAcutmask = (ra_mockg>raminGAMA) & (ra_mockg<ramaxGAMA) & (dec_mockg>decminGAMA) & (dec_mockg<decmaxGAMA) & (z_mockg>zmin) & (z_mockg<zmax)
    ra_mockg = ra_mockg[GAMAcutmask]
    dec_mockg = dec_mockg[GAMAcutmask]
    z_mockg = z_mockg[GAMAcutmask]

    print(Ngal)
    print(len(ra_mockg))
    np.save('/idia/projects/hi_im/meerpower/2021Lband/mocks/mockGAMAcat_%s'%i,[ra_mockg,dec_mockg,z_mockg])

    xp,yp,zp = grid.SkyCoordtoCartesian(ra_mockg,dec_mockg,z_mockg,ramean_arr=ra,decmean_arr=dec,doTile=False)
    n_g_mock_rg,W_fft,counts = grid.mesh(xp,yp,zp,dims=dims0_rg,window=window,compensate=compensate,interlace=interlace,verbose=True)

    ra_p,dec_p,nu_p,pixvals = grid.SkyPixelParticles(ra,dec,nu,wproj,map=map_HI,W=W_HI,Np=Np)
    xp,yp,zp = grid.SkyCoordtoCartesian(ra_p,dec_p,HItools.Freq2Red(nu_p),ramean_arr=ra,decmean_arr=dec,doTile=False)
    delta_HI_rg,W_fft,counts = grid.mesh(xp,yp,zp,pixvals,dims0_rg,window,compensate,interlace,verbose=True)

    Pk[i],k,nmodes = power.Pk(delta_HI_rg,n_g_mock_rg,dims_rg,kbins,corrtype='Cross',w1=W01_HI_rg,w2=W01_g_rg,W1=W01_HI_rg,W2=W01_g_rg)

np.save('/idia/projects/hi_im/steve/gridimp/data/MKtestingPks_window=%s_interlace=%s_wRSD'%(window,interlace),[Pk,k])
exit()
#'''
if local==True: Pk,k = np.load('/Users/user/Documents/gridimp/data/MKtestingPks_window=%s_interlace=%s_wRSD.npy'%(window,interlace),allow_pickle=True)
if local==False: Pk,k = np.load('/idia/projects/hi_im/steve/gridimp/data/MKtestingPks_window=%s_interlace=%s_wRSD.npy'%(window,interlace),allow_pickle=True)

dpix = 0.3
d_c = cosmo.d_com(HItools.Freq2Red(np.min(nu)))
s_pix = d_c * np.radians(dpix)
s_para = np.mean( cosmo.d_com(HItools.Freq2Red(nu[:-1])) - cosmo.d_com(HItools.Freq2Red(nu[1:])) )
print(s_pix)
print(s_para)

pkmod,k = model.PkMod(Pmod,dims_rg,kbins,b_HI,b_g,f,sig_v=0,Tbar1=Tbar,Tbar2=1,r=1,R_beam1=np.mean(R_beam),R_beam2=0,w1=W01_HI_rg,w2=W01_g_rg,W1=W01_HI_rg,W2=W01_g_rg,s_para=s_para,s_pix=s_pix,interpkbins=True,MatterRSDs=False,gridinterp=True)[0:2]
plt.plot(k,k**2*pkmod,color='black',ls='--')
plt.errorbar(k,k**2*np.mean(Pk,0),k**2*np.std(Pk,0),ls='none',marker='o')
plt.figure()

plt.axhline(1,color='black',ls='--')
plt.errorbar(k,np.mean(Pk,0)/pkmod,np.std(Pk,0)/pkmod,ls='none',marker='o')
plt.show()
