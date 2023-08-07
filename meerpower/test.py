import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
sys.path.insert(1, '/idia/projects/hi_im/meerpower/meerpower')
import Init
import plot

# Read-in some level6 data just to obtain coordinates:
filestem = '/idia/projects/hi_im/raw_vis/MeerKLASS2021/level6/0.3/sigma_3/data/'
map_file = filestem + 'Nscan966_Tsky_cube_p0.3d_sigma3.0_iter2.fits'
counts_file = filestem + 'Nscan966_Npix_count_cube_p0.3d_sigma3.0_iter2.fits'
null0,null1,null2,null3,dims,ra,dec,nu,wproj = Init.ReadIn(map_file,counts_file)

######################################################################
############## CHOSE LEVEL 5 MAPS TO ANALYSE HERE: ###################
## - where individial dish and scan maps are stored - ################
level5path = '/idia/projects/hi_im/raw_vis/MeerKLASS2021/level5/0.3/data/' # previous sky model calibration technique
#level5path = '/scratch3/users/jywang/MeerKLASS2021/level5/0.3/sigma4_count40/re_cali1_round5/' # updated self-calibration technique
level5subset_folder = level5path.replace("/", "_" ) # name subset folder after root data path to keep track of different subset map versions
output_path = '/idia/projects/hi_im/raw_vis/MeerKLASS2021/subsetmaps/' + level5subset_folder + '/'  # where to save created subsets of combined dish and scan maps
print(output_path)

MKmap_B,w_B,W_B,counts_B = np.load(output_path+'dish10-10_scan0-40.npy')
MKmap_A,w_A,W_A,counts_A = np.load(output_path+'dish28-28_scan0-40.npy')
#MKmap_A,w_A,W_A,counts_A = np.load(output_path+'dish0-63_scan20-20.npy')

### Remove incomplete LoS pixels from maps:
MKmap_A,w_A,W_A,counts_A = Init.FilterIncompleteLoS(MKmap_A,w_A,W_A,counts_A)
MKmap_B,w_B,W_B,counts_B = Init.FilterIncompleteLoS(MKmap_B,w_B,W_B,counts_B)

### Trim map edges (can fine-tune boarders if needed):
MKmap_A,w_A,W_A,counts_A = Init.MapTrim(MKmap_A,w_A,W_A,counts_A,ra,dec,ramin=334,ramax=357,decmin=-35,decmax=-26.5)
MKmap_B,w_B,W_B,counts_B = Init.MapTrim(MKmap_B,w_B,W_B,counts_B,ra,dec,ramin=334,ramax=357,decmin=-35,decmax=-26.5)

#plot.Map(MKmap_A,W=W_A,map_ra=ra,map_dec=dec,wproj=wproj,title='Time block - 20')
plot.Map(MKmap_B,W=W_B,map_ra=ra,map_dec=dec,wproj=wproj,title='[m010 - all scans]')
plot.Map(MKmap_A,W=W_A,map_ra=ra,map_dec=dec,wproj=wproj,title='[m028 - all scans]')
plt.figure()
#exit()

### Initialise some fiducial cosmology and survey parameters:
import cosmo
nu_21cm = 1420.405751 #MHz
zeff = (nu_21cm/np.median(nu)) - 1 # Effective redshift (redshift of median frequency)
cosmo.SetCosmology(builtincosmo='Planck18',z=zeff)
Pmod = cosmo.GetModelPk(zeff)
f = cosmo.f(zeff)
sig_v = 0
b_HI = 1.5
OmegaHI = 1.333333e-3
OmegaHIbHI = 0.85e-3 # MKxWiggleZ constraint
OmegaHI = OmegaHIbHI/b_HI
import HItools
import telescope
Tbar = HItools.Tbar(zeff,OmegaHI)
D_dish = 13.5 # Dish-diameter [metres]
theta_FWHM,R_beam = telescope.getbeampars(D_dish,np.median(nu))

import power # All power spectrum calculations performed in this script
import model
nkbin = 15
kmin,kmax = 0.05,0.28
kbins = np.linspace(kmin,kmax,nkbin+1) # k-bin edges [using linear binning]
nx,ny,nz = np.shape(MKmap_A)

import grid
nx_rg,ny_rg,nz_rg = int(nx/2),int(ny/2),int(nz/2) # number of pixels in Comoving space to grid to
ndim = nx_rg,ny_rg,nz_rg
blackman = 1

### Chose 2D k-bins:
kperpbins = np.linspace(0.00882353,0.3,34)
kparabins = np.linspace(0.003,0.6,20)
kperpcen = (kperpbins[1:] + kperpbins[:-1])/2
kparacen = (kparabins[1:] + kparabins[:-1])/2
kperpgrid = np.tile(kperpcen,(len(kparacen),1))
kparagrid = np.tile(kparacen,(len(kperpcen),1))
kparagrid = np.swapaxes(kparagrid,0,1)
kgrid = kperpcen * kparacen[:,np.newaxis]

### Create split colorbar for 2D Pk negative values:
import matplotlib.colors as mcolors
colors1 = plt.cm.Blues_r(np.linspace(0., 1, 128))
colors2 = plt.cm.gist_heat_r(np.linspace(0, 1, 128))
colors = np.vstack((colors1, colors2))
mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

MKmap_clean_A = np.copy(MKmap_A)
MKmap_clean_rg_A,dims,dims0 = grid.comoving(blackman*MKmap_clean_A,ra,dec,nu,W=W_A,ndim=ndim)
w_rg_A,dims,dims0 = grid.comoving(blackman*w_A,ra,dec,nu,W=W_A,ndim=ndim)
W_rg_A,dims,dims0 = grid.comoving(blackman*W_A,ra,dec,nu,W=W_A,ndim=ndim)
Pk2D_A,k2d,nmodes = power.Pk2D(MKmap_clean_rg_A,MKmap_clean_rg_A,dims,kperpbins,kparabins,w1=w_rg_A,w2=w_rg_A,W1=W_rg_A,W2=W_rg_A)
MKmap_clean_B = np.copy(MKmap_B)
MKmap_clean_rg_B,dims,dims0 = grid.comoving(blackman*MKmap_clean_B,ra,dec,nu,W=W_B,ndim=ndim)
w_rg_B,dims,dims0 = grid.comoving(blackman*w_B,ra,dec,nu,W=W_B,ndim=ndim)
W_rg_B,dims,dims0 = grid.comoving(blackman*W_B,ra,dec,nu,W=W_B,ndim=ndim)
Pk2D_B,k2d,nmodes = power.Pk2D(MKmap_clean_rg_B,MKmap_clean_rg_B,dims,kperpbins,kparabins,w1=w_rg_B,w2=w_rg_B,W1=W_rg_B,W2=W_rg_B)

plt.pcolormesh(kperpbins,kparabins,np.log10(Pk2D_A),cmap='viridis')
plt.colorbar(label=r'log[$P(k_\perp,k_\parallel)$]')
plt.xlabel(r'$k_\perp [h\,{\rm Mpc}^{-1}]$')
plt.ylabel(r'$k_\parallel [h\,{\rm Mpc}^{-1}]$')
#plt.title(r'Auto one timeblock ($N_{\rm fg}=0$) ')
plt.title(r'Auto [m028 - all scans] ($N_{\rm fg}=0$) ')
plt.figure()
plt.pcolormesh(kperpbins,kparabins,np.log10(Pk2D_B),cmap='viridis')
plt.colorbar(label=r'log[$P(k_\perp,k_\parallel)$]')
plt.xlabel(r'$k_\perp [h\,{\rm Mpc}^{-1}]$')
plt.ylabel(r'$k_\parallel [h\,{\rm Mpc}^{-1}]$')
#plt.title(r'Auto one timeblock ($N_{\rm fg}=0$) ')
plt.title(r'Auto [m010 - all scans] ($N_{\rm fg}=0$) ')
plt.figure()

plt.pcolormesh(kperpbins,kparabins,np.log10(Pk2D_A)-np.log10(Pk2D_B),cmap='viridis')
plt.colorbar(label=r'log[$P$] - log[$P$]')
plt.xlabel(r'$k_\perp [h\,{\rm Mpc}^{-1}]$')
plt.ylabel(r'$k_\parallel [h\,{\rm Mpc}^{-1}]$')
#plt.title(r'Auto one timeblock ($N_{\rm fg}=0$) ')
plt.title(r'Residual autos m028 and m010')
plt.show()
