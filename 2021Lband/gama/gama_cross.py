import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
sys.path.insert(1, '/idia/projects/hi_im/meerpower/meerpower')
import Init
import plot

# Read-in level6 MeerKAT data:
filestem = '/idia/users/jywang/MeerKLASS/calibration2021/level6/0.3/sigma4_count40/re_cali1_round5/'
map_file = filestem + 'Nscan961_Tsky_cube_p0.3d_sigma4.0_iter2.fits'
MKmap,w_HI,W_HI,counts_HI,dims,ra_map,dec_map,nu,wproj = Init.ReadIn(map_file)

### Initialise some fiducial cosmology and survey parameters:
import cosmo
nu_21cm = 1420.405751 #MHz
zeff = (nu_21cm/np.median(nu)) - 1 # Effective redshift (redshift of median frequency)
cosmo.SetCosmology(builtincosmo='Planck18',z=zeff)
Pmod = cosmo.GetModelPk(zeff)
f = cosmo.f(zeff)
sig_v = 0
b_HI = 1.5
OmegaHIbHI = 0.85e-3 # MKxWiggleZ constraint
OmegaHI = OmegaHIbHI/b_HI
import HItools
import telescope
Tbar = HItools.Tbar(zeff,OmegaHI)
D_dish = 13.5 # Dish-diameter [metres]
theta_FWHM,R_beam = telescope.getbeampars(D_dish,np.median(nu))

### Remove incomplete LoS pixels from maps:
MKmap,w_HI,W_HI,counts_HI = Init.FilterIncompleteLoS(MKmap,w_HI,W_HI,counts_HI)

### Trim map edges (can fine-tune boarders if needed):
MKmap,w_HI,W_HI,counts_HI = Init.MapTrim(ra_map,dec_map,MKmap,w_HI,W_HI,counts_HI,ramin=334,ramax=357,decmin=-35,decmax=-26.5)

### Use counts map as weights:
w_HI = np.copy(counts_HI) # Not factorised along LoS so will add rank if used in foreground cleaning in current form

# Read-in GAMA G-23 field galaxies:
from astropy.io import fits
Fits = '/idia/projects/hi_im/GAMA_DR4/G23TilingCatv11.fits'
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
'''
### Trim final footprint down to ~GAMA footprint then regridding will remove empty pixel space after foreground cleaning:
ramin = 337
ramax = 353
decmin = -35
decmax = -29
W_regridfoot = Init.MapTrim(ra_map,dec_map,map1=W_HI,ramin=ramin,ramax=ramax,decmin=decmin,decmax=decmax)
'''
W_regridfoot = np.copy(W_HI)

### Obtain pixel and channel sizes for damping models:
d_c = cosmo.d_com( HItools.Freq2Red(np.median(nu))) # Comoving distance to frequency bin
ra_map[ra_map>180] = ra_map[ra_map>180] - 360 # Make continuous RA i.e. 359,360,1 -> -1,0,1 so mean RA is correct
s_pix_ra = d_c * np.radians(np.mean(np.diff(ra_map,axis=0)))
s_pix_dec = d_c * np.radians(np.mean(np.diff(dec_map,axis=1)))
ra_map[ra_map<0] = ra_map[ra_map<0] + 360 # Reset negative coordinates to 359,360,1 convention
s_pix = np.mean([s_pix_ra,s_pix_dec]) #Pix size [mpc/h] # use for generic pixelisation function
deltanu = abs(nu[0]-nu[1])
z1 = HItools.Freq2Red(np.median(nu))
z0 = HItools.Freq2Red(np.median(nu) + deltanu)
s_para = cosmo.d_com(z1) - cosmo.d_com(z0)

import foreground
import power # All power spectrum calculations performed in this script
import model
from scipy import signal
nkbin = 15
kmin,kmax = 0.05,0.28
kbins = np.linspace(kmin,kmax,nkbin+1) # k-bin edges [using linear binning]
nx,ny,nz = np.shape(MKmap)

import grid # use this for going from (ra,dec,freq)->(x,y,z) Cartesian-comoving grid
nx_rg,ny_rg,nz_rg = int(nx/2),int(ny/2),int(nz/2) # number of pixels in Comoving space to grid to
ndim = nx_rg,ny_rg,nz_rg

doWindow = True # Implement window functions to apply to maps and weights to taper edges
tukey_alpha = 0.3 # (set =0 to turn off) fraction of the window inside the cosine tapered region: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.tukey.html
if doWindow==True:
    window_x = np.tile(signal.windows.tukey(nx_rg,tukey_alpha)[:,np.newaxis,np.newaxis],(1,ny_rg,nz_rg))
    window_y = np.tile(signal.windows.tukey(ny_rg,tukey_alpha)[np.newaxis,:,np.newaxis],(nx_rg,1,nz_rg))
    window_z = np.tile(signal.windows.tukey(nz_rg,tukey_alpha)[np.newaxis,np.newaxis,:],(nx_rg,ny_rg,1))
    window = window_x*window_y*window_z # combine all windows into 3D taper
else: window = 1

#z_g_shuffle = np.copy(z_g)
#np.random.shuffle(z_g_shuffle)

dims_rg,dims0 = grid.comoving_dims(ra_map,dec_map,nu,wproj,ndim=ndim,W=W_HI)
print(dims0)
exit()
lx,ly,lz,nx_rg,ny_rg,nz_rg = dims_rg

b_g = 1.9 # tuned by eye

n_g = grid.CartesianGridGalaxies(ra_g,dec_g,z_g,dims0,ra_map,dec_map)
w_g_rg = np.ones(np.shape(n_g))
W_g_rg = np.ones(np.shape(n_g))

### Trim (set to zero) pixels outside the GAMA field and its weights/windows
n_g,w_g_rg,W_g_rg = Init.GridTrim(n_g,w_g_rg,W_g_rg,x0=16,x1=-18,y0=2,y1=-15)
Ngal = np.sum(n_g)
print(Ngal)
n_g_plot = np.copy(n_g)
n_g_plot[W_g_rg==0] = np.nan
plt.imshow(np.mean(n_g_plot,2))
plt.colorbar()
plt.show()
exit()

'''
# Create and trim galaxies sky masks to GAMA window for n_g mocks in TF
w_g = np.ones((nx,ny,nz))
W_g = np.ones((nx,ny,nz))
w_g,W_g = Init.MapTrim(ra_map,dec_map,map1=w_g,map2=W_g,ramin=ramin,ramax=ramax,decmin=decmin,decmax=decmax)
ndens = Ngal/(np.sum(W_g))
W_g *= ndens # Uniform survey selection function where footprint covered
print(ndens)
'''
# Multiply Tukey taper window by all maps that undergo Fourier transforms
n_g,w_g_rg,W_g_rg = window*n_g,window*w_g_rg,window*W_g_rg

### GAMA Auto-power (use to constrain bias):
Pk_g,k,nmodes = power.Pk(n_g,n_g,dims_rg,kbins,corrtype='Galauto',w1=w_g_rg,w2=w_g_rg,W1=W_g_rg,W2=W_g_rg,doNGPcorrect=True,Pmod=Pmod,Pnoise=0,b1=b_g,b2=b_g,f=f,sigv=sig_v,Tbar1=1,Tbar2=1,r=1,R_beam1=0,R_beam2=0,sig_N=0)
nbar = np.sum(n_g)/(lx*ly*lz) # Calculate number density inside survey footprint
P_SN = np.ones(len(k))*1/nbar # approximate shot-noise for errors (already subtracted in Pk estimator)
pkmod,k = model.PkMod(Pmod,dims_rg,kbins,b_g,b_g,f,sig_v,Tbar1=1,Tbar2=1,r=1,R_beam1=0,R_beam2=0,w1=w_g_rg,w2=w_g_rg,W1=W_g_rg,W2=W_g_rg,s_pix=s_pix,s_para=s_para,interpkbins=True,MatterRSDs=True,gridinterp=True)[0:2]
plt.plot(k,Pk_g)
plt.plot(k,pkmod,color='black',ls='--',label=r'Model [$\Omega_{\rm HI}b_{\rm HI} = %s \times 10^{-3}]$'%np.round(OmegaHI*b_HI*1e3,2))
plt.loglog()
plt.title('GAMA auto-correlation')
plt.xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
plt.ylabel(r'$P_{\rm g}(k)\,[h^{-3}{\rm Mpc}^{3}]$')
plt.axhline(0,lw=0.8,color='black')
plt.figure()

N_fgs = [6]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
for i in range(len(N_fgs)):
    print(i)
    MKmap_clean = foreground.PCAclean(MKmap,N_fg=N_fgs[i],W=W_HI,w=None)
    MKmap_clean_rg,dims_rg,dims0 = grid.cartesian(MKmap_clean,ra_map,dec_map,nu,wproj,W=W_regridfoot,ndim=ndim,Np=1)
    w_rg_HI,dims_rg,dims0 = grid.cartesian(w_HI,ra_map,dec_map,nu,wproj,W=W_regridfoot,ndim=ndim,Np=1)
    W_rg_HI,dims_rg,dims0 = grid.cartesian(W_HI,ra_map,dec_map,nu,wproj,W=W_regridfoot,ndim=ndim,Np=1)

    # Multiply Tukey taper window by all maps that undergo Fourier transforms
    MKmap_clean_rg,w_rg_HI,W_rg_HI = window*MKmap_clean_rg,window*w_rg_HI,window*W_rg_HI

    # Measure and plot power spectrum:
    norm = np.ones(nkbin)
    norm = k**2
    Pk_HI,k,nmodes = power.Pk(MKmap_clean_rg,MKmap_clean_rg,dims_rg,kbins,corrtype='HIauto',w1=w_rg_HI,w2=w_rg_HI,W1=W_rg_HI,W2=W_rg_HI,doNGPcorrect=True,Pmod=Pmod,Pnoise=0,b1=b_HI,b2=b_HI,f=f,sigv=sig_v,Tbar1=Tbar,Tbar2=Tbar,r=1,R_beam1=R_beam,R_beam2=R_beam,sig_N=0)
    Pk_X,k,nmodes = power.Pk(MKmap_clean_rg,n_g,dims_rg,kbins,corrtype='Cross',w1=w_rg_HI,w2=w_g_rg,W1=W_rg_HI,W2=W_g_rg,doNGPcorrect=True,Pmod=Pmod,Pnoise=0,b1=b_HI,b2=b_g,f=f,sigv=sig_v,Tbar1=Tbar,Tbar2=1,r=1,R_beam1=R_beam,R_beam2=0,sig_N=0)

    sig_err = 1/np.sqrt(2*nmodes) * np.sqrt( Pk_X**2 + Pk_HI*( Pk_g + P_SN ) ) # Error estimate
    plt.errorbar(k+(k/200*i),norm*Pk_X,norm*sig_err,label=r'$N_{\rm fg}=%s$'%N_fgs[i],ls='none',marker='o',color=colors[i])

pkmod,k = model.PkMod(Pmod,dims_rg,kbins,b_HI,b_g,f,sig_v,Tbar1=Tbar,Tbar2=1,r=1,R_beam1=R_beam,R_beam2=0,w1=w_rg_HI,w2=w_g_rg,W1=W_rg_HI,W2=W_g_rg,interpkbins=True,MatterRSDs=True,gridinterp=True)[0:2]
plt.plot(k,norm*pkmod,color='black',ls='--',label=r'Model [$\Omega_{\rm HI}b_{\rm HI} = %s \times 10^{-3}]$'%np.round(OmegaHI*b_HI*1e3,2))
if norm[0]==1.0: plt.loglog()
plt.legend(fontsize=12,loc='upper left',bbox_to_anchor=[1,1])
#plt.title('Null Test (shuffled GAMA redshifts) MeerKAT 2021 x GAMA (no TF)')
plt.xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
if norm[0]==1.0: plt.ylabel(r'$P_{\rm g,HI}(k)\,[{\rm mK}\,h^{-3}{\rm Mpc}^{3}]$')
else: plt.ylabel(r'$k^2\,P_{\rm g,HI}(k)\,[{\rm mK}\,h^{-1}{\rm Mpc}]$')
plt.axhline(0,lw=0.8,color='black')
plt.show()
exit()

### Calculate TF:
Nmock = 10
N_fg = N_fgs[-1]
TFfile = 'TFdata/Nfgs%s.npy'%N_fg
LoadTF = False
T_i,T_nosub_i,k = foreground.TransferFunction(MKmap,Nmock,TFfile,N_fg,LoadTF=LoadTF,corrtype='Cross',Pmod=Pmod,dims=dims,kbins=kbins,w_HI=w_HI,W_HI=W_HI,w_g=w_g,W_g=W_g,taper=window,regrid=True,ndim=ndim,W_regridfoot=W_regridfoot,b_HI=b_HI,b_g=b_g,f=f,Tbar=Tbar,Ngal=Ngal,ra=ra_map,dec=dec_map,nu=nu)

print(np.shape(T_i))
print(np.shape(k))

k = k[0]
for i in range(Nmock):
    plt.plot(k,T_i[i],lw=0.8,color='black')
plt.plot(k,np.mean(T_i,0))
plt.figure()

Pk_X_TF = Pk_X/np.mean(T_i,0)
plt.errorbar(k,norm*Pk_X_TF,norm*sig_err,label=r'$N_{\rm fg}=%s$'%N_fg,ls='none',marker='o')
plt.plot(k,norm*pkmod,color='black',ls='--',label=r'Model [$\Omega_{\rm HI}b_{\rm HI} = %s \times 10^{-3}]$'%np.round(OmegaHI*b_HI*1e3,2))
if norm[0]==1.0: plt.loglog()
plt.legend(fontsize=12,loc='upper left',bbox_to_anchor=[1,1])
plt.xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
if norm[0]==1.0: plt.ylabel(r'$P_{\rm g,HI}(k)\,[{\rm mK}\,h^{-3}{\rm Mpc}^{3}]$')
else: plt.ylabel(r'$k^2\,P_{\rm g,HI}(k)\,[{\rm mK}\,h^{-1}{\rm Mpc}]$')
plt.axhline(0,lw=0.8,color='black')
plt.show()
