import numpy as np
import matplotlib.pyplot as plt
import math
#'''
import scipy.ndimage
import scipy as sp
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import SkyCoord
import astropy.wcs
from astropy.wcs.utils import skycoord_to_pixel
from astropy.io import fits
from astropy.wcs import WCS
import astropy.coordinates as ac
#'''
from astropy.wcs.utils import pixel_to_skycoord
import astropy
from astropy.coordinates import SkyCoord
from astropy import units as u
import HItools
import model

import pmesh

#from astropy.cosmology import Planck15 as cosmo
#h = cosmo.H(0).value/100 # use to convert astopy Mpc distances to Mpc/h

import cosmo
h = cosmo.H(0)/100 # use to convert astopy Mpc distances to Mpc/h

v_21cm = 1420.405751#MHz

def comoving_dims(ra,dec,nu,wproj,ndim=None,W=None,frame='icrs'):
    '''Obtain lengths and origins of Cartesian comoving grid that encloses a
    sky map with (RA,Dec,nu) input coordinates for the map voxels'''
    # ndim = tuple of pixel dimensions (nx,ny,nz) use if want added to dims arrays
    # W = binary mask - dimensions will be calculated to close fit around only filled
    #      pixels (W==1) if this is given
    nra,ndec = np.shape(ra)
    nnu = len(nu)
    dnu = np.mean(np.diff(nu))
    # Produce particles at pixel indices (0,nra/ndec/nz) values then use astropy to convert back to (RA,Dec)
    xp = np.arange(0,nra)
    yp = np.arange(0,ndec)
    nu_p = np.copy(nu)
    xp,yp,nu_p = np.tile(xp[:,np.newaxis,np.newaxis],(1,ndec,nnu)),np.tile(yp[np.newaxis,:,np.newaxis],(nra,1,nnu)),np.tile(nu_p[np.newaxis,np.newaxis,:],(nra,ndec,1))
    if W is not None: xp,yp,nu_p = xp[W==1],yp[W==1],nu_p[W==1]
    # Cut core particles since only need edges of map to convert and obtain fitted grid:
    coremask = (xp>np.min(xp)) & (xp<np.max(xp)) & (yp>np.min(yp)) & (yp<np.max(yp)) & (nu_p>np.min(nu_p)) & (nu_p<np.max(nu_p))
    xp,yp,nu_p = xp[~coremask],yp[~coremask],nu_p[~coremask]
    # Extend boundaries particles by 10 map pixels in all directions for a buffer
    #   since later random particles (with assignment convolution) will be kicked
    #   way beyond cell centre and can fall off grid:
    xp[xp==np.min(xp)] -= 10
    xp[xp==np.max(xp)] += 10
    yp[yp==np.min(yp)] -= 10
    yp[yp==np.max(yp)] += 10
    nu_p[nu_p==np.min(nu_p)] -= 10*dnu
    nu_p[nu_p==np.max(nu_p)] += 10*dnu
    skycoords = pixel_to_skycoord(xp,yp,wproj)
    ra_p = skycoords.ra.degree
    dec_p = skycoords.dec.degree
    red_p = HItools.Freq2Red(nu_p)
    x_p,y_p,z_p = SkyCoordtoCartesian(ra_p,dec_p,red_p,ramean_arr=ra,decmean_arr=dec,frame=frame,doTile=False)
    x0,y0,z0 = np.min(x_p),np.min(y_p),np.min(z_p)
    lx,ly,lz = np.max(x_p)-x0, np.max(y_p)-y0, np.max(z_p)-z0
    if ndim is None: nx,ny,nz = nra,ndec,nnu
    else: nx,ny,nz = ndim
    dims = [lx,ly,lz,nx,ny,nz]
    dims0 = [lx,ly,lz,nx,ny,nz,x0,y0,z0]
    return dims,dims0

def cartesian(map,ra,dec,nu,wproj=None,nside=None,W=None,ndim=None,Np=3,frame='icrs',verbose=False):
    '''regrid (RA,Dec,z) map into comoving Cartesian coordinates (Lx,Ly,Lz [Mpc/h])'''
    # wproj: projection kernel to be used to WCS astropy coordinates
    # nside: healpix nside if using 1D healpix map coordinates
    ### Produce Np test particles per map voxel for regridding:
    if wproj is not None:
        ra_p,dec_p,nu_p,pixvals = SkyPixelParticles(ra,dec,nu,wproj,map=map,W=W,Np=Np)
        x_p,y_p,z_p = SkyCoordtoCartesian(ra_p,dec_p,HItools.Freq2Red(nu_p),ramean_arr=ra,decmean_arr=dec,frame=frame,doTile=False)
    if nside is not None:
        ra_p,dec_p,nu_p = SkyPixelParticles_healpy(ra,dec,nu,nside,map=map,W=W,Np=Np)
        x_p,y_p,z_p = SkyCoordtoCartesian(ra_p.to(u.deg).value,dec_p.to(u.deg).value,HItools.Freq2Red(nu_p),ramean_arr=ra.to(u.deg).value,decmean_arr=dec.to(u.deg).value,frame=frame,doTile=False)

    # Don't use particle mins/maxs to obtain dims, instead use the generic function
    #   below to ensure consistent dimensions throughout code rather than being
    #   random particle dependent:
    dims,dims0 = comoving_dims(ra,dec,nu,wproj,ndim=ndim,W=W,frame=frame)
    lx,ly,lz,nx,ny,nz,x0,y0,z0 = dims0
    xbins,ybins,zbins = np.linspace(x0,lx+x0,nx+1),np.linspace(y0,ly+y0,ny+1),np.linspace(z0,lz+z0,nz+1)
    physmap = np.histogramdd((x_p,y_p,z_p),bins=(xbins,ybins,zbins),weights=pixvals)[0]
    # Average the physmap since multiple map values may enter same cartesian pixel:
    counts = np.histogramdd((x_p,y_p,z_p),bins=(xbins,ybins,zbins))[0]

    '''
    print(np.sum(counts))
    print(len(x_p))
    print('----')
    print(np.min(xbins),np.max(xbins))
    print(np.min(ybins),np.max(ybins))
    print(np.min(zbins),np.max(zbins))
    print('----')
    print(np.min(x_p),np.max(x_p))
    print(np.min(y_p),np.max(y_p))
    print(np.min(z_p),np.max(z_p))
    print('----')
    plt.scatter(x_p,y_p)
    plt.scatter(x_p[x_p<np.min(xbins)],y_p[x_p<np.min(xbins)],color='red')
    plt.scatter(x_p[x_p>np.max(xbins)],y_p[x_p>np.max(xbins)],color='red')
    plt.scatter(x_p[y_p<np.min(ybins)],y_p[y_p<np.min(ybins)],color='red')
    plt.scatter(x_p[y_p>np.max(ybins)],y_p[y_p>np.max(ybins)],color='red')
    plt.show()
    exit()
    '''

    physmap[physmap!=0] = physmap[physmap!=0]/counts[physmap!=0]
    #physmap[physmap!=0] = physmap[physmap!=0] / np.sqrt(counts[physmap!=0])


    if verbose==True:
        print('\nCartesian regridding summary:')
        print(' - Minimum number of particles in grid cell: '+str(np.min(counts)))
        print(' - Maximum number of particles in grid cell: '+str(np.max(counts)))
        print(' - Mean number of particles in grid cell: '+str(np.round(np.mean(counts),3)))
        print(' - Number of missing particles: '+str(int(len(ra_p) - np.sum(counts))))
    dims = [lx,ly,lz,nx,ny,nz]
    dims0 = [lx,ly,lz,nx,ny,nz,x0,y0,z0]
    return physmap,dims,dims0
    #return physmap,counts,dims,dims0

def lightcone(physmap,dims0,ra,dec,nu,wproj,W=None,Np=3,frame='icrs',verbose=False):
    '''Regrid density/temp field in comoving [Mpc/h] cartesian space, into lightcone with (RA,Dec,z)'''
    ### Produce Np test particles per map voxel for regridding:
    ra_p,dec_p,nu_p = SkyPixelParticles(ra,dec,nu,wproj,map=None,W=W,Np=Np)
    red_p = HItools.Freq2Red(nu_p)
    x_p,y_p,z_p = SkyCoordtoCartesian(ra_p,dec_p,red_p,ramean_arr=ra,decmean_arr=dec,frame=frame,doTile=False)
    ### Bin particles into Cartesian bins to match each with input cell values:
    lx,ly,lz,nx,ny,nz,x0,y0,z0 = dims0
    dx,dy,dz = lx/nx,ly/ny,lz/nz
    xbins = np.linspace(x0,x0+lx,nx+1)
    ybins = np.linspace(y0,y0+ly,ny+1)
    zbins = np.linspace(z0,z0+lz,nz+1)
    ixbin = np.digitize(x_p,xbins)-1
    iybin = np.digitize(y_p,ybins)-1
    izbin = np.digitize(z_p,zbins)-1
    ### Remove any particles beyond physmap boarder (may happen if producing simulation
    #     and physmap defined as the trimmed map region but a lightcone for the full
    #     footprint is demanded so edge effects from e.g. beam can be thrown away
    cutmask = (ixbin!=-1) & (ixbin!=nx) & (iybin!=-1) & (iybin!=ny) & (izbin!=-1) & (izbin!=nz)
    ixbin,iybin,izbin = ixbin[cutmask],iybin[cutmask],izbin[cutmask]
    ra_p,dec_p,red_p = ra_p[cutmask],dec_p[cutmask],red_p[cutmask]
    ## Gather cellvals within footprint and map:
    cellvals = physmap[ixbin,iybin,izbin] # cell values associated with each particle
    map,counts = AstropyGridding(ra_p,dec_p,red_p,nu,particleweights=cellvals,obsdata='2021')
    # Average the map since multiple particle values may enter same pixel:
    map[map!=0] = map[map!=0]/counts[map!=0]
    if verbose==True:
        print('\nLightcone sampling summary:')
        print(' - Minimum number of particles in map voxel: '+str(np.min(counts)))
        print(' - Maximum number of particles in map voxel: '+str(np.max(counts)))
        print(' - Mean number of particles in map voxel: '+str(np.round(np.mean(counts),3)))
        print(' - Number of missing particles: '+str(int(len(ra_p) - np.sum(counts))))
    return map

def healpy_gaussian(ra,dec,nu,nside):
    '''Produce a Gaussian random field with sigma=1 on healpy map at input coordinates'''
    import astropy_healpix
    from astropy_healpix import HEALPix
    hp0 = HEALPix(nside)
    ipix = hp0.lonlat_to_healpix(ra,dec)

    hpmap = np.zeros((hp0.npix,len(nu)))
    for i in range(len(nu)):
        np.add.at( hpmap[:,i] , ipix , np.random.normal(0,1,len(ipix)) )

    return hpmap


'''
def lightcone_new(physmap,dims0,ra,dec,nu,wproj,W=None,Np=3,frame='icrs',verbose=False):
    #Use even number of cell values
    #Regrid density/temp field in comoving [Mpc/h] cartesian space, into lightcone with (RA,Dec,z)
    ### Produce Np test particles per cell for regridding to sky voxel:

    xp,yp,zp,cellvals = GridParticles(dims0,delta=physmap,W=W,Np=Np)

    ra_p,dec_p,red_p = Cart2SphericalCoords(xp,yp,zp,ra,dec,frame)

    map,counts = AstropyGridding(ra_p,dec_p,red_p,nu,particleweights=cellvals/Np,obsdata='2021')

    #map[map!=0] = map[map!=0]/counts[map!=0]

    if verbose==True:
        print('\nLightcone sampling summary:')
        print(' - Minimum number of particles in map voxel: '+str(np.min(counts)))
        print(' - Maximum number of particles in map voxel: '+str(np.max(counts)))
        print(' - Mean number of particles in map voxel: '+str(np.round(np.mean(counts),3)))
        print(' - Number of missing particles: '+str(int(len(ra_p) - np.sum(counts))))
    return map
    #return map,counts
'''

def transform(delta,dims0,dims0_new):
    '''Regrid input cartesian voxels onto another cartesian grid with different dimensions.
    Used for trimming large high-res mock field onto courser observation field'''
    lx,ly,lz,nx,ny,nz,x0,y0,z0 = dims0
    xbins,ybins,zbins = np.linspace(x0,x0+lx,nx+1),np.linspace(y0,y0+ly,ny+1),np.linspace(z0,z0+lz,nz+1)
    xp,yp,zp = (xbins[1:] + xbins[:-1])/2,(ybins[1:] + ybins[:-1])/2,(zbins[1:] + zbins[:-1])/2  #centre of voxels
    xp,yp,zp = np.tile(xp[:,np.newaxis,np.newaxis],(1,ny,nz)),np.tile(yp[np.newaxis,:,np.newaxis],(nx,1,nz)),np.tile(zp[np.newaxis,np.newaxis,:],(nx,ny,1))
    xp,yp,zp,delta = np.ravel(xp),np.ravel(yp),np.ravel(zp),np.ravel(delta)
    # Define voxel boundaries for new grid to sample onto:
    lx,ly,lz,nx,ny,nz,x0,y0,z0 = dims0_new
    xbins,ybins,zbins = np.linspace(x0,x0+lx,nx+1),np.linspace(y0,y0+ly,ny+1),np.linspace(z0,z0+lz,nz+1)
    delta_rg = np.histogramdd((xp,yp,zp),bins=(xbins,ybins,zbins),weights=delta)[0]
    counts = np.histogramdd((xp,yp,zp),bins=(xbins,ybins,zbins))[0]
    delta_rg[counts!=0] /= counts[counts!=0]
    return delta_rg

def SkyPixelParticles(ra,dec,nu,wproj,map=None,W=None,Np=1):
    '''Create particles that lie in centre of ra,dec,nu cells, then randomly generate
    additional particles kicked by random half-pixel distances away from pixel centre'''
    # Np = number of particles generated in each cell (default 1 only assigns particles at cell centre)
    nra,ndec = np.shape(ra)
    nz = len(nu)
    dnu = np.mean(np.diff(nu))
    # Produce particles at pixel indices (0,nra/ndec/nz) values then use astropy to convert back to (RA,Dec)
    xp = np.arange(0,nra)
    yp = np.arange(0,ndec)
    xp,yp,nu_p = np.tile(xp[:,np.newaxis,np.newaxis],(1,ndec,nz)),np.tile(yp[np.newaxis,:,np.newaxis],(nra,1,nz)),np.tile(nu[np.newaxis,np.newaxis,:],(nra,ndec,1))
    if Np==1: xp,yp,nu_p = np.ravel(xp),np.ravel(yp),np.ravel(nu_p)
    if Np>1:
        xpkick,ypkick,nupkick = np.array([]),np.array([]),np.array([])
        for i in range(Np-1):
            xpkick = np.append(xpkick, xp + np.random.uniform(-0.5,0.5,np.shape(xp)) )
            ypkick = np.append(ypkick, yp + np.random.uniform(-0.5,0.5,np.shape(yp)) )
            nupkick = np.append(nupkick, nu_p + np.random.uniform(-dnu/2,dnu/2,np.shape(nu_p)) )
        xp = np.append(xp,xpkick)
        yp = np.append(yp,ypkick)
        nu_p = np.append(nu_p,nupkick)
    if map is not None or W is not None:
        ixbin = np.rint(xp).astype(int) # integising assigns random particle to pixel index
        iybin = np.rint(yp).astype(int) #    used to associate each pixel with its cell value
        nubins = np.linspace(nu[0]-dnu/2,nu[-1]+dnu/2,len(nu)+1)
        inubin = np.digitize(nu_p,nubins)-1
    if map is not None: pixvals = map[ixbin,iybin,inubin] # cell values associated with each particle
    if W is not None:
        W_p = W[ixbin,iybin,inubin] # use to discard particles from empty pixels
        xp,yp,nu_p = xp[W_p==1],yp[W_p==1],nu_p[W_p==1]
        if map is not None: pixvals = pixvals[W_p==1]
    skycoords = pixel_to_skycoord(xp,yp,wproj)
    ra_p = skycoords.ra.degree
    dec_p = skycoords.dec.degree
    if map is None: return ra_p,dec_p,nu_p
    if map is not None: return ra_p,dec_p,nu_p,pixvals

def get_healpy_grid_dims(ra,dec,nu,nside,d_c,dims0):
    '''Approximate the healpy pixel dimensions in Mpc/h for modelling'''
    # d_c: comoving distances to each frequnecy channel
    import astropy_healpix
    from astropy_healpix import HEALPix
    hp0 = HEALPix(nside)
    ipix = hp0.lonlat_to_healpix(ra,dec)
    dang = hp0.pixel_resolution.to(u.deg).value
    s_pix = np.mean(d_c) * np.radians(dang)
    s_para = np.mean( d_c[:-1] - d_c[1:] )
    lx,ly,lz,nx,ny,nz,x0,y0,z0 = dims0
    nxhp,nyhp,nzhp = round(lx/s_pix), round(ly/s_pix), round(lz/s_para)
    if nzhp % 2 != 0: # number is odd. Needs to be even
        if int(lz/s_para)!=nzhp: # originally rounded up, so round down to nearest even intg
            nzhp = int(lz/s_para)
        if int(lz/s_para)==nzhp: # originall rounded down, so round up to nearest even intg
            nzhp = round(1 + lz/s_para)
    return [lx,ly,lz,nxhp,nyhp,nzhp,x0,y0,z0]

def SkyCoordtoCartesian(ra_,dec_,z,ramean_arr=None,decmean_arr=None,doTile=True,LoScentre=True,frame='icrs'):
    '''***From gridimp: https://github.com/stevecunnington/gridimp/blob/main/gridimp/grid.py***'''
    '''Convert (RA,Dec,z) sky coordinates into Cartesian (x,y,z) comoving coordinates
    with [Mpc/h] units.
    doTile: set True (default) if input (ra,dec,z) are coordinates of map pixels of lengths ra=(nx,ny),dec=(nz,ny),z=nz)
            set False if input are test particles/galaxy coordinates already with equal length (RA,Dec,z) for every input
    LoScentre: set True (default) to align footprint with ra=dec=0 so LoS is aligned with
                one axis (x-axis by astropy default)
    ramean_arr/decmean_arr: arrays to use for mean ra/dec values. Use if want to subtract the exact same means as done for
                              another map e.g if gridding up galaxy map and want to subtract the IM mean for consistency.
    '''
    ra = np.copy(ra_);dec = np.copy(dec_) # Define new arrays so amends don't effect global coordinates
    if ramean_arr is None: ramean_arr = np.copy(ra)
    if decmean_arr is None: decmean_arr = np.copy(dec)
    if LoScentre==True: # subtract RA Dec means to align the footprint with ra=dec=0 for LoS
        ra[ra>180] = ra[ra>180] - 360 # Make continuous RA i.e. 359,360,1 -> -1,0,1 so mean RA is correct
        ramean_arr[ramean_arr>180] = ramean_arr[ramean_arr>180] - 360 # Make continuous RA i.e. 359,360,1 -> -1,0,1 so mean RA is correct
        ra = ra - np.mean(ramean_arr)
        ra[ra<0] = ra[ra<0] + 360 # Reset negative coordinates to 359,360,1 convention
        ramean_arr[ramean_arr<0] = ramean_arr[ramean_arr<0] + 360 # Reset negative coordinates to 359,360,1 convention
        dec = dec - np.mean(decmean_arr)
    #d = cosmo.comoving_distance(z).value
    d = cosmo.d_com(z)/h # [Mpc]
    # Build array in shape of map to assign each entry a Cartesian (x,y,z) coordinate:
    if doTile==True:
        nx,ny = np.shape(ra)
        nz = len(z)
        ra = np.repeat(ra[:, :, np.newaxis], nz, axis=2)
        dec = np.repeat(dec[:, :, np.newaxis], nz, axis=2)
        d = np.tile(d[np.newaxis,np.newaxis,:],(nx,ny,1))
    c = SkyCoord(ra*u.degree, dec*u.degree, d*u.Mpc, frame=frame)
    # Astropy does x-axis as LoS by default, so change this by assigning z=x, x=y, y=z:
    z,x,y = c.cartesian.x.value*h, c.cartesian.y.value*h, c.cartesian.z.value*h
    return x,y,z

def mesh(x,y,z,T=None,dims=None,window='nnb',compensate=True,interlace=False,verbose=False):
    '''***From gridimp: https://github.com/stevecunnington/gridimp/blob/main/gridimp/grid.py***'''
    '''Utilises pmesh to place interpolate particles onto a grid for a given
    interpolation window function'''
    # window options are zeroth to 3rd order: ['ngp','cic','tsc','pcs']
    lx,ly,lz,nx,ny,nz,x0,y0,z0 = dims
    if window=='ngp': window = 'nnb' # pmesh uses nnb for nearest grid point
    # Correct for pmesh half-cell shifting in output numpy array:
    Hx,Hy,Hz = lx/nx,ly/ny,lz/nz
    pos = np.swapaxes( np.array([x-x0-Hx/2,y-y0-Hy/2,z-z0-Hz/2]), 0,1)
    # Use pmesh to create field and paint with chosed anssigment window:
    pm0 = pmesh.pm.ParticleMesh(BoxSize=[lx,ly,lz], Nmesh=[nx,ny,nz])
    real1 = pmesh.pm.RealField(pm0)
    real1[:] = 0
    counts = pm0.paint(pos, resampler=window)
    if T is not None: # assign temps (not galaxy counts) and normalise
        pm0.paint(pos, mass=T, resampler=window, hold=True, out=real1)
        ### Normalise by particle count entering cell:
        real1[counts!=0] /= counts[counts!=0]
    else: # Galaxy case, map is should just count ones
        pm0.paint(pos, mass=np.ones(len(x)), resampler=window, hold=True, out=real1)
    # Create binary mask (W01), required because convolved pmesh is non-zero everywhere:
    xbins,ybins,zbins = np.linspace(x0,x0+lx,nx+1),np.linspace(y0,y0+ly,ny+1),np.linspace(z0,z0+lz,nz+1)
    W01 = np.histogramdd((x,y,z),bins=(xbins,ybins,zbins))[0]
    W01[W01!=0] = 1
    if verbose==True:
        counts = counts.preview()
        print('\nCartesian regridding summary:')
        print(' - Minimum number of particles in grid cell: '+str(np.min(counts)))
        print(' - Maximum number of particles in grid cell: '+str(np.max(counts)))
        print(' - Mean number of particles in grid cell: '+str(np.round(np.mean(counts[counts!=0]),3)))
        print(' - Number of missing particles: '+str(int(len(x) - np.sum(counts))))
    if compensate==True: # apply W(k) correction in Fourier space to field:
        c1 = real1.r2c()
        c1 /= model.W_mas(dims,window)
        if interlace==True: # Create a second shifted field (following Nbodykit:
            # https://github.com/bccp/nbodykit/blob/376c9d78204650afd9af81d148b172804432c02f/nbodykit/source/mesh/catalog.py#L11
            real2 = pmesh.pm.RealField(pm0)
            real2[:] = 0
            shifted = pm0.affine.shift(0.5)
            counts = pm0.paint(pos, resampler=window, transform=shifted)
            if T is not None: # assign temps (not galaxy counts) and normalise
                pm0.paint(pos, mass=T, resampler=window, transform=shifted, hold=True, out=real2)
                ### Normalise by particle count entering cell:
                real2[counts!=0] /= counts[counts!=0]
            else: # Galaxy case, map is should just count ones
                pm0.paint(pos, mass=np.ones(len(x)), resampler=window, hold=True, out=real2)
            ### Apply W(k) correction in Fourier space to field:
            c2 = real2.r2c()
            c2 /= model.W_mas(dims,window)
            # Interlace both fields (again following NBK example):
            H = [lx/nx,ly/ny,lz/nz]
            for k, s1, s2 in zip(c1.slabs.x, c1.slabs, c2.slabs):
                kH = sum(k[i] * H[i] for i in range(3))
                s1[...] = s1[...] * 0.5 + s2[...] * 0.5 * np.exp(0.5 * 1j * kH)
        c1.c2r(real1) # FFT back to real-space
    map = real1.preview()
    return map,W01,counts

def AstropyGridding(ra_p,dec_p,red_p,nu,particleweights=None,obsdata=None):
    '''Gridding the rac,dec pointings (or particles) from katcali example notebook:
    https://github.com/meerklass/katcali/blob/master/examples/level5/katcali_multi_level5.ipynb
    https://github.com/meerklass/katcali/blob/master/examples/MeerKLASS2021/python3/level5/KATcali_multi_level5_py3.ipynb
    '''
    # obsdata: chose either '2019' or '2021' for correct sky placement
    # particleweights: for histogram binning. Use if constructing sim with M_HI for each galaxy
    pix_deg = 0.3
    #set the sky area to be pixelized - SETUP FOR 11hr FIELD PILOT DATA:
    if obsdata=='2019':
        x_cen = 163 #deg #RA
        x_half = 20 #deg
        y_cen = 3.5 #deg #DEC
        y_half = 6 #deg
    if obsdata=='2021':
        x_cen=-18 #deg #RA
        x_half=20 #deg
        y_cen=-34 #deg #DEC
        y_half=11 #deg
    N_half_x = int(x_half/pix_deg)
    N_half_y = int(y_half/pix_deg)
    nx = 2*N_half_x+1
    ny = 2*N_half_y+1
    nz = len(nu)
    # Create bin boundaries at +/- 0.5 of integer pixels values (+/- dnu/2 from channel centres):
    xbins,ybins = np.arange(-0.5,nx+0.5,1),np.arange(-0.5,ny+0.5,1)
    dnu = nu[1]-nu[0]
    nubins = np.linspace(nu[0]-dnu/2,nu[-1]+dnu/2,nz+1)
    zbins = (v_21cm/nubins) - 1
    w = astropy.wcs.WCS(naxis=2)
    w.wcs.crval = [x_cen-x_half, y_cen-y_half] # reference pointing of the image #deg
    w.wcs.crpix = [1.0, 1.0] # pixel index corresponding to the reference pointing (try either 1 or 0 to see if the behaviour agrees to your expectation!)
    w.wcs.cdelt = np.array([pix_deg, pix_deg]) # resolution
    w.wcs.ctype = ['RA---ZEA', 'DEC--ZEA'] #projection
    ##check the (min ra, min dec) of sky area will fall into pix (0,0)
    p0 = ac.SkyCoord(ra=(x_cen-x_half)*u.deg, dec=(y_cen-y_half)*u.deg)
    ref_p = skycoord_to_pixel(p0, w)
    assert(ref_p[0]<1e-12) #should be zero
    assert(ref_p[1]<1e-12) #shoule be zero
    map = np.zeros((nx,ny,nz))
    counts = np.zeros((nx,ny,nz))
    for i in range(nz):
        zmask = (red_p>zbins[i+1]) & (red_p<zbins[i])
        p_list = ac.SkyCoord(ra=ra_p[zmask]*u.deg, dec=dec_p[zmask]*u.deg)
        x_pix_list,y_pix_list = skycoord_to_pixel(p_list,w) #observation (ra,dec) to pix
        map[:,:,i] = np.histogram2d(x_pix_list,y_pix_list,bins=(xbins,ybins),weights=particleweights[zmask])[0]
        counts[:,:,i] = np.histogram2d(x_pix_list,y_pix_list,bins=(xbins,ybins))[0] # use for averaging particle values
    return map,counts

def CartesianGridGalaxies(ra_g,dec_g,red_g,dims0,ra,dec,frame='icrs'):
    '''Place input galaxy coordinates directly onto Cartesian comoving grid '''
    #ra,dec: Array of coordinates from intensity map to use in subtracting mean
    #          coordinates in sky->cartesian conversion
    x_g,y_g,z_g = SkyCoordtoCartesian(ra_g,dec_g,red_g,doTile=False,ramean_arr=ra,decmean_arr=dec)
    lx,ly,lz,nx,ny,nz,x0,y0,z0 = dims0
    xbins,ybins,zbins = np.linspace(x0,x0+lx,nx+1),np.linspace(y0,y0+ly,ny+1),np.linspace(z0,z0+lz,nz+1)
    galmap = np.histogramdd((x_g,y_g,z_g),bins=(xbins,ybins,zbins))[0]
    return galmap

def Cart2SphericalCoords(xcoord,ycoord,zcoord,ramean_arr,decmean_arr,frame='icrs'):
    '''Mainly used for converting galaxy sims in comoving space into sky light-cone (ra,dec,z) coords
    '''
    ### Following astropy example in: https://docs.astropy.org/en/stable/coordinates/
    ### *** Input is assumed in units of Mpc/h so a conversion is done in to Mpc for astopy convention
    # ramean_arr/decmean_arr: used to centre the ra/dec coordinates to mean of footprint
    # Feed x=zcoord for astropy convention that x-axis is along LoS:
    c = SkyCoord(x=zcoord/h, y=xcoord/h, z=ycoord/h, unit='Mpc', frame=frame, representation_type='cartesian')
    c.representation_type = 'spherical'
    ra,dec,dist = c.ra.value,c.dec.value,c.distance.value
    # Convert distance into a redshift:
    redarr = np.linspace(0,2,1000) # assumes redshift doesn't exceed 2
    #distarr = cosmo.comoving_distance(redarr).value
    distarr = cosmo.d_com(redarr)/h # [Mpc]
    z = np.interp(dist,distarr,redarr)
    # Add on RA Dec map means to unalign the phys-footprint with ra=dec=0 and align coordinates on footprint centre
    ra[ra>180] = ra[ra>180] - 360 # Make continuous RA i.e. 359,360,1 -> -1,0,1 so mean RA is correct
    ramean_arr[ramean_arr>180] = ramean_arr[ramean_arr>180] - 360 # Make continuous RA i.e. 359,360,1 -> -1,0,1 so mean RA is correct
    ra = ra + np.mean(ramean_arr)
    ra[ra<0] = ra[ra<0] + 360 # Reset negative coordinates to 359,360,1 convention
    ramean_arr[ramean_arr<0] = ramean_arr[ramean_arr<0] + 360 # Reset negative coordinates to 359,360,1 convention
    dec = dec + np.mean(decmean_arr)
    return ra,dec,z

def GridParticles(dims0,delta=None,W=None,Np=1):
    '''Create particles that lie in centre of x,y,z cells and then randomly generate
    additional particles kicked by random half-cell distance away from cell centre'''
    lx,ly,lz,nx,ny,nz,x0,y0,z0 = dims0
    dx,dy,dz = lx/nx,ly/ny,lz/nz
    xbins = np.linspace(x0,x0+lx,nx+1)
    ybins = np.linspace(y0,y0+ly,ny+1)
    zbins = np.linspace(z0,z0+lz,nz+1)
    # First create particles at cell centres:
    xp,yp,zp = (xbins[1:]+xbins[:-1])/2,(ybins[1:]+ybins[:-1])/2,(zbins[1:]+zbins[:-1])/2 #centre of bins
    xp,yp,zp = np.tile(xp[:,np.newaxis,np.newaxis],(1,ny*Np,nz*Np)),np.tile(yp[np.newaxis,:,np.newaxis],(nx*Np,1,nz*Np)),np.tile(zp[np.newaxis,np.newaxis,:],(nx*Np,ny*Np,1))
    if Np==1: xp,yp,zp = np.ravel(xp),np.ravel(yp),np.ravel(zp)
    if Np>1:
        xpkick,ypkick,zpkick = np.array([]),np.array([]),np.array([])
        for i in range(Np-1):
            xpkick = np.append(xpkick, xp + np.random.uniform(-dx/2,dx/2,np.shape(xp)) )
            ypkick = np.append(ypkick, yp + np.random.uniform(-dy/2,dy/2,np.shape(yp)) )
            zpkick = np.append(zpkick, zp + np.random.uniform(-dz/2,dz/2,np.shape(zp)) )
        xp = np.append(xp,xpkick)
        yp = np.append(yp,ypkick)
        zp = np.append(zp,zpkick)
    '''
    ### Sanity plot showing particles on grid for one z-slice:
    izbin = np.digitize(zp,zbins)-1
    plt.scatter(xp[izbin==0],yp[izbin==0],s=1)
    for i in range(len(xbins)):
        plt.axvline(xbins[i],color='grey',lw=1)
    for i in range(len(ybins)):
        plt.axhline(ybins[i],color='grey',lw=1)
    plt.show()
    exit()
    '''
    if delta is None: return xp,yp,zp
    else:
        ixbin = np.digitize(xp,xbins)-1
        iybin = np.digitize(yp,ybins)-1
        izbin = np.digitize(zp,zbins)-1
        cellvals = delta[ixbin,iybin,izbin] # cell values associated with each particle
        if W is not None:
            W_p = W[ixbin,iybin,izbin] # use to discard particles from empty pixels
            xp,yp,zp,cellvals = xp[W_p==1],yp[W_p==1],zp[W_p==1],cellvals[W_p==1]
        return xp,yp,zp,cellvals

def UniformGridParticles(dims0,delta=None,Np=2):

    # Produce evenly distributed particles throughout each cell
    ### Np!=1 gives results with a differing amplitude to input model power spectrum - RECHECK:
    lx,ly,lz,nx,ny,nz,x0,y0,z0 = dims0
    xp,yp,zp = np.linspace(x0,x0+lx,nx*Np+1),np.linspace(y0,y0+ly,ny*Np+1),np.linspace(z0,z0+lz,nz*Np+1)
    xp,yp,zp = (xp[1:]+xp[:-1])/2,(yp[1:]+yp[:-1])/2,(zp[1:]+zp[:-1])/2 #centre of bins
    xp,yp,zp = np.tile(xp[:,np.newaxis,np.newaxis],(1,ny*Np,nz*Np)),np.tile(yp[np.newaxis,:,np.newaxis],(nx*Np,1,nz*Np)),np.tile(zp[np.newaxis,np.newaxis,:],(nx*Np,ny*Np,1))
    '''
    ### Plot to check random particles are being distributed correctly:
    plt.scatter(xp,yp,s=1)
    xbins = np.linspace(x0,x0+lx,nx+1)
    ybins = np.linspace(y0,y0+ly,ny+1)
    for i in range(len(xbins)):
        plt.axvline(xbins[i],color='grey',lw=1)
    for i in range(len(ybins)):
        plt.axhline(ybins[i],color='grey',lw=1)
    plt.show()
    exit()
    '''
    if delta is None: return np.ravel(xp),np.ravel(yp),np.ravel(zp)
    else:
        # Replicate host cells delta values each produced particle (these are later normalised):
        cellvals = np.repeat(delta,Np,axis=0)
        cellvals = np.repeat(cellvals,Np,axis=1)
        cellvals = np.repeat(cellvals,Np,axis=2)
        return np.ravel(xp),np.ravel(yp),np.ravel(zp),np.ravel(cellvals)

def SkyPixelParticlesOld(ra,dec,z,map=None,niter=2):
    ''' Produce evenly distributed particles inbetween each (RA,Dec) pixel coordinate
    # niter: number of iterations to loop over, each one adds particles between the mid
    positions of the previous particles. Quickly becomes expensive so keep low.
    '''

    ra[ra>180] = ra[ra>180] - 360 # Make continuous RA i.e. 359,360,1 -> -1,0,1 so mean RA is correct
    for i in range(niter):
        if i>0: # use previous particle positions and values beyond the first iteration
            ra = np.copy(ra_p)
            dec = np.copy(dec_p)
            if map is not None: map = np.copy(pixvals)
        nra,ndec = np.shape(ra)
        ra_p = np.zeros((2*(nra-1)+1,2*(ndec-1)+1))
        dec_p = np.zeros((2*(nra-1)+1,2*(ndec-1)+1))
        if map is not None: pixvals = np.zeros((2*(nra-1)+1,2*(ndec-1)+1,len(z)))
        # Loop over all coordinates/previous iterations particles and add mid-point particles
        for i in range(nra):
            for j in range(ndec):
                ra_p[i*2,j*2] = ra[i,j]
                dec_p[i*2,j*2] = dec[i,j]
                if i<(nra-1): ra_p[i*2+1,j*2] = np.mean(ra[i:i+2,j])
                if j<(ndec-1): dec_p[i*2,j*2+1] = np.mean(dec[i,j:j+2])
                if j<(ndec-1): ra_p[i*2,j*2+1] = np.mean(ra[i,j:j+2])
                if i<(nra-1): dec_p[i*2+1,j*2] = np.mean(dec[i:i+2,j])
                if (i<(nra-1)) & (j<(ndec-1)):
                    ra_p[i*2+1,j*2+1] = np.mean(ra[i:i+2,j:j+2])
                    dec_p[i*2+1,j*2+1] = np.mean(dec[i:i+2,j:j+2])
                if map is not None:
                    pixvals[i*2,j*2,:] = map[i,j,:]
                    if i<(nra-1): pixvals[i*2+1,j*2,:] = np.mean(map[i:i+2,j,:],axis=0)
                    if j<(ndec-1): pixvals[i*2,j*2+1,:] = np.mean(map[i,j:j+2,:],axis=0)
                    if (i<(nra-1)) & (j<(ndec-1)): pixvals[i*2+1,j*2+1,:] = np.mean(map[i:i+2,j:j+2,:],axis=(0,1))

    ra_p[ra_p<0] = ra_p[ra_p<0] + 360 # Reset negative coordinates to 359,360,1 convention

    nz = len(z)*niter
    nra,ndec = np.shape(ra_p)

    ra_p,dec_p = np.tile(ra_p[:,:,np.newaxis],(1,1,nz)),np.tile(dec_p[:,:,np.newaxis],(1,1,nz))
    z_p = np.tile(z[np.newaxis,np.newaxis,:],(nra,ndec,1))
    if map is None: return np.ravel(ra_p),np.ravel(dec_p),np.ravel(z_p)
    else:
        pixvals = np.tile(pixvals[:,:,np.newaxis],(1,1,niter))
        return np.ravel(ra_p),np.ravel(dec_p),np.ravel(z_p),np.ravel(pixvals)
