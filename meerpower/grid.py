import numpy as np
'''
import scipy.ndimage
import scipy as sp
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import SkyCoord
import astropy.wcs
from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs.utils import pixel_to_skycoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.coordinates as ac
'''

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
import HItools
h = cosmo.H(0).value/100 # use to convert astopy Mpc distances to Mpc/h

def comoving(map,ra,dec,nu,frame='icrs',W=None,ndim=None):
    '''regrid (RA,Dec,z) map into comoving Cartesian coordinates (Lx,Ly,Lz [Mpc/h])'''
    if ndim is None: nx,ny,nz = np.shape(map)
    else: nx,ny,nz = ndim
    z = HItools.Freq2Red(nu)
    x,y,z = SkyCoordtoCartesian(ra,dec,z,frame=frame)
    x,y,z,map = np.ravel(x),np.ravel(y),np.ravel(z),np.ravel(map)
    if W is not None: # use window to only grid filled cartesian pixels
        W = np.ravel(W)
        x,y,z,map = x[W==1],y[W==1],z[W==1],map[W==1]
    x0,y0,z0 = np.min(x),np.min(y),np.min(z)
    lx,ly,lz = np.max(x)-x0, np.max(y)-y0, np.max(z)-z0
    xbins,ybins,zbins = np.linspace(np.min(x),np.max(x),nx+1),np.linspace(np.min(y),np.max(y),ny+1),np.linspace(np.min(z),np.max(z),nz+1)
    physmap = np.histogramdd((x,y,z),bins=(xbins,ybins,zbins),weights=map)[0]
    # Average the physmap since multiple map values may enter same cartesian pixel:
    counts = np.histogramdd((x,y,z),bins=(xbins,ybins,zbins))[0]
    physmap[physmap!=0] = physmap[physmap!=0]/counts[physmap!=0]
    dims = [lx,ly,lz,nx,ny,nz]
    dims0 = [lx,ly,lz,nx,ny,nz,x0,y0,z0]
    return physmap,dims,dims0

def SkyCoordtoCartesian(ra_,dec_,z,ramean_arr=None,decmean_arr=None,doTile=True,LoScentre=True,frame='icrs'):
    '''Convert (RA,Dec,z) sky coordinates into Cartesian (x,y,z) comoving coordinates
    with [Mpc/h] units.
    doTile: set True (default) if input (ra,dec,z) are coordinates of map pixels of lengths ra=(nx,ny),dec=(nz,ny),z=nz)
            set False if input are galaxy coordinates already with equal length (RA,Dec,z) for every input
    LoScentre: set True (default) to align footprint with ra=dec=0 so LoS is aligned with
                one axis (x-axis by astropy default)
    ramean_arr/decmean_arr: arrays to use for mean ra/dec values. Use if want to subtract the exact same means as done for
                              another map e.g if gridding up galaxy map and want to subtract the IM mean for consistency.
    '''
    ra = np.copy(ra_);dec = np.copy(dec_) # Define new arrays so amends don't effect global coordinates
    if ramean_arr is None: ramean_arr = ra
    if decmean_arr is None: decmean_arr = dec
    if LoScentre==True: # subtract RA Dec means to align the footprint with ra=dec=0 for LoS
        if np.max(abs(np.diff(ra,0)))>180: # Jump in coordinates i.e. 359,360,1 - need to correct for mean centring
            ra[ra>180] = ra[ra>180] - 360 # Make continuous RA i.e. 359,360,1 -> -1,0,1 so mean RA is correct
            ramean_arr[ramean_arr>180] = ramean_arr[ramean_arr>180] - 360 # Make continuous RA i.e. 359,360,1 -> -1,0,1 so mean RA is correct
            ra = ra - np.mean(ramean_arr)
            ra[ra<0] = ra[ra<0] + 360 # Reset negative coordinates to 359,360,1 convention
        else: ra = ra - np.mean(ramean_arr)
        dec = dec - np.mean(decmean_arr)
    d = cosmo.comoving_distance(z).value
    # Build array in shape of map to assign each entry a Cartesian (x,y,z) coordinate:
    if doTile==True:
        nx,ny = np.shape(ra)
        nz = len(z)
        ra = np.repeat(ra[:, :, np.newaxis], nz, axis=2)
        dec = np.repeat(dec[:, :, np.newaxis], nz, axis=2)
        d = np.tile(d[np.newaxis,np.newaxis,:],(nx,ny,1))
    c = SkyCoord(ra*u.degree, dec*u.degree, d*u.Mpc, frame=frame)
    # Astropy does x-axis as LoS by default, so below changes this
    #   by assigning z=x, x=y, y=z
    z,x,y = c.cartesian.x.value*h, c.cartesian.y.value*h, c.cartesian.z.value*h
    return x,y,z

def ComovingGridGalaxies(dims0,ra,dec,doWeights=False,Load=False,frame='icrs'):
    ''' Grid galaxies directy onto Cartesian comoving grid - more accurate but slower
    approach apposed to gridding and saving the (ra,dec,z) galaxy maps then regridding
    their pixels
    '''
    if Load==True:
        #galmap,randgrid = np.load('/users/scunnington/MeerKAT/LauraShare/ComivingWiggleZMaps.npy')
        galmap,randgrid = np.load('/idia/projects/hi_im/crosspower/2019/data/ComivingWiggleZMaps.npy')
        return galmap,randgrid
    ### Load galaxy catalogue:
    galcat = np.genfromtxt('/users/scunnington/MeerKAT/LauraShare/wigglez_reg11hrS_z0pt30_0pt50/reg11data.dat', skip_header=1)
    ra_gal,dec_gal,z_gal = galcat[:,0],galcat[:,1],galcat[:,2]
    # Subtract same RA Dec map means removed in MeerKAT map to align the footprint with ra=dec=0 for LoS
    noncontinuos_mask = ra>=ra[0]
    if len(ra[noncontinuos_mask])<len(ra): # if true there is discontinuity in RA array (360->0)
        ra[noncontinuos_mask] = ra[noncontinuos_mask] - 360 # Make continuous RA i.e. 359,360,1 -> -1,0,1 so mean RA is correct
        ra_gal = ra_gal - np.mean(ra)
        ra[ra<0] = ra[ra<0] + 360 # Reset negative coordinates to 359,360,1 convention
    else: ra_gal = ra_gal - np.mean(ra)
    dec_gal = dec_gal - np.mean(dec)
    x,y,z = SkyCoordtoCartesian(ra_gal,dec_gal,z_gal,doTile=False,LoScentre=False,frame=frame)
    lx,ly,lz,nx,ny,nz,x0,y0,z0 = dims0
    xbins,ybins,zbins = np.linspace(x0,x0+lx,nx+1),np.linspace(y0,y0+ly,ny+1),np.linspace(z0,z0+lz,nz+1)
    galmap = np.histogramdd((x,y,z),bins=(xbins,ybins,zbins))[0]
    if doWeights==False: return galmap
    if doWeights==True:
        datapath = '/users/scunnington/MeerKAT/LauraShare/wigglez_reg11hrS_z0pt30_0pt50/'
        rarand=np.empty(0)
        decrand=np.empty(0)
        zrand=np.empty(0)
        Nmock = 1000 # Number of WigZ randoms to stack in selection function
        for i in range(1,Nmock):
            galcat = np.genfromtxt( datapath + 'reg11rand%s.dat' %'{:04d}'.format(i), skip_header=1)
            rarand = np.append(rarand, galcat[:, 0])
            decrand = np.append(decrand, galcat[:,1])
            zrand = np.append(zrand, galcat[:,2])
        # Subtract same RA Dec map means removed in MeerKAT map to align the footprint with ra=dec=0 for LoS
        noncontinuos_mask = ra>=ra[0]
        if len(ra[noncontinuos_mask])<len(ra): # if true there is discontinuity in RA array (360->0)
            ra[noncontinuos_mask] = ra[noncontinuos_mask] - 360 # Make continuous RA i.e. 359,360,1 -> -1,0,1 so mean RA is correct
            rarand = rarand - np.mean(ra)
            ra[ra<0] = ra[ra<0] + 360 # Reset negative coordinates to 359,360,1 convention
        else: rarand = rarand - np.mean(ra)
        decrand = decrand - np.mean(dec)
        xrand,yrand,zrand = SkyCoordtoCartesian(rarand,decrand,zrand,doTile=False,LoScentre=False,frame=frame)
        randgrid = np.histogramdd((xrand,yrand,zrand),bins=(xbins,ybins,zbins))[0]
        randgrid = randgrid/Nmock # average
        #np.save('/users/scunnington/MeerKAT/LauraShare/ComivingWiggleZMaps',[galmap,randgrid])
        np.save('/idia/projects/hi_im/crosspower/2019/data/ComivingWiggleZMaps',[galmap,randgrid])
        return galmap,randgrid

def Cart2SphericalCoords(xcoord,ycoord,zcoord,cosmo,frame='icrs'):
    '''Mainly used for converting galaxy sims in comoving space into sky light-cone (ra,dec,z) coords
    '''
    ### Following astropy example in: https://docs.astropy.org/en/stable/coordinates/
    ### *** Input is assumed in units of Mpc/h so a conversion is done in to Mpc for astopy convention
    h = cosmo.H(0).value/100 # use to convert astopy Mpc distances to Mpc/h
    # Feed x=zcoord for astropy convention that x-axis is along LoS:
    c = SkyCoord(x=zcoord/h, y=xcoord/h, z=ycoord/h, unit='Mpc', frame=frame, representation_type='cartesian')
    c.representation_type = 'spherical'
    ra,dec,dist = c.ra.value,c.dec.value,c.distance.value
    # Convert distance into a redshift:
    redarr = np.linspace(0,2,1000) # assumes redshift doesn't exceed 2
    distarr = cosmo.comoving_distance(redarr).value
    z = np.interp(dist,distarr,redarr)
    return ra,dec,z

def AstropyGridding(ra,dec,z,nu,galaxyweights=None,data=None):
    '''Some code from katcali example notebook for gridding the rac,dec pointings
    onto ra,dec pixelised map:
    https://github.com/meerklass/katcali/blob/master/examples/level5/katcali_multi_level5.ipynb
    '''
    # data: chose either '2019' or '2021' for correct sky placement
    # galaxyweights: for histogram binning. Use if constructing sim with M_HI for each galaxy
    pix_deg = 0.3
    #set the sky area to be pixelized - SETUP FOR 11hr FIELD PILOT DATA:
    if data=='2019':
        x_cen = 163 #deg #RA
        x_half = 20 #deg
        y_cen = 3.5 #deg #DEC
        y_half = 6 #deg
    if data=='2021':
        x_cen=-18 #deg #RA
        x_half=20 #deg
        y_cen=-34 #deg #DEC
        y_half=11 #deg
    N_half_x = int(x_half/pix_deg)
    N_half_y = int(y_half/pix_deg)
    nx = 2*N_half_x+1
    ny = 2*N_half_y+1
    nz = len(nu)
    xbins,ybins = np.arange(0,nx+1,1),np.arange(0,ny+1,1)
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
    ref_p=skycoord_to_pixel(p0, w)
    assert(ref_p[0]<1e-12) #should be zero
    assert(ref_p[1]<1e-12) #shoule be zero
    map = np.zeros((nx,ny,nz))
    for i in range(nz):
        zmask = (z>zbins[i+1]) & (z<zbins[i])
        p_list = ac.SkyCoord(ra=ra[zmask]*u.deg, dec=dec[zmask]*u.deg)
        x_pix_list,y_pix_list = skycoord_to_pixel(p_list,w) #observation (ra,dec) to pix
        map[:,:,i] = np.histogram2d(x_pix_list,y_pix_list,bins=(xbins,ybins),weights=galaxyweights[zmask])[0]
    return map

def regrid_Kiyo(map,ra,dec,nu,pad=0,order=0,mode='constant'):
    # Re-grid the sky-coordinates (ra,dec,freq) map into a Cartesian comoving
    #  grid (x,y,z) [Mpc/h] physical distance units.
    # map: input (ra,dec,nu) structure with smallest frequency first i.e. nu[0] - np.min(nu)
    # ra,dec,nu : coordinates for each pixel in map. RA and Dec in [deg] nu in [MHz]
    ''' Function is essentially rewritten from Yi-Chao/Kiyo's method into condensed format
          to work in crosspower pipeline. Function is thereofore based on the code in:
     - https://github.com/meerklass/meerKAT_sim/blob/ycli/sim/meerKAT_sim/ps/physical_gridding.py
     - https://github.com/kiyo-masui/analysis_IM/blob/master/map/physical_gridding.py
    '''
    nx,ny,nz = np.shape(map)
    norig = np.copy([nx,ny,nz])
    ### Define the grid size:
    dec_centre = np.mean(dec)
    rafact = np.cos(np.radians(dec_centre))
    thetax, thetay = np.abs(ra[0]-ra[-1]), np.abs(dec[0]-dec[-1]) # span in RA and Dec
    thetax *= rafact
    z1 = v_21cm/np.max(nu) - 1
    z2 = v_21cm/np.min(nu) - 1
    c1 = cosmo.comoving_distance(z1).value * h
    c2 = cosmo.comoving_distance(z2).value * h
    c_center = (c1 + c2) / 2
    lx,ly,lz = np.radians(thetax)*c2, np.radians(thetay)*c2 , c2-c1
    # Enlarge cube size by `pad` in each dimension, so raytraced cube sits exactly
    #   within the gridded points. Set pad=0 if map already comfortably inside grid
    #   boundaries:
    if pad==0: nxpad,nypad,nzpad = nx,ny,nz
    else:
        nxpad,nypad,nzpad = nx+pad,ny+pad,nz+pad
        lx,ly,lz = lx*nxpad/nx,ly*nypad/ny,lz*nzpad/nz
        c1 = c_center - (c_center - c1) * (nzpad) / nz
        c2 = c_center + (c2 - c_center) * (nzpad) / nz
    dz = abs(c2 - c1) / (nzpad - 1)
    dz_centre = c1 + dz * (nzpad / 2)
    x_axis = np.linspace(-lx/2,lx/2,nxpad)
    y_axis = np.linspace(-ly/2,ly/2,nypad)
    radius_axis = np.linspace( dz_centre+lz/2, dz_centre-lz/2, nzpad ) #start from largest distance and descend since nu starts small

    # Obtain a redshift/freq corresponding to lz radius array
    _xp = np.linspace(z1 * 0.9, z2 * 1.1, 2000)
    _fp = cosmo.comoving_distance(_xp).value * h
    za = np.interp(radius_axis, _fp, _xp)
    nua = v_21cm / (1 + za)
    gridy, gridx = np.meshgrid(y_axis, x_axis)
    interpol_grid = np.zeros((3, nxpad, nypad))
    ## Switch axes order to Yi-Chao's convention:
    map = np.moveaxis(map,-1,0)

    ### Populate map in Cartesian-comoving space:
    physmap = np.zeros((nzpad,nxpad,nypad)) # This will be the final map in physical comoving space
    mask = np.ones_like(physmap)
    for i in range(nzpad):
        interpol_grid[0, :, :] = (nua[i] - nu[0]) / (nu[-1] - nu[0]) * nz
        proper_z = cosmo.comoving_transverse_distance(za[i]).value * h
        angscale = ((proper_z * u.deg).to(u.rad)).value
        interpol_grid[1, :, :] = gridx/angscale/thetax*nx + nx/2
        interpol_grid[2, :, :] = gridy/angscale/thetay*ny + ny/2
        physmap[i, :, :] = sp.ndimage.map_coordinates(map, interpol_grid, order=order, mode=mode)
        interpol_grid[1, :, :] = np.logical_or(interpol_grid[1, :, :] >= nx, interpol_grid[1, :, :] < 0)
        interpol_grid[2, :, :] = np.logical_or(interpol_grid[2, :, :] >= ny, interpol_grid[2, :, :] < 0)
        mask = np.logical_not(np.logical_or(interpol_grid[1, :, :], interpol_grid[2, :, :]))
        physmap *= mask
    ## Switch back axes order to Steves's convention:
    physmap = np.moveaxis(physmap,0,-1)
    dx,dy,dz = (x_axis[1]-x_axis[0]),(y_axis[1]-y_axis[0]),(radius_axis[1]-radius_axis[0])
    lx,ly,lz = lx+dx,ly+dy,lz+dz
    dims = [lx,ly,lz,nxpad,nypad,nzpad]
    return physmap,dims
