'''
Mock generation code: have taken parts from Steve Murray's Power box [https://github.com/steven-murray/powerbox/blob/master/powerbox]
however this only works for cubic boxes where nx=ny=nz - so need to use this generalised script
'''
import numpy as np
import plot
import matplotlib.pyplot as plt
try: # See if pyfftw is installed and use this if so for increased speed performance - see powerbox documentation
    from pyfftw import empty_aligned as empty
    HAVE_FFTW = True
except ImportError:
    HAVE_FFTW = False

#### Setting false by default since getting FFTW errors
HAVE_FFTW = False

import dft # power box script
import grid

def getkspec(nx,ny,nz,dx,dy,dz,doRSD=False):
    # Obtain 3D grid of k-modes
    kx = dft.fftfreq(nx, d=dx, b=1)
    ky = dft.fftfreq(ny, d=dy, b=1)
    kz = dft.fftfreq(nz, d=dz, b=1)
    kspec = np.sqrt(kx[:,np.newaxis,np.newaxis]**2 + ky[np.newaxis,:,np.newaxis]**2 + kz[np.newaxis,np.newaxis,:]**2)
    if doRSD==False: return kspec
    if doRSD==True:
        # Calculate mu-spectrum as needed for RSD application:
        k0mask = kspec==0
        kspec[k0mask] = 1.
        muspec = np.absolute(kz[np.newaxis,np.newaxis,:])/kspec
        kspec[k0mask] = 0.
        del k0mask
        return kspec,muspec

def _make_hermitian(mag, pha):
    #### Credit Steven Murray and Powerbox: https://github.com/steven-murray/powerbox/blob/master/powerbox
    revidx = (slice(None, None, -1),) * len(mag.shape)
    mag = (mag + mag[revidx]) / np.sqrt(2)
    pha = (pha - pha[revidx]) / 2 + np.pi
    return mag * (np.cos(pha) + 1j * np.sin(pha))

def gauss_hermitian(lx,ly,lz,nx,ny,nz):
    #### Credit Steven Murray and Powerbox: https://github.com/steven-murray/powerbox/blob/master/powerbox
    "A random array which has Gaussian magnitudes and Hermitian symmetry"
    np.random.seed(seed_)
    mag = np.random.normal(0,1,(nx+1,ny+1,nz+1))
    pha = 2*np.pi * np.random.uniform(size=(nx+1,ny+1,nz+1))
    dk = _make_hermitian(mag, pha)
    return dk[:-1,:-1,:-1] # Put back to even array by trimming off pixels

def Generate(Pmod,dims,b=1,f=0,Tbar=1,doRSD=False,LogNorm=True,seed=None,W=None,LightCone=False,ra=None,dec=None,nu=None,wproj=None,Np=3,dims0=None,w_noise=None,sigma_N=None,PossionSampGalaxies=False,Ngal=None,ObtainExactNgal=True):
    ### Generate a mock field
    # Default is to do a logN mock but if a Gaussian mock is required set LogNorm=False
    # seed: manually set this to same number to geneate fields to cross-correlate
    # W: optional binary survey selection function
    # LightCone: Set True (and supply ra/dec/nu) to project mock field onto a lightcone
    #    with (ra,dec,z) coordinates to match real data.
    # w_noise: optional inverse variance noise weights - use these to create a noise map constistent with the weight
    # sigma_N: noise variance in map. Only used if w_noise is given
    # PossionSampGalaxies: set true to generate a number counts field of galaxies, requires Ngal
    # Ngal: if Poisson sampling galaxies, the number of galaxies to aim to include in counts map
    # ObtainExactNgal: if Poisson sampling galaxies, set True to obtain exact target Ngal
    # ReturnGalCat: if True, randomly assign coordinates within voxels to all counted galaxies and return catalogue
    if seed is None: seed = np.random.randint(0,1e6)
    global seed_; seed_=seed
    lx,ly,lz,nx,ny,nz = dims
    if LightCone==True: # Run regrid code to get size of physcial grid that encloses full (RA,Dec,z) map:
        dims,dims0 = grid.comoving_dims(ra,dec,nu,wproj,ndim=[nx,ny,nz],W=W)
    lx,ly,lz,nx,ny,nz = dims
    if f!=0: doRSD=True
    # Works for even number of cells - if odd required add one and remove at end:
    x_odd,y_odd,z_odd = False,False,False # Assume even dimensions
    if nx%2!=0:
        x_odd = True
        lx = lx + lx/nx
        nx = nx + 1
    if ny%2!=0:
        y_odd = True
        ly = ly + ly/ny
        ny = ny + 1
    if nz%2!=0:
        z_odd = True
        lz = lz + lz/nz
        nz = nz + 1
    dx,dy,dz = lx/nx,ly/ny,lz/nz # Resolution
    vol = lx*ly*lz # Define volume from new grid size
    delta = empty((nx,ny,nz), dtype='complex128')
    if doRSD==False: kspec = getkspec(nx,ny,nz,dx,dy,dz)
    if doRSD==True: kspec,muspec = getkspec(nx,ny,nz,dx,dy,dz,doRSD)
    pkspec = np.zeros(np.shape(kspec))
    pkspec[kspec!=0] = 1/vol * Pmod(kspec[kspec!=0])
    if doRSD==False: pkspec = b**2 * pkspec
    if doRSD==True: pkspec = b**2 * (1 + (f/b)*muspec**2)**2 * pkspec
    if LogNorm==True:
        # Inverse Fourier transform to obtain the correlation function
        xigrid = vol * np.real(dft.ifft(pkspec, L=[lx,ly,lz], a=1, b=1)[0])
        xigrid = np.log(1 + xigrid) # Transform the correlation function
        pkspec = np.abs( dft.fft(xigrid, L=[lx,ly,lz], a=1, b=1)[0] )
        pkspec[kspec==0] = 0
    delta = np.sqrt(pkspec) * gauss_hermitian(lx,ly,lz,nx,ny,nz)
    if LogNorm==True: delta = np.sqrt(vol) * dft.ifft(delta, L=[lx,ly,lz], a=1, b=1)[0]
    if LogNorm==False: delta = vol * dft.ifft(delta, L=[lx,ly,lz], a=1, b=1)[0]
    delta = np.real(delta)
    if LogNorm==True:
        # Return log-normal density field delta_LN
        delta = np.exp(delta - np.var(delta)/2) - 1
    if x_odd==True: delta = delta[:-1,:,:]
    if y_odd==True: delta = delta[:,:-1,:]
    if z_odd==True: delta = delta[:,:,:-1]
    delta *= Tbar # multiply by Tbar if not set to 1
    if w_noise is not None:
        if sigma_N is None: print('\nError! provide sigma_N to estimate noise in mock');exit()
        w_noise[w_noise!=0] = 1/w_noise[w_noise!=0] # invert inverse variance map
        w_noise = w_noise/np.max(w_noise) # normalise to max value is 1
        dT_noise = np.random.normal(0,sigma_N,(np.shape(delta)))
        dT_noise *= w_noise # non-uniform noise map
        delta += dT_noise
    if PossionSampGalaxies==False:
        if W is not None: # Assumes binary window W(0,1)
            # a more complex window can be used for galaxy mocks if Poisson sampling
            delta[W==0] = 0
        if LightCone==False: return delta
        if LightCone==True: #return grid.lightcone(delta,dims0,ra,dec,nu)
            map = grid.lightcone(delta,dims0,ra,dec,nu,wproj,Np=Np,verbose=True)
            return delta,map
    if PossionSampGalaxies==True:
        if LightCone==True: delta = grid.lightcone(delta,dims0,ra,dec,nu,wproj,Np=Np)
        if Ngal is None: print('\Error!: define Ngal to Poisson sample galaxies in mock');exit()
        return PoissonSampleGalaxies(delta,dims,Ngal,W,ObtainExactNgal)

def PoissonSampleGalaxies(delta_g,dims,Ngal,W=None,ObtainExactNgal=True):
    # delta_g: galaxy overdensity field with minimum -1
    # W is an optional survey selection function
    Ngal = int(Ngal) # ensure Ngal is an integer and not float
    lx,ly,lz,nx,ny,nz = dims
    if W is None:
        ndens = Ngal/(nx*ny*nz)
        W = np.ones((nx,ny,nz)) * ndens # Uniform survey selection function
    n_g = (delta_g + 1) * W
    n_g_poisson = np.random.poisson( n_g )
    if ObtainExactNgal==True:
        # Above Poisson method is fast but doesn't deliver the exact target Ngal,
        #    if this is required then keep running Poisson samplings on the remaining
        #    galaxy count differences between acheived and target, until target is met
        W_g = np.ones((nx,ny,nz))
        W_g[W==0] = 0
        for i in range(100):
            if i==99: print('\nPoisson sampling ran out of loops - check code!'); exit()
            Ngal_diff = np.sum(n_g_poisson) - Ngal
            if Ngal_diff==0: break
            W_diff = W_g/np.sum(W_g) * np.abs(Ngal_diff) # Should sum to Ngal
            n_g_diff = (delta_g + 1) * W_diff
            n_g_poisson_diff = np.random.poisson( n_g_diff ) # new sampled galaxies to add/subtract
            if Ngal_diff<0: n_g_poisson += n_g_poisson_diff
            if Ngal_diff>0: n_g_poisson -= n_g_poisson_diff
            if np.min(n_g_poisson)<0:
                # If Ngal is low for mock, some pixels may contain negative counts,
                #   use this code to fix that by resetting those pixels to zero, and
                #   instead subtract galaxies from those with already filled pixles
                n_g_poisson[n_g_poisson<0] = 0
                Ngal_diff = np.sum(n_g_poisson) - Ngal
                if Ngal_diff>0:
                    ### Select random Ngal_diff pixels with galaxies already in to remove
                    ix,iy,iz = np.where(n_g_poisson>0)
                    randindx = np.random.randint(0,len(ix),np.abs(Ngal_diff))
                    n_g_poisson[ix[randindx],iy[randindx],iz[randindx]] -= 1
    ###### This is NOT working ###########
    # - currently just using RSD factors in input power spectrum
    #N_g = ApplyRSD(N_g,nx,ny,nz,lx,ly,lz,z_eff,deltak,kspec)
    ############################################
    return n_g_poisson

def LogGalaxyCatalogue(n_g,dims0,ra,dec):
    '''From lognormal poission sample n_g generate random galaxy coordinates within
    voxels for all counted galaxies, then return their transformed positions in RA,Dec,z'''
    lx,ly,lz,nx,ny,nz,x0,y0,z0 = dims0
    xbins,ybins,zbins = np.linspace(x0,x0+lx,nx+1),np.linspace(y0,y0+ly,ny+1),np.linspace(z0,z0+lz,nz+1)
    # Obtain coordinates (xp0,yp0,zp0) of cell centres:
    xp0,yp0,zp0 = (xbins[1:]+xbins[:-1])/2,(ybins[1:]+ybins[:-1])/2,(zbins[1:]+zbins[:-1])/2 #centre of bins
    xp0,yp0,zp0 = np.tile(xp0[:,np.newaxis,np.newaxis],(1,ny,nz)),np.tile(yp0[np.newaxis,:,np.newaxis],(nx,1,nz)),np.tile(zp0[np.newaxis,np.newaxis,:],(nx,ny,1))
    # Loop over all number of counts appearing in n_g and generate a galaxy
    #   for each with coordinates generated within voxel by a random kick
    #   between 0 and H/2 (cell size) in all directions
    Hx,Hy,Hz = lx/nx,ly/ny,lz/nz
    xp,yp,zp = np.array([]),np.array([]),np.array([])
    for i in range(np.max(n_g)):
        xp0_i = xp0[n_g>=(i+1)]
        yp0_i = yp0[n_g>=(i+1)]
        zp0_i = zp0[n_g>=(i+1)]
        # Satellite xp_s particles uniformly random 0<x<H/2 from cell centred particels xp0:
        xp = np.append(xp,xp0_i + np.random.uniform(-Hx/2,Hx/2,np.shape(xp0_i)))
        yp = np.append(yp,yp0_i + np.random.uniform(-Hy/2,Hy/2,np.shape(yp0_i)))
        zp = np.append(zp,zp0_i + np.random.uniform(-Hz/2,Hz/2,np.shape(zp0_i)))
    # Transform Cartesian coordinates of galaxies into sky coordinates:
    return grid.Cart2SphericalCoords(xp,yp,zp,ramean_arr=ra,decmean_arr=dec)

def GetMock_OLDVERSION(Pmod,dims,b=1,f=0,doRSD=False,LogNorm=True):
    '''
    Default is to do a lognormal mock but if a Gaussian mock is required instead
    set LogNorm=False
    '''
    lx,ly,lz,nx,ny,nz = dims
    if f!=0: doRSD=True
    # Works for even number of cells - if odd required add one and remove at end:
    x_odd,y_odd,z_odd = False,False,False # Assume even dimensions
    if nx%2!=0:
        x_odd = True
        lx = lx + lx/nx
        nx = nx + 1
    if ny%2!=0:
        y_odd = True
        ly = ly + ly/ny
        ny = ny + 1
    if nz%2!=0:
        z_odd = True
        lz = lz + lz/nz
        nz = nz + 1
    dx,dy,dz = lx/nx,ly/ny,lz/nz # Resolution
    vol = lx*ly*lz # Define volume from new grid size
    if HAVE_FFTW==True: delta = empty((nx,ny,nz), dtype='complex128')
    else: delta = np.zeros((nx,ny,nz), dtype='complex128')
    if doRSD==False: kspec = getkspec(nx,ny,nz,dx,dy,dz)
    if doRSD==True: kspec,muspec = getkspec(nx,ny,nz,dx,dy,dz,doRSD)
    pkspec = np.zeros(np.shape(kspec))
    pkspec[kspec!=0] = 1/vol * Pmod(kspec[kspec!=0])
    if doRSD==False: pkspec = b**2 * pkspec
    if doRSD==True: pkspec = b**2 * (1 + (f/b)*muspec**2)**2 * pkspec
    if LogNorm==True:
        # Inverse Fourier transform to obtain the correlation function
        xigrid = vol * np.real(dft.ifft(pkspec, L=[lx,ly,lz], a=1, b=1)[0])
        xigrid = np.log(1 + xigrid) # Transform the correlation function
        pkspec = np.abs( dft.fft(xigrid, L=[lx,ly,lz], a=1, b=1)[0] )
        pkspec[kspec==0] = 0
    delta = np.sqrt(pkspec) * gauss_hermitian(lx,ly,lz,nx,ny,nz)
    if LogNorm==True: delta = np.sqrt(vol) * dft.ifft(delta, L=[lx,ly,lz], a=1, b=1)[0]
    if LogNorm==False: delta = vol * dft.ifft(delta, L=[lx,ly,lz], a=1, b=1)[0]
    delta = np.real(delta)
    if LogNorm==True:
        # Return log-normal density field delta_LN
        delta = np.exp(delta - np.var(delta)/2) - 1
    if x_odd==True: delta = delta[:-1,:,:]
    if y_odd==True: delta = delta[:,:-1,:]
    if z_odd==True: delta = delta[:,:,:-1]
    return delta
