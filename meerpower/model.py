import numpy as np
import scipy
from scipy.interpolate import interp1d
import power
import HItools
import grid
import plot
import matplotlib.pyplot as plt
H_0 = 67.7 # Planck15

def FitPolynomial(x,y,n):
    ### Fit a polynomial of order n to a generic 1D data array [x,y]
    coef = np.polyfit(x,y,n)
    func = np.zeros(len(x)) # fitted function
    for i in range(n+1):
        func += coef[-(i+1)]*x**i
    return func

def fix_pixels(input,W,IncludeDeadLoS=False):

    import astrofix # shoved here to remove depedency on it for default running

    ### Fill in missing pixels to later remove
    '''
    ##### Function by Paula Soares - with edits by Steve #######
    '''
    input[W==0] = np.nan # Ensure all empty pixels set to NaN so can be identified by astrofix
    axes = np.shape(input)
    if len(axes)==2: nx,ny = axes
    if len(axes)==3: nx,ny,nz = axes
    if len(axes)==3 and IncludeDeadLoS==False: # exlude completely empty LoS
        # Define 2D mask which will be used to remove any LoS which are completely
        #   as do not want to include these in fixed image:
        W_2D = np.ones((nx,ny))
        W_2D[np.sum(W,2)==0] = 0
    if len(axes)==3:
        for i in range(input.shape[2]):
            plottools.ProgressBar(i,input.shape[2],'\nRunning astrofix ...')
            if np.isnan(input[:,:,i]).all(): continue # if channel is empty, don't apply correction
            try: fixed_image, para, TS = astrofix.Fix_Image(input[:,:,i], "asnan", max_clip=0.1, sig_clip=-100)
            except ValueError:
                print('\nToo many empty pixels in channel number %s for astrofix. Deleting channel'%i)
                input[:,:,i] = np.nan
                continue
            # Add fixed pixels to input map wherever a LoS is not completely dead
            input[:,:,i][W_2D==1] = fixed_image[W_2D==1]

    else: # if just one (nx,ny) input map given:
        fixed_image, para, TS = astrofix.Fix_Image(input, "asnan", max_clip=0.1, sig_clip=-100)
        input[W==0] = fixed_image[W==0]
    # Create array marking the fixed pixles so they can be excluding from certain analyses
    W_fix = np.ones(np.shape(W))
    W_fix[np.isnan(input)] = 0
    W_fix = W_fix - W
    input[np.isnan(input)] = 0 # convert NaNs back to zeros
    return input,W_fix

def PkModSpec(Pmod,dims,kspec,muspec,b1,b2,f,sig_v=0,Tbar1=1,Tbar2=1,r=1,R_beam1=0,R_beam2=0,sig_N=0,w1=None,w2=None,W1=None,W2=None,MatterRSDs=False,lwin=None,pixwin=None,s_para=0,Damp=None,gridinterp=False):
    ### Separate function to PkMod which leaves model Pk in 3D spectrum format
    if len(dims)==6: lx,ly,lz,nx,ny,nz = dims
    if len(dims)==9: lx,ly,lz,nx,ny,nz,x0,y0,z0 = dims
    kspec[kspec==0] = 1 # avoid model interpolation error from k=0
    # Collect damping terms from beam/FG/channels/heapy pixelisation:
    if Damp is None: Damp = B_beam(muspec,kspec,R_beam1)*B_beam(muspec,kspec,R_beam2)
    if sig_N!=0: P_N = sig_N**2 * (lx*ly*lz)/(nx*ny*nz) # noise term
    else: P_N = 0
    if MatterRSDs==False: beta1,beta2 = f/b1,f/b2 # Include bias in Kaiser term (sensitive in quadrupole)
    if MatterRSDs==True: beta1,beta2 = f,f # Exclude bias in Kaiser term, i.e. only apply RSD to dark matter field, leaving a single amplitude parameter to constrain


    # Do full model i.e. eq 12 in Wolz+21.
    ###### Old method applying beam, bias and Kaiser on the interpolared grid ####
    if gridinterp==True: # Do full grid interp, creates spiky model due to uneven distribution of kperp/kpara pixels into bins
        pkspecmod = Damp * Tbar1*Tbar2 * b1*b2*( r + (beta1 + beta2)*muspec**2 + beta1*beta2*muspec**4 ) / (1 + (kspec*muspec*sig_v/H_0)**2) * Pmod(kspec) + P_N
        if w1 is not None or w2 is not None or W1 is not None or W2 is not None: # Convolve with window
            pkspecmod = power.getpkconv(pkspecmod,dims,w1,w2,W1,W2)
        return pkspecmod
    ############################################################################

    #'''
    kmod = np.linspace(np.min(kspec),np.max(kspec),1000)
    Pk_int = lambda mu: Tbar1*Tbar2 * b1*b2*( r + (beta1 + beta2)*mu**2 + beta1*beta2*mu**4 ) / (1 + (k_i*mu*sig_v/H_0)**2) * Pmod(k_i) * B_beam(mu,k_i,R_beam1) * B_beam(mu,k_i,R_beam2) + P_N
    pkmod = np.zeros(len(kmod))
    nmodes = np.zeros(len(kmod))
    for i in range(len(kmod)):
        k_i = kmod[i]
        pkmod[i] = scipy.integrate.quad(Pk_int, 0, 1)[0]
    pkmod = interp1d(kmod, pkmod)
    #'''

    ##### BELOW IS WORKING ATTEMPT TO DEVELOP A HIGHLY SAMPLED ANISOTROPIC MODEL WHICH IS THEN
    #####    INTERPOLATABLE ONTO THE DATA GRID SO THAT CONVOLUTION WITH WINDOW AND WEIGHT
    #####    FUNCTIONS IS POSSIBLE
    '''
    from scipy.interpolate import interpn
    nxgrid,nygrid,nzgrid = np.linspace(1,nx,nx),np.linspace(1,ny,ny),np.linspace(1,nz,nz)
    ngrid = (nxgrid,nygrid,nzgrid)
    nxmod,nymod,nzmod = 256,256,256

    lxmod,lymod,lzmod = 1000,1000,1000
    dims = [lxmod,lymod,lzmod,nxmod,nymod,nzmod]
    kspecmod,muspecmod,indep = power.getkspec(dims,FullPk=True)
    kspecmod[kspecmod==0] = 1 # avoid model interpolation error from k=0

    Damp = B_beam(muspecmod,kspecmod,R_beam1)*B_beam(muspecmod,kspecmod,R_beam2)

    ###
    nxgrid,nygrid,nzgrid = np.linspace(1,nx,nx),np.linspace(1,ny,ny),np.linspace(1,nz,nz)
    ngrid = (nxgrid,nygrid,nzgrid)
    nxmod,nymod,nzmod = 256,256,256
    dimsmod = [lx,ly,lz,nxmod,nymod,nzmod]
    nxgrid_mod,nygrid_mod,nzgrid_mod = np.linspace(1,nx,nxmod),np.linspace(1,ny,nymod),np.linspace(1,nz,nzmod)
    i_coords,j_coords,k_coords = np.meshgrid(nxgrid_mod,nygrid_mod,nzgrid_mod, indexing='ij')
    coordinate_grid = np.array([i_coords,j_coords,k_coords])
    coordinate_grid = np.swapaxes(coordinate_grid,0,1)
    coordinate_grid = np.swapaxes(coordinate_grid,1,2)
    coordinate_grid = np.swapaxes(coordinate_grid,2,3)
    print(np.shape(ngrid))
    print(ngrid[0])
    print(np.shape(coordinate_grid))
    exit()
    ###

    pkspecmod = Damp * Tbar1*Tbar2 * b1*b2*( r + (beta1 + beta2)*muspecmod**2 + beta1*beta2*muspecmod**4 ) / (1 + (kspecmod*muspecmod*sig_v/H_0)**2) * Pmod(kspecmod) + P_N

    pkspec = interpn(kspec, pkspecmod, kspecmod)
    print(np.shape(pkspec))
    exit()

    nxgrid,nygrid,nzgrid = np.linspace(1,nx,nx),np.linspace(1,ny,ny),np.linspace(1,nz,nz)
    ngrid = (nxgrid,nygrid,nzgrid)
    nxmod,nymod,nzmod = 256,256,256
    #kbinsmod = np.linspace(kbins[0],kbins[-1],200)
    dimsmod = [lx,ly,lz,nxmod,nymod,nzmod]
    nxgrid_mod,nygrid_mod,nzgrid_mod = np.linspace(1,nx,nxmod),np.linspace(1,ny,nymod),np.linspace(1,nz,nzmod)
    i_coords,j_coords,k_coords = np.meshgrid(nxgrid_mod,nygrid_mod,nzgrid_mod, indexing='ij')
    coordinate_grid = np.array([i_coords,j_coords,k_coords])
    coordinate_grid = np.swapaxes(coordinate_grid,0,1)
    coordinate_grid = np.swapaxes(coordinate_grid,1,2)
    coordinate_grid = np.swapaxes(coordinate_grid,2,3)
    w_HI_mod = interpn(ngrid, w_HI, coordinate_grid)
    '''

    pkspecmod = pkmod(kspec)
    '''
    if MatterRSDs==True: # Only apply Kaiser term to matter power spectrum leaving a single amplitude parameter
        pkspecmod = Tbar1*Tbar2 * b1*b2 * r * (1 + f*muspec**2)**2 / (1 + (kspec*muspec*sig_v/H_0)**2) * Pmod(kspec) * B_beam(muspec,kspec,R_beam1) * B_beam(muspec,kspec,R_beam2)
        #pkspecmod[kspec==1] = 0
    '''
    if w1 is not None or w2 is not None or W1 is not None or W2 is not None: # Convolve with window
        pkspecmod = power.getpkconv(pkspecmod,dims,w1,w2,W1,W2)
    return pkspecmod

def PkMod(Pmod,dims,kbins,b1=1,b2=1,f=0,sig_v=0,Tbar1=1,Tbar2=1,r=1,R_beam1=0,R_beam2=0,sig_N=0,w1=None,w2=None,W1=None,W2=None,doMultipole=False,Pk2D=False,kperpbins=None,kparabins=None,MatterRSDs=False,interpkbins=False,lwin=None,pixwin=None,s_para=0,Damp=None,gridinterp=False):
    ### r is cross-correlation coeficient if doing a cross-correlation, set all _1 and _2 parameters
    ###  equal if doing an auto correlation
    #if len(dims)==6: lx,ly,lz,nx,ny,nz = dims
    #if len(dims)==9: lx,ly,lz,nx,ny,nz,x0,y0,z0 = dims
    if interpkbins==True: # If True, interpolate model Pk over same grid and bin using same pipeline as data
        kspec,muspec,indep = power.getkspec(dims,FullPk=True)
        pkspecmod = PkModSpec(Pmod,dims,kspec,muspec,b1,b2,f,sig_v,Tbar1,Tbar2,r,R_beam1,R_beam2,sig_N,w1,w2,W1,W2,MatterRSDs,lwin,pixwin,s_para,Damp=Damp,gridinterp=gridinterp)
        if doMultipole==False:
            if Pk2D==False:
                pkmod,k,nmodes = power.binpk(pkspecmod,dims[:6],kbins,FullPk=True,doindep=False)
                return pkmod,k,nmodes
            if Pk2D==True:
                pk2d,nmodes = power.bin2DPk(pkspecmod,dims[:6],kperpbins,kparabins,FullPk=True)
                return pk2d,nmodes
        if doMultipole==True:
            pk0,pk2,pk4,k,nmodes = power.binpole(pkspecmod,dims[:6],kbins,FullPk=True,doindep=False)
            return pk0,pk2,pk4,k,nmodes
    if interpkbins==False: # If False, run a more approximate model build, using integration over analytical function
        if doMultipole==False:
            kmod = (kbins[1:] + kbins[:-1]) / 2 #centre of k-bins
            beta1,beta2 = f/b1,f/b2
            deltak = [kbins[i]-kbins[i-1] for i in range(1,len(kbins))]
            if sig_N!=0: P_N = sig_N**2 * (lx*ly*lz)/(nx*ny*nz) # noise term
            else: P_N = 0
            Pk_int = lambda mu: Tbar1*Tbar2 * b1*b2*( r + (beta1 + beta2)*mu**2 + beta1*beta2*mu**4 ) / (1 + (k_i*mu*sig_v/H_0)**2) * Pmod(k_i) * B_beam(mu,k_i,R_beam1) * B_beam(mu,k_i,R_beam2) + P_N
            pkmod = np.zeros(len(kmod))
            nmodes = np.zeros(len(kmod))
            for i in range(len(kmod)):
                k_i = kmod[i]
                pkmod[i] = scipy.integrate.quad(Pk_int, 0, 1)[0]
                nmodes[i] = 1 / (2*np.pi)**3 * (lx*ly*lz) * (4*np.pi*k_i**2*deltak[i]) # Based on eq14 in https://arxiv.org/pdf/1509.03286.pdf
            return pkmod,kmod,nmodes

def B_beam(mu,k,R_beam):
    if R_beam==0: return 1
    return np.exp( -(1-mu**2)*k**2*R_beam**2/2 )

### Detection Calculation:
def DetectionSigma(data,model,errors,nullmodel=None):
    # Calculate detection significance relative to some null model
    if nullmodel is None: nullmodel = np.zeros(len(data))
    Chisq = ChiSquare(data,model,errors)
    nullChisq = ChiSquare(data,nullmodel,errors)
    if Chisq>nullChisq:
        print('\nDetection Significance: Null' )
        print('Model Chi^2 = '+str(Chisq))
        print('Null Chi^2 =  '+str(nullChisq))
        return np.nan
    else:
        det_sig = np.sqrt(nullChisq - Chisq)
        print('\nDetection Significance: ' + str(np.round(det_sig,3)) + ' sigma' )
        return det_sig

#Reduced Chi-squared function
def ChiSquare(x_obs,x_mod,x_err,dof=None):
    if dof is None: return np.sum( ((x_obs-x_mod)/x_err)**2 )
    else: # return reduced ChiSquare
        return np.sum( ((x_obs-x_mod)/x_err)**2 ) / dof

########################################################################
#  Least-Squares Fitting Functions                                     #
########################################################################
from scipy.optimize import curve_fit

def LSqFitCrossPkAmplitude(pk_gHI,sig_err,Pmod,zeff_,dims,kbins,kmin=None,kmax=None,b_g=1,f=0,R_beam=0,w1=None,w2=None,W1=None,W2=None):
    ### Least-squares fit of cross-power single scaling power spectrum amplitude
    ### default assumes b_HI=1 and r=1, so fitting joint Omega_HI b_HI r parameter
    global pkmod; global zeff; global kcut
    zeff = zeff_
    pkmod,k,nmodes = PkMod(Pmod,dims,kbins,b2=b_g,f=f,R_beam1=R_beam,w1=w1,w2=w2,W1=W1,W2=W2,interpkbins=True,MatterRSDs=True)
    # Implement any k-cuts:
    if kmin is None: kmin = kbins[0]
    if kmax is None: kmax = kbins[-1]
    kcut = (k>kmin) & (k<kmax)

    popt, pcov = curve_fit(PkAmplitude, k[kcut], pk_gHI[kcut], p0=0.4e-3, sigma=sig_err[kcut], bounds=(0, 10))
    OmHIbHI, OmHIbHI_err = popt[0],np.sqrt(pcov[0,0]) # take errors on parameter estimates as root of the covariance (since only 1 parameter, cov = [1,1] matrix)
    return OmHIbHI, OmHIbHI_err

def PkAmplitude(k,OmHIbHI):
    b_HI = 1
    OmegaHI = OmHIbHI/b_HI
    Tbar = HItools.Tbar(zeff,OmegaHI)
    return Tbar*b_HI*pkmod[kcut]

########################################################################
# MCMC Fitting Functions                                               #
########################################################################
import emcee

def model(theta,k):
    if ndim==1:
        OmHI = theta
        Tbar = HItools.Tbar(zeff,OmHI)
        return Tbar * pkmod
    if ndim==2:
        OmHI,b_HI = theta
        Tbar = HItools.Tbar(zeff,OmHI)
        return PkMod(Pmod,dims,kbins,b_HI,b_g,f,sig_v,Tbar1=Tbar,Tbar2=1,r=r_HIg,R_beam1=R_beam,R_beam2=0,w1=w_HI,w2=w_g,W1=W_HI,W2=W_g,interpkbins=True)[0]

#log likelihood
def lnlike(theta, k, Pk, Pkerr):
    return -0.5 * ChiSquare(Pk,model(theta,k),Pkerr)

#priors
def lnprior(theta):
    if ndim==1:
        OmHI = theta
        if 0 < OmHI < 1: #imposing a prior
            return 0.0
    if ndim==2:
        OmHI,b_HI = theta
        if 0 < OmHI < 1 and 0 < b_HI < 4: #imposing a prior
            return 0.0
    return -np.inf

def lnprob(theta, k, Pk, Pkerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta,k,Pk,Pkerr)

def runMCMC(k,Pk,Pkerr,Omega_HI_fid,b_HI_fid,zeff_,Pmod_,b_g_,f_,sig_v_,r_HIg_,R_beam_,dims_,kbins_,w_g_=None,W_g_=None,w_HI_=None,W_HI_=None,nwalkers=200,niter=500,ndim_=2,ContinueBackend=False,backendfile=None):
    '''
    Main run function for MCMC
    '''
    global zeff; global Pmod; global b_g; global f; global sig_v; global r_HIg; global R_beam; global dims; global kbins; global w_g; global W_g; global w_HI; global W_HI; global ndim
    zeff=zeff_; Pmod=Pmod_; b_g=b_g_; f=f_; sig_v=sig_v_; r_HIg=r_HIg_; R_beam=R_beam_; dims=dims_; kbins=kbins_; w_g=w_g_; W_g=W_g_; w_HI=w_HI_; W_HI=W_HI_; ndim=ndim_

    if ndim==1:
        # Calculate model with b_HI = Tbar = 1 to use in amplitude fitting:
        global pkmod
        pkmod = PkMod(Pmod,dims,kbins,1,b_g,f,sig_v,Tbar1=1,Tbar2=1,r=r_HIg,R_beam1=R_beam,R_beam2=0,w1=w_HI,w2=w_g,W1=W_HI,W2=W_g,interpkbins=True,MatterRSDs=True)[0]
        OmHI_p0 = np.random.normal(Omega_HI_fid,scale=0.1*Omega_HI_fid,size=nwalkers)
        p0 = np.swapaxes(np.array([OmHI_p0]),0,1)
    if ndim==2: # Fit Omega_HI and b_HI independently
        OmHI_p0 = np.random.normal(Omega_HI_fid,scale=0.1*Omega_HI_fid,size=nwalkers)
        b_HI_p0 = np.random.normal(b_HI_fid,scale=0.1*b_HI_fid,size=nwalkers)
        p0 = np.swapaxes(np.array([OmHI_p0,b_HI_p0]),0,1)

    backend = emcee.backends.HDFBackend(backendfile)
    if ContinueBackend==False: backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend, args=(k,Pk,Pkerr))

    print("\nRunning production...")
    if ContinueBackend==False: pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
    if ContinueBackend==True: #continue from previous chains
        pos, prob, state = sampler.run_mcmc(None, niter, progress=True)
    samples = sampler.chain.reshape((-1, ndim))
    return samples

def JacknifeSpectrum(dT_HI,n_g,dims,kbins,njack,T,w1=None,w2=None,W1=None,W2=None,ra=None,dec=None,nu=None,FullCov=False):
    lx,ly,lz,nx,ny,nz = dims
    nkbin = len(kbins)-1
    blackman = np.reshape( np.tile(np.blackman(nz), (nx,ny)) , (nx,ny,nz) ) # Blackman function along every LoS
    PS_jack = np.zeros((njack,nkbin))
    njackcubed = np.int(np.cbrt(njack))
    jackcount = 0
    npix = np.sum(W1)
    npixnjack = int(npix/njack)
    for i in range(njack):
        print(i)
        # Jackknife available pixels:
        dT_HI_jack = np.copy(dT_HI)
        w1_jack = np.copy(w1)
        W1_jack = np.copy(W1)
        dT_HI_masked = dT_HI[np.nonzero(W1)]
        w1_masked = w1[np.nonzero(W1)]
        W1_masked = W1[np.nonzero(W1)]
        x0_jack = i * npixnjack
        x1_jack = (i+1) * npixnjack

        dT_HI_masked[x0_jack:x1_jack] = 0
        w1_masked[x0_jack:x1_jack] = 0
        W1_masked[x0_jack:x1_jack] = 0

        '''
        mask = np.zeros(np.shape(W1_masked))
        mask[x0_jack:x1_jack] = 1
        maskmap = np.zeros(np.shape(dT_HI))
        maskmap[np.nonzero(W1)] = mask
        plottools.PlotMap(maskmap)
        plt.show()
        exit()
        plottools.PlotMap(dT_HI_jack,W=W1)
        plottools.PlotMap(W1_jack)
        '''
        dT_HI_jack[np.nonzero(W1)] = dT_HI_masked
        w1_jack[np.nonzero(W1)] = w1_masked
        W1_jack[np.nonzero(W1)] = W1_masked
        '''
        plottools.PlotMap(dT_HI_jack,W=W1)
        plottools.PlotMap(W1_jack)
        dT_HI_masked = dT_HI[np.nonzero(W1)]
        dT_HI_jack[np.nonzero(W1)] = dT_HI_masked
        plottools.PlotMap(dT_HI_jack,W=W1)
        plt.show()
        exit()
        '''
        dT_HI_jack_rg,dims = grid.regrid(blackman*dT_HI_jack,ra,dec,nu)
        w1_jack_rg,dims = grid.regrid(blackman*w1_jack,ra,dec,nu)
        W1_jack_rg,dims = grid.regrid(blackman*W1_jack,ra,dec,nu)

        pk_gHI,k,nmodes = power.Pk(dT_HI_jack,n_g,dims,kbins,corrtype='Cross',w1=w1_jack,w2=w2,W1=W1_jack,W2=W2)
        ### Apply foreground transfer function to power spectra:
        pk_gHI_TF = pk_gHI/T
        PS_jack[i] = pk_gHI_TF
    '''
    for i in range(njackcubed):
        plottools.ProgressBar(i,njackcubed,'Jackknifing:')
        for j in range(njackcubed):
            for k in range(njackcubed):
                dT_HI_jack = np.copy(dT_HI)
                w1_jack = np.copy(w1)
                W1_jack = np.copy(W1)
                #Create mask to jackknife data with:
                x0_jack = i * (nx/njackcubed)
                x1_jack = (i+1) * (nx/njackcubed)
                y0_jack = j * (ny/njackcubed)
                y1_jack = (j+1) * (ny/njackcubed)
                z0_jack = k * (nz/njackcubed)
                z1_jack = (k+1) * (nz/njackcubed)
                z,y,x = np.indices(np.shape(dT_HI))
                jackmask = (x >= x0_jack) & (x < x1_jack) & (y >= y0_jack)\
                & (y < y1_jack)  & (z >= z0_jack) & (z < z1_jack)
                dT_HI_jack[jackmask] = 0
                w1_jack[jackmask] = 0
                W1_jack[jackmask] = 0

                map = np.ones(np.shape(dT_HI))
                map[jackmask] = 0
                plt.imshow(np.sum(map,0))
                plt.colorbar()
                plt.figure()
                plt.imshow(np.sum(dT_HI_jack,0))
                plt.colorbar()
                plt.figure()
                plt.imshow(np.sum(dT_HI_jack,1))
                plt.colorbar()
                plt.figure()
                plt.imshow(np.sum(dT_HI_jack,2))
                plt.colorbar()
                plt.show()
                exit()

                pk_gHI,k,nmodes = power.Pk(dT_HI_jack,n_g,dims,kbins,corrtype='Cross',w1=w1_jack,w2=w2,W1=W1_jack,W2=W2)
                ### Apply foreground transfer function to power spectra:
                pk_gHI_TF = pk_gHI/T
                PS_jack[jackcount] = pk_gHI_TF
                jackcount += 1
    '''

    #Formula for variance of jackknived data, from wiki: https://en.wikipedia.org/wiki/Jackknife_resampling
    # Also see Norberg (https://arxiv.org/pdf/0810.1885.pdf):
    mu = np.mean(PS_jack,0)
    if FullCov==False:
        var = (njack - 1) / njack * np.sum( ( PS_jack - mu )**2 , 0)
        sig = np.sqrt(var) #Get standard.dev for error bars
        return sig
    if FullCov==True:
        C = np.zeros((nkbin,nkbin)) # Covariance matrix
        for i in range(nkbin):
            for j in range(nkbin):
                #var = (njack - 1) / njack * np.sum( ( PS_jack - mu )**2 , 0)
                C[i,j] = (njack - 1) / njack * np.sum( (PS_jack[:,i] - mu[i])*(PS_jack[:,j] - mu[j]) , 0 )
        return C
