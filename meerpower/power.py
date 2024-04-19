#### Steve Cunnington code with contributions from Chris Blake
import numpy as np
import model

########################################################################
# Estimate the 3D power spectrum of a density field.                   #
########################################################################

def Pk(f1,f2,dims,kbins,corrtype='HIauto',w1=None,w2=None,W1=None,W2=None,kcuts=None,doMultipole=False):
    ### *** IF CROSS-CORRELATING: assumes f1 = HI field and f2 = galaxy field ****
    ### if auto-correlating then set f1=f2, norm1=norm2, w1=w2 and W1=W2
    ######################################################################
    # f1/f2: fields to correlate (f=n_g for galaxies, f=dT_HI for IM)
    # corrtype: type of correlation to compute, options are:
    #   - corrtype='HIauto': (default) for HI auto-correlation of temp fluctuation field dT_HI = T_HI - <T_HI>
    #   - corrtype='Galauto': for galaxy auto-correlation of number counts field n_g
    #   - corrtype='Cross': for HI-galaxy cross-correlation <dT_HI,n_g>
    #   - for HI IM:   norm = None (assuming fluctuation field is the input f field)
    #   - for galxies: norm = N_g (total galaxy count in input f field)
    # w1/w2: optional field weights
    # W1/W2: optional survey selection functions
    # noise: optional noise term to subtract (only if doing auto-correlation)
    #   - for HI IM:   noise = P_N (thermal noise [*DEVELOP THIS FOR WORKING AUTO-HI-CORRELATION*])
    #   - for galxies: noise = 'shot' will include shot-noise subtraction
    # kcuts = [kperpmin,kparamin,kperpmax,kparamax]: If given, power spectrum only measured within this scale range
    ######################################################################
    pkspec = getpkspec(f1,f2,dims,corrtype,w1,w2,W1,W2)
    if doMultipole==False:
        Pk,k,nmodes = binpk(pkspec,dims,kbins,kcuts)
        return Pk,k,nmodes
    if doMultipole==True: # Do multipole decomposition and return Monopole (P0), Quadrupole (P2) and Hexadecapole (P4)
        pk0,pk2,pk4,k,nmodes = binpole(pkspec,dims,kbins)
        return pk0,pk2,pk4,k,nmodes

def Pk2D(f1,f2,dims,kperpbins,kparabins,corrtype='HIauto',w1=None,w2=None,W1=None,W2=None):
    '''
    Same as 1D Pk() function but for 2D cylindrical power spectrum
    '''
    lx,ly,lz,nx,ny,nz = dims
    pkspec = getpkspec(f1,f2,dims,corrtype,w1,w2,W1,W2)
    Pk2d,k2d,nmodes = binpk2D(pkspec,dims,kperpbins,kparabins)
    return Pk2d,k2d,nmodes

def getpkspec(f1,f2,dims,corrtype='HIauto',w1=None,w2=None,W1=None,W2=None,Ngal=None):
    # Obtain full 3D unbinned power spectrum - follows formalism in Blake+10[https://arxiv.org/pdf/1003.5721.pdf] (sec 3.1)
    #  - see Pk function for variable definitions
    if corrtype=='Galauto' or corrtype=='Cross': Ngal = np.sum(f2) # assumes f2 is galaxy field
    lx,ly,lz,nx,ny,nz = dims
    Vcell = (lx*ly*lz) / (nx*ny*nz)
    # Apply default unity weights/windows if none given:
    if w1 is None: w1 = np.ones((nx,ny,nz))# set to unity if no weights provided
    if w2 is None: w2 = np.ones((nx,ny,nz))# set to unity if no weights provided
    if W1 is None: W1 = np.ones((nx,ny,nz))# set to unity if no weights provided
    if W2 is None: W2 = np.ones((nx,ny,nz))# set to unity if no weights provided
    if corrtype=='Galauto': W1,W2 = W1/np.sum(W1),W2/np.sum(W2) # Normalise galaxy window functions so sum(W)=1
    if corrtype=='Cross': W2 = W2/np.sum(W2) # Normalise galaxy window functions so sum(W)=1
    if corrtype=='HIauto':
        S = 0 ### DEVELOP THIS ### for thermal noise HI IM subtraction
        F1k = np.fft.rfftn(w1*f1)
        F2k = np.fft.rfftn(w2*f2)
        pkspec = np.real( F1k * np.conj(F2k) )
        return Vcell / np.sum(w1*w2) * (pkspec-S) # Normalisation with windows is NOT needed
    if corrtype=='Galauto':

        Wk1 = np.fft.rfftn(w1*W1)
        F1k = np.fft.rfftn(w1*f1) - Ngal*Wk1
        Wk2 = np.fft.rfftn(w2*W2)
        F2k = np.fft.rfftn(w2*f2) - Ngal*Wk2
        pkspec = np.real( F1k * np.conj(F2k) )

        S = Ngal * np.sum(w1**2*W1) * np.ones(np.shape(pkspec)) # shot-noise term
        S /= model.W_mas(dims,window='ngp')**2

        return Vcell / np.sum(w1*w2*W1*W2) * (pkspec-S) * 1/(Ngal**2) # Normalisation with windows is needed

    if corrtype=='Cross':
        S = 0 # noise drops out in cross-correlation
        F1k = np.fft.rfftn(w1*f1)
        Wk2 = np.fft.rfftn(w2*W2)
        F2k = np.fft.rfftn(w2*f2) - Ngal*Wk2
        pkspec = np.real( F1k * np.conj(F2k) )
        return Vcell / np.sum(w1*w2*W2) * (pkspec-S) * 1/Ngal # Only normalisation with galaxy window is needed

########################################################################
# Bin 3D power spectrum in angle-averaged bins.                        #
########################################################################

def binpk(pkspec,dims,kbins,kcuts=None,FullPk=False,doindep=True):
    #### Bin 3D power spectrum in angle-averaged bins
    lx,ly,lz,nx,ny,nz = dims
    kspec,muspec,indep = getkspec(dims,FullPk)
    if kcuts is not None: # Remove kspec (by setting to -1) to exclude from bin average:
        if doindep==True: kperp,kpara,indep_perp,indep_para = getkspec2D(dims,FullPk=FullPk,doindep=doindep)
        if doindep==False: kperp,kpara = getkspec2D(dims,FullPk=FullPk,doindep=doindep)
        kperpcutmin,kparacutmin,kperpcutmax,kparacutmax = kcuts
        if kperpcutmin is not None: kspec[kperp<kperpcutmin] = -1
        if kperpcutmax is not None: kspec[kperp>kperpcutmax] = -1
        if kparacutmin is not None: kspec[kpara<kparacutmin] = -1
        if kparacutmax is not None: kspec[kpara>kparacutmax] = -1
    if doindep==True:
        pkspec = pkspec[indep==True]
        kspec = kspec[indep==True]
    ikbin = np.digitize(kspec,kbins)
    nkbin = len(kbins)-1
    pk,k,nmodes = np.zeros(nkbin),np.zeros(nkbin),np.zeros(nkbin,dtype=int)
    for ik in range(nkbin):
        nmodes[ik] = int(np.sum(np.array([ikbin==ik+1])))
        k[ik] = np.nanmean( kspec[ikbin==ik+1] ) # average k-bin value for notional k
        if (nmodes[ik] > 0): #if nmodes==0 for this k then remains Pk=0
            pk[ik] = np.nanmean(pkspec[ikbin==ik+1])
    return pk,k,nmodes

def binpk2D(pkspec,dims,kperpbins,kparabins,FullPk=False,doindep=True):
    kperpspec,kparaspec,indep_perp,indep_para = getkspec2D(dims,FullPk)
    if doindep==True: # Identify and remove non-independent modes
        pkspec = pkspec[(indep_perp==True) & (indep_para==True)]
        kperpspec = kperpspec[(indep_perp==True) & (indep_para==True)]
        kparaspec = kparaspec[(indep_perp==True) & (indep_para==True)]
    # Get indices where kperp and kpara values fall in bins
    ikbin_perp = np.digitize(kperpspec,kperpbins)
    ikbin_para = np.digitize(kparaspec,kparabins)
    lx,ly,lz,nx,ny,nz = dims
    nkperpbin,nkparabin = len(kperpbins)-1,len(kparabins)-1
    pk2d,k2d,nmodes2d = np.zeros((nkparabin,nkperpbin)),np.zeros((nkparabin,nkperpbin)),np.zeros((nkparabin,nkperpbin),dtype=int)
    for i in range(nkperpbin):
        for j in range(nkparabin):
            ikmask = (ikbin_perp==i+1) & (ikbin_para==j+1) # Use for identifying all kperp,kpara modes that fall in 2D k-bin
            nmodes2d[j,i] = int(np.sum(np.array([ikmask])))
            k2d[j,i] = np.mean( np.sqrt( kperpspec[ikmask]**2 + kparaspec[ikmask]**2 )  ) # average k-bin value for notional kperp, kpara combination
            if (nmodes2d[j,i] > 0):
                # Average power spectrum into (kperp,kpara) cells
                pk2d[j,i] = np.mean(pkspec[ikmask])
    return pk2d,k2d,nmodes2d

########################################################################
# Obtain 3D grid of k-modes.                                           #
########################################################################

def getkspec(dims,FullPk=False,decomp=False):
    lx,ly,lz,nx,ny,nz = dims
    kx = 2*np.pi*np.fft.fftfreq(nx,d=lx/nx)
    ky = 2*np.pi*np.fft.fftfreq(ny,d=ly/ny)
    if FullPk==True or decomp==True: kz = 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)
    else: kz = 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)[:int(nz/2)+1]
    if decomp==True: # Return coordinate tuple (kx,ky,kz) at every point on grid
        kxi,kyj,kzk = np.meshgrid(kx,ky,kz, indexing='ij')
        kspec = np.array([kxi,kyj,kzk])
        kspec = np.swapaxes(kspec,0,1)
        kspec = np.swapaxes(kspec,1,2)
        kspec = np.swapaxes(kspec,2,3)
        return kspec
    indep = getindep(nx,ny,nz)
    indep[0,0,0] = False
    if FullPk==True:
        indep = fthalftofull(nx,ny,nz,indep)
    kspec = np.sqrt(kx[:,np.newaxis,np.newaxis]**2 + ky[np.newaxis,:,np.newaxis]**2 + kz[np.newaxis,np.newaxis,:]**2)
    kspec[0,0,0] = 1 # to avoid divide by zero error
    muspec = np.absolute(kz[np.newaxis,np.newaxis,:])/kspec
    muspec[0,0,0] = 1 # divide by k=0, means mu->1
    kspec[0,0,0] = 0 # reset
    return kspec,muspec,indep

def getkspec2D(dims,do2D=False,FullPk=False,doindep=True):
    '''
    Obtain two 3D arrays specifying kperp and kpara values at every point in
    pkspec array - if do2D==True - return 2D arrays of kperp,kpara
    '''
    lx,ly,lz,nx,ny,nz = dims
    kx = 2*np.pi*np.fft.fftfreq(nx,d=lx/nx)
    ky = 2*np.pi*np.fft.fftfreq(ny,d=ly/ny)
    '''
    if do2D==True:
        if nx!=ny or lx!=ly:
            print('\nError: Perpendicular dimensions must be the same!')
            exit()
        kperp = np.tile(np.sqrt(kx**2 + ky**2),(nx,1))
        kz = 2*np.pi*np.fft.fftfreq(nx,d=lx/nx)
        kpara = np.tile(kz,(nz,1))
        kpara = np.swapaxes(kpara,0,1)
        return kperp,kpara
    '''

    kperp = np.sqrt(kx[:,np.newaxis]**2 + ky[np.newaxis,:]**2)
    if FullPk==False:
        kpara = np.abs( 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)[:int(nz/2)+1] )
        kperpspec = np.reshape( np.repeat(kperp,int(nz/2)+1) , (nx,ny,int(nz/2)+1) )
    if FullPk==True:
        #indep = fthalftofull(nx,ny,nz,indep)
        kpara = 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)
        kperpspec = np.reshape( np.repeat(kperp,nz) , (nx,ny,nz) )
    kparaspec = np.tile(kpara,(nx,ny,1))

    if doindep==True:
        indep = getindep(nx,ny,nz)
        indep[0,0,0] = False
        indep_perp,indep_para = np.copy(indep),np.copy(indep)
        indep_perp[kperpspec==0] = False
        indep_para[kparaspec==0] = False

    kparaspec[0,0,0],kperpspec[0,0,0] = 0.,0.

    '''
    # Sanity check: kspec from kpara,kperp matches kspec function
    kspec = np.sqrt(kperpspec**2 + kparaspec**2)
    kspec0 = getkspec(dims)[0]
    plt.imshow(kspec[:,:,-5])
    plt.colorbar()
    plt.figure()
    plt.imshow(kspec0[:,:,-5])
    plt.colorbar()
    plt.show()
    exit()
    '''

    if doindep==True: return kperpspec,kparaspec,indep_perp,indep_para
    if doindep==False: return kperpspec,kparaspec

########################################################################
# Bin 3D power spectrum in angle-averaged bins, weighting by Legendre  #
# polynomials.                                                         #
########################################################################

def binpole(pkspec,dims,kbins,FullPk=False,doindep=True):
    lx,ly,lz,nx,ny,nz = dims
    kspec,muspec,indep = getkspec(dims,FullPk)
    if doindep==True:
        pkspec = pkspec[indep==True]
        kspec = kspec[indep==True]
        muspec = muspec[indep==True]
    leg2spec = ((3*(muspec**2))-1)/2
    leg4spec = ((35*(muspec**4))-(30*(muspec**2))+3)/8
    ikbin = np.digitize(kspec,kbins)
    nkbin = len(kbins)-1
    nmodes,pk0,pk2,pk4,k = np.zeros(nkbin,dtype=int),np.zeros(nkbin),np.zeros(nkbin),np.zeros(nkbin),np.zeros(nkbin)
    for ik in range(nkbin):
        nmodes[ik] = int(np.sum(np.array([ikbin==ik+1])))
        k[ik] = np.mean( kspec[ikbin==ik+1] ) # average k-bin value for notional k
        if (nmodes[ik] > 0):
            pk0[ik] = np.mean(pkspec[ikbin==ik+1])
            pk2[ik] = 5*np.mean((pkspec*leg2spec)[ikbin==ik+1])
            pk4[ik] = 9*np.mean((pkspec*leg4spec)[ikbin==ik+1])
    return pk0,pk2,pk4,k,nmodes

def getindep(nx,ny,nz):
    ### Obtain array of independent 3D modes
    indep = np.full((nx,ny,int(nz/2)+1),False,dtype=bool)
    indep[:,:,1:int(nz/2)] = True
    indep[1:int(nx/2),:,0] = True
    indep[1:int(nx/2),:,int(nz/2)] = True
    indep[0,1:int(ny/2),0] = True
    indep[0,1:int(ny/2),int(nz/2)] = True
    indep[int(nx/2),1:int(ny/2),0] = True
    indep[int(nx/2),1:int(ny/2),int(nz/2)] = True
    indep[int(nx/2),0,0] = True
    indep[0,int(ny/2),0] = True
    indep[int(nx/2),int(ny/2),0] = True
    indep[0,0,int(nz/2)] = True
    indep[int(nx/2),0,int(nz/2)] = True
    indep[0,int(ny/2),int(nz/2)] = True
    indep[int(nx/2),int(ny/2),int(nz/2)] = True
    return indep

def fthalftofull(nx,ny,nz,halfspec):
    ### Fill full transform given half transform
    fullspec = np.empty((nx,ny,nz))
    ixneg,iyneg,izneg = nx-np.arange(nx),ny-np.arange(ny),nz-np.arange(int(nz/2)+1,nz)
    ixneg[0],iyneg[0] = 0,0
    fullspec[:,:,:int(nz/2)+1] = halfspec
    fullspec[:,:,int(nz/2)+1:nz] = fullspec[:,:,izneg][:,iyneg][ixneg]
    return fullspec

def fthalftofull2(nx,ny,nz,halfspec1,halfspec2):
    fullspec = np.empty((nx,ny,nz))
    ixneg,iyneg,izneg = nx-np.arange(nx),ny-np.arange(ny),nz-np.arange(int(nz/2)+1,nz)
    ixneg[0],iyneg[0] = 0,0
    fullspec[:,:,:int(nz/2)+1] = np.real(halfspec1*np.conj(halfspec2))
    fullspec[:,:,int(nz/2)+1:nz] = fullspec[:,:,izneg][:,iyneg][ixneg]
    return fullspec

def getpkconv(pkspecmod,dims,w1=None,w2=None,W1=None,W2=None):
    # w1/w2: optional field weights
    # W1/W2: optional survey selection functions
    lx,ly,lz,nx,ny,nz = dims
    pkspecmod[0,0,0] = 0
    # Apply default unity weights/windows if none given:
    if w1 is None: w1 = np.ones((nx,ny,nz))# set to unity if no weights provided
    if w2 is None: w2 = np.ones((nx,ny,nz))# set to unity if no weights provided
    if W1 is None: W1 = np.ones((nx,ny,nz))# set to unity if no weights provided
    if W2 is None: W2 = np.ones((nx,ny,nz))# set to unity if no weights provided
    W1,W2 = W1/np.sum(W1),W2/np.sum(W2) # Normalise window functions so sum(W)=1
    Wk1 = np.fft.rfftn(w1*W1)
    Wk2 = np.fft.rfftn(w2*W2)
    Wk = fthalftofull2(nx,ny,nz,Wk1,Wk2) ; del Wk1; del Wk2
    # FFT model P(k) and W(k) (despite already being in Fourier space) in order to
    #   use convolution theorem and multiply Fourier transforms together:
    pkspecmodFT = np.fft.rfftn(pkspecmod)
    Wk1FT = np.fft.rfftn(Wk); del Wk
    pkcongrid = np.fft.irfftn(pkspecmodFT*Wk1FT) # Inverse Fourier transform
    return pkcongrid / ( nx*ny*nz * np.sum(w1*w2*W1*W2) )

########################################################################
# Functions for Jing05 NGP correction
########################################################################
def W_field(dims,p=1,FullPk=False,interlace=False):
    lx,ly,lz,nx,ny,nz = dims
    nyqx,nyqy,nyqz = nx*np.pi/lx,ny*np.pi/ly,nz*np.pi/lz
    kx = 2*np.pi*np.fft.fftfreq(nx,d=lx/nx)[:,np.newaxis,np.newaxis]
    ky = 2*np.pi*np.fft.fftfreq(ny,d=ly/ny)[np.newaxis,:,np.newaxis]
    if FullPk==False: kz = 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)[:int(nz/2)+1][np.newaxis,np.newaxis,:]
    if FullPk==True: kz = 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)[np.newaxis,np.newaxis,:]
    qx,qy,qz = (np.pi*kx)/(2*nyqx),(np.pi*ky)/(2*nyqy),(np.pi*kz)/(2*nyqz)
    wx = np.divide(np.sin(qx),qx,out=np.ones_like(qx),where=qx!=0.)
    wy = np.divide(np.sin(qy),qy,out=np.ones_like(qy),where=qy!=0.)
    wz = np.divide(np.sin(qz),qz,out=np.ones_like(qz),where=qz!=0.)
    W = (wx*wy*wz)**(p)
    return W

def W(dims,p=1,field=None,FullPk=False):
    lx,ly,lz,nx,ny,nz = dims
    nyqx,nyqy,nyqz = nx*np.pi/lx,ny*np.pi/ly,nz*np.pi/lz
    kx = 2*np.pi*np.fft.fftfreq(nx,d=lx/nx)[:,np.newaxis,np.newaxis]
    ky = 2*np.pi*np.fft.fftfreq(ny,d=ly/ny)[np.newaxis,:,np.newaxis]
    if FullPk==False: kz = 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)[:int(nz/2)+1][np.newaxis,np.newaxis,:]
    if FullPk==True: kz = 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)[np.newaxis,np.newaxis,:]
    qx,qy,qz = (np.pi*kx)/(2*nyqx),(np.pi*ky)/(2*nyqy),(np.pi*kz)/(2*nyqz)
    wx = np.divide(np.sin(qx),qx,out=np.ones_like(qx),where=qx!=0.)
    wy = np.divide(np.sin(qy),qy,out=np.ones_like(qy),where=qy!=0.)
    wz = np.divide(np.sin(qz),qz,out=np.ones_like(qz),where=qz!=0.)
    W = (wx*wy*wz)**p
    #'''
    sum = 0
    shiftarray = np.linspace(-1,1,3) # integer vectors by which to nudge the nyquist freq.
    for ix in shiftarray:
        for iy in shiftarray:
            for iz in shiftarray:
                kx1 = kx + 2*nyqx*ix
                ky1 = ky + 2*nyqy*iy
                kz1 = kz + 2*nyqz*iz
                qx1,qy1,qz1 = (np.pi*kx1)/(2*nyqx),(np.pi*ky1)/(2*nyqy),(np.pi*kz1)/(2*nyqz)
                wx = np.divide(np.sin(qx1),qx1,out=np.ones_like(qx1),where=qx1!=0.)
                wy = np.divide(np.sin(qy1),qy1,out=np.ones_like(qy1),where=qy1!=0.)
                wz = np.divide(np.sin(qz1),qz1,out=np.ones_like(qz1),where=qz1!=0.)

                W1 = (wx*wy*wz)**p
                if field is None: sum += W1/W
                if field is not None: sum += field / W1

    W = sum
    #'''
    return W


def C_1(dims,p=1):
    # Analytical shot noise aliasing correction - from Jing eq:20
    # not required for NGP
    lx,ly,lz,nx,ny,nz = dims
    if p==1: return 1 # Jing05
    kx = 2*np.pi*np.fft.fftfreq(nx,d=lx/nx)[:,np.newaxis,np.newaxis]
    ky = 2*np.pi*np.fft.fftfreq(ny,d=ly/ny)[np.newaxis,:,np.newaxis]
    kz = 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)[:int(nz/2)+1][np.newaxis,np.newaxis,:]
    kx[kx==0]=1e-30; ky[ky==0]=1e-30; kz[kz==0]=1e-30 #Amend to avoid divide by zeros
    #if p==2: return (1 - 2/3*np.sin(kx*lx/(2*nx))**2)*(1 - 2/3*np.sin(ky*ly/(2*ny))**2)*(1 - 2/3*np.sin(kz*lz/(2*nz))**2)
    if p==2: return 1 - 2/3*np.sin(kx*lx/(2*nx))**2
    if p==3: return 1 - np.sin(kx*lx/(2*nx))**2 + 2/15*np.sin(kx*lx/(2*nx))**4
    # Below derived in Nbodykit CompensatePCSShotnoise: https://github.com/bccp/nbodykit/blob/376c9d78204650afd9af81d148b172804432c02f/nbodykit/source/mesh/catalog.py#L573
    if p==4: return 1 - 4/3*np.sin(kx*lx/(2*nx))**2 + 2/5*np.sin(kx*lx/(2*nx))**4 - 4/315*np.sin(kx*lx/(2*nx))**6

def PkSumOverNyquist(dims,Pmod,Pnoise,b1,b2,f,sigv=0,Tbar1=1,Tbar2=1,r=1,R_beam1=0,R_beam2=0,sig_N=0,w1=None,w2=None,W1=None,W2=None,s_pix=0,s_para=0,p=1):
    lx,ly,lz,nx,ny,nz = dims
    nyqx,nyqy,nyqz = nx*np.pi/lx,ny*np.pi/ly,nz*np.pi/lz
    kx = 2*np.pi*np.fft.fftfreq(nx,d=lx/nx)[:,np.newaxis,np.newaxis]
    ky = 2*np.pi*np.fft.fftfreq(ny,d=ly/ny)[np.newaxis,:,np.newaxis]
    kz = 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)[np.newaxis,np.newaxis,:]
    qx,qy,qz = (np.pi*kx)/(2*nyqx),(np.pi*ky)/(2*nyqy),(np.pi*kz)/(2*nyqz)
    wx = np.divide(np.sin(qx),qx,out=np.ones_like(qx),where=qx!=0.)
    wy = np.divide(np.sin(qy),qy,out=np.ones_like(qy),where=qy!=0.)
    wz = np.divide(np.sin(qz),qz,out=np.ones_like(qz),where=qz!=0.)
    W = (wx*wy*wz)**p
    # Sum correction for FFT grid aliasing over integers n=[-1,0,1] as in https://arxiv.org/pdf/1902.07439.pdf step(v) pg 13:
    kx = 2*np.pi*np.fft.fftfreq(nx,d=lx/nx)[:,np.newaxis,np.newaxis]
    ky = 2*np.pi*np.fft.fftfreq(ny,d=ly/ny)[np.newaxis,:,np.newaxis]
    kz = 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)[np.newaxis,np.newaxis,:]
    kspec,muspec,indep = getkspec(dims,FullPk=True)
    pkspecmod = model.PkModSpec(Pmod,dims,kspec,muspec,b1,b2,f,sigv,Tbar1,Tbar2,r,R_beam1,R_beam2,sig_N,w1,w2,W1,W2,s_pix=s_pix,s_para=s_para,gridinterp=True)
    Pnoisespec = Pnoise * np.ones((nx,ny,nz)) # Assumes constant, scale invariant noise
    sum = 0
    ### Choise of integer shifts: larger range the more accurate the correction but diminishing
    # returns above [-1,0,1] - should the x,y-directions cover more(?), since the Nyquist frequency
    # is much closer to k-range we measure in. LoS (freq) direction has large k~>4 Nyq freq.
    shiftarray = np.linspace(-1,1,3) # integer vectors by which to nudge the nyquist freq.
    #shiftarray = np.linspace(-2,2,5) # integer vectors by which to nudge the nyquist freq.
    #shiftarray = np.linspace(-3,3,7) # integer vectors by which to nudge the nyquist freq.
    #shiftarray = np.linspace(-5,5,11) # integer vectors by which to nudge the nyquist freq.
    for ix1 in shiftarray:
        print(ix1)
        for iy1 in shiftarray:
            for iz1 in shiftarray:
                for ix2 in shiftarray:
                    for iy2 in shiftarray:
                        for iz2 in shiftarray:
                            kx1 = kx + 2*nyqx*ix1
                            ky1 = ky + 2*nyqy*iy1
                            kz1 = kz + 2*nyqz*iz1
                            kspec1 = np.sqrt(kx1**2 + ky1**2 + kz1**2)
                            kspec1[0,0,0] = 1 # to avoid divide by zero error
                            muspec1 = kz1/kspec1
                            muspec1[0,0,0] = 1 # divide by k=0, means mu->1
                            kspec1[0,0,0] = 0 # reset
                            qx1,qy1,qz1 = (np.pi*kx1)/(2*nyqx),(np.pi*ky1)/(2*nyqy),(np.pi*kz1)/(2*nyqz)
                            wx1 = np.divide(np.sin(qx1),qx1,out=np.ones_like(qx1),where=qx1!=0.)
                            wy1 = np.divide(np.sin(qy1),qy1,out=np.ones_like(qy1),where=qy1!=0.)
                            wz1 = np.divide(np.sin(qz1),qz1,out=np.ones_like(qz1),where=qz1!=0.)
                            W1 = (wx1*wy1*wz1)**p

                            kx2 = kx + 2*nyqx*ix2
                            ky2 = ky + 2*nyqy*iy2
                            kz2 = kz + 2*nyqz*iz2
                            kspec2 = np.sqrt(kx2**2 + ky2**2 + kz2**2)
                            kspec2[0,0,0] = 1 # to avoid divide by zero error
                            muspec2 = kz2/kspec2
                            muspec2[0,0,0] = 1 # divide by k=0, means mu->1
                            kspec2[0,0,0] = 0 # reset
                            qx2,qy2,qz2 = (np.pi*kx2)/(2*nyqx),(np.pi*ky2)/(2*nyqy),(np.pi*kz2)/(2*nyqz)
                            wx2 = np.divide(np.sin(qx2),qx2,out=np.ones_like(qx2),where=qx2!=0.)
                            wy2 = np.divide(np.sin(qy2),qy2,out=np.ones_like(qy2),where=qy2!=0.)
                            wz2 = np.divide(np.sin(qz2),qz2,out=np.ones_like(qz2),where=qz2!=0.)
                            W2 = (wx2*wy2*wz2)**p

                            if Pmod==0: pkspecmod = 0
                            #else:
                                #pkspecmod1 = model.PkModSpec(Pmod,dims,kspec1,muspec1,b1,b2,f,sigv,Tbar1,Tbar2,r,R_beam1,R_beam2,sig_N,w1,w2,W1,W2,s_pix=s_pix,s_para=s_para,gridinterp=True)
                                #pkspecmod2 = model.PkModSpec(Pmod,dims,kspec2,muspec2,b1,b2,f,sigv,Tbar1,Tbar2,r,R_beam1,R_beam2,sig_N,w1,w2,W1,W2,s_pix=s_pix,s_para=s_para,gridinterp=True)
                            #Pmodspec = Pnoisespec + pkspecmod
                            Pmodspec = Pnoisespec + np.sqrt(pkspecmod*pkspecmod)
                            #Pmodspec = Pnoisespec + np.sqrt(pkspecmod1*pkspecmod2)
                            sum += W1/W * W2/W * Pmodspec # eq28 Sefusatti

    return sum

def C_2(dims,Pmod,Pnoise,b1,b2,f,sigv=0,Tbar1=1,Tbar2=1,r=1,R_beam1=0,R_beam2=0,sig_N=0,w1=None,w2=None,W1=None,W2=None,s_pix=0,s_para=0,Damp=None,p=1):
    # Sum correction for FFT grid aliasing.                                #
    '''
    Follow steps in Jing05 to obtain correction to model power spectrum from
    NGP assignment
    Defined in Blake+10 (eq 14) = Sum{ H**2(k1) * Pkmod(k1) } / Pkmod(k)
    For auto-correlation use b1=b2. For optical surveys use Rbeam=0
    '''
    lx,ly,lz,nx,ny,nz = dims
    nyqx,nyqy,nyqz = nx*np.pi/lx,ny*np.pi/ly,nz*np.pi/lz
    kx = 2*np.pi*np.fft.fftfreq(nx,d=lx/nx)[:,np.newaxis,np.newaxis]
    ky = 2*np.pi*np.fft.fftfreq(ny,d=ly/ny)[np.newaxis,:,np.newaxis]
    kz = 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)[np.newaxis,np.newaxis,:]
    #kz = 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)[:int(nz/2)+1][:int(nz/2)+1][np.newaxis,np.newaxis,:]
    kspec,muspec,indep = getkspec(dims,FullPk=True)
    # Use model power spectrum (if not available, need to use iterative method):
    Pnoisespec = Pnoise * np.ones(np.shape(kspec)) # Assumes constant, scale invariant noise
    if Pmod==0: pkspecmod = 0
    else: pkspecmod = model.PkModSpec(Pmod,dims,kspec,muspec,b1,b2,f,sigv,Tbar1,Tbar2,r,R_beam1,R_beam2,sig_N,w1,w2,W1,W2,s_pix=s_pix,s_para=s_para,Damp=Damp,gridinterp=True)
    Pmodspec = Pnoisespec + pkspecmod
    # Sum correction for FFT grid aliasing over integers n=[-1,0,1] as in https://arxiv.org/pdf/1902.07439.pdf step(v) pg 13:
    sum1 = 0

    ### Choise of integer shifts: larger range the more accurate the correction but diminishing
    # returns above [-1,0,1] - should the x,y-directions cover more(?), since the Nyquist frequency
    # is much closer to k-range we measure in. LoS (freq) direction has large k~>4 Nyq freq.
    shiftarray = np.linspace(-1,1,3) # integer vectors by which to nudge the nyquist freq.
    #shiftarray = np.linspace(-2,2,5) # integer vectors by which to nudge the nyquist freq.
    #shiftarray = np.linspace(-3,3,7) # integer vectors by which to nudge the nyquist freq.
    #shiftarray = np.linspace(-5,5,11) # integer vectors by which to nudge the nyquist freq.

    for ix in shiftarray:
        for iy in shiftarray:
            for iz in shiftarray:
                kx1 = kx + 2*nyqx*ix
                ky1 = ky + 2*nyqy*iy
                kz1 = kz + 2*nyqz*iz
                kspec1 = np.sqrt(kx1**2 + ky1**2 + kz1**2)
                kspec1[0,0,0] = 1 # to avoid divide by zero error
                muspec1 = kz1/kspec1
                muspec1[0,0,0] = 1 # divide by k=0, means mu->1
                kspec1[0,0,0] = 0 # reset

                qx1,qy1,qz1 = (np.pi*kx1)/(2*nyqx),(np.pi*ky1)/(2*nyqy),(np.pi*kz1)/(2*nyqz)
                wx = np.divide(np.sin(qx1),qx1,out=np.ones_like(qx1),where=qx1!=0.)
                wy = np.divide(np.sin(qy1),qy1,out=np.ones_like(qy1),where=qy1!=0.)
                wz = np.divide(np.sin(qz1),qz1,out=np.ones_like(qz1),where=qz1!=0.)
                W = (wx*wy*wz)**p

                if Pmod==0: pkspecmod = 0
                else: pkspecmod = model.PkModSpec(Pmod,dims,kspec1,muspec1,b1,b2,f,sigv,Tbar1,Tbar2,r,R_beam1,R_beam2,sig_N,w1,w2,W1,W2,s_pix=s_pix,s_para=s_para,Damp=Damp,gridinterp=True)
                Pmodspec1 = Pnoisespec + pkspecmod

                #sum1 += W**2
                #sum1 += Pmodspec1/W**2
                sum1 += W**2*Pmodspec1 * theta_n(ix,iy,iz)
                #sum1 += Pmodspec1
                #sum1 += W**2*Pmodspec1 * model.B_vox(dims)**2
    '''
    qx,qy,qz = (np.pi*kx)/(2*nyqx),(np.pi*ky)/(2*nyqy),(np.pi*kz)/(2*nyqz)
    wx = np.divide(np.sin(qx),qx,out=np.ones_like(qx),where=qx!=0.)
    wy = np.divide(np.sin(qy),qy,out=np.ones_like(qy),where=qy!=0.)
    wz = np.divide(np.sin(qz),qz,out=np.ones_like(qz),where=qz!=0.)
    W = (wx*wy*wz)**p
    #sum1 /= W**2
    #sum1 *= W**2
    '''

    #C2 = sum1 / Pmodspec
    C2 = sum1
    #return C2[:,:,:int(nz/2)+1] # Cut in half to make applicable to data
    return C2

def theta_n(ix,iy,iz):
    # eq36 Sefusatti
    return 1
    #return 1/2*(1 + (-1)**(ix+iy+iz) )
