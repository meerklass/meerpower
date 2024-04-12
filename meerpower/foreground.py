import numpy as np
import matplotlib.pyplot as plt
import Init
import power
import plot
import telescope
import grid
import cosmo
import HItools
import model
#import healpy as hp - only needed in some functions for sims which are currently not used

def PCAclean(M,N_fg,w=None,W=None,returnAnalysis=False,MeanCentre=False):
    # N_fg: number of eigenmodes for PCA to remove
    # w: inverse noise weights
    # W: binary window function, used to calculate correct mean-centring
    nx,ny,nz = np.shape(M)
    M = np.reshape(M,(nx*ny,nz))
    if w is not None: w = np.reshape(w,(nx*ny,nz))
    if W is not None: W = np.reshape(W,(nx*ny,nz))
    ### Mean Centring and axes configuration:
    if MeanCentre is True:
        if W is None: M = M - np.mean(M,0) # Mean centre data
        else:
            for i in range(np.shape(M)[1]): # Have to loop, since mask collapses arrange to 1D and doesn't allow mean along an axis
                if len(M[:,i][W[:,i]==1])>0: # skip empty channels to avoid mean of empty slice error
                    M[:,i][W[:,i]==1] = M[:,i][W[:,i]==1] - np.nanmean(M[:,i][W[:,i]==1]) # Mean centre data
    M = np.swapaxes(M,0,1) # [Npix,Nz]->[Nz,Npix]
    if W is not None: W = np.swapaxes(W,0,1) # [Npix,Nz]->[Nz,Npix]
    if W is None: W = np.ones(np.shape(M)) # Use unit window everywhere if no window provided
    if w is not None: w = np.swapaxes(w,0,1) # [Npix,Nz]->[Nz,Npix]
    if w is None: w = np.ones(np.shape(M)) # Use unit weights if no weighting provided
    ### Covariance calculation:
    C = np.cov(w*M) # include weight in frequency covariance estimate
    if returnAnalysis==True:
        eigenval = np.linalg.eigh(C)[0]
        eignumb = np.linspace(1,len(eigenval),len(eigenval))
        eigenval = eigenval[::-1] #Put largest eigenvals first
        V = np.linalg.eigh(C)[1][:,::-1] # Eigenvectors from covariance matrix with most dominant first
        return C,eignumb,eigenval,V
    ### Remove dominant modes:
    V = np.linalg.eigh(C)[1][:,::-1] # Eigenvectors from covariance matrix with most dominant first
    A = V[:,:N_fg] # Mixing matrix, first N_fg most dominant modes of eigenvectors
    S = np.dot(A.T,M) # not including weights in mode subtraction (as per Yi-Chao's approach)
    Residual = (M - np.dot(A,S))
    Residual = np.swapaxes(Residual,0,1) #[Nz,Npix]->[Npix,Nz]
    return np.reshape(Residual,(nx,ny,nz)) #Rebuild if M was 3D datacube

def PCAclean_dev(M,N_fg,w=None,W=None,returnAnalysis=False,M2=None,w2=None,W2=None,map4PCA=None):
    # N_fg: number of eigenmodes for PCA to remove
    # w: inverse noise weights
    # W: binary window function, used to calculate correct mean-centring
    # M2: Secondary map to do a cross-PCA clean where covariance is calculated
    #      between both maps M1 and M2 and shared modes projected out both maps
    # map4PCA: optional secondary map if given mixing matrix will be calculated
    #            based on this but then applied to input map M
    nx,ny,nz = np.shape(M)
    M = np.reshape(M,(nx*ny,nz))
    if map4PCA is not None: map4PCA = np.reshape(map4PCA,(nx*ny,nz))
    if w is not None: w = np.reshape(w,(nx*ny,nz))
    if W is not None: W = np.reshape(W,(nx*ny,nz))
    if M2 is not None: M2 = np.reshape(M2,(nx*ny,nz))
    if w2 is not None: w2 = np.reshape(w2,(nx*ny,nz))
    if W2 is not None: W2 = np.reshape(W2,(nx*ny,nz))
    '''
    ### Use for removing dead pixels - does not change results
    if W is not None: # Remove dead pixels for PCA analysis (restore them at the end)
        M_orig = np.copy(M)
        W_orig = np.copy(W)
        if w is not None: w_orig = np.copy(w)
        W_2D = np.ones((nx*ny)) # 2D mask to remove any LoS which are completely empty
        W_2D[np.sum(W,1)==0] = 0
        W_LoS = np.ones(nz) # For masking dead channels, likely removed for RFI
        W_LoS[np.sum(W,0)==0] = 0
        M = M[W_2D==1]
        W = W[W_2D==1]
        if w is not None: w = w[W_2D==1]
        M = M[:,W_LoS==1] # remove dead channels if not filling them with interpolated values
        W = W[:,W_LoS==1] # remove dead channels if not filling them with interpolated values
        if w is not None: w = w[:,W_LoS==1]
    '''
    ### Mean Centring and axes configuration:
    if W is None:
        M = M - np.mean(M,0) # Mean centre data
        if map4PCA is not None: map4PCA = map4PCA - np.mean(map4PCA,0) # Mean centre data
    else:
        for i in range(np.shape(M)[1]): # Have to loop, since mask collapses arrange to 1D and doesn't allow mean along an axis
            if len(M[:,i][W[:,i]==1])>0: # skip empty channels to avoid mean of empty slice error
                M[:,i][W[:,i]==1] = M[:,i][W[:,i]==1] - np.nanmean(M[:,i][W[:,i]==1]) # Mean centre data
                if map4PCA is not None: map4PCA[:,i][W[:,i]==1] = map4PCA[:,i][W[:,i]==1] - np.nanmean(map4PCA[:,i][W[:,i]==1]) # Mean centre data
    if M2 is not None:
        if W2 is None:
            M2 = M2 - np.mean(M2,0) # Mean centre data
        else:
            for i in range(np.shape(M2)[1]): # Have to loop, since mask collapses arrange to 1D and doesn't allow mean along an axis
                if len(M2[:,i][W2[:,i]==1])>0: # skip empty channels to avoid mean of empty slice error
                    M2[:,i][W2[:,i]==1] = M2[:,i][W2[:,i]==1] - np.nanmean(M2[:,i][W2[:,i]==1]) # Mean centre data

    M = np.swapaxes(M,0,1) # [Npix,Nz]->[Nz,Npix]
    if map4PCA is not None: map4PCA = np.swapaxes(map4PCA,0,1) # [Npix,Nz]->[Nz,Npix]
    if W is not None: W = np.swapaxes(W,0,1) # [Npix,Nz]->[Nz,Npix]
    if W is None: W = np.ones(np.shape(M)) # Use unit window everywhere if no window provided
    if w is not None: w = np.swapaxes(w,0,1) # [Npix,Nz]->[Nz,Npix]
    if w is None: w = np.ones(np.shape(M)) # Use unit weights if no weighting provided
    if M2 is not None:
        M2 = np.swapaxes(M2,0,1) # [Npix,Nz]->[Nz,Npix]
        if W2 is not None: W2 = np.swapaxes(W2,0,1) # [Npix,Nz]->[Nz,Npix]
        if W2 is None: W2 = np.ones(np.shape(M2)) # Use unit window everywhere if no window provided
        if w2 is not None: w2 = np.swapaxes(w2,0,1) # [Npix,Nz]->[Nz,Npix]
        if w2 is None: w2 = np.ones(np.shape(M2)) # Use unit weights if no weighting provided


    ### Covariance calculation:
    C = np.cov(w*M) # include weight in frequency covariance estimate
    if M2 is not None: # Cross-correlation covariance
        # Copying numpy.cov code [https://github.com/numpy/numpy/blob/v1.23.0/numpy/lib/function_base.py#L2486-L2705]
        #    but changed so it calculates covariance between two arrays
        wM = w*M
        wM2 = w2*M2
        avg = np.average(wM, axis=1)
        avg2 = np.average(wM2, axis=1)
        wM -= avg[:, None]
        wM2 -= avg2[:, None]
        fact = M.shape[1] - 1
        C = np.dot(wM,wM2.T.conj())
        C *= np.true_divide(1, fact)

    if map4PCA is not None: C = np.cov(w*map4PCA) # include weight in frequency covariance estimate
    '''
    ### Normalising method for covariance calculation to attempt to account for
    ###   freq-dependent mask - makes results worse.
    C = np.dot( (w*M), (w*M).T)
    norm = np.dot( w, w.T)
    #norm = np.dot( W, W.T)
    norm[norm==0] = 1e30 # avoid divide by zero error
    C /= norm
    '''
    if returnAnalysis==True:
        eigenval = np.linalg.eigh(C)[0]
        eignumb = np.linspace(1,len(eigenval),len(eigenval))
        eigenval = eigenval[::-1] #Put largest eigenvals first
        V = np.linalg.eigh(C)[1][:,::-1] # Eigenvectors from covariance matrix with most dominant first
        return C,eignumb,eigenval,V

    ### Remove dominant modes:
    V = np.linalg.eigh(C)[1][:,::-1] # Eigenvectors from covariance matrix with most dominant first
    A = V[:,:N_fg] # Mixing matrix, first N_fg most dominant modes of eigenvectors
    if M2 is None:
        S = np.dot(A.T,M) # not including weights in mode subtraction (as per Yi-Chao's approach)
        Residual = (M - np.dot(A,S))
        Residual = np.swapaxes(Residual,0,1) #[Nz,Npix]->[Npix,Nz]
    else:
        S = np.dot(A.T,M) # not including weights in mode subtraction (as per Yi-Chao's approach)
        S2 = np.dot(A.T,M2) # not including weights in mode subtraction (as per Yi-Chao's approach)
        Residual = (M - np.dot(A,S))
        Residual2 = (M2 - np.dot(A,S2))
        Residual = np.swapaxes(Residual,0,1) #[Nz,Npix]->[Npix,Nz]
        Residual2 = np.swapaxes(Residual2,0,1) #[Nz,Npix]->[Npix,Nz]

    '''
    ### Use for removing dead pixels - does not change results
    if W is not None: # Add back in empty lines-of-sight:
        FullResid = np.zeros((nx*ny,nz))
        W_comb = W_2D[:,np.newaxis] * W_LoS
        np.place(FullResid, W_comb==1, Residual)
        Residual = FullResid; del FullResid
    '''
    if M2 is None:
        return np.reshape(Residual,(nx,ny,nz)) #Rebuild if M was 3D datacube
    else:
        return np.reshape(Residual,(nx,ny,nz)), np.reshape(Residual2,(nx,ny,nz)) #Rebuild if M was 3D datacube

def PCAclean_Test(M,N_fg,w=None,W=None,returnEigenSpec=False,FillChannels=False,Sim=None):
    ''' Development function for testing PCA
    '''
    # N_fg: number of eigenmodes for PCA to remove
    # w: inverse noise weights
    # W: binary window function, used to calculate correct mean-centring
    # Sim: input mock simulation for TF calculation when using YC method (estimating modes on M+S then applying to just S)
    nx,ny,nz = np.shape(M)
    M = np.reshape(M,(nx*ny,nz))
    if w is not None: w = np.reshape(w,(nx*ny,nz))
    if W is not None: W = np.reshape(W,(nx*ny,nz))
    if Sim is not None: Sim = np.reshape(Sim,(nx*ny,nz))

    '''
    #### TODO: Investigate if this approach makes any difference when also weighting the field:
    if W is not None: # Remove dead pixels for PCA analysis (restore them at the end)
        W_2D = np.ones((nx*ny)) # 2D mask to remove any LoS which are completely empty
        W_2D[np.sum(W,1)==0] = 0
        M = M[W_2D==1]
        W = W[W_2D==1]
        if w is not None: w = w[W_2D==1]
    '''
    #'''
    if W is not None: # Remove dead pixels for PCA analysis (restore them at the end)
        M_orig = np.copy(M)
        W_orig = np.copy(W)
        if w is not None: w_orig = np.copy(w)
        W_2D = np.ones((nx*ny)) # 2D mask to remove any LoS which are completely empty
        W_2D[np.sum(W,1)==0] = 0
        W_LoS = np.ones(nz) # For masking dead channels, likely removed for RFI
        W_LoS[np.sum(W,0)==0] = 0
        M = M[W_2D==1]
        if Sim is not None: Sim = Sim[W_2D==1]
        W = W[W_2D==1]
        if w is not None: w = w[W_2D==1]
        #'''
        if FillChannels==False:
            M = M[:,W_LoS==1] # remove dead channels if not filling them with interpolated values
            W = W[:,W_LoS==1] # remove dead channels if not filling them with interpolated values
            if w is not None: w = w[:,W_LoS==1]
        #'''

        '''
        ### For only calculating covaraince based on complete LoS:
        W_fullLoS = np.sum(W_orig,1)==np.max(np.sum(W_orig,1))
        W = W_orig[W_fullLoS==True]
        M = M_orig[W_fullLoS==True]
        if w is not None: w = w_orig[W_fullLoS==True]
        M = M[:,W_LoS==1] # remove dead channels if not filling them with interpolated values
        W = W[:,W_LoS==1] # remove dead channels if not filling them with interpolated values
        if w is not None: w = w[:,W_LoS==1]
        '''

    #'''
    '''
    if FillChannels==True:
        if W is None: print('\n Error: Require W input for missing channel interpolation.'); exit()
        from scipy import interpolate
        ichan = np.arange(1,nz+1,1) # Channel numbers (reqd for interpolation)
        #W_cut = W[W_2D==1]
        for i in range(np.shape(M)[0]):
            deadchan = ichan[W[i,:]==0] # Find dead channels
            ### Currently just doing linear interpolation - EXPERIMENT WITH QUADRATIC & CUBIC?
            f = interpolate.interp1d(ichan[W[i,:]==1], M[i,:][W[i,:]==1],fill_value='extrapolate',assume_sorted=True)
            M[i,:][W[i,:]==0] = f(deadchan) # Replace dead channel with interpolated value
            if i==-1:
                plt.plot(ichan,M[i,:])
                plt.scatter(deadchan,M[i,:][W[i,:]==0],color='red',label='Interpolated Fills')
                plt.xlabel(r'Freq Channel')
                plt.ylabel('$T$ [mk]')
                plt.legend()
                plt.show()
                exit()
        W_LoS = np.ones(nz) # Reset LoS mask since dead channels have now been filled-in
    '''
    #'''
    if W is None: M = M - np.mean(M,0) # Mean centre data
    else:
        for i in range(np.shape(M)[1]): # Have to loop, since mask collapses arrange to 1D and doesn't allow mean along an axis
            M[:,i][W[:,i]==1] = M[:,i][W[:,i]==1] - np.mean(M[:,i][W[:,i]==1]) # Mean centre data
            if Sim is not None: Sim[:,i][W[:,i]==1] = Sim[:,i][W[:,i]==1] - np.mean(Sim[:,i][W[:,i]==1]) # Mean centre sim

    #'''
    #M = M - np.sum(M)/np.sum(w)

    M = np.swapaxes(M,0,1) # [Npix,Nz]->[Nz,Npix]
    if w is not None: w = np.swapaxes(w,0,1) # [Npix,Nz]->[Nz,Npix]
    if w is None: w = np.ones(np.shape(M)) # Use unit weights if no weighting provided
    #'''

    '''
    ##### Manul covariance calculation, gives same result as normalised weight method
    nznew,npix = np.shape(M)
    C = np.zeros((nznew,nznew))
    Mbar = np.sum(M)/np.sum(w)
    for i in range(nznew):
        print(i)
        for j in range(nznew):
            n = 0 # counter for number of mutually filled pixels used in calculation
            for p in range(npix):
                if M[i,p]!=0 and M[j,p]!=0:
                    C[i,j] += (M[i,p]-Mbar)*(M[j,p]-Mbar)
                    n += 1
            C[i,j] /= (n - 1) # 1/(n-1) for that element in covariance
    np.save('temp_covariance_manual_calc',C)
    C = np.load('temp_covariance_manual_calc.npy')
    C[C==0] = np.nan
    plt.imshow(C)
    plt.colorbar()
    plt.show()
    exit()
    '''
    if Sim is None: C = np.dot( (w*M), (w*M).T) # include weight in frequency covariance estimate
    else: # Estimate eigenmodes from M + S, then apply to just S
        Sim = np.swapaxes(Sim,0,1) # [Npix,Nz]->[Nz,Npix]
        C = np.dot( (w*M+w*Sim), (w*M+w*Sim).T)

    norm = np.dot( w, w.T)
    norm[norm==0] = 1e30 # avoid divide by zero error
    C /= norm

    if returnEigenSpec==True:
        Cplot = np.copy(C)

        ### normalised covariance i.e. corrleation matrix C = C_ij / sqrt(C_ii * C_jj)
        nznew,npix = np.shape(M)
        for i in range(nznew):
            for j in range(nznew):
                Cplot /= np.sqrt(Cplot[i,i] * Cplot[j,j])
        Cplot[C==0] = np.nan
        plt.figure()
        #plt.imshow(Cplot,vmin=290000,vmax=410000)
        plt.imshow(Cplot)
        plt.colorbar()
        #plt.title(r'$\nu\nu\prime$ Covariance (no missing pixels)',fontsize=18)
        #plt.title(r'$\nu\nu\prime$ Covariance (deleted LoS only)',fontsize=18)
        #plt.title(r'$\nu\nu\prime$ Covariance (deleted channels only)',fontsize=18)
        #plt.title(r'$\nu\nu\prime$ Covariance (deleted channels & delted LoS)',fontsize=18)
        #plt.title(r'$\nu\nu\prime$ Covariance (frequency varying pixel mask same as MeerKAT)',fontsize=18)
        #plt.title(r'$\nu\nu\prime$ Covariance (Same as MeerKAT but used Astrofix)',fontsize=18)
        plt.title(r'$\nu\nu\prime$ Covariance (additional freq masking)',fontsize=18)
        #plt.title(r'$\nu\nu\prime$ Covariance (MeerKAT 11hr field data)',fontsize=18)
        plt.figure()
        #exit()
    if returnEigenSpec==True:
        eigenval = np.linalg.eigh(C)[0]
        eignumb = np.linspace(1,len(eigenval),len(eigenval))
        eigenval = eigenval[::-1] #Put largest eigenvals first
        return eignumb,eigenval
    V = np.linalg.eigh(C)[1][:,::-1] # Eigenvectors from covariance matrix with most dominant first
    '''
    ### Use only if calculating covaraince based on complete LoS:
    print(np.shape(C))
    print(np.shape(V))
    M = np.copy(M_orig)
    W = np.copy(W_orig)
    w = np.copy(w_orig)
    M = M[W_2D==1]
    W = W[W_2D==1]
    w = w[W_2D==1]
    M = M[:,W_LoS==1] # remove dead channels if not filling them with interpolated values
    W = W[:,W_LoS==1] # remove dead channels if not filling them with interpolated values
    w = w[:,W_LoS==1]
    for i in range(np.shape(M)[1]): # Have to loop, since mask collapses arrange to 1D and doesn't allow mean along an axis
        M[:,i][W[:,i]==1] = M[:,i][W[:,i]==1] - np.mean(M[:,i][W[:,i]==1]) # Mean centre data
    M = np.swapaxes(M,0,1) # [Npix,Nz]->[Nz,Npix]
    w = np.swapaxes(w,0,1) # [Npix,Nz]->[Nz,Npix]
    '''
    A = V[:,:N_fg] # Mixing matrix, first N_fg most dominant modes of eigenvectors
    #S = np.dot(A.T,w*M)
    #Residual = 1/w*(M - np.dot(A,S))
    if Sim is None:
        S = np.dot(A.T,M) # not including weights in mode subtraction (as per Yi-Chao's approach)
        Residual = (M - np.dot(A,S))
    else:
        S = np.dot(A.T,Sim) # not including weights in mode subtraction (as per Yi-Chao's approach)
        Residual = (Sim - np.dot(A,S))
    Residual = np.swapaxes(Residual,0,1) #[Nz,Npix]->[Npix,Nz]

    #'''
    if W is not None: # Add back in empty lines-of-sight:
        FullResid = np.zeros((nx*ny,nz))
        W_comb = W_2D[:,np.newaxis] * W_LoS
        np.place(FullResid, W_comb==1, Residual)
        Residual = FullResid; del FullResid
    #'''
    if FillChannels==True: # Remove filled-in interpolated values in final cleaned data and mean centre
        Residual[W_orig==0] = 0
        Residual_unfill = np.zeros(np.shape(Residual))
        #Residual_unfill[W_orig==1] = Residual[W_orig==1] - np.mean(Residual[W_orig==1])
        Residual_unfill[W_orig==1] = Residual[W_orig==1]
        Residual = Residual_unfill; del Residual_unfill

    return np.reshape(Residual,(nx,ny,nz)) #Rebuild if M was 3D datacube

def CleanLevel5Map(cube,counts,nu,w=None,trimcut=None):
    # Use for cleaning dish and time maps individually before combining
    nx,ny,nz = np.shape(cube)
    cube[counts!=0] = cube[counts!=0]/counts[counts!=0] ### This method is same as loading Tsky_xy_p0.3d.fits without '_Sum_'
    if trimcut is not None: # trim edges in RA,Dec
        cube[trimcut] = 0
        counts[trimcut] = 0
    W = np.ones(np.shape(cube)) # Binary mask: 1 where pixel filled, 0 otherwise
    W[cube==0] = 0
    W = np.ones(np.shape(cube)) # Binary mask: 1 where pixel filled, 0 otherwise
    W[cube==0] = 0
    # Keep only complete LoS pixels:
    W_fullLoS = np.sum(W,2)==np.max(np.sum(W,2))
    W[np.logical_not(W_fullLoS)] = 0
    cube[np.logical_not(W_fullLoS)] = 0
    counts[np.logical_not(W_fullLoS)] = 0
    C,eignumb,eigenval,V = PCAclean(cube,N_fg=1,W=W,w=w,returnAnalysis=True) # weights included in covariance calculation
    flagthresh = 0.008 # Remove any PCA modes with mean absoloute fit above this
    Num = 6
    N_fg = 0
    for i in range(Num):
        poly = model.FitPolynomial(nu,V[:,i],n=3)
        cleanflag = False
        if np.mean(np.abs(poly)) > flagthresh: cleanflag = True
        if cleanflag==True: N_fg+=1
        if cleanflag==False: break
    '''
    Vnan = np.copy(V)
    Vnan[V==0] = np.nan
    M = np.reshape(cube,(nx*ny,nz))
    M = np.swapaxes(M,0,1) # [Npix,Nz]->[Nz,Npix]
    plt.figure(figsize=(18,10))
    panel = 0
    for i in range(Num):
        panel += 1
        plt.subplot(3,4,panel)
        plt.axhline(0,color='black',lw=1)
        plt.plot(nu,Vnan[:,i],label='eigenmode %s'%(i+1))
        poly = model.FitPolynomial(nu,V[:,i],n=3)
        cleanflag = False
        if np.mean(np.abs(poly)) > flagthresh: cleanflag = True
        if cleanflag==False: plt.legend(fontsize=16)
        if cleanflag==True: plt.legend(fontsize=16,labelcolor='red')
        plt.plot(nu,poly,ls='--',color='black')
        plt.xticks([])
        plt.yticks([])
        panel += 1
        plt.subplot(3,4,panel)
        A = np.swapaxes([V[:,i]],0,1) # Mixing matrix
        S = np.dot(A.T,M)
        sourcemap = np.dot(A,S)
        sourcemap = np.swapaxes(sourcemap,0,1) #[Nz,Npix]->[Npix,Nz]
        sourcemap = np.reshape(sourcemap,(nx,ny,nz))
        plt.imshow(np.mean(sourcemap,2).T)
        plt.xticks([])
        plt.yticks([])
        plt.title(r'eigensource %s'%(i+1),fontsize=16)
    #plt.suptitle(nights[n]+'_m0'+dishes[d])
    #plt.savefig('plots/Eigenvecs_%s.png'%(nights[n]+'_m0'+dishes[d]),bbox_inches='tight')
    #plt.close()
    plt.figure()
    #exit()
    '''
    cleancube = PCAclean(cube,N_fg,W=W)
    cleancube[counts!=0] = cleancube[counts!=0] * counts[counts!=0] # Reweight by counts for correct averaging later when all maps combined
    return cleancube,counts

def TransferFunction(dT_obs,Nmock,N_fg,corrtype,kbins,k,TFfile,ra,dec,nu,wproj,dims0_rg,Np,window,compensate,interlace,mockfilepath_HI,mockfilepath_g=None,gal_cat=None,gamma=None,D_dish=None,w_HI=None,W_HI=None,doWeightFGclean=False,PCAMeanCentre=False,w_HI_rg=None,W_HI_rg=None,w_g_rg=None,W_g_rg=None,kcuts=None,taper_HI=1,taper_g=1,LoadTF=False,TF2D=False,kperpbins=None,kparabins=None):
    # Loop over Nmock number of mocks injected into real data to compute transfer function
    # Assumes each mock is saved as ""[mockfilepath]_i.npy" where i = {0,Nmock-1}
    # corrtype: type of correlation to compute Transfer function for, options are:
    #   - corrtype='HIauto': for HI auto-correlation of temp fluctuation field dT_HI = T_HI - <T_HI>
    #   - corrtype='Cross': for HI-galaxy cross-correlation <dT_HI,n_g>
    dims_rg = dims0_rg[:6]
    ### Load pre-saved data if requested:
    if LoadTF==True:
        T,T_nosub = np.load(TFfile+'.npy',allow_pickle=True)
        k_TF = np.load(TFfile+'_karray.npy',allow_pickle=True)
        if TF2D==False:
            if k is not None:  # Check TF matches k-bins using - only possible if k is provided
                if len(k)!=len(k_TF):
                    print('\n Error: Loaded transfer function contains different k-bins\n'); exit()
                if np.allclose(np.array(k,dtype='float32'),np.array(k_TF,dtype='float32'))==False:
                    print('\n Error: Loaded transfer function contains different k-bins\n'); exit()
        return T,T_nosub,k_TF
    ### If no pre-saved TF, run calculation:
    if gamma is not None: dT_obs_resmooth,w_HI = telescope.weighted_reconvolve(dT_obs,w_HI,W_HI,ra,dec,nu,D_dish,gamma=gamma)
    else: dT_obs_resmooth = np.copy(dT_obs)
    if doWeightFGclean==True: w_FG = w_HI
    else: w_FG = None
    dT_clean_data = PCAclean(dT_obs_resmooth,N_fg,w=w_FG,W=W_HI,MeanCentre=PCAMeanCentre)
    ra_p,dec_p,nu_p,pixvals = grid.SkyPixelParticles(ra,dec,nu,wproj,map=dT_clean_data,W=W_HI,Np=Np)
    xp,yp,zp = grid.SkyCoordtoCartesian(ra_p,dec_p,HItools.Freq2Red(nu_p),ramean_arr=ra,decmean_arr=dec,doTile=False)
    dT_clean_data_rg,W_fft,counts = grid.mesh(xp,yp,zp,pixvals,dims0_rg,window,compensate,interlace,verbose=False)
    #dT_clean_data_rg = taper_HI*dT_clean_data_rg
    T,T_nosub = [],[]
    for i in range(Nmock):
        plot.ProgressBar(i,Nmock,header='\nConstructing transfer function...')
        # Read-in HI IM mock and mock galaxies:
        dT_mock = np.load(mockfilepath_HI + '_' + str(i) + '.npy')
        if corrtype=='Cross':
            ra_g,dec_g,z_g = np.load(mockfilepath_g + '_' + str(i) + '.npy')
            if gal_cat=='cmass':
                zmin,zmax = HItools.Freq2Red(np.max(nu)),HItools.Freq2Red(np.min(nu))
                ra_g,dec_g,z_g = Init.pre_process_2019Lband_CMASS_galaxies(ra_g,dec_g,z_g,ra,dec,zmin,zmax,W_HI)
            ### Grid mock galaxies:
            xp,yp,zp = grid.SkyCoordtoCartesian(ra_g,dec_g,z_g,ramean_arr=ra,decmean_arr=dec,doTile=False)
            n_g_rg,W_fft,counts = grid.mesh(xp,yp,zp,dims=dims0_rg,window=window,compensate=compensate,interlace=interlace,verbose=False)
            #n_g_rg = taper_g*n_g_rg
        # Inject HI mock into data, clean and regrid both cleaned and original mock:
        if gamma is not None:
            dT_obs_mock_resmooth = telescope.weighted_reconvolve(dT_mock + dT_obs,w_HI,W_HI,ra,dec,nu,D_dish,gamma=gamma)[0]
            dT_mock = telescope.weighted_reconvolve(dT_mock,w_HI,W_HI,ra,dec,nu,D_dish,gamma=gamma)[0]
        else: dT_obs_mock_resmooth = dT_mock + dT_obs


        dT_obs_mock_resmooth[W_HI==0] = 0 # ensure same pixels are flagged
        dT_mock[W_HI==0] = 0 # ensure same pixels are flagged


        dT_clean_mock = PCAclean(dT_obs_mock_resmooth,N_fg,w=w_FG,W=W_HI,MeanCentre=PCAMeanCentre)
        ra_p,dec_p,nu_p,pixvals = grid.SkyPixelParticles(ra,dec,nu,wproj,map=dT_clean_mock,W=W_HI,Np=Np)
        xp,yp,zp = grid.SkyCoordtoCartesian(ra_p,dec_p,HItools.Freq2Red(nu_p),ramean_arr=ra,decmean_arr=dec,doTile=False)
        dT_clean_mock_rg,W_fft,counts = grid.mesh(xp,yp,zp,pixvals,dims0_rg,window,compensate,interlace,verbose=False)
        ra_p,dec_p,nu_p,pixvals = grid.SkyPixelParticles(ra,dec,nu,wproj,map=dT_mock,W=W_HI,Np=Np)
        xp,yp,zp = grid.SkyCoordtoCartesian(ra_p,dec_p,HItools.Freq2Red(nu_p),ramean_arr=ra,decmean_arr=dec,doTile=False)
        dT_mock_rg,W_fft,counts = grid.mesh(xp,yp,zp,pixvals,dims0_rg,window,compensate,interlace,verbose=False)
        #dT_clean_mock_rg,dT_mock_rg = taper_HI*dT_clean_mock_rg,taper_HI*dT_mock_rg # multiply tapering window by HI map
        if corrtype=='HIauto':
            if TF2D==False:
                Pk_dm,k,nmodes = power.Pk(dT_clean_mock_rg-dT_clean_data_rg , dT_mock_rg , dims_rg,kbins,corrtype='HIauto',w1=w_HI_rg,w2=w_HI_rg,W1=W_HI_rg,W2=W_HI_rg,kcuts=kcuts)
                Pk_dm_nosub,k,nmodes = power.Pk(dT_clean_mock_rg , dT_mock_rg , dims_rg,kbins,corrtype='HIauto',w1=w_HI_rg,w2=w_HI_rg,W1=W_HI_rg,W2=W_HI_rg,kcuts=kcuts)
                Pk_mm,k,nmodes = power.Pk(dT_mock_rg , dT_mock_rg , dims_rg,kbins,corrtype='HIauto',w1=w_HI_rg,w2=w_HI_rg,W1=W_HI_rg,W2=W_HI_rg,kcuts=kcuts)
            if TF2D==True:
                Pk_dm,k,nmodes = power.Pk2D(dT_clean_mock_rg-dT_clean_data_rg , dT_mock_rg ,dims_rg,kperpbins,kparabins,corrtype='HIauto',w1=w_HI_rg,w2=w_HI_rg,W1=W_HI_rg,W2=W_HI_rg)
                Pk_dm_nosub,k,nmodes = power.Pk2D(dT_clean_mock_rg , dT_mock_rg ,dims_rg,kperpbins,kparabins,corrtype='HIauto',w1=w_HI_rg,w2=w_HI_rg,W1=W_HI_rg,W2=W_HI_rg)
                Pk_mm,k,nmodes = power.Pk2D(dT_mock_rg , dT_mock_rg ,dims_rg,kperpbins,kparabins,corrtype='HIauto',w1=w_HI_rg,w2=w_HI_rg,W1=W_HI_rg,W2=W_HI_rg)
        if corrtype=='Cross':
            if TF2D==False:
                Pk_dm,k,nmodes = power.Pk(dT_clean_mock_rg-dT_clean_data_rg , n_g_rg , dims_rg,kbins,corrtype='Cross',w1=w_HI_rg,w2=w_g_rg,W1=W_HI_rg,W2=W_g_rg,kcuts=kcuts)
                Pk_dm_nosub,k,nmodes = power.Pk(dT_clean_mock_rg , n_g_rg , dims_rg,kbins,corrtype='Cross',w1=w_HI_rg,w2=w_g_rg,W1=W_HI_rg,W2=W_g_rg,kcuts=kcuts)
                Pk_mm,k,nmodes = power.Pk(dT_mock_rg , n_g_rg , dims_rg,kbins,corrtype='Cross',w1=w_HI_rg,w2=w_g_rg,W1=W_HI_rg,W2=W_g_rg,kcuts=kcuts)
            if TF2D==True:
                Pk_dm,k,nmodes = power.Pk2D(dT_clean_mock_rg-dT_clean_data_rg , n_g_rg ,dims_rg,kperpbins,kparabins,corrtype='Cross',w1=w_HI_rg,w2=w_g_rg,W1=W_HI_rg,W2=W_g_rg)
                Pk_dm_nosub,k,nmodes = power.Pk2D(dT_clean_mock_rg , n_g_rg ,dims_rg,kperpbins,kparabins,corrtype='Cross',w1=w_HI_rg,w2=w_g_rg,W1=W_HI_rg,W2=W_g_rg)
                Pk_mm,k,nmodes = power.Pk2D(dT_mock_rg , n_g_rg ,dims_rg,kperpbins,kparabins,corrtype='Cross',w1=w_HI_rg,w2=w_g_rg,W1=W_HI_rg,W2=W_g_rg)

        # Append results and save in each loop so can access intermediate saved TF whilst looping over mocks:
        T.append( Pk_dm / Pk_mm )
        T_nosub.append( Pk_dm_nosub / Pk_mm )
        if TFfile is not None:
            #import warnings # Use for ignoring jagged array warning in pickle save
            #warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
            np.save(TFfile, [T,T_nosub] )
            np.save(TFfile+'_karray', k )
    return T,T_nosub,k

def TransferFunction_multimap(mapfilepaths,Nmock,N_fg,corrtype,kbins,k,TFfile,ra,dec,nu,wproj,dims0_rg,Np,window,compensate,interlace,W_HI,mockfilepath_HI,mockfilepath_g=None,w_HI_rg=None,W_HI_rg=None,w_g_rg=None,W_g_rg=None,N_fg2nd=0,taper_HI=1,taper_g=1,LoadTF=False,TF2D=False,kperpbins=None,kparabins=None):
    '''Transfer function calculation for cleaning of individual maps (scan or dish)
    and adding back together'''
    # if corrtype==Cross: must specify galaxy mock path and input w_g/W_g maps.
    # mapfilepaths: list of strings to numpy files where pre-divided maps are saved
    # N_fg2nd: option for second PCA clean once all maps are recombined
    nx,ny = np.shape(ra)
    nz = len(nu)
    dims_rg = dims0_rg[:6]
    ### Load pre-saved data if requested:
    if LoadTF==True:
        if TF2D==False:
            T,T_nosub,k_TF = np.load(TFfile+'.npy',allow_pickle=True)
            if k is not None:  # Check TF matches k-bins using - only possible if k is provided
                if len(k)!=len(k_TF):
                    print('\n Error: Loaded transfer function contains different k-bins\n'); exit()
                if np.allclose(k,k_TF)==False:
                    print('\n Error: Loaded transfer function contains different k-bins\n'); exit()
            return T,T_nosub,k_TF
        if TF2D==True:
            T2d,k_TF = np.load(TFfile+'.npy',allow_pickle=True)
            return T2d,k_TF
    ### If no pre-saved TF, run calculation:
    MKmap_weightedave = np.zeros((nx,ny,nz))
    counts_tot = np.zeros((nx,ny,nz))
    for m in range(len(mapfilepaths)):
        MKmap,W_HI,counts_HI = np.load(mapfilepaths[m])
        MKmap_clean = PCAclean(MKmap,N_fg=N_fg,W=W_HI,w=None)
        MKmap_weightedave += MKmap_clean*counts_HI
        counts_tot += counts_HI
    MKmap_weightedave[counts_tot!=0] /= counts_tot[counts_tot!=0]
    dT_clean_data = np.copy(MKmap_weightedave)
    if N_fg2nd!=0: # perform second foreground clean on recombined data:
        dT_clean_data = PCAclean(dT_clean_data,N_fg=N_fg2nd,W=W_HI,w=None)
    ra_p,dec_p,nu_p,pixvals = grid.SkyPixelParticles(ra,dec,nu,wproj,map=dT_clean_data,W=W_HI,Np=Np)
    xp,yp,zp = grid.SkyCoordtoCartesian(ra_p,dec_p,HItools.Freq2Red(nu_p),ramean_arr=ra,decmean_arr=dec,doTile=False)
    dT_clean_data_rg,W_fft,counts = grid.mesh(xp,yp,zp,pixvals,dims0_rg,window,compensate,interlace,verbose=False)
    dT_clean_data_rg = taper_HI*dT_clean_data_rg
    if TF2D==False:
        T = np.zeros((Nmock,len(kbins)-1))
        T_nosub = np.zeros((Nmock,len(kbins)-1))
    if TF2D==True:
        T = np.zeros((Nmock,len(kparabins)-1,len(kperpbins)-1))
    for i in range(Nmock):
        plot.ProgressBar(i,Nmock,header='\nConstructing transfer function...')
        # Read-in HI IM mock and mock galaxies:
        dT_mock = np.load(mockfilepath_HI + '_' + str(i) + '.npy')
        if corrtype=='Cross':
            ra_g,dec_g,z_g = np.load(mockfilepath_g + '_' + str(i) + '.npy')
            ### Grid mock galaxies:
            xp,yp,zp = grid.SkyCoordtoCartesian(ra_g,dec_g,z_g,ramean_arr=ra,decmean_arr=dec,doTile=False)
            n_g_rg,W_fft,counts = grid.mesh(xp,yp,zp,dims=dims0_rg,window=window,compensate=compensate,interlace=interlace,verbose=False)
            n_g_rg = taper_g*n_g_rg
        # Inject HI mock into each map, clean and regrid both cleaned and original mock:
        MKmap_weightedave = np.zeros((nx,ny,nz))
        counts_tot = np.zeros((nx,ny,nz))
        for m in range(len(mapfilepaths)):
            MKmap,W_HI,counts_HI = np.load(mapfilepaths[m])
            MKmap_clean = PCAclean(dT_mock + MKmap,N_fg=N_fg,W=W_HI,w=None)
            MKmap_weightedave += MKmap_clean*counts_HI
            counts_tot += counts_HI
        MKmap_weightedave[counts_tot!=0] /= counts_tot[counts_tot!=0]
        dT_clean_mock = np.copy(MKmap_weightedave)
        if N_fg2nd!=0: # perform second foreground clean on recombined data:
            dT_clean_mock = PCAclean(dT_clean_mock,N_fg=N_fg2nd,W=W_HI,w=None)
        ra_p,dec_p,nu_p,pixvals = grid.SkyPixelParticles(ra,dec,nu,wproj,map=dT_clean_mock,W=W_HI,Np=Np)
        xp,yp,zp = grid.SkyCoordtoCartesian(ra_p,dec_p,HItools.Freq2Red(nu_p),ramean_arr=ra,decmean_arr=dec,doTile=False)
        dT_clean_mock_rg,W_fft,counts = grid.mesh(xp,yp,zp,pixvals,dims0_rg,window,compensate,interlace,verbose=False)
        ra_p,dec_p,nu_p,pixvals = grid.SkyPixelParticles(ra,dec,nu,wproj,map=dT_mock,W=W_HI,Np=Np)
        xp,yp,zp = grid.SkyCoordtoCartesian(ra_p,dec_p,HItools.Freq2Red(nu_p),ramean_arr=ra,decmean_arr=dec,doTile=False)
        dT_mock_rg,W_fft,counts = grid.mesh(xp,yp,zp,pixvals,dims0_rg,window,compensate,interlace,verbose=False)
        dT_clean_mock_rg,dT_mock_rg = taper_HI*dT_clean_mock_rg,taper_HI*dT_mock_rg # multiply tapering window by HI map
        Pk_dm,k,nmodes = power.Pk(dT_clean_mock_rg-dT_clean_data_rg , dT_mock_rg , dims_rg,kbins,corrtype='HIauto',w1=w_HI_rg,w2=w_HI_rg,W1=W_HI_rg,W2=W_HI_rg)
        Pk_mm,k,nmodes = power.Pk(dT_mock_rg , dT_mock_rg , dims_rg,kbins,corrtype='HIauto',w1=w_HI_rg,w2=w_HI_rg,W1=W_HI_rg,W2=W_HI_rg)
        if corrtype=='HIauto':
            if TF2D==False:
                Pk_dm,k,nmodes = power.Pk(dT_clean_mock_rg-dT_clean_data_rg , dT_mock_rg , dims_rg,kbins,corrtype='HIauto',w1=w_HI_rg,w2=w_HI_rg,W1=W_HI_rg,W2=W_HI_rg)
                Pk_dm_nosub,k,nmodes = power.Pk(dT_clean_mock_rg , dT_mock_rg , dims_rg,kbins,corrtype='HIauto',w1=w_HI_rg,w2=w_HI_rg,W1=W_HI_rg,W2=W_HI_rg)
                Pk_mm,k,nmodes = power.Pk(dT_mock_rg , dT_mock_rg , dims_rg,kbins,corrtype='HIauto',w1=w_HI_rg,w2=w_HI_rg,W1=W_HI_rg,W2=W_HI_rg)
            if TF2D==True:
                Pk_dm,k2d,nmodes = power.Pk2D(dT_clean_mock_rg-dT_clean_data_rg , dT_mock_rg ,dims_rg,kperpbins,kparabins,corrtype='HIauto',w1=w_HI_rg,w2=w_HI_rg,W1=W_HI_rg,W2=W_HI_rg)
                Pk_mm,k2d,nmodes = power.Pk2D(dT_mock_rg , dT_mock_rg ,dims_rg,kperpbins,kparabins,corrtype='HIauto',w1=w_HI_rg,w2=w_HI_rg,W1=W_HI_rg,W2=W_HI_rg)
        if corrtype=='Cross':
            if TF2D==False:
                Pk_dm,k,nmodes = power.Pk(dT_clean_mock_rg-dT_clean_data_rg , n_g_rg , dims_rg,kbins,corrtype='Cross',w1=w_HI_rg,w2=w_g_rg,W1=W_HI_rg,W2=W_g_rg)
                Pk_dm_nosub,k,nmodes = power.Pk(dT_clean_mock_rg , n_g_rg , dims_rg,kbins,corrtype='Cross',w1=w_HI_rg,w2=w_g_rg,W1=W_HI_rg,W2=W_g_rg)
                Pk_mm,k,nmodes = power.Pk(dT_mock_rg , n_g_rg , dims_rg,kbins,corrtype='Cross',w1=w_HI_rg,w2=w_g_rg,W1=W_HI_rg,W2=W_g_rg)
            if TF2D==True:
                Pk_dm,k2d,nmodes = power.Pk2D(dT_clean_mock_rg-dT_clean_data_rg , n_g_rg ,dims_rg,kperpbins,kparabins,corrtype='Cross',w1=w_HI_rg,w2=w_g_rg,W1=W_HI_rg,W2=W_g_rg)
                Pk_mm,k2d,nmodes = power.Pk2D(dT_mock_rg , n_g_rg ,dims_rg,kperpbins,kparabins,corrtype='Cross',w1=w_HI_rg,w2=w_g_rg,W1=W_HI_rg,W2=W_g_rg)
        T[i] = Pk_dm / Pk_mm
        if TF2D==False:
            T_nosub[i] = Pk_dm_nosub / Pk_mm
    if TFfile is not None:
        if TF2D==False: np.save(TFfile,[T,T_nosub,k])
        if TF2D==True: np.save(TFfile,[T,k2d])
    if TF2D==True: return T,k2d
    else: return T,T_nosub,k

def TransferFunctionAuto_CrossDish(dT_obsA,dT_obsB,Nmock,TFfile,dims_orig,N_fg,corrtype='HIauto',Pmod=None,kbins=None,k=None,w1=None,W1=None,w2=None,W2=None,regrid=False,ndim=None,blackman=1,zeff=0,b_HI=1,f=0,Tbar=1,map_ra=None,map_dec=None,nu=None,LoadTF=False,TF2D=False,kperpbins=None,kparabins=None):
    '''OLD CODE????'''
    # Loop over Nmock number of mocks injected into real data to compute transfer function
    #np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) # Set to stop VisibleDeprecationWarning
    ### Load pre-saved data if requested:
    if LoadTF==True:
        if TF2D==False:
            T,T_nosubA,T_nosubB,k_TF = np.load(TFfile,allow_pickle=True)
            return T,T_nosubA,T_nosubB,k_TF
        if TF2D==True:
            T,k_TF = np.load(TFfile,allow_pickle=True)
            return T,k_TF
    ### If no pre-saved TF, run calculation:
    dT_clean_dataA = PCAclean(dT_obsA,N_fg,w=None,W=W1)
    dT_clean_dataB = PCAclean(dT_obsB,N_fg,w=None,W=W2)
    if regrid==True: # Regrid data from sky (ra,dec,freq)->(x,y,z) comoving
        dT_clean_dataA,dims,dims0 = grid.comoving(blackman*dT_clean_dataA,map_ra,map_dec,nu,W=W1,ndim=ndim)
        if w1 is not None: w1_rg,dims,dims0 = grid.comoving(blackman*w1,map_ra,map_dec,nu,W=W1,ndim=ndim) # use in PS measurement
        if W1 is not None: W1_rg,dims,dims0 = grid.comoving(blackman*W1,map_ra,map_dec,nu,W=W1,ndim=ndim) #   not FG clean
        dT_clean_dataB,dims,dims0 = grid.comoving(blackman*dT_clean_dataB,map_ra,map_dec,nu,W=W2,ndim=ndim)
        if w2 is not None: w2_rg,dims,dims0 = grid.comoving(blackman*w2,map_ra,map_dec,nu,W=W2,ndim=ndim) # use in PS measurement
        if W2 is not None: W2_rg,dims,dims0 = grid.comoving(blackman*W2,map_ra,map_dec,nu,W=W2,ndim=ndim) #   not FG clean
    else:
        if w1 is not None: w1_rg = blackman*w1
        if W1 is not None: W1_rg = blackman*W1
        if w2 is not None: w2_rg = blackman*w2
        if W2 is not None: W2_rg = blackman*W2
    if w1 is None: w1_rg = None
    if W1 is None: W1_rg = None
    if w2 is None: w2_rg = None
    if W2 is None: W2_rg = None

    T = np.zeros((Nmock,len(kbins)-1))
    if TF2D==False:
        T_nosubA = np.zeros((Nmock,len(kbins)-1))
        T_nosubB = np.zeros((Nmock,len(kbins)-1))
    if TF2D==True: T = np.zeros((Nmock,len(kparabins)-1,len(kperpbins)-1))
    for i in range(Nmock):
        plot.ProgressBar(i,Nmock,header='\nConstructing transfer function...')
        seed = np.random.randint(0,1e6) # Use to generate consistent HI IM and galaxies from same random seed
        dT_HImock = mock.Generate(Pmod,dims_orig,b=b_HI,f=f,Tbar=Tbar,doRSD=True,seed=seed,W=None)
        dT_HImock = telescope.smooth(dT_HImock,map_ra,map_dec,nu,D_dish=13.5)
        dT_HImockA = np.copy(dT_HImock)
        dT_HImockB = np.copy(dT_HImock)
        print(np.shape(dT_HImockA))
        print(np.shape(W1))

        dT_HImockA[W1==0] = 0 # Mock same empty pixels as astrofixed data
        dT_HImockB[W2==0] = 0 # Mock same empty pixels as astrofixed data
        dT_clean_mockA = PCAclean(dT_HImockA + dT_obsA,N_fg,w=None,W=W1)
        dT_clean_mockB = PCAclean(dT_HImockB + dT_obsB,N_fg,w=None,W=W2)
        if regrid==True: # Regrid cleaned maps from sky (ra,dec,freq)->(x,y,z) comoving
            dT_clean_mockA,dims,dims0 = grid.comoving(blackman*dT_clean_mockA,map_ra,map_dec,nu,W=W1,ndim=ndim)
            dT_HImockA,dims,dims0 = grid.comoving(blackman*dT_HImockA,map_ra,map_dec,nu,W=W1,ndim=ndim)
            dT_clean_mockB,dims,dims0 = grid.comoving(blackman*dT_clean_mockB,map_ra,map_dec,nu,W=W2,ndim=ndim)
            dT_HImockB,dims,dims0 = grid.comoving(blackman*dT_HImockB,map_ra,map_dec,nu,W=W2,ndim=ndim)
        else:
            dT_clean_mockA = blackman*dT_clean_mockA
            dT_HImockA = blackman*dT_HImockA
            dT_clean_mockB = blackman*dT_clean_mockB
            dT_HImockB = blackman*dT_HImockB
        if TF2D==False:
            Pk_dm,k,nmodes = power.Pk(dT_clean_mockA-dT_clean_dataA , dT_clean_mockB-dT_clean_dataB , dims,kbins,corrtype='HIauto',w1=w1_rg,w2=w2_rg,W1=W1_rg,W2=W2_rg)
            Pk_dm_nosubA,k,nmodes = power.Pk(dT_clean_mockA , dT_clean_mockB-dT_clean_dataB , dims,kbins,corrtype='HIauto',w1=w1_rg,w2=w2_rg,W1=W1_rg,W2=W2_rg)
            Pk_dm_nosubB,k,nmodes = power.Pk(dT_clean_mockA-dT_clean_dataA , dT_clean_mockB , dims,kbins,corrtype='HIauto',w1=w1_rg,w2=w2_rg,W1=W1_rg,W2=W2_rg)
            Pk_mm,k,nmodes = power.Pk(dT_HImockA , dT_HImockB , dims,kbins,corrtype='HIauto',w1=w1_rg,w2=w2_rg,W1=W1_rg,W2=W2_rg)
        if TF2D==True:
            Pk_dm,nmodes = power.Pk2D(dT_clean_mockA-dT_clean_dataA , dT_clean_mockB-dT_clean_dataB ,dims,kperpbins,kparabins,corrtype='HIauto',w1=w1_rg,w2=w2_rg,W1=W1_rg,W2=W2_rg)
            Pk_mm,nmodes = power.Pk2D(dT_HImockA , dT_HImockB ,dims,kperpbins,kparabins,corrtype='HIauto',w1=w1_rg,w2=w2_rg,W1=W1_rg,W2=W2_rg)
        T[i] = Pk_dm / Pk_mm
        if TF2D==False:
            T_nosubA[i] = Pk_dm_nosubA / Pk_mm
            T_nosubB[i] = Pk_dm_nosubB / Pk_mm
    if TFfile is not None:
        if TF2D==False: np.save(TFfile,[T,T_nosubA,T_nosubB,k])
        if TF2D==True: np.save(TFfile,[T,k])
    if TF2D==False: return T,T_nosubA,T_nosubB,k
    if TF2D==True: return T,k

def TransferFunctionOLD(dT_obs,Nmock,TFfile,dims,N_fg,Pmod=None,kbins=None,k=None,w_HI=None,W_HI=None,regrid=False,blackman=1,mockfilepath=None,zeff=0,b_HI=1,f=0,Tbar=1,ra=None,dec=None,nu=None,LoadTF=False):
    # Loop over Nmock number of mocks injected into real data to compute transfer function
    # Assumes each mock is saved as ""[mockfilepath]_i.npy" where i = {0,Nmock-1}
    # Set mockfilepath=False to generate lognormals on the fly
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) # Set to stop VisibleDeprecationWarning
    ### Load pre-saved data if requested:
    if LoadTF==True:
        T,k_TF = np.load(TFfile,allow_pickle=True)
        if k is not None:  # Check TF matches k-bins using - only possible if k is provided
            if len(k)!=len(k_TF):
                print('\n Error: Loaded transfer function contains different k-bins\n'); exit()
            if np.allclose(k,k_TF)==False:
                print('\n Error: Loaded transfer function contains different k-bins\n'); exit()
        return T,k_TF
    ### If no pre-saved TF, run calculation:
    dT_clean_data = PCAclean(dT_obs,N_fg,W=W_HI,freqmask=freqmask)
    if regrid==True: # Regrid data from sky (ra,dec,freq)->(x,y,z) comoving
        dT_clean_data,dims = grid.regrid(blackman*dT_clean_data,ra,dec,nu)
        w_HI_rg,dims = grid.regrid(blackman*w_HI,ra,dec,nu) # use in PS measurement
        W_HI_rg,dims = grid.regrid(blackman*W_HI,ra,dec,nu) #   not FG clean
    else:
        w_HI_rg = blackman*w_HI
        W_HI_rg = blackman*W_HI
    T = np.zeros((Nmock,len(kbins)-1))
    for i in range(Nmock):
        plot.ProgressBar(i,Nmock,header='\nConstructing transfer function...')
        if mockfilepath is None:
            dT_HImock = Tbar * mock.GetMock(Pmod,dims,b=b_HI,f=f,doRSD=True)
            dT_HImock[W_HI==0] = 0 # Mask pixels to match data
        else: dT_mock = np.load(mockfilepath + '_' + i + '.npy')
        dT_clean_mock = PCAclean(dT_HImock + dT_obs,N_fg,W=W_HI,freqmask=freqmask)
        if regrid==True: # Regrid cleaned maps from sky (ra,dec,freq)->(x,y,z) comoving
            dT_clean_mock,dims = grid.regrid(blackman*dT_clean_mock,ra,dec,nu)
            dT_HImock,dims = grid.regrid(blackman*dT_HImock,ra,dec,nu)
        else:
            dT_clean_mock = blackman*dT_clean_mock
            dT_HImock = blackman*dT_HImock
        Pk_dm,k,nmodes = power.Pk(dT_clean_mock-dT_clean_data , dT_HImock , dims,kbins,w1=w_HI,w2=w_HI,W1=W_HI,W2=W_HI)
        Pk_mm,k,nmodes = power.Pk(dT_HImock , dT_HImock , dims,kbins,w1=w_HI,w2=w_HI,W1=W_HI,W2=W_HI)
        T[i] = Pk_dm / Pk_mm
    if TFfile is not None: np.save(TFfile,[T,k])
    return T,k

def applyFG(map,dims,zeff,zmin,zmax,ra=None,dec=None,GSMfilepath='/Users/stevecunnington/Documents/IntensityKit/PyGSM/pygsm'):
    #print('\nBuilding Foreground maps ...')
    lx,ly,lz,nx,ny,nz = dims
    import sys
    sys.path.insert(1,GSMfilepath)
    from pygsm import GlobalSkyModel
    gsm = GlobalSkyModel()
    nside = 512
    vbins = np.linspace(HItools.Red2Freq(zmin),HItools.Red2Freq(zmax),nz+1)
    d_c = cosmo.D_com(zeff)  # Distance to central redshift
    if ra is None or dec is None:
        #Construct quasi-angular coordinates for FG map centred on Galactic centre (ra=dec=0):
        deltar = np.degrees(lx/d_c) #~R.A width in degrees
        deltad = np.degrees(ly/d_c) #~Dec width in degrees
        rmin,rmax,dmin,dmax = -deltar/2,deltar/2,-deltad/2,deltad/2
        rbins = np.linspace(rmin,rmax,nx+1)
        dbins = np.linspace(dmin,dmax,ny+1)
        rbincentres = rbins+(rbins[1]-rbins[0])/2
        rbincentres = rbincentres[:len(rbincentres)-1] #remove last value since this is outside of bins
        dbincentres = dbins+(dbins[1]-dbins[0])/2
        dbincentres = dbincentres[:len(dbincentres)-1] #remove last value since this is outside of bins
    else:
        rmin,rmax,dmin,dmax = np.min(ra),np.max(ra),np.min(dec),np.max(dec)
        rbins, dbins = np.linspace(rmin,rmax,nx+1), np.linspace(dmin,dmax,ny+1)
        rbincentres = (rbins+(rbins[1]-rbins[0])/2)
        rbincentres = rbincentres[:len(rbincentres)-1]
        dbincentres = dbins+(dbins[1]-dbins[0])/2
        dbincentres = dbincentres[:len(dbincentres)-1]
    npix = hp.nside2npix(nside)
    FGmaps = np.zeros((nz,npix))
    for i in range(nz):
        plot.ProgressBar(i,nz,header='\nBuilding foreground maps ...')
        vmin = vbins[i+1]
        vmax = vbins[i]
        vc = vmin + (vmax - vmin)/2
        FGmaps[i] = gsm.generate(vc)*1e3 #mK
    if ra is not None or dec is not None: FGmaps = change_coord(FGmaps, ['G', 'C']) # converting from galactic to equatorial coordinates
    return map + np.flip( HealpixtoCube(FGmaps,rbincentres,dbincentres) , 0 ) # flip the R.A coords for conventional consistency

def HealpixtoCube(hpmap,rbincentres,dbincentres):
    '''
    Convert healpy map array into data cube for use on flat skies.
    Healpy map to be input in [Nz,Npix]
    Returns data cube in [Nx,Ny,Nz] format
    '''
    nside = hp.get_nside(hpmap)
    x = rbincentres
    y = dbincentres[:,np.newaxis]
    pixindex = hp.ang2pix(nside, np.pi/2-np.radians(y), np.radians(x))
    pixindex = np.swapaxes(pixindex,0,1)
    datacube=[]
    for z in range(np.shape(hpmap)[0]):
        datacube.append( hpmap[z][pixindex] )
    datacube = np.array(datacube)
    datacube = np.swapaxes(datacube,0,1)
    datacube = np.swapaxes(datacube,1,2)
    return datacube

def change_coord(m, coord):
    '''
    ### Code from Paula Soares ###
    Rotates coordinate system of healpy map.
    m: map or array of maps to be rotated.
    coord: current coordinate system and new coordinate system, e.g. coord=['G', 'C'] rotates
    from galactic to equatorial coordinates. Ecliptic coordinates ('E') also allowed.
    '''
    # Basic HEALPix parameters
    npix = m.shape[-1]
    nside = hp.npix2nside(npix)
    ang = hp.pix2ang(nside, np.arange(npix))
    # Select the coordinate transformation
    rot = hp.Rotator(coord=reversed(coord))
    # Convert the coordinates
    new_ang = rot(*ang)
    new_pix = hp.ang2pix(nside, *new_ang)
    return m[..., new_pix]

def FGPeturbations(dT_MK,W,nu):
    '''Use real data to generate perturbations in the simulated FG spectra so they
    are no longer perfectly smooth continuum signals
    '''
    nx,ny,nz = np.shape(dT_MK)
    dT_MK[W==0] = 0 # Set back to zero for more poly fitting
    perturbs = np.ones((nx,ny,nz))
    for i in range(nx):
        for j in range(ny):
            if W[i,j,0]==0: continue
            poly = model.FitPolynomial(nu,dT_MK[i,j,:],n=2)
            perturbs[i,j,:] = dT_MK[i,j,:]/poly
    return perturbs
