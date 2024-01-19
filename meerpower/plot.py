import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import NullFormatter
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import mpl_style
plt.style.use(mpl_style.style1)
import copy
import sys

def ProgressBar(i,N,header=None):
    ### progress bar printing for looping process, for i'th element of N total
    if i==0:
        if header is None: print('\nPerforming loop:')
        else: print('\n'+header)
    barlength = 30 # text spaces in command line
    sys.stdout.write('\r')
    ratio = (i + 1) / N
    sys.stdout.write("[%-30s] %d%%" % ('='*int(barlength*ratio), 100*ratio))
    sys.stdout.flush()
    if i==(N-1): print('\n')

def Map(map,W=None,ra=None,dec=None,map_ra=None,map_dec=None,wproj=None,title=None,Gal=False,cbar_label=None,cbarshrink=1,ZeroCentre=False,vmin=None,vmax=None,cmap='magma'):
    plt.figure()
    if map_ra is not None:
        plt.subplot(projection=wproj)
        ax = plt.gca()
        lon = ax.coords[0]
        lat = ax.coords[1]
        lon.set_major_formatter('d')
        #lon.set_ticklabel_visible(False)
        #lat.set_ticklabel_visible(False)
        #lon.set_ticklabel(size=fontsize)
        #lat.set_ticklabel(size=fontsize)
        lon.set_ticks_position('b')
        lat.set_ticks_position('l')
        plt.grid(True, color='grey', ls='solid',lw=0.5)
    if len(np.shape(map))==3:
        map = np.mean(map,2) # Average along 3rd dimention (LoS) as default if 3D map given
        if W is not None: W = np.mean(W,2)
    if vmax is not None: map[map>vmax] = vmax
    if vmin is not None: map[map<vmin] = vmin
    if ZeroCentre==True:
        divnorm = colors.TwoSlopeNorm(vmin=np.min(map), vcenter=0, vmax=np.max(map))
        cmap = copy.copy(matplotlib.cm.get_cmap("seismic"))
        cmap.set_bad(color='grey')
    else: divnorm = None
    if W is not None: map[W==0] = np.nan
    if ra is not None: # Check ascending direction of R.A coordinates:
        if ra[0]>ra[1]: # R.A coordinates descend left to right therefore switch:
            ra = np.flip(ra) # Flip ra so lowest ra is first
            map = np.flip(map,0) # Flip map so lowest ra pixel appears left
    if map_ra is not None:
        data = plt.imshow(map.T,cmap=cmap,norm=divnorm)
        plt.xlabel('R.A [deg]',fontsize=18)
        plt.ylabel('Dec. [deg]',fontsize=18)
    else:
        if ra is None or dec is None: plt.imshow(map.T,cmap=cmap,norm=divnorm)
        else:
            data = plt.imshow(map.T,cmap=cmap,norm=divnorm,extent=[np.min(ra),np.max(ra),np.min(dec),np.max(dec)])
            plt.xlabel('R.A [deg]',fontsize=18)
            plt.ylabel('Dec. [deg]',fontsize=18)
    if vmax is not None or vmin is not None: plt.clim(vmin,vmax)
    cbar = plt.colorbar(orientation='horizontal',shrink=cbarshrink)
    if cbar_label is None:
        if Gal==False: cbar.set_label('mK')
    else: cbar.set_label(cbar_label)
    plt.title(title,fontsize=18)

def LoSspectra(map,W,zaxis=None,mapUnits='mK',xlabel=None,ylabel=None,lw=0.01,title=None):
    ### plot all amplitudes along the LoS (freq or redshift direction) for every pixel in given 3D map
    # Assumes input map is in form [x,y,z] or [RA,Dec,z] where z is LoS dimension
    # zaxis: specify zaxis values to include in plot e.g. zaxis = nu (where nu is frequency values)
    plt.figure()
    map_nan = np.copy(map)
    map_nan[W==0] = np.nan
    nx,ny,nz = np.shape(map)
    plt.figure()
    for i in range(nx):
        for j in range(ny):
            if zaxis is None: plt.plot(map_nan[i,j,:],lw=lw,color='black')
            else: plt.plot(zaxis,map_nan[i,j,:],lw=lw,color='black')
    del map_nan
    if zaxis is None: plt.xlim(left=0,right=nz)
    else:
        plt.xlim(left=np.min(zaxis),right=np.max(zaxis))
    if xlabel is None: plt.xlabel('Channel')
    else: plt.xlabel(xlabel)
    if ylabel is None: plt.ylabel('Map amplitude ['+mapUnits+']')
    else: plt.ylabel(ylabel)
    if title is not None: plt.title(title,fontsize=18)

def FrequencyCovariance(C,nu,title=None):
    plt.figure()
    plt.imshow(C,extent=[nu[0],nu[-1],nu[0],nu[-1]])
    plt.colorbar(label=r'mK$^2$')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Frequency [MHz]')
    if title is not None: plt.title(title,fontsize=18)
    else: plt.title('Frequency covariance')

def EigenSpectrum(eignumb,eigenval,title=None):
    plt.figure()
    ### Show eigenvalue spectrum from outputs of PCA clean
    eignumb_cut = 40 # highest eigenvalue to show
    plt.plot(eignumb,eigenval,'-o')
    plt.yscale('log')
    plt.ylim(bottom=eigenval[eignumb_cut])
    plt.xlim(left=0,right=eignumb_cut)
    plt.xlabel('Eigennumber')
    plt.ylabel('Eigenvalue')
    if title is not None: plt.title(title,fontsize=18)

def Eigenmodes(x,V,Num=6,title=None):
    # Num: number of eigenmodes selected to plot
    chart = 100*Num + 11
    plt.figure(figsize=(7,3*Num))
    for i in range(Num):
        plt.subplot(chart + i)
        plt.plot(x,V[:,i],label='eigenmode %s'%(i+1))
        plt.legend(fontsize=16)
    if title is not None: plt.title(title,fontsize=18)

def ProjectedEigenmodeMaps(map,W,V,ra,dec,wproj,Num=6):
    # Num: number of eigenmodes selected to project
    eigenvecs = np.arange(Num)
    nx,ny,nz = np.shape(map)
    M = np.reshape(map,(nx*ny,nz))
    M = np.swapaxes(M,0,1) # [Npix,Nz]->[Nz,Npix]
    for i in range(len(eigenvecs)):
        A = np.swapaxes([V[:,eigenvecs[i]]],0,1) # Mixing matrix
        S = np.dot(A.T,M)
        sourcemap = np.dot(A,S)
        sourcemap = np.swapaxes(sourcemap,0,1) #[Nz,Npix]->[Npix,Nz]
        sourcemap = np.reshape(sourcemap,(nx,ny,nz))
        Map(sourcemap,map_ra=ra,map_dec=dec,wproj=wproj,W=W,title=r'Projected map for Eigenvector %s'%(eigenvecs[i]+1))

def PlotPk(k,Pk,sig_err=None,Pkmod=None,datalabels=None,modellabel='Model',figsize=(8,8),legendfontsize=18,xlabel=None,ylabel=None,ylabel_unit=None,plottitle=None,norm=1,fill_between=False,ModelComp=False,DetectSig=False):

    #k,Pk: data to plot - input as array for multiple Pks e.g. [Pk1,Pk2]. If same k being used for all inputs, only pass it once e.g. k=k not k=[k,k]
    # datalabels: string to display in legend
    # modellabel: string to display model label in legend
    # legendfontsize: fontsize in legend
    # x[y]label: string to display as axes labels
    # ylabel_unit: Additional units to add into power ylabel e.g. r'${\rm mK}^2$' or r'${\rm mK}$' for cross
    # plottitle: string for plot title
    # norm: pre-factor to multiply Pk by. Options 1 (for standard loglog Pk) or 'ksq' (for k**2*Pk)
    # ModelComp: set True to add subpanel at bottom of plot showing model comparison
    # DetctSig: set True to add subpanel at bottom of plot showing null diagnostic

    ### Check for errors in inputs:
    if ModelComp==True and Pkmod is None:
        print('\nError: No model supplied for model comparison')
        exit()
    if DetectSig==True and sig_err is None:
        print('\nError: No error supplied for null diagnostic')
        exit()
    if ModelComp==True and DetectSig==True:
        print('\nError: Can not show model comparison and detection diagnostic in same subplot region. Chose only one.')
        exit()

    fontsize = 18
    markers = ['o','s','v','P','X'] # only currently allows 5 different Pk entries

    ### Prepare for multiple Pk entries plot overlaid:
    if len(np.shape(Pk))>1:
        Pkentries = np.shape(Pk)[0]
        if len(np.shape(k))==1: # Repeat k if same k are being used for all Pk inputs:
            k = np.tile(k, (Pkentries,1))
    else: # Put data into 2D array so looping code over 1 entry works
        Pkentries = 1
        k,Pk = [k],[Pk]
        if sig_err is not None: sig_err = [sig_err]
    if datalabels is None: datalabels = np.repeat(None,Pkentries)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    if Pkentries>1: # offset k values to avoid overlap
        offset = 0.02 # Proportional amount to shift k-values by
        for i in range(Pkentries-1):
            if norm==1: k[i+1] += np.log10(10**(offset*k[i+1]))
            else: k[i+1] += offset*k[i+1]

    ### Define axes labels:
    if xlabel is None: xlabel = r'$k\,[h/{\rm Mpc}]$'
    if ylabel is None:
        if norm==1:
            ylabel = r'$P(k)$'
            if ylabel_unit is None: ylabel = ylabel + r'$\,[({\rm Mpc}/h)^3]$'
            else: ylabel = ylabel + r'$\,[$' + ylabel_unit + r'$ ({\rm Mpc}/h)^3]$'
        if norm!=1:
            ylabel = r'$k^2\,P(k)$'
            if ylabel_unit is None: ylabel = ylabel + r'$\,[{\rm Mpc}/h]$'
            else: ylabel = ylabel + r'$\,[$' + ylabel_unit + r'$ {\rm Mpc}/h]$'

    ### Run Plot:
    #if figsize is None: fig = plt.figure(figsize=(8,8))
    #else: fig = plt.figure(figsize=figsize)
    fig = plt.figure(figsize=figsize)

    if ModelComp==True or DetectSig==True:
        gs = GridSpec(3,1) # 3 rows, 1 columns
        ax1 = fig.add_subplot(gs[0:2,0]) # First row, first column
    else: ax1 = plt.gca()

    # Plot model (currently assumed model is formed in first array of k):
    if Pkmod is not None:
        if norm==1: normfactor = 1
        if norm=='ksq': normfactor = k[0]**2
        ax1.plot(k[0],normfactor*Pkmod,ls='--',color='black',zorder=-1,label=modellabel)

    for i in range(Pkentries):
        if norm==1: normfactor = 1
        if norm=='ksq': normfactor = k[i]**2
        if sig_err is not None:
            if norm==1: ax1.errorbar(k[i],normfactor*np.abs(Pk[i]),normfactor*sig_err[i],color=colors[i],marker=markers[i],ls='none',zorder=0,label=datalabels[i])
            if norm=='ksq': ax1.errorbar(k[i],normfactor*Pk[i],normfactor*sig_err[i],color=colors[i],marker=markers[i],ls='none',zorder=0,label=datalabels[i])
            if fill_between==True: ax1.fill_between(k[i], normfactor*Pk[i]-normfactor*sig_err[i], normfactor*Pk[i]+normfactor*sig_err[i],color=colors[i],alpha=0.3)

        if norm==1:
            ax1.scatter(k[i][Pk[i]>0],normfactor*np.abs(Pk[i][Pk[i]>0]),color=colors[i],marker=markers[i],s=50)
            ax1.scatter(k[i][Pk[i]<0],normfactor*np.abs(Pk[i][Pk[i]<0]),color=colors[i],marker=markers[i],s=50,facecolors='white',zorder=1)
        if norm=='ksq': ax1.scatter(k[i],normfactor*Pk[i],color=colors[i],marker=markers[i],s=50)

    if ModelComp==False and DetectSig==False: ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(plottitle,fontsize=16)
    if norm==1: ax1.loglog()
    if norm=='ksq': ax1.set_xscale('log')
    #ax1.xaxis.set_major_formatter(NullFormatter())
    #ax1.xaxis.set_minor_formatter(NullFormatter())
    if datalabels[0] is not None: ax1.legend(fontsize=legendfontsize)

    if norm=='ksq': ax1.axhline(0,color='black',lw=1)

    if ModelComp==True or DetectSig==True:
        ax1.xaxis.set_major_formatter(NullFormatter())
        ax1.xaxis.set_minor_formatter(NullFormatter())
        ax2 = fig.add_subplot(gs[2,0]) # Last row, first column
        if ModelComp==True:
            ax2.axhline(1,color='black',ls='--')
            if len(np.ravel(Pk)[np.ravel(Pk)<=0])>0: ax2.axhline(0,color='grey',lw=1) # Show zero horizontal line if model some negative data points exist
            for i in range(Pkentries):
                ax2.errorbar(k[i],Pk[i]/Pkmod,sig_err[i]/Pkmod,color=colors[i],marker=markers[i],ls='none',zorder=0,label=datalabels[i])
                ax2.scatter(k[i],Pk[i]/Pkmod,color=colors[i],marker=markers[i],s=50)
            ax2.set_ylabel(r'$P(k)/P^{\rm mod}(k)$')
        if DetectSig==True:
            ax2.axhline(0,color='black')
            for i in range(Pkentries):
                ax2.scatter(k[i],Pk[i]/sig_err[i],color=colors[i],marker=markers[i],s=50)
            ax2.set_ylabel(r'$P(k)/\sigma_{P(k)}$')
        ax2.set_xlabel(xlabel)
        ax2.set_xscale('log')
        #ax2.xaxis.set_major_formatter(ScalarFormatter())
        #ax2.xaxis.set_minor_formatter(ScalarFormatter())
        ax2.tick_params(labelsize=fontsize,which="both")

        if plottitle==None: top=0.99
        else: top = 0.94
        plt.subplots_adjust(top=top,
        bottom=0.106,
        left=0.176,
        right=0.99,
        hspace=0.06,
        wspace=0.2)

def CovarianceMatrix(C,kbins,title=None):
    cmap = 'RdBu'
    #cmap = 'viridis'

    plt.imshow(C,vmin=-1,vmax=1, cmap=cmap,extent=[np.min(kbins),np.max(kbins),np.min(kbins),np.max(kbins)])
    #plt.imshow(C, cmap=cmap,extent=[np.min(kbins),np.max(kbins),np.min(kbins),np.max(kbins)])
    cbar = plt.colorbar()
    plt.xlabel(r'$k\,[h/{\rm Mpc}]$')
    plt.ylabel(r'$k\,[h/{\rm Mpc}]$')
    if title is not None: plt.title(title,fontsize=18)
    plt.figure()
