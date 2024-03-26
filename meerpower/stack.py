import numpy as np
import HItools
import plot
import matplotlib.pyplot as plt

def spectral(map,W,ra_g,dec_g,z_g,ra,dec,nu,nnu_stack,nnu_bins):
    Ngal = len(ra_g)
    dnu = np.mean(np.diff(nu))
    nu_stack_bins = np.linspace(-nnu_stack*dnu,nnu_stack*dnu,nnu_bins)
    nu_stack_bins_cent = (nu_stack_bins[1:] + nu_stack_bins[:-1])/2
    d = np.zeros(len(nu_stack_bins_cent))
    for i in range(Ngal):
        ra_stack,dec_stack,nu_stack = (ra-ra_g[i]),(dec-dec_g[i]),(nu-HItools.Red2Freq(z_g[i]))
        ibin = np.digitize(nu_stack,nu_stack_bins)
        dang = np.sqrt(ra_stack**2 + dec_stack**2) # angular distance from galaxy position
        map_nu = map[dang==np.min(np.abs(dang))][0] # map values along pixel closest to galaxy position
        for di in range(len(d)):
            d[di] += np.sum(map_nu[ibin==di+1])
    return d/Ngal,nu_stack_bins_cent

def angular(map,W,ra_g,dec_g,z_g,ra,dec,nu,degmax,ddegpix,w=None,dnu=2,doPhotoz=False,zphotolims=None):
    #dnu: width of channels to stack at angular position of galaxy [MHz]
    Ngal = len(ra_g)
    ddegbins = np.arange(ddegpix/2,degmax,ddegpix)
    ddegbins = -np.flip(ddegbins)
    ddegbins = np.append(ddegbins,np.abs(np.flip(ddegbins)))
    ddeg = np.zeros((len(ddegbins)-1,len(ddegbins)-1))
    counts = np.zeros((len(ddegbins)-1,len(ddegbins)-1))

    if w is None: w = np.ones(np.shape(map))
    if doPhotoz==True: Z_B_MAX,Z_B_MIN = zphotolims

    W_nu = np.mean(W,2)
    W_nu = np.ones(np.shape(W_nu))
    for i in range(Ngal):
        ra_stack,dec_stack,nu_stack = (ra[W_nu!=0]-ra_g[i]),(dec[W_nu!=0]-dec_g[i]),(nu-HItools.Red2Freq(z_g[i]))

        if doPhotoz==False:
            map_nu = np.mean( map[:,:,np.abs(nu_stack)<dnu/2] , 2)
            w_nu = np.mean( w[:,:,np.abs(nu_stack)<dnu/2] , 2)

        if doPhotoz==True:
            plot.ProgressBar(i,len(ra_g),header='Stacking photo-z galaxies')
            # Z_B_MIN/Z_B_MAX represent Lower/Upper bound of the 68% confidence interval of Z_B.
            #  therefore assuming Gaussian profile, sigma = (Z_B_MAX-Z_B_MIN) / 2
            nu_B_min,nu_B_max = HItools.Red2Freq(Z_B_MAX[i]),HItools.Red2Freq(Z_B_MIN[i])
            sig = (nu_B_max - nu_B_min) / 2
            # Create Gaussian weighting centred at Z_B, with width sigma as above and apply to map:
            w_pz_nu = 1/(sig*np.sqrt(2*np.pi)) * np.exp(-(nu_stack**2/(2*sig**2)))
            '''
            w_pz_nu /= np.sum(w_pz_nu)
            print(np.sum(w_pz_nu))
            plt.plot(nu_stack,w_pz_nu)
            plt.show()
            exit()
            '''
            map_nu = np.copy(map)
            map_nu = np.sum(w_pz_nu*map_nu,2)


        counts += np.histogram2d(np.ravel(ra_stack),np.ravel(dec_stack),bins=ddegbins,weights=np.ravel(w_nu[W_nu!=0]))[0]
        ddeg += np.histogram2d(np.ravel(ra_stack),np.ravel(dec_stack),bins=ddegbins,weights=np.ravel(map_nu[W_nu!=0]*w_nu[W_nu!=0]))[0]

    ddeg[counts!=0] = ddeg[counts!=0]/counts[counts!=0]

    return np.swapaxes(ddeg,0,1),np.swapaxes(counts,0,1)
