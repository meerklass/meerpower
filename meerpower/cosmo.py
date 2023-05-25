import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
from scipy.interpolate import interp1d
c_km = 299792.458 #m/s

def SetCosmology(builtincosmo='Planck18',z=0,UseCamb=True):
    global UseCamb_global; UseCamb_global = UseCamb
    ### Use below if using camb
    if UseCamb==True:
        import camb
        from camb import model, initialpower
        global H_0
        global h
        global D_z # Growth function D(z)
        global Om0
        global Ob0
        global n_s
        global A_s
        global delta_c
        if builtincosmo=='WMAP1':
            H_0 = 73 # [km/(Mpc s)]
            h = H_0/100
            Om0 = 0.25
            Ob0 = 0.045 # Omega_b
            n_s = 0.99
        if builtincosmo=='Planck15':
            H_0 = 67.7 # [km/(Mpc s)]
            h = H_0/100
            Om0 = 0.307
            Ob0 = 0.0486 # Omega_b
            n_s = 0.968
        if builtincosmo=='Planck18':
            H_0 = 67.4 # [km/(Mpc s)]
            h = H_0/100
            Om0 = 0.315 # Omega_m
            Ob0 = 0.0489 # Omega_b
            n_s = 0.965
        A_s = 2.14e-9 # Scalar amplitude
        D_z = D(z) # Growth Function (normalised to unity for z=0)
        delta_c = 1.686
        GetModelPk(z,1e-4,1e0,NonLinear=False) # Use to set global transfer function T
    else:
        global cosmo
        global cosmoNBK
        from nbodykit.lab import cosmology
        if builtincosmo=='Planck15':
            from astropy.cosmology import Planck15 as cosmo
            cosmoNBK = cosmology.Planck15

def f(z):
    if UseCamb_global==True:
        gamma = 0.545
        return Omega_M(z)**gamma
    else:
        return cosmoNBK.scale_independent_growth_rate(z) # NBK default growth rate for given fiducial cosmo

def E(z):
    return np.sqrt( 1 - Om0 + Om0*(1+z)**3 )

def H(z):
    if UseCamb_global==True: return E(z)*H_0
    else: return cosmo.H(z).value

def Omega_M(z):
    return H_0**2*Om0*(1+z)**3 / H(z)**2

def Omega_b(z=0):
    if z!=0: print('\nError: Cosmotools needs evoloution for Omega_b(z)!')
    return Ob0

def D_com(z):
    if UseCamb_global==True:
        #Comoving distance [Mpc/h]
        func = lambda z: (c_km/H_0)/E(z)
        h = H_0/100
        return scipy.integrate.romberg(func,0,z) * h
    else:
        h = cosmo.H(0).value/100 # use to convert astopy Mpc distances to Mpc/h
        return cosmo.comoving_distance(z).value*h

def Deltab(z,k,f_NL,b_HI):
    #Scale dependent modification of the Gaussian clustering bias
    # Eq 13 of https://arxiv.org/pdf/1507.03550.pdf
    T_k = np.ones(np.shape(k)) # Use to avoid evaluating T(k) outside interpolation
                               #   range. Uses convention T(k->0)=1
    T_k[k>kmin_interp] = T(k[k>kmin_interp]) # Evalutate transfer function for all k>0
    k[k==0] = 1e-30 # Avoid divive by zero error in final return statement
    return 3*f_NL*( (b_HI-1)*Om0*H_0**2*delta_c ) / (c_km**2*k**2*T_k*D_z)

def D(z):
    #Growth parameter - obtained from eq90 in:
    #   https://www.astro.rug.nl/~weygaert/tim1publication/lss2009/lss2009.linperturb.pdf
    integrand = lambda zi: (1+zi)/(H(zi)**3)
    D_0 = 5/2 * Om0 * H_0**2 * H(0) * integrate.quad(integrand, 0, 1e3)[0]
    D_z = 5/2 * Om0 * H_0**2 * H(z) * integrate.quad(integrand, z, 1e3)[0]
    return D_z / D_0 # Normalise such that D(z=0) = 1

'''
def T(k):
    #Transfer function - from Amendola DE textbook eq 4.207:
    k_eq = 2e-2
    x = k / k_eq
    return np.log(1+0.171*x)/(0.171*x) * ( 1 + 0.284*x + (1.18*x)**2 + (0.399*x)**3 + (0.49*x)**4 )**-0.25
'''

def GetModelPk(z,kmin=1e-3,kmax=10,NonLinear=False):
    '''
    Use pycamb to generate model power spectrum at redshift z
    '''
    import camb
    from camb import model, initialpower
    # Declare minium k value for avoiding interpolating outside this value
    global kmin_interp
    kmin_interp = kmin
    Oc0 = Om0 - Ob0 # Omega_c
    #Set up the fiducial cosmology
    pars = camb.CAMBparams()
    #Set cosmology
    pars.set_cosmology(H0=H_0,ombh2=Ob0*h**2,omch2=Oc0*h**2,omk=0,mnu=0)
    pars.set_dark_energy() #LCDM (default)
    pars.InitPower.set_params(ns=n_s, r=0, As=A_s)
    pars.set_for_lmax(2500, lens_potential_accuracy=0);
    #Calculate results for these parameters
    results = camb.get_results(pars)
    #Get matter power spectrum at some redshift
    pars.set_matter_power(redshifts=[z], kmax=kmax)
    if NonLinear==False: pars.NonLinear = model.NonLinear_none
    if NonLinear==True: pars.NonLinear = model.NonLinear_both # Uses HaloFit
    results.calc_power_spectra(pars)
    k, z, pk = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints = 200)
    # Define global transfer function to be called in other functions:
    trans = results.get_matter_transfer_data()
    k_trans = trans.transfer_data[0,:,0] #get kh - the values of k/h at which transfer function is calculated
    transfer_func = trans.transfer_data[model.Transfer_cdm-1,:,0]
    transfer_func = transfer_func/np.max(transfer_func)
    global T
    T = interp1d(k_trans, transfer_func) # Transfer function - set to global variable
    return interp1d(k, pk[0])
