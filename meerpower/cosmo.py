import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
from scipy.interpolate import interp1d
c_km = 299792.458 #m/s

def SetCosmology(builtincosmo='Planck18',z=0,UseAstropy=True,UseCAMB=True,UseNBK=False,UseCLASS=False):
    # UseAstropy: Set True to use astropy (where possible) for parameters and distances
    #             Set False to instead use manual calculations of parameters and distances
    # UseNBK: Use Nbodykit for power spectrum and cosmology calculations - not compatible with CAMB
    global astropy; astropy = UseAstropy
    if UseNBK==True:
        UseCAMB=False
        from nbodykit.lab import cosmology
    if UseCLASS==True:
        UseCAMB=False
        import classylss
    if UseCAMB==True:
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
        if astropy==False: H_0 = 73 # [km/(Mpc s)]
        Om0 = 0.25
        Ob0 = 0.045 # Omega_b
        n_s = 0.99
    if builtincosmo=='Planck15':
        if astropy==False: H_0 = 67.7 # [km/(Mpc s)]
        Om0 = 0.307
        Ob0 = 0.0486 # Omega_b
        n_s = 0.968
    if builtincosmo=='Planck18':
        if astropy==False: H_0 = 67.4 # [km/(Mpc s)]
        Om0 = 0.315 # Omega_m
        Ob0 = 0.0489 # Omega_b
        n_s = 0.965
    if astropy==True:
        global cosmo
        if builtincosmo=='Planck15': from astropy.cosmology import Planck15 as cosmo
        if builtincosmo=='Planck18': from astropy.cosmology import Planck18 as cosmo
        H_0 = cosmo.H(0).value
    h = H_0/100
    A_s = 2.14e-9 # Scalar amplitude
    D_z = D(z) # Growth Function (normalised to unity for z=0)
    delta_c = 1.686
    if UseCAMB==True: GetModelPk(z,1e-4,1e0,NonLinear=False,UseNBK=UseNBK) # Use to set global transfer function T

#def astropy_cosmo():
#    return cosmo

def f(z):
    gamma = 0.545
    return Omega_m(z)**gamma

def E(z):
    return np.sqrt( 1 - Om0 + Om0*(1+z)**3 )

def H(z):
    if astropy==False: return E(z)*H_0
    if astropy==True: return cosmo.H(z).value

def Omega_m(z):
    return H_0**2*Om0*(1+z)**3 / H(z)**2

def Omega_b(z=0):
    if z!=0: print('\nError: Cosmotools needs evoloution for Omega_b(z)!')
    return Ob0

def d_com(z,UseCamb=False): #Comoving distance [Mpc/h]
    if astropy==False:
        func = lambda z: (c_km/H_0)/E(z)
        return scipy.integrate.romberg(func,0,z) * h
    if astropy==True:
        return cosmo.comoving_distance(z).value * h

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

def GetModelPk(z,kmin=1e-3,kmax=10,NonLinear=False,UseCAMB=True,UseNBK=False,UseCLASS=False):
    '''
    Generate model power spectrum at redshift z using pycamb (default) or can use
    Nbodykit package
    '''
    if UseNBK==True or UseCLASS==True: UseCAMB = False
    if UseCAMB==True:
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
    if UseNBK==True:
        from nbodykit.lab import cosmology
        cosmo = cosmology.Planck15
        return cosmology.LinearPower(cosmo, z, transfer='EisensteinHu')
    if UseCLASS==True:
        import classylss.binding as CLASS
        #cosmo = CLASS.ClassEngine({'output': 'dTk vTk mPk', 'non linear': 'halofit', 'P_k_max_h/Mpc' : 20., "z_max_pk" : 100.0})
        cosmo = CLASS.ClassEngine({'output': 'dTk vTk mPk', 'P_k_max_h/Mpc' : 20., "z_max_pk" : 100.0})
        sp = CLASS.Spectra(cosmo)
        k = np.linspace(kmin,kmax,10000)
        return interp1d(k, sp.get_pk(k=k,z=z) )
