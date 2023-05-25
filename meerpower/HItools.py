import numpy as np
from astropy.cosmology import Planck15 as cosmo
import astropy
from astropy.cosmology import Planck15 as cosmo
# Some constants:
from astropy import constants
c = astropy.constants.c.value #m/s - speed of light
h_P = constants.h.value #m^2 kg / s - Planck's constant
k_B = constants.k_B.value #m^2 kg / s^2 K^1 - Boltzmann's constant
A_12 = 2.876e-15 #Hz - Einstein coefficient (D.Alonso https://arxiv.org/pdf/1405.1751.pdf)
m_H = 1.6737236e-27 #kg  - Hydrogen atom mass
M_sun = constants.M_sun.value #kg - Solar mass
H0 = cosmo.H(0).value #km / Mpc s - Hubble constant
h = H0/100
v_21cm = 1420.405751#MHz

def b_HI(z):
    '''
    Use 6 values for HI bias at redshifts 0 to 5 found in Table 5 of
    Villaescusa-Navarro et al.(2018) https://arxiv.org/pdf/1804.09180.pdf
    and get a polyfit function based on these values
    '''
    #### Code for finding polynomial coeficients: #####
    #z = np.array([0,1,2,3,4,5])
    #b_HI = np.array([0.84, 1.49, 2.03, 2.56, 2.82, 3.18])
    #coef = np.polyfit(z, b_HI,2)
    #A,B,C = coef[2],coef[1],coef[0]
    ###################################################
    A,B,C = 0.84178571,0.69289286,-0.04589286
    return A + B*z + C*z**2

def OmegaHImodel(z):
     # Matches SKAO red book and Alkistis early papers also consistent with GBT
     #   Masui + Wolz measurements at z=0.8
    return 0.00048 + 0.00039*z - 0.000065*z**2

def Tbar(z,OmegaHI):
    # Battye+13 formula
    Hz = cosmo.H(z).value #km / Mpc s
    H0 = cosmo.H(0).value #km / Mpc s
    h = H0/100
    return 180 * OmegaHI * h * (1+z)**2 / (Hz/H0)

def P_SN(z):
    '''
    Use 6 values for HI shot noise at redshifts 0 to 5 found in Table 5 of
    Villaescusa-Navarro et al.(2018) https://arxiv.org/pdf/1804.09180.pdf
    and get a polyfit function based on these values
    '''
    #### Code for finding polynomial coeficients: #####
    #z = np.array([0,1,2,3,4,5])
    #P_SN = np.array([104,124,65,39,14,7])
    #coef = np.polyfit(z, P_SN,4)
    #A,B,C,D,E = coef[4],coef[3],coef[2],coef[1],coef[0]
    ###################################################
    A,B,C,D,E = 104.76587301587332, 81.77513227513245, -87.78472222222258, 23.393518518518654, -1.9791666666666783
    return A + B*z + C*z**2 + D*z**3 + E*z**4

def Red2Freq(z):
    # Convert redshift to frequency for HI emission (freq in MHz)
    return v_21cm / (1+z)

def Freq2Red(v):
    # Convert frequency to redshift for HI emission (freq in MHz)
    return (v_21cm/v) - 1

def BrightnessTemps(z,dv,M_HI,pixarea):
    '''
    Takes M_HI field in (M_sun/h) units and makes into brightness temperature field.
    '''
    deltav = dv #Freq bin width (Hz)
    M_HI = M_HI * M_sun/h #convert to kg
    #Below following D.Alonso 2014 (https://arxiv.org/pdf/1405.1751.pdf)
    #   and S.Cunnington et al. (https://arxiv.org/pdf/1904.01479.pdf):
    r = cosmo.comoving_distance(z).value #Mpc
    r = r * 3.086e22 #Convert from Mpc to metres
    dOmega = np.radians(np.sqrt(pixarea))**2 #square size of pixel in radians
    T_HI = 3*h_P*c**2*A_12/(32*np.pi*m_H*k_B*v_21cm*1e6) * 1/((1+z)*r)**2 * M_HI/(deltav*dOmega)
    return T_HI
