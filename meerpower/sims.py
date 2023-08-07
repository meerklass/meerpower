import numpy as np
import h5py
import Init
import HItools

def MultiDarkSimBuild(MD_filename,ra,dec,nu):
    ''' Simulation builder for HI intensity maps. Uses multi-dark light-cone output
    which can be obtained from: https://tao.asvo.org.au for specified RA/Dec/Redshift
    range
    '''
    # MD_filename: path string for multi-dark .hdf5 lightcone catalogue

    '''
    with h5py.File(MD_filename,'r') as f:
        for key in f.keys():
            print(key)
        x = f['X'][()]
        y = f['Y'][()]
        z = f['Z'][()]
        Vz = f['Vz'][()]
        M_CGM = f['McoldDisk'][()] #h^-1 Solar Masses

    R_mol = 0.4 #From A.Zoldan https://arxiv.org/pdf/1610.02042.pdf pg5
    f_mol = R_mol / (R_mol + 1) #From https://arxiv.org/pdf/astro-ph/0605035.pdf eq 21
    f_H = 0.75 # fraction of hydrogen present in the cold gas mass

    M_HI_grid = np.histogramdd((x,y,z),bins=(xbins,ybins,zbins),weights=M_HI)[0]
    '''
