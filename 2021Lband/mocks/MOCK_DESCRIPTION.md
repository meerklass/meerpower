#2021Lband mocks for HI intensity maps and galaxy catalogues
Steve Cunnington [steven.cunnington@manchester.ac.uk]
05/04/24

This folder contains 500 realisations of lognormal mocks. The mocks come in two versions:
1) HI-only intensity maps (same map coordinates as 0.3 deg MeerKAT maps)
2) Galaxy catalogues (with GAMA-like number densities and footprint)

Mocks generates using ``meerpower/2021Lband/mocks/logNbuilder.py`` script.

Mocks created using gridimp [https://github.com/stevecunnington/gridimp] codes allowing the Cartesian lognormal field to be transformed to sky-map space so intensity maps and galaxy positions can be saved with [RA,Dec,z] coordinates, emulating input real data.

*Details are given below*:
Both fields begin with a lognormal field generation in a high-resolution Cartesian-space [512,512,512] cells over a region calculated to enclose the MeerKAT 2021Lband footprint. Both HI and galaxy lognormal are generated with a consistent random seed to assure they will correlate.

1) Intensity maps: Each lognormal HI field realisation assumes a HI bias (b_HI=1.5) and mean HI temp (based on Omega_HI b_HI=0.85e-3). Each Cartesian lognormal field is sampled onto a sky-map 'lightcone' with the same pixel size and frequency channels as MeerKAT data. Map is smoothed in sky coordinates with a frequency-dependent Gaussian beam, assuming dish-diameter of 13.5m. Map edges are then cut with same analysis mask used on real data. This mitigates any edge effects from smoothing.

2) Galaxy catalogues: A lognormal galaxy density field is generated in same high-resolution Cartesian grid as the HI but instead assumes a galaxy linear bias of b_g=2.35, consistent with the real GAMA data auto-correlation. Field is Poission sampled giving a number counts field for the entire Cartesian grid, with number counts being consistent with the real GAMA-G23 number densities. Each number count cell is assigned a uniform random coordinate within the cell for all the counts within it. These provide a catalogue of Cartesian coordinates for each galaxy. These are then transformed to sky (RA,Dec,z) coordinates and saved.
