SPEpy
=====
#### Description:

SPEpy (**S**implified **P**arquet **E**quations in **py**thon), is a python3 code to solve 
the one-band single-impurity Anderson model (SIAM) or the one-band Hubbard model within the 
_simplified parquet equation scheme_ (SPE) as described in Refs. [1-3].
This version is capable of solving SIAM with Lorentzian, Gaussian, semi-elliptic and the 
square- and simple-cubic-lattice non-interacting density of states (DoS).  

This solver can also be used self-consistently as a solver in the DMFT loop. 
The DMFT loop is implemented only for the Bethe lattice with semi-elliptic DoS and gives metallic 
solutions only. This code is more a proof of concept than a real DMFT solver, and should be used as that.  

This code is still subject to heavy development and 
big changes. It requires deep knowledge of the method to obtain reasonable results. 
Codes also contain internal switches that allow e.g. to change the level of self-consistency contitions. 
Use only as an example how to implement SPE. Code *siam_parquet.py* is yet in early stage of development.  

Code is now tested on python 3.7 with SciPy 1.1. Compatibility with older SciPy versions
is not guaranteed. Some parts of code use [mpmath](mpmath.org) library to calculate special functions 
of complex variable and to perform abitrary-precision calculations. Tested with mpmath 1.0.

SPEpy is a free software distributed under the GPL license.

#### Project homepage:
[github.com/pokornyv/SPEpy](https://github.com/pokornyv/SPEpy)

#### Usage:
- `python3 siam_static.py <float U> <float Delta> <float eps> <float T>`  
- `python3 siam_parquet.py <float U> <float Delta> <float eps> <float T> <LppIn> <LmpIn>`  
- `python3 2nd_order.py <float U> <float Delta> <float eps> <float T>`  
- `python3 dmft_parquet.py`  

where *python3* is the alias for the python 3 interpreter. Model and method parameters are declared in the
*siam.in* and *dmft.in* files, respectively. *LppIn* and *LmpIn* are values of the irreducible vertex
for which we already have results and we want to use them as the initial conditions. These parameters are 
not necessary but they occasionaly speed up the calculation.

#### TODO list:
- [x] Fix *RuntimeWarning: overflow encountered in exp* in `FermiDirac` and `BoseEinstein`.
- [ ] The quasiparticle residue *Z* is not always calculated correctly due to an instability in the 
numerical differentiation procedure (needs testing).
- [ ] Fix signed int overflow in `KramersKronigFFT`.
- [ ] Implement cubic lattice DoS for future **k**-dependent calculations to *siam_parquet.py*.
- [ ] Implement square lattice DoS for *d*-wave ordering calculations to *siam_parquet.py*.
- [ ] Implement magnetic field to *siam_parquet.py*.
- [ ] Temperature dependence in *siam_parquet.py* works only to small temperatures.
- [ ] Update the dmft code for the new concept (needs testing, works only at half-filling).
- [ ] Speed-up the simple-cubic/square lattice calculation by using the Hilbert transform (maybe not possible).

#### List of files:
- *parlib.py* - library of general functions and for *siam_static.py*
- *parlib2.py* - library of functions used in *siam_parquet.py*
- *siam_static.py* - solves the one-band single-impurity Anderson model or the Hubbard model using 
SPE with static Lambda vertex as described in Refs. [1-3]
- *siam_parquet.py* - solves the one-band single-impurity Anderson model using SPE with modified Lambda vertex,
still in development
- *dmft_parquet.py* - calculates the DMFT solution of the half-filled one-band Hubbard model on a Bethe lattice using static SPE
- *config_siam.py* - reads input from command line and from *siam.in*, sets up basic global arrays
- *config_dmft.py* - reads input file *dmft.in*, sets up basic global arrays
- *siam_infile.md* - description of the *siam.in* file
- *dmft_infile.md* - description of the *dmft.in* file
- *README.md* - this document

#### Output files:
*siam_static.py* generates two output files (see `WriteFileGreenF` and `WriteFileVertex` switches in *siam.in*). 
File *gf_xxxx.dat* contains the real and imaginary part of the non-interacting Green function, the spectral self-energy and the
interacting Green function. File *vertex_xxxx.dat* contains the two-particle bubble, dynamical part of the vertex *K* and the kernel
of the Schwinger-Dyson equation. File *siam.npz* contains raw data for further processing.

*siam_parquet.py* generates equivalent file named *gfyy_xxxx.dat* where *yy* = *Up* or *Dn* is the spin channel.
For non-magnetic setup only the *Up* file is generated. `WriteFileVertex` switch in *siam.in* is dummy for this code.

*dmft_parquet.py* generates output file *gf_iterxxxx.dat* with Greens function and self-energy for every 
iteration (see `WriteFiles` switch in *dmft.in*) and a final file *gf_Uxxxx_dmft.dat* at the end. Raw data from last iteration
are stored in *npz* archives. The evolution of the important quantities with dmft iterations is stored in *iterations.dat*.

#### References:
1. [V. Janiš, V. Pokorný, and A. Kauch, *Phys. Rev. B* **95**, 165113 (2017).](http://journals.aps.org/prb/abstract/10.1103/PhysRevB.95.045108)
2. [V. Janiš, A. Kauch, and V. Pokorný, *Phys. Rev. B* **95**, 045108 (2017).](http://journals.aps.org/prb/abstract/10.1103/PhysRevB.95.165113)
3. [V. Pokorný, M. Žonda, A. Kauch, and V. Janiš, *Acta Phys. Pol. A* **131**, 1042 (2017).](http://doi.org/10.12693/APhysPolA.131.1042)
4. [V. Janiš and P. Augustinský, *Phys. Rev. B* **77**, 085106 (2008).](https://doi.org/10.1103/PhysRevB.77.085106)
5. [V. Janiš and P. Augustinský, *Phys. Rev. B* **75**, 165108 (2007).](https://doi.org/10.1103/PhysRevB.75.165108)

