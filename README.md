SPEpy
=====
#### Description:

SPEpy (**S**implified **P**arquet **E**quations in **py**thon), is a python3 code to solve 
the one-band single-impurity Anderson model (SIAM) or the one-band Hubbard model within the 
_simplified parquet equation scheme_ (SPE) as described in Refs. [1-2]. 
This solver can also be used self-consistently as a solver in the DMFT loop (metallic solutions only). 
This version is capable of solving SIAM with Lorentzian, Gaussian, semi-elliptic and the simple-cubic-lattice 
non-interacting density of states (DoS).  

The DMFT loop is implemented only for the Bethe lattice with semi-elliptic DoS.  

This code is still subject to heavy development and 
big changes. It requires deep knowledge of the method to obtain reasonable results. 
Codes also contain internal switches that allow e.g. to change the level of self-consistency contitions. 
Use only as an example how to implement SPE.  

Code is tested on python 3.5 and 3.6 with SciPy 0.18. Compatibility with older SciPy versions
is not guaranteed. Some parts of code use [mpmath](mpmath.org) library to calculate special functions 
of complex variable and to perform abitrary-precision calculations. Tested with mpmath 0.19.

#### Project homepage:
https://github.com/pokornyv/SPEpy

#### Usage:
- `python3 siam_static.py <float U> <float Delta> <float eps> <float T>`  
- `python3 siam_dynamic.py <float U> <float Delta> <float eps> <float Uin>`  
- `python3 2nd_order.py <float U> <float Delta> <float eps> <float T>`  
- `python3 dmft_parquet.py`  

where *python3* is the alias for the python 3 interpreter. Model and method parameters are declared in the
*siam.in* and *dmft.in* files, respectively. *Uin* is the value of interaction strength for which we already 
have results and we want to use them as the initial condition.

#### TODO list:
- [x] Temperature dependence works only for the half-filling case (needs testing).
- [x] Implement simple-cubic lattice non-interacting DoS for future **k**-dependent calculations (needs testing).
- [x] The quasiparticle residue *Z* is not calculated correctly due to an instability in the numerical differentiation procedure (needs testing) .
- [x] Update the dmft code for the new concept (needs testing, implemented only for half-filling).
- [ ] Implement square lattice non-interacting DoS for d-wave ordering study in the Hubbard model.
- [ ] Speed-up the simple-cubic/square lattice calculation by using the Hilbert transform (maybe not possible).
- [ ] Fix signed int overflow in `KramersKronigFFT`.
- [ ] Fix *RuntimeWarning: overflow encountered in exp* in `FermiDirac` and `BoseEinstein`.
- [ ] Clean the *siam_dynamic.py* code.


#### List of files:
- *parlib.py* - library of functions  
- *siam_static.py* - solves the one-band single-impurity Anderson model or the Hubbard model using SPE with static Lambda vertex  
- *siam_dynamic.py* - solves the one-band single-impurity Anderson model or the Hubbard model using SPE with dynamic Lambda vertex  
- *2nd order.py* - solves the one-band single-impurity Anderson model in 2nd order of perturbation theory  
- *dmft_parquet.py* - calculates the DMFT solution of the half-filled one-band Hubbard model on a Bethe lattice using (static) SPE  
- *params_siam.py* - reads input file *siam.in* with model and method parameters  
- *params_dmft.py* - reads input file *dmft.in* with model and method parameters  
- *siam_infile.md* - description of the *siam.in* file  
- *dmft_infile.md* - description of the *dmft.in* file  

#### Output files:
*siam_static.py* generates two output files (see `WriteFileGreenF` and `WriteFileVertex` switches in *siam.in*). 
File *gf_xxxx.dat* contains the real and imaginary part of the non-interacting Green function, the spectral self-energy and the
interacting Green function. File *vertex_xxxx.dat* contains the two-particle bubble, dynamical part of the vertex *K* and the kernel
of the Schwinger-Dyson equation. File *siam.npz* contains raw data for further processing.  

*siam_dynamic.py* generates equivalent files named *gfdyn_xxxx.dat* containing the thermodynamic self-energy, 
thermodynamic Green function, spectral self-energy and spectral Green function, and *vertexdyn_xxxx.dat*
containing Lambda vertex, K vertex and the kernel of the Schwinger-Dyson equation. Data file is called *data_xxx.dat*
and can be used as an initial condition for a different calculation.

*dmft_parquet.py* generates output file *gf_iterxxxx.dat* with Greens function and self-energy for every 
iteration (see `WriteFiles` switch in *dmft.in*) and a final file *gf_Uxxxx_dmft.dat* at the end. Raw data from last iteration
are stored in *npz* archives. The evolution of the important quantities with dmft iterations is stored in *iterations.dat*.

#### References:
1. V. Janiš, V. Pokorný, and A. Kauch, *Phys. Rev. B* **95**, 165113 (2017).  
2. V. Janiš, A. Kauch, and V. Pokorný, *Phys. Rev. B* **95**, 045108 (2017).  
3. V. Janiš and P. Augustinský, *Phys. Rev. B* **77** (2008).  
4. V. Janiš and P. Augustinský, *Phys. Rev. B* **75** (2007).  

