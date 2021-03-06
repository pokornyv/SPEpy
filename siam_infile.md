Description of the *siam.in* file
================================

The file *siam.in* defines various parameters used in the *siam_parquet*, *siam_static* and *super_parquet* codes.

## Example input file *siam.in*:
```
[params]

NE           : 21
dE           : 1e-4
PrecLambda   : 1e-6
PrecSigmaT   : 1e-4
PrecN        : 1e-4
alpha        : 0.5

;SCsolver     : 'fixed'
;SCsolver     : 'iter'
SCsolver     : 'root'

GFtype        : lor
;GFtype       : semi
;GFtype       : gauss

;GFtype       : sc
;GFtype       : scinf

calcSusc      : 0

SC            : 0
FSC           : 0

[IO]

WriteOutput      : 1
WriteFileGreenF  : 1
WriteFileVertex  : 1
WriteDataFile    : 0

WriteMax         : 30
WriteStep        : 3
```
  
## Description

### [params] section

- *NE* - the energy axis contains 2^NE+1 points (caution, NE>21 can cause trouble with signed int overflow)  
- *dE* - discretization of the energy axis  
- *PrecLambda* - convergence criterium used in calculation of the vertex Lambda  
- *PrecSigmaT* - convergence criterium used in calculation of the thermodynamic self-energy, used only in *siam_static*  
- *PrecN* - convergence criterium used in calculation of the electron density  
- *alpha* - mixing parameter used in calculation of the thermodynamic self-energy (*siam_static*) or 
the Lambda vertex (*siam_parquet*, if *SCsolver* = 'iter'), alpha=1 means no mixing  
- *SCsolver* - method of solving the self-consistent equations for Lambda vertex, used only in *siam_parquet* and *super_parquet*  
- *GFtype* - input Green function, lor - Loretzian, semi - semielliptic, gauss - gaussian (only for *siam_parquet* and *siam_static*), 
sc - supercondcting, finite gap, scinf - superconducting, infinite gap (only for *super_parquet*)  
- *calcSusc* - 0/1 switch whether calculate susceptibilities  
- *SC* - 0/1 switch whether use the partial self-consistency where the spectral self-energy is calculated from
thermodynamic bubble but spectral 1p propagator   
- *FSC* - 0/1 switch whether use the full self-consistency where the spectral self-energy is calculated from
spectral 2p bubble and spectral 1p propagator  

### [IO] section

- *WriteOutput* - 0/1 switch whether write calculation details to standard output  
- *WriteFileGreenF* - 0/1 switch whether write output file with Green functions and self-energy  
- *WriteFileVertex* - 0/1 switch whether write output file with vertex functions, used only in *siam_static*  
- *WriteDataFile* - 0/1 switch whether write the Green function and the self-energy to 
the *siam.npz* archive for additional processing  
- *WriteMax* - maximum of the energy window for file output  
- *WriteStep* - step in energies for file output. The values will be written with 10^WriteStep x dE step 
(e.g. WriteStep = 3 means every thousandth point)  

