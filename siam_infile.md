Description of the *siam.in* file
================================

The file *siam.in* defines various parameters used in the *siam_parquet.py* and *2nd_order.py* codes.

## Example input file *siam.in*:
```
[params]

NE           : 21
dE           : 1e-4
PrecLambda   : 1e-6
PrecSigmaT   : 1e-4
PrecN        : 1e-4
alpha        : 0.5

;GFtype       : semi
;GFtype       : gauss
GFtype        : lor
;GFtype       : sc
;GFtype       : sq

calcSusc         : 0

[IO]

WriteOutput      : 1
WriteFileGreenF  : 1
WriteFileVertex  : 1
WriteDataFile    : 0

WriteMax         : 10
WriteStep        : 3
```
  
##Description

###[params] section

- *NE* - the energy axis contains 2^NE+1 points (caution, NE>21 can cause trouble with signed int overflow)  
- *dE* - discretization of the energy axis  
- *PrecLambda* - convergence criterium used in calculation of the vertex Lambda  
- *PrecSigmaT* - convergence criterium used in calculation of the thermodynamic self-energy  
- *PrecN* - convergence criterium used in calculation of the electron density  
- *alpha* - mixing parameter used in calculation of the thermodynamic self-energy or 
the Lambda vertex, alpha=1 means no mixing  
- *GFtype* - input Green function, lor - Loretzian, semi - semielliptic, 
sc - simple-cubic lattice, sq - square lattice  
- *calcSusc* - 0/1 switch whether calculate susceptibilities (it takes a lot of time)  

###[IO] section

- *WriteOutput* - 0/1 switch whether write output to standard IO (e.g. screen)  
- *WriteFileGreenF* - 0/1 switch whether write output file with Green functions and self-energy  
- *WriteFileVertex* - 0/1 switch whether write output file with vertex functions  
- *WriteDataFile* - 0/1 switch whether write the Green function and the self-energy to the *siam.npz* archive for additional processing  
- *WriteMax* - maximum of the energy window for file output  
- *WriteStep* - step in energies for file output. The values will be written with 10^WriteStep x dE step 
(e.g. WriteStep = 3 means every thousandth point)  
