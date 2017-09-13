Description of the *dmft.in* file
================================

The file *dmft.in* defines various parameters used in the *dmft_parquet.py* code.

## Example input file *dmft.in*:
```
[params]

U            : 1.0
Delta        : 1.0
ef           : 0.0
Temp         : 0.0

NE           : 21
dE           : 1e-4
PrecLambda   : 1e-6
PrecSigmaT   : 1e-4

NStart        : 1
NIter         : 10

alpha        : 1.0

[IO]

WriteFiles   : 1
WriteOutput  : 1
WriteMax     : 10.0
WriteStep    : 3
```
  
##Description

###[params] section

- *U* - Coulomb interaction
- *Delta* - hybridization, not used in this code, only for compatibility
- *ef* - local energy level, not used in this code, only for compatibility
- *Temp* - temperature

- *NE* - the energy axis contains 2^NE+1 points
- *dE* - discretization of the energy axis
- *PrecLambda* - convergence criterium used in calculation of the static vertex Lambda
- *PrecSigmaT* - convergence criterium used in calculation of the thermodynamic self-energy, not used in this code

- *NStart* - iteration to start with, allows to continue the calculation from the (NStart-1)th iteration
- *NIter* - number of iterations to do

- *alpha* - mixing parameter between self-energies from previous and current iterations; alpha=1 means no mixing

###[IO] section

- *WriteFiles* - 0/1 switch whether write output files with Green functions and self-energy
- *WriteOutput* - 0/1 switch whether write output to standard IO (e.g. screen)
- *WriteMax* - maximum of the energy window for file output
- *WriteStep* - step in energies for file output. The values will be written with 10^WriteStep x dE step

