###########################################################
# SPEpy - simplified parquet equation solver for SIAM     #
# Vladislav Pokorny; 2015-2019; pokornyv@fzu.cz           #
# homepage: github.com/pokornyv/SPEpy                     #
# developed and optimized using python 3.7.2              #
# params_siam.py - reading parameter file                 #
###########################################################

import scipy as sp
from configparser import SafeConfigParser
from os import listdir
from sys import argv

###########################################################
## reading parameters from command line ###################
U      = float(argv[1])
Delta  = float(argv[2])
ed     = float(argv[3])
T      = float(argv[4])
h      = 0.0         ## magnetic field not yet implemented

## reading guess for Lambdas from command line
try:
	LppIn = float(argv[5])
	LmpIn = float(argv[6])
	Lin = True
except IndexError:
	LppIn = LmpIn = 0.0
	Lin = False

###########################################################
## reading config file ####################################
cfile = 'siam.in'

if cfile not in listdir('.'): 
	print('- Parameter file '+cfile+' missing. Exit.')
	exit(1)

config = SafeConfigParser()
config.read(cfile)

## default values

NE               = 21
dE               = 1e-4
PrecLambda       = 1e-6
PrecSigmaT       = 1e-3
PrecN            = 1e-4
alpha            = 1.0
GFtype           = 'lor'
GFmethod         = 'H'
calcSusc         = True
cat              = True
WriteGF          = True
WriteVertex      = False
WriteNpz         = False
WriteMax         = 10
WriteStep        = 3

## read the inputs ########################################
## calculation parameters

if config.has_option('params','NE'):
	NE       = int(config.get('params','NE'))
if config.has_option('params','dE'):
	dE       = float(config.get('params','dE'))
if config.has_option('params','PrecLambda'):
	epsl     = float(config.get('params','PrecLambda'))
if config.has_option('params','PrecSigmaT'):
	epst     = float(config.get('params','PrecSigmaT'))
if config.has_option('params','PrecN'):
	epsn     = float(config.get('params','PrecN'))
if config.has_option('params','GFtype'):
	GFtype   = str(config.get('params','GFtype'))
if config.has_option('params','calcSusc'):
	calcSusc = bool(int(config.get('params','calcSusc')))
if config.has_option('params','alpha'):
	alpha    = float(config.get('params','alpha'))
if config.has_option('params','GFmethod'):
	GFmethod = str(config.get('params','GFmethod'))

## I/O parameters

if config.has_option('IO','WriteOutput'):
	chat        = bool(int(config.get('IO','WriteOutput')))
if config.has_option('IO','WriteFileGreenF'):
	WriteGF     = bool(int(config.get('IO','WriteFileGreenF')))
if config.has_option('IO','WriteFileVertex'):
	WriteVertex = bool(int(config.get('IO','WriteFileVertex')))
if config.has_option('IO','WriteDataFile'):
	WriteNpz    = bool(int(config.get('IO','WriteDataFile')))
if config.has_option('IO','WriteMax'):
	WriteMax    = sp.fabs(float(config.get('IO','WriteMax')))
if config.has_option('IO','WriteStep'):
	WriteStep   = int(config.get('IO','WriteStep'))

###########################################################
## energy axis ############################################
## RuntimeWarning: invalid value encountered in power: 
## for KK we need range(N)**3, for large arrays it can hit the limit of
## 9223372036854775808 == 2**63 of signed int, then decrease number of energy points

def FillEnergies(dE,N):
	"""	returns the symmetric array of energies 
	[Emin,Emin+dE,...,0,...,-Emin-dE,-Emin] of length N """
	dE_dec = int(-sp.log10(dE))
	En_A = sp.linspace(-(N-1)/2*dE,(N-1)/2*dE,N)
	return sp.around(En_A,dE_dec+2)

N = 2**NE-1
dE_dec = int(-sp.log10(dE))
En_A = FillEnergies(dE,N)

## config_siam.py end ##

