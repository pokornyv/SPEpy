###########################################################
# SPEpy - simplified parquet equation solver for SIAM     #
# Copyright (C) 2019  Vladislav Pokorny; pokornyv@fzu.cz  #
# homepage: github.com/pokornyv/SPEpy                     #
# params_siam.py - reading parameter file                 #
###########################################################

import scipy as sp
from configparser import SafeConfigParser
from os import listdir
from sys import argv

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
epsl             = 1e-6
epst             = 1e-3
epsn             = 1e-6
alpha            = 0.5
SCsolver         = 'fixed'
GFtype           = 'lor'
GFmethod         = 'H'
SC               = False
FSC              = False
calcSusc         = True
chat             = True
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
if config.has_option('params','alpha'):
	alpha    = float(config.get('params','alpha'))
if config.has_option('params','SCsolver'):
	SCsolver = str(config.get('params','SCsolver'))
if config.has_option('params','GFtype'):
	GFtype   = str(config.get('params','GFtype'))
if config.has_option('params','calcSusc'):
	calcSusc = bool(int(config.get('params','calcSusc')))
if config.has_option('params','GFmethod'):
	GFmethod = str(config.get('params','GFmethod'))
if config.has_option('params','SC'):
	SC   = bool(int(config.get('params','SC')))
if config.has_option('params','FSC'):
	FSC   = bool(int(config.get('params','FSC')))

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
## reading parameters from command line ###################
if GFtype in ['sc','scinf']: ## superconducting model, run super_parquet.py
	U      = float(argv[1])
	DeltaS = float(argv[2])
	GammaS = float(argv[3])
	GammaN = float(argv[4])
	P      = float(argv[5])
	T = eps = h = 0.0
	Phi = P*sp.pi
else:					## normal model, run siam_parquet.py or siam_static.py
	U      = float(argv[1])
	Delta  = float(argv[2])
	ed     = float(argv[3])
	T      = float(argv[4])
	try:
		h = float(argv[5])
	except IndexError:
		h = 0.0

## reading guess for Lambdas from command line
try:
	LIn = float(argv[6])
	Lin = True
except IndexError:
	LIn = 0.0
	Lin = False


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

izero = 0.0 ## imaginary shift of energies, useful for DoS with poles or singularities

N = 2**NE-1
dE_dec = int(-sp.log10(dE))
En_A = FillEnergies(dE,N)

Nhalf = int((len(En_A)-1)/2)	## zero on the energy axis

###########################################################
## particle distributions #################################

offE = 1e-12

FermiDirac      = lambda E,T: 1.0/(sp.exp((E+offE)/T)+1.0)
BoseEinstein    = lambda E,T: 1.0/(sp.exp((E+offE)/T)-1.0)
FermiDiracDeriv = lambda E,T: -(1.0/T)*sp.exp((E+offE)/T)/(sp.exp((E+offE)/T)+1.0)**2

def FillFD(En_A,T):
	""" fill an array with Fermi-Dirac distribution """
	N = int((len(En_A)-1)/2)
	sp.seterr(over='ignore') ## ignore overflow in exp, not important in this calculation
	if T == 0.0: FD_A = 1.0*sp.concatenate([sp.ones(N),[0.5],sp.zeros(N)])
	else:        FD_A = FermiDirac(En_A,T)
	sp.seterr(over='warn')
	return FD_A


def FillBE(En_A,T):
	""" fill an array with Bose-Einstein distribution """
	N = int((len(En_A)-1)/2)
	sp.seterr(over='ignore') ## ignore overflow in exp, not important in this calculation
	if T == 0.0: BE_A = -1.0*sp.concatenate([sp.ones(N),[0.5],sp.zeros(N)])
	else:        
		BE_A = BoseEinstein(En_A,T)
		BE_A[N] = -0.5
	sp.seterr(over='warn')
	return BE_A


def FillFDplusBE(En_A,T):
	""" fill an array with a som of Bose-Einstein and Fermi-Dirac
	numerically more precise than the sum of the above functions due to a pole in BE """
	N = int((len(En_A)-1)/2)
	sp.seterr(over=  'ignore') ## ignore overflow in sinh
	sp.seterr(divide='ignore') ## we deal with the pole ourselves
	if T == 0.0: FB_A = sp.zeros(len(En_A))
	else:        
		FB_A = 1.0/sp.sinh(En_A/T)
		FB_A[N] = 0.0
	sp.seterr(divide='warn')
	sp.seterr(over=  'warn')
	return FB_A

FD_A = FillFD(En_A,T)
BE_A = FillBE(En_A,T)
FB_A = FillFDplusBE(En_A,T)

## config_siam.py end ##

