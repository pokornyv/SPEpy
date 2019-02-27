###########################################################
# SPEpy - simplified parquet equation solver for SIAM     #
# Copyright (C) 2019  Vladislav Pokorny; pokornyv@fzu.cz  #
# homepage: github.com/pokornyv/SPEpy                     #
# params_dmft.py - reading parameter file                 #
###########################################################

import scipy as sp
from os import listdir
from configparser import SafeConfigParser

cfile = 'dmft.in'

if cfile not in listdir('.'): 
	print('- Parameter file '+cfile+' missing. Exit.')
	exit(1)

config = SafeConfigParser()
config.read(cfile)

###########################################################
## read the inputs ########################################

## there are no default values as sme make no sense for DMFT (default U?)

## calculation parameters
U      = float(config.get('params','U'))
Delta  = float(config.get('params','Delta'))
ef     = float(config.get('params','ef'))
T      = float(config.get('params','Temp'))

NE     = int(config.get('params','NE'))
dE     = float(config.get('params','dE'))
epsl   = float(config.get('params','PrecLambda'))
epss   = float(config.get('params','PrecSigmaT'))

NStart = int(config.get('params','NStart'))
NIter  = int(config.get('params','NIter'))

if config.has_option('params','alpha'):
	alpha = float(config.get('params','alpha'))
else: alpha = 1.0

## I/O parameters

WriteFiles = bool(int(config.get('IO','WriteFiles')))
chat       = bool(int(config.get('IO','WriteOutput')))
WriteMax   = float(config.get('IO','WriteMax'))
WriteStep  = int(config.get('IO','WriteStep'))

###########################################################
## energy axis ############################################

def FillEnergies(dE,N):
	"""	returns the symmetric array of energies 
	[Emin,Emin+dE,...,0,...,-Emin-dE,-Emin] of length N """
	dE_dec = int(-sp.log10(dE))
	En_A = sp.linspace(-(N-1)/2*dE,(N-1)/2*dE,N)
	return sp.around(En_A,dE_dec+2)

N = 2**NE-1
dE_dec = int(-sp.log10(dE))
En_F = FillEnergies(dE,N)

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

FD_A = FillFD(En_A,T)
BE_A = FillBE(En_A,T)

## config_dmft.py end ###

