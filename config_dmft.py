###########################################################
# SPEpy - simplified parquet equation solver for SIAM     #
# Vladislav Pokorny; 2015-2019; pokornyv@fzu.cz           #
# homepage: github.com/pokornyv/SPEpy                     #
# developed and optimized using python 3.7.2              #
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

## energy axis #######################################################

def FillEnergies(dE,N):
	"""	returns the symmetric array of energies 
	[Emin,Emin+dE,...,0,...,-Emin-dE,-Emin] of length N """
	dE_dec = int(-sp.log10(dE))
	En_A = sp.linspace(-(N-1)/2*dE,(N-1)/2*dE,N)
	return sp.around(En_A,dE_dec+2)

N = 2**NE-1
dE_dec = int(-sp.log10(dE))
En_F = FillEnergies(dE,N)

## config_dmft.py end ###

