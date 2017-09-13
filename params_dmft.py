import scipy as sp
from configparser import SafeConfigParser

config = SafeConfigParser()
config.read('dmft.in')

###########################################################
## read the inputs ########################################

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

## params_dmft.py end ###
