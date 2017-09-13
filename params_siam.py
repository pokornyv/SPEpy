import scipy as sp
from configparser import SafeConfigParser

config = SafeConfigParser()
config.read('siam.in')

###########################################################
# read the inputs #########################################

# calculation parameters

alpha    = 1.0
GFmethod = 'H'
calcSusc = True

NE       = int(config.get('params','NE'))
dE       = float(config.get('params','dE'))
epsl     = float(config.get('params','PrecLambda'))
epst     = float(config.get('params','PrecSigmaT'))
epsn     = float(config.get('params','PrecN'))
GFtype   = str(config.get('params','GFtype'))

if config.has_option('params','calcSusc'):
	calcSusc = bool(int(config.get('params','calcSusc')))
if config.has_option('params','alpha'):
	alpha    = float(config.get('params','alpha'))
if config.has_option('params','GFmethod'):
	GFmethod = str(config.get('params','GFmethod'))

# I/O parameters

chat        = bool(int(config.get('IO','WriteOutput')))
WriteGF     = bool(int(config.get('IO','WriteFileGreenF')))
WriteVertex = bool(int(config.get('IO','WriteFileVertex')))
WriteNpz    = bool(int(config.get('IO','WriteDataFile')))
WriteMax    = sp.fabs(float(config.get('IO','WriteMax')))
WriteStep   = int(config.get('IO','WriteStep'))

## siam.in end ##

