###########################################################
# SPEpy - simplified parquet equation solver for SIAM     #
# Copyright (C) 2019  Vladislav Pokorny; pokornyv@fzu.cz  #
# homepage: github.com/pokornyv/SPEpy                     #
# super_parquet.py - solver for superconducting model     #
# method described in <not published>                     #
###########################################################

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import scipy as sp
from scipy.integrate import simps
from scipy.optimize import brentq
from sys import argv,exit,version_info
from os import listdir
from time import ctime,time
from parlib  import *
from parlib2 import *

t = time()

hashes = '#'*80

## python version
ver = str(version_info[0])+'.'+str(version_info[1])+'.'+str(version_info[2])

## header for files so we store the parameters along with data
if GFtype  == 'sc':
	parline = '# U = {0: .4f}, DeltaS = {1: .4f}, GammaS = {2: .4f}, GammaN = {3: .4f}, Phi/pi = {4: .4f}, T = {5: .4f}'\
	.format(U,DeltaS,GammaS,GammaN,P,T)
	parfname = 'U'+str(U)+'D'+str(DeltaS)+'GS'+str(GammaS)+'GN'+str(GammaN)+'Phi'+str(P)
elif GFtype  == 'scinf':
	parline = '# U = {0: .4f}, DeltaS = inf, GammaS = {1: .4f}, GammaN = {2: .4f}, Phi/pi = {3: .4f}, T = {4: .4f}'\
	.format(U,GammaS,GammaN,P,T)
	parfname = 'U'+str(U)+'DinfGS'+str(GammaS)+'GN'+str(GammaN)+'Phi'+str(P)
header = parline+'\n# E\t\tRe Gn\t\tIm Gn\t\tRe Ga\t\tIm Ga\t\tRe G(rot)\t\tIm G(rot)'

###########################################################
## print the header #######################################
if chat:
	print(hashes+'\n# generated by '+str(argv[0])+', '+str(ctime()))
	print('# python version: '+str(ver)+', SciPy version: '+str(sp.version.version))
	print('# energy axis: [{0: .5f} ..{1: .5f}], step = {2: .5f}, length = {3: 3d}'\
     .format(En_A[0],En_A[-1],dE,len(En_A)))
	print(parline)
	print('# Kondo temperature from Bethe ansatz: Tk ~{0: .5f}'\
	.format(float(KondoTemperature(U,GammaS,0.0))))
	if SCsolver == 'fixed': 
		print('# using Steffensen fixed-point algorithm to calculate Lambda vertex')
	elif SCsolver == 'root': 
		print('# using MINPACK root to calculate Lambda vertex')
	else: 
		print('# using iteration algorithm to calculate Lambda vertex, mixing parameter alpha = {0: .5f}'\
		.format(float(alpha)))

## shift energies by small imaginary part
izero = 1e-8j   ## imaginary shift of energies to avoid poles at DeltaS
En_A = En_A+izero

###########################################################
## inicialize the non-interacting Green function ##########
if GFtype  == 'sc':
	if chat: print('# using non-interacting DoS for a superconducting quantum dot')
	Hyb_A  = HybFunctionSC(En_A,GammaS,GammaN,DeltaS,Phi,sgnA= 1)
	GFlambda = lambda x: GreensFunctionSC(x,Hyb_A,GammaN)
elif GFtype  == 'scinf':
	if chat: print('# using non-interacting DoS for a superconducting atomic limit (infinite gap)')
	GFlambda = lambda x: GreensFunctionSCinf(x,GammaS,GammaN,Phi)
else:
	if chat: print('# Error: the non-interacting GF is not compatible with the superconducting model')
	exit(1)


###########################################################
## calculate thermodynamic (HF) Green function ############
###########################################################
if chat: print('#\n# calculating the non-interacting Green function:')

GFzero_A = GFlambda(En_A)
nzero = Filling(GFzero_A)
if chat: print('# - norm[G0]     = {0: .8f}, n  = {1: .8f}'.format(IntDOS(GFzero_A),nzero))

## symmetrized Green function
GFzeroNorm_A = 0.5*(GFzero_A-sp.flipud(sp.conj(GFzero_A)))
GFzeroAnom_A = 0.5*(GFzero_A+sp.flipud(sp.conj(GFzero_A)))
nzero2  = Filling(GFzeroNorm_A)
nuzero2 = Filling(GFzeroAnom_A)
if chat: print('# - norm[G0norm] = {0: .8f}, nW = {1: .8f}'.format(IntDOS(GFzeroNorm_A),nzero2 ))
if chat: print('# - norm[G0anom] = {0: .8f}, nu = {1: .8f}'.format(IntDOS(GFzeroAnom_A),nuzero2))

#WriteFileX([GFzeroNorm_A,GFzeroAnom_A,GFzero_A],WriteMax,WriteStep,parline,'Gzero.dat')

###########################################################
## calculate the static self-energy #######################
if chat: print('#\n# calculating the Hartree-Fock Green function:')
eqn = lambda x: Filling(GFlambda(En_A-U*(x-0.5)))-x
nHF = brentq(eqn,0.0,1.0)

GFzero_A = GFlambda(En_A-U*(nHF-0.5))
nHF = Filling(GFzero_A)
if chat: print('# - norm[GHF]     = {0: .8f}, nW = {1: .8f}'.format(IntDOS(GFzero_A),nHF))

## symmetrized Green function
GFzeroNorm_A = 0.5*(GFzero_A-sp.flipud(sp.conj(GFzero_A)))
GFzeroAnom_A = 0.5*(GFzero_A+sp.flipud(sp.conj(GFzero_A)))
nHF2  = Filling(GFzeroNorm_A)
nuHF2 = Filling(GFzeroAnom_A)
if chat: print('# - norm[GHFnorm] = {0: .8f}, n  = {1: .8f}'.format(IntDOS(GFzeroNorm_A),nHF2 ))
if chat: print('# - norm[GHFanom] = {0: .8f}, nu = {1: .8f}'.format(IntDOS(GFzeroAnom_A),nuHF2))

JCHF = JosephsonCurrent(GFzeroAnom_A,En_A)
if chat: print('# - Josephson current: {0: .8f}'.format(JCHF))

'''
## print all combinations of bubbles, debug only
channel='ee'
Bubble_A = TwoParticleBubble(GFzeroNorm_A,GFzeroNorm_A,channel)
WriteFileX([Bubble_A],WriteMax,WriteStep,parline,'bubble'+str(GammaN)+'NNee.dat')
Bubble_A = TwoParticleBubble(GFzeroAnom_A,GFzeroAnom_A,channel)
WriteFileX([Bubble_A],WriteMax,WriteStep,parline,'bubble'+str(GammaN)+'AAee.dat')
Bubble_A = TwoParticleBubble(GFzeroNorm_A,GFzeroAnom_A,channel)
WriteFileX([Bubble_A],WriteMax,WriteStep,parline,'bubble'+str(GammaN)+'NAee.dat')
Bubble_A = TwoParticleBubble(GFzeroAnom_A,GFzeroNorm_A,channel)
WriteFileX([Bubble_A],WriteMax,WriteStep,parline,'bubble'+str(GammaN)+'ANee.dat')
exit()
'''

## write the output file ##################################
if WriteGF:
	if chat: print('# Writing output file(s):')
	header = parline+'\n# E\t\tRe GF0\t\tIm GF0\t\tRe SE\t\tIm SE\t\tRe GF\t\tIm GF'
	filename = 'gf_HF_'+parfname+'.dat'
	WriteFileX([GFzeroNorm_A,GFzeroAnom_A,GFzero_A],WriteMax,WriteStep,header,filename)
	RI1 = sp.real(GFzeroNorm_A)*sp.imag(GFzeroAnom_A)
	RI2 = sp.real(GFzeroAnom_A)*sp.imag(GFzeroNorm_A)
	WriteFileX([RI1,RI2],WriteMax,WriteStep,'','crossG.dat')

#print(simps(FD_A*RI1,En_A))
#print(simps(FD_A*RI2,En_A))

GG1  = CorrelatorImGGzero(GFzero_A,GFzero_A,1,1)
GG2n = CorrelatorImGGzero(GFzeroNorm_A,GFzeroNorm_A,1,1)
GG2a = CorrelatorImGGzero(GFzeroAnom_A,GFzeroAnom_A,1,1)

#print('{0: .8f} {1: .8f} {2: .8f} {3: .8f}'.format(sp.real(GG1),sp.imag(GG1),sp.real(GG2n+GG2a),sp.imag(GG2n+GG2a)))

###########################################################
## second order PT solution ###############################
###########################################################
if chat: print('#\n# calculating the second order PT solution for control:')
Bubble_A   = TwoParticleBubble(GFzero_A,GFzero_A,'eh')
ChiGamma_A = U**2*Bubble_A
Sigma2nd_A = SelfEnergy(GFzero_A,ChiGamma_A)
SE02nd     = sp.imag(Sigma2nd_A[int(N/2)])

## symmetrized self-energy
Sigma2ndNorm_A = 0.5*(Sigma2nd_A-sp.flipud(sp.conj(Sigma2nd_A)))
Sigma2ndAnom_A = 0.5*(Sigma2nd_A+sp.flipud(sp.conj(Sigma2nd_A)))
#WriteFileX([Sigma2ndNorm_A,Sigma2ndAnom_A,Sigma2nd_A],WriteMax,WriteStep,parline,'SE2nd.dat')

## correct the static part of the self-energy to fulfill charge consistency
eqn = lambda x: Filling(GFlambda(En_A-U*(x-0.5)-Sigma2nd_A))-x
n2nd = brentq(eqn,0.0,1.0)

GF2nd_A = GFlambda(En_A-U*(n2nd-0.5)-Sigma2nd_A)
n2nd = Filling(GF2nd_A)
if chat: print('# - norm[G2nd]     = {0: .8f}, nW = {1: .8f}'.format(IntDOS(GF2nd_A),n2nd))

## symmetrized Green function
GF2ndNorm_A = 0.5*(GF2nd_A-sp.flipud(sp.conj(GF2nd_A)))
GF2ndAnom_A = 0.5*(GF2nd_A+sp.flipud(sp.conj(GF2nd_A)))
n2nd2  = Filling(GF2ndNorm_A)
nu2nd2 = Filling(GF2ndAnom_A)
if chat: print('# - norm[G2ndNorm] = {0: .8f}, n  = {1: .8f}'.format(IntDOS(GF2ndNorm_A),n2nd2))
if chat: print('# - norm[G2ndAnom] = {0: .8f}, nu = {1: .8f}'.format(IntDOS(GFzeroAnom_A),nuzero2))

JC2nd = JosephsonCurrent(GF2ndAnom_A,En_A)
if chat: print('# - Josephson current: {0: .8f}'.format(JC2nd))

## DoS at Fermi energy
DOSF2nd = -sp.imag(GF2ndNorm_A[int(N/2)])/sp.pi

## write the output file ##################################
if WriteGF:
	if chat: print('# Writing output file(s):')
	filename = 'gf_2nd_'+parfname+'.dat'
	WriteFileX([GF2ndNorm_A,GF2ndAnom_A,GF2nd_A],WriteMax,WriteStep,header,filename)

#print('{0: .4f}\t{1: .4f}\t{2: .4f}\t{3: .4f}\t{4: .6f}\t{5: .6f}\t{6: .6f}\t{7: .6f}\t{8: .6f}\t{9: .6f}\t{10: .6f}\t{11: .6f}'\
#.format(U,DeltaS,GammaS,GammaN,P,0.0,nHF,n2nd,0.0,JCHF,JC2nd,0.0))

###########################################################
## calculate the Lambda vertex ############################
###########################################################
## use Lambda from the older method as a starting point
if not Lin:
	if chat: print('# calculating the fully static vertex at half-filling as a starting point:')
	Bubble_A = TwoParticleBubble(GFzero_A,GFzero_A,'eh')
	Uc = -1.0/sp.real(Bubble_A[Nhalf])
	if chat: print('# - Critical U: {0: .6f}'.format(Uc))
	#WriteFileX([Bubble_A],WriteMax,WriteStep,parline,'bubble'+str(GammaN)+'.dat')
	Lambda = CalculateLambda(Bubble_A,GFzero_A,GFzero_A)
	if chat: print('# - LambdaZero = {0: .8f}'.format(Lambda))
else:
	if chat: print('# Initial guess for Lambda read from input: {0: .6f}'.format(LIn))
	Lambda = LIn

## calculate new Lambda vertex
if chat: print('# - calculating Lambda vertex:')
Lambda = CalculateLambdaD(GFzero_A,GFzero_A,Lambda)
K      = KvertexD(Lambda,GFzero_A,GFzero_A)
XD     = ReBDDFDD(GFzero_A,GFzero_A,0)
if chat: print('# - - Lambda vertex:  Lambda: {0: .8f}'.format(Lambda))
if chat: print('# - - K vertex:            K: {0: .8f}'.format(K))
if chat: print('# - - aux. integral:       X: {0: .8f}'.format(XD))

Det_A = DeterminantGD(Lambda,GFzero_A,GFzero_A)
Dzero = Det_A[int((len(En_A)-1)/2)]
if chat: print('# - determinant at zero energy: {0: .8f} {1:+8f}i'.format(sp.real(Dzero),sp.imag(Dzero)))

###########################################################
## spectral self-energy ###################################
if chat: print('#\n# calculating the spectral self-energy:')
Sigma_A = SelfEnergyD2(GFzero_A,GFzero_A,Lambda,'up')
SE0int  = sp.imag(Sigma_A[int(N/2)])

## symmetrized self-energy
SigmaNorm_A = 0.5*(Sigma_A-sp.flipud(sp.conj(Sigma_A)))
SigmaAnom_A = 0.5*(Sigma_A+sp.flipud(sp.conj(Sigma_A)))
#WriteFileX([SigmaNorm_A,SigmaAnom_A,Sigma_A],WriteMax,WriteStep,parline,'SEint.dat')

## quasiparticle weight
[Z,dReSEdw] = QuasiPWeight(sp.real(SigmaNorm_A))
if chat: print('# - Z = {0: .8f}, DReSE/dw[0] = {1: .8f}, m*/m = {2: .8f}'\
.format(float(Z),float(dReSEdw),float(1.0/Z)))

###########################################################
## interacting Green function #############################
if chat: print('#\n# calculating the spectral Green function:')

## correct the static part of the self-energy to fulfill charge consistency
eqn = lambda x: Filling(GFlambda(En_A-U*(x-0.5)-Sigma_A))-x
nint = brentq(eqn,0.0,1.0)

GFint_A = GFlambda(En_A-U*(nint-0.5)-Sigma_A)
nint = Filling(GFint_A)
if chat: print('# - norm[Gint]      = {0: .8f}, nW = {1: .8f}'.format(IntDOS(GFint_A),nint))

## symmetrized Green function
GFintNorm_A = 0.5*(GFint_A-sp.flipud(sp.conj(GFint_A)))
GFintAnom_A = 0.5*(GFint_A+sp.flipud(sp.conj(GFint_A)))
nint2  = Filling(GFintNorm_A)
nuint2 = Filling(GFintAnom_A)
if chat: print('# - norm[GintNorm] = {0: .8f}, n  = {1: .8f}'.format(IntDOS(GFintNorm_A),nint2))
if chat: print('# - norm[GintAnom] = {0: .8f}, nu = {1: .8f}'.format(IntDOS(GFintAnom_A),nuint2))

JCint = JosephsonCurrent(GFintAnom_A,En_A)
if chat: print('# - Josephson current: {0: .8f}'.format(JCint))

## DoS at Fermi energy
DOSFint = -sp.imag(GFintNorm_A[int(N/2)])/sp.pi

## write the output file ##################################
if WriteGF:
	if chat: print('# Writing output file(s):')
	filename = 'gf_int_'+parfname+'.dat'
	WriteFileX([GFintNorm_A,GFintAnom_A,GFint_A],WriteMax,WriteStep,header,filename)

## write data to standard output
print('{0: .4f}\t{1: .4f}\t{2: .4f}\t{3: .4f}\t{4: .6f}\t{5: .6f}\t{6: .6f}\t{7: .6f}\t{8: .6f}\t{9: .6f}\t{10: .6f}\t{11: .6f}'\
.format(U,DeltaS,GammaS,GammaN,P,Lambda,nuHF2,nu2nd2,nuint2,JCHF,JC2nd,JCint))

if chat: print('# '+argv[0]+' DONE after {0: .2f} seconds.'.format(float(time()-t)))

## super_parquet.py end ###

