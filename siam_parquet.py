###########################################################
# SPEpy - simplified parquet equation solver for SIAM     #
# Copyright (C) 2019  Vladislav Pokorny; pokornyv@fzu.cz  #
# homepage: github.com/pokornyv/SPEpy                     #
# siam_dynamic.py - solver for SPE                        #
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
from parlib import *
from parlib2 import *

t = time()

hashes = '#'*80

## python version
ver = str(version_info[0])+'.'+str(version_info[1])+'.'+str(version_info[2])
## header for files so we store the parameters along with data
parline = '# U = {0: .4f}, Delta = {1: .4f}, ed = {2: .4f}, h = {3: .4f}, T = {4: .4f}'\
.format(U,Delta,ed,h,T)

## print the header #######################################
if chat: 
	print(hashes+'\n# generated by '+str(argv[0])+', '+str(ctime()))
	print('# python version: '+str(ver)+', SciPy version: '+str(sp.version.version))
	print('# energy axis: [{0: .5f} ..{1: .5f}], step = {2: .5f}, length = {3: 3d}'\
     .format(En_A[0],En_A[-1],dE,len(En_A)))
	print('# U = {0: .5f}, Delta = {1: .5f}, ed = {2: .5f}, h = {3: .5f}, T = {4: .5f}'\
	.format(U,Delta,ed,h,T))
	print('# Kondo temperature from Bethe ansatz: Tk ~{0: .5f}'\
	.format(float(KondoTemperature(U,Delta,ed))))
	if SCsolver == 'fixed': 
		print('# using Steffensen fixed-point algorithm to calculate Lambda vertex')
	elif SCsolver == 'root': 
		print('# using MINPACK root to calculate Lambda vertex')
	else: 
		print('# using iteration algorithm to calculate Lambda vertex, mixing parameter alpha = {0: .5f}'\
		.format(float(alpha)))

###########################################################
## inicialize the non-interacting Green function ##########
if GFtype == 'lor':
	if chat: print('# using Lorentzian non-interacting DoS')
	GFlambda = lambda x: GreensFunctionLorenz(x,Delta)
	DensityLambda = lambda x: DensityLorentz(x,Delta)
elif GFtype == 'semi':
	if chat: print('# using semielliptic non-interacting DoS')
	W = Delta 	## half-bandwidth 
	GFlambda = lambda x: GreensFunctionSemi(x,W)
	DensityLambda = lambda x: DensitySemi(x,W)
elif GFtype == 'gauss':
	if chat: print('# using Gaussian non-interacting DoS')
	GFlambda = lambda x: GreensFunctionGauss(x,Delta)
	DensityLambda = lambda x: DensityGauss(x,Delta)
elif GFtype == 'sc':
	if chat: print('# using simple cubic lattice non-interacting DoS')
	W = Delta # half-bandwidth 
	GFlambda = lambda x: GreensFunctionSC(x,W)
	print('# Error: this DoS is not yet implemented.')
	exit(1)
elif GFtype == 'sq':
	if chat: print('# using square lattice non-interacting DoS')
	W = Delta # half-bandwidth 
	izero = 1e-6    ## imaginary shift of energies to avoid poles
	GFlambda = lambda x: GreensFunctionSquare(x,izero,W)
	print('# Error: this DoS is not yet implemented.')
	exit(1)
else:
	print('# Error: DoS type "'+GFtype+'" not implemented.')
	exit(1)

## write the distributions to a file, development only
#WriteFileX([FD_A,BE_A,FB_A],1,4,parline,'dist.dat')

## using the Lambda from the older method as a starting point
if not Lin:
	if chat: print('# calculating the fully static vertex at half-filling as a starting point:')
	GFzero_A = GFlambda(En_A)
	Bubble_A = TwoParticleBubble(GFzero_A,GFzero_A,'eh')
	Lambda0 = CalculateLambda(Bubble_A,GFzero_A,GFzero_A)
	if chat: print('# - Lambda0 = {0: .8f}'.format(Lambda0))
else:
	if chat: print('# Initial guess for Lambda: L(++) = {0: .6f}, L(+-) = {1: .6f}'.format(LppIn,LmpIn))
	
########################################################
## calculate filling of the thermodynamic Green function
if chat: print('#\n# calculating the initial thermodynamic Green function:')
[nTup,nTdn] = [0.5,0.5]
[nTupOld,nTdnOld] = [1e8,1e8]

k = 1
while any([sp.fabs(nTupOld-nTup) > epsn, sp.fabs(nTdnOld-nTdn) > epsn]):
	[nTupOld,nTdnOld] = [nTup,nTdn]
	if T == 0.0:
		nup_dens = lambda x: DensityLambda(ed+U/2.0*(x+nTdn-1.0)-h) - x
		ndn_dens = lambda x: DensityLambda(ed+U/2.0*(nTup+x-1.0)+h) - x
	else:
		nup_dens = lambda x: Filling(GFlambda(En_A-ed-U/2.0*(x+nTdn-1.0)+h)) - x
		ndn_dens = lambda x: Filling(GFlambda(En_A-ed-U/2.0*(nTup+x-1.0)-h)) - x
	nTup = brentq(nup_dens,0.0,1.0,xtol = epsn)
	nTdn = brentq(ndn_dens,0.0,1.0,xtol = epsn)
	if chat: print('# - - {0: 3d}:   nUp: {1: .8f}, nDn: {2: .8f}'.format(k,nTup,nTdn))
	k += 1

## fill the Green functions
GFTup_A = GFlambda(En_A-ed-U/2.0*(nTup+nTdn-1.0)+h)
GFTdn_A = GFlambda(En_A-ed-U/2.0*(nTup+nTdn-1.0)-h)
## write non-interacting GF to a file, development only
#WriteFileX([GFTup_A,GFTdn_A],WriteMax,WriteStep,parline,'GFTzero.dat')

if chat: print('# - norm[GTup]: {0: .8f}, n[GTup]: {1: .8f}'\
.format(float(IntDOS(GFTup_A)),float(nTup)))
if chat: print('# - norm[GTdn]: {0: .8f}, n[GTdn]: {1: .8f}'\
.format(float(IntDOS(GFTdn_A)),float(nTdn)))
if chat: print('# - nT = {0: .8f}, mT = {1: .8f}'.format(float(nTup+nTdn),float(nTup-nTdn)))

###########################################################
## calculate the Lambda vertex ############################
if chat: print('#\n# calculating the Hartree-Fock self-energy:')
if Lin:
	## reading initial values from command line
	[LambdaPP,LambdaPM] = [LppIn,LmpIn]
else:
	## using the static guess
	[LambdaPP,LambdaPM] = [Lambda0,Lambda0]

[nTupOld,nTdnOld] = [1e8,1e8]
[Sigma0,Sigma1] = [U*(nTup+nTdn-1.0)/2.0,U*(nTup-nTdn)/2.0]

k = 1
while any([sp.fabs(nTupOld-nTup) > epsn, sp.fabs(nTdnOld-nTdn) > epsn]):
	if chat: print('#\n# Iteration {0: 3d}'.format(k))
	[nTupOld,nTdnOld] = [nTup,nTdn]
	## Lambda vertex
	[LambdaPP,LambdaPM] = CalculateLambdaD(GFTup_A,GFTdn_A,LambdaPP,LambdaPM)
	Kpp = KvertexD( 1,LambdaPP,LambdaPM,GFTup_A,GFTdn_A)
	Kmp = KvertexD(-1,LambdaPP,LambdaPM,GFTup_A,GFTdn_A)
	if chat: print('# - Lambda vertex: Lambda(++): {0: .8f} {1:+8f}i  Lambda(+-): {2: .8f} {3:+8f}i'\
	.format(sp.real(LambdaPP),sp.imag(LambdaPP),sp.real(LambdaPM),sp.imag(LambdaPM)))
	if chat: print('# - K vertex:           K(++): {0: .8f} {1:+8f}i       K(-+): {2: .8f} {3:+8f}i'\
	.format(sp.real(Kpp),sp.imag(Kpp),sp.real(Kmp),sp.imag(Kmp)))
	## check the integrals:
	RFDpp = ReBDDFDD( 1,GFTup_A,GFTdn_A,0)
	IFDpp = ImBDDFDD( 1,GFTup_A,GFTdn_A,0)
	RFDmp = ReBDDFDD(-1,GFTup_A,GFTdn_A,0)
	IFDmp = ImBDDFDD(-1,GFTup_A,GFTdn_A,0)
	if chat: print('# - aux. integrals:     X(++): {0: .8f} {1:+8f}i       X(-+): {2: .8f} {3:+8f}i'\
	.format(RFDpp,IFDpp,RFDmp,IFDmp))
	## HF self-energy
	[Sigma0,Sigma1] = CalculateSigmaT(LambdaPP,LambdaPM,Sigma0,Sigma1,GFlambda,DensityLambda)
	if chat: print('# - static self-energy: normal: {0: .8f} {1:+8f}i, anomalous: {2: .8f} {3:+8f}i'\
	.format(sp.real(Sigma0),sp.imag(Sigma0),sp.real(Sigma1),sp.imag(Sigma1)))
	GFTup_A = GFlambda(En_A-ed-Sigma0+(h-Sigma1))
	GFTdn_A = GFlambda(En_A-ed-Sigma0-(h-Sigma1))
	## symmetrize the Green function if possible
	if h == 0.0:
		if chat: print('# - h = 0, averaging Green functions over spin to avoid numerical errors')
		GFTup_A = sp.copy((GFTup_A+GFTdn_A)/2.0)
		GFTdn_A = sp.copy((GFTup_A+GFTdn_A)/2.0)
		Sigma1 = 0.0
	## recalculate filling and magnetization
	if any([ed!=0.0,h!=0.0]):
		if T == 0.0:
			nTup = DensityLambda(ed+Sigma0-(h-Sigma1))
			nTdn = DensityLambda(ed+Sigma0+(h-Sigma1))
		else:
			nTup = Filling(GFTup_A)
			nTdn = Filling(GFTdn_A)
	else: ## ed = 0 and h = 0
		nTup = nTdn = 0.5
	## this is to convert complex to float, the warning is just a sanity check
	if any([sp.fabs(sp.imag(nTup))>1e-6,sp.fabs(sp.imag(nTdn))>1e-6,]):
		print('# Warning: non-zero imaginary part of nT, up: {0: .8f}, dn: {1: .8f}.'\
		.format(sp.imag(nTup),sp.imag(nTdn)))
	[nTup,nTdn] = [sp.real(nTup),sp.real(nTdn)]
	## print integrals for control 
	IG0p = IntGdiff(GFTup_A,GFTdn_A)
	IG0m = IntGdiff(sp.conj(GFTup_A),sp.conj(GFTdn_A))
	if chat: print('# - integrals <Gup-Gdn>: I(+) {0: .8f} {1:+8f}i, I(-) {2: .8f} {3:+8f}i'\
	.format(sp.real(IG0p),sp.imag(IG0p),sp.real(IG0m),sp.imag(IG0m)))
	## print putput for given iteration
	if chat: 
		print('# - thermodynamic Green function filling: nTup = {0: .8f}, nTdn = {1: .8f}'.format(nTup,nTdn))
		print('# - ed = {0: .4f}, h = {1: .4f}:    nT = {2: .8f}, mT = {3: .8f}'.format(ed,h,nTup+nTdn,nTup-nTdn))
		print('{0: 3d}\t{1: .8f}\t{2: .8f}\t{3: .8f}\t{4: .8f}'.format(k,nTup,nTdn,nTup+nTdn,nTup-nTdn))
	k+=1

if chat: print('# - Calculation of the Hartree-Fock self-energy finished after {0: 3d} iterations.'.format(int(k-1)))

#print('{0: .4f}\t{1: .8f}\t{2: .8f}\t{3: .8f}\t{4: .8f}\t{5: .8f}\t{6: .8f}\t{7: .8f}\t{8: .8f}'\
#.format(T,RFDpp,IFDpp,RFDmp,IFDmp,sp.real(Kpp),sp.imag(Kpp),sp.real(Kmp),sp.imag(Kmp)))

Det_A = DeterminantGD(LambdaPP,LambdaPM,GFTup_A,GFTdn_A)
Dzero = Det_A[int((len(En_A)-1)/2)]
if chat: print('# - determinant at zero energy: {0: .8f} {1:+8f}i'.format(sp.real(Dzero),sp.imag(Dzero)))
## write the determinant to a file, for development only
#WriteFileX([GFTup_A,GFTdn_A,Det_A],WriteMax,WriteStep,parline,'DetG.dat')
if chat and h==0.0:
	print('# - thermodynamic susceptibility: {0: .8f}'.format(sp.real(SusceptibilityTherm(Dzero,GFTup_A))))

#print('{0: .4f}\t{1: .8f}\t{2: .8f}\t{3: .8f}\t{4: .8f}\t{5: .8f}\t{6: .8f}\t{7: .8f}\t{8: .8f}\t{9: .8f}'\
#.format(h,sp.real(LambdaPP),sp.imag(LambdaPP),sp.real(LambdaPM),sp.imag(LambdaPM),sp.real(Dzero),nTup,nTdn,nTup+nTdn,nTup-nTdn))
#exit()

## check the zero of the determinant, development only
#GG0 = CorrelatorGGzero(GFTup_A,GFTdn_A,1,1)
#CDD=1.0+2.0*sp.real(GG0*LambdaPP)+absC(GG0)*(absC(LambdaPP)-absC(LambdaPM))
#print("# - check D(0): {0: .8f} {1:+8f}i".format(sp.real(CDD),sp.imag(CDD)))

#print('{0: .4f}\t{1: .8f}\t{2: .8f}\t{3: .8f}\t{4: .8f}\t{5: .8f}\t{6: .8f}\t{7: .8f}\t{8: .8f}\t{9: .8f}'\
#.format(U,RFDpp,IFDpp,RFDmp,IFDmp,sp.real(Kpp),sp.imag(Kpp),sp.real(Kmp),sp.imag(Kmp),sp.real(Dzero)))

###########################################################
## spectral self-energy ###################################
if chat: print('#\n# calculating the spectral self-energy:')
SigmaUp_A = SelfEnergyD(GFTup_A,GFTdn_A,LambdaPP,LambdaPM,'up')
SigmaDn_A = SelfEnergyD(GFTup_A,GFTdn_A,LambdaPP,LambdaPM,'dn')
Sigma_A = (SigmaUp_A+SigmaDn_A)/2.0

## quasiparticle weights
[Zup,dReSEupdw] = QuasiPWeight(sp.real(SigmaUp_A))
[Zdn,dReSEdndw] = QuasiPWeight(sp.real(SigmaDn_A))
[Z,dReSEdw]     = QuasiPWeight(sp.real(Sigma_A))

if chat: print('# - Z = {0: .8f}, DReSE/dw[0] = {1: .8f}, m*/m = {2: .8f}'\
.format(float(Z),float(dReSEdw),float(1.0/Z)))

if chat and h!=0.0: 
	print('# - up spin: Z = {0: .8f}, DReSE/dw[0] = {1: .8f}, m*/m = {2: .8f}'\
	.format(float(Zup),float(dReSEupdw),float(1.0/Zup)))
	print('# - dn spin: Z = {0: .8f}, DReSE/dw[0] = {1: .8f}, m*/m = {2: .8f}'\
	.format(float(Zdn),float(dReSEdndw),float(1.0/Zdn)))

###########################################################
## interacting Green function #############################
if chat: print('#\n# calculating the spectral Green function:')
if chat: print('# - iterating the final density:')
[nUp,nDn] = [nTup,nTdn]
[nUpOld,nDnOld] = [1e8,1e8]

k = 1
while any([sp.fabs(nUpOld-nUp) > epsn, sp.fabs(nDnOld-nDn) > epsn]):
	[nUpOld,nDnOld] = [nUp,nDn]
	nup_dens = lambda x: Filling(GFlambda(En_A-ed-U/2.0*(x+nDn-1.0)+(h-Sigma1)-Sigma_A)) - x
	ndn_dens = lambda x: Filling(GFlambda(En_A-ed-U/2.0*(nUp+x-1.0)-(h-Sigma1)-Sigma_A)) - x
	nUp = brentq(nup_dens,0.0,1.0,xtol = epsn)
	nDn = brentq(ndn_dens,0.0,1.0,xtol = epsn)
	if chat: print('# - - {0: 3d}:   nUp: {1: .8f}, nDn: {2: .8f}'.format(k,nUp,nDn))
	k += 1

GFintUp_A = GFlambda(En_A-ed-U/2.0*(nUp+nDn-1.0)+(h-Sigma1)-Sigma_A)
GFintDn_A = GFlambda(En_A-ed-U/2.0*(nUp+nDn-1.0)-(h-Sigma1)-Sigma_A)

[nUp,nDn] = [Filling(GFintUp_A),Filling(GFintDn_A)]
if chat: 
	print('# - spectral Green function filling: nUp = {0: .8f}, nDn = {1: .8f}'.format(nUp,nDn))
	print('# - ed = {0: .4f}, h = {1: .4f}:    n = {2: .8f}, m = {3: .8f}'.format(ed,h,nUp+nDn,nUp-nDn))

## DoS at Fermi energy
DOSFup = -sp.imag(GFintUp_A[int(N/2)])/sp.pi
DOSFdn = -sp.imag(GFintDn_A[int(N/2)])/sp.pi

## HWHM of the spectral function
[HWHMup,DOSmaxUp,wmaxUp] = CalculateHWHM(GFintUp_A)
[HWHMdn,DOSmaxDn,wmaxDn] = CalculateHWHM(GFintDn_A)
if any([HWHMup == 0.0,HWHMdn == 0.0]) and chat: 
	print('# - Warning: HWHM cannot be calculated, setting it to zero.')
elif any([HWHMup < dE,HWHMdn < dE]): 
	print('# - Warning: HWHM smaller than energy resolution.')
if chat: print('# - spin-up: DOS[0] = {0: .8f}, maximum of DoS: {1: .8f} at w = {2: .8f}'\
.format(float(DOSFup),float(DOSmaxUp),float(wmaxUp)))
if h!=0.0 and chat:
	print('# - spin-dn: DOS[0] = {0: .8f}, maximum of DoS: {1: .8f} at w = {2: .8f}'\
	.format(float(DOSFdn),float(DOSmaxDn),float(wmaxDn)))
if chat: print('# - HWHM: spin-up: {0: .8f}, spin-dn: {1: .8f}'.format(float(HWHMup),float(HWHMdn)))

###########################################################
## write the output files #################################
if WriteGF:
	header = parline+'\n# E\t\tRe GF0\t\tIm GF0\t\tRe SE\t\tIm SE\t\tRe GF\t\tIm GF'
	filename = 'gfUp_'+str(GFtype)+'_U'+str(U)+'eps'+str(ed)+'T'+str(T)+'h'+str(h)+'.dat'
	WriteFileX([GFTup_A,SigmaUp_A,GFintUp_A],WriteMax,WriteStep,header,filename)
	if h!=0.0:	
		filename = 'gfDn_'+str(GFtype)+'_U'+str(U)+'eps'+str(ed)+'T'+str(T)+'h'+str(h)+'.dat'
		WriteFileX([GFTdn_A,SigmaDn_A,GFintDn_A],WriteMax,WriteStep,header,filename)
		filename = 'gfMag_'+str(GFtype)+'_U'+str(U)+'eps'+str(ed)+'T'+str(T)+'h'+str(h)+'.dat'
		WriteFileX([GFintUp_A,GFintDn_A,SigmaUp_A,SigmaDn_A],WriteMax,WriteStep,header,filename)


print('{0: .4f}\t{1: .4f}\t{2: .4f}\t{3: .4f}\t{4: .6f}\t{5: .6f}\t{6: .6f}\t{7: .6f}\t{8: .6f}\t{9: .6f}\t{10: .6f}\t{11: .6f}'\
.format(U,ed,T,h,sp.real(LambdaPP),sp.imag(LambdaPP),sp.real(LambdaPM),sp.imag(LambdaPM),HWHMup,Zup,DOSFup,sp.real(Dzero)))

print('{0: .4f}\t{1: .4f}\t{2: .4f}\t{3: .4f}\t{4: .6f}\t{5: .6f}\t{6: .6f}\t{7: .6f}'\
.format(U,ed,T,h,float(nTup),float(nTdn),float(nUp),float(nDn)))

if chat: print('# '+argv[0]+' DONE after {0: .2f} seconds.'.format(float(time()-t)))

## siam_dynamic.py end ###

