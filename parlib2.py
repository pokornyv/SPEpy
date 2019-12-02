###########################################################
# SPEpy - simplified parquet equation solver for SIAM     #
# Copyright (C) 2019  Vladislav Pokorny; pokornyv@fzu.cz  #
# homepage: github.com/pokornyv/SPEpy                     #
# parlib2.py - library of functions                       #
###########################################################

import scipy as sp
from scipy.integrate import simps
from scipy.fftpack import fft,ifft
from scipy.optimize import brentq,fixed_point,root
from parlib import KramersKronigFFT,Filling,TwoParticleBubble,WriteFileX
from time import time
from config_siam import *

absC = lambda z: sp.real(z)**2+sp.imag(z)**2

###########################################################
## integrals over the Green function ######################

def DeterminantGD(Lambda,Gup_A,Gdn_A):
	''' determinant '''
	Det_A = 1.0+Lambda*(GG1_A+GG2_A)
	return Det_A


def ReBDDFDD(Gup_A,Gdn_A,printint):
	''' function to calculate the sum of real parts of FD and BD integrals '''
	Int1_A = sp.imag(1.0/sp.flipud(sp.conj(Det_A)))*sp.real(Gup_A*sp.flipud(Gdn_A))
	Int2_A = sp.imag(Gup_A*sp.flipud(sp.conj(Gdn_A))/sp.flipud(sp.conj(Det_A)))
	## here we multiply big and small numbers for energies close to zero
	#RBF1_A    = sp.exp(sp.log(FB_A)+sp.log(Int1_A))
	RBF1_A    = -FB_A*Int1_A
	RBF1_A[Nhalf] =  (RBF1_A[Nhalf-1] + RBF1_A[Nhalf+1])/2.0
	RBF2_A    = -FD_A*Int2_A
	TailL2    = -0.5*RBF2_A[0]*En_A[0] ## leading-order, 1/x**3 tail correction to Int2_A
	RBF       =  (simps(RBF1_A+RBF2_A,En_A)+TailL2)/sp.pi
	if printint:
		WriteFileX([Int1_A,Int2_A,RBF1_A,RBF2_A],50.0,3,'','RBF.dat')
	print('{0: .5f}\t{1: .8f}\t{2: .8f}'\
	.format(T,simps(RBF1_A,En_A)/sp.pi,(simps(RBF2_A,En_A)+TailL2)/sp.pi),flush=True)
	#exit()
	return RBF


###########################################################
## correlators of Green functions #########################

def CorrelatorGG(G1_A,G2_A,En_A,i1,i2):
	''' <G1(x+i10)G2(x+w+i20)>, i1 and i2 are imaginary parts of arguments '''
	## zero-padding the arrays, G1 and G2 are complex functions
	FDex_A = sp.concatenate([FD_A[Nhalf:],sp.zeros(2*Nhalf+3),FD_A[:Nhalf]])
	G1ex_A = sp.concatenate([G1_A[Nhalf:],sp.zeros(2*Nhalf+3),G1_A[:Nhalf]])
	G2ex_A = sp.concatenate([G2_A[Nhalf:],sp.zeros(2*Nhalf+3),G2_A[:Nhalf]])
	if i1*i2 > 0: G1ex_A = sp.conj(G1ex_A)
	ftF1_A = fft(FDex_A*G1ex_A)
	ftF2_A = fft(G2ex_A)
	if i2 > 0: ftF1_A = sp.conj(ftF1_A)
	else:      ftF2_A = sp.conj(ftF2_A)
	GG_A = ifft(ftF1_A*ftF2_A*dE)
	## undo the zero padding
	GG_A = sp.concatenate([GG_A[3*Nhalf+4:],GG_A[:Nhalf+1]])
	TailL = -sp.real(G1_A)[0]*sp.real(G2_A)[0]*En_A[0] ## leading tail correction
	return -(GG_A+TailL)/sp.pi


def CorrelatorImGGzero(G1_A,G2_A,i1,i2):
	''' <G1(x+i10)G2(x+i20)>, w=0 element of the CorrelatorGG '''
	if i1 < 0: G1_A = sp.conj(G1_A)
	if i2 < 0: G2_A = sp.conj(G2_A)
	Int_A = FD_A*sp.imag(G1_A*G2_A)
	Int = simps(Int_A,En_A)
	#print(Int_A[ 0],Int_A[-1])
	#TailL = -sp.real(Int_A[ 0])*En_A[ 0]
	#TailR =  sp.real(Int_A[-1])*En_A[-1]
	#return -(Int+TailL+TailR)/sp.pi
	return -Int/sp.pi


def SusceptibilitySpecD(L,chiT,GFint_A):
	''' susceptibility '''
	Int = simps(FD_A*sp.imag(GFint_A**2),En_A)/sp.pi
	## what about tail???
	return (2.0+L*chiT)*Int


###########################################################
## two-particle vertex ####################################

def KvertexD(Lambda,Gup_A,Gdn_A):
	''' reducible K vertex Eq. (39ab) '''
	K = -Lambda**2*CorrelatorImGGzero(Gdn_A,Gup_A,1,1)
	return K


def LambdaVertexD(Gup_A,Gdn_A,Lambda):
	''' calculates the Lambda vertex for given i '''
	global GG1_A,GG2_A,Det_A
	Det_A  = DeterminantGD(Lambda,Gup_A,Gdn_A)
	K      = KvertexD(Lambda,Gup_A,Gdn_A)
#	GFn_A = 0.5*(Gup_A-sp.flipud(sp.conj(Gup_A)))
#	XD     = ReBDDFDD(GFn_A,GFn_A,0)
	XD     = ReBDDFDD(Gup_A,Gdn_A,0)
	Lambda = U/(1.0+K*XD)
	print('#  Lambda: {0: .8f} {1:+8f}i'.format(float(sp.real(Lambda)),float(sp.imag(Lambda))))
	print('#  X:      {0: .8f}'.format(XD))
	print('#  K:      {0: .8f} {1:+8f}i'.format(float(sp.real(K)),float(sp.imag(K))))
	return Lambda


def CalculateLambdaD(Gup_A,Gdn_A,Lambda):
	''' main solver for the Lambda vertex '''
	global GG1_A,GG2_A,alpha
	Lold = 1e8
	if SCsolver == 'iter': diffL = 1e8
	## correlators don't change with Lambda iterations
	t = time()
	if chat: print('# - - calculating correlators... ',end='',flush=True)
	GG1_A = CorrelatorGG(sp.imag(Gup_A),Gdn_A,En_A, 1, 1)
	GG2_A = CorrelatorGG(sp.imag(Gdn_A),Gup_A,En_A, 1,-1)
	if chat: print(' done in {0: .2f} seconds.'.format(time()-t))
	#from parlib import WriteFileX
	#WriteFileX([GG1_A,GG2_A],100.0,3,'','GGcorr.dat')
	k = 1
	while sp.fabs(sp.real(Lambda-Lold))>epsl:
		Lold = Lambda
		if SCsolver == 'fixed':
			Eqn = lambda x: LambdaVertexD(Gup_A,Gdn_A,x)
			try:
				Lambda = fixed_point(Eqn,Lambda,xtol=epsl)
			except RuntimeError:
				print("# - Error: CalculateLambdaD: No convergence in fixed-point algorithm.")
				print("# - Switch SCsolver to 'iter' or 'root' in siam.in and try again.")
				exit(1)
		elif SCsolver == 'brentq':
			Uc = -1.0/TwoParticleBubble(Gup_A,Gdn_A,'eh')[Nhalf]
			print(Uc)
			Eqn = lambda x: LambdaVertexD(Gup_A,Gdn_A,x)-x
			try:
				Lambda = brentq(Eqn,0.0,Uc-1e-12)
				if chat: print("# - - convergence check: {0: .5e}".format(Eqn(Lambda)))
			except RuntimeError:
				print("# - Error: CalculateLambdaD: No convergence in Brent algorithm.")
				print("# - Switch SCsolver to 'iter' or 'root' in siam.in and try again.")
				exit(1)
			break ## we don't need the outer loop here
		elif SCsolver == 'iter':
			print('# alpha: {0: .6f}'.format(alpha))
			diffLold = diffL
			Eqn = lambda x: LambdaVertexD(Gup_A,Gdn_A,x)
			Lambda = alpha*Eqn(Lambda) + (1.0-alpha)*Lold
			diffL = sp.fabs(sp.real(Lambda-Lold))
			if diffL<diffLold: alpha = sp.amin([1.05*alpha,1.0])
		elif SCsolver == 'root':
			## originally implemented for two complex Lambdas as 4-dimensional problem
			## now just a check... sad story
			ErrConv = 0
			eqn = lambda x: LambdaVertexD(Gup_A,Gdn_A,x)-x
			sol = root(eqn,[Lambda],method='hybr')
			if sol.success:
				Lambda = sol.x[0]
				if chat: print("# - - number of function calls: {0: 3d}".format(sol.nfev))
				if chat: print("# - - convergence check: {0: .5e}".format(sol.fun[0]))
				for x in sol.fun:
					if sp.fabs(x)>epsl:
						print('# - - Warning: CalculateLambdaD: Convergence criteria for Lambda not satisfied!')
						ErrConv = 1
				#print(sol.status) # 1 - gtol satisfied, 2 - ftol satisfied
			else:
				print("# - - Error: CalculateLambdaD: no solution by MINPACK root. Message from root:")
				print("# - - "+sol.message)
				exit(1)
			if ErrConv:
				print("# - - Error: CalculateLambdaD: no convergence in MINPACK root routine.")
				print("# - - Switch SCsolver to 'iter' or 'fixed' in siam.in and try again.")
				exit(1)
			break ## we don't need the outer loop here
		else:
			print('# - - Error: CalculateLambdaD: Unknown SCsolver')
			exit(1)
		if chat: print('# - - iter. {0: 3d}: Lambda: {1: .8f}'.format(k,sp.real(Lambda)))
		if k > 1000:
			print('# - - Error: CalculateLambdaD: No convergence after 1000 iterations. Exit.')
			exit(1)
		k += 1
	return Lambda


###########################################################
## static self-energy #####################################

def VecSigmaT(Sigma0in,Sigma1in,Lambda,GFlambda,DLambda):
	''' calculates normal and anomalous static self-energy, returns differences '''
	#Sigma1in = RSigma1in+1.0j*ISigma1in
	Gup_A = GFlambda(En_A-ed-Sigma0in+(h-Sigma1in))
	Gdn_A = GFlambda(En_A-ed-Sigma0in-(h-Sigma1in))
	if T == 0.0:
		nTup = sp.real(DLambda(ed+Sigma0in-(h-Sigma1in)))
		nTdn = sp.real(DLambda(ed+Sigma0in+(h-Sigma1in)))
	else:
		nTup = Filling(Gup_A)
		nTdn = Filling(Gdn_A)
	Sigma0 = U*(nTup+nTdn-1.0)/2.0
	Sigma1 = -Lambda*(nTup-nTdn)/2.0
	return [Sigma0-Sigma0in,Sigma1-Sigma1in]


def CalculateSigmaT(Lambda,S0,S1,GFlambda,DLambda):
	''' solver for the static self-energy '''
	eqn = lambda x: VecSigmaT(x[0],x[1],Lambda,GFlambda,DLambda)
	#sol = root(eqn,[S0,sp.real(S1),sp.imag(S1)],method='lm')	
	sol = root(eqn,[S0,S1],method='hybr')
	if sol.success:
		[Sigma0,Sigma1] = [sol.x[0],sol.x[1]]
		if chat: print("# - - number of function calls: {0: 3d}".format(sol.nfev))
		if chat: print("# - - convergence check: {0: .5e}, {1: .5e}".format(sol.fun[0],sol.fun[1]))
	else:
		print("# - Error: CalculateSigmaT: no solution by MINPACK root. Message from root:")
		print("# - - "+sol.message)
		exit(1)	
	return [Sigma0,Sigma1]


###########################################################
## dynamic self-energy ####################################

def CorrelatorsSE(Gup_A,Gdn_A,i1,i2):
	''' correlators to Theta function, updated '''
	## zero-padding the arrays, G1 and G2 are complex functions
	FDex_A   = sp.concatenate([FD_A[Nhalf:], sp.zeros(2*Nhalf+3), FD_A[:Nhalf]])
	Fup_A    = sp.concatenate([Gup_A[Nhalf:],sp.zeros(2*Nhalf+3),Gup_A[:Nhalf]])
	Fdn_A    = sp.concatenate([Gdn_A[Nhalf:],sp.zeros(2*Nhalf+3),Gdn_A[:Nhalf]])
	ftIGG1_A = fft(FDex_A*sp.imag(Fdn_A))*sp.conj(fft(Fup_A))*dE
	ftGG2_A  = sp.conj(fft(FDex_A*sp.conj(Fup_A)))*fft(Fdn_A)*dE
	ftGG3_A  = sp.conj(fft(FDex_A*Fup_A))*fft(Fdn_A)*dE
	IGGs1_A  = -ifft(ftIGG1_A)/sp.pi
	GGs2_A   = -ifft(ftGG2_A)/(2.0j*sp.pi)
	GGs3_A   = -ifft(ftGG3_A)/(2.0j*sp.pi)
	## undo the zero padding
	IGGs1_A = sp.concatenate([IGGs1_A[3*Nhalf+4:],IGGs1_A[:Nhalf+1]])
	GGs2_A  = sp.concatenate([ GGs2_A[3*Nhalf+4:], GGs2_A[:Nhalf+1]])
	GGs3_A  = sp.concatenate([ GGs3_A[3*Nhalf+4:], GGs3_A[:Nhalf+1]])
	return [IGGs1_A,GGs2_A,GGs3_A]


def BubbleD(G2_A,G1_A,Lambda,spin):
	''' auxiliary function (2P bubble) to calculate spectral self-energy '''
	sGG1_A = CorrelatorGG(sp.imag(G2_A),G1_A,En_A, 1, 1)
	sGG2_A = CorrelatorGG(sp.imag(G1_A),G2_A,En_A, 1,-1)
	return Lambda*(sGG1_A+sGG2_A)


def SelfEnergyD(Gup_A,Gdn_A,Lambda,spin):
	''' dynamic self-energy, uses Kramers-Kronig relations to calculate the real part '''
	if spin == 'up': 
		BubbleD_A = BubbleD(Gup_A,Gdn_A,Lambda,spin)
		GF_A = sp.copy(Gdn_A) 
		Det_A = DeterminantGD(Lambda,Gup_A,Gdn_A)
	else: ## spin='dn'
		BubbleD_A = BubbleD(Gdn_A,Gup_A,Lambda,spin)
		GF_A = sp.copy(Gup_A) 
		Det_A = sp.flipud(sp.conj(DeterminantGD(Lambda,Gup_A,Gdn_A)))
	Kernel_A = U*BubbleD_A/Det_A
	## zero-padding the arrays
	FDex_A     = sp.concatenate([FD_A[Nhalf:],sp.zeros(2*N+3),FD_A[:Nhalf]])
	BEex_A     = sp.concatenate([BE_A[Nhalf:],sp.zeros(2*N+3),BE_A[:Nhalf]])
	ImGF_A     = sp.concatenate([sp.imag(GF_A[Nhalf:]),sp.zeros(2*Nhalf+3),sp.imag(GF_A[:Nhalf])])
	ImKernel_A = sp.concatenate([sp.imag(Kernel_A[Nhalf:]),sp.zeros(2*Nhalf+3),sp.imag(Kernel_A[:Nhalf])])
	## performing the convolution
	ftImSE1_A  = -sp.conj(fft(BEex_A*ImKernel_A))*fft(ImGF_A)*dE
	ftImSE2_A  = -fft(FDex_A*ImGF_A)*sp.conj(fft(ImKernel_A))*dE
	ImSE_A     = sp.real(ifft(ftImSE1_A+ftImSE2_A))/sp.pi
	ImSE_A     = sp.concatenate([ImSE_A[3*Nhalf+4:],ImSE_A[:Nhalf+1]])
	Sigma_A    = KramersKronigFFT(ImSE_A) + 1.0j*ImSE_A
	return Sigma_A


def SelfEnergyD2(Gup_A,Gdn_A,Lambda,spin):
	''' dynamic self-energy, calculates the complex function from FFT '''
	if spin == 'up': 
		BubbleD_A = BubbleD(Gup_A,Gdn_A,Lambda,spin)
		GF_A = sp.copy(Gdn_A) 
		Det_A = DeterminantGD(Lambda,Gup_A,Gdn_A)
	else: ## spin='dn'
		BubbleD_A = BubbleD(Gdn_A,Gup_A,Lambda,spin)
		GF_A = sp.copy(Gup_A) 
		Det_A = sp.flipud(sp.conj(DeterminantGD(Lambda,Gup_A,Gdn_A)))
	Kernel_A = U*BubbleD_A/Det_A
	## zero-padding the arrays
	FDex_A     = sp.concatenate([FD_A[Nhalf:],sp.zeros(2*Nhalf+3),FD_A[:Nhalf]])
	BEex_A     = sp.concatenate([BE_A[Nhalf:],sp.zeros(2*Nhalf+3),BE_A[:Nhalf]])
	GFex_A     = sp.concatenate([GF_A[Nhalf:],sp.zeros(2*Nhalf+3),GF_A[:Nhalf]])
	Kernelex_A = sp.concatenate([Kernel_A[Nhalf:],sp.zeros(2*Nhalf+3),Kernel_A[:Nhalf]])
	## performing the convolution
	ftSE1_A  = -sp.conj(fft(BEex_A*sp.imag(Kernelex_A)))*fft(GFex_A)*dE
	ftSE2_A  = +fft(FDex_A*sp.imag(GFex_A))*sp.conj(fft(Kernelex_A))*dE
	SE_A     = ifft(ftSE1_A+ftSE2_A)/sp.pi
	SE_A     = sp.concatenate([SE_A[3*Nhalf+4:],SE_A[:Nhalf+1]])
	return SE_A


def SelfEnergyD_sc(Gup_A,Gdn_A,GTup_A,GTdn_A,Lambda,spin):
	''' dynamic self-energy, calculates the complex function from FFT '''
	if spin == 'up': 
		BubbleD_A = BubbleD(GTup_A,GTdn_A,Lambda,spin)
		GF_A = sp.copy(Gdn_A) 
		Det_A = DeterminantGD(Lambda,GTup_A,GTdn_A)
	else: ## spin='dn'
		BubbleD_A = BubbleD(GTdn_A,GTup_A,Lambda,spin)
		GF_A = sp.copy(Gup_A) 
		Det_A = sp.flipud(sp.conj(DeterminantGD(Lambda,GTup_A,GTdn_A)))
	Kernel_A = U*BubbleD_A/Det_A
	## zero-padding the arrays
	FDex_A     = sp.concatenate([FD_A[Nhalf:],sp.zeros(2*Nhalf+3),FD_A[:Nhalf]])
	BEex_A     = sp.concatenate([BE_A[Nhalf:],sp.zeros(2*Nhalf+3),BE_A[:Nhalf]])
	GFex_A     = sp.concatenate([GF_A[Nhalf:],sp.zeros(2*Nhalf+3),GF_A[:Nhalf]])
	Kernelex_A = sp.concatenate([Kernel_A[Nhalf:],sp.zeros(2*Nhalf+3),Kernel_A[:Nhalf]])
	## performing the convolution
	ftSE1_A  = -sp.conj(fft(BEex_A*sp.imag(Kernelex_A)))*fft(GFex_A)*dE
	ftSE2_A  = +fft(FDex_A*sp.imag(GFex_A))*sp.conj(fft(Kernelex_A))*dE
	SE_A     = ifft(ftSE1_A+ftSE2_A)/sp.pi
	SE_A     = sp.concatenate([SE_A[3*Nhalf+4:],SE_A[:Nhalf+1]])
	return SE_A

## parlib2.py end ###

