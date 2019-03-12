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
from parlib import KramersKronigFFT,Filling
from time import time
from config_siam import *

absC = lambda z: sp.real(z)**2+sp.imag(z)**2

###########################################################
## integrals over the Green function ######################

def DeterminantGD(Lpp,Lpm,Gup_A,Gdn_A):
	''' determinant '''
	Det_A = 1.0+GG1_A*Lpp-(GG2_A-GG3_A)*Lpm-GG4_A*sp.conj(Lpp)-GG1_A*GG4_A*(absC(Lpp)-absC(Lpm))
	return Det_A


def ReBDDFDD(i1,Gup_A,Gdn_A,printint):
	''' function to calculate the sum of real parts of FD and BD integrals '''
	N = int((len(En_A)-1)/2)
	if i1 == 1: 
		Int1_A = sp.imag(1.0/sp.flipud(sp.conj(Det_A)))*sp.real(Gup_A*sp.flipud(Gdn_A))
	elif i1 == -1:
		Int1_A = sp.imag(1.0/sp.flipud(sp.conj(Det_A)))*sp.real(Gup_A*sp.flipud(sp.conj(Gdn_A)))
	Int2_A = sp.imag(Gup_A*sp.flipud(sp.conj(Gdn_A))/sp.flipud(sp.conj(Det_A)))
	## here we multiply big and small numbers for energies close to zero
	#RBF1_A    = sp.exp(sp.log(FB_A)+sp.log(Int1_A))
	RBF1_A    =  FB_A*Int1_A
	RBF1_A[N] =  (RBF1_A[N-1] + RBF1_A[N+1])/2.0
	RBF2_A    = -FD_A*Int2_A
	TailL2    = -0.5*RBF2_A[0]*En_A[0] ## leading-order, 1/x**3 tail correction to Int2_A
	RBF       =  (simps(RBF1_A+RBF2_A,En_A)+TailL2)/sp.pi
	#print(' Re(BDD+FDD): {0: .8f} ({1: 2d},{2: 2d})'.format(float(RBF),i1))
	if printint:
		from parlib import WriteFileX
		WriteFileX([Int1_A,Int2_A,RBF1_A,RBF2_A],100.0,4,'','RBF'+str(i1)+'.dat')
	return RBF


def ImBDDFDD(i1,Gup_A,Gdn_A,printint):
	''' function to calculate the sum of imaginary parts of FD and BD integrals '''
	if i1 == 1:
		Int_A = sp.imag(1.0/sp.flipud(sp.conj(Det_A)))*sp.imag(Gup_A*sp.flipud(Gdn_A))
	elif i1 == -1:
		Int_A = sp.imag(1.0/sp.flipud(sp.conj(Det_A)))*sp.imag(Gup_A*sp.flipud(sp.conj(Gdn_A)))
	IBF = simps(FB_A*Int_A,En_A)/sp.pi
	#print(' Im(BDD+FDD): {0: .8f} ({1: 2d},{2: 2d})'.format(float(IBF),i1,i2))
	if printint:
		from parlib import WriteFileX
		WriteFileX([Int_A,FB_A*Int_A],1.0,4,'','IBF'+str(i1)+'.dat')
	return IBF


###########################################################
## correlators of Green functions #########################

def CorrelatorGG(G1_A,G2_A,En_A,i1,i2):
	''' <G1(x+i10)G2(x+w+i20)> 
     i1 and i2 are imaginary parts of arguments '''
	N = int((len(En_A)-1)/2)
	## zero-padding the arrays, G1 and G2 are complex functions
	FDex_A = sp.concatenate([FD_A[N:],sp.zeros(2*N+3),FD_A[:N]])
	G1ex_A = sp.concatenate([G1_A[N:],sp.zeros(2*N+3),G1_A[:N]])
	G2ex_A = sp.concatenate([G2_A[N:],sp.zeros(2*N+3),G2_A[:N]])
	if i1*i2 > 0: G1ex_A = sp.conj(G1ex_A)
	ftF1_A = fft(FDex_A*G1ex_A)
	ftF2_A = fft(G2ex_A)
	if i2 > 0: ftF1_A = sp.conj(ftF1_A)
	else:      ftF2_A = sp.conj(ftF2_A)
	GG_A = ifft(ftF1_A*ftF2_A*dE)
	## undo the zero padding
	GG_A = sp.concatenate([GG_A[3*N+4:],GG_A[:N+1]])
	TailL = -sp.real(G1_A)[0]*sp.real(G2_A)[0]*En_A[0] ## leading tail correction
	return -(GG_A+TailL)/(2.0j*sp.pi)


def CorrelatorGGzero(G1_A,G2_A,i1,i2):
	''' <G1(x+i10)G2(x+i20)>, w=0 element of the CorrelatorGG '''
	if i1 < 0: G1_A = sp.conj(G1_A)
	if i2 < 0: G2_A = sp.conj(G2_A)
	Int_A = FD_A*G1_A*G2_A
	Int = simps(Int_A,En_A)
	TailL = -sp.real(Int_A[ 0])*En_A[ 0]-0.5j*sp.imag(Int_A[ 0])*En_A[ 0]
	TailR =  sp.real(Int_A[-1])*En_A[-1]+0.5j*sp.imag(Int_A[-1])*En_A[-1]
	return -(Int+TailL+TailR)/(2.0j*sp.pi)


def IntGdiff(Gup_A,Gdn_A):
	''' integral of a Green function difference to anomalous HF self-energy '''
	Int_A = FD_A*(Gup_A-Gdn_A)
	Int = simps(Int_A,En_A)
	TailL = 0.0	## tails cancel out in the difference
	TailR = 0.0
	return -(Int+TailL+TailR)/(2.0j*sp.pi)


###########################################################
## two-particle vertex ####################################

def KvertexD(i1,Lpp,Lpm,Gup_A,Gdn_A):
	''' reducible K vertex Eq. (39ab) '''
	GG = CorrelatorGGzero(Gdn_A,Gup_A,1,1)
	#print('# GG0: {0: .8f} {1:+8f}i'.format(sp.real(GG0),sp.imag(GG)))
	if i1 == 1: ## K(+,+)
		K = -Lpp**2*GG-absC(Lpm)*sp.conj(GG)-Lpp*(absC(Lpp)-absC(Lpm))*absC(GG)
	else:       ## K(-,+)
		#K = -Lpm*(Lpp*GG+sp.conj(Lpp)*sp.conj(GG)+(absC(Lpp)-absC(Lpm))*absC(GG))
		K = -Lpm*(2.0*sp.real(Lpp*GG)+(absC(Lpp)-absC(Lpm))*absC(GG))
	return K


def LambdaVertexD(i2,Gup_A,Gdn_A,Lpp,Lpm):
	''' calculates the Lambda vertex for given i1,i2 '''
	global GG1_A,GG2_A,GG3_A,GG4_A,Det_A
	Det_A  = DeterminantGD(Lpp,Lpm,Gup_A,Gdn_A)
	K      = KvertexD(i2,Lpp,Lpm,Gup_A,Gdn_A)
	RXD    = ReBDDFDD(i2,Gup_A,Gdn_A,0)
	IXD    = ImBDDFDD(i2,Gup_A,Gdn_A,0)
	Lambda = U/(1.0+K*(RXD+1.0j*IXD))
	#print('{0: 2d} {1: 2d}\t{2: .8f} {3:+8f}i'.format(i1,i2,sp.real(Lambda),sp.imag(Lambda)))
	#print('{0: 2d} {1: 2d}\t{2: .8f} {3:+8f}i'.format(i1,i2,RFD,IFD))
	#print('{0: 2d} {1: 2d}\t{2: .8f} {3:+8f}i'.format(i1,i2,sp.real(K),sp.imag(K)))
	return Lambda


def VecLambdaD(Gup_A,Gdn_A,LppR,LppI,LpmR,LpmI):
	''' calculates both Lambda vertices L(++), L(+-), returns differences '''
	Lpp2 = LambdaVertexD( 1,Gup_A,Gdn_A,LppR+1.0j*LppI,LpmR+1.0j*LpmI)
	Lpm2 = LambdaVertexD(-1,Gup_A,Gdn_A,LppR+1.0j*LppI,LpmR+1.0j*LpmI)
	return [sp.real(Lpp2)-LppR,sp.imag(Lpp2)-LppI,sp.real(Lpm2)-LpmR,sp.imag(Lpm2)-LpmI]


def CalculateLambdaD(Gup_A,Gdn_A,Lpp,Lpm):
	''' main solver for the Lambda vertex '''
	global GG1_A,GG2_A,GG3_A,GG4_A,alpha
	[LppOld,LpmOld] = [1e8,1e8]
	if SCsolver == 'iter': [diffpp,diffpm] = [1e8,1e8]
	## correlators don't change with Lambda iterations
	t = time()
	if chat: print('# - calculating correlators... ',end='',flush=True)
	GG1_A = CorrelatorGG(Gup_A,Gdn_A,En_A, 1, 1)
	GG2_A = CorrelatorGG(Gup_A,Gdn_A,En_A,-1, 1)
	GG3_A = CorrelatorGG(Gdn_A,Gup_A,En_A, 1,-1)
	GG4_A = CorrelatorGG(Gdn_A,Gup_A,En_A,-1,-1)
	if chat: print(' done in {0: .2f} seconds.'.format(time()-t))
	#Np = int((len(En_A)-1)/2)
	#print('# corr. GG0 {0: .8f} {1:+8f}i'.format(sp.real(GG2_A[Np]),sp.imag(GG2_A[Np])))
	#print(CorrelatorGGzero(Gup_A,Gdn_A,-1,1))
	#exit()
	#from parlib import WriteFileX
	#WriteFileX([GG1_A,GG2_A,GG3_A,GG4_A],100.0,3,'','GGcorr.dat')
	k = 1
	while any([sp.fabs(sp.real(Lpp-LppOld))>epsl,sp.fabs(sp.real(Lpm-LpmOld))>epsl]):
		[LppOld,LpmOld] = [Lpp,Lpm]
		if SCsolver == 'fixed':
			Eqnpp = lambda x: LambdaVertexD( 1,Gup_A,Gdn_A,x,Lpm)
			Eqnpm = lambda x: LambdaVertexD(-1,Gup_A,Gdn_A,Lpp,x)
			try:
				Lpp = fixed_point(Eqnpp,Lpp,xtol=epsl)
				Lpm = fixed_point(Eqnpm,Lpm,xtol=epsl)
			except RuntimeError:
				print("# - CalculateLambdaD: No convergence in fixed-point algorithm.")
				print("# - Switch SCsolver to 'iter' or 'root' in siam.in and try again.")
				exit(1)
		elif SCsolver == 'iter':
			print('# alpha: {0: .6f}'.format(alpha))
			[diffppOld,diffpmOld] = [diffpp,diffpm]
			Eqnpp = lambda x: LambdaVertexD( 1,Gup_A,Gdn_A,x,Lpm)
			Eqnpm = lambda x: LambdaVertexD(-1,Gup_A,Gdn_A,Lpp,x)
			Lpp = alpha*Eqnpp(Lpp) + (1.0-alpha)*LppOld
			Lpm = alpha*Eqnpm(Lpm) + (1.0-alpha)*LpmOld
			diffpp = sp.fabs(sp.real(Lpp-LppOld))
			diffpm = sp.fabs(sp.real(Lpm-LpmOld))
			if all([diffpp<diffppOld,diffpm<diffpmOld]): alpha = sp.amin([1.05*alpha,1.0])
		elif SCsolver == 'root':
			## implemented for complex Lambdas as 4-dimensional problem
			eqn = lambda x: VecLambdaD(Gup_A,Gdn_A,x[0],x[1],x[2],x[3])
			sol = root(eqn,[sp.real(Lpp),sp.imag(Lpp),sp.real(Lpm),sp.imag(Lpm)],method='lm')
			[Lpp,Lpm] = [sol.x[0]+1.0j*sol.x[1],sol.x[2]+1.0j*sol.x[3]]
			break ## we don't need the outer loop here
		else:
			print('# - CalculateLambdaD: Unknown SCsolver')
			exit(1)
		if chat: print('# - - iter. {0: 3d}: Lambda(++): {1: .8f} {2:+8f}i  Lambda(+-): {3: .8f} {4:+8f}i'\
		.format(k,sp.real(Lpp),sp.imag(Lpp),sp.real(Lpm),sp.imag(Lpm)))
		if k > 1000:
			print('# - CalculateLambdaD: No convergence after 1000 iterations. Exit.')
			exit(1)
		k += 1
		print([Lpp,Lpm])
	return [Lpp,Lpm]


###########################################################
## static self-energy #####################################

def VecSigmaT(Sigma0in,RSigma1in,ISigma1in,LSpp,LSpm,GFlambda,DLambda):
	''' calculates normal and anomalous static self-energy, returns differences '''
	Sigma1in = RSigma1in+1.0j*ISigma1in
	Gup_A = GFlambda(En_A-ed-Sigma0in+(h-Sigma1in))
	Gdn_A = GFlambda(En_A-ed-Sigma0in-(h-Sigma1in))
	if T == 0.0:
		nTup = sp.real(DLambda(ed+Sigma0in-(h-Sigma1in)))
		nTdn = sp.real(DLambda(ed+Sigma0in+(h-Sigma1in)))
		Sigma1 = -0.5*LSpp*(nTup-nTdn)
	else:
		nTup = Filling(Gup_A)
		nTdn = Filling(Gdn_A)
		IG0p = IntGdiff(Gup_A,Gdn_A)
		IG0m = IntGdiff(sp.conj(Gup_A),sp.conj(Gdn_A))
		Sigma1 = -0.5*(LSpp*IG0p-LSpm*IG0m)
	Sigma0 = U*(nTup+nTdn-1.0)/2.0
	return [Sigma0-Sigma0in,sp.real(Sigma1)-RSigma1in,sp.imag(Sigma1)-ISigma1in]


def CalculateSigmaT(Lpp,Lpm,S0,S1,GFlambda,DLambda):
	''' solver for the static self-energy '''
	LSymmPP = Lpp
	LSymmPM = sp.real(Lpm)	## 0.5*(LambdaPM+sp.conj(LambdaPM))
	eqn = lambda x: VecSigmaT(x[0],x[1],x[2],LSymmPP,LSymmPM,GFlambda,DLambda)
	sol = root(eqn,[S0,sp.real(S1),sp.imag(S1)],method='lm')
	[Sigma0,Sigma1] = [sol.x[0],sol.x[1]+1.0j*sol.x[2]]
	return [Sigma0,Sigma1]


###########################################################
## dynamic self-energy ####################################

def CorrelatorsSE(Gup_A,Gdn_A,i1,i2):
	''' correlators to Theta function, updated '''
	N = int((len(En_A)-1)/2)
	## zero-padding the arrays, G1 and G2 are complex functions
	FDex_A   = sp.concatenate([FD_A[N:], sp.zeros(2*N+3), FD_A[:N]])
	Fup_A    = sp.concatenate([Gup_A[N:],sp.zeros(2*N+3),Gup_A[:N]])
	Fdn_A    = sp.concatenate([Gdn_A[N:],sp.zeros(2*N+3),Gdn_A[:N]])
	ftIGG1_A = fft(FDex_A*sp.imag(Fdn_A))*sp.conj(fft(Fup_A))*dE
	ftGG2_A  = sp.conj(fft(FDex_A*sp.conj(Fup_A)))*fft(Fdn_A)*dE
	ftGG3_A  = sp.conj(fft(FDex_A*Fup_A))*fft(Fdn_A)*dE
	IGGs1_A  = -ifft(ftIGG1_A)/sp.pi
	GGs2_A   = -ifft(ftGG2_A)/(2.0j*sp.pi)
	GGs3_A   = -ifft(ftGG3_A)/(2.0j*sp.pi)
	## undo the zero padding
	IGGs1_A = sp.concatenate([IGGs1_A[3*N+4:],IGGs1_A[:N+1]])
	GGs2_A  = sp.concatenate([ GGs2_A[3*N+4:], GGs2_A[:N+1]])
	GGs3_A  = sp.concatenate([ GGs3_A[3*N+4:], GGs3_A[:N+1]])
	return [IGGs1_A,GGs2_A,GGs3_A]


def Theta(G1_A,G2_A,Lpp,Lmp,spin):
	''' auxiliary function to calculate spectral self-energy '''
	GG0 = CorrelatorGGzero(G1_A,G2_A,1,1)
	#print('# GG0: {0: .8f} {1:+8f}i'.format(sp.real(GG0),sp.imag(GG)))
	gmp = Lmp if spin == 'up' else sp.conj(Lmp)	## gamma(-,+)
	gpp = Lpp+(absC(Lpp)-absC(Lmp))*sp.conj(GG0)	## gamma(+,+)
	[IGGs1_A,GGs2_A,GGs3_A] = CorrelatorsSE(G1_A,G2_A,1,1)
	#from parlib import WriteFileX
	#WriteFileX([IGGs1_A,GGs2_A,GGs3_A],100,3,'','ThetaGG.dat')
	Theta_A = gmp*IGGs1_A+gpp*GGs2_A-gmp*GGs3_A
	return Theta_A


def SelfEnergyD(Gup_A,Gdn_A,Lpp,Lmp,spin):
	''' dynamic self-energy, uses Kramers-Kronig relations to calculate the real part '''
	global GG1_A,GG2_A,GG3_A,GG4_A
	#GG1_A = CorrelatorGG(Gup_A,Gdn_A,En_A, 1, 1)
	#GG2_A = CorrelatorGG(Gup_A,Gdn_A,En_A,-1, 1)
	#GG3_A = CorrelatorGG(Gup_A,Gdn_A,En_A, 1,-1)
	#GG4_A = CorrelatorGG(Gup_A,Gdn_A,En_A,-1,-1)
	N = int((len(En_A)-1)/2)
	if spin == 'up': 
		Theta_A = Theta(Gup_A,Gdn_A,Lpp,Lmp,spin)
		GF_A = sp.copy(Gdn_A) 
		Det_A = DeterminantGD(Lpp,Lmp,Gup_A,Gdn_A)
	else: ## spin='dn'
		Theta_A = Theta(Gdn_A,Gup_A,Lpp,Lmp,spin)
		GF_A = sp.copy(Gup_A) 
		Det_A = sp.flipud(sp.conj(DeterminantGD(Lpp,Lmp,Gup_A,Gdn_A)))
	Kernel_A = U*Theta_A/Det_A
	## zero-padding the arrays
	FDex_A     = sp.concatenate([FD_A[N:],sp.zeros(2*N+3),FD_A[:N]])
	BEex_A     = sp.concatenate([BE_A[N:],sp.zeros(2*N+3),BE_A[:N]])
	ImGF_A     = sp.concatenate([sp.imag(GF_A[N:]),sp.zeros(2*N+3),sp.imag(GF_A[:N])])
	ImKernel_A = sp.concatenate([sp.imag(Kernel_A[N:]),sp.zeros(2*N+3),sp.imag(Kernel_A[:N])])
	## performing the convolution
	ftImSE1_A  = -sp.conj(fft(BEex_A*ImKernel_A))*fft(ImGF_A)*dE
	ftImSE2_A  = -fft(FDex_A*ImGF_A)*sp.conj(fft(ImKernel_A))*dE
	ImSE_A     = sp.real(ifft(ftImSE1_A+ftImSE2_A))/sp.pi
	ImSE_A     = sp.concatenate([ImSE_A[3*N+4:],ImSE_A[:N+1]])
	Sigma_A    = KramersKronigFFT(ImSE_A) + 1.0j*ImSE_A
	return Sigma_A


def SelfEnergyD2(Gup_A,Gdn_A,Lpp,Lmp,spin):
	''' dynamic self-energy, calculates the complex function from FFT '''
	global GG1_A,GG2_A,GG3_A,GG4_A
	N = int((len(En_A)-1)/2)
	if spin == 'up': 
		Theta_A = Theta(Gup_A,Gdn_A,Lpp,Lmp,spin)
		GF_A = sp.copy(Gdn_A) 
		Det_A = DeterminantGD(Lpp,Lmp,Gup_A,Gdn_A)
	else: ## spin='dn'
		Theta_A = Theta(Gdn_A,Gup_A,Lpp,Lmp,spin)
		GF_A = sp.copy(Gup_A) 
		Det_A = sp.flipud(sp.conj(DeterminantGD(Lpp,Lmp,Gup_A,Gdn_A)))
	Kernel_A = U*Theta_A/Det_A
	## zero-padding the arrays
	FDex_A     = sp.concatenate([FD_A[N:],sp.zeros(2*N+3),FD_A[:N]])
	BEex_A     = sp.concatenate([BE_A[N:],sp.zeros(2*N+3),BE_A[:N]])
	GFex_A     = sp.concatenate([GF_A[N:],sp.zeros(2*N+3),GF_A[:N]])
	Kernelex_A = sp.concatenate([Kernel_A[N:],sp.zeros(2*N+3),Kernel_A[:N]])
	## performing the convolution
	ftSE1_A  = -sp.conj(fft(BEex_A*sp.imag(Kernelex_A)))*fft(GFex_A)*dE
	ftSE2_A  = +fft(FDex_A*sp.imag(GFex_A))*sp.conj(fft(Kernelex_A))*dE
	SE_A     = ifft(ftSE1_A+ftSE2_A)/sp.pi
	SE_A     = sp.concatenate([SE_A[3*N+4:],SE_A[:N+1]])
	return SE_A


## parlib2.py end ###

