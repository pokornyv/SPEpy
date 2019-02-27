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
from parlib import KramersKronigFFT#,TwoParticleBubble
from time import time
from config_siam import *

absC = lambda z: sp.real(z)**2+sp.imag(z)**2

def LambdaVertexD(U,K,BDFD):
	''' irreducible Lambda vertex Eq. (35) '''
	return U/(1.0+K*BDFD)


def KvertexD(i1,i2,Lpp,Lpm,Gup_A,Gdn_A):
	''' reducible K vertex Eq. (39ab) '''
	GG = CorrelatorGGzero(Gdn_A,Gup_A)
	#print('# GG0: {0: .8f} {1:+8f}i'.format(sp.real(GG0),sp.imag(GG)))
	if i1 == 1: ## K(+,+)
		K = -Lpp**2*GG-absC(Lpm)*sp.conj(GG)-Lpp*(absC(Lpp)-absC(Lpm))*absC(GG)
	else:       ## K(-,+)
		#K = -Lpm*(Lpp*GG+sp.conj(Lpp)*sp.conj(GG)+(absC(Lpp)-absC(Lpm))*absC(GG))
		K = -Lpm*(2.0*sp.real(Lpp*GG)+(absC(Lpp)-absC(Lpm))*absC(GG))
	return K


def ReBDDFDD(i1,i2,Gup_A,Gdn_A,printint):
	''' function to check the sum of imaginary parts of FD and BD ints. '''
	N = int((len(En_A)-1)/2)
	if i1 == 1: ## i2 is always +1
		Int1_A = sp.imag(1.0/sp.flipud(sp.conj(Det_A)))*sp.real(Gup_A*sp.flipud(Gdn_A))
	elif i1 == -1:
		Int1_A = sp.imag(1.0/sp.flipud(sp.conj(Det_A)))*sp.real(Gup_A*sp.flipud(sp.conj(Gdn_A)))
	Int2_A = sp.imag(Gup_A*sp.flipud(sp.conj(Gdn_A))/sp.flipud(sp.conj(Det_A)))
	## here we multiply big and small numbers for energies close to zero
	RBF1_A    = sp.exp(sp.log(FB_A)+sp.log(Int1_A))
	RBF1_A[N] = (RBF1_A[N-1] + RBF1_A[N+1])/2.0
	RBF2_A    = -FD_A*Int2_A
	RBF       = simps(RBF1_A+RBF2_A,En_A)/sp.pi
	#print(' Re(BDD+FDD): {0: .8f} ({1: 2d},{2: 2d})'.format(float(RBF),i1,i2))
	if printint:
		from parlib import WriteFile2
		WriteFile2(Int1_A,Int2_A,RBF1_A,RBF2_A,1.0,4,'','RBF'+str(i1)+'.dat')
	return RBF


def ImBDDFDD(i1,i2,Gup_A,Gdn_A,printint):
	''' function to check the sum of imaginary parts of FD and BD ints. '''
	if i1 == 1: ## i2 is always +1
		Int_A = sp.imag(1.0/sp.flipud(sp.conj(Det_A)))*sp.imag(Gup_A*sp.conj(Gdn_A))
	elif i1 == -1:
		Int_A = sp.imag(1.0/sp.flipud(sp.conj(Det_A)))*sp.imag(Gup_A*sp.flipud(sp.flipud(Gdn_A)))
	IBF = simps(FB_A*Int_A,En_A)/sp.pi
	#print(' Im(BDD+FDD): {0: .8f} ({1: 2d},{2: 2d})'.format(float(IBF),i1,i2))
	if printint:
		from parlib import WriteFile2
		WriteFile2(Det_A,1.0/Det_A,Int_A,FB_A*Int_A,1.0,4,'','IBF'+str(i1)+'.dat')
	return IBF


def DeterminantGD(Lpp,Lpm,Gup_A,Gdn_A):
	''' determinant Eq. (40)  '''
	Det_A = 1.0+GG1_A*Lpp-(GG2_A-GG3_A)*Lpm-GG4_A*sp.conj(Lpp)-GG1_A*GG4_A*(absC(Lpp)-absC(Lpm))
	return Det_A


def CorrelatorGG(G1_A,G2_A,En_A,i1,i2):
	''' <G[s](x+w)G[s'](x+w')> Eq. (41) 
     i1 and i2 are imaginary parts of w or w' '''
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
	## recalculate real part via KK
	#GG_A = KramersKronigFFT(sp.imag(GG_A)) + 1.0j*sp.imag(GG_A)
	return -GG_A/(2.0j*sp.pi)


def CorrelatorGGzero(G1_A,G2_A):
	''' <G[s](x)G[s'](x)> '''
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


def LambdaD(i1,i2,Gup_A,Gdn_A,Lpp,Lpm):
	''' calculates the Lambda vertex for given i1,i2 '''
	global GG1_A,GG2_A,GG3_A,GG4_A,Det_A
	Det_A  = DeterminantGD(Lpp,Lpm,Gup_A,Gdn_A)
	K      = KvertexD(i1,i2,Lpp,Lpm,Gup_A,Gdn_A)
	RFD    = ReBDDFDD(i1,i2,Gup_A,Gdn_A,0)
	IFD    = ImBDDFDD(i1,i2,Gup_A,Gdn_A,0)
	Lambda = U/(1.0+K*(RFD+1.0j*IFD))
	#print('{0: 2d} {1: 2d}\t{2: .8f} {3:+8f}i'.format(i1,i2,sp.real(Lambda),sp.imag(Lambda)))
	#print('{0: 2d} {1: 2d}\t{2: .8f} {3:+8f}i'.format(i1,i2,RFD,IFD))
	#print('{0: 2d} {1: 2d}\t{2: .8f} {3:+8f}i'.format(i1,i2,sp.real(K),sp.imag(K)))
	return Lambda


def VecLambdaD(Gup_A,Gdn_A,LppR,LppI,LpmR,LpmI):
	''' calculates both Lambda vertices L(++), L(+-), returns differences '''
	Lpp2 = LambdaD( 1, 1,Gup_A,Gdn_A,LppR+1.0j*LppI,LpmR+1.0j*LpmI)
	Lpm2 = LambdaD(-1, 1,Gup_A,Gdn_A,LppR+1.0j*LppI,LpmR+1.0j*LpmI)
	#print("{0: .8f}\t{1: .8f}\t{2: .8f}\t{3: .8f}"\
	#.format(sp.real(Lpp2),sp.imag(Lpp2),sp.real(Lmp2),sp.imag(Lmp2)))
	return [sp.real(Lpp2)-LppR,sp.imag(Lpp2)-LppI,sp.real(Lpm2)-LpmR,sp.imag(Lpm2)-LpmI]


def CalculateLambdaD(Gup_A,Gdn_A,Lpp,Lpm):
	''' main solver for the Lambda vertex '''
	global GG1_A,GG2_A,GG3_A,GG4_A,alpha
	[LppOld,LpmOld] = [1e8,1e8]
	if SCsolver == 'iter': [diffpp,diffpm] = [1e8,1e8]
	## correlators not't change with Lambda iterations
	t = time()
	if chat: print('# - calculating correlators... ',end='',flush=True)
	GG1_A = CorrelatorGG(Gup_A,Gdn_A,En_A, 1, 1)
	GG2_A = CorrelatorGG(Gup_A,Gdn_A,En_A,-1, 1)
	GG3_A = CorrelatorGG(Gup_A,Gdn_A,En_A, 1,-1)
	GG4_A = CorrelatorGG(Gup_A,Gdn_A,En_A,-1,-1)
	if chat: print(' done in {0: .2f} seconds.'.format(time()-t))
	#print('# corr. GG0 {0: .8f} {1:+8f}i'.format(sp.real(GG1_A[int((len(En_A)-1)/2)]),sp.imag(GG1_A[int((len(En_A)-1)/2)])))
	#from parlib import WriteFile2
	#WriteFile2(GG1_A,GG2_A,GG3_A,GG4_A,100.0,3,'','GGcorr.dat')
	#exit()
	#print(LambdaD( 1, 1,Gup_A,Gdn_A,Lppmax,Lmpmax).real)
	#print(LambdaD( 1, 1,Gup_A,Gdn_A,Lppmax,Lmpmax).imag)
	k = 1
	while any([sp.fabs(sp.real(Lpp-LppOld))>epsl,sp.fabs(sp.real(Lpm-LpmOld))>epsl]):
		[LppOld,LpmOld] = [Lpp,Lpm]
		Eqnpp = lambda x: LambdaD( 1, 1,Gup_A,Gdn_A,x,Lpm)
		Eqnpm = lambda x: LambdaD(-1, 1,Gup_A,Gdn_A,Lpp,x)
		if SCsolver == 'fixed':
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
			Lpp = alpha*Eqnpp(Lpp) + (1.0-alpha)*LppOld
			Lpm = alpha*Eqnpm(Lpm) + (1.0-alpha)*LpmOld
			diffpp = sp.fabs(sp.real(Lpp-LppOld))
			diffpm = sp.fabs(sp.real(Lpm-LpmOld))
			if all([diffpp<diffppOld,diffpm<diffpmOld]): alpha = sp.amin([1.05*alpha,1.0])
		elif SCsolver == 'root':
			## implemented for complex Lambdas as 4-dimensional problem
			eqn = lambda x: VecLambdaD(Gup_A,Gdn_A,x[0],x[1],x[2],x[3])
			sol = root(eqn,[Lpp,0.0,Lpm,0.0],method='lm')
			#print("# Solution:",sol.x)
			[Lpp,Lpm] = [sol.x[0]+1.0j*sol.x[1],sol.x[2]+1.0j*sol.x[3]]
			break ## we don't need the outer loop here
		else:
			print('# CalculateLambdaD: Unknown SCsolver')
			exit(1)
		if chat: print('# - - iter. {0: 3d}: Lambda(++): {1: .8f} {2:+8f}i  Lambda(+-): {3: .8f} {4:+8f}i'\
		.format(k,sp.real(Lpp),sp.imag(Lpp),sp.real(Lpm),sp.imag(Lpm)))
		if k > 1000:
			print('# CalculateLambdaD: No convergence after 1000 iterations. Exit.')
			exit(1)
		k += 1
	return [Lpp,Lpm]


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


def Theta(Gup_A,Gdn_A,Lpp,Lmp):
	''' auxiliary function to calculate spectral self-energy '''
	GG0 = CorrelatorGGzero(Gup_A,Gdn_A)
	#print('# GG0: {0: .8f} {1:+8f}i'.format(sp.real(GG0),sp.imag(GG)))
	gpp = Lpp+(absC(Lpp)-absC(Lmp))*sp.conj(GG0)
	gmp = Lmp
	[IGGs1_A,GGs2_A,GGs3_A] = CorrelatorsSE(Gup_A,Gdn_A,1,1)
	#from parlib import WriteFile
	#WriteFile(IGGs1_A,GGs2_A,GGs3_A,100,3,'','ThetaGG.dat')
	Theta_A = gmp*IGGs1_A+gpp*GGs2_A-gmp*GGs3_A
	return Theta_A


def SelfEnergyD(Gup_A,Gdn_A,Lpp,Lmp,U,spin):
	''' dynamic self-energy for spin-up '''
	global GG1_A,GG2_A,GG3_A,GG4_A
	#GG1_A = CorrelatorGG(Gup_A,Gdn_A,En_A, 1, 1)
	#GG2_A = CorrelatorGG(Gup_A,Gdn_A,En_A,-1, 1)
	#GG3_A = CorrelatorGG(Gup_A,Gdn_A,En_A, 1,-1)
	#GG4_A = CorrelatorGG(Gup_A,Gdn_A,En_A,-1,-1)
	N = int((len(En_A)-1)/2)
	Theta_A  = Theta(Gup_A,Gdn_A,Lpp,Lmp)
	Det_A    = DeterminantGD(Lpp,Lmp,Gup_A,Gdn_A)
	Kernel_A = U*Theta_A/Det_A
	#from parlib import WriteFile2
	#if spin == 'up': WriteFile2(Gup_A,Gdn_A,Det_A,Theta_A,100,3,'','sedet.dat')
	if spin == 'up': GF_A = sp.copy(Gdn_A) 
	else:            GF_A = sp.copy(Gup_A) 
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


"""
def FDD(i1,i2,Gup_A,Gdn_A,Det_A,En_A,T):
	''' fermionic integral over the Green function Eq. (37) '''
	from parlib import WriteFile
	FD_A = FillFD(En_A,T)
	#Int_A = BE_A*Gup_A*sp.conj(sp.flipud(Gdn_A))*sp.imag(1.0/Det_A)
	if i1 == 1: ## i2 is always +1
		Int1_A = -Gup_A*sp.imag(sp.flipud(Gdn_A))/sp.conj(sp.flipud(Det_A))
		Int2_A =  sp.flipud(Gdn_A)*sp.imag(Gup_A)/sp.flipud(Det_A)
#		WriteFile(En_A,Int_A,Gup_A,Gdn_A,0,0,0,30.0,3,'','FDDpp.dat',1)
	elif i1 == -1:
		Int1_A = -Gup_A*sp.imag(sp.flipud(Gdn_A))/sp.flipud(Det_A)
		Int2_A =  sp.conj(sp.flipud(Gdn_A))*sp.imag(Gup_A)/sp.flipud(Det_A)
#		WriteFile(En_A,Int_A,Gup_A,Gdn_A,0,0,0,30.0,3,'','FDDmp.dat',1)
	#FDD1 = -simps(FD_A*Int1_A,En_A)/sp.pi
	#FDD2 = -simps(FD_A*Int2_A,En_A)/sp.pi
	FDD = -simps(FD_A*(Int1_A+Int2_A),En_A)/sp.pi
	#print(' FDD1: {0: .8f} {1:+8f}i ({2: 2d},{3: 2d})'.format(sp.real(FDD1),sp.imag(FDD1),i1,i2))
	#print(' FDD2: {0: .8f} {1:+8f}i ({2: 2d},{3: 2d})'.format(sp.real(FDD2),sp.imag(FDD2),i1,i2))
	#print(' FDD: {0: .8f} {1:+8f}i ({2: 2d},{3: 2d})'.format(sp.real(FDD),sp.imag(FDD),i1,i2))
	return FDD


def BDD(i1,i2,Gup_A,Gdn_A,Det_A,En_A,T):
	''' bosonic integral over the Green function Eq. (36) '''
	from parlib import WriteFile
	BE_A = FillBE(En_A,T)
	if i1 == 1: ## i2 is always +1
		Int_A = Gup_A*sp.flipud(Gdn_A)*sp.imag(1.0/sp.flipud(sp.conj(Det_A)))
#		WriteFile(En_A,Int_A,Gup_A,Gdn_A,0,0,0,30.0,3,'','BDDpp.dat',1)
	elif i1 == -1:
		Int_A = Gup_A*sp.flipud(sp.conj(Gdn_A))*sp.imag(1.0/sp.flipud(sp.conj(Det_A)))
#		WriteFile(En_A,Int_A,Gup_A,Gdn_A,0,0,0,30.0,3,'','BDDmp.dat',1)
	BDD = simps(BE_A*Int_A,En_A)/sp.pi
	#print(' BDD: {0: .8f} {1:+8f}i ({2: 2d},{3: 2d})'.format(sp.real(BDD),sp.imag(BDD),i1,i2))
	return BDD
"""

## parlib2.py end ###

