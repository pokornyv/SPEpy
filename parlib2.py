###########################################################
# SPEpy - simplified parquet equation solver for SIAM     #
# Vladislav Pokorny; 2015-2019; pokornyv@fzu.cz           #
# homepage: github.com/pokornyv/SPEpy                     #
# developed and optimized using python 3.7.2              #
# parlib2.py - library of functions                       #
###########################################################

import scipy as sp
from scipy.integrate import simps
from scipy.fftpack import fft,ifft
from scipy.optimize import brentq,fixed_point,root
from parlib import KramersKronigFFT,FillFD,FillBE,KramersKronigFFT,TwoParticleBubble
from time import time

absC = lambda z: sp.real(z)**2+sp.imag(z)**2

def LambdaVertexD(U,K,BDFD):
	''' irreducible Lambda vertex Eq. (35) '''
	return U/(1.0+K*BDFD)


def KvertexD(i1,i2,Lpp,Lmp,Gup_A,Gdn_A,En_A,T):
	''' reducible K vertex Eq. (42ab) '''
	GG = CorrelatorGGzero(Gdn_A,Gup_A,En_A,T)
	#print('# GG0: {0: .8f} {1:+8f}i'.format(sp.real(GG0),sp.imag(GG)))
	if i1 == 1: ## K(+,+)
		K = -Lpp**2*GG-absC(Lmp)*sp.conj(GG)-Lpp*(absC(Lpp)-absC(Lmp))*absC(GG)
	else:       ## K(-,+)
		K = -Lmp*(Lpp*GG+sp.conj(Lpp)*sp.conj(GG)+(absC(Lpp)-absC(Lmp))*absC(GG))
	return K


def ReBDDFDD(i1,i2,Gup_A,Gdn_A,En_A,T):
	''' function to check the sum of imaginary parts of FD and BD ints. '''
#	FD_A = FillFD(En_A,T)
#	BE_A = FillBE(En_A,T)
	from parlib import WriteFile
	if i1 == 1: ## i2 is always +1
		Int1_A = sp.imag(1.0/sp.flipud(sp.conj(Det_A)))*sp.real(Gup_A*sp.flipud(Gdn_A))
	elif i1 == -1:
		Int1_A = sp.imag(1.0/sp.flipud(sp.conj(Det_A)))*sp.real(Gup_A*sp.flipud(sp.conj(Gdn_A)))
	Int2_A = sp.imag(Gup_A*sp.flipud(sp.conj(Gdn_A))/sp.flipud(sp.conj(Det_A)))
	RBF = simps((FD_A+BE_A)*Int1_A-FD_A*Int2_A,En_A)/sp.pi
	#print(' Re(BDD+FDD): {0: .8f} ({1: 2d},{2: 2d})'.format(float(RBF),i1,i2))
	return RBF


def ImBDDFDD(i1,i2,Gup_A,Gdn_A,En_A,T):
	''' function to check the sum of imaginary parts of FD and BD ints. '''
#	FD_A = FillFD(En_A,T)
#	BE_A = FillBE(En_A,T)
	if i1 == 1: ## i2 is always +1
		Int_A = sp.imag(1.0/sp.flipud(sp.conj(Det_A)))*sp.imag(Gup_A*sp.conj(Gdn_A))
	elif i1 == -1:
		Int_A = sp.imag(1.0/sp.flipud(sp.conj(Det_A)))*sp.imag(Gup_A*sp.flipud(sp.flipud(Gdn_A)))
	IBF = simps((FD_A+BE_A)*Int_A,En_A)/sp.pi
	#print(' Im(BDD+FDD): {0: .8f} ({1: 2d},{2: 2d})'.format(float(IBF),i1,i2))
	return IBF


def DeterminantGD(Lpp,Lmp,Gup_A,Gdn_A,En_A,T):
	''' determinant Eq. (40)  '''
	Det_A = 1.0+GG1_A*Lpp-(GG2_A-GG3_A)*Lmp-GG4_A*sp.conj(Lpp)+GG1_A*GG4_A*(absC(Lmp)-absC(Lpp))
	return Det_A


def CorrelatorGG(G1_A,G2_A,En_A,i1,i2,T):
	''' <G[s](x+w)G[s'](x+w')> Eq. (41) 
     i1 and i2 are imaginary parts of w or w' '''
#	FD_A = FillFD(En_A,T)
	N = int((len(En_A)-1)/2)
	dE = sp.around(En_A[1]-En_A[0],8)
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
	GG_A = KramersKronigFFT(sp.imag(GG_A)) + 1.0j*sp.imag(GG_A)
	return -GG_A/(2.0j*sp.pi)


def CorrelatorGGzero(G1_A,G2_A,En_A,T):
	''' <G[s](x)G[s'](x)> '''
	#FD_A = FillFD(En_A,T)
	Int_A = FD_A*G1_A*G2_A
	Int = simps(Int_A,En_A)
	TailL = -sp.real(Int_A[ 0])*En_A[ 0]-0.5j*sp.imag(Int_A[ 0])*En_A[ 0]
	TailR =  sp.real(Int_A[-1])*En_A[-1]+0.5j*sp.imag(Int_A[-1])*En_A[-1]
	return -(Int+TailL+TailR)/(2.0j*sp.pi)


def LambdaD(i1,i2,Gup_A,Gdn_A,Lpp,Lmp,En_A,U,T):
	''' calculates the Lambda vertex for given i1,i2 '''
	global FD_A,BE_A,GG1_A,GG2_A,GG3_A,GG4_A,Det_A
	Det_A  = DeterminantGD(Lpp,Lmp,Gup_A,Gdn_A,En_A,T)
	K      = KvertexD(i1,i2,Lpp,Lmp,Gup_A,Gdn_A,En_A,T)
	RFD    = ReBDDFDD(i1,i2,Gup_A,Gdn_A,En_A,T)
	IFD    = ImBDDFDD(i1,i2,Gup_A,Gdn_A,En_A,T)
	Lambda = U/(1.0+K*(RFD+1.0j*IFD))
	#print('{0: 2d} {1: 2d}\t{2: .8f} {3:+8f}i'.format(i1,i2,sp.real(Lambda),sp.imag(Lambda)))
	#print('{0: 2d} {1: 2d}\t{2: .8f} {3:+8f}i'.format(i1,i2,RFD,IFD))
	#print('{0: 2d} {1: 2d}\t{2: .8f} {3:+8f}i'.format(i1,i2,sp.real(K),sp.imag(K)))
	return Lambda


def VecLambdaD(Gup_A,Gdn_A,Lpp,Lmp,En_A,U,T):
	''' calculates both Lambda vertices L(++), L(-+), returns differences '''
	Lpp2 = LambdaD( 1, 1,Gup_A,Gdn_A,Lpp,Lmp,En_A,U,T)
	Lmp2 = LambdaD(-1, 1,Gup_A,Gdn_A,Lpp,Lmp,En_A,U,T)
	print("{0: .8f}\t{1: .8f}\t{2: .8f}\t{3: .8f}"\
	.format(sp.real(Lpp2),sp.imag(Lpp2),sp.real(Lmp2),sp.imag(Lmp2)))
	return [Lpp2-Lpp, Lmp2-Lmp]


def CalculateLambdaD(Gup_A,Gdn_A,En_A,Lpp,Lmp,U,T,eps,alpha,chat,sc_scheme):
	''' main solver for the Lambda vertex '''
	Bubble_A = TwoParticleBubble(Gup_A,Gdn_A,En_A,T,'eh')
	Uc = -1.0/sp.real(Bubble_A[int(len(En_A)/2)])
	#print("# - Critical U from fully static limit: {0: .6f}".format(Uc))
	global FD_A,BE_A
	global GG1_A,GG2_A,GG3_A,GG4_A
	FD_A = FillFD(En_A,T)
	BE_A = FillBE(En_A,T)
	## correlators do not change with iterations
	t = time()
	if chat: print('# - calculating correlators... ',end='',flush=True)
	GG1_A = CorrelatorGG(Gup_A,Gdn_A,En_A, 1, 1,T)
	GG2_A = CorrelatorGG(Gup_A,Gdn_A,En_A,-1, 1,T)
	GG3_A = CorrelatorGG(Gup_A,Gdn_A,En_A, 1,-1,T)
	GG4_A = CorrelatorGG(Gup_A,Gdn_A,En_A,-1,-1,T)
	if chat: print(' done in {0: .2f} seconds.'.format(time()-t))
	#print('# corr. GG0 {0: .8f} {1:+8f}i'.format(sp.real(GG1_A[int((len(En_A)-1)/2)]),sp.imag(GG1_A[int((len(En_A)-1)/2)])))
	#from parlib import WriteFile2
	#WriteFile2(En_A,GG1_A,GG2_A,GG3_A,GG4_A,100.0,3,'','GG.dat',1)
	#exit()
	[LppOld,LmpOld] = [1e8,1e8]
	if sc_scheme == 'iter': [diffpp,diffmp] = [1e8,1e8]
	k = 1
	#print(LambdaD( 1, 1,Gup_A,Gdn_A,Lppmax,Lmpmax,En_A,U,T).real)
	#print(LambdaD( 1, 1,Gup_A,Gdn_A,Lppmax,Lmpmax,En_A,U,T).imag)
	while any([sp.fabs(sp.real(Lpp-LppOld))>eps,sp.fabs(sp.real(Lmp-LmpOld))>eps]):
		[LppOld,LmpOld] = [Lpp,Lmp]
		Eqnpp = lambda x: LambdaD( 1, 1,Gup_A,Gdn_A,x,Lmp,En_A,U,T)
		Eqnmp = lambda x: LambdaD(-1, 1,Gup_A,Gdn_A,Lpp,x,En_A,U,T)
		if sc_scheme == 'fixed':
			try:
				Lpp = fixed_point(Eqnpp,Lpp,xtol=eps)
				Lmp = fixed_point(Eqnmp,Lmp,xtol=eps)
			except RuntimeError:
				print("# - CalculateLambdaD: No convergence in fixed-point algorithm.")
				print("# - Switch SCsolver to 'iter' or 'root' in siam.in and try again.")
				exit(1)
		elif sc_scheme == 'iter':
			#print('alpha',alpha)
			[diffppOld,diffmpOld] = [diffpp,diffmp]
			Lpp = alpha*Eqnpp(Lpp) + (1.0-alpha)*LppOld
			Lmp = alpha*Eqnmp(Lmp) + (1.0-alpha)*LmpOld
			diffpp = sp.fabs(sp.real(Lpp-LppOld))
			diffmp = sp.fabs(sp.real(Lmp-LmpOld))
			if all([diffpp<diffppOld,diffmp<diffmpOld]): alpha = sp.amin([1.1*alpha,1.0])
		elif sc_scheme == 'root':
			## implemented only for real Lambdas
			eqn = lambda x: VecLambdaD(Gup_A,Gdn_A,x[0],x[1],En_A,U,T)
			sol = root(eqn,[Lpp,Lmp],method='lm')
			#print("# Solution:",sol.x)
			[Lpp,Lmp] = [sol.x[0],sol.x[1]]
			break ## we don't need the outer loop here
		else:
			print('# CalculateLambdaD: Unknown SCsolver')
			exit(1)
		if chat: print('# - - iter. {0: 3d}: Lambda(++): {1: .8f} {2:+8f}i  Lambda(-+): {3: .8f} {4:+8f}i'\
		.format(k,sp.real(Lpp),sp.imag(Lpp),sp.real(Lmp),sp.imag(Lmp)))
		k += 1
	return [Lpp,Lmp]


def CorrelatorsSE(Gup_A,Gdn_A,En_A,i1,i2,T):
	''' correlators to Theta function, updated '''
	N = int((len(En_A)-1)/2)
	dE = sp.around(En_A[1]-En_A[0],8)
	#FD_A = FillFD(En_A,T)
	## zero-padding the arrays, G1 and G2 are complex functions
	FDex_A = sp.concatenate([FD_A[N:], sp.zeros(2*N+3), FD_A[:N]])
	Fup_A  = sp.concatenate([Gup_A[N:],sp.zeros(2*N+3),Gup_A[:N]])
	Fdn_A  = sp.concatenate([Gdn_A[N:],sp.zeros(2*N+3),Gdn_A[:N]])
	ftIGG1_A = fft(FDex_A*sp.imag(Fdn_A))*sp.conj(fft(Fup_A))*dE
	ftGG2_A  = sp.conj(fft(FDex_A*sp.conj(Fup_A)))*fft(Fdn_A)*dE
	ftGG3_A  = sp.conj(fft(FDex_A*Fup_A))*fft(Fdn_A)*dE
	IGG1_A   = -ifft(ftIGG1_A)/sp.pi
	GG2_A    = -ifft(ftGG2_A)/(2.0j*sp.pi)
	GG3_A    = -ifft(ftGG3_A)/(2.0j*sp.pi)
	## undo the zero padding
	IGG1_A = sp.concatenate([IGG1_A[3*N+4:],IGG1_A[:N+1]])
	GG2_A  = sp.concatenate([ GG2_A[3*N+4:], GG2_A[:N+1]])
	GG3_A  = sp.concatenate([ GG3_A[3*N+4:], GG3_A[:N+1]])
	return [IGG1_A,GG2_A,GG3_A]


def Theta(Gup_A,Gdn_A,En_A,Lpp,Lmp,T):
	''' auxiliary function to calculate spectral self-energy '''
	GG0 = CorrelatorGGzero(Gup_A,Gdn_A,En_A,T)
	#print('# GG0: {0: .8f} {1:+8f}i'.format(sp.real(GG0),sp.imag(GG)))
	gpp = Lpp+(absC(Lpp)-absC(Lmp))*sp.conj(GG0)
	gmp = Lmp
	[IGG1_A,GG2_A,GG3_A] = CorrelatorsSE(Gup_A,Gdn_A,En_A,1,1,T)
	#from parlib import WriteFile
	#WriteFile(En_A,IGG1_A,GG2_A,GG3_A,100,3,'#','ThetaGG.dat',1)
	Theta_A = gmp*IGG1_A+gpp*GG2_A-gmp*GG3_A
	return Theta_A


def SelfEnergyD(Gup_A,Gdn_A,En_A,Lpp,Lmp,U,T,spin):
	''' dynamic self-energy for spin-up '''
	global FD_A,BE_A
	FD_A = FillFD(En_A,T)
	BE_A = FillBE(En_A,T)
	N = int((len(En_A)-1)/2)
	dE = sp.around(En_A[1]-En_A[0],8)
	Theta_A  = Theta(Gup_A,Gdn_A,En_A,Lpp,Lmp,T)
	Det_A    = DeterminantGD(Lpp,Lmp,Gup_A,Gdn_A,En_A,T)
	Kernel_A = U*Theta_A/Det_A
	#from parlib import WriteFile2
	#if spin == 'up': WriteFile2(En_A,Gup_A,Gdn_A,Det_A,Theta_A,100,3,'#','sedet.dat',1)
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
def CorrelatorsSE_old(Gup_A,Gdn_A,En_A,i1,i2,T):
	''' correlators to Theta function '''
	N = int((len(En_A)-1)/2)
	dE = sp.around(En_A[1]-En_A[0],8)
	#FD_A = FillFD(En_A,T)
	## zero-padding the arrays, G1 and G2 are complex functions
	FDex_A  = sp.concatenate([FD_A[N:], sp.zeros(2*N+3), FD_A[:N]])
	Fup_A = sp.concatenate([Gup_A[N:],sp.zeros(2*N+3),Gup_A[:N]])
	Fdn_A = sp.concatenate([Gdn_A[N:],sp.zeros(2*N+3),Gdn_A[:N]])
	ftIGG1_A = sp.conj(fft(FDex_A*sp.imag(Fup_A)))*fft(Fdn_A)*dE
	ftGG2_A  = fft(FDex_A*Fdn_A)*sp.conj(fft(Fup_A))*dE
	ftGG3_A  = fft(FDex_A*sp.conj(Fdn_A))*sp.conj(fft(Fup_A))*dE
	IGG1_A   = -ifft(ftIGG1_A)/sp.pi
	GG2_A    = -ifft(ftGG2_A)/(2.0j*sp.pi)
	GG3_A    = -ifft(ftGG3_A)/(2.0j*sp.pi)
	## undo the zero padding
	IGG1_A = sp.concatenate([IGG1_A[3*N+4:],IGG1_A[:N+1]])
	GG2_A  = sp.concatenate([ GG2_A[3*N+4:], GG2_A[:N+1]])
	GG3_A  = sp.concatenate([ GG3_A[3*N+4:], GG3_A[:N+1]])
	return [IGG1_A,GG2_A,GG3_A]
"""

"""
def CalculateLambdaD2(GFup_A,GFdn_A,En_A,Lpp,Lmp,alpha,U,T):
	''' calculates both Lambda vertices '''
	DetG_A = DeterminantGD(Lpp,Lmp,GFup_A,GFdn_A,En_A,T)
	# WriteFile(En_A,DetG_A,GFTup_A,GFTdn_A,p.WriteMax,p.WriteStep,'gf','DetGF.dat',p.chat)
	Kpp = KvertexD( 1, 1,Lpp,Lmp,GFup_A,GFdn_A,En_A,T)
	Kmp = KvertexD(-1, 1,Lpp,Lmp,GFup_A,GFdn_A,En_A,T)
	#BDetpp = BDD( 1, 1,GFup_A,GFdn_A,DetG_A,En_A,T)
	#FDetpp = FDD( 1, 1,GFup_A,GFdn_A,DetG_A,En_A,T)
	RFDpp = ReBDDFDD( 1, 1,GFup_A,GFdn_A,DetG_A,En_A,T)
	IFDpp = ImBDDFDD( 1, 1,GFup_A,GFdn_A,DetG_A,En_A,T)
	#print('  1 1 FDD+BDD: {0: .8f} {1:+8f}i'.format(sp.real(BDetpp+FDetpp),sp.imag(BDetpp+FDetpp)))
	#BDetmp = BDD(-1, 1,GFup_A,GFdn_A,DetG_A,En_A,T)
	#FDetmp = FDD(-1, 1,GFup_A,GFdn_A,DetG_A,En_A,T)
	RFDmp = ReBDDFDD(-1, 1,GFup_A,GFdn_A,DetG_A,En_A,T)
	IFDmp = ImBDDFDD(-1, 1,GFup_A,GFdn_A,DetG_A,En_A,T)
	#print(' -1 1 FDD+BDD: {0: .8f} {1:+8f}i'.format(sp.real(BDetmp+FDetmp),sp.imag(BDetmp+FDetmp)))
	#print('# - aux. integrals: (FDD+BDD)(++): {0: .8f} {1:+8f}i (FDD+BDD)(-+): {2: .8f} {3:+8f}i'\
	#.format(RFDpp,IFDpp,RFDmp,IFDmp))
	Lambdapp = alpha*LambdaVertexD(U,Kpp,RFDpp+1.0j*IFDpp)+(1.0-alpha)*Lpp
	Lambdamp = alpha*LambdaVertexD(U,Kmp,RFDmp+1.0j*IFDmp)+(1.0-alpha)*Lmp
	return [Lambdapp,Lambdamp]
"""

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

