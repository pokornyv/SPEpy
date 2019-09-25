###########################################################
# SPEpy - simplified parquet equation solver for SIAM     #
# Copyright (C) 2019  Vladislav Pokorny; pokornyv@fzu.cz  #
# homepage: github.com/pokornyv/SPEpy                     #
# parlib.py - library of functions                        #
###########################################################

import scipy as sp
from sys import exit
from time import ctime
from scipy.integrate import simps
from scipy.fftpack import fft,ifft
from scipy.interpolate import InterpolatedUnivariateSpline,UnivariateSpline
from scipy.optimize import brentq,fixed_point
from scipy.special import erf,wofz
from config_siam import *

## auxiliary functions ####################################

def KondoTemperature(U,Gamma,en):
	''' calculating Kondo temperature '''
	if U!=0.0 and Gamma!=0.0:
		return sp.sqrt(U*Gamma/2.0)*sp.exp(-sp.pi*sp.fabs(U**2-4.0*en**2)/(8.0*U*Gamma))
	else:
		return 0.0


def IntDOS(GF_A):
	''' the integral over the DOS, should be 1.0 '''
	TailL =  sp.imag(GF_A)[ 0]*En_A[ 0]/sp.pi	# left tail
	TailR = -sp.imag(GF_A)[-1]*En_A[-1]/sp.pi	# right tail
	return -simps(sp.imag(GF_A),En_A)/sp.pi+TailL+TailR


def Filling(GF_A):
	''' calculates filling, the integral over (-inf,inf) of f(w)G(w) '''
	DOS_A = -FD_A*sp.imag(GF_A)/sp.pi
	## old tail: fit by a/x**2
	TailL1 = -DOS_A[ 0]*En_A[ 0]	# left tail
	TailR1 =  DOS_A[-1]*En_A[-1]	# right tail
	return simps(DOS_A,En_A) + TailL1 + TailR1

## functions to calculate spectral density ################

def GreensFunctionSemi(E,W):
	'''	local Green's function for semielliptic band '''
	## small imaginary part helps to keep us on the right brach of the square root
	gz = E+1e-12j
	return (2.0*gz/W**2)*(1.0-sp.sqrt(1.0-W**2/gz**2))


def DensitySemi(x,W):
	''' particle denisty of a semi-elliptic band for T=0 '''
	return sp.real(0.5 - x*sp.sqrt(W**2-x**2)/(sp.pi*W**2) - sp.arcsin(x/W)/sp.pi)


def GreensFunctionLorenz(E,Delta):
	'''	local Green's function for Lorentzian band '''
	return 1.0/(E+1.0j*Delta)


def DensityLorentz(x,Delta):
	''' particle denisty of a Lorentzian band  for T=0 '''
	return 0.5 - sp.arctan(x/Delta)/sp.pi


def GreensFunctionGauss(E,Gamma):
	'''	local Green's function for Gaussian band '''
	return -1.0j*sp.sqrt(sp.pi/(2.0*Gamma**2))*wofz(E/sp.sqrt(2.0*Gamma**2))


def DensityGauss(x,Gamma):
	''' particle denisty of a Gaussian band  for T=0 '''
	return (1.0+erf(x/sp.sqrt(2.0*Gamma**2)))/2.0


def GreensFunctionSquare(x,izero,W):
	''' local Green's function electrons on a 2D square lattice,
	using elliptic integral of the first kind K(z) from mpmath '''
	from mpmath import ellipk
	K = sp.frompyfunc(ellipk,1,1)
	x = x+1.0j*izero
	return sp.array(2.0*K((W/x)**2)/(sp.pi*x),dtype=complex)


def GreensFunctionSCubic(x,W):
	''' local Green's function electrons on a 3D sc lattice '''
	## scipy version of hyp2f1 has precision issues arouns some points
	#from scipy.special import hyp2f1
	#K = lambda k: hyp2f1(0.5,0.5,1.0,k)*sp.pi/2.0 
	from mpmath import ellipk
	#K = sp.vectorize(ellipk)
	K = sp.frompyfunc(ellipk,1,1)
	x = (3.0/W)*x+1e-12j # scaling the half-width and securing the correct branch cut
	A = 0.5+1.0/(2.0*x**2)*(3.0-sp.sqrt(x**2-9.0)*sp.sqrt(x**2-1.0))
	B = A/(A-1.0)
	kpm1 = 0.25*B*sp.sqrt(4.0-B)
	kpm2 = 0.25*(2.0-B)*sp.sqrt(1.0-B)
	ellip1 = K(0.5+kpm1-kpm2)
	ellip2 = K(0.5-kpm1-kpm2)
	return sp.array((3.0/W)*4.0*sp.sqrt(1.0-0.75*A)/(1.0-A)*ellip1*ellip2/(sp.pi**2*x),dtype=complex)


## superconducting Green function

def GapFunction(x,Delta):
	''' returns one when -Delta<x<Delta, zero otherwise '''
	return sp.sign(Delta**2-x**2)/2.0+0.5


def BandFunction(x,Delta):
	''' returns one when |x|>Delta, zero otherwise '''
	return sp.sign(x**2-Delta**2)/2.0+0.5


def HybFunctionSC(x,izero,GammaS,GammaN,Delta,Phi):
	''' hybridization function for the rotated superconducting model at half-filling '''
	return GammaN+2.0*GammaS*sp.fabs(x)*BandFunction(x,Delta)/sp.sqrt((x+1.0j*izero)**2-Delta**2)\
	*(1.0-Delta/(x+1.0j*izero)*sp.cos(Phi/2.0))


def GreensFunctionSC(x,izero,GammaS,GammaN,Delta,Phi):
	return 1.0/(En_A+1.0j*izero+1.0j*HybFunctionSC(x,izero,GammaS,GammaN,Delta,Phi))

## functions to manipulate and process spectra

def ShiftGreensFunction(GF_A,shift):
	''' fill the GF array with GF shifted by real, static self-energy '''
	ReGF = InterpolatedUnivariateSpline(En_A,sp.real(GF_A))
	ImGF = InterpolatedUnivariateSpline(En_A,sp.imag(GF_A))
	GF_A = ReGF(En_A+shift)+1.0j*ImGF(En_A+shift)
	return GF_A


def CalculateHWHM(GF_A):
	''' calculates the half-width at half-maximum of the Kondo resonance 
	and the maximum of the spectral function '''
	N = len(En_A)
	IntMin = int((N+1)/2-int(0.5/dE))
	IntMax = int((N+1)/2+int(0.5/dE))
	DOSmaxPos = sp.argmax(-sp.imag(GF_A[IntMin:IntMax])/sp.pi)
	DOSmax    = -sp.imag(GF_A[IntMin+DOSmaxPos])/sp.pi # maximum of DoS
	wmax      = En_A[IntMin+DOSmaxPos]                 # position of the maximum at energy axis
	DOS = InterpolatedUnivariateSpline(En_A-1e-12,-sp.imag(GF_A)/sp.pi-DOSmax/2.0) 
	## 1e-12 breaks symmetry for half-filling, otherway DOS.roots() loses one solution.
	DOSroots_A = sp.sort(sp.fabs(DOS.roots()))
	try:
		HWHM = (DOSroots_A[0] + DOSroots_A[1])/2.0
	except IndexError:
		HWHM = 0.0
	return [HWHM,DOSmax,wmax]


def QuasiPWeight(ReSE_A):
	''' calculating the Fermi-liquid quasiparticle weight (residue) Z '''
	N = len(En_A)
	#M = int(1e-3/dE) if dE < 1e-3 else 1	# very fine grids lead to oscillations
	# replace 1 with M below to dilute the grid
	ReSE = UnivariateSpline(En_A[int(N/2-10):int(N/2+10):1],ReSE_A[int(N/2-10):int(N/2+10):1])
	dReSEdw = ReSE.derivatives(0.0)[1]
	Z = 1.0/(1.0-dReSEdw)
	return sp.array([Z,dReSEdw])

## bubbles and vertex functions ###########################

def KVertex(Lambda,Bubble_A):
	''' dynamical part of the two-particle vertex '''
	## offE helps to prevent the 'RuntimeWarning: invalid value encountered in true_divide'
	## that causes problems in non-symmetric cases while calculating SigmaT in siam_static.py
	return -Lambda**2*Bubble_A/(1.0+Lambda*Bubble_A+offE)


def PsiInt(K_A,GFup_A,GFdn_A):
	''' integrating the static bubble \Psi = KxGxG, extended for finite temperatures'''
	Int_A = BE_A*sp.imag(sp.flipud(sp.conj(K_A*GFdn_A))*GFup_A)
	TailL = -Int_A[0]*En_A[0]/2.0
	return (simps(Int_A,En_A)+TailL)/sp.pi


def LambdaFunction(Lambda,Bubble_A,GFup_A,GFdn_A):
	''' function to calculate new Lambda from previous iteration '''
	K_A = KVertex(Lambda,Bubble_A)
	Psi = PsiInt(K_A,GFup_A,GFdn_A)
	Lambda = U/(1.0+Psi)
	#if chat: print('# - - Psi = {0: .8f}, Lambda = {1: .8f}'.format(Psi,Lambda))
	return Lambda


def CalculateLambda(Bubble_A,GFup_A,GFdn_A):
	''' function to calculate static Lambda '''
	Uc = -1.0/sp.real(Bubble_A[int(len(En_A)/2)])
	print('# - - Critical U: {0: .6f}'.format(Uc))
	if Uc < 1e-6:
		print('# Warning: CalculateLambda: critical U is very small, please check the bubble.')	
	LMin = 0.0
	LMax = Uc-1e-10
	#if GFtype == 'sc': LMax = 10.0 ## :SCGF
	try:
		eqn = lambda x: x-LambdaFunction(x,Bubble_A,GFup_A,GFdn_A)
		w = 0.0
		Lambda = brentq(eqn,LMin,LMax,xtol=epsl)
	except ValueError:
		print('# Error: CalculateLambda: brentq failed to calculate Lambda, probably too close to critical U.')
#		print('#        Using Lambda = 0.99*pi')
#		Lambda = 0.99*sp.pi
		exit()
	return Lambda

## susceptibilities #######################################

def SusceptibilityTherm(a,GF_A):
	''' susceptibility calculated from the thermal self-energy derivative '''
	Int_A = FD_A*sp.imag(GF_A**2)
	## what about tail???
	return 2.0*simps(Int_A,En_A)/(a*sp.pi)


def SusceptibilitySpec(U,Lambda,X_A,GF_A,BubZero):
	''' susceptibility calculated from the spectral self-energy derivative '''
	Int_A = FD_A*sp.imag(GF_A**2*(1.0-U*X_A/(1.0+Lambda*BubZero)))
	## what about tail???
	return 2.0*simps(Int_A,En_A)/sp.pi


def SusceptibilityHF(U,GF_A,X_A):
	''' susceptibility calculated from the full spectral self-energy derivative '''
	Int1_A = FD_A*sp.imag(GF_A**2*(1.0-U*X_A))
	Int2_A = FD_A*sp.imag(GF_A**2*X_A)
	I1 = simps(Int1_A,En_A)/sp.pi
	I2 = simps(Int2_A,En_A)/sp.pi
	return 2.0*I1/(1.0+U**2*I2)


def AFbubble(GFzero_A,MuBar):
	''' non-local bubble for k=(pi,pi,pi) - antiferromagnetic case for sc lattice '''
	shFD_A = FillFD(En_A-MuBar,T)
	F_A = -sp.imag(GFzero_A)*shFD_A/(sp.pi*En_A)
	F_A[int(len(En_A)/2)] = 0.0 # we are taking the principal value
	return simps(F_A,En_A)


def SpecHWHM(GFint_A):
	''' Half-width at half-maximum of the spectral function '''
	N = len(En_A)
	DOSF = -sp.imag(GFint_A[N/2])/sp.pi	# value at Fermi energy
	DOS = InterpolatedUnivariateSpline(En_A,-sp.imag(GFint_A)/sp.pi-DOSF/2.0)
	return sp.amin(sp.fabs(DOS.roots()))

## convolutions in Matsubara frequencies ##################

def TwoParticleBubble(F1_A,F2_A,channel):
	''' calculates the two-particle bubble, channel = 'eh', 'ee' '''
	N = int((len(En_A)-1)/2)
	## zero-padding the arrays
	exFD_A = sp.concatenate([FD_A[N:],sp.zeros(2*N+3),FD_A[:N]])
	ImF1_A = sp.concatenate([sp.imag(F1_A[N:]),sp.zeros(2*N+3),sp.imag(F1_A[:N])])
	ImF2_A = sp.concatenate([sp.imag(F2_A[N:]),sp.zeros(2*N+3),sp.imag(F2_A[:N])])
	## performing the convolution
	if channel == 'eh':
		ftImChi1_A =  sp.conj(fft(exFD_A*ImF2_A))*fft(ImF1_A)*dE	# f(x)F2(x)F1(w+x)
		ftImChi2_A = -fft(exFD_A*ImF1_A)*sp.conj(fft(ImF2_A))*dE	# f(x)F1(x)F2(x-w)
	elif channel == 'ee':
		ftImChi1_A =  fft(exFD_A*ImF2_A)*fft(ImF1_A)*dE					# f(x)F2(x)F1(w-x)
		ftImChi2_A = -sp.conj(fft(exFD_A*sp.flipud(ImF1_A)))*fft(ImF2_A)*dE	# f(x)F1(-x)F2(w+x)
	ImChi_A = -sp.real(ifft(ftImChi1_A+ftImChi2_A))/sp.pi
	ImChi_A = sp.concatenate([ImChi_A[3*N+4:],ImChi_A[:N+1]])
	Chi_A = KramersKronigFFT(ImChi_A) + 1.0j*ImChi_A
	return Chi_A


def SelfEnergy(GF_A,ChiGamma_A):
	''' calculating the dynamical self-energy from the Schwinger-Dyson equation '''
	N = int((len(En_A)-1)/2)
	## zero-padding the arrays
	exFD_A = sp.concatenate([FD_A[N:],sp.zeros(2*N+3),FD_A[:N]])
	exBE_A = sp.concatenate([BE_A[N:],sp.zeros(2*N+3),BE_A[:N]])
	ImGF_A = sp.concatenate([sp.imag(GF_A[N:]),sp.zeros(2*N+3),sp.imag(GF_A[:N])])
	ImCG_A = sp.concatenate([sp.imag(ChiGamma_A[N:]),sp.zeros(2*N+3),sp.imag(ChiGamma_A[:N])])
	## performing the convolution
	ftImSE1_A = -sp.conj(fft(exBE_A*ImCG_A))*fft(ImGF_A)*dE
	ftImSE2_A = -fft(exFD_A*ImGF_A)*sp.conj(fft(ImCG_A))*dE
	ImSE_A = sp.real(ifft(ftImSE1_A+ftImSE2_A))/sp.pi
	ImSE_A = sp.concatenate([ImSE_A[3*N+4:],ImSE_A[:N+1]])
	Sigma_A = KramersKronigFFT(ImSE_A) + 1.0j*ImSE_A
	return Sigma_A


def KramersKronigFFT(ImX_A):
	'''	Hilbert transform used to calculate real part of a function from its imaginary part
	uses piecewise cubic interpolated integral kernel of the Hilbert transform
	use only if len(ImX_A)=2**m-1, uses fft from scipy.fftpack  '''
	X_A = sp.copy(ImX_A)
	N = int(len(X_A))
	## be careful with the data type, orherwise it fails for large N
	if N > 3e6: A = sp.arange(3,N+1,dtype='float64')
	else:       A = sp.arange(3,N+1)  
	X1 = 4.0*sp.log(1.5)
	X2 = 10.0*sp.log(4.0/3.0)-6.0*sp.log(1.5)
	## filling the kernel
	if N > 3e6: Kernel_A = sp.zeros(N-2,dtype='float64')
	else:       Kernel_A = sp.zeros(N-2)
	Kernel_A = (1-A**2)*((A-2)*sp.arctanh(1.0/(1-2*A))+(A+2)*sp.arctanh(1.0/(1+2*A)))\
	+((A**3-6*A**2+11*A-6)*sp.arctanh(1.0/(3-2*A))+(A+3)*(A**2+3*A+2)*sp.arctanh(1.0/(2*A+3)))/3.0
	Kernel_A = sp.concatenate([-sp.flipud(Kernel_A),sp.array([-X2,-X1,0.0,X1,X2]),Kernel_A])/sp.pi
	## zero-padding the functions for fft
	ImXExt_A = sp.concatenate([X_A[int((N-1)/2):],sp.zeros(N+2),X_A[:int((N-1)/2)]])
	KernelExt_A = sp.concatenate([Kernel_A[N:],sp.zeros(1),Kernel_A[:N]])
	## performing the fft
	ftReXExt_A = -fft(ImXExt_A)*fft(KernelExt_A)
	ReXExt_A = sp.real(ifft(ftReXExt_A))
	ReX_A = sp.concatenate([ReXExt_A[int((3*N+3)/2+1):],ReXExt_A[:int((N-1)/2+1)]])
	return ReX_A


def XIntegralsFFT(GF_A,Bubble_A,Lambda,BubZero):
	''' calculate X integral to susceptibilities using FFT '''
	N = int((len(En_A)-1)/2)
	Kappa_A  = TwoParticleBubble(GF_A,GF_A**2,'eh')
	Bubble_A = TwoParticleBubble(GF_A,GF_A,'eh')
	#print(Kappa_A[N],Bubble_A[N])
	V_A   = 1.0/(1.0+Lambda*Bubble_A)
	KV_A  = Lambda*Kappa_A*V_A**2
	KmV_A = Lambda*sp.flipud(sp.conj(Kappa_A))*V_A**2
	## zero-padding the arrays
	exFD_A  = sp.concatenate([FD_A[N:],sp.zeros(2*N+2),FD_A[:N+1]])
	ImGF_A  = sp.concatenate([sp.imag(GF_A[N:]),sp.zeros(2*N+2),sp.imag(GF_A[:N+1])])
	ImGF2_A = sp.concatenate([sp.imag(GF_A[N:]**2),sp.zeros(2*N+2),sp.imag(GF_A[:N+1]**2)])
	ImV_A   = sp.concatenate([sp.imag(V_A[N:]),sp.zeros(2*N+2),sp.imag(V_A[:N+1])])
	ImKV_A  = sp.concatenate([sp.imag(KV_A[N:]),sp.zeros(2*N+2),sp.imag(KV_A[:N+1])])
	ImKmV_A = sp.concatenate([sp.imag(KmV_A[N:]),sp.zeros(2*N+2),sp.imag(KmV_A[:N+1])])
	## performing the convolution
	ftImX11_A = -sp.conj(fft(exFD_A*ImV_A))*fft(ImGF2_A)*dE
	ftImX12_A =  fft(exFD_A*ImGF2_A)*sp.conj(fft(ImV_A))*dE
	ftImX21_A = -sp.conj(fft(exFD_A*ImKV_A))*fft(ImGF_A)*dE
	ftImX22_A =  fft(exFD_A*ImGF_A)*sp.conj(fft(ImKV_A))*dE
	ftImX31_A = -sp.conj(fft(exFD_A*ImKmV_A))*fft(ImGF_A)*dE
	ftImX32_A =  fft(exFD_A*ImGF_A)*sp.conj(fft(ImKmV_A))*dE
	## inverse transform
	ImX1_A =  sp.real(ifft(ftImX11_A+ftImX12_A))/sp.pi
	ImX2_A =  sp.real(ifft(ftImX21_A+ftImX22_A))/sp.pi
	ImX3_A = -sp.real(ifft(ftImX31_A+ftImX32_A))/sp.pi
	ImX1_A =  sp.concatenate([ImX1_A[3*N+4:],ImX1_A[:N+1]])
	ImX2_A =  sp.concatenate([ImX2_A[3*N+4:],ImX2_A[:N+1]])
	ImX3_A =  sp.concatenate([ImX3_A[3*N+4:],ImX3_A[:N+1]])
	## getting real part from imaginary
	X1_A = KramersKronigFFT(ImX1_A) + 1.0j*ImX1_A + BubZero # constant part !!!
	X2_A = KramersKronigFFT(ImX2_A) + 1.0j*ImX2_A
	X3_A = KramersKronigFFT(ImX3_A) + 1.0j*ImX3_A
	return [X1_A,X2_A,X3_A]


## output functions #######################################

def WriteFileX(X_L,WriteMax,de_dec,header,filename):
	''' writes data arrays to file, writes multiple complex arrays at once 
	X_L is a list of complex arrays, WriteMax is the cutoff in energies,
	de_dec is the density of the output and header is the header line '''
	LN = len(X_L)
	f = open(filename,'w')
	f.write("# file written "+ctime()+'\n')
	f.write(header+'\n')
	line = ''
	for i in range(len(En_A)):
		if sp.fabs(En_A[i]) <= WriteMax and sp.fabs(En_A[i] - sp.around(En_A[i],de_dec)) == 0:
			line = '{0: .6f}\t'.format(float(En_A[i]))
			for k in range(LN):
				line += '{0: .6f}\t{1: .6f}\t'.format(float(sp.real(X_L[k][i])),float(sp.imag(X_L[k][i])))
			f.write(line+'\n')
	f.close
	if chat: print('#   File '+filename+' written.')

## parlib.py end ###

