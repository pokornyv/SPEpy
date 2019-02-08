###########################################################
# SPEpy - simplified parquet equation solver for SIAM     #
# Vladislav Pokorny; 2015-2019; pokornyv@fzu.cz           #
# homepage: github.com/pokornyv/SPEpy                     #
# developed and optimized using python 3.7.2              #
# parlib.py - library of functions                        #
###########################################################

import scipy as sp
import warnings
from sys import exit
from scipy.integrate import simps
from scipy.fftpack import fft,ifft
from scipy.interpolate import InterpolatedUnivariateSpline,UnivariateSpline
from scipy.optimize import brentq,fixed_point
from scipy.special import erf,wofz
#from scipy.optimize import curve_fit

## particle distributions #################################

offE = 1e-12

FermiDirac      = lambda E,T: 1.0/(sp.exp((E+offE)/T)+1.0)
BoseEinstein    = lambda E,T: 1.0/(sp.exp((E+offE)/T)-1.0)
FermiDiracDeriv = lambda E,T: -(1.0/T)*sp.exp((E+offE)/T)/(sp.exp((E+offE)/T)+1.0)**2

def FillFD(En_A,T):
	""" fill an array with Fermi-Dirac distribution """
	N = int((len(En_A)-1)/2)
	sp.seterr(over='ignore') ## ignore overflow in exp, not important in this calculation
	if T == 0.0: FD_A = 1.0*sp.concatenate([sp.ones(N),[0.5],sp.zeros(N)])
	else:        FD_A = FermiDirac(En_A,T)
	sp.seterr(over='warn')
	return FD_A


def FillBE(En_A,T):
	""" fill an array with Bose-Einstein distribution """
	N = int((len(En_A)-1)/2)
	sp.seterr(over='ignore') ## ignore overflow in exp, not important in this calculation
	if T == 0.0: BE_A = -1.0*sp.concatenate([sp.ones(N),[0.5],sp.zeros(N)])
	else:        
		BE_A = BoseEinstein(En_A,T)
		BE_A[N] = -0.5
	sp.seterr(over='warn')
	return BE_A


## auxiliary functions ####################################

def KondoTemperature(U,Gamma,en):
	""" calculating Kondo temperature """
	if U!=0.0 and Gamma!=0.0:
		return sp.sqrt(U*Gamma/2.0)*sp.exp(-sp.pi*sp.fabs(U**2-4.0*en**2)/(8.0*U*Gamma))
	else:
		return 0.0


def IntDOS(GF_A,En_A):
	""" the integral over the DOS, should be 1.0 """
	TailL =  sp.imag(GF_A)[ 0]*En_A[ 0]/sp.pi	# left tail
	TailR = -sp.imag(GF_A)[-1]*En_A[-1]/sp.pi	# right tail
	return -simps(sp.imag(GF_A),En_A)/sp.pi+TailL+TailR


def Filling(GF_A,En_A,T):
	""" calculates filling, the integral over (-inf,inf) of f(w)G(w) """
	FD_A = FillFD(En_A,T)
	DOS_A = -FD_A*sp.imag(GF_A)/sp.pi
	## old tail: fit by a/x**2
	TailL1 = -DOS_A[0] *En_A[0]	# left tail
	TailR1 =  DOS_A[-1]*En_A[-1]	# right tail
	## new tail: fit by a/x**2+b/x**4
	## this fit depends on the length of the fitting interval
	#K = int(len(En_A)/100) 	## length of the fitting interval
	#a = lambda x1,x2,f1,f2:        (f1*x1**4-f2*x2**4)/(x1**2-x2**2)
	#b = lambda x1,x2,f1,f2,ab: 0.5*(f1*x1**4+f2*x2**4-ab*(x1**2+x2**2))
	#aL = a(En_A[ 0],En_A[ K],DOS_A[ 0],DOS_A[ K])
	#bL = b(En_A[ 0],En_A[ K],DOS_A[ 0],DOS_A[ K],aL)
	#aR = a(En_A[-K],En_A[-1],DOS_A[-K],DOS_A[-1])
	#bR = b(En_A[-K],En_A[-1],DOS_A[-K],DOS_A[-1],aR)
	#TailL2 = -aL/En_A[ 0]-bL/(3.0*En_A[ 0]**3)
	#TailR2 = -aR/En_A[-1]-bL/(3.0*En_A[-1]**3)
	#print('# - Filling(): tails: L1 {0: .8f},  R1 {1: .8f}'.format(TailL,TailR))
	#print('#                     L2 {0: .8f},  R2 {1: .8f}'.format(TailL2,TailR2))
	return simps(DOS_A,En_A) + TailL1 + TailR1

## functions to calculate spectral density ################

def GreensFunctionSemi(E,W):
	"""	local Green's function for semielliptic band """
	## small imaginary part helps to keep us on the right brach of the square root
	gz = E+1e-12j
	return (2.0*gz/W**2)*(1.0-sp.sqrt(1.0-W**2/gz**2))


def DensitySemi(x,W):
	""" particle denisty of a semi-elliptic band for T=0 """
	return sp.real(0.5 - x*sp.sqrt(W**2-x**2)/(sp.pi*W**2) - sp.arcsin(x/W)/sp.pi)


def GreensFunctionLorenz(E,Delta):
	"""	local Green's function for Lorentzian band """
	return 1.0/(E+1.0j*Delta)


def DensityLorentz(x,Delta):
	""" particle denisty of a Lorentzian band  for T=0 """
	return 0.5 - sp.arctan(x/Delta)/sp.pi


def GreensFunctionGauss(E,Gamma):
	"""	local Green's function for Gaussian band """
	return -1.0j*sp.sqrt(sp.pi/(2.0*Gamma**2))*wofz(E/sp.sqrt(2.0*Gamma**2))


def DensityGauss(x,Gamma):
	""" particle denisty of a Gaussian band  for T=0 """
	return (1.0+erf(x/sp.sqrt(2.0*Gamma**2)))/2.0


def GreensFunctionSquare(x,izero,W):
	""" local Green's function electrons on a 2D square lattice,
	using elliptic integral of the first kind K(z) from mpmath """
	from mpmath import ellipk
	K = sp.frompyfunc(ellipk,1,1)
	x = (1.0/W)*x+1.0j*izero
	return sp.array(2.0*K(1.0/x**2)/(sp.pi*x),dtype=complex)


def GreensFunctionSC(x,W):
	""" local Green's function electrons on a 3D sc lattice 
	using elliptic integral of the first kind K(z) calculated via hyp2f1
	Warning: precision issues with hyp2f1, use the ellipk from mpmath instead """
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


def ShiftGreensFunction(GF_F,En_F,shift):
	""" fill the GF array with GF shifted by real, static self-energy """
	ReGF = InterpolatedUnivariateSpline(En_F,sp.real(GF_F))
	ImGF = InterpolatedUnivariateSpline(En_F,sp.imag(GF_F))
	GF_F = ReGF(En_F+shift)+1.0j*ImGF(En_F+shift)
	return GF_F


def CalculateHWHM(GF_F,En_F):
	""" calculates the half-width at half-maximum of the Kondo resonance 
	and the maximum of the spectral function """
	N = len(En_F)
	dE = sp.around(En_F[1]-En_F[0],8)
	IntMin = int((N+1)/2-int(0.5/dE))
	IntMax = int((N+1)/2+int(0.5/dE))
	DOSmaxPos = sp.argmax(-sp.imag(GF_F[IntMin:IntMax])/sp.pi)
	DOSmax    = -sp.imag(GF_F[IntMin+DOSmaxPos])/sp.pi # maximum of DoS
	wmax      = En_F[IntMin+DOSmaxPos]                 # position of the maximum at energy axis
	DOS = InterpolatedUnivariateSpline(En_F-1e-12,-sp.imag(GF_F)/sp.pi-DOSmax/2.0) 
	## 1e-12 breaks symmetry for half-filling, otherway DOS.roots() loses one solution.
	DOSroots_F = sp.sort(sp.fabs(DOS.roots()))
	try:
		HWHM = (DOSroots_F[0] + DOSroots_F[1])/2.0
	except IndexError:
		HWHM = 0.0
	return [HWHM,DOSmax,wmax]


def QuasiPWeight(En_F,ReSE_F):
	""" calculating the Fermi-liquid quasiparticle weight (residue) Z """
	dE = En_F[1]-En_F[0]
	N = len(En_F)
	#M = int(1e-3/dE) if dE < 1e-3 else 1	# very fine grids lead to oscillations
	# replace 1 with M below to dilute the grid
	ReSE = UnivariateSpline(En_F[int(N/2-10):int(N/2+10):1],ReSE_F[int(N/2-10):int(N/2+10):1])
	dReSEdw = ReSE.derivatives(0.0)[1]
	Z = 1.0/(1.0-dReSEdw)
	return sp.array([Z,dReSEdw])

## bubbles and vertex functions ###########################

def KVertex(Lambda,Bubble_F):
	""" dynamical part of the two-particle vertex """
	## offE helps to prevent the 'RuntimeWarning: invalid value encountered in true_divide'
	## that causes problems in non-symmetric cases while calculating SigmaT in siam_static.py
	return -Lambda**2*Bubble_F/(1.0+Lambda*Bubble_F+offE)


def PsiInt(K_F,GFup_F,GFdn_F,En_F,T):
	""" integrating the static bubble \Psi = KxGxG, extended for finite temperatures"""
	BE_F = FillBE(En_F,T)
	Int_F = BE_F*sp.imag(sp.flipud(sp.conj(K_F*GFdn_F))*GFup_F)
	TailL = -Int_F[0]*En_F[0]/2.0
	return (simps(Int_F,En_F)+TailL)/sp.pi


def LambdaFunction(U,Lambda,Bubble_F,GFup_F,GFdn_F,En_F,T,chat):
	""" function to calculate new Lambda from previous iteration """
	K_F = KVertex(Lambda,Bubble_F)
	Psi = PsiInt(K_F,GFup_F,GFdn_F,En_F,T)
	Lambda = U/(1.0+Psi)
	#if chat: print('# - - Psi = {0: .8f}, Lambda = {1: .8f}'.format(Psi,Lambda))
	return Lambda


def CalculateLambda(U,Bubble_F,GFup_F,GFdn_F,En_F,T,chat,epsl):
	""" function to calculate new Lambda from previous iteration """
	Uc = -1.0/sp.real(Bubble_F[int(len(En_F)/2)])
	if Uc < 1e-6:
		print('# Warning: CalculateLambda: critical U is very small, please check the bubble.')	
	LMin = 0.0
	LMax = Uc-1e-10
	try:
		eqn = lambda x: x-LambdaFunction(U,x,Bubble_F,GFup_F,GFdn_F,En_F,T,chat)
		w = 0.0
		Lambda = brentq(eqn,LMin,LMax,xtol=epsl)
	except ValueError:
		print('# Error: CalculateLambda: brentq failed to calculate Lambda, probably too close to critical U.')
		exit()
	return Lambda

## susceptibilities #######################################

def SusceptibilityTherm(a,GF_F,En_F,T):
	""" susceptibility calculated from the thermal self-energy derivative """
	FD_F = FillFD(En_F,T)
	Int_F = FD_F*sp.imag(GF_F**2)
	## what about tail???
	return 2.0*simps(Int_F,En_F)/(a*sp.pi)


def SusceptibilitySpec(U,Lambda,X_F,GF_F,BubZero,En_F,T):
	""" susceptibility calculated from the spectral self-energy derivative """
	FD_F = FillFD(En_F,T)
	Int_F = FD_F*sp.imag(GF_F**2*(1.0-U*X_F/(1.0+Lambda*BubZero)))
	## what about tail???
	return 2.0*simps(Int_F,En_F)/sp.pi


def SusceptibilityHF(U,GF_F,X_F,En_F,T):
	""" susceptibility calculated from the full spectral self-energy derivative """
	FD_F = FillFD(En_F,T)
	Int1_F = FD_F*sp.imag(GF_F**2*(1.0-U*X_F))
	Int2_F = FD_F*sp.imag(GF_F**2*X_F)
	I1 = simps(Int1_F,En_F)/sp.pi
	I2 = simps(Int2_F,En_F)/sp.pi
	return 2.0*I1/(1.0+U**2*I2)


def AFbubble(En_F,GFzero_F,MuBar,T):
	""" non-local bubble for k=(pi,pi,pi) - antiferromagnetic case for sc lattice """
	FD_F = FillFD(En_F-MuBar,T)
	F_F = -sp.imag(GFzero_F)*FD_F/(sp.pi*En_F)
	F_F[int(len(En_F)/2)] = 0.0 # we are taking the principal value
	return simps(F_F,En_F)


def SpecHWHM(GFint_F,En_F):
	""" Half-width at half-maximum of the spectral function """
	N = len(En_F)
	DOSF = -sp.imag(GFint_F[N/2])/sp.pi	# value at Fermi energy
	DOS = InterpolatedUnivariateSpline(En_F,-sp.imag(GFint_F)/sp.pi-DOSF/2.0)
	return sp.amin(sp.fabs(DOS.roots()))


## convolutions in Matsubara frequencies ##################

def TwoParticleBubble(F1_F,F2_F,En_F,T,channel):
	""" calculates the two-particle bubble, channel = 'eh', 'ee' """
	N = int((len(En_F)-1)/2)
	dE = sp.around(En_F[1]-En_F[0],8)
	FD_F = FillFD(En_F,T)
	## zero-padding the arrays
	FD_F    = sp.concatenate([FD_F[N:],sp.zeros(2*N+3),FD_F[:N]])
	ImF1_F = sp.concatenate([sp.imag(F1_F[N:]),sp.zeros(2*N+3),sp.imag(F1_F[:N])])
	ImF2_F = sp.concatenate([sp.imag(F2_F[N:]),sp.zeros(2*N+3),sp.imag(F2_F[:N])])
	## performing the convolution
	if channel == 'eh':
		ftImChi1_F =  sp.conj(fft(FD_F*ImF2_F))*fft(ImF1_F)*dE		# f(x)F2(x)F1(w+x)
		ftImChi2_F = -fft(FD_F*ImF1_F)*sp.conj(fft(ImF2_F))*dE		# f(x)F1(x)F2(x-w)
	elif channel == 'ee':
		ftImChi1_F =  fft(FD_F*ImF2_F)*fft(ImF1_F)*dE					# f(x)F2(x)F1(w-x)
		ftImChi2_F = -sp.conj(fft(FD_F*sp.flipud(ImF1_F)))*fft(ImF2_F)*dE	# f(x)F1(-x)F2(w+x)
	ImChi_F = -sp.real(ifft(ftImChi1_F+ftImChi2_F))/sp.pi
	ImChi_F = sp.concatenate([ImChi_F[3*N+4:],ImChi_F[:N+1]])
	Chi_F = KramersKronigFFT(ImChi_F) + 1.0j*ImChi_F
	return Chi_F


def SelfEnergy(GF_F,ChiGamma_F,En_F,T):
	""" calculating the dynamical self-energy from the Schwinger-Dyson equation """
	N = int((len(En_F)-1)/2)
	dE = sp.around(En_F[1]-En_F[0],8)
	FD_F = FillFD(En_F,T)
	BE_F = FillBE(En_F,T)
	## zero-padding the arrays
	FD_F   = sp.concatenate([FD_F[N:],sp.zeros(2*N+3),FD_F[:N]])
	BE_F   = sp.concatenate([BE_F[N:],sp.zeros(2*N+3),BE_F[:N]])
	ImGF_F = sp.concatenate([sp.imag(GF_F[N:]),sp.zeros(2*N+3),sp.imag(GF_F[:N])])
	ImCG_F = sp.concatenate([sp.imag(ChiGamma_F[N:]),sp.zeros(2*N+3),sp.imag(ChiGamma_F[:N])])
	## performing the convolution
	ftImSE1_F = -sp.conj(fft(BE_F*ImCG_F))*fft(ImGF_F)*dE
	ftImSE2_F = -fft(FD_F*ImGF_F)*sp.conj(fft(ImCG_F))*dE
	ImSE_F = sp.real(ifft(ftImSE1_F+ftImSE2_F))/sp.pi
	ImSE_F = sp.concatenate([ImSE_F[3*N+4:],ImSE_F[:N+1]])
	Sigma_F = KramersKronigFFT(ImSE_F) + 1.0j*ImSE_F
	return Sigma_F


def KramersKronigFFT(ImX_F):
	"""	Hilbert transform used to calculate real part of a function from its imaginary part
	uses piecewise cubic interpolated integral kernel of the Hilbert transform
	use only if len(ImX_F)=2**m-1, uses fft from scipy.fftpack  """
	X_F = sp.copy(ImX_F)
	N = int(len(X_F))
	## be careful with the data type, orherwise it fails for large N
	if N > 3e6: A = sp.arange(3,N+1,dtype='float64')
	else:       A = sp.arange(3,N+1)  
	X1 = 4.0*sp.log(1.5)
	X2 = 10.0*sp.log(4.0/3.0)-6.0*sp.log(1.5)
	## filling the kernel
	if N > 3e6: Kernel_F = sp.zeros(N-2,dtype='float64')
	else:       Kernel_F = sp.zeros(N-2)
	Kernel_F = (1-A**2)*((A-2)*sp.arctanh(1.0/(1-2*A))+(A+2)*sp.arctanh(1.0/(1+2*A)))\
	+((A**3-6*A**2+11*A-6)*sp.arctanh(1.0/(3-2*A))+(A+3)*(A**2+3*A+2)*sp.arctanh(1.0/(2*A+3)))/3.0
	Kernel_F = sp.concatenate([-sp.flipud(Kernel_F),sp.array([-X2,-X1,0.0,X1,X2]),Kernel_F])/sp.pi
	## zero-padding the functions for fft
	ImXExt_F = sp.concatenate([X_F[int((N-1)/2):],sp.zeros(N+2),X_F[:int((N-1)/2)]])
	KernelExt_F = sp.concatenate([Kernel_F[N:],sp.zeros(1),Kernel_F[:N]])
	## performing the fft
	ftReXExt_F = -fft(ImXExt_F)*fft(KernelExt_F)
	ReXExt_F = sp.real(ifft(ftReXExt_F))
	ReX_F = sp.concatenate([ReXExt_F[int((3*N+3)/2+1):],ReXExt_F[:int((N-1)/2+1)]])
	return ReX_F


def XIntegralsFFT(GF_F,Bubble_F,Lambda,BubZero,T,En_F):
	""" calculate X integral using FFT """
	N = int((len(En_F)-1)/2)
	dE = sp.around(En_F[1]-En_F[0],8)
	Kappa_F  = TwoParticleBubble(GF_F,GF_F**2,En_F,T,'eh')
	Bubble_F = TwoParticleBubble(GF_F,GF_F,En_F,T,'eh')
	V_F = 1.0/(1.0+Lambda*Bubble_F)
	KV_F = Lambda*Kappa_F*V_F**2
	KmV_F = Lambda*sp.flipud(sp.conj(Kappa_F))*V_F**2
	FD_F = FillFD(En_F,T)
	## zero-padding the arrays
	FD_F = sp.concatenate([FD_F[N:],sp.zeros(2*N+2),FD_F[:N+1]])
	ImGF_F  = sp.concatenate([sp.imag(GF_F[N:]),sp.zeros(2*N+2),sp.imag(GF_F[:N+1])])
	ImGF2_F = sp.concatenate([sp.imag(GF_F[N:]**2),sp.zeros(2*N+2),sp.imag(GF_F[:N+1]**2)])
	ImV_F = sp.concatenate([sp.imag(V_F[N:]),sp.zeros(2*N+2),sp.imag(V_F[:N+1])])
	ImKV_F = sp.concatenate([sp.imag(KV_F[N:]),sp.zeros(2*N+2),sp.imag(KV_F[:N+1])])
	ImKmV_F = sp.concatenate([sp.imag(KmV_F[N:]),sp.zeros(2*N+2),sp.imag(KmV_F[:N+1])])
	## performing the convolution
	ftImX11_F = -sp.conj(fft(FD_F*ImV_F))*fft(ImGF2_F)*dE
	ftImX12_F = fft(FD_F*ImGF2_F)*sp.conj(fft(ImV_F))*dE
	ftImX21_F = -sp.conj(fft(FD_F*ImKV_F))*fft(ImGF_F)*dE
	ftImX22_F = fft(FD_F*ImGF_F)*sp.conj(fft(ImKV_F))*dE
	ftImX31_F = -sp.conj(fft(FD_F*ImKmV_F))*fft(ImGF_F)*dE
	ftImX32_F = fft(FD_F*ImGF_F)*sp.conj(fft(ImKmV_F))*dE
	## inverse transform
	ImX1_F =  sp.real(ifft(ftImX11_F+ftImX12_F))/sp.pi
	ImX2_F =  sp.real(ifft(ftImX21_F+ftImX22_F))/sp.pi
	ImX3_F = -sp.real(ifft(ftImX31_F+ftImX32_F))/sp.pi
	ImX1_F = sp.concatenate([ImX1_F[3*N+4:],ImX1_F[:N+1]])
	ImX2_F = sp.concatenate([ImX2_F[3*N+4:],ImX2_F[:N+1]])
	ImX3_F = sp.concatenate([ImX3_F[3*N+4:],ImX3_F[:N+1]])
	## getting real part from imaginary
	X1_F = KramersKronigFFT(ImX1_F) + 1.0j*ImX1_F + BubZero # constant part !!!
	X2_F = KramersKronigFFT(ImX2_F) + 1.0j*ImX2_F
	X3_F = KramersKronigFFT(ImX3_F) + 1.0j*ImX3_F
	return [X1_F,X2_F,X3_F]


## output functions #######################################

def WriteFile(En_F,X1_F,X2_F,X3_F,WriteMax,de_dec,header,filename,chat):
	""" writes data arrays to file, writes three complex arrays at once """
	f = open(filename,'w')
	f.write(header+'\n')
	for i in range(len(En_F)):
		if sp.fabs(En_F[i]) <= WriteMax and sp.fabs(En_F[i] - sp.around(En_F[i],de_dec)) == 0:
			f.write('{0: .6f}\t{1: .6f}\t{2: .6f}\t{3: .6f}\t{4: .6f}\t{5: .6f}\t{6: .6f}\n'\
			.format(float(En_F[i]),float(sp.real(X1_F[i])),float(sp.imag(X1_F[i])),float(sp.real(X2_F[i]))\
			,float(sp.imag(X2_F[i])),float(sp.real(X3_F[i])),float(sp.imag(X3_F[i]))))
	f.close
	if chat: print('#   File '+filename+' written.')


def WriteFile2(En_F,X1_F,X2_F,X3_F,X4_F,WriteMax,de_dec,header,filename,chat):
	""" writes data arrays to file, writes four complex arrays at once """
	f = open(filename,'w')
	f.write(header+'\n')
	for i in range(len(En_F)):
		if sp.fabs(En_F[i]) <= WriteMax and sp.fabs(En_F[i] - sp.around(En_F[i],de_dec)) == 0:
			f.write('{0: .6f}\t{1: .6f}\t{2: .6f}\t{3: .6f}\t{4: .6f}\t{5: .6f}\t{6: .6f}\t{7: .6f}\t{8: .6f}\n'\
			.format(float(En_F[i]),float(sp.real(X1_F[i])),float(sp.imag(X1_F[i])),float(sp.real(X2_F[i]))\
			,float(sp.imag(X2_F[i])),float(sp.real(X3_F[i])),float(sp.imag(X3_F[i]))\
			,float(sp.real(X4_F[i])),float(sp.imag(X4_F[i]))))
	f.close
	if chat: print('#   File '+filename+' written.')

## parlib.py end ###

