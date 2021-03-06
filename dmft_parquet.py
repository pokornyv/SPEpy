###########################################################
# SPEpy - simplified parquet equation solver for SIAM     #
# Copyright (C) 2019  Vladislav Pokorny; pokornyv@fzu.cz  #
# homepage: github.com/pokornyv/SPEpy                     #
# dmft_parquet.py - solver for Hubbard model using DMFT   #
# self-consistency and static parqet solver               #
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

from sys import argv,exit,version_info
from os import listdir,remove
from time import ctime

import scipy as sp
from scipy.optimize import fixed_point,brentq
from scipy.interpolate import InterpolatedUnivariateSpline

from parlib import *
from config_dmft import *

method = 'SPE'
#method = '2nd'

hashes = '#'*80

ver = str(version_info[0])+'.'+str(version_info[1])+'.'+str(version_info[2])
if chat: 
	print(hashes+'\n# DMFT solver for one-band Hubbard model using simplified parquet equations')
	print('# generated by '+str(argv[0])+', '+str(ctime()))
	print('# python version: '+str(ver)+', SciPy version: '+str(sp.version.version))
	print('# U = {0: .4f}, Delta = {1: .4f}, eps = {2: .4f}, T = {3: .4f}'.format(U,Delta,ef,T))
	print('# energy axis: [{0: .5f} ..{1: .5f}], step = {2: .5f}, length = {3: 3d}'\
      .format(En_A[0],En_A[-1],dE,len(En_A)))
	print("# Kondo temperature Tk ~{0: .5f}".format(float(KondoTemperature(U,Delta,ef))))
	print("# mixing parameter alpha ={0: .5f}".format(float(alpha)))

	if   method == 'SPE': print("# using simplified parquet equation (SPE) solver")
	elif method == '2nd': print("# using 2nd-order PT solver")	

#####################################################################
## define the non-interacting DoS and allocate memory for arrays

if chat: print('# using semielliptic non-interacting DoS')
W = Delta # half-bandwidth for semielliptic DoS
GFlambda = lambda x: GreensFunctionSemi(x,W)
GFzero_F = GFlambda(En_A)

if chat: print('# norm[G0]: {0: .6f}, n[G0]: {1: .6f}'\
.format(float(IntDOS(GFzero_F,En_A)),float(Filling(GFzero_F,En_A))))

GFint_F = sp.copy(GFzero_F)
SE_F    = sp.zeros_like(GFzero_F)
SEold_F = sp.zeros_like(GFzero_F)

#####################################################################
## start DMFT loop 
LambdaOld = 1e5
for NLoop in range(NStart,NStart+NIter):
	if chat: print ('#\n'+'#'*40+'\n# DMFT iteration {0: 3d} /{1: 3d}'.format(NLoop,NStart+NIter-1))
	iter_file = open('iterations.dat','a')
	if NLoop == 1: 
		iter_file.write('# '+argv[0]+': New calculation started on '+ctime()+'\n')
		iter_file.write('# iter\tLambda\t\ta\t\tDosInt\t\tDosF\t\tHWHM\t\tUc\t\tConv(Lambda)\n')

	## load data if necessary
	if NStart > 1 and NLoop == NStart:
		fname_in = 'dmft_'+str(NLoop-1)+'.npz'
		if chat: print('# loading interacting GF and self-energy from file')
		GFint_F = sp.load(fname_in)['GFint_F']
		try:                      
			SEold_F = sp.load(fname_in)['SE_F']
		except FileNotFoundError: 
			print('File not found, using zero self-energy.')
			SEold_F = sp.zeros_like(GFzero_F)
		try: 
			LambdaOld = sp.load(fname_in)['Lambda']
		except KeyError: 
			pass

	## calculate bath GF from lattice Dyson equation
	GFbath_F = 1.0/(1.0/GFint_F + SEold_F)
	#WriteFileX([En_A,GFzero_F,GFbath_F,GFbath_F],WriteMax,WriteStep,'bath.dat',chat)

	## claculate the bubble
	if chat: print('# calculating the two-particle bubble...')
	Bubble_F = TwoParticleBubble(GFbath_F,GFbath_F,En_A,'eh')
	BubZero = Bubble_F[int(N/2)]
	Uc = -1.0/sp.real(BubZero)
	if chat: 
		print('# Bubble[0] = {0: .5f} + {1: .5f}i, critical U = {2: .5f}'\
		.format(float(sp.real(BubZero)),float(sp.imag(BubZero)),float(Uc)))

	## calculate Lambda part of the vertex
	if method == 'SPE':
		Lambda = CalculateLambda(U,Bubble_F,GFbath_F,GFbath_F,En_A,chat,epsl)
		if chat: print('# Lambda = {0: .5f}'.format(Lambda))
		a = 1.0 + Lambda*sp.real(BubZero)
		DLambda = sp.fabs(Lambda-LambdaOld)
		print('# Convergence: |Lambda - LambdaOld| ={0: .8f}'.format(DLambda))
		LambdaOld = Lambda
	else:
		Lambda = LambdaOld = DLambda = 0.0
		a = 1.0

	## calculate the self-energy from SD equation
	if chat: print('# calculating self-energy and the interacting Green function...')
	if method == 'SPE':
		K_F = KVertex(Lambda,Bubble_F)
		ChiDelta_F = U*Bubble_F*(K_F + Lambda)
	elif method == '2nd':
		ChiDelta_F = U**2*Bubble_F
	SE_F = SelfEnergy(GFbath_F,ChiDelta_F,En_A)

	## mix self-energy with previous iteration
	SE_F = (alpha*SE_F + (1.0-alpha)*SEold_F)
	SEold_F = sp.copy(SE_F)

	## calculate the interacting GF from local Dyson equation
	GFint_F = GFlambda(En_A-SE_F)
	DOSF = -sp.imag(GFint_F[int(N/2)])/sp.pi
	Norm = IntDOS(GFint_F,En_A)
	[HWHM,DOSmax,wmax] = CalculateHWHM(GFint_F,En_A)
	if chat: print('# Int A(w)dw (int) = {0: .5f}, DOS[0] = {1: .5f}, HWHM = {2: .6f}'\
	.format(float(Norm),float(DOSF),float(HWHM)))

	## write intermediate step to file
	if WriteFiles:
		filename = 'gf_iter'+str(NLoop)+'.dat'
		WriteFileX([En_A,GFbath_F,SE_F,GFint_F],WriteMax,WriteStep,'gf',filename,chat)
	## save int. GF and SE in case we want to continue iterations and remove an old one (npz files are large)
	sp.savez_compressed('dmft_'+str(NLoop),GFint_F = GFint_F, SE_F = SE_F, Lambda = Lambda)
	rm_filename = 'dmft_'+str(NLoop-2)+'.npz'
	if rm_filename in listdir('.'):
		print('# Removing npz file from iteration {0: 3d} to save disk space.'.format(NLoop-2))
		remove(rm_filename)
	## write data about current iteration fo file
	iter_file.write('{0: 3d}\t{1: .6f}\t{2: .6f}\t{3: .6f}\t{4: .6f}\t{5: .6f}\t{6: .6f}\t{7: .6f}\n'\
	.format(NLoop,Lambda,float(a),float(Norm),float(DOSF),float(HWHM),float(Uc),float(DLambda)))
	iter_file.close()

## write the final GF to file
filename = 'gf_U'+str(U)+'_dmft.dat'
WriteFileX([En_A,GFbath_F,SE_F,GFint_F],WriteMax,WriteStep,'gf',filename,chat)

print('{0: .3f}\t{1: .3f}\t{2: .3f}\t{3: .3f}\t{4: .6f}\t{5: .6f}\t{6: .6f}\t{7: .6f}\t{8: .6f}\t{9: .6f}\n'\
.format(U,Delta,ef,T,Lambda,float(a),float(Norm),float(DOSF),float(HWHM),float(DLambda)))

if chat: print('# '+argv[0]+' DONE.')

## dmft_parquet.py end ###

