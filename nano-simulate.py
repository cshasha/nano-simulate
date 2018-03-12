import numpy as np
import pylab as pl
import itertools
from numpy.matlib import repmat, repeat
import numpy.ma as ma
from math import factorial
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")
import os.path
from decimal import Decimal
import sys
from functools import reduce
#from mpl_toolkits.mplot3d import axes3d


#---about output:
# a .csv file will be saved after this simulation finishes. the first column records
# the applied field (in mT), and the second column records the average z-component
# of the particle magnetization. a .txt file is also written with the same name, containing 
# the values of the input parameters. the M(H) curve is plotted once the simulation finishes.
# to turn off, comment out "pl.show()", the last line of the program. the iteration number
# is printed (up to X)

w = float(sys.argv[1])

#---parameters---
I = 10   #number of particles
N = 1000   #number of time steps (min 10000-1000000). dt = 1e-8 at least (ideally 1e-10)
X = 1   #number of repetitions

cluster = 0   #0 = no cluster (random). 1 = chain. 2 = + (I = 5)

minT = 10   #minimum temperature
maxT = 80   #maximum temperature
tstep = 0.5   #temp step

"""
minF = 3   #100
maxF = 8   #10000000
Fsteps = 40
freqs = np.logspace(minF,maxF,num=Fsteps)
"""

save = "test"  #filename to save
text = save + ".txt"
save1 = save + ".csv"
#save2 = "hyst_16nm_20mT_frozen_2D_z.csv"   #filename.csv
interactions = "off"   #"on" or "off"
brownian = "on"   #on or off
aligned = "no"   #yes or no. if yes, sets all axes and moments to z direction

#---constants---
kb = 1.381e-23   #boltzmann constant
mu0 = 1e-7   #mu0/4pi 

#C = 1e15   #concentration

h0 = .02  #applied field amp (T/mu0). 10 Oe = 0.001 T
f = 25500   #frequency (Hz)
cycs = 2   #number of field cycles
twoD = "off"   #on or off
N *= cycs

num_sizes = 1   #number of different sizes (1 or 2)
diam1 = 21.9e-9   #average particle diameter 1
diam2 = 25e-9   #second average diameter
sigma = .09   #polydispersity index. typical good: 0.05
#s = 3.2e-9
#sigma = np.sqrt(np.log(1 + s**2*diam1**(-2)))
coating = 35e-9   #added diameter due to coating

shape = "c"   #either "u" for uniaxial or "c" for cubic

tauPercent = 0.1

C = np.sqrt(3)**3*(2*(diam1+coating))**(-3)

KB = -13000
KS = -5.1e-5
K = KB + 6*KS/diam1   #anisotropy constant (J/m^3). 1 J/m^3 = 10 erg/cm^3
K2 = -5000.
k_sigma = 0.2 #variance for k values
rho = 4.9e6   #np density (g/m^3)
#gamma = 1.3e9   #gyromag ratio (Hz/T) 1.3e9
gyro = 1.76e11
lam = 1   #damping
Ms = 420000.   #saturation magnetization in A/m (420000 for Magnetite) scaled - bulk values 476k magnetite, 426k maghemite. 367.5?
M0 = 491511.   #magnetization at 0K, for Bloch's law
a = 1.5   #values for Bloch's law
b = 2.8e-5   #values for Bloch's law
eta = 8.9e-4   #viscosity. water: 8.9e-4 in Pa*s = kg/m/s. viscosity of air = 1.81e-5
H_k = 2*np.abs(K)/(Ms)
gamma = gyro*H_k/(2*np.pi)

temperature = 300.

np.savetxt(text, ["I = %d\nN = %d\nX = %d\ncluster=%d\ninteractions=%s\nbrownian=%s\naligned=%s\nconcentration=%E\nh0=%f\nf=%f\ncycs=%d\ntwoD=%s\nnumsizes=%d\ndiam1=%E\ndiam2=%E\nsigma=%f\ncoating=%E\nshape=%s\nK=%d\nlambda=%f\nviscosity=%E" % (I,N,X,cluster,interactions,brownian,aligned,C,h0,f,cycs,twoD,num_sizes,diam1,diam2,sigma,coating,shape,K,lam,eta)], fmt='%s')

if os.path.isfile(save1):
	print('file exists')
	

#---values---
#d = (1+lam**2)*kb*(gamma*Ms*lam)**(-1)   #equal to DV/T, if Ms is constant
L = (I*C**(-1))**(1/3.)   #length of one box side
rAvg = 3**(1/2.)*C**(-1/3.)  #estimated avg interparticle distance

w = 2*np.pi*f   #angular frequency (Rad/sec). constraint: f*N must be integer
dt = cycs*(f*N)**(-1)   #time step
#dt = 1e-12
T = np.arange(0,dt*N,dt)   #array of time steps


#---initialization---
M = np.zeros((I,3,N))   #initialize moment matrix
An = np.zeros((I,3))   #initialize anisotropy matrix
nx = np.zeros((I,3))
ny = np.zeros((I,3))
Axes = np.zeros((I,3,N))   #initialize anisotropy axes

XX1 = np.zeros((X,int((maxT-minT)/tstep)+1,2))   #initialize matrices of susceptibilities
XX2 = np.zeros((X,int((maxT-minT)/tstep)+1,2))
#XF1 = np.zeros((X,Fsteps,2))   #initialize matrices of susceptibilities
#XF2 = np.zeros((X,Fsteps,2))

sizes = np.zeros(I)   #create size distribution
if num_sizes == 1:
	sizes = np.random.lognormal(np.log(diam1), sigma, I)
if num_sizes == 2:
	sizes[0:I/2.] = np.random.lognormal(np.log(diam1), sigma, I/2.)
	sizes[I/2.:I] = np.random.lognormal(np.log(diam2), sigma, I/2.)

volumes = (4/3.)*np.pi*(sizes/2.)**3   #create array of volumes
h_volumes = (4/3.)*np.pi*((sizes+coating)/2.)**3   #hydrodynamic volumes (calculate later)

k_values = np.random.lognormal(np.log(abs(K)), k_sigma, I)   #create array of k values
k_values *= np.sign(K)
k_values2 = np.random.lognormal(np.log(abs(K2)), k_sigma, I)
k_values2 *= np.sign(K2)
betas = Ms**(-1)*k_values   #create array of betas if Ms is constant
betas2 = Ms**(-1)*k_values2   #create array of betas

def initialize():
	global M 
	global mcGhost
	global M_coords
	global H_app_z
	global H_app_y
	global H_dip 
	global An
	global nx, ny
	global R 
	global RR 
	global rBar
	global Start
	global Axes
	global aStart

	if cluster == 0:
		M_coords = np.random.rand(I,3)*L   #positions of particles. need to add limits
	if cluster == 1:
		M_coords = np.zeros((I,3))
		c_theta = np.random.rand(1)*np.pi/2.
		c_phi = np.random.rand(1)*np.pi/2.
		#fig = pl.figure()
		#ax = fig.add_subplot(111, projection='3d')
		for c in range(I):
			#M_coords[c,2] = c*diam1
			#M_coords[c,0] = c*diam1
			M_coords[c,0] = c*diam1*np.sin(c_theta)*np.cos(c_phi)
			M_coords[c,1] = c*diam1*np.sin(c_theta)*np.sin(c_phi)
			M_coords[c,2] = c*diam1*np.cos(c_theta)
			#ax.scatter(M_coords[c,0], M_coords[c,1], M_coords[c,2], c='m')

		#pl.show()

	if cluster == 2:
		M_coords = np.zeros((I,3))
		M_coords[0,0] = diam1
		M_coords[1,1] = diam1
		M_coords[2,0] = diam1
		M_coords[2,1] = diam1
		M_coords[3,0] = 2*diam1
		M_coords[3,1] = diam1
		M_coords[4,0] = diam1
		M_coords[4,1] = 2*diam1

	mcGhost = M_coords[:]

	M_theta = np.random.rand(I)*np.pi   #theta
	M_phi = np.random.rand(I)*2*np.pi   #phi
	if aligned == "yes":
		M[:,0,0] = 0
		M[:,1,0] = 0
		M[:,2,0] = 1
	else:
		M[:,0,0] = np.sin(M_theta[:])*np.cos(M_phi[:]) #random orientations
		M[:,1,0] = np.sin(M_theta[:])*np.sin(M_phi[:])
		M[:,2,0] = np.cos(M_theta[:])

	Start = np.copy(M[:,:,0])   #preserves initial conditions

	An_theta = np.random.rand(I)*np.pi   #theta (n)
	An_phi = np.random.rand(I)*2*np.pi   #phi (n)
	if aligned == "yes":
		An[:,0] = 0
		An[:,1] = 0
		An[:,2] = 1
	elif aligned == "y":
		An[:,0] = 0
		An[:,1] = 1
		An[:,2] = 0
	elif aligned == "y50":
		An[:I/2.,0] = 0
		An[:I/2.,1] = 1
		An[:I/2.,2] = 0
		An[I/2.:,0] = np.sin(An_theta[I/2.:])*np.cos(An_phi[I/2.:]) #random orientations
		An[I/2.:,1] = np.sin(An_theta[I/2.:])*np.sin(An_phi[I/2.:])
		An[I/2.:,2] = np.cos(An_theta[I/2.:])
	else:
		An[:,0] = np.sin(An_theta[:])*np.cos(An_phi[:]) #random orientations
		An[:,1] = np.sin(An_theta[:])*np.sin(An_phi[:])
		An[:,2] = np.cos(An_theta[:])

	if aligned == "yes":
		nx[:,0] = 1
		nx[:,1] = 0
		nx[:,2] = 0
		ny[:,0] = 0
		ny[:,1] = 1
		ny[:,2] = 0
	else:
		R_theta = np.random.rand(I)*np.pi   #theta (n)
		R_phi = np.random.rand(I)*2*np.pi   #phi (n)	
		for i in range(I):
			R_n = np.array([np.sin(R_theta[i])*np.cos(R_phi[i]),np.sin(R_theta[i])*np.sin(R_phi[i]),np.cos(R_theta[i])])
			ny[i,:] = np.cross(An[i,:],R_n)
			nx[i,:] = np.cross(An[i,:],ny[i,:])

	Axes[:,:,0] = An
	aStart = np.copy(Axes[:,:,0])

	H_dip = np.zeros((I,3))
	H_app_z = h0*np.cos(w*T)
	if twoD == "on":
		w2 = 1.05*w
		H_app_y = h0*np.sin(w2*T)
	else:
		H_app_y = 0*np.sin(w*T)
	
	#H_app_z = np.zeros(N)
	#H_app_z.fill(h0)
	
	#H_app[0:N/5.].fill(h0)
	#H_app[N/5.:N].fill(0)

	#---make ghost coordinate matrix
	def makecGhost(mcGhost):
		global g_mask_x1, g_mask_x2, g_mask_y1, g_mask_y2, g_mask_z1, g_mask_z2

		g_mask_x1 = M_coords[:,0] < rAvg
		g_mask_x2 = M_coords[:,0] > L-rAvg

		j1 = mcGhost[g_mask_x1]
		j1[:,0] += L
		j2 = mcGhost[g_mask_x2]
		j2[:,0] -= L

		mcGhost = np.vstack((mcGhost,j1))
		mcGhost = np.vstack((mcGhost,j2))

		g_mask_y1 = mcGhost[:,1] < rAvg
		g_mask_y2 = mcGhost[:,1] > L-rAvg

		k1 = mcGhost[g_mask_y1]
		k1[:,1] += L
		k2 = mcGhost[g_mask_y2]
		k2[:,1] -= L

		mcGhost = np.vstack((mcGhost,k1))
		mcGhost = np.vstack((mcGhost,k2))

		g_mask_z1 = mcGhost[:,2] < rAvg
		g_mask_z2 = mcGhost[:,2] > L-rAvg

		l1 = mcGhost[g_mask_z1]
		l1[:,2] += L
		l2 = mcGhost[g_mask_z2]
		l2[:,2] -= L

		mcGhost = np.vstack((mcGhost,l1))
		mcGhost = np.vstack((mcGhost,l2))

		return mcGhost
	
	#---create distance matrix. stays fixed
	mcGhost = makecGhost(mcGhost)

	numPoints = len(mcGhost)
	dM = repmat(mcGhost, numPoints, 1) - repeat(mcGhost, numPoints, axis=0)
	RR = dM.reshape((numPoints, numPoints, 3))
	R = np.sqrt(np.sum(RR**2, axis = 2))

	rBar = np.average(R[:,:], weights = (R[:,:]>0))


#---make ghost matrix of moments
def makeGhost(mcGhost, n):
	mGhost = M[:,:,n]

	mX1 = np.copy(mGhost[g_mask_x1])
	mX2 = np.copy(mGhost[g_mask_x2])

	mGhost = np.vstack((mGhost,mX1))
	mGhost = np.vstack((mGhost,mX2))

	mY1 = np.copy(mGhost[g_mask_y1])
	mY2 = np.copy(mGhost[g_mask_y2])

	mGhost = np.vstack((mGhost,mY1))
	mGhost = np.vstack((mGhost,mY2))

	mZ1 = np.copy(mGhost[g_mask_z1])
	mZ2 = np.copy(mGhost[g_mask_z2])

	mGhost = np.vstack((mGhost,mZ1))
	mGhost = np.vstack((mGhost,mZ2))

	return mGhost

#---thermalizes for N/5 steps
def thermalize(temp):
	#Ms = M0*(1 - b*temp**a)
	global nx, ny
	betas = Ms**(-1)*k_values
	betas2 = Ms**(-1)*k_values2
	d = (1+lam**2)*kb*(gamma*Ms*lam)**(-1) 
	for n in range(int(N/3)):
		dW = np.random.normal(loc = 0.0, scale = 1, size = (I,3))
		dW *= np.sqrt(dt*2*d*temp*volumes[:,None]**(-1))

		if brownian == "on":
			dT = np.random.normal(loc = 0.0, scale = 1, size = (I,3))
			dT *= np.sqrt(dt*12*kb*temp*eta*h_volumes[:,None])
				
		if shape == "u":
			if brownian == "on":
				Torque = -2*(k_values*volumes)[:,None]*np.absolute(np.sum(M[:,:,n]*Axes[:,:,n],axis=1))[:,None]*np.cross(Axes[:,:,n],M[:,:,n])

			H = 2*betas[:,None]*np.sum(M[:,:,n]*Axes[:,:,n],axis=1)[:,None]*Axes[:,:,n]
		else:
			mx = np.sum(M[:,:,n]*nx,axis=1)[:,None]
			my = np.sum(M[:,:,n]*ny,axis=1)[:,None]
			mz = np.sum(M[:,:,n]*Axes[:,:,n],axis=1)[:,None]
			if brownian == "on":
				Torque = -2*(k_values*volumes)[:,None]*(my**2 + mx**2)*mz*np.cross(Axes[:,:,n],M[:,:,n]) - 2*(k_values2*volumes)[:,None]*my**2*mx**2*mz*np.cross(Axes[:,:,n],M[:,:,n]) \
					     -2*(k_values*volumes)[:,None]*(my**2 + mz**2)*mx*np.cross(nx,M[:,:,n]) - 2*(k_values2*volumes)[:,None]*my**2*mz**2*mx*np.cross(nx,M[:,:,n]) \
					     -2*(k_values*volumes)[:,None]*(mz**2 + mx**2)*my*np.cross(ny,M[:,:,n]) - 2*(k_values2*volumes)[:,None]*mz**2*mx**2*my*np.cross(ny,M[:,:,n])

			H = -betas[:,None]*(mx**2*mz*Axes[:,:,n] + my**2*mz*Axes[:,:,n] + mx**2*my*ny + mz**2*my*ny + my**2*mx*nx + mz**2*mx*nx) - 2*betas2[:,None]*(mx**2*my**2*mz*Axes[:,:,n] + mx**2*mz**2*my*ny + my**2*mz**2*mx*nx)

		if interactions == "on":
			mGhost = makeGhost(mcGhost,n)
			masks = [R == 0]
			z = reduce(np.logical_or, masks)
			y = np.zeros((len(mcGhost),len(mcGhost),3))
			y[:,:,0] = z
			y[:,:,1] = z
			y[:,:,2] = z
			Q = ma.masked_array((mu0*diam1**3/8.*Ms*(3*np.sum(RR*mGhost,axis=2)[:,:,None]*RR/R[:,:,None]**2 - mGhost))/R[:,:,None]**3, mask=y, fill_value = 0)
			H_dip = np.sum(Q.filled(0)[0:I,:,:],axis=1)
			H += H_dip
		
		mb = M[:,:,n] + gamma*(1+lam**2)**(-1)*((np.cross(M[:,:,n],H) - lam*np.cross(M[:,:,n], np.cross(M[:,:,n],H)))*dt + np.cross(M[:,:,n],dW) - lam*np.cross(M[:,:,n], np.cross(M[:,:,n],dW)))
		norm_m = np.sqrt(np.einsum('...i,...i', mb, mb))
		mb /= norm_m[:,None]

		if brownian == "on":
			nb = Axes[:,:,n] + (6*eta*h_volumes[:,None])**(-1)*(np.cross(Torque,Axes[:,:,n])*dt + np.cross(dT,Axes[:,:,n]))
			norm_n = np.sqrt(np.einsum('...i,...i', nb, nb))
			nb /= norm_n[:,None]
			if shape == "u":
				tbar = 2*(k_values*volumes)[:,None]*np.absolute(np.sum(mb*nb,axis=1))[:,None]*np.cross(nb,mb)
				hbar = 2*betas[:,None]*np.sum(mb*nb,axis=1)[:,None]*nb
			else:
				nxb = nx + (6*eta*h_volumes[:,None])**(-1)*(np.cross(Torque,nx)*dt + np.cross(dT,nx))
				nyb = ny + (6*eta*h_volumes[:,None])**(-1)*(np.cross(Torque,ny)*dt + np.cross(dT,ny))
				mxb = np.sum(mb*nxb,axis=1)[:,None]
				myb = np.sum(mb*nyb,axis=1)[:,None]
				mzb = np.sum(mb*nb,axis=1)[:,None]
				tbar = -2*(k_values*volumes)[:,None]*(myb**2 + mxb**2)*mzb*np.cross(nb,mb) - 2*(k_values2*volumes)[:,None]*myb**2*mxb**2*mzb*np.cross(nb,mb) \
					   -2*(k_values*volumes)[:,None]*(myb**2 + mzb**2)*mxb*np.cross(nxb,mb) - 2*(k_values2*volumes)[:,None]*myb**2*mzb**2*mxb*np.cross(nxb,mb) \
					   -2*(k_values*volumes)[:,None]*(mzb**2 + mxb**2)*myb*np.cross(nyb,mb) - 2*(k_values2*volumes)[:,None]*mzb**2*mxb**2*myb*np.cross(nyb,mb)
				hbar = -betas[:,None]*(mxb**2*mzb*nb + myb**2*mzb*nb + mxb**2*myb*nyb + mzb**2*myb*nyb + myb**2*mxb*nxb + mzb**2*mxb*nxb) - 2*betas2[:,None]*(mxb**2*myb**2*mzb*nb + mxb**2*mzb**2*myb*nyb + myb**2*mzb**2*mxb*nxb)
			if interactions == "on":
				hbar += H_dip
			Axes[:,:,n+1] = Axes[:,:,n] + (6*eta*h_volumes[:,None])**(-1)*0.5*(dt*(np.cross(Torque,Axes[:,:,n]) + np.cross(tbar,nb)) + np.cross(dT,Axes[:,:,n]) + np.cross(dT,nb))
			norm2 = np.sqrt(np.einsum('...i,...i', Axes[:,:,n+1], Axes[:,:,n+1]))
			Axes[:,:,n+1] /= norm2[:,None]
		else:
			nb = Axes[:,:,n]
			if shape == "u":
				hbar = 2*betas[:,None]*np.sum(mb*An,axis=1)[:,None]*An
			else: 
				mxb = np.sum(mb*nx,axis=1)[:,None]
				myb = np.sum(mb*ny,axis=1)[:,None]
				mzb = np.sum(mb*nb,axis=1)[:,None]
				hbar = -betas[:,None]*(mxb**2*mzb*nb + myb**2*mzb*nb + mxb**2*myb*ny + mzb**2*myb*ny + myb**2*mxb*nx + mzb**2*mxb*nx) - 2*betas2[:,None]*(mxb**2*myb**2*mzb*nb + mxb**2*mzb**2*myb*ny + myb**2*mzb**2*mxb*nx)

			Axes[:,:,n+1] = An
			if interactions == "on":
				hbar += H_dip		

		M[:,:,n+1] = M[:,:,n] + gamma*(1+lam**2)**(-1)*(0.5*(dt*(np.cross(M[:,:,n],H) - lam*np.cross(M[:,:,n], np.cross(M[:,:,n],H)) + np.cross(mb,hbar) - lam*np.cross(mb, np.cross(mb,hbar))) + np.cross(M[:,:,n],dW) - lam*np.cross(M[:,:,n], np.cross(M[:,:,n],dW)) + np.cross(mb,dW) - lam*np.cross(mb, np.cross(mb,dW))))
		
		norm = np.sqrt(np.einsum('...i,...i', M[:,:,n+1], M[:,:,n+1]))
		M[:,:,n+1] /= norm[:,None]
		if shape == "c" and brownian == "on":
			U = np.cross(Axes[:,:,n],Axes[:,:,n+1],axis=1)
			U /= np.sqrt(U[:,0]**2 + U[:,1]**2 + U[:,2]**2)[:,None]
			al = np.arccos(np.sum(Axes[:,:,n]*Axes[:,:,n+1],axis=1))
			Rot = np.array([[np.cos(al) + (U[:,0]**2)*(1 - np.cos(al)), U[:,0]*U[:,1]*(1 - np.cos(al)) - U[:,2]*np.sin(al), U[:,0]*U[:,2]*(1 - np.cos(al)) + U[:,1]*np.sin(al)],[U[:,0]*U[:,1]*(1 - np.cos(al)) + U[:,2]*np.sin(al), np.cos(al) + (U[:,1]**2)*(1 - np.cos(al)), U[:,2]*U[:,1]*(1 - np.cos(al)) - U[:,0]*np.sin(al)],[U[:,0]*U[:,2]*(1 - np.cos(al)) - U[:,1]*np.sin(al),U[:,2]*U[:,1]*(1 - np.cos(al)) + U[:,0]*np.sin(al), np.cos(al) + (U[:,2]**2)*(1 - np.cos(al))]])
			nx2 = np.zeros((I,3))
			ny2 = np.zeros((I,3))
			for p in range(3):
				nx2[:,p] = Rot[p,0,:]*nx[:,0] + Rot[p,1,:]*nx[:,1] + Rot[p,2,:]*nx[:,2]
				ny2[:,p] = Rot[p,0,:]*ny[:,0] + Rot[p,1,:]*ny[:,1] + Rot[p,2,:]*ny[:,2]

			nx = nx2
			ny = ny2

			al2 = np.sqrt(dW[:,0]**2 + dW[:,1]**2 + dW[:,2]**2)
			nx3 = np.zeros((I,3))
			ny3 = np.zeros((I,3))
			Rot2 = np.array([[np.cos(al2) + (Axes[:,0,n+1]**2)*(1 - np.cos(al2)), Axes[:,0,n+1]*Axes[:,1,n+1]*(1 - np.cos(al2)) - Axes[:,2,n+1]*np.sin(al2), Axes[:,0,n+1]*Axes[:,2,n+1]*(1 - np.cos(al2)) + Axes[:,1,n+1]*np.sin(al2)],[Axes[:,0,n+1]*Axes[:,1,n+1]*(1 - np.cos(al2)) + Axes[:,2,n+1]*np.sin(al2), np.cos(al2) + (Axes[:,1,n+1]**2)*(1 - np.cos(al2)), Axes[:,2,n+1]*Axes[:,1,n+1]*(1 - np.cos(al2)) - Axes[:,0,n+1]*np.sin(al2)],[Axes[:,0,n+1]*Axes[:,2,n+1]*(1 - np.cos(al2)) - Axes[:,1,n+1]*np.sin(al2),Axes[:,2,n+1]*Axes[:,1,n+1]*(1 - np.cos(al2)) + Axes[:,0,n+1]*np.sin(al2), np.cos(al2) + (Axes[:,2,n+1]**2)*(1 - np.cos(al2))]])
			for pp in range(3):
				nx3[:,pp] = Rot2[pp,0,:]*nx[:,0] + Rot2[pp,1,:]*nx[:,1] + Rot2[pp,2,:]*nx[:,2]
				ny3[:,pp] = Rot2[pp,0,:]*ny[:,0] + Rot2[pp,1,:]*ny[:,1] + Rot2[pp,2,:]*ny[:,2]
			
			nx = nx3
			ny = ny3


	M[:,:,0] = np.copy(M[:,:,int(N/5)])
	Axes[:,:,0] = np.copy(Axes[:,:,int(N/5)])
	return M


def runMC(temp, bias):
	#Ms = M0*(1 - b*temp**a)
	global nx, ny, U1
	betas = Ms**(-1)*k_values
	betas2 = Ms**(-1)*k_values2
	d = (1+lam**2)*kb*(gamma*Ms*lam)**(-1) 
	H_bias = np.zeros(N)
	H_bias.fill(bias)

	for n in range(N-1):
		dW = np.random.normal(loc = 0.0, scale = 1, size = (I,3))
		dW *= np.sqrt(dt*2*d*temp*volumes[:,None]**(-1))

		if brownian == "on":
			dT = np.random.normal(loc = 0.0, scale = 1, size = (I,3))
			dT *= np.sqrt(dt*12*kb*temp*eta*h_volumes[:,None])
				
		if shape == "u":
			if brownian == "on":
				Torque = -2*(k_values*volumes)[:,None]*np.absolute(np.sum(M[:,:,n]*Axes[:,:,n],axis=1))[:,None]*np.cross(Axes[:,:,n],M[:,:,n])

			H = 2*betas[:,None]*np.sum(M[:,:,n]*Axes[:,:,n],axis=1)[:,None]*Axes[:,:,n]
		else:
			mx = np.sum(M[:,:,n]*nx,axis=1)[:,None]
			my = np.sum(M[:,:,n]*ny,axis=1)[:,None]
			mz = np.sum(M[:,:,n]*Axes[:,:,n],axis=1)[:,None]
			#print(nx)
			#print(my)
			#print(mz)
			if brownian == "on":
				Torque = -2*(k_values*volumes)[:,None]*(my**2 + mx**2)*mz*np.cross(Axes[:,:,n],M[:,:,n]) - 2*(k_values2*volumes)[:,None]*my**2*mx**2*mz*np.cross(Axes[:,:,n],M[:,:,n]) \
					     -2*(k_values*volumes)[:,None]*(my**2 + mz**2)*mx*np.cross(nx,M[:,:,n]) - 2*(k_values2*volumes)[:,None]*my**2*mz**2*mx*np.cross(nx,M[:,:,n]) \
					     -2*(k_values*volumes)[:,None]*(mz**2 + mx**2)*my*np.cross(ny,M[:,:,n]) - 2*(k_values2*volumes)[:,None]*mz**2*mx**2*my*np.cross(ny,M[:,:,n])

			H = -betas[:,None]*(mx**2*mz*Axes[:,:,n] + my**2*mz*Axes[:,:,n] + mx**2*my*ny + mz**2*my*ny + my**2*mx*nx + mz**2*mx*nx) - 2*betas2[:,None]*(mx**2*my**2*mz*Axes[:,:,n] + mx**2*mz**2*my*ny + my**2*mz**2*mx*nx)

		H[:,1] += H_app_y[n] + H_bias[n]
		H[:,2] += H_app_z[n] + H_bias[n]

		#H[:,0] = 0
		#H[:,1] = 0
		#H[:,2] = 2
		#print(Torque)
		if interactions == "on":
			mGhost = makeGhost(mcGhost,n)
			masks = [R == 0]
			z = reduce(np.logical_or, masks)
			y = np.zeros((len(mcGhost),len(mcGhost),3))
			y[:,:,0] = z
			y[:,:,1] = z
			y[:,:,2] = z
			Q = ma.masked_array((mu0*diam1**3/8.*Ms*(3*np.sum(RR*mGhost,axis=2)[:,:,None]*RR/R[:,:,None]**2 - mGhost))/R[:,:,None]**3, mask=y, fill_value = 0)
			H_dip = np.sum(Q.filled(0)[0:I,:,:],axis=1)
			H += H_dip
		
		#print(M[:,:,n])
		mb = M[:,:,n] + gamma*(1+lam**2)**(-1)*((np.cross(M[:,:,n],H) - lam*np.cross(M[:,:,n], np.cross(M[:,:,n],H)))*dt + np.cross(M[:,:,n],dW) - lam*np.cross(M[:,:,n], np.cross(M[:,:,n],dW)))
		norm_m = np.sqrt(np.einsum('...i,...i', mb, mb))
		mb /= norm_m[:,None]
		#print(H)
		#print(mb)
		if brownian == "on":
			nb = Axes[:,:,n] + (6*eta*h_volumes[:,None])**(-1)*(np.cross(Torque,Axes[:,:,n])*dt + np.cross(dT,Axes[:,:,n]))
			norm_n = np.sqrt(np.einsum('...i,...i', nb, nb))
			#print(norm_n)
			nb /= norm_n[:,None]
			
			if shape == "u":
				tbar = 2*(k_values*volumes)[:,None]*np.absolute(np.sum(mb*nb,axis=1))[:,None]*np.cross(nb,mb)
				hbar = 2*betas[:,None]*np.sum(mb*nb,axis=1)[:,None]*nb
			else:
				nxb = nx + (6*eta*h_volumes[:,None])**(-1)*(np.cross(Torque,nx)*dt + np.cross(dT,nx))
				nyb = ny + (6*eta*h_volumes[:,None])**(-1)*(np.cross(Torque,ny)*dt + np.cross(dT,ny))
				mxb = np.sum(mb*nxb,axis=1)[:,None]
				myb = np.sum(mb*nyb,axis=1)[:,None]
				mzb = np.sum(mb*nb,axis=1)[:,None]
				tbar = -2*(k_values*volumes)[:,None]*(myb**2 + mxb**2)*mzb*np.cross(nb,mb) - 2*(k_values2*volumes)[:,None]*myb**2*mxb**2*mzb*np.cross(nb,mb) \
					   -2*(k_values*volumes)[:,None]*(myb**2 + mzb**2)*mxb*np.cross(nxb,mb) - 2*(k_values2*volumes)[:,None]*myb**2*mzb**2*mxb*np.cross(nxb,mb) \
					   -2*(k_values*volumes)[:,None]*(mzb**2 + mxb**2)*myb*np.cross(nyb,mb) - 2*(k_values2*volumes)[:,None]*mzb**2*mxb**2*myb*np.cross(nyb,mb)
				hbar = -betas[:,None]*(mxb**2*mzb*nb + myb**2*mzb*nb + mxb**2*myb*nyb + mzb**2*myb*nyb + myb**2*mxb*nxb + mzb**2*mxb*nxb) - 2*betas2[:,None]*(mxb**2*myb**2*mzb*nb + mxb**2*mzb**2*myb*nyb + myb**2*mzb**2*mxb*nxb)
				#tbar = 2*(k_values*volumes)[:,None]*(my**2 + mx**2)*np.sum(mb*nb,axis=1)[:,None]*np.cross(nb,mb) + 2*(k_values2*volumes)[:,None]*my**2*mx**2*np.sum(mb*nb,axis=1)[:,None]*np.cross(nb,mb)
				#hbar = -betas[:,None]*(mxb**2*mzb*nb + myb**2*mzb*nb + mxb**2*myb*ny + mzb**2*myb*ny + myb**2*mxb*nx + mzb**2*mxb*nx) - 2*betas2[:,None]*(mxb**2*myb**2*mzb*nb + mxb**2*mzb**2*myb*ny + myb**2*mzb**2*mxb*nx)
				#tbar = 2*(k_values*volumes)[:,None]*(my**2 + mx**2)*np.sum(mb*nb,axis=1)[:,None]*np.cross(nb,mb) + 2*(k_values2*volumes)[:,None]*my**2*mx**2*np.sum(mb*nb,axis=1)[:,None]*np.cross(nb,mb)
			
				#hbar = 2*betas[:,None]*np.sum(mb*nb,axis=1)[:,None]*nb
				#tbar = 2*(k_values*volumes)[:,None]*np.absolute(np.sum(mb*nb,axis=1))[:,None]*np.cross(nb,mb)
			if interactions == "on":
				hbar += H_dip
			hbar[:,1] += H_app_y[n+1]	
			hbar[:,2] += H_app_z[n+1] + H_bias[n+1]  #added bias here
			#print(hbar)	
			#hbar[:,0] = 0
			#hbar[:,1] = 0
			#hbar[:,2] = 20
			Axes[:,:,n+1] = Axes[:,:,n] + (6*eta*h_volumes[:,None])**(-1)*0.5*(dt*(np.cross(Torque,Axes[:,:,n]) + np.cross(tbar,nb)) + np.cross(dT,Axes[:,:,n]) + np.cross(dT,nb))
			norm2 = np.sqrt(np.einsum('...i,...i', Axes[:,:,n+1], Axes[:,:,n+1]))
			Axes[:,:,n+1] /= norm2[:,None]
		else:
			nb = Axes[:,:,n]
			if shape == "u":
				hbar = 2*betas[:,None]*np.sum(mb*An,axis=1)[:,None]*An
			else: 
				mxb = np.sum(mb*nx,axis=1)[:,None]
				myb = np.sum(mb*ny,axis=1)[:,None]
				mzb = np.sum(mb*nb,axis=1)[:,None]
				hbar = -betas[:,None]*(mxb**2*mzb*nb + myb**2*mzb*nb + mxb**2*myb*ny + mzb**2*myb*ny + myb**2*mxb*nx + mzb**2*mxb*nx) - 2*betas2[:,None]*(mxb**2*myb**2*mzb*nb + mxb**2*mzb**2*myb*ny + myb**2*mzb**2*mxb*nx)

			hbar[:,1] += H_app_y[n+1]
			hbar[:,2] += H_app_z[n+1] + H_bias[n+1]  #added bias here
			Axes[:,:,n+1] = An
			if interactions == "on":
				hbar += H_dip		

		M[:,:,n+1] = M[:,:,n] + gamma*(1+lam**2)**(-1)*(0.5*(dt*(np.cross(M[:,:,n],H) - lam*np.cross(M[:,:,n], np.cross(M[:,:,n],H)) + np.cross(mb,hbar) - lam*np.cross(mb, np.cross(mb,hbar))) + np.cross(M[:,:,n],dW) - lam*np.cross(M[:,:,n], np.cross(M[:,:,n],dW)) + np.cross(mb,dW) - lam*np.cross(mb, np.cross(mb,dW))))
		norm = np.sqrt(np.einsum('...i,...i', M[:,:,n+1], M[:,:,n+1]))
		M[:,:,n+1] /= norm[:,None]

		if shape == "c" and brownian == "on":
			U = np.cross(Axes[:,:,n],Axes[:,:,n+1],axis=1)
			if np.any(np.isnan(U)) == 1:
				print('trueU')
				U = U1

			U /= np.sqrt(U[:,0]**2 + U[:,1]**2 + U[:,2]**2)[:,None]
			U1 = U
			#j = np.sum(Axes[:,:,n]*Axes[:,:,n+1],axis=1)
			#al = np.arccos(j)
			al = np.arccos(np.sum(Axes[:,:,n]*Axes[:,:,n+1],axis=1))
			if np.any(np.isnan(al)) == 1:
				#print('trueA')
				al.fill(0)

			#print(j)
			#print('%e' % Axes[0,0,n])
			#print('%e' % Axes[0,1,n])
			#print('%e' % Axes[0,2,n])
			#print(U)
			Rot = np.array([[np.cos(al) + (U[:,0]**2)*(1 - np.cos(al)), U[:,0]*U[:,1]*(1 - np.cos(al)) - U[:,2]*np.sin(al), U[:,0]*U[:,2]*(1 - np.cos(al)) + U[:,1]*np.sin(al)],[U[:,0]*U[:,1]*(1 - np.cos(al)) + U[:,2]*np.sin(al), np.cos(al) + (U[:,1]**2)*(1 - np.cos(al)), U[:,2]*U[:,1]*(1 - np.cos(al)) - U[:,0]*np.sin(al)],[U[:,0]*U[:,2]*(1 - np.cos(al)) - U[:,1]*np.sin(al),U[:,2]*U[:,1]*(1 - np.cos(al)) + U[:,0]*np.sin(al), np.cos(al) + (U[:,2]**2)*(1 - np.cos(al))]])
			nx2 = np.zeros((I,3))
			ny2 = np.zeros((I,3))
			for p in range(3):
				nx2[:,p] = Rot[p,0,:]*nx[:,0] + Rot[p,1,:]*nx[:,1] + Rot[p,2,:]*nx[:,2]
				ny2[:,p] = Rot[p,0,:]*ny[:,0] + Rot[p,1,:]*ny[:,1] + Rot[p,2,:]*ny[:,2]

			nx = nx2
			ny = ny2
			#print(Rot[:,:,:])
			"""
			al2 = np.sqrt(dW[:,0]**2 + dW[:,1]**2 + dW[:,2]**2)
			nx3 = np.zeros((I,3))
			ny3 = np.zeros((I,3))
			Rot2 = np.array([[np.cos(al2) + (Axes[:,0,n+1]**2)*(1 - np.cos(al2)), Axes[:,0,n+1]*Axes[:,1,n+1]*(1 - np.cos(al2)) - Axes[:,2,n+1]*np.sin(al2), Axes[:,0,n+1]*Axes[:,2,n+1]*(1 - np.cos(al2)) + Axes[:,1,n+1]*np.sin(al2)],[Axes[:,0,n+1]*Axes[:,1,n+1]*(1 - np.cos(al2)) + Axes[:,2,n+1]*np.sin(al2), np.cos(al2) + (Axes[:,1,n+1]**2)*(1 - np.cos(al2)), Axes[:,2,n+1]*Axes[:,1,n+1]*(1 - np.cos(al2)) - Axes[:,0,n+1]*np.sin(al2)],[Axes[:,0,n+1]*Axes[:,2,n+1]*(1 - np.cos(al2)) - Axes[:,1,n+1]*np.sin(al2),Axes[:,2,n+1]*Axes[:,1,n+1]*(1 - np.cos(al2)) + Axes[:,0,n+1]*np.sin(al2), np.cos(al2) + (Axes[:,2,n+1]**2)*(1 - np.cos(al2))]])
			for pp in range(3):
				nx3[:,pp] = Rot2[pp,0,:]*nx[:,0] + Rot2[pp,1,:]*nx[:,1] + Rot2[pp,2,:]*nx[:,2]
				ny3[:,pp] = Rot2[pp,0,:]*ny[:,0] + Rot2[pp,1,:]*ny[:,1] + Rot2[pp,2,:]*ny[:,2]
			
			nx = nx3
			ny = ny3
			"""

		vstep = (N)**(-1)
		#vect = np.arange(vstep,1+vstep,vstep)
		#vect = np.logspace(vstep,1,num=N)
		#vect /= 10
		#vect *=20
		#if n % 10 == 0:
		#	pl.plot(Axes[:,1,n], Axes[:,2,n], linestyle = 'none', marker = '.',color = 'blue', alpha = vect[n])
		#	pl.plot(M[:,1,n], M[:,2,n], linestyle = 'none', marker = '.',color = 'm', alpha = vect[n])
		#pl.plot(nx[:,0], nx[:,2], linestyle = 'none', marker = '.',color = 'red', alpha = vect[n])
		#pl.plot(ny[:,0], ny[:,2], linestyle = 'none', marker = '.',color = 'c', alpha = vect[n])
		#if n % 1000 == 0:
		#	print(M[:,:,n+1])
		#	print(n)
		#print(H)
		#print(Torque)


	#pl.ylim([-0.1,1.1])
	#pl.xlim([-0.1,1.1])
	#pl.show()
	return M

def runAC():	
	temp = minT
	X1 = np.zeros((int((maxT-minT)/tstep)+1,2))
	X2 = np.zeros((int((maxT-minT)/tstep)+1,2))
	x = 0
	while temp <= maxT:
		thermalize(temp)
		print(temp)
		runMC(temp,0)
		X1[x,0] = temp
		X2[x,0] = temp

		X1[x,1] = np.sum(2*(h0*N)**(-1)*np.mean(M[:,2,:], axis=0)*np.cos(w*T[0:N]))
		X2[x,1] = np.sum(2*(h0*N)**(-1)*np.mean(M[:,2,:], axis=0)*np.sin(w*T[0:N]))

		print(X2[x,1])

		M[:,:,0] = Start
		Axes[:,:,0] = aStart

		x += 1
		temp += tstep

	return X1, X2

def runACF():	
	X1 = np.zeros((Fsteps,2))
	X2 = np.zeros((Fsteps,2))
	global f
	global w
	global dt
	global T
	for ff in range(Fsteps):
		f = freqs[ff]
		w = 2*np.pi*f   #angular frequency (Rad/sec). constraint: f*N must be integer
		dt = cycs*(f*N)**(-1)   #time step
		T = np.arange(0,dt*N,dt)   #array of time steps
		initialize()
		thermalize(297)
		runMC(297,0)
		X1[ff,0] = freqs[ff]
		X2[ff,0] = freqs[ff]

		X1[ff,1] = np.sum(2*(h0*N)**(-1)*np.mean(M[:,2,:], axis=0)*np.cos(w*T[0:N]))
		X2[ff,1] = np.sum(2*(h0*N)**(-1)*np.mean(M[:,2,:], axis=0)*np.sin(w*T[0:N]))

		M[:] = Start
		Axes[:] = aStart

	return X1, X2

def main():
	xx1 = np.zeros((int((maxT-minT)/tstep)+1,2))
	xx2 = np.zeros((int((maxT-minT)/tstep)+1,2))
	for x in range(X):
		initialize()
		X1, X2 = runAC()
		XX1[x,:,:] = X1[:,:]
		XX2[x,:,:] = X2[:,:]

	xx1[:,0] = XX1[0,:,0]
	xx2[:,0] = XX2[0,:,0]

	xx1[:,1] = np.mean(XX1[:,:,1],axis=0)
	xx2[:,1] = np.mean(XX2[:,:,1],axis=0)

	return xx1, xx2

def mainF():
	xx1 = np.zeros((Fsteps,2))
	xx2 = np.zeros((Fsteps,2))
	for x in range(X):
		X1, X2 = runACF()
		XF1[x,:,:] = X1[:,:]
		XF2[x,:,:] = X2[:,:]

	xx1[:,0] = XF1[0,:,0]
	xx2[:,0] = XF2[0,:,0]

	xx1[:,1] = np.mean(XF1[:,:,1],axis=0)
	xx2[:,1] = np.mean(XF2[:,:,1],axis=0)

	return xx1, xx2


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
	order_range = range(order+1)
	half_window = (window_size -1) // 2
	# precompute coefficients
	b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
	m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
	# pad the signal at the extremes with
	 # values taken from the signal itself
	firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
	lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
	y = np.concatenate((firstvals, y, lastvals))
	return np.convolve( m[::-1], y, mode='valid')



"""
#X'' vs. Temp

xx1, xx2 = main()

np.savetxt(save, xx2, delimiter=",")

pl.plot(xx2[:,0], xx2[:,1], '.', color='darkblue')
pl.plot(xx2[:,0], xx2[:,1], color='darkblue', linewidth = 0.5)
pl.ylabel('X"')
pl.xlabel('Temperature (K)')
pl.title('X" vs. Temperature')
pl.show()

pl.plot(xx1[:,0], xx1[:,1], linewidth = 0.5)
pl.ylabel('X')
pl.xlabel('Temperature (K)')
pl.title('X vs. Temperature')
#pl.show()


#np.savetxt("5nm_sigma3_X1.csv", xx1, delimiter=",")
#np.savetxt("X2_6p3nm_K7000_f10000_t1.csv", xx2, delimiter=",")
"""


"""
#MPS harmonics at different temps

Harm = np.zeros((20,2,5))

for xx in range(5):
	initialize()
	for x in range(20):
		tem = 270+x*5
		print x
		runMC(tem)
		fM = np.fft.fft(np.mean(M[:,2,:],axis=0))
		Har = abs(fM[1:9])
		Harm[x,0,xx] = tem
		Harm[x,1,xx] = Har[4]/Har[2]
"""
"""
#MPS harmonics at different amplitudes

nAmps = 7
Harm = np.zeros((nAmps,9,X))

for xx in range(X):
	h0 = 0.015
	initialize()
	for x in range(nAmps):
		print h0
		runMC(300,0)
		fM = np.fft.fft(np.mean(M[:,2,:],axis=0))
		Har = abs(fM[1:9])
		Harm[x,0,xx] = h0
		#Harm[x,1,xx] = Har[4]/Har[2]
		Harm[x,1:,xx] = Har[:]
		h0 += 0.002

Harms = np.mean(Harm, axis=2)

np.savetxt("Harmonics_17nm_varamp_test3.csv", Harms, delimiter=",")
"""
"""
#MPS harmonics at different viscosities

Harm = np.zeros((20,9,X))

for xx in range(X):
	initialize()
	for x in range(20):
		print eta
		runMC(300,0)
		fM = np.fft.fft(np.mean(M[:,2,:],axis=0))
		Har = abs(fM[1:9])
		Harm[x,0,xx] = eta
		#Harm[x,1,xx] = Har[4]/Har[2]
		Harm[x,1:,xx] = Har[:]
		eta *= 1.5 

Harms = np.mean(Harm, axis=2)

np.savetxt("Harmonics_1234_varamp.csv", Harms, delimiter=",")
"""
"""

#MPS harmonics at different bias fields

fields = 12
Harm = np.zeros((X,21,fields))
amp = np.zeros(fields)
bmin = 0
bmax = 0.15
#h0 = 0.01
bias = np.linspace(bmin,bmax,fields)
for q in range(fields):
	amp[q] = bias[q]
	#print(bias[q])
	for xx in range(X):
		initialize()
		thermalize(300)
		runMC(300,bias[q])
		#yhat = savitzky_golay(np.mean(M[:,2,:],axis=0), 51, 3)
		#yhat = savitzky_golay(yhat2, 51, 3)
		#fM = np.fft.fft(yhat)
		fM = np.fft.fft(np.mean(M[:,2,:],axis=0))

		##fM = np.fft.fft(M[:,2,:])
		##fM = np.sum(fM,axis=0)
		Harm[xx,0,q] = amp[q]
		Harm[xx,1:,q] = abs(fM[1:21])
		##Harm[xx,1:21,q] = np.real(fM[1:21])
		##Harm[xx,21:41,q] = np.imag(fM[1:21])
		##Harm[xx,1:,q] = abs(fM[1:21])**2
		if q == 0:
			phase = np.angle(fM[1:21])
			print phase.shape

	print(q)
	#h0 += 0.002

#ratio = Harm[:,4]/Harm[:,2]
Har = np.mean(Harm, axis=0)
np.savetxt(save, Har, delimiter=",")
np.savetxt(save2, phase, delimiter=",")
#pl.plot(H_app[0:N]*1000,np.mean(M[:,2,:],axis=0), color='darkblue', linewidth = 0.5)
#amp = np.array([10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200])
pl.plot(amp,Har[3,:])
pl.show()

"""
"""
#MPS moment/axis/hysteresis


#pl.plot(T,np.mean(M[:,2,:],axis=0), color='darkblue', linewidth = 0.5)

#pl.plot(Axes[0,2,:])
#pl.plot(M[0,2,:], color='darkblue', linewidth = 0.5)

#motion = np.zeros((N,2))
#motion[:,0] = np.mean(Axes[:,2,:],axis=0)
#motion[:,1] = np.mean(M[:,2,:],axis=0)

hystX = np.zeros((N,3,X))
for xx in range(X):
	initialize()
	thermalize(300)
	runMC(300,0)
	hystX[:,0,xx] = H_app_z[0:N]*1000
	hystX[:,1,xx] = np.mean(M[:,1,:],axis=0)
	hystX[:,2,xx] = np.mean(M[:,2,:],axis=0)
	print xx


hyst = np.mean(hystX, axis=2)


np.savetxt(save1, hyst, delimiter=",")
#np.savetxt(save2, np.mean(M[:,2,:],axis=0), delimiter=",")

#pl.plot(hyst[:,0],hyst[:,1])
#pl.plot(hyst[190000:,0],hyst[190000:,1])
#pl.plot(hyst[:10000,0],hyst[:10000,1])

#pl.ylabel('Magnetic Moment')
#pl.xlabel('Mag Field')
#pl.title('18nm, 310 K')
#pl.ylim([-1.1,1.1])
#pl.xlim([-100,100])
#pl.show()
pl.plot(hyst[:,0],hyst[:,2])
pl.show()

pl.plot(hyst[0,2])
pl.plot(Axes[0,2,:])
pl.show()

pl.plot(hyst[0,2])
pl.plot(Axesx[0,2,:])
pl.show()

pl.plot(hyst[0,2])
pl.plot(Axesy[0,2,:])
pl.show()
"""

"""
#for calculating relaxation time

Zmax = np.max(hyst[:,1])
#tau = 0
#for n in range(N):
#	if np.abs(hyst[n,1]) < Zmax - tauPercent*Zmax:
#		tau += 1

#tau /= 2
#tau *= 1e-8
tau0 = np.sqrt(np.pi)*Ms*(2*lam*gamma*K*2)**(-1)
vol = (4/3.)*np.pi*(diam1/2.)**3
volH = (4/3.)*np.pi*((diam1+coating)/2.)**3
sig = K*vol*(kb*297)**(-1)
tauN = tau0*np.exp(sig)*(sig)**(-0.5)
tauB = 3*volH*eta*(kb*297)**(-1)
tauE = tauB*tauN*(tauB+tauN)**(-1)

y = np.arange(N)


yhat = savitzky_golay(hyst[:,1], 51, 3)
#yhat2 = savitzky_golay(yhat, 51, 3)
q = 15
x = np.arange(q)
popt, pcov = curve_fit(func, x, hyst[N/5:N/5+q,1], maxfev=1000)
z = np.polyfit(x,np.log(hyst[N/5:N/5+q,1]),1)
p = np.poly1d(z)

#pl.plot(hyst[:,0],yhat, color='darkblue', linewidth = 0.5)
#pl.plot(yhat, color='darkblue', linewidth = 0.5)
#pl.plot(x, yhat[N/5:N/5+20], color='darkblue', linewidth = 0.5)
pl.plot(x, np.log(hyst[N/5:N/5+q,1]))
#pl.plot(x, func(y[N/5:N/5+20],*popt), color='red')
pl.plot(x, p(x), color='red')
print z

pl.ylabel('Magnetic Moment')
pl.xlabel('Mag Field')
pl.title('297 K')
pl.show()
"""

"""
#for evaluating taus from Debye model

diam1 = 10e-9
acs = np.zeros((Fsteps,21))
taus = np.zeros((5,20))
for j in range(20):
	sizes = np.random.lognormal(np.log(diam1), sigma, I)
	volumes = (4/3.)*np.pi*(sizes/2.)**3   #create array of volumes
	h_volumes = (4/3.)*np.pi*((sizes+40e-9)/2.)**3   #hydrodynamic volumes (calculate later)
	xx1, xx2 = mainF()
	acs[:,j] = xx2[:,1]
	#pl.plot(xx2[:,0],xx2[:,1])
	#pl.show()
	taus[0,j] = (xx2[np.where(xx2[:,1] == np.max(xx2[:,1])),0])**(-1)

	vol = (4/3.)*np.pi*(diam1/2.)**3
	volH = (4/3.)*np.pi*((diam1+coating)/2.)**3
	sig = K*vol*(kb*297)**(-1)
	tau0 = np.sqrt(np.pi)*Ms*(2*lam*gamma*K*2)**(-1)
	taus[1,j] = tau0*np.exp(sig)*(sig)**(-0.5)
	taus[2,j] = 3*volH*eta*(kb*297)**(-1)
	taus[3,j] = taus[1,j]*taus[2,j]*(taus[1,j]+taus[2,j])**(-1)
	taus[4,j] = diam1

	print diam1
	diam1 += 1e-9

acs[:,20] = xx2[:,0]
np.savetxt("tau_K9000_ac_5mT_g.csv", taus, delimiter=",")
np.savetxt("ac_K9000_ac_5mT_g.csv", acs, delimiter=",")
"""
"""
taus = np.zeros((5,5,20))   #simulations, neel, brownian, reduced
tau0 = np.sqrt(np.pi)*Ms*(2*lam*gamma*K*2)**(-1)
diam1 = 10e-9
for j in range(20):
	for xx in range(5):
		sizes = np.random.lognormal(np.log(diam1), sigma, I)
		volumes = (4/3.)*np.pi*(sizes/2.)**3   #create array of volumes
		h_volumes = (4/3.)*np.pi*((sizes+40e-9)/2.)**3   #hydrodynamic volumes (calculate later)
		initialize()
		thermalize(297)
		runMC(297,0)
		hyst = np.zeros((N,2))
		hyst[:,0] = H_app[0:N]*1000
		hyst[:,1] = np.mean(M[:,2,:],axis=0)
		#pl.plot(hyst[:,1])
		#pl.show()

		#q = 15
		#x = np.arange(q)
		#z = np.polyfit(x,np.log(hyst[N/5:N/5+q,1]),1)
		#taus[xx,0,j] = -(z[0])**(-1)
		#taus[xx,0,j] *= 1e-8

		Zmax = np.max(hyst[:,1])
		tau = 0
		for n in range(N):
			if np.abs(hyst[n,1]) < Zmax - tauPercent*Zmax:
				tau += 1

		tau /= 2
		tau *= 1e-8
		taus[xx,0,j] = tau

		vol = (4/3.)*np.pi*(diam1/2.)**3
		volH = (4/3.)*np.pi*((diam1+coating)/2.)**3
		sig = K*vol*(kb*297)**(-1)
		taus[xx,1,j] = tau0*np.exp(sig)*(sig)**(-0.5)
		taus[xx,2,j] = 3*volH*eta*(kb*297)**(-1)
		taus[xx,3,j] = taus[xx,1,j]*taus[xx,2,j]*(taus[xx,1,j]+taus[xx,2,j])**(-1)
		taus[xx,4,j] = diam1


	print diam1
	diam1 += 1e-9

tau = np.mean(taus, axis=0)
np.savetxt("tau_K9000_hyst90.csv", tau, delimiter=",")

#pl.plot(yhat)
#pl.plot(H_app)
#pl.show()
#pl.plot(hyst[N/5:N/5+100,1])
#pl.show()

#np.savetxt("cubic_test_boff_K25te.csv", yhat, delimiter=",")
"""

"""
#for calculating PSF

yhat = savitzky_golay(np.mean(M[:,2,:],axis=0), 51, 3)
yhat2 = savitzky_golay(yhat, 51, 3)
pl.plot(H_app[0:N]*1000,yhat2, color='darkblue', linewidth = 0.5)
pl.show()

dMdH = np.zeros((N-1,3))
dMdH[:,0] = H_app[0:N-1]*1000
dMdH[:,1] = np.diff(yhat2)/np.diff(H_app[0:N]*1000)
dMdH[:,2] = yhat2[0:N-1]
pl.plot(H_app[0:N-1]*1000, dMdH[:,1])
pl.ylim(0,0.3)
pl.show()

#pl.plot(np.mean(Axes[:,2,:],axis=0))
#pl.plot(np.mean(M[:,2,:],axis=0), color='darkblue', linewidth = 0.5)
#pl.show()

"""

"""
#for calculating PSF for multiple sizes

dMdH = np.zeros((N-1,3,22))
der = np.zeros((3,N-1,22))
der2 = np.zeros((3,N-1,22))
diam1 = 18e-9
for q in range(22):
	for xx in range(3):
		sizes = np.random.lognormal(np.log(diam1), sigma, I)
		volumes = (4/3.)*np.pi*(sizes/2.)**3   #create array of volumes
		h_volumes = (4/3.)*np.pi*((sizes+40e-9)/2.)**3   #hydrodynamic volumes (calculate later)
		initialize()
		thermalize(300)
		runMC(300)
		yhat = savitzky_golay(np.mean(M[:,2,:],axis=0), 51, 3)
		yhat2 = savitzky_golay(yhat, 51, 3)
		der[xx,:,q] = np.diff(yhat2)/np.diff(H_app[0:N]*1000)
		der2[xx,:,q] = yhat2[0:N-1]

	print diam1
	diam1 += 1e-9

dMdH[:,0,:] = H_app[0:N-1,None]*1000
dMdH[:,1,:] = np.mean(der, axis = 0)
dMdH[:,2,:] = np.mean(der2, axis = 0)
pl.plot(H_app[0:N-1]*1000, dMdH[:,1,0])
pl.ylim(0,0.3)
pl.show()

np.savetxt("dMdH_K12000_H.csv", dMdH[:,0,:], delimiter=",")
np.savetxt("dMdH_K12000_psf.csv", dMdH[:,1,:], delimiter=",")
np.savetxt("dMdH_K12000_hyst.csv", dMdH[:,2,:], delimiter=",")

"""
hystX = np.zeros((N,2,X))

for xx in range(X):
	initialize()
	thermalize(temperature)
	runMC(temperature,0)
	hystX[:,0,xx] = H_app_z[0:N]*1000
	hystX[:,1,xx] = np.mean(M[:,2,:],axis=0)
	#print(xx)

hyst = np.mean(hystX, axis=2)
np.savetxt(save1, hyst, delimiter=",")

#pl.plot(hyst[:,0],hyst[:,1])
#pl.plot(hyst[:,1])
#pl.ylabel('Magnetic Moment')
#pl.xlabel('Mag Field')
#pl.show()

