from __future__ import division
import numpy as np
import numpy.ma as ma
import itertools
from functools import reduce
from numpy.matlib import repmat, repeat

#---global constants----
kb = 1.381e-23   #boltzmann constant
mu0 = 1e-7       #permeability/4pi
gyro = 1.76e11   #gyromagnetic ratio
#-----------------------

def calculate_values(shape, kBulk, kSurface, K, K2, kSigma, Ms, fieldFreq, fieldAmp, concentration, \
	numTimeSteps, numParticles, diameter, sigma, hDiameter, hSigma, coating, cycles):
	"""
	Sets initial values for parameters based on inputs.

	Returns:
		gamma					  (float) : resonance frequency (Hz)
		boxLength				  (float) : length of one side of the cube that particles are placed in (m)
		rAvg 					  (float) : average separation of particles (m)
		angFreq					  (float) : angular frequency of applied field (Rad/s)
		dt 						  (float) : time step (s)
		timeSteps 				  (array) : array of each time step for one field cycle
		volumes 				  (array) : array of core volumes for corresponding particles
		hVolumes 				  (array) : array of hydrodynamic volumes for corresponding particles
		betas 					  (array) : array of parameter beta for anisotropy energy calculation 
		betas2 					  (array) : array of parameter beta2 for anisotropy energy calculation
		kValues 				  (array) : array of K1 values for corresponding particles 
		k2Values 				  (array) : array of K2 values for corresponding particles 
		fieldAmp				  (float) : amplitude of applied magnetic field in Tesla
	"""
	#adjust units
	diameter *= 1e-9    #convert to meters
	hDiameter *= 1e-9   #convert to meters
	fieldFreq *= 1000 	#convert to Hz
 	fieldAmp *= 0.001   #convert to Tesla
 	numTimeSteps *= cycles

 	if coating != None: hDiameter = diameter + coating*1e-9   #hydrodynamic diameter
	if shape == "cubic": K = kBulk + 6*kSurface/diameter   #calculate effective K from Bulk and Surface
	
	H_k = 50*np.abs(K)/float(Ms)   #anisotropy field
	gamma = gyro*H_k/(2*np.pi)   #resonance frequency

	boxLength = (numParticles*concentration**(-1))**(1/3.)   #length of one box side
	rAvg = 3**(1/2.)*concentration**(-1/3.)  #estimated avg interparticle distance

	angFreq = 2*np.pi*fieldFreq   #angular frequency (Rad/sec)
	dt = cycles*(fieldFreq*numTimeSteps)**(-1)   #time step
	timeSteps = np.arange(0,dt*numTimeSteps,dt)   #array of time steps

	diameters = np.random.lognormal(np.log(diameter), sigma, numParticles)	   #create array of diameters
	hDiameters = np.random.lognormal(np.log(hDiameter), hSigma, numParticles)	#create array of hydrodynamic diameters

	volumes = (4/3.)*np.pi*(diameters/2.)**3   #create array of volumes
	hVolumes = (4/3.)*np.pi*(hDiameters/2.)**3   #create array of hydrodynamic volumes

	kValues = (np.random.lognormal(np.log(abs(K)), kSigma, numParticles))*np.sign(K)   #create array of K values
	k2Values = (np.random.lognormal(np.log(abs(K2)), kSigma, numParticles))*np.sign(K2)   #create array of K2 values

	betas = Ms**(-1)*kValues   #create array of betas
	betas2 = Ms**(-1)*k2Values   #create array of betas from K2

	return gamma, boxLength, rAvg, angFreq, dt, timeSteps, volumes, hVolumes, betas, betas2, kValues, \
		k2Values, fieldAmp, numTimeSteps


def generate_fluctuations(numParticles, alpha, Ms, gamma, viscosity, temperature, volumes, hVolumes, dt, type):
	"""
	Generates vector of random thermal fluctuations. Specify field or torque with "type".

	Returns:
		dW					  (array) : array of fluctuations
	"""
	#set standard deviation for fluctuations based on type		
	if type == "H": sd = np.sqrt(dt*2*(1+alpha**2)*kb*(gamma*Ms*alpha)**(-1)*temperature*volumes[:,None]**(-1))
	if type == "T": sd = np.sqrt(dt*12*kb*temperature*viscosity*hVolumes[:,None])
	#generate randomly distributed fluctuations
	dW = np.random.normal(loc = 0.0, scale = sd, size = (numParticles,3))

	return dW

def calculate_cosines(M, A):
	"""
	Calculate direction cosines (dot product of moment and axis vectors).

	Returns:
		mx					  (array) : direction cosines for x 
		my					  (array) : direction cosines for y 
		mz					  (array) : direction cosines for z 
	"""
	mx = np.sum(M*A[:,3:6],axis=1)[:,None]
	my = np.sum(M*A[:,6:],axis=1)[:,None]
	mz = np.sum(M*A[:,:3],axis=1)[:,None]

	return mx, my, mz

def calculate_torque(M, A, kValues, volumes, shape, *args):
	"""
	Calculate torque on each particle. Add args for cubic anisotropy: mx, my, mz, k2values

	Returns:
		torque					  (array) : array of torques
	"""
	if shape == "uniaxial": torque = -2*(kValues*volumes)[:,None]*np.absolute(np.sum(M*A,axis=1))[:,None]*np.cross(A,M)
	if shape == "cubic": 
		mx = args[0]
		my = args[1]
		mz = args[2]
		nz = A[:,:3]
		nx = A[:,3:6]
		ny = A[:,6:]
		k2Values = args[3]
		torque = -2*(kValues*volumes)[:,None]*(my**2 + mx**2)*mz*np.cross(nz,M) \
				-2*(k2Values*volumes)[:,None]*my**2*mx**2*mz*np.cross(nz,M) \
				-2*(kValues*volumes)[:,None]*(my**2 + mz**2)*mx*np.cross(nx,M) \
				-2*(k2Values*volumes)[:,None]*my**2*mz**2*mx*np.cross(nx,M) \
				-2*(kValues*volumes)[:,None]*(mz**2 + mx**2)*my*np.cross(ny,M) \
				-2*(k2Values*volumes)[:,None]*mz**2*mx**2*my*np.cross(ny,M)

	return torque

def calculate_field(M, A, betas, shape, *args):
	"""
	Calculate field on each particle. Add args for cubic anisotropy: mx, my, mz, betas2

	Returns:
		field					  (array) : array of fields
	"""
	if shape == "uniaxial": field = 2*betas[:,None]*np.sum(M*A,axis=1)[:,None]*A
	if shape == "cubic": 
		mx = args[0]
		my = args[1]
		mz = args[2]
		nz = A[:,:3]
		nx = A[:,3:6]
		ny = A[:,6:]
		betas2 = args[3]
		field = -betas[:,None]*(mx**2*mz*nz + my**2*mz*nz + mx**2*my*ny + mz**2*my*ny + my**2*mx*nx + \
			mz**2*mx*nx) - 2*betas2[:,None]*(mx**2*my**2*mz*nz + mx**2*mz**2*my*ny + my**2*mz**2*mx*nx)

	return field

def calculate_interaction_field(M, ghostCoords, RR, R, numParticles, masks, diameter, Ms):
	"""
	Calculate dipole interaction field on each particle.

	Returns:
		dipleField					  (array) : array of dipole fields
	"""
	#generate "ghost" matrix for periodic BCs
	ghostMatrix = make_ghost_matrix(M, masks)

	#mask out zero values along diagonal
	mask = np.zeros((len(ghostCoords),len(ghostCoords),3))
	mask[:,:,0] = reduce(np.logical_or, [R == 0])
	mask[:,:,1] = reduce(np.logical_or, [R == 0])
	mask[:,:,2] = reduce(np.logical_or, [R == 0])

	#calculate interactions between each particle
	Mask = ma.masked_array((mu0*diameter**3/8.*Ms*(3*np.sum(RR*ghostMatrix,axis=2)[:,:,None]*RR/R[:,:,None]**2 - ghostMatrix))/R[:,:,None]**3, mask=mask, fill_value = 0)
	
	#sum over interactions on each particle
	dipoleField = np.sum(Mask.filled(0)[0:numParticles,:,:],axis=1)

	return dipoleField

def rotate_axes(A, numParticles):
	"""Rotate two minor axes of cubic particle"""
	#generate normal unit vector to rotate around
	U = np.cross(A[:,:3,0],A[:,:3,1],axis=1)
	U /= np.sqrt(U[:,0]**2 + U[:,1]**2 + U[:,2]**2)[:,None]
	#generate angle to rotate around
	angle = np.arccos(np.sum(A[:,:3,0]*A[:,:3,1],axis=1))
	#create rotation matrix based on normal vector and angle
	rotMatrix = np.array([[np.cos(angle) + (U[:,0]**2)*(1 - np.cos(angle)), U[:,0]*U[:,1]*(1 - np.cos(angle)) \
		- U[:,2]*np.sin(angle), U[:,0]*U[:,2]*(1 - np.cos(angle)) + U[:,1]*np.sin(angle)],[U[:,0]*U[:,1]*(1 - np.cos(angle)) \
		+ U[:,2]*np.sin(angle), np.cos(angle) + (U[:,1]**2)*(1 - np.cos(angle)), U[:,2]*U[:,1]*(1 - np.cos(angle)) \
		- U[:,0]*np.sin(angle)],[U[:,0]*U[:,2]*(1 - np.cos(angle)) - U[:,1]*np.sin(angle),U[:,2]*U[:,1]*(1 - np.cos(angle)) \
		+ U[:,0]*np.sin(angle), np.cos(angle) + (U[:,2]**2)*(1 - np.cos(angle))]])
	#rotate each vector
	for p in range(3):
		A[:,3+p,1] = rotMatrix[p,0,:]*A[:,3,0] + rotMatrix[p,1,:]*A[:,4,0] + rotMatrix[p,2,:]*A[:,5,0]
		A[:,6+p,1] = rotMatrix[p,0,:]*A[:,6,0] + rotMatrix[p,1,:]*A[:,7,0] + rotMatrix[p,2,:]*A[:,8,0]

	return A