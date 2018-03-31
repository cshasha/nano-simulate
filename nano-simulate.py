
from __future__ import division
import numpy as np
import numpy.ma as ma
import itertools
import os.path, sys, time, warnings
import pylab as pl
from datetime import datetime, timedelta
from functools import reduce
from math import factorial
from numpy.matlib import repmat, repeat
#import timeit
warnings.filterwarnings("ignore")

#---global constants----
kb = 1.381e-23   #boltzmann constant
mu0 = 1e-7       #permeability/4pi
gyro = 1.76e11   #gyromagnetic ratio
#-----------------------

def simulate_MH(numParticles=100, numReps=1, diameter=25, shape="cubic", savefile=None, numTimeSteps=10000, \
	brownian="on", interactions="off", aligned="off", temperature=300, concentration=1e15, cluster=0, \
	fieldAmp=20, fieldFreq=25, cycles=2, sigma=0.1, hDiameter=50, coating=None, hSigma=0.1, kBulk=-13000, \
	kSurface=-3.9e-5, K=None, K2=-5000, kSigma=0.2, alpha=1, Ms=420000, viscosity=8.9e-4, time="off", \
	field="ac",cut=10):

	"""
	Main function to simulate M(H) curve.

	Args:
		numParticles 	  (int, optional) : number of particles. defaults to 100
		numReps 		  (int, optional) : number of overall iterations of the simulation. defaults to 1
		diameter	    (float, optional) : average particle diameter, in nanometers. defaults to 25
		shape 			  (str, optional) : magnetocrystalline anisotropy. either "cubic" or "uniaxial". 
											defaults to "cubic"
		savefile	 	  (str, optional) : name of file to save (e.g. "test"). defaults to None
		numTimeSteps      (int, optional) : number of time steps in each iteration. defaults to 10000
		brownian 	  	  (str, optional) : sets brownian rotation on or off. either "on" or "off". 
											defaults to "on"
		interactions	  (str, optional) : turns interparticle interactions on or off. either "on" or "off". 
											defaults to "off"
		aligned 	 	  (str, optional) : sets moments and axes aligned along the z direction. either "on" 
											or "off". defaults to "off"
		temperature 	(float, optional) : temperature in Kelvin. defaults to 300
		concentration 	(float, optional) : concentration in particles/m^3. relevant when interactions are on. 
											defaults to 1e15
		cluster 		  (int, optional) : controls cluster type. 0 = no cluster, 1 = chain. defaults to 0
		fieldAmp        (float, optional) : amplitude of applied field, in mT. defaults to 20
		fieldFreq       (float, optional) : frequency of applied field, in kHz. defaults to 25
		cycles			  (int, optional) : number of field cycles. defaults to 2
		sigma           (float, optional) : distribution parameter for particle diameters. defaults to 0.1
		hDiameter       (float, optional) : average hydrodynamic diameter in nanometers. defaults to 50
		coating         (float, optional) : added non-magnetic coating in nanometers. can use in place of 
											hDiameter. defaults to None
		hSigma			(float, optional) : size distribution parameter for hydrodynamic size. defaults to 0.1
		kBulk			(float, optional) : bulk anisotropy constant in J/m^3. only relevant for cubic 
											anisotropy. defaults to -13000.
		kSurface        (float, optional) : surface anisotropy in J/m^3. only relevant for cubic anisotropy. 
											defaults to -3.9e-5
		K               (float, optional) : uniaxial anisotropy constant in J/m^3, or K1 for cubic anisotropy, 
											if not using bulk/surface. defaults to None
		K2              (float, optional) : K2 in J/m^3 for cubic anisotropy. defaults to -5000
		kSigma			(float, optional) : distribution parameter for anisotropy values. defaults to 0.2
		alpha			(float, optional) : Gilbert damping constant. defaults to 1
		Ms 				(float, optional) : saturation magnetization in A/m. defaults to 420000
		viscosity       (float, optional) : medium viscosity, in Pa*s. defaults to 8.9e-4
		time 			  (str, optional) : turn on to print timestamp after parts of the simulation. 
											defaults to "off"
		field  			  (str, optional) : "ac" or "dc". defaults to "ac"
		cut  			  (int, optional) : cuts data points saved down by a factor. defaults to 10 (saves 
											every 10th point)

	Returns:
		magData					  (array) : first column records applied field, second column records
											average z magnetization for all particles
	"""
	#set initial values
	gamma, boxLength, rAvg, angFreq, dt, timeSteps, volumes, hVolumes, betas, betas2, kValues, k2Values, \
		fieldAmp = calculate_values(shape, kBulk, kSurface, K, K2, kSigma, Ms, fieldFreq, fieldAmp, \
		concentration, numTimeSteps, numParticles, diameter, sigma, hDiameter, hSigma, coating, cycles)
	
	#initialize empty matrix to be saved
	magData = np.zeros((int(numTimeSteps/(cycles*cut)),2,numReps))

	#loop over iterations
	for x in range(numReps):
		#initialize particles and fields
		particleMoments, particleAxes, mStart, nStart, hApplied, particleCoords, ghostCoords, masks, \
			distMatrixSq, distMatrix = initialize_all(numParticles, numTimeSteps, cluster, boxLength, \
			hDiameter, aligned, field, fieldAmp, angFreq, timeSteps, shape, rAvg)

		#thermalize without applied field
		particleMoments, particleAxes = thermalize(numTimeSteps, numParticles, particleMoments, particleAxes, \
			diameter, alpha, Ms, gamma, viscosity, temperature, volumes, hVolumes, kValues, k2Values, betas, \
			betas2, shape, brownian, dt, interactions, ghostCoords, distMatrix, distMatrixSq, masks)

		#main simulation with applied field
		particleMoments = run_simulation(numTimeSteps, numParticles, particleMoments, particleAxes, diameter, \
			alpha, Ms, gamma,viscosity, temperature, volumes, hVolumes, kValues, k2Values, betas, betas2, \
			shape, brownian, dt, interactions, ghostCoords, distMatrix, distMatrixSq, masks, hApplied)

		#fill in data matrix to save
		magData[:,0,x] = hApplied[:int(numTimeSteps/cycles):cut]*1000   #convert field back to mT

		#throws away data if NaN values. otherwise saves last cycle of points
		if np.isnan(particleMoments).any() == True: print("Iteration " + str(x + 1) + " skipped from nan")
		else: 
			#average over all particles
			magData[:,1,x] = np.mean(particleMoments[:,2,:],axis=0)[int(numTimeSteps/cycles)*(cycles-1)::cut]   
			print("Iteration " + str(x + 1) + " successful")

	#average over all iterations
	magData = np.mean(magData, axis=2)

	#export to .csv
	if savefile: np.savetxt(str(savefile)+".csv", magData, delimiter=",")

	return magData

def timeit(method):
	"""Returns time elapsed by a particular function."""
	def timed(*args, **kw):
		ts = time.time()   #start time
		result = method(*args, **kw)
		te = time.time()   #end time
		microsec = timedelta(microseconds=(te-ts)*1000*1000)   #elapsed time in microseconds
		d = datetime(1,1,1) + microsec   #time format
		print(method.__name__ +  " %d:%d:%d:%d.%d" % (d.day-1, d.hour, d.minute, d.second, d.microsecond/1000))
		return result
	return timed

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
	#---adjust units
	diameter *= 1e-9    #convert to meters
	hDiameter *= 1e-9   #convert to meters
	fieldFreq *= 1000 	#convert to Hz
 	fieldAmp *= 0.001    #convert to Tesla

 	if coating != None: hDiameter = diameter + coating*1e-9   #hydrodynamic diameter
	if shape == "cubic": K = kBulk + 6*kSurface/diameter   #calculate effective K from Bulk and Surface

	H_k = 2*np.abs(K)/float(Ms)   #anisotropy field
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
		k2Values, fieldAmp

def initialize_empty_matrices(numParticles, numTimeSteps, shape):

	"""
	Creates empty matrices for particle moments and axes.

	Returns:
		particleMoments 	      (array) : empty array of particle moments
		particleAxes			  (array) : empty array of particle axes
	"""
	particleMoments = np.zeros((numParticles,3,numTimeSteps))   #initialize moment matrix
	if shape == "uniaxial": particleAxes = np.zeros((numParticles,3,2))   #initialize anisotropy axes
	if shape == "cubic": particleAxes = np.zeros((numParticles,9,2))   #initialize anisotropy axes

	return particleMoments, particleAxes

def initialize_particles(particleMoments, particleAxes, numParticles, numTimeSteps, cluster, boxLength, \
	hDiameter, aligned, field, fieldAmp, angFreq, timeSteps, shape):

	"""
	Initializes particle positions, moments, axes, and applied field.

	Returns:
		particleMoments 	      (array) : array of particle moments
		particleAxes			  (array) : array of particle axes
		mStart					  (array) : initial particle moments
		nStart 					  (array) : initial particle axes
		particleCoords			  (array) : particle spatial coordinates
	"""

	if cluster == 0: particleCoords = np.random.rand(numParticles,3)*boxLength   #random positions
	if cluster == 1:   #chain
		cTheta = np.random.rand(1)*np.pi/2.   #generate random angle for chain orientation
		cPhi = np.random.rand(1)*np.pi/2.
		particleCoords = np.zeros((numParticles,3))
		particleCoords[:,0] = np.arange(0,numParticles,1)*hDiameter*np.sin(cTheta)*np.cos(cPhi)   #stack particles in chain formation
		particleCoords[:,1] = np.arange(0,numParticles,1)*hDiameter*np.sin(cTheta)*np.sin(cPhi)
		particleCoords[:,2] = np.arange(0,numParticles,1)*hDiameter*np.cos(cTheta)

	if aligned == "on": particleMoments[:,2,0] = 1   #aligns all moments to z-direction
	else:
		mTheta = np.random.rand(numParticles)*np.pi   #generate random angles for moment orientation
		mPhi = np.random.rand(numParticles)*2*np.pi
		particleMoments[:,0,0] = np.sin(mTheta)*np.cos(mPhi) #set moments to random orientations
		particleMoments[:,1,0] = np.sin(mTheta)*np.sin(mPhi)
		particleMoments[:,2,0] = np.cos(mTheta)

	mStart = np.copy(particleMoments[:,:,0])   #preserves initial conditions

	if aligned == "on": particleAxes[:,2,0] = 1   #aligns all axes to z-direction
	else:
		nTheta = np.random.rand(numParticles)*np.pi   #generate random angle for axis orientation
		nPhi = np.random.rand(numParticles)*2*np.pi
		particleAxes[:,0,0] = np.sin(nTheta)*np.cos(nPhi) #set axes to random orientations
		particleAxes[:,1,0] = np.sin(nTheta)*np.sin(nPhi)
		particleAxes[:,2,0] = np.cos(nTheta)

	if shape == "cubic":
		if aligned == "on": particleAxes[:,3,0] = particleAxes[:,7,0] = 1   #aligns secondary axes to x and y directions
		else:
			rTheta = np.random.rand(numParticles)*np.pi   #generate random angles for secondary axes
			rPhi = np.random.rand(numParticles)*2*np.pi	
			rN = np.zeros((numParticles, 3))
			rN[:,0] = np.sin(rTheta)*np.cos(rPhi)   #generate random vector
			rN[:,1] = np.sin(rTheta)*np.sin(rPhi)
			rN[:,2] = np.cos(rTheta)
			particleAxes[:,3:6,0] = np.cross(particleAxes[:,:3,0], rN)  #set secondary axis direction
			particleAxes[:,6:,0] = np.cross(particleAxes[:,:3,0], particleAxes[:,3:6,0])   #set secondary axis direction

	nStart = np.copy(particleAxes[:,:,0])   #preserve initial condition

	hApplied = fieldAmp*np.cos(angFreq*timeSteps)   #generate applied field
	if field == "dc": hApplied.fill(fieldAmp)   #generate applied dc field

	return particleMoments, particleAxes, mStart, nStart, hApplied, particleCoords

def initialize_ghost_coords(particleCoords, rAvg, boxLength):

	"""
	Initialize "ghost" coordinates to implement periodic boundary conditions. Copies particles within rAvg/3 
	of the edges.

	Returns:
		ghostCoords 	 	      (array) : array of coordinates of "ghost" matrix
		masks					   (list) : list containing 6 arrays of masks used for periodic BCs
	"""

	ghostCoords = particleCoords

	#copy and extend in the x direction
	maskX1 = ghostCoords[:,0] < rAvg/3
	maskX2 = ghostCoords[:,0] > boxLength-rAvg/3

	x_ext1 = ghostCoords[maskX1]
	x_ext1[:,0] += boxLength
	x_ext2 = ghostCoords[maskX2]
	x_ext2[:,0] -= boxLength

	ghostCoords = np.vstack((ghostCoords,x_ext1))
	ghostCoords = np.vstack((ghostCoords,x_ext2))

	#copy and extend in the y direction
	maskY1 = ghostCoords[:,1] < rAvg/3
	maskY2 = ghostCoords[:,1] > boxLength-rAvg/3

	y_ext1 = ghostCoords[maskY1]
	y_ext1[:,1] += boxLength
	y_ext2 = ghostCoords[maskY2]
	y_ext2[:,1] -= boxLength

	ghostCoords = np.vstack((ghostCoords,y_ext1))
	ghostCoords = np.vstack((ghostCoords,y_ext2))

	#copy and extend in the z direction
	maskZ1 = ghostCoords[:,2] < rAvg/3
	maskZ2 = ghostCoords[:,2] > boxLength-rAvg/3

	z_ext1 = ghostCoords[maskZ1]
	z_ext1[:,2] += boxLength
	z_ext2 = ghostCoords[maskZ2]
	z_ext2[:,2] -= boxLength

	ghostCoords = np.vstack((ghostCoords,z_ext1))
	ghostCoords = np.vstack((ghostCoords,z_ext2))

	#preserve and combine masks
	masks = [maskX1, maskX2, maskY1, maskY2, maskZ1, maskZ2]

	return ghostCoords, masks

@timeit
def initialize_all(numParticles, numTimeSteps, cluster, boxLength, hDiameter, aligned, field, \
	fieldAmp, angFreq, timeSteps, shape, rAvg):
	"""Initializes all particles and fields."""
	#create empty matrices
	particleMoments, particleAxes = initialize_empty_matrices(numParticles, numTimeSteps, shape)

	#fill matrices, initialize particles and fields
	particleMoments, particleAxes, mStart, nStart, hApplied, particleCoords = initialize_particles(particleMoments, \
		particleAxes, numParticles, numTimeSteps, cluster, boxLength, hDiameter, aligned, field, fieldAmp, angFreq, \
		timeSteps, shape)
	
	#initialize "ghost" coordinates for periodic boundary conditions
	ghostCoords, masks = initialize_ghost_coords(particleCoords, rAvg, boxLength)
	distMatrixSq = repmat(ghostCoords, len(ghostCoords), 1) - repeat(ghostCoords, len(ghostCoords), axis=0)
	distMatrixSq = distMatrixSq.reshape((len(ghostCoords), len(ghostCoords), 3))
	distMatrix = np.sqrt(np.sum(distMatrixSq**2, axis = 2))

	return particleMoments, particleAxes, mStart, nStart, hApplied, particleCoords, ghostCoords, masks, \
		distMatrixSq, distMatrix

def make_ghost_matrix(M, masks):

	"""
	Initialize "ghost" matrix to implement periodic boundary conditions.

	Returns:
		ghostMatrix					  (array) : array of "ghost" particle moments for periodic BCs
	"""

	#initialize ghost matrix
	ghostMatrix = M

	#copy moments in x direction
	mX1 = np.copy(ghostMatrix[masks[0]])
	mX2 = np.copy(ghostMatrix[masks[1]])

	ghostMatrix = np.vstack((ghostMatrix,mX1))
	ghostMatrix = np.vstack((ghostMatrix,mX2))

	#copy moments in y direction
	mY1 = np.copy(ghostMatrix[masks[2]])
	mY2 = np.copy(ghostMatrix[masks[3]])

	ghostMatrix = np.vstack((ghostMatrix,mY1))
	ghostMatrix = np.vstack((ghostMatrix,mY2))

	#copy moments in z direction
	mZ1 = np.copy(ghostMatrix[masks[4]])
	mZ2 = np.copy(ghostMatrix[masks[5]])

	ghostMatrix = np.vstack((ghostMatrix,mZ1))
	ghostMatrix = np.vstack((ghostMatrix,mZ2))

	return ghostMatrix

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
	if shape == "uniaxial": field = 2*betas[:,None]*np.sum(M*A,axis=1)[:,None]*A
	if shape == "cubic": 
		mx = args[0]
		my = args[1]
		mz = args[2]
		nz = A[:,:3]
		nx = A[:,3:6]
		ny = A[:,6:]
		betas2 = args[3]
		field = -betas[:,None]*(mx**2*mz*nz + my**2*mz*nz + mx**2*my*ny + mz**2*my*ny + my**2*mx*nx + mz**2*mx*nx) - 2*betas2[:,None]*(mx**2*my**2*mz*nz + mx**2*mz**2*my*ny + my**2*mz**2*mx*nx)

	return field

def calculate_interaction_field(M, ghostCoords, RR, R, numParticles, masks, diameter, Ms):
	ghostMatrix = make_ghost_matrix(M, masks)
	mask = np.zeros((len(ghostCoords),len(ghostCoords),3))
	mask[:,:,0] = reduce(np.logical_or, [R == 0])
	mask[:,:,1] = reduce(np.logical_or, [R == 0])
	mask[:,:,2] = reduce(np.logical_or, [R == 0])
	Mask = ma.masked_array((mu0*diameter**3/8.*Ms*(3*np.sum(RR*ghostMatrix,axis=2)[:,:,None]*RR/R[:,:,None]**2 - ghostMatrix))/R[:,:,None]**3, mask=mask, fill_value = 0)
	dipoleField = np.sum(Mask.filled(0)[0:numParticles,:,:],axis=1)
	return dipoleField

def rotate_axes(A, numParticles):
	
	U = np.cross(A[:,:3,0],A[:,:3,1],axis=1)
	U /= np.sqrt(U[:,0]**2 + U[:,1]**2 + U[:,2]**2)[:,None]
	angle = np.arccos(np.sum(A[:,:3,0]*A[:,:3,1],axis=1))
	rotMatrix = np.array([[np.cos(angle) + (U[:,0]**2)*(1 - np.cos(angle)), U[:,0]*U[:,1]*(1 - np.cos(angle)) - U[:,2]*np.sin(angle), U[:,0]*U[:,2]*(1 - np.cos(angle)) + U[:,1]*np.sin(angle)],[U[:,0]*U[:,1]*(1 - np.cos(angle)) + U[:,2]*np.sin(angle), np.cos(angle) + (U[:,1]**2)*(1 - np.cos(angle)), U[:,2]*U[:,1]*(1 - np.cos(angle)) - U[:,0]*np.sin(angle)],[U[:,0]*U[:,2]*(1 - np.cos(angle)) - U[:,1]*np.sin(angle),U[:,2]*U[:,1]*(1 - np.cos(angle)) + U[:,0]*np.sin(angle), np.cos(angle) + (U[:,2]**2)*(1 - np.cos(angle))]])
	nx2 = np.zeros((numParticles,3))
	ny2 = np.zeros((numParticles,3))
	for p in range(3):
		A[:,3+p,1] = rotMatrix[p,0,:]*A[:,3,0] + rotMatrix[p,1,:]*A[:,4,0] + rotMatrix[p,2,:]*A[:,5,0]
		A[:,6+p,1] = rotMatrix[p,0,:]*A[:,6,0] + rotMatrix[p,1,:]*A[:,7,0] + rotMatrix[p,2,:]*A[:,8,0]

	return A

@timeit
def thermalize(numTimeSteps, numParticles, particleMoments, particleAxes, diameter, alpha, Ms, gamma, viscosity, temperature, volumes, \
	hVolumes, kValues, k2Values, betas, betas2, shape, brownian, dt, interactions, ghostCoords, distMatrix, distMatrixSq, masks):
	for n in range(int(numTimeSteps/3)):
		dW = generate_fluctuations(numParticles, alpha, Ms, gamma, viscosity, temperature, volumes, hVolumes, dt, "H")

		if shape == "cubic": mx, my, mz = calculate_cosines(particleMoments[:,:,n], particleAxes[:,:,0])

		if brownian == "on": 
			dT = generate_fluctuations(numParticles, alpha, Ms, gamma, viscosity, temperature, volumes, hVolumes, dt, "T")
			if shape == "uniaxial": torque = calculate_torque(particleMoments[:,:,n], particleAxes[:,:,0], kValues, volumes, shape)
			if shape == "cubic": torque = calculate_torque(particleMoments[:,:,n], particleAxes[:,:,0], kValues, volumes, shape, mx, my, mz, k2Values)		

		if shape == "uniaxial": field = calculate_field(particleMoments[:,:,n], particleAxes[:,:,0], betas, shape)
		if shape == "cubic": field = calculate_field(particleMoments[:,:,n], particleAxes[:,:,0], betas, shape, mx, my, mz, betas2)

		if interactions == "on":
			dipoleField = calculate_interaction_field(particleMoments[:,:,n], ghostCoords, distMatrixSq, distMatrix, numParticles, masks, diameter, Ms)
			field += dipoleField
		
		mBar = particleMoments[:,:,n] + gamma*(1+alpha**2)**(-1)*((np.cross(particleMoments[:,:,n],field) \
			- alpha*np.cross(particleMoments[:,:,n], np.cross(particleMoments[:,:,n],field)))*dt + np.cross(particleMoments[:,:,n],dW) \
			- alpha*np.cross(particleMoments[:,:,n], np.cross(particleMoments[:,:,n],dW)))
		mBar /= np.sqrt(np.einsum('...i,...i', mBar, mBar))[:,None]

		if brownian == "on":
			nBar = particleAxes[:,:3,0] + (6*viscosity*hVolumes[:,None])**(-1)*(np.cross(torque,particleAxes[:,:3,0])*dt + np.cross(dT,particleAxes[:,:3,0]))
			nBar /= np.sqrt(np.einsum('...i,...i', nBar, nBar))[:,None]

			if shape == "uniaxial": tBar = calculate_torque(mBar, nBar, kValues, volumes, shape)
				
			if shape == "cubic":
				nxBar = particleAxes[:,3:6,0] + (6*viscosity*hVolumes[:,None])**(-1)*(np.cross(torque,particleAxes[:,3:6,0])*dt + np.cross(dT,particleAxes[:,3:6,0]))
				nyBar = particleAxes[:,6:,0] + (6*viscosity*hVolumes[:,None])**(-1)*(np.cross(torque,particleAxes[:,6:,0])*dt + np.cross(dT,particleAxes[:,6:,0]))
				nBar = np.concatenate((nBar, nxBar, nyBar), axis=1)
				mxBar, myBar, mzBar = calculate_cosines(mBar, nBar)
				tBar = calculate_torque(mBar, nBar, kValues, volumes, shape, mxBar, myBar, mzBar, k2Values)

			particleAxes[:,:3,1] = particleAxes[:,:3,0] + (6*viscosity*hVolumes[:,None])**(-1)*0.5*(dt*(np.cross(torque,particleAxes[:,:3,0]) + np.cross(tBar,nBar[:,:3])) + np.cross(dT,particleAxes[:,:3,0]) + np.cross(dT,nBar[:,:3]))
			particleAxes[:,:3,1] /= np.sqrt(np.einsum('...i,...i', particleAxes[:,:3,1], particleAxes[:,:3,1]))[:,None]

		else: 
			nBar, particleAxes[:,:,1] = particleAxes[:,:,0] 
			if shape == "cubic": mxBar, myBar, mzBar = calculate_cosines(mBar, nBar)

		if shape == "uniaxial": hBar = calculate_field(mBar, nBar, betas, shape)
		if shape == "cubic": hBar = calculate_field(mBar, nBar, betas, shape, mxBar, myBar, mzBar, betas2)

		if interactions == "on":
			dipoleFieldBar = calculate_interaction_field(mBar, ghostCoords, distMatrixSq, distMatrix, numParticles, masks, diameter, Ms)
			hBar += dipoleFieldBar

		particleMoments[:,:,n+1] = particleMoments[:,:,n] + gamma*(1+alpha**2)**(-1)*(0.5*(dt*(np.cross(particleMoments[:,:,n],field) - alpha*np.cross(particleMoments[:,:,n], np.cross(particleMoments[:,:,n],field)) + np.cross(mBar,hBar) - alpha*np.cross(mBar, np.cross(mBar,hBar))) + np.cross(particleMoments[:,:,n],dW) - alpha*np.cross(particleMoments[:,:,n], np.cross(particleMoments[:,:,n],dW)) + np.cross(mBar,dW) - alpha*np.cross(mBar, np.cross(mBar,dW))))
		particleMoments[:,:,n+1] /= np.sqrt(np.einsum('...i,...i', particleMoments[:,:,n+1], particleMoments[:,:,n+1]))[:,None]

		if shape == "cubic" and brownian == "on":
			particleAxes = rotate_axes(particleAxes, numParticles)

		particleAxes[:,:,0] = particleAxes[:,:,1]

		if n == int(numTimeSteps/5): 
			copyM = np.copy(particleMoments[:,:,n+1])
			copyA = np.copy(particleAxes[:,:3,0])

	particleMoments[:,:,0] = copyM
	particleAxes[:,:3,0] = copyA

	return particleMoments, particleAxes

@timeit
def run_simulation(numTimeSteps, numParticles, particleMoments, particleAxes, diameter, alpha, Ms, gamma, viscosity, temperature, volumes, \
	hVolumes, kValues, k2Values, betas, betas2, shape, brownian, dt, interactions, ghostCoords, distMatrix, distMatrixSq, masks, hApplied):
	for n in range(numTimeSteps-1):
		dW = generate_fluctuations(numParticles, alpha, Ms, gamma, viscosity, temperature, volumes, hVolumes, dt, "H")

		if shape == "cubic": mx, my, mz = calculate_cosines(particleMoments[:,:,n], particleAxes[:,:,0])

		if brownian == "on": 
			dT = generate_fluctuations(numParticles, alpha, Ms, gamma, viscosity, temperature, volumes, hVolumes, dt, "T")
			if shape == "uniaxial": torque = calculate_torque(particleMoments[:,:,n], particleAxes[:,:,0], kValues, volumes, shape)
			if shape == "cubic": torque = calculate_torque(particleMoments[:,:,n], particleAxes[:,:,0], kValues, volumes, shape, mx, my, mz, k2Values)		

		if shape == "uniaxial": field = calculate_field(particleMoments[:,:,n], particleAxes[:,:,0], betas, shape)
		if shape == "cubic": field = calculate_field(particleMoments[:,:,n], particleAxes[:,:,0], betas, shape, mx, my, mz, betas2)

		field[:,2] += hApplied[n]

		if interactions == "on":
			dipoleField = calculate_interaction_field(particleMoments[:,:,n], ghostCoords, distMatrixSq, distMatrix, numParticles, masks, diameter, Ms)
			field += dipoleField
		
		mBar = particleMoments[:,:,n] + gamma*(1+alpha**2)**(-1)*((np.cross(particleMoments[:,:,n],field) \
			- alpha*np.cross(particleMoments[:,:,n], np.cross(particleMoments[:,:,n],field)))*dt + np.cross(particleMoments[:,:,n],dW) \
			- alpha*np.cross(particleMoments[:,:,n], np.cross(particleMoments[:,:,n],dW)))
		mBar /= np.sqrt(np.einsum('...i,...i', mBar, mBar))[:,None]

		if brownian == "on":
			nBar = particleAxes[:,:3,0] + (6*viscosity*hVolumes[:,None])**(-1)*(np.cross(torque,particleAxes[:,:3,0])*dt + np.cross(dT,particleAxes[:,:3,0]))
			nBar /= np.sqrt(np.einsum('...i,...i', nBar, nBar))[:,None]

			if shape == "uniaxial": tBar = calculate_torque(mBar, nBar, kValues, volumes, shape)
				
			if shape == "cubic":
				nxBar = particleAxes[:,3:6,0] + (6*viscosity*hVolumes[:,None])**(-1)*(np.cross(torque,particleAxes[:,3:6,0])*dt + np.cross(dT,particleAxes[:,3:6,0]))
				nyBar = particleAxes[:,6:,0] + (6*viscosity*hVolumes[:,None])**(-1)*(np.cross(torque,particleAxes[:,6:,0])*dt + np.cross(dT,particleAxes[:,6:,0]))
				nBar = np.concatenate((nBar, nxBar, nyBar), axis=1)
				mxBar, myBar, mzBar = calculate_cosines(mBar, nBar)
				tBar = calculate_torque(mBar, nBar, kValues, volumes, shape, mxBar, myBar, mzBar, k2Values)

			particleAxes[:,:3,1] = particleAxes[:,:3,0] + (6*viscosity*hVolumes[:,None])**(-1)*0.5*(dt*(np.cross(torque,particleAxes[:,:3,0]) + np.cross(tBar,nBar[:,:3])) + np.cross(dT,particleAxes[:,:3,0]) + np.cross(dT,nBar[:,:3]))
			particleAxes[:,:3,1] /= np.sqrt(np.einsum('...i,...i', particleAxes[:,:3,1], particleAxes[:,:3,1]))[:,None]

		else: 
			nBar, particleAxes[:,:,1] = particleAxes[:,:,0] 
			if shape == "cubic": mxBar, myBar, mzBar = calculate_cosines(mBar, nBar)

		if shape == "uniaxial": hBar = calculate_field(mBar, nBar, betas, shape)
		if shape == "cubic": hBar = calculate_field(mBar, nBar, betas, shape, mxBar, myBar, mzBar, betas2)

		hBar[:,2] += hApplied[n+1]

		if interactions == "on":
			dipoleFieldBar = calculate_interaction_field(mBar, ghostCoords, distMatrixSq, distMatrix, numParticles, masks, diameter, Ms)
			hBar += dipoleFieldBar

		particleMoments[:,:,n+1] = particleMoments[:,:,n] + gamma*(1+alpha**2)**(-1)*(0.5*(dt*(np.cross(particleMoments[:,:,n],field) - alpha*np.cross(particleMoments[:,:,n], np.cross(particleMoments[:,:,n],field)) + np.cross(mBar,hBar) - alpha*np.cross(mBar, np.cross(mBar,hBar))) + np.cross(particleMoments[:,:,n],dW) - alpha*np.cross(particleMoments[:,:,n], np.cross(particleMoments[:,:,n],dW)) + np.cross(mBar,dW) - alpha*np.cross(mBar, np.cross(mBar,dW))))
		particleMoments[:,:,n+1] /= np.sqrt(np.einsum('...i,...i', particleMoments[:,:,n+1], particleMoments[:,:,n+1]))[:,None]

		if shape == "cubic" and brownian == "on":
			particleAxes = rotate_axes(particleAxes, numParticles)

		particleAxes[:,:,0] = particleAxes[:,:,1]

	return particleMoments


simulate_MH(numParticles=10, numReps=1, numTimeSteps=10, shape = "cubic", interactions="on")




