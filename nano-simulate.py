from __future__ import division
import numpy as np
import pylab as pl
from utils import *
from initialize import *
from calculations import *
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
		fieldAmp, numTimeSteps = calculate_values(shape, kBulk, kSurface, K, K2, kSigma, Ms, fieldFreq, fieldAmp, \
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

@timeit
def thermalize(numTimeSteps, numParticles, particleMoments, particleAxes, diameter, alpha, Ms, gamma, viscosity, \
	temperature, volumes, hVolumes, kValues, k2Values, betas, betas2, shape, brownian, dt, interactions, \
	ghostCoords, distMatrix, distMatrixSq, masks):
	"""Runs simulation without applied field"""
	#main loop over time steps
	for n in range(int(numTimeSteps/3.)):
		#generate thermal fluctuations for field
		dW = generate_fluctuations(numParticles, alpha, Ms, gamma, viscosity, temperature, volumes, hVolumes, \
			dt, "H")
		#calculate direction cosines for cubic anisotropy
		if shape == "cubic": mx, my, mz = calculate_cosines(particleMoments[:,:,n], particleAxes[:,:,0])

		if brownian == "on": 
			#generate thermal fluctuations for torque
			dT = generate_fluctuations(numParticles, alpha, Ms, gamma, viscosity, temperature, volumes, hVolumes, \
				dt, "T")
			#calculate torque on particles
			if shape == "uniaxial": torque = calculate_torque(particleMoments[:,:,n], particleAxes[:,:,0], \
				kValues, volumes, shape)
			if shape == "cubic": torque = calculate_torque(particleMoments[:,:,n], particleAxes[:,:,0], kValues, \
				volumes, shape, mx, my, mz, k2Values)		

		#calculate field on particles
		if shape == "uniaxial": field = calculate_field(particleMoments[:,:,n], particleAxes[:,:,0], betas, \
			shape)
		if shape == "cubic": field = calculate_field(particleMoments[:,:,n], particleAxes[:,:,0], betas, shape, \
			mx, my, mz, betas2)

		if interactions == "on":
			#calculate dipole interaction field
			dipoleField = calculate_interaction_field(particleMoments[:,:,n], ghostCoords, distMatrixSq, \
				distMatrix, numParticles, masks, diameter, Ms)
			#add to total field
			field += dipoleField
		
		#calculate moment predictor
		mBar = particleMoments[:,:,n] + gamma*(1+alpha**2)**(-1)*((np.cross(particleMoments[:,:,n],field) \
			- alpha*np.cross(particleMoments[:,:,n], np.cross(particleMoments[:,:,n],field)))*dt \
			+ np.cross(particleMoments[:,:,n],dW) \
			- alpha*np.cross(particleMoments[:,:,n], np.cross(particleMoments[:,:,n],dW)))
		#normalize moment predictor
		mBar /= np.sqrt(np.einsum('...i,...i', mBar, mBar))[:,None]

		if brownian == "on":
			#calculate axis predictor
			nBar = particleAxes[:,:3,0] + (6*viscosity*hVolumes[:,None])**(-1)*(np.cross(torque,particleAxes[:,:3,0])*dt \
				+ np.cross(dT,particleAxes[:,:3,0]))
			#normalize axis predictor
			nBar /= np.sqrt(np.einsum('...i,...i', nBar, nBar))[:,None]

			#calculate torque predictor(s)
			if shape == "uniaxial": tBar = calculate_torque(mBar, nBar, kValues, volumes, shape)
			if shape == "cubic":
				nxBar = particleAxes[:,3:6,0] + (6*viscosity*hVolumes[:,None])**(-1)*(np.cross(torque,particleAxes[:,3:6,0])*dt \
					+ np.cross(dT,particleAxes[:,3:6,0]))
				nyBar = particleAxes[:,6:,0] + (6*viscosity*hVolumes[:,None])**(-1)*(np.cross(torque,particleAxes[:,6:,0])*dt \
					+ np.cross(dT,particleAxes[:,6:,0]))
				nBar = np.concatenate((nBar, nxBar, nyBar), axis=1)
				mxBar, myBar, mzBar = calculate_cosines(mBar, nBar)
				tBar = calculate_torque(mBar, nBar, kValues, volumes, shape, mxBar, myBar, mzBar, k2Values)

			#calculate new particle axes
			particleAxes[:,:3,1] = particleAxes[:,:3,0] + (6*viscosity*hVolumes[:,None])**(-1)*0.5*(dt*(np.cross(torque,particleAxes[:,:3,0]) \
				+ np.cross(tBar,nBar[:,:3])) + np.cross(dT,particleAxes[:,:3,0]) + np.cross(dT,nBar[:,:3]))
			#normalize new particle axes
			particleAxes[:,:3,1] /= np.sqrt(np.einsum('...i,...i', particleAxes[:,:3,1], particleAxes[:,:3,1]))[:,None]

		else: 
			#set axis predictors equal
			nBar = particleAxes[:,:,0]
			particleAxes[:,:,1] = particleAxes[:,:,0] 
			if shape == "cubic": mxBar, myBar, mzBar = calculate_cosines(mBar, nBar)

		#calculate field predictors
		if shape == "uniaxial": hBar = calculate_field(mBar, nBar, betas, shape)
		if shape == "cubic": hBar = calculate_field(mBar, nBar, betas, shape, mxBar, myBar, mzBar, betas2)

		if interactions == "on":
			dipoleFieldBar = calculate_interaction_field(mBar, ghostCoords, distMatrixSq, distMatrix, \
				numParticles, masks, diameter, Ms)
			hBar += dipoleFieldBar

		#calculate new particle moments
		particleMoments[:,:,n+1] = particleMoments[:,:,n] + gamma*(1+alpha**2)**(-1)*(0.5*(dt*(np.cross(particleMoments[:,:,n],field) \
			- alpha*np.cross(particleMoments[:,:,n], np.cross(particleMoments[:,:,n],field)) + np.cross(mBar,hBar) \
			- alpha*np.cross(mBar, np.cross(mBar,hBar))) + np.cross(particleMoments[:,:,n],dW) \
			- alpha*np.cross(particleMoments[:,:,n], np.cross(particleMoments[:,:,n],dW)) + np.cross(mBar,dW) \
			- alpha*np.cross(mBar, np.cross(mBar,dW))))
		#normalize new particle moments
		particleMoments[:,:,n+1] /= np.sqrt(np.einsum('...i,...i', particleMoments[:,:,n+1], particleMoments[:,:,n+1]))[:,None]

		if shape == "cubic" and brownian == "on":
			#rotate minor axes
			particleAxes = rotate_axes(particleAxes, numParticles)

		#update particle axes
		particleAxes[:,:,0] = particleAxes[:,:,1]

		#save configuration
		if n == int(numTimeSteps/5): 
			copyM = np.copy(particleMoments[:,:,n+1])
			copyA = np.copy(particleAxes[:,:3,0])

	particleMoments[:,:,0] = copyM
	particleAxes[:,:3,0] = copyA

	return particleMoments, particleAxes

@timeit
def run_simulation(numTimeSteps, numParticles, particleMoments, particleAxes, diameter, alpha, Ms, gamma, \
	viscosity, temperature, volumes, hVolumes, kValues, k2Values, betas, betas2, shape, brownian, dt, \
	interactions, ghostCoords, distMatrix, distMatrixSq, masks, hApplied):
	"""Run simulation with applied field."""
	#loop over time steps
	for n in range(numTimeSteps-1):
		#generate thermal fluctuations for field
		dW = generate_fluctuations(numParticles, alpha, Ms, gamma, viscosity, temperature, volumes, hVolumes, \
			dt, "H")
		#calculate direction cosines for cubic anisotropy
		if shape == "cubic": mx, my, mz = calculate_cosines(particleMoments[:,:,n], particleAxes[:,:,0])

		if brownian == "on": 
			#generate thermal fluctuations for torque
			dT = generate_fluctuations(numParticles, alpha, Ms, gamma, viscosity, temperature, volumes, \
				hVolumes, dt, "T")
			#calculate torque on particles
			if shape == "uniaxial": torque = calculate_torque(particleMoments[:,:,n], particleAxes[:,:,0], \
				kValues, volumes, shape)
			if shape == "cubic": torque = calculate_torque(particleMoments[:,:,n], particleAxes[:,:,0], kValues, \
				volumes, shape, mx, my, mz, k2Values)		

		#calculate field on particles
		if shape == "uniaxial": field = calculate_field(particleMoments[:,:,n], particleAxes[:,:,0], betas, \
			shape)
		if shape == "cubic": field = calculate_field(particleMoments[:,:,n], particleAxes[:,:,0], betas, shape, \
			mx, my, mz, betas2)

		#add applied field
		field[:,2] += hApplied[n]

		if interactions == "on":
			#calculate dipole interaction field
			dipoleField = calculate_interaction_field(particleMoments[:,:,n], ghostCoords, distMatrixSq, \
				distMatrix, numParticles, masks, diameter, Ms)
			#add to total field
			field += dipoleField
		
		#calculate moment predictor
		mBar = particleMoments[:,:,n] + gamma*(1+alpha**2)**(-1)*((np.cross(particleMoments[:,:,n],field) \
			- alpha*np.cross(particleMoments[:,:,n], np.cross(particleMoments[:,:,n],field)))*dt \
			+ np.cross(particleMoments[:,:,n],dW) \
			- alpha*np.cross(particleMoments[:,:,n], np.cross(particleMoments[:,:,n],dW)))
		#normalize moment predictor
		mBar /= np.sqrt(np.einsum('...i,...i', mBar, mBar))[:,None]

		if brownian == "on":
			#calculate axis predictor
			nBar = particleAxes[:,:3,0] + (6*viscosity*hVolumes[:,None])**(-1)*(np.cross(torque,particleAxes[:,:3,0])*dt \
				+ np.cross(dT,particleAxes[:,:3,0]))
			#normalize axis predictor
			nBar /= np.sqrt(np.einsum('...i,...i', nBar, nBar))[:,None]

			#calculate torque predictor(s)
			if shape == "uniaxial": tBar = calculate_torque(mBar, nBar, kValues, volumes, shape)
			if shape == "cubic":
				nxBar = particleAxes[:,3:6,0] + (6*viscosity*hVolumes[:,None])**(-1)*(np.cross(torque,particleAxes[:,3:6,0])*dt \
					+ np.cross(dT,particleAxes[:,3:6,0]))
				nyBar = particleAxes[:,6:,0] + (6*viscosity*hVolumes[:,None])**(-1)*(np.cross(torque,particleAxes[:,6:,0])*dt \
					+ np.cross(dT,particleAxes[:,6:,0]))
				nBar = np.concatenate((nBar, nxBar, nyBar), axis=1)
				mxBar, myBar, mzBar = calculate_cosines(mBar, nBar)
				tBar = calculate_torque(mBar, nBar, kValues, volumes, shape, mxBar, myBar, mzBar, k2Values)

			#calculate new particle axes
			particleAxes[:,:3,1] = particleAxes[:,:3,0] + (6*viscosity*hVolumes[:,None])**(-1)*0.5*(dt*(np.cross(torque,particleAxes[:,:3,0]) \
				+ np.cross(tBar,nBar[:,:3])) + np.cross(dT,particleAxes[:,:3,0]) + np.cross(dT,nBar[:,:3]))
			#normalize new particle axes
			particleAxes[:,:3,1] /= np.sqrt(np.einsum('...i,...i', particleAxes[:,:3,1], particleAxes[:,:3,1]))[:,None]

		else: 
			#set axis predictors equal
			nBar = particleAxes[:,:,0]
			particleAxes[:,:,1] = particleAxes[:,:,0] 
			if shape == "cubic": mxBar, myBar, mzBar = calculate_cosines(mBar, nBar)

		#calculate field predictors
		if shape == "uniaxial": hBar = calculate_field(mBar, nBar, betas, shape)
		if shape == "cubic": hBar = calculate_field(mBar, nBar, betas, shape, mxBar, myBar, mzBar, betas2)

		#add applied field predictor
		hBar[:,2] += hApplied[n+1]

		#add dipole field predictor
		if interactions == "on":
			dipoleFieldBar = calculate_interaction_field(mBar, ghostCoords, distMatrixSq, distMatrix, numParticles, masks, diameter, Ms)
			hBar += dipoleFieldBar

		#calculate new particle moments
		particleMoments[:,:,n+1] = particleMoments[:,:,n] + gamma*(1+alpha**2)**(-1)*(0.5*(dt*(np.cross(particleMoments[:,:,n],field) \
			- alpha*np.cross(particleMoments[:,:,n], np.cross(particleMoments[:,:,n],field)) + np.cross(mBar,hBar) \
			- alpha*np.cross(mBar, np.cross(mBar,hBar))) + np.cross(particleMoments[:,:,n],dW) \
			- alpha*np.cross(particleMoments[:,:,n], np.cross(particleMoments[:,:,n],dW)) + np.cross(mBar,dW) \
			- alpha*np.cross(mBar, np.cross(mBar,dW))))
		#normalize new particle moments
		particleMoments[:,:,n+1] /= np.sqrt(np.einsum('...i,...i', particleMoments[:,:,n+1], particleMoments[:,:,n+1]))[:,None]

		if shape == "cubic" and brownian == "on":
			#rotate minor axes
			particleAxes = rotate_axes(particleAxes, numParticles)

		#update particle axes
		particleAxes[:,:,0] = particleAxes[:,:,1]

	return particleMoments


magData = simulate_MH(numParticles=100, numReps=1, numTimeSteps=1000, shape = "uniaxial",K = 6000,brownian="on",interactions="off", cut=10)
#pl.plot(magData[:,0], magData[:,1])
#pl.show()




