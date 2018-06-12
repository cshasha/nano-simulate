from __future__ import division
import numpy as np

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

