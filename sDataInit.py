import numpy as np
import matplotlib.pyplot as plt
import VTLib as vt



########################################
# Import Data
npFile = np.load('/home/brian/local/VT/sample.npz')
xx = npFile['x']
xx = xx[0]
xx = xx[0:5]
yy = npFile['y']
yy = yy[0]
yy = yy[0:5]

nParticles = len(xx)

x = np.zeros(nParticles, dtype=object)
for i in range(0,nParticles):
    x[i] = np.concatenate((xx[i],yy[i]), axis=1)


########################################
# Set constants
tol = 65E-9
theta0 = 0
fps = 200.0
dt = 1/fps
dx = 162E-9
dy = dx
lengthMin = 200E-9
maxTau = 500E-3
slopeMin = 100E-3
slopeMax = 200E-3
smoothFactor = 30
cutoffRad = 350E-9
minTime = 200


params = {'tol':tol, 'theta0':theta0, 'fps':fps, 'dt':dt, 'dx':dx,'dy':dy, 'lengthMin':lengthMin, 'maxTau':maxTau, 'slopeMin':slopeMin, 'slopeMax':slopeMax, 'smoothFactor':smoothFactor, 'cutoffRad':cutoffRad, 'minTime':minTime}


v1 = vt.tmsd(params, x)
v1.indexVec = [0]

