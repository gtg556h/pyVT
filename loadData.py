import numpy as np
import matplotlib.pyplot as plt
#import VTLib as vt



########################################
# Import Data
npFile = np.load('/home/brian/local/VT/sample.npz')
x = npFile['x']
x = x[0]
x = x[0:25]
y = npFile['y']
y = y[0]
y = y[0:25]


