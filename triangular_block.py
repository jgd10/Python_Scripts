import pySALESetup as pss
import numpy as np
import scipy.spatial as scsp
import scipy as sc
import random
import matplotlib.pyplot as plt
import time


L_cells    = 500 				# T - Transverse, L - Longitudinal
T_cells    = 200 
T_length   = 200.e-6
L_length   = 500.e-6
GRIDSPC    = L_length/L_cells

pss.generate_mesh(L_cells,T_cells,mat_no=1)
mats = pss.mats

X1 = np.arange(0,100,3.88)
									# THESE ARE JUST LINES. NEED TO COMPLETE THE SHAPES. MORE VERTICES REQD.
Y1 = np.arange(0,25,.97)

X2 = np.copy(X1)
Y2 = np.copy(Y1)
Y2 = Y2[::-1]

Y1 += 37.5
Y2 += 37.5 
X1 += 100.
#r1 = np.array([0.,62.5])
#r2 = np.array([100.,37.5])   # in microns
#r3 = np.array([200.,62.5])
#r4 = np.array([300.,67.5])   # in microns
#r5 = np.array([400.,92.5])
#r6 = np.array([500.,67.5])   # in microns

#r1 = r1[::-1]
#r2 = r2[::-1]
#r3 = r3[::-1]
assert np.size(X1) == np.size(Y1)
p = 1.
N = np.size(X1)
for i in range(N):
	if i%2==0:
		pass
	else:
		Y1 += p*2.5/2.
		Y2 += p*2.5/2.
		p  *= -1.

X1 /= GRIDSPC * 1.e-6
Y1 /= GRIDSPC * 1.e-6
X2 /= GRIDSPC * 1.e-6
Y2 /= GRIDSPC * 1.e-6


#r1 /= GRIDSPC*1.e6
#r2 /= GRIDSPC*1.e6
#r3 /= GRIDSPC*1.e6

pss.fill_arbitrary_shape(X1,Y1,mats[0])

#pss.fill_plate(0.,300.,mats[1])

pss.save_general_mesh()

fig, ax = plt.subplots()
cax = ax.imshow(pss.mesh,cmap='Greys',interpolation='nearest',vmin=0,vmax=1)
cbar = fig.colorbar(cax, orientation='horizontal')
plt.show()
