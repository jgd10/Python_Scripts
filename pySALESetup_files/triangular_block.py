import pySALESetup as pss
import numpy as np
import scipy.spatial as scsp
import scipy as sc
import random
import matplotlib.pyplot as plt
import time


L_cells    = 1000 				# T - Transverse, L - Longitudinal
T_cells    = 400 
T_length   = 200.e-6
L_length   = 500.e-6
GRIDSPC    = L_length/L_cells

pss.generate_mesh(L_cells,T_cells,mat_no=1)
mats = pss.mats

X1 = np.arange(0.,100.,2.)
Y1 = X1*25./100.
Y1 = Y1[::-1]

X2 = np.arange(0.,101.,2.)
Y2 = X2*25./100.
X2+= 100.

X  = np.append(X1,X2)
Y  = np.append(Y1,Y2)

Y += 37.5

assert np.size(X) == np.size(Y)

p = 1.
N = np.size(X)
for i in range(N):
	if i%2==0:
		pass
	else:
		Y[i] += p*2.5
		p  *= -1.

X  = np.append(X,(X[-1],X[0])) 
Y  = np.append(Y,([0.,0.])) 
X /= (GRIDSPC/1.e-6)
Y /= (GRIDSPC/1.e-6)

pss.fill_arbitrary_shape_p(Y,X,mats[0])

#pss.fill_plate(0.,300.,mats[1])

pss.save_general_mesh()

fig, ax = plt.subplots()
cax = ax.imshow(pss.mesh,cmap='Greys',interpolation='nearest',vmin=0,vmax=1)
cbar = fig.colorbar(cax, orientation='horizontal')
plt.show()
