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

r1 = np.array([0.,92.5])
r2 = np.array([100.,67.5])   # in microns
r3 = np.array([200.,92.5])
#r4 = np.array([300.,67.5])   # in microns
#r5 = np.array([400.,92.5])
#r6 = np.array([500.,67.5])   # in microns

r1 = r1[::-1]
r2 = r2[::-1]
r3 = r3[::-1]

r1 /= GRIDSPC*1.e6
r2 /= GRIDSPC*1.e6
r3 /= GRIDSPC*1.e6

pss.fill_above_line(r1,r2,mats[0],mixed=True)
pss.fill_above_line(r2,r3,mats[0],mixed=True)

#pss.fill_plate(0.,300.,mats[1])

pss.save_general_mesh()

fig, ax = plt.subplots()
cax = ax.imshow(pss.mesh,cmap='Greys',interpolation='nearest',vmin=0,vmax=1)
cbar = fig.colorbar(cax, orientation='horizontal')
plt.show()
