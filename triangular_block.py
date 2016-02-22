import pySALESetup as pss
import numpy as np
import scipy.spatial as scsp
import scipy as sc
import random
import matplotlib.pyplot as plt
import time


X_cells    = 1000 
Y_cells    = 1000 
x_length   = 500.e-6
y_length   = 500.e-6
GRIDSPC    = x_length/X_cells

pss.generate_mesh(X_cells,Y_cells,mat_no=1)
mats = pss.mats

r1 = np.array([0.,67.5])   # in microns
r2 = np.array([100.,92.5])
r3 = np.array([200.,67.5])
r4 = np.array([300.,92.5])
r5 = np.array([400.,67.5])
r6 = np.array([500.,92.5])

r1 = r1[::-1]
r2 = r2[::-1]
r3 = r3[::-1]
r4 = r4[::-1]
r5 = r5[::-1]
r6 = r6[::-1]

r1 /= GRIDSPC*1.e6
r2 /= GRIDSPC*1.e6
r3 /= GRIDSPC*1.e6
r4 /= GRIDSPC*1.e6
r5 /= GRIDSPC*1.e6
r6 /= GRIDSPC*1.e6

pss.fill_above_line(r1,r2,mats[0],mixed=True)
pss.fill_above_line(r2,r3,mats[0],mixed=True)
pss.fill_above_line(r3,r4,mats[0],mixed=True)
pss.fill_above_line(r4,r5,mats[0],mixed=True)
pss.fill_above_line(r5,r6,mats[0],mixed=True)

#pss.fill_plate(0.,300.,mats[1])

pss.save_general_mesh()

fig, ax = plt.subplots()
cax = ax.imshow(pss.mesh,cmap='Greys',interpolation='nearest',vmin=0,vmax=1)
cbar = fig.colorbar(cax, orientation='horizontal')
plt.show()
