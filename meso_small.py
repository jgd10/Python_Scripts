import numpy as np
import scipy.spatial as scsp
import scipy as sc
import random
import matplotlib.pyplot as plt
import pySALESetup as pss
import time


vol_frac   = .55
X_cells    = 56 
Y_cells    = 56 
PR         = 0.
cppr       = 8 
vfraclimit = .495                               # The changeover point from random to forced contacts. > 1.0 => least contacts; = 0. Max contacts
x_length   = 1.e-4
y_length   = 1.e-4
GRIDSPC    = x_length/X_cells
mat_no     = 5

pss.generate_mesh(X_cells,Y_cells,mat_no,cppr,PR,vol_frac)
mats = pss.mats

part_area  = np.zeros((1))
cppr_range = pss.cppr_max - pss.cppr_min
r = pss.cppr_mid
pss.mesh_Shps[0,:,:],part_area[0] = pss.gen_circle(r)

ANGLE = 60
ANG   = ANGLE*np.pi/180.

pss.place_shape(pss.mesh_Shps[0,:,:],0,28,0)
pss.place_shape(pss.mesh_Shps[0,:,:],16,28,1)

dX = 2*cppr*np.sin(ANG)
dY = 2*cppr*np.cos(ANG) 
pss.place_shape(pss.mesh_Shps[0,:,:],16+dY,28+dX,2)
pss.place_shape(pss.mesh_Shps[0,:,:],32+dY,28+dX,3)
pss.place_shape(pss.mesh_Shps[0,:,:],32+dY,28+dX-16,4)

#pss.place_shape(pss.mesh_Shps[0,:,:],50,25,3)
#pss.place_shape(pss.mesh_Shps[0,:,:],50,41,4)
#pss.place_shape(pss.mesh_Shps[0,:,:],50,9,2)






plt.figure()
plt.imshow(pss.materials[0,:,:],interpolation='nearest',cmap='binary')
plt.imshow(pss.materials[1,:,:],interpolation='nearest',cmap='binary',alpha=.5)
plt.imshow(pss.materials[2,:,:],interpolation='nearest',cmap='binary',alpha=.5)
plt.imshow(pss.materials[3,:,:],interpolation='nearest',cmap='binary',alpha=.5)
plt.imshow(pss.materials[4,:,:],interpolation='nearest',cmap='binary',alpha=.5)
plt.show()

#S = float(np.sum(UC)-4*3)#There are 4 particles and 3 cells overlap per particle
#print "Approximate Volume Fraction = {:3.1f}".format(S/float(lx*ly))

pss.save_general_mesh(fname='meso_m_test-{:3.1f}.iSALE'.format(ANGLE),mixed=False)


