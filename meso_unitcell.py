import numpy as np
import scipy.spatial as scsp
import scipy as sc
import random
import matplotlib.pyplot as plt
import pySALESetup as pss
import time


vol_frac   = .55
X_cells    = 500 
Y_cells    = 500 
PR         = 0.
cppr       = 8 
vfraclimit = .495                               # The changeover point from random to forced contacts. > 1.0 => least contacts; = 0. Max contacts
x_length   = 1.e-3
y_length   = 1.e-3
GRIDSPC    = x_length/X_cells
mat_no     = 5

pss.generate_mesh(X_cells,Y_cells,mat_no,cppr,PR,vol_frac)
mats = pss.mats

part_area  = np.zeros((1))
part_radii = []
cppr_range = pss.cppr_max - pss.cppr_min
r = pss.cppr_mid
pss.mesh_Shps[0,:,:],part_area[0] = pss.gen_circle(r)
part_radii.append(r)

xcoords = []
ycoords = []
radii   = []
I_shape = []
J_shape = []
lx, ly = 44, 44
UC = pss.unit_cell(LX=lx,LY=ly)

#pss.place_shape(pss.mesh_Shps[0,:,:],0,0,1,UC,LX=lx,LY=ly)
pss.place_shape(pss.mesh_Shps[0,:,:],11,11,2,UC,LX=lx,LY=ly)
pss.place_shape(pss.mesh_Shps[0,:,:],22,22,3,UC,LX=lx,LY=ly)
pss.place_shape(pss.mesh_Shps[0,:,:],33,33,4,UC,LX=lx,LY=ly)
#pss.place_shape(pss.mesh_Shps[0,:,:],lx,lx,1,UC,LX=lx,LY=ly)
pss.place_shape(pss.mesh_Shps[0,:,:],22,lx,1,UC,LX=lx,LY=ly)
pss.place_shape(pss.mesh_Shps[0,:,:],lx-22,0,1,UC,LX=lx,LY=ly)

plt.figure()
plt.imshow(UC[0,:,:],interpolation='nearest',cmap='binary')
plt.imshow(UC[1,:,:],interpolation='nearest',cmap='binary',alpha=.5)
plt.imshow(UC[2,:,:],interpolation='nearest',cmap='binary',alpha=.5)
plt.imshow(UC[3,:,:],interpolation='nearest',cmap='binary',alpha=.5)

pss.copypasteUC(UC)

plt.figure()
plt.imshow(pss.materials[0,:,:],interpolation='nearest',cmap='binary')
plt.imshow(pss.materials[1,:,:],interpolation='nearest',cmap='binary',alpha=.5)
plt.imshow(pss.materials[2,:,:],interpolation='nearest',cmap='binary',alpha=.5)
plt.imshow(pss.materials[3,:,:],interpolation='nearest',cmap='binary',alpha=.5)
plt.show()



