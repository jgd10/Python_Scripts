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
cppr_range = pss.cppr_max - pss.cppr_min
r = pss.cppr_mid
pss.mesh_Shps[0,:,:],part_area[0] = pss.gen_circle(r)


ANGLE = 30
ANGLE*= np.pi/180.

TOT_Area   = (cppr*2)*pss.meshx
No_per_row = TOT_Area*vol_frac/np.floor(np.pi*cppr**2.)
Separation = np.floor(pss.meshx/No_per_row)
di         = np.round(np.sqrt(Separation**2. + (cppr*2.)**2.))
dj         = np.round(cppr*2*np.cos(ANGLE))
dind       = np.round(2*cppr*np.sin(ANGLE))
print dind
# The Separation is only for the vert lines case. here we need it for the rotated case
# and must rotate the separation appropriately

M     = 0
j     = cppr
ind_i = np.arange(0,pss.meshx,di)
N = np.size(ind_i)
#ind_i = np.append(ind_i,ind_i[-1]+Separation)
while j < pss.meshy+cppr:
    for item in ind_i:
        if item>-cppr and item < pss.meshx+cppr:
            pss.place_shape(pss.mesh_Shps[0,:,:],j,item,M)
        M += 1
        if M > 4: M = 0
    ind_i += dind
    for k in range(N-1,-1,-1):
        if ind_i[k] > (pss.meshx): 
            ind_i[k] = ind_i[0] - di
            ind_i = np.roll(ind_i,1)
    j   += dj






plt.figure()
plt.imshow(pss.materials[0,:,:],interpolation='nearest',cmap='binary')
plt.imshow(pss.materials[1,:,:],interpolation='nearest',cmap='binary',alpha=.5)
plt.imshow(pss.materials[2,:,:],interpolation='nearest',cmap='binary',alpha=.5)
plt.imshow(pss.materials[3,:,:],interpolation='nearest',cmap='binary',alpha=.5)
plt.imshow(pss.materials[4,:,:],interpolation='nearest',cmap='binary',alpha=.5)
plt.show()

#S = float(np.sum(UC)-4*3)#There are 4 particles and 3 cells overlap per particle
#print "Approximate Volume Fraction = {:3.1f}".format(S/float(lx*ly))

pss.save_general_mesh(mixed=False)


