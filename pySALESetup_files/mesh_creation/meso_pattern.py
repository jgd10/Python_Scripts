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

r = float(pss.cppr_min)

# Define the pattern
xc = np.array([-1.,0.,0.,1.])
yc = np.array([0.,1.,-1.,0.])
rc = .5
w  = rc*4
h  = rc*4

# scale as rc = r
# i.e. scale factor (sc) = 2r
sc = 2.*r
w *= sc
h *= sc
xc*= sc*GRIDSPC
yc*= sc*GRIDSPC
rc*= sc*GRIDSPC

# Calculate number of occurences
No = X_cells*Y_cells/(w*h)
Nx = int(X_cells/w)
Ny = int(Y_cells/h)
w *= GRIDSPC
h *= GRIDSPC

X = []
Y = []
for i in range(Nx):
	for j in range(Ny):
		X.append(xc+i*w)
		Y.append(yc+j*h)

X       = np.array(X)
Y       = np.array(Y)
N       = np.size(X)
radii   = np.full((N),r)

MAT = pss.mat_assignment(mats,X,Y,radii)
A,B = pss.part_distance(X,Y,radii,MAT,plot=True)

pss.save_spherical_parts(X,Y,radii,MAT,A)
pss.view_mesoiSALE(filepath='meso_A-{:3.4f}.iSALE'.format(A))


