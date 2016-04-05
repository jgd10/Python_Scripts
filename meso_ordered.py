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

r = pss.cppr_min
pss.mesh_Shps[0,:,:],part_area = pss.gen_circle(r)
part_radii = r


NC   = np.ceil(X_cells*r*2.*vol_frac/(np.pi*r**2.))
sepx = x_length/(NC+1)
# NC+1 as centres are nodes and there will be 1 more gap than nodes

r*= GRIDSPC
sepy    = 2.*r
NR      = y_length/sepy + 1
ycoords = np.linspace(0,y_length,num=NR,endpoint=True)
NR      = np.size(ycoords)

xcoords = np.linspace(0,x_length,num=NC,endpoint=True)
NC      = np.size(xcoords)

ycoords = np.repeat(ycoords,NC)
xcoords = np.tile(xcoords,NR)

N       = np.size(xcoords)
radii   = np.full((N),r)

MAT = pss.mat_assignment(mats,xcoords,ycoords,radii)
A,B = pss.part_distance(xcoords,ycoords,radii,MAT,plot=True)

pss.save_spherical_parts(xcoords,ycoords,radii,MAT,A)
pss.view_mesoiSALE(filepath='meso_A-{:3.4f}.iSALE'.format(A))


