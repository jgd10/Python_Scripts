import pySALESetup as pss
import numpy as np
import scipy.spatial as scsp
import scipy as sc
import random
import matplotlib.pyplot as plt
import time


T_length   = .5e-3
L_length   = 1.e-3
GRIDSPC    = 2.e-5
T_cells    = int(T_length/GRIDSPC) 
L_cells    = int(L_length/GRIDSPC)	# T - Transverse, L - Longitudinal

pss.generate_mesh(L_cells,T_cells,mat_no=1,GS=GRIDSPC)
pss.mesh_Shps[0,:,:],part_area = pss.gen_circle(r)
mats = pss.mats

l1 = 0.5*L_length
l2 = 0.9*L_length
t1 = 0.1*T_length
t2 = 0.9*T_length

pss.fill_rectangle(l1,t1,l2,t2,mats[0])

plt.figure()
#plt.imshow(pss.mesh,interpolation='nearest',cmap='binary')
plt.imshow(pss.materials[0,:,:],interpolation='nearest',cmap='binary')
plt.show()

pss.save_general_mesh()
