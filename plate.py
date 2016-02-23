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


pss.fill_plate(0.,50.,mats[0])

pss.save_general_mesh()

fig, ax = plt.subplots()
cax = ax.imshow(pss.mesh,cmap='Greys',interpolation='nearest',vmin=0,vmax=1)
cbar = fig.colorbar(cax, orientation='horizontal')
plt.show()
