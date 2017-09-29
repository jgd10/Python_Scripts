import pySALESetup as pss
import numpy as np
import scipy.spatial as scsp
import scipy as sc
import random
import matplotlib.pyplot as plt
import time


L_cells    = 500 				# T - Transverse, L - Longitudinal
T_cells    = 500 
T_length   = 1.e-3
L_length   = 1.e-3
GRIDSPC    = L_length/L_cells

pss.generate_mesh(L_cells,T_cells,mat_no=1)
mats = pss.mats


pss.fill_Allmesh(mats[0])

pss.save_general_mesh(fname='meso_m_500x500block.iSALE',noVel=True)

fig, ax = plt.subplots()
cax = ax.imshow(pss.materials[0,:,:],interpolation='nearest',vmin=0,vmax=1)
cbar = fig.colorbar(cax, orientation='horizontal')
plt.show()
