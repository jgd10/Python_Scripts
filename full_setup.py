import pySALESetup as pss
import numpy as np
import scipy.spatial as scsp
import scipy as sc
import random
import matplotlib.pyplot as plt
import time


L_cells    = 224 				# T - Transverse, L - Longitudinal
T_cells    = 407 
T_length   = 25.4e-3
L_length   = 14.0e-3
GRIDSPC    = L_length/L_cells

pss.generate_mesh(L_cells,T_cells,mat_no=5,GS=GRIDSPC)
mats = pss.mats

cu_1x = np.array([0,25.4,0,25.4,0])*1.e-3
cu_1y = np.array([27,27,28,28,27])*1.e-3

cu_2x = np.array([7.7,17.7,7.7,17.7,7.7])*1.e-3
cu_2y = np.array([28,28,29,29,28])*1.e-3

pss.fill_arbitrary_shape_Phys(cu_1y,cu_1x,mats[0])
pss.fill_arbitrary_shape_Phys(cu_2y,cu_2x,mats[0])

#################

al_1x = np.array([6.7,7.7,6.7,7.7,6.7])*1.e-3
al_1y = np.array([28,28,41,41,28])*1.e-3

al_2x = np.array([17.7,18.7,17.7,18.7,17.7])*1.e-3
al_2y = np.array([28,28,41,41,28])*1.e-3

pss.fill_arbitrary_shape_Phys(al_1y,al_1x,mats[1])
pss.fill_arbitrary_shape_Phys(al_2y,al_2x,mats[1])

################

sip_x = np.array([7.7,17.7,7.7,17.7,7.7])*1.e-3
sip_y = np.array([29,29,35,35,29])*1.e-3

pss.fill_arbitrary_shape_Phys(sip_y,sip_x,mats[2])

################

pmma_x= np.array([7.7,17.7,7.7,17.7,7.7])*1.e-3
pmma_y= np.array([35,35,41,41,35])*1.e-3

pss.fill_arbitrary_shape_Phys(pmma_y,pmma_x,mats[3])

plt.figure()
plt.imshow(pss.materials[0,:,:],interpolation='nearest',cmap='binary')
plt.imshow(pss.materials[1,:,:],interpolation='nearest',cmap='binary',alpha=.5)
plt.imshow(pss.materials[2,:,:],interpolation='nearest',cmap='binary',alpha=.5)
plt.imshow(pss.materials[3,:,:],interpolation='nearest',cmap='binary',alpha=.5)
plt.imshow(pss.materials[4,:,:],interpolation='nearest',cmap='binary',alpha=.5)
plt.show()

pss.save_general_mesh()
