import pySALESetup as pss
import numpy as np
import scipy.spatial as scsp
import scipy as sc
import random
import matplotlib.pyplot as plt
import time


L_cells    = 224 				# T - Transverse, L - Longitudinal
T_cells    = 407 
r = 8
T_length   = 25.4e-3
L_length   = 14.0e-3
GRIDSPC    = L_length/L_cells

pss.generate_mesh(L_cells,T_cells,CPPR=r,mat_no=5,GridSpc=GRIDSPC)
mats = pss.mats

Cu1_T1 = 0.
Cu1_T2 = 25.4e-3
Cu1_L1 = 0.
Cu1_L2 = 1.e-3
pss.fill_rectangle(Cu1_L1,Cu1_T1,Cu1_L2,Cu1_T2,mats[0])

Cu2_T1 = 7.7e-3
Cu2_T2 = 17.7e-3
Cu2_L1 = 1.e-3
Cu2_L2 = 2.e-3
pss.fill_rectangle(Cu2_L1,Cu2_T1,Cu2_L2,Cu2_T2,mats[0])

Al1_T1 = 6.7e-3
Al1_T2 = 7.7e-3
Al1_L1 = 1.e-3
Al1_L2 = 14.e-3
pss.fill_rectangle(Al1_L1,Al1_T1,Al1_L2,Al1_T2,mats[1])

Al2_T1 = 17.7e-3
Al2_T2 = 18.7e-3
Al2_L1 = 1.e-3
Al2_L2 = 14.e-3
pss.fill_rectangle(Al2_L1,Al2_T1,Al2_L2,Al2_T2,mats[1])

Sip_T1 = 7.7e-3
Sip_T2 = 17.7e-3
Sip_L1 = 2.e-3
Sip_L2 = 8.e-3
pss.fill_rectangle(Sip_L1,Sip_T1,Sip_L2,Sip_T2,mats[2])

PMMA_T1 = 7.7e-3
PMMA_T2 = 17.7e-3
PMMA_L1 = 8.e-3
PMMA_L2 = 14.e-3
pss.fill_rectangle(PMMA_L1,PMMA_T1,PMMA_L2,PMMA_T2,mats[3])

pss.mesh_Shps[0,:,:],part_area = pss.gen_circle(r)

i = np.where(abs(pss.yh-12.7e-3)<pss.GS/2.)
j = np.where(abs(pss.xh-3.e-3)<pss.GS/2.)
I, J, M = i[0],j[0],mats[4]
pss.place_shape(pss.mesh_Shps[0,:,:],J,I,M)

plt.figure()
#plt.imshow(pss.mesh,interpolation='nearest',cmap='binary')
view_mesh = np.zeros_like((pss.materials[0,:,:]))
for item in mats:
    view_mesh += pss.materials[item-1,:,:]*item
plt.imshow(view_mesh,interpolation='nearest',cmap = 'Reds')
#plt.imshow(pss.materials[0,:,:],interpolation='nearest',cmap='copper_r')
#plt.imshow(pss.materials[1,:,:],interpolation='nearest',cmap='BuPu')
#plt.imshow(pss.materials[2,:,:],interpolation='nearest',cmap='viridis')
#plt.imshow(pss.materials[3,:,:],interpolation='nearest',cmap='binary')
#plt.imshow(pss.materials[4,:,:],interpolation='nearest',cmap='Reds_r')
plt.show()

pss.save_general_mesh(mixed=False)
