import pySALESetup as pss
import numpy as np
import scipy.spatial as scsp
import scipy as sc
import random
import matplotlib.pyplot as plt
import time

def func(x):
    off_set = 5.32e-3
    theta = 2.*np.pi*x/1.e-3
    y = .24e-3*np.cos(theta) + off_set
    return y
    

#L_cells    = 1000 				# T - Transverse, L - Longitudinal
#T_cells    = 300 
#L_cells    = 100 				# T - Transverse, L - Longitudinal
#T_cells    = 407 
r = 8
L_length   = 4.e-3
T_length   = 10.32e-3
GRIDSPC    = 11.85e-6
T_cells    = int(T_length/GRIDSPC)
L_cells    = int(L_length/GRIDSPC)
pss.generate_mesh(L_cells,T_cells,CPPR=r,mat_no=2,GridSpc=GRIDSPC)
mats = pss.mats
mats = mats.astype(int)
"""
Cu1_T1 = 0.
Cu1_T2 = 10.e-3
Cu1_L1 = 0.
Cu1_L2 = 1.3e-3
pss.fill_rectangle(Cu1_L1,Cu1_T1,Cu1_L2,Cu1_T2,mats[0])


Al1_T1 = 6.55e-3
Al1_T2 = 7.7e-3
Al1_L1 = 1.3e-3
Al1_L2 = 13.5e-3
pss.fill_rectangle(Al1_L1,Al1_T1,Al1_L2,Al1_T2,mats[1])

Al2_T1 = 17.7e-3
Al2_T2 = 18.85e-3
Al2_L1 = 1.3e-3
Al2_L2 = 13.5e-3
pss.fill_rectangle(Al2_L1,Al2_T1,Al2_L2,Al2_T2,mats[1])

Al3_T1 = 18.85e-3
Al3_T2 = 25.4e-3
Al3_L1 = 1.3e-3
Al3_L2 = 2.5e-3
pss.fill_rectangle(Al3_L1,Al3_T1,Al3_L2,Al3_T2,mats[1])

Al4_T1 = 0.e-3
Al4_T2 = 6.55e-3
Al4_L1 = 1.3e-3
Al4_L2 = 2.5e-3
pss.fill_rectangle(Al4_L1,Al4_T1,Al4_L2,Al4_T2,mats[1])
"""
Cu2_L1 = 0.e-3
Cu2_L2 = 4.e-3
Cu2_T1 = 0.e-3
Cu2_T2 = 5.32e-3
pss.fill_sinusoid(Cu2_L1,Cu2_T1,func,Cu2_L2,mats[0],mixed=False)

Sip_L1 = 0.e-3
Sip_L2 = 4.e-3
Sip_T1 = 0.e-3
Sip_T2 = 10.32e-3
pss.fill_rectangle(Sip_L1,Sip_T1,Sip_L2,Sip_T2,mats[1])



pss.materials = pss.materials[:,::-1,]
#pss.materials = np.transpose(pss.materials)
plt.figure()
#plt.imshow(pss.mesh,interpolation='nearest',cmap='binary')
view_mesh = np.zeros_like((pss.materials[0,:,:]))
for item in mats:
    view_mesh += pss.materials[item-1,:,:]*item
plt.imshow(view_mesh,interpolation='nearest',cmap = 'viridis')
plt.show()

pss.save_general_mesh(fname='meso_m_Alwave_{}x{}.iSALE'.format(T_cells,L_cells),mixed=False,noVel=True)


