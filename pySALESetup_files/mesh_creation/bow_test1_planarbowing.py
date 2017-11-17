import pySALESetup as pss
import numpy as np
import scipy.spatial as scsp
import scipy as sc
import random
import matplotlib.pyplot as plt
import time

# Cu top hat part 1: Main disc 14mm x 1.mm
DR1_L1 = 0.
DR1_L2 = 14.e-3
DR1_T1 = 0.
DR1_T2 = 1.e-3

# Cu top hat part 2: Central protrusion 10mm x 1.mm
DR2_L1 = 2.e-3
DR2_L2 = 12.e-3
DR2_T1 = 1.e-3
DR2_T2 = 2.e-3

# Al cell part 1a (LHS): wall that is 1.mm x 11mm in size
CL1_L1 = 1.e-3
CL1_L2 = 2.e-3
CL1_T1 = 1.e-3
CL1_T2 = 12.e-3

# Al cell part 1b (RHS): wall that is 1.mm x 11mm in size
CL2_L1 = 12.e-3
CL2_L2 = 13.e-3
CL2_T1 = 1.e-3
CL2_T2 = 12.e-3
"""
# Al cell part 2b (RHS): extension that is 6.6mm x 1.2mm
CL3_L1 = 18.8e-3
CL3_L2 = 25.4e-3
CL3_T1 = 1.3e-3
CL3_T2 = 2.5e-3

# Al cell part 2a (LHS): extension that is 6.6mm x 1.2mm
CL4_L1 = 0.e-3
CL4_L2 = 6.6e-3
CL4_T1 = 1.3e-3
CL4_T2 = 2.5e-3
"""
# Sipernat bed:  10.mm x 10.mm
Sip_L1 = 2.e-3
Sip_L2 = 12.e-3
Sip_T1 = 2.e-3
Sip_T2 = 12.e-3


L_length   = 14.e-3
T_length   = 12.e-3
GRIDSPC    = 50.e-6
T_cells    = int(T_length/GRIDSPC)				# T - Transverse, L - Longitudinal
L_cells    = int(L_length/GRIDSPC)				# T - Transverse, L - Longitudinal
print T_cells, L_cells
pss.generate_mesh(L_cells,T_cells,mat_no=3,GridSpc=GRIDSPC)
mats = pss.mats


pss.fill_rectangle(DR1_L1,DR1_T1,DR1_L2,DR1_T2,int(mats[0]))
pss.fill_rectangle(DR2_L1,DR2_T1,DR2_L2,DR2_T2,int(mats[0]))
pss.fill_rectangle(CL1_L1,CL1_T1,CL1_L2,CL1_T2,mats[1])
pss.fill_rectangle(CL2_L1,CL2_T1,CL2_L2,CL2_T2,mats[1])
#pss.fill_rectangle(CL3_L1,Al3_T1,Al3_L2,Al3_T2,mats[1])
#pss.fill_rectangle(CL4_L1,Al4_T1,Al4_L2,Al4_T2,mats[1])
pss.fill_rectangle(Sip_L1,Sip_T1,Sip_L2,Sip_T2,mats[2])

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.set_xlim(0,500)
#plt.imshow(pss.mesh,interpolation='nearest',cmap='binary')
view_mesh = np.zeros_like((pss.materials[0,:,:]))
pss.materials = pss.materials[:,::-1,]
for KK in range(pss.Ms):
    matter = np.copy(pss.materials[KK,:,:])*(KK+1)
    matter = np.ma.masked_where(matter==0.,matter)
    ax.imshow(matter, cmap='rainbow',vmin=0,vmax=pss.Ms,interpolation='nearest',origin='lower')
    
plt.show()

pss.save_general_mesh(fname='meso_m_bowtest1_{}x{}.iSALE'.format(T_cells,L_cells),mixed=False,noVel=True)
