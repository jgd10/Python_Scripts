import pySALESetup as pss
import numpy as np
import scipy.spatial as scsp
import scipy as sc
import random
import matplotlib.pyplot as plt
import time
# ROD 1
#a  = 1.79e-3
#b  = 5.1e-3
#c  = 6.83e-3
# ROD 2 
#a  = 1.83e-3
#b  = 4.84e-3
#c  = 6.65e-3
# ROD 4 (there is no 3) 
a  = 1.967e-3
b  = 4.835e-3
c  = 6.305e-3

# Cu top hat part 1: Main disc 25.4mm x 1.3mm
Cu1_L1 = 0.
Cu1_L2 = 25.4e-3
Cu1_T1 = 0.
Cu1_T2 = 1.3e-3

# Cu top hat part 2: Central protrusion 9.8mm x 1.3mm
Cu2_L1 = 7.8e-3
Cu2_L2 = 17.6e-3
Cu2_T1 = 1.3e-3
Cu2_T2 = 2.6e-3

# Al cell part 1a (LHS): wall that is 1.2mm x 12.3mm in size
Al1_L1 = 6.6e-3
Al1_L2 = 7.8e-3
Al1_T1 = 1.3e-3
Al1_T2 = 13.6e-3

# Al cell part 1b (RHS): wall that is 1.2mm x 12.3mm in size
Al2_L1 = 17.6e-3
Al2_L2 = 18.8e-3
Al2_T1 = 1.3e-3
Al2_T2 = 13.6e-3

# Al cell part 2b (RHS): extension that is 6.6mm x 1.2mm
Al3_L1 = 18.8e-3
Al3_L2 = 25.4e-3
Al3_T1 = 1.3e-3
Al3_T2 = 2.5e-3

# Al cell part 2a (LHS): extension that is 6.6mm x 1.2mm
Al4_L1 = 0.e-3
Al4_L2 = 6.6e-3
Al4_T1 = 1.3e-3
Al4_T2 = 2.5e-3

# Sipernat bed:  9.8mm x 6.0mm
Sip_L1 = 7.8e-3
Sip_L2 = 17.6e-3
Sip_T1 = 2.6e-3
Sip_T2 = Sip_T1+c

# PMMA window:  9.8mm x 6.0mm
PMMA_L1 = 7.8e-3
PMMA_L2 = 17.6e-3
PMMA_T1 = Sip_T2
PMMA_T2 = Sip_T2+6.e-3

L_length   = 25.4e-3
T_length   = PMMA_T2+.1e-3
r_real     = .5e-3
GRIDSPC    = 11.85e-6
T_cells    = int(T_length/GRIDSPC)				# T - Transverse, L - Longitudinal
L_cells    = int(L_length/GRIDSPC)				# T - Transverse, L - Longitudinal
r          = int(r_real/GRIDSPC)
print T_cells, L_cells, r
pss.generate_mesh(L_cells,T_cells,CPPR=r,mat_no=5,GridSpc=GRIDSPC)
mats = pss.mats
pss.mesh_Shps[0,:,:] = pss.gen_circle(r)

# Borosilicate Rod (1mm diameter) placed at position a,b
t0 = Cu2_T2 + a
l0 = Sip_L1 + b
j = np.where(abs(pss.yh-l0)<pss.GS/2.)
i = np.where(abs(pss.xh-t0)<pss.GS/2.)
I, J, M = i[0],j[0],mats[4]
pss.place_shape(pss.mesh_Shps[0,:,:],J,I,M)

pss.fill_rectangle(Cu1_L1,Cu1_T1,Cu1_L2,Cu1_T2,int(mats[0]))
pss.fill_rectangle(Cu2_L1,Cu2_T1,Cu2_L2,Cu2_T2,int(mats[0]))
pss.fill_rectangle(Al1_L1,Al1_T1,Al1_L2,Al1_T2,mats[1])
pss.fill_rectangle(Al2_L1,Al2_T1,Al2_L2,Al2_T2,mats[1])
pss.fill_rectangle(Al3_L1,Al3_T1,Al3_L2,Al3_T2,mats[1])
pss.fill_rectangle(Al4_L1,Al4_T1,Al4_L2,Al4_T2,mats[1])
pss.fill_rectangle(Sip_L1,Sip_T1,Sip_L2,Sip_T2,mats[2])
pss.fill_rectangle(PMMA_L1,PMMA_T1,PMMA_L2,PMMA_T2,mats[3])

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

pss.save_general_mesh(fname='meso_m_ROD4_{}x{}.iSALE'.format(T_cells,L_cells),mixed=False,noVel=True)
