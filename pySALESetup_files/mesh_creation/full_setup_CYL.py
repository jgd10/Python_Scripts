import pySALESetup as pss
import numpy as np
import scipy.spatial as scsp
import scipy as sc
import random
import matplotlib.pyplot as plt
import time


additional_length = .301e-3

L_length   = 12.7e-3
T_length   = 15.e-3+additional_length
GRIDSPC    = 11.85e-6# * 3.
T_cells    = int(T_length/GRIDSPC) 				# T - Transverse, L - Longitudinal
L_cells    = int(L_length/GRIDSPC)

print 'mesh shape: {} x {}'.format(L_cells,T_cells)
Radius = 500.e-6
r      = int(Radius/GRIDSPC)
pss.generate_mesh(L_cells,T_cells,CPPR=r,mat_no=4,GridSpc=GRIDSPC)
pss.mesh_Shps[0,:,:] = pss.gen_circle(r)
mats = pss.mats


j = np.where(abs(pss.yh-0.e-3)<pss.GS/2.)
i = np.where(abs(pss.xh-3.6e-3)<pss.GS/2.)
#I, J, M = i[0],j[0],mats[4]
#pss.place_shape(pss.mesh_Shps[0,:,:],J,I,M)

# Cu top hat part 1: Main disc 25.4mm x 1.3mm
Cu1_L1 = 0.
Cu1_L2 = 12.7e-3
Cu1_T1 = 0.
Cu1_T2 = 1.3e-3
pss.fill_rectangle(Cu1_L1,Cu1_T1,Cu1_L2,Cu1_T2,mats[0])

# Cu top hat part 2: Central protrusion 9.8mm x 1.3mm
Cu2_L1 = 0.e-3
Cu2_L2 = 4.9e-3
Cu2_T1 = 1.3e-3
Cu2_T2 = 2.6e-3
pss.fill_rectangle(Cu2_L1,Cu2_T1,Cu2_L2,Cu2_T2,mats[0])

# Al cell part 1: wall that is 1.2mm x 12.3mm in size
Al2_L1 = 4.9e-3
Al2_L2 = 6.1e-3
Al2_T1 = 1.3e-3
Al2_T2 = 12.5e-3
pss.fill_rectangle(Al2_L1,Al2_T1,Al2_L2,Al2_T2,mats[1])

# Al cell part 2: extension that is 6.6mm x 1.2mm
Al2_L1 = 6.1e-3
Al2_L2 = 12.7e-3
Al2_T1 = 1.3e-3
Al2_T2 = 2.5e-3
pss.fill_rectangle(Al2_L1,Al2_T1,Al2_L2,Al2_T2,mats[1])

# Sipernat bed:  9.8mm x 6.0mm(ish)
dx = 4.9e-3/4.
dx4 = 0.1e-3
dx3 = 0.2e-3
dx2 = 0.3e-3
dx1 = 4.3e-3
Sip1_L1 = 0.e-3
Sip1_L2 = 4.9e-3
Sip1_T1 = 2.6e-3
Sip1_T2 = 8.6e-3+additional_length
#Sip2_L1 = dx1
#Sip2_L2 = dx1+dx2
#Sip2_T1 = 2.6e-3
#Sip2_T2 = 8.6e-3+additional_length
#Sip3_L1 = dx1+dx2
#Sip3_L2 = dx1+dx2+dx3
#Sip3_T1 = 2.6e-3
#Sip3_T2 = 8.6e-3+additional_length
#Sip4_L1 = dx1+dx2+dx3
#Sip4_L2 = dx1+dx2+dx3+dx4
#Sip4_T1 = 2.6e-3
#Sip4_T2 = 8.6e-3+additional_length
pss.fill_rectangle(Sip1_L1,Sip1_T1,Sip1_L2,Sip1_T2,mats[2])
#pss.fill_rectangle(Sip2_L1,Sip2_T1,Sip2_L2,Sip2_T2,mats[3])
#pss.fill_rectangle(Sip3_L1,Sip3_T1,Sip3_L2,Sip4_T2,mats[4])
#pss.fill_rectangle(Sip4_L1,Sip4_T1,Sip4_L2,Sip3_T2,mats[5])

# PMMA window:  9.8mm x 6.0mm
PMMA_L1 = 0.e-3
PMMA_L2 = 4.9e-3
PMMA_T1 = 8.6e-3+additional_length
PMMA_T2 = 14.6e-3+additional_length
pss.fill_rectangle(PMMA_L1,PMMA_T1,PMMA_L2,PMMA_T2,mats[3])

print pss.meshx,pss.meshy
fig = plt.figure()
ax = fig.add_subplot(111,aspect='equal')
#plt.imshow(pss.mesh,interpolation='nearest',cmap='binary')
view_mesh = np.zeros_like((pss.materials[0,:,:]))

#pss.materials = pss.materials[:,:,::-1]
pss.materials = pss.materials[:,::-1,]
cmap = plt.cm.rainbow
for item in mats:
    M = np.copy(pss.materials[item-1])
    M = np.ma.masked_where(M==0.,M)*item
    p = ax.imshow(M,interpolation='nearest',cmap = cmap,vmin=1,vmax=np.amax(mats))#,alpha=0.5)#,vmin=0,vmax=np.amax(mats))
    p.cmap.set_under('grey')
    if item == 1.:  fig.colorbar(p)
#plt.imshow(pss.materials[0,:,:],interpolation='nearest',cmap='copper_r')
#plt.imshow(pss.materials[1,:,:],interpolation='nearest',cmap='BuPu')
#plt.imshow(pss.materials[2,:,:],interpolation='nearest',cmap='viridis')
#plt.imshow(pss.materials[3,:,:],interpolation='nearest',cmap='binary')
#plt.imshow(pss.materials[4,:,:],interpolation='nearest',cmap='Reds_r')
plt.show()

pss.save_general_mesh(noVel=True)
