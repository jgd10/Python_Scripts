import pySALESetup as pss
import numpy as np
import scipy.spatial as scsp
import scipy as sc
import random
import matplotlib.pyplot as plt
import time

def rotate_by_xdeg(V,x):
    t = x*np.pi/180.
    M = [[np.cos(t), -np.sin(t)],
         [np.sin(t),  np.cos(t)]]
    N,junk = np.shape(V)
    for i in range(N):
        V[i,:] = np.dot(M,V[i,:])
    return V
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

Cu1_a = [0.,         0.]
Cu1_b = [25.4e-3,    0.]
Cu1_c = [25.4e-3,1.3e-3]
Cu1_d = [0.,     1.3e-3]

Cu1_ = np.row_stack((Cu1_a,Cu1_b,Cu1_c,Cu1_d))
#Cu1_ = rotate_by_xdeg(Cu1_,3)

#Cu1_L1 = 0.
#Cu1_L2 = 25.4e-3
#Cu1_T1 = 0.
#Cu1_T2 = 1.3e-3

"""
#######################################
# UNNEEDED WHEN ROTATING WHOLE SYSTEM #
#######################################
Cu3_L = np.zeros((3))
Cu3_T = np.zeros((3))
# Cu top hat 3 deg incline
Cu3_a = [0., 0.]
Cu3_b = [25.4e-3,0.]
Cu3_c = [25.4e-3, 25.4e-3 * np.tan(3.*np.pi/180.)]
Cu3_ = np.row_stack((Cu3_a,Cu3_b,Cu3_c,Cu3_d))
Cu3_ = rotate_by_xdeg(Cu3_,3)
#Cu3_L[0] = 0.
#Cu3_T[0] = 0.
#Cu3_L[1] = 25.4e-3
#Cu3_T[1] = 0.e-3
#Cu3_L[2] = 25.4e-3
#Cu3_T[2] = 25.4e-3 * np.tan(3.*np.pi/180.)
#print Cu3_T,Cu3_L

"""
# Cu top hat part 2: Central protrusion 9.8mm x 1.3mm
Cu2_a = [7.8e-3, 1.3e-3]
Cu2_b = [7.8e-3, 2.6e-3]
Cu2_c = [17.6e-3,2.6e-3]
Cu2_d = [17.6e-3,1.3e-3]
Cu2_ = np.row_stack((Cu2_a,Cu2_b,Cu2_c,Cu2_d))
#Cu2_ = rotate_by_xdeg(Cu2_,3)
#Cu2_L1 = 7.8e-3
#Cu2_L2 = 17.6e-3
#Cu2_T1 = 1.3e-3
#Cu2_T2 = 2.6e-3

# Al cell part 1a (LHS): wall that is 1.2mm x 12.3mm in size
Al1_a  = [6.6e-3,1.3e-3]
Al1_b  = [6.6e-3,13.6e-3]
Al1_c  = [7.8e-3,13.6e-3]
Al1_d  = [7.8e-3,1.3e-3]
Al1_ = np.row_stack((Al1_a,Al1_b,Al1_c,Al1_d))
#Al1_ = rotate_by_xdeg(Al1_,3)
#Al1_L1 = 6.6e-3
#Al1_L2 = 7.8e-3
#Al1_T1 = 1.3e-3
#Al1_T2 = 13.6e-3

# Al cell part 1b (RHS): wall that is 1.2mm x 12.3mm in size
Al2_a = [17.6e-3,1.3e-3]
Al2_b = [17.6e-3,13.6e-3]
Al2_c = [18.8e-3,13.6e-3]
Al2_d = [18.8e-3,1.3e-3]
Al2_ = np.row_stack((Al2_a,Al2_b,Al2_c,Al2_d))
#Al2_ = rotate_by_xdeg(Al2_,3)
#Al2_L1 = 17.6e-3
#Al2_L2 = 18.8e-3
#Al2_T1 = 1.3e-3
#Al2_T2 = 13.6e-3

# Al cell part 2b (RHS): extension that is 6.6mm x 1.2mm
Al3_a = [18.8e-3,1.3e-3]
Al3_b = [18.8e-3,2.5e-3]
Al3_c = [25.4e-3,2.5e-3]
Al3_d = [25.4e-3,1.3e-3]
Al3_ = np.row_stack((Al3_a,Al3_b,Al3_c,Al3_d))
#Al3_ = rotate_by_xdeg(Al3_,3)
#Al3_L1 = 18.8e-3
#Al3_L2 = 25.4e-3
#Al3_T1 = 1.3e-3
#Al3_T2 = 2.5e-3

# Al cell part 2a (LHS): extension that is 6.6mm x 1.2mm
Al4_a = [0.,1.3e-3]
Al4_b = [0.,2.5e-3]
Al4_c = [6.6e-3,2.5e-3]
Al4_d = [6.6e-3,1.3e-3]
Al4_ = np.row_stack((Al4_a,Al4_b,Al4_c,Al4_d))
#Al4_ = rotate_by_xdeg(Al4_,3)
#Al4_L1 = 0.e-3
#Al4_L2 = 6.6e-3
#Al4_T1 = 1.3e-3
#Al4_T2 = 2.5e-3

# Sipernat bed:  1.4mm x 6.0mm
dx = 9.8e-3/4.
Sip_a = [7.8e-3,2.6e-3]
Sip_b = [7.8e-3,2.6e-3+c]
Sip_c = [7.8e-3+dx,2.6e-3+c]
Sip_d = [7.8e-3+dx,2.6e-3]
Sip_1 = np.row_stack((Sip_a,Sip_b,Sip_c,Sip_d))
Sip_a = [7.8e-3+dx,2.6e-3]
Sip_b = [7.8e-3+dx,2.6e-3+c]
Sip_c = [7.8e-3+2*dx,2.6e-3+c]
Sip_d = [7.8e-3+2*dx,2.6e-3]
Sip_2 = np.row_stack((Sip_a,Sip_b,Sip_c,Sip_d))
Sip_a = [7.8e-3+2*dx,2.6e-3]
Sip_b = [7.8e-3+2*dx,2.6e-3+c]
Sip_c = [7.8e-3+3*dx,2.6e-3+c]
Sip_d = [7.8e-3+3*dx,2.6e-3]
Sip_3 = np.row_stack((Sip_a,Sip_b,Sip_c,Sip_d))
Sip_a = [7.8e-3+3*dx,2.6e-3]
Sip_b = [7.8e-3+3*dx,2.6e-3+c]
Sip_c = [7.8e-3+4*dx,2.6e-3+c]
Sip_d = [7.8e-3+4*dx,2.6e-3]
Sip_4 = np.row_stack((Sip_a,Sip_b,Sip_c,Sip_d))
#Sip_ = rotate_by_xdeg(Sip_,3)
#Sip_L1 = 7.8e-3
#Sip_L2 = 17.6e-3
#Sip_T1 = 2.6e-3
#Sip_T2 = Sip_T1+c

# PMMA window:  9.8mm x 6.0mm
PMMA_a = [7.8e-3,2.6e-3+c]
PMMA_b = [7.8e-3,2.6e-3+c+6.e-3]
PMMA_c = [17.6e-3,2.6e-3+c+6.e-3]
PMMA_d = [17.6e-3,2.6e-3+c]
PMMA_ = np.row_stack((PMMA_a,PMMA_b,PMMA_c,PMMA_d))
#PMMA_ = rotate_by_xdeg(PMMA_,3)
#PMMA_L1 = 7.8e-3
#PMMA_L2 = 17.6e-3
#PMMA_T1 = Sip_T2
#PMMA_T2 = Sip_T2+6.e-3

L_length   = 25.4e-3
T_length   = PMMA_c[1]+.1e-3
r_real     = .5e-3
GRIDSPC    = 11.85e-6 * 3.
T_cells    = int(T_length/GRIDSPC)				# T - Transverse, L - Longitudinal
L_cells    = int(L_length/GRIDSPC)				# T - Transverse, L - Longitudinal
r          = int(r_real/GRIDSPC)
pss.generate_mesh(L_cells,T_cells,mat_no=4+3,GridSpc=GRIDSPC)
mats = pss.mats
#pss.mesh_Shps[0,:,:] = pss.gen_circle(r)

# Borosilicate Rod (1mm diameter) placed at position a,b
#t0 = Cu2_T2 + a
#l0 = Sip_L1 + b
#j = np.where(abs(pss.yh-l0)<pss.GS/2.)
#i = np.where(abs(pss.xh-t0)<pss.GS/2.)
#I, J, M = i[0],j[0],mats[4]
#pss.place_shape(pss.mesh_Shps[0,:,:],J,I,M)

#pss.fill_rectangle(Cu1_L1,Cu1_T1,Cu1_L2,Cu1_T2,int(mats[0]))
#pss.fill_rectangle(Cu2_L1,Cu2_T1,Cu2_L2,Cu2_T2,int(mats[0]))
#pss.fill_rectangle(Al1_L1,Al1_T1,Al1_L2,Al1_T2,mats[1])
#pss.fill_rectangle(Al2_L1,Al2_T1,Al2_L2,Al2_T2,mats[1])
#pss.fill_rectangle(Al3_L1,Al3_T1,Al3_L2,Al3_T2,mats[1])
#pss.fill_rectangle(Al4_L1,Al4_T1,Al4_L2,Al4_T2,mats[1])
#pss.fill_rectangle(Sip_L1,Sip_T1,Sip_L2,Sip_T2,mats[2])
#pss.fill_rectangle(PMMA_L1,PMMA_T1,PMMA_L2,PMMA_T2,mats[3])
pss.fill_polygon(Cu1_[:,0],Cu1_[:,1],int(mats[0]))
pss.fill_polygon(Cu2_[:,0],Cu2_[:,1],int(mats[0]))
pss.fill_polygon(Al1_[:,0],Al1_[:,1],int(mats[1]))
pss.fill_polygon(Al2_[:,0],Al2_[:,1],int(mats[1]))
pss.fill_polygon(Al3_[:,0],Al3_[:,1],int(mats[1]))
pss.fill_polygon(Al4_[:,0],Al4_[:,1],int(mats[1]))

pss.fill_polygon(Sip_1[:,0],Sip_1[:,1],int(mats[2]))
pss.fill_polygon(Sip_2[:,0],Sip_2[:,1],int(mats[3]))
pss.fill_polygon(Sip_3[:,0],Sip_3[:,1],int(mats[4]))
pss.fill_polygon(Sip_4[:,0],Sip_4[:,1],int(mats[5]))

pss.fill_polygon(PMMA_[:,0],PMMA_[:,1],int(mats[6]))

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
    
pss.save_general_mesh(fname='meso_m_Sip4_{}x{}_discretesip.iSALE'.format(T_cells,L_cells),mixed=False,noVel=True)
plt.show()

