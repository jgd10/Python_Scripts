import random
import matplotlib.pyplot as plt
import pySALESetup as pss
import numpy as np

All = np.genfromtxt('meso_A-1.5203.iSALE')
m,x,y,r = All[:,0],All[:,1],All[:,2],All[:,3] 
m       = m.astype(int)
Nr      = np.size(np.unique(r)) # Number of different sizes
Nm      = np.size(np.unique(m)) # Number of different materials
Np      = np.size(x)            # Number of different particles
                                                                    # Define details of the particle bed 
cppr       = 1
diameter   = 32e-6
GRIDSPC    = 32e-6 * 0.5 / float(cppr)
x_length   = 1.e-3
y_length   = 1.e-3
X_cells    = int(x_length/GRIDSPC)                                  # Physical distance/cell
Y_cells    = int(y_length/GRIDSPC)                                  # Physical distance/cell
mat_no     = int(Nm)

pss.generate_mesh(X_cells,Y_cells,mat_no,cppr,GridSpc=GRIDSPC)          # Generate & set up the mesh
matter = pss.materials
pss.mesh_Shps[0] = pss.gen_circle(cppr)
#print X_cells,Y_cells,cppr,GRIDSPC
for i in range(Np):
    x0 = int(x[i]/GRIDSPC)
    y0 = int(y[i]/GRIDSPC)
    pss.place_shape(pss.mesh_Shps[0],x0,y0,m[i]-1)

pss.save_general_mesh(fname='meso_A-1.5203_cppr-{}_{}x{}.iSALE'.format(cppr,X_cells,Y_cells),noVel=True)

fig = plt.figure()
ax  = fig.add_subplot(111,aspect='equal')
for j in range(Nm):
    matter = np.copy(pss.materials[j,:,:]*(j+1))
    matter = np.ma.masked_where(matter==0.,matter)
    ax.imshow(matter, cmap='plasma',vmin=0,vmax=pss.Ms,interpolation='nearest')
ax.set_xlim(0,pss.meshx)
ax.set_ylim(0,pss.meshy)
plt.show()
