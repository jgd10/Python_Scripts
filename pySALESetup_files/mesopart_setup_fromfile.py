import random
import matplotlib.pyplot as plt
import pySALESetup as pss
import numpy as np

fpath = 'meso_A-1.5203.iSALE'
All = np.genfromtxt(fpath)
m,x,y,r = All[:,0],All[:,1],All[:,2],All[:,3] 
m       = m.astype(int)
Nr      = np.size(np.unique(r)) # Number of different sizes
Nm      = np.size(np.unique(m)) # Number of different materials
Np      = np.size(x)            # Number of different particles
                                                                    # Define details of the particle bed 
cppr       = 8
diameter   = 32e-6
GRIDSPC    = 32e-6 * 0.5 / float(cppr)
x_length   = 1.e-3
y_length   = 1.e-3
X_cells    = int(x_length/GRIDSPC)                                  # Physical distance/cell
Y_cells    = int(y_length/GRIDSPC)                                  # Physical distance/cell
mat_no     = int(Nm)

frac = 90.
theta = np.pi* frac/180.
c = np.cos(theta)
s = np.sin(theta)

xr = c*x - s*y 
yr = s*x + c*y 

if np.size(xr[xr<0])>0.:
    xr += np.amax(abs(xr))
if np.size(yr[yr<0])>0.:
    yr += np.amax(abs(yr))

print xr-x
fpath = fpath.replace('.iSALE','_rot-{:2.1f}.iSALE'.format(frac))

#pss.view_mesoiSALE(fpath,save=False)
A = np.column_stack((m,xr,yr,r))
np.savetxt(fpath,A)
pss.view_mesoiSALE(fpath,save=False)

pss.generate_mesh(X_cells,Y_cells,mat_no,cppr,GridSpc=GRIDSPC)          # Generate & set up the mesh
matter = pss.materials
pss.mesh_Shps[0] = pss.gen_circle(cppr)
#print X_cells,Y_cells,cppr,GRIDSPC
for i in range(Np):
    x0 = int(x[i]/GRIDSPC)
    y0 = int(y[i]/GRIDSPC)
    pss.place_shape(pss.mesh_Shps[0],x0,y0,m[i]-1)

pss.save_general_mesh(fname='meso_m_A-1.5203_rot90.iSALE',noVel=True)

fig = plt.figure()
ax  = fig.add_subplot(111,aspect='equal')
for j in range(Nm):
    matter = np.copy(pss.materials[j,:,:]*(j+1))
    matter = np.ma.masked_where(matter==0.,matter)
    ax.imshow(matter, cmap='plasma',vmin=0,vmax=pss.Ms,interpolation='nearest')
ax.set_xlim(0,pss.meshx)
ax.set_ylim(0,pss.meshy)
plt.show()
