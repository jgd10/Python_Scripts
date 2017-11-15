import pySALESetup as pss
import numpy as np
import random
import matplotlib.pyplot as plt

vol_frac   = .3
X_cells    = 500 
Y_cells    = 500 
x_length   = 1.e-3                                                  # Transverse width of bed
y_length   = 1.e-3                                                  # Longitudinal width of bed
PR         = 0.                                                     # Particle size range (as a fraction)
cppr       = 20                                                     # Cells per particle radius
GRIDSPC    = x_length/X_cells                                       # Physical distance/cell
mat_no     = 5


pss.generate_mesh(X=X_cells,Y=Y_cells,mat_no=mat_no,CPPR=cppr,pr=PR,VF=vol_frac,GridSpc=GRIDSPC) 

n = pss.N                                                           # n is the number of particles to generated for the 'library' which can be selected from this and placed into the mesh
part_area  = np.zeros((n))                                          # N.B. particles can (and likely will) be placed more than once
part_radii = np.linspace(pss.cppr_min,pss.cppr_max,n)               # part_area and part_radii are arrays to contain the area and radii of the 'library' particles
                                                                    # if PR is non-zero then a range of radii will be generated

for i in range(n):                                                                      
    pss.mesh_Shps[i,:,:] = pss.gen_circle(part_radii[i])            # Generate a circle and store it in mesh_Shps (an array of meshes) 
    part_area[i]         = np.sum(pss.mesh_Shps[i,:,:])             # Record the shape's area


A_        = []                                                      # Generate empty lists to store the area
xc        = []                                                      # x coord
yc        = []                                                      # y coord
I_        = []                                                      # Library index
J_        = []                                                      # shape number of each placed particle
vf_pld    = 0                                                       # Volume Fraction of material inserted into the mesh so far
J         = 0
old_vfrac = 0.
try:                                                                # Begin loop placing grains in.
    while vf_pld < vol_frac:                                        # Keep attempting to add particles until the required volume fraction is achieved      
        I    = random.randint(0,n-1)                                  # Generate a random index to randomly select a particle from the 'library'
        fail = 1
        while fail == 1:                                            # Whilst fail == 1 continue to try to generate a coordinate
            x,y,fail = pss.gen_coord(pss.mesh_Shps[I,:,:])          # Function generates a coordinate where the placed shape does not overlap with any placed so far
        area = pss.insert_shape_into_mesh(pss.mesh_Shps[I,:,:],x,y)
        J   += 1                                                  
        # pss.place_shape(pss.mesh_Shps[0,:,:],J,I,M)               # This function allows you to place a shape in AND specify its material. However it overwrites previous placings
        A_.append(area)                                             # Save the area
        xc.append(x)                                                # Save the coords of that particle
        yc.append(y)
        J_.append(J)                                                # Save the particle number
        I_.append(I)                                                # Save the library index
        
        vf_pld = np.sum(A_)/(pss.meshx*pss.meshy)                   # update the volume fraction
        if vf_pld == old_vfrac: 
            print '##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##########'
            print '# volume fraction no longer increasing. Break here #'
            print '##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##########'
            break
        old_vfrac = vf_pld
        print "volume fraction achieved so far: {:3.3f}%".format(vf_pld*100)
        
except KeyboardInterrupt:
    pass                
I_shape = np.array(I_)                                              # Convert the lists of index, shape number, and coordinates to arrays
J_shape = np.array(J_)

xcSI = np.array(xc)
ycSI = np.array(yc)
xcSI*= GRIDSPC                                                      # Turn the coordinates into physical units
ycSI*= GRIDSPC
MAT  = pss.mat_assignment(pss.mats[1:],xcSI,ycSI)                   # Assign materials to the particles. This returns an array that is the same shape as xc, 
                                                                    # and contains the optimum corresponding material number for each particle
pss.populate_materials(I_,xc,yc,MAT,J)                              # Now populate the materials meshes (NB these are different to the 'mesh' and are
                                                                    # the ones used in the actual iSALE input file)

pss.fill_rectangle(0,0,pss.meshx*GRIDSPC,pss.meshy*GRIDSPC,pss.mats[0])
                                                                    # pySALESetup prioritises material placed sooner. So later things will NOT overwrite previous ones
pss.save_general_mesh()                                             # Save the mesh as meso_m.iSALE (default)

plt.figure()                                                        # plot the resulting mesh. Skip this bit if you do not need to.
for KK in range(pss.Ms):
    matter = np.copy(pss.materials[KK,:,:])*(KK+1)
    matter = np.ma.masked_where(matter==0.,matter)
    plt.imshow(matter, cmap='viridis',vmin=0,vmax=pss.Ms,interpolation='nearest')

matter = np.copy(pss.materials[pss.Ms-1,:,:])*(pss.Ms)
matter = np.ma.masked_where(matter==0.,matter)


#plt.axis('equal')
plt.xlim(0,pss.meshx)
plt.ylim(0,pss.meshy)
plt.show()

