import numpy as np
import random
import matplotlib.pyplot as plt
import pySALESetup as pss
import sys

# All parameters required to make a particle bed
vol_frac   = .5
X_cells    = 500 
Y_cells    = 500 
# Particle size range, as a fraction. e.g. 0.1 gives a variation in size of 10%
PR         = 0.
cppr       = 8
# Threshold at which algorithm begins walking particles into contacts
vfraclimit = .5                     
x_length   = 1.e-3
y_length   = 1.e-3
GRIDSPC    = x_length/X_cells
mat_no     = 5

# generate the mesh using the above inputs
pss.generate_mesh(X_cells,Y_cells,mat_no,cppr,PR,vol_frac,NS=50,NP=5)

# Volume fraction of particles placed
vol_placed_frac = 0.

# No. of particles to generate across the Part size range; default = 20
n = pss.N                                 

# Instead of generating circles use pre-made meshes and proceed the same way.
pss.mesh_Shps[0,:,:] = pss.gen_shape_fromtxt('../../grain_library/grain_aspect-1.026_mesh.txt')
pss.mesh_Shps[1,:,:] = pss.gen_shape_fromtxt('../../grain_library/grain_aspect-1.085_mesh.txt')
pss.mesh_Shps[2,:,:] = pss.gen_shape_fromtxt('../../grain_library/grain_aspect-2.005_mesh.txt')
pss.mesh_Shps[3,:,:] = pss.gen_shape_fromtxt('../../grain_library/grain_aspect-1.516_mesh.txt')
pss.mesh_Shps[4,:,:] = pss.gen_shape_fromtxt('../../grain_library/grain_aspect-1.249_mesh.txt')

# set nmax and nmin for random choices later
nmin = 0
nmax = n-1                          

# Initialise all arrays to record placing details
xcoords = np.ones((X_cells*Y_cells))*-9999.
ycoords = np.ones((X_cells*Y_cells))*-9999.
I_shape = np.ones((X_cells*Y_cells))*-9999.
J_shape = np.ones((X_cells*Y_cells))*-9999.
placed_part_area = np.ones((X_cells*Y_cells))*-9999.

# List of all the record arrays
rec_arrays = [xcoords,ycoords,I_shape,J_shape,placed_part_area]

# Set old_vfrac for the first check.
old_vfrac = 0.

# Number of particles placed
J = 0

# Try to fill mesh. If it gets stuck, option to ctrl-c and keep progress still available.
try:
    # Whilst the fraction of placed particles < target: loop.
    while vol_placed_frac < vol_frac:    

        # If first particle ALWAYS insert 
        # Or, if placed frac less than threshold, insert.
        if vol_placed_frac < vfraclimit*vol_frac or J == 0:

            # Choose a shape at random from mesh_Shps
            I = random.randint(nmin,nmax)        
            fail = True

            # Find a working coordinate
            while fail:
                x,y,fail = pss.gen_coord(pss.mesh_Shps[I,:,:])

            # Place the shape, record the new area
            area = pss.insert_shape_into_mesh(pss.mesh_Shps[I,:,:],x,y)

            # enter all parameters into record arrays
            part_data = [x,y,I,J+1,area]
            for rec,elm in zip(rec_arrays,part_data):
                rec[J] = elm

            # Increment placed particle counter by 1
            J += 1                              
        else: 
            # Random shape as before
            I = random.randint(nmin,nmax)        

            # This time, 'drop' shape in, to perform random walk until contact made.
            x,y,area = pss.drop_shape_into_mesh(pss.mesh_Shps[I,:,:])
            
            # Record parameters in the arrays
            part_data = [x,y,I,J+1,area]
            for rec,elm in zip(rec_arrays,part_data):
                rec[J] = elm

            # Increment placed particle counter
            J += 1

        # Calc the volume fraction placed
        vol_placed_frac = np.sum(placed_part_area[placed_part_area!=-9999.])/float(pss.meshx*pss.meshy)    
        
        # If no increase in volume fraction after last particle, break.
        if vol_placed_frac == old_vfrac: 
            print '##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##########'
            print '# volume fraction no longer increasing. Break here #'
            print '##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##########'
            break
        
        # Update the 'old' volume fraction every fifth particle
        if J%5 == 0:
            old_vfrac = vol_placed_frac
        print "\rvolume fraction achieved so far: {:5.3f}%".format(vol_placed_frac*100),
        sys.stdout.flush()
    print

except KeyboardInterrupt:
    pass

# Remove all dud elements from arrays
xcr     = xcoords[xcoords!=-9999.]
ycr     = ycoords[ycoords!=-9999.]
I_shape = I_shape[I_shape!=-9999.] 
J_shape = J_shape[J_shape!=-9999.]

# Integer arrays of coords
XINT      = np.copy(xcr).astype(int)
YINT      = np.copy(ycr).astype(int)

# Convert coords to physical space
xcr     *= GRIDSPC
ycr     *= GRIDSPC

# Assign materials to particles based on their coords
MAT      = pss.mat_assignment(pss.mats,xcr,ycr)

# Populate materials meshes with the material numbers that have just been assigned
pss.populate_materials(I_shape,XINT,YINT,MAT,J)

# Calculate the maximum variation in porosity across P cols/rows in the bed
P = 6
K = pss.max_porosity_variation(partitions=P)

# Calculate the discrete contact number (physically calculated, with cells and all)
Z2, contact_matrix = pss.discrete_contacts_number(I_shape,XINT,YINT,J,J_shape)

print "Max Porosity variation across {} partitions: {:3.2f}%".format(P,K*100.)

# Total no. particles placed and final volume fraction achieved
placed_part_area = placed_part_area[placed_part_area!=-9999.]
print "total particles placed: {}".format(J)
vol_frac_calc = np.sum(placed_part_area)/(pss.meshx*pss.meshy)

# If within 2% of target, deem this a success; else failure
if abs(vol_frac_calc - vol_frac) <= 0.02:
    print "GREAT SUCCESS! Volume Fraction = {:3.3f}%".format(vol_frac_calc*100.)
    # save the resulting bed and use the contacts number as an identifier
    #pss.save_general_mesh(mixed=True)
else:
    print "FAILURE. Volume Fraction = {:3.3f}%".format(vol_frac_calc*100.)

# view resulting bed
pss.view_all_materials()

