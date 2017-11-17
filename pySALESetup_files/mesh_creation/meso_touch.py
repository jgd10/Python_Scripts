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
pss.generate_mesh(X_cells,Y_cells,mat_no,cppr,PR,vol_frac)

# Volume fraction of particles placed
vol_placed_frac = 0.

# No. of particles to generate across the Part size range; default = 20
n = pss.N                                 
part_area  = np.zeros((n))
part_radii = []
cppr_range = pss.cppr_max - pss.cppr_min

# Generate n particles with a slowly increasing size
for i in range(n):                            
    r = pss.cppr_min + i*cppr_range/(n-1)    
    pss.mesh_Shps[i,:,:] = pss.gen_circle(r)
    part_area[i] = np.sum(pss.mesh_Shps[i,:,:])
    part_radii.append(r)

# Sort these particles into ascending area order and invert
#sort_indices  = np.argsort(part_area)          
#part_area     = part_area[sort_indices]        
#pss.mesh_Shps = pss.mesh_Shps[sort_indices,:,:]
#part_area     = part_area[::-1]                 
#pss.mesh_Shps = pss.mesh_Shps[::-1,:,:]
#part_radii    = part_radii[::-1]			

# set nmax and nmin for random choices later
nmin = 0
nmax = n-1                          

# Initialise all arrays to record placing details
xcoords = np.ones((X_cells*Y_cells))*-9999.
ycoords = np.ones((X_cells*Y_cells))*-9999.
radii   = np.ones((X_cells*Y_cells))*-9999.
I_shape = np.ones((X_cells*Y_cells))*-9999.
J_shape = np.ones((X_cells*Y_cells))*-9999.
placed_part_area = np.ones((X_cells*Y_cells))*-9999.

# List of all the record arrays
rec_arrays = [xcoords,ycoords,radii,I_shape,J_shape,placed_part_area]

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
            part_data = [x,y,part_radii[I],I,J+1,area]
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
            part_data = [x,y,part_radii[I],I,J+1,area]
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
radii   =   radii[radii  !=-9999.]
I_shape = I_shape[I_shape!=-9999.] 
J_shape = J_shape[J_shape!=-9999.]

# Integer arrays of coords
XINT      = np.copy(xcr).astype(int)
YINT      = np.copy(ycr).astype(int)

"""
# kdtree algorithm calculating Z
XY = np.column_stack((xcr,ycr))

import scipy.spatial as scsp
mytree = scsp.cKDTree(XY,leafsize=100)
for item in XY:
	print mytree.query(item, k=np.size(pss.mats)+1, distance_upper_bound=pss.cppr_max*3)
"""
# Convert coords to physical space
xcr     *= GRIDSPC
ycr     *= GRIDSPC
radii   *= GRIDSPC

# Assign materials to particles based on their coords
MAT      = pss.mat_assignment(pss.mats,xcr,ycr)

# Calculate coordination number
Z,B = pss.part_distance(xcr,ycr,radii,MAT,False)
print "The Coordination number, Z = {}".format(Z)
print "Avg Contacts Between the Same Materials, B = {}".format(B)
print 'Total contacts between same materials = {}, Total particles = {}'.format(B*J,J)

# Populate materials meshes with the material numbers that have just been assigned
pss.populate_materials(I_shape,XINT,YINT,MAT,J)

# Calculate the maximum variation in porosity across P cols/rows in the bed
P = 6
K = pss.max_porosity_variation(partitions=P)

# Calculate the discrete contact number (physically calculated, with cells and all)
Z2, contact_matrix = pss.discrete_contacts_number(I_shape,XINT,YINT,J,J_shape)

# Calculate the fabric tensor for this system
Z3,A,F = pss.Fabric_Tensor_disks(xcr,ycr,radii,tolerance=1.e-6)
print '\n'
print "Z; discrete Z; Fabric Tensor Z: {:3.2f}, {:3.2f}, {:3.2f}".format(Z, Z2, Z3)
print "Fabric Anisotropy A: {:3.4f}".format(A)
print "Max Porosity variation across {} partitions: {:3.2f}%".format(P,K*100.)
print '\n'

# Total no. particles placed and final volume fraction achieved
placed_part_area = placed_part_area[placed_part_area!=-9999.]
print "total particles placed: {}".format(J)
vol_frac_calc = np.sum(placed_part_area)/(pss.meshx*pss.meshy)

# If within 2% of target, deem this a success; else failure
if abs(vol_frac_calc - vol_frac) <= 0.02:
    print "GREAT SUCCESS! Volume Fraction = {:3.3f}%".format(vol_frac_calc*100.)
    # save the resulting bed and use the contacts number as an identifier
    pss.save_spherical_parts(xcr,ycr,radii,MAT,Z)
else:
    print "FAILURE. Volume Fraction = {:3.3f}%".format(vol_frac_calc*100.)

# view resulting bed
pss.view_all_materials()

