import numpy as np
import random
import matplotlib.pyplot as plt
import pySALESetup as pss
import sys

                                                                    # Define details of the particle bed 
vol_frac   = .5
X_cells    = 500 
Y_cells    = 500 
PR         = 0.
cppr       = 20
vfraclimit = 1.                                                     # The changeover point from 'placed' to 'dropped' contacts. particles
x_length   = 1.e-3
y_length   = 1.e-3
GRIDSPC    = x_length/X_cells                                       # Physical distance/cell
mat_no     = 5

pss.generate_mesh(X_cells,Y_cells,mat_no,cppr,PR,vol_frac)          # Generate & set up the mesh


n = pss.N                                                           # n is the number of particles to generated as a 'library' to be placed into the mesh
part_area  = np.zeros((n))                                          # N.B. particles can (and likely will) be placed more than once
part_radii = np.zeros((n))                                          # part_area and part_radii are arrays to contain the area and radii of the 'library' particles

for i in range(n):                                                  
    eccen = np.sqrt(8./9.)
    theta = random.random()*np.pi
    Rad   = cppr
    pss.mesh_Shps[i,:,:] = pss.gen_ellipse(Rad,theta,eccen)         # Generate an ellipse of radius: Rad, angled at theta, and with eccentricity: eccen.
    part_area[i]         = np.sum(pss.mesh_Shps[i,:,:])             # Record the shape's area
    part_radii[i]        = cppr                                     # Record the shape's radius (semi-major radius in this case)


vol_placed_frac = 0.
placed_part_area = []

nmin = 0                                                            # Max and min elements of the shape 'library'
nmax = n-1                                                          

xcoords   = []                                                      # Lists to record the coords, radii, 'library' number and particle number of each particle
ycoords   = []                                                      # Start empty and append to them as we do not know how many particles will be generated
radii     = []                                                        
I_shape   = []
J_shape   = []
old_vfrac = 0.
J         = 0                                                       # counter for the number of particles placed in the mesh

try:
    while vol_placed_frac<vol_frac:                                 # Keep attempting to add particles until the required volume fraction is achieved
        
        if J == 0 or (vol_placed_frac < vfraclimit*vol_frac):       # FIRST PARTICLE must ALWAYS be randomly placed
            I        = random.randint(nmin,nmax)                    # Generate a rrandom number to randomly select a particle from the 'library'
            fail     = 1
            while fail == 1:                                        # Whilst fail = True continue to try to generate a coordinate
                x,y,fail = pss.gen_coord(pss.mesh_Shps[I,:,:])
            area     = pss.insert_shape_into_mesh(pss.mesh_Shps[I,:,:],x,y)
            J       += 1                                                  
    
        else: 
            I        = random.randint(nmin,nmax)                    # Generate a random number to randomly select one of the generated shapes, to be tried for this loop
            x,y,area = pss.drop_shape_into_mesh(pss.mesh_Shps[I,:,:])
            J       += 1
        
        placed_part_area.append(area)                               # Save the area
        xcoords.append(x)                                           # Save the coords of that particle
        ycoords.append(y)
        radii.append(part_radii[I])                                 # Save the radii
        J_shape.append(J)                                           # Save the particle number
        I_shape.append(I)                                           # Save the number within the 'library'
        
        vol_placed_frac = np.sum(placed_part_area)/(pss.meshx*pss.meshy)    
                                                                    # update the volume fraction
        if vol_placed_frac == old_vfrac: 
            print '##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##########'
            print '# volume fraction no longer increasing. Break here #'
            print '##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##########'
            break
        old_vfrac = vol_placed_frac
        print "\rvolume fraction achieved so far: {:3.3f}%".format(vol_placed_frac*100),
        sys.stdout.flush()
        
except KeyboardInterrupt:
    pass

I_shape   = np.array(I_shape)                                       # Convert the lists to numpy arrays    
J_shape   = np.array(J_shape)
xcr       = np.array(xcoords)
ycr       = np.array(ycoords)
radii     = np.array(radii)

XINT      = np.copy(xcr)                                            # Coordinates and radii are in units of 'cells' up to here
YINT      = np.copy(ycr)                                            # This keeps a copy of these before they're converted to physical units

xcr       =   xcr.astype(float)                                     # Convert to floats
ycr       =   ycr.astype(float)
radii     = radii.astype(float)



xcr     *= GRIDSPC                                                  # Convert to physical units
ycr     *= GRIDSPC
MAT      = pss.mat_assignment(pss.mats,xcr,ycr)                     # Assign materials to the particles
radii   *= GRIDSPC

pss.save_spherical_parts(xcr,ycr,radii,MAT,A,fname='coords.txt')    # Save particle coordinates, radii and material number as a txt file with file name fname
                                                                    # When fname is not 'meso' A does not need to be anything and can just be zero as it is not used
pss.populate_materials(I_shape,XINT,YINT,MAT,J)                     # populate the materials meshes 
pss.save_general_mesh()                                             # Save full mesh as a meso_m.iSALE file. NB This uses the integer coords we had before


print "total particles placed: {}".format(J)

placed_part_area = np.array(placed_part_area)
vol_frac_calc = np.sum(placed_part_area)/(pss.meshx*pss.meshy)

if abs(vol_frac_calc - pss.vol_frac) <= 0.02:
    print "GREAT SUCCESS! Volume Fraction = {:3.3f}%".format(vol_frac_calc*100.)
else:
    print "FAILURE. Volume Fraction = {:3.3f}%".format(vol_frac_calc*100.)


plt.figure()                                                        # Generate a figure of the mesh you have just created, with a different colour for each material
for KK in range(pss.Ms):
    matter = np.copy(pss.materials[KK,:,:])*(KK+1)
    matter = np.ma.masked_where(matter==0.,matter)
    plt.imshow(matter, cmap='plasma',vmin=0,vmax=pss.Ms,interpolation='nearest')
plt.xlim(0,pss.meshx)
plt.ylim(0,pss.meshy)
plt.show()

