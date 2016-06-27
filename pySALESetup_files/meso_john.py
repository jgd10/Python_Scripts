import numpy as np
import random
import matplotlib.pyplot as plt
import pySALESetup as pss

                                                                    # Define details of the particle bed 
vol_frac   = .5  ### default 0.5
X_cells    = 500 ### defaults = 500
Y_cells    = 500 
PR         = 0. ### default = 0.1, have to increase to cover the high eccen. shapes
cppr       = 20     ### default = 20
vfraclimit = 1.                                                     # The changeover point from 'placed' to 'dropped' contacts. particles
x_length   = 1.e-3
y_length   = 1.e-3
GRIDSPC    = x_length/X_cells                                       # Physical distance/cell
mat_no     = 5

pss.generate_mesh(X_cells,Y_cells,mat_no,cppr,PR,vol_frac,NS=cppr*5)          # Generate & set up the mesh


n = pss.N                                                           # n is the number of particles to generated as a 'library' to be placed into the mesh
ns = pss.Ns ### added ns





######################################## generate mesh_Shps for each eccentricity, and add it into the mesh_Shps_list #####################################

axis_ratios     = np.array([2.5,2.,1.5,1.])                 ### list semi-major/semi-minor axis ratios (must be float)
# numpy arrays are defined as np.array([...]) and not np.array((...))

eccen_list  = np.sqrt(1 - (1/np.square(axis_ratios)))        ### calculate list of eccentricities from axis ratios
n_eccen     = np.size(eccen_list)                                   ### number of eccentricities
# length works, but it is better to use size IMO as size is for numpy arrays and len for lists

#mesh_Shps_list = []                                                ### list of the generated mesh_Shps for each eccentricity
mesh_Shps_array = np.zeros((n_eccen,n,ns,ns))
#part_area_list = []
#part_radii_list = []
#part_angle_list = []

#angles = [random.random() * np.pi for _ in range(n)]            ### generate list of random angles, to be kept constant
# A nice feature of numpy arrays means that you can do the above as:
angles = np.random.rand(n)*np.pi

part_area      = np.zeros((n))                                          
part_radii     = np.zeros((n))

for j in range(n_eccen):                                                ### generate a mesh_Shps array for each eccentricity and add it to mesh_Shps_list
    ### reset mesh shapes array, and the particle area/radius arrays:
    #pss.mesh_Shps = np.zeros((n,ns,ns))    
    # Here, you are re-defining mesh_Shps, it is already defined in pySALESetup and unnecessary. If you just wish to reset them, do this:
    pss.mesh_Shps *= 0.
    part_area     *= 0.
    part_radii    *= 0.
    #part_angle = np.zeros((n))                                         ### add in list of angles

    for i in range(n):                                                  
        theta = angles[i]                                                ### use the already generated list of angles
        Rad   = np.sqrt(axis_ratios[j]) * cppr                             ### radius calculated so that area stays constant, where cppr is radius of perfect circle 
        pss.mesh_Shps[i,:,:] = pss.gen_ellipse(Rad,theta,eccen_list[j]) ### Generate an ellipse of radius: Rad, angled at theta, and with eccentricity taken from eccen_list      
        part_area[i]         = np.sum(pss.mesh_Shps[i,:,:])             # Record the shape's area 
        part_radii[i]        = cppr                                     # Record the shape's radius (semi-major radius in this case) ###need to change this?     
    #part_angle[i]        = theta

    #mesh_Shps_list.append(pss.mesh_Shps)                                ### append new mesh_Shps to mesh_Shps_list
    # try to avoid appending things as it's a very costly programming technique (computationally)
    mesh_Shps_array[j,:,:,:] = pss.mesh_Shps
    #part_area_list.append(part_area)
    #part_radii_list.append(part_radii)
    #part_angle_list.append(part_angle)

############################################################################################################################################################

pss.mesh_Shps = mesh_Shps_array[0,:,:,:]    ### pick which eccentricity to use when finding positions






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
        print "volume fraction achieved so far: {:3.3f}%".format(vol_placed_frac*100)
        
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

#pss.save_spherical_parts(xcr,ycr,radii,MAT,A,fname='coords.txt')   # Save particle coordinates, radii and material number as a txt file with file name fname
                                                                    # When fname is not 'meso' A doe snot need to be anything and can just be zero as it is not used
pss.save_particle_mesh(I_shape,XINT,YINT,MAT,J)                     # Save full mesh as a meso_m.iSALE file. NB This uses the integer coords we had before


print "total particles placed: {}".format(J)

placed_part_area = np.array(placed_part_area)
vol_frac_calc = np.sum(placed_part_area)/(pss.meshx*pss.meshy)

if abs(vol_frac_calc - pss.vol_frac) <= 0.02:
    print "GREAT SUCCESS! Volume Fraction = {:3.3f}%".format(vol_frac_calc*100.)
else:
    print "FAILURE. Volume Fraction = {:3.3f}%".format(vol_frac_calc*100.)




######################################## loop through the list of eccentricities and plot each one ################################################

k = 0
while k < n_eccen:

    pss.generate_mesh(X_cells,Y_cells,mat_no,cppr,PR,vol_frac,NS=cppr*5)           ### clear mesh

    pss.mesh_Shps = mesh_Shps_array[k,:,:,:]                            ### change eccentricity

                                        
    pss.save_particle_mesh(I_shape,XINT,YINT,MAT,J,fname='meso_m_aspect-{:2.2f}.iSALE'.format(axis_ratios[k]))                     # Save full mesh as a meso_m.iSALE file. NB This uses the integer coords we had before
    ###still need to save individually!    
    
    plt.figure()                                                        # Generate a figure of the mesh you have just created, with a different colour for each material
    for KK in range(pss.Ms):
        matter = np.copy(pss.materials[KK,:,:])*(KK+1)
        matter = np.ma.masked_where(matter==0.,matter)
        plt.imshow(matter, cmap='BuPu',vmin=0,vmax=pss.Ms,interpolation='nearest')
    plt.xlim(0,pss.meshx)
    plt.ylim(0,pss.meshy)
    #plt.savefig('test.png')    
    plt.show()
    
    
    k += 1

####################################################################################################################################################

