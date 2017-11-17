import numpy as np
import scipy.spatial as scsp
import scipy as sc
import random
import matplotlib.pyplot as plt
import pySALESetup as pss
import time


X_cells    = 1000 
Y_cells    = 1000 
vfraclimit = .5                               # The changeover point from random to forced contacts. > 1.0 => least contacts; = 0. Max contacts
x_length   = 1.e-3
y_length   = 1.e-3
GRIDSPC    = x_length/X_cells
mat_no     = 5
CPPR = 24

pss.generate_mesh(X_cells,Y_cells,mat_no,CPPR=24)
mats = pss.mats

pss.N = 10                                 # Particles can be placed MORE than once!
MM = pss.Ms
ii = 0
n = pss.N
part_radii = []
pss.mesh_Shps[0,:,:] = pss.gen_shape_fromtxt('../grain_library/grain_aspect-1.026_mesh.txt')
pss.mesh_Shps[1,:,:] = pss.gen_shape_fromtxt('../grain_library/grain_aspect-1.085_mesh.txt')
pss.mesh_Shps[2,:,:] = pss.gen_shape_fromtxt('../grain_library/grain_aspect-2.005_mesh.txt')
pss.mesh_Shps[3,:,:] = pss.gen_shape_fromtxt('../grain_library/grain_aspect-1.516_mesh.txt')
pss.mesh_Shps[4,:,:] = pss.gen_shape_fromtxt('../grain_library/grain_aspect-1.249_mesh.txt')

pss.mesh_Shps[5,:,:] = pss.gen_shape_fromtxt('../grain_library/grain_aspect-1.158_mesh.txt')
pss.mesh_Shps[6,:,:] = pss.gen_shape_fromtxt('../grain_library/grain_aspect-1.834_mesh.txt')
pss.mesh_Shps[7,:,:] = pss.gen_shape_fromtxt('../grain_library/grain_aspect-1.824_mesh.txt')
pss.mesh_Shps[8,:,:] = pss.gen_shape_fromtxt('../grain_library/grain_aspect-1.402_mesh.txt')
pss.mesh_Shps[9,:,:] = pss.gen_shape_fromtxt('../grain_library/grain_aspect-1.723_mesh.txt')


nmin = 0
nmax = n-1                                # max integer generated

xcoords = []
ycoords = []
I_shape = []
J_shape = []
old_vfrac = 0.
vol_placed_frac = 0.
placed_part_area = []
J = 0
try:
    while vol_placed_frac<pss.vol_frac:                        # Keep attempting to add particles until the required volume fraction is achieved
        if J == 0:                                    # FIRST PARTICLE must ALWAYS be randomly placed
            I = random.randint(nmin,nmax)                    # Generate a random number to randomly select one of the generated shapes, to be tried for this loop
            fail = 1
            while fail == 1:
                x,y,fail = pss.gen_coord(pss.mesh_Shps[I,:,:])
            area = pss.insert_shape_into_mesh(pss.mesh_Shps[I,:,:],x,y)
            ii+= 1
            J += 1                                # J is now the total number of particles inserted
            placed_part_area.append(area)                    # Update the list of areas
            xcoords.append(x)
            ycoords.append(y)
            I_shape.append(I)
            J_shape.append(J)
    
        elif vol_placed_frac < vfraclimit*pss.vol_frac:
            #elif J%2 == 0:
            if ii >= MM: ii = 0
            I = random.randint(nmin,nmax)                    # Generate a random number to randomly select one of the generated shapes, to be tried for this loop
            fail = 1
            while fail == 1:
                x,y,fail = pss.gen_coord(pss.mesh_Shps[I,:,:])
            area = pss.insert_shape_into_mesh(pss.mesh_Shps[I,:,:],x,y)
            ii += 1
            J += 1                                # J is now the total number of particles inserted
            placed_part_area.append(area)                    # Update the list of areas
            xcoords.append(x)
            ycoords.append(y)
            I_shape.append(I)
            J_shape.append(J)
        else: 
            if ii >= MM: ii = 0
            I = random.randint(nmin,nmax)                    # Generate a random number to randomly select one of the generated shapes, to be tried for this loop
            x,y,area = pss.drop_shape_into_mesh(pss.mesh_Shps[I,:,:])
            J += 1
            ii+= 1
            placed_part_area.append(area)                    # Update the list of areas
            xcoords.append(x)
            ycoords.append(y)
            I_shape.append(I)
            J_shape.append(J)
        
        #print placed_part_area,np.sum(placed_part_area)# After it reaches 50%, the algorithm really slows, so output the results of each step to see progress
        vol_placed_frac = np.sum(placed_part_area)/(pss.meshx*pss.meshy)    # update the volume fraction
        if vol_placed_frac == old_vfrac: 
            print '##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##########'
            print '# volume fraction no longer increasing. Break here #'
            print '##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##########'
            break
        old_vfrac = vol_placed_frac
        print "volume fraction achieved so far: {:3.3f}%".format(vol_placed_frac*100)
        
except KeyboardInterrupt:
    pass
I_shape   = np.array(I_shape)    
J_shape   = np.array(J_shape)    
xcr       = np.array(xcoords)
ycr       = np.array(ycoords)
zcr       = np.zeros_like(xcr)
#radii     = np.array(radii)

XINT      = np.copy(xcr)
YINT      = np.copy(ycr)

xcr   =   xcr.astype(float)
ycr   =   ycr.astype(float)
zcr   =   zcr.astype(float)
#radii = radii.astype(float)



"""
XY = np.column_stack((xcr,ycr))

mytree = scsp.cKDTree(XY,leafsize=100)
for item in XY:
	print mytree.query(item, k=np.size(mats)+1, distance_upper_bound=pss.cppr_max*3)
"""


MAT      = pss.mat_assignment(mats,xcr,ycr)
DMY      = np.zeros_like(xcoords)
xcr     *= GRIDSPC
ycr     *= GRIDSPC
zcr     *= GRIDSPC
#radii   *= GRIDSPC


#A,B = pss.part_distance(xcr,ycr,radii,MAT,False)
#print "The Contacts Measure, A = {}".format(A)
#print "Avg Contacts Between the Same Materials, B = {}".format(B)
#print 'Total contacts between same materials = {}, Total particles = {}'.format(B*J,J)
#ALL = np.column_stack((MAT,xcr,ycr,radii))

#pss.save_spherical_parts(xcr,ycr,radii,MAT,A)
#print 'save to meso_A-{:3.4f}.iSALE'.format(A)
#pss.save_particle_mesh(I_shape,XINT,YINT,MAT,J,mixed=True)
pss.populate_materials(I_shape,XINT,YINT,MAT,J)                              # Now populate the materials meshes (NB these are different to the 'mesh' and are
pss.save_general_mesh(mixed=True)


timestr = time.strftime('%d-%m-%Y_%H-%M-%S')
#np.savetxt('{}cppr_{}vfrlim_A{:1.3f}_{}.iSALE'.format(cppr,vfraclimit,A,timestr),ALL)
placed_part_area = np.array(placed_part_area)
print "total particles placed: {}".format(J)
vol_frac_calc = np.sum(placed_part_area)/(pss.meshx*pss.meshy)

if abs(vol_frac_calc - pss.vol_frac) <= 0.02:
    print "GREAT SUCCESS! Volume Fraction = {:3.3f}%".format(vol_frac_calc*100.)
else:
    print "FAILURE. Volume Fraction = {:3.3f}%".format(vol_frac_calc*100.)


plt.figure()
for KK in range(pss.Ms):
    matter = np.copy(pss.materials[KK,:,:])*(KK+1)
    matter = np.ma.masked_where(matter==0.,matter)
    plt.imshow(matter, cmap='plasma',vmin=0,vmax=pss.Ms,interpolation='nearest')
#plt.axis('equal')
plt.xlim(0,pss.meshx)
plt.ylim(0,pss.meshy)
plt.show()

