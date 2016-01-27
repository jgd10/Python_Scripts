import numpy as np
import scipy as sc
import random
import matplotlib.pyplot as plt
import pySALESetup as pss
import time


vol_frac   = .55
X_cells    = 500 
Y_cells    = 500 
PR         = 0.
cppr       = 6 
vfraclimit = .4                                # The changeover point from random to forced contacts. > 1.0 => least contacts; = 0. Max contacts
x_length   = 1.e-3
y_length   = 1.e-3
GRIDSPC    = x_length/X_cells
#X_cells   -= int(2*cppr*1.1)
#Y_cells   -= int(2*cppr*1.1)

pss.generate_mesh(X_cells,Y_cells,cppr,PR,vol_frac)


""" #################################################################################### """
""" #######   Initialise variables and estimate the number of particles needed   ####### """
""" #################################################################################### """
counter   = 0
counter_2 = 0
j = 0                                    # No. particles placed
J = 0                                    # No. particles tried
N = pss.N                                    # test value
vol_placed_frac = 0.
placed_part_area = []
#print "Estimated No. Particles = {}".format(Nps)            # Calculate rough no. particles required

mats = np.array([1.,2.,3.,4.,5.])
ii = 0
MM = np.size(mats)


""" #################################################################################### """
""" #######   GENERATE n+1 Particles - to be placed in appropriate distribution  ####### """
""" #################################################################################### """
n = pss.N                                 # Particles can be placed MORE than once!
part_area        = np.zeros((n))
part_radii = []
cppr_range = pss.cppr_max - pss.cppr_min
for i in range(n):                            # n+1 as range starts at 0; i.e. you'll never get to i = n unless range goes to n+1!
    r = pss.cppr_min + i*cppr_range/(n-1)                # generate radii that are incrementally greater for each circle produced
    pss.mesh_Shps[i,:,:],part_area[i] = pss.gen_circle(r)
    part_radii.append(r)


""" #################################################################################### """
""" #######              sort particles into order of Large -> Small             ####### """
""" #################################################################################### """

sort_indices  = np.argsort(part_area)                    # Sort areas into smallest to largest
part_area     = part_area[sort_indices]                    # Arrange as appropriate
pss.mesh_Shps = pss.mesh_Shps[sort_indices,:,:]
part_area     = part_area[::-1]                        # Sort shapes into same order as areas
pss.mesh_Shps = pss.mesh_Shps[::-1,:,:]
part_radii    = part_radii[::-1]			# Radii are not 'sorted' as they are created in size order already
#print part_radii, part_area[0:n]

""" #################################################################################### """
""" #######    Loop generating coords and inserting particles into the mesh     ######## """
""" #################################################################################### """
nmin = 0
nmax = n-1                                # max integer generated

xcoords = []
ycoords = []
radii   = []
I_shape = []
J_shape = []

try:
    while vol_placed_frac<vol_frac:                        # Keep attempting to add particles until the required volume fraction is achieved
        if J == 0:                                    # FIRST PARTICLE must ALWAYS be randomly placed
            I = random.randint(nmin,nmax)                    # Generate a random number to randomly select one of the generated shapes, to be tried for this loop
            fail = 1
            while fail == 1:
                x,y,fail = pss.gen_coord(pss.mesh_Shps[I,:,:])
            area = pss.insert_shape_into_mesh(pss.mesh_Shps[I,:,:],x,y,mats[ii])
            ii+= 1
            J += 1                                # J is now the total number of particles inserted
            placed_part_area.append(area)                    # Update the list of areas
            xcoords.append(x)
            ycoords.append(y)
            radii.append(part_radii[I])
            I_shape.append(I)
            J_shape.append(J)
    
        if vol_placed_frac < vfraclimit*vol_frac:
            if ii >= MM: ii = 0
            I = random.randint(nmin,nmax)                    # Generate a random number to randomly select one of the generated shapes, to be tried for this loop
            fail = 1
            while fail == 1:
                x,y,fail = pss.gen_coord(pss.mesh_Shps[I,:,:])
            area = pss.insert_shape_into_mesh(pss.mesh_Shps[I,:,:],x,y,mats[ii])
            ii += 1
            J += 1                                # J is now the total number of particles inserted
            placed_part_area.append(area)                    # Update the list of areas
            xcoords.append(x)
            ycoords.append(y)
            radii.append(part_radii[I])
            I_shape.append(I)
            J_shape.append(J)
        else: 
            if ii >= MM: ii = 0
            I = random.randint(nmin,nmax)                    # Generate a random number to randomly select one of the generated shapes, to be tried for this loop
            x,y,area = pss.drop_shape_into_mesh(pss.mesh_Shps[I,:,:],part_radii[I],mats[ii])
            J += 1
            ii+= 1
            placed_part_area.append(area)                    # Update the list of areas
            xcoords.append(x)
            ycoords.append(y)
            radii.append(part_radii[I])
            I_shape.append(I)
            J_shape.append(J)
        
        #print placed_part_area,np.sum(placed_part_area)# After it reaches 50%, the algorithm really slows, so output the results of each step to see progress
        vol_placed_frac = np.sum(placed_part_area)/(pss.meshx*pss.meshy)    # update the volume fraction
        print "volume fraction achieved so far: {:3.3f}%".format(vol_placed_frac*100)
        
except KeyboardInterrupt:
    pass
I_shape   = np.array(I_shape)    
J_shape   = np.array(J_shape)    
xcr       = np.array(xcoords)
ycr       = np.array(ycoords)
radii     = np.array(radii)

xcr   =   xcr.astype(float)
ycr   =   ycr.astype(float)
radii = radii.astype(float)


"""
Mats      = np.zeros_like(xcoords)
Mats     += 2

for h in range(np.amax(J_shape)):                    
    Xc = xcr - xcr[h]
    Yc = ycr - ycr[h]
    R  = np.sqrt(Xc**2. + Yc**2.)
    for material in Mats[(R<(3.*radii[h]))*(R>0.)]:
        if material == Mats[h]:
            MatNum = np.roll(MatNum,1)
            Mats[h] = MatNum[0]
            pss.mesh[(xcr[h]-radii[h]):(xcr[h]+radii[h]), (ycr[h]-radii[h]):(ycr[h]+radii[h])] *= Mats[h]


sort_indices  = np.argsort(Mats)                    # Sort areas into smallest to largest
xcoords       = xcoords[sort_indices]                    # Arrange as appropriate
ycoords       = ycoords[sort_indices]                    # Arrange as appropriate
zcoords       = zcoords[sort_indices]                    # Arrange as appropriate
radii         =   radii[sort_indices]                    # Arrange as appropriate
"""
MAT = pss.mat_assignment(mats,xcr,ycr,radii)
zcr      = np.zeros_like(xcoords)
DMY      = np.zeros_like(xcoords)
xcr     *= GRIDSPC
ycr     *= GRIDSPC
zcr     *= GRIDSPC
radii   *= GRIDSPC
A,B = pss.part_distance(xcr,ycr,radii,MAT,True)
print "The Contacts Measure, A = {}".format(A)
print "Avg Contacts Between the Same Materials, B = {}".format(B)
print 'Total contacts between same materials = {}, Total particles = {}'.format(B*J,J)
ALL = np.column_stack((MAT,xcr,ycr,radii))

timestr = time.strftime('%d-%m-%Y_%H-%M-%S')
np.savetxt('{}cppr_{}vfrlim_A{:1.3f}_{}.iSALE'.format(cppr,vfraclimit,A,timestr),ALL)
placed_part_area = np.array(placed_part_area)
print "total particles placed: {}".format(J)
vol_frac_calc = np.sum(placed_part_area)/(pss.meshx*pss.meshy)

if abs(vol_frac_calc - pss.vol_frac) <= 0.02:
    print "GREAT SUCCESS! Volume Fraction = {:3.3f}%".format(vol_frac_calc*100.)
else:
    print "FAILURE. Volume Fraction = {:3.3f}%".format(vol_frac_calc*100.)

plt.figure(3)
plt.imshow(pss.mesh, cmap='Greys',  interpolation='nearest')
plt.show()

