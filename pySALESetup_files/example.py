import pySALESetup as pss
import numpy as np

# Here we will set up a simple model iSALE is typically unable
# to investigate with its traditional set up routine

# This generates the mesh will be filling and all the arrays
# associated with it. We will use a 100 x 100 mesh (1 mm x 1 mm)
# You also need to add either the largest cppr you will use or 
# a cppr and the range of sizes to be used.
pss.generate_mesh(100,100,GridSpc=10.e-6,mat_no=5,CPPR=25)

# Fill the bottom half of the mesh with material #1
# fill_plate takes the lower y coord and the upper y coord
# and fills between them
pss.fill_plate(0.,.5e-3,1)

# Provide a series of x vertices and y vertices and fill_polygon
# will fill the shape (with material #2 this time)
pss.fill_polygon([0,0.5e-3,0.7e-3,0.2e-3,0],[0.1e-3,0.25e-3,0.2e-3,0.e-3,0.1e-3],2)

# This lets us look at the mesh, however the polygon does not
# appear! pySALESetup gives priority to materials already present
pss.display_mesh()

# If we add the material '-1', this is treated as void
pss.fill_plate(0.,.25e-3,-1)
pss.fill_polygon([0,0.5e-3,0.7e-3,0.2e-3,0],[0.1e-3,0.25e-3,0.2e-3,0.e-3,0.1e-3],2)

# Now the shape appears!
pss.display_mesh()

# If we refill the portion we deleted with a new material, material #3
pss.fill_plate(0.,.25e-3,3)
pss.display_mesh()

# Now we can set about flling the top half of the mesh, let's put some
# grains in, each inside the other
r1 = 0.1e-3
cppr1 = int(r1/10.e-6)
r2 = 0.175e-3
cppr2 = int(r2/10.e-6)
r3 = 0.25e-3
cppr3 = int(r3/10.e-6)
pss.mesh_Shps[0] = pss.gen_circle(cppr1)
pss.mesh_Shps[1] = pss.gen_circle(cppr2)
pss.mesh_Shps[2] = pss.gen_circle(cppr3)
for i in range(3):
    pss.place_shape(pss.mesh_Shps[i],0.5e-3,0.75e-3,i+2)
pss.display_mesh()

# Finally let's fill the rest of the mesh with material #5 which
# has not yet been used. And give the top half a velocity of -1 km/s
pss.fill_Allmesh(5)
pss.plate_vel(0.5e-3,1.e-3,-1.e3)
# Now lets check the mesh and the velocities
pss.display_mesh(showvel=True)
# And save!
pss.save_general_mesh(fname='meso_m.iSALE')
