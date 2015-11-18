import numpy as np
import random
import matplotlib.pyplot as plt

def generate_mesh(X,Y,CPPR,pr,VF):
	global meshx, meshy, cppr_mid,PR,cppr_min,cppr_max,vol_frac,mesh,xh,yh,Ns,N,Nps,part_area,mesh0,mesh_Shps
	meshx 	  = X
	meshy 	  = Y
	cppr_mid  = CPPR
	PR        = pr
	cppr_min  = int((1-PR)*cppr_mid)					    	# Min No. cells/particle radius 
	cppr_max  = int((1+PR)*cppr_mid)						# Max No. cells/particle radius
	vol_frac  = VF									# Target fraction by volume of parts:void

	mesh      = np.zeros((meshx,meshy))
	xh        = np.arange(meshx)							#arrays of physical positions of cell BOUNDARIES (not centres)
	yh        = np.arange(meshy)
	Ns        = 2*(cppr_max)+2							# Dimensions of the mini-mesh for individual shapes. MUST BE EVEN.
	N         = 4									# Artifact number from an earlier version. serves as a safety net for Nps
	Nps       = estimate_no_particles() + N**2
	part_area = np.zeros((Nps))
	mesh0     = np.zeros((Ns,Ns))							# Generate mesh that is square and slightly larger than the max particle size
	mesh_Shps = np.zeros((Nps,Ns,Ns))						# Generate array of meshes of this size.

def estimate_no_particles():
	global cppr_min, meshx, meshy, vol_frac
	Area_total = meshx*meshy					# Total Area of mesh
	Area_1part = np.pi*cppr_mid**2.					# Area of one particle
	No_parts   = int(Area_total*vol_frac/Area_1part)+1		# Approx No. Particles, covering the volume fraction reqd.
	No_parts   = int(No_parts*1.1)					# There will be overlaps so overestimate by 10%
        return No_parts							

def gen_circle(r_):
	global mesh0, Ns
	x0 = cppr_max + 1.  
	y0 = cppr_max + 1.						# Define x0, y0 to be the centre of the mesh
	AREA  = 0.
	for j in range(Ns):						# Iterate through all the x- and y-coords
	    for i in range(Ns):						
		xc = 0.5*(i + (i+1)) - x0				# Convert current coord to position relative to (x0,y0) 
		yc = 0.5*(j + (j+1)) - y0				# Everything is now in indices, with (x0,y0)
		
		r = (xc/r_)**2. + (yc/r_)**2.
		if r<=1:
		    mesh0[i,j] = 1.0
		    AREA += 1
	#plt.figure(1)
	#plt.imshow(mesh0, cmap='Greys',  interpolation='nearest')
	#plt.show()
	return mesh0, AREA
	

def gen_polygon(sides,radii):						# To match the new method, generate a polygon from given radii and number of sides
	global mesh0, cppr_min, cppr_max, n_min, n_max, Ns		# Only the angles used are now randomly selected.
	AREA  = 0.
	#n    = int((random.random())*(n_max-n_min)+n_min)		# Random number of sides between n_min and n_max
	n     = sides
	R     = np.zeros((2,n))						# Array for the coords of the vertices
	delr  = (cppr_max-cppr_min)					# Difference between min and max radii
	
	I   = np.arange(n)						# array of vertex numbers
	ang = np.random.rand(n)											
	#rad = np.random.rand(n)						# Generate 'n' random radii 					
	phi = np.pi/2. - ang*np.pi*2./n - I*np.pi*2./n                  # Generate 'n' random angles 
	rho = radii				    			# Ensure each vertex is within an arc, 1/nth of a circle
	R[0,:] = rho*np.cos(phi)					# Each vertex will also be successive, such that drawing a line between
	R[1,:] = rho*np.sin(phi)					# each one in order, will result in no crossed lines.
									# Convert into cartesian coords and store in R
	qx = 0.								# Make the reference point (q) zero, i.e. the centre of the shape
	qy = 0.								# All dimensions are in reference to the central coordinates.
	x0 = cppr_max + 1.  
	y0 = cppr_max + 1.							# Define x0, y0 to be the centre of the mesh
	for j in range(Ns-1):						# Iterate through all the x- and y-coords
		for i in range(Ns-1):					# N-1 because we want the centres of each cell, and there are only N-1 of these!
			xc = 0.5*(i + (i+1)) - x0       		# Convert current coord to position relative to (x0,y0) 
			yc = 0.5*(j + (j+1)) - y0			# Everything is now in indices, with (x0,y0)

			sx = xc - qx					# s = vector difference between current coord and ref coord
			sy = yc - qy		
			intersection = 0				# Initialise no. intersections as 0
			for l in range(n-1):				# cycle through each edge, bar the last
				rx = R[0,l+1] - R[0,l]			# Calculate vector of each edge (r), i.e. the line between the lth vertex and the l+1th vertex
				ry = R[1,l+1] - R[1,l]
				RxS = (rx*sy-ry*sx)			# Vector product of r and s (with z = 0), technically produces only a z-component
				if RxS!=0.:				# If r x s  = 0 then lines are parallel
									# If r x s != 0 then lines are NOT parallel, but since they are of finite length, may not intersect
					t = ((qx-R[0,l])*sy - (qy-R[1,l])*sx)/RxS
					u = ((qx-R[0,l])*ry - (qy-R[1,l])*rx)/RxS
									# Consider two points along each line. They are t and u fractions of the way along each line
									# i.e. at q + us and p + tr. To find where lines intersect, consider these two equal and find t & u
									# if 0 <= t,u <= 1 Then there is intersection  between the lines of finite length!
					if t<=1. and t>=0. and u<=1. and u>=0.:
						intersection = intersection + 1
			rx = R[0,0] - R[0,n-1]				# Do the last edge. Done separately to avoid needing a circular 'for' loop
			ry = R[1,0] - R[1,n-1]
			if (rx*sy-ry*sy)!=0.:
				RxS = (rx*sy-ry*sx)
				t = ((qx-R[0,n-1])*sy - (qy-R[1,n-1])*sx)/RxS
				u = ((qx-R[0,n-1])*ry - (qy-R[1,n-1])*rx)/RxS
				if t<=1. and t>=0. and u<=1. and u>=0.:
					intersection = intersection + 1
			if (intersection%2==0.):			# If number of intersections is divisible by 2 (or just zero) -> fill that cell!
				mesh0[i,j] = 1.0
				AREA += 1
	#print 'Area = ',AREA

	return mesh0, AREA

def check_coords_full(shape,x,y):					# Function to check if the generated shape will fit at the generated coords
    global mesh, meshx, meshy, cppr_mid					# use global parameters
    Nx, Ny = meshx, meshy
    X, Y   = np.shape(shape)						# Dimensions of particle's mesh
    i_edge = int(x - X/2)						# Location of the edge of the shape to be checked, mesh, within the main mesh.
    j_edge = int(y - Y/2)
    i_finl = i_edge + X
    j_finl = j_edge + Y
    CHECK  = 0								# Initialise the CHECK as 0. 0 == all fine

    if i_edge < 0:							# If the coords have the particle being generated over the mesh boundary
	I_initial = abs(i_edge)						# This bit checks for this, and reassigns a negative starting index to zero
	i_edge    = 0							# However, the polygon's mesh will not completely be in the main mesh 
    else:								# So I_initial defines the cut-off point
	    I_initial = 0							# If the polygon's mesh does not extend beyond the main mesh, then I_initial is just 0
    if j_edge < 0:							# Repeat for the j-coordinate
	J_initial = abs(j_edge) 
	j_edge = 0
    else:
	J_initial = 0
    
    if i_finl > Nx: i_finl = Nx						# If coords place shape outside of other end of mesh, redefine end edges.		
    if j_finl > Ny: j_finl = Ny	
    
    if i_edge < cppr_mid:    CHECK = 1					# If shape overlaps the boundary in i, (x?) then fail the check.
    if i_finl > Nx-cppr_mid: CHECK = 1

    for i in range(i_edge, i_finl, 1):					# Iterate over the section of the full mesh of interest
	I = i - i_edge + I_initial					# The equivalent index within the polygon's mesh
	for j in range(j_edge, j_finl, 1):
	    J = j - j_edge + J_initial
	    if mesh[i,j]!=0. and shape[I,J] != 0.:
		CHECK = 1						# If there's a point in the polygon's mesh that has material, AND the same point    
		#print "check failed"
    #print "check = {}".format(CHECK)					# in the main mesh contains material. Then this placement has failed. CHECK = 1
    #if x <= cppr_max or x >= (Nx-cppr_max) or y <= cppr_max or y>= (Ny-cppr_max): CHECK = 1 initial attempt to make spheres appear within mesh only (no partials)
    return CHECK


def check_overlap(x, y, shape):
    global mesh, meshx, meshy
    
    Nx, Ny     = meshx, meshy
    Px, Py     = np.shape(shape)
    while touching == 0:	
	i_edge   = x - cppr_max - 1					# Location of the edge of the polygon's mesh, within the main mesh.
	j_edge   = y - cppr_max - 1
	i_finl   = x + cppr_max + 1					# As the indices refer to the close edge to the origin, the extra cell should be added on the other side
	j_finl   = y + cppr_max + 1					# i.e. the side furthest from the origin
	
	if i_edge < 0:							# If the coords have the particle being generated over the mesh boundary
	    I_initial = abs(i_edge)					# This bit checks for this, and reassigns a negative starting index to zero
	    i_edge    = 0						# However, the polygon's mesh will not completely be in the main mesh 
	else:								# So I_initial defines the cut-off point
	    I_initial = 0						# If the polygon's mesh does not extend beyond the main mesh, then I_initial is just 0
	if j_edge < 0:							# Repeat for the j-coordinate
	    J_initial = abs(j_edge) 
	    j_edge = 0
	else:
	    J_initial = 0

        I_final = Px 
        if (i_finl)>Nx:							# Similarly for the other end of the mesh
	    I_final -= abs(Nx-i_finl) 					# The above only sorts out two sides of the mesh
	    i_finl   = Nx
        J_final = Py
        if (j_finl)>Ny:
	    J_final -= abs(Ny-j_finl) 
	    j_finl   = Ny
	temp_shape = shape[I_initial:I_final,J_initial:J_final]		# record the shape as a temporary array (not outside the loop as it may be cut up)
	temp_mesh  = mesh[i_edge:i_finl,j_edge:j_finl]			# Same for the bit of mesh we are in right now
	test       = temp_mesh[temp_shape==1.]				# An array of all the points in the full mesh that are covered by the shape
	plt.figure(1)
	plt.imshow(test, cmap='Greys',  interpolation='nearest')
	plt.show()


def drop_shape_into_mesh(shape):
    global mesh, meshx, meshy, cppr_max, cppr_min

    touching   = 0
    passes     = 1
    counter    = 0
    
    x,y = gen_coord_basic()
    
    Nx, Ny     = meshx, meshy
    Px, Py     = np.shape(shape)
    while touching == 0:	
	if x > Ny-cppr_max or x < cppr_max: x,y = gen_coord_basic()
	if y > Nx-cppr_max or y < cppr_max: x,y = gen_coord_basic()
	i_edge   = y - cppr_max - 1					# Location of the edge of the polygon's mesh, within the main mesh.
	j_edge   = x - cppr_max - 1
	i_finl   = y + cppr_max + 1					# As the indices refer to the close edge to the origin, the extra cell should be added on the other side
	j_finl   = x + cppr_max + 1					# i.e. the side furthest from the origin
        	
	if i_edge < 0:							# If the coords have the particle being generated over the mesh boundary
	    I_initial = abs(i_edge)					# This bit checks for this, and reassigns a negative starting index to zero
	    i_edge    = 0						# However, the polygon's mesh will not completely be in the main mesh 
	else:								# So I_initial defines the cut-off point
	    I_initial = 0						# If the polygon's mesh does not extend beyond the main mesh, then I_initial is just 0
	if j_edge < 0:							# Repeat for the j-coordinate
	    J_initial = abs(j_edge) 
	    j_edge = 0
	else:
	    J_initial = 0

        I_final = Px 
        if (i_finl)>Nx:							# Similarly for the other end of the mesh
	    I_final -= abs(Nx-i_finl) 					# The above only sorts out two sides of the mesh
	    i_finl   = Nx
        J_final = Py
        if (j_finl)>Ny:
	    J_final -= abs(Ny-j_finl) 
	    j_finl   = Ny
	#print i_edge, i_finl, I_initial, I_final
	#print j_edge, j_finl, J_initial, J_final
	#print '\n'


	temp_shape = shape[I_initial:I_final,J_initial:J_final]		# The rectangular array containing the portion of shape to be placed into mesh 
	temp_mesh  = mesh[i_edge:i_finl,j_edge:j_finl]			# The equivalent rectangular array within the mesh, in the correct place
	#test       = temp_mesh[temp_shape!=1.]				# An array containing any points that have material in, in the same place, in BOTH arrays.
	test       = np.minimum(temp_shape,temp_mesh)
	#plt.figure(1)
	#plt.imshow(test, cmap='Greys',  interpolation='nearest')
	#plt.show()
	

	if (np.sum(test) > 2. or np.sum(test) == 0.):    		# If 'test' is > 2, then there are more than 2 cells overlapping with other objects at this position
	    y += random.randint(-cppr_min,cppr_min) 
	    x += random.randint(-cppr_min,cppr_min)
    	elif(np.sum(test) <= 2.):					# If there are fewer than 2 overlapping cells, but MORE than 0, place shape here.
	    mesh[i_edge:i_finl,j_edge:j_finl] = np.maximum(shape[I_initial:I_final,J_initial:J_final],mesh[i_edge:i_finl,j_edge:j_finl])
	    touching = 1						# Assign 'touching' a value of 1 to break loop
	    temp_shape[temp_shape>0.] = 1.				# Change all values in temp_shape to 1., if not already 
	    area = np.sum(temp_shape) - np.sum(test)			# Area placed into mesh is the sum of all positive points in temp_shape - overlap
    return x,y,area
            


"""
	






	if y < cppr_max or y > (meshy - cppr_max) or x < cppr_max or x > (meshx-cppr_max):
	    if np.size(test)>0. and np.amax(test) >= 0. and np.sum(test) <= 2. and touching == 0 :
	    #x,y,passes = gen_coord(shape)
		mesh[i_edge:i_finl,j_edge:j_finl] = np.maximum(shape[I_initial:I_final,J_initial:J_final],mesh[i_edge:i_finl,j_edge:j_finl])
		touching = 1
		temp_shape[temp_shape>0.] = 1.
		area = np.sum(temp_shape)

	    elif y<=cppr_max or y >= (meshy - cppr_max):		# Extra conditions as above is linked by 'or' statements and not 'and'
	    	x,y,passes = gen_coord(shape)
	    elif x<=cppr_max or x >= (meshx - cppr_max):
	    	x,y,passes = gen_coord(shape)
    	    else:
	    	x,y,passes = gen_coord(shape)
		
	elif np.size(test)>0. and np.amax(test) > 0. and np.sum(test) <= 2. and touching == 0 :						# If the max of this array is >0 the shape is touching something!
	    mesh[i_edge:i_finl,j_edge:j_finl] = np.maximum(shape[I_initial:I_final,J_initial:J_final],mesh[i_edge:i_finl,j_edge:j_finl])
	    touching = 1
	    temp_shape[temp_shape>0.] = 1.
	    area = np.sum(temp_shape) - np.sum(test)
	else:
	    #print x,y
	    y += random.randint(-cppr_min,cppr_min) 
	    x += random.randint(-cppr_min,cppr_min) 
"""

def insert_shape_into_mesh(shape,x0,y0):
    global mesh, meshx, meshy, cppr_max
    Nx, Ny = meshx, meshy
    Px, Py = np.shape(shape)
    nn     = cppr_max								# 1/2 the width of the polygon's mesh, as an integer
    i_edge = x0 - nn - 1							# Location of the edge of the polygon's mesh, within the main mesh.
    j_edge = y0 - nn - 1
    i_finl = x0 + nn + 1							# As the indices refer to the close edge to the origin, the extra cell should be added on the other side
    j_finl = y0 + nn + 1							# i.e. the side furthest from the origin
    if i_edge < 0:								# If the coords have the particle being generated over the mesh boundary
	I_initial = abs(i_edge)							# This bit checks for this, and reassigns a negative starting index to zero
	i_edge    = 0								# However, the polygon's mesh will not completely be in the main mesh 
    else:									# So I_initial defines the cut-off point
	I_initial = 0								# If the polygon's mesh does not extend beyond the main mesh, then I_initial is just 0
    if j_edge < 0:								# Repeat for the j-coordinate
	J_initial = abs(j_edge) 
	j_edge = 0
    else:
	J_initial = 0

    I_final = Px 
    if (i_finl)>Nx:
	I_final -= abs(Nx-i_finl) 
	i_finl   = Nx
    J_final = Py
    if (j_finl)>Ny:
	J_final -= abs(Ny-j_finl) 
	j_finl   = Ny

    temp_shape = shape[I_initial:I_final,J_initial:J_final]			# record the shape as a temporary array for area calculation
    #print shape
    mesh[i_edge:i_finl,j_edge:j_finl] = np.maximum(shape[I_initial:I_final,J_initial:J_final],mesh[i_edge:i_finl,j_edge:j_finl])
    temp_shape[temp_shape>0.] = 1.						# All non-zero elements become 1, as particle may have different material number
    area = np.sum(temp_shape)							# Area is sum of all these points
    return area


def insert_shape_into_mesh_extra(shape,x0,y0,J):
    global mesh, meshx, meshy, cppr_max
    Nx, Ny = meshx, meshy
    Px, Py = np.shape(shape)
    nn     = cppr_max								# 1/2 the width of the polygon's mesh, as an integer
    i_edge = x0 - nn - 1							# Location of the edge of the polygon's mesh, within the main mesh.
    j_edge = y0 - nn - 1
    i_finl = x0 + nn + 1							# As the indices refer to the close edge to the origin, the extra cell should be added on the other side
    j_finl = y0 + nn + 1							# i.e. the side furthest from the origin
    if i_edge < 0:								# If the coords have the particle being generated over the mesh boundary
	I_initial = abs(i_edge)							# This bit checks for this, and reassigns a negative starting index to zero
	i_edge    = 0								# However, the polygon's mesh will not completely be in the main mesh 
    else:									# So I_initial defines the cut-off point
	I_initial = 0								# If the polygon's mesh does not extend beyond the main mesh, then I_initial is just 0
    if j_edge < 0:								# Repeat for the j-coordinate
	J_initial = abs(j_edge) 
	j_edge = 0
    else:
	J_initial = 0

    I_final = Px 
    if (i_finl)>Nx:
	I_final -= abs(Nx-i_finl) 
	i_finl   = Nx
    J_final = Py
    if (j_finl)>Ny:
	J_final -= abs(Ny-j_finl) 
	j_finl   = Ny

    temp_shape = shape[I_initial:I_final,J_initial:J_final]			# record the shape as a temporary array for area calculation
    temp_shape *= (J)								# Each particle is numbered!
    mesh[i_edge:i_finl,j_edge:j_finl] = np.maximum(temp_shape,mesh[i_edge:i_finl,j_edge:j_finl])
    temp_shape[temp_shape>0.] = 1.						# All non-zero elements become 1, as particle may have different material number
    area = np.sum(temp_shape)							# Area is sum of all these points
    return i_edge,i_finl,j_edge,j_finl,area

def gen_coord_basic():								# Quick routine to just generate 2 coords without checking for overlap
	global mesh,meshx,meshy,cppr_max					# Use the global mesh
	good = 0
	indices = np.where(mesh==0.)						# Generates 2 arrays one containing all the x coords and one, the corresponding y coords.
	indices = np.column_stack(indices)					# Where mesh = 0. column_stack mushes them together
	x,y = random.choice(indices)						# This randomly selects one pair of coordinates!
	while good == 0:
		if y < cppr_max or y > meshy-cppr_max:
			x,y = random.choice(indices)				# This randomly selects one pair of coordinates!
		else:
			good = 1
	return x,y


def gen_coord(shape):								# Method no. 2 generates 1 coord at a time
	global mesh, meshx, meshy, cppr_mid, cppr_max				# use global parameters
	check   = 1
	counter = 0
	passes  = 0
	Nx, Ny = meshx, meshy
	indices = np.where(mesh==0.)	# Generates 2 arrays one containing all the x coords and one, the corresponding y coords.
	indices = np.column_stack(indices)					# Where mesh = 0. column_stack mushes them together
	while check == 1:
	    x,y = random.choice(indices)					# This randomly selects one pair of coordinates!
	    check = check_coords_full(shape,x,y)				# Function to check if the polygon generated will fit in the generated coords
	    counter += 1
	    if counter>5000: 
	        check = 0
	        passes= 1							# if counter exceeds 1000 -> break loop and assign passes[k] = 1
		break
	return x,y,passes

#def assign_mat_number(I,x,y):
#	global mesh

def gen_N_coords(N,shape):							# generate N particle coords
	global mesh, meshx, meshy, cppr_mid					# use global parameters
	Nx, Ny= meshx, meshy							# set meshx and meshy to the local parameters Nx, Ny
	x     = np.zeros((N*N))							# Create arrays of all the x and y coords to be generated	
	y     = np.zeros((N*N))							
	passes= np.zeros((N*N))							# Array of the number of 'passes' if the Jth counter exceeds 1000, passes[J] = 1
        check =  np.ones((N*N))							# Check array. if check[J] = 1 then the Jth term has failed placement
	coord = np.zeros((N*N))							# Coordinate number, i.e. coord[J] == the Jth coordinate
	k = 0
	imin = 0
	jmin = 0
	for l in range(4):
	    imax = N
	    jmax = N
	    if l==1: 
		imin+=1	
		imax+=1
	    if l==2:
		jmin+=1
		jmax+=1
	    if l==3: 
		imin-=1
		imax-=1
	    for i in range(imin,imax,2):
		for j in range(jmin,jmax,2):
		    counter = 0
		    
		    while check[k] == 1:
			counter += 1						# increment counter each time a new coord is generated
			
			x[k]     = random.choice

			x[k]     = int(random.random()*(Nx/N) + i*Nx/N)		# regenerate coords, but, keep SAME phase.
			y[k]     = int(random.random()*(Ny/N) + j*Ny/N)
			x[k]     = int(x[k])
			y[k]     = int(y[k])
			if x[k]>Nx: x[k] = Nx					# ensure coords are inside mesh
			if y[k]>Ny: y[k] = Ny
			if x[k]<0:  x[k] = 0
			if y[k]<0:  y[k] = 0
			#check[k] = check_coords_quick(x[k],y[k])		# Function to check if the polygon generated will fit in the generated coords
			check[k] = check_coords_full(shape[:,:],x[k],y[k])	# Function to check if the polygon generated will fit in the generated coords
			if counter>5000: 
			    check[k] = 0
			    passes[k]= 1					# if counter exceeds 1000 -> break loop and assign passes[k] = 1
		    k+=1
	
	#print 'Number of passes = {}'.format(sum(passes))		
	return x,y,passes							# return N cell coords for particle placement, as well as their successes(=0)/failures(=1)


"""
meshx     = 500									# No. cells in mesh in x-direction
meshy     = 500									# No. cells in mesh in y-direction
cppr_mid  = 10 
PR        = 0.3
cppr_min  = int((1-PR)*cppr_mid)					    	# Min No. cells/particle radius 
cppr_max  = int((1+PR)*cppr_mid)						# Max No. cells/particle radius
L         = 3.2e-6								# Distance across one cell (m) (GRIDSPC)
n_min     = 10									# Minimum No. Sides/particle
n_max     = 10									# Maximum No. Sides/particle
vol_frac  = .5									# Target fraction by volume of parts:void

mesh      = np.zeros((meshx,meshy))
xh        = np.arange(meshx)							#arrays of physical positions of cell BOUNDARIES (not centres)
yh        = np.arange(meshy)
MX        = meshx*L								#mesh length and width in metres
MY        = meshy*L
Ns        = 2*(cppr_max)+2							# minimum of a pentagon max of a ...10-sided one

N         = 4
Nps       = estimate_no_particles() + N**2
part_area = np.zeros((Nps))
mesh0     = np.zeros((Ns,Ns))							# Generate mesh that is square and slightly larger than the max particle size
mesh_Shps = np.zeros((Nps,Ns,Ns))						# Generate array of meshes of this size.
"""

