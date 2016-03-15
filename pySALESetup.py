"""  This is a module in-progress for use generating meso-setups to be put into iSALE """
""" There is much to work on and more is being added all the time                     """
""" Last edited - 19/01/16 - JGD													  """


import numpy as np
import random
import matplotlib.pyplot as plt

def generate_mesh(X,Y,mat_no,CPPR=10,pr=0.,VF=.5,e = 0.):
	"""
	This function generates the global mesh that all particles will be inserted into.
	Initially it reads in several parameters and renames them within the module. Then 
	several arrays are initialised, both of the main mesh and the 'shape' meshes (mesh0)

	X    : The x-length of the mesh, in cells
	Y    : The y-length of the mesh, in cells
	CPPR : The chosen value of cells per particle radius
	pr   : The particle size range (a fraction). The range of particle sizes that will be produced
	VF   : The volume fraction. AKA the ratio of particle to void we want to achieve

	Nothing is returned but many factors become global.

	NB. Recently I have updated 'N' to now be the number of different particles that can be generated
	and NOT N**2. 19-01-16
	"""
	global meshx, meshy, cppr_mid,PR,cppr_min,cppr_max,vol_frac,mesh,xh,yh,Ns,N,Nps,part_area,mesh0,mesh_Shps,eccen,materials,Ms,mats,objects,FRAC,OBJID

	Ms        = mat_no	 																				# M is the number of materials within the mesh
	mats      = np.arange(Ms)+1.
	meshx 	  = X
	meshy 	  = Y
	cppr_mid  = CPPR
	PR        = pr
	eccen     = e																						# Eccentricity of ellipses. e = 0. => circle. 0 <= e < 1
	cppr_min  = int((1-PR)*cppr_mid)													    			# Min No. cells/particle radius 
	cppr_max  = int((1+PR)*cppr_mid)																	# Max No. cells/particle radius
	vol_frac  = VF																						# Target fraction by volume of parts:void
	mesh      = np.zeros((meshx,meshy))
	materials = np.zeros((Ms,meshx,meshy))																# The materials array contains a mesh for each material number
	objects   = np.zeros((Ms,meshx,meshy))																# The materials array contains a mesh for each material number
	xh        = np.arange(meshx)																		# arrays of physical positions of cell BOUNDARIES (not centres)
	yh        = np.arange(meshy)
	Ns        = 2*(cppr_max)+2															    			# Dimensions of the mini-mesh for individual shapes. MUST BE EVEN.
	N         = 20																						# N is the number of different particles that can be generated  
	part_area = np.zeros((N))
	mesh0     = np.zeros((Ns,Ns))																		# Generate mesh that is square and slightly larger than the max particle size
																										# This will be the base for any shapes that are generated
	mesh_Shps = np.zeros((N,Ns,Ns))																		# Generate an array of meshes of this size. of size N (Ns x Ns x N)
	FRAC      = np.zeros((Ms,meshx*meshy))																# An array for storing the fractions of material 
	OBJID     = np.zeros((Ms,meshx*meshy))																# An array for storing the fractions of material 

def estimate_no_particles(R = 10, X = 1000, Y = 1000, VF = 0.5):
	"""
	A simple function to estimate the number of particles needed for a specific job. 
	Usually this overestimates. Only considers circular particles.

	It simply calculates the area of one particle, then the fraction that is to be
	filled, then the number that would go into that. This number is returned.

	R  : Radius of a particle in cells (default = 10)
	X  : X-length of mesh in cells (default = 1000)
	Y  : Y-length of mesh in cells (default = 1000)
	VF : Volume fraction of mesh to be filled (default = 0.5)
	
	Returns a float value.
	"""
	global cppr_mid, meshx, meshy, vol_frac
	Area_total = meshx*meshy																			# Total Area of mesh
	Area_1part = np.pi*cppr_mid**2.																		# Area of one particle
	No_parts   = int(Area_total*vol_frac/Area_1part)+1													# Approx No. Particles, covering the volume fraction reqd.
	No_parts   = int(No_parts)																			# There will be overlaps so overestimate by 10%
	return No_parts							

def gen_circle(r_):
	"""
	This function generates a circle within the base mesh0. It very simply converts
	each point to a radial coordinate from the origin (the centre of the shape.
	Then assesses if the radius is less than that of the circle in question. If it 
	is, the cell is filled.

	r_ : radius of the circle, origin is assumed to be the centre of the mesh0

	mesh0 and an AREA are returned
	"""
	global mesh0, Ns																					# mesh0 is Ns x Ns in size
	mesh0[:] = 0. 																						# Ensure that mesh0 is all zeros (as it should be before this is called)
	x0 = cppr_max + 1.	  																				# Ns = 2*cppr_max + 2, so half well be cppr_max + 1
	y0 = cppr_max + 1.																					# Define x0, y0 to be the centre of the mesh
	AREA  = 0.																							# Initialise AREA as 0.
	for j in range(Ns):																					# Iterate through all the x- and y-coords
	    for i in range(Ns):						
		xc = 0.5*(i + (i+1)) - x0																		# Convert current coord to position relative to (x0,y0) 
		yc = 0.5*(j + (j+1)) - y0																		# Everything is now in cartesian coords relative to the mesh centre
		
		r = (xc/r_)**2. + (yc/r_)**2.																	# Find the radial distance from current coord to (x0,y0), normalised by r_
		if r<=1:																						# If this is less than 1 (i.e. r_) then the coord is in the circle => fill
		    mesh0[i,j] = 1.0																			# Fill cell
		    AREA += 1																					# Increment area
	return mesh0, AREA

def gen_ellipse(r_,a_,e_):
	"""
	This function generates an ellipse in mesh0. It uses a semi-major axis of r_
	a rotation of a_ and an eccentricity of e_. It otherwise works on
	principles similar to those used in gen_circle.
	
	r_ : the semi major axis (in cells)
	a_ : the angle of rotation (in radians)
	e_ : the eccentricity of the ellipse
	"""
	global mesh0, Ns
	x0 = cppr_max + 1.
	y0 = cppr_max + 1.
	AREA = 0.
	mesh0[:] = 0.																						# A safety feature to ensure that mesh0 is always all 0. before starting
	A = r_
	B = A*np.sqrt(1.-e_**2.)																			# A is the semi-major radius, B is the semi-minor radius
	for j in range(Ns):
	    for i in range(Ns):
	        xc = 0.5*(i + (i+1)) - x0
	        yc = 0.5*(j + (j+1)) - y0 
	
	        xct = xc * np.cos(a_) - yc * np.sin(a_)
	        yct = xc * np.sin(a_) + yc * np.cos(a_)
	        r = (xct/A)**2. + (yct/B)**2.
	
	        if r<=1:
	            mesh0[i,j] = 1.
	            AREA+=1
	"""
	def makeplot():
	    fig=plt.figure()
	    ax=fig.add_subplot(111,aspect='equal')
	    ax.imshow(mesh0)
	    plt.show()
	    fig.clf()
	"""
	return mesh0, AREA
	
def gen_circle_p(r_):
	"""	
	This function generates a circle within the base mesh0. It very simply converts
	each point to a radial coordinate from the origin (the centre of the shape.
	Then assesses if the radius is less than that of the circle in question. If it 
	is, the cell is filled.
	
	This version partially fills cells by breaking the cells nearest the edge into 10 x 10 mini meshes
	and then it calculates how many of these are within the mesh to get a partial fill value for that 
	cell.
	
	r_ : radius of the circle, origin is assumed to be the centre of the mesh0 (in cells)
	
	mesh0 and an AREA are returned
	"""
	global mesh0, Ns																					# mesh0 is Ns x Ns in size
	CL = 1.																								# This is mesh0 so cell length is simply '1.'
	x0 = cppr_max + 1.	  																				# Ns = 2*cppr_max + 2, so half well be cppr_max + 1
	y0 = cppr_max + 1.																					# Define x0, y0 to be the centre of the mesh
	AREA  = 0.																							# Initialise AREA as 0.
	mesh0[:] = 0.																						# This is necessary to ensure the mesh is empty before use.
	for j in range(Ns):																					# Iterate through all the x- and y-coords
	    for i in range(Ns):						
			xc = 0.5*(i + (i+1)) - x0																	# Convert current coord (centre of cell) to position relative to (x0,y0) 
			yc = 0.5*(j + (j+1)) - y0																	# Everything is now in cartesian coords relative to the mesh centre
			r = np.sqrt((xc)**2. + (yc)**2.)															# Find the radial distance from current coord to (x0,y0)_
			if r<=(r_-1):																				# If this is less than r_-1 then the coord is in the circle 
																										# => fill COMPLETELY
		   		mesh0[i,j] = 1.0																		# Fill cell
		   		AREA += 1																				# Increment area
			elif abs(r-r_)<=np.sqrt(2):																	# BUT if the cell centre is with root(2) of the circle edge, 
																										# then partially fill
				xx = np.linspace(i,i+1,11)																# Create 2 arrays of 11 elements each and with divisions of 0.1
				yy = np.linspace(j,j+1,11)																# Essentially this splits the cell into a mini-mesh 
																										# of 10 x 10 mini cells
				for I in range(10):
					for J in range(10):																	# Iterate through these
						xxc = 0.5*(xx[I] + xx[J+1]) - x0												# Change their coordinates as before
						yyc = 0.5*(yy[J] + yy[J+1]) - y0
						r = np.sqrt((xxc)**2. + (yyc)**2. )												# Find the radial distance from current mini coord to (x0,y0)
						if r <= r_:																		# If this is less than r_ then the mini coord is in the circle.
							mesh0[i,j] += (.1**2.)														# Fill cell by 0.1**2 (the area of one mini-cell)
							AREA += (.1**2.)															# Increment area by the same amount
	"""
	plt.figure()
	plt.imshow(mesh0,cmap='Greys')
	plt.show()
	"""
	return mesh0, AREA

def gen_polygon(sides,radii):						
	"""
	This function generates a polygon, given a specified number of sides and
	specific radii. The angles between vertices are randomly generated in this
	method.

	This is done principally by defining the x,y position of each point then
	considering lines, drawn from each mesh point to a reference point (taken to be the origin
	and centre of the shape). If they cross the edge of the shape an even number
	of times, the cell is considered to be in the shape and is filled in.

	sides : An integer number greater than 2
	radii : An array of radii, size = sides

	The area of the shape and it's mesh0 are returned.
	"""
	global mesh0, cppr_min, cppr_max, n_min, n_max, Ns													# Only the angles used are now randomly selected.
	AREA  = 0.
	n     = sides
	R     = np.zeros((2,n))																				# Array for the coords of the vertices
	delr  = (cppr_max-cppr_min)																			# Difference between min and max radii
	mesh0[:] = 0.																						# This is necessary to ensure the mesh is empty before use.
	I   = np.arange(n)																					# array of vertex numbers
	ang = np.random.rand(n)																				
	phi = np.pi/2. - ang*np.pi*2./n - I*np.pi*2./n														# Generate 'n' random angles 
	rho = radii				    																		# Ensure each vertex is within an arc, 1/nth of a circle
	R[0,:] = rho*np.cos(phi)																			# Each vertex will also be successive, such that drawing a line between
	R[1,:] = rho*np.sin(phi)																			# each one in order, will result in no crossed lines.
																										# Convert into cartesian coords and store in R
	qx = 0.																								# Make the reference point (q) zero, i.e. the centre of the shape
	qy = 0.																								# All dimensions are in reference to the central coordinates.
	x0 = cppr_max + 1.  
	y0 = cppr_max + 1.																					# Define x0, y0 to be the centre of the mesh
	for j in range(Ns-1):																				# Iterate through all the x- and y-coords
		for i in range(Ns-1):																			# N-1 because we want the centres of each cell, 
																										# and there are only N-1 of these!
			xc = 0.5*(i + (i+1)) - x0																	# Convert current coord to position relative to (x0,y0) 
			yc = 0.5*(j + (j+1)) - y0																	# Everything is now in indices, with (x0,y0)

			sx = xc - qx		         																# s = vector difference between current coord and ref coord
			sy = yc - qy		   
			intersection = 0																			# Initialise no. intersections as 0
			for l in range(n-1):																		# cycle through each edge, bar the last
				rx = R[0,l+1] - R[0,l]																	# Calculate vector of each edge (r), i.e. the line between 
																										# the lth vertex and the l+1th vertex
				ry = R[1,l+1] - R[1,l]
				RxS = (rx*sy-ry*sx)																		# Vector product of r and s (with z = 0), technically 
																										# produces only a z-component
				if RxS!=0.:																				# If r x s  = 0 then lines are parallel
																										# If r x s != 0 then lines are NOT parallel, 
																										# but since they are of finite length, may not intersect
					t = ((qx-R[0,l])*sy - (qy-R[1,l])*sx)/RxS
					u = ((qx-R[0,l])*ry - (qy-R[1,l])*rx)/RxS
																										# Consider two points along each line. 
																										# They are t and u fractions of the way along each line
																										# i.e. at q + us and p + tr. To find where lines intersect,
																										# consider these two equal and find t & u
																										# if 0 <= t,u <= 1 Then there is intersection between 
																										# the lines of finite length!
					if t<=1. and t>=0. and u<=1. and u>=0.:
						intersection = intersection + 1
			rx = R[0,0] - R[0,n-1]																		# Do the last edge. Done separately to avoid needing a circular
																										# 'for' loop
			ry = R[1,0] - R[1,n-1]
			if (rx*sy-ry*sy)!=0.:
				RxS = (rx*sy-ry*sx)
				t = ((qx-R[0,n-1])*sy - (qy-R[1,n-1])*sx)/RxS
				u = ((qx-R[0,n-1])*ry - (qy-R[1,n-1])*rx)/RxS
				if t<=1. and t>=0. and u<=1. and u>=0.:
					intersection = intersection + 1
			if (intersection%2==0.):																	# If number of intersections is divisible by 2 (or just zero)
																										# -> fill that cell!
				mesh0[i,j] = 1.0
				AREA += 1

	return mesh0, AREA

def check_coords_full(shape,x,y):																		
	"""
	This function checks if the chosen shape will overlap with any other material,
	should it be placed.
	
	It works by initially checking the location of the generated coords and 
	ammending them if the shape overlaps the edge of the mesh. Then the two arrays
	can be compared.
	
	shape : the array based on mesh0 containg the shape
	x     : The x coord of the shape's origin
	y     : The equivalent y coord
	
	the value of CHECK is returned. 1 is a failure, 0 is success.
	"""
	global mesh, meshx, meshy, cppr_mid																	# use global parameters
	X, Y   = np.shape(shape)																			# Dimensions of particle's mesh
	i_edge = int(x - X/2)																				# Location of the edge of the shape to be checked's mesh; 
																										# within the main mesh.
	j_edge = int(y - Y/2)
	i_finl = i_edge + X
	j_finl = j_edge + Y
	CHECK  = 0																							# Initialise the CHECK as 0; 0 == all fine
	
	if i_edge < 0:																						# If the coords have the particle being generated over the mesh boundary
		I_initial = abs(i_edge)																			# This bit checks for this, 
																										# and reassigns a negative starting index to zero
		i_edge    = 0																					# However, the polygon's mesh will not completely be in the main mesh 
	else:																								# So I_initial defines the cut-off point
	    I_initial = 0																					# If the polygon's mesh does not extend beyond the main mesh,
																										# then I_initial is just 0
	if j_edge < 0:																						# Repeat for the j-coordinate
		J_initial = abs(j_edge) 
		j_edge = 0
	else:
		J_initial = 0
	
	if i_finl > meshx: i_finl = meshx																	# If coords place shape outside of other end of mesh,
																										# redefine end edges.		
	if j_finl > meshy: j_finl = meshy	
	
	for i in range(i_edge, i_finl, 1):																	# Iterate over the section of the full mesh of interest
		I = i - i_edge + I_initial																		# The equivalent index within the polygon's mesh
		for j in range(j_edge, j_finl, 1):
			J = j - j_edge + J_initial
	    		if mesh[i,j]!=0. and shape[I,J] != 0.:
					CHECK = 1																			# If there's a point in the polygon's mesh that has material,
																										# AND the corresponding point    
	return CHECK																						# has material in it in the main mesh => failure and CHECK = 1


def drop_shape_into_mesh(shape,rr):
	"""
	This function 'drops' a particle into the mesh and has it undergo a random walk
	until it overlaps sufficiently with another particle and is declared 'touching'.
	Only then is a particle fully inserted into the mesh.
	
	"""
	global mesh, meshx, meshy, cppr_max, cppr_min, materials
	
	cell_limit = (np.pi*float(cppr_max)**2.)/100.														# Max number of overlapping cells should scale with area.
																										# area ~= 110 cells for 6cppr
																										# Does NOT need to be integer since values in the mesh are floats, 
																										# and it is their sum that is calculated.
	touching   = 0																						# Initialise the indicators for this function
	passes     = 1																						# here are 'passes' and 'counter' similar to check_coords_full
	counter    = 0																						# But this time there is 'touching' which indicates
																										# contact between particles
	x,y = gen_coord_basic()
	
	Nx, Ny     = meshx, meshy
	Px, Py     = np.shape(shape)
	while touching == 0:	
		#if x > Nx-rr or x < rr: x,y = gen_coord_basic()
		#if y > Ny-rr or y < rr: x,y = gen_coord_basic()
		if x > Nx: x = 0
		if x < 0:  x = Nx
		if y > Ny: y = 0
		if y < 0:  y = Ny																				# If the coord moves out of the mesh, wrap back around.
		i_edge   = x - cppr_max - 1																		# Location of the edge of the polygon's mesh, within the main mesh.
		j_edge   = y - cppr_max - 1
		i_finl   = x + cppr_max + 1																		# The indices refer to the close edge to the origin,
																										# the extra cell should be added on the other side
		j_finl   = y + cppr_max + 1																		# i.e. the side furthest from the origin
		    	
		if i_edge < 0:																					# If the coords have the particle being generated over the mesh boundary
		    I_initial = abs(i_edge)																		# This bit checks for this, and reassigns a negative starting
																										# index to zero
		    i_edge    = 0																				# However, the polygon's mesh will not completely be in the main mesh 
		else:																							# So I_initial defines the cut-off point
		    I_initial = 0																				# If the polygon's mesh does not extend beyond the main mesh,
																										# then I_initial is just 0
		if j_edge < 0:																					# Repeat for the j-coordinate
		    J_initial = abs(j_edge) 
		    j_edge = 0
		else:
		    J_initial = 0
		
		I_final = Px 																					# In a numpy array '-1' indicates the last element.		
		if (i_finl)>Nx:																					# Similarly for the other end of the mesh
			I_final -= abs(Nx-i_finl) 																	# The above only sorts out two sides of the mesh
			i_finl   = Nx
		J_final = Py
		if (j_finl)>Ny:
			J_final -= abs(Ny-j_finl) 
			j_finl   = Ny
		
		
		temp_shape = shape[I_initial:I_final,J_initial:J_final]											# The rectangular array containing the portion of shape 
																										# to be placed into mesh 
		temp_mesh  = mesh[i_edge:i_finl,j_edge:j_finl]													# The equivalent rectangular array within the mesh, in the correct place
		test       = np.minimum(temp_shape,temp_mesh)													# An array containing any points that have material in,
																										# in the same place, in BOTH arrays		

		if (np.sum(test) > cell_limit or np.sum(test) == 0.):  											# If 'test' is > 2, then there are more than 2 cells 
																										# overlapping with other objects at this position
		    rx = random.randint(-cppr_min,cppr_min)
		    ry = random.randint(-cppr_min,cppr_min)
		    y += ry 
		    x += rx
		elif(np.sum(test) <= cell_limit):																# If there are fewer than 2 overlapping cells, but MORE than 0,
																										# place shape here.
			"""
		    #plt.figure(1)
		    #plt.imshow(np.maximum(mesh[i_edge:i_finl,j_edge:j_finl],shape[I_initial:I_final,J_initial:J_final]), cmap='Greys',  interpolation='nearest')
		    #plt.plot(x,y,color='r',marker='o',ms='5')
		    #plt.show()
			"""
			mesh[i_edge:i_finl,j_edge:j_finl] = np.maximum(shape[I_initial:I_final,J_initial:J_final],mesh[i_edge:i_finl,j_edge:j_finl])	
																										# materials contains each cell's material number.
																										# Prior to mat assignment 
																										# it is just the particle number
			touching = 1																				# Assign 'touching' a value of 1 to break loop
			#temp_shape[temp_shape>0.] = 1.																# Change all values in temp_shape to 1., if not already 
			area = np.sum(temp_shape) - np.sum(test)													# Area placed into mesh is the sum of all positive points
																										# in temp_shape - overlap
		else:
		    pass
	return x,y,area
            
def insert_shape_into_mesh(shape,x0,y0):
	"""
	This function inserts the shape (passed as the array 'shape') into the
	mesh at coordinate x0, y0.
	
	The bulk of this function calculates the necessary i and j values in the
	case that the shape is partially in the mesh. Once the necessary i's and 
	j's for both the 'shape' mesh and global mesh are calculated, the shape
	is inserted by taking the maximum of the two array sections.
	The two array sections MUST be the same size and MUST consist of ones 
	and zeros exclusively to work. This is accounted for, but untested as of 
	yet.
	
	NB: If material numbers are to be assigned during 
	placement, then this function will have to be altered, although its current
	set up SHOULD work as this has been considered
	
	shape  : mesh0 array of the shape to be inserted
	x0, y0 : The x and y coords at which the shape is to be placed. (These are the shape's origin point)
	
	The area of the placed shape (it may not be the same as the original
	if the shape is clipped) is returned and the global mesh is altered.
	"""
	global mesh, meshx, meshy, cppr_max, materials
	Px, Py = np.shape(shape)																			# Px and Py are the dimensions of the 'shape' array
	i_edge = x0 - cppr_max - 1																			# Location of the edge of the polygon's mesh, within the main mesh.
	j_edge = y0 - cppr_max - 1																			# This is calculated explicitly in case the mesh has a non-constant size
	i_finl = x0 + cppr_max + 1																			# The indices refer to the closest edge to the origin,
																										# an extra cell is added either side
	j_finl = y0 + cppr_max + 1																			# to ensure the shape is completely encompassed within the box
	
	""" 'i' refers to the main mesh indices whereas 'I' refers to 'shape' indices """
	if i_edge < 0:																						# Condition if the coords have the particle being generated 
																										# over the mesh boundary
		I_initial = abs(i_edge)																			# a negative starting index is reassigned to zero
		i_edge    = 0																					# The polygon's mesh will not completely be in the main mesh 
	else:																								# So I_initial defines the cut-off point
		I_initial = 0																					# If the polygon's mesh does not extend beyond the main mesh,
																										# then I_initial is just 0
	if j_edge < 0:																						# Repeat for the j-coordinate
		J_initial = abs(j_edge) 
		j_edge = 0
	else:
		J_initial = 0
	
	I_final = Px 
	if (i_finl)>meshx:																					# This section deals with the occurrence of the shape overlapping 
																										# with the opposite ends of
		I_final -= abs(meshx-i_finl) 																	# meshes. And works on the same principles as above, 
																										# although the maths is slightly different
		i_finl   = meshx
	J_final = Py
	if (j_finl)>meshy:
		J_final -= abs(meshy-j_finl) 
		j_finl   = meshy
	
	temp_shape = shape[I_initial:I_final,J_initial:J_final]												# record the shape as a temporary array for area calculation
	mesh[i_edge:i_finl,j_edge:j_finl] = np.maximum(shape[I_initial:I_final,J_initial:J_final],mesh[i_edge:i_finl,j_edge:j_finl])
	
														
	""" The shape is inserted by comparing, and taking the maximum, of the two arrays  """
	area = np.sum(temp_shape)																			# Area is sum of all these points
	return area

def place_shape(shape,x0,y0,mat,obj):
	"""
	This function inserts the shape (passed as the array 'shape') into the
	correct materials mesh at coordinate x0, y0.
	
	shape  : mesh0 array of the shape to be inserted
	x0, y0 : The x and y coords at which the shape is to be placed. (These are the shape's origin point)
	mat    : this is the index of the material
	
	This method relies on there being no particles of identical materials being in contact (i.e. no overlapping particles within the materials mesh)

	nothing is returned.
	"""
	global mesh, meshx, meshy, cppr_max, materials
	Px, Py = np.shape(shape)																			# Px and Py are the dimensions of the 'shape' array
	i_edge = x0 - cppr_max - 1																			# Location of the edge of the polygon's mesh, within the main mesh.
	j_edge = y0 - cppr_max - 1																			# This is calculated explicitly in case the mesh has a non-constant size
	i_finl = x0 + cppr_max + 1																			# The indices refer to the closest edge to the origin,
																										# an extra cell is added either side
	j_finl = y0 + cppr_max + 1																			# to ensure the shape is completely encompassed within the box
	
	""" 'i' refers to the main mesh indices whereas 'I' refers to 'shape' indices """
	if i_edge < 0:																						# Condition if the coords have the particle being generated 
																										# over the mesh boundary
		I_initial = abs(i_edge)																			# a negative starting index is reassigned to zero
		i_edge    = 0																					# The polygon's mesh will not completely be in the main mesh 
	else:																								# So I_initial defines the cut-off point
		I_initial = 0																					# If the polygon's mesh does not extend beyond the main mesh, 
																										# then I_initial is just 0
	if j_edge < 0:																						# Repeat for the j-coordinate
		J_initial = abs(j_edge) 
		j_edge = 0
	else:
		J_initial = 0
	
	I_final = Px 
	if (i_finl)>meshx:																					# This section deals with the occurrence of the shape 
																										# overlapping with the opposite ends of
		I_final -= abs(meshx-i_finl) 																	# meshes. And works on the same principles as above, 
																										# although the maths is slightly different
		i_finl   = meshx
	J_final = Py
	if (j_finl)>meshy:
		J_final -= abs(meshy-j_finl) 
		j_finl   = meshy
	temp_shape = shape[I_initial:I_final,J_initial:J_final]												# record the shape as a temporary array for area calculation
	materials[mat-1,i_edge:i_finl,j_edge:j_finl] = np.maximum(shape[I_initial:I_final,J_initial:J_final],materials[mat-1,i_edge:i_finl,j_edge:j_finl])
	objects_temp                                 = np.ceil(np.maximum(shape[I_initial:I_final,J_initial:J_final],objects[mat-1,i_edge:i_finl,j_edge:j_finl]))
	objects_temp[objects_temp>0.]                = obj
	objects[mat-1,i_edge:i_finl,j_edge:j_finl]   = objects_temp 
	
	return
														

def gen_coord_basic():																					
	"""
	This function simply generates a coordinate to place a shape, but does
	NOT check the shape can go there. It simply generates a coord from
	all the cells that have no material. 
	
	Note: this function has the capability to  not allow
	any coords to be generated within one radius of the mesh-edge. It is
	currently commented out, but this can be amended to prevent 'partial particles'

	returns coordinates in x,y form.
	"""
	global mesh,meshy,cppr_max																			# Use the global mesh
	good = 0																							# Here 'good' is the indicator for the loop, when this equals 1 we are 'good' and the loop breaks
	indices = np.where(mesh==0.)																		# Generates 2 arrays containing the indices of all values within mesh which == 0
	indices = np.column_stack(indices)																	# column_stack mushes them together
	x,y = random.choice(indices)																		# This randomly selects one pair of coordinates!
	"""
	while good == 0:
		if y < cppr_max or y > meshy-cppr_max:															# If the coord is within one radius of the edge, => rechoose
			x,y = random.choice(indices)																# This randomly selects one pair of coordinates!
		else:
			good = 1																					# Otherwise the coord is fine and is returned.
	"""
	return x,y


def gen_coord(shape):																					
	"""
	Function generating a coordinate at which the 'shape' can go.
	The function takes a 'shape' as an argument (e.g. a circle or ellipse)
	then it selects a cell at random from the main mesh. However, it only 
	includes cells that are not filled with material initilally in its
	selection. Then the coordinates are checked using the check_coords_full
	function. If this is succesful, the loop breaks and the coords are 
	returned. If not, then the loop generates new coords and repeats until
	either success is achieved or 5000 tries are exceeded. If the counter
	exceeds 5000, the loop is ended and failure noted in the 'passes' variable.

	passes, x and y are all returned at the end of the function. Success has 
	passes = 0, and failure has passes = 1.

	shape : array containing the shape to be checked. This does not have to be regular 
	"""

	global mesh																							# use global parameters (the mesh itself, mesh size)
	check   = 1																							# Initialise the indicator variables
	counter = 0																							# and the counter
	passes  = 0																							# passes should start as 0 and check should start as 1.
	indices = np.where(mesh==0.)																		# Generates an array of all indices in the main mesh of unassigned cells
	indices = np.column_stack(indices)																	# column_stack mushes them together such that random choice 
																										# can choose pairs of coords
	while check == 1:																					# Begin the loop. whilst check = 1, continue to loop 
	    x,y = random.choice(indices)																	# This randomly selects one pair of coordinates!
	    check = check_coords_full(shape,x,y)															# Function to check if the polygon generated will fit 
																										# in the generated coords
	    counter += 1																					# Increment the counter
	    if counter>5000: 																				# If the counter exceeds 5000, this is the break clause
	        check = 0																					# Break the loop and report the failure in the return
	        passes= 1																					# if counter exceeds 1000 -> break loop and assign passes[k] = 1
		break
	return x,y,passes



def mat_basic(mats,N):
	"""
	Function assigning material numbers to particles. This is
	the very simple version which randomly assigns each one
	with no optimisation at all.

	mats : array containing all material numbers to be assigned
	N    : Number of particles

	Returns array 'MAT' of all assigned mat numbers
	"""
	MAT = np.random.choice(mats,N)																		# This function randomly chooses values from 'mats' 
																										# to populate an array of size N
	return MAT

def mat_assignment_1(mats,xc,yc,r):						
	"""
	Function to assign material numbers to each particle
	This function tries to optimise the assignments such
	that as few particles of the same material are in co
	ntact as possible.
	
	mats : array containing all the material numbers to be assigned
	xc   : All the x coords of each particle centre. (array)
	yc   :  "   "  y  "     "   "      "       "   . (array)
	r    :  "   "  radii    "   "      "    . (array)

	Returns array 'MAT' containg a material number for every particle
	"""
	global cppr_max
	N    = np.size(r)																					# No. of particles
	L    = np.amax(r)/cppr_max
	MAT  = np.zeros((N))																				# Array for all material numbers of all particles
	i = 0																								# Counts the number of particles that have been assigned
	while i < N:																						# Loop every particle and assign each one in turn.	
		lowx   = xc[i] - cppr_max*L*6.																	# Create a 'box' around each particle (in turn) that is
																										# 4 diameters by 4 diameters
		higx   = xc[i] + cppr_max*L*6.
		lowy   = yc[i] - cppr_max*L*6.
		higy   = yc[i] + cppr_max*L*6.
		boxmat = MAT[(lowx<xc)*(xc<higx)*(lowy<yc)*(yc<higy)*(MAT!=0.)] 								# Array containing a list of all material numbers within the 'box' 
																										# (disregarding unassigned materials)
		M      = mats[np.in1d(mats,boxmat,invert=True)]													# Array containing all values in 'mats' that are NOT present in the box
		if np.size(M) == 0:																				# If M contains no values, then all the  materials are assigned at 
																										# least once in the box already
			elements = []																				# If the particles surrounding this one have all been assigned, 
																										# check for duplicates!
			for item in boxmat:																			# If there are duplicates already, DON'T assign these if possible
				H = boxmat - item																		# Cycle through each item in the box array. Subtract it from the array
				if np.size(H[H==0])==1:																	# If the new array (H) has more than one zero, 
																										# there was a duplicate, and it should be ignored
					elements.append(item)																# Add all non-duplicates to the list 'elements'
			if np.size(elements) == 0:																	# If 'elements' is empty (only duplicates present) choose at random
				MAT[i] = np.random.choice(mats,1)														# If there is no optimum value it randomly selects one from mats
			else:																						# else randomly select an element from the 'elements' list
				MAT[i] = np.random.choice(elements,1)													# which should only contain materials that appear once
		elif np.size(M) == 1:																			# If M only contains one value (i.e. one material doesn't 
																										# appear in the box)
			MAT[i] = M																					# Use that!
		else:																							# Otherwise there may be multiple elements not present
			MAT[i] = M[0]																				# So select the first in the array for assignment
		i += 1
	return MAT


def mat_assignment(mats,xc,yc,r):						
	"""
	This function has the greatest success and is based on that used in JP Borg's work with CTH.

	Function to assign material numbers to each particle
	This function tries to optimise the assignments such
	that as few particles of the same material are in co
	ntact as possible. It works by creating an array of
	all the particle material numbers within a box, 6 x 6
	radii around each particle coord, as well as the corres
	ponding coords.

	Then they are sorted into the order closest -> furthest.
	Only the first M are considered (in this order). M is
	the number of different materials being used. The
	corresponding array of materials is checked against mats.
	
	If they contain all the same elements, there are no repeats
	and all material numbers are used up => use that of the 
	particle furthest away.

	If they do not, there is at least one repeat, select the
	remaining material number or randomly select one of those
	left, if there are more than one.

	Continue until all the particles are assigned.
	
	mats : array containing all the material numbers to be assigned
	xc   : All the x coords of each particle centre. (array)
	yc   :  "   "  y  "     "   "      "       "   . (array)
	r    :  "   "  radii    "   "      "    . (array)

	Returns array 'MAT' containg a material number for every particle
	"""
	global cppr_max
	N    = np.size(r)																					# No. of particles
	M    = np.size(mats)
	L    = np.amax(r)/cppr_max																			# Length of one cell
	MAT  = np.zeros((N))																				# Array for all material numbers of all particles
	i = 0																								# Counts the number of particles that have been assigned
	while i < N:																						# Loop every particle and assign each one in turn.	
		lowx   = xc[i] - cppr_max*L*6.																	# Create a 'box' around each particle (in turn) 
																										# that is 4 diameters by 4 diameters
		higx   = xc[i] + cppr_max*L*6.
		lowy   = yc[i] - cppr_max*L*6.
		higy   = yc[i] + cppr_max*L*6.
		boxmat = MAT[(lowx<xc)*(xc<higx)*(lowy<yc)*(yc<higy)] 											# Array containing a list of all material numbers within the 'box' 
		boxx   =  xc[(lowx<xc)*(xc<higx)*(lowy<yc)*(yc<higy)]											# Array containing the corresponding xcoords
		boxy   =  yc[(lowx<xc)*(xc<higx)*(lowy<yc)*(yc<higy)]											# and the ycoords
		nn     =  np.size(boxmat)
		D      = np.zeros_like(boxmat)																	# Empty array for the distances
		for ii in range(nn):																			# Loop over each elelment and calc the distances to the current particle
			D[ii] = (boxx[ii] - xc[i])**2. + (boxy[ii] - yc[i])**2.										# No need to root it as we are only interestd in the order
		ind = np.argsort(D)																				# Sort the particles into order of distance from the considered particle
		BXM= boxmat[ind]																				# Sort the materials into the corresponding order
		DU = np.unique(BXM[:M])																			# Only select the M closest particles
		if np.array_equal(DU, mats):																	# If the unique elements in this array equate the array of 
																										# materials then all are taken
			mm     = BXM[M-1]                                                                           
			MAT[i] = mm		  						           											# Set the particle material to be of the one furthest from 
																										# the starting particle
			materials[materials==-1*(i+1)] = mm                                                         # Assign all filled cells 
		else:																							# Else there is a material in mats that is NOT in DU
			indices = np.in1d(mats,DU,invert=True)														# This finds the indices of all elements that only appear in 
																										# mats and not DU
			mm      = np.random.choice(mats[indices],1)
			MAT[i]  = mm                            													# Randomly select one to be the current particle's material number
			materials[materials==-1*(i+1)] = mm
		i += 1																							# Increment i
	return MAT


def part_distance(X,Y,radii,MAT,plot=False):
	"""
	This is a simple function to calculate the average number of contacts between
	particles and plot the contacts graphically, if needed.

	*** NB THIS FUNCTION WILL NOT WORK WELL FOR PR != 0. ***

	The particle centres and radii are taken. The distance between two touching 
	particles should be approximately 1 diameter (centre to centre). So an 
	upper bound on this distance is calculated. This is set as half the radius of
	a particle divided by its cppr_mid, i.e. approximately one cell. 

	Graphically, if plot = True, the full mesh is generated with the appropriate
	materials, and red lines are drawn between the centres of particles in contact.

	X     : Full set of x coords (SI units) for each particle centre
	Y     : Full set of y coords (SI units) for each particle centre
	radii : Full set of radii for each particle (SI)
	MAT   : Array containing the material number for each particle
	plot  : Condition for plotting. True => Will plot a figure, False (default) => will not.

	returns the contact measure A, and saves the plot if wanted
	"""
	global cppr_mid, meshx, meshy,cppr_max
	
	N           = np.size(X)					  														# No. of particles
	mean_radii  = np.mean(radii)				   														# Calculate the mean radii	
	D           = np.zeros((N))																			# D is an array for the distances between particles
	GRIDSPC     = np.amax(radii)/cppr_max																# Physical distance/cell (SI)
	Dtouch      = []																					# The final list of distances for parts in contact
	diameters   = radii*2.																				# All the diameters
	error       = radii/cppr_mid																		# The assumed error on the distances between particles
	upper_bound = diameters + error/2.																	# The upper bound on this distance
	print 'Max, Min, Mean radii = {},{},{}'.format(np.amax(radii),np.amin(radii),mean_radii)
	B           = 0																						# B is the number of contacts that exist between the same materials
	
	if plot == True:																					# If plot == True then produce a figure
	    fig = plt.figure()
	    ax = fig.add_subplot(111,aspect='equal')
	    ax.set_xlim(0,meshy*GRIDSPC)																	# limits are set like this to ensure the final graphic 
																										# matches the mesh in orientation
	    ax.set_ylim(meshx*GRIDSPC,0)
	    for i in range(N):																				# Plot each circle in turn
	        circle = plt.Circle((X[i],Y[i]),radii[i],color='{:1.2f}'.format((MAT[i])*.5/np.amax(MAT)))  # give each one a color based on their material number. 
																										# NB any color = 1. will be WHITE 
	        ax.add_patch(circle)
	
	for i in range(N-1):
		D *= 0.																							 # Initialise D each loop, in case it is full and 
																										 # to ensure nothing carries over
		D -= 1.
		for j in range(N-1):				  															 # find distances between current particle and 
																										 # ALL other particles and store these in D
			dx = X[i] - X[j]
			dy = Y[i] - Y[j]
			distance = np.sqrt(dx**2. + dy**2.)
			D[j] = distance
			if D[j]>0. and D[j]<upper_bound[i]:			   												 # If particles overlap draw a red line between their centres
				if plot == True: 																			
					ax.plot([X[i],X[j]],[Y[i],Y[j]],lw=1.,color='r')
				Dtouch.append(D[j])
				if MAT[i] == MAT[j]: B += 1
	    
	
	Dtouch = np.array(Dtouch)				   															 # Convert to numpy array
	 
	A = float(np.size(Dtouch))/float(N)		   															 # The size of Dtouch/total part number is the mean
	B = float(B)/float(N)																				 # B is the average number of contacts/particle that are 
																										 # between identical materials
	if plot == True: 
		ax.set_title('$A = ${:1.3f}'.format(A))
		plt.savefig('contacts_figure_A-{:1.3f}_B-{:1.3f}.png'.format(A,B),dpi=400)						 # Save the figure
		plt.show()
	
	return A, B

def save_particle_mesh(SHAPENO,X,Y,MATS,n,fname='meso_m.iSALE'):
	"""
	A function that saves the current mesh as a text file that can be read, verbatim into iSALE.
	This compiles the integer indices of each cell, as well as the material in them and the fraction
	of matter present. It saves all this as the filename specified by the user, with the default as 
	meso_m.iSALE

	fname   : The filename to be used for the text file being used
	SHAPENO : The indexes of each shape within mesh_Shps
	X       : The xcoord of the shape centre (in cells)
	Y       : The ycoord of the shape centre (in cells)
	MATS    : The array of each corresponding material number to each particle
	n       : The total number of particles

	returns nothing but saves all the info as a txt file called 'fname' and populates the materials mesh.

	NB This function will remake the mesh.
	"""
	global mesh, mesh_Shps,meshx,meshy,FRAC,OBJID,materials
	XI    = np.zeros((meshx*meshy))	
	YI    = np.zeros((meshx*meshy))
	for k in range(n):
		place_shape(mesh_Shps[SHAPENO[k]],X[k],Y[k],MATS[k],k)

	K = 0
	materials = materials[:,::-1,:]    																	#Reverse array vertically, as it is read into iSALE 
																										# upside down otherwise
	for i in range(meshx):
		for j in range(meshy):
			XI[K] = i
			YI[K] = j
			for mm in range(Ms):
				FRAC[mm,K] = materials[mm,i,j]
				OBJID[mm,K]= objects[mm,i,j]															# each particle number
			K += 1
	FRAC = check_FRACs(FRAC)
	HEAD = '{},{}'.format(K,Ms)
	ALL  = np.column_stack((XI,YI,FRAC.transpose()))                                                	# ,OBJID.transpose())) Only include if particle number needed
	np.savetxt(fname,ALL,header=HEAD,fmt='%5.3f',comments='')
	return

def save_general_mesh(fname='meso_m.iSALE'):
	"""
	A function that saves the current mesh as a text file that can be read, verbatim into iSALE.
	This compiles the integer indices of each cell, as well as the material in them and the fraction
	of matter present. It saves all this as the filename specified by the user, with the default as 
	meso_m.iSALE

	This version of the function works for continuous and solid materials, such as a multiple-plate setup.
	It does not need to remake the mesh as there is no particular matter present.

	fname   : The filename to be used for the text file being used

	returns nothing but saves all the info as a txt file called 'fname' and populates the materials mesh.
	"""
	global mesh, mesh_Shps,meshx,meshy,FRAC,OBJID,materials
	K = 0
	XI    = np.zeros((meshx*meshy))	
	YI    = np.zeros((meshx*meshy))
	materials = materials[:,::-1,:]    																	# Reverse array vertically, 
																										# as it is read into iSALE upside down otherwise
	for i in range(meshx):
		for j in range(meshy):
			XI[K] = i
			YI[K] = j
			for mm in range(Ms):
				FRAC[mm,K] = materials[mm,i,j]
			K += 1
	FRAC = check_FRACs(FRAC)
	HEAD = '{},{}'.format(K,Ms)
	ALL  = np.column_stack((XI,YI,FRAC.transpose()))                                                	# ,OBJID.transpose())) Only include if particle number needed
    
	#fig, ax = plt.subplots()
	#cax = ax.imshow(materials[0,:,:],cmap='Greys',interpolation='nearest',vmin=0,vmax=1)
	#cbar = fig.colorbar(cax, orientation='horizontal')
	#plt.show()
	np.savetxt(fname,ALL,header=HEAD,fmt='%5.3f',comments='')
	return

def populate_from_bmp(A):
	global materials,Ms,FRAC
	fname = 'meso_m.iSALE'
	A = A[:,::-1]
	ny, nx = np.shape(A)
	generate_mesh(nx,ny,1)
	XI    = np.zeros((nx*ny))	
	YI    = np.zeros((nx*ny))
	K = 0
	for i in range(nx):
		for j in range(ny):
			XI[K] = i
			YI[K] = j
			FRAC[0,K] = 1. - A[j,i]
			K += 1
		
	HEAD = '{},{}'.format(K,Ms)
	ALL  = np.column_stack((XI,YI,FRAC.transpose()))                                                # ,OBJID.transpose())) Only include if particle number needed
    
	fig, ax = plt.subplots()
	cax = ax.imshow(1.-A,cmap='Greys',interpolation='nearest',vmin=0,vmax=1)
	cbar = fig.colorbar(cax, orientation='horizontal')
	plt.show()
	np.savetxt(fname,ALL,header=HEAD,fmt='%5.3f',comments='')
	return

def check_FRACs(FRAC):
	"""
	This function checks all the volume fractions in each cell and deals with any occurrences where they add to more than one
	by scaling down ALL fractions in that cell, such that it is only 100% full.

	FRAC : Array containing the full fractions in each cell of each material
	"""

	global Ms,meshx,meshy

	for i in range(meshx*meshy):
		SUM = np.sum(FRAC[:,i])
		if SUM > 1.:
			FRAC[:,i] /= SUM
		else:
			pass
	return FRAC

def fill_plate(y1,y2,mat,invert=False):
	"""
	This function creates a 'plate' structure across the mesh, filled with the material of your choice. Similar to PLATE in iSALE
	It fills all cells between y1 and y2.
	"""
	global meshx,meshy,materials,mesh
	assert y2>y1, 'ERROR: 2nd y value is less than the first, the function accepts them in ascending order' 
	for j in range(meshx):
		if j <= y2 and j >= y1:
			materials[:,j,:]     = 0. 																	# This ensures that plates override in the order they are placed.
			materials[mat-1,j,:] = 1.
			mesh[j,:]            = 1.
	return

def fill_above_line(r1,r2,mat,invert=False,mixed=False):
	"""
	This function takes two points and fills all cells above the line drawn between them.
	If invert is True then it will fill all cells below the line. if mixed is True, then it
	will also partially fill cells where necessary.
	
	x1, y1 : x and y coords of point 1
	x2, y2 : x and y coords of point 2
	invert : logical to indicate if inversion necessary
	mixed  : logical to indicate if partially filled cells necessary
	"""
	
	global meshx,meshy,materials,mesh
	AREA = 0.
	if mixed == True:																					# If there are to be mixed cells, then set MIX to 1.
		MIX = 1.
	else:
		MIX = 0.
	x2 = r2[0]
	x1 = r1[0]
	y2 = r2[1]
	y1 = r1[1]
	assert x1!=x2,'ERROR: x2 is equal to x1, this is a vertical line, use fill_left or fill_right'
	M = (y2-y1)/(x2-x1)																					# Calculate the equation of the line between x1,y1 and x2,y2
	C = y1 - M*x1
	for i in range(meshx):
		for j in range(meshy):
			xc = 0.5*(i + (i+1))																	
			yc = 0.5*(j + (j+1))						
			A = (yc - C)/M		
			#A = M*xc + C
			if (j+.5) < y2 and (j+.5) > y1:
				if invert == False:																		# If the fill is not inverted do below
					if (xc-A) < -1.*np.sqrt(2.)*MIX:													# If the cell centre falls under this line then fill fully
				   		mesh[i,j]            = 1.0														# If MIX=0. then this will do it for all cells
						materials[mat-1,i,j] = 1.0
				   		AREA += 1																		# If MIX=1. then the ones right next to the line will be mixed
					elif abs(xc-A) <= np.sqrt(2.) and mixed == True:									# If mixed cells wanted and the cell is within the necessary range
						xx = np.linspace(i,i+1,11)														# Split into 10x10 grid and fill, depending on these values.
						yy = np.linspace(j,j+1,11)						
						for I in range(10):
							for J in range(10):							
								xxc = 0.5*(xx[I] + xx[J+1])		
								yyc = 0.5*(yy[J] + yy[J+1])
								#A = M*xxc + C							
								A = (yyc-C)/M
								if (xxc-A) < 0.:				
									mesh[i,j]            += (.1**2.)				
									materials[mat-1,i,j] += (.1**2.)
									AREA                 += (.1**2.)			
				elif invert == True:
					if (xc-A) > np.sqrt(2.)*MIX:				
				   		mesh[i,j]            = 1.0														# If MIX=0. then this will do it for all cells
						materials[mat-1,i,j] = 1.0
					elif abs(xc-A) <= np.sqrt(2) and mixed == True:					
						xx = np.linspace(i,i+1,11)						
						yy = np.linspace(j,j+1,11)						
						for I in range(10):
							for J in range(10):							
								xxc = 0.5*(xx[I] + xx[J+1])		
								yyc = 0.5*(yy[J] + yy[J+1])
								#A = M*xxc + C							
								A = (yyc-C)/M
								if (xxc-A) > 0.:				
									mesh[i,j]            += (.1**2.)				
									materials[mat-1,i,j] += (.1**2.)
									AREA                 += (.1**2.)			
			else:
				pass
	return

def fill_arbitrary_shape(X,Y,mat):						
	"""
	Function to fill an arbitrary shape in the mesh based on arrays of vertices.
	This version does NOT partially fill cells.
	NB for this to work the coordinates of the vertices MUST be relative to the centre of the object.

	X, Y      : Vertices in cells
	"""
	global mesh, materials,meshx,meshy																	# Only the angles used are now randomly selected.
	N      = np.size(X)
	R      = np.zeros((2,N))																			# Array for the coords of the vertices
	x0,y0 = find_centroid(X,Y)
	R[0,:] = X - x0																						# Each vertex will also be successive, such that drawing a line between
	R[1,:] = Y - y0																						# each one in order, will result in no crossed lines.
	qx = 0.																								# Make the reference point (q) zero, i.e. the centre of the shape
	qy = 0.																								# All dimensions are in reference to the central coordinates.
	for j in range(meshy):																				# Iterate through all the x- and y-coords
		for i in range(meshx):																			
			xc = 0.5*(i+(i+1))-x0																		# Convert current coord to position relative to (x0,y0) 
			yc = 0.5*(j+(j+1))-y0																		# Everything is now in indices, with (x0,y0)
			sx = xc - qx																				# s = vector difference between current coord and ref coord
			sy = yc - qy		
			intersection = 0																			# Initialise no. intersections as 0
			for l in range(N-1):																		# cycle through each edge, bar the last
				rx = R[0,l+1] - R[0,l]																	# Calculate vector of each edge (r), 
																										# i.e. the line between the lth vertex and the l+1th vertex
				ry = R[1,l+1] - R[1,l]
				RxS = (rx*sy-ry*sx)																		# Vector product of r and s (with z = 0), 
																										# technically produces only a z-component
				if RxS!=0.:																				# If r x s  = 0 then lines are parallel
					t = ((qx-R[0,l])*sy - (qy-R[1,l])*sx)/RxS
					u = ((qx-R[0,l])*ry - (qy-R[1,l])*rx)/RxS	
					if t<=1. and t>=0. and u<=1. and u>=0.:
						intersection = intersection + 1
			rx = R[0,0] - R[0,N-1]																		# Do the last edge. 
																										# Done separately to avoid needing a circular 'for' loop
			ry = R[1,0] - R[1,N-1]
			if (rx*sy-ry*sy)!=0.:
				RxS = (rx*sy-ry*sx)
				t = ((qx-R[0,N-1])*sy - (qy-R[1,N-1])*sx)/RxS
				u = ((qx-R[0,N-1])*ry - (qy-R[1,N-1])*rx)/RxS
				if t<=1. and t>=0. and u<=1. and u>=0.:
					intersection = intersection + 1
			if (intersection%2==0.):																	# If number of intersections is divisible by 2 (or just zero) 
																										#-> fill that cell!
				mesh[i,j]            = 1.0
				materials[mat-1,i,j] = 1.0

	return

def fill_arbitrary_shape_p(X,Y,mat):						
	"""
	Function to fill an arbitrary shape in the mesh based on arrays of vertices.
	This version DOES partially fill cells.
	"""
	global mesh, materials,meshx,meshy												# Only the angles used are now randomly selected.
	x0,y0  = find_centroid(X,Y)
	N      = np.size(X)
	X      = np.append(X,X[0])
	Y      = np.append(Y,Y[0])
	R      = np.zeros((2,N+1))																# Array for the coords of the vertices
	R[0,:] = X - x0															# Each vertex will also be successive, such that drawing a line between
	R[1,:] = Y - y0														# each one in order, will result in no crossed lines.
																			# Convert into cartesian coords and store in R
	qx = 0.																		# Make the reference point (q) zero, i.e. the centre of the shape
	qy = 0.																		# All dimensions are in reference to the central coordinates.
	for j in range(meshy):																# Iterate through all the x- and y-coords
		print j
		for i in range(meshx):															# N-1 because we want the centres of each cell, and there are only N-1 of these!
			xc = 0.5*(i + i+1) - x0		
			yc = 0.5*(j + j+1) - y0
			sx = xc - qx															# s = vector difference between current coord and ref coord
			sy = yc - qy		
			intersection = 0																# Initialise no. intersections as 0
			for l in range(N):															# cycle through each edge, bar the last
				rx = R[0,l+1] - R[0,l]										# Calculate vector of each edge (r), i.e. the line between the lth vertex and the l+1th vertex
				ry = R[1,l+1] - R[1,l]
				RxS = (rx*sy-ry*sx)															# Vector product of r and s (with z = 0), technically produces only a z-component
				if RxS!=0.:																	# If r x s  = 0 then lines are parallel
					t = ((qx-R[0,l])*sy - (qy-R[1,l])*sx)/RxS
					u = ((qx-R[0,l])*ry - (qy-R[1,l])*rx)/RxS	
					if t<=1. and t>=0. and u<=1. and u>=0.:
						intersection = intersection + 1
			if (intersection%2==0.):							# If number of intersections is divisible by 2 (or just zero) -> fill that cell!
				
				xx = np.linspace(i,i+1,11)						
				yy = np.linspace(j,j+1,11)						
				for I in range(10):
					for J in range(10):							
						xc = 0.5*(xx[I] + xx[I+1]) - x0		
						yc = 0.5*(yy[J] + yy[J+1]) - y0
						sx = xc - qx															# s = vector difference between current coord and ref coord
						sy = yc - qy		
						intersection = 0																# Initialise no. intersections as 0
						for l in range(N):															# cycle through each edge, bar the last
							rx = R[0,l+1] - R[0,l]										# Calculate vector of each edge (r), i.e. the line between the lth vertex and the l+1th vertex
							ry = R[1,l+1] - R[1,l]
							RxS = (rx*sy-ry*sx)															# Vector product of r and s (with z = 0), technically produces only a z-component
							if RxS!=0.:																	# If r x s  = 0 then lines are parallel
								t = ((qx-R[0,l])*sy - (qy-R[1,l])*sx)/RxS
								u = ((qx-R[0,l])*ry - (qy-R[1,l])*rx)/RxS	
								if t<=1. and t>=0. and u<=1. and u>=0.:
									intersection = intersection + 1
						if (intersection%2==0.):							# If number of intersections is divisible by 2 (or just zero) -> fill that cell!
							mesh[i,j]            += (.1**2.)				
							materials[mat-1,i,j] += (.1**2.)

	return

def find_centroid(x,y):
	"""
	Simple function to return the centroid coordinate of an arbitrary, non-intersecting polygon.
	with n vertices. see https://en.wikipedia.org/wiki/Centroid#Locating_the_centroid for further
	information.
	"""
	x = np.append(x,x[0])
	y = np.append(y,y[0])

	n = np.size(x)
	A  = 0
	Cx = 0
	Cy = 0
	for i in range(n-1):
		A += (x[i]*y[i+1]-x[i+1]*y[i])*.5
	for j in range(n-1):
		Cx += (x[j]+x[j+1])*(x[j]*y[j+1]-x[j+1]*y[j])
		Cy += (y[j]+y[j+1])*(x[j]*y[j+1]-x[j+1]*y[j])
	
	Cx /= (6.*A)
	Cy /= (6.*A)
	return Cx,Cy
