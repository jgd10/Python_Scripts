"""  This is a module in-progress for use generating meso-setups to be put into iSALE """
""" There is much to work on and more is being added all the time                     """
""" - 19/01/16 - JGD                                                                  """

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.path   as mpath
from matplotlib.widgets import Slider, Button, RadioButtons
import Tkinter
import tkSimpleDialog as tksd
from mpl_toolkits.axes_grid1 import make_axes_locatable

def interactive_setup():
    fig, ax = plt.subplots(figsize=(15,10))
    plt.subplots_adjust(left=0.4, bottom=0.2)
    plt.axis([0, 1, -10, 10])
    root = Tkinter.Tk()
    root.withdraw()
    
    axcolor = 'lightgoldenrodyellow'
    axcppr = plt.axes([0.1, 0.95, 0.15, 0.02], facecolor=axcolor)
    axmatn = plt.axes([0.1, 0.90, 0.15, 0.02], facecolor=axcolor)
    axpr__ = plt.axes([0.1, 0.85, 0.15, 0.02], facecolor=axcolor)
    axXcls = plt.axes([0.1, 0.80, 0.15, 0.02], facecolor=axcolor)
    axYcls = plt.axes([0.1, 0.75, 0.15, 0.02], facecolor=axcolor)
    axvfrc = plt.axes([0.1, 0.60, 0.15, 0.02], facecolor=axcolor)
    
    scppr_ = Slider(axcppr, 'C.P.P.R.', 5, 50, valinit=15,valfmt='%1d')
    smatno = Slider(axmatn, 'No. of Materials', 1, 9, valinit=5,valfmt='%1d')
    spr___ = Slider(axpr__, 'Grain Size Range', 0, 1, valinit=0.2)
    svfrac = Slider(axvfrc, 'Target \n Volume Fraction', 0., 1., valinit=0.5)
    sXcell = Slider(axXcls, 'X cells', 10, 1000, valinit=200,valfmt='%03d')
    sYcell = Slider(axYcls, 'Y cells', 10, 1000, valinit=200,valfmt='%03d')
    
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button1 = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    
    generateax = plt.axes([0.13, 0.675, 0.1, 0.06])
    button2 = Button(generateax, 'Generate \n New Mesh', color=axcolor, hovercolor='0.975')
    
    addboxax = plt.axes([0.13, 0.5, 0.1, 0.04])
    button3 = Button(addboxax, 'Draw Box', color=axcolor, hovercolor='0.975')
    
    saveax = plt.axes([0.65, 0.025, 0.1, 0.04])
    button4 = Button(saveax, 'Save', color='pink', hovercolor='0.975')
    
    
    def draw_box(event):
        x1 = tksd.askfloat('Bottom left x-coord [m]','Value:',initialvalue=0.)
        y1 = tksd.askfloat('Bottom left y-coord [m]','Value:',initialvalue=0.)
        x2 = tksd.askfloat('Top right x-coord [m]','Value:',initialvalue=1.)
        y2 = tksd.askfloat('Top right y-coord [m]','Value:',initialvalue=1.)
        mat= tksd.askinteger('Material No.','Value:',initialvalue=1)
        fill_rectangle(x1,y1,x2,y2,mat)
        update()
    button3.on_clicked(draw_box)

#    def make_parts



    def reset(event):
        svfrac.reset()
        sXcell.reset()
        sYcell.reset()
        smatno.reset()
        spr___.reset()
        scppr_.reset()
    button1.on_clicked(reset)

    def generate(event):
        X = int(sXcell.val)
        Y = int(sYcell.val)
        M = int(smatno.val)
        C = int(scppr_.val)
        P = spr___.val
        G = tksd.askfloat('GRIDSPC [m/cell]','Value:',initialvalue=2.e-6)
        generate_mesh(X,Y,M,C,P,G)
        update()
    button2.on_clicked(generate)
    def update():
        ax.cla()
        ax.imshow(mesh,cmap='binary',vmin=0,vmax=1,interpolation='nearest',origin='lower')
        for KK in range(Ms):
            matter = np.copy(materials[KK,:,:])*(KK+1)
            matter = np.ma.masked_where(matter==0.,matter)
            ax.imshow(matter, cmap='viridis',vmin=0,vmax=Ms,interpolation='nearest',origin='lower')        
        ax.set_xlabel('metres')
        ax.set_ylabel('metres')
        ax.axvline(0,color='k',linestyle='--')
        ax.axvline(meshx,color='k',linestyle='--')
        ax.axhline(0,color='k',linestyle='--')
        ax.axhline(meshy,color='k',linestyle='--')
        ax.set_xticks([0,meshx/2,meshx])
        ax.set_xticklabels([0,meshx*GS/2.,meshx*GS])
        ax.set_yticks([0,meshy/2,meshy])
        ax.set_yticklabels([0,meshy*GS/2.,meshy*GS])
        ax.axis('equal')

    rax = plt.axes([0.27, 0.92, 0.1, 0.05], facecolor=axcolor)
    radio = RadioButtons(rax, ('circles', 'ellipses'), active=0)
    
    def generate_shapes(label):
        rot = tksd.askstring('Randomly rotate shapes?','[Y/N]',initialvalue='N')
        generate_particle_list(label,rot)
    radio.on_clicked(generate_shapes)

    def save_mesh(event):
        f = tksd.askstring('Save as:','filename',initialvalue='meso_m.iSALE')
        save_general_mesh(fname=f)
    button4.on_clicked(save_mesh)



    plt.show()

def clear_mesh():
    global mesh, materials
    mesh      *= 0
    materials *= 0
    return

def generate_particle_list(shape,rotation=False):
    cppr_range = cppr_max-cppr_min
    if shape == 'ellipses': e = tksd.askfloat('Eccentricity of Ellipses','Value:',initialvalue=.5)
    for i in range(N):
        r = cppr_min + i*cppr_range/(N-1)
        if rotation == 'Y':   
            rot = random.random()*np.pi
        else:
            rot = 0.
        if shape == 'circles':    mesh_Shps[i] = gen_circle(r)
        elif shape == 'ellipses': mesh_Shps[i] = gen_ellipse(r,rot,e)
        Shps_Area[i] = np.sum(mesh_Shps[i])
    return

def place_grains(VF,R1,R2):
    vfrc = 0.
    mesh_tmp = np.zeros_like(materials[0])
    box = mesh_tmp[(XX>R1[0])*(XX<R2[0])*(YY<R2[1])*(YY>R1[1])]
    nbx,nby = np.shape(box)
    try:
        while vfrc < VF:
            for mm in range(Ms):
                mesh_tmp = np.maxmimum(mesh_tmp,materials[mm])
            I = random.randint(0,N-1)
            shape   = mesh_Shps[I]
            indices = np.where(box==0.)                                                            # Generates an array of all indices in the main mesh of unassigned cells
            indices = np.column_stack(indices)                                                     # column_stack mushes them together such that random choice 
            y,x     = random.choice(indices) 
            imin    = x-Ns/2
            jmin    = y-Ns/2
            imax    = x+Ns/2
            jmax    = y+Ns/2
            sub_box = box[imin:imax,jmin:jmax]
            imin   += Ns/2 - x
            imax   += Ns/2 - x
            jmin   += Ns/2 - y
            jmax   += Ns/2 - y
            ind_shp = np.nonzero(shape[imin:imax,jmin:jmax])
            T       = np.sum(sub_box[ind_shp])
            if T<=2.:
                place_shape(mesh_Shps[I],x,y,mm)
                A = Shps_Area[I] - T
                vfrc += A
                update()
            else:
                pass
    except KeyboardInterrupt:
        pass


def generate_mesh(X=500,Y=500,mat_no=5,CPPR=10,pr=0.,GridSpc=2.e-6,NS=None,NP=20):
    """
    This function generates the global mesh that all particles will be inserted into.
    Initially it reads in several parameters and renames them within the module. Then 
    several arrays are initialised, both of the main mesh and the 'shape' meshes (mesh0)

    X        : The x-length of the mesh, in cells
    Y        : The y-length of the mesh, in cells
    CPPR     : The chosen value of cells per particle radius
    pr       : The particle size range (a fraction). The range of particle sizes that will be produced
    GridSpc  : The physical length of one cell. [m]
    NS       : Length [cells] of one side of the 'mini mesh' used to hold the generated shapes/grains 

    Nothing is returned but many variables become global.

    NB. Recently I have updated 'N' to now be the number of different particles that can be generated
    and NOT N**2. 19-01-16
    """
    global meshx,meshy,cppr_mid,PR,cppr_min,cppr_max,mesh,xh,yh,Ns,N,Nps
    global Shps_Area,mesh0,mesh_Shps,materials,Ms,mats,objects,FRAC,OBJID,GS
    global trmesh,XX,YY,x_c,y_c,VX_,VY_,part_no,alpha

    # Length of the side of one cell in m
    GS        = GridSpc
    # Number of materials in the simulation
    Ms        = mat_no                     
    # List of the material numbers (in integers!)
    mats      = np.arange(Ms,dtype=int)+1
    # No. cells in the transverse direction
    meshx     = X
    # No. cells in the longitudinal direction
    meshy     = Y
    # Average number of cells per particle radius (when linear distribution)
    cppr_mid  = CPPR
    # Particle size range
    PR        = pr
    
    # if cppr_max and min predefined
    if np.size(PR)>1:
        # Minimum particle size
        cppr_min  = int(PR[0]*cppr_mid)
        # Maximum particle size
        cppr_max  = int(PR[1]*cppr_mid)
    # else if just PR defined
    else:
        # Maximum particle size
        cppr_max  = int(cppr_mid*(1.+PR))
        # Minimum particle size
        cppr_min  = int(cppr_mid*(1.-PR))

    # Volume fraction field 
    mesh      = np.zeros((meshy,meshx))
    # Tracer mesh
    trmesh    = np.zeros((meshy,meshx))
    # Velocity field (Transverse)
    VX_       = np.zeros((meshy,meshx))
    # Velocity field (Longitudinal)
    VY_       = np.zeros((meshy,meshx))
    # Material fields
    materials = np.zeros((Ms,meshy,meshx))                                                     
    # Object field
    objects   = np.zeros((meshy,meshx))                                                        
    # Particle number field
    part_no   = np.zeros((meshy,meshx))                                                        
    # Disension field (default & min value: 1)
    alpha   = np.ones((meshy,meshx))                                                        
    # Transverse positions of the nodes
    xh        = np.arange(meshx+1)*GS                                                          
    # Longitudinal positions of the nodes
    yh        = np.arange(meshy+1)*GS

    # Transverse positions of the cells (centres)
    x_c       = (np.arange(meshx)+.5)*GS
    # Longitudinal positions of the cells (centres)
    y_c       = (np.arange(meshy)+.5)*GS
    # Coordinate fields
    XX,YY     = np.meshgrid(x_c,y_c)
    if NS is None:
        Ns        = int(2*(cppr_max)+2)                                                                    # Dimensions of the mini-mesh for individual shapes. MUST BE EVEN.
    else:
        Ns = NS
    N         = NP                                                                                         # N is the number of different particles that can be generated  
    Shps_Area = np.zeros((N))
    mesh0     = np.zeros((Ns,Ns))                                                                          # Generate mesh that is square and slightly larger than the max 
                                                                                                           # particle size
                                                                                                           # This will be the base for any shapes that are generated
    mesh_Shps = np.zeros((N,Ns,Ns))                                                                        # Generate an array of meshes of this size. of size 
                                                                                                           # N (Ns x Ns x N)
    FRAC      = np.zeros((Ms,meshx*meshy))                                                                 # An array for storing the fractions of material 
    OBJID     = np.zeros((Ms,meshx*meshy))                                                                 # An array for storing the fractions of material 

def max_porosity_variation(partitions=2):
    global mesh
    """
    Function that finds the largest varition in porosity across the entire mesh. 
    This will give incorrect answers when the mesh is not purely granular.
    returns the maximum difference between two partitions of the same orientation.
    """
    #mesh = np.sum(materials,axis=0)
    #mesh[mesh>1.] = 1.
    pores = 1.-mesh

    # create arrays to store vert and horiz partition porosities
    pores_T = np.ones(partitions)*-9999.
    pores_L = np.ones(partitions)*-9999.

    # divide the mesh into intervals divT and divL wide
    divT = int(meshx/float(partitions))
    divL = int(meshy/float(partitions))
    for p in range(partitions):
        #print mesh[p*divT:(p+1)*divT,:]
        pores_T[p] = np.mean(mesh[p*divT:(p+1)*divT,:])
        pores_L[p] = np.mean(mesh[:,p*divL:(p+1)*divL])
    
    # find the maximum difference between partitions
    maxdiff_T = np.amax(pores_T) - np.amin(pores_T)
    maxdiff_L = np.amax(pores_L) - np.amin(pores_L)
    
    # Find the largest of these twp
    maxdiff = max(maxdiff_T,maxdiff_L)
    return maxdiff
        

def setup_mesh_from_mesom(fpath='meso_m.iSALE',GSInput=2.e-6):
    """
    Sets up a mesh from a meso_m.iSALE file. Once complete, the new mesh can be edited further. 
    NB the cellsize is a required input as this is not stored in meso_m.iSALE!
    """
    global materials,VX_,VY_
    with open(fpath, 'r') as f:
            first_line = f.readline().strip()
    topline = np.fromstring(first_line,sep=',')
    mat_no = int(np.amin(topline))
    N      = int(np.amax(topline))
    I     = np.genfromtxt(fpath,usecols=(0),skip_header=1).astype(int)
    J     = np.genfromtxt(fpath,usecols=(1),skip_header=1).astype(int)
    VX    = np.genfromtxt(fpath,usecols=(2),skip_header=1)
    VY    = np.genfromtxt(fpath,usecols=(3),skip_header=1)
    FRAC = np.zeros((mat_no,N))
    for col in range(mat_no):
        FRAC[col] = np.genfromtxt(fpath,usecols=(col+4),skip_header=1)

    meshx = np.amax(I) + 1
    meshy = np.amax(J) + 1
    generate_mesh(meshx, meshy, mat_no = mat_no, GridSpc = GSInput)

    for i in range(N):
        for m in range(mat_no):
            materials[m,I[i],J[i]] = FRAC[m,i]
        VX_[I[i],J[i]] = VX[i]
        VY_[I[i],J[i]] = VY[i]

    return


def unit_cell(LX=None,LY=None):
    global meshx,meshy,cppr_mid,Ms,trmesh
    if LX == None: LX = int(meshx/10) 
    if LY == None: LY = int(meshy/10) 
    UC = np.zeros((Ms,LY,LX))
    return UC

def copypasteUC(UC,UCX,UCY,RAD,MATS):
    global meshx,meshy,materials,trmesh
    LY,LX = np.shape(UC[0,:,:])
    i  = 0
    ii = LX
    xcoords = np.copy(UCX)
    ycoords = np.copy(UCY)
    mats    = np.copy(MATS)
    rad     = np.copy(RAD)
    while i < meshx:
        j = 0
        jj = LY
        I = i + LX
        if I > meshx: 
            ii= abs(i - meshx)
            I = meshx
        while j < meshy:
            J = j + LY
            if J > meshy: 
                jj = abs(j - meshy)
                J  = meshy
            materials[:,j:J,i:I] = UC[:,:jj,:ii]
            ycoords = np.append(ycoords,UCY+i)
            xcoords = np.append(xcoords,UCX+j)
            mats    = np.append(mats,MATS)
            rad     = np.append(rad,RAD)
            j += LY
        i += LX
    R = np.mean(rad)
    xc       = xcoords[(xcoords<meshx+R)*(ycoords<meshy+R)]
    yc       = ycoords[(xcoords<meshx+R)*(ycoords<meshy+R)]
    rads     =     rad[(xcoords<meshx+R)*(ycoords<meshy+R)]
    mat      =    mats[(xcoords<meshx+R)*(ycoords<meshy+R)]
    coords   = np.column_stack((mat,xc,yc,rads))

    coords = np.vstack({tuple(row) for row in coords})

    #c,indices = np.unique(coords,return_index = True)
    #print xcoords
    #coords = coords[indices]
    return coords[:,0],coords[:,1],coords[:,2],coords[:,3]

def convert_to_impactor(R1,R2,mm):
    """
    In iSALE velocities are determined by the host object, and until velocity
    assignment is coded into pySALESetup this method will be the main way of 
    making the target and impactor out of the same meso_m.iSALE file.

    A material can not have more than one host object; therefore all material numbers
    in the impactor must be different to those in the target. This function
    takes the section to be the impactor and alters all material numbers within it
    such that they are distinct from the target.
    """
    assert R1[0]<=R2[0] and R1[1]<=R2[1], 'ERROR: Second coordinates must be greater than the first'

    materials[:][(XX>=R1[0])*(XX<=R2[0])*(YY>=R1[1])*(YY<=R2[1])] = 0.
#    print np.shape(materials[:mm]),np.shape(materials[mm:]),np.shape(materials)
#    impactor = np.roll(impactor,shift=1,axis=0)
#    print np.amax(materials[:mm]),np.amax(materials[mm:])

    return

def gen_circle(r_):
    """
    This function generates a circle within the base mesh0. It very simply converts
    each point to a radial coordinate from the origin (the centre of the shape.
    Then assesses if the radius is less than that of the circle in question. If it 
    is, the cell is filled.
    
    r_ : radius of the circle, origin is assumed to be the centre of the mesh0
    
    mesh0 and an AREA are returned
    """
    global mesh0, Ns                                                                                    # mesh0 is Ns x Ns in size
    mesh0[:] = 0.                                                                                        # Ensure that mesh0 is all zeros (as it should be before this is called)
    x0 = float(Ns)/2.                                                                                    # Ns = 2*cppr_max + 2, so half well be cppr_max + 1
    y0 = float(Ns)/2.                                                                                    # Define x0, y0 to be the centre of the mesh
    for j in range(Ns):                                                                                    # Iterate through all the x- and y-coords
        for i in range(Ns):                        
            xc = 0.5*(i + (i+1)) - x0                                                                    # Convert current coord to position relative to (x0,y0) 
            yc = 0.5*(j + (j+1)) - y0                                                                    # Everything is now in cartesian coords relative to the mesh centre
            
            r = (xc/r_)**2. + (yc/r_)**2.                                                                # Find the radial distance from current coord to (x0,y0), normalised by r_
            if r<=1:                                                                                    # If this is less than 1 (i.e. r_) then the coord is in the circle => fill
                mesh0[j,i] = 1.0                                                                        # Fill cell
    return mesh0

def polygon_area(X,Y):
    N = np.size(X)
    A = 0
    for i in range(1,N):
        A += (X[i-1]*Y[i]-X[i]*Y[i-1])*.5
    return abs(A)

def gen_shape_fromtxt(fname='shape.txt'):
    """
    This function generates a mesh0 from a text file of ones and 0s.
    shape.txt MUST be of the same shape as mesh0
    """
    global mesh0, Ns                                                                                    # mesh0 is Ns x Ns in size
    M = np.genfromtxt(fname,comments='#',delimiter=',')
    assert np.shape(M) == np.shape(mesh0), 'ERROR: the shapes mesh is not the same as the mesh0'
    mesh0 = M
    return mesh0

def gen_shape_fromvertices(fname='shape.txt',mixed=False,areascale=1.,rot=0.,min_res=5):
    """
    This function generates a mesh0 from a text file containing a list of its vertices
    in normalised coordinates over a square grid of dimensions 1 x 1. Centre = (0,0)
    coordinates must be of the form:
    j   i
    x   x
    x   x
    x   x
    .   .
    .   .
    .   .
    and the last coordinate MUST be identical to the first
    --------------------------------------------------------------------------
    |kwargs    |  Meaning                                                    | 
    --------------------------------------------------------------------------
    |mixed     |  partially filled cells on or off                           |
    |rot       |  rotation of the grain (radians)                            |
    |areascale |  Fraction between 0 and 1, indicates how to scale the grain |
    |min_res   |  Minimum resolution allowed for a grain                     |
    --------------------------------------------------------------------------

    """
    global mesh0, Ns,cppr_max                                            # mesh0 is Ns x Ns in size
    mesh0 *= 0.
    theta = 2.*np.pi*rot 
    ct    = np.cos(theta)
    st    = np.sin(theta)
    J_    = np.genfromtxt(fname,comments='#',usecols=0,delimiter=',')
    I_    = np.genfromtxt(fname,comments='#',usecols=1,delimiter=',')
    MAXI  = np.amax(abs(I_))
    MAXJ  = np.amax(abs(J_))
    MAX   = max(MAXI,MAXJ)
    J_   /= MAX 
    I_   /= MAX 
    
    if J_[0] != J_[-1]:
        J_ = np.append(J_,J_[0])
        I_ = np.append(I_,I_[0])

    A_shape = polygon_area(I_,J_)
    lengthscale  = np.sqrt(areascale)                                                       # Shape area is scaled and equivalent radius found
    """
    areascale is ratio of areas, thus lengthscale is the square root of this
    """
                                                                                                            # This is used as the lengthscale
    #print areascale,lengthscale
    J_   *= (Ns/2.)*lengthscale
    I_   *= (Ns/2.)*lengthscale
    J     = J_*ct - I_*st
    I     = J_*st + I_*ct
    
    radii     = np.sqrt(I**2+J**2)
    min_radii = np.amin(radii)
    max_radii = int(np.amax(radii))
    min_size  = min(max_radii+1,Ns/2.)               
    min_size *= 2
    # Possible to have a max radius larger than half the box width
    # But not to have a sub-mesh that big!
    mesh_     = np.zeros((min_size,min_size))
    Failure   = False
    if min_radii < min_res: Failure = True
    n  = np.size(J)-1
    qx = 0.                                                                                 
    qy = 0.                                                                                 
    y0 = float(max_radii*2+2)/2.                                                                                    # Define x0, y0 to be the centre of the mesh
    x0 = y0
    if Failure != True:
        I += x0
        J += y0
        path = mpath.Path(np.column_stack((I,J)))
        for i in range(Ns):
            for j in range(Ns):
                in_shape = path.contains_point([i+.5,j+.5])
                if in_shape and mixed == False: mesh_[i,j] = 1.
                elif in_shape and mixed == True:
                    for ii in np.arange(i,i+1,.1):
                        for jj in np.arange(j,j+1,.1):
                            in_shape2 = path.contains_point([ii+.05,jj+.05])
                            if in_shape2: mesh_[i,j] += .01

    ind = (Ns - min_size)/2
    #print ind,Ns,min_size
    if ind != 0.:
        mesh0[ind:-ind,ind:-ind] = mesh_
    else:
        mesh0 = mesh_
    return mesh0,Failure

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
    assert e_ >= 0. and e_ < 1., 'error: invalid value of eccentricity. e must be: 0 <= e < 1'
    x0 = float(Ns)/2.                                                                                    # Ns = 2*cppr_max + 2, so half well be cppr_max + 1
    y0 = float(Ns)/2.                                                                                    # Define x0, y0 to be the centre of the mesh
    mesh0 *= 0.                                                                                        # A safety feature to ensure that mesh0 is always all 0. before starting
    A = r_
    B = A*np.sqrt(1.-e_**2.)                                                                            # A is the semi-major radius, B is the semi-minor radius
    for j in range(Ns):
        for i in range(Ns):
            xc = 0.5*(i + (i+1)) - x0
            yc = 0.5*(j + (j+1)) - y0 
            
            xct = xc * np.cos(a_) - yc * np.sin(a_)
            yct = xc * np.sin(a_) + yc * np.cos(a_)
            r = (xct/A)**2. + (yct/B)**2.
            
            if r<=1:
                mesh0[j,i] = 1.
    return mesh0
    
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
    global mesh0, Ns                                                                                # mesh0 is Ns x Ns in size
    CL = 1.                                                                                            # This is mesh0 so cell length is simply '1.'
    x0 = float(Ns)/2.                                                                                    # Ns = 2*cppr_max + 2, so half well be cppr_max + 1
    y0 = float(Ns)/2.                                                                                    # Define x0, y0 to be the centre of the mesh
    mesh0[:] = 0.                                                                                        # This is necessary to ensure the mesh is empty before use.
    for j in range(Ns):                                                                                    # Iterate through all the x- and y-coords
        for i in range(Ns):                        
            xc = 0.5*(i + (i+1)) - x0                                                                    # Convert current coord (centre of cell) to position relative to (x0,y0) 
            yc = 0.5*(j + (j+1)) - y0                                                                    # Everything is now in cartesian coords relative to the mesh centre
            r = np.sqrt((xc)**2. + (yc)**2.)                                                            # Find the radial distance from current coord to (x0,y0)_
            if r<=(r_-1.):                                                                                # If this is less than r_-1 then the coord is in the 
                                                                                                        # circle => fill COMPLETELY
                mesh0[j,i] = 1.0                                                                        # Fill cell
                AREA += 1                                                                                # Increment area
            elif abs(r-r_)<np.sqrt(2.):                                                                    # BUT if the cell centre is with root(2) 
                                                                                                        # of the circle edge, then partially fill
                xx = np.arange(i,i+1,.1)                                                                # Create 2 arrays of 11 elements each and with divisions of 0.1
                yy = np.arange(j,j+1,.1)                                                                # Essentially this splits the cell into a mini-mesh 
                                                                                                        # of 10 x 10 mini cells
                for I in range(10):
                    for J in range(10):                                                                    # Iterate through these
                        xxc = 0.5+xx[I] - x0                                                            # Change their coordinates as before
                        yyc = 0.5+yy[J] - y0
                        r = np.sqrt((xxc)**2. + (yyc)**2. )                                                # Find the radial distance from current mini coord to (x0,y0)
                        if r <= r_:                                                                        # If this is less than r_ then the mini coord is in the circle.
                            mesh0[j,i] += (.1**2.)                                                        # Fill cell by 0.1**2 (the area of one mini-cell)
    """    
    plt.figure()
    plt.imshow(mesh0,cmap='Greys',interpolation='nearest')
    plt.show()
    """
    return mesh0

def gen_polygon(vertices,radii,angles=None):                        
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
    global mesh0, cppr_min, cppr_max, n_min, n_max, Ns                                                    # Only the angles used are now randomly selected.
    mesh0 *= 0.                                                                                        # This is necessary to ensure the mesh is empty before use.
    n      = vertices
    R      = np.zeros((2,n))                                                                                # Array for the coords of the vertices
    delr   = (cppr_max-cppr_min)                                                                            # Difference between min and max radii

    if angles == None:
        I     = np.arange(n)                                                                                    # array of vertex numbers
        phase = random.random()*np.pi*2.
        phi   = 2.*np.pi*I/n + phase                                                                               # Generate 'n' random angles 
    else:
        phi = angles

    rho    = radii                                                                                        # Ensure each vertex is within an arc, 1/nth of a circle
    R[0,:] = rho*np.cos(phi)                                                                            # Each vertex will also be successive, such that drawing a line between
    R[1,:] = rho*np.sin(phi)                                                                            # each one in order, will result in no crossed lines.
                                                                                                        # Convert into cartesian coords and store in R
    qx = 0.                                                                                                # Make the reference point (q) zero, i.e. the centre of the shape
    qy = 0.                                                                                                # All dimensions are in reference to the central coordinates.
    x0 = float(Ns)/2.                                                                                    # Ns = 2*cppr_max + 2, so half well be cppr_max + 1
    y0 = float(Ns)/2.                                                                                    # Define x0, y0 to be the centre of the mesh
    for j in range(Ns-1):                                                                                # Iterate through all the x- and y-coords
        for i in range(Ns-1):                                                                            # N-1 because we want the centres of each cell, 
                                                                                                        # and there are only N-1 of these!
            xc = 0.5*(i + (i+1)) - x0                                                                    # Convert current coord to position relative to (x0,y0) 
            yc = 0.5*(j + (j+1)) - y0                                                                    # Everything is now in indices, with (x0,y0)

            sx = xc - qx                                                                            # s = vector difference between current coord and ref coord
            sy = yc - qy           
            intersection = 0                                                                            # Initialise no. intersections as 0
            for l in range(n-1):                                                                        # cycle through each edge, bar the last
                rx = R[0,l+1] - R[0,l]                                                                    # Calculate vector of each edge (r), i.e. the line between 
                                                                                                        # the lth vertex and the l+1th vertex
                ry = R[1,l+1] - R[1,l]
                RxS = (rx*sy-ry*sx)                                                                        # Vector product of r and s (with z = 0), technically 
                                                                                                        # produces only a z-component
                if RxS!=0.:                                                                                # If r x s  = 0 then lines are parallel
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
            rx = R[0,0] - R[0,n-1]                                                                        # Do the last edge. Done separately to avoid needing a circular
                                                                                                        # 'for' loop
            ry = R[1,0] - R[1,n-1]
            if (rx*sy-ry*sy)!=0.:
                RxS = (rx*sy-ry*sx)
                t = ((qx-R[0,n-1])*sy - (qy-R[1,n-1])*sx)/RxS
                u = ((qx-R[0,n-1])*ry - (qy-R[1,n-1])*rx)/RxS
                if t<=1. and t>=0. and u<=1. and u>=0.:
                    intersection = intersection + 1
            if (intersection%2==0.):                                                                    # If number of intersections is divisible by 2 (or just zero)
                                                                                                        # -> fill that cell!
                mesh0[j,i] = 1.0
    #plt.figure()
    #plt.imshow(mesh0,cmap='binary',interpolation='nearest')
    #plt.plot(x0,y0,color='r',marker='o')
    #plt.show()
    return mesh0

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
    global mesh, meshx, meshy, cppr_mid,cppr_max                                                        # use global parameters
    cell_limit = (np.pi*float(cppr_max)**2.)/100.                                                        # Max number of overlapping cells should scale with area.
    CHECK  = 0                                                                                            # Initialise the CHECK as 0; 0 == all fine
    Py, Px     = np.shape(shape)

    if x > meshx+cppr_max: CHECK = 1
    if x < 0-cppr_max:  CHECK = 1
    if y > meshy+cppr_max: CHECK = 1
    if y < 0-cppr_max:  CHECK = 1                                                                        # If the coord moves out of the mesh, wrap back around.
    i_edge    = x - Ns/2
    j_edge    = y - Ns/2
    i_finl    = x + Ns/2
    j_finl    = y + Ns/2
        
    if i_edge < 0:                                                                                        # If the coords have the particle being generated over the mesh boundary
        I_initial = abs(i_edge)                                                                            # This bit checks for this, and reassigns a negative starting
                                                                                                        # index to zero
        i_edge    = 0                                                                                    # However, the polygon's mesh will not completely be in the main mesh 
    else:                                                                                                # So I_initial defines the cut-off point
        I_initial = 0                                                                                    # If the polygon's mesh does not extend beyond the main mesh,
                                                                                                        # then I_initial is just 0
    if j_edge < 0:                                                                                        # Repeat for the j-coordinate
        J_initial = abs(j_edge) 
        j_edge = 0
    else:
        J_initial = 0
    
    I_final = Px                                                                                        # In a numpy array '-1' indicates the last element.        
    if (i_finl)>meshx:                                                                                    # Similarly for the other end of the mesh
        I_final -= abs(meshx-i_finl)                                                                    # The above only sorts out two sides of the mesh
        i_finl   = meshx
    J_final = Py
    if (j_finl)>meshy:
        J_final -= abs(meshy-j_finl) 
        j_finl   = meshy
    
    """    
    for i in range(i_edge, i_finl, 1):                                                                    # Iterate over the section of the full mesh of interest
        I = i - i_edge + I_initial                                                                        # The equivalent index within the polygon's mesh
        for j in range(j_edge, j_finl, 1):
            J = j - j_edge + J_initial
            if mesh[i,j]!=0. and shape[I,J] != 0.:
                    CHECK = 1                                                                            # If there's a point in the polygon's mesh that has material,
    """                                                                                                    # AND the corresponding point    
    temp_shape = np.copy(shape[J_initial:J_final,I_initial:I_final])                                    # The rectangular array containing the portion of shape 
    temp_mesh  = np.copy(mesh[j_edge:j_finl,i_edge:i_finl])                                                # The equivalent rectangular array within the mesh, in the correct place
    test       = np.minimum(temp_shape,temp_mesh)                                                        # An array containing any points that have material in,
                                                                                                        # in the same place, in BOTH arrays        
    if (np.sum(test) > 0):                                                                                # If 'test' is > 2, then there are more than 2 cells 
        CHECK=1
        pass                                                                                            # overlapping with other objects at this position
    elif(np.sum(test) == 0):                                                                            # If there are fewer than 2 overlapping cells, but MORE than 0,
        pass
    return CHECK                                                                                        # has material in it in the main mesh => failure and CHECK = 1


def drop_shape_into_mesh(shape):
    """
    This function 'drops' a particle into the mesh and has it undergo a random walk
    until it overlaps sufficiently with another particle and is declared 'touching'.
    Only then is a particle fully inserted into the mesh.
    
    """
    global mesh, meshx, meshy, cppr_max, cppr_min, materials
    
    cell_limit = max((np.pi*float(cppr_max)**2.)/100.,1)                                                # Max number of overlapping cells should scale with area.
                                                                                                        # area ~= 110 cells for 6cppr
                                                                                                        # Does NOT need to be integer since values in the mesh are floats, 
                                                                                                        # and it is their sum that is calculated.
    touching   = 0                                                                                        # Initialise the indicators for this function
    passes     = 1                                                                                        # here are 'passes' and 'counter' similar to check_coords_full
    counter    = 0                                                                                        # But this time there is 'touching' which indicates
                                                                                                        # contact between particles
    x,y = gen_coord_basic()
    
    Nx, Ny     = meshx, meshy
    Py, Px     = np.shape(shape)
    while touching == 0:    
        #if x > Nx-rr or x < rr: x,y = gen_coord_basic()
        #if y > Ny-rr or y < rr: x,y = gen_coord_basic()
        if x > Nx+cppr_max: x = 0-cppr_max+1
        if x < 0-cppr_max:  x = Nx+cppr_max-1
        if y > Ny+cppr_max: y = 0-cppr_max+1
        if y < 0-cppr_max:  y = Ny+cppr_max-1                                                            # If the coord moves out of the mesh, wrap back around.
        i_edge    = x - Ns/2
        j_edge    = y - Ns/2
        i_finl    = x + Ns/2
        j_finl    = y + Ns/2
        #i_edge   = x - cppr_max - 1                                                                        # Location of the edge of the polygon's mesh, within the main mesh.
        #j_edge   = y - cppr_max - 1
        #i_finl   = x + cppr_max + 1                                                                        # The indices refer to the close edge to the origin,
                                                                                                        # the extra cell should be added on the other side
        #j_finl   = y + cppr_max + 1                                                                        # i.e. the side furthest from the origin
        if i_edge < 0:                                                                                  # If the coords have the particle being generated over the mesh boundary
            I_initial = abs(i_edge)                                                                     # This bit checks for this, and reassigns a negative starting
                                                                                                        # index to zero
            i_edge    = 0                                                                               # However, the polygon's mesh will not completely be in the main mesh 
        else:                                                                                           # So I_initial defines the cut-off point
            I_initial = 0                                                                               # If the polygon's mesh does not extend beyond the main mesh,
                                                                                                        # then I_initial is just 0
        if j_edge < 0:                                                                                  # Repeat for the j-coordinate
            J_initial = abs(j_edge) 
            j_edge = 0
        else:
            J_initial = 0
        
        I_final = Px                                                                                    # In a numpy array '-1' indicates the last element.        
        if (i_finl)>Nx:                                                                                 # Similarly for the other end of the mesh
            I_final -= abs(Nx-i_finl)                                                                   # The above only sorts out two sides of the mesh
            i_finl   = Nx
        J_final = Py
        if (j_finl)>Ny:
            J_final -= abs(Ny-j_finl) 
            j_finl   = Ny
        
        temp_shape = np.copy(shape[J_initial:J_final,I_initial:I_final])                                # The rectangular array containing the portion of shape 
                                                                                                        # to be placed into mesh 
        temp_mesh  = np.copy(mesh[j_edge:j_finl,i_edge:i_finl])                                         # The equivalent rectangular array within the mesh, in the correct place
        test       = np.minimum(temp_shape,temp_mesh)                                                   # An array containing any points that have material in,
                                                                                                        # in the same place, in BOTH arrays        
        if abs(I_initial-I_final)<=cppr_min or abs(J_initial-J_final)<= cppr_min:
            rx = random.randint(-cppr_min,cppr_min)
            ry = random.randint(-cppr_min,cppr_min)
            y += ry 
            x += rx
        elif (np.sum(test) > cell_limit or np.sum(test) == 0.):                                         # If 'test' is > 2, then there are more than 2 cells 
                                                                                                        # overlapping with other objects at this position
            rx = random.randint(-cppr_min,cppr_min)
            ry = random.randint(-cppr_min,cppr_min)
            y += ry 
            x += rx
        elif(np.sum(test) <= cell_limit):                                                               # If there are fewer than 2 overlapping cells, but MORE than 0,
                                                                                                        # place shape here.
            mesh[j_edge:j_finl,i_edge:i_finl] = np.maximum(shape[J_initial:J_final,I_initial:I_final],mesh[j_edge:j_finl,i_edge:i_finl])    
            #plt.figure(1)
            #plt.imshow(mesh, cmap='Greys',  interpolation='nearest')
            #plt.plot(x,y,color='r',marker='o',ms='5')
            #plt.show()
                                                                                                        # materials contains each cell's material number.
                                                                                                        # Prior to mat assignment 
                                                                                                        # it is just the particle number
            touching = 1                                                                                # Assign 'touching' a value of 1 to break loop
            #temp_shape[temp_shape>0.] = 1.                                                             # Change all values in temp_shape to 1., if not already 
            area = np.sum(temp_shape) - np.sum(test)                                                    # Area placed into mesh is the sum of all positive points
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

    This differs from 'place shape' as it does not require the material number to be known
    and is predominantly used when arranging particles, rather than 'placing them'(permanently)
    at the final stage of particle bed construction.
    """
    global mesh, meshx, meshy, cppr_max, materials
    Py, Px = np.shape(shape)                                                                            # Px and Py are the dimensions of the 'shape' array
    i_edge    = x0 - Ns/2
    j_edge    = y0 - Ns/2
    i_finl    = x0 + Ns/2
    j_finl    = y0 + Ns/2
    #i_edge = x0 - cppr_max - 1                                                                            # Location of the edge of the polygon's mesh, within the main mesh.
    #j_edge = y0 - cppr_max - 1                                                                            # This is calculated explicitly in case the mesh has a non-constant size
    #i_finl = x0 + cppr_max + 1                                                                            # The indices refer to the closest edge to the origin,
                                                                                                        # an extra cell is added either side
    #j_finl = y0 + cppr_max + 1                                                                            # to ensure the shape is completely encompassed within the box
    
    """ 'i' refers to the main mesh indices whereas 'I' refers to 'shape' indices """
    if i_edge < 0:                                                                                        # Condition if the coords have the particle being generated 
                                                                                                        # over the mesh boundary
        I_initial = abs(i_edge)                                                                            # a negative starting index is reassigned to zero
        i_edge    = 0                                                                                    # The polygon's mesh will not completely be in the main mesh 
    else:                                                                                                # So I_initial defines the cut-off point
        I_initial = 0                                                                                    # If the polygon's mesh does not extend beyond the main mesh,
                                                                                                        # then I_initial is just 0
    if j_edge < 0:                                                                                        # Repeat for the j-coordinate
        J_initial = abs(j_edge) 
        j_edge = 0
    else:
        J_initial = 0
    
    I_final = Px 
    if (i_finl)>meshx:                                                                                    # This section deals with the occurrence of the shape overlapping 
                                                                                                        # with the opposite ends of
        I_final -= abs(meshx-i_finl)                                                                    # meshes. And works on the same principles as above, 
                                                                                                        # although the maths is slightly different
        i_finl   = meshx
    J_final = Py
    if (j_finl)>meshy:
        J_final -= abs(meshy-j_finl) 
        j_finl   = meshy
    
    temp_shape = shape[J_initial:J_final,I_initial:I_final]                                                # record the shape as a temporary array for area calculation
    mesh[j_edge:j_finl,i_edge:i_finl] = np.maximum(shape[J_initial:J_final,I_initial:I_final],mesh[j_edge:j_finl,i_edge:i_finl])
    #plt.figure(1)
    #plt.imshow(mesh, cmap='Greys',  interpolation='nearest')
    #plt.plot(x0,y0,color='r',marker='o',ms='5')
    #plt.show()
    
                                                        
    """ The shape is inserted by comparing, and taking the maximum, of the two arrays  """
    area = np.sum(temp_shape)                                                                            # Area is sum of all these points
    return area

def place_shape(shape,x0,y0,mat,MATS=None,LX=None,LY=None,Mixed=False,info=False,shapeno=0):
    """
    This function inserts the shape (passed as the array 'shape') into the
    correct materials mesh at coordinate x0, y0.
    
    shape  : mesh0 array of the shape to be inserted
    x0, y0 : The x and y coords at which the shape is to be placed. (These are the shape's origin point)
    mat    : this is the index of the material
    
    This method relies on there being no particles of identical materials being in contact (i.e. no overlapping particles within the materials mesh)
    
    nothing is returned.
    """
    global mesh, meshx, meshy, cppr_max, materials,Ns,part_no,GS
    if MATS == None: MATS = materials                                                                 # Now the materials mesh is only the default. Another mesh can be used!
    if LX   == None: LX   = meshx                                                                     # The code should still work as before.
    if LY   == None: LY   = meshy
    if type(x0) == float or type(x0) == np.float64: x0 = int(x0/GS)
    if type(y0) == float or type(y0) == np.float64: y0 = int(y0/GS)
    Py, Px = np.shape(shape)                                                                          # Px and Py are the dimensions of the 'shape' array
    i_edge    = x0 - Ns/2
    j_edge    = y0 - Ns/2
    i_finl    = x0 + Ns/2
    j_finl    = y0 + Ns/2
    #i_edge = x0 - cppr_max - 1                                                                      # Location of the edge of the polygon's mesh, within the main mesh.
    #j_edge = y0 - cppr_max - 1                                                                      # This is calculated explicitly in case the mesh has a non-constant size
    #i_finl = x0 + cppr_max + 1                                                                      # The indices refer to the closest edge to the origin,
                                                                                                     # an extra cell is added either side
    #j_finl = y0 + cppr_max + 1                                                                      # to ensure the shape is completely encompassed within the box
    mat = int(mat) 
    """ 'i' refers to the main mesh indices whereas 'I' refers to 'shape' indices """
    if i_edge < 0:                                                                                   # Condition if the coords have the particle being generated 
                                                                                                     # over the mesh boundary
        I_initial = abs(i_edge)                                                                      # a negative starting index is reassigned to zero
        i_edge    = 0                                                                                # The polygon's mesh will not completely be in the main mesh 
    else:                                                                                            # So I_initial defines the cut-off point
        I_initial = 0                                                                                # If the polygon's mesh does not extend beyond the main mesh, 
                                                                                                     # then I_initial is just 0
    if j_edge < 0:                                                                                   # Repeat for the j-coordinate
        J_initial = abs(j_edge) 
        j_edge = 0
    else:
        J_initial = 0
    
    I_final = Px 
    if (i_finl)>LX:                                                                                        # This section deals with the occurrence of the shape 
                                                                                                        # overlapping with the opposite ends of
        I_final -= abs(LX-i_finl)                                                                        # meshes. And works on the same principles as above, 
                                                                                                        # although the maths is slightly different
        i_finl   = LX
    J_final = Py
    if (j_finl)>LY:
        J_final -= abs(LY-j_finl) 
        j_finl   = LY
    
    temp_shape = shape[J_initial:J_final,I_initial:I_final]                                                # record the shape as a temporary array for area calculation
    NJ, NI = np.shape(temp_shape)
    if Mixed == False:
        for o in range(J_final-J_initial):
            for p in range(I_final-I_initial):
                if temp_shape[o,p] == 1.: 
                    if MATS == None:
                        if np.sum(materials[:,o+j_edge,p+i_edge]) == 0.:
                            materials[mat-1,o+j_edge,p+i_edge] = 1.
                            if info: part_no[o+j_edge,p+i_edge] = shape_no
                        else:
                            pass
                    else:
                        if np.sum(MATS[:,o+j_edge,p+i_edge]) == 0.:
                            MATS[mat-1,o+j_edge,p+i_edge] = 1.
                            if info: part_no[o+j_edge,p+i_edge] = shapeno
                        else:
                            pass
    elif Mixed == True:
        for o in range(J_final-J_initial):
            for p in range(I_final-I_initial):
                if temp_shape[o,p] > 0.: 
                    if MATS is None:
                        tot_present = np.sum(materials[:,o+j_edge,p+i_edge])
                        space_left = 1. - tot_present
                        if temp_shape[o,p] > space_left: 
                            new_mat = space_left
                        else:
                            new_mat = temp_shape[o,p]
                        materials[mat-1,o+j_edge,p+i_edge] += new_mat
                        if info and temp_shape[o,p]>0.5: part_no[o+j_edge,p+i_edge] = shapeno
                    else:
                        tot_present = np.sum(MATS[:,o+j_edge,p+i_edge])
                        space_left = 1. - tot_present
                        if temp_shape[o,p] > space_left: 
                            new_mat = space_left
                        else:
                            new_mat = temp_shape[o,p]
                        MATS[mat-1,o+j_edge,p+i_edge] += new_mat
                        if info and temp_shape[o,p]>0.5: part_no[o+j_edge,p+i_edge] = shapeno

    #objects_temp                                 = np.ceil(np.maximum(shape[I_initial:I_final,J_initial:J_final],objects[mat-1,i_edge:i_finl,j_edge:j_finl]))
    #objects_temp[objects_temp>0.]                = obj
    #objects[mat-1,i_edge:i_finl,j_edge:j_finl]   = objects_temp 
    # Objects mesh is no longer necessary and has been tentatively removed
    
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
    global mesh,meshy,cppr_max                                                                            # Use the global mesh
    good = 0                                                                                            # Here 'good' is the indicator for the loop, when this equals 1 we are 'good' and the loop breaks
    indices = np.where(mesh==0.)                                                                        # Generates 2 arrays containing the indices of all values within mesh which == 0
    indices = np.column_stack(indices)                                                                    # column_stack mushes them together
    y,x = random.choice(indices)                                                                        # This randomly selects one pair of coordinates!
    """
    while good == 0:
        if y < cppr_max or y > meshy-cppr_max:                                                            # If the coord is within one radius of the edge, => rechoose
            x,y = random.choice(indices)                                                                # This randomly selects one pair of coordinates!
        else:
            good = 1                                                                                    # Otherwise the coord is fine and is returned.
    """
    return x,y

def gen_coord_in_box(shape,box):                                                                                    
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

    This version takes values for a 'box' in which to generate the coordinates
    
    shape : array containing the shape to be checked. This does not have to be regular 
    """
    
    global mesh                                                                                            # use global parameters (the mesh itself, mesh size)
    check   = 1                                                                                            # Initialise the indicator variables
    counter = 0                                                                                            # and the counter
    passes  = 0                                                                                            # passes should start as 0 and check should start as 1.
    indices = np.where(box==0.)                                                                        # Generates an array of all indices in the main mesh of unassigned cells
    indices = np.column_stack(indices)                                                                    # column_stack mushes them together such that random choice 
                                                                                                        # can choose pairs of coords
    while check == 1:                                                                                    # Begin the loop. whilst check = 1, continue to loop 
        y,x   = random.choice(indices)                                                                    # This randomly selects one pair of coordinates!
        check = check_coords_full(shape,x,y)                                                            # Function to check if the polygon generated will fit 
        counter += 1                                                                                    # Increment the counter
        if counter>5000:                                                                                # If the counter exceeds 5000, this is the break clause
            check = 0                                                                                    # Break the loop and report the failure in the return
            passes= 1                                                                                    # if counter exceeds 1000 -> break loop and assign passes[k] = 1
            break
    return x,y,passes

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
    
    global mesh                                                                                            # use global parameters (the mesh itself, mesh size)
    check   = 1                                                                                            # Initialise the indicator variables
    counter = 0                                                                                            # and the counter
    passes  = 0                                                                                            # passes should start as 0 and check should start as 1.
    indices = np.where(mesh==0.)                                                                        # Generates an array of all indices in the main mesh of unassigned cells
    indices = np.column_stack(indices)                                                                    # column_stack mushes them together such that random choice 
                                                                                                        # can choose pairs of coords
    while check == 1:                                                                                    # Begin the loop. whilst check = 1, continue to loop 
        y,x   = random.choice(indices)                                                                    # This randomly selects one pair of coordinates!
        check = check_coords_full(shape,x,y)                                                            # Function to check if the polygon generated will fit 
        counter += 1                                                                                    # Increment the counter
        if counter>5000:                                                                                # If the counter exceeds 5000, this is the break clause
            check = 0                                                                                    # Break the loop and report the failure in the return
            passes= 1                                                                                    # if counter exceeds 1000 -> break loop and assign passes[k] = 1
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
    MAT = np.random.choice(mats,N)                                                                        # This function randomly chooses values from 'mats' 
                                                                                                        # to populate an array of size N
    return MAT



def mat_assignment(mats,xc,yc):                        
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
    global cppr_max,GS,Ns
    N    = np.size(xc)                                                               # No. of particles
    M    = np.size(mats)
    L    = GS                                                                        # Length of one cell
    MAT  = np.zeros((N))                                                             # Array for all material numbers of all particles
    i = 0                                                                            # Counts the number of particles that have been assigned
    while i < N:                                                                     # Loop every particle and assign each one in turn.    
        lowx   = xc[i] - 3.*Ns*L                                                     # Create a 'box' around each particle (in turn) that is 6Ns x 6Ns
        higx   = xc[i] + 3.*Ns*L
        lowy   = yc[i] - 3.*Ns*L
        higy   = yc[i] + 3.*Ns*L
        boxmat = MAT[(lowx<=xc)*(xc<=higx)*(lowy<=yc)*(yc<=higy)]                    # Array containing a list of all material numbers within the 'box' 
        boxx   =  xc[(lowx<=xc)*(xc<=higx)*(lowy<=yc)*(yc<=higy)]                    # Array containing the corresponding xcoords
        boxy   =  yc[(lowx<=xc)*(xc<=higx)*(lowy<=yc)*(yc<=higy)]                    # and the ycoords
        nn     =  np.size(boxmat)
        D   = np.sqrt((boxx - xc[i])**2. + (boxy - yc[i])**2.)                       # Calculate the distances to the nearest particles
        ind = np.argsort(D)                                                          # Sort the particles into order of distance from the considered particle
        BXM = boxmat[ind]                                                            # Sort the materials into the corresponding order
        DU  = np.unique(BXM[:M])                                                     # Only select the M closest particles
        if np.array_equal(DU, mats):                                                 # If the unique elements in this array equate the array of 
                                                                                     # materials then all are taken
            mm     = BXM[M-1]                                                                           
            MAT[i] = mm                                                              # Set the particle material to be of the one furthest from 
                                                                                     # the starting particle
            materials[materials==-1*(i+1)] = mm                                      # Assign all filled cells 
        else:                                                                        # Else there is a material in mats that is NOT in DU
            indices = np.in1d(mats,DU,invert=True)                                   # This finds the indices of all elements that only appear in 
                                                                                     # mats and not DU
            mm      = np.random.choice(mats[indices],1)
            MAT[i]  = mm                                                             # Randomly select one to be the current particle's material number
            materials[materials==-1*(i+1)] = mm
        i += 1                                                                       # Increment i
    MAT = MAT.astype(int)
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
    
    N           = np.size(X)                                                                            # No. of particles
    mean_radii  = np.mean(radii)                                                                        # Calculate the mean radii    
    D           = np.zeros((N))                                                                            # D is an array for the distances between particles
    GRIDSPC     = np.amax(radii)/cppr_max                                                                # Physical distance/cell (SI)
    Dtouch      = []                                                                                    # The final list of distances for parts in contact
    diameters   = radii*2.                                                                                # All the diameters
    error       = radii/cppr_mid                                                                        # The assumed error on the distances between particles
    upper_bound = diameters + error/2.                                                                    # The upper bound on this distance
    print 'Max, Min, Mean radii = {},{},{}'.format(np.amax(radii),np.amin(radii),mean_radii)
    B           = 0                                                                                        # B is the number of contacts that exist between the same materials
    
    if plot == True:                                                                                    # If plot == True then produce a figure
        fig = plt.figure()
        ax = fig.add_subplot(111,aspect='equal')
        ax.set_xlim(0,meshx*GRIDSPC)                                                                    # limits are set like this to ensure the final graphic 
                                                                                                        # matches the mesh in orientation
        ax.set_ylim(meshy*GRIDSPC,0)
        for i in range(N):                                                                                # Plot each circle in turn
            ax.plot([X[i]],[Y[i]],color='k',marker='o',linestyle=' ',ms=3)
            circle = plt.Circle((X[i],Y[i]),radii[i],color='{:1.2f}'.format((MAT[i])*.5/np.amax(MAT)))  # give each one a color based on their material number. 

                                                                                                        # NB any color = 1. will be WHITE 
            ax.add_patch(circle)
    
    for i in range(N-1):
        D *= 0.                                                                                             # Initialise D each loop, in case it is full and 
                                                                                                         # to ensure nothing carries over
        D -= 1.
        for j in range(N-1):                                                                             # find distances between current particle and 
                                                                                                         # ALL other particles and store these in D
            dx = X[i] - X[j]
            dy = Y[i] - Y[j]
            distance = np.sqrt(dx**2. + dy**2.)
            D[j] = distance
            if D[j]>0. and D[j]<upper_bound[i]:                                                             # If particles overlap draw a red line between their centres
                if plot == True:                                                                            
                    ax.plot([X[i],X[j]],[Y[i],Y[j]],lw=1.,color='r')
                Dtouch.append(D[j])
                if MAT[i] == MAT[j]: B += 1
        
    
    Dtouch = np.array(Dtouch)                                                                             # Convert to numpy array
     
    A = float(np.size(Dtouch))/float(N)                                                                     # The size of Dtouch/total part number is the mean
    B = float(B)/float(N)                                                                                 # B is the average number of contacts/particle that are 
                                                                                                         # between identical materials
    if plot == True: 
        ax.set_title('$Z = ${:1.3f}'.format(A))
        plt.savefig('contacts_figure_Z-{:1.3f}_B-{:1.3f}.png'.format(A,B),dpi=600)                         # Save the figure
        plt.show()
    
    return A, B


def save_spherical_parts(X,Y,R,MATS,Z,fname='meso'):
    global mesh, mesh_Shps,meshx,meshy,FRAC,OBJID,materials,trmesh
    """
    Saves particle placement information in a meso.iSALE file in format:
    MATERIAL : X0 : Y0 : RADIUS
    This can be read by iSALE. NB X,Y and R are in physical units
    """
    if fname == 'meso':
        fname += '_Z-{:3.4f}.iSALE'.format(Z)
    else:
        pass
    
    ALL  = np.column_stack((MATS,X,Y,R))                                                
    np.savetxt(fname,ALL,comments='')
    return

def view_mesoiSALE(filepath = 'meso.iSALE',save=False):
    """
    This function allows you to view the arrangement a meso.iSALE file would produce
    """
    M = np.genfromtxt(filepath,dtype=float,usecols=(0))
    X = np.genfromtxt(filepath,dtype=float,usecols=(1))
    Y = np.genfromtxt(filepath,dtype=float,usecols=(2))
    R = np.genfromtxt(filepath,dtype=float,usecols=(3))
    N = np.size(X)

    fig = plt.figure()
    ax  = fig.add_subplot(111,aspect='equal')
    ax.set_xlim([np.amin(X)-np.amax(R),np.amax(X)+np.amax(R)])
    ax.set_ylim([np.amin(Y)-np.amax(R),np.amax(Y)+np.amax(R)])
    for i in range(N):                                                                                     # Plot each circle in turn
        circle = plt.Circle((X[i],Y[i]),R[i],color='{:1.2f}'.format((M[i])*.5/np.amax(M)))               # give each one a color based on their material number. NB any color = 1. will be WHITE 
        ax.add_patch(circle)
    if save:
        filepath=filepath.replace('.iSALE','.png')
        plt.savefig(filepath,dpi=500)
    plt.show()
    return
"""
def view_meso_miSALE(MX,MY,filepath = 'meso_m.iSALE',save=False):
    
    N,mats = np.genfromtxt(filepath,dtype=float,max_rows=1)
    F      = np.zeros(N,mats)
    X      = np.genfromtxt(filepath,dtype=float,usecols=(0),skip_header=1)
    Y      = np.genfromtxt(filepath,dtype=float,usecols=(1),skip_header=1)

    for j in range(mats):
        F[:,j] = np.genfromtxt(filepath,dtype=float,usecols=(2+j),skip_header=1)

    Nx   = np.amax(X)
    Ny   = np.amax(Y)
    assert Nx*Ny == N
    mesh = np.zeros(Nx,Ny)
    for i in range(N):
        mesh[X[i],Y[i]] = np.sum(F[i,:])
    fig = plt.figure()
    ax  = fig.add_subplot(111,aspect='equal')
    ax.imshow(mesh,interpolation='nearest')
    if save:
        filepath=filepath.replace('.iSALE','.png')
        plt.savefig(filepath,dpi=500)
    plt.show()
    return
"""
def particle_gap_measure(filepath = 'meso.iSALE',plot=False):
    """
    This function calculates the average distance between a particle and the next one below it, that is NOT in contact: 'G'.
    G is calculated in multiples of radii.
    """
    M = np.genfromtxt(filepath,dtype=float,usecols=(0))
    X = np.genfromtxt(filepath,dtype=float,usecols=(1))
    Y = np.genfromtxt(filepath,dtype=float,usecols=(2))
    R = np.genfromtxt(filepath,dtype=float,usecols=(3))
    N = np.size(X)
    
    crit_angle  = 40                                                                                    # The angle after which Force chains no longer propagate
    crit_angle *= np.pi/180.                                                                            # i.e. they are pushed out the way (need in radians)
    mean_radii  = np.mean(R)                                                                            # Calculate the mean radii    
    D           = np.zeros((N))                                                                            # D is an array for the distances between particles
    Dgap        = []                                                                                    # The final list of distances for parts in contact
    UB          = X + R*np.sin(crit_angle)                                                              # Check all particles below, within this window
    LB          = X - R*np.sin(crit_angle)                                                              # that is within the critical angle arc

    X /= R
    Y /= R
    UB/= R
    LB/= R
    R /= R
    if plot == True:                                                                                    # If plot == True then produce a figure
        fig = plt.figure()
        ax = fig.add_subplot(111,aspect='equal')
        ax.set_xlim(np.amin(X),np.amax(X))
        ax.set_ylim(np.amin(Y),np.amax(Y))
        for i in range(N):                                                                                # Plot each circle in turn
            ax.plot([X[i]],[Y[i]],color='k',marker='x',linestyle=' ',ms=3,mew=1.)
    
    for i in range(N-1):
        best_dy = Y[i] - np.amin(Y)
        old_dy  = best_dy
        best_X = X[i] 
        best_Y = np.amin(Y) 
        old_X  = best_X
        old_Y  = best_Y
        for j in range(N-1):    # find distances between current particle and 
            if i == j:
                pass
            elif X[j] < UB[i] and X[j] > LB[i] and Y[j] < Y[i]:                                            # Only consider particles within the critical arc
                dy = abs(Y[j] - Y[i]) #- R[i] - R[j]                                                         # And ones below the current particle
                if dy < old_dy: 
                    best_dy = dy
                    best_X  = X[j]
                    best_Y  = Y[j]
                else:
                    best_dy = old_dy 
                    best_X  = old_X
                    best_Y  = old_Y
                old_dy  = best_dy
                old_X   = best_X
                old_Y   = best_Y
            else:
                pass
        D[i] = best_dy 
        if best_dy <= 2.:
            if plot == True:
                ax.plot([X[i],best_X],[Y[i],best_Y],lw=1.5,color='r')
        else:
            Dgap.append(D[i])
            if plot == True:
                ax.plot([X[i],best_X],[Y[i],best_Y],lw=1.,color='b')
    Dgap  = np.array(Dgap)                                                                             # Convert to numpy array
    Dgap -= 2                                                                                           # Now Dgap is the distance between parts and NOT centres
     
    G = np.mean(Dgap)                                                                 # The size of Dtouch/total part number is the mean
                                                                                                         # gap length, in radii
    if plot == True: 
        ax.set_title('$G = ${:1.3f} radii'.format(G))
        ax.set_xlabel('Transverse Position [Radii]')
        ax.set_ylabel('Longitudinal Position [Radii]')
        plt.savefig('gaps_figure_G-{:1.3f}.png'.format(G),dpi=600)                                         # Save the figure
        plt.show()
    return G

def discrete_contacts_number(SHAPENO,X,Y,n,PN):
    global mesh, mesh_Shps,meshx,meshy,FRAC,OBJID,materials
    mesh                *= 0.
    contacts             = np.zeros_like(PN)
    contact_matrix = np.zeros((n,n))
    for k in range(n):
        shape = np.copy(mesh_Shps[SHAPENO[k]])
        shape[shape>0] = PN[k]
        area = insert_shape_into_mesh(shape,X[k],Y[k])

    for i in range(meshx):
        for j in range(meshy):
            if mesh[j,i] > 0:
                J = min(j,meshy-2)
                J = max(1,J)
                I = min(i,meshx-2)
                I = max(1,I)
                cel = mesh[J,I]
                top = mesh[J+1,I]
                bot = mesh[J-1,I]
                lef = mesh[J,I-1]
                rit = mesh[J,I+1]
                if top > 0. and top != cel and contact_matrix[int(top-1),int(cel-1)] == 0.:
                    contacts[PN == cel]   += 1
                    contacts[PN == top] += 1
                    contact_matrix[int(top-1),int(cel-1)] += 1.
                    contact_matrix[int(cel-1),int(top-1)] += 1.
                if bot > 0. and bot != cel and contact_matrix[int(bot-1),int(cel-1)] == 0.:
                    contacts[PN == cel]   += 1
                    contacts[PN == bot] += 1
                    contact_matrix[int(bot-1),int(cel-1)] += 1.
                    contact_matrix[int(cel-1),int(bot-1)] += 1.
                if lef > 0. and lef != cel and contact_matrix[int(lef-1),int(cel-1)] == 0.:
                    contacts[PN == cel]   += 1
                    contacts[PN == lef] += 1
                    contact_matrix[int(lef-1),int(cel-1)] += 1.
                    contact_matrix[int(cel-1),int(lef-1)] += 1.
                if rit > 0. and rit != cel and contact_matrix[int(rit-1),int(cel-1)] == 0.:
                    contacts[PN == cel]   += 1
                    contacts[PN == rit] += 1
                    contact_matrix[int(rit-1),int(cel-1)] += 1.
                    contact_matrix[int(cel-1),int(rit-1)] += 1.
    A = np.mean(contacts)

    return A, contact_matrix

def populate_materials(SHAPENO,X,Y,MATS,n,mixed=True,info=False): 
    """
    Function populates the materials meshes. Must be used after material assignment to particles before
    any block material assignment, e.g. a matrix.

    SHAPENO : The indexes of each shape within mesh_Shps
    X       : The xcoord of the shape centre (in cells)
    Y       : The ycoord of the shape centre (in cells)
    MATS    : The array of each corresponding material number to each particle
    n       : The total number of particles
    mixed   : Whether cells can have multiple materials or not (yes => True, no => False)
    
    NB This function will remake the mesh but in the correct materials spaces.
    """
    global mesh, mesh_Shps,meshx,meshy,FRAC,OBJID,materials,trmesh
    for k in range(n):
        mmm = MATS[k]
        if info:
            place_shape(mesh_Shps[SHAPENO[k]],X[k],Y[k],mmm,Mixed=mixed,info=info,shapeno=k+1)
        else:
            place_shape(mesh_Shps[SHAPENO[k]],X[k],Y[k],mmm,Mixed=mixed)
    return


def save_general_mesh(fname='meso_m.iSALE',mixed=False,noVel=False,info=False):
    """
    A function that saves the current mesh as a text file that can be read, verbatim into iSALE.
    This compiles the integer indices of each cell, as well as the material in them and the fraction
    of matter present. It saves all this as the filename specified by the user, with the default as 
    meso_m.iSALE
    
    This version of the function works for continuous and solid materials, such as a multiple-plate setup.
    It does not need to remake the mesh as there is no particular matter present.
    
    fname   : The filename to be used for the text file being used
    mixed   : Are mixed cells used?
    noVel   : Does not include velocities in meso_m.iSALE file
    info    : Include particle ID (i.e. #) as a column in the final file 
    
    returns nothing but saves all the info as a txt file called 'fname' and populates the materials mesh.
    """
    global mesh, mesh_Shps,meshx,meshy,FRAC,OBJID,materials,Ms,trmesh,VX_,VY_
    if info:
        OI    = np.zeros((meshx*meshy))
        PI    = np.zeros((meshx*meshy))

    XI    = np.zeros((meshx*meshy))    
    YI    = np.zeros((meshx*meshy))
    UX    = np.zeros((meshx*meshy))
    UY    = np.zeros((meshx*meshy))
    K     = 0
    #materials          = materials[:,::-1,:]    #Reverse array vertically, as it is read into iSALE upside down otherwise
    #VX_                = VX_[::-1,:]
    #VY_                = VY_[::-1,:]
    for i in range(meshx):
        for j in range(meshy):
            XI[K] = j
            YI[K] = i
            UX[K] = VX_[j,i]
            UY[K] = VY_[j,i]
            if info:
                PI[K] = part_no[j,i]
            for mm in range(Ms):
                FRAC[mm,K] = materials[mm,j,i]
                #OBJID[mm,K]= objects[mm,j,i]                                                        # each particle number
            K += 1
    FRAC = check_FRACs(FRAC,mixed)
    HEAD = '{},{}'.format(K,Ms)
    if noVel:
        ALL  = np.column_stack((XI,YI,FRAC.transpose()))                                                # ,OBJID.transpose())) Only include if particle number needed
    elif info:
        ALL  = np.column_stack((XI,YI,UX,UY,FRAC.transpose(),PI))                                                # ,OBJID.transpose())) Only include if particle number needed
    elif info and noVel:
        ALL  = np.column_stack((XI,YI,FRAC.transpose(),PI))                                                # ,OBJID.transpose())) Only include if particle number needed
    else:
        ALL  = np.column_stack((XI,YI,UX,UY,FRAC.transpose()))                                                # ,OBJID.transpose())) Only include if particle number needed
    np.savetxt(fname,ALL,header=HEAD,fmt='%5.3f',comments='')
    return


def populate_from_bmp(A):
    """
    Function that populates the materials meshes from an array of grayscale values
    (0-255), typically derived from a BMP image, but others will work.
    Different shades are treated as different materials, however, white is ignored
    and treated as 'VOID'. This puts a cap of 255 on the maximum number of materials possible
    in a simulation.
    ***********************************************************
    **NB This function does not yet generate a velocity field**
    ***********************************************************
    --------------------------------------------------------------------------
    |args      |  Meaning                                                    | 
    --------------------------------------------------------------------------
    |A         |  2D array of grayscale integer                              |
    |          |  values: (0 - 255)                                          |
    --------------------------------------------------------------------------
    """
    global materials,Ms,FRAC
    fname = 'meso_m.iSALE'
    #A = A[:,::-1]
    ny, nx = np.shape(A)
    ms    = np.unique(A[A!=255])    #white is considered 'VOID' and should not be included
    Nms   = np.size(ms)
    generate_mesh(nx,ny,mat_no=Nms)
    XI    = np.zeros((nx*ny))    
    YI    = np.zeros((nx*ny))
    K     = 0
    for i in range(nx):
        for j in range(ny):
            XI[K] = i
            YI[K] = j
            M     = 0
            for item in ms:
                if A[j,i] == item:
                    FRAC[M,K] = 1. 
                else:
                    FRAC[M,K] *= 0.
                M += 1
            K += 1
        
    HEAD = '{},{}'.format(np.size(XI),Ms)
    ALL  = np.column_stack((XI,YI,FRAC.transpose()))                                                # ,OBJID.transpose())) Only include if particle number needed
    
    fig, ax = plt.subplots()
    cax = ax.imshow(A,cmap='Greys',interpolation='nearest')#,vmin=0,vmax=1)
    cbar = fig.colorbar(cax, orientation='horizontal')
    plt.show()
    np.savetxt(fname,ALL,header=HEAD,fmt='%5.3f',comments='')
    return

def check_FRACs(FRAC,mixed):
    """
    This function checks all the volume fractions in each cell and deals with any occurrences where they add to more than one
    by scaling down ALL fractions in that cell, such that it is only 100% full.
    
    FRAC : Array containing the full fractions in each cell of each material
    """
    
    global Ms,meshx,meshy
    if mixed==True:
        for i in range(meshx*meshy):
            SUM = np.sum(FRAC[:,i])
            if SUM > 1.:
                FRAC[:,i] /= SUM
            else:
                pass
    else:
        for i in range(meshx*meshy):
            SUM = np.sum(FRAC[:,i])
            if SUM > 1.:
                done = False
                for j in range(Ms):
                    if FRAC[j,i] > 0 and done == False: 
                        FRAC[j,i] = 1.
                        done = True
                    else:
                        FRAC[j,i] = 0.
            else:
                pass
    
    return FRAC

def fill_plate(y1,y2,mat):
    """
    This function creates a 'plate' structure across the mesh, filled with the material of your choice. Similar to PLATE in iSALE
    It fills all cells between y1 and y2. if material -1 is used, this is treated as void and overwrites previous materials; 
    additionally setting their velocities to 0.
    """
    global meshx,meshy,materials,mesh,GRIDSPC
    assert y2>y1, 'ERROR: 2nd y value is less than the first, the function accepts them in ascending order' 
    # overwrite = 1. if material -1 is used => VOID
    ovrwt = 1
    if mat == -1: ovrwt = 0
    for j in range(meshy):
        pos = j*GS
        if pos <= y2 and pos >= y1:
            present_mat = np.sum(materials[:,j,:],axis=0)                                        # This ensures that plates override in the order they are placed.
            materials[mat-1,j,:] = (1. - present_mat)*(ovrwt)
            mesh[j,:]            = (1. - present_mat)*(ovrwt)
            if ovrwt == 0: 
                materials[:,j,:] = 0.
                VX_[j,:] = 0.
                VY_[j,:] = 0.
    return

def fill_Allmesh(mat,overwrite=False):
    """
    This function fills the entire mesh with a material. does NOT overwrite by default
    """
    global meshx,meshy,materials,mesh
    if overwrite:
        # Erase all materials
        materials[:,:,:] = 0.
        # fill with chosen material
        materials[mat-1,:,:] = 1.
    else:
        # sum across material axes
        present_mat = np.sum(materials,axis=0)                      
        # If cell [i,j] is full, present_mat[i,j] = 1., so mat_tofill[i,j] = 0.
        mat_tofill  = 1. - present_mat
        # Fill chosen material mesh with the appropriate quantities
        materials[mat-1,:,:] = mat_tofill 
    return

def rectangle_material_vel(L1,L2,T1,T2,mat,vel,dim=1):
    """
    A combination of rectangle and material vel. Gives all cells of material mat, in box constrained by
    (L1,T1) and (L2,T2), a velocity vel in direction dim.
    """
    global VX_,VY_,materials
    assert L2>L1, 'ERROR: 2nd L value is less than the first, the function accepts them in ascending order' 
    assert T2>T1, 'ERROR: 2nd T value is less than the first, the function accepts them in ascending order' 
    assert dim == 0 or dim ==1, 'ERROR: Dimension must be 0 or 1, x or y'
    if dim == 0:
        VX_[(XX<=L2)*(XX>=L1)*(YY<=T2)*(YY>=T1)*(materials[mat-1]>0.)] = vel #- np.sum(materials,axis=0)  
    elif dim == 1:
        VY_[(XX<=L2)*(XX>=L1)*(YY<=T2)*(YY>=T1)*(materials[mat-1]>0.)] = vel #- np.sum(materials,axis=0)  
    return

def material_vel(mat,vel,dim=1):
    """
    This function gives all the cells containing material mat, a velocity vel, in direction dim
    """
    global VX_,VY_,materials
    assert dim == 0 or dim ==1, 'ERROR: Dimension must be 0 or 1, x or y'
    if dim == 0:
        VX_[materials[mat-1]>0.] = vel #- np.sum(materials,axis=0)  
    elif dim == 1:
        VY_[materials[mat-1]>0.] = vel #- np.sum(materials,axis=0)  
    return

def rectangle_vel(L1,T1,L2,T2,vel,dim=1):
    """
    This function gives all the cells within the box (defined by (L1,T1) and (L2,T2) a velocity vel in direction dim
    provided they contain some material! Void is given no velocity.
    """
    global VX_,VY_,materials
    assert L2>L1, 'ERROR: 2nd L value is less than the first, the function accepts them in ascending order' 
    assert T2>T1, 'ERROR: 2nd T value is less than the first, the function accepts them in ascending order' 
    assert dim == 0 or dim ==1, 'ERROR: Dimension must be 0 or 1, x or y'
    if dim == 0:
        VX_[(XX<=L2)*(XX>=L1)*(YY<=T2)*(YY>=T1)*(np.sum(materials,axis=0)>0.)] = vel #- np.sum(materials,axis=0)  
    elif dim == 1:
        VY_[(XX<=L2)*(XX>=L1)*(YY<=T2)*(YY>=T1)*(np.sum(materials,axis=0)>0.)] = vel #- np.sum(materials,axis=0)  
    return

def plate_vel(L1,L2,vel,dim=1):
    """
    This function gives all the cells within the box (defined by (L1,T1) and (L2,T2) a velocity vel in direction dim
    provided they contain some material! Void is given no velocity.
    """
    global VX_,VY_,materials
    assert L2>L1, 'ERROR: 2nd L value is less than the first, the function accepts them in ascending order' 
    assert dim == 0 or dim ==1, 'ERROR: Dimension must be 0 or 1, x or y'
    if dim == 0:
        VX_[(YY<=L2)*(YY>=L1)*(np.sum(materials,axis=0)>0.)] = vel #- np.sum(materials,axis=0)  
    elif dim == 1:
        VY_[(YY<=L2)*(YY>=L1)*(np.sum(materials,axis=0)>0.)] = vel #- np.sum(materials,axis=0)  
    return

def material_alpha(mat,alp):
    """
    This function gives all the cells containing material mat, a distension: alp
    """
    global alpha,materials
    assert dim == 0 or dim ==1, 'ERROR: Dimension must be 0 or 1 (x or y)'
    alpha[materials[mat-1]>0.] = alp #- np.sum(materials,axis=0)  
    return

def material_alpha_Fof(mat,f,dim=0):
    """
    This function gives all the cells containing material mat, a distension: alp = f(r)
    if dim = 0, r = x
    if dim = 1, r = y
    """
    global alpha,materials,XX,YY
    assert dim == 0 or dim ==1, 'ERROR: Dimension must be 0 or 1 (x or y)'
    if dim != 0: 
        alpha[materials[mat-1]>0.] = f(YY[materials[mat-1]>0.]) #- np.sum(materials,axis=0)  
    else:
        alpha[materials[mat-1]>0.] = f(XX[materials[mat-1]>0.]) #- np.sum(materials,axis=0)  
    return

def material_alpha_Fofxy(mat,f,dim=0):
    """
    This function gives all the cells containing material mat, a distension: alp = f(x,y)
    """
    global alpha,materials,XX,YY
    alpha[materials[mat-1]>0.] = f(XX[materials[mat-1]>0.],YY[materials[mat-1]>0.]) #- np.sum(materials,axis=0)  
    return

def fill_rectangle(L1,T1,L2,T2,mat):
    """
    This function creates a 'plate' structure across the mesh, filled with the material of your choice. Similar to PLATE in iSALE
    It fills all cells between y1 and y2.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~if mat == -1. this fills the cells with VOID and overwrites everything.~
    ~and additionally sets all velocities in those cells to zero            ~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    global meshx,meshy,materials,mesh,trmesh,VX_,VY_,Ms,XX,YY
    assert L2>L1, 'ERROR: 2nd L value is less than the first, the function accepts them in ascending order' 
    assert T2>T1, 'ERROR: 2nd T value is less than the first, the function accepts them in ascending order' 
    if mat == -1:
        for mm in range(Ms):
            materials[mm][(XX<=L2)*(XX>=L1)*(YY<=T2)*(YY>=T1)] *= 0.
        VX_[(XX<=L2)*(XX>=L1)*(YY<=T2)*(YY>=T1)] *= 0.
        VY_[(XX<=L2)*(XX>=L1)*(YY<=T2)*(YY>=T1)] *= 0.
        part_no[(XX<=L2)*(XX>=L1)*(YY<=T2)*(YY>=T1)] *= 0.
    else:
        temp_materials = np.copy(materials[int(mat)-1])
        temp_materials[(XX<=L2)*(XX>=L1)*(YY<=T2)*(YY>=T1)*(np.sum(materials,axis=0)<1.)] = 1. #- np.sum(materials,axis=0)  
        temp_2 = np.sum(materials,axis=0)*temp_materials
        temp_materials -= temp_2
        materials[int(mat)-1] += temp_materials
    return
def fill_polygon(L,T,mat):
    """
    This function fills all cells within a polygon defined by the vertices in arrays L and T
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~if mat == -1. this fills the cells with VOID and overwrites everything.~
    ~and additionally sets all velocities in those cells to zero            ~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    global meshx,meshy,materials,mesh,trmesh,VX_,VY_,Ms,XX,YY,GS
    path = mpath.Path(np.column_stack((L,T)))
    print np.column_stack((L,T))
    # find the box that entirely encompasses the polygon
    L1 = np.amin(L) 
    L2 = np.amax(L)
    T1 = np.amin(T)
    T2 = np.amax(T)
    
    # find the coordinates of every point in that box
    Xc_TEMP = XX[(XX<=L2)*(XX>=L1)*(YY<=T2)*(YY>=T1)]
    Yc_TEMP = YY[(XX<=L2)*(XX>=L1)*(YY<=T2)*(YY>=T1)]

    # store all indices of each coord that is in the polygon in two arrays
    x_success = np.ones_like(Xc_TEMP,dtype=int)*-9999
    y_success = np.ones_like(Yc_TEMP,dtype=int)*-9999

    # cycle through all these points
    # store successes in arrays
    k = 0
    for x, y in zip(Xc_TEMP,Yc_TEMP):
        in_shape = path.contains_point([x,y])
        if in_shape:
            x_success[k]   = np.where((XX==x)*(YY==y))[0][0] 
            y_success[k]   = np.where((XX==x)*(YY==y))[1][0]
            k += 1
    x_suc = x_success[x_success!=-9999] #np.ma.masked_where(x_success==-9999,x_success)
    y_suc = y_success[y_success!=-9999] #np.ma.masked_where(y_success==-9999,y_success)
    #fig = plt.figure()
    #ax = fig.add_subplot(111,aspect='equal')
    #ax.plot(XX[(XX==x_suc)],YY[(YY==y_suc)],linestyle=' ',marker='.')
    #ax.plot(L,T)
    #ax.set_xlim(L1,L2)
    #ax.set_ylim(T1,T2)
    #plt.show()

    if mat == -1:
        # select the necessary material using new arrays of indices
        for mm in range(Ms):
           materials[mm][x_suc,y_suc] *= 0.
        VX_[x_suc,y_suc] *= 0.
        VY_[x_suc,y_suc] *= 0.
        part_no[x_suc,y_suc] *= 0.
    else:
        temp_materials = np.copy(materials[int(mat)-1,x_suc,y_suc])
        temp_materials[(np.sum(materials[:,x_suc,y_suc],axis=0)<1.)] = 1. #- np.sum(materials,axis=0)  
        temp_2 = np.sum(materials[:,x_suc,y_suc],axis=0)*temp_materials
        temp_materials -= temp_2
        materials[int(mat)-1,x_suc,y_suc] += temp_materials
    return
def overwrite_rectangle(L1,T1,L2,T2,mat):
    """
    This function creates a 'plate' structure across the mesh, filled with the material of your choice. Similar to PLATE in iSALE
    It fills all cells between y1 and y2. This version overwrites any material already placed, unlike fill-rectangle, which does not
    """
    global meshx,meshy,materials,mesh,xh,yh,trmesh
    assert L2>L1, 'ERROR: 2nd L value is less than the first, the function accepts them in ascending order' 
    assert T2>T1, 'ERROR: 2nd T value is less than the first, the function accepts them in ascending order' 
    temp_materials = np.copy(materials[mat-1])
    for mm in range(Ms):
        materials[mm][(XX<=L2)*(XX>=L1)*(YY<=T2)*(YY>=T1)] = 0. #- np.sum(materials,axis=0)  
    materials[mat-1][(XX<=L2)*(XX>=L1)*(YY<=T2)*(YY>=T1)] = 1. #- np.sum(materials,axis=0)  
    part_no[(XX<=L2)*(XX>=L1)*(YY<=T2)*(YY>=T1)] = 0.
    return

def fill_sinusoid(L1,T1,A,w,p,c,L2,mat,mixed=False,tracers=False,ON=None):
    def sinfunc(D,A,w,p):
        theta = (2.*np.pi*D/w) + p 
        E = A*np.sin(theta) + c
    return E
    """
    This function creates a 'plate' structure across the mesh, filled with the material of your choice. Similar to PLATE in iSALE
    It fills all cells between y1 and y2.
    """
    global meshx,meshy,materials,mesh,xh,yh,mats,trmesh
    #assert L2>L1, 'ERROR: 2nd L value is less than the first, the function accepts them in ascending order' 
    assert L2>L1, 'ERROR: 2nd L value is less than the first, the function accepts them in ascending order' 
    for j in range(meshy):
        for i in range(meshx):
            Tc = 0.5*(yh[j] + (yh[j+1]))                        
            Lc = 0.5*(xh[i] + (xh[i+1]))                                                                    
            dx = abs(xh[i+1] - xh[i])
            if Tc <= sinfunc(Lc) and Tc >= T1 and Lc >= L1 and Lc <= L2:
                #materials[:,i,j]     = 0.                                                            # This ensures that plates override in the order they are placed.
                if mixed:
                    ll = np.linspace(xh[i],xh[i+1],11)                                                        # Split into 10x10 grid and fill, depending on these values.
                    tt = np.linspace(yh[j],yh[j+1],11)                        
                    counter = 0
                    for I in range(10):
                        for J in range(10):
                            llc = (ll[I] + ll[I+1])/2.
                            ttc = (tt[J] + tt[J+1])/2.
                            if ttc <= sinfunc(llc) and llc >= L1 and ttc >= T1 and llc <= L2:
                                counter += 1
                    present_mat = np.sum(materials[:,j,i])
                    new_mat     = counter*.1**2.
                    space_left = 1.-present_mat
                    new_mat = min(new_mat,space_left)
                    materials[mat-1,j,i] += new_mat
                    if tracers and new_mat>0.5: trmesh[j,i] = ON


                else:
                    present_mat = np.sum(materials[:,j,i])
                    if present_mat == 0.:
                        materials[mat-1,j,i] = 1.
                        mesh[j,i]            = 1.
                    else:
                        pass
            elif mixed and abs(Tc-sinfunc(Lc)) <= 2.*dx and Lc >= L1 and Tc >= T1 and Lc <= L2:
                ll = np.linspace(xh[i],xh[i+1],11)                                                        # Split into 10x10 grid and fill, depending on these values.
                tt = np.linspace(yh[j],yh[j+1],11)                        
                counter = 0
                for I in range(10):
                    for J in range(10):
                        llc = (ll[I] + ll[I+1])/2.
                        ttc = (tt[J] + tt[J+1])/2.
                        if ttc <= sinfunc(llc) and llc >= L1 and ttc >= T1 and llc <= L2:
                            counter += 1
                present_mat = np.sum(materials[:,j,i])
                new_mat     = counter*.1**2.
                space_left = 1.-present_mat
                new_mat = min(new_mat,space_left)
                materials[mat-1,j,i] += new_mat
                #if tracers and new_mat>0.5: trmesh[j,i] = ON
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
    
    global meshx,meshy,materials,mesh,trmesh
    AREA = 0.
    if mixed == True:                                                                                    # If there are to be mixed cells, then set MIX to 1.
        MIX = 1.
    else:
        MIX = 0.
    x2 = r2[0]
    x1 = r1[0]
    y2 = r2[1]
    y1 = r1[1]
    assert x1!=x2,'ERROR: x2 is equal to x1, this is a vertical line, use fill_left or fill_right'
    M = (y2-y1)/(x2-x1)                                                                                    # Calculate the equation of the line between x1,y1 and x2,y2
    C = y1 - M*x1
    for i in range(meshx):
        for j in range(meshy):
            xc = 0.5*(i + (i+1))                                                                    
            yc = 0.5*(j + (j+1))                        
            A = (yc - C)/M        
            #A = M*xc + C
            if (j+.5) < y2 and (j+.5) > y1:
                if invert == False:                                                                        # If the fill is not inverted do below
                    if (xc-A) < -1.*np.sqrt(2.)*MIX:                                                    # If the cell centre falls under this line then fill fully
                        present_mat = np.sum(materials[:,j,i])
                        new_mat     = 1.
                        space_left  = 1.-present_mat
                        new_mat     = min(new_mat,space_left)
                        materials[mat-1,j,i] += new_mat
                    elif abs(xc-A) <= np.sqrt(2.) and mixed == True:                                    # If mixed cells wanted and the cell is within the necessary range
                        xx = np.linspace(i,i+1,11)                                                        # Split into 10x10 grid and fill, depending on these values.
                        yy = np.linspace(j,j+1,11)                        
                        counter = 0
                        for I in range(10):
                            for J in range(10):                            
                                xxc = 0.5*(xx[I] + xx[J+1])        
                                yyc = 0.5*(yy[J] + yy[J+1])
                                #A = M*xxc + C                            
                                A = (yyc-C)/M
                                if (xxc-A) < 0.:                
                                    counter += 1
                                    mesh[j,i]            += (.1**2.)                
                        present_mat = np.sum(materials[:,j,i])
                        new_mat     = counter*.1**2.
                        space_left  = 1.-present_mat
                        new_mat     = min(new_mat,space_left)
                        materials[mat-1,j,i] += new_mat
                elif invert == True:
                    if (xc-A) > np.sqrt(2.)*MIX:                
                        mesh[j,i]            = 1.0                                                        # If MIX=0. then this will do it for all cells
                        materials[mat-1,j,i] = 1.0
                    elif abs(xc-A) <= np.sqrt(2) and mixed == True:                    
                        xx = np.linspace(i,i+1,11)                        
                        yy = np.linspace(j,j+1,11)                        
                        counter = 0
                        for I in range(10):
                            for J in range(10):                            
                                xxc = 0.5*(xx[I] + xx[J+1])        
                                yyc = 0.5*(yy[J] + yy[J+1])
                                #A = M*xxc + C                            
                                A = (yyc-C)/M
                                if (xxc-A) > 0.:                
                                    counter += 1
                                    mesh[j,i]            += (.1**2.)                
                        present_mat = np.sum(materials[:,j,i])
                        new_mat     = counter*.1**2.
                        space_left  = 1.-present_mat
                        new_mat     = min(new_mat,space_left)
                        materials[mat-1,j,i] += new_mat
            else:
                pass
    return

def fill_arbitrary_shape(X,Y,mat,tracers=False,ON=None):                        
    """
    Function to fill an arbitrary shape in the mesh based on arrays of vertices.
    This version does NOT partially fill cells.
    NB for this to work the coordinates of the vertices MUST be relative to the centre of the object.

    X, Y      : Vertices in cells
    """
    global mesh, materials,meshx,meshy,trmesh                                                                    # Only the angles used are now randomly selected.
    N      = np.size(X)
    R      = np.zeros((2,N))                                                                            # Array for the coords of the vertices
    x0,y0 = find_centroid(X,Y)
    R[0,:] = X - x0                                                                                        # Each vertex will also be successive, such that drawing a line between
    R[1,:] = Y - y0                                                                                        # each one in order, will result in no crossed lines.
    qx = 0.                                                                                                # Make the reference point (q) zero, i.e. the centre of the shape
    qy = 0.                                                                                                # All dimensions are in reference to the central coordinates.
    for j in range(meshy):                                                                                # Iterate through all the x- and y-coords
        for i in range(meshx):                                                                            
            xc = 0.5*(i+(i+1))-x0                                                                        # Convert current coord to position relative to (x0,y0) 
            yc = 0.5*(j+(j+1))-y0                                                                        # Everything is now in indices, with (x0,y0)
            sx = xc - qx                                                                                # s = vector difference between current coord and ref coord
            sy = yc - qy        
            intersection = 0                                                                            # Initialise no. intersections as 0
            for l in range(N-1):                                                                        # cycle through each edge, bar the last
                rx = R[0,l+1] - R[0,l]                                                                    # Calculate vector of each edge (r), 
                                                                                                        # i.e. the line between the lth vertex and the l+1th vertex
                ry = R[1,l+1] - R[1,l]
                RxS = (rx*sy-ry*sx)                                                                        # Vector product of r and s (with z = 0), 
                                                                                                        # technically produces only a z-component
                if RxS!=0.:                                                                                # If r x s  = 0 then lines are parallel
                    t = ((qx-R[0,l])*sy - (qy-R[1,l])*sx)/RxS
                    u = ((qx-R[0,l])*ry - (qy-R[1,l])*rx)/RxS    
                    if t<=1. and t>=0. and u<=1. and u>=0.:
                        intersection = intersection + 1
            rx = R[0,0] - R[0,N-1]                                                                        # Do the last edge. 
                                                                                                        # Done separately to avoid needing a circular 'for' loop
            ry = R[1,0] - R[1,N-1]
            if (rx*sy-ry*sy)!=0.:
                RxS = (rx*sy-ry*sx)
                t = ((qx-R[0,N-1])*sy - (qy-R[1,N-1])*sx)/RxS
                u = ((qx-R[0,N-1])*ry - (qy-R[1,N-1])*rx)/RxS
                if t<=1. and t>=0. and u<=1. and u>=0.:
                    intersection = intersection + 1
            if (intersection%2==0.):                                                                    # If number of intersections is divisible by 2 (or just zero) 
                present_mat = np.sum(materials[:,j,i])
                new_mat     = 1.
                space_left  = 1.-present_mat
                new_mat     = min(new_mat,space_left)
                materials[mat-1,j,i] += new_mat
                if tracers and new_mat>0.5: trmesh[j,i] = ON

    return


def find_centroid(x,y):
    """
    Simple function to return the centroid coordinate of an arbitrary, non-intersecting polygon.
    with n vertices. see https://en.wikipedia.org/wiki/Centroid#Locating_the_centroid for further
    information.
    """
    if ((x[0],y[0]) != (x[-1],y[-1])):
        x = np.append(x,x[0])
        y = np.append(y,y[0])
    else:
        pass
    rectangle = False
    if ((x[0],x[1]) != (x[2],x[3]) and (y[0],y[1]) == (y[2],y[3]) or ((y[0],y[1]) != (y[2],y[3]) and (x[0],x[1]) == (x[2],x[3]))): rectangle = True

    n = np.size(x)
    A  = 0
    Cx = 0
    Cy = 0
    if rectangle:
        Cx = (np.amax(x)+np.amin(x))/2.
        Cy = (np.amax(y)+np.amin(y))/2.
    else:
        for i in range(n-1):
            A += (x[i]*y[i+1]-x[i+1]*y[i])*.5
            print A
        for j in range(n-1):
            Cx += (x[j]+x[j+1])*(x[j]*y[j+1]-x[j+1]*y[j])
            Cy += (y[j]+y[j+1])*(x[j]*y[j+1]-x[j+1]*y[j])
        
        Cx /= (6.*A)
        Cy /= (6.*A)
    return Cx,Cy


def smooth_mesh(MMM):
    ny, nx = np.shape(MMM)
    edges = np.zeros_like(MMM)

    for i in range(nx-1):
        for j in range(ny-1):
            if MMM[j-1,i] >= .99: edges[j,i] += 1.
            if MMM[j+1,i] >= .99: edges[j,i] += 1.
            if MMM[j,i-1] >= .99: edges[j,i] += 1.
            if MMM[j,i+1] >= .99: edges[j,i] += 1.
    MMM[(edges<2)*(MMM>0.)] *= .5 
    MMM[edges==2] = 0
    MMM[(edges>2.)*(MMM<1.)] += .5
    MMM[MMM>1.] = 1.

    #plt.figure()
    #plt.imshow(MMM,vmin=0,vmax=1,cmap='binary',interpolation='nearest')
    #plt.show()

    return MMM
def shape_info(fname):
    """
    This function calculates the Elongation, area ratio and geologic 'roundness'
    of a given shape, from its vertices.

    returns:

    A_ratio    = (Area of shape)/(Area of largest box containing it)
    Elongation = ratio of long side to short side of largest box containing shape
    Roundness  = (average radius of curvature)/(smallest inscribed circle radius)
    """
    VX    = np.genfromtxt(fname,comments='#',usecols=0,delimiter=',')
    VY    = np.genfromtxt(fname,comments='#',usecols=1,delimiter=',')
    V     = np.genfromtxt(fname,comments='#',usecols=(0,1),delimiter=',')
    
    
    nv    = np.size(VX)

    vxmax = np.amax(VX)
    vxmin = np.amin(VX)
    xspan = abs(vxmax-vxmin)
    
    vymax = np.amax(VY)
    vymin = np.amin(VY)
    yspan = abs(vymax-vymin)
    L     = max(xspan,yspan)
    B     = min(xspan,yspan)

    VX /= L
    VY /= L
    V  /= L
    L  /= L
    B  /= L

    elongation = L/B
    A_ratio    = polygon_area(VX,VY)/(L*B) 
    a = 0.
    for i in range(nv-1):
        I1 = i-1
        I2 = i
        I3 = i+1
        
        if i == 0:    I1 = -1

        # Vectors to the midpoint of each face connecting vertex i
        M1 = np.array([(VX[I1]+VX[I2])/2.,(VY[I1]+VY[I2])/2.])
        M2 = np.array([(VX[I2]+VX[I3])/2.,(VY[I2]+VY[I3])/2.])
        
        # Vector magnitudes
        m1 = np.linalg.norm(M1)
        m2 = np.linalg.norm(M2)
        
        # Normalise the two midpoint vectors
        U1 = M1/m1
        U2 = M2/m2
        
        # Their average magnitude
        m  = (m1+m2)/2.
        # New vectors for use in calculation
        N1 = V[i] + U1*m
        N2 = V[i] + U2*m

        # Length of the chord between them
        S  = np.linalg.norm(N1-N2)
        
        # Calculate angle between N1 & N2
        cost = np.sum(N1*N2)/(np.linalg.norm(N1)*np.linalg.norm(N2))

        # Do not allow angles > pi
        thet = np.arccos(cost)
        if thet > np.pi: thet -= np.pi
        alph = np.pi-thet
        
        # radius = rad of circle that is tangential at N1 & N2
        a += S*.5/(np.sin(alph/2.))
    ac = a/float(nv-1)
    q  = B/3.
    roundness = ac/q

    return elongation,A_ratio,roundness

def display_mesh(showvel=False):
    global mesh, materials, Ms, meshx, meshy
    fig = plt.figure()
    if showvel:
        ax1 = fig.add_subplot(131,aspect='equal')
        ax2 = fig.add_subplot(132,aspect='equal')
        ax3 = fig.add_subplot(133,aspect='equal')
        vL_ = ax2.pcolormesh(XX,YY,VY_,cmap='bone',vmin=np.amin(VY_),vmax=np.amax(VY_))
        vT_ = ax3.pcolormesh(XX,YY,VX_,cmap='bone',vmin=np.amin(VY_),vmax=np.amax(VY_))
        ax2.set_title('Longitudinal Vel')
        ax3.set_title('Transverse Vel')
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right',size='5%',pad=0.05)
        cb = fig.colorbar(vL_,cax=cax,orientation='vertical')
        cb.set_label('Velocity  [ms$^{-1}$]')

    else:
        ax1 = fig.add_subplot(111,aspect='equal')
    for KK in range(Ms):
        matter = np.copy(materials[KK,:,:])*(KK+1)
        matter = np.ma.masked_where(matter==0.,matter)
        ax1.imshow(matter, cmap='copper',vmin=0,vmax=Ms,interpolation='nearest')
    #ax1.set_xlim(0,np.amax(XX))
    #ax1.set_ylim(0,np.amax(YY))
    ax1.set_xlim(0,meshx)
    ax1.set_ylim(0,meshy)

    #ax1.set_title('Mesh')
    ax1.set_title('Materials')
    fig.tight_layout()
    plt.show()
    return

def Fabric_Tensor_disks(ic,jc,r,tolerance=0.):
    """
    This function calculates the fabric tensor of a 2D particle ensemble consisting of
    perfectly circular disks. They do NOT have to be identical in size!

    input:

    ic = x-coordinates of each particle centre [float,array]
    jc = y-coordinates of each particle centre [float,array]
    a  = corresponding radii of each particle  [float,array]

    output:

    Z  = The coordination number (called A, in the past by myself)
    A  = The Fabric Anisotropy
    F  = The fabric tensor (2nd order)
    """
    # simple function to compute all possible differences between elements in an array
    def difference_matrix(a):
        x = np.reshape(a, (len(a), 1))
        return x - x.transpose()
    # This function similarly finds all possible combinations of elements in an array.
    def addition_matrix(a):
        x = np.reshape(a, (len(a), 1))
        return x + x.transpose()

    # Number of particles
    Np = np.size(ic)
    # Maximum number of possible contacts (no particle can have more than 6 each)
    nc = np.zeros((Np*6))
    # two arrays; one with all possible x-distances bewteen particles (li)
    # one with all possible y-distances (lj)
    # NB each has a shape of Np x Np and the matrices are 'flipped' sign-wise along the diagonal
    # and the diagonals are zeros
    li = difference_matrix(ic)
    lj = difference_matrix(jc)
    # s is all possible combinations of radii in the same order. i.e. 
    # elements of s are the minimum corresponding distances required for a 'contact'
    # Add a tolerance because we are not necessarily working with perfect circles, there is 
    # an uncertainty on the minimum length required for a contact 
    s  = addition_matrix(r) + tolerance
    # The magnitude (length) of each branch vector is
    L  = np.sqrt(li**2. + lj**2.)
    L[L==0] = 1.
    # normalise all branch vectors
    li /= L 
    lj /= L
    # set all branch vectors that are not contacts equal to 9999.
    li[L>=s] = 9999.
    lj[L>=s] = 9999.
    # Remove any branches of zero length (impossibilities) and flatten the arrays
    # NB every contact appears twice in each array!! Each has a size of 2*Nc (No. contacts)
    ni = li[li!=9999.]
    nj = lj[lj!=9999.]
    
    F = np.zeros((2,2))
    F[0,0] = np.sum(ni**2.)/float(Np)
    F[1,1] = np.sum(nj**2.)/float(Np)
    
    F[0,1] = np.sum(ni*nj)/float(Np)
    F[1,0] = F[0,1]

    Z = F[0,0] + F[1,1]
    A = F[0,0] - F[1,1]
    return Z, A, F

def view_all_materials(save=False,fname='targetbed.png'):
    global Ms,materials,meshx,meshy

    fig = plt.figure()
    ax  = fig.add_subplot(111,aspect='equal')
    for KK in range(Ms):
        matter = np.copy(materials[KK,:,:])*(KK+1)
        matter = np.ma.masked_where(matter==0.,matter)
        ax.imshow(matter, cmap='plasma',vmin=0,vmax=Ms,interpolation='nearest')
    ax.set_xlim(0,meshx)
    ax.set_ylim(0,meshy)
    if save: fig.savefig(fname,bbox_inches='tight',dpi=300)
    plt.show()
    return
