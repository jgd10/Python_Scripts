import pySALESetup as pss
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import glob

def flip_grain(mesh0):
    flip_options = [0,1,2,3]
    F            = random.choice(flip_options)
    if F == 0 or F == 2:
        pass
    elif F == 1:
        mesh0 = mesh0[::-1,:]
    elif F == 3:
        mesh0 = mesh0[:,::-1]
    return mesh0

def polygon_area(X,Y):
    N = np.size(X)
    A = 0
    for i in range(1,N):
        A += (X[i-1]*Y[i]-X[i]*Y[i-1])*.5
    return abs(A)

def function(x):
    A = 2.908
    B = 0.028
    C = 0.320
    L = 0.643
    a = 99.4
    return (1./a)*A/(B+C*np.exp(-L*x))

def lunar_pdf(x,x_tol):                                                   
    P     = abs(function(x+x_tol) - function(x-x_tol))
    return P
def PHI_(x):
    return -1.*np.log2(x)

def DD_(x):
    return 0.5*2.**(-x)

vol_frac   = .5
Y_cells    = 750 
X_cells    = 750 
y_length   = 5.e-3                                                 # Transverse width of bed
x_length   = 5.e-3                                                 # Longitudinal width of bed
cppr       = 25                                                     # Cells per particle radius
LB_cppr    = 5.                                                     # Lowest allowed cppr
GRIDSPC    = x_length/X_cells                                       # Physical distance/cell
mat_no     = 7

#############################
largest_grain = 5e-3
smallst_grain = 10e-6
largest_cppr  = (largest_grain/GRIDSPC)
smallst_cppr  = max((smallst_grain/GRIDSPC),LB_cppr)
print smallst_cppr
refrnce_cppr  = float(cppr)
large_frac    = largest_cppr/refrnce_cppr
small_frac    = smallst_cppr/refrnce_cppr
#############################

PR            = (small_frac,large_frac)                                              # Particle size range
print PR
A_est         = vol_frac*X_cells*Y_cells #/(np.pi*(.25 * 1.e-3 * 2.**(-2.6))**2.)
A_total       = y_length*x_length


pss.generate_mesh(X=X_cells,Y=Y_cells,mat_no=mat_no,CPPR=cppr,pr=PR,VF=vol_frac,GridSpc=GRIDSPC,NP=30) 

n = pss.N                                                           # n is the number of particles to generated for the 'library' which can be 
                                                                    # selected from this and placed into the mesh
part_area = np.zeros((n))                                           # N.B. particles can (and likely will) be placed more than once
part_phi  = np.linspace(PHI_(pss.cppr_max*GRIDSPC*1.e3*2.),PHI_(pss.cppr_min*GRIDSPC*1.e3*2.),n)
guide_radi= (DD_(part_phi)/(GRIDSPC*1.e3)).astype(int)              # particle radii generated as linear range from the min to max
part_radi = np.zeros_like(guide_radi)
                                                                    # guide only! Frequency will be calculated based on final size only!
phi_tol   = abs(part_phi[1] - part_phi[0])/2.                       # The 'tolerance' on each radius. A level of uncertainty
part_freq = np.zeros((n)).astype(int)                               # An array to store the frequency of each grain size
wasted_area = 0

grains = glob.glob('../grain_library/regshapes/*.txt')
MAXR   = np.amax(guide_radi.astype(float))
ii = 0
II = 0
lost_area = 0
while ii < n:
    GRN                  = random.choice(grains)
    x                    = float(guide_radi[ii])
    rot                  = random.random()
    ascale               = (x/MAXR)**2.
    
    pss.mesh_Shps[ii],fail= pss.gen_shape_fromvertices(fname=GRN,mixed=False,areascale=ascale, rot=rot, min_res=4)
    #plt.figure(i)
    #plt.axis('equal')
    #plt.imshow(pss.mesh_Shps[i,:,:],cmap='binary',interpolation='nearest')
    #plt.savefig('test{}.png'.format(i))
    if fail != True:
        equiv_diam           = np.sqrt(np.sum(pss.mesh_Shps[ii])/np.pi)*2.
        part_radi[ii]        = equiv_diam/2.
        part_area[ii]        = np.sum(pss.mesh_Shps[ii])
        phi                  = -np.log2(equiv_diam*GRIDSPC*1.e3)   # For each radius, generate an equivalent phi. phi = -log2(2R)
        theta                = lunar_pdf(phi,phi_tol)              # as a percentage by Weight! *A_est/part_area[ii]
        frq                  = .5*theta*A_total/(part_area[ii]*(GRIDSPC**2.))
        part_freq[ii]        = int(np.around(frq))                      # The frequency of that size is the integral of the pdf over the tolerance range
        lost_area           += (frq-part_freq[ii])*part_area[ii]
        print frq,(frq-part_freq[ii])
                                                                    # i.e. Prob = |CDF(x+dx) - CDF(x-dx)|; expected freq is No. * Prob!
    ii += 1
    II += 1
    if fail == True: ii -= 1
    if II > n*2: break
print 100*lost_area/(X_cells*Y_cells)

#print np.sum(part_freq*part_area)/A_est
print 'Radii|Frequencies'
print np.column_stack((part_radi,part_freq))
#print part_freq,np.sum(part_freq),N_est



A_        = []                                                      # Generate empty lists to store the area
xc        = []                                                      # x coord
yc        = []                                                      # y coord
I_        = []                                                      # Library index
J_        = []                                                      # shape number of each placed particle
vf_pld    = 0                                                       # Volume Fraction of material inserted into the mesh so far
J         = 0
old_vfrac = 0.
I         = 0
freq      = 0
total_freq = np.sum(part_freq)
actul_freq = np.zeros_like(part_freq)
try:                                                                # Begin loop placing grains in.
    while I < n:#while vf_pld < vol_frac:                                        # Keep attempting to add particles until the required volume fraction is achieved      
        afreq       = 0
        for k in range(part_freq[I]):
            fail    = 1
            fail_no = 0
            while fail == 1:
                x,y,fail = pss.gen_coord(pss.mesh_Shps[I])              # Function generates a coordinate where the placed shape does not overlap with any placed so far
                fail_no += 1
                if fail_no > 10: 
                    fail    = 0
            if fail_no <= 10: 
                afreq += 1
                area = pss.insert_shape_into_mesh(pss.mesh_Shps[I],x,y)
                J   += 1       
            fail_no = 0
            A_.append(area)                                             # Save the area
            xc.append(x)                                                # Save the coords of that particle
            yc.append(y)
            J_.append(J)                                                # Save the particle number
            I_.append(I)                                                # Save the library index
            vf_pld = np.sum(A_)/(pss.meshx*pss.meshy)                   # update the volume fraction
            old_vfrac = vf_pld
            print "\rvolume fraction achieved so far: {:5.3f}%".format(vf_pld*100),
            sys.stdout.flush()
        actul_freq[I] = afreq
        I += 1
    print
except KeyboardInterrupt:
    pass

I_shape = np.array(I_)                                              # Convert the lists of index, shape number, and coordinates to arrays
J_shape = np.array(J_)

xcSI = np.array(xc).astype(float)
ycSI = np.array(yc).astype(float)
xcSI*= GRIDSPC                                                      # Turn the coordinates into physical units
ycSI*= GRIDSPC
MAT  = pss.mat_assignment(pss.mats[1:],xcSI,ycSI)                   # Assign materials to the particles. This returns an array that is the same shape as xc, 
pss.populate_materials(I_,xc,yc,MAT,J)                              # Now populate the materials meshes (NB these are different to the 'mesh' and are
                                                                    # and contains the optimum corresponding material number for each particle
                                                                    # the ones used in the actual iSALE input file)

#pss.fill_rectangle(0,0,pss.meshx*GRIDSPC,pss.meshy*GRIDSPC,pss.mats[0])
                                                                    # pySALESetup prioritises material placed sooner. So later things will NOT overwrite previous ones
pss.save_general_mesh()                                             # Save the mesh as meso_m.iSALE (default)

plt.figure()                                                        # plot the resulting mesh. Skip this bit if you do not need to.
for KK in range(pss.Ms):
    matter = np.copy(pss.materials[KK,:,:])*(KK+1)
    matter = np.ma.masked_where(matter==0.,matter)
    plt.imshow(matter, cmap='viridis',vmin=0,vmax=pss.Ms,interpolation='nearest')

matter = np.copy(pss.materials[pss.Ms-1,:,:])*(pss.Ms)
matter = np.ma.masked_where(matter==0.,matter)


#plt.axis('equal')
plt.axvline(0)
plt.axvline(pss.meshx)
plt.axhline(0)
plt.axhline(pss.meshy)

#plt.xlim(0,pss.meshx)
#plt.ylim(0,pss.meshy)
plt.axis('off')
plt.savefig('Regolith_vfrac-{:3.2f}.png'.format(vf_pld*100),dpi=500)
plt.show()

