import pySALESetup as pss
import numpy as np
import random
import matplotlib.pyplot as plt
import sys

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

vol_frac   = 1.
Y_cells    = 500 
X_cells    = 500 
y_length   = 20.e-3                                                  # Transverse width of bed
x_length   = 20.e-3                                                  # Longitudinal width of bed
PR         = (0.1,3.0)                                              # Particle size range
cppr       = 50                                                     # Cells per particle radius
GRIDSPC    = x_length/X_cells                                        # Physical distance/cell
mat_no     = 7
A_est      = vol_frac*X_cells*Y_cells #/(np.pi*(.25 * 1.e-3 * 2.**(-2.6))**2.)


pss.generate_mesh(X=X_cells,Y=Y_cells,mat_no=mat_no,CPPR=cppr,pr=PR,VF=vol_frac,GridSpc=GRIDSPC,NP=10) 

n = pss.N                                                           # n is the number of particles to generated for the 'library' which can be 
                                                                    # selected from this and placed into the mesh
part_area = np.zeros((n))                                           # N.B. particles can (and likely will) be placed more than once
part_phi  = np.linspace(PHI_(pss.cppr_max*GRIDSPC*1.e3*2.),PHI_(pss.cppr_min*GRIDSPC*1.e3*2.),n)
part_radi = (DD_(part_phi)/(GRIDSPC*1.e3)).astype(int)                     # particle radii generated as linear range from the min to max
phi_tol   = abs(part_phi[1] - part_phi[0])                          # The 'tolerance' on each radius. A level of uncertainty
part_freq = np.zeros((n)).astype(int)                               # An array to store the frequency of each grain size
wasted_area = 0
for i in range(n):
    x                    = float(part_radi[i])
    pss.mesh_Shps[i,:,:] = pss.gen_circle(int(x))                   # Generate a circle and store it in mesh_Shps (an array of meshes) 
    part_area[i]         = np.sum(pss.mesh_Shps[i,:,:])
    phi                  = -np.log2(part_radi[i]*2.*GRIDSPC*1.e3)   # For each radius, generate an equivalent phi. phi = -log2(2R)
    frq                  = lunar_pdf(phi,phi_tol)*A_est/part_area[i]
    part_freq[i]         = int(np.around(frq))                      # The frequency of that size is the integral of the pdf over the tolerance range
    if part_freq[i] == 0:
        wasted_area += frq*part_area[i]
    if (wasted_area/part_area[i])>1.:
        part_freq[i] += 1
                                                                    # i.e. Prob = |CDF(x+dx) - CDF(x-dx)|; expected freq is No. * Prob!

print np.sum(part_freq*part_area)/A_est
print 'Radii_/_Frequencies'
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
try:                                                                # Begin loop placing grains in.
    while I<n:#while vf_pld < vol_frac:                                        # Keep attempting to add particles until the required volume fraction is achieved      
        fail = 1
        fail_no   = 0
        while fail == 1 and part_freq[I]>0:                                            # Whilst fail == 1 continue to try to generate a coordinate
            x,y,fail = pss.gen_coord(pss.mesh_Shps[I,:,:])          # Function generates a coordinate where the placed shape does not overlap with any placed so far
            fail_no += 1
            if fail_no >= 5: 
                fail    = 0
                fail_no = 0
        freq += 1
        if part_freq[I] > 0:
            area = pss.insert_shape_into_mesh(pss.mesh_Shps[I,:,:],x,y)
            #print 'particle number: {}, radius = {:3.0f}, number placed = {}'.format(I,part_radi[I],freq)#,freq,part_freq[I]
            J   += 1       
        # pss.place_shape(pss.mesh_Shps[0,:,:],J,I,M)               # This function allows you to place a shape in AND specify its material. However it overwrites previous placings
            A_.append(area)                                             # Save the area
            xc.append(x)                                                # Save the coords of that particle
            yc.append(y)
            J_.append(J)                                                # Save the particle number
            I_.append(I)                                                # Save the library index
        
            vf_pld = np.sum(A_)/(pss.meshx*pss.meshy)                   # update the volume fraction
            old_vfrac = vf_pld
            print "\rvolume fraction achieved so far: {:5.3f}%".format(vf_pld*100),
            sys.stdout.flush()
        else:
            I   += 1
        if freq>part_freq[I]:
            I   += 1
            freq = 0

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
plt.xlim(0,pss.meshx)
plt.ylim(0,pss.meshy)
plt.axis('off')
plt.savefig('Regolith_vfrac-{:3.2f}.png'.format(vf_pld*100),dpi=500)
plt.show()

