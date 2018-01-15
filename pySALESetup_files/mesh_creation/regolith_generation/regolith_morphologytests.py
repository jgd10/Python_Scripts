import pySALESetup as pss
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import glob


def function(x):
    A = 2.908
    B = 0.028
    C = 0.320
    L = 0.643
    a = 99.4
    return (1./a)*A/(B+C*np.exp(-L*x))
def diff_function(x):
    A = 2.908
    B = 0.028
    C = 0.320
    L = 0.643
    a = 99.4
    pdf  = (1./a)*A*L*C*np.exp(L*x)/(B*np.exp(L*x)+C)**2.
    return pdf

def lunar_pdf(x,LB_tol,UB_tol):                                                   
    P     = abs(function(x+UB_tol) - function(x-LB_tol))
    return P
def PHI_(x):
    """
    x must be in SI (metres) for this function to work
    PHI is only calculated correctly when the arg is mm
    """
    return -1.*np.log2(x*1000.)

def DD_(x):
    return 0.5*2.**(-x)

def setupTOPhalf(V_I,vf_pld,void_pld,mat_por,Plot=True):
   # fill whole mesh with matrix
   pss.fill_rectangle(0,0,pss.meshx*pss.GS,pss.meshy*pss.GS,pss.mats[0])
   # fill top 6 cells with matrix
   pss.overwrite_rectangle(0.,(pss.meshy-6.)*pss.GS,pss.meshx*pss.GS,pss.meshy*pss.GS,pss.mats[0])
   # fill top 3 cells of mesh with void
   pss.fill_rectangle(0.,(pss.meshy-3.)*pss.GS,pss.meshx*pss.GS,pss.meshy*pss.GS,-1.)
   
   prev_voids = np.copy(pss.materials[-1])
   pss.materials[-1] *= 0.
   print 'volume fraction of explicit void: {:3.3f} %, matrix porosity - {:3.3f}%'.format(void_pld*100.,mat_por*100.)
   #pss.rectangle_vel(0,y_length*.5,x_length,y_length,-5.e2,dim=1)
   pss.rectangle_vel(0,0,x_length,y_length,-V_I,dim=1)
   pss.save_general_mesh(fname = 'meso_m_vf-{:3.2f}_vof-{:3.2f}_mpor={:3.2f}_partB.iSALE'.format(vf_pld*100,void_pld*100,mat_por),info=True)
   if Plot:
       fig2 = plt.figure(1)                                                        # plot the resulting mesh. Skip this bit if you do not need to.
       ax2  = fig2.add_subplot(111)
       for KK in range(pss.Ms):
           matter = np.copy(pss.materials[KK,:,:])*(KK+1)
           matter = np.ma.masked_where(matter==0.,matter)
           ax2.imshow(matter, cmap='viridis',vmin=0,vmax=pss.Ms,interpolation='nearest',origin='lower')
       fig2.savefig('vfrac-{:3.2f}_void-{:3.2f}_mpor-{:3.2f}-PartB.png'.format(vf_pld*100,void_pld*100,mat_por))
       fig2.clf()
   # Reverse changes for next loop to how they were before
   pss.materials[0] *= 0.
   pss.materials[-1] = prev_voids
   return

def setupBOTTOMhalf(V_I,vf_pld,void_pld,mat_por,Plot=True):
   # fill whole mesh with matrix
   pss.fill_rectangle(0,0,pss.meshx*pss.GS,pss.meshy*pss.GS,pss.mats[0])
   # fill bottom 6 cells with matrix
   pss.overwrite_rectangle(0.,0.,pss.meshx*pss.GS,6.*pss.GS,pss.mats[0])
   # fill bottom 3 cells of mesh with void
   pss.fill_rectangle(0.,0.,pss.meshx*pss.GS,3.*pss.GS,-1.)
   
   prev_voids = np.copy(pss.materials[-1])
   pss.materials[-1] *= 0.
   print 'volume fraction of explicit void: {:3.3f} %, matrix porosity - {:3.3f}%'.format(void_pld*100.,mat_por*100.)
   #pss.rectangle_vel(0,y_length*.5,x_length,y_length,-5.e2,dim=1)
   pss.rectangle_vel(0,0,x_length,y_length,V_I,dim=1)
   pss.save_general_mesh(fname = 'meso_m_vf-{:3.2f}_vof-{:3.2f}_mpor={:3.2f}_partA.iSALE'.format(vf_pld*100,void_pld*100,mat_por),info=True)
   if Plot:
       fig2 = plt.figure(1)                                                        # plot the resulting mesh. Skip this bit if you do not need to.
       ax2  = fig2.add_subplot(111)
       for KK in range(pss.Ms):
           matter = np.copy(pss.materials[KK,:,:])*(KK+1)
           matter = np.ma.masked_where(matter==0.,matter)
           ax2.imshow(matter, cmap='viridis',vmin=0,vmax=pss.Ms,interpolation='nearest',origin='lower')
       fig2.savefig('vfrac-{:3.2f}_void-{:3.2f}_mpor-{:3.2f}-PartA.png'.format(vf_pld*100,void_pld*100,mat_por))
       fig2.clf()
   # Reverse changes for next loop to how they were before
   pss.materials[0] *= 0.
   pss.materials[-1] = prev_voids
   return


vol_frac   = .5
Y_cells    = 1200
X_cells    = 500
y_length   = 3.e-3                                                 # Transverse width of bed
x_length   = 1.25e-3                                                 # Longitudinal width of bed
cppr       = 25                                                     # Cells per particle radius
LB_cppr    = 5.                                                     # Lowest allowed cppr
GRIDSPC    = x_length/X_cells                                       # Physical distance/cell
mat_no     = 6

#############################
largest_cppr  = 100 
smallst_cppr  = 4
largest_grain = largest_cppr*GRIDSPC
smallst_grain = smallst_cppr*GRIDSPC
#print smallst_cppr
refrnce_cppr  = float(cppr)
large_frac    = largest_cppr/refrnce_cppr
small_frac    = smallst_cppr/refrnce_cppr
#############################
PR            = (small_frac,large_frac)                                              # Particle size range
A_est         = vol_frac*X_cells*Y_cells #/(np.pi*(.25 * 1.e-3 * 2.**(-2.6))**2.)
A_total       = y_length*x_length


pss.generate_mesh(X=X_cells,Y=Y_cells,mat_no=mat_no,CPPR=cppr,pr=PR,GridSpc=GRIDSPC,NP=100) 


n = pss.N                                                           # n is the number of particles to generated for the 'library' which can be 
                                                                    # selected from this and placed into the mesh
part_area = np.zeros((n))                                           # N.B. particles can (and likely will) be placed more than once
guide_radi= np.linspace(smallst_grain,largest_grain,n)
guide_radi= guide_radi[::-1]
part_phi  = PHI_(guide_radi*2.) 
#part_phi  = np.linspace(PHI_(pss.cppr_max*GRIDSPC*1.e3*2.),PHI_(pss.cppr_min*GRIDSPC*1.e3*2.),n)
#guide_radi= (DD_(part_phi)/(GRIDSPC*1.e3)).astype(int)              # particle radii generated as linear range from the min to max
part_radi = np.zeros_like(guide_radi)
                                                                    # guide only! Frequency will be calculated based on final size only!
phi_tol   = abs(part_phi[1:] - part_phi[:-1])/2.                       # The 'tolerance' on each radius. A level of uncertainty
part_freq = np.zeros((n)).astype(int)                               # An array to store the frequency of each grain size
wasted_area = 0

grains = glob.glob('../grain_library/regshapes/grain_area*.txt')
MAXR   = np.amax(guide_radi)
ii = 0
II = 0
lost_area = 0

rotation = np.zeros((n))
elongtin = np.zeros((n))
areratio = np.zeros((n))
roundnes = np.zeros((n))
while ii < n:
    GRN                  = random.choice(grains)
    rot                  = random.random()
    rotation[ii]         = rot
    ascale               = (guide_radi[ii]/MAXR)**2.
    
    pss.mesh_Shps[ii],fail = pss.gen_shape_fromvertices(fname=GRN,mixed=False,areascale=ascale, rot=rot, min_res=4)
    elongtin[ii],areratio[ii],roundnes[ii] = pss.shape_info(GRN)
    """
    plt.figure()
    plt.axis('equal')
    plt.imshow(pss.mesh_Shps[ii,:,:],cmap='binary',interpolation='nearest')
    plt.show()
    """
    if fail != True:
        equiv_diam           = np.sqrt(np.sum(pss.mesh_Shps[ii])/np.pi)*2.
        part_radi[ii]        = equiv_diam/2.
        part_area[ii]        = np.sum(pss.mesh_Shps[ii])
        phi                  = -np.log2(equiv_diam*GRIDSPC*1.e3)   # For each radius, generate an equivalent phi. phi = -log2(2R)
        if ii == n-1: 
            UB_tol = 0
        else:
            UB_tol = phi_tol[ii]
        if ii == 0:   
            LB_tol = 0
        else:
            LB_tol = phi_tol[ii]

        theta                = lunar_pdf(phi,LB_tol,UB_tol)              # as a percentage by Weight! *A_est/part_area[ii]
        area_fraction        = (vol_frac*A_total)/(part_area[ii]*(GRIDSPC**2.))
        frq                  = theta*area_fraction
        part_freq[ii]        = int(np.around(frq))                      # The frequency of that size is the integral of the pdf over the tolerance range
        lost_area           += (frq-part_freq[ii])*part_area[ii]
                                                                    # i.e. Prob = |CDF(x+dx) - CDF(x-dx)|; expected freq is No. * Prob!
    ii += 1
    if fail == True: 
        ii -= 1
        II += 1
        fail = False
    if II > n*2: break
lost_area = 100*lost_area/(X_cells*Y_cells)

#print np.sum(part_freq*part_area)/A_est
#print part_freq,np.sum(part_freq),N_est
print 'highest achievable vol frac = {:3.4f}%'.format(np.sum(part_freq*part_area)*100./(X_cells*Y_cells))



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
equiv_radi = np.zeros((total_freq))
equiv_cppr = np.zeros((total_freq))
Angle      = np.zeros((total_freq))
Elong = np.zeros_like(Angle)
A_rat = np.zeros_like(Angle)
Round = np.zeros_like(Angle)

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
                equiv_radi[J] = part_radi[I]*pss.GS
                equiv_cppr[J] = int(part_radi[I])
                Angle[J]      = rotation[I]
                Elong[J]      = elongtin[I]
                A_rat[J]      = areratio[I]
                Round[J]      = roundnes[I]
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
MAT  = pss.mat_assignment(pss.mats[1:-1],xcSI,ycSI)                   # Assign materials to the particles. This returns an array that is the same shape as xc, 
pss.populate_materials(I_,xc,yc,MAT,J,info=True)                    # Now populate the materials meshes (NB these are different to the 'mesh' and are
                                                                    # and contains the optimum corresponding material number for each particle
                                                                    # the ones used in the actual iSALE input file)
data = np.vstack((J_,equiv_radi,equiv_cppr,MAT,Elong,A_rat,Round,xcSI,ycSI,Angle,I_))
index = ('particle number, Equivalent Radii, Equivalent Cppr, Material Number, Elongation, Area Ratio, Roundness, x coord, y coord, Rotation, Shape Number')
data = np.transpose(data)

dframe = np.core.records.array(data,names=index)
np.savetxt('info_m_Regolith_vfrac-{:3.2f}_partA.csv'.format(vf_pld*100),dframe,header=index,comments='',delimiter=',')

# pySALESetup prioritises material placed sooner. So later things will NOT overwrite previous ones

for i in range(J):
    void_pld   = np.sum(pss.materials[-1])/float(np.size(pss.materials[-1]))
    if (i==int(J*.25)) or (i==int(J*.5)) or (i==int(J*.75)):
        mat_por = ((.5-void_pld)/(1.-vf_pld-void_pld))
        #setupBOTTOMhalf(1.5e3,vf_pld,void_pld,mat_por)
        setupTOPhalf(1.5e3,vf_pld,void_pld,mat_por)
    II = max(I_[i]-1,0)
    II = min(II,np.amax(I_))
    pss.place_shape(pss.mesh_Shps[II],xc[i],yc[i],pss.mats[-1])

#setupBOTTOMhalf(1.5e3,vf_pld,void_pld,mat_por)
setupTOPhalf(1.5e3,vf_pld,void_pld,mat_por)





#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
###########################
# plotting; not algorithm #
###########################


fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
actul_freq = actul_freq*part_area
act_CDF = np.cumsum(actul_freq)
act_CDF/= (pss.meshx*pss.meshy*vol_frac)
phi_ = PHI_(part_radi*2*GRIDSPC)

#print 'Radii|PHI|Frequencies'
#print np.column_stack((part_radi*2*GRIDSPC,PHI_(part_radi*2*GRIDSPC),part_freq))
phibase = np.linspace(-2,9,1000)
phi0 = np.amin(phi_[act_CDF>0])
ax.plot(phi_,100.*(act_CDF+function(phi0)),label='actual',linestyle=' ',marker='.',mfc='None')
ax.plot(phibase,function(phibase)*100.,label='target')
ax.set_xlabel('$\phi = -$log$_2$(Diameter [mm])')
ax.set_ylabel('AREA%')
ax.set_title('Bed : {} x {}, GRIDSPC = {:3.2g}m'.format(X_cells,Y_cells,GRIDSPC))
ax.text(6,10,
    'Area achieved = {:3.2f}% \nLost Area = {:3.2f}% \nPorosity of matrix = {:3.2f}%'.format(vf_pld*100.,lost_area,((.5-void_pld)/(1.-vf_pld-void_pld)*100.)),fontsize=7)

ax.axvspan(np.amin(phi_),np.amax(phi_[:-1]),color='r',alpha=0.2,label='target region')
ax.axvspan(np.amin(phi_[act_CDF>0]),np.amax(phi_[:-1]),color='b',alpha=0.2,label='actual region')

ax.legend(loc='best',numpoints=1,fontsize='small')
fig.savefig('Regolith_vfrac-{:3.2f}_CDF.png'.format(vf_pld*100),dpi=500,bbox_inches='tight')

fig2 = plt.figure()                                                        # plot the resulting mesh. Skip this bit if you do not need to.
ax2  = fig2.add_subplot(111)
for KK in range(pss.Ms):
    matter = np.copy(pss.materials[KK,:,:])*(KK+1)
    matter = np.ma.masked_where(matter==0.,matter)
    ax2.imshow(matter, cmap='viridis',vmin=0,vmax=pss.Ms,interpolation='nearest',origin='lower')

matter = np.copy(pss.materials[pss.Ms-1,:,:])*(pss.Ms)
matter = np.ma.masked_where(matter==0.,matter)


#plt.axis('equal')
ax2.axvline(0)
ax2.axvline(pss.meshx)
ax2.axhline(0)
ax2.axhline(pss.meshy)

#plt.xlim(0,pss.meshx)
#plt.ylim(0,pss.meshy)
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.imshow(pss.VY_,cmap='plasma',origin = 'lower')

fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
pss.part_no = np.ma.masked_where(pss.part_no==0.,pss.part_no)
pl = ax4.imshow(pss.part_no,origin='lower')
fig4.colorbar(pl)

plt.show()

