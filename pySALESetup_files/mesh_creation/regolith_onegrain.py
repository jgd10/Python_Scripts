import numpy as np
import pySALESetup as pss
import matplotlib.pyplot as plt
import glob
import random


Y_cells    = 2000
X_cells    = 1000
y_length   = 5.e-3                                                 # Transverse width of bed
x_length   = 2.5e-3                                                 # Longitudinal width of bed
cppr       = 25                                                     # Cells per particle radius
LB_cppr    = 5.                                                     # Lowest allowed cppr
GRIDSPC    = x_length/X_cells                                       # Physical distance/cell
mat_no     = 3

############################################
largest_cppr  = 350                      ###
smallst_cppr  = 4                        ###  
largest_grain = largest_cppr*GRIDSPC     ###
smallst_grain = smallst_cppr*GRIDSPC     ### 
refrnce_cppr  = float(cppr)              ###
large_frac    = largest_cppr/refrnce_cppr###
small_frac    = smallst_cppr/refrnce_cppr###
############################################

PR            = (small_frac,large_frac)                                              # Particle size range

pss.generate_mesh(X=X_cells,Y=Y_cells,mat_no=mat_no,CPPR=cppr,pr=PR,GridSpc=GRIDSPC,NP=500) 

grains = glob.glob('../grain_library/regshapes/grain_area*.txt')
GRN                  = random.choice(grains)

pss.mesh_Shps[0],fail = pss.gen_shape_fromvertices(fname=GRN,mixed=False,areascale=1, min_res=4)

pss.place_shape(pss.mesh_Shps[0],x_length*.5,y_length*.25,2)


GRN                  = random.choice(grains)
pss.mesh_Shps[1],fail = pss.gen_shape_fromvertices(fname=GRN,mixed=False,areascale=1, min_res=4)
pss.place_shape(pss.mesh_Shps[1],x_length*.5,y_length*.75,3)

pss.fill_rectangle(0,0,pss.meshx*GRIDSPC,pss.meshy*GRIDSPC,1)
pss.overwrite_rectangle(0.,(pss.meshy-6.)*GRIDSPC,pss.meshx*GRIDSPC,pss.meshy*GRIDSPC,1)
pss.overwrite_rectangle(0.,0.,pss.meshx*GRIDSPC,6.*GRIDSPC,1)
pss.fill_rectangle(0.,(pss.meshy-3.)*GRIDSPC,pss.meshx*GRIDSPC,pss.meshy*GRIDSPC,-1.)
pss.fill_rectangle(0.,0.,pss.meshx*GRIDSPC,3.*GRIDSPC,-1.)
pss.rectangle_vel(0,y_length*.5,x_length,y_length,-2.5e2,dim=1)
pss.rectangle_vel(0,0,x_length,y_length*.5,2.5e2,dim=1)
pss.save_general_mesh(fname = 'meso_m_2grain_mirror.iSALE',info=True)                                             # Save the mesh as meso_m.iSALE (default)




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
ax2.axis('off')
fig2.savefig('test.png',dpi=500,bbox_inches='tight')


fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
pss.part_no = np.ma.masked_where(pss.part_no==0.,pss.part_no)
pl = ax4.imshow(pss.part_no,origin='lower')
fig4.colorbar(pl)

plt.show()


