import numpy as np
import pySALESetup as pss

# 3 repeating materials: Al, Cu, Fe, PC impactor
# Bed of size 1 mm x 9 mm
# Repeat pattern 6 times, 50 cells/band, 150 cells/sandwich
pss.generate_mesh(100,900,mat_no=3,GridSpc=10.e-6)

Al1 = np.array([0.0e-3,0.5e-3])
Cu1 = np.array([0.5e-3,1.0e-3])
Fe1 = np.array([1.0e-3,1.5e-3])

for i in range(6):
    pss.fill_plate(Al1[0],Al1[1],1)
    pss.fill_plate(Cu1[0],Cu1[1],2)
    pss.fill_plate(Fe1[0],Fe1[1],3)
    Al1 += 1.5e-3
    Cu1 += 1.5e-3
    Fe1 += 1.5e-3
    print Al1
    print 
    print Cu1
    print 
    print Fe1
    print '-----------------------'

#pss.display_mesh()

pss.save_general_mesh(fname='meso_m_repeatingsandwich_{}x{}.iSALE'.format(pss.meshx,pss.meshy),noVel=True)







