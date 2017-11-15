import pySALESetup as pss
import numpy as np
import matplotlib.pyplot as plt

L_length   = 10.e-3
T_length   = 1.e-3
GRIDSPC    = 2.e-6 
T_cells    = 5 #int(T_length/GRIDSPC)				# T - Transverse, L - Longitudinal
L_cells    = int(L_length/GRIDSPC)				# T - Transverse, L - Longitudinal
pss.generate_mesh(T_cells,L_cells,mat_no=1,GridSpc=GRIDSPC)
pss.fill_Allmesh(1)
pss.save_general_mesh(noVel=True)

fig = plt.figure()
ax = fig.add_subplot(111,aspect='equal')
ax.imshow(pss.materials[0])
plt.show()
