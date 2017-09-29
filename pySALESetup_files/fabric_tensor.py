import numpy as np
import pySALESetup as pss
from sys import argv


A = np.genfromtxt(argv[1])


xcoords = A[:,1]
ycoords = A[:,2]
radii = A[:,3]

toler = 2.e-6 *.5
Z, A, F = pss.Fabric_Tensor_disks(xcoords,ycoords,radii,toler) 

print Z
print A
print F
