import numpy as np
import glob


Bfile = glob.glob('*partA*.iSALE')
Afile = glob.glob('*partB*.iSALE')
A = np.genfromtxt(Afile[0],dtype=float,skip_header=1)
B = np.genfromtxt(Bfile[0],dtype=float,skip_header=1)


print B[:,0]
B[:,0] += (np.amax(A[:,0])+1)
partnums2 = B[:,-1] 

partnums2[partnums2>0] += (np.amax(A[:,-1]))
B[:,-1] = partnums2

full = np.row_stack((A,B))

np.savetxt('meso_m.iSALE',full,fmt='%5.3f')
