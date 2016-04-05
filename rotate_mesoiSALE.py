import numpy as np
import pySALESetup as pss
import sys
mats = np.genfromtxt(sys.argv[1],dtype=float,usecols=(0))
x    = np.genfromtxt(sys.argv[1],dtype=float,usecols=(1))
y    = np.genfromtxt(sys.argv[1],dtype=float,usecols=(2))
r    = np.genfromtxt(sys.argv[1],dtype=float,usecols=(3))
frac = 1./2.
theta = np.pi* frac
c = np.cos(theta)
s = np.sin(theta)

xr = c*x - s*y 
yr = s*x + c*y 

if np.size(xr[xr<0])>0.:
    xr += np.amax(abs(xr))
if np.size(yr[yr<0])>0.:
    yr += np.amax(abs(yr))
fpath = sys.argv[1]
fpath = fpath.replace('.iSALE','_rot-{:2.2f}pi.iSALE'.format(frac))

pss.view_mesoiSALE(sys.argv[1],save=True)
A = np.column_stack((mats,xr,yr,r))
np.savetxt(fpath,A)
pss.view_mesoiSALE(fpath,save=True)
