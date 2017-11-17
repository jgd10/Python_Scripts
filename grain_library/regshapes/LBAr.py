# Remove the randomness from the function.
# Can we make a grain both highly angluar and highly elliptical, then morph it to be less angular, more spherical? Iterative process?
# Shrink radii slowly? Simulate it as if geological. Moving average around shape to round it?
# Construct a bunch of highly elliptical, angular shapes, then reduce these effects. All to construct this library!
# Geologists ACTUALLY look at the angularity and the eccentricity (or sphericity) and we will know this info.
# Gareth's idea is to have 2 algorithms, one to generate a grain, one to smooth it. Numbers with no random element can go in/be stored for later use.


import numpy as np
import random
import matplotlib.pyplot as plt
import pySALESetup as pss
from matplotlib.path import Path
import matplotlib.patches as patches

def polygon_area(X,Y):
    N = np.size(X)
    A = 0
    for i in range(1,N):
        A += (X[i-1]*Y[i]-X[i]*Y[i-1])*.5
    return abs(A)

def aspect_ratio(A):
    Diameters = []
    N = np.size(A)
    for i in range(N/2 - 1):                    # To calculate the diamaters, cycle through N/2-1 times. e.g. if N=6, then you only need to do 0,1,2 radii
        Diameters.append(A[i]+A[i+N/2])
    aspect = np.amax(Diameters)/np.amin(Diameters)
    return aspect
def angle_between_vertices(x1,x2):
    ct    = x1[0]*x2[0] + x1[1]*x2[1]
    mod1  = np.sqrt(x1[0]**2.+x1[1]**2.)
    mod2  = np.sqrt(x2[0]**2.+x2[1]**2.)
    theta = np.arccos(ct/(mod1*mod2))
    return theta


N = 8
L = 1.
B = 1./1.35
A = 0.7*L*B
q = 0.2*B/2.
for i in range(500):
    A_ratio = 0
    while A_ratio < .65 or A_ratio > .75:
    
        X1 = np.array([-L/2.,random.random()*B - B/2.])
        X3 = np.array([random.random()*L - L/2.,-B/2.])
        X5 = np.array([L/2., random.random()*B - B/2.])
        X7 = np.array([random.random()*L - L/2., B/2.])
        
        X2 = (X1 + X3)/2.
        X4 = (X3 + X5)/2.
        X6 = (X5 + X7)/2.
        X0 = (X7 + X1)/2.
        
        evenvert = [X2,X4,X6,X0]
        
        for v in evenvert:
            x0 = v[0]
            y0 = v[1]
            ang = np.arctan2(v[1],v[0])
            rad = np.sqrt(v[0]**2. + v[1]**2.)
            rad = rad + random.choice([-1,1])*q
            v[0]= max(rad*np.cos(ang),-L/2.)
            v[0]= min(v[0],L/2.)
            v[1]= max(rad*np.sin(ang),-B/2.)
            v[1]= min(v[1],B/2.)
        
        R = np.row_stack((X0,X1,X2,X3,X4,X5,X6,X7,X0))
        Area = polygon_area(R[:,0],R[:,1])
        A_ratio = Area/(L*B)
    np.savetxt('grain_arearatio-{:2.3f}.txt'.format(A_ratio),R,delimiter=',')

fig = plt.figure()
ax = fig.add_subplot(111,aspect='equal')
codes = []
codes.append(Path.MOVETO)
for i in range(7):
	codes.append(Path.LINETO)

codes.append(Path.CLOSEPOLY)
path = Path(R, codes)
patch = patches.PathPatch(path, facecolor='red', lw=0,alpha=.8)
ax.add_patch(patch)
ax.axvline(-L/2.)
ax.axvline(L/2.)
ax.axhline(-B/2.)
ax.axhline(B/2.)
ax.set_xlim(-.8,.8)
ax.set_ylim(-.8,.8)
plt.show()
#plt.savefig('./grain_test.png',dpi=300)

