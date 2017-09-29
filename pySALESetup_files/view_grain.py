import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from sys import argv

R = np.genfromtxt('grain_aspect-{}.txt'.format(argv[1]),dtype=float,delimiter=',')
#print R
N1,N2 = np.shape(R)
RHO = np.sqrt(R[:,1]**2. + R[:,0]**2.)		# Convert to polar coords
PHI = np.zeros((N1))
PHI = np.arctan2(R[:,1],R[:,0])			# NB y coord comes FIRST in this function. arctan2 chooses the correct quadrant as necessary
PHI+= np.pi/2.
RHO /= (np.amax(RHO))				# Normalise RHO relative to its largest radius
codes = []
codes.append(Path.MOVETO)
A = np.zeros((N1+1,N2))
A[:N1,:] = R
A[N1,:]= R[0,:]
for i in range(N1-1):
	codes.append(Path.LINETO)

codes.append(Path.CLOSEPOLY)
path = Path(A, codes)

fig = plt.figure()
ax = fig.add_subplot(111)
patch = patches.PathPatch(path, facecolor='orange', lw=0)
ax.add_patch(patch)
ax.set_xlim(-500,500)
ax.set_ylim(-500,500)
plt.savefig('./grain_aspect-{}.png'.format(argv[1]),dpi=300)
plt.show()
