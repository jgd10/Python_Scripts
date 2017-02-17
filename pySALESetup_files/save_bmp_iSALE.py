import pySALESetup as pss
import numpy as np
from PIL import Image
import sys
import matplotlib.pyplot as plt

im = Image.open(sys.argv[1])

A = np.asarray(im)
B = np.copy(A)
#B[B>0.] = 1.

B = B[:,::-1]
B = B.transpose()
print np.shape(B)
pss.populate_from_bmp(B)
plt.figure()
plt.imshow(B,interpolation='nearest')
plt.show()
