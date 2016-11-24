# base code acquired from here:
# https://zulko.wordpress.com/2012/09/29/extract-data-from-graph-pictures-with-python/
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
import tkSimpleDialog as tksd
import Tkinter
import sys,threading

root = Tkinter.Tk()
root.withdraw()
 
def tellme(s):
    print s
    plt.title(s,fontsize=16)
    plt.draw()
def promptPoint(text=None):     
    if text is not None: tellme('{}: zoom in on point, if necessary. \n hit any key to continue'.format(text))
    while plt.waitforbuttonpress(timeout=-1) == False:  #returns false for clicks, and true for keypress
        pass                                            #timeout < 0 means it never timesout
    if text is not None: tellme('{}: click on the necessary pixel'.format(text))
    return  np.array(plt.ginput(1,timeout=-1)[0]) 
def askValue(text='',initialvalue=0.0):
    return tksd.askfloat(text, 'Value:',initialvalue=initialvalue)
def askValueI(text='',initialvalue=0.0):
    return tksd.askinteger(text, 'Value:',initialvalue=initialvalue)
def askValueS(text='',initialvalue='viridis'):
    return tksd.askstring(text, 'Value:',initialvalue=initialvalue)

""" GUI to get data from a XY graph image. Either provide the graph
    as a path to an image in 'source' or copy it to the clipboard.
"""
 
##### GET THE IMAGE

fname = sys.argv[1]
image  = Image.open(fname)
rgb_im = image.convert('RGB')
img  = np.array(rgb_im)
#image = mpimg.imread(argv[1])
origin = 'upper'

###### DISPLAY THE IMAGE
plt.ion() 
fig, ax = plt.subplots(1)
imgplot = ax.imshow(image, origin=origin)
fig.canvas.draw()
plt.show()
 
##### PROMPT THE AXES





RH_ref = promptPoint('select the top-right corner of the cmap')
LH_ref = promptPoint('select the bottom-left corner of the cmap')

imin = int(min(LH_ref[0],RH_ref[0]))
imax = int(max(LH_ref[0],RH_ref[0]))
jmin = int(min(LH_ref[1],RH_ref[1]))
jmax = int(max(LH_ref[1],RH_ref[1]))
data = img[jmin:jmax,imin:imax,:].astype(float)
C0_ref = promptPoint('select the bottom of the cbar')
C1_ref = promptPoint('select the top of the cbar')
plt.ioff()

C0_val = askValue('colourscale min',0.)
C1_val = askValue('colourscale max',1.)
range1 = abs(C0_ref[0]-C1_ref[0])
range2 = abs(C0_ref[1]-C1_ref[1])

cimin = int(min(C0_ref[0],C1_ref[0]))
cimax = int(max(C0_ref[0],C1_ref[0]))
cjmin = int(min(C0_ref[1],C1_ref[1]))
cjmax = int(max(C0_ref[1],C1_ref[1]))
plt.close()

cbar_ = img[cjmin:cjmax,cimin:cimax,:]


if range1 > range2: 
    AXS = 0
elif range1 < range2:
    AXS = 1

CB_ = np.mean(cbar_,axis=AXS).astype(float)
CN  = np.size(CB_[:,0])
 

nx,ny,dmy = np.shape(data)
dummy     = np.ones((nx,ny,3))*-1.

thresh  = np.array([8.,8.,8.])
counter = 0
total   = float(nx*ny)

for k in np.arange(CN):
    B = np.all(abs(data-CB_[k,:]) <= thresh,axis=2)
    dummy[B] *= 0. 
    dummy[B] += 1. - k/(CN-1.)

data2 = dummy[:,:,0].copy()
data2  = np.ma.masked_where(data2<0.,data2)
data2 *= abs(C0_val - C1_val)
data2 += min(C0_val,C1_val)

"""
X0val = askValue('input the horizontal axis lower limit',0.)
X1val = askValue('input the horizontal axis upper limit',1.)
Y0val = askValue('input the vertical axis lower limit',0.)
Y1val = askValue('input the vertical axis upper limit',1.)

Nx,Ny = np.shape(data2)
xc = np.linspace(X0val,X1val,Nx)
yc = np.linspace(Y1val,Y0val,Ny)
"""

np.savetxt('ripped_data_from_{}.csv'.format(fname[:-4]),data2)
cpal = askValueS('New cmap string','viridis')

fig  = plt.figure()
ax1  = fig.add_subplot(221)
base = np.linspace(C1_val,C0_val,CN)

ax1.set_title('original cbar RGB')
ax1.plot(base,CB_[:,0],linestyle=' ',marker='o',color='r',mec='None',alpha=0.5,label='red')
ax1.plot(base,CB_[:,1],linestyle=' ',marker='^',color='g',mec='None',alpha=0.5,label='blue')
ax1.plot(base,CB_[:,2],linestyle=' ',marker='s',color='b',mec='None',alpha=0.5,label='green')
ax1.set_xlabel('data values')
ax1.set_ylabel('RGB value')
ax1.legend(loc='best',numpoints=1,fontsize='xx-small')

ax2 = fig.add_subplot(222)
mplimage = mpimg.imread(fname)
ax2.set_title('original image')
ax2.imshow(mplimage,origin=origin)
ax2.axis('off')

#ax3 = fig.add_subplot(223)
#ax3.set_title('new image (pcolormesh)')
#pcl = ax3.pcolormesh(xc,yc,data2,cmap=cpal,vmin=C0_val,vmax=C1_val)
#ax3.axis('equal')
#fig.colorbar(pcl)

ax4 = fig.add_subplot(224)
ax4.set_title('new image (imshow)')
ims = ax4.imshow(data2,cmap=cpal,interpolation='nearest',vmin=C0_val,vmax=C1_val)
fig.colorbar(ims)

fig.tight_layout()
fig.savefig('ripped_data_from_{}.png'.format(fname[:-4]),dpi=300,bbox_inches='tight')
sys.exit()
