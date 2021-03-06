# base code acquired from here:
# https://zulko.wordpress.com/2012/09/29/extract-data-from-graph-pictures-with-python/
from PIL import Image
from scipy.misc import imread
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
import tkSimpleDialog as tksd
import Tkinter
import sys,threading
# This script is designed to extract data from colormaps.
# It has some success with stepped plots but still struggles with interpolated 
# plots. Needs more develoment.
root = Tkinter.Tk()
root.withdraw()
 
def tellme(s):
    print s
    plt.title(s,fontsize=16)
    plt.draw()
def promptPoint(text=None):     
    if text is not None: tellme('{}: zoom in on point, if necessary. \n hit any key when ready'.format(text))
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
#image  = Image.open(fname)
#rgb_im = image.convert('RGB')
image  = imread(sys.argv[1]) #np.array(rgb_im)
img    = image
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

# data square coords
imin = int(min(LH_ref[0],RH_ref[0]))
imax = int(max(LH_ref[0],RH_ref[0]))
jmin = int(min(LH_ref[1],RH_ref[1]))
jmax = int(max(LH_ref[1],RH_ref[1]))

# slice the image to the selected region
data = img[jmin:jmax,imin:imax,:].astype(float)

# find the colorbar scale location and values
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

# slice the image to the colorbar
cbar_ = img[cjmin:cjmax,cimin:cimax,:]

# check if colorbar horizontally or vertically oriented
if range1 > range2: 
    AXS = 0
elif range1 < range2:
    AXS = 1

# collapse the colorbar along the axis
cb_ = np.mean(cbar_,axis=AXS).astype(float)

# linearly interpolate between colors along the colobar
# in R, G, and B
M,stuff   = np.shape(cb_)
cb_interp = np.zeros((M*10,3))
x         = np.arange(M)+1
xinterp   = np.linspace(1,M,M*10)
cb_interp[:,0] = np.interp(xinterp,x,cb_[:,0])
cb_interp[:,1] = np.interp(xinterp,x,cb_[:,1])
cb_interp[:,2] = np.interp(xinterp,x,cb_[:,2])

CB_ = np.copy(cb_interp)

CN  = np.size(CB_[:,0])
 
# dummy is new array to contain extracted data
nx,ny,dmy = np.shape(data)
dummy     = np.ones((nx,ny,3))*-1.

# threshold RGB values for each pixel to be close to
thresh  = np.array([10.,10.,10.])
counter = 0
total   = float(nx*ny)

for k in np.arange(CN):
    # find all data points  within thresh of the kth color
    B = np.all(abs(data-CB_[k,:]) <= thresh,axis=2)
    # overwrite exisiting data there and insert extracted data
    dummy[B] *= 0. 
    dummy[B] += 1. - float(k)/(CN-1.)

# all dummy values should be > 0.; mask those that are not
data2 = dummy[:,:,0].copy()
data2  = np.ma.masked_where(data2<0.,data2)

# convert dummy values to actual data values
data2 *= abs(C0_val - C1_val)
data2 += min(C0_val,C1_val)

# Save ripped data as txt file
np.savetxt('ripped_data_from_{}.txt'.format(fname[:-4]),data2)

# plot data in new cmap
cpal = askValueS('New cmap string','viridis')

fig  = plt.figure()
ax1  = fig.add_subplot(221)
base = np.linspace(C1_val,C0_val,CN)
ax1.set_title('original cbar RGB')
# plot original cbar as RGB
ax1.plot(base,CB_[:,0],linestyle=' ',marker='o',color='r',mec='None',alpha=0.5,label='red')
ax1.plot(base,CB_[:,1],linestyle=' ',marker='^',color='g',mec='None',alpha=0.5,label='blue')
ax1.plot(base,CB_[:,2],linestyle=' ',marker='s',color='b',mec='None',alpha=0.5,label='green')
ax1.set_xlabel('data values')
ax1.set_ylabel('RGB value')
ax1.legend(loc='best',numpoints=1,fontsize='xx-small')

# plot original imae for comparison
ax2 = fig.add_subplot(222)
mplimage = imread(fname)
ax2.set_title('original image')
ax2.imshow(mplimage,origin=origin)
ax2.axis('off')

# plot new cmap
ax4 = fig.add_subplot(224)
ax4.set_title('new image (imshow)')
ims = ax4.imshow(data2,cmap=cpal,interpolation='nearest',vmin=C0_val,vmax=C1_val)
fig.colorbar(ims)

# save figure
fig.tight_layout()
fig.savefig('ripped_data_from_{}.png'.format(fname[:-4]),dpi=300,bbox_inches='tight')
sys.exit()
