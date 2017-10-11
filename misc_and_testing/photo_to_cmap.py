import matplotlib.pyplot as plt
from scipy.misc import imread
from sys import argv

im = imread(argv[1],flatten=True)

fname = argv[1]
fname = fname[:-4]


fig = plt.figure(figsize=(8,8))

ax1 = fig.add_subplot(4,4,1,aspect='equal')
ax2 = fig.add_subplot(4,4,2,aspect='equal')
ax3 = fig.add_subplot(4,4,3,aspect='equal')
ax4 = fig.add_subplot(4,4,4,aspect='equal')
ax5 = fig.add_subplot(4,4,5,aspect='equal')
ax6 = fig.add_subplot(4,4,6,aspect='equal')
ax7 = fig.add_subplot(4,4,7,aspect='equal')
ax8 = fig.add_subplot(4,4,8,aspect='equal')
ax9 = fig.add_subplot(4,4,9,aspect='equal')
ax10 = fig.add_subplot(4,4,10,aspect='equal')
ax11 = fig.add_subplot(4,4,11,aspect='equal')
ax12 = fig.add_subplot(4,4,12,aspect='equal')
ax13 = fig.add_subplot(4,4,13,aspect='equal')
ax14 = fig.add_subplot(4,4,14,aspect='equal')
ax15 = fig.add_subplot(4,4,15,aspect='equal')
ax16 = fig.add_subplot(4,4,16,aspect='equal')

axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16]
cmaps= ['PiYG','rainbow','viridis','inferno','magma','plasma','RdYlBu_r','cubehelix',
        'coolwarm','BuPu_r','bone','jet','YlGnBu_r','gnuplot','ocean','copper']

for ax,c in zip(axes,cmaps):
    ax.imshow(im,cmap=c)
    ax.axis('off')
    ax.set_title(c)

fig.tight_layout()
fig.savefig('{}_in_many_colors1.png'.format(fname),bbox_inches='tight')
