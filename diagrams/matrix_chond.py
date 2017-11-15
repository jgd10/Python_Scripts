import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

fig = plt.figure()
ax = fig.add_subplot(111,aspect='equal')
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)

chond1 = plt.Circle((3.5,0),1,color=plt.cm.viridis(0))
chond2 = plt.Circle((2.3,-3.2),1,color=plt.cm.viridis(0))
chond3 = plt.Circle((-.5,3.9),1,color=plt.cm.viridis(0))
chond4 = plt.Circle((-2.,-3.),1,color=plt.cm.viridis(0))
chond5 = plt.Circle((-3.7,1.5),1,color=plt.cm.viridis(0))

Mx = np.random.uniform(-5,5,10000)
My = np.random.uniform(-5,5,10000)

ax.plot(Mx,My,linestyle=' ',marker='.',color=plt.cm.viridis(.5),zorder=0,ms=1)
ax.text(-.5,3.9,'\'chondrules\'',ha='center',va='center',color='w',size=7)
ax.text(-3.7,1.5,'solidified...',ha='center',va='center',color='w',size=7)
ax.text(3.5,0,'...melt...',ha='center',va='center',color='w',size=7)
ax.text(-2.,-3.,'...droplets',ha='center',va='center',color='w',size=7)
ax.text(2.3,-3.2,'Scale: [mm]',ha='center',va='center',color='w',size=7)

font = FontProperties()
font.set_weight('bold')
font.set_size('large')

bbox_props = dict(boxstyle="round,pad=0.3", fc="w", ec=plt.cm.viridis(.5), lw=2)
ax.text(0.,0.,'\'matrix\'\n porous ($\phi \gtrsim$ 60%)\n dust \n Scale: [$\mu$m]',ha='center',va='center',color='k',fontproperties=font,bbox=bbox_props)

ax.axis('off')
ax.add_artist(chond1)
ax.add_artist(chond2)
ax.add_artist(chond3)
ax.add_artist(chond4)
ax.add_artist(chond5)

fig.savefig('matrixchondrules_scale.png',dpi=700,bbox_inches='tight')
