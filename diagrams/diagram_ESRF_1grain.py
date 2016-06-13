import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig = plt.figure(figsize=(9.,6.))
ax = fig.add_subplot(111,aspect='equal')


width  = 25.4
length = 41.0

ax.set_xlim(0.,8)
ax.set_ylim(-width/3.,width/3.)

drivera = patches.Rectangle((0,-width/2.), width=1. ,height=25.4,color='c')
driverb = patches.Rectangle((1.,-5.),width=1. ,height=10. ,color='c')
Alwalla = patches.Rectangle((1.,-6.),width=13.,height=1.  ,color='.5')
Alwallb = patches.Rectangle((1.,+5.),width=13.,height=1.  ,color='.5')
sipernt = patches.Rectangle((2.,-5.),width=6. ,height=10  ,color='lightsteelblue')
window  = patches.Rectangle((8.,-5.),width=6. ,height=10  ,color='tan')
bead    = patches.Circle((3.,0.),.5,color='k')

ax.annotate("1 mm",xy=(5.,0.),color='k',ha='center',va='center')
ax.annotate("",xy=(4,-.5),xytext=(4.,.5), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3",color='k'),rotation=90,color='k')

ax.annotate("1 mm",xy=(3.,-2.),color='k',ha='center',va='center')
ax.annotate("",xy=(2.5,-1),xytext=(3.5,-1), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3",color='k'),color='k')






ax.set_xticks([])
ax.set_yticks([])


ax.add_patch(drivera)
ax.add_patch(driverb)
ax.add_patch(Alwalla)
ax.add_patch(Alwallb)
ax.add_patch(sipernt)
ax.add_patch(window)
ax.add_patch(bead)
ax.axis('off')
plt.tight_layout()
plt.savefig('ESRF_onegrain_diagram.pdf',format='pdf',dpi=500,bbox_inxhes='tight')
plt.savefig('ESRF_onegrain_diagram.png',format='png',dpi=500,bbox_inxhes='tight')

plt.show()


