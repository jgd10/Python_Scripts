import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

fig = plt.figure(figsize=(3.5,7.))
ax = fig.add_subplot(111,aspect='equal')


width  = 24.
length = 48.

ax.set_xlim(0,width)
ax.set_ylim(length,0)

impactor = patches.Rectangle((0.,0.),   width=10.,height=30. ,color='k')
part_bed = patches.Rectangle((0.,30.),   width=10.,height=10. ,color='gray')

ax.annotate("Al" ,(5.,15.),ha='center',color='w',weight='bold',size=16)
ax.annotate("particles\n(WC)" ,(5.,35.),ha='center',color='w',weight='bold',size=16)
ax.annotate(r"1 cell = 2 $\boldsymbol{\mu}$m" ,(5.,5.),ha='center',color='w',weight='bold',size=14)
ax.annotate(r"$\boldsymbol{\phi}$ = 0.5" ,(5,38.),ha='center',color='w',weight='bold',size=16)
ax.annotate("cppr  = 8" ,(15,45.),ha='left',color='k',weight='bold',size=16)
ax.text(5.,20.,r"$V_{impact}$",bbox=dict(boxstyle='rarrow',fc='w'),color='k',rotation=-90,ha='center',size=18)

ax.annotate("",xy=(12., 0.),xytext=(12., 30.), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3"))
ax.annotate("3 mm" ,(13.,15.),ha='left',size=16)

ax.annotate("",xy=(12., 30.),xytext=(12., 40.), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3"))
ax.annotate("1 mm" ,(13.,35.),ha='left',size=16)

ax.annotate("",xy=(0., 42.),xytext=(10., 42.), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3"))
ax.annotate("1 mm" ,(5.,44.),ha='center',size=16)


ax.set_xticks([])
ax.set_yticks([])


ax.add_patch(impactor)
ax.add_patch(part_bed)
ax.axis('off')
plt.tight_layout()
plt.savefig('BNHMK_setup_toscale.pdf',format='pdf',dpi=500,bbox_inxhes='tight')
#plt.savefig('ESRF_setup_toscale.png',format='png',dpi=500,bbox_inxhes='tight')

plt.show()


