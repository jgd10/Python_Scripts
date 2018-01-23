import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
plt.rc('text', usetex=True)
plt.rc('font',family='sans serif')
plt.rc('font',serif='helvetica')
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

fig = plt.figure(figsize=(5.,5.))
ax = fig.add_subplot(111,aspect='equal')



circle1 = plt.Circle((0.,0.),radius=1.,color='k',fill=False)     ##  # give each one a color based on their material number. 
circle2 = plt.Circle((np.sqrt(2.),np.sqrt(2.)),radius=1.,color='k',fill=False) #
circle3 = plt.Circle((-1.,2.6),radius=1.,color='k',fill=False) #
circle4 = plt.Circle((2.6,-1.),radius=1.,color='k',fill=False) #

ax.plot([0,np.sqrt(2.)],[0,np.sqrt(2.)],marker='x',mew=2.,color='k',linestyle=':')

ax.annotate('',xycoords='data',xytext=(np.sqrt(2.)*.5,np.sqrt(2.)*.5),xy=(1.,1.),arrowprops=dict(facecolor='black',width=.5))
label = '$n^{k} = (n_{x}^{k},n_{y}^{k})$' 
ax.text(.9,0.7,label,ha='left',va='center',size=16)
ax.text(0.,-0.2,'$k$th particle',ha='center',va='center',size=16)
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')

ax.add_patch(circle1)
ax.add_patch(circle2)
ax.add_patch(circle3)
ax.add_patch(circle4)

ax.set_xlim(-1.1,2.6)
ax.set_ylim(-1.1,2.6)
plt.tight_layout()
plt.savefig('contacts_diagram2.pdf',format='pdf',bbox_inches='tight',dpi=500)
#plt.savefig('contacts_diagram.png',format='png',bbox_inches='tight',dpi=500)

plt.show()
