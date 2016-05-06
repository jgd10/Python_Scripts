import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig = plt.figure(figsize=(3.,3.))
ax = fig.add_subplot(111,aspect='equal')



circle1 = plt.Circle((0.,0.3),.5,color='k',alpha=.5)  # give each one a color based on their material number. 
circle2 = plt.Circle((.77,-.64),.5,color='k',alpha=.25)
circle3 = plt.Circle((-1.,0.),.5,color='k',fill=False)
arc     = patches.Arc((0.,0.3),1.25,1.25,angle=90,theta1=180,theta2=230)

plt.plot([0.,0.77],[0.3,-0.64],color='k',linestyle='--',marker='o',markerfacecolor='None')
plt.plot([0.,0.77],[0.3,-0.34],color='k',linestyle='-')
plt.plot([0.,0.],[0.3,-.7],color='k',marker=' ')
plt.plot([0.,-1.],[0.3,0],color='k',linestyle='--',marker='o',markerfacecolor='None')
plt.axvline(x=0.77,linestyle=':',color='k')
plt.axvline(x=-0.77,linestyle=':',color='k')
plt.text(0.05,-0.5,r'$\theta_{crit}$')

ax.annotate("",xy=(-.77, -1.7),xytext=(0.77, -1.7), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3"))
ax.annotate("critical range",(-.55,-2.))
ax.set_xticks([])
ax.set_yticks([])

ax.add_patch(circle1)
ax.add_patch(circle2)
ax.add_patch(circle3)
ax.add_patch(arc)
ax.set_xlim(-1.5,1.27)
ax.set_ylim(-2.27,.8)
plt.tight_layout()
plt.savefig('critical_angle_diagram.pdf',format='pdf',bbox_inches='tight')

plt.show()
