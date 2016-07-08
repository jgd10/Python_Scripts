import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig = plt.figure(figsize=(3.,3.))
ax = fig.add_subplot(111,aspect='equal')



circle1 = plt.Circle((0.,0.3),.5,color='k',alpha=.5)     ##  # give each one a color based on their material number. 
circle2 = plt.Circle((1.73,-.64),.5,color='k',alpha=.25) #
circle3 = plt.Circle((-1.,0.3),.5,color='k',alpha=.75)   ##
circle4 = plt.Circle((0.,-1.5),.5,color='k',alpha=.25)
circle5 = plt.Circle((-0.9,-1.),.5,color='k',alpha=.75)
circle6 = plt.Circle((1.,2.),.5,color='k',alpha=.25)     ##
circle7 = plt.Circle((.6,1.1),.5,color='k')              ##
circle8 = plt.Circle((-2,1),.5,color='k')
circle9 = plt.Circle((-2.5,2),.5,color='k',alpha=.75)
circle10= plt.Circle((-1.2,1.9),.5,color='k',alpha=.25)
circle11= plt.Circle((-1.4,-1.9),.5,color='k')
circle12= plt.Circle((1.2,-2.5),.5,color='k',alpha=.5)
circle13= plt.Circle((-2.4,-1),.5,color='k',alpha=.75)
circle14= plt.Circle((2.1,-1.56),.5,color='k',alpha=.5)  #
circle15= plt.Circle((2.5,0.),.5,color='k')              #
circle16= plt.Circle((2.2,1.5),.5,color='k',alpha=.5)
circle17= plt.Circle((0.1,-2.7),.5,color='k',alpha=.75)
circle18= plt.Circle((-1.3,-3.),.5,color='k',alpha=.5)
circle19= plt.Circle((3.,2.8),.5,color='k')
circle20= plt.Circle((3,-2.7),.5,color='k',alpha=.75)
circle21= plt.Circle((-2.6,-2.8),.5,color='k',alpha=.5)
circle22= plt.Circle((-2.8,3.2),.5,color='k',alpha=.75)
circle23= plt.Circle((-0.01,2.3),.5,color='k',alpha=.25)
circle24= plt.Circle((-3.3,0.01),.5,color='k',alpha=.75)

plt.plot([-1,0,.6,1],[0.3,0.3,1.1,2],color='w',lw=2.)
plt.plot([2.1,1.73,2.5],[-1.56,-.64,0],color='w',lw=2.)

ax.annotate("1",(-1.1,-0.1),color='w')
ax.annotate("2",(0.,0.),color='w')
ax.annotate("2",(0.7,.9),color='w')
ax.annotate("1",(1.15,1.9),color='w')

ax.annotate("1",(2.2,-1.56),color='w')
ax.annotate("2",(1.4,-.8),color='w')
ax.annotate("1",(2.6,0),color='w')

ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
ax.add_patch(circle1)
ax.add_patch(circle2)
ax.add_patch(circle3)
ax.add_patch(circle4)
ax.add_patch(circle5)
ax.add_patch(circle6)
ax.add_patch(circle7)
ax.add_patch(circle8)
ax.add_patch(circle9)
ax.add_patch(circle10)
ax.add_patch(circle11)
ax.add_patch(circle12)
ax.add_patch(circle13)
ax.add_patch(circle14)
ax.add_patch(circle15)
ax.add_patch(circle16)
ax.add_patch(circle17)
ax.add_patch(circle18)
ax.add_patch(circle19)
ax.add_patch(circle20)
ax.add_patch(circle21)
ax.add_patch(circle22)
ax.add_patch(circle23)
ax.add_patch(circle24)
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
plt.tight_layout()
plt.savefig('contacts_diagram.pdf',format='pdf',bbox_inches='tight',dpi=500)
plt.savefig('contacts_diagram.png',format='png',bbox_inches='tight',dpi=500)

plt.show()
