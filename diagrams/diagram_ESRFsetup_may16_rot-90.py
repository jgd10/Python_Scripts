import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig = plt.figure(figsize=(6.*1.77777778,6.))
ax = fig.add_subplot(111,aspect='equal')


width  = 41.0
length = 25.4

ax.set_xlim(5.,width+7)
ax.set_ylim(-length/2.-3,length/2.+3)

mag = plt.cm.magma

sabot   = patches.Rectangle((9.,-6.35),       width=15.,height=12.7,color=mag(0.0))#'darkblue')

sabota  = patches.Rectangle((7.5,-6.35),      width=1., height=12.7,color=mag(0.0))#'darkblue')
sabotb  = patches.Rectangle((6.25,-6.35),      width=.75,height=12.7,color=mag(0.0))#'darkblue')
sabotc  = patches.Rectangle((5.25,-6.35),      width=.5, height=12.7,color=mag(0.0))#'darkblue')

flyer   = patches.Rectangle((24.,-6.35),      width=2. ,height=12.7,color=mag(0.9))#'orchid')
drivera = patches.Rectangle((27.,-length/2.),  width=1.,height=25.4,color=mag(0.8))#'c')
driverb = patches.Rectangle((28.,-5.),        width=1. ,height=10. ,color=mag(0.8))#'c')
Alwalla = patches.Rectangle((28.,-6.),        width=11.,height=1.  ,color=mag(0.7))#'.5')
Alwallb = patches.Rectangle((28.,5.),         width=11.,height=1.  ,color=mag(0.7))#'.5')
sipernt = patches.Rectangle((29.,-5.),        width=6. ,height=10  ,color=mag(0.3))#'lightsteelblue')
window  = patches.Rectangle((35.,-5.),        width=6. ,height=10  ,color=mag(0.5))#'tan')
rod     = patches.Circle((31.,0.),radius=.5,color=mag(0.6))#'purple')

ax.text(10,12,"[TO SCALE]",fontsize=24)

ax.annotate("",xy=(5., 7.),xytext=(24., 7.), textcoords='data',arrowprops=dict(arrowstyle="<-",connectionstyle="arc3"))
ax.annotate("25 mm" ,(12.5,8.),ha='center')
ax.annotate("Sabot: \nPolycarbonate (PC)",(15.,3.),va='center',color='w',ha='center')
ax.annotate("12.7 mm" ,(22.,0.),va='center',ha='center',color='w',rotation=90)
ax.annotate("",xy=(23,6.35),xytext=(23.,-6.35), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3",color='w'),rotation=90,color='w')
ax.text(13.,0.,r"$V_{impact}\approx 600ms^{-1}$",bbox=dict(boxstyle='rarrow',fc='w'),color='k')

ax.annotate("",xy=(24., -6.),xytext=(26., -6.), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3"))
ax.annotate("2 mm",xytext=(23.,-9.),xy=(25.,-6.5),arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=+0.2"),fontsize=12)
ax.annotate("Flyer: Cu/PC",xy=(25,0.),rotation=90,ha = 'center',va='center',fontsize=10)

ax.annotate("",xy=(27., 12.7),xytext=(25.5, 12.7), textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
ax.annotate("",xy=(29.5, 12.7),xytext=(28., 12.7), textcoords='data',arrowprops=dict(arrowstyle="<-",connectionstyle="arc3"))
ax.annotate("1 mm",xy=(27.5,14.),textcoords='data',ha='center',va='center')

ax.annotate("",xy=(27., 8.),xytext=(25.5,8.), textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
ax.annotate("",xy=(30.5,8.),xytext=(29., 8.), textcoords='data',arrowprops=dict(arrowstyle="<-",connectionstyle="arc3"))
plt.plot([29,29],[5,9],color='k',linestyle='--')
ax.annotate("2 mm",xy=(28.25,10.),textcoords='data')
ax.annotate("Driver: Cu/PC",xy=(28.,0.),rotation=90,ha = 'center',va='center',fontsize=10)

ax.annotate("",xy=(29., -7.),xytext=(35., -7.), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3"))
ax.annotate("6 mm" ,(32,-8.5),ha='center',va='center')
ax.annotate("Matrix (Silica $\phi = 70$%)",xy=(34.,0.),rotation=90,ha = 'center',va='center',fontsize=10,color='w')
ax.annotate("Rod (1 mm Diameter)",xy=(30.,0.),rotation=90,ha = 'center',va='center',fontsize=10,color='w')
ax.annotate("",xy=(35., -7.),xytext=(41., -7.), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3"))
ax.annotate("6 mm" ,(38,-8.5),ha='center',va='center')
ax.annotate("Window: PMMA",xy=(37.,0.),ha = 'center',va='center',fontsize=10,color='w',rotation=90)

ax.annotate("9.8 mm" ,(39.,0.),va='center',ha='center',rotation=90,color='w')
ax.annotate("",xy=(40,4.9),xytext=(40,-4.9), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3",color='w'),rotation=90,color='w')

ax.annotate("",xy=(41.,6.) ,xytext=(41.,7.5), textcoords='data'  ,arrowprops=dict(arrowstyle="->",connectionstyle="arc3"),rotation=90)
ax.annotate("",xy=(41.,3.5),xytext=(41., 5.), textcoords='data'  ,arrowprops=dict(arrowstyle="<-",connectionstyle="arc3"),rotation=90)
ax.annotate("",xy=(41.,-6.) ,xytext=(41.,-7.5), textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3"),rotation=90)
ax.annotate("",xy=(41.,-3.5),xytext=(41., -5.), textcoords='data',arrowprops=dict(arrowstyle="<-",connectionstyle="arc3"),rotation=90)
plt.plot([39,41],[6,6],linestyle='--',color='k')
plt.plot([39,41],[-6,-6],linestyle='--',color='k')
ax.annotate("1 mm",xy=(41.5,5.5),textcoords='data',va='center')
ax.annotate("1 mm",xy=(41.5,-5.5),textcoords='data',va='center')
ax.annotate("Cell: Al",(35,5.5),ha='center',va='center',color='w',fontsize=10)
ax.annotate("Cell: Al",(35,-5.5),ha='center',va='center',color='w',fontsize=10)
ax.annotate("",xy=(28., 7.),xytext=(39., 7.), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3"))
ax.annotate("11 mm" ,(34,8.5),ha='center',va='center')

ax.annotate("25.4 mm" ,(46.,0.),va='center',ha='center',rotation=90,color='k')
ax.annotate("",xy=(45,12.7),xytext=(45.,-12.7), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3",color='k'),rotation=90)
plt.plot([28,45],[12.7,12.7],linestyle='--',color='k')
plt.plot([28,45],[-12.7,-12.7],linestyle='--',color='k')


ax.set_xticks([])
ax.set_yticks([])


ax.add_patch(sabot)
ax.add_patch(sabota)
ax.add_patch(sabotb)
ax.add_patch(sabotc)
ax.add_patch(flyer)
ax.add_patch(drivera)
ax.add_patch(driverb)
ax.add_patch(Alwalla)
ax.add_patch(Alwallb)
ax.add_patch(sipernt)
ax.add_patch(window)
ax.add_patch(rod)
ax.axis('off')
"""
ax.plot([9,35],[5,5],linestyle=':',color='r')
ax.plot([9,35],[-5,-5],linestyle=':',color='r')
ax.plot([35,35],[-5,5],linestyle=':',color='r')
"""
yl = np.linspace(-6.35,6.35,100)
xl = 0.5*np.sin(yl*3.)+8.5
#ax.annotate("10.0 mm \n (Simulated Region)" ,(11.5,0.),va='center',ha='center',rotation=90,color='r')
#ax.annotate("",xy=(10.,5),xytext=(10.,-5), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3",color='r'),rotation=90,color='r')
plt.tight_layout()
fig.savefig('ESRF_setup_toscale_MAY16_ROT-90.pdf',format='pdf',dpi=500,bbox_inches='tight',transparent=True)
fig.savefig('ESRF_setup_toscale_MAY16_ROT-90.png',format='png',dpi=700,bbox_inches='tight',transparent=True)



plt.show()


