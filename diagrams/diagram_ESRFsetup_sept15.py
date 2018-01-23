import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig = plt.figure(figsize=(9.,6.))
ax = fig.add_subplot(111,aspect='equal')


width  = 25.4
length = 41.0

ax.set_xlim(-1.,length+7)
ax.set_ylim(-width/2.-3,width/2.+3)

sabot   = patches.Rectangle((9.,-6.35),   width=15.,height=12.7 ,color='darkblue')
flyer   = patches.Rectangle((24.,-6.35),width=2. ,height=12.7 ,color='orchid')
drivera = patches.Rectangle((27.,-width/2.), width=1. ,height=25.4,color='c')
driverb = patches.Rectangle((28.,-5.),width=1. ,height=10. ,color='c')
Alwalla = patches.Rectangle((28.,-6.),width=11.,height=1.  ,color='.5')
Alwallb = patches.Rectangle((28.,+5.),width=11.,height=1.  ,color='.5')
sipernt = patches.Rectangle((29.,-5.),width=4. ,height=10  ,color='lightsteelblue')
window  = patches.Rectangle((33.,-5.),width=6. ,height=10  ,color='tan')

ax.text(10,12,"[TO SCALE]",rotation = 90)

ax.annotate("",xy=(10., 7.),xytext=(24., 7.), textcoords='data',arrowprops=dict(arrowstyle="<-",connectionstyle="arc3"))
ax.annotate("25 mm" ,(12.5,10.),ha='center',rotation= 90)
ax.annotate("Sabot: \nPolycarb",(15.,3.),va='center',color='w',rotation=90,ha='center')
ax.annotate("12.7 mm" ,(22.,0.),va='center',ha='center',rotation=90,color='w')
ax.annotate("",xy=(23,6.35),xytext=(23.,-6.35), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3",color='w'),rotation=90,color='w')
ax.text(13.,0.,r"$V_{impact}\approx 600ms^{-1}$",bbox=dict(boxstyle='rarrow',fc='w'),color='k')

ax.annotate("",xy=(24., -6.),xytext=(26., -6.), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3"))
ax.annotate("2 mm",xytext=(23.,-9.),xy=(25.,-6.5),arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=+0.2"),fontsize=12,rotation=90)
ax.annotate("Flyer: Cu/Polycarb",xy=(25,0.),rotation=90,ha = 'center',va='center',fontsize=10)

ax.annotate("",xy=(27., 12.7),xytext=(25.5, 12.7), textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
ax.annotate("",xy=(29.5, 12.7),xytext=(28., 12.7), textcoords='data',arrowprops=dict(arrowstyle="<-",connectionstyle="arc3"))
ax.annotate("1 mm",xy=(27.5,14.),textcoords='data',rotation=90,ha='center',va='center')

ax.annotate("",xy=(27., 8.),xytext=(25.5,8.), textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
ax.annotate("",xy=(30.5,8.),xytext=(29., 8.), textcoords='data',arrowprops=dict(arrowstyle="<-",connectionstyle="arc3"))
plt.plot([29,29],[5,9],color='k',linestyle='--')
ax.annotate("2 mm",xy=(28.25,10.),textcoords='data',rotation=90)
ax.annotate("Driver: Cu/Polycarb",xy=(28.,0.),rotation=90,ha = 'center',va='center',fontsize=10)

ax.annotate("",xy=(29., -7.),xytext=(33., -7.), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3"))
ax.annotate("4 mm" ,(31,-8.5),ha='center',va='center',rotation=90)
ax.annotate("Matrix/Chondrules:",xy=(30.,0.),rotation=90,ha = 'center',va='center',fontsize=10,color='k')
ax.annotate("Silica/Soda Lime",xy=(31.5,0.),rotation=90,ha = 'center',va='center',fontsize=10,color='k')
ax.annotate("",xy=(33., -7.),xytext=(39., -7.), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3"))
ax.annotate("6 mm" ,(36,-8.5),ha='center',va='center',rotation=90)
ax.annotate("Window: PMMA",xy=(37.,0.),rotation=90,ha = 'center',va='center',fontsize=10,color='w')

ax.annotate("",xy=(39.,6.) ,xytext=(39.,7.5), textcoords='data'  ,arrowprops=dict(arrowstyle="->",connectionstyle="arc3"),rotation=90)
ax.annotate("",xy=(39.,3.5),xytext=(39., 5.), textcoords='data'  ,arrowprops=dict(arrowstyle="<-",connectionstyle="arc3"),rotation=90)
ax.annotate("",xy=(39.,-6.) ,xytext=(39.,-7.5), textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3"),rotation=90)
ax.annotate("",xy=(39.,-3.5),xytext=(39., -5.), textcoords='data',arrowprops=dict(arrowstyle="<-",connectionstyle="arc3"),rotation=90)
ax.annotate("1 mm",xy=(39.5,5.5),textcoords='data',va='center',rotation=90)
ax.annotate("1 mm",xy=(39.5,-5.5),textcoords='data',va='center',rotation=90)
ax.annotate("Cell: Al",(35,5.5),ha='center',va='center',color='w',fontsize=10)
ax.annotate("Cell: Al",(35,-5.5),ha='center',va='center',color='w',fontsize=10)

ax.annotate("25.4 mm" ,(46.,0.),va='center',ha='center',rotation=90,color='k')
ax.annotate("",xy=(45,12.7),xytext=(45.,-12.7), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3",color='k'),rotation=90)
plt.plot([28,45],[12.7,12.7],linestyle='--',color='k')
plt.plot([28,45],[-12.7,-12.7],linestyle='--',color='k')


ax.set_xticks([])
ax.set_yticks([])


ax.add_patch(sabot)
ax.add_patch(flyer)
ax.add_patch(drivera)
ax.add_patch(driverb)
ax.add_patch(Alwalla)
ax.add_patch(Alwallb)
ax.add_patch(sipernt)
ax.add_patch(window)
ax.axis('off')
ax.plot([9,33],[5,5],linestyle=':',color='r')
ax.plot([9,33],[-5,-5],linestyle=':',color='r')
ax.plot([33,33],[-5,5],linestyle=':',color='r')

yl = np.linspace(-6.35,6.35,100)
xl = 0.5*np.sin(yl*3.)+8.5
ax.fill_betweenx(yl,xl,10.,linestyle='-',color='darkblue')
ax.annotate("10.0 mm \n (Simulated Region)" ,(11.5,0.),va='center',ha='center',rotation=90,color='r')
ax.annotate("",xy=(10.,5),xytext=(10.,-5), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3",color='r'),rotation=90,color='r')
plt.tight_layout()
fig.savefig('ESRF_setup_toscale_SPT15.pdf',format='pdf',dpi=500,bbox_inches='tight',transparent=True)
#fig.savefig('ESRF_setup_toscale.png',format='png',dpi=700,bbox_inches='tight',transparent=True)



plt.show()


