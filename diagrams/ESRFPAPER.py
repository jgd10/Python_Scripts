import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{wasysym}')

fig = plt.figure(figsize=(6.,3.))
ax = fig.add_subplot(111,aspect='equal')


width  = 41.0
length = 25.4

ax.set_xlim(20.,80)
ax.set_ylim(-25,+25)

mag = plt.cm.magma

sabot   = patches.Rectangle((9.,-6.35),       width=15.,height=12.7,color='silver')#'darkblue')
ax.annotate(r"$600 \text{ms}^{-1}$",xy=(22,0.),textcoords='data',ha='center',va='center',rotation=90,fontsize=7)
impact = patches.Arrow(20,-10,5,0,color='k',width=4,fill=False)

#sabota  = patches.Rectangle((7.5,-6.35),      width=1., height=12.7,color=mag(0.0))#'darkblue')
#sabotb  = patches.Rectangle((6.25,-6.35),      width=.75,height=12.7,color=mag(0.0))#'darkblue')
#sabotc  = patches.Rectangle((5.25,-6.35),      width=.5, height=12.7,color=mag(0.0))#'darkblue')

flyer   = patches.Rectangle((24.,-6.35),      width=2. ,height=12.7,color='brown')#'orchid')
drivera = patches.Rectangle((27.,-length/2.),  width=1.,height=25.4,color='silver')#'c')
driverb = patches.Rectangle((28.,-5.),        width=1. ,height=10. ,color='silver')#'c')
Alwalla = patches.Rectangle((28.,-6.),        width=11.,height=1.  ,color='grey')#'.5')
Albrima = patches.Rectangle((28.,-12.7),        width=1., height=7.55  ,color='grey')#'.5')
Alwallb = patches.Rectangle((28.,5.),         width=11.,height=1.  ,color='grey')#'.5')
Albrimb = patches.Rectangle((28.,5.),         width=1., height=7.7  ,color='grey')#'.5')
sipernt = patches.Rectangle((29.,-5.),        width=6. ,height=10.  ,color='plum',ls='--')#'lightsteelblue')
window  = patches.Rectangle((35.,-5.),        width=6. ,height=10.  ,color='beige')#'tan')

# scale bar
#scale_1 = patches.Rectangle((31.5,-12.7),        width=3. ,height=6.  ,color='k')#'tan')
#ax.annotate("6 mm",xy=(33.,-9.7),textcoords='data',ha='center',va='center',rotation=90,color='w',fontsize=9)


# 2.5x bigger
siperntZOOM = patches.Rectangle((50.,-12.5),        width=15. ,height=25.  ,color='plum')#'lightsteelblue')
siperntZOOMbox = patches.Rectangle((50.,-12.5),        width=15. ,height=25.  ,color='k',fill=False,ls='--')#'lightsteelblue')
ax.annotate(r"$\times 2.5$ zoom",xy=(52,6.),textcoords='data',ha='center',va='center',rotation=90)
ax.annotate("",xy=(50., -13.5),xytext=(65., -13.5), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3"))
ax.annotate("6 mm",xy=(57.5,-14.5),textcoords='data',ha='center',va='center',rotation=180)
ax.annotate("",xy=(64., -12.5),xytext=(64., 12.5), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3"))
ax.annotate("10 mm",xy=(62.5,0.),textcoords='data',ha='center',va='center',rotation=90)
rod     = patches.Rectangle((55.-1.25,-10.), width=2.5,height=20,color='blue')
rod2    = patches.Circle((55.,16.), radius = 1.25,color='b')
ax.plot([55.,55.],[9.,18.],linestyle=':',color='k',lw=.8)
ax.plot([53.,57.],[16.,16.],linestyle=':',color='k',lw=.8)
ax.annotate("borosilicate rod",xy=(55,0.),textcoords='data',ha='center',va='center',rotation=90,color='w')

# simulated region
simregn = patches.Rectangle((18,-12.7),width=24,height=25.4,fill=False,color='k',ls='-.',zorder=3,label='iSALE')
# x-ray fov
xrayfov = patches.Rectangle((27,-6.05),width=12.1,height=12.1,fill=False,color='r',ls=':',zorder=3,label='X-ray FoV')

# legend for the above
ax.annotate("iSALE FoV" ,(74,-14),ha='center',va='center',fontsize=8,rotation=90)
ax.annotate("X-ray FoV" ,(78,-14),ha='center',va='center',fontsize=8,rotation=90)
ax.plot([74,74],[-25,-20],color='k',linestyle='-.')
ax.plot([78,78],[-25,-20],color='r',linestyle=':')

# sipernat grains zoomed in
zmcirc  = patches.Circle((72.,17.), radius=5.,fill=False,ls='--',lw=1.,color='k')
grainsx = [69,74,72,70,75,69,71,74,73,71]
grainsy = [15,20,16,16,19,17,14,14,15,20]
ax.plot(grainsx,grainsy,marker='.',color='plum',linestyle=' ',mec='k')
ax.annotate("Matrix (Silica)" ,(69,4),ha='center',va='center',fontsize=8,rotation=90)
ax.annotate(r"$\diameter\approx7~\mu$m" ,(72,4),ha='center',va='center',fontsize=8,rotation=90)
ax.annotate(r"$\phi \approx 70\%$" ,(75,4),ha='center',va='center',fontsize=8,rotation=90)
ax.annotate(r"iSALE: $\epsilon-\alpha$ model" ,(78,4),ha='center',va='center',fontsize=8,rotation=90)

# zoom lines
ax.plot([35,50],[-5,-12.5],color='k',ls='--',zorder=10,lw=1)
ax.plot([35,50],[5,12.5],color='k',ls='--',zorder=10,lw=1)
ax.plot([61,67],[8,18],color='k',ls='--',zorder=10,lw=1)
ax.plot([61,73],[8,12],color='k',ls='--',zorder=10,lw=1)

#pre-shot xrays
xrays1 = patches.Arrow(28,-25,0,10,color='k',width=5,fill=False)
xrays2 = patches.Arrow(32,-25,0,10,color='k',width=5,fill=False)
xrays3 = patches.Arrow(36,-25,0,10,color='k',width=5,fill=False)
ax.annotate("X-RAYS" ,(25,-20.5),ha='center',va='center',fontsize=8,rotation=90)

# post-shot xrays & beamsplitter
ax.plot([32,32,23],[12.7,18,18],linestyle='-',color='k',lw=.8,zorder=0)
ax.plot([32,32],[18,22],linestyle='-',color='k',lw=.8,zorder=0)
ax.plot([31,33],[19,17],linestyle='-',color='grey',zorder=0)

#pimax cameras
pimax1 = patches.Rectangle((28,22),width=8.,height=3,fill=False,lw=1.,color='k')
pimax2 = patches.Rectangle((20.,14),width=3.,height=8,fill=False,lw=1.,color='k')
ax.annotate("PIMAX" ,(32,23.5),ha='center',va='center',fontsize=8)
ax.annotate("PIMAX" ,(21.5,18),ha='center',va='center',rotation=90,fontsize=8)


#rod     = patches.Circle((31.,0.),radius=.5,color=mag(0.6))#'purple')

#ax.text(10,12,"[TO SCALE]",fontsize=24)
#
#ax.annotate("",xy=(5., 7.),xytext=(24., 7.), textcoords='data',arrowprops=dict(arrowstyle="<-",connectionstyle="arc3"))
#ax.annotate("25 mm" ,(12.5,8.),ha='center')
#ax.annotate("Sabot: \nPolycarbonate (PC)",(15.,3.),va='center',color='w',ha='center')
#ax.annotate("12.7 mm" ,(22.,0.),va='center',ha='center',color='w',rotation=90)
#ax.annotate("",xy=(23,6.35),xytext=(23.,-6.35), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3",color='w'),rotation=90,color='w')
#ax.text(13.,0.,r"$V_{impact}\approx 600ms^{-1}$",bbox=dict(boxstyle='rarrow',fc='w'),color='k')
#
#ax.annotate("",xy=(24., -6.),xytext=(26., -6.), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3"))
#ax.annotate("2 mm",xytext=(23.,-9.),xy=(25.,-6.5),arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=+0.2"),fontsize=12)
#ax.annotate("Flyer: Cu/PC",xy=(25,0.),rotation=90,ha = 'center',va='center',fontsize=10)
#
#ax.annotate("",xy=(27., 12.7),xytext=(25.5, 12.7), textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
#ax.annotate("",xy=(29.5, 12.7),xytext=(28., 12.7), textcoords='data',arrowprops=dict(arrowstyle="<-",connectionstyle="arc3"))
#ax.annotate("1 mm",xy=(27.5,14.),textcoords='data',ha='center',va='center')
#
#ax.annotate("",xy=(27., 8.),xytext=(25.5,8.), textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
#ax.annotate("",xy=(30.5,8.),xytext=(29., 8.), textcoords='data',arrowprops=dict(arrowstyle="<-",connectionstyle="arc3"))
#plt.plot([29,29],[5,9],color='k',linestyle='--')
#ax.annotate("2 mm",xy=(28.25,10.),textcoords='data')
#ax.annotate("Driver: Cu/PC",xy=(28.,0.),rotation=90,ha = 'center',va='center',fontsize=10)
#
#ax.annotate("",xy=(29., -7.),xytext=(35., -7.), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3"))
#ax.annotate("6 mm" ,(32,-8.5),ha='center',va='center')
#ax.annotate("Matrix (Silica $\phi = 70$%)",xy=(34.,0.),rotation=90,ha = 'center',va='center',fontsize=10,color='w')
#ax.annotate("Rod (1 mm Diameter)",xy=(30.,0.),rotation=90,ha = 'center',va='center',fontsize=10,color='w')
#ax.annotate("",xy=(35., -7.),xytext=(41., -7.), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3"))
#ax.annotate("6 mm" ,(38,-8.5),ha='center',va='center')
#ax.annotate("Window: PMMA",xy=(37.,0.),ha = 'center',va='center',fontsize=10,color='w',rotation=90)
#
#ax.annotate("9.8 mm" ,(39.,0.),va='center',ha='center',rotation=90,color='w')
#ax.annotate("",xy=(40,4.9),xytext=(40,-4.9), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3",color='w'),rotation=90,color='w')
#
#ax.annotate("",xy=(41.,6.) ,xytext=(41.,7.5), textcoords='data'  ,arrowprops=dict(arrowstyle="->",connectionstyle="arc3"),rotation=90)
#ax.annotate("",xy=(41.,3.5),xytext=(41., 5.), textcoords='data'  ,arrowprops=dict(arrowstyle="<-",connectionstyle="arc3"),rotation=90)
#ax.annotate("",xy=(41.,-6.) ,xytext=(41.,-7.5), textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3"),rotation=90)
#ax.annotate("",xy=(41.,-3.5),xytext=(41., -5.), textcoords='data',arrowprops=dict(arrowstyle="<-",connectionstyle="arc3"),rotation=90)
#plt.plot([39,41],[6,6],linestyle='--',color='k')
#plt.plot([39,41],[-6,-6],linestyle='--',color='k')
#ax.annotate("1 mm",xy=(41.5,5.5),textcoords='data',va='center')
#ax.annotate("1 mm",xy=(41.5,-5.5),textcoords='data',va='center')
#ax.annotate("Cell: Al",(35,5.5),ha='center',va='center',color='w',fontsize=10)
#ax.annotate("Cell: Al",(35,-5.5),ha='center',va='center',color='w',fontsize=10)
#ax.annotate("",xy=(28., 7.),xytext=(39., 7.), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3"))
#ax.annotate("11 mm" ,(34,8.5),ha='center',va='center')
#
#ax.annotate("25.4 mm" ,(46.,0.),va='center',ha='center',rotation=90,color='k')
#ax.annotate("",xy=(45,12.7),xytext=(45.,-12.7), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3",color='k'),rotation=90)
#plt.plot([28,45],[12.7,12.7],linestyle='--',color='k')
#plt.plot([28,45],[-12.7,-12.7],linestyle='--',color='k')


ax.set_xticks([])
ax.set_yticks([])


ax.add_patch(sabot)
#ax.add_patch(sabota)
#ax.add_patch(sabotb)
#ax.add_patch(sabotc)
ax.add_patch(sipernt)
ax.add_patch(window)
ax.add_patch(flyer)
ax.add_patch(drivera)
ax.add_patch(driverb)
ax.add_patch(Alwalla)
ax.add_patch(Alwallb)
ax.add_patch(Albrima)
ax.add_patch(Albrimb)
ax.add_patch(siperntZOOM)
ax.add_patch(siperntZOOMbox)
ax.add_patch(zmcirc)
ax.add_patch(xrays1)
ax.add_patch(xrays2)
ax.add_patch(xrays3)
ax.add_patch(pimax1)
ax.add_patch(pimax2)
ax.add_patch(simregn)
ax.add_patch(xrayfov)
ax.add_patch(rod)
ax.add_patch(rod2)
ax.add_patch(impact)
#ax.add_patch(scale_1)

#ax.add_patch(rod)
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
#fig.savefig('ESRF_setup_toscale_MAY16_ROT-90.pdf',format='pdf',dpi=500,bbox_inches='tight',transparent=True)
fig.savefig('figure1.pdf',format='pdf',dpi=500,bbox_inches='tight',transparent=True)
fig.savefig('figure1.png',format='png',dpi=500,bbox_inches='tight',transparent=True)



plt.show()


