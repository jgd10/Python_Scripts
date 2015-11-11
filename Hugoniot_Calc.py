import numpy as np
import scipy as sc
import scipy.stats as stats
import pySALEPlot as psp
import matplotlib.pyplot as plt

def find_shock_front(imax,jmax,sfronts,jfronts,Pre,YC):
    dy = abs(YC[0,1] - YC[0,0])
    for i in range(0,imax,1):
	j = 1
	while sfronts[i] == 0.: 
	    j = min(j,jmax-1)							    # Prevent loop exceeding end of array
	    
	    if (Pre[i,j+1]-Pre[i,j-1])/dy > 1.:					    # 'Max' function ensures no div by zero. Pressure cutoff at 1nPa
		sfronts[i] = YC[i,j+1]
		jfronts[i] = j+1
	    else:
		pass
	    j += 1
    SF        = np.mean(sfronts)						    # Shock Front position, as a mean of all the shock fronts in j

    return sfronts, jfronts, SF
    
def plot_var_colormaps(A,B,step,sfronts,XC,YC,t,field,dirname):
    fig = plt.figure()
    fig.set_size_inches(10,10)
    ax  = fig.add_subplot(1,1,1)
    #p   = ax.pcolormesh(Pre,cmap=psp.viridis,vmin=0,vmax=3)
    den = ax.imshow(A,cmap=psp.viridis,interpolation='nearest',vmin=0,vmax=2.,origin='right')
#    tmp = ax.imshow(B,cmap='plasma',interpolation='nearest',vmin=24,vmax=350.,origin='right')
    ax.set_xlabel('y [mm]')
    ax.set_ylabel('x [mm]')
    ax.set_xticklabels(['0','1','2','3','4','5','6','7','8'])
    ax.set_yticklabels(['0','2','4','6','8','10'])
    ax.set_xlim(800,0)
    ax.set_ylim(0,1000)
    ax.set_title('t = {:3.2f} $\mu$s'.format(step.time*1.e6))
    cb = plt.colorbar(den,orientation='vertical') 
#    cb = plt.colorbar(tmp,orientation='vertical')
    cb.set_label('Density [gcm$^{-3}$] ')
#    cb.set_label(r'Temperature [$^{o}$C] ')
    ax.plot((sfronts+6.+1.)*100.,(XC[:,0])*100.,label='Shock Front',color='k')
    if t == tstart:
	plt.axvline(x=(-2.+6.+1.)*100.,label='Driver/Bed interface',color='k',linestyle='--',linewidth=2.)
	plt.axvline(x=(0.+6.+1.)*100.,label='Flyer/Driver interface',color='k',linestyle='-.',linewidth=2.)
    ax.legend(fontsize='small',loc='upper right',framealpha=0.5)
    fig.savefig('../{}/{}_shockfront{:05d}.png'.format(dirname,field,t),dpi=500)
    ax.cla

def calc_postshock_PBD_state(SF, particle_bed,YC,imax,jmax,jfronts,sfronts,posback,Den,VY):
    sfronts_n = sfronts/SF						     # Normalised shock front positions relative to mean
    SF_std    = np.std(sfronts)
    SF_std_n  = np.std(sfronts_n)
    if SF_std_n < 0.001 and particle_bed == 0:
	# The shock front is in a non-meso-scopic-porous medium (i.e. continous)
	print "At time {:3.2f} $\mu$s the Shock Front is in a continuous object".format(step.time*1.e6)
	particle_bed = 0
	PBD          = 0
    else:
	particle_bed = 1
	PBD          = 1
	# Shock front in the particle bed
    if particle_bed == 1:
	for i in range(0,imax,1):
	    j = jfronts[i]
	    while posback[i] == 0.: 
		j += 1
		j = min(j,jmax)
		Den_avg = np.mean(Den[i,j:(j+20)])			     # 20 cells being 2 particle diameters
		Den_n   = Den[i,j:(j+20)]/Den_avg

		if np.std(Den_n)<0.1:
		    posback[i]  = YC[i,j]
		    jback[i]    = j
		    #print posback[i],'back position'
		else:
		    pass
	QUP = np.mean( VY[ (YC > np.amin(sfronts)) * (YC < np.amax(posback))] )	# Quasi UP. Measured as average behind slowest SF and fastest bit of driver.

    else:
	QUP = 0.

    return QUP,PBD,posback,jback


datadir = 'All_touch/'
N = 10 
model_01 = psp.opendatfile('../{}/v150/jdata.dat'.format(datadir))					     # Open datafile 
model_02 = psp.opendatfile('../{}/v250/jdata.dat'.format(datadir))					     # Open datafile 
model_03 = psp.opendatfile('../{}/v350/jdata.dat'.format(datadir))					     # Open datafile 
model_04 = psp.opendatfile('../{}/v450/jdata.dat'.format(datadir))					     # Open datafile 
model_05 = psp.opendatfile('../{}/v550/jdata.dat'.format(datadir))					     # Open datafile 
model_06 = psp.opendatfile('../{}/v650/jdata.dat'.format(datadir))					     # Open datafile 
model_07 = psp.opendatfile('../{}/v750/jdata.dat'.format(datadir))					     # Open datafile 
model_08 = psp.opendatfile('../{}/v850/jdata.dat'.format(datadir))					     # Open datafile 
model_09 = psp.opendatfile('../{}/v950/jdata.dat'.format(datadir))					     # Open datafile 
model_10 = psp.opendatfile('../{}/v1050/jdata.dat'.format(datadir))					     # Open datafile 

models = [model_01,model_02,model_03,model_04,model_05,model_06,model_07,model_08,model_09,model_10]
modelS = ['model_01','model_02','model_03','model_04','model_05','model_06','model_07','model_08','model_09','model_10']
VI = ([150,250,350,450,550,650,750,850,950,1050])
[model.setScale('mm') for model in models]

t1 = .2 
t2 = .9 
III =0

field1 = 'Pre'                                                       
field2 = 'V_y'
field3 = 'Den'
field4 = 'TPS'
field5 = 'Tmp'
dirname= 'Hugoniots'
                                                      
tstart = 1									    # Start at t>0 such that there IS a shock present
tfin   = 250
tintv  = 1



for model in models:
        print "current model = {}".format(modelS[III])
	YC     = model.yc
	XC     = model.xc
	TME    = np.zeros(( int((tfin-tstart+1)/tintv) ))				    # time 
	YSF    = np.zeros(( int((tfin-tstart+1)/tintv) ))				    # Shock Front Y position
	PBD    = np.zeros(( int((tfin-tstart+1)/tintv) ))				    # Shock in Particle Bed array. 1 is yes 0 is no
	QUP    = np.zeros(( int((tfin-tstart+1)/tintv) ))				    # Quasi steady state Particle velocity

	Us     = np.zeros(( int((tfin-tstart+1)/(5*tintv))+1 ))
	Up     = np.zeros(( int((tfin-tstart+1)/(5*tintv))+1 ))				    # Shock and particle velocities calculated every 5 timesteps
	Time   = np.zeros(( int((tfin-tstart+1)/(5*tintv))+1 ))				    # Equivalent time every 5 timesteps (=.1 us)

	dy     = abs(YC[0,1] - YC[0,0])
	dx     = abs(XC[1,0] - XC[0,0])
	for t in range(tstart,tfin,tintv):   

	    particle_bed = 0

	    step = model.readStep(['{}'.format(field1),'{}'.format(field2),'{}'.format(field3),'{}'.format(field4),'{}'.format(field5)],t)
	    Den  = step.data[2]*1.e-3
	    Pre  = step.data[0]*1.e-9
	    VY   = step.data[1]
	    TPS  = step.data[3]*1.e-3
	    Tmp  = step.data[4] - 273						     # Convert Temp to Celsius
	    
	    imax,jmax = np.shape(Den)
	    sfronts   = np.zeros((imax))
	    jfronts   = np.zeros((imax))
	    posback   = np.zeros((imax))					     # position of back of bed
	    jback     = np.zeros((imax))					     # j coord of back of bed
            
	    sfronts, jfronts, SF = find_shock_front(imax,jmax,sfronts,jfronts,Pre,YC)	    

	    print 'Time: {:3.2f} us'.format(step.time*1.e6)
	    print 'Average Shock Front position = {:3.2f} mm'.format(SF)
	    print 'Max SF position: {:3.2f} mm, Min SF position: {:3.2f} mm \n'.format(np.amax(sfronts),np.amin(sfronts))
	    
	    YSF[t]  = np.mean(sfronts)
	    TME[t]  = step.time*1.e6

	    QUP,PBD,posback,jback = calc_postshock_PBD_state(SF,particle_bed,YC,imax,jmax,jfronts,sfronts,posback,Den,VY) 

	    if t%5 == 0:
		slope, intercept, other, things, too = stats.linregress(TME[t-4:t],YSF[t-4:t])
		Us[t/5] = abs(slope*1000.)
		Up[t/5] =abs( np.mean(QUP[t-4:t]))
		Time[t/5] = TME[t]

	plt.figure()
	plt.plot(TME,abs(YSF),linestyle=' ',marker = 'p',color = 'm')
	plt.ylabel('Shock Position [mm]')
	plt.xlabel('Time [$\mu$s]')
	plt.savefig('../{}/{}/YSF-TME.png'.format(dirname,ModelS[III]),dpi=300)

	plt.figure()
	plt.plot(Time, Us, linestyle=' ',marker = 'o', color = 'r')
	plt.ylabel('Shock Velocity [m$^{-1}$]')
	plt.xlabel('Time [$\mu$s]')
	plt.savefig('../{}/{}/Us-Time.png'.format(dirname,ModelS[III]),dpi=300)

	plt.figure()
	plt.plot(Time, Up, linestyle=' ',marker = 'd', color = 'b')
	plt.xlabel('Time [$\mu$s]')
	plt.ylabel('Particle Velocity [m$^{-1}$]')
	plt.savefig('../{}/{}/Up-Time.png'.format(dirname,ModelS[III]),dpi=300)
	III += 1




