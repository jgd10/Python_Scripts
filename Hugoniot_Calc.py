import numpy as np
import scipy as sc
import scipy.stats as stats
import pySALEPlot as psp
import matplotlib.pyplot as plt

def two_lines(x, a, b, c, d):
    one = a*x + b
    two = c*x + d
    return np.minimum(one, two)

def thr_lines(x, a, b, c, d, e, f):
    one = a*x + b
    two = c*x + d
    thr = e*x + f
    A = np.minimum(one,two)
    B = np.maximum(A,thr)
    return B
def twoline_piecewise_fit(Up,Us,g):
    guess = g	       #slope, incpt, slope, incpt
    fit, cov = sc.optimize.curve_fit(two_lines, Up, Us, p0 = guess)
 
    crossover = (fit[3] - fit[1]) / (fit[0] - fit[2])
    X_up = float(crossover)
    Y_up = fit[0]*X_up + fit[1]

    up_lin1 = np.arange(np.amin(Up),X_up,0.01)
    up_lin2 = np.arange(X_up,np.amax(Up),0.01)
    us_lin1 = up_lin1*fit[0] + fit[1]
    us_lin2 = up_lin2*fit[2] + fit[3]

    return up_lin1,us_lin1,up_lin2,us_lin2

def outlier_mean(A):
    q75, q25 = np.percentile(A,[75,25])
    IQR = q75 - q25

    B = A[A>(q25-1.5*IQR)]

    b = np.mean(B)

    return b


def thrline_piecewise_fit(Up,Us,g):
    guess = g					#slope, incpt, slope, incpt, slope, incpt
    fit, cov = sc.optimize.curve_fit(thr_lines, Up, Us, p0 = guess)
 
    crossover_a = (fit[3] - fit[1]) / (fit[0] - fit[2])
    crossover_b = (fit[5] - fit[3]) / (fit[2] - fit[4])
    X_upa = float(crossover_a)
    Y_upa = fit[0]*X_upa + fit[1]
    X_upb = float(crossover_b)
    Y_upb = fit[2]*X_upb + fit[3]
    print X_upa,Y_upa
    print X_upb,Y_upb
    print 'a={},b={},c={},d={},e={},f={}'.format(fit[0],fit[1],fit[2],fit[3],fit[4],fit[5])
    up_lin1 = np.arange(np.amin(Up),X_upa,0.01)
    up_lin2 = np.arange(X_upa,X_upb,0.01)
    up_lin3 = np.arange(X_upb,np.amax(Up),0.01)
    us_lin1 = up_lin1*fit[0] + fit[1]
    us_lin2 = up_lin2*fit[2] + fit[3]
    us_lin3 = up_lin3*fit[4] + fit[5]

    return up_lin1,us_lin1,up_lin2,us_lin2,up_lin3,us_lin3

def find_shock_front(imax,jmax,sfronts,jfronts,Pre,YC):
    dy = abs(YC[0,1] - YC[0,0])
    for i in range(0,imax,1):
	j = 1
	while sfronts[i] == 0. and YC[i,j] < (0.0 - dy): 
	    j = min(j,jmax-3)							    # Prevent loop exceeding end of array
	    if YC[i,j] > (0.0-dy): J = j
	    
	    if (Pre[i,j+1]-Pre[i,j])/dy >= 1.: 
#if Pre[i,j+3] > 0. and Pre[i,j-3] > 0.: #if (Pre[i,j+1]-Pre[i,j-1])/dy > 1.:					    # 'Max' function ensures no div by zero. Pressure cutoff at 1nPa
		sfronts[i] = min(YC[i,j+1],0.)
		if sfronts[i] == 0.: 
			jfronts[i] = J
		else:
			jfronts[i] = j+1
	    else:
		pass
	    j += 1
    SF = np.mean(sfronts)						    # Shock Front position, as a mean of all the shock fronts in j

    return sfronts, jfronts, SF

def find_shock_front_2(imax,jmax,sfronts,jfronts,VY,YC,A_lim):
    for i in range(0,imax,1):
        j = 1
        while sfronts[i] == 0. and YC[i,j] < (0.0 - dy): 
            j = min(j,jmax-3)							    # Prevent loop exceeding end of array
            if YC[i,j] > (0.0-dy): J = j
            if -1.*VY[i,j] > A_lim: 
                sfronts[i] = min(YC[i,j+1],0.)
                if sfronts[i] == 0.: 
                    jfronts[i] = J
                else:
                    jfronts[i] = j+1
                break
            else:
                pass
            j += 1
    SF = np.mean(sfronts)						    # Shock Front position, as a mean of all the shock fronts in j
    
    return sfronts, jfronts, SF
    
def plot_var_colormaps(A,model,step,sfronts,posback,t,field,dirname,datadir,modeldir):
    fig = plt.figure()
    A  /= np.mean(A)
    ASD = np.std(A)
    ax  = fig.add_subplot(111,aspect='equal')
    var = ax.pcolormesh(model.x,model.y,A,cmap=psp.viridis,vmin=0.,vmax=1.+2.*ASD)
    ax.set_xlim(0.,1.)
    ax.set_ylim(-1.,0.)
    ax.set_xlabel('Transverse Position [mm]')
    ax.set_ylabel('Longitudinal Position [mm]')
    ax.set_title('t = {:3.2f} $\mu$s'.format(step.time*1.e6))
    N = np.size(sfronts)
    X = np.arange(0.,1.0,1./N)
    MSF = np.mean(A)
    ax.axhline(y=MSF,linestyle='--',color='0.75',label = 'Mean Shock Front',lw=2.)
    ax.plot(X,sfronts,label='Shock Front',color='c',lw=2.)
    ax.plot(X,posback,label='Impactor Front',color='r',lw=2.)
    ax.legend(fontsize='medium',loc='upper right',framealpha=0.5)
    cb=fig.colorbar(var)
    cb.set_label('Velocity/[Mean vel over mesh](={:3.2f}km/s)]'.format(MSF),fontsize=10)
    fig.savefig('./{}/cmaps/{}_{}_{}_shockfront{:05d}.png'.format(dirname,datadir,modeldir,field,t),dpi=300)
    #plt.show()
    plt.close()

def calc_postshock_PBD_state(SF,particle_bed,YC,imax,jfronts,sfronts,posback,Den,VY):
    """
    ***
    This function calculates the post-shock particle bed state. It does this by individually looking
    at each longitudinal column of cells. It starts at the current shock position and scans back
    though the bed until it reaches the impactor.
    ***
    It will only activate if the shock wave is in a non-continuous material
    """
    sfronts_n = sfronts/SF						    # Normalised shock front positions relative to mean
    SF_std_n  = np.std(sfronts_n)
    if SF_std_n < 0.001 and particle_bed == 0:
        # The shock front is in a non-meso-scopic-porous medium (i.e. continous)
        # The same condition is not applicable in reverse
        print "At time {:3.2f} $\mu$s the Shock Front is in a continuous object".format(step.time*1.e6)
        particle_bed = 0
        PBD          = 0
    else:
        particle_bed = 1
        PBD          = 1
    								    # Shock front in the particle bed
    QUP = 0.
    NN  = 0.                                            # No. cells used to calculate QUP, initialise as 1 in case NN = 0
    VY  = np.ma.masked_invalid(VY)
    if particle_bed == 1:
        for i in range(0,imax,1):
            j          = jfronts[i]					    # The jth position of the shock front for this column of 'i'
            posback[i] = sfronts[i]					    # And the starting point for this algorithm
            den_parts  = Den[i,j]					    # The density at the shock position
            up_grad    = np.gradient(VY[i,j:])
            while posback[i] == sfronts[i]: 
    	        j     = min(j,jmax)					    # Failsafe in case algorithm reaches the end of the array
    	        den_1 = Den[i,j:(j+20)]
    	        den_1 = den_1.astype(int)
    	        A = np.size(den_1[den_1==2]) 
    	        B = np.size(den_1)	
    	        if A==B or YC[i,j] >= 0.: 
    	            posback[i]  = YC[i,j]
    	            jback[i]    = j
    	        else:
    	            pass
                #if up_grad[j] <= up_grad[jfronts[i]]*.1: QUP = np.ma.mean(VY[i,j:j+10]) 
                j += 1
            QUP += np.sum(VY[i,jfronts[i]:jback[i]])
            NN  += np.size(VY[i,jfronts[i]:jback[i]])
    
        #QUP = np.mean(VY[(YC > np.amin(sfronts))*(YC < np.amax(posback))])  # Quasi UP. Measured as average behind slowest SF and fastest bit of driver.
    
    else:
       NN = 1.
    if NN == 0.: NN = 1. 
    print QUP,NN,QUP/NN,PBD#,posback
    QUP /= NN
    return QUP#,PBD,posback,jback

def hugoniot_point(QUP,YSF,TME):
    global t1,t2
    t1 = .0 
    t2 = .8 
    time = TME[(YSF<0.)*(YSF>-1.)*(TME<t2)]					    # Only use times and particle velocities for results WITHIN the particle bed
    up   = QUP[(YSF<0.)*(YSF>-1.)*(TME<t2)]					    # .8mm into the particle bed is given, for the shock to stabilise
    sf   = YSF[(YSF<0.)*(YSF>-1.)*(TME<t2)]					    # Shock front positions
    
    #UP   = outlier_mean(up)
    UP = np.mean(up[1:])
    UPSD = np.std(up[1:])
    US, intercept, other, things, too = stats.linregress(time[1:],sf[1:])
    US*=-1.
    UP*=-1.
    """
    plt.figure()
    plt.plot(time,-1*sf,linestyle=' ',marker='o')
    plt.plot(TME,-1*YSF,linestyle=' ',marker='*')
    plt.plot(time,time*US-intercept,linestyle='-',marker='o')
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(time, up)
    plt.title('Up - Time')
    plt.subplot(2,1,2)
    plt.plot(time,-1.*sf)
    plt.title('Shock Position - time')
    plt.show()
    """
    return UP,US*1000,-intercept,UPSD

def hugoniot_point_ND(QUP,YSF,TME,N):
    global t1,t2
    t1 = .0 
    t2 = 0.8 
    D  = -3.2e-2								    # in mm and -ve as up is +ve
    time = TME[(YSF<(N*D))*(YSF>-1.)*(TME<t2)]					    # Only use times and particle velocities for results WITHIN the particle bed
    up   = QUP[(YSF<(N*D))*(YSF>-1.)*(TME<t2)]					    # .8mm into the particle bed is given, for the shock to stabilise
    sf   = YSF[(YSF<(N*D))*(YSF>-1.)*(TME<t2)]					    # Shock front positions
    
    #UP   = outlier_mean(up)
    UP   = np.mean(up[1:])
    UPSD = np.std(up[1:])
    US, intercept, other, things, too = stats.linregress(time[1:],sf[1:])
    US*=-1.
    UP*=-1.
    """
    plt.figure()
    plt.plot(time,-1*sf,linestyle=' ',marker='o')
    plt.plot(TME,-1*YSF,linestyle=' ',marker='*')
    plt.plot(time,time*US-intercept,linestyle='-',marker='o')
    """
    return UP,US*1000,-intercept,UPSD
    
dir_0  = './Same_A_ESRFmethod'								     # Directory name for the output files
dirs   = ['A-1.272','A-1.275','A-1.2770','A-1.2725']
psp.mkdir_p(dir_0)
psp.mkdir_p(dir_0+'/cmaps')

N = 5

UP     = np.zeros((N))
US     = np.zeros((N))
UPSD_0 = np.zeros((N))

UP_2   = np.zeros((N))
US_2   = np.zeros((N))
UPSD_2 = np.zeros((N))

UP_4   = np.zeros((N))
US_4   = np.zeros((N))
UPSD_4 = np.zeros((N))

UP_6   = np.zeros((N))
US_6   = np.zeros((N))
UPSD_6 = np.zeros((N))

III =0

field1 = 'Pre'                                                       
field2 = 'V_y'
field3 = 'Den'
dirname= 'Hugoniots'
                                                      
tstart = 1									    # Start at t>0 such that there IS a shock present
tfin   = 56 
tintv  = 1



for directory in dirs:
    modelv150 = psp.opendatfile('{}/v150/jdata.dat'.format(directory))					     # Open datafile 
    modelv200 = psp.opendatfile('{}/v200/jdata.dat'.format(directory))					     # Open datafile 
    modelv250 = psp.opendatfile('{}/v250/jdata.dat'.format(directory))					     # Open datafile 
    modelv300 = psp.opendatfile('{}/v300/jdata.dat'.format(directory))					     # Open datafile 
    modelv350 = psp.opendatfile('{}/v350/jdata.dat'.format(directory))					     # Open datafile 
    models = [modelv150,modelv200,modelv250,modelv300,modelv350]
    modelS = (['v150','v200','v250','v300','v350'])
    VI = ([150,200,250,300,350])
    [model.setScale('mm') for model in models]
    III =0
    hh = 0
    for model in models:
        sf = 0.										    # Initialise the shock front position
        psp.mkdir_p('./{}/{}/{}'.format(dir_0,directory,modelS[III]))
        print "########################################"
        print "current model = {}".format(modelS[III])
        print "########################################"
        YC     = model.yc
        XC     = model.xc
        TME    = np.zeros(( int((tfin-tstart+1)/tintv) ))				    # time 
        YSF    = np.zeros(( int((tfin-tstart+1)/tintv) ))				    # Shock Front Y position
        PBD    = np.zeros(( int((tfin-tstart+1)/tintv) ))				    # Shock in Particle Bed array. 1 is yes 0 is no
        QUP    = np.zeros(( int((tfin-tstart+1)/tintv) ))				    # Quasi steady state Particle velocity
        PRE    = np.zeros(( int((tfin-tstart+1)/tintv) ))				    # Quasi steady state Particle velocity
    
        Us     = np.zeros(( int((tfin-tstart+1)/(5*tintv))+1 ))
        Up     = np.zeros(( int((tfin-tstart+1)/(5*tintv))+1 ))				    # Shock and particle velocities calculated every 5 timesteps
        Time   = np.zeros(( int((tfin-tstart+1)/(5*tintv))+1 ))				    # Equivalent time every 5 timesteps (=.1 us)
    
        dy     = abs(YC[0,1] - YC[0,0])
        dx     = abs(XC[1,0] - XC[0,0])
        t = 1
        tfin2 = model.nsteps
        counter = 0
        AMAX = 0. 
        while sf < 1.0 and t < tfin:
    
            particle_bed = 0
    
            step = model.readStep(['{}'.format(field1),'{}'.format(field2),'{}'.format(field3)],t)
            Den  = np.ma.filled(step.data[2],0.)*1.e-3*step.cmc[0]
            """
            plt.figure()
            plt.imshow(step.data[0],cmap='viridis')
            plt.show()
            """
            Pre  = np.ma.filled(step.data[0],0.)*1.e-9
            VY   = step.data[1]
    
            imax,jmax = np.shape(Den)
            sfronts   = np.zeros((imax))
            jfronts   = np.zeros((imax),dtype=int)
            posback   = np.zeros((imax))					     # position of back of bed
            jback     = np.zeros((imax),dtype=int)					     # j coord of back of bed
            AMAX      = VI[III]/3.
            sfronts, jfronts, SF = find_shock_front_2(imax,jmax,sfronts,jfronts,VY,YC,AMAX)	    
    
            print 'Time: {:3.2f} us'.format(step.time*1.e6)
            print 'Average Shock Front position = {:3.2f} mm'.format(SF)
            print 'Max SF position: {:3.2f} mm, Min SF position: {:3.2f} mm \n'.format(np.amax(sfronts),np.amin(sfronts))
            
            YSF[(t-1)/tintv]  = np.mean(sfronts)
            TME[(t-1)/tintv]  = step.time*1.e6
            sf            = np.amin(sfronts)*-1.
            
            #QUP[(t-1)/tintv],posback,jback = calc_postshock_PBD_state(SF,particle_bed,YC,imax,jfronts,sfronts,posback,Den,VY) 
            QUP[(t-1)/tintv] = calc_postshock_PBD_state(SF,particle_bed,YC,imax,jfronts,sfronts,posback,Den,VY) 
            #posback = np.maximum(sfronts,posback)	# Failsafe. The back of the bed should ALWAYS be behind the Shock Front
            #AMAX = QUP[(t-1)/tintv]*.9
            """
            if counter%5 == 0 and counter>0:
        	slope, intercept, other, things, too = stats.linregress(TME[t-4:t],YSF[t-4:t])
        	Us[counter/5] = abs(slope*1000.)
        	Time[counter/5] = TME[t]
            counter += 1
            """
            #plot_var_colormaps(step.data[1],model,step,sfronts,posback,t,'Up',dir_0,directory,modelS[III])
            
            t += tintv
    
        TME = TME[(YSF!=0.)]
        QUP = QUP[(YSF!=0.)]
        YSF = YSF[(YSF!=0.)]
        UP[III],US[III],C_0,UPSD_0[III]     = hugoniot_point(QUP[:-1],YSF[:-1],TME[:-1])
        UP_2[III],US_2[III],C_2,UPSD_2[III] = hugoniot_point_ND(QUP[:-1],YSF[:-1],TME[:-1],2.)
        UP_4[III],US_4[III],C_4,UPSD_4[III] = hugoniot_point_ND(QUP[:-1],YSF[:-1],TME[:-1],4.)
        UP_6[III],US_6[III],C_6,UPSD_6[III] = hugoniot_point_ND(QUP[:-1],YSF[:-1],TME[:-1],6.)
        #plt.close()
        """
        plt.figure()
        plt.plot(Time, Us, linestyle=' ',marker = 'o', color = 'r')
        plt.ylabel('Shock Velocity [ms$^{-1}$]')
        plt.xlabel('Time [$\mu$s]')
        plt.savefig('./{}/{}_{}_Us-Time.png'.format(dir_0,directory,modelS[III]),dpi=300)
        plt.close()
        """
    
        plt.figure(figsize=(3.5,3.5))
        plt.plot(TME[1:],QUP[1:]*-1.,linestyle=' ',marker = '+',mew=1.5, color = 'b')
        plt.xlabel('Time [$\mu$s]')
        plt.ylabel('Particle Velocity [ms$^{-1}$]')
        plt.savefig('./{}/{}_{}_Up-Time.png'.format(dir_0,directory,modelS[III]),dpi=500,bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(3.5,3.5))
        plt.plot(TME[1:],abs(YSF[1:]),linestyle=' ',marker = 'x',mew=1.5,color = 'm')
        plt.plot(TME[1:],1.e-3*US[III]*TME[1:]+C_0,linestyle='-',color='k',label='0D')
        plt.plot(TME[1:],1.e-3*US_2[III]*TME[1:]+C_2,linestyle=':',color='k',label='2D')
        plt.plot(TME[1:],1.e-3*US_4[III]*TME[1:]+C_4,linestyle='-.',color='k',label='4D')
        plt.plot(TME[1:],1.e-3*US_6[III]*TME[1:]+C_6,linestyle='--',color='k',label='6D')
        plt.ylabel('Shock Position [mm]')
        plt.xlabel('Time [$\mu$s]')
        plt.legend(loc='best',fontsize='small')
        plt.savefig('./{}/{}_{}_YSF-TME.png'.format(dir_0,directory,modelS[III]),dpi=500,bbox_inches='tight')
        
        plt.figure(figsize=(3.5,3.5))
        plt.plot(YSF[1:]*-1.,QUP[1:]*-1.,linestyle=' ',marker = 'o', color = 'b')
        plt.axhline(y=1.*UP[III],linestyle='-',color='k',label='0D')
        plt.axhline(y=1.*UP_2[III],linestyle=':',color='k',label='2D')
        plt.axhline(y=1.*UP_4[III],linestyle='-.',color='k',label='4D')
        plt.axhline(y=1.*UP_6[III],linestyle='--',color='k',label='6D')
    
        plt.axvline(x=2.*3.2e-2,linestyle=':', color = 'k')
        plt.axvline(x=4.*3.2e-2,linestyle='-.', color = 'k')
        plt.axvline(x=6.*3.2e-2,linestyle='--', color = 'k')
        plt.title('Up - Shock Position')
        plt.xlabel('Shock Position [mm]')
        plt.ylabel('Particle Velocity [ms$^{-1}$]')
        plt.legend(loc='best',fontsize='small')
        plt.savefig('./{}/{}_{}_QUP-TME.png'.format(dir_0,directory,modelS[III]),dpi=500,bbox_inches='tight')
        #plt.show()
        plt.close()
    
        III += 1
    
    g = [2.,50.,1.5,400.]
    #up1,us1,up2,us2 = twoline_piecewise_fit(UP,US,g)
    np.savetxt('./{}/{}/Up-Us_{}_original.csv'.format(dir_0,directory,directory),np.column_stack((UP,US,UP_2,US_2,UP_4,US_4,UP_6,US_6,UPSD_0,UPSD_2,UPSD_4,UPSD_6)),delimiter=',')
    plt.figure(figsize=(3.5,3.5))
    plt.plot(UP,US,linestyle=' ',marker='o',mew=1,markerfacecolor='None',label='all touching',ms=5)
    #plt.plot(up1,us1,up2,us2,color='k')
    plt.xlabel('Particle Velocity [ms$^{-1}$]')
    plt.ylabel('Shock Velocity [ms$^{-1}$]')
    plt.savefig('./{}/{}/hugoniot_us-up.png'.format(dir_0,directory),dpi=500,bbox_inches='tight')




