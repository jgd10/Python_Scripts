import numpy as np
import scipy as sc
import scipy.stats as stats
import pySALEPlot as psp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    #SF = np.mean(sfronts)						    # Shock Front position, as a mean of all the shock fronts in j
    SF = np.median(sfronts)						    # Shock Front position, as a median of all the shock fronts in j

    return sfronts, jfronts, SF

def find_shock_front_2(imax,jmax,sfronts,jfronts,VY,YC,A_lim):
    jj = np.median(jfronts)
    for i in range(0,imax,1):
        j = jj - 100
        j = max(j,0)
        Shock = False
        while Shock == False and YC[i,j] < (0.0 - dy): 
            j = min(j,jmax-3)							    # Prevent loop exceeding end of array
            if YC[i,j] > (0.0-dy): J = j
            if -1.*VY[i,j] > A_lim: 
                sfronts[i] = min(YC[i,j+1],0.)
                Shock = True
                if sfronts[i] == 0.: 
                    jfronts[i] = J
                else:
                    jfronts[i] = j+1
                break
            else:
                pass
            j += 1
    SF = np.median(sfronts)						    # Shock Front position, as a mean of all the shock fronts in j
    
    return sfronts, jfronts, SF

def find_shock_front_3(imax,jmax,sfronts,jfronts,A,YC,A_lim):
    jj = np.median(jfronts)
    for i in range(0,imax,1):
        j = 0#jj - 300
        j = max(j,0)
        Shock = False
        while Shock == False and YC[i,j] < (0.0 - dy): 
            j = min(j,jmax-3)							    # Prevent loop exceeding end of array
            if YC[i,j] > (0.0-dy): J = j
            if A[i,j] > A_lim: 
                sfronts[i] = min(YC[i,j+1],0.)
                Shock = True
                if sfronts[i] == 0.: 
                    jfronts[i] = J
                else:
                    jfronts[i] = j+1
                break
            else:
                pass
            j += 1
    SF = np.median(sfronts)						    # Shock Front position, as a mean of all the shock fronts in j
    
    return sfronts, jfronts, SF
    
def plot_var_colormaps(A,model,step,sfronts,posback,t,field,dirname,datadir,modeldir):
    fig = plt.figure()
    A  /= np.mean(A)
    ASD = np.std(A)
    ax  = fig.add_subplot(111,aspect='equal')
    var = ax.pcolormesh(model.x,model.y,A,cmap='plasma',vmin=0.,vmax=1.)
    ax.set_xlim(0.,1.)
    ax.set_ylim(-1.,0.)
    ax.set_xlabel('Transverse Position [mm]')
    ax.set_ylabel('Longitudinal Position [mm]')
    ax.set_title('t = {:3.2f} $\mu$s'.format(step.time*1.e6))
    N = np.size(sfronts)
    X = np.arange(0.,1.0,1./N)
    MSF = np.ma.median(A)
    ax.axhline(y=MSF,linestyle='--',color='powderblue',label = 'Median Shock Front',lw=2.)
    ax.plot(X,sfronts,label='Shock Front',color='deepskyblue',lw=1.5)
    ax.plot(X,posback,label='Impactor Front',color='brown',lw=2.)
    ax.legend(fontsize='small',loc='lower right',framealpha=0.5)
    cb=fig.colorbar(var)
    cb.set_label('Velocity/[Mean vel over mesh](={:3.2f}km/s)]'.format(MSF),fontsize=10)
    fig.savefig('./{}/cmaps/{}_{}_{}_shockfront{:05d}.png'.format(dirname,datadir,modeldir,field,t),dpi=400)
    #plt.show()
    plt.close()

def calc_postshock_PBD_state2(SF,particle_bed,YC,imax,jfronts,sfronts,posback,Den,VY,impactor):
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
            #up_grad    = np.gradient(VY[i,j:])
            while posback[i] == sfronts[i]: 
    	        j     = min(j,jmax)					    # Failsafe in case algorithm reaches the end of the array
                #den_1 = Den[i,j:(j+20)]
                #den_1 = den_1.astype(int)
                #A = np.size(den_1[den_1==1]) 
                #B = np.size(den_1)	
    	        if impactor[i,j]>0. or YC[i,j] >= 0.: 
    	            posback[i]  = YC[i,j]
    	            jback[i]    = j
    	        else:
    	            pass
                #if up_grad[j] <= up_grad[jfronts[i]]*.1: QUP = np.ma.mean(VY[i,j:j+10]) 
                j += 1
            if posback[i]!=sfronts[i]:
                QUP += np.ma.sum(VY[i,jfronts[i]:jback[i]])
                NN  += np.ma.MaskedArray.count(VY[i,jfronts[i]:jback[i]])
            else:
                QUP += 0
                NN  += 0
    
        #QUP = np.mean(VY[(YC > np.amin(sfronts))*(YC < np.amax(posback))])  # Quasi UP. Measured as average behind slowest SF and fastest bit of driver.
    
    else:
       NN = 1.
    if NN == 0.: NN = 1. 
    #print QUP,NN,QUP/NN
    QUP /= NN
    return QUP,jback

def calc_postshock_PBD_state(SF,particle_bed,YC,imax,jfronts,sfronts,posback,Den,VY,F):
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
    QF  = 0.
    NN  = 0. # No. cells used to calculate QUP, initialise as 1 in case NN = 0
    FF = 0.
    #VY  = np.ma.masked_invalid(VY)
    #mesh = np.zeros_like(VY)
    if particle_bed == 1:
        for i in range(0,imax,1):
            j          = jfronts[i]					    # The jth position of the shock front for this column of 'i'
            posback[i] = sfronts[i]					    # And the starting point for this algorithm
            den_parts  = Den[i,j]					    # The density at the shock position
            up_grad    = np.gradient(VY[i,j:])
            while posback[i] == sfronts[i]: 
                #j     = min(j,jmax)					    # Failsafe in case algorithm reaches the end of the array
                #den_1 = Den[i,j:(j+20)]
                #den_1 = den_1.astype(int)
                #A = np.size(den_1[den_1==2]) 
                #B = np.size(den_1)	
                #if A==B or YC[i,j] >= 0.: 
                if A==B or YC[i,j] >= 0.: 
                    posback[i]  = YC[i,j]
                    jback[i]    = j
    	        else:
    	            pass
                #if up_grad[j] <= up_grad[jfronts[i]]*.1: QUP = np.ma.mean(VY[i,j:j+10]) 
                j += 1
            ss   = np.ma.MaskedArray.count(VY[i,jfronts[i]:jback[i]]) 
            FP   = F[i,jfronts[i]:jback[i]]
            #ff   = np.ma.MaskedArray.count(F[i,jfronts[i]:jback[i]]) 
            ff   = np.size(FP[FP>0.])
            if ss == 0:
                QUP += 0
            else:
                QUP += np.ma.sum(VY[i,jfronts[i]:jback[i]])
            if ff == 0:
                QF += 0
            else:
                QF += np.sum(FP[FP>0.])
            #mesh[i,jfronts[i]:jback[i]] += VY[i,jfronts[i]:jback[i]]
            #mesh[mesh!=0.] = 1.0
            #mesh[i,jfronts[i]] += 1.0
            #mesh[i,jback[i]] += 1.0
            NN  += np.ma.MaskedArray.count(VY[i,jfronts[i]:jback[i]])
            FF  += ff
    
        #QUP = np.mean(VY[(YC > np.amin(sfronts))*(YC < np.amax(posback))])  # Quasi UP. Measured as average behind slowest SF and fastest bit of driver.
    
    else:
       NN = 1.
       FF = 1.
    if NN == 0.: NN = 1. 
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(mesh,interpolation='nearest',cmap = 'inferno')
    plt.show()
    """
    #print QUP,NN,QUP/NN#,posback
    QUP /= NN
    QF  /= FF
    #print QF
    return QUP,QF#,PBD,posback,jback

def hugoniot_point(QUP,YSF,TME):
    global t1,t2
    t1 = .0 
    QUP = np.ma.masked_invalid(QUP)
    time = TME[(YSF<0.)*(YSF>-.85)]					    # Only use times and particle velocities for results WITHIN the particle bed
    up   = QUP[(YSF<0.)*(YSF>-.85)]					    # .8mm into the particle bed is given, for the shock to stabilise
    sf   = YSF[(YSF<0.)*(YSF>-.85)]					    # Shock front positions

    
    #UP   = outlier_mean(up)
    UP   = np.ma.mean(up[1:])
    UPSD = np.ma.std(up[1:])
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
    QUP = np.ma.masked_invalid(QUP)
    time = TME[(YSF<(N*D))*(YSF>-.85)]					    # Only use times and particle velocities for results WITHIN the particle bed
    up   = QUP[(YSF<(N*D))*(YSF>-.85)]					    # .8mm into the particle bed is given, for the shock to stabilise
    sf   = YSF[(YSF<(N*D))*(YSF>-.85)]					    # Shock front positions
    
    #UP   = outlier_mean(up)
    UP   = np.ma.mean(up[1:])
    UPSD = np.ma.std(up[1:])
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

def hugoniot_point_general_ND(AAA,YSF,TME,N):
    global t1,t2
    t1 = .0 
    t2 = 0.8 
    D  = -3.2e-2								    # in mm and -ve as up is +ve
    AAA= np.ma.masked_invalid(AAA)
    time = TME[(YSF<(N*D))*(YSF>-.85)]					    # Only use times and particle velocities for results WITHIN the particle bed
    a    = AAA[(YSF<(N*D))*(YSF>-.85)]					    # .8mm into the particle bed is given, for the shock to stabilise
    sf   = YSF[(YSF<(N*D))*(YSF>-.85)]					    # Shock front positions
    #UP   = outlier_mean(up)
    A   = np.ma.mean(a[1:])
    ASD = np.ma.std(a[1:])
    return A,ASD
    
dir_0  = './hugoniot_data'								     # Directory name for the output files
dirs   = ['A-1.5203','A-0.8520','A-0.7226','A-1.1201','A-0.4051','A-1.2878']
#dirs = dirs[::-1]
psp.mkdir_p(dir_0)
psp.mkdir_p(dir_0+'/cmaps')

N = 11

UP     = np.zeros((N))
US     = np.zeros((N))
UPSD_0 = np.zeros((N))

UP_2   = np.zeros((N))
US_2   = np.zeros((N))
UPSD_2 = np.zeros((N))

SGY_   = np.zeros((N))
SGYSD  = np.zeros((N))
TPS_   = np.zeros((N))
TPSSD  = np.zeros((N))
PRE_   = np.zeros((N))
PRESD  = np.zeros((N))
DEN_   = np.zeros((N))
DENSD  = np.zeros((N))
ALP_   = np.zeros((N))
ALPSD  = np.zeros((N))
DENs_  = np.zeros((N))
DENsSD = np.zeros((N))

DEN0   = np.zeros((N))
ALP0   = np.zeros((N))

III =0

field1 = 'Pre'                                                       
field2 = 'V_y'
field3 = 'Den'
field4 = 'TPS'
field5 = 'Syy'
field6 = 'Tmp'
dirname= 'Hugoniots'
                                                      
tstart = 1									    # Start at t>0 such that there IS a shock present
tfin   = 201 
tintv  = 1



for directory in dirs:
    #modelv150 = psp.opendatfile('{}/v150/jdata.dat'.format(directory))					     # Open datafile 
    #modelv200 = psp.opendatfile('{}/v200/jdata.dat'.format(directory))					     # Open datafile 
    #modelv250 = psp.opendatfile('{}/v250/jdata.dat'.format(directory))					     # Open datafile 
    #modelv300 = psp.opendatfile('{}/v300/jdata.dat'.format(directory))					     # Open datafile 
    modelv350 = psp.opendatfile('{}/v350/jdata.dat'.format(directory))					     # Open datafile 
    #modelv450 = psp.opendatfile('{}/v450/jdata.dat'.format(directory))					     # Open datafile 
    #modelv550 = psp.opendatfile('{}/v550/jdata.dat'.format(directory))					     # Open datafile 
    modelv650 = psp.opendatfile('{}/v650/jdata.dat'.format(directory))					     # Open datafile 
    #modelv750 = psp.opendatfile('{}/v750/jdata.dat'.format(directory))					     # Open datafile 
    #modelv850 = psp.opendatfile('{}/v850/jdata.dat'.format(directory))					     # Open datafile 
    modelv950 = psp.opendatfile('{}/v950/jdata.dat'.format(directory))					     # Open datafile 
    models = [modelv350,modelv650,modelv950]#[modelv150,modelv200,modelv250,modelv300,modelv350,modelv450,modelv550,modelv650,modelv750,modelv850,modelv950]
    modelS = (['v350','v650','v950'])#(['v150','v200','v250','v300','v350','v450','v550','v650','v750','v850','v950'])
    VI = (['350','650','950'])#([150,200,250,300,350,450,550,650,750,850,950])
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
        nnn    = int((tfin-tstart+1)/tintv)
        TME    = np.zeros(( nnn ))				    # time 
        YSF    = np.zeros(( nnn ))				    # Shock Front Y position
        PBD    = np.zeros(( nnn ))				    # Shock in Particle Bed array. 1 is yes 0 is no
        QUP    = np.zeros(( nnn ))				    # Quasi steady state Particle velocity
        SGY    = np.zeros(( nnn ))				    # Quasi steady state Longitudinal Stress
        DEN    = np.zeros(( nnn ))				    # Quasi steady state Bulk Density
        DENs   = np.zeros(( nnn ))				    # Quasi steady state Solid Density
        ALP    = np.zeros(( nnn ))				    # Quasi steady state Distension
        PRE    = np.zeros(( nnn ))				    # Quasi steady state Pressure
        TPS    = np.zeros(( nnn ))				    # Quasi steady state Total Plastic Strain
        
        QUPstd    = np.zeros(( nnn ))				
        SGYstd    = np.zeros(( nnn ))			
        PREstd    = np.zeros(( nnn ))				
    
        dy     = abs(YC[0,1] - YC[0,0])
        dx     = abs(XC[1,0] - XC[0,0])
        t = 1
        tfin2 = model.nsteps
        counter = 0
        step      = model.readStep(['{}'.format(field3)],0)
        conc      = step.cmc[0]
        Den       = np.ma.filled(step.data[0],0.)*1.e-3*conc
        poros     = 1. - np.mean(conc[(YC<=0)*(YC>=-1.)])
        print 'vfrac = {} %'.format(1. - poros)
        ALP0[III] = 1./(1.-poros)
        DEN0[III] = np.mean(Den[(YC<=0.)*(YC>=-1.)]) 
        print DEN0[III],ALP0[III],poros
        imax,jmax = np.shape(Den)
        sfronts   = np.zeros((imax))
        jfronts   = np.zeros((imax),dtype=int)
        posback   = np.zeros((imax))					     # position of back of bed
        jback     = np.zeros((imax),dtype=int)					     # j coord of back of bed
        AMAX = 0.
        while np.around(sf,decimals=2) < 1. and t < tfin:
    
            particle_bed = 0
    
            step = model.readStep(['{}'.format(field1),'{}'.format(field2),'{}'.format(field3),'{}'.format(field4),'{}'.format(field5),'Tmp'],t)
            Den  = np.ma.filled(step.data[2],0.)*1.e-3*step.cmc[0]
            Pre  = np.ma.filled(step.data[0],0.)*1.e-9
            Syy  = np.ma.filled(step.data[4],0.)*1.e-9
            tps  = np.ma.filled(step.data[3],-1.)
            Tmp  = step.data[5]
            Sgy  = (Syy - Pre)*-1.
            VY   = step.data[1]
    
            #AMAX      = VI[III]/3.
            sfronts, jfronts, SF = find_shock_front_3(imax,jmax,sfronts,jfronts,Pre,YC,AMAX/2.)	    
    
            print 'Time: {:3.2f} us'.format(step.time*1.e6)
            print 'Shock Front position = {:3.2f} mm'.format(SF)
            print 'Max SF position: {:3.2f} mm, Min SF position: {:3.2f} mm \n'.format(np.amax(sfronts),np.amin(sfronts))
            
            YSF[(t-1)/tintv]  = SF #np.mean(sfronts)
            TME[(t-1)/tintv]  = step.time*1.e6
            sf                = abs(np.amin(sfronts))#-SF #
            
            #QUP[(t-1)/tintv],posback,jback = calc_postshock_PBD_state(SF,particle_bed,YC,imax,jfronts,sfronts,posback,Den,VY) 
            QUP[(t-1)/tintv],jback = calc_postshock_PBD_state2(SF,particle_bed,YC,imax,jfronts,sfronts,posback,Den,VY,step.cmc[1]) 
            #posback = np.maximum(sfronts,posback)	# Failsafe. The back of the bed should ALWAYS be behind the Shock Front
            
            SPre = []
            SSgy = []
            SDen = []
            Sup  = []
            Stps = []
            Conc = []
            STmp = []
            fill = step.cmc[0]
            for i in range(imax):
                for j in range(jfronts[i],jback[i]+1,1):
                    #if np.bool(step.data[0][i,j]): 
                    SPre.append(Pre[i,j])
                    #if np.bool(step.data[0][i,j]): 
                    SSgy.append(Sgy[i,j])
                    SDen.append(Den[i,j])
                    if np.bool(VY[i,j]): Sup.append(abs(VY[i,j]))
                    if tps[i,j]!=-1.: Stps.append(tps[i,j])
                    if np.bool(Tmp[i,j]): STmp.append(Tmp[i,j])
                    Conc.append(fill[i,j])
            Sup  = np.array(Sup)
            STmp = np.array(STmp)
            Conc = np.array(Conc)
            SDen = np.array(SDen)
            QPre = np.mean(SPre)

            porosity  = 1. - np.mean(Conc)
            print porosity, 1./(1.-porosity)
            
            
            ALP[(t-1)/tintv] = 1./(1.-porosity)
            PRE[(t-1)/tintv] = np.mean(SPre)
            DEN[(t-1)/tintv] = np.mean(SDen)
            SGY[(t-1)/tintv] = np.mean(SSgy)
            TPS[(t-1)/tintv] = np.mean(Stps)
            DENs[(t-1)/tintv] = np.mean(SDen[Conc>0.])
            
            
            QUPstd[(t-1)/tintv] = np.std(SPre)
            PREstd[(t-1)/tintv] = np.std(SDen)
            SGYstd[(t-1)/tintv] = np.std(SSgy)


            if t%20 == 0 and t!=1:
                rho00 = .5*15.56
                derived_SF = -np.sqrt(SGY[(t-1)/tintv]/(rho00 - rho00**2./DEN[(t-1)/tintv]))*step.time*1.e6
                fig23 = plt.figure(figsize=(15,5))
                ax231 = fig23.add_subplot(131,aspect='equal')
                ax232 = fig23.add_subplot(132)
                ax233 = fig23.add_subplot(133)
                #ax234 = fig23.add_subplot(144)
                
                pp = ax231.pcolormesh(model.x,model.y,step.data[0]*1.e-9,cmap='viridis',vmin=0,vmax=QPre*3.)
    	        divider = make_axes_locatable(ax231)
                cax = divider.append_axes("right", size="5%", pad=0.05)
    	        cb = fig23.colorbar(pp,cax=cax)
                cb.set_label('Pressure [GPa]',fontsize=10)
                
                ax231.plot(model.xc,sfronts,marker=' ',lw=.5,color='pink',linestyle='-')
                ax231.axhline(y=SF,color='gray')
                ax231.set_xlabel('Transverse\nPosition [mm]')
                ax231.set_ylabel('Longitudinal\nPosition [mm]')
                ax231.set_xlim(0, 1)
                ax231.set_ylim(-1,0)
                
                n2, bins2, patches2 = ax232.hist(Sup,100,color=plt.cm.viridis(0))
                ax232.axvline(x=-QUP[(t-1)/tintv],linestyle='--')
                #ax232.set_xlim(0,5)
                ax232.set_xlabel('$u_p$ [ms$^{-1}$]')
                ax232.set_ylabel('Frequency')
                
                #n3, bins3, patches3 = ax233.hist(STmp,100,color=plt.cm.viridis(0))
                #ax233.set_xlim(0,2870+273+300)
                #ax233.axvline(x=2870+273,linestyle='-',color='orange')
                #ax233.set_xlabel('Temperature [K]')

                n4, bins4, patches4 = ax233.hist(SSgy,100,color=plt.cm.viridis(0))
                ax233.set_xlim(0,SGY[(t-1)/tintv]*3.)
                ax233.axvline(x=SGY[(t-1)/tintv],linestyle='--')
                ax233.set_xlabel('-$\sigma_{LL}$ [GPa]')
                #n3, bins3, patches3 = ax233.hist(sfronts,50,color=plt.cm.viridis(0.))
                #ax233.axvline(x=SF,linestyle='--',lw=.5)
                #ax233.set_xlabel('Shock Position [mm]')
                #ax233.set_ylabel('Frequency')
                
                #n4, bins4, patches4 = ax234.hist(Sup,50,color=plt.cm.viridis(0.5))
                #ax234.axvline(x=QUP[(t-1)/tintv],linestyle='--',color='k',lw=.5)

                #Pre  = step.data[0]*1.e-9
                #plt.plot(model.yc[0,:],np.mean(abs(VY),axis=0)/np.amax(np.mean(abs(VY),axis=0)),color='k',lw=2.)
                #plt.pcolormesh(model.yc[0,:],model.xc[:,0],abs(VY),cmap='plasma',alpha=0.4)
                #plt.plot(model.yc[0,:],np.mean(-VY,axis=0)/np.amax(np.mean(-VY,axis=0)),color='k',lw=2.)
                #plt.pcolormesh(model.yc[0,:],model.xc[:,0],-VY,cmap='viridis',alpha=0.4)
                #plt.axvline(x=SF,linestyle='--',color='k',lw=1.5)
                #plt.axvline(x=derived_SF,linestyle=':',color='k',lw=1.5)
                #plt.axhline(y=abs(QUP[(t-1)/tintv])/np.amax(np.mean(abs(VY),axis=0)),linestyle='--',color='k',lw=1.5)
                #plt.xlabel('Position [mm]')
                #plt.ylabel('Normalised Velocity [x {:3.1f} m/s] \n Position [mm]'.format(np.amax(np.mean(abs(VY),axis=0))))
                #plt.axis('equal')
                #plt.xlim(1,-1)
                #plt.ylim(0,1)
                #plt.savefig('./{}/cmap_lines_{}_{}_{}_cmap_{:05d}.png'.format(dir_0,directory,modelS[III],field2,t),dpi=500,bbox_inches='tight')
                fig23.tight_layout()
                fig23.savefig('./histograms/{}{}_paper_hist_figure_{:05d}.png'.format(directory,modelS[III],t),dpi=500,bbox_inches='tight')
                plt.close()
            AMAX = QPre
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
        PRE = PRE[(YSF!=0.)]
        SGY = SGY[(YSF!=0.)]
        TPS = TPS[(YSF!=0.)]
        DEN = DEN[(YSF!=0.)]
        ALP = ALP[(YSF!=0.)]
        
        PREstd = PREstd[(YSF!=0.)]
        SGYstd = SGYstd[(YSF!=0.)]
        QUPstd = QUPstd[(YSF!=0.)]

        UP[III],US[III],C_0,UPSD_0[III]     = hugoniot_point(QUP[:-1],YSF[:-1],TME[:-1])
        UP_2[III],US_2[III],C_2,UPSD_2[III] = hugoniot_point_ND(QUP[:-1],YSF[:-1],TME[:-1],2.)
        
        SGY_[III],SGYSD[III] = hugoniot_point_general_ND(SGY[:-1],YSF[:-1],TME[:-1],2.)
        DEN_[III],DENSD[III] = hugoniot_point_general_ND(DEN[:-1],YSF[:-1],TME[:-1],2.)
        TPS_[III],TPSSD[III] = hugoniot_point_general_ND(TPS[:-1],YSF[:-1],TME[:-1],2.)
        PRE_[III],PRESD[III] = hugoniot_point_general_ND(PRE[:-1],YSF[:-1],TME[:-1],2.)
        ALP_[III],ALPSD[III] = hugoniot_point_general_ND(ALP[:-1],YSF[:-1],TME[:-1],2.)
        ###############################################################################
        ###############################################################################
        # ignore early weirdness in distension by putting N = 6
        ###############################################################################
        ###############################################################################
        
        plt.figure(figsize=(3.5,3.5))
        plt.plot(TME,QUPstd,linestyle=' ',marker = '+',mew=1.5, color = 'b')
        #plt.axhline(y=ALP_[III],linestyle='-',color='k',label='2D')
        plt.ylabel('Standard Deviation on $u_p$ [ms$^{-1}$]')
        plt.xlabel('Time [$\mu$s]')
        #plt.legend(loc='best',fontsize='small')
        plt.savefig('./histograms/{}_{}_QUPstd-Time.png'.format(directory,modelS[III]),dpi=500,bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(3.5,3.5))
        plt.plot(TME,PREstd,linestyle=' ',marker = '+',mew=1.5, color = 'b')
        #plt.axhline(y=ALP_[III],linestyle='-',color='k',label='2D')
        plt.ylabel('Standard Deviation on\nPressure [GPa]')
        plt.xlabel('Time [$\mu$s]')
        #plt.legend(loc='best',fontsize='small')
        plt.savefig('./histograms/{}_{}_PREstd-Time.png'.format(directory,modelS[III]),dpi=500,bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(3.5,3.5))
        plt.plot(TME,SGYstd,linestyle=' ',marker = '+',mew=1.5, color = 'b')
        #plt.axhline(y=ALP_[III],linestyle='-',color='k',label='2D')
        plt.ylabel('Standard Deviation on $-\sigma_{LL}$ [GPa]')
        plt.xlabel('Time [$\mu$s]')
        #plt.legend(loc='best',fontsize='small')
        plt.savefig('./histograms/{}_{}_SGYstd-Time.png'.format(directory,modelS[III]),dpi=500,bbox_inches='tight')
        plt.close()
         
        """
        DENs_[III],DENsSD[III] = hugoniot_point_general_ND(DENs[:-1],YSF[:-1],TME[:-1],2.)
        print "0D up = {}, Us = {}; 2D up = {}, Us = {}".format(UP[III],US[III],UP_2[III],US_2[III])
        #plt.close()
        plt.figure()
        plt.plot(TME, DEN, linestyle=' ',marker = 'h', color = 'g')
        plt.axhline(y=(6.*3.2e-2),linestyle='-.',color='k',label='6 Diameters')
        plt.axhline(y=1.*DEN_[III],linestyle='-',color='k',label='6D average')
        plt.ylabel('Post-Shock Density [gcm$^{-3}$]')
        plt.xlabel('Time [$\mu$s]')
        plt.savefig('./{}/{}_{}_Den-Time.png'.format(dir_0,directory,modelS[III]),dpi=300)
        plt.close()
    
        plt.figure(figsize=(3.5,3.5))
        plt.plot(TME[1:],QUP[1:]*-1.,linestyle=' ',marker = '+',mew=1.5, color = 'b')
        plt.xlabel('Time [$\mu$s]')
        plt.ylabel('Particle Velocity [ms$^{-1}$]')
        plt.savefig('./{}/{}_{}_Up-Time.png'.format(dir_0,directory,modelS[III]),dpi=500,bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(3.5,3.5))
        plt.plot(TME[1:],abs(YSF[1:]),linestyle=' ',marker = 'x',mew=1.5,color = 'm')
        plt.plot(TME[1:],1.e-3*US[III]*TME[1:]+C_0,linestyle='-',color='k',label='0D')
        plt.plot(TME[1:],1.e-3*US_2[III]*TME[1:]+C_2,linestyle=':',color='k',label='6D')
        #plt.plot(TME[1:],1.e-3*US_4[III]*TME[1:]+C_4,linestyle='-.',color='k',label='4D')
        #plt.plot(TME[1:],1.e-3*US_6[III]*TME[1:]+C_6,linestyle='--',color='k',label='6D')
        plt.axhline(y=(6.*3.2e-2),linestyle='-.',color='k',label='6D')
        plt.ylabel('Shock Position [mm]')
        plt.xlabel('Time [$\mu$s]')
        plt.legend(loc='best',fontsize='small')
        plt.savefig('./{}/{}/{}_{}_YSF-TME.png'.format(dir_0,directory,directory,modelS[III]),dpi=500,bbox_inches='tight')
        
        plt.figure(figsize=(3.5,3.5))
        plt.plot(YSF[1:]*-1.,QUP[1:]*-1.,linestyle=' ',marker = 'o', color = 'b')
        plt.axhline(y=1.*UP[III],linestyle='-',color='k',label='0D')
        plt.axhline(y=1.*UP_2[III],linestyle=':',color='k',label='6D')
        #plt.axhline(y=1.*UP_4[III],linestyle='-.',color='k',label='4D')
        #plt.axhline(y=1.*UP_6[III],linestyle='--',color='k',label='6D')
    
        plt.axvline(x=6.*3.2e-2,linestyle=':', color = 'k')
        #plt.axvline(x=4.*3.2e-2,linestyle='-.', color = 'k')
        #plt.axvline(x=6.*3.2e-2,linestyle='--', color = 'k')
        plt.title('Up - Shock Position')
        plt.xlabel('Shock Position [mm]')
        plt.ylabel('Particle Velocity [ms$^{-1}$]')
        plt.legend(loc='best',fontsize='small')
        plt.savefig('./{}/{}/{}_{}_QUP-YSF.png'.format(dir_0,directory,directory,modelS[III]),dpi=500,bbox_inches='tight')
        #plt.show()
        plt.close()
        
        plt.figure(figsize=(3.5,3.5))
        plt.plot(YSF[1:]*-1.,DEN[1:],linestyle=' ',marker = 'o', color = 'g')
        plt.axhline(y=1.*DEN_[III],linestyle='-',color='k',label='0D')
        plt.axvline(x=6.*3.2e-2,linestyle=':', color = 'k')
        plt.xlabel('Shock Position [mm]')
        plt.ylabel('Post-Shock Density [gcm$^{-3}$]')
        plt.legend(loc='best',fontsize='small')
        plt.savefig('./{}/{}/{}_{}_DEN-YSF.png'.format(dir_0,directory,directory,modelS[III]),dpi=500,bbox_inches='tight')
        #plt.show()
        plt.close()
        """
    
        III += 1
        model.closeFile()
    
    #np.savetxt('./{}/{}/Up-Us_{}.csv'.format(dir_0,directory,directory),np.column_stack((UP,US,UP_2,US_2,UPSD_0,UPSD_2)),delimiter=',')
    #np.savetxt('./{}/{}/All-Hugoniot-points_{}_6D.csv'.format(dir_0,directory,directory),np.column_stack((UP_2,UPSD_2,US_2,DEN_,DENSD,SGY_,SGYSD,PRE_,PRESD,TPS_,TPSSD,ALP_,ALPSD,DENs_,DENsSD,DEN0,ALP0)),delimiter=',')
    #plt.figure(figsize=(3.5,3.5))
    #plt.plot(UP,US,linestyle=' ',marker='o',mew=1,markerfacecolor='None',label='all touching',ms=5)
    #plt.plot(up1,us1,up2,us2,color='k')
    #plt.xlabel('Particle Velocity [ms$^{-1}$]')
    #plt.ylabel('Shock Velocity [ms$^{-1}$]')
    #plt.savefig('./{}/{}/hugoniot_us-up.png'.format(dir_0,directory),dpi=500,bbox_inches='tight')




