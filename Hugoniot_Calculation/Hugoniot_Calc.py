import numpy as np
import scipy as sc
import scipy.stats as stats
import pySALEPlot as psp
import matplotlib.pyplot as plt

def find_shock_front(imax,jmax,sfronts,jfronts,A,YC,A_lim):
    for i in range(0,imax,1):
        j = 0
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
    

def calc_postshock_state(SF,particle_bed,YC,imax,jfronts,sfronts,posback,Den,VY,impactor):
    """
    ***
    This function calculates the post-shock particle bed state. It does this by individually looking
    at each longitudinal column of cells. It starts at the current shock position and scans back
    though the bed until it reaches the impactor.
    ***
    It will only activate if the shock wave is in a non-continuous material
    """
    sfronts_n = sfronts/SF						        # Normalised shock front positions relative to mean
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
            while posback[i] == sfronts[i]: 
    	        j = min(j,jmax) 					    # Failsafe in case algorithm reaches the end of the array
    	        if impactor[i,j]>0. or YC[i,j] >= 0.: 
    	            posback[i] = YC[i,j]
    	            jback[i]   = j
    	        else:
    	            pass
                j += 1
            if posback[i]!=sfronts[i]:
                QUP += np.ma.sum(VY[i,jfronts[i]:jback[i]])
                NN  += np.ma.MaskedArray.count(VY[i,jfronts[i]:jback[i]])
            else:
                QUP += 0
                NN  += 0
    else:
       NN = 1.
    if NN == 0.: NN = 1. 
    QUP /= NN
    return QUP,jback

def hugoniot_point_Usup_ND(QUP,YSF,TME,N):
    global t1,t2
    t1   = .0 
    t2   = 0.8 
    D    = -3.2e-2								                    # in mm and -ve as up is +ve
    QUP  = np.ma.masked_invalid(QUP)
    time = TME[(YSF<(N*D))*(YSF>-1.)*(TME<t2)]					    # Only use times and particle velocities for results WITHIN the particle bed
    up   = QUP[(YSF<(N*D))*(YSF>-1.)*(TME<t2)]					    # .8mm into the particle bed is given, for the shock to stabilise
    sf   = YSF[(YSF<(N*D))*(YSF>-1.)*(TME<t2)]					    # Shock front positions
    
    UP   = np.ma.mean(up[1:])
    UPSD = np.ma.std(up[1:])
    US, intercept, other, things, too = stats.linregress(time[1:],sf[1:])
    US  *=-1.
    UP  *=-1.
    """
    plt.figure()
    plt.plot(time,-1*sf,linestyle=' ',marker='o')
    plt.plot(TME,-1*YSF,linestyle=' ',marker='*')
    plt.plot(time,time*US-intercept,linestyle='-',marker='o')
    """
    return UP,US*1000,-intercept,UPSD

def hugoniot_point_general_ND(AAA,YSF,TME,N):
    global t1,t2
    t1   = .0 
    t2   = 0.8 
    D    = -3.2e-2								                    # in mm and -ve as up is +ve
    AAA  = np.ma.masked_invalid(AAA)
    time = TME[(YSF<(N*D))*(YSF>-1.)*(TME<t2)]					    # Only use times and particle velocities for results WITHIN the particle bed
    a    = AAA[(YSF<(N*D))*(YSF>-1.)*(TME<t2)]					    # .8mm into the particle bed is given, for the shock to stabilise
    sf   = YSF[(YSF<(N*D))*(YSF>-1.)*(TME<t2)]					    # Shock front positions
    A    = np.ma.mean(a[1:])
    ASD  = np.ma.std(a[1:])
    return A,ASD
    
dir_0  = './hugoniot_data'								            # Directory name for the output files
dirs   = ['A-1.1201_K-28.673']
psp.mkdir_p(dir_0)
psp.mkdir_p(dir_0+'/cmaps')

N = 11

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

III =0

field1 = 'Pre'                                                       
field2 = 'V_y'
field3 = 'Den'
field4 = 'TPS'
field5 = 'Syy'
                                                      
tstart = 1									    # Start at t>0 such that there IS a shock present
tfin   = 56 
tintv  = 1



for directory in dirs:
    modelv150 = psp.opendatfile('{}/v150/jdata.dat'.format(directory))					     # Open datafile 
    modelv200 = psp.opendatfile('{}/v200/jdata.dat'.format(directory))					     # Open datafile 
    modelv250 = psp.opendatfile('{}/v250/jdata.dat'.format(directory))					     # Open datafile 
    modelv300 = psp.opendatfile('{}/v300/jdata.dat'.format(directory))					     # Open datafile 
    modelv350 = psp.opendatfile('{}/v350/jdata.dat'.format(directory))					     # Open datafile 
    modelv450 = psp.opendatfile('{}/v450/jdata.dat'.format(directory))					     # Open datafile 
    modelv550 = psp.opendatfile('{}/v550/jdata.dat'.format(directory))					     # Open datafile 
    modelv650 = psp.opendatfile('{}/v650/jdata.dat'.format(directory))					     # Open datafile 
    modelv750 = psp.opendatfile('{}/v750/jdata.dat'.format(directory))					     # Open datafile 
    modelv850 = psp.opendatfile('{}/v850/jdata.dat'.format(directory))					     # Open datafile 
    modelv950 = psp.opendatfile('{}/v950/jdata.dat'.format(directory))					     # Open datafile 
    models = [modelv150,modelv200,modelv250,modelv300,modelv350,modelv450,modelv550,modelv650,modelv750,modelv850,modelv950]
    modelS = (['v150','v200','v250','v300','v350','v450','v550','v650','v750','v850','v950'])
    VI     = ([150,200,250,300,350,450,550,650,750,850,950])
    [model.setScale('mm') for model in models]
    III = 0
    hh  = 0
    for model in models:
        sf = 0.										    # Initialise the shock front position
        psp.mkdir_p('./{}/{}/{}'.format(dir_0,directory,modelS[III]))
        print "########################################"
        print "current model = {}".format(modelS[III])
        print "########################################"
        YC        = model.yc
        XC        = model.xc
        TME       = np.zeros(( int((tfin-tstart+1)/tintv) ))				    # time 
        YSF       = np.zeros(( int((tfin-tstart+1)/tintv) ))				    # Shock Front Y position
        PBD       = np.zeros(( int((tfin-tstart+1)/tintv) ))				    # Shock in Particle Bed array. 1 is yes 0 is no
        QUP       = np.zeros(( int((tfin-tstart+1)/tintv) ))				    # Quasi steady state Particle velocity
        SGY       = np.zeros(( int((tfin-tstart+1)/tintv) ))				    # Quasi steady state Longitudinal Stress
        DEN       = np.zeros(( int((tfin-tstart+1)/tintv) ))				    # Quasi steady state Bulk Density
        PRE       = np.zeros(( int((tfin-tstart+1)/tintv) ))				    # Quasi steady state Pressure
        TPS       = np.zeros(( int((tfin-tstart+1)/tintv) ))				    # Quasi steady state Total Plastic Strain
    
        dy        = abs(YC[0,1] - YC[0,0])
        dx        = abs(XC[1,0] - XC[0,0])
        
        t         = 1
        tfin2     = model.nsteps
        counter   = 0
        step      = model.readStep(['{}'.format(field3)],0)
        Den       = np.ma.filled(step.data[0],0.)*1.e-3*step.cmc[0]
        print 'vfrac = {} %'.format(np.mean(Den[(YC<0.)*(YC>-1.)])*100./15.56)
        imax,jmax = np.shape(Den)
        sfronts   = np.zeros((imax))
        jfronts   = np.zeros((imax),dtype=int)
        posback   = np.zeros((imax))		                    			     # position of back of bed
        jback     = np.zeros((imax),dtype=int)					                 # j coord of back of bed
        AMAX      = 0.
        while np.around(sf,decimals=2) < 1. and t < tfin:
            particle_bed = 0
    
            step = model.readStep(['{}'.format(field1),'{}'.format(field2),'{}'.format(field3),'{}'.format(field4),'{}'.format(field5)],t)
            Den  = np.ma.filled(step.data[2],0.)*1.e-3*step.cmc[0]
            Pre  = np.ma.filled(step.data[0],0.)*1.e-9
            Syy  = np.ma.filled(step.data[4],0.)*1.e-9
            tps  = np.ma.filled(step.data[3],-1.)
            Sgy  = (Syy - Pre)*-1.
            VY   = step.data[1]
    
            #AMAX      = VI[III]/3.
            sfronts, jfronts, SF = find_shock_front(imax,jmax,sfronts,jfronts,Pre,YC,AMAX/2.)	    
    
            print 'Time: {:3.2f} us'.format(step.time*1.e6)
            print 'Shock Front position = {:3.2f} mm'.format(SF)
            print 'Max SF position: {:3.2f} mm, Min SF position: {:3.2f} mm \n'.format(np.amax(-sfronts),np.amin(-sfronts))
            
            YSF[(t-1)/tintv]  = SF 
            TME[(t-1)/tintv]  = step.time*1.e6
            sf                = -SF 
            
            QUP[(t-1)/tintv],jback = calc_postshock_state(SF,particle_bed,YC,imax,jfronts,sfronts,posback,Den,VY,step.cmc[1]) 
            
            SPre = []
            SSgy = []
            SDen = []
            Stps = []
            for i in range(imax):
                for j in range(jfronts[i],jback[i]+1,1):
                    SPre.append(Pre[i,j])
                    SSgy.append(Sgy[i,j])
                    SDen.append(Den[i,j])
                    if tps[i,j]!=-1.: Stps.append(tps[i,j])
            
            
            PRE[(t-1)/tintv] = np.mean(SPre)
            DEN[(t-1)/tintv] = np.mean(SDen)
            SGY[(t-1)/tintv] = np.mean(SSgy)
            TPS[(t-1)/tintv] = np.mean(Stps)

            if t%10 == 0:
                plt.figure()
                plt.plot(model.yc[0,:],np.mean(abs(VY),axis=0)/np.amax(np.mean(abs(VY),axis=0)),color='k',lw=2.)
                plt.pcolormesh(model.yc[0,:],model.xc[:,0],abs(VY),cmap='plasma',alpha=0.4)
                plt.axvline(x=SF,linestyle='--',color='k',lw=1.5)
                plt.axhline(y=abs(QUP[(t-1)/tintv])/np.amax(np.mean(abs(VY),axis=0)),linestyle='--',color='k',lw=1.5)
                plt.xlabel('Position [mm]')
                plt.ylabel('Normalised Velocity [x {:3.1f} m/s] \n Position [mm]'.format(np.amax(np.mean(abs(VY),axis=0))))
                plt.axis('equal')
                plt.xlim(1,-1)
                plt.ylim(0,1)
                plt.savefig('./{}/cmap_lines_{}_{}_{}_cmap_{:05d}.png'.format(dir_0,directory,modelS[III],field2,t),dpi=500,bbox_inches='tight')
                plt.close()
            AMAX = np.mean(SPre)
            
            t += tintv
    
        TME = TME[(YSF!=0.)]
        QUP = QUP[(YSF!=0.)]
        YSF = YSF[(YSF!=0.)]
        PRE = PRE[(YSF!=0.)]
        SGY = SGY[(YSF!=0.)]
        TPS = TPS[(YSF!=0.)]
        DEN = DEN[(YSF!=0.)]

        UP_2[III],US_2[III],C_2,UPSD_2[III] = hugoniot_point_Usup_ND(QUP[:-1],YSF[:-1],TME[:-1],2.)
        
        SGY_[III], SGYSD[III] = hugoniot_point_general_ND(SGY[:-1],YSF[:-1],TME[:-1],2.)
        DEN_[III], DENSD[III] = hugoniot_point_general_ND(DEN[:-1],YSF[:-1],TME[:-1],2.)
        TPS_[III], TPSSD[III] = hugoniot_point_general_ND(TPS[:-1],YSF[:-1],TME[:-1],2.)
        PRE_[III], PRESD[III] = hugoniot_point_general_ND(PRE[:-1],YSF[:-1],TME[:-1],2.)

        print "up = {} m/s, Us = {} m/s ~ Density = {} g/cc, Stress (L) = {} GPa".format(UP_2[III],US_2[III],DEN_[III],SGY_[III])
        
        plt.figure(figsize=(3.5,3.5))
        plt.plot(TME[1:],abs(YSF[1:]),linestyle=' ',marker = 'x',mew=1.5,color = 'm')
        plt.plot(TME[1:],1.e-3*US_2[III]*TME[1:]+C_2,linestyle=':',color='k',label='2D')
        plt.axhline(y=(2.*3.2e-2),linestyle='-.',color='k',label='2D')
        plt.ylabel('Shock Position [mm]')
        plt.xlabel('Time [$\mu$s]')
        plt.legend(loc='best',fontsize='small')
        plt.savefig('./{}/{}/{}_{}_YSF-TME.png'.format(dir_0,directory,directory,modelS[III]),dpi=500,bbox_inches='tight')
        
        plt.figure(figsize=(3.5,3.5))
        plt.plot(YSF[1:]*-1.,QUP[1:]*-1.,linestyle=' ',marker = 'o', color = 'b')
        plt.axhline(y=1.*UP_2[III],linestyle=':',color='k',label='2D')
        plt.axvline(x=2.*3.2e-2,linestyle=':', color = 'k')
        plt.title('Up - Shock Position')
        plt.xlabel('Shock Position [mm]')
        plt.ylabel('Particle Velocity [ms$^{-1}$]')
        plt.legend(loc='best',fontsize='small')
        plt.savefig('./{}/{}/{}_{}_QUP-TME.png'.format(dir_0,directory,directory,modelS[III]),dpi=500,bbox_inches='tight')
        plt.close()
        
        state_vars = np.column_stack((TME,YSF,QUP,SGY,DEN,TPS,PRE))
        HEAD = '{}:time [us]; Shock Position [mm]; particle velocity [m/s]; Stress [GPa]; Density [g/cc]; Total Plastic Strain [/m]; Pressure [GPa]'.format(modelS[III])
        np.savetxt('./{}/{}/state_variables_{}.csv'.format(dir_0,directory,modelS[III]),state_vars,header=HEAD,delimiter = ',')
    
        III += 1
        model.closeFile()
    
    g = [2.,50.,1.5,400.]
    np.savetxt('./{}/{}/Up-Us_{}.csv'.format(dir_0,directory,directory),np.column_stack((UP,US,UP_2,US_2,UPSD_0,UPSD_2)),delimiter=',')
    np.savetxt('./{}/{}/All-Hugoniot-points_{}.csv'.format(dir_0,directory,directory),
            np.column_stack((UP_2,UPSD_2,US_2,DEN_,DENSD,SGY_,SGYSD,PRE_,PRESD,TPS_,TPSSD)),delimiter=',')

    plt.figure(figsize=(3.5,3.5))
    plt.plot(UP,US,linestyle=' ',marker='o',mew=1.5,markerfacecolor='None',label='all touching',ms=8)
    plt.xlabel('Particle Velocity [ms$^{-1}$]')
    plt.ylabel('Shock Velocity [ms$^{-1}$]')
    plt.savefig('./{}/{}/hugoniot_us-up.png'.format(dir_0,directory),dpi=500,bbox_inches='tight')




