import numpy as np
import scipy as sc
import scipy.stats as stats
import pySALEPlot as psp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def find_shock_front_3(imax,jmax,sfronts,jfronts,A,YC,A_lim,YDV,jback):
    jj = np.median(jfronts)
    for i in range(0,imax,1):
        j = 0#jj - 300
        j = max(j,0)
        Shock = False
        while Shock == False and YC[i,j] < YDV[i]:#(0.0 - dy): 
            j = min(j,jmax-3)							    # Prevent loop exceeding end of array
            #if YC[i,j] >= YDV[i]: J = j #> (0.0-dy): J = j
            if A[i,j] > A_lim: 
                sfronts[i] = min(YC[i,j+1],0.)
                jfronts[i] = j
                Shock = True
                break
            else:
                pass
            j += 1
        if YC[i,j] >= YDV[i] : 
            sfronts[i] = YDV[i]
            jfronts[i] = jback[i]
    SF_med = np.median(sfronts)						    # Shock Front position, as a mean of all the shock fronts in j
    SF_avg = np.mean(sfronts)						    # Shock Front position, as a mean of all the shock fronts in j
    
    return sfronts, jfronts, SF_med, SF_avg

def find_jback(YC,imax,impactor):
    jback = np.zeros((imax))
    jback = jback.astype(int)
    
    yy = YC[0,:]
    Ytemp = np.copy(YC)
    Ytemp[impactor==0] = 9999.

    YDV = np.amin(Ytemp,axis=1)
    jback = np.argmin(Ytemp,axis=1)
    return YDV,jback

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

def hugoniot_point_ND(QUP,YSF,TME,N):
    global t1,t2
    t1 = .0 
    t2 = 0.8 
    D  = 3.2e-2								    # in mm and -ve as up is +ve
    YSF = abs(YSF)
    QUP = np.ma.masked_invalid(QUP)
    time = TME[(YSF>(N*D))*(YSF<.85)]					    # Only use times and particle velocities for results WITHIN the particle bed
    up   = QUP[(YSF>(N*D))*(YSF<.85)]					    # .8mm into the particle bed is given, for the shock to stabilise
    sf   = YSF[(YSF>(N*D))*(YSF<.85)]					    # Shock front positions
    
    #UP   = outlier_mean(up)
    UP   = np.ma.mean(up[1:])
    UPSD = np.ma.std(up[1:])
    US, intercept, other, things, too = stats.linregress(time[1:],sf[1:])
    US = abs(US)
    #UP*=-1.
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
    D  = 3.2e-2								    # in mm and -ve as up is +ve
    AAA= np.ma.masked_invalid(AAA)
    YSF = abs(YSF)
    time = TME[(YSF>(N*D))*(YSF<.85)]					    # Only use times and particle velocities for results WITHIN the particle bed
    a    = AAA[(YSF>(N*D))*(YSF<.85)]					    # .8mm into the particle bed is given, for the shock to stabilise
    sf   = YSF[(YSF>(N*D))*(YSF<.85)]					    # Shock front positions
    #UP   = outlier_mean(up)
    A   = np.ma.mean(a[1:])
    ASD = np.ma.std(a[1:])
    return A,ASD
    
dir_0  = './hugoniot_data'								     # Directory name for the output files
dirs   = ['A-1.5203','A-0.8520','A-0.7226','A-1.1201','A-0.4051','A-1.2878']

N = 11

UPMIN     = np.zeros((N))
UPMED     = np.zeros((N))
UPMAX     = np.zeros((N))

USMIN     = np.zeros((N))
USMED     = np.zeros((N))
USMAX     = np.zeros((N))

#UPSD_0 = np.zeros((N))

#UP_2   = np.zeros((N))
#US_2   = np.zeros((N))
#UPSD_2 = np.zeros((N))

#QVX_   = np.zeros((N))
#QVXSD  = np.zeros((N))
#QVY_   = np.zeros((N))
#QVYSD  = np.zeros((N))
SGY_MIN   = np.zeros((N))
SGY_MED   = np.zeros((N))
SGY_MAX   = np.zeros((N))
#SGYSD  = np.zeros((N))
#PRE_   = np.zeros((N))
#PRESD  = np.zeros((N))
DEN_MIN   = np.zeros((N))
DEN_MED   = np.zeros((N))
DEN_MAX   = np.zeros((N))
#DENSD  = np.zeros((N))
#ALP_   = np.zeros((N))
#ALPSD  = np.zeros((N))
#DENs_  = np.zeros((N))
#DENsSD = np.zeros((N))

DEN0   = np.zeros((N))
ALP0   = np.zeros((N))

III =0

field1 = 'Pre'                                                       
field2 = 'V_y'
field3 = 'Den'
field4 = 'TPS'
field5 = 'Syy'
field6 = 'Tmp'
field7 = 'V_x'
dirname= 'Hugoniots'
                                                      
tstart = 1									    # Start at t>0 such that there IS a shock present
tfin   = 201 
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
    VI = ([150,200,250,300,350,450,550,650,750,850,950])
    VI = VI[::-1]
    models = models[::-1]
    modelS = modelS[::-1]
    [model.setScale('mm') for model in models]
    III =0
    hh = 0
    for model in models:
        sf = 0.										    # Initialise the shock front position
        print "########################################"
        print "current model = {}".format(modelS[III])
        print "########################################"
        YC     = model.yc
        XC     = model.xc
        nnn    = int((tfin-tstart+1)/tintv)
        TME    = np.zeros(( nnn ))				    # time 
        YSFMIN    = np.zeros(( nnn ))				    # Shock Front Y position
        YSFMED    = np.zeros(( nnn ))				    # Shock Front Y position
        YSFMAX    = np.zeros(( nnn ))				    # Shock Front Y position
        PBD    = np.zeros(( nnn ))				    # Shock in Particle Bed array. 1 is yes 0 is no
        QUPMIN    = np.zeros(( nnn ))				    # Quasi steady state Particle velocity
        QUPMED    = np.zeros(( nnn ))				    # Quasi steady state Particle velocity
        QUPMAX    = np.zeros(( nnn ))				    # Quasi steady state Particle velocity
        #QVX    = np.zeros(( nnn ))				    # Quasi steady state Particle velocity
        #QVY    = np.zeros(( nnn ))				    # Quasi steady state Particle velocity
        SGYMIN    = np.zeros(( nnn ))				    # Quasi steady state Longitudinal Stress
        SGYMED    = np.zeros(( nnn ))				    # Quasi steady state Longitudinal Stress
        SGYMAX    = np.zeros(( nnn ))				    # Quasi steady state Longitudinal Stress
        DENMIN    = np.zeros(( nnn ))				    # Quasi steady state Bulk Density
        DENMED    = np.zeros(( nnn ))				    # Quasi steady state Bulk Density
        DENMAX    = np.zeros(( nnn ))				    # Quasi steady state Bulk Density
        #DENs   = np.zeros(( nnn ))				    # Quasi steady state Solid Density
        #ALP    = np.zeros(( nnn ))				    # Quasi steady state Distension
        PRE    = np.zeros(( nnn ))				    # Quasi steady state Pressure
        
        #QUPstd    = np.zeros(( nnn ))				
        #SGYstd    = np.zeros(( nnn ))			
        #PREstd    = np.zeros(( nnn ))				
    
        dy     = abs(YC[0,1] - YC[0,0])
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
        imax,jmax = np.shape(Den)
        sfronts   = np.zeros((imax))
        jfronts   = np.zeros((imax),dtype=int)
        posback   = np.zeros((imax))					     # position of back of bed
        jback     = np.zeros((imax),dtype=int)					     # j coord of back of bed
        AMAX = 0.
        while np.around(sf,decimals=2) < 1. and t < tfin2:
    
            particle_bed = 0
    
            step = model.readStep(['{}'.format(field1),'{}'.format(field2),'{}'.format(field3),'{}'.format(field4),'{}'.format(field5),'V_x'],t)
            Den  = np.ma.filled(step.data[2],0.)*1.e-3*step.cmc[0]
            Pre  = np.ma.filled(step.data[0],0.)*1.e-9
            Syy  = np.ma.filled(step.data[4],0.)*1.e-9
            tps  = np.ma.filled(step.data[3],-1.)
            Tmp  = step.data[5]
            Sgy  = (Syy - Pre)*-1.
            VY   = step.data[1]
            VX   = step.data[5]
    
            YDV,jback = find_jback(YC,imax,step.cmc[1])
            
            #fig2 = plt.figure()
            #ax1 = fig2.add_subplot(111,aspect='equal')
            #ax1.pcolormesh(model.xc,model.yc,Den)
            #ax1.plot(XC[:,0],YC[0,jback])
            #ax1.set_ylim(-0.5,0.5)
            #plt.show()

            sfronts, jfronts, SF, SF_mean = find_shock_front_3(imax,jmax,sfronts,jfronts,Pre,YC,AMAX/2.,YDV,jback)	    
    
            #print 'Time: {:3.2f} us'.format(step.time*1.e6)
            print 'Max SF; Mean SF; Min SF.  {:3.2f} mm; {:3.2f} mm; {:3.2f} mm\n'.format(np.amax(-sfronts),-SF,np.amin(-sfronts))
            
            T = (t-1)/tintv
            YSFMED[T]  = -SF #np.mean(sfronts)
            YSFMIN[T]  = np.amin(-sfronts)
            YSFMAX[T]  = np.amax(-sfronts)
            TME[T]     = step.time*1.e6
            sf         = abs(np.amin(sfronts))#-SF #
            
            SPre    = np.ones((250000))*-9999.
            SSgyMIN = np.ones((250000))*-9999.
            SSgyMED = np.ones((250000))*-9999.
            SSgyMAX = np.ones((250000))*-9999.
            SDenMIN = np.ones((250000))*-9999.
            SDenMED = np.ones((250000))*-9999.
            SDenMAX = np.ones((250000))*-9999.
            Sup_MIN = np.ones((250000))*-9999.
            Sup_MED = np.ones((250000))*-9999.
            Sup_MAX = np.ones((250000))*-9999. 
                
            fill = step.cmc[0]
            iV = 0              # index for fields that include voids (Stress, Pressure, Density)
            iS = 0              # index for fields that do NOT include voids (Velocity)
            for i in range(imax):
                for j in range(jfronts[i],jback[i]+1,1):
                    SPre[iV]    = Pre[i,j]
                    SSgyMAX[iV] = Sgy[i,j] # All cells will necessary be behind the max position by definition!
                    SDenMAX[iV] = Den[i,j]
                    #print iV,iS,i,j
                    if fill[i,j] > 0.: Sup_MAX[iS] = np.sqrt(VX[i,j]**2.+VY[i,j]**2.)

                    if abs(YC[i,j])<abs(YSFMED[T]):
                        SSgyMED[iV] = Sgy[i,j] # All cells will necessary be behind the max position by definition!
                        SDenMED[iV] = Den[i,j]
                        if fill[i,j] > 0.: Sup_MED[iS] = np.sqrt(VX[i,j]**2.+VY[i,j]**2.)

                    if abs(YC[i,j])<abs(YSFMIN[T]):
                        SSgyMIN[iV] = Sgy[i,j] # All cells will necessary be behind the max position by definition!
                        SDenMIN[iV] = Den[i,j]
                        if fill[i,j] > 0.: Sup_MIN[iS] = np.sqrt(VX[i,j]**2.+VY[i,j]**2.)
            
                    iV += 1
                    if fill[i,j] > 0.: iS += 1 

            SPre    = SPre[SPre!=-9999.]
            SSgyMIN = SSgyMIN[SSgyMIN!=-9999.]
            SSgyMED = SSgyMED[SSgyMED!=-9999.]
            SSgyMAX = SSgyMAX[SSgyMAX!=-9999.]
            SDenMIN = SDenMIN[SDenMIN!=-9999.]
            SDenMED = SDenMED[SDenMED!=-9999.]
            SDenMAX = SDenMAX[SDenMAX!=-9999.]
            Sup_MIN = Sup_MIN[Sup_MIN!=-9999.]
            Sup_MED = Sup_MED[Sup_MED!=-9999.]
            Sup_MAX = Sup_MAX[Sup_MAX!=-9999.]
            
            #porosity  = 1. - np.mean(Conc)
            
            QUPMIN[T] = np.mean(Sup_MIN)
            QUPMAX[T] = np.mean(Sup_MAX)
            QUPMED[T] = np.mean(Sup_MED)
            
            #ALP[(t-1)/tintv] = 1./(1.-porosity)
            PRE[T] = np.mean(SPre)
            DENMIN[T] = np.mean(SDenMIN)
            DENMED[T] = np.mean(SDenMED)
            DENMAX[T] = np.mean(SDenMAX)
            SGYMIN[T] = np.mean(SSgyMIN)
            SGYMED[T] = np.mean(SSgyMED)
            SGYMAX[T] = np.mean(SSgyMAX)
            #DENs[T] = np.mean(SDen[Conc>0.])
            
            #QUPstd[(t-1)/tintv] = np.std(Sup)
            #PREstd[(t-1)/tintv] = np.std(SPre)
            #SGYstd[(t-1)/tintv] = np.std(SSgy)

            #print QUP[(t-1)/tintv]

            AMAX = PRE[T]
            
            t += tintv
    
        TME = TME[(YSFMIN!=0.)]
        QUPMIN = QUPMIN[(YSFMIN!=0.)]
        QUPMED = QUPMED[(YSFMIN!=0.)]
        QUPMAX = QUPMAX[(YSFMIN!=0.)]
        PRE = PRE[(YSFMIN!=0.)]
        SGYMIN = SGYMIN[(YSFMIN!=0.)]
        SGYMED = SGYMED[(YSFMIN!=0.)]
        SGYMAX = SGYMAX[(YSFMIN!=0.)]
        DENMIN = DENMIN[(YSFMIN!=0.)]
        DENMED = DENMED[(YSFMIN!=0.)]
        DENMAX = DENMAX[(YSFMIN!=0.)]
        
        YSFMED = YSFMED[(YSFMIN!=0.)]
        YSFMAX = YSFMAX[(YSFMIN!=0.)]
        YSFMIN = YSFMIN[(YSFMIN!=0.)]

        UPMAX[III],USMAX[III],CMAX,junk = hugoniot_point_ND(QUPMAX,YSFMAX,TME,2.)
        UPMED[III],USMED[III],CMED,junk = hugoniot_point_ND(QUPMED,YSFMED,TME,2.)
        UPMIN[III],USMIN[III],CMIN,junk = hugoniot_point_ND(QUPMIN,YSFMIN,TME,2.)
        
        SGY_MAX[III],junk = hugoniot_point_general_ND(SGYMAX,YSFMAX,TME,2.)
        SGY_MED[III],junk = hugoniot_point_general_ND(SGYMED,YSFMED,TME,2.)
        SGY_MIN[III],junk = hugoniot_point_general_ND(SGYMIN,YSFMIN,TME,2.)
        
        DEN_MAX[III],junk = hugoniot_point_general_ND(DENMAX,YSFMAX,TME,2.)
        DEN_MED[III],junk = hugoniot_point_general_ND(DENMED,YSFMED,TME,2.)
        DEN_MIN[III],junk = hugoniot_point_general_ND(DENMIN,YSFMIN,TME,2.)
        
        ###############################################################################
        ###############################################################################
        # ignore early weirdness in distension by putting N = 6
        ###############################################################################
        ###############################################################################
        
        plt.figure(figsize=(3.5,3.5))
        plt.plot(YSFMAX,DENMAX,linestyle='-', marker = ' ',mew=1.5, color = 'r',label='max')
        plt.plot(YSFMED,DENMED,linestyle='-', marker = ' ',mew=1.5, color = 'k',label='med')
        plt.plot(YSFMIN,DENMIN,linestyle='-', marker = ' ',mew=1.5, color = 'b',label='min')
        plt.axhline(y=DEN_MAX[III], color = 'r')
        plt.axhline(y=DEN_MED[III], color = 'k')
        plt.axhline(y=DEN_MIN[III], color = 'b')
        plt.ylabel('Density')
        plt.xlabel('Shockpos')
        plt.legend(loc='best',fontsize='small')
        plt.savefig('./paper_review/MAXMEDMIN_{}_{}_QUPstd-Time.png'.format(directory,modelS[III]),dpi=500,bbox_inches='tight')
        plt.close()
        
         
    
        III += 1
        model.closeFile()
    
    np.savetxt('./paper_review/MAXMEDMIN_UPUSDENSGY_Hugoniot_{}_2D.csv'.format(directory),np.column_stack((UPMAX,UPMED,UPMIN,USMAX,USMED,USMIN,DEN_MAX,DEN_MED,DEN_MIN,SGY_MAX,SGY_MED,SGY_MIN)),delimiter=',')



