# ADDED 220 280 550

import pySALEPlot as psp
from pylab import figure,arange,colorbar
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy import stats
def find_behind_shock(jmax,AMAX,P,Den,Den_I):
    #print "####################### New Function Call #########################"
    
    j = 0								          # Counter for compaction wave location
    n = 0
    J = 0
    I_j = 0								          # Counter to ensure compaction front only found once.
    for j in xrange(jmax):
	#print Den[j]
        #print P[j], AMAX/2.						          # Iterate along array
	P_avg = np.mean(P[j-1:j+1])
        if P_avg > (AMAX/2.) and n == 0:				          # Condition for existence of compaction front 
            n   = 1							          # Counter to ensure only one wavefront located 
            I_j = 0 
            jj  = j 
             
            while (I_j==0):
                jj += 1 
		"""
                Den_std = sc.std(Den[jj:jj+10]/np.max(Den[jj:jj+10])) 
                #print Den,Den_std, jj 
                if ((Den_std < .001) and I_j == 0): 
                    I_j = jj 
                    #print abs(I_j-j) 
                else: 
                    pass                                            # Iterate along array
                """
		Den_std = sc.std(Den[jj:jj+100]/Den_I)
		Den_mean= abs(sc.mean(Den[jj:jj+100]/Den_I) - 1)    # Gives the fraction difference from the impactor density (either +10% or -10% for example)
		#print Den_std, jj
		if ((Den_std < .01) and Den_mean < .1 and I_j == 0):
		    I_j = jj-0
		    #print abs(I_j-j)
		else:
		    pass                                            # Iterate along array
	    J = j
	    #print J
            n = 1                                                  # Counter to ensure only one wavefront located
        else:
            pass
    return J, I_j

def find_shock_width(J,I_j,P,P_Quasi, Den_I):
    dj      = I_j - J
    i_i,i_f = J, J+dj
    j_i,j_f = J, J-dj
    II = 0
    JJ = 0

    for i in range(i_i,i_f,1):
	P_avg = np.mean(P[i-1:i+1])
	if (P_avg>0.8*P_Quasi and II == 0):
	    II = i
 
    for j in range(j_i,j_f,-1):
	P_avg = np.mean(P[j-1:j+1])
	if (P_avg<0.2*P_Quasi and JJ == 0):
	    JJ = j
    return II,JJ
	
    
    

field1 = 'Pre'                                                       
field2 = 'V_x'                                                       
field3 = 'V_y'
field4 = 'Den'                                                      
field5 = 'Syy' 
dirname='./del-t'								     # Directory name for the output files

modelv150 = psp.opendatfile('./v150/jdata.dat')					     # Open datafile 
modelv250 = psp.opendatfile('./v250/jdata.dat')					     # Open datafile 
modelv350 = psp.opendatfile('./v350/jdata.dat')					     # Open datafile 
modelv450 = psp.opendatfile('./v450/jdata.dat')					     # Open datafile 
modelv550 = psp.opendatfile('./v550/jdata.dat')					     # Open datafile 
modelv650 = psp.opendatfile('./v650/jdata.dat')					     # Open datafile 
modelv750 = psp.opendatfile('./v750/jdata.dat')					     # Open datafile 
modelv850 = psp.opendatfile('./v850/jdata.dat')					     # Open datafile 
modelv950 = psp.opendatfile('./v950/jdata.dat')					     # Open datafile 
modelv1050 = psp.opendatfile('./v1050/jdata.dat')					     # Open datafile 

models = [modelv150,modelv250,modelv350,modelv450,modelv550,modelv650,modelv750,modelv850,modelv950,modelv1050]
VI = ([150,250,350,450,550,650,750,850,950,1050])
[model.setScale('mm') for model in models]
N = 10 
t1 = .2 
t2 = .9 
imax = 45									     # total number of steps to iterate through 
intv = 1									     # interval between steps
rho_00= np.zeros((N))
Us_   = np.zeros((N))
Up_   = np.zeros((N))
rho_  = np.zeros((N))
sigL_ = np.zeros((N))
SW_   = np.zeros((N))
SWstd = np.zeros((N))

#Us_total   = np.zeros((N,imax/intv))
#Up_total   = np.zeros((N,imax/intv))
#rho_total  = np.zeros((N,imax/intv))
#sigL_total = np.zeros((N,imax/intv))
#SW_total   = np.zeros((N,imax/intv))
y2 = -0.
y1 = -1.0
hh = 0
III =0

for model in models:
        print "current model = {}".format(VI[III])
	III += 1

	time  = np.zeros(imax/intv)                                          # Initialise all variables of interest as arrays of zeros
	Y     = np.zeros(imax/intv)					     # Shock Position
	P_wf  = np.zeros(imax/intv)					     # Quasi-steady state pressure
	V_x   = np.zeros(imax/intv)					     # Transverse velocity
	V_y   = np.zeros(imax/intv)					     # Longitudinal velocity
	rho   = np.zeros(imax/intv)					     # Density
	SL    = np.zeros(imax/intv)					     # Longitudinal Stress
	SW    = np.zeros(imax/intv)					     # Shock Width


	step = model.readStep(['{}'.format(field1),'{}'.format(field4)],0)
	A    = sc.mean(abs(step.data[0][:,:]), axis = 0)                        # Ath array. used for definition of compaction front
	Den0 = np.ma.filled(step.data[1][:,:],0.)*1.e-3
	Den_I= Den0[0,0]
	Den0 = sc.mean(Den0, axis = 0)						# Ath array. used for definition of compaction front
	AMAX = np.amax(A)                                                       # Then AMAX is max value of A. May be redundant (uses V_x)
	jmax = np.size(A)  


	rho_00[hh] = sc.mean(Den0[(model.yc[0,:]>-1.0)*(model.yc[0,:]<0.0)])
						      
	for i in arange(0,imax,intv):                                                     # loop over steps
	    
	    step = model.readStep(['{}'.format(field1),'{}'.format(field2),'{}'.format(field3),'{}'.format(field4),'{}'.format(field5)],i)
	    P    = step.data[0][:,:]							  # Arrays of the variables
	    Vx1  = step.data[1][:,:]
	    Vy1  = step.data[2][:,:]
	    Den  = step.data[3][:,:]*step.cmc[0]
	    Syy  = step.data[4][:,:]
	    Syy  = np.ma.filled(Syy,0.)							  # Fill all masked values in the arrays (VOID) with 0.0
	    Sgy  = sc.mean((Syy-P), axis=0)*-1.e-9 
	    Den  = np.ma.filled(Den,0.)							  # Fill all masked values in the arrays (VOID) with 0.0 
	    P    = np.ma.filled(P,  0.)						          # Fill all masked values in the arrays (VOID) with 0.0 
	    Den  = sc.mean(Den, axis=0)*1.e-3
	    P    = sc.mean(P,   axis=0)*1.e-9
	    Vx   = sc.mean(abs(Vx1[:][:]), axis=0)
	    Vy   = sc.mean(abs(Vy1[:][:]), axis=0)
	 
	    J, I_j         = find_behind_shock(jmax,abs(AMAX),abs(Sgy),Den,Den_I)
	    #print J, I_j
            #II, JJ         = find_shock_width(J,I_j,abs(Sgy),abs(AMAX),Den_I)
	    #SW[i/intv]     = abs(model.yc[0,II]-model.yc[0,JJ])
	    rho[i/intv]    = sc.mean(Den[J:I_j])
	    SL[i/intv]     = sc.mean(Sgy[J:I_j])
	    P_wf[i/intv]   =   sc.mean(P[J:I_j])
	    V_x[i/intv]    =  sc.mean(Vx[J:I_j])
	    V_y[i/intv]    =  sc.mean(Vy[J:I_j])
	    #print Vy[J:I_j], "shock position = {}, impactor position = {}".format(model.yc[0,J],model.yc[0,I_j])
	    AMAX           = SL[i/intv]


	    Y[i/intv]      = model.yc[0,J] 
	    time[i/intv]   = step.time*1.e6
	    if (J == I_j == 0):
		AMAX = np.amax(P)*1.e9
	    else:
		pass
	tlin =  time[(Y<y2)*(Y>y1)]
	Ylin = abs(Y[(Y<y2)*(Y>y1)])
	Up   =   V_y[(Y<y2)*(Y>y1)]
	rho  =   rho[(Y<y2)*(Y>y1)]
	SL   =    SL[(Y<y2)*(Y>y1)]
	slope, intercept, r_value, p_value, std_err = stats.linregress(tlin,Ylin)
	#print Up,Ylin
	Up_mean   = sc.mean(Up)
	Up_err    = sc.std(Up)/(np.sqrt(np.size(Up)))
	print "for v_I = {} m/s: Up = {} +/- {} m/s, Us = {} m/s, With intercept = {}".format(VI[hh],Up_mean,Up_err,slope*1.e3,intercept)
	Up_[hh]   = Up_mean
	Us_[hh]   = slope*1.e3
	rho_[hh]  = sc.mean(rho)
	sigL_[hh] = sc.mean(SL)
#        SW_[hh]   = SW_avg
#	SWstd[hh] = SW_std
#	print VI[hh], SW_
	plt.figure()
	plt.plot(tlin,Up,linestyle=' ',marker='o',color='r')
	plt.xlabel('time (us)')
	plt.ylabel('particle vel (m/s)')
	plt.figure()
	plt.plot(Ylin,Up,linestyle=' ',marker='D',color='b')
	plt.xlabel('position (mm)')
	plt.ylabel('particle vel (m/s)')
	plt.figure()
	plt.plot(tlin,Ylin,linestyle=' ',marker='d',color='k')
	plt.xlabel('time (us)')
	plt.ylabel('position (mm)')
	plt.figure()
	plt.plot(tlin,rho,linestyle=' ',marker='<',color='g')
	plt.xlabel('time (us)')
	plt.ylabel('density (gcm^-3)')
	plt.show()
	plt.close()
	
	hh+=1

"""
plt.figure()
plt.errorbar(VI,SW_,yerr=SWstd*0.5,marker='o',label=r'$\pm 0.5 \sigma$')
plt.xlabel('impact velocity (m/s)')
plt.ylabel('shock width (mm)')
plt.legend(numpoints=1)
plt.show()
"""

I=0

DATA = np.column_stack((Up_,Us_,VI,rho_00,rho_,sigL_))
sc.savetxt('up-us-VI-rho00-rho-sigL.csv',DATA,delimiter=',',header='Up [m/s] - Us [m/s] - Impact Velocity [m/s] - Initial Density [g/cm^3] - Final Density [g/cm^-3] - Longitudinal Stress [GPa]')


#print np.size(L), np.size(Up), np.size(Us)
#print L, Up, Us
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.text(225,1000,"Linear fit: $U_s$ = {:.3f}$U_p$ + {:.3f}".format(S,c_s))
#print Up, Us
#ax.text(225,675,"Quadratic fit: $U_s$ = {:.3f}*$U_p^2$ + {:.3f}*$U_p$ + {:.3f}".format(M1,M2,M3))
#ax.set_title("Hugoniot in $U_s-u_p$ plane Y = 5GPa \n With Identical Initial Meshes (53.22\% PARFRAC)")
ax.plot(Up_,Us_, linestyle= ' ', marker = 'p', markersize = 16)
#Up = Up[:]
#L = L[:]
#ax.plot(Up,M,color='g')
ax.set_xlabel("Particle Velocity $u_{p}$ [m/s]")
ax.set_ylabel("Shock Velocity $U_{s}$ [m/s]")
plt.show()

