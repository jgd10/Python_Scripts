import numpy as np
import matplotlib.pyplot as plt
import sys


def particle_gap_measure(A,filepath = 'meso.iSALE',plot=False):
    """
    This function calculates the average distance between a particle and the next one below it, that is NOT in contact: 'G'.
    G is calculated in multiples of radii.
    """
    M = np.genfromtxt(filepath,dtype=float,usecols=(0))
    X = np.genfromtxt(filepath,dtype=float,usecols=(1))
    Y = np.genfromtxt(filepath,dtype=float,usecols=(2))
    R = np.genfromtxt(filepath,dtype=float,usecols=(3))
    N = np.size(X)
    
    crit_angle  = 40                                                                                    # The angle after which Force chains no longer propagate
    crit_angle *= np.pi/180.                                                                            # i.e. they are pushed out the way (need in radians)
    mean_radii  = np.mean(R)																		    # Calculate the mean radii	
    rad_err     = .1+2.                                                                                 # 10% uncertainty on radius 
    D           = np.zeros((N))																			# D is an array for the distances between particles
    Dgap        = []																					# The final list of distances for parts in contact
    Dtotal      = []																					# The final list of distances for parts in contact
    UB          = X + 2.*R*np.sin(crit_angle)                                                              # Check all particles below, within this window
    LB          = X - 2.*R*np.sin(crit_angle)                                                              # that is within the critical angle arc

    X /= R
    Y /= R
    UB/= R
    LB/= R
    R /= R
    if plot == True:																					# If plot == True then produce a figure
        fig = plt.figure()
        ax = fig.add_subplot(111,aspect='equal')
        ax.set_xlim(0,np.amax(X))
        ax.set_ylim(np.amax(Y),0)
        for i in range(N):																				# Plot each circle in turn
            ax.plot([X[i]],[Y[i]],color='k',marker='x',linestyle=' ',ms=3,mew=1.)
    
    for i in range(N):
        best_dy = np.amax(Y) - Y[i]
        old_dy  = best_dy
        best_X = X[i] 
        best_Y = np.amax(Y) 
        old_X  = best_X
        old_Y  = best_Y
        for j in range(N):	# find distances between current particle and 
            if i == j:
                pass
            elif X[j] < UB[i] and X[j] > LB[i] and Y[j] > Y[i]:                                            # Only consider particles within the critical arc
                dX = abs(X[j] - X[i])
                dY = abs(Y[j] - Y[i]) #- R[i] - R[j]                                                         # And ones below the current particle
                dy = np.sqrt(dX**2. + dY**2.)
                if dy < old_dy: 
                    best_dy = dy
                    best_X  = X[j]
                    best_Y  = Y[j]
                else:
                    best_dy = old_dy 
                    best_X  = old_X
                    best_Y  = old_Y
                old_dy  = best_dy
                old_X   = best_X
                old_Y   = best_Y
            else:
                pass
        D[i] = best_dy 
        if best_dy <= rad_err:
            Dtotal.append(D[i])
            if plot == True:
                ax.plot([X[i],best_X],[Y[i],best_Y],lw=1.5,color='r')
        else:
            Dgap.append(D[i])
            Dtotal.append(D[i])
            if plot == True:
                ax.plot([X[i],best_X],[Y[i],best_Y],lw=1.,color='b')
    Dgap  = np.array(Dgap)			                                                                 # Convert to numpy array
    Dgap -= 2                                                                                           # Now Dgap is the distance between parts and NOT centres
     
    #G = np.sum(Dgap)/N																 # The size of Dtouch/total part number is the mean
    if np.size(Dgap) == 0:
        G = 0
    else:
        G = np.sum(Dgap)																									 # gap length, in radii
    L = np.sum(Dtotal)
    K = G/L * 100.
    if plot == True: 
    	ax.set_title('$K = ${:1.3f}% '.format(K))
        ax.set_xlabel('Transverse Position [Radii]')
        ax.set_ylabel('Longitudinal Position [Radii]')
        plt.savefig('gaps_figure_A-{}_K-{:1.3f}percent.png'.format(A,K),dpi=600)		                				 # Save the figure
    	plt.show()
    return K

filepath = sys.argv[1]
A = sys.argv[2]
plot = True

K = particle_gap_measure(A,filepath,plot)

print K
