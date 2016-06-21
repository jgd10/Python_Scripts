import numpy as np
from scipy.sparse.csgraph import dijkstra
import matplotlib.pyplot as plt
import sys


def Dstra_1(A,filepath = 'meso.iSALE',plot=False):
    """
    This function calculates the average distance between a particle and the next one below it, that is NOT in contact: 'G'.
    G is calculated in multiples of radii.
    """
    M = np.genfromtxt(filepath,dtype=float,usecols=(0))
    X = np.genfromtxt(filepath,dtype=float,usecols=(1))
    Y = np.genfromtxt(filepath,dtype=float,usecols=(2))
    R = np.genfromtxt(filepath,dtype=float,usecols=(3))
    indices = np.argsort(Y)
    
    Y  = Y[indices]/R
    X  = X[indices]/R
    R  = R[indices]/R
    Xx = X[(X>=0)*(Y>=0)]
    Yy = Y[(X>=0)*(Y>=0)]
    R  = R[(X>=0)*(Y>=0)]
    X  = np.copy(Xx)
    Y  = np.copy(Yy)
    crit_angle  = 40                                                                                    # The angle after which Force chains no longer propagate
    crit_angle *= np.pi/180.                                                                            # i.e. they are pushed out the way (need in radians)
    mean_radii  = np.mean(R)																		    # Calculate the mean radii	
    rad_err     = .1+2.                                                                                 # 10% uncertainty on radius 
    UB = X + 2.*R*np.sin(crit_angle)                                                              # Check all particles below, within this window
    LB = X - 2.*R*np.sin(crit_angle)                                                              # that is within the critical angle arc
    
    N = np.size(X)
    
    
    graph    = np.zeros((N,N))
    gapgraph = np.zeros((N,N))
    for i in range(N):
        count = 0
        for j in range(N):
            if i == j:
                graph[i,j]    = np.inf
                gapgraph[i,j] = np.inf
            elif X[j] < UB[i] and X[j] > LB[i] and Y[j] > Y[i] and count<=1:                    # Only consider particles within the critical arc
                dx = abs(X[i] - X[j])
                dy = abs(Y[i] - Y[j])
                dr = np.sqrt(dx**2. + dy**2.)
                graph[i,j]    = dr
                gapgraph[i,j] = dr - 2. # weights are just the gaps, no gap => zero weight
                count += 1
            else:
                graph[i,j]    = np.inf
                gapgraph[i,j] = np.inf
    
    gapgraph[gapgraph<=0.] = 0.
    
    gapgraph = np.ma.masked_where(gapgraph==np.inf,gapgraph)
    graph    = np.ma.masked_where(graph==np.inf,graph)
    
    nn    = int((np.amax(Y)-np.amin(Y)+2.*np.amax(R))/2.*np.mean(R))
    end   = np.zeros((nn),dtype=int)
    start = np.arange(nn,dtype = int)						# Start points are the 10 closest to the impactor
    dist  = np.zeros((nn))
    gdist = np.zeros((nn))
    for k in range(nn):
        D1 = 0
        end[k] = k
        for l in range(N):
            if X[l]<(X[k]+R[k]) and X[l]>(X[k]-R[k]) and Y[l] > Y[k]:
                dx = abs(X[l] - X[k])
                dy = abs(Y[l] - Y[k])
                dr = np.sqrt(dx**2. + dy**2.)
                D0 = dr
                if D0 > D1: 
                    D1     = D0
                    end[k] = l
                else:
                    pass
    dgraph,predD = dijkstra(graph,return_predecessors=True)
    Ggraph,predG = dijkstra(gapgraph,return_predecessors=True)
    for m in range(nn):
    	dist[m]  = dgraph[start[m],end[m]]
    	gdist[m] = Ggraph[start[m],end[m]]
    
    if plot == True:
        fig = plt.figure(figsize=(3.5,3.5))
        ax  = fig.add_subplot(111,aspect='equal')
        ax.set_ylim([np.amin(X)-np.amax(R),np.amax(X)+np.amax(R)])
        ax.set_xlim([np.amin(Y)-np.amax(R),np.amax(Y)+np.amax(R)])
        for i in range(N):																				# Plot each circle in turn
            circle = plt.Circle((X[i],Y[i]),R[i],color='k',lw=.5,fill=False)#'{:1.2f}'.format((M[i])*.5/np.amax(M))) 
            ax.add_patch(circle)
        ax.set_xlim(0,np.amax(X))
        ax.set_ylim(np.amax(Y),0)
    
    	for o in range(nn):
            ax.plot([X[start[o]],X[end[o]]],[Y[start[o]],Y[end[o]]],marker='x',mew=1.,ms=4,color='k',lw=2.,linestyle=' ')
            ii = end[o]
            steps = 0
            II = ii
            while ii != start[o]:
                steps += 1
                #ax.plot(X[ii],Y[ii],marker='x',mew=2,ms=5,color='r')
                ax.plot([X[II],X[ii]],[Y[II],Y[ii]],color='k',lw=1.)
                II = ii
                ii = predG[start[o],ii]
                if ii == -9999: break
            if ii == -9999:
                pass
            else:
                ax.plot([X[II],X[ii]],[Y[II],Y[ii]],color='k',lw=1.)
    
        ax.set_xlabel('Transverse \nPosition [Radii]',fontsize=14)
        ax.set_ylabel('Longitudinal \nPosition [Radii]',fontsize=14)
    gdist = np.array(gdist) 
    K = (np.mean(gdist[gdist!=np.inf])/np.mean(gdist[gdist!=np.inf]+2*(steps+1))) * 100.
    ax.set_title('$K = ${:1.1f}% '.format(K),fontsize=16)
    plt.tight_layout()
    plt.savefig('gaps_Dijk1_figure_A-{}_K-{:1.3f}percent.pdf'.format(A,K),format='pdf',dpi=600,bbox_inches='tight')		                				 # Save the figure
    plt.show()
    return K

filepath = sys.argv[1]
A = sys.argv[2]
plot = True

K = Dstra_1(A,filepath,plot)

print K
