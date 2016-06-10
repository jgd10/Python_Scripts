# Remove the randomness from the function.
# Can we make a grain both highly angluar and highly elliptical, then morph it to be less angular, more spherical? Iterative process?
# Shrink radii slowly? Simulate it as if geological. Moving average around shape to round it?
# Construct a bunch of highly elliptical, angular shapes, then reduce these effects. All to construct this library!
# Geologists ACTUALLY look at the angularity and the eccentricity (or sphericity) and we will know this info.
# Gareth's idea is to have 2 algorithms, one to generate a grain, one to smooth it. Numbers with no random element can go in/be stored for later use.


import numpy as np
import random
import matplotlib.pyplot as plt

def aspect_ratio(A):
    Diameters = []
    N = np.size(A)
    for i in range(N/2 - 1):                    # To calculate the diamaters, cycle through N/2-1 times. e.g. if N=6, then you only need to do 0,1,2 radii
        Diameters.append(A[i]+A[i+N/2])
    aspect = np.amax(Diameters)/np.amin(Diameters)
    return aspect

NUMBER = 100
Ns = 50
mesh0 = np.zeros((Ns,Ns))
D = 30
S = 0.0                                    # Default atm
N = 12
A = 0.1
mixed = True

if N%2 != 0:
    N += 1
else:
    pass
for u in range(NUMBER):
    mesh0*= 0.
    AREA  = 0.
    Pi    = np.pi
    rho   = np.ones((N))*D/2.                      # Array of all the radii in this shape. Start all at mean radii
    R     = np.zeros((2,N))                        # Array for the coords of the vertices
    beta  = 2.*Pi/N
    a     = D/2.                                   # Semi-major axis. b is semi-minor
    b     = a*np.sqrt(1.-S**2.)                    # S, equivalent to eccentricity. S = 0 => circle
    theta = 0                                      # Always start the first point on the horizontal
    rotat = random.uniform(-Pi,Pi)                    # Randomly rotate the shape by between -pi and pi radians
    for item in range(N):
        rho[item] = a*b/np.sqrt((b*np.cos(theta+rotat))**2. + (a*np.sin(theta+rotat))**2.)
        # radius of an ellipse
        rho[item] = np.random.normal(rho[item],A*rho[item])    # Randomly select a radius, with the radius of the ellipse as a mean and A*radius as the std
        R[0,item] = rho[item]*np.cos(theta)            # x coordinate
        R[1,item] = rho[item]*np.sin(theta)            # y coordinate
        theta    += beta                    # increment theta
    aspect     = aspect_ratio(rho)
    max_radius = np.amax(rho)
    I = np.copy(R[0,:])
    J = np.copy(R[1,:])
    n = np.size(J)
    I = np.append(I,I[0])
    J = np.append(J,J[0])
    R/= Ns
    np.savetxt('grain_aspect-{:3.3f}.txt'.format(aspect),R.transpose(),delimiter=',',header='normalised x, y coords of grain with aspect ratio {} and centre at (0,0)'.format(aspect))
    assert I[0] == I[-1],J[0] == J[-1] 
    qx = 0.                                                                                 
    qy = 0.                                                                                 
    x0 = Ns/2 
    y0 = Ns/2
    plt.figure()
    for j in range(Ns):                                                                     
        for i in range(Ns):                                                                 
            xc = i - x0                                                       
            yc = j - y0                                                       
            sx = xc - qx                                                                    
            sy = yc - qy           
            intersection = 0                                                                
            r_ = np.sqrt(xc**2. + yc**2.)
            if r_ <= (max_radius+1): 
               xx = np.linspace(i-.5,i+.5,10)                                                
               yy = np.linspace(j-.5,j+.5,10)                                                
               for ii in range(10):
                   for jj in range(10):                                                                    
                       xxc = xx[ii] - x0                                                          
                       yyc = yy[jj] - y0
                       sx  = xxc - qx                                                                        
                       sy  = yyc - qy           
                       Xn = 0                                                                      
                       for l in range(n):                                                                  
                           rx = I[l+1] - I[l]                                                                
                           ry = J[l+1] - J[l]
                           RxS = (rx*sy-ry*sx)                                                               
                           if RxS!=0.:                                                                       
                               t = ((qx-I[l])*sy - (qy-J[l])*sx)/RxS
                               u = ((qx-I[l])*ry - (qy-J[l])*rx)/RxS
                               if t<=1. and t>=0. and u<=1. and u>=0.: Xn += 1
                       if (Xn%2==0):                                                        
                           mesh0[j,i] += 0.1**2.
    np.savetxt('grain_aspect-{:3.3f}_mesh.txt'.format(aspect),mesh0,delimiter=',', fmt='%1.4f',header='# full mini-mesh of grain. Ns = {}, mixed = True, D, S, N, A = {}, {}, {}, {}'.format(aspect,D,S,N,A))
    """
    plt.imshow(mesh0,cmap='binary',interpolation='nearest',vmin=0,vmax=1)
    #plt.plot(x0,y0,color='r',marker='o')
    #plt.plot(I+Ns/2,J+Ns/2,marker='o',color='b')
    plt.xlim(0,Ns)
    plt.ylim(0,Ns)
    #plt.show()
    """
