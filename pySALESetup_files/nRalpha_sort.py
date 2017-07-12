import glob
import numpy as np
import os

"""
This script takes all the grain files in a directory and selects only
ones that have a specified elongation (n), specific surface area (a)
and 'roundness' (R). For full definitions of each of these parameters
See the Lunar Sourcebook, Chapter 9, & Appendix A (Mahmood et al 1974b)

"""
def polygon_area(x,y,N):                               # Defined as taking the determinant around each vertex
    A = 0.
    for i in range(1,N):
        A += 0.5*(x[i-1]*y[i] - x[i]*y[i-1])
    A += 0.5*(x[-1]*y[0] - x[0]*y[-1])
    return abs(A)

def angles_with_faces(r,f1,f2):
    theta1 = np.arccos(np.dot(r,f1)/(np.linalg.norm(r)*np.linalg.norm(f1)))
    theta2 = np.arccos(np.dot(r,f2)/(np.linalg.norm(r)*np.linalg.norm(f2)))
    return theta1,theta2

def side_angles(V,N):
    Th = np.zeros((N))
    for i in range(N):
        k1 = i-1
        k2 = i
        line = V[k1,:]-V[k2,:]
        Th[i] = np.arctan2(line[1],line[0])
    return Th
def interior_angles(V,N):
    Th = np.zeros((N))
    for i in range(N):
        k0 = i-1
        k1 = i
        k2 = i+1
        if k2 == N: k2 = 0
        
        line1 = V[k1,:]-V[k0,:]
        line2 = V[k1,:]-V[k2,:]

        Th[i] = np.arccos(np.dot(line1,line2)/(np.linalg.norm(line1)*np.linalg.norm(line2)))
        if Th[i] > np.pi: Th[i] = 2*np.pi - Th[i]
    return Th

n     = 1.35        # elongation: Length/Width
n_tol = 0.03            
a     = 0.72        # Specific Area: Area/(Length x Width)
a_tol = 0.02
R     = 0.215       # Roundness: Average of radii of curvature at each point/radius of max inscribed circle
R_tol = 0.005


filenames = glob.glob('grain*.txt')
directory = './N-10_n-1.35_alpha-.72/'
if not os.path.exists(directory):
    os.mkdir(directory)
for fname in filenames:
    V = np.genfromtxt(fname,delimiter=',',dtype=float) # Vertices
    N,d = np.shape(V)                                  # N = No. vertices, d = No. dimensions
    
    l = 0.                                             # Length
    b = 0.                                             # Width
    d = 0.
    l = np.amax(V[:,0])-np.amin(V[:,0])
    b = np.amax(V[:,1])-np.amin(V[:,1])
    """
    for i in range(N):
        for j in range(N):
            l = max(l,abs(V[i,0]-V[j,0]))              # Max length and width between vertices
            b = max(b,abs(V[i,1]-V[j,1]))
    """
    L    = max(l,b)
    #print L,B
    B    = min(l,b)
    n_   = L/B
    #print n_
    area = polygon_area(V[:,0],V[:,1],N)
    a_   = area/(L*B)

    if n_ >= (n-n_tol) and n_ <= (n+n_tol) and a_ >= (a-a_tol) and a_ <= (a+a_tol): 
        q     = 0
        rad   = np.zeros((N))
        ang   = np.zeros((N))
        angle2= np.zeros((N*2))
        radii2= np.zeros((N*2))
        minrad= L
        for k in range(N):
            q      = min(minrad,np.linalg.norm(V[k,:]))
            rad[k] = np.linalg.norm(V[k,:])
            ang[k] = np.arctan2(V[k,1],V[k,0])

        theta = interior_angles(V,N)
        S     = 0.2*q/np.tan(theta/2.)
        m     = 0 
        phi   = side_angles(V,N)
        V2    = np.zeros((N*2,2))
        for h in range(N):
            V2[m,0] = V[h,0] + S[h]*np.cos(phi[h])
            V2[m,1] = V[h,1] + S[h]*np.sin(phi[h])
            m += 1
            h2 = h+1
            if h2 == N: h2 = 0
            V2[m,0] = V[h,0] - S[h]*np.cos(phi[h2])
            V2[m,1] = V[h,1] - S[h]*np.sin(phi[h2])
            m += 1
        np.savetxt('{}v2_{}'.format(directory,fname),V2,delimiter=',',header='x, y coords (in cells) of grain {} with roundness = 0.2'.format(fname))
        

    
            
        


