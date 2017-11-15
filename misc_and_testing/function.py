import numpy as np
import matplotlib.pyplot as plt

def polynom_shape(n,a,r,x,y,x0,y0):
    assert a[0] == a[-1] 
    # the first coeff MUST be equal to the last
    x -= x0
    y -= y0
    R = 0
    k = 0
    for ak in a:
        R += ak * x**(2.*k) * y**(2.*(n-k)) 
        k += 1
    return R <= r**(2*n)

def polar_shape(n,a,b,c,r,x,y,x0,y0):
    # the first coeff MUST be equal to the last
    x -= x0
    y -= y0
    rr     = np.sqrt(x**2. + y**2.)
    #if x == 0: x = 1.e-10
    theta = np.arctan2(y,x)
    R = r
    k = 1
    #ind = np.argsort(abs(a))
    #a = a[ind]
    #a = a[::-1]
    N = np.size(a)
    for h in range(N):
        R += a[h]*r*np.sin(k*theta)*np.exp(-k*c)
        R += b[h]*r*np.cos(k*theta)*np.exp(-k*c)
        k += 1
    return rr <= R

M = np.zeros((1000,1000))

x0,y0 = 50,50
N     = 7
mu    = 0. 
sigma = 0.1
i_start = 0
j_start = 0
for K in range(10):
    c = np.linspace(0.05,0.14,10)
    a = np.linspace(0,.01,10)*K
    b = np.linspace(0,.01,10)*K
    #a = a[::-1]
    #b = b[::-1]
    for L in range(10):
        #if N%2 == 0:
        #    a = sigma*np.random.randn(N)
        #    b = sigma*np.random.randn(N)
        #else:
        #    a = sigma*np.random.randn(N)
        #    b = sigma*np.random.randn(N)
        for i in range(i_start,i_start+100):
            for j in range(j_start,j_start+100):
                if polar_shape(N,a,b,c[L],30,i,j,x0,y0)==True: M[j,i] = 1.
        x0      += 100
        i_start += 100
        if x0 > 950: 
            x0 = 50
            i_start = 0
    y0      += 100
    j_start += 100

plt.figure()
plt.imshow(M,cmap='binary',interpolation='nearest')
plt.axis('equal')
plt.xlim([0,1000])
plt.ylim([0,1000])
plt.show()
