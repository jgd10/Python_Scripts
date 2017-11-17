import numpy as np
import matplotlib.pyplot as plt

def vel_factor(M,m,N):
    psi = 1
    for k in range(N):        
        psi *= (M+k*m)/(M+(k+1)*m)
    return psi


# Al density is 2.7   g/cc
# WC density is 15.56 g/cc
# 20 mm x 1 mm Al impactor has mass/length of 2.7*1000*20*1*1.e-6
# 1 32um diameter WC grain has mass/length of 15.56*1000*pi*((32*1.e-6)**2)
Al_mass = 2.7*1000*20*1*1.e-6
WC_mass = 15.56*1000*np.pi*((32*1.e-6)**2)
print Al_mass, WC_mass
M   = Al_mass
#m   = (np.arange(20)+1)*WC_mass
v0  = 950.
N   = 616.
A   = np.array([0.4,0.6,0.8,1.0,1.2,1.4,1.6])
# k (number of clumps) will be some function of N, number of clumps will scale with number 
# of particles and A. This is a test function; as A increases k will drop
k   = 0.1*N/(1.+A)
# nm is the avg number of grains per clump, with a total of k clumps
nm  = N/k 
print nm
# i.e. (nm = 1+A)
m   = nm*WC_mass
L   = np.size(k)


PSI = np.zeros_like((k),dtype=float)
for i in range(L):
    PSI[i] = vel_factor(M,m[i],int(k[i]))


fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(223,sharex=ax1)
ax3 = fig.add_subplot(222)
ax4 = fig.add_subplot(224,sharex=ax3)
ax1.plot(k,PSI)
ax1.set_ylabel(r'Coefficient, $\psi$')

ax2.plot(k,PSI*v0,marker='o',mfc='None',mec='crimson',linestyle=' ')
ax2.set_xlabel('Number, k, of little masses in m\nN=600')
ax2.set_ylabel('Final velocity [ms$^{-1}$]\n$v_0 =$ '+'{:2.2f}'.format(v0))

ax3.plot(A,PSI)
ax3.set_ylabel(r'Coefficient, $\psi$')

ax4.plot(A,PSI*v0,marker='o',mfc='None',mec='crimson',linestyle=' ')
ax4.set_xlabel('Coordination Number $A$')
ax4.set_ylabel('Final velocity [ms$^{-1}$]\n$v_0 =$ '+'{:2.2f}'.format(v0))
fig.tight_layout()
plt.show()
