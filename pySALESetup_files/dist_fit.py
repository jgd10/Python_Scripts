import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def function(x,A,B,C,L):
    y = A/(B+C*np.exp(-L*x))
    return y
def func2(x,A,B,C,L):
    pdf  = A*L*C*np.exp(L*phi)/(B*np.exp(L*phi)+C)**2.
    return pdf


data  = np.genfromtxt('Lunar_distribution.dat',dtype=float)

A = 1
B = .5
C = 4.665
L = .7


phi = np.linspace(-2,9,100)
popt, pcov = curve_fit(function,data[:,0],data[:,1],p0 = [A,B,C,L])
print 'fit: A = {:3.3f}, B = {:3.3f}, C = {:3.3f}, L = {:3.3f}'.format(popt[0],popt[1],popt[2],popt[3])
NORM = function(9.,popt[0],popt[1],popt[2],popt[3])-function(-2.,popt[0],popt[1],popt[2],popt[3])
pdf  = A*L*C*np.exp(L*phi)/(B*np.exp(L*phi)+C)**2.
pdf /= NORM
print 'Norm factor = {}'.format(NORM)


D = 2**(-phi)
D*= 1000

fig = plt.figure()
ax = fig.add_subplot(211)
ph = np.linspace(-2,9,100)
ax.plot(ph,function(ph,popt[0],popt[1],popt[2],popt[3]),color='orange')
ax.axvline(x=5.5)
ax.axvline(x=2)
ax.set_ylabel('AREA% (Cumulative)')
ax1 = fig.add_subplot(212,sharex=ax)
ax1.plot(ph,func2(ph,popt[0],popt[1],popt[2],popt[3]),color='orange')
ax1.axvline(x=5.5)
ax1.axvline(x=2)
ax1.set_xlabel('$\phi$')
ax1.set_ylabel('AREA%')
ax1.text(-1,10,'Bottom?')
ax1.text(7,10,'Top?')
ax1.text(3,5,'Middle?')


plt.figure()
plt.plot(data[:,0],data[:,1],marker='o', linestyle=' ')
plt.plot(D,function(phi,popt[0],popt[1],popt[2],popt[3]))

plt.figure()
plt.plot(D,pdf)
plt.show()
