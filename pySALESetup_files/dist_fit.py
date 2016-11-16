import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def function(x,A,B,C,L):
    y = A/(B+C*np.exp(-L*x))
    return y

data  = np.genfromtxt('Lunar_distribution.dat',dtype=float)

A = 1
B = .5
C = 4.665
L = .7

phi = np.linspace(-2,9,100)
popt, pcov = curve_fit(function,data[:,0],data[:,1],p0 = [A,B,C,L])
print 'fit: A = {:3.3f}, B = {:3.3f}, C = {:3.3f}, L = {:3.3f}'.format(popt[0],popt[1],popt[2],popt[3])
NORM = function(9.,A,B,C,L)-function(-2.,A,B,C,L)
pdf  = A*L*C*np.exp(L*phi)/(B*np.exp(L*phi)+C)**2.
pdf /= NORM

plt.figure()
plt.plot(data[:,0],data[:,1],marker='o', linestyle=' ')
plt.plot(phi,function(phi,popt[0],popt[1],popt[2],popt[3]))

plt.figure()
plt.plot(phi,pdf)
plt.show()
