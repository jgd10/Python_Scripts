import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def func1(x,a,b,c,d,e):
    return a/(b+x**c) + d*np.exp(e*x)

data = np.genfromtxt('MassDensitySensitivity3.csv',delimiter=',',dtype=float)

p0 = [1,0,0,0,0]
popt,pcov = curve_fit(func1,data[:,0],data[:,1],p0=p0)

print 'a = {:2.3f}, b = {:2.3f}, c = {:2.3f}, d = {:2.3f}, e = {:2.3f}'.format(*popt) 

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(data[:,0],data[:,1],linestyle=' ',marker='+',ms=4)
ax.plot(data[:,0],func1(data[:,0],*popt))

plt.show()


