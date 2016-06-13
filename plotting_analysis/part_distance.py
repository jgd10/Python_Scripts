import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.stats import expon,linregress
from matplotlib import cm

def func(x,a,b,c):
    return a * np.exp(-b * x) + c
def func1(x,a,b,c):
    return a/(b*x) + c

fname1  = '6cpprleasttouch_11-09-15.iSALE' 
fname2  = '6cpprhalftouch_11-09-15.iSALE' 
fname4  = '6cpprmosttouch_07-09-15.iSALE' 

fname1a = '8cpprleasttouch_02-10-15.iSALE'  
fname2a = '8cpprhalftouch_02-10-15.iSALE'  
fname4a = '8cpprmosttouch_02-10-15.iSALE'  

fname1b = '10cpprleasttouch_02-10-15.iSALE'
fname2b = '10cpprhalftouch_02-10-15.iSALE' 
fname4b = '10cpprmosttouch_02-10-15.iSALE' 

fname1c = '12cpprleasttouch_10-09-15.iSALE' 
fname2c = '12cpprhalftouch_10-09-15.iSALE'
fname4c = '12cpprmosttouch_10-09-15.iSALE'

fname1d = 'meso_multitouch_11-08-15.iSALE'
fname2d = 'meso_fewtouch_13-08-15.iSALE'

group   = [fname1,fname2,fname4,fname1a,fname2a,fname4a,fname1b,fname2b,fname4b,fname1c,fname2c,fname4c]
cont    = ['Fewest Contacts','Half Contacts','Most Contacts']
#group = [fname1d,fname2d]
M = np.size(group)
error = 2.e-4						   # The uncertainty on the radius will be +/- 1 cell width. In this case it's 2um (2.e-4cm cos CTH works in cm)
diameters   = np.array([.0024,.0024,.0024,.0032,.0032,.0032,.0040,.0040,.0040,.0048,.0048,.0048])
upper_bound = diameters + error/2.
DIST = np.zeros((M))


MPN = 4							   # Max No. (contact) Points (MPD says 6, but this isn't realistic so we use 4
I = 0
DIST_avg = []    
plt.figure(figsize=(3., 3.0))
C = []
for fname in group:

    X = np.genfromtxt(fname,usecols=(1),dtype=float)	   # X and Y coords of each particle centre
    Y = np.genfromtxt(fname,usecols=(2),dtype=float)
    radii = np.genfromtxt(fname,usecols=(3),dtype=float)   # Sim for radius of each particle
    N = np.size(X)					   # No. of particles
    #print N
    dist_avgs = []
    mean_radii = np.mean(radii)				   # Sanity check the mean radii are as expected
    print 'Max, Min, Mean radii = {},{},{} from file {}'.format(np.amax(radii),np.amin(radii),mean_radii,fname)
    
    distances = np.zeros((N*MPN))			   # Total number of distances will be number of particles * possible distances per particle (MAX)
    D  = np.ones((N))
    Dtouch = []
    for i in range(N-1):
	D *= 0.
	D -= 1.
	for j in range(N-1):				   # find distances between current particle and ALL other particles and store these in D
	    dx = X[i] - X[j]
	    dy = Y[i] - Y[j]
	    distance = np.sqrt(dx**2. + dy**2.)
	    D[j] = distance
	Dtemp = D[(D>0.)*(D<upper_bound[I])]		   # Array containing all distances between centres, greater than 0., but less than upperbound on the diameter
	
	for item in Dtemp:				   # Append all these distances to the array Dtouch. This contains all distances to particles that are in contact w\ this one
	    Dtouch.append(item)

    Dtouch = np.array(Dtouch)				   # Convert to numpy array
    C.append(float(np.size(Dtouch))/float(N))		   # The size of Dtouch/total part number is the mean
	
    if I<3:
    	plt.bar(I,C[-1],color=cm.binary(1.*I/3),label = '{}'.format(cont[I]))
    elif I>2 and I<6:
    	plt.bar(I,C[-1],color=cm.binary(1.*(I-3)/3))
    elif I>5 and I<9:
    	plt.bar(I,C[-1],color=cm.binary(1.*(I-6)/3))
    else:
    	plt.bar(I,C[-1],color=cm.binary(1.*(I-9)/3))

    plt.ylabel(r'Average Contact Points [Particle$^{-1}$]')
    I += 1
C = np.array(C)
labels = (['  ',' 6','  ','  ',' 8','  ','  ','10','  ','  ','12','  '])
plt.xticks(range(I),labels,size='small',ha='left')#,rotation='vertical',ha='left')
plt.xlabel('No. Cells Per Particle Radius (cppr)')
plt.legend(fontsize='x-small',loc='best',framealpha=0.5)
cppr = np.array([6,8,10,12])
plt.savefig('contacts_bar.pdf', format='pdf',dpi=500,bbox_inches='tight')

