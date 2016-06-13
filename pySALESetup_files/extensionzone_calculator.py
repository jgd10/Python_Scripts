from numpy import *
print "###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###"
print "###Script calculating the number of extension zone cells required for a given length###"
print "###Script accepts standard form, and all input must be REAL                         ###"
print "###                           J.G. Derrick - 05/03/2015                             ###"
print "###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###"

L=float(raw_input("Input the length of zone (m): "))
GRIDSPC=float(raw_input("Input min. width of each cell GRIDSPC (m): "))
GRIDSPCM=float(raw_input("Input max cellsize as multiple of original width of each cell GRIDSPCM: "))
#GRIDSPCM does not take non-multiples yet. i.e. it won't work if in iSALE the value is greater than 0
GRIDEXT=float(raw_input("Input factor of extension of each subsequent cell GRIDEXT: "))

l=0                                      # Physical size of extension zone during calculation
n=0                                      # No. cells in extension zone calculator
while (l<=L):                            # calculate actual length (metres)
    if (GRIDEXT**n>GRIDSPCM):                  # cap cell size at 20 times original
        l+=GRIDSPCM*GRIDSPC                    # add on capped size
        #print n
    else:
	#print 'things are happening'
        l+=GRIDSPC*(GRIDEXT**(n+1))          # increment by new cell size
    n+=1
print "Number of cells in extension zone: {}" .format(n)

Cont = str(raw_input("Would you like to calculate the OBJRES of an object containing an extension zone? [y/n]: "))
if (Cont == 'y'):
     ObjLength = float(raw_input("Input the total Length of the object in question (m): "))
     print "OBJRES = {}".format(.5*ObjLength/GRIDSPC)
else:
     print "No worries, maybe next time."
