import sys
import time
"""
This script contains several loading bars and examples of them being used
all are text-based and only require the sys module. time is used to slow
them down sufficiently to see what's going on!
"""

print 'Simple Pinwheel\n'
# Simple pinwheel
k = 0
for i in range(100):
    percent_comp = int(100.*float(i)/99.)
    # remember to escape the backslash otherwise you're going to have a bad time
    wheel = ['\r / ','\r - ','\r \\ ','\r | ']
    print wheel[k]+' {}%'.format(percent_comp),
    sys.stdout.flush()
    k+=1
    if k == 4: k = 0
    time.sleep(.05)

print 
print 
print 'Method 1\n'
# Method 1, remake a raw string each time but with the marker in a different place
for i in range(500):
    # percent complete of the loop; this can be any iterator or quantity
    percent_comp = int(100.*float(i)/499.)
    # Make the progress bar as a combination of strings. The \r is required for
    # stdout.flush() to work properly
    # In each iteration change the location of the marker to be at the approx
    # percentage completion position
    prog_bar1 = '\r[' + ' '*(percent_comp-1)+'>'+' '*(100-percent_comp) + ']'+' {}%'.format(percent_comp)
    # to continually update the text the , at the end is necessary
    print prog_bar1,
    # flush the output
    sys.stdout.flush()
    # sleep; this just slows the process down a bit
    time.sleep(.01)

print 
print 
print 'Method 2\n'
# Method 2, make a list of strings, change the appropriate position to the marker, repeat
for i in range(500):
    percent_comp = int(100.*float(i)/499.)
    # this time we use a list of strings for the indexing properties
    prog_bar2 = ['\r[']+[' ']*100+[']']+[' {}%'.format(percent_comp)]
    # We can use the index to change the correct position to the marker
    # The key advantage of this method is you can present multiple tasks
    # on the same bar
    if percent_comp == 0:
        pass
    else:
        prog_bar2[percent_comp]='~'
    # The list of strings must be joined to be printed appropriately
    print ''.join(prog_bar2),
    sys.stdout.flush()
    time.sleep(.01)
print 
print 
print 'Method 3\n'

# Method 3, same as method 2 but don't update the bar to remove prev entries
prog_bar3 = ['\r[']+[' ']*100+[']']+[' ']
for i in range(500):
    percent_comp = int(100.*float(i)/499.)
    if percent_comp == 0:
        pass
    else:
        prog_bar3[percent_comp]='#'
    prog_bar3[-1]= ' {}%'.format(percent_comp)
    print ''.join(prog_bar3),
    sys.stdout.flush()
    time.sleep(.01)


