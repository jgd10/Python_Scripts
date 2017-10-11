import sys
import numpy as np
import time

for i in range(500):
    percent_comp = int(100.*float(i)/500.)
    prog_bar = '\r[' + ' '*(percent_comp-1)+'>'+' '*(100-percent_comp) + ']'
    print prog_bar,
    sys.stdout.flush()
    time.sleep(.01)

for i in range(500):
    percent_comp = max(int(100.*float(i)/500.),1)
    prog_bar = ['\r[']+[' ']*100+[']']
    prog_bar[percent_comp]='|'
    print ''.join(prog_bar),
    sys.stdout.flush()
    time.sleep(.01)

for i in range(500):
    percent_comp = max(int(100.*float(i)/500.),1)
    prog_bar = ['\r[']+[' ']*(100-percent_comp)+[']']
    prog_bar[1]='#'*percent_comp
    print ''.join(prog_bar),
    sys.stdout.flush()
    time.sleep(.01)

c = 0
bar = '#'
prog_bar = ['\r[']+[' ']*(100-percent_comp)+[']']
for i in range(1500):
    if i == 500:  
        c = 500
        bar=':'
    if i == 1000: 
        c = 1000 
        bar='>'
    percent_comp = max(int(100.*float(i-c)/500.),1)

    prog_bar[1]= bar*percent_comp
    print ''.join(prog_bar),
    sys.stdout.flush()
    time.sleep(.01)
