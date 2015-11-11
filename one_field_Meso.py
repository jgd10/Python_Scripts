import pySALEPlot as psp
from pylab import figure,arange,colorbar
import numpy as np
# This example plotting script designed to plot 
# damage in the demo2D example simulation

# Make an output directory
field1 = 'Pre'#raw_input('Which field is to be considered? (3 chars) =>')
field2 = 'V_y'#raw_input('Which field is to be considered? (3 chars) =>')
dirname='./%s'%field2
psp.mkdir_p(dirname)

# Open the datafile
model=psp.opendatfile('jdata.dat')

# Set the distance units to km
model.setScale('mm')

# Set up a pylab figure
fig=figure()
ax=fig.add_subplot(111,aspect='equal')
fig.set_size_inches(4,8)
# Loop over timesteps, in increments of 5

#step_1=model.readStep('%s'%field2,0)
#MAXvalue = abs(np.amax(step_1.data[0]))
for i in arange(0,301,1):
    # Set the axis labels
    ax.set_xlabel('Transverse position [mm]')
    ax.set_ylabel('Longitudinal postion [mm]')

    # Set the axis limits
    ax.set_xlim([0,2])
    ax.set_ylim([-3.5,.5])

    # Read the time step 'i' from the datafile
    step=model.readStep(['%s'%field2,'%s'%field1],i)
    norm_data = (step.data[0])

    # Plot the pressure field
    p=ax.pcolormesh(model.x,model.y,norm_data,
            cmap='plasma_r',vmin=-200,vmax=0)

    # Add a colourbar, but only need to do this once
    if i == 0:
        cb=fig.colorbar(p)
        cb.set_label('Longitudinal velocity [m/s]')
    #ax.set_xlim(0,1)
    #ax.set_ylim(-3,1)
    ax.set_title('t = {: 5.2f}$\mu s$'.format(step.time*1.e6))
   
    # Save the figure
    fig.savefig('{}/{}_full{:05d}.png'.format(dirname,field2,i),dpi=500)

    # Clear the axis for the next timestep
    ax.cla()
