import pySALEPlot as psp
from matplotlib.pyplot import figure,colorbar
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
# This example plotting script designed to plot 
# damage in the demo2D example simulation

# Make an output directory
#field1 = 'Den'#raw_input('Which field is to be considered? (3 chars) =>')
field2 = 'Den'#raw_input('Which field is to be considered? (3 chars) =>')
dirname='./%s'%field2
psp.mkdir_p(dirname)

# Open the datafile
model=psp.opendatfile('jdata.dat')

# Set the distance units to km
model.setScale('um')

# Set up a pylab figure
fig=figure()
ax=fig.add_subplot(111,aspect='equal')
fig.set_size_inches(4,3)
# Loop over timesteps, in increments of 5

#step_1=model.readStep('%s'%field2,0)
#MAXvalue = abs(np.amax(step_1.data[0]))
plt.tight_layout()
for i in np.arange(0,37,1):
    # Set the axis labels
    ax.set_xlabel('Transverse position [$\mu$m]',fontsize=10)
    ax.set_ylabel('Longitudinal postion [$\mu$m]',fontsize=10)

    # Set the axis limits
    ax.set_xlim([0,200])
    ax.set_ylim([-500,0])
    ax.set_xticklabels(np.arange(0,201,25),fontsize=10)
    ax.set_yticklabels(np.arange(500,-1,-100),fontsize=10)
    # Read the time step 'i' from the datafile
    step=model.readStep(['%s'%field2],i)#,'%s'%field1],i)
    norm_data = step.data[0]*1.e-3
    #norm_data  = step.data[0]*-1.e-3
    #norm_data  = step.data[0]*1.e-9

    # Plot the pressure field
    p=ax.pcolormesh(model.x,model.y,norm_data,
            cmap='viridis',vmax=4.,vmin=0.)

    # Add a colourbar, but only need to do this once
    if i == 0:
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="5%", pad=0.05)
		cb=fig.colorbar(p,cax=cax)
		cb.set_label('density [gcm$^{-3}$]',fontsize=10)
		#cb.set_label('Longitudinal Velocity [kms$^{-1}$]',fontsize=10)
		#cb.set_label('Pressure [GPa]',fontsize=10)
    #ax.set_xlim(0,1)
    #ax.set_ylim(-3,1)
    ax.set_title('t = {: 5.2f}$ns$'.format(step.time*1.e9))
    # Save the figure
    fig.savefig('{}/{}_full{:05d}.png'.format(dirname,field2,i),dpi=500)

    # Clear the axis for the next timestep
    ax.cla()
