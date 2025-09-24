import os
import numpy as np
import netCDF4 as NC
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
from scipy.io import loadmat
from pyproj import Proj
from tdr_tc_centering_with_example import distance
import matplotlib.lines as mlines
from matplotlib.lines import Line2D

domain_width_value= 150
azimuth_line = [136,113,94,65,110,106,98,56,42,105] 

#All crystals
time1= 0
time1_full=0
time2= 70 #1400 seconds
time2_full=1400
time3= 155 #3100 (SB3)
time3_full=3100
time4= 279 #5580 (only SB2) 
time4_full=5580

#SB1
time5= 380 # 7600
time5_full=7600
time6= 750 # 15000
time6_full=15000
time7= 900 #18000
time7_full=18000

#SB2
time8= 350 #7000
time8_full=7000
time9= 425 #8500
time9_full=8500
time10=515 #10,300
time10_full=10300

marker_times_all = [time1, time2]
marker_times_sb1= [time5, time6, time7]
marker_times_sb2= [time4, time8, time9, time10]
marker_times_sb3=[time3]

# Function to calculate azimuth
def calculate_azimuth(x1, y1, x2_array, y2_array):
    dx = x2_array - x1
    dy = y2_array - y1
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    met_ang = (-angle_deg + 90) % 360
    for i in range(len(met_ang) - 1):
        met_ang[i+1] = np.where(met_ang[i] > 330 and met_ang[i+1] < 100 or met_ang[i] < 100 and met_ang[i+1] > 330, np.nan, met_ang[i+1])
    return met_ang

os.chdir('/rita/s0/scratch/nrb171/Harvey/') # Get Harvey data from Nick's folder
levels_dbz = np.arange(-15, 75, 3)
levels_W = np.arange(-5, 5.001, 0.5)

# Import ICTG and cropped data
os.chdir('/rita/s0/bsr5234/modelling/ICTG/forBruno/model2/')
fname = "ictg2_stormrel3DlogD_tenicfzrfzPSD_m1-10_28800s_x117y108z23_201708250400_newspreadtest.mat" # New run with riming
ictg = loadmat(fname)
part_mass = ictg['part_m']  # (x,y,z,size,time)

# Import shear data
os.chdir('/rita/s0/scratch/nrb171/harvey_postproc/9km/')
fname = "wrf_shear_nondivirrot02.mat"  # Nick's shear data
from scipy.io import loadmat
mat = loadmat(fname)
shear_u = np.array(mat['shru'][0, 1:])
shear_v = np.array(mat['shrv'][0, 1:])

# Interpolate the shear data to 5-min resolution
interp_to = np.arange(1, 361, 1)
interp_from = np.arange(0, 360, 12)
interp_shear_u = np.interp(interp_to, interp_from, shear_u)
interp_shear_v = np.interp(interp_to, interp_from, shear_v)

# Plot dbz at 4z
os.chdir('/rita/s0/scratch/nrb171/harvey_postproc/1km')
file= 'wrfout_d04_2017-08-25_04:00:00_dbz_z.nc'
level_index=15
data = NC.Dataset(file)
dbz=data.variables['dbz_interp'][level_index] 
levels=data.variables['level'][:]
formatted_level = int(levels[level_index])
lat = data.variables['XLAT']
lon = data.variables['XLONG']

xx, yy = np.meshgrid(np.arange(0,np.shape(dbz)[0]),np.arange(0,np.shape(dbz)[1]))
crop_adjx = 407
crop_adjy = 416

#Wind centers
os.chdir('/rita/s0/bsr5234/modelling/tilt_analysis/')
allstks = np.load('stored_centers_Harvey.npy')
ctrlons = allstks[:,0,0]
ctrlats = allstks[:,1,0]
wctrys = allstks[:,5,0]
wctrxs = allstks[:,6,0]
tstep = 120

radius_grid = distance(ctrlats[tstep], ctrlons[tstep], lat, lon)
x_axes = np.concatenate((-radius_grid[415,:407],radius_grid[415,407:]))
y_axes = np.concatenate((-radius_grid[:415,407],radius_grid[415:,407]))

#Get time values
for i in range(len(part_mass[:, 0, 0, 0, 0])): 
    for j in range(len(part_mass[0, :, 0, 0, 0])): 
        for k in range(len(part_mass[0, 0, :, 0, 0])): 
            for l in range(len(part_mass[0, 0, 0, :, 0])): 
                if not np.isnan(ictg['part_posx'][i, j, k, 0, 0]):
                     part_m_values = ictg['part_m'][i, j, k, l, :]
                     time_values = np.arange(len(part_m_values)) * 20


specific_particle_indices = [(26,16,5,0)
]

# Loop over each particle and generate plots
for particle_indices in specific_particle_indices:
    m, n, o, p = particle_indices
    fig = plt.figure(figsize=(16, 14), dpi=300)
    gs = gridspec.GridSpec(10, 8)

    # Create subplots
    ax1 = fig.add_subplot(gs[0:4,0:4]) # Reflectivity
    ax2 = fig.add_subplot(gs[0:2,4:8]) # Rime Mass
    ax3 = fig.add_subplot(gs[2:4,4:8]) # Particle Mass
    ax4 = fig.add_subplot(gs[4:6,0:4]) # Temperature
    ax5 = fig.add_subplot(gs[4:6,4:8]) # Aspect Ratio
    ax6 = fig.add_subplot(gs[6:8,0:4]) # Ice Supersaturation
    ax7 = fig.add_subplot(gs[6:8,4:8]) # Z
    ax8 = fig.add_subplot(gs[8:10,0:4]) # Radial
    ax9 = fig.add_subplot(gs[8:10,4:8]) # Azimuthal

    k=o
    azimuths_single = calculate_azimuth(0, 0, ictg['part_posx'][m,n,o,p,:], ictg['part_posy'][m,n,o,p,:])
    
   # Define the colors for the legend
    legend_colors = ['darkblue', 'darkred', 'darkgreen']  #Largest Mass Family; extrme/unique family
    crystal_indices = [(26, 16, 5,0), (26, 16, 5,1), (26, 16, 5,2)]  # Unique and Extreme Family

    # Generate the legend labels based on p+1
    legend_labels = [f'FC{p + 4}' for (m, n, o, p) in crystal_indices]  # Labels for the 3 crystals
    legend_entries = []  # Store final masses for the legend

    # Add context trajectories for the specified crystal indices
    for i, (m, n, o, p) in enumerate(crystal_indices):
        color = legend_colors[i]  # Get the corresponding color for the crystal index
        
        # Retrieve particle data for the specific crystal index (m, n, o, p)
        x = ictg['part_posx'][m, n, o, p, :]
        y = ictg['part_posy'][m, n, o, p, :]
        
        # Plot particle trajectories and properties in different colors
        linewidth = 2.25 if i < 2 else 1.25
        ax1.plot(x, y, linewidth=linewidth, color=color)
        ax1.plot(x[0], y[0], 'o', color='black', markersize=5)

        azimuths = calculate_azimuth(0, 0, ictg['part_posx'][m, n, o, p, :], ictg['part_posy'][m, n, o, p, :])

        linewidth = 1.5 if i < 2 else 1.00
        ax2.plot(time_values, np.cumsum(ictg['part_mrime'][m, n, o, p, :]), linewidth=linewidth, color=color)
        ax3.plot(time_values, ictg['part_m'][m, n, o, p, :], linewidth=linewidth, color=color)
        ax4.plot(time_values, ictg['record_T'][m, n, o, p, :] - 273.15, linewidth=linewidth, color=color)
        ax5.plot(time_values, ictg['part_c'][m, n, o, p, :] / ictg['part_a'][m, n, o, p, :], linewidth=linewidth, color=color)
        ax6.plot(time_values, (ictg['record_evapor'][m, n, o, p, :] / ictg['record_eice'][m, n, o, p, :]) - 1, linewidth=linewidth, color=color)
        ax7.plot(time_values, ictg['record_w'][m, n, o, p, :], linewidth=linewidth, color=color)
        
        
        ax8.plot(azimuths, ictg['part_posz'][m, n, o, p, :], linewidth=linewidth, color=color)
        ax8.plot(azimuths[0], ictg['part_posz'][m, n, o, p, 0], 'o', color='black', markersize=4)

        d = np.sqrt(ictg['part_posx'][m, n, o, p, :]**2 + ictg['part_posy'][m, n, o, p, :]**2)
        ax9.plot(d, ictg['part_posz'][m, n, o, p, :], linewidth=linewidth, color=color)
        ax9.plot(d[0], ictg['part_posz'][m, n, o, p, 0], 'o', color='black', markersize=4)

    # Create custom legend handles with the specified colors
    hardcoded_masses = {
    1: 4.13e-05,   # Size Bin 1 (p=4)
    2: 1.52e-04,  # Size Bin 2 (p=14)
    3: 3.33e-04   # Size Bin 3 (p=23)
    }

    legend_entries = []
    legend_colors = ['darkblue', 'darkred', 'darkgreen']  # Make sure this matches your plotting colors
    legend_handles = []

    # Add a grey line for the spatially adjacent crystals
    adjacent_color = 'grey'
    adjacent_handle = plt.Line2D([0], [0], color=adjacent_color, lw=2)
    legend_handles = [plt.Line2D([0], [0], color=legend_colors[i], lw=2) for i in range(3)] + [adjacent_handle]
    #legend_labels.append('Spatially Adjacent\nCrystals')  # Add the multi-line label
    ax1.legend(legend_handles, legend_labels, loc='lower left')

    ax1.scatter(0, 0, alpha=1, color='r', marker='+') #Add center

    # Add box showing the boundaries for the 45-degree arc
    azimuth_right = (azimuths_single[0] + 22.5) % 360
    azimuth_left = (azimuths_single[0] - 22.5) % 360
    plotymax, plotymin = 17, 5

    # convert km altitude to normed values
    def conv_to_norm(km_alt, plotymax, plotymin):
        normed_alt = float(km_alt - plotymin) / (float(plotymax)-5)
        return normed_alt

    vspanymax = conv_to_norm(ictg['part_posz'][m,n,o,p,0]+1, plotymax, plotymin)
    vspanymin = conv_to_norm(ictg['part_posz'][m,n,o,p,0]-1, plotymax, plotymin)
    ax8.set_ylim(plotymin, plotymax)

    crystal_id = f"{m}_{n}_{o}_{p}"
    PT=ax1.contourf(x_axes, y_axes, dbz, cmap=cm.gist_ncar, levels=levels_dbz, extend='both', alpha=0.10)

    cl=plt.colorbar(PT,ax=ax1)  
    domain_width=domain_width_value
    ax1.set_xlim(0- domain_width, 0+ domain_width)
    ax1.set_ylim(0 -domain_width,0 +domain_width)
    xtlabs=np.concatenate((np.arange(domain_width,0,-25),np.arange(0,domain_width+1,25)))
    ax1.set_xticks(np.arange(-domain_width,domain_width+1,25), xtlabs)
    ax1.set_yticks(np.arange(-domain_width,domain_width+1,25), xtlabs)

    cl.ax.tick_params()
    cl.ax.set_title('dbz')

    # Add title, axes, etc to plot
    for ax in [ax2,ax3, ax5]: # Set mass variables to use log scale
        ax.set_yscale('log')

    for ax in [ax2,ax3, ax4,ax5,ax6,ax7]:
        ax.set_xlim(left=0)

    ax1.set_xlabel("Distance from storm center (km)")
    ax1.set_ylabel("Distance from storm center (km)")
    ax1.text(0.02,0.88,"a)",transform=ax1.transAxes, fontsize=18)

    ax2.set_title('Particle Accumulated Rime Mass')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Mass (mg)')
    ax2.text(0.02,0.88,"e)",transform=ax2.transAxes, fontsize=18)
    ax2.grid()

    ax3.set_title('Particle Mass')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Mass (mg)')
    ax3.text(0.02,0.88,"f)",transform=ax3.transAxes, fontsize=18)
    ax3.grid()

    ax4.set_title('Temperature')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Temperature (C)')
    ax4.text(0.02,0.88,"b)",transform=ax4.transAxes, fontsize=18)
    ax4.invert_yaxis()
    ax4.grid()

    ax5.set_title('Particle Aspect Ratio')
    ax5.set_xlabel('Time (seconds)')
    ax5.set_ylabel('c/a')
    ax5.text(0.02,0.88,"g)",transform=ax5.transAxes, fontsize=18)
    ax5.grid()

    ax6.set_title('Ice Supersaturation')
    ax6.set_xlabel('Time (seconds)')
    ax6.text(0.02,0.88,"c)",transform=ax6.transAxes, fontsize=18)
    ax6.grid(True)

    ax7.set_title('Vertical Velocity')
    ax7.set_xlabel('Time (seconds)')
    ax7.set_ylabel('Vertical Velocity (m/s)')
    ax7.text(0.02,0.88,"h)",transform=ax7.transAxes, fontsize=18)
    ax7.grid()

    ax8.set_title('Azimuthal Position')
    ax8.set_xlabel('Azimuth (degrees)')
    ax8.set_ylabel('Altitude (km)')
    ax8.set_xlim(0,361)
    ax8.text(0.02,0.88,"d)",transform=ax8.transAxes, fontsize=18)
    ax8.grid()

    ax9.set_title('Radial Position')
    ax9.set_xlabel('Radius (km)')
    ax9.set_ylabel('Altitude (km)')
    ax9.set_xlim(0,right=None)
    ax9.text(0.02,0.88,"i)",transform=ax9.transAxes, fontsize=18)
    ax9.grid()

    #Highlight zero line
    ax5.axhline(y=1, color='black', linewidth=3)
    ax6.axhline(y=0, color='black', linewidth=3)
    ax7.axhline(y=0, color='black', linewidth=3)

    #Add quadrant markers and labels
    for x in [15, 105, 195, 285]:
        ax8.axvline(x=x, color='gray', linestyle='--', linewidth=3)
    
    # Add labels in the middle of each section
    label_positions = [40, 150, 240, 322.5]
    labels = ['DL', 'DR', 'UR', 'UL']

    ymin, ymax = ax8.get_ylim()
    y_pos = ymin + 0.75 * (ymax - ymin)  # Position labels near the top

    for x, label in zip(label_positions, labels):
        ax8.text(x, y_pos, label, ha='center', va='center', fontsize=10, color='black', alpha=0.8)

    #Add dashed line in reflectivity
    radius_1 = 38  # Can change
    radius_2= 50
    radius_3= 110
    radius_4=25
    radius_5=10

    fig.tight_layout()
    PTG=plt.gcf()

    # Save the figure
    plt.savefig('/rita/s0/mjv5638/plots/A_thesis/NEWTEST_all_marked_summaryplot_withoutmarkers_' + crystal_id,bbox_inches="tight")

    print(crystal_id)
    plt.close()
