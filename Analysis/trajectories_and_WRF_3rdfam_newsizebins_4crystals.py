import os
import numpy as np
import netCDF4 as NC
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
from scipy.io import loadmat
from pyproj import Proj
from matplotlib.lines import Line2D
import re
from matplotlib.colors import to_rgba
import matplotlib.lines as mlines

LIGHT_RED = '#FFA07A'  # Light red; sublimate in eye
GREY = '#808080'     # medium grey
BLACK=  '#000000' #Melting Level
LIGHT_BLUE= "#4682B4"    ## sublimation: if any d_single > 60 km

#Function to count revolutions
def calculate_azimuth_revolutions(x1, y1, x2_array, y2_array):
    dx = x2_array - x1
    dy = y2_array - y1
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    met_ang = (-angle_deg + 90) % 360  # Adjust the angle for correct orientation
    return met_ang

# Function to count revolutions based on azimuth
def count_revolutions(azimuths):
    rev_count = 0.0  # Initialize total revolution counter (float to track cumulative revolutions)
    revolution_values = []  # Store cumulative revolution values to track progress

    for i in range(1, len(azimuths)):
        # Calculate the difference between the current and previous azimuth
        if azimuths[i]> azimuths[i-1]:
            delta = (azimuths[i-1]-0)+(360-azimuths[i])
        else:
            delta = azimuths[i] - azimuths[i-1]
        # Calculate the revolution increment using the absolute difference in azimuth
        rev_increment = abs(delta) / 360.0 
        rev_count += rev_increment # Update the cumulative revolution count by adding the increment to the previous total
        revolution_values.append(rev_count)

    return revolution_values

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
legend_entries = []  # Store legend labels

# Import ICTG and cropped data
os.chdir('/rita/s0/bsr5234/modelling/ICTG/forBruno/model2/')
fname = "ictg2_stormrel3DlogD_tenicfzrfzPSD_m1-10_28800s_x195y156z43_201708250400_cyclesplit_np45.mat"  #45 crystals
ictg = loadmat(fname)
part_mrime_array = ictg['part_posx'][:] # (x,y,z,size,time)

match = re.search(r'np(\d+)\.mat', fname)
if match:
    num_crystals = int(match.group(1))
print(num_crystals)  # Output: 25

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

mass_min = 3.62e-5 * 0.95  # 5% below smallest mass
mass_max = 3.33e-4 * 1.05  # 5% above largest mass

for i in range(len(part_mrime_array[:, 0, 0, 0, 0])): 
    for j in range(len(part_mrime_array[0, :, 0, 0, 0])): 
        for k in range(len(part_mrime_array[0, 0, :, 0, 0])): 
            for l in range(len(part_mrime_array[0, 0, 0, :, 0])): 
                if not np.isnan(ictg['part_posx'][i, j, k, 0, 0]):
                     part_m_values = ictg['part_m'][i, j, k, l, :]
                     time_values = np.arange(len(part_m_values)) * 20

# Plot the trajectory for the particles
fig = plt.figure(figsize=(11, 14), dpi=300) 
gs = gridspec.GridSpec(16, 8)

# Create subplots
ax1 = fig.add_subplot(gs[0:4,0:4]) # Reflectivity 1
ax3 = fig.add_subplot(gs[4:6,0:4]) # Temperature 1
ax5 = fig.add_subplot(gs[6:8,0:4]) # Mass 1
ax7 = fig.add_subplot(gs[8:10,0:4]) # VV 1
ax9 = fig.add_subplot(gs[10:12,0:4]) # Azimuthal
ax11 = fig.add_subplot(gs[12:14,0:4]) # Radial


ax2=  fig.add_subplot(gs[0:4,4:8]) #Refletivity 2
ax4 = fig.add_subplot(gs[4:6,4:8]) # Temperature 2
ax6 = fig.add_subplot(gs[6:8,4:8]) # Mass 2
ax8 = fig.add_subplot(gs[8:10,4:8]) # VV1
ax10 = fig.add_subplot(gs[10:12,4:8]) # Azimithal
ax12 = fig.add_subplot(gs[12:14,4:8]) # Radial

# Create a colormap 
shape = part_mrime_array.shape  # [m, n, o, p, t]
num_crystals = shape[0] * shape[1] * shape[2] * shape[3]

# Manual color overrides
mass_color_map = {
    3.62e-5: 'darkblue', 1.34e-4: 'darkred', 3.33e-4: 'darkgreen'
}

colors = []
indices_to_plot = []  # keep track of which crystals to plot

special_bins = []
legend_lines_1 = []
legend_labels_1 = []

legend_lines_2 = []
legend_labels_2 = []

color_map_1 = {12: LIGHT_RED, 18: GREY}
color_map_2 = {21: LIGHT_RED, 23: 'darkgreen'}
combined_color_map = {**color_map_1, **color_map_2}

# Loop through each p value you want to plot
for p, color in combined_color_map.items():
    found = False  # flag to stop after first match

    for m in range(shape[0]):
        for n in range(shape[1]):
            for o in range(shape[2]):
                if np.isnan(ictg['part_posx'][m, n, o, 0, 0]):
                    continue

                # Check if this (m, n, o, p) is valid
                if not np.isnan(ictg['part_posx'][m, n, o, p, 0]):
                    found = True
                    break
            if found:
                break
        if found:
            break

    if not found:
        continue  # Skip if no valid entry found for this p

    # Pull out data
    x = ictg['part_posx'][m, n, o, p, :]
    y = ictg['part_posy'][m, n, o, p, :]
    z = ictg['part_posz'][m, n, o, p, :]
    T = ictg['record_T'][m, n, o, p, :] - 273.15
    mass = ictg['part_m'][m, n, o, p, :]
    w = ictg['record_w'][m, n, o, p, :]
    azimuths = calculate_azimuth(0, 0, x, y)
    d = np.sqrt(x**2 + y**2)
    linewidth = 2

    # Plot based on which panel this p belongs to
    if p in color_map_1:
        line1, = ax1.plot(x, y, linewidth=linewidth, color=color)
        ax1.plot(x[0], y[0], 'o', color='black', markersize=5)

        ax3.plot(time_values, T, linewidth=linewidth, color=color)
        ax5.plot(time_values, mass, linewidth=linewidth, color=color)
        ax7.plot(time_values, w, linewidth=linewidth, color=color)

        ax9.plot(azimuths, z, linewidth=linewidth, color=color)
        ax9.plot(azimuths[0], z[0], 'o', color='black', markersize=5)

        ax11.plot(d, z, linewidth=linewidth, color=color)
        ax11.plot(d[0], z[0], 'o', color='black', markersize=5)

    
        legend_lines_1.append(line1)
        legend_labels_1.append(f'Crystal {p}')

    elif p in color_map_2:
        line2, = ax2.plot(x, y, linewidth=linewidth, color=color)
        ax2.plot(x[0], y[0], 'o', color='black', markersize=5)
        
        ax4.plot(time_values, T, linewidth=linewidth, color=color)
        ax6.plot(time_values, mass, linewidth=linewidth, color=color)
        ax8.plot(time_values, w, linewidth=linewidth, color=color)

        ax10.plot(azimuths, z, linewidth=linewidth, color=color)
        ax10.plot(azimuths[0], z[0], 'o', color='black', markersize=5)

        ax12.plot(d, z, linewidth=linewidth, color=color)
        ax12.plot(d[0], z[0], 'o', color='black', markersize=5)

        legend_lines_2.append(line2)
        legend_labels_2.append(f'Crystal {p}')
        
# Create legend handles
legend_elements = []

# Add specific size bin legends if relevant values were plotted
if any(p == 4 for _, _, _, p in indices_to_plot):
    legend_elements.append(mlines.Line2D([], [], color='darkblue', label='Size Bin 1'))
if any(p == 13 for _, _, _, p in indices_to_plot):
    legend_elements.append(mlines.Line2D([], [], color='darkred', label='Size Bin 2'))
if any(p == 22 for _, _, _, p in indices_to_plot):
    legend_elements.append(mlines.Line2D([], [], color='darkgreen', label='Size Bin 3'))

# Add color-based condition legends
legend_elements.append(mlines.Line2D([], [], color= LIGHT_RED, label='Downshear Eye Sublimation'))
legend_elements.append(mlines.Line2D([], [], color= GREY , label='Upshear Eye Sublimation'))
legend_elements.append(mlines.Line2D([], [], color= BLACK, label='Melted Crystal'))

crystal_id = f"{m}_{n}_{o}_{p}"
k = o

# Plot dbz at 4z
os.chdir('/rita/s0/scratch/nrb171/harvey_postproc/1km')
file = 'wrfout_d04_2017-08-25_04:00:00_dbz_z.nc'
level_index = 15
data = NC.Dataset(file)
ideal = data.variables['dbz_interp'][level_index] 
levels = data.variables['level'][:]
formatted_level = int(levels[level_index])

xx, yy = np.meshgrid(np.arange(0, np.shape(ideal)[0]), np.arange(0, np.shape(ideal)[1]))
crop_adjx = 407
crop_adjy = 416

PT = ax1.contourf(xx-crop_adjx, yy-crop_adjx, ideal, cmap=cm.gist_ncar, levels=levels_dbz, extend='both', alpha=0.15)
ax1.scatter(0, 0, alpha=1, color='r', marker='+') # Add center

PT = ax2.contourf(xx-crop_adjx, yy-crop_adjx, ideal, cmap=cm.gist_ncar, levels=levels_dbz, extend='both', alpha=0.15)
ax2.scatter(0, 0, alpha=1, color='r', marker='+') # Add center

cl = plt.colorbar(PT, ax=ax1)  
cl.ax.set_title('dbz')
domain_width = 75
ax1.set_xlim(0 - domain_width, 0 + domain_width)
ax1.set_ylim(0 - domain_width, 0 + domain_width)

domain_width = 75
cb = plt.colorbar(PT, ax=ax2)  
cb.ax.set_title('dbz')
ax2.set_xlim(0 - domain_width, 0 + domain_width)
ax2.set_ylim(0 - domain_width, 0 + domain_width)

# Add title, axes, etc to plot
for ax in [ax3,ax4]:
    ax.set_yticks(np.arange(-60,0.1, 20))  # Ticks every 2 km

for ax in [ax5,ax6]: # Set mass variables to use log scale
    ax.set_yscale('log')

for ax in [ax5,ax6]:
    ax.set_ylim(0.0000001,0.01)
    ax.set_yticks([1e-7, 1e-5, 1e-3])

for ax in [ax7,ax8]:
    ax.set_ylim(-7,14.1)
    ax.set_yticks(np.arange(-4, 12.1, 4))  # Ticks every 2 km

for ax in [ax9,ax10, ax11,ax12]:
    ax.set_ylim(10,17.1)
    ax.set_yticks(np.arange(10, 17, 2))  # 10 to 16 km every 1 km

ax3.set_title('Temperature')
ax3.set_xlabel('Time (seconds)')
ax3.set_ylabel('Temperature (C)')
ax3.invert_yaxis()
ax3.set_ylim(bottom=0)
ax3.grid()

ax4.set_title('Temperature', pad=10)
ax4.set_xlabel('Time (seconds)')
ax4.invert_yaxis()
ax4.set_ylim(bottom=0)
ax4.grid()

ax5.set_title('Particle Mass')
ax5.set_xlabel('Time (seconds)')
ax5.set_ylabel('Mass (mg)')
ax5.grid()

ax6.set_title('Particle Mass')
ax6.set_xlabel('Time (seconds)')
ax6.grid()

ax7.set_title('Vertical Velocity')
ax7.set_xlabel('Time (seconds)')
ax7.set_ylabel('Altitude (km)')
ax7.grid()

ax8.set_title('Vertical Velocity')
ax8.set_xlabel('Time (seconds)')
ax8.grid()

ax9.set_title('Azimuthal Position')
ax9.set_xlabel('Azimuth (degrees)')
ax9.set_ylabel('Altitude (km)')
ax9.set_xlim(0, 361)
ax9.grid()

ax10.set_title('Azimuthal Position')
ax10.set_xlabel('Azimuth (degrees)')
ax10.set_xlim(0, 361)
ax10.grid()

ax11.set_title('Radial Position')
ax11.set_xlabel('Radius (km)')
ax11.set_ylabel('Altitude (km)')
ax11.set_xlim(0, right=None)
ax11.grid()

ax12.set_title('Radial Position')
ax12.set_xlabel('Radius (km)')
ax12.set_xlim(0, right=None)
ax12.grid()

ax1.text(0.02,0.90,"a)",transform=ax1.transAxes, fontsize=18)
ax2.text(0.02,0.90,"g)",transform=ax2.transAxes, fontsize=18)
ax3.text(0.02,0.75,"b)",transform=ax3.transAxes, fontsize=18)
ax4.text(0.02,0.75,"h)",transform=ax4.transAxes, fontsize=18)
ax5.text(0.02,0.75,"c)",transform=ax5.transAxes, fontsize=18)
ax6.text(0.02,0.75,"i)",transform=ax6.transAxes, fontsize=18)
ax7.text(0.02,0.75,"d)",transform=ax7.transAxes, fontsize=18)
ax8.text(0.02,0.75,"j)",transform=ax8.transAxes, fontsize=18)
ax9.text(0.02,0.75,"e)",transform=ax9.transAxes, fontsize=18)
ax10.text(0.02,0.75,"k)",transform=ax10.transAxes, fontsize=18)
ax11.text(0.02,0.75,"f)",transform=ax11.transAxes, fontsize=18)
ax12.text(0.02,0.75,"l)",transform=ax12.transAxes, fontsize=18)

ax7.axhline(y=0, color='black', linewidth=3)
ax8.axhline(y=0, color='black', linewidth=3)

#Add quadrant markers and labels
for x in [15, 105, 195, 285]:
    ax9.axvline(x=x, color='gray', linestyle='--', linewidth=3)
    ax10.axvline(x=x, color='gray', linestyle='--', linewidth=3)

#Add labels in the middle of each section
label_positions = [60, 150, 240, 322.5]
labels = ['DL', 'DR', 'UR', 'UL']

ymin, ymax = ax9.get_ylim()
y_pos = ymin + 0.75 * (ymax - ymin)  # Position labels near the top

for x, label in zip(label_positions, labels):
    ax9.text(x, y_pos, label, ha='center', va='center', fontsize=10, color='black', alpha=0.8)
    ax10.text(x, y_pos, label, ha='center', va='center', fontsize=10, color='black', alpha=0.8)

legend_lines_1 = [
    mlines.Line2D([], [], color=LIGHT_RED, linewidth=2),    # Choose correct color
    mlines.Line2D([], [], color=GREY, linewidth=2)
]
legend_labels_1 = ['Crystal 14', 'Crystal 15']

legend_lines_2 = [
    mlines.Line2D([], [], color=LIGHT_RED, linewidth=2),  # Adjust colors to match your plot
    mlines.Line2D([], [], color='darkgreen', linewidth=2)
]
legend_labels_2 = ['Crystal 16', 'FC6']

ax1.legend(legend_lines_1, legend_labels_1, loc='best')
ax2.legend(legend_lines_2, legend_labels_2, loc='best')

azimuth_lines = [93, 91]  # Two azimuth values
radius = 150  # Adjustable

# List of axes to match each azimuth
axes = [ax1, ax2]

for azimuth_line, ax in zip(azimuth_lines, axes):
    # Convert azimuth to radians (rotate clockwise from north)
    azimuth_radians = np.radians(90 - azimuth_line)

    # Compute x, y coordinates for the azimuth line
    x_azimuth = radius * np.cos(azimuth_radians)
    y_azimuth = radius * np.sin(azimuth_radians)

    # Plot on the corresponding axis
    ax.plot([0, x_azimuth], [0, y_azimuth], linestyle='--', color='black', linewidth=1)

fig.subplots_adjust(
    top=0.96, bottom=0.05, left=0.06, right=0.96,
    hspace=5, wspace=1.0  
)
PTG = plt.gcf()

# Save the figure
os.chdir('/rita/s0/mjv5638/plots/ICTG_and_WRF/More_Sizes/45')
plt.savefig(f'summary_plot_family3_newsizebins_4crystals', bbox_inches="tight")
print("done")
plt.close()
