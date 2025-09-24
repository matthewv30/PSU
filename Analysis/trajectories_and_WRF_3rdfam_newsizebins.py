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
from matplotlib.gridspec import GridSpec

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

# Set up figure
fig = plt.figure(figsize=(26, 10))

# === ax1: square ===
ax1_left = 0.05
ax1_size = 0.6  # square (width = height)
ax1_bottom = 0.2

ax1 = fig.add_axes([ax1_left, ax1_bottom, ax1_size, ax1_size])
ax1.set_aspect('equal')

# === Right-side layout settings ===
right_gap_h = 0.035  # space between right-side axes
right_gap_v = 0.09   # vertical gap between top and bottom rows
n_cols = 3

# Reduce horizontal gap between ax1 and others
right_start_x = ax1_left + ax1_size

# Calculate width and height of right-side axes
total_width = ax1_size
right_width = (total_width - 2 * right_gap_h) / n_cols
right_height = (ax1_size - right_gap_v) / 2

# Positions for top and bottom rows
top_row_y = ax1_bottom + right_height + right_gap_v
bottom_row_y = ax1_bottom

# Top row axes
ax4 = fig.add_axes([right_start_x + 0 * (right_width + right_gap_h), top_row_y, right_width, right_height])
ax3 = fig.add_axes([right_start_x + 1 * (right_width + right_gap_h), top_row_y, right_width, right_height])
ax6 = fig.add_axes([right_start_x + 2 * (right_width + right_gap_h), top_row_y, right_width, right_height])

# Bottom row axes
ax7 = fig.add_axes([right_start_x + 0 * (right_width + right_gap_h), bottom_row_y, right_width, right_height])
ax8 = fig.add_axes([right_start_x + 1 * (right_width + right_gap_h), bottom_row_y, right_width, right_height])
ax9 = fig.add_axes([right_start_x + 2 * (right_width + right_gap_h), bottom_row_y, right_width, right_height])

# Create a colormap 
# colors = plt.cm.viridis(np.linspace(0, 1, num_crystals))
shape = part_mrime_array.shape  # [m, n, o, p, t]
num_crystals = shape[0] * shape[1] * shape[2] * shape[3]

LIGHT_RED = '#FFA07A'  # Light red; sublimate in eye
GREY = '#808080'     # medium grey
BLACK=  '#000000' #Melting Level
LIGHT_BLUE= "#4682B4"    ## sublimation: if any d_single > 60 km

# Manual color overrides
mass_color_map = {
    3.62e-5: 'darkblue',1.34e-4: 'darkred', 3.33e-4: 'darkgreen'
}

colors = []
indices_to_plot = []  # keep track of which crystals to plot
special_bins = []

for m in range(shape[0]):
    for n in range(shape[1]):
        for o in range(shape[2]):
            for p in range(shape[3]):
                init_mass = ictg['part_m'][m, n, o, p, 0]

                # Skip if initial mass outside the buffered range
                if 4<= p <= 13 or 15 <= p <= 23: #10-50
                    if p in [4, 13, 22]:
                        special_bins.append((m, n, o, p))  # Save for later
                        continue  # Skip plotting for now

                    # Compute d_single
                    x = ictg['part_posx'][m, n, o, p, :]
                    y = ictg['part_posy'][m, n, o, p, :]
                    d_single = np.sqrt(x**2 + y**2)

                    valid_idx = np.where(~np.isnan(d_single))[0]

                    if valid_idx.size == 0:
                        colors.append( '#000000')  #Black
                    else:
                        last_idx = valid_idx[-1]
                        last_val = d_single[last_idx]
                        ends_with_nan = (last_idx + 1 < len(d_single)) and np.isnan(d_single[last_idx + 1])
                        
                        if p==18 or p==20:
                            colors.append(GREY) #grey
                        elif last_val <= 20 and ends_with_nan:
                            colors.append('#FFA07A')  # salmon
        
                        elif np.any(d_single > 60):
                            colors.append("#4682B4")  # light blue
                        else:
                            colors.append(GREY)  # black

                    indices_to_plot.append((m, n, o, p))

for m, n, o, p in special_bins:
    if p == 4:
        colors.append('darkblue')
    elif p == 13:
        colors.append('darkred')
    elif p == 22:
        colors.append('darkgreen')

    indices_to_plot.append((m, n, o, p))

# Plot the specified crystals with the gradient of colors
for idx, (m, n, o, p) in enumerate(indices_to_plot):
    if not np.isnan(ictg['part_posx'][m,n,o,0,0]):
        x = ictg['part_posx'][m,n,o,p, :] 
        y = ictg['part_posy'][m,n,o,p, :] 
        initial_mass = ictg['part_m'][m, n, o, p, 0] 

        linewidth = 1

        ax1.plot(x, y, linewidth=linewidth, color=colors[idx], label=f'Crystal {idx}')
        ax1.plot(x[0], y[0], 'o', color='black', markersize=5)
        ax3.plot(time_values, ictg['part_m'][m,n,o,p,:], linewidth=linewidth, color=colors[idx])
        ax4.plot(time_values, ictg['record_T'][m,n,o,p,:]-273.15, linewidth=linewidth, color=colors[idx])
        ax6.plot(time_values, (ictg['record_evapor'][m,n,o,p,:]/ictg['record_eice'][m,n,o,p,:])-1, linewidth=linewidth, color=colors[idx])
        ax7.plot(time_values, ictg['record_w'][m, n, o, p, :], linewidth=linewidth, color=colors[idx])
        
        azimuths_single = calculate_azimuth(0, 0, ictg['part_posx'][m,n,o,p,:], ictg['part_posy'][m,n,o,p,:])
        ax8.plot(azimuths_single, ictg['part_posz'][m,n,o,p,:], linewidth=linewidth, color=colors[idx])
        ax8.plot(azimuths_single[0], ictg['part_posz'][m,n,o,p,0], 'o', color='black', markersize=5)
        
        d_single = np.sqrt(ictg['part_posx'][m,n,o,p,:]**2 + ictg['part_posy'][m,n,o,p,:]**2)
        ax9.plot(d_single, ictg['part_posz'][m,n,o,p,:], linewidth=linewidth, color=colors[idx])
        ax9.plot(d_single[0], ictg['part_posz'][m,n,o,p,0], 'o', color='black', markersize=5)

        #legend_entries.append(f'Crystal {p+1}, Initial Mass: {initial_mass:.2e} mg')
        print(f'Crystal {p}, Initial Mass: {initial_mass:.2e} mg')


# Create legend handles
legend_elements = []

# Add specific size bin legends if relevant values were plotted
if any(p == 4 for _, _, _, p in indices_to_plot):
    legend_elements.append(mlines.Line2D([], [], color='darkblue', label='FC4'))
if any(p == 13 for _, _, _, p in indices_to_plot):
    legend_elements.append(mlines.Line2D([], [], color='darkred', label='FC5'))
if any(p == 22 for _, _, _, p in indices_to_plot):
    legend_elements.append(mlines.Line2D([], [], color='darkgreen', label='FC6'))

# Add color-based condition legends
legend_elements.append(mlines.Line2D([], [], color= LIGHT_RED, label='Downshear Eye Sublimation'))
legend_elements.append(mlines.Line2D([], [], color= GREY , label='Large Azimuthal Distance'))
legend_elements.append(mlines.Line2D([], [], color= LIGHT_BLUE , label='Radially Ejected'))

ax1.legend(handles=legend_elements, loc='lower left')

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

cl = plt.colorbar(PT, ax=ax1)  
domain_width = 150
ax1.set_xlim(0 - domain_width, 0 + domain_width)
ax1.set_ylim(0 - domain_width, 0 + domain_width)
cl.ax.set_title('dbz')

for ax in [ax3]: # Set mass variables to use log scale
    ax.set_yscale('log')

ax3.set_title('Particle Mass')
ax3.set_xlabel('Time (seconds)')
ax3.set_ylabel('Mass (mg)')
ax3.grid()

ax4.set_title('Temperature')
ax4.set_xlabel('Time (seconds)')
ax4.set_ylabel('Temperature (C)')
ax4.invert_yaxis()
ax4.grid()

ax6.set_title('Ice Supersaturation')
ax6.set_xlabel('Time (seconds)')
ax6.grid()

ax7.set_title('Vertical Velocity')
ax7.set_xlabel('Time (seconds)')
ax7.set_ylabel('Altitude (km)')
ax7.grid()

ax8.set_title('Azimuthal Position')
ax8.set_xlabel('Azimuth (degrees)')
ax8.set_ylabel('Altitude (km)')
ax8.set_xlim(0, 361)
ax8.grid()

ax9.set_title('Radial Position')
ax9.set_xlabel('Radius (km)')
ax9.set_ylabel('Altitude (km)')
ax9.set_xlim(0, right=None)
ax9.grid()

ax1.text(0.02,0.88,"a)",transform=ax1.transAxes, fontsize=18)
ax3.text(0.02,0.88,"c)",transform=ax3.transAxes, fontsize=18)
ax4.text(0.02,0.88,"b)",transform=ax4.transAxes, fontsize=18)
ax6.text(0.02,0.88,"d)",transform=ax6.transAxes, fontsize=18)
ax7.text(0.02,0.88,"e)",transform=ax7.transAxes, fontsize=18)
ax8.text(0.02,0.88,"f)",transform=ax8.transAxes, fontsize=18)
ax9.text(0.02,0.88,"g)",transform=ax9.transAxes, fontsize=18)

ax6.axhline(y=0, color='black', linewidth=3)
ax7.axhline(y=0, color='black', linewidth=3)

#Add quadrant markers and labels
for x in [15, 105, 195, 285]:
    ax8.axvline(x=x, color='gray', linestyle='--', linewidth=3)

# Add labels in the middle of each section
label_positions = [60, 150, 240, 322.5]
labels = ['DL', 'DR', 'UR', 'UL']

ymin, ymax = ax8.get_ylim()
y_pos = ymin + 0.75 * (ymax - ymin)  # Position labels near the top

for x, label in zip(label_positions, labels):
    ax8.text(x, y_pos, label, ha='center', va='center', fontsize=10, color='black', alpha=0.8)

fig.tight_layout()
PTG = plt.gcf()

# Save the figure
os.chdir('/rita/s0/mjv5638/plots/ICTG_and_WRF/More_Sizes/45')
plt.savefig(f'newlayout_summary_plot_family3_1050_trajectories_and_WRF', bbox_inches="tight")
print("done")
plt.close()
