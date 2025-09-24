import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
import glob
import netCDF4 as NC
import datetime as dt
import numpy.ma as ma
from scipy.io import loadmat
from tdr_tc_centering_with_example import distance, get_bearing
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties

# Define start and end azimuths and trajectories
azimuth_step = 1
trajectory_indices = [(26,16,5,0)] #Unique and Extreme Family
radmin, radmax= 25, 35 #Range for one or more rev

time1= 380 #7600 seconds
start_azimuth = 110
end_azimuth = 110
time2=750 #16,000 seconds
azimuth_2=106
time3= 900 #18,000 seconds
azimuth_3=98

# Constants
e0 = 611  # Reference vapor pressure at T0 (Pa)
T0 = 273.15  # Reference temperature (K)
Lv= 2.5e6  # Latent heat of vaporization (J/kg)
Lf= 3.33e5  # Latent heat of fusion (J/kg)
Rv = 461.5  # Specific gas constant for water vapor (J/(kg·K))


def calculate_end_point(start_point, distance, azimuth):
    azimuth_rad = np.radians(azimuth)
    x0, y0 = start_point
    end_x = x0 + distance * np.cos(azimuth_rad)
    end_y = y0 + distance * np.sin(azimuth_rad)
    return round(end_x), round(end_y)

def get_line_coordinates(start_point, end_point):
    x0, y0 = start_point
    x1, y1 = end_point
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))
    x_increment = dx / steps
    y_increment = dy / steps
    line = [(x0 + i * x_increment, y0 + i * y_increment) for i in range(steps + 1)]
    return line

def Bresenham_line(start_point, end_point):
    """
    Bresenham's Line Algorithm to calculate the points between (x1, y1) and (x2, y2).

    Parameters:
    - x1, y1: Coordinates of the starting point.
    - x2, y2: Coordinates of the ending point.

    Returns:
    - line_points: List of tuples containing the coordinates of points along the line.

    For more information see: https://en.m.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    """

    line_points = []

    x1, y1 = start_point
    x2, y2 = end_point

    # Calculate differences and absolute differences
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    # Initial decision parameter
    if dx > dy:
        err = dx / 2
    else:
        err = -dy / 2

    while True:
        line_points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = err
        if e2 > -dx:
            err -= dy
            x1 += sx
        if e2 < dy:
            err += dx
            y1 += sy
    return line_points

def calculate_cross_section_values(line_coordinates, grid_values):
    values = []
    for point in line_coordinates:
        x, y = point
        values.append(grid_values[int(x), int(y)])
    return values

def calculate_cross_section(latlon_center, idx_center, field_variable, azimuth, max_distance=200):
    '''
    Calculates a vertical cross section at a specified azimuth.

    Parameters:
    - field variable: any of the variables from the get_field() function, e.g. reflectivity.
    - azimuth: in degrees, meteorological azimuth.

    Returns:
    - 2-D vertical cross section values
    - The radial distance axes for the cross section. 
        Note the vertical axis can be accessed using the get_levels() function in the TCR_toolkit class.
    '''

    # max_distance=200    #value larger than the number of grid points a cross section would span
    start_point = idx_center    #

    #Calculate the line of the cross-section based on the storm center and azimuth
    end_point = calculate_end_point(start_point, max_distance, azimuth)

    #Calculate the grid points that are closest to the line using the Bresenham Line Algorithm
    line_coordinates = Bresenham_line(start_point, end_point)

    from tdr_tc_centering_with_example import distance
    
    #calculate the radius of each gridbox
    lat_coordinates = [int(x[0]) for x in line_coordinates]
    lon_coordinates = [int(x[1]) for x in line_coordinates]
    distances_along_cross_section = distance(latitude[lat_coordinates[0], lon_coordinates[0]], longitude[lat_coordinates[0], lon_coordinates[0]], latitude[lat_coordinates, lon_coordinates], longitude[lat_coordinates, lon_coordinates])

    crosssection=[]
    for level in range(len(field_variable[:,0,0])):
        crosssection_layer = calculate_cross_section_values(line_coordinates, field_variable[level,:,:])
        crosssection.append(crosssection_layer)

    return crosssection, distances_along_cross_section

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

def radial_mean(center, datavariable, modellevel):
    altitude = zlevels[modellevel]
    #print(altitude)

    azimuthgrid = get_bearing(center[0],center[1],latitude,longitude)
    ang = (-np.rad2deg(azimuthgrid) + 90)%360
    # ang = ang.astype(np.int32)        #use to bin by 1 degree azimuths
 
    base = 5 #Binning
    ang = (base * np.around(ang/base, decimals=0)).astype(np.int32)
 
    # y, x = np.indices((datavariable.shape)) #gives indexes for the data array
    # radiusgrid = np.sqrt((x - center[0])**2 + (y - center[1])**2) #calculates at what radius each grid box is at
    radiusgrid = distance(center[0],center[1],latitude,longitude)
    radiusgrid = radiusgrid.astype(np.int32) #rounds radii to nearest grid box
    radiusmin, radiusmax = radmin, radmax 
    for i in range(np.shape(datavariable)[0]):
        for j in range(np.shape(datavariable)[1]):
            if radiusgrid[i][j] > radiusmax or radiusgrid[i][j] < radiusmin: #use for a donut/ring
                datavariable[i][j] = np.nan


    DR=datavariable.ravel() #flattens the array in to one long 1D array
    RR=ang.ravel() # Uses the azimuth grid to sort
    # print(type(DR[0]), type(RR[0]))
    keep = ~np.isnan(DR) #using this as an index allows to only look at existing data values
    rbin = np.bincount(RR[keep]) #gives the amount of grid boxes with data at each radius
    tbin = np.bincount(RR[keep],DR[keep]) #creates a sum of all the existing data values at each radius
    azimean=tbin/rbin #takes the summed data values and divides them by the amount of grid boxes that have data at each radius

    return azimean, altitude, radiusmin, radiusmax

# Wind centers
os.chdir('/rita/s0/bsr5234/modelling/tilt_analysis/')
allstks = np.load('stored_centers_Harvey.npy')

ctrlons = allstks[:, 0, 0]
ctrlats = allstks[:, 1, 0]
wctrys = allstks[:, 5, 0]
wctrxs = allstks[:, 6, 0]

# Set the time for 0400Z
ctrlon_0400Z = ctrlons[120]
ctrlat_0400Z = ctrlats[120]
ctry_0400Z = wctrys[120]
ctrx_0400Z = wctrxs[120]

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
tstep = 120

radius_grid = distance(ctrlats[tstep], ctrlons[tstep], lat, lon)
x_axes = np.concatenate((-radius_grid[415,:407],radius_grid[415,407:]))
y_axes = np.concatenate((-radius_grid[:415,407],radius_grid[415:,407]))
levels_dbz = np.arange(-15, 75, 3)

os.chdir('/rita/s0/bsr5234/modelling/ICTG/forBruno/model2/')
fname = "ictg2_stormrel3DlogD_tenicfzrfzPSD_m1-10_28800s_x117y108z23_201708250400_newspreadtest.mat"  # New run with riming
ictg = loadmat(fname)

# Change dir to folder with 1km runs and all the U/V wind vbl files
os.chdir('/rita/s0/bsr5234/modelling/ICTG/modeldatafiles/wrf_vbl_files/0400Z/')
#os.chdir('/rita/s0/scratch/nrb171/harvey_postproc/1km')

# Load u and v wind data files outside the function
uwind_file = 'wrfout_d04_2017-08-25_04:00:00_ua_z.nc'
vwind_file = 'wrfout_d04_2017-08-25_04:00:00_va_z.nc'
uwind_data = NC.Dataset(uwind_file)
vwind_data = NC.Dataset(vwind_file)

uwind = uwind_data.variables['ua_interp'][:]
vwind = vwind_data.variables['va_interp'][:]

wwind_file = 'wrfout_d04_2017-08-25_04:00:00_wa_z.nc'
dbz_vbl_file = 'wrfout_d04_2017-08-25_04:00:00_REFL_10CM_z.nc'

wwind_data = NC.Dataset(wwind_file)
wwind = wwind_data.variables['wa_interp'][:]  # W wind | units: m s-1 | shape = (39, 831, 831) = (level, south_north, west_east)

longitude = wwind_data.variables['XLONG'][:]
latitude = wwind_data.variables['XLAT'][:]
zlevels = wwind_data.variables['level']

#Ice Superstaturation Calculation 
#Get temperature
temp_file= 'wrfout_d04_2017-08-25_04:00:00_tk_z.nc'
temp_data= NC.Dataset(temp_file)
temp = temp_data.variables['temp_interp'][:] #In Kelvin

longitude = temp_data.variables['XLONG'][:]
latitude = temp_data.variables['XLAT'][:]
zlevels = temp_data.variables['level']

#Convert to saturation vapor pressure (Using Clausius Clapeyron)
es = e0 * np.exp((Lv / Rv) * (1 / T0 - 1 / temp)) # Saturation vapor pressure in Pascals
ei = e0 * np.exp(((Lv+Lf) / Rv) * (1 / T0 - 1 / temp)) # Saturation vapor pressure in Pascals

#Get q vapor
qvapor_file= 'wrfout_d04_2017-08-25_04:00:00_QVAPOR_z.nc'
qvapor_data= NC.Dataset(qvapor_file)
qvapor = qvapor_data.variables['QVAPOR_interp'][:]

#Get Pressure
p_file = 'wrfout_d04_2017-08-25_04:00:00_pres_z.nc'  # Assuming a pressure file
p_data = NC.Dataset(p_file)
pressure = p_data.variables['pressure_interp'][:]  # Pressure in Pascals

epsilon = 0.622  # Ratio of gas constants for dry air and water vapor
vapor_pressure = (qvapor * pressure) / (epsilon + qvapor) #Vapor pressure in Pa
supersaturation_ice = (vapor_pressure / ei) - 1

# W wind Calculation
altkmlist = []
allew_ud = []
radiusmins = []  # To store radiusmin values
radiusmaxs = []  # To store radiusmax values

# Plot Radial Velocity
LLctr = (ctrlat_0400Z, ctrlon_0400Z)
index_center = (ctry_0400Z, ctrx_0400Z)

ws = np.sqrt(uwind**2 + vwind**2)  # Calculate wind speed
azimuthgrid = get_bearing(LLctr[0], LLctr[1], latitude, longitude)
ang = (-np.rad2deg(azimuthgrid) + 90) % 360

radn = np.arctan2(vwind, uwind)
wdir = (-np.rad2deg(radn) + 90) % 360
angle_diff = np.deg2rad(ang - wdir)
radial_wind = ws * np.cos(angle_diff)
tangential_wind = ws * np.sin(angle_diff)

# Create a 2x2 subplot layout, Loop through the azimuths
for Xsec_azimuth in range(start_azimuth, end_azimuth + 1, azimuth_step):
    print(Xsec_azimuth)
    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    plt.subplots_adjust(wspace=0.28, hspace=0.35)

    # Flatten axs to handle as a 1D array
    axs = axs.flatten()
    colors = ['darkblue'] #Unique and Extreme Family
    legend_labels = ['FC4'] 
    
    #Xsec_azimuth = azimuth_step  
    Wcrosssection_1, Xsec_dists_1 = calculate_cross_section(LLctr, index_center, radial_wind, start_azimuth)
    tanwindcrosssection_1, Xsec_dists_4 = calculate_cross_section(LLctr, index_center, tangential_wind, start_azimuth)
    Wcrosssection_2, Xsec_dists_2 = calculate_cross_section(LLctr, index_center, radial_wind, azimuth_2)
    tanwindcrosssection_2, Xsec_dists_5 = calculate_cross_section(LLctr, index_center, tangential_wind, azimuth_2)
    Wcrosssection_3, Xsec_dists_3 = calculate_cross_section(LLctr, index_center, radial_wind, azimuth_3)
    tanwindcrosssection_3, Xsec_dists_6 = calculate_cross_section(LLctr, index_center, tangential_wind, azimuth_3)
    rv_range = np.arange(-20,20.0001,1)
    
    # Reflectivity Plan View (Top Left Panel - ax0)
    PT = axs[0].contourf(x_axes, y_axes, dbz, cmap=cm.gist_ncar, levels=levels_dbz, extend='both', alpha=0.15)
    axs[0].scatter(0, 0, alpha=1, color='r', marker='+')  # Add center
    cl = plt.colorbar(PT, ax=axs[0]) 

    domain_width = 125
    axs[0].set_xlim(-domain_width, domain_width)
    axs[0].set_ylim(-domain_width, domain_width)
    xtlabs = np.concatenate((np.arange(domain_width, 0, -25), np.arange(0, domain_width+1, 25)))
    axs[0].set_xticks(np.arange(-domain_width, domain_width+1, 25), xtlabs)
    axs[0].set_yticks(np.arange(-domain_width, domain_width+1, 25), xtlabs)

    cl.ax.tick_params()
    cl.ax.set_title('dbz')
    axs[0].set_xlabel("Distance from storm center (km)")
    axs[0].set_ylabel("Distance from storm center (km)")
    axs[0].text(0.02, 0.92, "a)", transform=axs[0].transAxes, fontsize=18)
    axs[0].set_title("Reflectivity", fontsize=10)

    # Add legend for crystal sizes
    for color, label in zip(colors, legend_labels):
        axs[0].plot([], [], color=color, label=label, linewidth=2)  
    axs[0].legend(loc='lower left', fontsize=10, frameon=True)

    # Add Trajectories to the Reflectivity Plot (Top Left - ax1)
    for idx, color, label in zip(trajectory_indices, colors, legend_labels):
        i, j, k, s = idx  # Unpack the indices for clarity
        k_adj = k 

        # Extract the trajectory x and y positions for the specified indices
        x = ictg['part_posx'][i, j, k_adj, s, :]
        y = ictg['part_posy'][i, j, k_adj, s, :]

        # Plot the trajectory on ax1
        axs[0].plot(x, y,color=color, label=label,linewidth=1.5)  # Plot trajectory with specified line style
        axs[0].plot(x[0], y[0], marker="o", markersize=7, color='black')
        axs[0].plot(x[time1], y[time1], marker="X", markersize=10, color=color, markeredgecolor='white', markeredgewidth=1)
        axs[0].plot(x[time2], y[time2], marker="X", markersize=10, color=color, markeredgecolor='white', markeredgewidth=1)
        axs[0].plot(x[time3], y[time3], marker="X", markersize=10, color=color, markeredgecolor='white', markeredgewidth=1)

    # Azimuthal Position Plot- Loop through the specific trajectory indices for the Azimuthal Position Plot
    for idx, color, label in zip(trajectory_indices, colors, legend_labels):
        i, j, k, s = idx 
        azimuths = calculate_azimuth(0, 0, ictg['part_posx'][i, j, k, s, :], ictg['part_posy'][i, j, k, s, :])  # Calculate azimuths for the particle trajectory
        d = np.sqrt(ictg['part_posx'][i, j, k, s, :]**2 + ictg['part_posy'][i, j, k, s, :]**2)

        axs[1].plot(d[time1], ictg['part_posz'][i, j, k, s, time1], marker="X", markersize=10, markeredgecolor='white', markeredgewidth=1, color=color)
        axs[2].plot(d[time2], ictg['part_posz'][i, j, k, s, time2], marker="X", markersize=10, markeredgecolor='white', markeredgewidth=1, color=color)
        axs[3].plot(d[time3], ictg['part_posz'][i, j, k, s, time3], marker="X", markersize=10, markeredgecolor='white', markeredgewidth=1, color=color)
        

    axs[1].set_title(f"Radial Cross Section- Radial Velocity\n and Tangential Velocity at {Xsec_azimuth} Degrees", fontsize=10)
    axs[1].set_xlabel('Radial distance from center (km)')
    axs[1].set_ylabel('Altitude (km)')
    axs[1].text(0.02, 0.92, "b)", transform=axs[1].transAxes, fontsize=18)

    axs[2].set_title(f"Radial Cross Section- Radial Velocity\n and Tangential Velocity at {azimuth_2} Degrees", fontsize=10)
    axs[2].set_xlabel('Radial distance from center (km)')
    axs[2].set_ylabel('Altitude (km)')
    axs[2].text(0.02, 0.92, "c)", transform=axs[2].transAxes, fontsize=18)

    axs[3].set_title(f"Radial Cross Section- Radial Velocity\n and Tangential Velocity at {azimuth_3} Degrees", fontsize=10)
    axs[3].set_xlabel('Radial distance from center (km)')
    axs[3].set_ylabel('Altitude (km)')
    axs[3].text(0.02, 0.92, "d)", transform=axs[3].transAxes, fontsize=18)

    # Vertical Velocity Cross Section 1
    Wcfill = axs[1].contourf(Xsec_dists_1, zlevels[:] / 1000, Wcrosssection_1, 
                            cmap=cm.seismic, levels=rv_range, extend='both', alpha=0.9)
    cbar = plt.colorbar(Wcfill, ax=axs[1])
    cbar.ax.set_title('m/s')
    tangential=axs[1].contour(Xsec_dists_4, zlevels[:] / 1000, tanwindcrosssection_1, linewidths=0.7,
                            levels = np.arange(-10,30.1,5), colors='black', extend='both')
    # Define y-locations where you want to label contours at x=30
    y_positions = np.linspace(np.min(zlevels) / 1000, np.max(zlevels) / 1000, 20)
    manual_positions = [(40, y) for y in y_positions]  # Force labels at x=30

    # Add labels manually at those positions
    labels= axs[1].clabel(
        tangential,
        inline=False,
        inline_spacing=0,
        fontsize=6,
        fmt='%d',
        manual=manual_positions
    )
    
    for txt in labels:
        txt.set_rotation(0)
        x, y = txt.get_position()
        txt.set_position((x, y + 0.2))  # Adjust the vertical offset as needed


     # Vertical Velocity Cross Section 2
    tangential=axs[2].contour(Xsec_dists_5, zlevels[:] / 1000, tanwindcrosssection_2, linewidths=0.7,
                            levels = np.arange(-10,30.1,5), colors='black', extend='both')
    # Define y-locations where you want to label contours at x=30
    y_positions = np.linspace(np.min(zlevels) / 1000, np.max(zlevels) / 1000, 10)
    manual_positions = [(100, y) for y in y_positions]  # Force labels at x=30

    # Add labels manually at those positions

    labels= axs[2].clabel(
        tangential,
        inline=False,
        inline_spacing=0,
        fontsize=6,
        fmt='%d',
        manual=manual_positions
    )
    
    for txt in labels:
        txt.set_rotation(0)
        x, y = txt.get_position()
        txt.set_position((x, y + 0.2))  # Adjust the vertical offset as needed

    Wcfill = axs[2].contourf(Xsec_dists_2, zlevels[:] / 1000, Wcrosssection_2, 
                            cmap=cm.seismic, levels=rv_range, extend='both', alpha=0.9)
    cbar = plt.colorbar(Wcfill, ax=axs[2])
    cbar.ax.set_title('m/s')

     # Vertical Velocity Cross Section 3
    Wcfill = axs[3].contourf(Xsec_dists_3, zlevels[:] / 1000, Wcrosssection_3, 
                            cmap=cm.seismic, levels=rv_range, extend='both', alpha=0.9)
    cbar = plt.colorbar(Wcfill, ax=axs[3])
    cbar.ax.set_title('m/s')

    tangential=axs[3].contour(Xsec_dists_6, zlevels[:] / 1000, tanwindcrosssection_3, linewidths=0.7,
                            levels = np.arange(-10,30.1,5), colors='black', extend='both')
    # Define y-locations where you want to label contours at x=30
    y_positions = np.linspace(np.min(zlevels) / 1000, np.max(zlevels) / 1000, 10)
    manual_positions = [(100, y) for y in y_positions]  # Force labels at x=30

    # Add labels manually at those positions
    labels= axs[3].clabel(
        tangential,
        inline=False,
        inline_spacing=0,
        fontsize=6,
        fmt='%d',
        manual=manual_positions
    )
    
    for txt in labels:
        txt.set_rotation(0)
        x, y = txt.get_position()
        txt.set_position((x, y + 0.2))  # Adjust the vertical offset as needed

    #Plotting changes
    axs[1].set_xlim(35,45)
    axs[2].set_xlim(80,105)
    axs[3].set_xlim(95,105)

    #1st line
    azimuth_line = start_azimuth  #Azimuth value from ax3
    radius = 150  # Can change

    azimuth_radians = np.radians(90 - azimuth_line)  # Adjust the azimuth angle for clockwise rotation
    # Calculate the x, y coordinates for the line at this azimuth (from the center to the edge of the plot)
    x_azimuth_1 = radius * np.cos(azimuth_radians)
    y_azimuth_1 = radius * np.sin(azimuth_radians)

    axs[0].plot([0, x_azimuth_1], [0, y_azimuth_1], linestyle='--', color='black', linewidth=1)
    axs[0].plot([0, x_azimuth_1], [0, y_azimuth_1], linestyle='--', color='black', linewidth=1)

    #2nd line
    azimuth_line_2 = azimuth_2 #Azimuth value from ax3
    radius = 150  # Can change
    azimuth_radians_2 = np.radians(90 - azimuth_line_2)  # Adjust the azimuth angle for clockwise rotation
    # Calculate the x, y coordinates for the line at this azimuth (from the center to the edge of the plot)
    x_azimuth_2 = radius * np.cos(azimuth_radians_2)
    y_azimuth_2 = radius * np.sin(azimuth_radians_2)

    axs[0].plot([0, x_azimuth_2], [0, y_azimuth_2], linestyle='--', color='black', linewidth=1)
    axs[0].plot([0, x_azimuth_2], [0, y_azimuth_2], linestyle='--', color='black', linewidth=1)
    
     #3rd line
    azimuth_line_3 = azimuth_3 #Azimuth value from ax3
    radius = 150  # Can change

    azimuth_radians_3 = np.radians(90 - azimuth_line_3)  # Adjust the azimuth angle for clockwise rotation
    
    # Calculate the x, y coordinates for the line at this azimuth (from the center to the edge of the plot)
    x_azimuth_3 = radius * np.cos(azimuth_radians_3)
    y_azimuth_3 = radius * np.sin(azimuth_radians_3)

    # Plot the dashed line at the specified azimuth angle in ax1
    axs[0].plot([0, x_azimuth_3], [0, y_azimuth_3], linestyle='--', color='black', linewidth=1)
    axs[0].plot([0, x_azimuth_3], [0, y_azimuth_3], linestyle='--', color='black', linewidth=1)

    axs[1].set_ylim(5, 18)
    axs[2].set_ylim(5, 18)
    axs[3].set_ylim(5, 18)

    # Save plots
    first_trajectory = trajectory_indices[0]
    file_name = '/rita/s0/mjv5638/plots/cross_sections/26_16_5_0/v3/TEST_cross_section_{:03d}_trajectory_{}_{}_{}_{}.png'.format(
        Xsec_azimuth, *first_trajectory
    )
    plt.savefig(file_name, bbox_inches="tight", dpi=300)
