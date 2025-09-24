import os
import netCDF4 as NC
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
import matplotlib.gridspec as gridspec

colors = ['darkgreen'] #Unique and Extreme Family
legend_labels = ['FC6'] 
start_azimuth = 90
azimuth_line = [start_azimuth] 

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
    end_point = calculate_end_point(start_point, max_distance, azimuth)
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

    azimuthgrid = get_bearing(center[0],center[1],latitude,longitude)
    ang = (-np.rad2deg(azimuthgrid) + 90)%360
 
    base = 5 #Binning
    ang = (base * np.around(ang/base, decimals=0)).astype(np.int32)
 
    # y, x = np.indices((datavariable.shape)) #gives indexes for the data array
    # radiusgrid = np.sqrt((x - center[0])**2 + (y - center[1])**2) #calculates at what radius each grid box is at
    radiusgrid = distance(center[0],center[1],latitude,longitude)
    radiusgrid = radiusgrid.astype(np.int32) #rounds radii to nearest grid box
    radiusmin, radiusmax = 25, 40
    for i in range(np.shape(datavariable)[0]):
        for j in range(np.shape(datavariable)[1]):
            if radiusgrid[i][j] > radiusmax or radiusgrid[i][j] < radiusmin: #use for a donut/ring
                datavariable[i][j] = np.nan


    DR=datavariable.ravel() #flattens the array in to one long 1D array
    RR=ang.ravel() # Uses the azimuth grid to sort
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

# W wind Calculation
altkmlist_w = []
allew_ud_w = []
radiusmins_w = []  # To store radiusmin values
radiusmaxs_w = []  # To store radiusmax values

# Loop through model levels and perform calculations
for idx, modellevel in enumerate(range(2, 36 * 2, 2 * 2)):
    os.chdir('/rita/s0/scratch/nrb171/harvey_postproc/1km')
    LLctr = (ctrlat_0400Z, ctrlon_0400Z)

    #W Wind
    wwind_copy = np.copy(tangential_wind)
    ew_mean_ud_w, altitude_w, radiusmin_w, radiusmax_w = radial_mean(LLctr, wwind_copy[modellevel, :, :], modellevel)
    keep_w = ~np.isnan(ew_mean_ud_w)  # Using this as an index to only look at existing data values
    altitude_km_w = altitude_w / 1000.  # Convert altitude to kilometers
    altkmlist_w.append(altitude_km_w)
    allew_ud_w.append(ew_mean_ud_w[keep_w])
    radiusmins_w.append(radiusmin_w)
    radiusmaxs_w.append(radiusmax_w)

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

idx = (26,16,5,2)
i, j, k, s = idx  # Unpack the indices for clarity
    
os.chdir('/rita/s0/bsr5234/modelling/ICTG/forBruno/model2/')
fname = "ictg2_stormrel3DlogD_tenicfzrfzPSD_m1-10_28800s_x117y108z23_201708250400_newspreadtest.mat"  # New run with riming
ictg = loadmat(fname)

os.chdir('/rita/s0/bsr5234/modelling/ICTG/modeldatafiles/wrf_vbl_files/0400Z/')

#w data
wwind_file = 'wrfout_d04_2017-08-25_04:00:00_wa_z.nc'
wwind_data = NC.Dataset(wwind_file)
wwind = wwind_data.variables['wa_interp'][:]  # W wind | units: m s-1 | shape = (39, 831, 831) = (level, south_north, west_east)

#v data
vwind_file = 'wrfout_d04_2017-08-25_04:00:00_va_z.nc'
vwind_data = NC.Dataset(vwind_file)
vwind = vwind_data.variables['va_interp'][:]  # V wind | units: m s-1 | shape = (39, 831, 831) = (level, south_north, west_east)

# Extract the trajectory x and y positions for the specified indices
x_position= ictg['part_posx'][i, j, k, s, :]
y_position = ictg['part_posy'][i, j, k, s, :]

yoffset=412 #Center location in x coordinates
xdist= np.arange(29,31,1) # In km, distance from center
tan_range=np.arange(-15,30.1,1)

for x in xdist:
    fig, axs = plt.subplots(2,2, figsize=(10,9))
    plt.subplots_adjust(wspace=0.28, hspace=0.35)

    axs = axs.flatten()
   
    #Calculate Cross Section
    vvcrosssection_1, Xsec_dists_1 = calculate_cross_section(LLctr, index_center, wwind, start_azimuth)
    W_cbrange = np.arange(-10,10.0001,1)

    Wcfill = axs[1].contourf(Xsec_dists_1, zlevels[:] / 1000, vvcrosssection_1, 
                            cmap=cm.seismic, levels=W_cbrange , extend='both', alpha=0.9)
    cbar = plt.colorbar(Wcfill, ax=axs[1])
    cbar.ax.set_title('m/s')

    # Reflectivity 
    PT = axs[0].contourf(x_axes, y_axes, dbz, cmap=cm.gist_ncar, levels=levels_dbz, extend='both', alpha=0.15)
    axs[0].scatter(0, 0, alpha=1, color='r', marker='+')  # Add center
    cl = plt.colorbar(PT, ax=axs[0]) 
    axs[0].plot(x_position, y_position,color='white',linewidth=2.5)
    axs[0].plot(x_position, y_position,color='darkgreen',linewidth=1.5)  # Plot trajectory with specified line style
    axs[0].plot(x_position[0], y_position[0], marker="o", markersize=7, color='black')
    axs[0].axvline(x, color='black', linestyle='--', linewidth=1)

    domain_width = 40
    axs[0].set_xlim(-domain_width, domain_width)
    axs[0].set_ylim(-domain_width, domain_width)
    xtlabs = np.concatenate((np.arange(domain_width, 0, -10), np.arange(0, domain_width+1, 10)))
    axs[0].set_xticks(np.arange(-domain_width, domain_width+1, 10), xtlabs)
    axs[0].set_yticks(np.arange(-domain_width, domain_width+1, 10), xtlabs)

    cl.ax.tick_params()
    cl.ax.set_title('dbz')
    axs[0].set_xlabel("Distance from storm center (km)")
    axs[0].set_ylabel("Distance from storm center (km)")
    axs[0].text(0.02, 0.92, "a)", transform=axs[0].transAxes, fontsize=18)
    axs[0].set_title("Reflectivity", fontsize=10)

    # W component
    w= axs[3].contourf(np.arange(360,460),zlevels[:]/1000, wwind[:,360:460, yoffset+x],levels = np.arange(-6,6.0001,0.1),cmap=cm.seismic, extend='both')
    idx = (26,16,5,2)
    i, j, k, s = idx  # Unpack the indices for clarity 
    
    axs[3].plot(ictg['part_posy'][i, j, k, s, :]+yoffset, ictg['part_posz'][i, j, k, s, :], linewidth=2.5, color='white')
    axs[3].plot(ictg['part_posy'][i, j, k, s, :]+yoffset, ictg['part_posz'][i, j, k, s, :], linewidth=1.5, color='darkgreen')
    axs[3].plot(ictg['part_posy'][i, j, k, s, 0]+yoffset, ictg['part_posz'][i, j, k, s, 0], marker="o", markersize=7, color='black')
    axs[3].set_ylim(10,17)

    #Edit
    axs[3].set_xlim(yoffset-30,yoffset+30)
    tick_labels = np.arange(-30, 31, 10)  # Tick labels
    tick_positions = tick_labels + yoffset  # Adjust positions by adding yoffset

    axs[3].set_title("South to North Cross Section- \nVertical Velocity", fontsize=10)
    axs[3].set_xlabel("South to North Distance from Center (km)")
    axs[3].set_ylabel('Altitude (km)')
    axs[3].text(0.02, 0.92, "d)", transform=axs[3].transAxes, fontsize=18)
    cbar = plt.colorbar(w, ax=axs[3], ticks=np.arange(-6, 7, 1))  # Set ticks
    cbar.ax.set_title('m/s')

    # Set the ticks and tick labels
    axs[3].set_xticks(tick_positions)  # Set tick positions
    axs[3].set_xticklabels([f"{int(tick)}" for tick in tick_labels])  # Set tick labels

    azimuths = calculate_azimuth(0, 0, ictg['part_posx'][i, j, k, s, :], ictg['part_posy'][i, j, k, s, :])  # Calculate azimuths for the particle trajectory
    d = np.sqrt(ictg['part_posx'][i, j, k, s, :]**2 + ictg['part_posy'][i, j, k, s, :]**2)

    axs[2].plot(azimuths, ictg['part_posz'][i, j, k, s, :], linewidth=2.5, color='white')
    axs[2].plot(azimuths, ictg['part_posz'][i, j, k, s, :], linewidth=1.5, color='darkgreen')
    axs[2].plot(azimuths[0], ictg['part_posz'][i, j, k, s, 0], marker="o", markersize=7, color='black')

    axs[1].plot(d, ictg['part_posz'][i, j, k, s, :], linewidth=1.5, color='white')
    axs[1].plot(d, ictg['part_posz'][i, j, k, s, :], linewidth=1.5, color='darkgreen')
    axs[1].plot(d[0], ictg['part_posz'][i, j, k, s, 0], marker="o", markersize=7, color='black')

    axs[2].set_ylim(10,17)
    axs[2].set_title(f"Azimuthal Cross Section-\nTangential Velocity", fontsize=10)
    axs[2].set_xlabel("Azimuth (degrees)")
    axs[2].set_ylabel('Altitude (km)')
    axs[2].set_xlim(60, 141)
    axs[2].set_xticks(np.arange(60,141,10))
    axs[2].text(0.02, 0.92, "c)", transform=axs[2].transAxes, fontsize=18)

    axs[1].set_title(f"Radial Cross Section-\nVertical Velocity at {start_azimuth} Degrees", fontsize=10)
    axs[1].set_xlabel('Radial distance from center (km)')
    axs[1].set_ylabel('Altitude (km)')
    axs[1].text(0.02, 0.92, "b)", transform=axs[1].transAxes, fontsize=18)
    axs[1].set_xlim(25,40)
    axs[1].set_ylim(10,17)

    azimuthalbins=np.arange(0,361,1)
    w= axs[2].contourf(azimuthalbins[keep_w], altkmlist_w, allew_ud_w, levels = tan_range, cmap=cm.jet, extend='both') #Vertical Velocity
    axs[2].contour(azimuthalbins[keep_w], altkmlist_w, allew_ud_w, levels=[0], colors='gray', linewidths=1.5)
    cbar = plt.colorbar(w, ax=axs[2])
    cbar.ax.set_title('m/s')

     # Add legend for crystal sizes
    for color, label in zip(colors, legend_labels):
        axs[0].plot([], [], color=color, label=label, linewidth=2)  
    axs[0].legend(loc='lower left', fontsize=10, frameon=True)

    radius = 200  # Can change
    for angle in azimuth_line:
        azimuth_radians = np.radians(90 - angle)  # Adjust for clockwise rotation
        x_azimuth_1 = radius * np.cos(azimuth_radians)
        y_azimuth_1 = radius * np.sin(azimuth_radians)
        axs[0].plot([0, x_azimuth_1], [0, y_azimuth_1], linestyle='--', color='black', linewidth=1)

    plt.savefig('/rita/s0/mjv5638/plots/A_thesis/unique_and_extreme_family/TEST_four_panel_distance_' + str(x)+ '.png', dpi=300, bbox_inches="tight")
    plt.clf()