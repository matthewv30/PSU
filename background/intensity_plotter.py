## Adopted from Bruno Rojas

# Import required libraries
import os
import numpy as np
import netCDF4 as NC
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import datetime as dt
import matplotlib.dates as mdates

def getime(filename): 
    """        
    Extracts datetime information from a WRF output filename.
    
    Assumes the filename format is YYYY-MM-DD_HH:MM... 
    Returns a Python datetime object.
    """
    filestrparts = filename.split('_')

    datestring = filestrparts[0]
    timestring = filestrparts[1]
    yr = int(datestring[0:4])
    mo = int(datestring[5:7])
    dy = int(datestring[8:10])
    hr = int(timestring[0:2])
    mn = int(timestring[3:5])
    
    fdatetime = dt.datetime(year=yr, month=mo, day=dy, hour=hr, minute=mn)
    return fdatetime


# Change directory to where results are stored
os.chdir('/rita/s0/bsr5234/modelling')

# If processed intensity data already exists, load it
if os.path.isfile('windspeed_Harvey.npy') == True:
    max_ws = np.load('windspeed_Harvey.npy', allow_pickle=True)    # Maximum windspeed
    wrftimes = np.load('datetimes_Harvey.npy', allow_pickle=True)  # Datetime stamps
    min_mslp = np.load('minpressure_Harvey.npy', allow_pickle=True) # Minimum sea-level pressure

else:
    min_mslp = []
    times = []
    max_ws = []

    
    os.chdir('/rita/s0/scratch/nrb171/Harvey/') # Change to directory containing raw WRF files
    for findex, file in enumerate(sorted(glob.glob("wrfout*"))): # Loop through WRF output files in chronological order

        # Extract datetime from filename
        wrfdt = getime(file[11:])
        data = NC.Dataset(file) # Open NetCDF dataset

        # Read variables
        lat = data.variables['XLAT'][0]
        lon = data.variables['XLONG'][0]
        sfc_pressure = data.variables['PSFC'][0]   # Surface pressure (Pa)
        terrain_height = data.variables['HGT'][0]  # Terrain height (m)
        temp2m = data.variables['T2'][0]           # 2-m temperature (K)

        # 10-m winds
        U10m = data.variables['U10'][0]
        V10m = data.variables['V10'][0]
        windspeed = np.sqrt(U10m**2 + V10m**2)     # Wind speed magnitude

        print(wrfdt, np.max(windspeed))            # Diagnostic printout

        # Convert surface pressure to mean sea-level pressure (hPa)
        g = 9.81
        Rgas = 287
        MSLP = sfc_pressure * np.exp((g * terrain_height) / (temp2m * Rgas))
        MSLP_mb = MSLP / 100

        # Save intensity metrics
        min_mslp.append(np.min(MSLP_mb))          # Minimum pressure
        max_ws.append(np.max(windspeed))          # Maximum wind speed
        times.append(wrfdt)                       # Corresponding time
    
    # Convert lists to arrays
    min_mslp = np.array(min_mslp)
    wrftimes = np.array(times)
    max_ws = np.array(max_ws)
    
    # Save processed data for future use
    os.chdir('/rita/s0/bsr5234/modelling')
    np.save('windspeed_Harvey.npy', max_ws)
    np.save('datetimes_Harvey.npy', wrftimes)
    np.save('minpressure_Harvey.npy', min_mslp)


# Smooth the max windspeed time series with a moving average
max_ws_smth = []
averaging_window = 12      # Number of time steps to average (12 = 1 hr if timestep = 5 min)
avglen_label = averaging_window * 5
averaging_sidelength = averaging_window / 2

for i in range(len(max_ws)):
    if i > averaging_sidelength and i < (len(max_ws) - averaging_sidelength):
        sma = np.sum(max_ws[int(i - averaging_sidelength):int(i + averaging_sidelength)]) / averaging_window
        max_ws_smth.append(sma)
    else:
        max_ws_smth.append(np.nan)

max_ws_smth = np.array(max_ws_smth)
max_ws_smth_kts = max_ws_smth * 1.944 # Convert smoothed wind speed from m/s to knots

# Define Rapid Intensification (RI) thresholds
# Values are (wind speed increase in knots, time window in hours)
RI_criteria1 = (20, 12)
RI_criteria2 = (25, 24)
RI_criteria3 = (30, 24)

def RI_identifier(RI_criteria):
    """
    Identify rapid intensification events given a threshold.
    RI_criteria = (threshold in knots, time window in hours).
    Returns an array with RI events flagged.
    """
    RI_threshold = RI_criteria[0]       # Speed increase threshold
    RI_window = RI_criteria[1]          # Time window in hours
    RI_check = []
    RI_window_tsteps = (RI_window * 60) / 5  # Convert hours to model time steps (5-min interval)
    
    for i in range(len(max_ws)):
        if i < (len(max_ws) - RI_window_tsteps):
            deltaV = max_ws_smth_kts[int(i + RI_window_tsteps)] - max_ws_smth_kts[int(i)]
            if deltaV >= RI_threshold:
                RI_check.append(1)      # Mark RI event
            else:
                RI_check.append(np.nan)
        else:
            RI_check.append(np.nan)

    return np.array(RI_check)


# ---- PLOTTING ----
fig, ax1 = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
ax2 = ax1.twinx()   # Second y-axis for wind speed

# Plot minimum pressure
ax1.plot(wrftimes, min_mslp, 'b', label='Minimum MSLP')
ax1.set_ylim(950, 990)
ax1.set_ylabel("Minimum MSLP (hPa)")
ax1.set_xlabel("Date/Time (UTC)")
ax1.grid(axis='x', color='black', linestyle='-', linewidth=0.3)
ax1.grid(axis='y', color='green', linestyle='--', linewidth=0.5)
ax1.legend(loc='lower right')

# Plot max wind speed
ax2.plot(wrftimes, max_ws, 'r', label='Maximum Wind Speed', alpha=0.2)
ax2.plot(wrftimes, max_ws_smth, 'r', label='Maximum Wind Speed (%i-min avg)' % avglen_label)
ax2.set_ylim(0, 65)
ax2.set_ylabel("Maximum Wind Speed (m/s)")
ax2.legend(loc='upper right')

plt.title("Harvey Model Intensity")

# Format x-axis as day-hour
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H'))

# Save plot
os.chdir('/rita/s0/mjv5638/plots/manuscript/new_figures')
plt.savefig('harvey_intensity.png', bbox_inches="tight", dpi=200)