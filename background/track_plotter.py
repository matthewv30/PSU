import os
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeat

# LOAD WIND CENTER TRACK (from .npy file)
os.chdir('/rita/s0/bsr5234/modelling/tilt_analysis/')
allstks = np.load('stored_centers_Harvey.npy')   # Array of storm center positions

# Extract longitude and latitude of storm center track (1 km wind centers)
ctrlons = allstks[:,0,0]   # longitudes
ctrlats = allstks[:,1,0]   # latitudes
print(np.shape(ctrlats))   # diagnostic check


# LOAD PRESSURE CENTER DATA (from Nick’s .mat file)
os.chdir('/rita/s0/scratch/nrb171/harvey_postproc/1km')
fname = "centersslp5_04.mat"  # File with pressure center data

from scipy.io import loadmat
mat = loadmat(fname)
print(mat.keys())   # Show available variables inside .mat file

# Extract relevant arrays
clat = np.array(mat['clat'][0, :])   # center latitudes
clon = np.array(mat['clon'][0, :])   # center longitudes
ctrx = np.array(mat['cx1'][0, :])    # x-coordinates (grid-based)
ctry = np.array(mat['cy1'][0, :])    # y-coordinates (grid-based)
print(len(clat))                     # number of time steps


# CREATE TIME ARRAY
basetime = dt.datetime(year=2017, month=8, day=24, hour=18, minute=5, second=0)

times = []
for tind, tincrement in enumerate(range(0, 360, 1)):  # loop over 360 five-minute intervals
    truetime = basetime + dt.timedelta(minutes=5) * int(tincrement)
    print(truetime, tind)   
    times.append(truetime)
times = np.array(times)     # convert list to NumPy array


# PLOTTING
# Create figure with fixed size and resolution
fig = plt.figure(figsize=(6,6), dpi=200)

# Add a Cartopy map with PlateCarree projection
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.add_feature(cfeat.COASTLINE, alpha=0.6)   # add coastlines
ax.add_feature(cfeat.BORDERS, alpha=0.6)     # add country borders
ax.gridlines(draw_labels=True)               # add labeled gridlines
# ax.set_extent([93,98,22,29])               # (optional) zoom in to a specific domain

# PLOT STORM CENTER TRACK
ax.plot(ctrlons[:], ctrlats[:], 'k', label='1 km wind center track')

# Optionally plot one highlighted point
# ax.scatter(ctrlons[119], ctrlats[119], s=30, marker='o', color='r', zorder=5)
# ax.text(ctrlons[119] + 0.05, ctrlats[119] + 0.05, dt.datetime.strftime(times[119], "%d %H%M"), color='r')

# Plot every 36th point (≈ every 3 hours) along the track
ax.scatter(ctrlons[35::36], ctrlats[35::36], s=30, marker='o', color='k', zorder=5)

# Label those points with date/time strings
timestrings = [dt.datetime.strftime(tstp, "%d %H%M") for tstp in times[35::36]]
for tsidx, ts in enumerate(timestrings):
    ax.text(ctrlons[35::36][tsidx] + 0.03,   
            ctrlats[35::36][tsidx] + 0.03,  
            ts)                              # label with day/hour-minute


# TITLES AND LABELS
ax.set_title('Harvey Storm Center Track', fontsize=14)
fig.text(0.5, 0.15, 'Longitude', ha='center', va='center', fontsize=12)
fig.text(0.02, 0.5, 'Latitude', ha='center', va='center', rotation='vertical', fontsize=12)
plt.legend()

# SAVE AND SHOW FIGURE
os.chdir('/rita/s0/mjv5638/mesoscale/')
plt.savefig('harvey_track.png', dpi=200)   # save figure
plt.show()