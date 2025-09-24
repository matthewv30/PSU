## Combination of initial_distributions.py (modified for azimuthal distribution at initialization) and 
# azimuthal_EW_updraft.py which plots radial mean updrafts of the eyewall, these two will be overlaid.
# Additionally, this will plot an additional 2 panels for the initial radial and altitudinal distributions

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import netCDF4 as NC
import datetime as dt
from scipy.io import loadmat
from tdr_tc_centering_with_example import distance, get_bearing

def getime(filename): 
    """        
    Grabs the time of the analysis based on the file name

    Returns a datetime variable
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

### IMPORT SHEAR ###
os.chdir('/rita/s0/scratch/nrb171/harvey_postproc/9km/')
shearfname = "wrf_shear_nondivirrot02.mat" # Nick's shear data
mat = loadmat(shearfname)
shear_dir = np.array(mat['shdi'][0,:])
shD = shear_dir[11]     # 19 is for 12Z on 25th, 11 is for 04Z

ups = (shD + 180) % 360  # Upshear direction
dws = shD                # Downshear direction
rts = (shD + 90) % 360   # Right-of-shear 
lfs = (shD - 90) % 360   # Left-of-shear

UR = [rts, ups, "UpShear Right", 'UR']
DR = [dws, rts, "DownShear Right", 'DR']
DL = [lfs, dws, "DownShear Left", 'DL']
UL = [ups, lfs, "UpShear Left", 'UL']
SQ = [UR, DR, DL, UL]    # All shear quadrants

### GRAB THE RADAR UNDERLAY FOR THE PLAN VIEW ###
os.chdir('/rita/s0/scratch/nrb171/Harvey/')
file = 'wrfout_d04_2017-08-25_04:00:00'
data = NC.Dataset(file)
reflectivity = data.variables['REFL_10CM'][0,25,:,:]  # Crystal level (~6.5 km index 25)
levels_dbz = np.arange(-15,75,3)

print(np.shape(reflectivity))
xx, yy = np.meshgrid(np.arange(0,np.shape(reflectivity)[0]), np.arange(0,np.shape(reflectivity)[1]))

altitude_indices = [23, 27, 31, 35, 39, 45]
alt_km = [6, 7, 8, 9, 10, 11.5]

### LOAD PARTICLE DATA ###
os.chdir('/rita/s0/bsr5234/modelling/ICTG/forBruno/model2/')
fname = "ictg2_stormrel3DlogD_tenicfzrfzPSD_m1-10_28800s_x117y108z23_201708250400_newspreadtest.mat"

filename_elements = fname.split('_')
cornerloc = filename_elements[5]
filedt = filename_elements[6]
fyear = int(filedt[0:4])
fmonth = int(filedt[4:6])
fday = int(filedt[6:8])
fhour = int(filedt[8:10])
fminute = int(filedt[10:12])
fdt = dt.datetime(year=fyear, month=fmonth, day=fday, hour=fhour, minute=fminute)
print(fdt)
mat = loadmat(fname)

### LIST OF ALL VARIABLES AVAILABLE ###
init_indx = mat['init_indx'][0]
init_indy = mat['init_indy'][0]
init_indz = mat['init_indz'][0]

part_V = mat['part_V']
part_a = mat['part_a']
part_c = mat['part_c']
part_m = mat['part_m']
part_mrime = mat['part_mrime']

# Storm relative position 
part_posx = mat['part_posx']
part_posy = mat['part_posy']
part_posz = mat['part_posz']

part_re = mat['part_re']
part_rho = mat['part_rho']
part_rhorime = mat['part_rhorime']
part_vel = mat['part_vel']
part_y = mat['part_y']
record_T = mat['record_T']
record_eice = mat['record_eice']
record_evapor = mat['record_evapor']
record_p = mat['record_p']
record_u = mat['record_u']
record_v = mat['record_v']
record_w = mat['record_w']
tot_trajtime = mat['tot_trajtime']
total_t = mat['total_t']
xtick_time = np.arange(0, mat['total_t'][0,0]+1, mat['delt'][0,0])
xaxistimemax = 3600 * 4

maximum_initial_particle_radial_dist = 40

### LOOP THROUGH PARTICLE INITIALIZATIONS ###
size_idx = 2
allmass_allz = []
allazi = []
allhgt = []
allrad = []
allpx = []
allpy = []
allmass = []

for k in range(len(init_indz)):
    for i in range(len(init_indx)):
        for j in range(len(init_indy)):
            if np.isnan(part_posx[i,j,k,size_idx,0]) == False:  # survived initialization
                particle_radialpos = np.sqrt(part_posx[i,j,k,size_idx,:]**2 + part_posy[i,j,k,size_idx,:]**2)
                if particle_radialpos[0] <= maximum_initial_particle_radial_dist:
                    particle_azimuthpos = np.arctan2(part_posy[i,j,k,size_idx,:], part_posx[i,j,k,size_idx,:]) * 180 / np.pi
                    particle_azimuthpos = ((-particle_azimuthpos) + 90) % 360

                    azimuth_delta = np.diff(particle_azimuthpos)
                    for t in range(len(particle_azimuthpos[1:])):
                        particle_azimuthpos[t+1] = np.where(np.abs(azimuth_delta[t]) > 180, np.nan, particle_azimuthpos[t+1])

                    allazi.append(particle_azimuthpos[:])
                    allhgt.append(part_posz[i,j,k,size_idx,:])
                    allrad.append(particle_radialpos[:])
                    allpx.append(part_posx[i,j,k,size_idx,:])
                    allpy.append(part_posy[i,j,k,size_idx,:])
                    allmass.append(part_m[i,j,k,size_idx,:])

allmass = np.array(allmass)
allhgt = np.array(allhgt)
allrad = np.array(allrad)

crop_adjx = 176
crop_adjy = 176
adjx = 240
adjy = 247
ts = 0

allazi = np.array(allazi)
mass_distribution = allazi[:, ts]
nonlinbins = np.logspace(-8, 3, 40)
azibinsize = 10
azibins = np.arange(0, 361, azibinsize)

### PLOT INITIAL ALTITUDE DISTRIBUTION ###
fig, axs = plt.subplots(1, 1, figsize=(6, 4.75))
axs.set_xlabel('Frequency (number of initialization locations)')
axs.set_xticks(np.arange(0,301,50))
axs.set_xlim(0,300)
axs.set_ylim(4,16)
axs.set_title("Crystal Initialization Locations by Altitude")
axs.set_ylabel('Altitude (km)')
axs.hist(allhgt[:,ts], bins=np.arange(4,14,0.25), orientation='horizontal')

### WIND CENTERS ###
os.chdir('/rita/s0/bsr5234/modelling/tilt_analysis/')
allstks = np.load('stored_centers_Harvey.npy')

ctrlons = allstks[:,0,0]
ctrlats = allstks[:,1,0]
wctrys = allstks[:,5,0]
wctrxs = allstks[:,6,0]

### INTERPOLATED WIND FILES ###
os.chdir('/rita/s0/scratch/nrb171/harvey_postproc/1km')
interpolated_vbl_file = 'wrfout_d04_2017-08-25_04:00:00_wa_z.nc'

### FUNCTION TO CALCULATE AZIMUTHAL AVERAGE ###
def radial_mean(center, wfile, modellevel):
    wwind_data = NC.Dataset(wfile)
    zlevels = wwind_data.variables['level']
    altitude = zlevels[modellevel]
    print(altitude)

    wwind = wwind_data.variables['wa_interp'][modellevel,:,:]
    longitude = wwind_data.variables['XLONG'][:]
    latitude = wwind_data.variables['XLAT'][:]

    azimuthgrid = get_bearing(center[0], center[1], latitude, longitude)
    ang = (-np.rad2deg(azimuthgrid) + 90) % 360

    base = 10
    ang = (base * np.around(ang/base, decimals=0)).astype(np.int32)

    datavariable = wwind
    radiusgrid = distance(center[0], center[1], latitude, longitude)
    radiusgrid = radiusgrid.astype(np.int32)
    rmin, rmax = 15, 40
    for i in range(np.shape(datavariable)[0]):
        for j in range(np.shape(datavariable)[1]):
            if radiusgrid[i][j] > rmax or radiusgrid[i][j] < rmin:
                datavariable[i][j] = np.nan

    DR = datavariable.ravel()
    RR = ang.ravel()
    keep = ~np.isnan(DR)
    rbin = np.bincount(RR[keep])
    tbin = np.bincount(RR[keep], DR[keep])
    azimean = tbin / rbin

    return azimean, altitude

timeindex = 120
ctr = (ctrlats[timeindex], ctrlons[timeindex])  # WIND CENTERS (lat, lon)

ascending_cmap = ['#9b5fe0', '#16a4d8', '#60dbe8', '#8bd346', '#efdf48', '#f9a52c', '#d64e12'][1::2]

for idx, modellevel in enumerate(range(16,28,4)):
    os.chdir('/rita/s0/scratch/nrb171/harvey_postproc/1km')
    ew_mean_ud, altitude = radial_mean(ctr, interpolated_vbl_file, modellevel)
    keep = ~np.isnan(ew_mean_ud)
    altitude_km = altitude/1000.

    radialbins = np.arange(0,501,1)
    azimuthalbins = np.arange(0,361,1)

plt.tight_layout()
figsavepath = '/rita/s0/mjv5638/plots/A_thesis'
plt.savefig('TEST_intro_spatial_distributions_{:02d}{:02d}_{}_initazimuthal_bin{}.png'.format(fhour,fminute,cornerloc,azibinsize), bbox_inches="tight", dpi=200)
plt.clf()
plt.close()