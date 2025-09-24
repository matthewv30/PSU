## This code plots all the initialization locations and all the trajectories
## on the plan view and also in height-azimuth and height-radius space

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import netCDF4 as NC
import datetime as dt
from scipy.io import loadmat
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks

### IMPORT SHEAR ###
os.chdir('/rita/s0/scratch/nrb171/harvey_postproc/9km/')
shearfname="wrf_shear_nondivirrot02.mat"
mat = loadmat(shearfname)
shear_dir = np.array(mat['shdi'][0,:])
shD = shear_dir[11]

ups = (shD + 180) % 360
dws = shD
rts = (shD + 90) % 360
lfs = (shD - 90) % 360

UR = [rts, ups, "UpShear Right", 'UR']
DR = [dws, rts, "DownShear Right", 'DR']
DL = [lfs, dws, "DownShear Left", 'DL']
UL = [ups, lfs, "UpShear Left", 'UL']
SQ = [UR, DR, DL, UL]

# Grab the radar underlay for the plan view
os.chdir('/rita/s0/scratch/nrb171/Harvey/')
file = 'wrfout_d04_2017-08-25_04:00:00'
data = NC.Dataset(file)
reflectivity = data.variables['REFL_10CM'][0, 25, :, :]
levels_dbz = np.arange(-15, 75, 3)

xx, yy = np.meshgrid(np.arange(0, np.shape(reflectivity)[0]), np.arange(0, np.shape(reflectivity)[1]))

# Load other data
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

mat = loadmat(fname)
init_indx = mat['init_indx'][0]
init_indy = mat['init_indy'][0]
init_indz = mat['init_indz'][0]

part_posx = mat['part_posx']
part_posy = mat['part_posy']
part_posz = mat['part_posz']
part_m = mat['part_m']

maximum_initial_particle_radial_dist = 40

# Indices to highlight in red
highlighted_particles = [(26,16,5),(17,32,1),(14,32,2)]  # replace with the actual i,j,k tuples you want to highlight

for size_idx in range(len(part_posx[0, 0, 0, :, 0])):
    allazi = []
    allhgt = []
    allrad = []
    allpx = []
    allpy = []
    allmass = []

    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(6, 10, figure=fig)
    
    ax1 = fig.add_subplot(gs[3:6, 0:4])
    ax1.contourf(xx - 176 - 240, yy - 176 - 240, reflectivity, cmap=cm.gist_ncar, levels=levels_dbz, extend='both', alpha=0.2)
    ax1.scatter(0, 0, alpha=1, color='r', marker='+')
    planviewdomainspan = 100
    ax1.set_xlim(-planviewdomainspan, planviewdomainspan)
    ax1.set_ylim(-planviewdomainspan, planviewdomainspan)
    ax1.set_xlabel("Horizontal distance from storm center (km)")
    ax1.set_ylabel("Horizontal distance from storm center (km)")

    ax4 = fig.add_subplot(gs[0:3, 0:4])
    ax4.contourf(xx - 176 - 240, yy - 176 - 240, reflectivity, cmap=cm.gist_ncar, levels=levels_dbz, extend='both', alpha=0.2)
    ax4.scatter(0, 0, alpha=1, color='r', marker='+')
    ax4.set_xlim(-planviewdomainspan, planviewdomainspan)
    ax4.set_ylim(-planviewdomainspan, planviewdomainspan)
    ax4.set_xlabel("Horizontal distance from storm center (km)")
    ax4.set_ylabel("Horizontal distance from storm center (km)")

    for i in range(len(init_indx)):
        for j in range(len(init_indy)):
            for k in range(len(init_indz)):
                if not np.isnan(part_posx[i, j, k, size_idx, 0]):
                    particle_radialpos = np.sqrt(part_posx[i, j, k, size_idx, :]**2 + part_posy[i, j, k, size_idx, :]**2)
                    if particle_radialpos[0] <= maximum_initial_particle_radial_dist:
                        particle_azimuthpos = np.arctan2(part_posy[i, j, k, size_idx, :], part_posx[i, j, k, size_idx, :]) * 180 / np.pi
                        particle_azimuthpos = ((-particle_azimuthpos) + 90) % 360
                        
                        # Add particle data to lists for plotting
                        allazi.append(particle_azimuthpos[:])
                        allhgt.append(part_posz[i, j, k, size_idx, :])
                        allrad.append(particle_radialpos[:])
                        allpx.append(part_posx[i, j, k, size_idx, :])
                        allpy.append(part_posy[i, j, k, size_idx, :])
                        allmass.append(part_m[i, j, k, size_idx, :])

                        # Plot initial particle position with highlighting
                        particle_indices = (i, j, k)  # Use loop indices directly
                        
                        # Check if this particle is highlighted
                        if particle_indices in highlighted_particles:
                            ax4.scatter(part_posx[i, j, k, size_idx, 0], part_posy[i, j, k, size_idx, 0], alpha=0.5, color='r', s=30)
                        else:
                            ax4.scatter(part_posx[i, j, k, size_idx, 0], part_posy[i, j, k, size_idx, 0], alpha=0.1, color='b', s=10)

    # Plot trajectories in ax1
    for timestep in range(len(allmass)):
        ax1.plot(allpx[timestep], allpy[timestep], alpha=0.1, color='k')

    fig.tight_layout()
    plt.savefig(f'/rita/s0/mjv5638/plots/manuscript/background/alltrajectorieslocations/alltrajectorieslocations_{fhour:02d}{fminute:02d}_{cornerloc}_size{size_idx}.png', bbox_inches="tight", dpi=200)
    plt.clf()
    plt.close()