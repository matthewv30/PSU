## This code plots the initial IWC curve and crystal initial sizes and the mass distribution of all the locations 
# and the smallest, middle and largest size bin. 

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import netCDF4 as NC
import datetime as dt
from scipy.io import loadmat
from matplotlib.gridspec import GridSpec
from scipy.integrate import quad

### IMPORT SHEAR ###
os.chdir('/rita/s0/scratch/nrb171/harvey_postproc/9km/')
shearfname="wrf_shear_nondivirrot02.mat" #nick's shear data
from scipy.io import loadmat
mat = loadmat(shearfname)
shear_dir = np.array(mat['shdi'][0,:])
shD = shear_dir[11]     #19 is for 12Z on 25th, 11 is for 04Z

ups=(shD+180) % 360 #Upshear direction
dws=shD #Downshear Direction
rts=(shD+90) % 360 #right-of-shear 
lfs=(shD-90) % 360 #left-of-shear

UR=[rts,ups,"UpShear Right", 'UR']#UpShear Right Quadrant
DR=[dws,rts,"DownShear Right", 'DR']#DownShear Right Quadrant
DL=[lfs,dws,"DownShear Left", 'DL']#DownShear Left Quadrant
UL=[ups,lfs,"UpShear Left", 'UL']#UpShear Left Quadrant
SQ=[UR,DR,DL,UL]#All shear quadrants

#grab the radar underlay for the plan view
os.chdir('/rita/s0/scratch/nrb171/Harvey/')
file='wrfout_d04_2017-08-25_04:00:00'
data = NC.Dataset(file)
reflectivity=data.variables['REFL_10CM'][0,25,:,:] # crystal level (~6.5 km index 25)
levels_dbz=np.arange(-15,75,3)

print(np.shape(reflectivity))
xx, yy = np.meshgrid(np.arange(0,np.shape(reflectivity)[0]),np.arange(0,np.shape(reflectivity)[1]))

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

#### LIST OF ALL VARIABLES AVAILABLE 
# Cropped grid values where crystals are placed
init_indx = mat['init_indx'][0]
init_indy = mat['init_indy'][0]
init_indz = mat['init_indz'][0]

part_V = mat['part_V']      #shape (x, y, z, particle number, time)
part_a = mat['part_a']
part_c = mat['part_c']
part_m = mat['part_m']
part_mrime = mat['part_mrime']

# storm relative position 
part_posx = mat['part_posx']    #shape (x, y, z, particle number, time)
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

maximum_initial_particle_radial_dist = 40    #distance from storm center, not particle size radius
os.chdir('/rita/s0/bsr5234/modelling/ICTG/modeldatafiles/cropped_domains/')
fname = 'varsWRFv3.8.1HarveyV2_201708250400.mat'
mat = loadmat(fname)
crop_lambdai = mat['crop_lambdai']  #(351, 351, 71)
crop_qnice = mat['crop_qnice']      #(351, 351, 71)
crop_Tv = mat['crop_Tv']            #(351, 351, 71)
crop_p = mat['crop_p']              #(351, 351, 71)

# ========================= Calculate PSD ====================
# Constants
rhoi = 890  # Density of ice kg/m3 (from the provided thermodynamic constants)
minmass = 1e-12  # in kg
Dmin = ((minmass * 3 / (4 * np.pi * rhoi))**(1/3)) * 2  # radiusx2 in m
Dmax = 5 * (200e-6)  # 5x min. snow size in m (=1mm)
logD = True  # Flag for logarithmic spacing
np_bins = 100  # Number of bins

# Define some thermodynamic constants
g = 9.81  # gravitational acceleration in m/s^2
cp = 1004.0  # specific heat capacity of air at constant pressure (J/kg/K)
cpi = 2108.0  # specific heat capacity of ice at constant pressure (J/kg/K)
cpw = 4187.0  # specific heat capacity of liquid water at constant pressure (J/kg/K)
Rv = 461.5  # gas constant for vapor (J/kg/K)
Rd = 287.05  # gas constant for dry air (J/kg/K)    #! fixed from 278 in Chelsey matlab code
p0 = 100000.0  # reference surface pressure (Pa)
es0 = 611  # equilibrium vapor pressure at T=T0=273.15 K (Pa)
T0 = 273.15  # reference temperature (K)
lv = 2.5e6  # enthalpy of vaporization (J/kg)
lf = 3.33e5  # enthalpy of fusion/melting (J/kg)
ls = lv + lf  # enthalpy of sublimation (J/kg)
rhol = 1000.0  # density of liquid water (kg/m^3)
rhoi = 917  # density of ice (kg/m^3)       # 890 in Thompson
rhoa = crop_p / (Rd * crop_Tv)

# Calculate Ntot and N0i
Ntot = crop_qnice * rhoa  # m-3
N0i = crop_lambdai * Ntot  # m-3 m-1
PSD_numbins = 100

# Create Darray and dDarray
if logD:
    Darray_edges = np.zeros(PSD_numbins+1)
    Darray = np.zeros(PSD_numbins)
    dDarray = np.zeros(PSD_numbins)
    Darray_edges[0] = Dmin
    Darray_edges[PSD_numbins] = Dmax
    for n in range(1, PSD_numbins):
        Darray_edges[n] = np.exp(((n - 1) / PSD_numbins) * np.log(Dmax / Dmin) + np.log(Dmin))
    for n in range(PSD_numbins):
        Darray[n] = np.sqrt(Darray_edges[n] * Darray_edges[n + 1])
        dDarray[n] = Darray_edges[n + 1] - Darray_edges[n]
else:
    Darray_edges = np.linspace(0, Dmax, PSD_numbins+1)  # 100 bins (in m units)
    dD = np.diff(Darray_edges)  # in m units
    dD = dD[0]
    Darray = Darray_edges[:-1] + (dD / 2)

# Calculate IWC using cloud ice PSD
def fun_IWC(DD):
    return (np.pi / 6) * rhoi * DD**3 * N0i_const * np.exp(-lambdai_const * DD)  # kg/m3 for IWC

ts=0
allmass_allbins=[]
allsize_allbins=[]
for size_idx in range(len(part_posx[0,0,0,:,0])):
    allmass=[]
    allsize=[]
    for k in range(len(init_indz)):
        for i in range(len(init_indx)):
            for j in range(len(init_indy)):
                if np.isnan(part_posx[i,j,k,size_idx,0]) == False:     #check if the particle survived initialization
                    particle_radialpos = np.sqrt(part_posx[i,j,k,size_idx,:]**2 + part_posy[i,j,k,size_idx,:]**2)
                    if particle_radialpos[0] <= maximum_initial_particle_radial_dist:
                        allmass.append(part_m[i,j,k,size_idx,ts])              
                        allsize.append(part_y[i,j,k,size_idx,ts])
    allmass_allbins.append(allmass)
    allsize_allbins.append(allsize)

allmass_allbins_array = np.array(allmass_allbins)
allsize_allbins_array = np.array(allsize_allbins)


# Function to sample bin indices based on IWC distribution
def sample_iwc_bin_indices(IWC_bins, num_particles):
    # Check if all elements in IWC_bins are zero
    if np.all(IWC_bins == 0):
        # If all elements are zero, handle it accordingly (e.g., return an empty array or predefined indices)
        print("All elements in IWC_bins are zero. No valid bins to sample from.")
        return np.zeros(num_particles, dtype=int)  # You can change this based on how you want to handle it

    # Calculate the cumulative IWC distribution
    cumulative_IWC = np.cumsum(IWC_bins)
    cumulative_IWC = cumulative_IWC / cumulative_IWC[-1]  # Normalize to make it a proper CDF

    # Generate quantiles for the given number of particles
    quantiles = np.linspace(0.5 / num_particles, 1 - 0.5 / num_particles, num_particles)

    # Interpolate to find the corresponding bin indices
    bin_indices = np.interp(quantiles, cumulative_IWC, np.arange(len(IWC_bins)))
    bin_indices = np.round(bin_indices).astype(int)  # Round to the nearest integer to get indices

    return bin_indices

ND = np.zeros(len(Darray))
iidx = 168
jidx = 204
kidx = 27

# Parameters needed to compute cloud ice PSD that are taken from WRF sim
N0i_const = N0i[iidx, jidx, kidx]
lambdai_const = crop_lambdai[iidx, jidx, kidx]

IWC_int, _ = quad(fun_IWC, 0, Dmax)  # integrate above function

# Iteratively computing the IWC contributed by each size bin within the cloud ice PSD
IWC_cumint = np.zeros(len(Darray))
IWC_bins = np.zeros(len(Darray))
for dd in range(len(Darray)):
    D = Darray[dd]
    D1 = Darray_edges[dd]
    D2 = Darray_edges[dd + 1]
    IWC_cumint[dd], _ = quad(fun_IWC, 0, D)
    IWC_bins[dd], _ = quad(fun_IWC, D1, D2)

# print(Darray[:])
ncnct = N0i_const * np.exp(-lambdai_const * Darray[:])  # number concentration (?)
print(ncnct)

# Find the np (number) largest IWC bins and corresponding diameters
number_of_particles = 5  # Number of largest bins to find, change as needed
bin_indices = sample_iwc_bin_indices(IWC_bins, number_of_particles)
IWC_selected = IWC_bins[bin_indices]
Danchors = Darray[bin_indices]
Danchors_mm = Danchors * 1e3  # convert to mm

fig = plt.figure(figsize=(10, 8))
gs = GridSpec(2, 2, figure=fig)
ax2 = fig.add_subplot(gs[0, 0]) #distribution
# ax2.plot()

num_of_bins = len(Danchors)
bin_colors = ['#003f5c','#58508d','#bc5090','#ff6361','#ffa600']
bin_labels=['Smallest bin', '2nd Smallest Bin', 'Middle Bin', '2nd Largest Bin', 'Largest Bin']

ax2.plot(Darray * 1e6, IWC_bins, 'b-', label='IWC Distribution')  # Convert Darray to micrometers (um) for plotting
ax2.bar(Darray * 1e6, IWC_bins, width=dDarray * 1e6, align='center', edgecolor='black', alpha=0.2)
for bn in range(num_of_bins):
    ax2.vlines(Danchors[bn] * 1e6, ymin=0, ymax=np.max(IWC_selected), colors=bin_colors[bn], linewidth=2.5, label=bin_labels[bn])
ax2.set_xscale('log')
ax2.set_xlabel('Particle Diameter (μm)')
ax2.set_ylabel('Ice Water Content (kg/m³)')
ax2.set_title('IWC Distribution Across Size Bins')
ax2.legend(fontsize=8)
ax2.grid(True)

ax3 = fig.add_subplot(gs[0, 1]) #distribution
ax3.plot(Darray * 1e6, ncnct, 'b-', label='Number\nConcentration')  # Convert Darray to micrometers (um) for plotting
ax3.set_title('Number Concentration by Size Bin')
for bn in range(num_of_bins):
    ax3.vlines(Danchors[bn] * 1e6, ymin=0, ymax=np.max(ncnct), colors=bin_colors[bn], linewidth=2.5, label=bin_labels[bn])
ax3.set_xscale('log')
ax3.set_xlabel('Particle Diameter (μm)')
ax3.set_ylabel('Number Concentration (#/m³)')

ax3.legend(fontsize=8)
ax3.grid(True)

nonlinbins = np.logspace(-6, 0, 40)
nonlinbins_size = np.logspace(1, 3, 20)

fig.tight_layout()
figsavepath = '/rita/s0/mjv5638/plots/manuscript/background'
if os.path.exists(figsavepath):
    os.chdir(figsavepath)
else:
    os.mkdir(figsavepath)
    os.chdir(figsavepath)
plt.savefig('TEST_init_summary_{:02d}{:02d}_{}_crysID{}x{}y{}z.png'.format(fhour,fminute,cornerloc, iidx, jidx, kidx), bbox_inches="tight", dpi=100)
plt.clf()
plt.close()