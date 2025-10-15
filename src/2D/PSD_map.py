import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from flow_properties import *
from flow_signal_auxiliary import *

# =========================================================================
# CFD DATA PROCESSING SCRIPT -> PSD MAP
# This script compute PSD (power spectral density) map over time for
# variable throughout a region
# =========================================================================

print("Starting CFD data reading and processing...")
script_start_time = time.time()

# =========================================================================
# GRID LOADING
# Load O-grid for further analysis
# =========================================================================
print("Loading O-grid...")
grid_setup_start = time.time()

# Load O-grid from Python/Numpy file
grid_file = flow_path_python + 'grid.npz'
grid = np.load(grid_file)
x = grid['x']
# y = grid['y']

# New x-coordinate system (Sucton side: 0 to 1 ;  Pressure side 1 to 2)
x_wall = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(x[:, 0])))])

# Clean memory
del(grid)
del(x)

grid_setup_time = time.time() - grid_setup_start

# =========================================================================
# TIME LOADING AND SETUP
# Load simulation time for further analysis and calculate time step
# =========================================================================
print("Loading time...")
time_setup_start = time.time()

# Load time from Python/Numpy file
time_file = flow_path_python + 'time.npy'
t = np.load(time_file)

# Time correction
t *= Ma

# Time step
dt = t[1] - t[0]

time_setup_time = time.time() - time_setup_start

# =========================================================================
# PSD PROCESSING SETUP
# Calculate frequency, number of bins, sample, segment length, overlap
# for segments, and window necessary for PSD analysis
# =========================================================================
print("Starting PSD setup...")
PSD_setup_start = time.time()

# Frequency calculation
f = 1.0 / dt

# Number of time instants (number of samples)
nt = len(t)

# Number of points to overlap between segments
overlap = 0.6667

# Number of bins in signal to be used
nbins = 4

# Calculate welch parameters
nperseg, noverlap, psd_size = welch_parameters(num_bins=nbins, num_points=nt, overlap=overlap)

# Compute window
windows = np.hanning(nperseg)

PSD_setup_time = time.time() - PSD_setup_start

# =========================================================================
# VARIABLE LOADING
# Load flow variable for further analysis (PSD)
# =========================================================================
print("Loading variable...")
variable_setup_start = time.time()

# Load variable from Python/Numpy file
variable_file = flow_path_python + variable_name
var = np.load(variable_file, mmap_mode="r")

variable_setup_time = time.time() - variable_setup_start

# =========================================================================
# SIGNAL REGION SETUP
# Separates the pressure and tangential velocity signals in the analysis
# region
# =========================================================================
print("Setup signal region...")
signal_init_start = time.time()

# Initialize pressure signal array
var_signal = var[imin:imax,jarg,:]

signal_init_time = time.time() - signal_init_start

# =========================================================================
# VARIABLE PSD ANALYSIS
# Spectral analysis of variable with PSD (power spectral density)
# =========================================================================
print("Starting PSD analysis...")
PSD_analysis_start = time.time()

# Compute variable signal PSD and frequency
freq, var_PSD = sp.signal.welch(var_signal, fs=f, window=windows, nperseg=nperseg, noverlap=noverlap,
    return_onesided=True, scaling='density', axis=1, average='mean')

# Frequency correction for SBLI analysis
freq *= bubble_length

PSD_analysis_time = time.time() - PSD_analysis_start

# =========================================================================
# PSD MAP PLOT
# Results plot generate and save
# =========================================================================
# print("Starting plot generation...")
Plot_generation_start = time.time()

# =====================================
# Plot variable PSD map
# =====================================

# Create figure and axis
fig1, ax = plt.subplots(figsize=(25,16), dpi=300)

# Generate Strouhal-X_wall mesh
St, XX = np.meshgrid(freq, x_wall[imin:imax])

# Generate levels
level = np.linspace(-6.8, -2.4, 11)

# Contour plot
var_cf = ax.contourf(XX, St, np.log10(var_PSD), levels=level, cmap="OrRd", extend='both')

# Colorbar
cbar = fig1.colorbar(var_cf, ax=ax, orientation="vertical", location='right', shrink=0.9, pad=0.05)
# cbar.set_label("Pressure PSD Map", fontsize=12, labelpad=10)
cbar.set_ticks([-2.4, -4.6, -6.8])
# cbar.set_ticklabels(["Low", "Mid", "High"])
cbar.ax.tick_params(labelsize=10, colors="black", direction="in", length=5, width=1)

# Titles and labels
ax.set_title("PSD Map - " + variable_plot_title, fontsize=18, pad=15, loc="center")
ax.set_xlabel("x", fontsize=14, labelpad=10)
ax.set_ylabel("St", fontsize=14, labelpad=10)

# Scales
ax.set_xscale("linear")
ax.set_yscale("log")

# Limits
ax.set_xlim(0.5, 1.01)
ax.set_ylim(4e-2, 2e1)

# Major and minor ticks
ax.tick_params(axis="x", which="major", direction="inout", color='black', length=8, width=1.2, labelsize=12, pad=6, top=True, bottom=True)
ax.tick_params(axis="x", which="minor", direction="in",    color='black', length=4, width=0.6, labelsize=0,  pad=0, top=False, bottom=True)
ax.tick_params(axis="y", which="major", direction="inout", color='black', length=8, width=1.2, labelsize=12, pad=6, right=True, left=True)
ax.tick_params(axis="y", which="minor", direction="in",    color='black', length=4, width=0.6, labelsize=0,  pad=0, right=True, left=False)

# Custom locators and formatters (ticks)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
# ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

ax.yaxis.set_major_locator(ticker.LogLocator(base=10))
# ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(1,10)*0.1))

# Grid
ax.grid(True, which="major", axis="both", linestyle="--", color="gray", alpha=0.7)
ax.grid(True, which="minor", axis="both", linestyle=":", color="gray", alpha=0.5)

# Auto layout adjustment
fig1.tight_layout()

# Save figure
fig1.savefig(fname=figure_path+variable_save_name+f"_PSD_map.png", dpi=300, format='png', bbox_inches="tight")

Plot_generation_time = time.time() - Plot_generation_start

# =========================================================================
# EXECUTION SUMMARY
# Display comprehensive timing information and script completion status
# =========================================================================
total_execution_time = time.time() - script_start_time

print("\n" + "="*73)
print("CFD DATA PROCESSING COMPLETED SUCCESSFULLY")
print("="*73)
print(f"Loading grid:                {grid_setup_time:.2f} seconds")
print(f"Loading time:                {time_setup_time:.2f} seconds")
print(f"Setup PSD parameters:        {PSD_setup_time:.2f} seconds")
print(f"Loading flow variable:       {variable_setup_time:.2f} seconds")
print(f"Initializing flow variable:  {signal_init_time:.2f} seconds")
print(f"PSD analysis:                {PSD_analysis_time:.2f} seconds")
print(f"Generate graphics:           {Plot_generation_time:.2f} seconds")
print("-"*73)
print(f"TOTAL EXECUTION TIME:        {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
print("="*73)
