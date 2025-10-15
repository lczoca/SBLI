import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from flow_properties import *
from flow_signal_auxiliary import *

# =========================================================================
# CFD DATA PROCESSING SCRIPT -> PSD
# This script compute the PSD (power spectral density) over time for a 
# variable and a set of probes
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
y = grid['y']

# Clean memory
del(grid)

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
# TIME SERIES PROCESSING SETUP
# Calculate number of files to process
# =========================================================================
print("Setting up time series processing...")

# Calculate number of output files to process
nqout = int((last_qout - first_qout) / skip_step_qout) + 1

print(f"    Processing {nqout} files from qout {first_qout} to {last_qout} (step: {skip_step_qout})")

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
# VARIABLE PROBES LOADING
# Load probes for flow variable on suction side
# =========================================================================
print("Loading variable probes...")
probes_init_start = time.time()

# Initialize variable probes array
var_probes = np.zeros((nqout, num_probes))

# Load variable probes in points in the suction side
for i in range(num_probes):
    ix, iy = probe[i]

    var_probes[:,i] = var[ix,iy,:]

probes_init_time = time.time() - probes_init_start

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
# VARIABLE PSD ARRAYS INITIALIZATION
# Initialize arrays to save variable PSD at it probe and the frequencies
# =========================================================================
print("Initializing PSD arrays...")
PSD_probes_init_start = time.time()

# Initialize PSD array
var_PSD = np.zeros((psd_size, num_probes))

# Initialize frequency array
freq = np.zeros((psd_size, num_probes))

PSD_probes_init_time = time.time() - PSD_probes_init_start

# =========================================================================
# VARIABLE PSD ANALYSIS
# Spectral analysis of variable with PSD (power spectral density)
# =========================================================================
print("Starting PSD analysis...")
PSD_analysis_start = time.time()

# Compute PSD and frequency for it probe
for i in range(num_probes):
    freq[:,i], var_PSD[:,i] = sp.signal.welch(var_probes[:,i], fs=f, window=windows, nperseg=nperseg, noverlap=noverlap,
        return_onesided=True, scaling='density', axis=- 1, average='mean')

# Frequency correction for SBLI analysis
freq *= bubble_length

PSD_analysis_time = time.time() - PSD_analysis_start

# =========================================================================
# VARIABLE PSD PLOT
# Results plot generate and save
# =========================================================================
print("Starting plot generation...")
Plot_generation_start = time.time()

# =====================================
# Plot variable probes signal
# =====================================
for i in range(num_probes):
    ix, iy = probe[i]

    # Create figure and axis
    fig1, ax = plt.subplots(figsize=(18,6), dpi=300)

    # Simple plot
    ax.plot(t, var_probes[:,i], label=f"[{x[ix, iy]},{y[ix, iy]}]", color="black", linestyle="-", linewidth=2, marker="", markersize=4)

    # Titles and labels
    ax.set_title("Signal " + variable_plot_title, fontsize=18, pad=15, loc="center")
    ax.set_xlabel("t", fontsize=14, labelpad=10)
    ax.set_ylabel("P", fontsize=14, labelpad=10)

    # Scales
    ax.set_xscale("linear")
    ax.set_yscale("linear")

    # Limits
    ax.set_xlim(15, 60)
    # ax.set_ylim(0.35, 0.70)

    # Major and minor ticks
    ax.tick_params(axis="x", which="major", direction="inout", color='black', length=8, width=1.2, labelsize=12, pad=6, top=True, bottom=True)
    ax.tick_params(axis="x", which="minor", direction="in",    color='black', length=4, width=0.6, labelsize=0,  pad=0, top=False, bottom=True)
    ax.tick_params(axis="y", which="major", direction="inout", color='black', length=8, width=1.2, labelsize=12, pad=6, right=True, left=True)
    ax.tick_params(axis="y", which="minor", direction="in",    color='black', length=4, width=0.6, labelsize=0,  pad=0, right=True, left=False)

    # Custom locators and formatters (ticks)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    ax.yaxis.set_major_locator(ticker.MaxNLocator(8))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # Grid
    ax.grid(True, which="major", axis="both", linestyle="--", color="gray", alpha=0.7)
    ax.grid(True, which="minor", axis="both", linestyle=":", color="gray", alpha=0.5)

    # Legend
    ax.legend(loc="upper right", fontsize=12, frameon=True, shadow=True, fancybox=True)

    # Extra text/annotation
    # ax.text(5, 0.8, "Annotation", fontsize=12, color="darkred", rotation=15)

    # Auto layout adjustment
    fig1.tight_layout()

    # Save figure
    # fig1.savefig(fname=figure_path+f"probe{ix}_{iy}.png", dpi=300, format='png', bbox_inches="tight")

    # Close figures
    plt.close()

    # =====================================
    # Plot variable probes PSD
    # =====================================

    # Create figure and axis
    fig2, ax = plt.subplots(figsize=(12,10), dpi=300)

    # Simple plot
    ax.plot(freq[:,i], var_PSD[:,i], label=f"[{x[ix, iy]:.2f},{y[ix, iy]:.2f}]", color="black", linestyle="-", linewidth=2, marker="", markersize=4)

    # ax.axvline(0.0175)
    # ax.axvline(0.057) 
    # ax.axvline(0.079) 
    # ax.axvline(0.12)  
    # ax.axvline(0.19)  
    # ax.axvline(0.3)   
    # ax.axvline(0.4)

    ax.axvline(0.043)
    ax.axvline(0.075) 
    ax.axvline(0.095) 
    ax.axvline(0.125)  
    ax.axvline(0.15)  
    ax.axvline(0.17)  
    ax.axvline(0.23)  
    ax.axvline(0.33)   
    ax.axvline(0.37)   
    ax.axvline(0.69)   

    # Titles and labels
    ax.set_title("PSD " + variable_plot_title, fontsize=18, pad=15, loc="center")
    ax.set_xlabel(r"$S_t$", fontsize=14, labelpad=10)
    ax.set_ylabel("PSD", fontsize=14, labelpad=10)

    # Scales
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Limits
    # ax.set_xlim(1e-2, 1e1)
    # ax.set_ylim(1e-6, 1e-1)

    # Major and minor ticks
    ax.tick_params(axis="x", which="major", direction="inout", color='black', length=8, width=1.2, labelsize=12, pad=6, top=True, bottom=True)
    ax.tick_params(axis="x", which="minor", direction="in",    color='black', length=4, width=0.6, labelsize=0,  pad=0, top=False, bottom=True)
    ax.tick_params(axis="y", which="major", direction="inout", color='black', length=8, width=1.2, labelsize=12, pad=6, right=True, left=True)
    ax.tick_params(axis="y", which="minor", direction="in",    color='black', length=4, width=0.6, labelsize=0,  pad=0, right=True, left=False)

    # Custom locators and formatters (ticks)
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(1,10)*0.1))

    ax.yaxis.set_major_locator(ticker.LogLocator(base=10))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(1,10)*0.1))

    # Grid
    ax.grid(True, which="major", axis="both", linestyle="--", color="gray", alpha=0.7)
    ax.grid(True, which="minor", axis="both", linestyle=":", color="gray", alpha=0.5)

    # Legend
    ax.legend(loc="upper right", fontsize=12, frameon=True, shadow=True, fancybox=True)

    # Extra text/annotation
    # ax.text(5, 0.8, "Annotation", fontsize=12, color="darkred", rotation=15)

    # Auto layout adjustment
    fig2.tight_layout()

    # Save figure
    fig2.savefig(fname=figure_path+f"PSD_probe{ix}_{iy}.png", dpi=300, format='png', bbox_inches="tight")

    # Close figures
    plt.close()

Plot_generation_time = time.time() - Plot_generation_start

# =========================================================================
# EXECUTION SUMMARY
# Display comprehensive timing information and script completion status
# =========================================================================
total_execution_time = time.time() - script_start_time

print("\n" + "="*73)
print("CFD DATA PROCESSING COMPLETED SUCCESSFULLY")
print("="*73)
print(f"Loading grid:            {grid_setup_time:.2f} seconds")
print(f"Loading time data:       {time_setup_time:.2f} seconds")
print(f"Loading variable data:   {variable_setup_time:.2f} seconds")
print(f"Loading variable probes: {probes_init_time:.2f} seconds")
print(f"Setup PSD parameters:    {PSD_setup_time:.2f} seconds")
print(f"PSD initialization:      {PSD_probes_init_time:.2f} seconds")
print(f"PSD analysis:            {PSD_analysis_time:.2f} seconds")
print(f"Generate graphics:       {Plot_generation_time:.2f} seconds")
print("-"*73)
print(f"TOTAL EXECUTION TIME:    {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
print("="*73)


# =========================================================================
# SHOW PLOTS
# Display figures generated
# =========================================================================
# plt.show()
