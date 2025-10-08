# %% Cell 1
import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from filepath import *
from flow_properties import *
from flow_signal_auxiliary import *

# =========================================================================
# CFD DATA PROCESSING SCRIPT -> PRESSURE AND VELOCITY PSD
# This script compute pressure and velocity (x;y) PSD (power spectral
# density) over time for a set of probes
# =========================================================================

print("Starting CFD data reading and processing...")
script_start_time = time.time()

# =========================================================================
# PRESSURE LOADING
# Load flow pressure for further analysis
# =========================================================================
print("Loading pressure...")
pressure_setup_start = time.time()

# Load pressure from Python/Numpy file
pressure_file = flow_path_python + 'pressure_o.npy'
P = np.load(pressure_file, mmap_mode='r')

pressure_setup_time = time.time() - pressure_setup_start

# =========================================================================
# VELOCITY LOADING
# Load flow velocity for further analysis
# =========================================================================
print("Loading velocity...")
velocity_setup_start = time.time()

# Load velocity from Python/Numpy file
velocity_x_file = flow_path_python + 'velocity_x_o.npy'
velocity_y_file = flow_path_python + 'velocity_y_o.npy'
U = np.load(velocity_x_file, mmap_mode='r')
V = np.load(velocity_y_file, mmap_mode='r')

velocity_setup_time = time.time() - velocity_setup_start

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
print('dt =', dt)

time_setup_time = time.time() - time_setup_start

# =========================================================================
# TIME SERIES PROCESSING SETUP
# Calculate number of files to process
# =========================================================================
print("Setting up time series processing...")

# Calculate number of output files to process
nqout = int((last_qout - first_qout) / skip_step_qout) + 1

# =========================================================================
# PROBES LOADING
# Load probes for flow pressure and velocity on suction side
# =========================================================================
print("Loading probes...")
probes_init_start = time.time()

# # Number of probes
# num_probes = 4

# # Initialize pressure probes array
# P_probes = np.zeros((nqout, num_probes))

# Load pressure probes in points in the suction side
P_probes = P[320,49,:]

# Load velocity probes in points in the suction side
U_probes = U[320,49,:]
V_probes = V[320,49,:]

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
# PSD ANALYSIS
# Spectral analysis with PSD (power spectral density)
# =========================================================================
print("Starting PSD analysis...")
PSD_analysis_start = time.time()

# Compute pressure PSD and frequency for it probe
P_freq, P_PSD = sp.signal.welch(P_probes, fs=f, window=windows, nperseg=nperseg, noverlap=noverlap,
    return_onesided=True, scaling='density', axis=-1, average='mean')

# Compute pressure PSD and frequency for it probe
U_freq, U_PSD = sp.signal.welch(U_probes, fs=f, window=windows, nperseg=nperseg, noverlap=noverlap,
    return_onesided=True, scaling='density', axis=-1, average='mean')
V_freq, V_PSD = sp.signal.welch(V_probes, fs=f, window=windows, nperseg=nperseg, noverlap=noverlap,
        return_onesided=True, scaling='density', axis=-1, average='mean')


# Frequency correction for SBLI analysis
P_freq *= 0.1
U_freq *= 0.1
V_freq *= 0.1

PSD_analysis_time = time.time() - PSD_analysis_start

# %% Cell 2
# =========================================================================
# PRESSURE PSD PLOT
# Results plot generate and save
# =========================================================================
print("Starting plot generation...")
Plot_generation_start = time.time()

# =====================================
# Plot probes signal
# =====================================

# Create figure and axis
fig1, ax = plt.subplots(figsize=(18,6), dpi=300)

# Simple plot
ax.plot(t, P_probes, label="Presure", color="red", linestyle="-", linewidth=2, marker="", markersize=4)
ax.plot(t, U_probes, label="Velocity-X", color="green", linestyle="-", linewidth=2, marker="", markersize=4)
ax.plot(t, V_probes, label="Velocity-Y", color="blue", linestyle="-", linewidth=2, marker="", markersize=4)

# Titles and labels
ax.set_title("Signal - point [320,49]", fontsize=18, pad=15, loc="center")
ax.set_xlabel("t", fontsize=14, labelpad=10)
ax.set_ylabel("P, U, V", fontsize=14, labelpad=10)

# Scales
ax.set_xscale("linear")
ax.set_yscale("linear")

# Limits
# ax.set_xlim(15, 41)
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
fig1.savefig(fname=figure_path+"PUV_probes.png", dpi=300, format='png', bbox_inches="tight")

# %% Cell 3
# =====================================
# Plot probes PSD
# =====================================

# Create figure and axis
fig2, ax = plt.subplots(figsize=(12,10), dpi=300)

# Simple plot
ax.plot(P_freq, P_PSD, label="Presure", color="red", linestyle="-", linewidth=2, marker="", markersize=4)
ax.plot(U_freq, U_PSD, label="Velocity-X", color="green", linestyle="-", linewidth=2, marker="", markersize=4)
ax.plot(V_freq, V_PSD, label="Velocity-Y", color="blue", linestyle="-", linewidth=2, marker="", markersize=4)

# Titles and labels
ax.set_title("Signal PSD - point [320,49]", fontsize=18, pad=15, loc="center")
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
fig2.savefig(fname=figure_path+"PUV_PSD.png", dpi=300, format='png', bbox_inches="tight")

Plot_generation_time = time.time() - Plot_generation_start

# =========================================================================
# EXECUTION SUMMARY
# Display comprehensive timing information and script completion status
# =========================================================================
total_execution_time = time.time() - script_start_time

print("\n" + "="*73)
print("CFD DATA PROCESSING COMPLETED SUCCESSFULLY")
print("="*73)
print(f"Loading pressure data:     {pressure_setup_time:.2f} seconds")
print(f"Loading velocity data:     {velocity_setup_time:.2f} seconds")
print(f"Loading time data:         {time_setup_time:.2f} seconds")
print(f"Loading pressure probes:   {probes_init_time:.2f} seconds")
print(f"Setup PSD parameters:      {PSD_setup_time:.2f} seconds")
print(f"PSD analysis:              {PSD_analysis_time:.2f} seconds")
print(f"Generate graphics:         {Plot_generation_time:.2f} seconds")
print("-"*73)
print(f"TOTAL EXECUTION TIME:      {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
print("="*73)


# =========================================================================
# SHOW PLOTS
# Display figures generated
# =========================================================================
# plt.show()
