import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from flow_properties import *
from flow_compute import *
from flow_signal_auxiliary import *
from flow_bubble_auxiliary import *

# =========================================================================
# CFD DATA PROCESSING SCRIPT -> BUBBLE PSD
# This script compute bubble PSD (power spectral density) over time
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
# SKIN FRICTION LOADING
# Load flow skin friction for further analysis
# =========================================================================
print("Loading skin friction...")
skin_friction_setup_start = time.time()

# Load skin friction from Python/Numpy file
c_f = load_skin_friction(skin_friction_filename='skin_friction.npy')

skin_friction_setup_time = time.time() - skin_friction_setup_start

# =========================================================================
# TANGENTIAL VELOCITY LOADING
# Load flow tangential velocity for further analysis
# =========================================================================
print("Loading tangential velocity...")
tangential_velocity_setup_start = time.time()

# Load tangential velocity from Python/Numpy file
_, v_t = compute_velocity(['velocity_n.npy', 'velocity_t.npy'])

tangential_velocity_setup_time = time.time() - tangential_velocity_setup_start

# =========================================================================
# COMPUTE BUBBLE LENGTH
# Compute bubble length, separation point and reattachment points
# =========================================================================
print("Compute bubble length...")
bubble_length_start = time.time()

# Compute bubble length, separation, and reattachment points
LSB, x_sep, x_reatt = bubble_length(x_grid=x, y_grid=y,
                                    skin_friction_coeff=c_f,
                                    analysis_region=region_analysis,
                                    invert_separation_order=False)

# Compute and print mean bubble length, separation, and reattachment points
LSB_mean = np.mean(LSB)
print("Bubble lenght (mean value) =", LSB_mean)
x_sep_mean = np.mean(x_sep)
print("Separation point (mean value) =", x_sep_mean)
x_reatt_mean = np.mean(x_reatt)
print("Reattachment point (mean value) =", x_reatt_mean)

bubble_length_time = time.time() - bubble_length_start

# =========================================================================
# COMPUTE BUBBLE AREA
# Compute bubble area over time for further analysis
# =========================================================================
print("Compute bubble area...")
bubble_area_start = time.time()

# Compute bubble area over time
ASB = bubble_area(x_grid=x, y_grid=y,
                  contour_level=-1e-6,
                  velocity_field=v_t,
                  spatial_analysis_region=region_analysis)

# Compute average bubble area
ASB_bar = np.mean(ASB)

# Compute flutuation bubble area
ASB_prime = ASB - ASB_bar

bubble_area_time = time.time() - bubble_area_start

# =========================================================================
# BUBBLE AREA PSD ANALYSIS
# Spectral analysis of bubble area PSD (power spectral density)
# =========================================================================
print("Starting PSD analysis...")
PSD_analysis_start = time.time()

# # Compute separation bubble length PSD and frequency
# LSB_freq, LSB_PSD = sp.signal.welch(LSB, fs=f, window=windows, nperseg=nperseg, noverlap=noverlap,
#     return_onesided=True, scaling='density', axis=-1, average='mean')

# # Compute separation bubble length PSD and frequency
# x_sep_freq, x_sep_PSD = sp.signal.welch(x_sep, fs=f, window=windows, nperseg=nperseg, noverlap=noverlap,
#     return_onesided=True, scaling='density', axis=-1, average='mean')

# # Compute separation bubble length PSD and frequency
# x_reatt_freq, x_reatt_PSD = sp.signal.welch(x_reatt, fs=f, window=windows, nperseg=nperseg, noverlap=noverlap,
#     return_onesided=True, scaling='density', axis=-1, average='mean')

# Compute separation bubble flutuation area PSD and frequency
ASB_prime_freq, ASB_prime_PSD = sp.signal.welch(ASB_prime, fs=f, window=windows, nperseg=nperseg, noverlap=noverlap,
    return_onesided=True, scaling='density', axis=-1, average='mean')

# Frequency correction for SBLI analysis
# LSB_freq *= LSB_mean
# x_sep_freq *= LSB_mean
# x_reatt_freq *= LSB_mean
ASB_prime_freq *= LSB_mean

PSD_analysis_time = time.time() - PSD_analysis_start

# =========================================================================
# BUBBLE PSD PLOT
# Results plot generate and save
# =========================================================================
print("Starting plot generation...")
Plot_generation_start = time.time()

# =====================================
# Plot bubble signal
# =====================================

# Create figure and axis
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18,12), dpi=300, sharex=True, sharey=False)
fig1.suptitle("Bubble - Suction Side", fontsize=18)
# fig1.suptitle("Bubble - Pressure Side", fontsize=18)

# =================
# First plot
# =================
ax1.plot(t, x_sep, label=r"$x_S$", color="orange", linestyle="-", linewidth=2, marker="", markersize=4)
ax1.plot(t, x_reatt, label=r"$x_R$", color="magenta", linestyle="-", linewidth=2, marker="", markersize=4)

# Titles and labels
# ax1.set_title("Bubble - Suction Side", fontsize=18, pad=15, loc="center")
ax1.set_xlabel("t", fontsize=14, labelpad=10)
ax1.set_ylabel("x", fontsize=14, labelpad=10)

# Scales
ax1.set_xscale("linear")
ax1.set_yscale("linear")

# Limits
ax1.set_xlim(15, 60)
ax1.set_ylim(0.7, 0.9)

# Major and minor ticks
ax1.tick_params(axis="x", which="major", direction="inout", color='black', length=8, width=1.2, labelsize=12, pad=6, top=True, bottom=True)
ax1.tick_params(axis="x", which="minor", direction="in",    color='black', length=4, width=0.6, labelsize=0,  pad=0, top=False, bottom=True)
ax1.tick_params(axis="y", which="major", direction="inout", color='black', length=8, width=1.2, labelsize=12, pad=6, right=True, left=True)
ax1.tick_params(axis="y", which="minor", direction="in",    color='black', length=4, width=0.6, labelsize=0,  pad=0, right=True, left=False)

# Custom locators and formatters (ticks)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))

ax1.yaxis.set_major_locator(ticker.MaxNLocator(3))
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# Grid
ax1.grid(True, which="major", axis="both", linestyle="--", color="gray", alpha=0.7)
ax1.grid(True, which="minor", axis="both", linestyle=":", color="gray", alpha=0.5)

# Legend
ax1.legend(loc="upper right", fontsize=12, frameon=True, shadow=True, fancybox=True)

# =================
# Second plot
# =================
ax2.plot(t, LSB, label='Length', color="black", linestyle="-", linewidth=2, marker="", markersize=4)
ax2.plot(t, np.ones_like(LSB)*LSB_mean, label=r"$mean$", color="red", linestyle="--", linewidth=2, marker="", markersize=4)

# Titles and labels
# ax2.set_title("Bubble - Suction Side", fontsize=18, pad=15, loc="center")
ax2.set_xlabel("t", fontsize=14, labelpad=10)
ax2.set_ylabel(r"$L_{SB}$", fontsize=14, labelpad=10)

# Scales
ax2.set_xscale("linear")
ax2.set_yscale("linear")

# Limits
ax2.set_xlim(15, 60)
ax2.set_ylim(0.0, 0.15)

# Major and minor ticks
ax2.tick_params(axis="x", which="major", direction="inout", color='black', length=8, width=1.2, labelsize=12, pad=6, top=True, bottom=True)
ax2.tick_params(axis="x", which="minor", direction="in",    color='black', length=4, width=0.6, labelsize=0,  pad=0, top=False, bottom=True)
ax2.tick_params(axis="y", which="major", direction="inout", color='black', length=8, width=1.2, labelsize=12, pad=6, right=True, left=True)
ax2.tick_params(axis="y", which="minor", direction="in",    color='black', length=4, width=0.6, labelsize=0,  pad=0, right=True, left=False)

# Custom locators and formatters (ticks)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1))

ax2.yaxis.set_major_locator(ticker.MaxNLocator(4))
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# Grid
ax2.grid(True, which="major", axis="both", linestyle="--", color="gray", alpha=0.7)
ax2.grid(True, which="minor", axis="both", linestyle=":", color="gray", alpha=0.5)

# Legend
ax2.legend(loc="upper right", fontsize=12, frameon=True, shadow=True, fancybox=True)

# =================
# Third plot
# =================
# ax3.plot(t, ASB/np.max(ASB), label='Area', color="black", linestyle="-", linewidth=2, marker="", markersize=4)
ax3.plot(t, ASB, label='A', color="black", linestyle="-", linewidth=2, marker="", markersize=4)
ax3.plot(t, ASB_prime, label="A'", color="blue", linestyle="-", linewidth=2, marker="", markersize=4)
ax3.plot(t, np.ones_like(ASB)*np.mean(ASB), label=r"$mean$", color="red", linestyle="--", linewidth=2, marker="", markersize=4)

# Titles and labels
# ax3.set_title("Bubble - Suction Side", fontsize=18, pad=15, loc="center")
ax3.set_xlabel("t", fontsize=14, labelpad=10)
ax3.set_ylabel(r"$A_{SB}$", fontsize=14, labelpad=10)

# Scales
ax3.set_xscale("linear")
ax3.set_yscale("linear")

# Limits
ax3.set_xlim(15, 60)
# ax3.set_ylim(0.0, 0.15)

# Major and minor ticks
ax3.tick_params(axis="x", which="major", direction="inout", color='black', length=8, width=1.2, labelsize=12, pad=6, top=True, bottom=True)
ax3.tick_params(axis="x", which="minor", direction="in",    color='black', length=4, width=0.6, labelsize=0,  pad=0, top=False, bottom=True)
ax3.tick_params(axis="y", which="major", direction="inout", color='black', length=8, width=1.2, labelsize=12, pad=6, right=True, left=True)
ax3.tick_params(axis="y", which="minor", direction="in",    color='black', length=4, width=0.6, labelsize=0,  pad=0, right=True, left=False)

# Custom locators and formatters (ticks)
ax3.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax3.xaxis.set_minor_locator(ticker.MultipleLocator(1))

ax3.yaxis.set_major_locator(ticker.MaxNLocator(4))
ax3.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# Grid
ax3.grid(True, which="major", axis="both", linestyle="--", color="gray", alpha=0.7)
ax3.grid(True, which="minor", axis="both", linestyle=":", color="gray", alpha=0.5)

# Legend
ax3.legend(loc="upper right", fontsize=12, frameon=True, shadow=True, fancybox=True)

# Auto layout adjustment
fig1.tight_layout()

# Save figure
fig1.savefig(fname=figure_path+"bubble_SS_signal.png", dpi=300, format='png', bbox_inches="tight")

# =====================================
# Plot pressure probes signal
# =====================================

# Create figure and axis
fig2, ax = plt.subplots(figsize=(12,10), dpi=300)

# Simple plot
# ax.plot(LSB_freq, LSB_PSD, label=r"$L_{SB}$", color="black", linestyle="-", linewidth=2, marker="", markersize=4)
# ax.plot(x_sep_freq, x_sep_PSD, label=r"$x_S$", color="orange", linestyle="-", linewidth=2, marker="", markersize=4)
# ax.plot(x_reatt_freq, x_reatt_PSD, label=r"$x_R$", color="magenta", linestyle="-", linewidth=2, marker="", markersize=4)
ax.plot(ASB_prime_freq, ASB_prime_PSD, label=r"$A'_{SB}$", color="blue", linestyle=":", linewidth=2, marker="", markersize=4)

# Titles and labels
ax.set_title("Bubble PSD - Suction Side", fontsize=18, pad=15, loc="center")
# ax.set_title("Bubble PSD - Pressure Side", fontsize=18, pad=15, loc="center")
ax.set_xlabel(r"$S_t$", fontsize=14, labelpad=10)
ax.set_ylabel("PSD", fontsize=14, labelpad=10)

# Scales
ax.set_xscale("log")
ax.set_yscale("log")

# Limits
ax.set_xlim(1e-2, 1e1)
ax.set_ylim(1e-14, 1e-7)

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
fig2.savefig(fname=figure_path+"bubble_SS_PSD.png", dpi=300, format='png', bbox_inches="tight")

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
print(f"Loading skin friction:       {skin_friction_setup_time:.2f} seconds")
print(f"Loading tangential velocity: {tangential_velocity_setup_time:.2f} seconds")
print(f"Computing bubble length:     {bubble_length_time:.2f} seconds")
print(f"Computing bubble area:       {bubble_area_time:.2f} seconds")
print(f"PSD analysis:                {PSD_analysis_time:.2f} seconds")
print(f"Generate graphics:           {Plot_generation_time:.2f} seconds")
print("-"*73)
print(f"TOTAL EXECUTION TIME:        {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
print("="*73)

# =========================================================================
# SHOW PLOTS
# Display figures generated
# =========================================================================
# plt.show()
