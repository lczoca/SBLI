import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as anime

from flow_properties import *
from flow_compute import *
from flow_signal_auxiliary import *

# =========================================================================
# CFD DATA PROCESSING SCRIPT -> BUBBLE ANIMATION
# This script makes an animation of the bubble
# =========================================================================

print("Starting CFD bubble animation...")
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
# Load simulation time for further analysis
# =========================================================================
print("Loading time...")
time_setup_start = time.time()

# Load time from Python/Numpy file
time_file = flow_path_python + 'time.npy'
t = np.load(time_file)

# Time correction
t *= Ma

# Number of snapshots
nt = np.size(t)

time_setup_time = time.time() - time_setup_start

# =========================================================================
# TANGENTIAL VELOCITY LOADING
# Load flow tangential velocity for further analysis
# =========================================================================
print("Loading tangential velocity...")
tangential_velocity_setup_start = time.time()

# Load tangential velocity from Python/Numpy file
_, v_t = compute_velocity(['velocity_n.npy', 'velocity_t.npy'])

# Compute average tangential velocity from Python/Numpy file
v_t_bar = np.mean(v_t, axis=-1)

tangential_velocity_setup_time = time.time() - tangential_velocity_setup_start

# =========================================================================
# BUBBLE PLOT
# Results plot generate and save
# =========================================================================
print("Starting plot generation...")
Plot_generation_start = time.time()

# =====================================
# Plot bubble over time
# =====================================

# Loop over time
for i in range(nt):
    print(f"Iteration: {i}", end='\r', flush=True)

    # Create figure and axis
    fig1, ax = plt.subplots(figsize=(25,16), dpi=300)

    # Generate levels
    # level = -1e-5  # Suction Side
    level = 1e-5   # Pressure Side

    # Region to analyze
    ra = region_analysis

    # Contour plot
    airfoil = ax.fill(x[:,0], y[:,0], label=None, facecolor="lightgray", edgecolor="black", linestyle="solid", linewidth=2)
    cont = ax.contour(x[ra,:], y[ra,:], v_t[ra,:,i], levels=[level], colors=["red"], linestyles=["solid"], linewidths=[2])
    bubble = ax.contour(x[ra,:], y[ra,:], v_t_bar[ra,:], levels=[level], colors=["blue"], linestyles=["dotted"], linewidths=[2])

    # Titles and labels
    ax.set_title(f"Bubble - Suction Side - t = {t[i]:.2f} s", fontsize=18, pad=15, loc="center")
    ax.set_xlabel("x", fontsize=14, labelpad=10)
    ax.set_ylabel("y", fontsize=14, labelpad=10)

    # Scales
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set_aspect("equal")

    # Limits
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.2, 0.05)

    # Major and minor ticks
    ax.tick_params(axis="x", which="major", direction="inout", color='black', length=8, width=1.2, labelsize=12, pad=6, top=True, bottom=True)
    ax.tick_params(axis="x", which="minor", direction="in",    color='black', length=4, width=0.6, labelsize=0,  pad=0, top=False, bottom=True)
    ax.tick_params(axis="y", which="major", direction="inout", color='black', length=8, width=1.2, labelsize=12, pad=6, right=True, left=True)
    ax.tick_params(axis="y", which="minor", direction="in",    color='black', length=4, width=0.6, labelsize=0,  pad=0, right=True, left=False)

    # # Custom locators and formatters (ticks)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    # ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(1,10)*0.1))

    # Grid
    ax.grid(True, which="major", axis="both", linestyle="--", color="gray", alpha=0.7)
    ax.grid(True, which="minor", axis="both", linestyle=":", color="gray", alpha=0.5)

    # -----------------
    # Zoom
    # -----------------
    axin = ax.inset_axes([-0.05, 0.1, 0.45, 0.45])

    airfoilin = axin.fill(x[:,0], y[:,0], label=None, facecolor="lightgray", edgecolor="black", linestyle="solid", linewidth=2)
    contin = axin.contour(x[ra,:], y[ra,:], v_t[ra,:,i], levels=[level], colors=["red"], linestyles=["solid"], linewidths=[2])
    bubblein = axin.contour(x[ra,:], y[ra,:], v_t_bar[ra,:], levels=[level], colors=["blue"], linestyles=["dotted"], linewidths=[2])

    # Titles and labels
    axin.set_title("Zoom on Bubble", fontsize=12, pad=15, loc="center")
    axin.set_xlabel("x", fontsize=10, labelpad=10)
    axin.set_ylabel("y", fontsize=10, labelpad=10)

    # Scales
    axin.set_xscale("linear")
    axin.set_yscale("linear")
    axin.set_aspect("equal")

    # Limits
    # axin.set_xlim(0.7, 0.9)       # Suction Side
    # axin.set_ylim(-0.15, -0.05)   # Suction Side
    axin.set_xlim(0.5, 0.7)       # Pressure Side
    axin.set_ylim(-0.15, -0.05)   # Pressure Side

    # Major and minor ticks
    axin.tick_params(axis="x", which="major", direction="inout", color='black', length=8, width=1.2, labelsize=6, pad=6, top=True, bottom=True)
    axin.tick_params(axis="x", which="minor", direction="in",    color='black', length=4, width=0.6, labelsize=0,  pad=0, top=False, bottom=True)
    axin.tick_params(axis="y", which="major", direction="inout", color='black', length=8, width=1.2, labelsize=6, pad=6, right=True, left=True)
    axin.tick_params(axis="y", which="minor", direction="in",    color='black', length=4, width=0.6, labelsize=0,  pad=0, right=True, left=False)

    # Custom locators and formatters (ticks)
    axin.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    # axin.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    axin.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    # axin.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(1,10)*0.1))

    # Grid
    axin.grid(True, which="major", axis="both", linestyle="--", color="gray", alpha=0.7)
    axin.grid(True, which="minor", axis="both", linestyle=":", color="gray", alpha=0.5)

    # Auto layout adjustment
    fig1.tight_layout()

    # Save figure
    # fig1.savefig(fname=figure_path+f"bubble_SS/bubble_t{i:04d}.png", dpi=300, format='png', bbox_inches="tight")  # Suction Side
    fig1.savefig(fname=figure_path+f"bubble_PS/bubble_t{i:04d}.png", dpi=300, format='png', bbox_inches="tight")  # Pressure Side

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
print(f"Loading grid:                {grid_setup_time:.2f} seconds")
print(f"Loading time:                {time_setup_time:.2f} seconds")
print(f"Loading tangential velocity: {tangential_velocity_setup_time:.2f} seconds")
print(f"Generate graphics:           {Plot_generation_time:.2f} seconds")
print("-"*73)
print(f"TOTAL EXECUTION TIME:        {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
print("="*73)