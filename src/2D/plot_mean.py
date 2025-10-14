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
# CFD DATA PROCESSING SCRIPT -> PLOT MEAN FLOW
# This script plot mean flow (velocity in X direction) and the probes 
# positions 
# =========================================================================

print("Starting CFD mean flow plot...")
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
# VELOCITIES (X,Y) LOADING
# Load flow velocity-X and velocity-Y for further analysis
# =========================================================================
print("Loading velocities...")
velocity_setup_start = time.time()

# Load velocity-X and velocity-Y from Python/Numpy file
u, v = load_velocity(['mean_velocity_x.npy', 'mean_velocity_y.npy'])

velocity_setup_time = time.time() - velocity_setup_start

# =========================================================================
# BUBBLE PLOT
# Results plot generate and save
# =========================================================================
print("Starting plot generation...")
Plot_generation_start = time.time()

# =====================================
# Plot velocity-X mean
# =====================================

# Create figure and axis
fig1, ax = plt.subplots(figsize=(25,16), dpi=300)

# Generate levels
level = np.linspace(-0.2, 2.0, 101)

# Contour plot
airfoil = ax.fill(x[:,0], y[:,0], label=None, facecolor="lightgray", edgecolor="black", linestyle="solid", linewidth=2)
flow = ax.contourf(x, y, u, levels=level, cmap="seismic", extend='both')

# # Plot probes
for i in range(num_probes):
    ix, iy = probe[i]
    ax.plot(x[ix, iy], y[ix, iy], label=None, color="black", linestyle=None, linewidth=0, marker=".", markersize=4)

# Colorbar
cbar = fig1.colorbar(flow, ax=ax, orientation="vertical", location='right', shrink=0.9, pad=0.05)
# cbar.set_label("Pressure PSD Map", fontsize=12, labelpad=10)
# cbar.set_ticks([-2.4, -4.6, -6.8])
# cbar.set_ticklabels(["Low", "Mid", "High"])
cbar.ax.tick_params(labelsize=10, colors="black", direction="in", length=5, width=1)

# Titles and labels
ax.set_title(f"Mean Flow - velocity X (and Probes)", fontsize=18, pad=15, loc="center")
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

# # Grid
# ax.grid(True, which="major", axis="both", linestyle="--", color="gray", alpha=0.7)
# ax.grid(True, which="minor", axis="both", linestyle=":", color="gray", alpha=0.5)

# Auto layout adjustment
fig1.tight_layout()

# Save figure
fig1.savefig(fname=figure_path+"mean_probes.png", dpi=300, format='png', bbox_inches="tight")

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
print(f"Loading grid:         {grid_setup_time:.2f} seconds")
print(f"Loading velocities:   {velocity_setup_time:.2f} seconds")
print(f"Generate graphics:    {Plot_generation_time:.2f} seconds")
print("-"*73)
print(f"TOTAL EXECUTION TIME: {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
print("="*73)