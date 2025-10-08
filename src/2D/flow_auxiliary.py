import numpy as np
import time

from filepath import *
from flow_properties import *
from flow_compute import *

# =========================================================================
# CFD DATA PROCESSING AND AVERAGING SCRIPT
# This script processes multiple files to compute auxiliaries flow
# properties and its mean for computational grids (O-grid)
# =========================================================================

print("Starting CFD data processing...")
script_start_time = time.time()

# =========================================================================
# GRID FILE LOADING AND SETUP
# Read dimensions and coordinates for the O-type structured grid
# =========================================================================
print("Loading grid file and initializing grids...")
grid_setup_start = time.time()

# Load O-grid coordinates (X and Y coordinates)
x, y = load_grid('grid.npz', ['x', 'y'])

# Define number of point for O-grid (2D structured grid)
nx, ny = np.shape(x)
print(f"    Grid dimensions: {nx} x {ny}")

# Compute metric terms
metricterm = metric_terms(grid_x=x, grid_y=y)

grid_setup_time = time.time() - grid_setup_start
print(f"Grid loading completed in {grid_setup_time:.2f} seconds")

# =========================================================================
# TIME SERIES PROCESSING SETUP
# Calculate number of files to process and create file list
# =========================================================================
print("Setting up time series processing...")

# Calculate number of output files to process
nqout = int((last_qout - first_qout) / skip_step_qout) + 1
qouts = range(first_qout, last_qout + 1, skip_step_qout)

print(f"Processing {nqout} files from qout {first_qout} to {last_qout} (step: {skip_step_qout})")

# =========================================================================
# VELOCITY (N,T) LOADING AND MEAN COMPUTING
# Read noraml and tangential velocity fields and compute their mean over 
# time
# =========================================================================
print("Loading velocity field and computing mean...")
velocity_setup_start = time.time()

# Velocities shape
shape = nx, ny, nqout

# Load normal and tangential velocity fields
vn, vt = compute_velocity(velocity_nt_filenames=['velocity_n.npy', 'velocity_t.npy'],
                          velocity_xy_filenames=['velocity_x.npy', 'velocity_y.npy'],
                          metric_terms=metricterm, velocity_shape=shape,
                          load=False, save=True)

# Compute normal and tangential velocity time average (mean)
vn_bar = np.mean(vn, axis=-1)
vt_bar = np.mean(vt, axis=-1)

# Save normal and tangential velocity time average fields
file_python = flow_path_python + 'mean_velocity_x.npy'
np.save(file_python, vn_bar)
file_python = flow_path_python + 'mean_velocity_y.npy'
np.save(file_python, vt_bar)

# Clean memory
del(vn)
del(vt)
del(vn_bar)
del(vt_bar)

velocity_setup_time = time.time() - velocity_setup_start
print(f"Velocity mean completed in {velocity_setup_time:.2f} seconds")

# =========================================================================
# SHEAR VISCOSITY LOADING AND MEAN COMPUTING
# Read shear viscosity field and compute shear viscosity mean over time
# =========================================================================
print("Loading shear viscosity field and computing mean...")
shear_viscosity_setup_start = time.time()

# Load shear viscosity field
mu = load_shear_viscosity(shear_viscosity_filename='shear_viscosity.npy', 
                          temperature_filename='temperature.npy',
                          load=False, save=True)

# Compute shear viscosity time average (mean)
mu_bar = np.mean(mu, axis=-1)

# Save shear viscosity time average field
file_python = flow_path_python + 'mean_shear_viscosity.npy'
np.save(file_python, mu_bar)

# Clean memory
del(mu)
del(mu_bar)

shear_viscosity_setup_time = time.time() - shear_viscosity_setup_start
print(f"Shear viscosity mean completed in {shear_viscosity_setup_time:.2f} seconds")

# =========================================================================
# WALL SHEAR STRESS LOADING AND MEAN COMPUTING
# Read wall shear stress field and compute wall shear stress mean over time
# =========================================================================
print("Loading wall shear stress field and computing mean...")
shear_stress_setup_start = time.time()

# Load wall shear stress field
tal_wall = load_shear_stress(shear_stress_filename='shear_stress.npy', 
                             grid_filename='grid.npz',
                             shear_viscosity_filename='shear_viscosity.npy',
                             velocity_nt_filename=['velocity_n.npy', 'velocity_t.npy'],
                             load=False, save=True)

print('wall =', np.shape(tal_wall))

# Compute wall shear stress time average (mean)
tal_wall_bar = np.mean(tal_wall, axis=-1)

# Save wall shear stress time average field
file_python = flow_path_python + 'mean_shear_stress.npy'
np.save(file_python, tal_wall_bar)

# Clean memory
del(tal_wall)
del(tal_wall_bar)

shear_stress_setup_time = time.time() - shear_stress_setup_start
print(f"Wall shear stress mean completed in {shear_stress_setup_time:.2f} seconds")

# =========================================================================
# SKIN FRICTION LOADING AND MEAN COMPUTING
# Read skin friction field and compute skin friction mean over time
# =========================================================================
print("Loading skin friction field and computing mean...")
skin_friction_setup_start = time.time()

# Load skin friction field
c_f = load_skin_friction(skin_friction_filename='skin_friction.npy', 
                         shear_stress_filename='shear_stress.npy',
                         load=False, save=True)

# Compute skin friction time average (mean)
c_f_bar = np.mean(c_f, axis=-1)

# Save skin friction time average field
file_python = flow_path_python + 'mean_skin_friction.npy'
np.save(file_python, c_f_bar)

# Clean memory
del(c_f)
del(c_f_bar)

skin_friction_setup_time = time.time() - skin_friction_setup_start
print(f"Skin friction mean completed in {skin_friction_setup_time:.2f} seconds")

# =========================================================================
# EXECUTION SUMMARY
# Display comprehensive timing information and script completion status
# =========================================================================
total_execution_time = time.time() - script_start_time

print("\n" + "="*73)
print("CFD DATA PROCESSING COMPLETED SUCCESSFULLY")
print("="*73)
print(f"Grid setup time:                    {grid_setup_time:.2f} seconds")
print(f"Velocity mean calculation:          {velocity_setup_time:.2f} seconds")
print(f"Shear viscosity mean calculation:   {shear_viscosity_setup_time:.2f} seconds")
print(f"Wall shear stress mean calculation: {shear_stress_setup_time:.2f} seconds")
print(f"Skin friction mean calculation:     {skin_friction_setup_time:.2f} seconds")
print("-"*73)
print(f"TOTAL EXECUTION TIME:               {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
print("="*73)
