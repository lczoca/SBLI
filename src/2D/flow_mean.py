import numpy as np
import time

from filepath import *
from flow_properties import *
from flow_compute import *

# =========================================================================
# CFD DATA PROCESSING AND AVERAGING SCRIPT
# This script processes multiple files to compute time-averaged flow
# properties for computational grids (O-grid)
# =========================================================================

print("Starting CFD data processing and averaging...")
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

grid_setup_time = time.time() - grid_setup_start
print(f"Grid loading completed in {grid_setup_time:.2f} seconds")

# =========================================================================
# DENSITY LOADING AND MEAN COMPUTING
# Read density field and compute density mean over time
# =========================================================================
print("Loading density field and computing mean...")
density_setup_start = time.time()

# Load density field
rho = load_density(density_filename='density.npy')

# Compute density time average (mean)
rho_bar = np.mean(rho, axis=-1)

# Save density time average field
file_python = flow_path_python + 'mean_density.npy'
np.save(file_python, rho_bar)

# Clean memory
del(rho)
del(rho_bar)

density_setup_time = time.time() - density_setup_start
print(f"Density mean completed in {density_setup_time:.2f} seconds")

# =========================================================================
# PRESSURE LOADING AND MEAN COMPUTING
# Read pressure field and compute pressure mean over time
# =========================================================================
print("Loading pressure field and computing mean...")
pressure_setup_start = time.time()

# Load pressure field
P = load_pressure(pressure_filename='pressure.npy')

# Compute pressure time average (mean)
P_bar = np.mean(P, axis=-1)

# Save pressure time average field
file_python = flow_path_python + 'mean_pressure.npy'
np.save(file_python, P_bar)

# Clean memory
del(P)
del(P_bar)

pressure_setup_time = time.time() - pressure_setup_start
print(f"Pressure mean completed in {pressure_setup_time:.2f} seconds")

# =========================================================================
# MOMENTUM (X,Y) LOADING AND MEAN COMPUTING
# Read momentum-X and momentum-Y fields and compute their mean over time
# =========================================================================
print("Loading momentum field and computing mean...")
momentum_setup_start = time.time()

# Load momentum-X and momentum-Y fields
mx, my = load_momentum(momentum_filenames=['momentum_x.npy', 'momentum_y.npy'])

# Compute momentum-X and momentum-Y time average (mean)
mx_bar = np.mean(mx, axis=-1)
my_bar = np.mean(my, axis=-1)

# Save momentum-X and momentum-Y time average fields
file_python = flow_path_python + 'mean_momentum_x.npy'
np.save(file_python, mx_bar)
file_python = flow_path_python + 'mean_momentum_y.npy'
np.save(file_python, my_bar)

# Clean memory
del(mx)
del(my)
del(mx_bar)
del(my_bar)

momentum_setup_time = time.time() - momentum_setup_start
print(f"Momentum mean completed in {momentum_setup_time:.2f} seconds")

# =========================================================================
# VELOCITY (X,Y) LOADING AND MEAN COMPUTING
# Read velocity-X and velocity-Y fields and compute their mean over time
# =========================================================================
print("Loading velocity field and computing mean...")
velocity_setup_start = time.time()

# Load velocity-X and velocity-Y fields
vx, vy = load_velocity(velocity_filenames=['velocity_x.npy', 'velocity_y.npy'],
                       momentum_filenames=['momentum_x.npy', 'momentum_y.npy'],
                       density_filename='density.npy',
                       load=False, save=True)

# Compute velocity-X and velocity-Y time average (mean)
vx_bar = np.mean(vx, axis=-1)
vy_bar = np.mean(vy, axis=-1)

# Save velocity-X and velocity-Y time average fields
file_python = flow_path_python + 'mean_velocity_x.npy'
np.save(file_python, vx_bar)
file_python = flow_path_python + 'mean_velocity_y.npy'
np.save(file_python, vy_bar)

# Clean memory
del(vx)
del(vy)
del(vx_bar)
del(vy_bar)

velocity_setup_time = time.time() - velocity_setup_start
print(f"Velocity mean completed in {velocity_setup_time:.2f} seconds")

# =========================================================================
# TEMPERATURE LOADING AND MEAN COMPUTING
# Read temperature field and compute temperature mean over time
# =========================================================================
print("Loading temperature field and computing mean...")
temperature_setup_start = time.time()

# Load temperature field
T = load_temperature(temperature_filename='temperature.npy',
                     density_filename='density.npy',
                     pressure_filename='pressure.npy',
                     load=False, save=True)

# Compute temperature time average (mean)
T_bar = np.mean(T, axis=-1)

# Save temperature time average field
file_python = flow_path_python + 'mean_temperature.npy'
np.save(file_python, T_bar)

# Clean memory
del(T)
del(T_bar)

temperature_setup_time = time.time() - temperature_setup_start
print(f"Temperature mean completed in {temperature_setup_time:.2f} seconds")

# =========================================================================
# EXECUTION SUMMARY
# Display comprehensive timing information and script completion status
# =========================================================================
total_execution_time = time.time() - script_start_time

print("\n" + "="*73)
print("CFD DATA PROCESSING COMPLETED SUCCESSFULLY")
print("="*73)
print(f"Grid setup time:              {grid_setup_time:.2f} seconds")
print(f"Density mean calculation:     {density_setup_time:.2f} seconds")
print(f"Pressure mean calculation:    {pressure_setup_time:.2f} seconds")
print(f"Momentum mean calculation:    {momentum_setup_time:.2f} seconds")
print(f"Velocity mean calculation:    {velocity_setup_time:.2f} seconds")
print(f"Temperature mean calculation: {temperature_setup_time:.2f} seconds")
print("-"*73)
print(f"TOTAL EXECUTION TIME:         {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
print("="*73)
