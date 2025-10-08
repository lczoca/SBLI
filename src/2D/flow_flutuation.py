import numpy as np
import time
import sys

from filepath import *
from flow_properties import *
from flow_compute import *

# =========================================================================
# CFD DATA PROCESSING AND FLUTUATION SCRIPT
# This script processes multiple files to compute time-flutuation flow
# properties for computational O-grid
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
# DENSITY LOADING AND FLUTUATION COMPUTING
# Read density field and compute density flutuation over time
# =========================================================================
print("Loading density field and computing flutuation...")
density_setup_start = time.time()

# Load density field
rho = load_density(density_filename='density.npy')

# Load density time average (mean)
rho_bar = load_density(density_filename='mean_density.npy')

# Compute density time flutuation
rho_prime = rho - rho_bar[:, :, np.newaxis]

# Save density time flutuation field
file_python = flow_path_python + 'flutuation_density.npy'
np.save(file_python, rho_prime)

# Clean memory
del(rho)
del(rho_bar)
del(rho_prime)

density_setup_time = time.time() - density_setup_start
print(f"Density flutuation completed in {density_setup_time:.2f} seconds")

# =========================================================================
# PRESSURE LOADING AND FLUTUATION COMPUTING
# Read pressure field and compute pressure flutuation over time
# =========================================================================
print("Loading pressure field and computing flutuation...")
pressure_setup_start = time.time()

# Load pressure field
P = load_pressure(pressure_filename='pressure.npy')

# Compute pressure time average (mean)
P_bar = load_pressure(pressure_filename='mean_pressure.npy')

# Compute pressure time flutuation
P_prime = P - P_bar[:, :, np.newaxis]

# Save pressure time flutuation field
file_python = flow_path_python + 'flutuation_pressure.npy'
np.save(file_python, P_prime)

# Clean memory
del(P)
del(P_bar)
del(P_prime)

pressure_setup_time = time.time() - pressure_setup_start
print(f"Pressure flutuation completed in {pressure_setup_time:.2f} seconds")

sys.exit()

# =========================================================================
# MOMENTUM (X,Y) LOADING AND FLUTUATION COMPUTING
# Read momentum-X and momentum-Y fields and compute their flutuation over 
# time
# =========================================================================
print("Loading momentum field and computing flutuation...")
momentum_setup_start = time.time()

# Load momentum-X and momentum-Y fields
mx, my = load_momentum(momentum_filenames=['momentum_x.npy', 'momentum_y.npy'])

# Compute momentum-X and momentum-Y time average (mean)
mx_bar, my_bar = load_momentum(momentum_filenames=['mean_momentum_x.npy', 'mean_momentum_y.npy'])

# Compute momentum-X and momentum-Y time flutuation
mx_prime = mx - mx_bar[:, :, np.newaxis]
my_prime = my - my_bar[:, :, np.newaxis]

# Save momentum-X and momentum-Y time flutuation fields
file_python = flow_path_python + 'flutuation_momentum_x.npy'
np.save(file_python, mx_prime)
file_python = flow_path_python + 'flutuation_momentum_y.npy'
np.save(file_python, my_prime)

# Clean memory
del(mx)
del(my)
del(mx_bar)
del(my_bar)
del(mx_prime)
del(my_prime)

momentum_setup_time = time.time() - momentum_setup_start
print(f"Momentum flutuation completed in {momentum_setup_time:.2f} seconds")

# =========================================================================
# VELOCITY (X,Y) LOADING AND FLUTUATION COMPUTING
# Read velocity-X and velocity-Y fields and compute their flutuation over
# time
# =========================================================================
print("Loading velocity field and computing flutuation...")
velocity_setup_start = time.time()

# Load velocity-X and velocity-Y fields
vx, vy = load_velocity(velocity_filenames=['velocity_x.npy', 'velocity_y.npy'])

# Compute velocity-X and velocity-Y time average (mean)
vx_bar, vy_bar = load_velocity(velocity_filenames=['mean_velocity_x.npy', 'mean_velocity_y.npy'])

# Compute velocity-X and velocity-Y time flutuation
vx_prime = vx - vx_bar[:, :, np.newaxis]
vy_prime = vy - vy_bar[:, :, np.newaxis]

# Save velocity-X and velocity-Y time flutuation fields
file_python = flow_path_python + 'flutuation_velocity_x.npy'
np.save(file_python, vx_prime)
file_python = flow_path_python + 'flutuation_velocity_y.npy'
np.save(file_python, vy_prime)

# Clean memory
del(vx)
del(vy)
del(vx_bar)
del(vy_bar)
del(vx_prime)
del(vy_prime)

velocity_setup_time = time.time() - velocity_setup_start
print(f"Velocity flutuation completed in {velocity_setup_time:.2f} seconds")

# =========================================================================
# TEMPERATURE LOADING AND FLUTUATION COMPUTING
# Read temperature field and compute temperature flutuation over time
# =========================================================================
print("Loading temperature field and computing flutuation...")
temperature_setup_start = time.time()

# Load temperature field
T = load_temperature(temperature_filename='temperature.npy')

# Compute temperature time average (mean)
T_bar = load_temperature(temperature_filename='mean_temperature.npy')

# Compute temperature time flutuation
T_prime = T - T_bar[:, :, np.newaxis]

# Save temperature time flutuation field
file_python = flow_path_python + 'flutuation_temperature.npy'
np.save(file_python, T_prime)

# Clean memory
del(T)
del(T_bar)
del(T_prime)

temperature_setup_time = time.time() - temperature_setup_start
print(f"Temperature flutuation completed in {temperature_setup_time:.2f} seconds")

# =========================================================================
# EXECUTION SUMMARY
# Display comprehensive timing information and script completion status
# =========================================================================
total_execution_time = time.time() - script_start_time

print("\n" + "="*73)
print("CFD DATA PROCESSING COMPLETED SUCCESSFULLY")
print("="*73)
print(f"Grid setup time:                    {grid_setup_time:.2f} seconds")
print(f"Density flutuation calculation:     {density_setup_time:.2f} seconds")
print(f"Pressure flutuation calculation:    {pressure_setup_time:.2f} seconds")
print(f"Momentum flutuation calculation:    {momentum_setup_time:.2f} seconds")
print(f"Velocity flutuation calculation:    {velocity_setup_time:.2f} seconds")
print(f"Temperature flutuation calculation: {temperature_setup_time:.2f} seconds")
print("-"*73)
print(f"TOTAL EXECUTION TIME:               {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
print("="*73)
