import numpy as np
import scipy as sp
import time
import gc

from filepath import *
from flow_properties import *
from flow_auxiliary import *

# =========================================================================
# CFD DATA READING AND PROCESSING SCRIPT
# This script processes flow properties: read flow properties and calculate
# and save other flow properties in Python/Numpy readable format (.npy files)
#  for further analysis
# =========================================================================

print("Starting CFD data reading and processing...")
script_start_time = time.time()

# # =========================================================================
# # GRID LOADING
# # Load O-grid for further analysis
# # =========================================================================
# print("Loading O-grid...")
# grid_setup_start = time.time()

# # Load O-grid from Python/Numpy file
# grid_file = flow_path_python + 'o_grid.npz'
# grid = np.load(grid_file)
# x = grid['x']
# y = grid['y']

# # Clean memory
# del(grid)
# gc.collect()

# grid_setup_time = time.time() - grid_setup_start

# # =========================================================================
# # DENSITY LOADING
# # Load flow density for further analysis
# # =========================================================================
# print("Loading density...")
# density_setup_start = time.time()

# # Load density from Python/Numpy file
# density_file = flow_path_python + 'density_o.npy'
# density = np.load(density_file)

# density_setup_time = time.time() - density_setup_start

# # =========================================================================
# # MOMENTUM LOADING
# # Load flow momentum for further analysis
# # =========================================================================
# print("Loading momentum...")
# momentum_setup_start = time.time()

# # Load momentum from Python/Numpy file
# momentum_x_file = flow_path_python + 'momentum_x_o.npy'
# momentum_y_file = flow_path_python + 'momentum_y_o.npy'
# momentum_x = np.load(momentum_x_file)
# momentum_y = np.load(momentum_y_file)

# momentum_setup_time = time.time() - momentum_setup_start

# # =========================================================================
# # PRESSURE LOADING
# # Load flow pressure for further analysis
# # =========================================================================
# print("Loading pressure...")
# pressure_setup_start = time.time()

# # Load pressure from Python/Numpy file
# pressure_file = flow_path_python + 'pressure_o.npy'
# pressure = np.load(pressure_file)

# pressure_setup_time = time.time() - pressure_setup_start

# # =========================================================================
# # COMPUTE VELOCITIES
# # Compute flow x-velocity and y-velocity from density, x-momentum and
# # y-momentum
# # =========================================================================
# print("Compute and save velocities (x,y)...")
# velocity_comp1_start = time.time()

# # Compute x-velocity from density and x-momentum
# velocity_x = calculate_velocity(density, momentum_x)
# del(momentum_x)
# gc.collect()

# # Compute y-velocity from density and x-momentum
# velocity_y = calculate_velocity(density, momentum_y)
# del(momentum_y)
# gc.collect()

# # Save x-velocity fields
# file_python = flow_path_python + 'velocity_x_o.npy'
# np.save(file_python, velocity_x)

# # Save y-velocity fields
# file_python = flow_path_python + 'velocity_y_o.npy'
# np.save(file_python, velocity_y)

# velocity_comp1_time = time.time() - velocity_comp1_start

# # =========================================================================
# # VELOCITY LOADING
# # Load flow velocity for further analysis
# # =========================================================================
# print("Loading velocity...")
# velocity_setup1_start = time.time()

# # Load velocity from Python/Numpy file
# velocity_x_file = flow_path_python + 'velocity_x_o.npy'
# velocity_y_file = flow_path_python + 'velocity_y_o.npy'
# velocity_x = np.load(velocity_x_file)
# velocity_y = np.load(velocity_y_file)

# velocity_setup1_time = time.time() - velocity_setup1_start

# # =========================================================================
# # COMPUTE TEMPERATURE
# # Compute flow temperature from density and pressure
# # =========================================================================
# print("Compute and save temperature...")
# temperature_comp_start = time.time()

# # Compute temperature from density and pressure
# temperature = calculate_temperature(density, pressure)
# del(density)
# del(pressure)
# gc.collect()

# # Save temperature fields
# file_python = flow_path_python + 'temperature_o.npy'
# np.save(file_python, temperature)

# temperature_comp_time = time.time() - temperature_comp_start

# # =========================================================================
# # TEMPERATURE LOADING
# # Load flow temperature for further analysis
# # =========================================================================
# print("Loading temperature...")
# temperature_setup_start = time.time()

# # Load temperature from Python/Numpy file
# temperature_file = flow_path_python + 'temperature_o.npy'
# temperature = np.load(temperature_file)

# temperature_setup_time = time.time() - temperature_setup_start

# # =========================================================================
# # COMPUTE VELOCITIES
# # Compute flow normal and tangential velocity from x-velocity and y-velocity
# # =========================================================================
# print("Compute and save velocities (n,y)...")
# velocity_comp2_start = time.time()

# # Compute normal and tangential velocity from x/y-velocity
# velocity_n, velocity_t = transform_velocity_to_normal_tangential(x, y, velocity_x, velocity_y)
# del(velocity_x)
# del(velocity_y)
# gc.collect()

# # Save normal-velocity fields
# file_python = flow_path_python + 'velocity_n_o.npy'
# np.save(file_python, velocity_n)

# # Save tangential-velocity fields
# file_python = flow_path_python + 'velocity_t_o.npy'
# np.save(file_python, velocity_t)

# velocity_comp2_time = time.time() - velocity_comp2_start

# # =========================================================================
# # VELOCITY LOADING
# # Load flow velocity for further analysis
# # =========================================================================
# print("Loading velocity...")
# velocity_setup2_start = time.time()

# # Load velocity from Python/Numpy file
# velocity_n_file = flow_path_python + 'velocity_n_o.npy'
# velocity_t_file = flow_path_python + 'velocity_t_o.npy'
# velocity_n = np.load(velocity_n_file)
# velocity_t = np.load(velocity_t_file)

# velocity_setup2_time = time.time() - velocity_setup2_start

# # =========================================================================
# # COMPUTE SHEAR VISCOSITY
# # Compute shear viscosity from temperature
# # =========================================================================
# print("Compute and save shear viscosity...")
# shear_visc_comp_start = time.time()

# # Compute shear viscosity from temperature
# shear_viscosity = calculate_shear_viscosity(temperature)
# del(temperature)
# gc.collect()

# # Save shear viscosity fields
# file_python = flow_path_python + 'shear_viscosity_o.npy'
# np.save(file_python, shear_viscosity)

# shear_visc_comp_time = time.time() - shear_visc_comp_start

# # =========================================================================
# # SHEAR VISCOSITY LOADING
# # Load flow shear viscosity for further analysis
# # =========================================================================
# print("Loading shear viscosity...")
# shear_visc_setup_start = time.time()

# # Load shear viscosity from Python/Numpy file
# shear_viscosity_file = flow_path_python + 'shear_viscosity_o.npy'
# shear_viscosity = np.load(shear_viscosity_file)

# shear_visc_setup_time = time.time() - shear_visc_setup_start

# # =========================================================================
# # COMPUTE WALL SHEAR STRESS
# # Compute wall shear stress from tangential-velocity and shear viscosity
# # =========================================================================
# print("Compute and save shear stress...")
# shear_stress_comp_start = time.time()

# # Compute wall shear stress from tangential-velocity and shear viscosity
# shear_stress = calculate_wall_shear_stress(x, y, velocity_t, shear_viscosity)
# del(velocity_n)
# del(velocity_t)
# gc.collect()

# # Save normal-velocity fields
# file_python = flow_path_python + 'wall_shear_stress_o.npy'
# np.save(file_python, shear_stress)

# shear_stress_comp_time = time.time() - shear_stress_comp_start

# # =========================================================================
# # WALL SHEAR STRESS LOADING
# # Load flow wall shear stress for further analysis
# # =========================================================================
# print("Loading shear stress...")
# shear_stress_setup_start = time.time()

# # Load wall shear stress from Python/Numpy file
# shear_stress_file = flow_path_python + 'wall_shear_stress_o.npy'
# shear_stress = np.load(shear_stress_file)

# shear_stress_setup_time = time.time() - shear_stress_setup_start

# # =========================================================================
# # COMPUTE SKIN FRICTION
# # Compute skin friction from wall shear stress
# # =========================================================================
# print("Compute and save skin friction...")
# skin_friction_comp_start = time.time()

# # Compute skin friction from wall shear stress
# skin_friction = calculate_skin_friction(shear_stress)
# del(shear_stress)
# gc.collect()

# # Save normal-velocity fields
# file_python = flow_path_python + 'skin_friction_o.npy'
# np.save(file_python, skin_friction)

# skin_friction_comp_time = time.time() - skin_friction_comp_start

# # =========================================================================
# # SKIN FRICTION LOADING
# # Load flow skin friction for further analysis
# # =========================================================================
# print("Loading skin friction...")
# skin_friction_setup_start = time.time()

# # Load skin friction from Python/Numpy file
# skin_friction_file = flow_path_python + 'skin_friction_o.npy'
# skin_friction = np.load(skin_friction_file)

# skin_friction_setup_time = time.time() - skin_friction_setup_start

# =========================================================================
# EXECUTION SUMMARY
# Display comprehensive timing information and script completion status
# =========================================================================
total_execution_time = time.time() - script_start_time

print("\n" + "="*73)
print("CFD DATA PROCESSING COMPLETED SUCCESSFULLY")
print("="*73)
# print(f"Loading grid:              {grid_setup_time:.2f} seconds")
# print(f"Loading density:           {density_setup_time:.2f} seconds")
# print(f"Loading momentum:          {momentum_setup_time:.2f} seconds")
# print(f"Loading pressure:          {pressure_setup_time:.2f} seconds")
# print(f"Computing velocity (x,y):  {velocity_comp1_time:.2f} seconds")
# print(f"Loading velocity (x,y):    {velocity_setup1_time:.2f} seconds")
# print(f"Computing temperature:     {temperature_comp_time:.2f} seconds")
# print(f"Loading temperature:       {temperature_setup_time:.2f} seconds")
# print(f"Computing velocity (n,t):  {velocity_comp2_time:.2f} seconds")
# print(f"Loading velocity (n,t):    {velocity_setup2_time:.2f} seconds")
# print(f"Computing shear viscosity: {shear_visc_comp_time:.2f} seconds")
print(f"Loading shear viscosity:   {shear_visc_setup_time:.2f} seconds")
# print(f"Computing shear stress:    {shear_stress_comp_time:.2f} seconds")
print(f"Loading shear stress:    {shear_stress_setup_time:.2f} seconds")
# print(f"Computing skin friction:   {skin_friction_comp_time:.2f} seconds")
print(f"Loading skin friction:   {skin_friction_setup_time:.2f} seconds")
print("-"*73)
print(f"TOTAL EXECUTION TIME:      {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
print("="*73)
