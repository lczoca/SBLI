import numpy as np
import sys
import time

import CGNS

# =========================================================================
# CFD DATA READING AND PROCESSING SCRIPT
# This script processes multiple CGNS files to read flow properties for one
# computational grid (O-grid) and save it in Python/Numpy readable format
# (.npy files) for further analysis
# =========================================================================

print("Starting CFD data reading and processing...")
script_start_time = time.time()

# =========================================================================
# FILES AND PATHS
# Paths and file names to read CGNS data
# =========================================================================
print("Setup paths and names...")
path_setup_start = time.time()

# Grid file
grid_path = '/media/lczoca/hd_leonardo2/Hugo_Data/'
grid_name = 'grid_2D'
grid_extension = '.cgns'

# Qout files
qout_path = '/media/lczoca/hd_leonardo2/Hugo_Data/'
qout_name = 'qout'
qout_extension = '_2D.cgns'
first_qout     = 6284
last_qout      = 43809
skip_step_qout = 5

# Maximum number of point to read in coordinates (x, y)
nx_max = 1280
ny_max = 190

# Python/Numpy save path
save_path = '/media/lczoca/hd_leonardo2/Hugo_Data_Processed/Python_data2/'

path_setup_time = time.time() - path_setup_start
print(f"Grid loading completed in {path_setup_time:.2f} seconds")

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
# GRID DIMENSION LOADING AND SETUP
# =========================================================================
print("Loading grid dimension and initializing...")
grid_setup_start = time.time()

# Open the mesh/grid file
grid_file = grid_path + grid_name + grid_extension
ifile, nbases = CGNS.open_file_read(grid_file)
ibase = 1

# =========================================================================
# O-GRID MESH READING
# Read dimensions for the O-type structured grid
# =========================================================================
print("  Reading O-grid mesh dimension...")
izone = 1  # Zone 1 corresponds to O-grid
idim = CGNS.zonedim_read(ifile, ibase, izone)
isize, nx, ny, nz = CGNS.zone_size_read(ifile, ibase, izone, idim)

# Define index ranges for O-grid (2D structured grid)
ijk_min = [1, 1]
ijk_max = [nx_max, ny_max]

# Load O-grid coordinates (X and Y coordinates)
x = CGNS.read_2d_coord("CoordinateX", ifile, ibase, izone, ijk_min, ijk_max, nx_max, ny_max)
y = CGNS.read_2d_coord("CoordinateY", ifile, ibase, izone, ijk_min, ijk_max, nx_max, ny_max)
print(f"    O-grid dimensions: {nx} x {ny}")

# Save mesh coordinates to Python-readable format (.npz files)
save_file = save_path + 'grid.npz'
np.savez(save_file, x=x, y=y)

grid_setup_time = time.time() - grid_setup_start
print(f"Grid loading completed in {grid_setup_time:.2f} seconds")

# =========================================================================
# FLOW FIELD ARRAYS INITIALIZATION
# Initialize arrays to save flow properties over time
# =========================================================================
print("Initializing flow field arrays...")
arrays_init_start = time.time()

# Initialize save arrays for O-grid flow properties
density = np.zeros((nx_max, ny_max, nqout))        # Density accumulator
pressure = np.zeros((nx_max, ny_max, nqout))       # Pressure accumulator
momentumx = np.zeros((nx_max, ny_max, nqout))      # X-momentum accumulator
momentumy = np.zeros((nx_max, ny_max, nqout))      # Y-momentum accumulator

# Initialize save array for simulation time
times = np.zeros((nqout))

arrays_init_time = time.time() - arrays_init_start
print(f"Flow field arrays initialized in {arrays_init_time:.2f} seconds")

# =========================================================================
# MAIN PROCESSING LOOP
# Process each CGNS file and save flow properties over time
# =========================================================================
print("Starting main processing loop...")
main_loop_start = time.time()

# Counter
nq = 0

# Loop over all CGNS output files
for i, qout in enumerate(qouts):
    file_start_time = time.time()

    # Format qout number as 6-digit string and construct file path
    nqout_str = str(qout).zfill(6)
    file_path_name = qout_path + qout_name + nqout_str + qout_extension
    print(f"Reading {qout_name + nqout_str}", end=" -> ")

    # Open current CGNS file for reading
    ifile, nbases = CGNS.open_file_read(file_path_name)

    # =========================================================================
    # O-GRID DATA PROCESSING
    # Read flow variables from O-grid zone and save
    # =========================================================================

    # Read and save simulation time
    times[nq] = CGNS.descriptors_read(ifile)

    # Read and save density field
    density[:,:,nq] = CGNS.read_2d_flow('Density', ifile, ibase, izone, ijk_min, ijk_max, nx_max, ny_max)

    # Read and accumulate pressure field
    pressure[:,:,nq] = CGNS.read_2d_flow('Pressure', ifile, ibase, izone, ijk_min, ijk_max, nx_max, ny_max)

    # Read and accumulate momentum components
    momentumx[:,:,nq] = CGNS.read_2d_flow('MomentumX', ifile, ibase, izone, ijk_min, ijk_max, nx_max, ny_max)
    momentumy[:,:,nq] = CGNS.read_2d_flow('MomentumY', ifile, ibase, izone, ijk_min, ijk_max, nx_max, ny_max)

    # Close current CGNS file
    CGNS.close_file(ifile)

    # Update counter
    nq += 1

    # Calculate processing time for this file
    file_time = time.time() - file_start_time
    print(f"Time to read file: {file_time:.2f} seconds")

main_loop_time = time.time() - main_loop_start
print(f"Main processing loop completed in {main_loop_time:.2f} seconds")
print(f"Average time per file: {main_loop_time/nqout:.2f} seconds")

# =========================================================================
# PYTHON/NUMPY OUTPUT FILES WRITING
# Save all arrays to Python-readable format (.npy files) for further analysis
# =========================================================================
print("Writing results to Python/NumPy files...")
numpy_write_start = time.time()

# Save density fields
file_python = save_path + 'time.npy'
np.save(file_python, times)

# Save density fields
file_python = save_path + 'density.npy'
np.save(file_python, density)

# Save pressure fields
file_python = save_path + 'pressure.npy'
np.save(file_python, pressure)

# Save momentum fields
file_python = save_path + 'momentum_x.npy'
np.save(file_python, momentumx)
file_python = save_path + 'momentum_y.npy'
np.save(file_python, momentumy)

numpy_write_time = time.time() - numpy_write_start
print(f"NumPy files writing completed in {numpy_write_time:.2f} seconds")

# =========================================================================
# EXECUTION SUMMARY
# Display comprehensive timing information and script completion status
# =========================================================================
total_execution_time = time.time() - script_start_time

print("\n" + "="*73)
print("CFD DATA PROCESSING COMPLETED SUCCESSFULLY")
print("="*73)
print(f"Path and name setup time:  {path_setup_time:.2f} seconds")
print(f"Grid setup time:           {grid_setup_time:.2f} seconds")
print(f"Array initialization:      {arrays_init_time:.2f} seconds")
print(f"Main processing loop:      {main_loop_time:.2f} seconds")
print(f"NumPy files writing:       {numpy_write_time:.2f} seconds")
print("-"*73)
print(f"TOTAL EXECUTION TIME:      {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
print(f"Files processed:           {nqout}")
print(f"Average time per file:     {main_loop_time/nqout:.2f} seconds")
print("="*73)
