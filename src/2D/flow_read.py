import numpy as np
from tqdm import tqdm
import psutil
import time

import CGNS

from filepath import *
from flow_properties import *

# =========================================================================
# CFD DATA READING AND PROCESSING SCRIPT
# This script processes multiple CGNS files to read flow properties for one
# computational grid (O-grid) and save it in Python/Numpy readable format
# (.npy files) for further analysis
# =========================================================================

print("Starting CFD data reading and processing...")
script_start_time = time.time()

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
idim_o = CGNS.zonedim_read(ifile, ibase, izone)
isize_o, nx_o, ny_o, nz_o = CGNS.zone_size_read(ifile, ibase, izone, idim_o)

# Maximum number of point to read in coordinates (x, y)
nx_o_max = 1280
ny_o_max = 190

# Define index ranges for O-grid (2D structured grid)
ijk_min_o = [1, 1]
ijk_max_o = [nx_o_max, ny_o_max]

# Load O-grid coordinates (X and Y coordinates)
xo = CGNS.read_2d_coord("CoordinateX", ifile, ibase, izone, ijk_min_o, ijk_max_o, nx_o_max, ny_o_max)
yo = CGNS.read_2d_coord("CoordinateY", ifile, ibase, izone, ijk_min_o, ijk_max_o, nx_o_max, ny_o_max)
print(f"    O-grid dimensions: {nx_o} x {ny_o}")

# Save mesh coordinates to Python-readable format (.npz files)
mean_file_python = mean_path_python + 'o_grid.npz'
np.savez(mean_file_python, x=xo, y=yo)

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
# FLOW FIELD ARRAYS INITIALIZATION
# Initialize arrays to save flow properties over time
# =========================================================================
print("Initializing flow field arrays...")
arrays_init_start = time.time()

# Initialize save arrays for O-grid flow properties
density_o = np.zeros((nx_o_max, ny_o_max, nqout))        # Density accumulator
pressure_o = np.zeros((nx_o_max, ny_o_max, nqout))       # Pressure accumulator
momentumx_o = np.zeros((nx_o_max, ny_o_max, nqout))      # X-momentum accumulator
momentumy_o = np.zeros((nx_o_max, ny_o_max, nqout))      # Y-momentum accumulator

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

# Initialize process monitoring for memory usage tracking
process = psutil.Process()

# Counter
nq = 0

# Generate progress bar with detailed information
with tqdm(total=nqout, desc="Processing qouts") as pbar:

    # Loop over all CGNS output files
    for i, qout in enumerate(qouts):
        file_start_time = time.time()

        # Format qout number as 6-digit string and construct file path
        nqout_str = str(qout).zfill(6)
        file_path_name = qout_path + qout_name + nqout_str + qout_extension

        # Open current CGNS file for reading
        ifile, nbases = CGNS.open_file_read(file_path_name)

        # =========================================================================
        # O-GRID DATA PROCESSING
        # Read flow variables from O-grid zone and save
        # =========================================================================

        # Read and save simulation time
        times[nq] = CGNS.descriptors_read(ifile)

        # Read and save density field
        density_o[:,:,nq] = CGNS.read_2d_flow('Density', ifile, ibase, izone, ijk_min_o, ijk_max_o, nx_o_max, ny_o_max)

        # Read and accumulate pressure field
        pressure_o[:,:,nq] = CGNS.read_2d_flow('Pressure', ifile, ibase, izone, ijk_min_o, ijk_max_o, nx_o_max, ny_o_max)

        # Read and accumulate momentum components
        momentumx_o[:,:,nq] = CGNS.read_2d_flow('MomentumX', ifile, ibase, izone, ijk_min_o, ijk_max_o, nx_o_max, ny_o_max)
        momentumy_o[:,:,nq] = CGNS.read_2d_flow('MomentumY', ifile, ibase, izone, ijk_min_o, ijk_max_o, nx_o_max, ny_o_max)

        # Close current CGNS file
        CGNS.close_file(ifile)

        # Update counter
        nq += 1

        # Calculate processing time for this file
        file_time = time.time() - file_start_time

        # Get current memory usage
        mem_usage = process.memory_info().rss / (1024 ** 2)  # Convert to MB

        # Calculate estimated time remaining
        elapsed_total = time.time() - main_loop_start
        avg_time_per_file = elapsed_total / (i + 1)
        remaining_files = nqout - (i + 1)
        eta_seconds = remaining_files * avg_time_per_file
        eta_minutes = eta_seconds / 60

        # Update progress bar with comprehensive information
        pbar.set_postfix({
            "Qout": nqout_str,
            "Time/file": f"{file_time:.1f}s",
            "RAM": f"{mem_usage:.0f}MB",
            "ETA": f"{eta_minutes:.1f}min"
        })
        pbar.update(1)

main_loop_time = time.time() - main_loop_start
print(f"\nMain processing loop completed in {main_loop_time:.2f} seconds")
print(f"Average time per file: {main_loop_time/nqout:.2f} seconds")

# =========================================================================
# PYTHON/NUMPY OUTPUT FILES WRITING
# Save all arrays to Python-readable format (.npy files) for further analysis
# =========================================================================
print("Writing results to Python/NumPy files...")
numpy_write_start = time.time()

# Save density fields
file_python = flow_path_python + 'time.npy'
np.save(file_python, times)

# Save density fields
file_python = flow_path_python + 'density_o.npy'
np.save(file_python, density_o)

# Save pressure fields
file_python = flow_path_python + 'pressure_o.npy'
np.save(file_python, pressure_o)

# Save momentum fields
file_python = flow_path_python + 'momentumx_o.npy'
np.save(file_python, momentumx_o)
file_python = flow_path_python + 'momentumy_o.npy'
np.save(file_python, momentumy_o)

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
print(f"Grid setup time:           {grid_setup_time:.2f} seconds")
print(f"Array initialization:      {arrays_init_time:.2f} seconds")
print(f"Main processing loop:      {main_loop_time:.2f} seconds")
print(f"NumPy files writing:       {numpy_write_time:.2f} seconds")
print("-"*73)
print(f"TOTAL EXECUTION TIME:      {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
print(f"Files processed:           {nqout}")
print(f"Average time per file:     {main_loop_time/nqout:.2f} seconds")
print(f"Final memory usage:        {process.memory_info().rss / (1024 ** 2):.1f} MB")
print("="*73)
