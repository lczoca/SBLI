import numpy as np
from tqdm import tqdm
import psutil
import time

import CGNS

from filepath import *
from flow_properties import *

# =========================================================================
# CFD DATA PROCESSING AND AVERAGING SCRIPT
# This script processes multiple CGNS files to compute time-averaged flow
# properties for two computational grids (O-grid and H-grid)
# =========================================================================

print("Starting CFD data processing and averaging...")
script_start_time = time.time()

# =========================================================================
# GRID FILE LOADING AND SETUP
# =========================================================================
print("Loading grid file and initializing grids...")
grid_setup_start = time.time()

# Open the mesh/grid file
grid_file = grid_path + grid_name + grid_extension
ifile, nbases = CGNS.open_file_read(grid_file)
ibase = 1

# =========================================================================
# O-GRID MESH READING
# Read dimensions and coordinates for the O-type structured grid
# =========================================================================
print("  Reading O-grid mesh...")
izone = 1  # Zone 1 corresponds to O-grid
idim_o = CGNS.zonedim_read(ifile, ibase, izone)
isize_o, nx_o, ny_o, nz_o = CGNS.zone_size_read(ifile, ibase, izone, idim_o)

# Define index ranges for O-grid (2D structured grid)
ijk_min_o = [1, 1]
ijk_max_o = [nx_o, ny_o]

# Load O-grid coordinates (X and Y coordinates)
xo = CGNS.read_2d_coord("CoordinateX", ifile, ibase, izone, ijk_min_o, ijk_max_o, nx_o, ny_o)
yo = CGNS.read_2d_coord("CoordinateY", ifile, ibase, izone, ijk_min_o, ijk_max_o, nx_o, ny_o)
print(f"    O-grid dimensions: {nx_o} x {ny_o}")

# =========================================================================
# H-GRID MESH READING
# Read dimensions and coordinates for the H-type structured grid
# =========================================================================
print("  Reading H-grid mesh...")
izone = 2  # Zone 2 corresponds to H-grid
idim_h = CGNS.zonedim_read(ifile, ibase, izone)
isize_h, nx_h, ny_h, nz_h = CGNS.zone_size_read(ifile, ibase, izone, idim_h)

# Define index ranges for H-grid (2D structured grid)
ijk_min_h = [1, 1]
ijk_max_h = [nx_h, ny_h]

# Load H-grid coordinates (X and Y coordinates)
xh = CGNS.read_2d_coord("CoordinateX", ifile, ibase, izone, ijk_min_h, ijk_max_h, nx_h, ny_h)
yh = CGNS.read_2d_coord("CoordinateY", ifile, ibase, izone, ijk_min_h, ijk_max_h, nx_h, ny_h)
print(f"    H-grid dimensions: {nx_h} x {ny_h}")

grid_setup_time = time.time() - grid_setup_start
print(f"Grid loading completed in {grid_setup_time:.2f} seconds")

# =========================================================================
# OUTPUT FILES CREATION
# Create CGNS and Python output files for storing averaged results
# =========================================================================
print("Creating output files...")
output_setup_start = time.time()

# Create a CGNS file to store the averaged flow data
mean_file_CGNS = mean_path_CGNS + mean_name + mean_extenrion
CGNS.create_file_cgns(mean_file_CGNS, '2D')

# Write mesh coordinates to the CGNS output file
CGNS.write_2d_coord(mean_file_CGNS, 1, nx_o, ny_o, xo, yo)  # O-grid coordinates
CGNS.write_2d_coord(mean_file_CGNS, 2, nx_h, ny_h, xh, yh)  # H-grid coordinates

# Save mesh coordinates to Python-readable format (.npz files)
mean_file_python = mean_path_python + 'o_grid.npz'
np.savez(mean_file_python, x=xo, y=yo)

mean_file_python = mean_path_python + 'h_grid.npz'
np.savez(mean_file_python, x=xh, y=yh)

output_setup_time = time.time() - output_setup_start
print(f"Output files created in {output_setup_time:.2f} seconds")

# =========================================================================
# FLOW FIELD ARRAYS INITIALIZATION
# Initialize arrays to accumulate flow properties for time averaging
# =========================================================================
print("Initializing flow field arrays...")
arrays_init_start = time.time()

# Initialize accumulation arrays for O-grid flow properties
mean_density_o = np.zeros((nx_o, ny_o))        # Density accumulator
mean_pressure_o = np.zeros((nx_o, ny_o))       # Pressure accumulator
mean_temperature_o = np.zeros((nx_o, ny_o))    # Temperature accumulator
mean_momentumx_o = np.zeros((nx_o, ny_o))      # X-momentum accumulator
mean_momentumy_o = np.zeros((nx_o, ny_o))      # Y-momentum accumulator
mean_velocityx_o = np.zeros((nx_o, ny_o))      # X-velocity accumulator
mean_velocityy_o = np.zeros((nx_o, ny_o))      # Y-velocity accumulator

# Initialize accumulation arrays for H-grid flow properties
mean_density_h = np.zeros((nx_h, ny_h))        # Density accumulator
mean_pressure_h = np.zeros((nx_h, ny_h))       # Pressure accumulator
mean_temperature_h = np.zeros((nx_h, ny_h))    # Temperature accumulator
mean_momentumx_h = np.zeros((nx_h, ny_h))      # X-momentum accumulator
mean_momentumy_h = np.zeros((nx_h, ny_h))      # Y-momentum accumulator
mean_velocityx_h = np.zeros((nx_h, ny_h))      # X-velocity accumulator
mean_velocityy_h = np.zeros((nx_h, ny_h))      # Y-velocity accumulator

# Initialize Iblank arrays (computational mask for overset grids)
iblank_o = np.zeros((nx_o, ny_o))  # O-grid computational mask
iblank_h = np.zeros((nx_h, ny_h))  # H-grid computational mask

arrays_init_time = time.time() - arrays_init_start
print(f"Flow field arrays initialized in {arrays_init_time:.2f} seconds")

# =========================================================================
# TIME SERIES PROCESSING SETUP
# Calculate number of files to process and create file list
# =========================================================================
print("Setting up time series processing...")

# Calculate number of output files to process
nqout = int((last_qout - first_qout) / skip_step_qout) + 1
qouts = range(first_qout, last_qout + 1, skip_step_qout)

print(f"Processing {nqout} files from qout {first_qout} to {last_qout} (step: {skip_step_qout})")

# Initialize process monitoring for memory usage tracking
process = psutil.Process()

# =========================================================================
# MAIN PROCESSING LOOP
# Process each CGNS file and accumulate flow properties for time averaging
# =========================================================================
print("Starting main processing loop...")
main_loop_start = time.time()

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
        ibase = 1

        # =========================================================================
        # O-GRID DATA PROCESSING
        # Read flow variables from O-grid zone and accumulate for averaging
        # =========================================================================
        izone = 1  # O-grid zone

        # Load Iblank mask only from the first file (assuming it's constant)
        if qout == first_qout:
            iblank_o = CGNS.read_2d_flow('Iblank', ifile, ibase, izone, ijk_min_o, ijk_max_o, nx_o, ny_o)

        # Read and accumulate density field
        density_o = CGNS.read_2d_flow('Density', ifile, ibase, izone, ijk_min_o, ijk_max_o, nx_o, ny_o)
        mean_density_o += density_o

        # Read and accumulate pressure field
        pressure_o = CGNS.read_2d_flow('Pressure', ifile, ibase, izone, ijk_min_o, ijk_max_o, nx_o, ny_o)
        mean_pressure_o += pressure_o

        # Calculate and accumulate temperature using ideal gas law: T = P/(ρR)
        mean_temperature_o += pressure_o / (density_o * R)

        # Read and accumulate momentum components
        momentumx_o = CGNS.read_2d_flow('MomentumX', ifile, ibase, izone, ijk_min_o, ijk_max_o, nx_o, ny_o)
        momentumy_o = CGNS.read_2d_flow('MomentumY', ifile, ibase, izone, ijk_min_o, ijk_max_o, nx_o, ny_o)
        mean_momentumx_o[:,:] += momentumx_o
        mean_momentumy_o[:,:] += momentumy_o

        # Calculate and accumulate velocity components: v = momentum/density
        mean_velocityx_o[:,:] += momentumx_o / density_o
        mean_velocityy_o[:,:] += momentumy_o / density_o

        # =========================================================================
        # H-GRID DATA PROCESSING
        # Read flow variables from H-grid zone and accumulate for averaging
        # =========================================================================
        izone = 2  # H-grid zone

        # Load Iblank mask only from the first file (assuming it's constant)
        if qout == first_qout:
            iblank_h = CGNS.read_2d_flow('Iblank', ifile, ibase, izone, ijk_min_h, ijk_max_h, nx_h, ny_h)

        # Read and accumulate density field
        density_h = CGNS.read_2d_flow('Density', ifile, ibase, izone, ijk_min_h, ijk_max_h, nx_h, ny_h)
        mean_density_h += density_h

        # Read and accumulate pressure field
        pressure_h = CGNS.read_2d_flow('Pressure', ifile, ibase, izone, ijk_min_h, ijk_max_h, nx_h, ny_h)
        mean_pressure_h += pressure_h

        # Calculate and accumulate temperature using ideal gas law: T = P/(ρR)
        mean_temperature_h += pressure_h / (density_h * R)

        # Read and accumulate momentum components
        momentumx_h = CGNS.read_2d_flow('MomentumX', ifile, ibase, izone, ijk_min_h, ijk_max_h, nx_h, ny_h)
        momentumy_h = CGNS.read_2d_flow('MomentumY', ifile, ibase, izone, ijk_min_h, ijk_max_h, nx_h, ny_h)
        mean_momentumx_h += momentumx_h
        mean_momentumy_h += momentumy_h

        # Calculate and accumulate velocity components: v = momentum/density
        mean_velocityx_h[:,:] += momentumx_h / density_h
        mean_velocityy_h[:,:] += momentumy_h / density_h

        # Close current CGNS file
        CGNS.close_file(ifile)

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
# TIME AVERAGING COMPUTATION
# Divide accumulated values by number of files to get time averages
# =========================================================================
print("Computing time averages...")
averaging_start = time.time()

# Compute time-averaged flow properties for O-grid
mean_density_o /= nqout
mean_pressure_o /= nqout
mean_temperature_o /= nqout
mean_momentumx_o /= nqout
mean_momentumy_o /= nqout
mean_velocityx_o /= nqout
mean_velocityy_o /= nqout

# Compute time-averaged flow properties for H-grid
mean_density_h /= nqout
mean_pressure_h /= nqout
mean_temperature_h /= nqout
mean_momentumx_h /= nqout
mean_momentumy_h /= nqout
mean_velocityx_h /= nqout
mean_velocityy_h /= nqout

averaging_time = time.time() - averaging_start
print(f"Time averaging completed in {averaging_time:.2f} seconds")

# =========================================================================
# CGNS OUTPUT FILE WRITING
# Write all averaged flow properties to CGNS format output file
# =========================================================================
print("Writing results to CGNS file...")
cgns_write_start = time.time()

# Write computational masks (Iblank) for both grids
CGNS.write_soln_2d(mean_file_CGNS, 1, nx_o, ny_o, iblank_o, 'Iblank')
CGNS.write_soln_2d(mean_file_CGNS, 2, nx_h, ny_h, iblank_h, 'Iblank')

# Write density fields for both grids
CGNS.write_soln_2d(mean_file_CGNS, 1, nx_o, ny_o, mean_density_o, 'Density')
CGNS.write_soln_2d(mean_file_CGNS, 2, nx_h, ny_h, mean_density_h, 'Density')

# Write pressure fields for both grids
CGNS.write_soln_2d(mean_file_CGNS, 1, nx_o, ny_o, mean_pressure_o, 'Pressure')
CGNS.write_soln_2d(mean_file_CGNS, 2, nx_h, ny_h, mean_pressure_h, 'Pressure')

# Write temperature fields for both grids
CGNS.write_soln_2d(mean_file_CGNS, 1, nx_o, ny_o, mean_temperature_o, 'Temperature')
CGNS.write_soln_2d(mean_file_CGNS, 2, nx_h, ny_h, mean_temperature_h, 'Temperature')

# Write momentum fields for both grids
CGNS.write_soln_2d(mean_file_CGNS, 1, nx_o, ny_o, mean_momentumx_o, 'MomentumX')
CGNS.write_soln_2d(mean_file_CGNS, 2, nx_h, ny_h, mean_momentumx_h, 'MomentumX')
CGNS.write_soln_2d(mean_file_CGNS, 1, nx_o, ny_o, mean_momentumy_o, 'MomentumY')
CGNS.write_soln_2d(mean_file_CGNS, 2, nx_h, ny_h, mean_momentumy_h, 'MomentumY')

# Write velocity fields for both grids
CGNS.write_soln_2d(mean_file_CGNS, 1, nx_o, ny_o, mean_velocityx_o, 'VelocityX')
CGNS.write_soln_2d(mean_file_CGNS, 2, nx_h, ny_h, mean_velocityx_h, 'VelocityX')
CGNS.write_soln_2d(mean_file_CGNS, 1, nx_o, ny_o, mean_velocityy_o, 'VelocityY')
CGNS.write_soln_2d(mean_file_CGNS, 2, nx_h, ny_h, mean_velocityy_h, 'VelocityY')

cgns_write_time = time.time() - cgns_write_start
print(f"CGNS file writing completed in {cgns_write_time:.2f} seconds")

# =========================================================================
# PYTHON/NUMPY OUTPUT FILES WRITING
# Save all arrays to Python-readable format (.npy files) for further analysis
# =========================================================================
print("Writing results to Python/NumPy files...")
numpy_write_start = time.time()

# Save computational masks
mean_file_python = mean_path_python + 'iblank_o.npy'
np.save(mean_file_python, iblank_o)
mean_file_python = mean_path_python + 'iblank_h.npy'
np.save(mean_file_python, iblank_h)

# Save density fields
mean_file_python = mean_path_python + 'mean_density_o.npy'
np.save(mean_file_python, mean_density_o)
mean_file_python = mean_path_python + 'mean_density_h.npy'
np.save(mean_file_python, mean_density_h)

# Save pressure fields
mean_file_python = mean_path_python + 'mean_pressure_o.npy'
np.save(mean_file_python, mean_pressure_o)
mean_file_python = mean_path_python + 'mean_pressure_h.npy'
np.save(mean_file_python, mean_pressure_h)

# Save temperature fields
mean_file_python = mean_path_python + 'mean_temperature_o.npy'
np.save(mean_file_python, mean_temperature_o)
mean_file_python = mean_path_python + 'mean_temperature_h.npy'
np.save(mean_file_python, mean_temperature_h)

# Save momentum fields
mean_file_python = mean_path_python + 'mean_momentumx_o.npy'
np.save(mean_file_python, mean_momentumx_o)
mean_file_python = mean_path_python + 'mean_momentumx_h.npy'
np.save(mean_file_python, mean_momentumx_h)
mean_file_python = mean_path_python + 'mean_momentumy_o.npy'
np.save(mean_file_python, mean_momentumy_o)
mean_file_python = mean_path_python + 'mean_momentumy_h.npy'
np.save(mean_file_python, mean_momentumy_h)

# Save velocity fields
mean_file_python = mean_path_python + 'mean_velocityx_o.npy'
np.save(mean_file_python, mean_velocityx_o)
mean_file_python = mean_path_python + 'mean_velocityx_h.npy'
np.save(mean_file_python, mean_velocityx_h)
mean_file_python = mean_path_python + 'mean_velocityy_o.npy'
np.save(mean_file_python, mean_velocityy_o)
mean_file_python = mean_path_python + 'mean_velocityy_h.npy'
np.save(mean_file_python, mean_velocityy_h)

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
print(f"Output files creation:     {output_setup_time:.2f} seconds")
print(f"Array initialization:      {arrays_init_time:.2f} seconds")
print(f"Main processing loop:      {main_loop_time:.2f} seconds")
print(f"Time averaging:            {averaging_time:.2f} seconds")
print(f"CGNS file writing:         {cgns_write_time:.2f} seconds")
print(f"NumPy files writing:       {numpy_write_time:.2f} seconds")
print("-"*73)
print(f"TOTAL EXECUTION TIME:      {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
print(f"Files processed:           {nqout}")
print(f"Average time per file:     {main_loop_time/nqout:.2f} seconds")
print(f"Final memory usage:        {process.memory_info().rss / (1024 ** 2):.1f} MB")
print("="*73)
