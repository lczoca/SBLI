import numpy as np
import time
import sys

from filepath import *
from flow_properties import *
from flow_compute import *

# =========================================================================
# CFD DATA PROCESSING AND AVERAGING SCRIPT
# This script processes multiple files to compute time-averaged, 
# time-flutuation and time-RMS flow properties for computational O-grid
# =========================================================================

# =========================================================================
# GRID FILE AND TIME SIRIES PROCESSING SETUP
# Read dimensions and coordinates for the O-type structured grid and
# calculate the number of files to process
# =========================================================================
def dimension_aux():
    """
    Function used to compute the grid dimension (nx, ny), the metric terms
    and the number of snapshots. 
    """
    # Load O-grid coordinates (X and Y coordinates)
    x, y = load_grid('grid.npz', ['x', 'y'])

    # Define number of point for O-grid (2D structured grid)
    nx, ny = np.shape(x)
    print(f"Grid dimensions: {nx} x {ny}")

    # Compute metric terms
    metricterm = metric_terms(grid_x=x, grid_y=y)

    # Calculate number of output files to process
    nqout = int((last_qout - first_qout) / skip_step_qout) + 1

    return nx, ny, metricterm, nqout

# =========================================================================
# DENSITY LOADING AND MEAN COMPUTING
# Read density field and compute density mean, flutuation and RMS over time
# =========================================================================
def mean_density():
    """
    Function used to compute the mean values, fluctuations 
    and rms value of the density field. 
    """

    print("=" * 50)
    print("Loading density field and computing mean...")
    density_setup_start = time.time()

    # -------------------------------------
    # Load density field
    rho = load_density(density_filename='density.npy')

    # -------------------------------------
    # Compute density time average (mean)
    density_setup_temp = time.time()
    rho_bar = np.mean(rho, axis=-1)

    # Save density time average field
    file_python = flow_path_python + 'mean_density.npy'
    np.save(file_python, rho_bar)

    density_setup_time = time.time() - density_setup_temp
    print(f"    Density mean completed in {density_setup_time:.2f} seconds")

    # -------------------------------------
    # Compute density time flutuation
    density_setup_temp = time.time()
    rho_prime = rho - rho_bar[:, :, np.newaxis]

    # Save density time flutuation field
    file_python = flow_path_python + 'flutuation_density.npy'
    np.save(file_python, rho_prime)

    density_setup_time = time.time() - density_setup_temp
    print(f"    Density flutuation completed in {density_setup_time:.2f} seconds")

    # -------------------------------------
    # Compute density time RMS
    density_setup_temp = time.time()
    rho_rms = np.sqrt(np.mean(rho_prime**2, axis=-1))

    # Save density time RMS field
    file_python = flow_path_python + 'rms_density.npy'
    np.save(file_python, rho_rms)

    density_setup_time = time.time() - density_setup_temp
    print(f"    Density RMS completed in {density_setup_time:.2f} seconds")

    # -------------------------------------
    density_setup_time = time.time() - density_setup_start
    print(f"  Density completed in {density_setup_time:.2f} seconds")
    print("=" * 50)

    return density_setup_time

# =========================================================================
# PRESSURE LOADING AND MEAN COMPUTING
# Read pressure field and compute pressure mean over time
# =========================================================================
def mean_pressure():
    """
    Function used to compute the mean values, fluctuations 
    and rms value of the pressure field. 
    """

    print("=" * 50)
    print("Loading pressure field and computing mean...")
    pressure_setup_start = time.time()

    # -------------------------------------
    # Load pressure field
    P = load_pressure(pressure_filename='pressure.npy')

    # -------------------------------------
    # Compute pressure time average (mean)
    pressure_setup_temp = time.time()
    P_bar = np.mean(P, axis=-1)

    # Save pressure time average field
    file_python = flow_path_python + 'mean_pressure.npy'
    np.save(file_python, P_bar)

    pressure_setup_time = time.time() - pressure_setup_temp
    print(f"    Pressure mean completed in {pressure_setup_time:.2f} seconds")

    # -------------------------------------
    # Compute pressure time flutuation
    pressure_setup_temp = time.time()
    P_prime = P - P_bar[:, :, np.newaxis]

    # Save pressure time flutuation field
    file_python = flow_path_python + 'flutuation_pressure.npy'
    np.save(file_python, P_prime)

    pressure_setup_time = time.time() - pressure_setup_temp
    print(f"    Pressure flutuation completed in {pressure_setup_time:.2f} seconds")

    # -------------------------------------
    # Compute pressure time RMS
    pressure_setup_temp = time.time()
    P_rms = np.sqrt(np.mean(P_prime**2, axis=-1))

    # Save pressure time RMS field
    file_python = flow_path_python + 'rms_pressure.npy'
    np.save(file_python, P_rms)

    pressure_setup_time = time.time() - pressure_setup_temp
    print(f"    Pressure RMS completed in {pressure_setup_time:.2f} seconds")

    # -------------------------------------
    pressure_setup_time = time.time() - pressure_setup_start
    print(f"  Pressure completed in {pressure_setup_time:.2f} seconds")
    print("=" * 50)

    return pressure_setup_time

# =========================================================================
# MOMENTUM (X,Y) LOADING AND MEAN COMPUTING
# Read momentum-X and momentum-Y fields and compute their mean over time
# =========================================================================
def mean_momentum():
    """
    Function used to compute the mean values, fluctuations 
    and rms value of the momentum-X and momentum-Y field. 
    """

    print("=" * 50)
    print("Loading momentum field and computing mean...")
    momentum_setup_start = time.time()

    # -------------------------------------
    # Load momentum-X and momentum-Y fields
    mx, my = load_momentum(momentum_filenames=['momentum_x.npy', 'momentum_y.npy'])

    # -------------------------------------
    # Compute momentum-X and momentum-Y time average (mean)
    momentum_setup_temp = time.time()
    mx_bar = np.mean(mx, axis=-1)
    my_bar = np.mean(my, axis=-1)

    # Save momentum-X and momentum-Y time average fields
    file_python = flow_path_python + 'mean_momentum_x.npy'
    np.save(file_python, mx_bar)
    file_python = flow_path_python + 'mean_momentum_y.npy'
    np.save(file_python, my_bar)

    momentum_setup_time = time.time() - momentum_setup_temp
    print(f"    Momentum mean completed in {momentum_setup_time:.2f} seconds")

    # -------------------------------------
    # Compute momentum-X and momentum-Y time flutuation
    momentum_setup_temp = time.time()
    mx_prime = mx - mx_bar[:, :, np.newaxis]
    my_prime = my - my_bar[:, :, np.newaxis]

    # Save momentum-X and momentum-Y time flutuation fields
    file_python = flow_path_python + 'flutuation_momentum_x.npy'
    np.save(file_python, mx_prime)
    file_python = flow_path_python + 'flutuation_momentum_y.npy'
    np.save(file_python, my_prime)

    momentum_setup_time = time.time() - momentum_setup_temp
    print(f"    Momentum flutuation completed in {momentum_setup_time:.2f} seconds")

    # -------------------------------------
    # Compute momentum-X and momentum-Y time RMS
    momentum_setup_temp = time.time()
    mx_rms = np.sqrt(np.mean(mx_prime**2, axis=-1))
    my_rms = np.sqrt(np.mean(my_prime**2, axis=-1))

    # Save momentum-X and momentum-Y time RMS fields
    file_python = flow_path_python + 'rms_momentum_x.npy'
    np.save(file_python, mx_rms)
    file_python = flow_path_python + 'rms_momentum_y.npy'
    np.save(file_python, my_rms)

    momentum_setup_time = time.time() - momentum_setup_temp
    print(f"    Momentum RMS completed in {momentum_setup_time:.2f} seconds")

    # -------------------------------------
    momentum_setup_time = time.time() - momentum_setup_start
    print(f"  Momentum completed in {momentum_setup_time:.2f} seconds")
    print("=" * 50)

    return momentum_setup_time

# =========================================================================
# VELOCITY (X,Y) LOADING AND MEAN COMPUTING
# Read velocity-X and velocity-Y fields and compute their mean over time
# =========================================================================
def mean_velocity():
    """
    Function used to compute the mean values, fluctuations 
    and rms value of the velocity-X and velocity-Y field. 
    """

    print("=" * 50)
    print("Loading velocity field and computing mean...")
    velocity_setup_start = time.time()

    # -------------------------------------
    # Load velocity-X and velocity-Y fields
    vx, vy = load_velocity(velocity_filenames=['velocity_x.npy', 'velocity_y.npy'],
                        momentum_filenames=['momentum_x.npy', 'momentum_y.npy'],
                        density_filename='density.npy',
                        load=False, save=True)

    # -------------------------------------
    # Compute velocity-X and velocity-Y time average (mean)
    velocity_setup_temp = time.time()
    vx_bar = np.mean(vx, axis=-1)
    vy_bar = np.mean(vy, axis=-1)

    # Save velocity-X and velocity-Y time average fields
    file_python = flow_path_python + 'mean_velocity_x.npy'
    np.save(file_python, vx_bar)
    file_python = flow_path_python + 'mean_velocity_y.npy'
    np.save(file_python, vy_bar)

    velocity_setup_time = time.time() - velocity_setup_temp
    print(f"    Velocity mean completed in {velocity_setup_time:.2f} seconds")

    # -------------------------------------
    # Compute velocity-X and velocity-Y time flutuation
    velocity_setup_temp = time.time()
    vx_prime = vx - vx_bar[:, :, np.newaxis]
    vy_prime = vy - vy_bar[:, :, np.newaxis]

    # Save velocity-X and velocity-Y time flutuation fields
    file_python = flow_path_python + 'flutuation_velocity_x.npy'
    np.save(file_python, vx_prime)
    file_python = flow_path_python + 'flutuation_velocity_y.npy'
    np.save(file_python, vy_prime)

    velocity_setup_time = time.time() - velocity_setup_temp
    print(f"    Velocity flutuation completed in {velocity_setup_time:.2f} seconds")

    # -------------------------------------
    # Compute velocity-X and velocity-Y time RMS
    velocity_setup_temp = time.time()
    vx_rms = np.sqrt(np.mean(vx_prime**2, axis=-1))
    vy_rms = np.sqrt(np.mean(vy_prime**2, axis=-1))

    # Save velocity-X and velocity-Y time RMS fields
    file_python = flow_path_python + 'rms_velocity_x.npy'
    np.save(file_python, vx_rms)
    file_python = flow_path_python + 'rms_velocity_y.npy'
    np.save(file_python, vy_rms)

    velocity_setup_time = time.time() - velocity_setup_temp
    print(f"    Velocity RMS completed in {velocity_setup_time:.2f} seconds")

    # -------------------------------------
    velocity_setup_time = time.time() - velocity_setup_start
    print(f"  Velocity completed in {velocity_setup_time:.2f} seconds")
    print("=" * 50)

    return velocity_setup_time

# =========================================================================
# TEMPERATURE LOADING AND MEAN COMPUTING
# Read temperature field and compute temperature mean over time
# =========================================================================
def mean_temperature():
    """
    Function used to compute the mean values, fluctuations 
    and rms value of the temperature field. 
    """

    print("=" * 50)
    print("Loading temperature field and computing mean...")
    temperature_setup_start = time.time()

    # -------------------------------------
    # Load temperature field
    T = load_temperature(temperature_filename='temperature.npy',
                        density_filename='density.npy',
                        pressure_filename='pressure.npy',
                        load=False, save=True)

    # -------------------------------------
    # Compute temperature time average (mean)
    temperature_setup_temp = time.time()
    T_bar = np.mean(T, axis=-1)

    # Save temperature time average field
    file_python = flow_path_python + 'mean_temperature.npy'
    np.save(file_python, T_bar)

    temperature_setup_time = time.time() - temperature_setup_temp
    print(f"    Temperature mean completed in {temperature_setup_time:.2f} seconds")

    # -------------------------------------
    # Compute temperature time flutuation
    temperature_setup_temp = time.time()
    T_prime = T - T_bar[:, :, np.newaxis]

    # Save temperature time flutuation field
    file_python = flow_path_python + 'flutuation_temperature.npy'
    np.save(file_python, T_prime)

    temperature_setup_time = time.time() - temperature_setup_temp
    print(f"    Temperature flutuation completed in {temperature_setup_time:.2f} seconds")

    # -------------------------------------
    # Compute temperature time RMS
    temperature_setup_temp = time.time()
    T_rms = np.sqrt(np.mean(T_prime**2, axis=-1))

    # Save temperature time RMS field
    file_python = flow_path_python + 'rms_temperature.npy'
    np.save(file_python, T_rms)

    temperature_setup_time = time.time() - temperature_setup_temp
    print(f"    Temperature RMS completed in {temperature_setup_time:.2f} seconds")

    # -------------------------------------
    temperature_setup_time = time.time() - temperature_setup_start
    print(f"  Temperature completed in {temperature_setup_time:.2f} seconds")
    print("=" * 50)

    return temperature_setup_time

# =========================================================================
# VELOCITY (N,T) LOADING AND MEAN COMPUTING
# Read noraml and tangential velocity fields and compute their mean over 
# time
# =========================================================================
def mean_velocity_nt():
    """
    Function used to compute the mean values and fluctuations
    value of the normal and tangential velocities field. 
    """
    
    print("=" * 50)
    print("Loading velocity field and computing mean...")
    velocity_setup_start = time.time()

    # -------------------------------------
    # Velocities shape
    nx, ny, metricterm, nqout = dimension_aux()
    shape = nx, ny, nqout

    # -------------------------------------
    # Load normal and tangential velocity fields
    vn, vt = compute_velocity(velocity_nt_filenames=['velocity_n.npy', 'velocity_t.npy'],
                              velocity_xy_filenames=['velocity_x.npy', 'velocity_y.npy'],
                              metric_terms=metricterm, velocity_shape=shape,
                              load=False, save=True)

    # -------------------------------------
    # Compute normal and tangential velocity time average (mean)
    velocity_setup_temp = time.time()
    vn_bar = np.mean(vn, axis=-1)
    vt_bar = np.mean(vt, axis=-1)

    # Save normal and tangential velocity time average fields
    file_python = flow_path_python + 'mean_velocity_n.npy'
    np.save(file_python, vn_bar)
    file_python = flow_path_python + 'mean_velocity_t.npy'
    np.save(file_python, vt_bar)

    velocity_setup_time = time.time() - velocity_setup_temp
    print(f"    Velocity mean completed in {velocity_setup_time:.2f} seconds")

    # -------------------------------------
    # Compute normal and tangential velocity time flutuation
    velocity_setup_temp = time.time()
    vn_prime = vn - vn_bar[:, :, np.newaxis]
    vt_prime = vt - vt_bar[:, :, np.newaxis]

    # Save normal and tangential velocity time flutuation fields
    file_python = flow_path_python + 'flutuation_velocity_n.npy'
    np.save(file_python, vn_prime)
    file_python = flow_path_python + 'flutuation_velocity_t.npy'
    np.save(file_python, vt_prime)

    velocity_setup_time = time.time() - velocity_setup_temp
    print(f"    Velocity flutuation completed in {velocity_setup_time:.2f} seconds")

    # -------------------------------------
    velocity_setup_time = time.time() - velocity_setup_start
    print(f"  Velocity completed in {velocity_setup_time:.2f} seconds")
    print("=" * 50)

    return velocity_setup_time

# =========================================================================
# SHEAR VISCOSITY LOADING AND MEAN COMPUTING
# Read shear viscosity field and compute shear viscosity mean over time
# =========================================================================
def mean_shear_viscosity():
    """
    Function used to compute the mean values and fluctuations 
    value of the shear viscosity.
    """

    print("=" * 50)
    print("Loading shear viscosity field and computing mean...")
    shear_viscosity_setup_start = time.time()

    # -------------------------------------
    # Load shear viscosity field
    mu = load_shear_viscosity(shear_viscosity_filename='shear_viscosity.npy', 
                            temperature_filename='temperature.npy',
                            load=False, save=True)

    # -------------------------------------
    # Compute shear viscosity time average (mean)
    shear_viscosity_setup_temp = time.time()
    mu_bar = np.mean(mu, axis=-1)

    # Save shear viscosity time average field
    file_python = flow_path_python + 'mean_shear_viscosity.npy'
    np.save(file_python, mu_bar)

    shear_viscosity_setup_time = time.time() - shear_viscosity_setup_temp
    print(f"    Shear viscosity mean completed in {shear_viscosity_setup_time:.2f} seconds")

    # -------------------------------------
    # Compute shear viscosity time flutuation
    shear_viscosity_setup_temp = time.time()
    mu_prime = mu - mu_bar[:, :, np.newaxis]

    # Save shear viscosity time flutuation field
    file_python = flow_path_python + 'flutuation_shear_viscosity.npy'
    np.save(file_python, mu_prime)

    shear_viscosity_setup_time = time.time() - shear_viscosity_setup_temp
    print(f"    Shear viscosity flutuation completed in {shear_viscosity_setup_time:.2f} seconds")

    # -------------------------------------
    shear_viscosity_setup_time = time.time() - shear_viscosity_setup_start
    print(f"  Shear viscosity completed in {shear_viscosity_setup_time:.2f} seconds")
    print("=" * 50)

    return shear_viscosity_setup_time

# =========================================================================
# EXECUTION AND SUMMARY
# Display comprehensive timing information and script completion status
# =========================================================================

def main_flow_compute():
    """
    Main function used to run all functions in this script.
    """
    print("="*73)
    print("Starting CFD data processing and averaging...")
    script_start_time = time.time()

    # Density
    density_setup_time = mean_density()

    # Pressure
    pressure_setup_time = mean_pressure()

    # Momentum
    momentum_setup_time = mean_momentum()

    # Velocity (X,Y)
    velocity_setup_time_xy = mean_velocity()

    # Temperature
    temperature_setup_time = mean_temperature()

    # Velocity (N,T)
    velocity_setup_time_nt = mean_velocity_nt()

    # Shear viscosity
    shear_viscosity_setup_time = mean_shear_viscosity()

    total_execution_time = time.time() - script_start_time

    print("\n" + "="*73)
    print("CFD DATA PROCESSING COMPLETED SUCCESSFULLY")
    print("="*73)
    print(f"Density mean calculation:         {density_setup_time:.2f} seconds")
    print(f"Pressure mean calculation:        {pressure_setup_time:.2f} seconds")
    print(f"Momentum mean calculation:        {momentum_setup_time:.2f} seconds")
    print(f"Velocity (x,y) mean calculation:  {velocity_setup_time_xy:.2f} seconds")
    print(f"Temperature mean calculation:     {temperature_setup_time:.2f} seconds")
    print(f"Velocity (n,t) mean calculation:  {velocity_setup_time_nt:.2f} seconds")
    print(f"Shear viscosity mean calculation: {shear_viscosity_setup_time:.2f} seconds")
    print("-"*73)
    print(f"TOTAL EXECUTION TIME:             {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
    print("="*73)


if __name__ == "__main__":
    main_flow_compute()
