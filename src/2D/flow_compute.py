import numpy as np

from flow_properties import *
from compact_scheme_10th import *

# ====================================================================================================
# CFD DATA READING AND PROCESSING SCRIPT
# This script processes flow properties: read flow properties or calculate and save other flow
# properties in Python/Numpy readable format (.npy files) for further analysis
# ====================================================================================================

# =========================================================================
# METRIC TERMS COMPUTING
# Compute metric terms for x-y grid
# =========================================================================
def metric_terms(grid_x: np.ndarray, grid_y: np.ndarray) -> tuple:
    """
    Calculate metric transformation coefficients for curvilinear coordinate systems.

    This function computes the metric terms required for transforming derivatives
    between physical coordinates (x, y) and computational coordinates (ξ, η) in
    structured grid systems.

    The transformation relates physical coordinates (x, y) to computational
    coordinates (ξ, η) through:
    - Forward transformation: (x, y) = f(ξ, η)
    - Inverse transformation: (ξ, η) = f⁻¹(x, y)

    Parameters:
    -----------
    grid_x : np.ndarray, shape (nx, ny)
        Physical x-coordinates of the computational grid points.
        Array representing the x-position of each grid point in the physical domain.
    grid_y : np.ndarray, shape (nx, ny)
        Physical y-coordinates of the computational grid points.
        Array representing the y-position of each grid point in the physical domain.
        Must have the same shape as grid_x.

    Returns:
    --------
    dx_dxi : np.ndarray, shape (nx, ny)
        Partial derivative ∂x/∂ξ. Rate of change of x with respect to ξ.
    dx_deta : np.ndarray, shape (nx, ny)
        Partial derivative ∂x/∂η. Rate of change of x with respect to η.
    dy_dxi : np.ndarray, shape (nx, ny)
        Partial derivative ∂y/∂ξ. Rate of change of y with respect to ξ.
    dy_deta : np.ndarray, shape (nx, ny)
        Partial derivative ∂y/∂η. Rate of change of y with respect to η.
    dxi_dx : np.ndarray, shape (nx, ny)
        Partial derivative ∂ξ/∂x. Rate of change of ξ with respect to x.
    dxi_dy : np.ndarray, shape (nx, ny)
        Partial derivative ∂ξ/∂y. Rate of change of ξ with respect to y.
    deta_dx : np.ndarray, shape (nx, ny)
        Partial derivative ∂η/∂x. Rate of change of η with respect to x.
    deta_dy : np.ndarray, shape (nx, ny)
        Partial derivative ∂η/∂y. Rate of change of η with respect to y.
    jacobian_determinant : np.ndarray, shape (nx, ny)
        Determinant of the Jacobian transformation matrix.
        det(J) = (∂x/∂ξ)(∂y/∂η) - (∂x/∂η)(∂y/∂ξ)

    Notes:
    ------
    Mathematical Background:
    ------------------------
    The Jacobian matrix of the coordinate transformation is:

        J = │ ∂x/∂ξ   ∂x/∂η │
            │ ∂y/∂ξ   ∂y/∂η │

    The inverse transformation coefficients are computed using:

        J⁻¹ = (1/det(J)) │  ∂y/∂η   -∂x/∂η │ = │ ∂ξ/∂x   ∂ξ/∂y │
                         │ -∂y/∂ξ    ∂x/∂ξ │   │ ∂η/∂x   ∂η/∂y │

    Dependencies:
    -------------
    Requires the function `compact_scheme_10th` to be available in the namespace.
    This function should implement a 10th-order compact finite difference scheme
    for calculating derivatives.

    Examples:
    ---------
    metrics = calculate_metric_terms(x, y)
    dx_dxi, dx_deta, dy_dxi, dy_deta, dxi_dx, dxi_dy, deta_dx, deta_dy, det_J = metrics

    Raises:
    -------
    ValueError
        If input arrays have different shapes or are not 2D
    NameError
        If the required function `compact_scheme_10th` is not available
    RuntimeError
        If the Jacobian determinant becomes zero or negative (invalid transformation)
    """

    # Validate input arrays
    if grid_x.shape != grid_y.shape:
        raise ValueError("Input arrays grid_x and grid_y must have the same shape")

    if grid_x.ndim != 2:
        raise ValueError("Input arrays must be 2D")

    # Extract grid dimensions
    nx, ny = grid_x.shape

    # Computational grid spacing (uniform grid assumption)
    dxi = 1.0   # Spacing in ξ-direction [-]
    deta = 1.0  # Spacing in η-direction [-]

    # Pre-allocate arrays for metric terms
    dx_dxi = np.zeros((nx, ny))   # ∂x/∂ξ
    dx_deta = np.zeros((nx, ny))  # ∂x/∂η
    dy_dxi = np.zeros((nx, ny))   # ∂y/∂ξ
    dy_deta = np.zeros((nx, ny))  # ∂y/∂η

    # # Compute derivatives in ξ-direction (along axis 0, constant j)
    # for j in range(ny):
    #     dx_dxi[:,j] = compact_scheme_10th(dxi, grid_x[:,j])
    #     dy_dxi[:,j] = compact_scheme_10th(dxi, grid_y[:,j])

    # # Compute derivatives in η-direction (along axis 1, constant i)
    # for i in range(nx):
    #     dx_deta[i,:] = compact_scheme_10th(deta, grid_x[i,:])
    #     dy_deta[i,:] = compact_scheme_10th(deta, grid_y[i,:])

    # Compute derivatives in ξ-direction (along axis 0, constant j)
    dx_dxi = np.apply_along_axis(compact_scheme_10th, 0, grid_x, dxi)
    dy_dxi = np.apply_along_axis(compact_scheme_10th, 0, grid_y, dxi)

    # Compute derivatives in η-direction (along axis 1, constant i)
    dx_deta = np.apply_along_axis(compact_scheme_10th, 1, grid_x, deta)
    dy_deta = np.apply_along_axis(compact_scheme_10th, 1, grid_y, deta)

    # Calculate Jacobian determinant using cross product formula
    # det(J) = ∂x/∂ξ * ∂y/∂η - ∂x/∂η * ∂y/∂ξ
    jacobian_determinant = dx_dxi * dy_deta - dx_deta * dy_dxi

    # Check for invalid transformations (non-positive Jacobian)
    if np.any(jacobian_determinant <= 0):
        raise RuntimeError(
            "Invalid grid transformation detected: Jacobian determinant is "
            "non-positive. This indicates grid folding or degenerate cells."
        )

    # Compute inverse transformation coefficients using Jacobian inverse formula
    # These coefficients transform derivatives from physical to computational space
    dxi_dx = dy_deta / jacobian_determinant    # ∂ξ/∂x = (∂y/∂η) / det(J)
    dxi_dy = -dx_deta / jacobian_determinant   # ∂ξ/∂y = -(∂x/∂η) / det(J)
    deta_dx = -dy_dxi / jacobian_determinant   # ∂η/∂x = -(∂y/∂ξ) / det(J)
    deta_dy = dx_dxi / jacobian_determinant    # ∂η/∂y = (∂x/∂ξ) / det(J)

    return dx_dxi, dx_deta, dy_dxi, dy_deta, dxi_dx, dxi_dy, deta_dx, deta_dy, jacobian_determinant

# =========================================================================
# GRID LOADING
# Load O-grid for further analysis
# =========================================================================
def load_grid(grid_filename: str, coordinates: list) -> tuple:
    """
    Load grid coordinate data from a NumPy binary file.

    This function reads a structured grid stored in NumPy's .npz format and extracts
    the x and y coordinate arrays.

    The function assumes the grid file is located in a predefined directory path
    (specified by the global variable `flow_path_python`) and contains arrays
    named 'x' and 'y' representing the coordinate meshgrid.

    Parameters:
    -----------
    grid_filename : str
        Name of the grid file to load (including extension, typically '.npz').
        The file should be a NumPy archive containing at least two arrays:
        - 'x': x-coordinates of the grid points
        - 'y': y-coordinates of the grid points
    coordinates : list of str
        List containing the names of the coordinate arrays to extract from the .npz file.
        Must contain exactly two string elements in the following order:
        - First element: name of the grid "X" coordinate array key in the .npz file
        - Second element: name of the grid "Y" coordinate array key in the .npz file

    Returns:
    --------
    x_grid : np.ndarray
        2D array containing grid "X" coordinate values for each grid point.
        Shape typically (nx, ny) where nx and ny are the grid dimensions.
    y_grid : np.ndarray
        2D array containing grid "Y" coordinate values for each grid point.
        Shape typically (nx, ny)where nx and ny are the grid dimensions.

    Notes:
    ------
    Global Dependencies:
    - This function relies on a global variable `flow_path_python` that must be
      defined before calling this function. This variable should contain the
      directory path where grid files are stored.
    File Format:
    - Expected format: NumPy .npz archive (compressed or uncompressed)

    Examples:
    ---------
    x, y = load_grid('grid.npz', ['x', 'y'])
    """

    # Construct full path to grid file by combining base directory with filename
    # flow_path_python is expected to be a global variable containing the base directory
    grid_filepath = flow_path_python + grid_filename

    # Load NumPy archive file containing grid coordinate data
    grid_data = np.load(grid_filepath)

    # Extract x-coordinate array from the loaded grid data
    x_grid = grid_data[coordinates[0]]

    # Extract y-coordinate array from the loaded grid data
    y_grid = grid_data[coordinates[1]]

    # Return both coordinate arrays as a tuple
    return x_grid, y_grid

# =========================================================================
# DENSITY LOADING
# Load flow density for further analysis
# =========================================================================
def load_density(density_filename: str) -> np.ndarray:
    """
    Load density field data from a NumPy binary file.

    This function reads density field data stored in NumPy's .npy binary format.

    The function assumes the density file is located in a predefined directory path
    (specified by the global variable `flow_path_python`).

    Parameters:
    -----------
    density_filename : str
        Name of the density file to load (including extension, typically '.npy').
        The file should be a NumPy binary array containing density field data.

    Returns:
    --------
    density_field : np.ndarray
        NumPy array containing the density field data.

    Notes:
    ------
    Global Dependencies:
    - This function relies on a global variable `flow_path_python` that must be
      defined before calling this function. This variable should contain the
      directory path where density files are stored.
    File Format:
    - Expected format: NumPy .npy binary format

    Examples:
    ---------
    rho = load_density('density.npy')
    """

    # Construct full path to density file by combining base directory with filename
    # flow_path_python is expected to be a global variable containing the base directory
    density_filepath = flow_path_python + density_filename

    # Load NumPy binary file containing density field data
    density_field = np.load(density_filepath, mmap_mode='r')

    # Return the loaded density field array
    return density_field

# =========================================================================
# PRESSURE LOADING
# Load flow pressure for further analysis
# =========================================================================
def load_pressure(pressure_filename: str) -> np.ndarray:
    """
    Load pressure field data from a NumPy binary file.

    This function reads pressure field data stored in NumPy's .npy binary format.

    The function assumes the pressure file is located in a predefined directory path
    (specified by the global variable `flow_path_python`).

    Parameters:
    -----------
    pressure_filename : str
        Name of the pressure file to load (including extension, typically '.npy').
        The file should be a NumPy binary array containing pressure field data.

    Returns:
    --------
    pressure_field : np.ndarray
        NumPy array containing the pressure field data.

    Notes:
    ------
    Global Dependencies:
    - This function relies on a global variable `flow_path_python` that must be
      defined before calling this function. This variable should contain the
      directory path where pressure files are stored.
    File Format:
    - Expected format: NumPy .npy binary format

    Examples:
    ---------
    P = load_pressure('pressure.npy')
    """

    # Construct full path to pressure file by combining base directory with filename
    # flow_path_python is expected to be a global variable containing the base directory
    pressure_filepath = flow_path_python + pressure_filename

    # Load NumPy binary file containing pressure field data
    pressure_field = np.load(pressure_filepath, mmap_mode='r')

    # Return the loaded pressure field array
    return pressure_field

# =========================================================================
# MOMENTUM LOADING
# Load flow momentum for further analysis
# =========================================================================
def load_momentum(momentum_filenames: list) -> tuple:
    """
    Load momentum field data from a NumPy binary file.

    This function reads momentum (x and y) field data stored in NumPy's .npy binary
    format.

    The function assumes the momentum file is located in a predefined directory path
    (specified by the global variable `flow_path_python`).

    Parameters:
    -----------
    momentum_filenames : list of str
        Name of the momentum-X and momentum-Y file to load (including extension,
        typically '.npy').
        The file should be a list of NumPy binary array containing momentum-X and
        momentum-Y field data.

    Returns:
    --------
    momentum_X_field : np.ndarray
        NumPy array containing the momentum-X field data.
    momentum_Y_field : np.ndarray
        NumPy array containing the momentum-Y field data.

    Notes:
    ------
    Global Dependencies:
    - This function relies on a global variable `flow_path_python` that must be
      defined before calling this function. This variable should contain the
      directory path where momentum files are stored.
    File Format:
    - Expected format: NumPy .npy binary format

    Examples:
    ---------
    mx, my = load_momentum(['momentum_x.npy', 'momentum_y.npy'])
    """

    # Construct full path to momentum-X and momentum-Y file by combining base directory with filename
    # flow_path_python is expected to be a global variable containing the base directory
    momentum_X_filepath = flow_path_python + momentum_filenames[0]
    momentum_Y_filepath = flow_path_python + momentum_filenames[1]

    # Load NumPy binary file containing momentum-X and momentum-Y field data
    momentum_X_field = np.load(momentum_X_filepath, mmap_mode='r')
    momentum_Y_field = np.load(momentum_Y_filepath, mmap_mode='r')

    # Return the loaded momentum-X and momentum-Y field array
    return momentum_X_field, momentum_Y_field

# =========================================================================
# VELOCITY COMPUTING/LOADING
# Compute flow velocity-X and velocity-Y (from density, momentum-X and
# momentum-Y) or load flow velocity for further analysis
# =========================================================================
def load_velocity(velocity_filenames: list, momentum_filenames: list = None,
                  density_filename: str = None, load: bool = True, save: bool = False) -> tuple:
    """
    Load or compute velocity field data from NumPy binary files.

    This function provides flexible loading of velocity field components (u, v) through
    two methods:
    1. Direct loading: Read pre-computed velocity fields from .npy files
    2. Computation: Calculate velocity from momentum and density fields
       using the relation: velocity = momentum / density

    The function assumes all files are located in a predefined directory path
    (specified by the global variable `flow_path_python`).

    Parameters:
    -----------
    velocity_filenames : list of str
        Name of the velocity-X and velocity-Y file to load (including extension,
        typically '.npy').
        The file should be a list of NumPy binary array containing velocity-X and
        velocity-Y field data.
    momentum_filenames : list of str
        Name of the momentum-X and momentum-Y file to load (including extension,
        typically '.npy').
        The file should be a list of NumPy binary array containing momentum-X and
        momentum-Y field data.
        Note: Must be provided if `load=False`, otherwise raises an error.
    density_filename : str
        Name of the density file to load (including extension, typically '.npy').
        The file should be a NumPy binary array containing density field data.
        Note: Must be provided if `load=False`, otherwise raises an error.
    load : bool, optional (default=True)
        Control flag for loading method:
        - True: Load velocity fields directly from `velocity_filenames`
        - False: Compute velocity from momentum and density fields
        Set to False when velocity fields are not pre-computed but momentum
        and density are available from simulation output.
    save : bool, optional (default=False)
        Control flag for saving computed velocity fields:
        - True: Save computed velocity fields to `velocity_filenames` paths
        - False: Do not save (default behavior)
        Useful when `load=False` to save computed velocities for future use,
        avoiding repeated computation. Has no effect when `load=True`.

    Returns:
    --------
    velocity_X_field : np.ndarray
        NumPy array containing the velocity-X field data.
    velocity_Y_field : np.ndarray
        NumPy array containing the velocity-Y field data.

    Notes:
    ------
    Global Dependencies:
    - This function relies on a global variable `flow_path_python` that must be
      defined before calling this function. This variable should contain the
      directory path where density, momentum and velocity files are stored.
    File Format:
    - Expected format: NumPy .npy binary format

    Examples:
    ---------
    vx, vy = load_velocity(['velocity_x.npy', 'velocity_y.npy'])
    vx, vy = load_velocity(
        ...     velocity_filenames=['velocity_x.npy', 'velocity_y.npy'],
        ...     momentum_filenames=['momentum_x.npy', 'momentum_y.npy'],
        ...     density_filename='density.npy',
        ...     load=False)
    vx, vy = load_velocity(
        ...     velocity_filenames=['velocity_x.npy', 'velocity_y.npy'],
        ...     momentum_filenames=['momentum_x.npy', 'momentum_y.npy'],
        ...     density_filename='density.npy',
        ...     load=False,
        ...     save=True)
    """

    # Construct full path to velocity-X and velocity-Y files by combining base directory with filename
    # flow_path_python is expected to be a global variable containing the base directory
    velocity_X_filepath = flow_path_python + velocity_filenames[0]
    velocity_Y_filepath = flow_path_python + velocity_filenames[1]

    # Direct loading mode: Load pre-computed velocity fields
    if load:
        # Load NumPy binary file containing velocity-X and velocity-Y field data
        velocity_X_field = np.load(velocity_X_filepath, mmap_mode='r')
        velocity_Y_field = np.load(velocity_Y_filepath, mmap_mode='r')

    # Computation mode: Calculate velocity from momentum and density
    else:
        # Load NumPy binary file containing momentum-X and momentum-Y field data
        momentum_X_field, momentum_Y_field = load_momentum(momentum_filenames)
        # Load NumPy binary file containing density field data
        density_field = load_density(density_filename)

        # Compute velocity components using the fundamental relation: velocity = momentum / density
        velocity_X_field = momentum_X_field / density_field
        velocity_Y_field = momentum_Y_field / density_field

    # Optional saving: Write computed velocity fields to disk for future use
    if save:
        # Save velocity-X field to specified filepath
        np.save(velocity_X_filepath, velocity_X_field)
        # Save velocity-Y field to specified filepath
        np.save(velocity_Y_filepath, velocity_Y_field)

    # Return the loaded velocity-X and velocity-Y field array
    return velocity_X_field, velocity_Y_field

# =========================================================================
# TEMPERATURE COMPUTING/LOADING
# Compute flow temperature (from density and pressure) or load flow
# temperature for further analysis
# =========================================================================
def load_temperature(temperature_filename: str, density_filename: str = None,
                     pressure_filename: str = None, load: bool = True, save: bool = False) -> np.ndarray:
    """
    Load or compute temperature field data from NumPy binary files.

    This function provides flexible loading of temperature field through two methods:
    1. Direct loading: Read pre-computed temperature fields from .npy files
    2. Computation: Calculate temperature from density and pressure fields
       using the relation: temperature = pressure / (density * R)

    The function assumes all files are located in a predefined directory path
    (specified by the global variable `flow_path_python`).

    Parameters:
    -----------
    temperature_filename : str
        Name of the temperature file to load (including extension, typically '.npy').
        The file should be a NumPy binary array containing temperature field data.
    density_filename : str
        Name of the density file to load (including extension, typically '.npy').
        The file should be a NumPy binary array containing density field data.
        Note: Must be provided if `load=False`, otherwise raises an error.
    pressure_filename : str
        Name of the pressure file to load (including extension, typically '.npy').
        The file should be a NumPy binary array containing pressure field data.
        Note: Must be provided if `load=False`, otherwise raises an error.
    load : bool, optional (default=True)
        Control flag for loading method:
        - True: Load temperature fields directly from `temperature_filenames`
        - False: Compute temperature from density and pressure fields
        Set to False when temperature fields are not pre-computed but density and
        pressure are available from simulation output.
    save : bool, optional (default=False)
        Control flag for saving computed temperature fields:
            - True: Save computed temperature fields to `temperature_filenames` paths
        - False: Do not save (default behavior)
        Useful when `load=False` to save computed temperature for future use,
        avoiding repeated computation. Has no effect when `load=True`.

    Returns:
    --------
    temperature_field : np.ndarray
        NumPy array containing the temperature field data.

    Notes:
    ------
    Global Dependencies:
    - This function relies on a global variable `flow_path_python` that must be
      defined before calling this function. This variable should contain the
      directory path where density, pressure and temperature files are stored.
    File Format:
    - Expected format: NumPy .npy binary format

    Examples:
    ---------
    T = load_temperature('temperature.npy')
    T = load_temperature(
        ...     temperature_filenames='temperature.npy',
        ...     density_filename='density.npy',
        ...     pressure_filename='pressure.npy',
        ...     load=False)
    T = load_temperature(
        ...     temperature_filenames='temperature.npy',
        ...     density_filename='density.npy',
        ...     pressure_filename='pressure.npy',
        ...     load=False,
        ...     save=True)
    """

    # Construct full path to temperature file by combining base directory with filename
    # flow_path_python is expected to be a global variable containing the base directory
    temperature_filepath = flow_path_python + temperature_filename

    # Direct loading mode: Load pre-computed temperature fields
    if load:
        # Load NumPy binary file containing temperature field data
        temperature_field = np.load(temperature_filepath, mmap_mode='r')

    # Computation mode: Calculate temperature from density and pressure
    else:
        # Load NumPy binary file containing density field data
        density_field = load_density(density_filename)
        # Load NumPy binary file containing pressure field data
        pressure_field = load_pressure(pressure_filename)

        # Compute temperature using ideal gas law rearranged: T = P/(ρR)
        temperature_field = pressure_field / (density_field * R)

    # Optional saving: Write computed temperature field to disk for future use
    if save:
        # Save temperature field to specified filepath
        np.save(temperature_filepath, temperature_field)

    # Return the loaded temperature field array
    return temperature_field

# =========================================================================
# VELOCITY COMPUTING/LOADING
# Compute flow normal and tangential velocities (from velocity-X, velocity-Y,
# and metric terms) or load flow velocity for further analysis
# =========================================================================
def compute_velocity(velocity_nt_filenames: list, velocity_xy_filenames: list = None,
                     metric_terms: tuple = None, velocity_shape: tuple = None,
                     load: bool = True, save: bool = False) -> tuple:
    """
    Load or compute normal and tangential velocities fields data from NumPy binary files.

    This function provides flexible loading of velocity field components in curvilinear coordinates
    (normal and tangential to a surface) through two methods:
    1. Direct loading: Read pre-computed normal and tangential velocity fields from .npy files
    2. Computation: Calculate normal and tangential velocities from velocity-X and velocity-Y fields
    and metric terms from curvilinear coordinate transformation

    The function assumes all files are located in a predefined directory path
    (specified by the global variable `flow_path_python`).

    Parameters:
    -----------
    velocity_nt_filenames : list of str
        Name of the normal and tangetial velocities files to load (including extension,
        typically '.npy').
        The file should be a list of NumPy binary array containing normal and tangential velocities
        field data.
    velocity_xy_filenames : list of str, optional (default=None)
        Name of the velocity-X and velocity-Y file to load (including extension,
        typically '.npy').
        The file should be a list of NumPy binary array containing velocity-X and
        velocity-Y field data.
        Note: Must be provided if `load=False`, otherwise raises an error.
    metric_terms : tuple of np.ndarray, optional (default=None)
        Tuple containing 9 metric term arrays for curvilinear coordinate transformation.
        Required only when `load=False`.
        Expected order: dx_dxi, dx_deta, dy_dxi, dy_deta, dxi_dx, dxi_dy, deta_dx, deta_dy, det_J
        where:
        - dx_dxi, dx_deta: Derivatives of x with respect to ξ and η
        - dy_dxi, dy_deta: Derivatives of y with respect to ξ and η
        - dxi_dx, dxi_dy: Derivatives of ξ with respect to x and y
        - deta_dx, deta_dy: Derivatives of η with respect to x and y
        - det_J: Jacobian determinant of the transformation
        All arrays should have shape (nx, ny) matching the spatial grid dimensions.
        These terms are computed from the grid geometry and define the transformation
        between physical (x,y) and computational (ξ,η) coordinates.
    velocity_shape : tuple of int, optional (default=None)
            Tuple specifying the shape of the velocity field arrays.
            Required only when `load=False`.
            Format: (nx, ny, nt)
            - nx: Number of grid points in ξ-direction
            - ny: Number of grid points in η-direction
            - nt: Number of temporal snapshots
    load : bool, optional (default=True)
        Control flag for loading method:
        - True: Load normal and tangential velocity fields directly from `velocity_nt_filenames`
        - False: Compute normal and tangential velocities from Cartesian components using
                 coordinate transformation with metric terms
        Set to False when velocity fields are not pre-computed but momentum
        and density are available from simulation output.
    save : bool, optional (default=False)
        Control flag for saving computed velocity fields:
        - True: Save computed normal and tangential velocity fields to `velocity_nt_filenames` paths
        - False: Do not save (default behavior)
        Useful when `load=False` to save computed velocities for future use,
        avoiding repeated computation. Has no effect when `load=True`.

    Returns:
    --------
    velocity_N_field : np.ndarray
        NumPy array containing the normal velocity field data.
    velocity_T_field : np.ndarray
        NumPy array containing the tangential velocity field data.

    Notes:
    ------
    Global Dependencies:
    - This function relies on a global variable `flow_path_python` that must be
      defined before calling this function. This variable should contain the
      directory path where density, momentum and velocity files are stored.
    File Format:
    - Expected format: NumPy .npy binary format

    Examples:
    ---------
    vn, vt = compute_velocity(['velocity_n.npy', 'velocity_t.npy'])
    vn, vt = compute_velocity(
        ...     velocity_nt_filenames=['velocity_n.npy', 'velocity_t.npy'],
        ...     velocity_xy_filenames=['velocity_x.npy', 'velocity_y.npy'],
        ...     metric_terms=metrics,
        ...     velocity_shape=(1280, 190, 7506),
        ...     load=False)
    vn, vt = compute_velocity(
        ...     velocity_nt_filenames=['velocity_n.npy', 'velocity_t.npy'],
        ...     velocity_xy_filenames=['velocity_x.npy', 'velocity_y.npy'],
        ...     metric_terms=metrics,
        ...     velocity_shape=(1280, 190, 7506),
        ...     load=False,
        ...     save=True)
    """

    # Construct full path to normal velocity and tangential velocity files (curvilinear coordinates)
    # by combining base directory with filename
    # flow_path_python is expected to be a global variable containing the base directory
    velocity_N_filepath = flow_path_python + velocity_nt_filenames[0]
    velocity_T_filepath = flow_path_python + velocity_nt_filenames[1]

    # Direct loading mode: Load pre-computed velocity fields
    if load:
        # Load NumPy binary file containing normal velocity and tangential velocity fields data
        velocity_N_field = np.load(velocity_N_filepath, mmap_mode='r')
        velocity_T_field = np.load(velocity_T_filepath, mmap_mode='r')

    # Computation mode: Transform Cartesian velocities to curvilinear coordinates
    else:
        # Load NumPy binary file containing velocity-X and velocity-Y field data
        velocity_X_field, velocity_Y_field = load_velocity(velocity_xy_filenames)

        # Unpack metric terms from the tuple
        # These terms define the transformation between physical and computational coordinates
        dx_dxi, dx_deta, dy_dxi, dy_deta, dxi_dx, dxi_dy, deta_dx, deta_dy, det_J = metric_terms

        # Extract dimensions from velocity_shape tuple
        nx, ny, nt = velocity_shape

        # Pre-allocate output arrays for normal and tangential velocity components
        velocity_N_field = np.zeros((nx, ny, nt))
        velocity_T_field = np.zeros((nx, ny, nt))

        # Pre-compute normalization factor (grid-invariant)
        eta_magnitude = np.sqrt(deta_dx**2 + deta_dy**2)

        # Compute normal velocity component (perpendicular to surface)
        # Formula: v_n = (∂η/∂x · u + ∂η/∂y · v) / |∇η|
        velocity_N_field = np.einsum('ij,ijk->ijk', deta_dx / eta_magnitude, velocity_X_field) \
                + np.einsum('ij,ijk->ijk', deta_dy / eta_magnitude, velocity_Y_field)

        # Compute tangential velocity component (parallel to surface)
        # Formula: v_t = (∂η/∂y · u - ∂η/∂x · v) / |∇η|
        velocity_T_field = np.einsum('ij,ijk->ijk', deta_dy / eta_magnitude, velocity_X_field) \
                - np.einsum('ij,ijk->ijk', deta_dx / eta_magnitude, velocity_Y_field)

        # Apply sign correction for the pressure side (second half of domain in ξ-direction)
        # This accounts for opposite surface normal orientation on different sides of the body
        velocity_N_field[nx//2:, :, :] *= -1

    # Optional saving: Write computed velocity fields to disk for future use
    if save:
        # Save normal velocity field to specified filepath
        np.save(velocity_N_filepath, velocity_N_field)
        # Save tangential velocity field to specified filepath
        np.save(velocity_T_filepath, velocity_T_field)

    # Return the loaded normal velocity and tangential velocity fields array
    return velocity_N_field, velocity_T_field

# =========================================================================
# SHEAR VISCOSITY COMPUTING/LOADING
# Compute flow shear viscosity (from temperature) or load flow shear
# viscosity for further analysis
# =========================================================================
def load_shear_viscosity(shear_viscosity_filename: str, temperature_filename: str = None,
                         load: bool = True, save: bool = False) -> np.ndarray:
    """
    Load or compute shear viscosity field data from NumPy binary files.

    This function provides flexible loading of shear viscosity (μ) field through two methods:
    1. Direct loading: Read pre-computed shear viscosity fields from .npy files
    2. Computation: Calculate shear viscosity from temperature field using Sutherland's law

    The function assumes all files are located in a predefined directory path
    (specified by the global variable `flow_path_python`).

    Parameters:
    -----------
    shear_viscosity_filename : str
        Name of the shear viscosity file to load or save (including extension, typically '.npy').
        The file should be a NumPy binary array containing shear viscosity field data.
    temperature_filename : str, optional (default=None)
        Name of the temperature file to load (including extension, typically '.npy').
        The file should be a NumPy binary array containing temperature field data.
        Note: Must be provided if `load=False`, otherwise raises an error.
    load : bool, optional (default=True)
        Control flag for loading method:
        - True: Load shear viscosity field directly from `shear_viscosity_filename`
        - False: Compute shear viscosity from temperature field using Sutherland's law
        Set to False when viscosity fields are not pre-computed but temperature
        field is available from simulation output.
    save : bool, optional (default=False)
        Control flag for saving computed temperature fields:
        - True: Save computed shear viscosity field to `shear_viscosity_filename` path
        - False: Do not save (default behavior)
        Useful when `load=False` to save computed viscosity for future use,
        avoiding repeated computation. Has no effect when `load=True`.

    Returns:
    --------
    shear_viscosity_field : np.ndarray
        NumPy array containing the shear viscosity (μ) field data.

    Notes:
    ------
    Global Dependencies:
    - This function relies on a global variable `flow_path_python` that must be
      defined before calling this function. This variable should contain the
      directory path where density, pressure and temperature files are stored.
    File Format:
    - Expected format: NumPy .npy binary format
    Sutherland's Law:
        The function implements a modified Sutherland's law for compressible flows:
            μ/μ_ref = [T/T_ref]^(3/2) × [(T_ref + S)/(T + S)]
        In non-dimensional form with T* = T/T_ref and using compressible flow parameters:
            μ* = [(γ-1)T*]^(3/2) × [(1 + Pr) / ((γ-1)T* + Pr)]
    where:
    - T* = non-dimensional temperature (T/T_ref)
    - γ = ratio of specific heats (cp/cv)
    - Pr = Prandtl number (μcp/k)
    - The term (γ-1)T* appears from the energy equation in compressible flow
    - The Sutherland constant S is embedded in the Prandtl number term

    Examples:
    ---------
    mu = load_shear_viscosity('shear_viscosity.npy')
    mu = load_shear_viscosity(
        ...     shear_viscosity_filename='shear_viscosity.npy',
        ...     temperature_filename='temperature.npy',
        ...     load=False)
    mu = load_shear_viscosity(
        ...     shear_viscosity_filename='shear_viscosity.npy',
        ...     temperature_filename='temperature.npy',
        ...     load=False,
        ...     save=True)
    """

    # Construct full path to shear viscosity file by combining base directory with filename
    # flow_path_python is expected to be a global variable containing the base directory
    shear_viscosity_filepath = flow_path_python + shear_viscosity_filename

    # Direct loading mode: Load pre-computed shear viscosity fields
    if load:
        # Load NumPy binary file containing shear viscosity field data
        shear_viscosity_field = np.load(shear_viscosity_filepath, mmap_mode='r')

    # Computation mode: Calculate shear viscosity from temperature
    else:
        # Load NumPy binary file containing temperature field data
        temperature_field = load_temperature(temperature_filename)

        # Calculate shear viscosity using modified Sutherland's law for compressible flows
        # Formula: μ* = [(γ-1)T*]^(3/2) × [(1 + Pr) / ((γ-1)T* + Pr)]
        shear_viscosity_field = (((gamma - 1.0) * temperature_field)**(3.0 / 2.0)) * \
                                ((1.0 + Pr) / (temperature_field * (gamma - 1) + Pr))

    # Optional saving: Write computed shear viscosity field to disk for future use
    if save:
        # Save shear viscosity field to specified filepath
        np.save(shear_viscosity_filepath, shear_viscosity_field)

    # Return the loaded shear viscosity field array
    return shear_viscosity_field

# =========================================================================
# WALL SHEAR STRESS COMPUTING/LOADING
# Compute wall shear stress (from tangential velocity and shear viscosity)
# or load wall shear stress for further analysis
# =========================================================================
def load_shear_stress(shear_stress_filename: str, grid_filename: str = None,
                      shear_viscosity_filename: str = None,
                      velocity_nt_filename: list = None,
                      load: bool = True, save: bool = False) -> np.ndarray:
    """
    Load or compute wall shear stress field data from NumPy binary files.

    This function provides flexible loading of wall shear stress (τ_wall) field through
    two methods:
    1. Direct loading: Read pre-computed wall shear stress field from .npy files
    2. Computation: Calculate wall shear stress from tangential velocity gradient,
        shear viscosity, and grid geometry using the fundamental relation:
        τ_w = μ (∂u_t/∂n)|_wall

    The function assumes all files are located in a predefined directory path
    (specified by the global variable `flow_path_python`).

    Parameters:
    -----------
    shear_stress_filename : str
        Name of the wall shear stress file to load or save (including extension, typically '.npy').
        The file should be a NumPy binary array containing wall shear stress field data.
    grid_filename : str, optional (default=None)
        Name of the grid file to load (including extension, typically '.npz').
        The file should be a NumPy binary array containing grid data.
        Note: Must be provided if `load=False`, otherwise raises an error.
    shear_viscosity_filename : str, optional (default=None)
        Name of the shear viscosity file to load (including extension, typically '.npy').
        The file should be a NumPy binary array containing dynamic viscosity field data.
        Note: Must be provided if `load=False`, otherwise raises an error.
    velocity_nt_filename : list of str, optional (default=None)
        Name of the normal and tangetial velocities files to load (including extension,
        typically '.npy').
        The file should be a list of NumPy binary array containing normal and tangential velocities
        field data.
        Note: Must be provided if `load=False`, otherwise raises an error.
    load : bool, optional (default=True)
        Control flag for loading method:
        - True: Load wall shear stress field directly from `shear_stress_filename`
        - False: Compute wall shear stress from velocity gradient and viscosity
        Set to False when shear stress is not pre-computed but velocity fields,
        viscosity, and grid data are available from simulation output.
    save : bool, optional (default=False)
        Control flag for saving computed wall shear stress field:
        - True: Save computed wall shear stress field to `shear_stress_filename` path
        - False: Do not save (default behavior)
        Useful when `load=False` to save computed shear stress for future use,
        avoiding repeated computation. Has no effect when `load=True`.

    Returns:
    --------
    shear_stress_field : np.ndarray
        NumPy array containing the wall shear stress (τ_wall) field data.

    Notes:
    ------
    Global Dependencies:
    - This function relies on a global variable `flow_path_python` that must be
      defined before calling this function. This variable should contain the
      directory path where grid, shear viscosity and normal and tangential velocities
      files are stored.
    File Format:
    - Expected format: NumPy .npy binary format
    The wall shear stress is computed as:
        τ_w = μ_w × (∂u_t/∂n)|_wall × (Ma/Re)
    where:
    - μ_w: Shear viscosity at the wall
    - ∂u_t/∂n: Normal gradient of tangential velocity at wall
    - Ma/Re: Non-dimensionalization factor (Mach/Reynolds number ratio)

    Examples:
    ---------
    tau_w = load_shear_stress('shear_stress.npy')
    tau_w = load_shear_stress(
        ...     shear_stress_filename='shear_stress.npy',
        ...     grid_filename='grid.npz',
        ...     shear_viscosity_filename='shear_viscosity.npy',
        ...     velocity_nt_filename=['velocity_n.npy', 'velocity_t.npy'],
        ...     load=False)
    tau_w = load_shear_stress(
        ...     shear_stress_filename='shear_stress.npy',
        ...     grid_filename='grid.npz',
        ...     shear_viscosity_filename='shear_viscosity.npy',
        ...     velocity_nt_filename=['velocity_n.npy', 'velocity_t.npy'],
        ...     load=False,
        ...     save=True)
    """

    # Construct full path to wall shear stress file by combining base directory with filename
    # flow_path_python is expected to be a global variable containing the base directory
    shear_stress_filepath = flow_path_python + shear_stress_filename

    # Direct loading mode: Load pre-computed wall shear stress fields
    if load:
        # Load NumPy binary file containing wall shear stress field data
        shear_stress_field = np.load(shear_stress_filepath, mmap_mode='r')

    # Computation mode: Calculate wall shear stress
    else:
        # Load NumPy binary file containing x-grid and y-grid
        x_grid, y_grid = load_grid(grid_filename, ['x', 'y'])

        # Load NumPy binary file containing tangential velocity field
        _, velocity_T_field = compute_velocity(velocity_nt_filename)

        # Load NumPy binary file containing shear viscosity field data
        shear_viscosity_field = load_shear_viscosity(shear_viscosity_filename)

        # Calculate distance vectors between adjacent grid points in wall-normal direction
        delta_x_12 = x_grid[:, 1] - x_grid[:, 0]
        delta_y_12 = y_grid[:, 1] - y_grid[:, 0]
        delta_x_23 = x_grid[:, 2] - x_grid[:, 1]
        delta_y_23 = y_grid[:, 2] - y_grid[:, 1]

        # Compute grid spacing as arc length between points (accounts for grid curvature)
        spacing_12 = np.sqrt(delta_x_12**2 + delta_y_12**2)
        spacing_23 = np.sqrt(delta_x_23**2 + delta_y_23**2)

        # Pre-compute finite difference coefficients
        # Second-order accurate scheme for non-uniform grid spacing
        spacing_sum = spacing_12 + spacing_23
        spacing_prod = spacing_12 * spacing_23
        spacing_ratio = spacing_12 / spacing_23

        # Finite difference weights for ∂u_t/∂n calculation
        coeff_1 = coeff_1 = spacing_ratio / spacing_sum - spacing_sum / spacing_prod
        coeff_2 = spacing_sum / spacing_prod
        coeff_3 = - spacing_ratio / spacing_sum

        # Compute velocity gradient normal to wall
        velocity_gradient_normal = (
            np.einsum('i,ij->ij', coeff_1, velocity_T_field[:, 0, :])
            + np.einsum('i,ij->ij', coeff_2, velocity_T_field[:, 1, :])
            + np.einsum('i,ij->ij', coeff_3, velocity_T_field[:, 2, :])
        )

        # Non-dimensional factors
        nondim_factor = Ma / Re

        # Calculate wall shear stress
        shear_stress_field = shear_viscosity_field[:, 0, :] * nondim_factor * velocity_gradient_normal

    # Optional saving: Write computed wall shear stress field to disk for future use
    if save:
        # Save wall shear stress field to specified filepath
        np.save(shear_stress_filepath, shear_stress_field)

    # Return the loaded wall shear stress field array
    return shear_stress_field

# =========================================================================
# SKIN FRICTION COMPUTING/LOADING
# Compute skin friction (from wall shear stress) or load skin friction for
# further analysis
# =========================================================================
def load_skin_friction(skin_friction_filename: str, shear_stress_filename: str = None,
                       load: bool = True, save: bool = False) -> np.ndarray:
    """
    Load or compute skin friction coefficient data from NumPy binary files.

    This function provides flexible loading of skin friction coefficient (C_f) through
    two methods:
    1. Direct loading: Read pre-computed skin friction coefficient from .npy files
    2. Computation: Calculate skin friction coefficient from wall shear
        stress using the fundamental relation:
        C_f = τ_w / (0.5 × Ma²)

    The function assumes all files are located in a predefined directory path
    (specified by the global variable `flow_path_python`).

    Parameters:
    -----------
    skin_friction_filename : str
        Name of the skin friction coefficient file to load or save (including extension,
        typically '.npy').
        The file should be a NumPy binary array containing skin friction coefficient data.
    shear_stress_filename : str, optional (default=None)
        Name of the wall shear stress file to load (including extension, typically '.npy').
        The file should be a NumPy binary array containing wall shear stress field data.
        Note: Must be provided if `load=False`, otherwise raises an error.
    load : bool, optional (default=True)
        Control flag for loading method:
        - True: Load skin friction coefficient directly from `skin_friction_filename`
        - False: Compute skin friction coefficient from wall shear stress field
        Set to False when skin friction coefficient is not pre-computed but wall shear
        stress is available.
    save : bool, optional (default=False)
        Control flag for saving computed skin friction coefficient:
        - True: Save computed skin friction coefficient to `skin_friction_filename` path
        - False: Do not save (default behavior)
        Useful when `load=False` to save computed skin friction coefficient for future use,
        avoiding repeated computation. Has no effect when `load=True`.

    Returns:
    --------
    skin_friction_coefficient : np.ndarray
        NumPy array containing the skin friction coefficient (C_f) field data.

    Notes:
    ------
    Global Dependencies:
    - This function relies on a global variable `flow_path_python` that must be
      defined before calling this function. This variable should contain the
      directory path where wall shear stress file are stored.
    File Format:
    - Expected format: NumPy .npy binary format
    The skin friction coefficient is computed as:
        C_f = τ_w / (0.5 × Ma²)
    where:
    - τ_w: Wall shear stress (non-dimensional)
    - Ma: Mach number (U_∞/a_∞)
    - a_∞: Freestream sound velocity (reference value)
    - U_∞: Freestream velocity (reference value)

    Examples:
    ---------
    c_f = load_skin_friction('skin_friction.npy')
    c_f = load_skin_friction(
        ...     skin_friction_filename='skin_friction.npy',
        ...     shear_stress_filename='shear_stress.npy',
        ...     load=False)
    c_f = load_skin_friction(
        ...     skin_friction_filename='skin_friction.npy',
        ...     shear_stress_filename='shear_stress.npy',
        ...     load=False,
        ...     save=True)
    """

    # Construct full path to skin friction coefficient file by combining base directory
    # with filename
    # flow_path_python is expected to be a global variable containing the base directory
    skin_friction_filepath = flow_path_python + skin_friction_filename

    # Direct loading mode: Load pre-computed skin friction coefficient
    if load:
        # Load NumPy binary file containing skin friction coefficient data
        skin_friction_coeff = np.load(skin_friction_filepath, mmap_mode='r')

    # Computation mode: Calculate skin friction coefficient
    else:
        # Load NumPy binary file containing wall shear stress field data
        shear_stress_field = load_shear_stress(shear_stress_filename)

        # Non-dimensional factors
        dynamic_pressure = 0.5 * Ma**2

        # Compute skin friction coefficient using:
        # c_f = τ_w / (0.5 × Ma²)
        skin_friction_coeff = shear_stress_field / dynamic_pressure

    # Optional saving: Write computed wall shear stress field to disk for future use
    if save:
        # Save wall shear stress field to specified filepath
        np.save(skin_friction_filepath, skin_friction_coeff)

    # Return the loaded wall shear stress field array
    return skin_friction_coeff
