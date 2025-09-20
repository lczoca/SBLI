import numpy as np
from tqdm import tqdm
import psutil
import time

import CGNS

from filepath import *
from flow_properties import *
from compact_scheme_10th import *

# =========================================================================
# CFD DATA PROCESSING SCRIPT
# This script read flow properties for one computational grid (O-grid) from
# Python/Numpy readable format (.npy files) and calculates others properties
# for further analysis
# =========================================================================

def calculate_metric_terms(grid_x: np.ndarray, grid_y: np.ndarray) -> tuple:
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
    >>> import numpy as np
    >>> # Create a simple rectangular grid
    >>> xi = np.linspace(0, 1, 21)
    >>> eta = np.linspace(0, 1, 11)
    >>> XI, ETA = np.meshgrid(xi, eta, indexing='ij')
    >>> x = XI + 0.1 * np.sin(2*np.pi*XI) * np.sin(np.pi*ETA)
    >>> y = ETA + 0.05 * np.cos(np.pi*XI) * np.sin(2*np.pi*ETA)
    >>>
    >>> # Calculate metric terms
    >>> metrics = calculate_metric_terms(x, y)
    >>> dx_dxi, dx_deta, dy_dxi, dy_deta, dxi_dx, dxi_dy, deta_dx, deta_dy, det_J = metrics
    >>>
    >>> # Check Jacobian determinant is positive (valid transformation)
    >>> assert np.all(det_J > 0), "Invalid grid transformation detected"

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
# =========================================================================

def calculate_velocity(fluid_density: np.ndarray, fluid_momentum: np.ndarray) -> np.ndarray:
    """
    Calculate fluid velocity from momentum and density using the fundamental relationship.

    This function computes the velocity field of a fluid by applying the basic kinematic
    relationship between momentum, density, and velocity. The velocity is obtained by
    dividing the momentum vector by the local density at each point in the flow field.

    Parameters:
    -----------
    fluid_density : np.ndarray
        Fluid density field. Must be a numpy array with shape corresponding to the
        computational grid. Values should be positive and non-zero to avoid division
        by zero errors.
    fluid_momentum : np.ndarray
        Fluid momentum field. Must be a numpy array with the same shape as fluid_density.
        For multi-dimensional flows, this represents the momentum
        density vector (ρu, ρv, ρw for 3D flows).

    Returns:
    --------
    velocity : np.ndarray
        Velocity field. Returns a numpy array with the same shape as the input arrays,
        representing the local fluid velocity at each grid point. For vector quantities,
        returns velocity components (u, v, w for 3D flows).

    Notes:
    ------
    - The calculation is based on the fundamental relationship: v = m/ρ, where
      v is velocity, m is momentum density, and ρ is fluid density
    - This function assumes the momentum is already in the form of momentum density
      (momentum per unit volume)
    - Division by zero will occur if fluid_density contains zero values

    Examples:
    ---------
    >>> import numpy as np
    >>> density = np.array([1.2, 1.1, 1.0])  # kg/m³
    >>> momentum = np.array([2.4, 3.3, 4.0])  # kg/(m²·s)
    >>> velocity = calculate_velocity(density, momentum)
    >>> print(velocity)  # [2.0, 3.0, 4.0] m/s

    Raises:
    -------
    ZeroDivisionError
        If fluid_density contains zero values
    ValueError
        If input arrays have incompatible shapes
    """

    # Calculate velocity using the fundamental kinematic relationship: v = momentum/density
    velocity = fluid_momentum / fluid_density

    return velocity


# =========================================================================
# =========================================================================


def calculate_temperature(fluid_density: np.ndarray, fluid_pressure: np.ndarray) -> np.ndarray:
    """
    Calculate fluid temperature from pressure and density using the ideal gas law.

    This function computes the temperature field of a fluid by applying the ideal gas
    equation of state. The temperature is determined using the relationship between
    pressure, density, and temperature for an ideal gas.

    Parameters:
    -----------
    fluid_density : np.ndarray
        Fluid density field. Must be a numpy array with shape corresponding to the
        computational grid. Values should be positive and non-zero.
    fluid_pressure : np.ndarray
        Fluid pressure field. Must be a numpy array with the same shape as fluid_density.
        Represents the thermodynamic pressure at each grid point.

    Returns:
    --------
    temperature : np.ndarray
        Temperature field. Returns a numpy array with the same shape as the input array,
        representing the local fluid temperature at each grid point.

    Notes:
    ------
    - The calculation is based on the ideal gas law: P = ρRT, rearranged as T = P/(ρR)
    - R is the specific gas constant and must be defined globally
    - This assumes the fluid behaves as an ideal gas
    - Valid for gases at moderate pressures and temperatures where real gas effects
      are negligible

    Global Variables Required:
    --------------------------
    R : float
        Specific gas constant.

    Examples:
    ---------
    >>> import numpy as np
    >>> R = 287.0  # J/(kg·K) for air
    >>> density = np.array([1.225, 1.0, 0.8])  # kg/m³
    >>> pressure = np.array([101325, 85000, 68000])  # Pa
    >>> temperature = calculate_temperature(density, pressure)
    >>> print(temperature)  # [288.2, 296.2, 297.0] K

    Raises:
    -------
    NameError
        If the global constant R is not defined
    ZeroDivisionError
        If fluid_density contains zero values
    ValueError
        If input arrays have incompatible shapes
    """

    # Calculate temperature using ideal gas law rearranged: T = P/(ρR)
    temperature = fluid_pressure / (fluid_density * R)

    return temperature

# =========================================================================
# =========================================================================

def calculate_shear_viscosity(fluid_temperature: np.ndarray) -> np.ndarray:
    """
    Calculate dynamic shear viscosity from temperature using Sutherland's law.

    This function computes the dynamic (shear) viscosity of a fluid as a function
    of temperature using Sutherland's viscosity law. This empirical correlation
    accounts for the temperature dependence of viscosity in gases and is widely
    used in computational fluid dynamics applications.

    Parameters:
    -----------
    fluid_temperature : np.ndarray
        Fluid temperature field. Must be a numpy array representing the local
        temperature at each grid point. Values should be positive and within the
        valid range for the Sutherland model.

    Returns:
    --------
    shear_viscosity : np.ndarray
        Dynamic shear viscosity field. Returns a numpy array with the same shape
        as the input temperature array, representing the local dynamic viscosity
        at each grid point.

    Notes:
    ------
    - Uses a modified form of Sutherland's law incorporating compressible flow parameters
    - The formula used: μ = ((γ-1)T)^(3/2) * (1+Pr)/(T(γ-1)+Pr)
    - This formulation is commonly used in compressible flow simulations
    - Valid for gases where viscosity increases with temperature

    Global Variables Required:
    --------------------------
    gamma : float
        Specific heat ratio (γ = Cp/Cv).
    Pr : float
        Prandtl number.

    Examples:
    ---------
    >>> import numpy as np
    >>> gamma = 1.4  # specific heat ratio for air
    >>> Pr = 0.72    # Prandtl number for air
    >>> temperature = np.array([288.15, 300.0, 350.0])  # K
    >>> viscosity = calculate_shear_viscosity(temperature)
    >>> print(viscosity)  # Dynamic viscosity values in Pa·s

    Raises:
    -------
    NameError
        If global constants gamma or Pr are not defined
    ValueError
        If fluid_temperature contains negative values or invalid temperatures
    """
    # Calculate shear viscosity using modified Sutherland's law for compressible flows
    shear_viscosity = (((gamma - 1.0) * fluid_temperature)**(3.0 / 2.0)) * \
                     ((1.0 + Pr) / (fluid_temperature * (gamma - 1) + Pr))
    return shear_viscosity

# =========================================================================
# =========================================================================

def transform_velocity_to_normal_tangential(x_grid: np.ndarray,
                                            y_grid: np.ndarray,
                                            x_velocity: np.ndarray,
                                            y_velocity: np.ndarray) -> tuple:
    """
    Transforms velocity components from Cartesian (x,y) to normal-tangential coordinates.

    This function converts velocity vectors from the standard x-y coordinate system
    to a curvilinear normal-tangential coordinate system based on the grid geometry.

    Parameters:
    -----------
    x_grid : np.ndarray, shape (nx, ny)
        2D grid of x-coordinates
    y_grid : np.ndarray, shape (nx, ny)
        2D grid of y-coordinates
    x_velocity : np.ndarray, shape (nx, ny, n_snapshots)
        Velocity field in Cartesian coordinates (x-velocity components)
    y_velocity : np.ndarray, shape (nx, ny, n_snapshots)
        Velocity field in Cartesian coordinates (y-velocity components)

    Returns:
    --------
    n_velocity : np.ndarray, shape (nx, ny, n_snapshots)
        Velocity field in normal-tangential coordinates (normal velocity components)
    t_velocity : np.ndarray, shape (nx, ny, n_snapshots)
        Velocity field in normal-tangential coordinates (tangential velocity components)

    Notes:
    ------
    - Normal velocity is corrected for pressure side (second half of domain)
    - Uses grid metric terms for coordinate transformation

    Usage Example:
    --------------
    n_velocity, t_velocity = transform_velocity_to_normal_tangential(x, y, velocity_x, velocity_y)
    """

    # Get dimensions
    nx, ny, n_snapshots = np.shape(x_velocity)

    # Pre-allocate output array -> normal and tangential velocity
    n_velocity = np.zeros_like(x_velocity)
    t_velocity = np.zeros_like(x_velocity)

    # Compute metric terms once (assumed to be grid-invariant)
    grid_metrics = metric_terms(x_grid, y_grid)
    dx_dxi, dx_deta, dy_dxi, dy_deta, dxi_dx, dxi_dy, deta_dx, deta_dy, jacobian_det = grid_metrics

    # Pre-compute normalization factor (grid-invariant)
    eta_magnitude = np.sqrt(deta_dx**2 + deta_dy**2)

    # Normal velocity component (perpendicular to surface): (deta_dy * u - deta_dx * v) / |eta|
    n_velocity = np.einsum('ij,ijk->ijk', deta_dx / eta_magnitude, x_velocity) \
               + np.einsum('ij,ijk->ijk', deta_dy / eta_magnitude, y_velocity)

    # Tangential velocity component (parallel to surface): (deta_dx * u + deta_dy * v) / |eta|
    t_velocity = np.einsum('ij,ijk->ijk', deta_dy / eta_magnitude, x_velocity) \
               - np.einsum('ij,ijk->ijk', deta_dx / eta_magnitude, y_velocity)

    # Apply sign correction for pressure side (second half of domain)
    n_velocity[nx//2:, :, :] *= -1

    return n_velocity, t_velocity

# =========================================================================
# =========================================================================


def calculate_wall_shear_stress(x_grid: np.ndarray, y_grid: np.ndarray,
                                t_velocity: np.ndarray, shear_viscosity: np.ndarray) -> np.ndarray:
    """
    Computes wall shear stress for 2D flow.

    This function calculates the wall shear stress (tau_wall) second-order
    finite difference scheme for the velocity gradient normal to the wall.

    Parameters:
    -----------
    x_grid : np.ndarray, shape (nx, ny)
        2D grid of x-coordinates
    y_grid : np.ndarray, shape (nx, ny)
        2D grid of y-coordinates
    t_velocity : np.ndarray, shape (nx, ny, n_snapshots)
        Tangential velocity component at wall-normal grid points
    shear_viscosity : np.ndarray, shape (nx, ny, n_snapshots)
        Shear viscosity field

    Returns:
    --------
    wall_shear_stress : np.ndarray, shape (nx, n_snapshots)
        Wall shear stress (tau_wall) at each surface point

    Notes:
    ------
    - Uses 3-point finite difference stencil for velocity gradient calculation
    - Assumes first grid line (j=0) represents the wall boundary
    - Second-order accurate discretization of du/dn at the wall

    Usage Example:
    --------------
    tau_wall = calculate_wall_shear_stress(x, y, t_velocity, shear_visc)
    """

    # Distance vectors between adjacent grid points
    delta_x_12 = x_grid[:, 1] - x_grid[:, 0]
    delta_y_12 = y_grid[:, 1] - y_grid[:, 0]
    delta_x_23 = x_grid[:, 2] - x_grid[:, 1]
    delta_y_23 = y_grid[:, 2] - y_grid[:, 1]

    # Grid spacing (arc length between points)
    spacing_12 = np.sqrt(delta_x_12**2 + delta_y_12**2)
    spacing_23 = np.sqrt(delta_x_23**2 + delta_y_23**2)

    # Pre-compute finite difference coefficients
    # Second-order accurate scheme for non-uniform grid spacing
    spacing_sum = spacing_12 + spacing_23
    spacing_prod = spacing_12 * spacing_23
    spacing_ratio = spacing_12 / spacing_23

    # Finite difference weights for du/dn calculation
    coeff_1 = coeff_1 = spacing_ratio / spacing_sum - spacing_sum / spacing_prod
    coeff_2 = spacing_sum / spacing_prod
    coeff_3 = - spacing_ratio / spacing_sum

    # Ultra-vectorized computation using einsum
    velocity_gradient_normal = (
          np.einsum('i,ij->ij', coeff_1, t_velocity[:, 0, :])
        + np.einsum('i,ij->ij', coeff_2, t_velocity[:, 1, :])
        + np.einsum('i,ij->ij', coeff_3, t_velocity[:, 2, :])
    )

    # Non-dimensional factors
    nondim_factor = Ma / Re

    # Calculate wall shear stress
    wall_shear_stress = shear_viscosity[:, 0, :] * nondim_factor * velocity_gradient_normal

    return wall_shear_stress

# =========================================================================
# =========================================================================

def calculate_skin_friction(wall_shear_stress: np.ndarray) -> np.ndarray:
    """
    Computes skin friction coefficient for 2D flow.

    This function calculates the skin friction coefficient (c_f) using wall
    shear stress.

    Parameters:
    -----------
    wall_shear_stress : np.ndarray, shape (nx, n_snapshots)
        Wall shear stress (tau_wall) at each surface point

    Returns:
    --------
    skin_friction_coeff : np.ndarray, shape (nx, n_snapshots)
        Skin friction coefficient (c_f) at each surface point

    Notes:
    ------
    - Uses 3-point finite difference stencil for velocity gradient calculation
    - Assumes first grid line (j=0) represents the wall boundary
    - Second-order accurate discretization of du/dn at the wall

    Usage Example:
    --------------
    c_f = calculate_skin_friction(tau_wall)
    """
    # Non-dimensional factors
    dynamic_pressure = 0.5 * Ma**2

    # Skin friction coefficient: c_f = tau_wall / (0.5 * rho * U_ref^2)
    # In non-dimensional form: c_f = tau_wall / (0.5 * Ma^2)
    skin_friction_coeff = wall_shear_stress / dynamic_pressure

    return skin_friction_coeff

# =========================================================================
# =========================================================================
