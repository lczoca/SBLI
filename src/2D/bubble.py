import numpy as np
import scipy as sp

# ====================================================================================================
# ====================================================================================================

def analyze_flow_separation_bubble(x_coords: np.ndarray,
                                   y_coords: np.ndarray,
                                   skin_friction_coeff: np.ndarray,
                                   analysis_region: np.ndarray | slice,
                                   invert_separation_order: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyzes flow separation bubbles by computing bubble length and separation/reattachment points.

    This function identifies regions of flow separation (where skin friction coefficient < 0)
    and calculates the geometric properties of separation bubbles for multiple flow snapshots.

    Parameters:
    -----------
    x_coords : np.ndarray, shape (nx, ny)
        Grid x-coordinates
    y_coords : np.ndarray, shape (nx, ny)
        Grid y-coordinates
    skin_friction_coeff : np.ndarray, shape (n_points, num_snapshots)
        Skin friction coefficient field (cf < 0 indicates separation)
    analysis_region : np.ndarray or slice
        Indices or slice defining the region to analyze for separation
    invert_separation_order : bool, default=False
        If True, swaps separation/reattachment point definitions (for pressure side analysis)

    Returns:
    --------
    bubble_length : np.ndarray, shape (num_snapshots,)
        Length of separation bubble for each snapshot
    separation_point : np.ndarray, shape (num_snapshots,)
        x-coordinate of flow separation point
    reattachment_point : np.ndarray, shape (num_snapshots,)
        x-coordinate of flow reattachment point

    Notes:
    -----
    - Separation occurs where skin_friction_coeff < 0
    - Bubble length is computed as Euclidean distance between separation/reattachment points
    - Function handles cases with no separation gracefully (returns 0)
    - Fully vectorized for maximum performance across all snapshots

    Usage Example:
    --------------
    LSB, sep_x, reatt_x = analyze_flow_separation_bubble(x, y, c_f, analysis_region, invert_separation_order=False)
    """

    # Get number of snapshots
    _, n_snapshots = np.shape(skin_friction_coeff)

    # Extract analysis region coordinates (works with both indices and slices)
    region_x_coords = x_coords[analysis_region, 0]
    region_y_coords = y_coords[analysis_region, 0]
    region_skin_friction = skin_friction_coeff[analysis_region, :]

    # Separation detection
    # Create boolean mask for separated flow regions
    is_separated = region_skin_friction < 0  # Shape: (n_region_points, num_snapshots)

    # Check which snapshots contain separation
    has_separation = np.any(is_separated, axis=0)  # Shape: (num_snapshots,)

    # Separation boundary identification - find separation boundaries
    # For snapshots with separation, find first and last separated points
    # First separated point per snapshot
    separation_start_idx = np.where(has_separation, np.argmax(is_separated, axis=0), 0)
    # Last separated point per snapshot
    separation_end_idx = np.where(has_separation, is_separated.shape[0] - 1 - np.argmax(is_separated[::-1, :], axis=0), 0)


    # Bubble geometry calculation
    # Extract coordinates at separation boundaries
    start_x = region_x_coords[separation_start_idx]
    start_y = region_y_coords[separation_start_idx]
    end_x = region_x_coords[separation_end_idx]
    end_y = region_y_coords[separation_end_idx]

    # Compute bubble length as Euclidean distance
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    bubble_length = np.where(has_separation, np.sqrt(delta_x**2 + delta_y**2), 0.0)

    # Separation point assignment - flow direction conventions
    if invert_separation_order:
        # Pressure side convention: separation = downstream, reattachment = upstream
        separation_point = np.where(has_separation, end_x, 0.0)
        reattachment_point = np.where(has_separation, start_x, 0.0)
    else:
        # Suction side convention: separation = upstream, reattachment = downstream
        separation_point = np.where(has_separation, start_x, 0.0)
        reattachment_point = np.where(has_separation, end_x, 0.0)

    return bubble_length, separation_point, reattachment_point


# =========================================================================
# Specialized functions for some cases
# =========================================================================

def analyze_suction_side_separation(x_coords: np.ndarray,
                                    y_coords: np.ndarray,
                                    skin_friction_coeff: np.ndarray,
                                    chord_fraction_limits: tuple[float, float] = (0.4, 0.92)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Specialized function for suction side separation analysis.

    Parameters:
    -----------
    chord_fraction_limits : tuple[float, float]
        (min_fraction, max_fraction) defining analysis region as fraction of chord

    Usage Example:
    --------------
    LSB_SS, x_separation_SS, x_reattachment_SS = analyze_suction_side_separation(x, y, c_f)
    """

    # Create analysis mask for suction side region
    half_chord_idx = x_coords.shape[0] // 2
    suction_region_mask = ((x_coords[:half_chord_idx, 0] > chord_fraction_limits[0]) &
                          (x_coords[:half_chord_idx, 0] < chord_fraction_limits[1]))
    analysis_indices = np.where(suction_region_mask)[0]

    # Analyze separation
    bubble_length, sep_point, reatt_point = analyze_flow_separation_bubble(
        x_coords, y_coords, skin_friction_coeff, analysis_indices, invert_separation_order=False)

    return bubble_length, sep_point, reatt_point

# =========================================================================

def analyze_pressure_side_separation(x_coords: np.ndarray,
                                     y_coords: np.ndarray,
                                     skin_friction_coeff: np.ndarray,
                                     analysis_slice: slice = slice(850, 1200)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Specialized function for pressure side separation analysis.

    Parameters:
    -----------
    analysis_slice : slice
        Slice object defining pressure side analysis region

    Notes:
    ------
    - Pressure side separation/reattachment point correction is done by inverting the values

    Usage Example:
    --------------
    LSB_PS, x_separation_PS, x_reattachment_PS = analyze_pressure_side_separation(x, y, c_f)
    """

    # Analyze separation
    bubble_length, sep_point, reatt_point = analyze_flow_separation_bubble(
        x_coords, y_coords, skin_friction_coeff, analysis_slice, invert_separation_order=True)

    return bubble_length, sep_point, reatt_point

# =========================================================================

def analyze_both_flow_sides_separation(x_coords: np.ndarray,
                                       y_coords: np.ndarray,
                                       skin_friction_coeff: np.ndarray) -> tuple[tuple, tuple]:
    """
    Function to analyze both suction and pressure side separation simultaneously.

    Returns:
    --------
    suction_results : tuple, shape (3, num_snapshots,)
        bubble_length : np.ndarray, shape (num_snapshots,)
            Length of separation bubble for each snapshot
        separation_point : np.ndarray, shape (num_snapshots,)
            x-coordinate of flow separation point
        reattachment_point : np.ndarray, shape (num_snapshots,)
            x-coordinate of flow reattachment point
    pressure_results : tuple, shape (3, num_snapshots,)
        bubble_length : np.ndarray, shape (num_snapshots,)
            Length of separation bubble for each snapshot
        separation_point : np.ndarray, shape (num_snapshots,)
            x-coordinate of flow separation point
        reattachment_point : np.ndarray, shape (num_snapshots,)
            x-coordinate of flow reattachment point

    Usage Example:
    --------------
    SS_results, SP_results = analyze_both_flow_sides_separation(x, y, c_f)
    """

    # Analyze both sides in parallel
    suction_results = analyze_suction_side_separation(x_coords, y_coords, skin_friction_coeff)
    pressure_results = analyze_pressure_side_separation(x_coords, y_coords, skin_friction_coeff)

    return suction_results, pressure_results
