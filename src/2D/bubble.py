from matplotlib import pyplot
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# ====================================================================================================
# CFD BUBBLE PROCESSING SCRIPT
# This script implement functions to calculate the mean length of the separation bubble and the area
# of the separation bubble for an airfoil region
# ====================================================================================================

def bubble_length(x_grid: np.ndarray, y_grid: np.ndarray,
                  skin_friction_coeff: np.ndarray, analysis_region: np.ndarray | slice,
                  invert_separation_order: bool = False) -> tuple:
    """
    Analyzes flow separation bubbles by computing bubble length and separation/reattachment points.

    This function identifies regions of flow separation (where skin friction coefficient < 0)
    and calculates the geometric properties of separation bubbles for mean flow.

    Parameters:
    -----------
    x_grid : np.ndarray, shape (nx, ny)
        Grid X coordinates
    y_grid : np.ndarray, shape (nx, ny)
        Grid Y coordinates
    skin_friction_coeff : np.ndarray, shape (n_points)
        Skin friction coefficient (mean value) (cf < 0 indicates separation)
    analysis_region : np.ndarray or slice
        Indices or slice defining the region to analyze for separation
    invert_separation_order : bool, default=False
        If True, swaps separation/reattachment point definitions (for pressure side analysis)

    Returns:
    --------
    bubble_length : float
        Mean length of separation bubble for each snapshot
    separation_point : float
        Grid X coordinate of mean flow separation point
    reattachment_point : float
        Grid Y coordinate of mean flow reattachment point

    Notes:
    -----
    - Separation occurs where skin_friction_coeff < 0
    - Bubble length is computed as Euclidean distance between separation/reattachment points
    - Function handles cases with no separation gracefully (returns 0)

    Usage Example:
    --------------
    LSB, sep_x, reatt_x = analyze_flow_separation_bubble(x, y, c_f, analysis_region, invert_separation_order=False)
    """

    # Extract analysis region coordinates
    region_x_grid = x_grid[analysis_region, 0]
    region_y_grid = y_grid[analysis_region, 0]
    region_skin_friction = skin_friction_coeff[analysis_region]

    # Separation detection
    # Create boolean mask for separated flow regions
    is_separated = region_skin_friction < 0

    # Check if there is any separation
    has_separation = np.any(is_separated)

    if not has_separation:
        # No separation found - return zeros
        return 0.0, 0.0, 0.0

    # Separation boundary identification - find separation boundaries
    # Find first and last separated points
    separation_indices = np.where(is_separated)[0]
    separation_start_idx = separation_indices[0]  # First separated point
    separation_end_idx = separation_indices[-1]   # Last separated point

    # Bubble geometry calculation
    # Extract coordinates at separation boundaries
    start_x = region_x_grid[separation_start_idx]
    start_y = region_y_grid[separation_start_idx]
    end_x = region_x_grid[separation_end_idx]
    end_y = region_y_grid[separation_end_idx]

    # Compute bubble length as Euclidean distance
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    bubble_length = np.sqrt(delta_x**2 + delta_y**2)

    # Separation point assignment - flow direction conventions
    if invert_separation_order:
        # Pressure side convention: separation = downstream, reattachment = upstream
        separation_point = end_x
        reattachment_point = start_x
    else:
        # Suction side convention: separation = upstream, reattachment = downstream
        separation_point = start_x
        reattachment_point = end_x

    return bubble_length, separation_point, reattachment_point

# ====================================================================================================
# ====================================================================================================

def bubble_area(x_grid: np.ndarray, y_grid: np.ndarray, contour_level: float,
                velocity_field: np.ndarray, spatial_analysis_region: np.ndarray | slice) -> np.ndarray:
    """
    Calculate the total area enclosed by contours at a specified level for each temporal snapshot.

    This function computes the area within regions bounded by contour lines at a given level
    across multiple time snapshots of a velocity field. It uses the Shoelace formula to
    calculate the area of each closed contour polygon and sums them to get the total area
    for each temporal snapshot.

    The function is particularly useful for analyzing bubble dynamics, convection patterns,
    or any other phenomena where tracking enclosed areas over time is important.

    Parameters:
    -----------
    x_grid : np.ndarray, shape (nx, ny)
        2D meshgrid array containing grid X coordinate values for each grid point.
    y_grid : np.ndarray, shape (nx, ny)
        2D meshgrid array containing grid Y coordinate values for each grid point.
    contour_level : float
        The scalar value defining the contour level at which to calculate enclosed areas.
        Areas will be computed for regions where the velocity field equals this value.
    velocity_field : np.ndarray, shape (nx, ny) or (nx, ny, nt)
        2D or 3D array containing velocity field data:
        - If 2D (nx, ny): Single snapshot of the velocity field
        - If 3D (nx, ny, nt): Multiple temporal snapshots where nt is the number of time steps
        The function automatically handles both cases by expanding 2D arrays to 3D.
    spatial_analysis_region : np.ndarray or slice
        Spatial subset specification for analysis (in X direction):
        - If slice: Standard Python slice object (e.g., slice(10, 50))
        - If np.ndarray: Boolean or integer array for advanced indexing
        Applied to both coordinate arrays and velocity field to focus analysis
        on a specific spatial region of interest.

    Returns:
    --------
    bubbles_areas : np.ndarray, shape (nt,)
        1D array containing the total enclosed area for each temporal snapshot.
        - Length equals the number of time snapshots in the velocity field
        - Each element represents the sum of all enclosed contour areas at the
          specified level for that particular time step

    Notes:
    ------
    Mathematical Foundation:
    The Shoelace formula (also known as the Surveyor's formula) is used to calculate
    the area of a simple polygon given its vertices:
    Area = 0.5 * |Σ(x_i * y_{i+1} - x_{i+1} * y_i)| for i = 0 to n-1


    Examples:
    ---------
    area = bubble_area(x, y, contour_level=-1e-5, velocity_field=u, spatial_analysis_region=slice(300:450))
    area = bubble_area(x, y, contour_level=-1e-5, velocity_field=u, spatial_analysis_region=[300:450])
    """

    # Handle both 2D and 3D input arrays by ensuring 3D structure
    # If input is 2D (single snapshot), expand to 3D by adding temporal dimension
    if velocity_field.ndim == 2:
        velocity_field = np.expand_dims(velocity_field, axis=2)

    # Apply spatial analysis region to coordinate grids and velocity field
    # This allows focusing the analysis on a specific spatial subset of the domain
    x_grid = x_grid[spatial_analysis_region, :]
    y_grid = y_grid[spatial_analysis_region, :]
    velocity_field = velocity_field[spatial_analysis_region, :, :]

    # Extract number of temporal snapshots from the last dimension
    num_snapshots = velocity_field.shape[-1]

    # Initialize array to store total enclosed areas for each temporal snapshot
    bubbles_areas = np.zeros(num_snapshots)

    # Process each temporal snapshot to calculate enclosed areas
    for snapshot_index in range(num_snapshots):
        # Create matplotlib figure and axis for contour generation
        # Note: This is required to extract contour line coordinates
        fig, ax = plt.subplots()

        # Generate contour lines at the specified level for current snapshot
        contour_set = ax.contour(x_grid, y_grid,
                                velocity_field[:, :, snapshot_index],
                                levels=[contour_level])

        # Initialize total area accumulator for current snapshot
        snapshot_bubble_area = 0.0

        # Process each contour line segment at the specified level
        # contour_set.allsegs[0] contains all line segments for the first (and only) contour level
        for contour_segment in contour_set.allsegs[0]:
            # Extract x and y coordinates of the contour segment vertices
            x_vertices = contour_segment[:, 0]
            y_vertices = contour_segment[:, 1]

            # Apply shoelace formula to calculate area enclosed by the contour
            # Formula: Area = 0.5 * |Σ(x_i * y_{i+1} - x_{i+1} * y_i)|
            segment_area = 0.5 * np.abs(np.dot(x_vertices, np.roll(y_vertices, 1)) -
                                       np.dot(y_vertices, np.roll(x_vertices, 1)))

            # Accumulate area from this contour segment to snapshot total
            snapshot_bubble_area += segment_area

        # Store total area for current snapshot
        bubbles_areas[snapshot_index] = snapshot_bubble_area

        # Explicitly close matplotlib figure to prevent memory leaks
        plt.close(fig)

    return bubbles_areas

# # ====================================================================================================
# # ====================================================================================================
