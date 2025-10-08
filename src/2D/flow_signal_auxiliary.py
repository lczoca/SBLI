import numpy as np

# =========================================================================
# DATA ANALYSIS SCRIPT -> AUXILIARY FUNCTIONS FOR SIGNAL PROCESSING
# This script contains auxiliary functions for signal processing, like:
#   welch_parameters
# =========================================================================

def welch_parameters(num_bins: int, num_points: int, overlap: float):
    """
    Calculate parameters for welch method based on desired number of segments.

    This function divides the signal into a specific number of segments (bins)
    and calculates the necessary parameters for the welch method, including the
    size of each segment and the overlap between them.

    Parameters:
    -----------
    num_bins : int
        Desired number of segments to divide the signal into.
        Must be positive and cannot be larger than num_points.
    num_points : int
        Total number of points (samples) in the input signal.
        Must be positive and greater than num_bins.
    overlap : float, optional
        Overlap fraction between consecutive segments, in range [0, 1).

    Returns:
    --------
    nperseg : int
        Length of each segment in number of samples.
        This value is used as the 'nperseg' parameter in scipy.signal.welch.
    noverlap : int
        Number of overlapping points between consecutive segments.
        This value is used as the 'noverlap' parameter in scipy.signal.welch.
    freq_size : int
        Expected size of the frequency array returned by scipy.signal.welch.
        For real signals: freq_size = nperseg // 2 + 1 (due to FFT symmetry)

    psd_size : int
        Expected size of the frequency array and the power spectral density (PSD)
        array returned by scipy.signal.welch.
        For real signals: psd_size = nperseg // 2 + 1 (due to FFT symmetry)

    Raises:
    -------
    ValueError
        If overlap is not in range [0, 1) or if n_bins is too large
        for the available number of points (resulting in nperseg < 1).

    Notes:
    ------
    - This function assumes real signals (not complex)
    - The actual number of segments processed may be higher than n_bins due to overlap
    - Higher overlap results in better variance reduction, but higher computational cost
    - Frequency resolution is determined by nperseg: Δf = fs / nperseg

    Examples:
    ---------
    >>> # Divide a 10000-point signal into 10 segments with 50% overlap
    >>> nperseg, noverlap, freq_size, psd_size = welch_setup(10, 10000, 0.5)
    >>> print(f"Segments of {nperseg} points with {noverlap} overlap points")
    Segments of 1000 points with 500 overlap points

    >>> # Use parameters with scipy.signal.welch
    >>> import scipy.signal as signal
    >>> frequencies, psd = signal.welch(x, fs=1000, nperseg=nperseg, noverlap=noverlap)
    >>> print(f"Got {len(frequencies)} frequency points and {len(psd)} PSD values")

    See Also:
    ---------
    scipy.signal.welch : Function for power spectral density estimation
    """

    # Input validation: overlap must be in the correct range
    if not (0 <= overlap < 1):
        raise ValueError("overlap must be between 0 and 1 (not inclusive).")

    # Calculate the size of each segment by dividing the total signal by number of bins
    # This is the direct approach:
    #   total_signal / [number_of_segments - (number_of_segments - 1) * overlap] = size_per_segment
    nperseg = int(num_points / (num_bins - ((num_bins - 1) * overlap)))

    # Check if the number of bins is viable for the signal size
    # If nperseg < 1, it means we have more bins than points, which is impossible
    if nperseg < 1:
        raise ValueError("n_bins is too large for the number of points.")

    # Calculate the number of overlapping points between consecutive segments
    # The overlap is a fraction of the segment size
    noverlap = int(nperseg * overlap)

    # For real signals, the FFT has Hermitian symmetry, so only half of the
    # frequencies are unique. The number of frequencies is (nperseg // 2) + 1
    # The +1 includes the DC component (zero frequency)
    psd_size = (nperseg // 2) + 1

    # Calculate the starting index of each segment window
    segment_start_indices = np.arange(num_bins) * int(nperseg - noverlap)

    # Calculate the ending index of each segment window
    segment_end_indices = segment_start_indices + int(nperseg)

    # Display formatted table header with diagnostic information
    print("=" * 73)
    # Display the key parameters that define the segmentation
    print(f"Size of each segment         = {nperseg}")
    print(f"Number of overlapping points = {noverlap}")
    print("=" * 73)
    # Computes the starting and ending indices for each segment window in the
    # Welch method and displays them in a formatted table
    print("Index  | Initial |   Final |    Size | Overlap")
    print("-" * 73)
    for segment_index in range(len(segment_start_indices)):
        print(f"{segment_index:6d} | {segment_start_indices[segment_index]:7.0f} | "
              f"{segment_end_indices[segment_index]:7.0f} | {nperseg:7.0f} | "
              f"{noverlap:7.0f}")
    print("=" * 73)

    # Return all calculated parameters
    # These can be used directly with scipy.signal.welch
    return nperseg, noverlap, psd_size
