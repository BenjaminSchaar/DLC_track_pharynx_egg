import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import re

def replace_outliers_with_nan(dataframe, columns, threshold):
    """
    Replaces outliers in specified columns of a DataFrame with NaN. Outliers are defined as
    values that fall below (mean - threshold * standard deviation) or above
    (mean + threshold * standard deviation).

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the data.
    columns (str or list): The column(s) to check for outliers.
    threshold (float): The number of standard deviations used to define the upper and lower limit.

    Returns:
    pd.DataFrame: A DataFrame with outliers replaced by NaN in the specified column(s).
    """
    if isinstance(columns, str):
        columns = [columns]  # Convert to list if a single column is provided

    for column_name in columns:
        # Calculate mean and standard deviation
        mean = dataframe[column_name].mean()
        std = dataframe[column_name].std()

        # Define the cut-off for outliers
        lower_limit = mean - threshold * std
        upper_limit = mean + threshold * std

        # Replace outliers with NaN
        dataframe[column_name] = dataframe[column_name].mask(
            (dataframe[column_name] < lower_limit) | (dataframe[column_name] > upper_limit), np.nan)

    return dataframe


def apply_smoothing(df, columns, fps):
    """
    Smooths specified columns in the DataFrame using a rolling window average.
    Window sizes are scaled by the frame rate (fps).

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (str or list): The column(s) to smooth.
        fps (float): Frames per second, used to scale window sizes.

    Returns:
        pd.DataFrame: DataFrame with smoothed columns added as new columns.
    """
    # Default smoothing parameters (in seconds, will be multiplied by fps)
    smoothing_params = {
        'speed_centroid': 1,
        'speed_center_0': 1,  # Assuming this is the format for the center point column
        'radial_speed': 1,
        'reversal_frequency': 1,
        'bearing_angle': 1,
        'NI': 1,
        'curving_angle': 1,
        'distance_to_odor_centroid': 1,
        'conc_at_centroid': 1,
        'conc_at_0': 1,
        'dC_centroid': 1,
        'dC_0': 1,
    }

    if isinstance(columns, str):
        columns = [columns]  # Convert to list if a single column is provided

    for column in columns:
        # Use get() with a default value of 10 if the column isn't in the dictionary
        base_window_size = smoothing_params.get(column, 10)

        # Scale window size by fps
        window_size = max(1, int(base_window_size * fps))

        # Print info about what window size is being used
        if column in smoothing_params:
            print(
                f"Smoothing column '{column}' with window size: {window_size} frames ({base_window_size} seconds at {fps} fps)")
        else:
            print(
                f"Column '{column}' is not in the smoothing dictionary (using default window size: {window_size} frames)")

        smoothed_column_name = f"{column}_smoothed"
        df[smoothed_column_name] = df[column].rolling(
            window=window_size, center=True, min_periods=1
        ).mean()

    return df


def smooth_trajectory_savitzky_golay_filter(df_column, window_length=11, poly_order=3):
    """
    Smooth a DataFrame column using Savitzky-Golay filter with linear interpolation for any NaN values.
    If data length is less than window_length, returns original data with a warning.

    Parameters:
    -----------
    df_column : pandas.Series
        DataFrame column containing trajectory data
    window_length : int
        Window length for Savitzky-Golay filter (will be made odd if even)
    poly_order : int
        Polynomial order for the filter (must be < window_length)

    Returns:
    --------
    smoothed_column : pandas.Series
        Smoothed version of input column with same index, or original column if smoothing not possible
    """
    # Convert window_length to integer explicitly
    window_length = int(window_length)
    poly_order = int(poly_order)

    # Check for NaN values and interpolate if any exist
    nan_count = df_column.isna().sum()
    if nan_count > 0:
        print(f"Found {nan_count} NaN values. Applying linear interpolation.")
        filled_data = df_column.interpolate(method='linear')
        # Handle edge cases if they exist
        filled_data = filled_data.fillna(method='bfill').fillna(method='ffill')
    else:
        filled_data = df_column

    # Check if window_length is odd, if not make it odd
    if window_length % 2 == 0:
        window_length += 1
        print(f"Window length was even, adjusted to {window_length}")

    # Validate poly_order is less than window_length
    if poly_order >= window_length:
        poly_order = window_length - 1
        print(f"Polynomial order was too high, adjusted to {poly_order}")

    # Check if we have enough data points for the Savitzky-Golay filter
    if len(filled_data) < window_length:
        print(
            f"WARNING: Not enough data points ({len(filled_data)}) for Savitzky-Golay filter with window_length={window_length}.")
        print("Skipping smoothing step and returning original data.")
        return df_column

    # Apply Savitzky-Golay smoothing
    smoothed_data = savgol_filter(filled_data, window_length, poly_order)

    # Return as pandas Series with original index
    return pd.Series(smoothed_data, index=df_column.index)