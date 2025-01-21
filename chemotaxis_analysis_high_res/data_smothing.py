import numpy as np
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

def apply_smoothing(df, columns):
    """
    Smooths specified columns in the DataFrame using a rolling window average.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (str or list): The column(s) to smooth.

    Returns:
        pd.DataFrame: DataFrame with smoothed columns added as new columns.
    """
    # Default smoothing parameters
    smoothing_params = {
        'speed': 50,
        'radial_speed': 50,
        'reversal_frequency': 50,
        'bearing_angle': 50,
        'NI': 50,
        'curving_angle': 50,
        'distance_to_odor_centroid': 50,
        'conc_at_centroid': 50,
        'conc_at_0': 10,
        'dC_centroid': 50,
        'dC_0': 10,
    }

    if isinstance(columns, str):
        columns = [columns]  # Convert to list if a single column is provided

    for column in columns:
        if column in smoothing_params:
            window_size = smoothing_params[column]
            smoothed_column_name = f"{column}_smoothed"
            df[smoothed_column_name] = df[column].rolling(
                window=window_size, center=True, min_periods=1
            ).mean()
        else:
            raise ValueError(f"Column '{column}' is not in the smoothing dictionary.")

    return df



