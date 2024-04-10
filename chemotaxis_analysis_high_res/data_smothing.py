import numpy as np
def replace_outliers_with_nan(dataframe, column_name, threshold):
    """
    Replaces outliers in a specified column of a DataFrame with NaN. Outliers are defined as
    values that fall below (mean - threshold * standard deviation) or above
    (mean + threshold * standard deviation).

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to check for outliers.
    threshold (float): The number of standard deviations used to define the upper and lower limit.

    Returns:
    pd.DataFrame: A DataFrame with outliers replaced by NaN in the specified column.
    """
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

