import pandas as pd
import numpy as np
from scipy.spatial import distance
import math

def interpolate_df(vector, length) -> np.ndarray:
    return np.interp(np.linspace(0, len(vector) - 1, length), np.arange(len(vector)), vector)

def correct_stage_pos_with_skeleton(
        worm_pos: pd.DataFrame,
        spline_X: pd.DataFrame,
        spline_Y: pd.DataFrame,
        skel_pos: int,
        video_resolution_x: int,
        video_resolution_y: int,
        factor_px_to_mm: float
) -> pd.DataFrame:
    '''
    This function uses the relative stage position and the skeleton to calculate the real relative
    worm position inside the arena!

    Parameters:
    worm_pos (pd.DataFrame): DataFrame containing the worm's position.
    spline_X (pd.DataFrame): DataFrame with X coordinates of the worm's spline.
    spline_Y (pd.DataFrame): DataFrame with Y coordinates of the worm's spline.
    skel_pos (int): Index of the skeleton position (0-99) to use for correction.
                    -> if position = 100 , centroid is calculated as average of all positions (1-99)
    video_resolution_x (int): Width of the video in pixels.
    video_resolution_y (int): Height of the video in pixels.
    factor_px_to_mm (float): Conversion factor from pixels to millimeters.

    Returns:
    pd.DataFrame: Updated DataFrame with the worm's corrected position.
    '''

    print("running func correct_stage_pos_with_skeleton for skel_pos:", skel_pos)

    video_resolution_x = int(video_resolution_x)
    video_resolution_y = int(video_resolution_y)
    factor_px_to_mm = float(factor_px_to_mm)

    center_x = video_resolution_x / 2
    center_y = video_resolution_y / 2

    if skel_pos == 100:  # calculate centroid
        # Calculate the mean of all columns for each row dynamically
        column_skel_pos_x = spline_X.mean(axis=1).to_numpy().astype(float)
        column_skel_pos_y = spline_Y.mean(axis=1).to_numpy().astype(float)
    else:
        column_skel_pos_x = spline_X.iloc[:, skel_pos].to_numpy().astype(float)
        column_skel_pos_y = spline_Y.iloc[:, skel_pos].to_numpy().astype(float)

    difference_x_px = column_skel_pos_x - center_x
    difference_y_px = column_skel_pos_y - center_y

    difference_center_x_mm = difference_x_px * factor_px_to_mm
    difference_center_y_mm = difference_y_px * factor_px_to_mm

    if skel_pos == 100:
        worm_pos['X_rel_skel_pos_centroid'] = worm_pos['X_rel'] - difference_center_y_mm
        worm_pos['Y_rel_skel_pos_centroid'] = worm_pos['Y_rel'] - difference_center_x_mm
    else:
        worm_pos[f'X_rel_skel_pos_{skel_pos}'] = worm_pos['X_rel'] - difference_center_y_mm
        worm_pos[f'Y_rel_skel_pos_{skel_pos}'] = worm_pos['Y_rel'] - difference_center_x_mm

    return worm_pos



# Define a function to calculate distance while handling NaN
def calculate_distance(row: pd.Series, x_col: str, y_col: str, x_odor: float, y_odor: float) -> float:
    '''
    calculates distance to odor from x and y coordinates row by row from df.apply function

    :param row: df row as pd.series
    :param x_col: x column name as string
    :param y_col: y column name as string
    :param x_odor: relative x position of odor as float
    :param y_odor: relative y position of odor as float
    :return: new column wit euclidian distance to odor
    '''
    x_rel, y_rel = row[x_col], row[y_col]
    if np.isnan(x_rel) or np.isnan(y_rel):
        return np.nan  # or you can return a default value, like -1 or 0
    return distance.euclidean((x_rel, y_rel), (x_odor, y_odor))


def calculate_time_in_seconds(df: pd.DataFrame, fps: int):
    '''
    calculates new column with time in seconds passed from column index (1 frame) and fps
    :param df:
    :param fps: fps of recording
    :return:
    '''
    print('calc time in seconds for:', df.head())
    fps = float(fps)
    # Convert index to numeric if necessary (this assumes index should be integer frame numbers)
    if not pd.api.types.is_numeric_dtype(df.index):
        df.index = pd.to_numeric(df.index, errors='coerce')  # 'coerce' turns invalid values into NaNs, adjust as needed

    df['time_seconds'] = df.index.to_series().astype(float) / fps
    return df


def calculate_preceived_conc(distance: float, time_seconds: float, conc_array: np.ndarray, distance_array: np.ndarray) -> float:

    # Type enforcement
    distance = float(distance)
    time_seconds = float(time_seconds)
    conc_array = np.asarray(conc_array)
    distance_array = np.asarray(distance_array)

    # finds the proper frame of the sim, the sim runs in 10 second steps with 0.1 fps
    sim_time = int(0)
    sim_time = int(time_seconds // 10 * 10)  # finds the simulation time that is closest to the real time

    sim_time_array = int(sim_time / 10)

    # Get all distance values in the chosen frame
    distances_in_frame = distance_array[sim_time_array]

    # Find the closest value to variable_value
    closest_distance = min(distances_in_frame, key=lambda x: abs(x - distance))

    # Finding the index of the closest value to closest_distance
    index_of_closest_distance = np.where(distances_in_frame == closest_distance)[0]

    conc_value = conc_array[sim_time_array][index_of_closest_distance]

    return conc_value

def calculate_displacement_vector(df_worm_parameter):
    '''
    Calculate and add the displacement vector components to the DataFrame.

    This function computes the components of the displacement vector
    for each time point in the worm's movement data. It approximates
    the instantaneous velocity vector, which represents the rate of
    change of position with respect to time.

    The function uses numpy.gradient to estimate the rate of change
    of x and y positions, then applies numpy.arctan2 to compute the
    angle of the displacement vector.

    Parameters:
    df_worm_parameter (pd.DataFrame): DataFrame with 'X_rel_skel_pos_centroid_corrected'
                                      and 'Y_rel_skel_pos_centroid_corrected' columns
                                      representing worm positions over time.

    Returns:
    pd.DataFrame: Original DataFrame with new columns added:
                  - 'dx_dt': Rate of change in x direction
                  - 'dy_dt': Rate of change in y direction
                  - 'direction': Angle of the displacement vector in degrees

    Note:
    The 'direction' column contains angles in degrees, where:
    - 0° points right (+x direction)
    - 90° points up (+y direction)
    - 180° or -180° points left (-x direction)
    - -90° points down (-y direction)
    Angles increase counterclockwise and range from -180° to 180°.

        90°
       |
       |
    180° --- 0° (or 360°)
       |
       |
     -90°

    '''
    # Ensure the required columns exist
    required_columns = ['X_rel_skel_pos_centroid_corrected', 'Y_rel_skel_pos_centroid_corrected']
    if not all(col in df_worm_parameter.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Extract x and y coordinates
    x = df_worm_parameter['X_rel_skel_pos_centroid_corrected'].values
    y = df_worm_parameter['Y_rel_skel_pos_centroid_corrected'].values

    # Calculate gradients
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)

    # Calculate direction
    direction_radians = np.arctan2(dy_dt, dx_dt)
    direction_degrees = np.degrees(direction_radians)

    df_worm_parameter['displacement_vector_degrees'] = direction_degrees

    return df_worm_parameter

def calculate_curving_angle(df_worm_parameter, bearing_range=1):
    '''
    Calculate the change in bearing angle over a specified range of frames.

    Parameters:
    df_worm_parameter (pd.DataFrame): DataFrame with 'displacement_vector_degrees' column
                                      (output of calculate_displacement_vector).
    bearing_range (int): Number of frames to look back when calculating bearing change.

    Returns:
    pd.DataFrame: Original DataFrame with new 'bearing_change' column added.
    '''
    if 'displacement_vector_degrees' not in df_worm_parameter.columns:
        raise ValueError(
            "DataFrame must contain 'displacement_vector_degrees' column. Run calculate_displacement_vector first.")

    displacement_vector_degrees = df_worm_parameter['displacement_vector_degrees'].values
    bearing_change = np.zeros_like(displacement_vector_degrees)

    for i in range(bearing_range, len(displacement_vector_degrees)):
        change = displacement_vector_degrees[i] - displacement_vector_degrees[i - bearing_range]
        # Ensure the change is in the range [-180, 180]
        bearing_change[i] = (change + 180) % 360 - 180

    df_worm_parameter['curving_angle'] = bearing_change

    return df_worm_parameter

def calculate_bearing_angle(df_worm_parameter, x_odor, y_odor):
    '''
    Calculate the bearing angle between the worm's displacement vector and the vector towards a defined odor source in 2D space.

    Mathematical description:
    1. Use two vectors:
       a) Displacement vector of the worm (already provided in degrees)
       b) Vector from worm to odor source (calculated)
    2. Compute the angle between these vectors by subtracting the displacement vector angle from the angle to the odor source.
    3. Normalize the resulting angle to be within the range [-180, 180] degrees.

    Angle interpretation:
    - The angle ranges from -180 to +180 degrees.
    - Positive angle: The worm's trajectory is to the left of the vector pointing to the odor source.
    - Negative angle: The worm's trajectory is to the right of the vector pointing to the odor source.
    - 0 degrees: The worm is moving directly towards the odor source.
    - ±180 degrees: The worm is moving directly away from the odor source.

    Parameters:
    df_worm_parameter (pd.DataFrame): DataFrame with 'X_rel_skel_pos_centroid_corrected',
    'Y_rel_skel_pos_centroid_corrected', and 'displacement_vector_degrees' columns.
    x_odor (float): X-coordinate of the odor source.
    y_odor (float): Y-coordinate of the odor source.

    Returns:
    pd.DataFrame: Original DataFrame with new 'bearing_angle_degrees' column added.
    '''
    required_columns = ['X_rel_skel_pos_centroid_corrected', 'Y_rel_skel_pos_centroid_corrected', 'displacement_vector_degrees']
    if not all(col in df_worm_parameter.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {', '.join(required_columns)}")

    # Calculate vector from worm to odor source
    dx_to_odor = x_odor - df_worm_parameter['X_rel_skel_pos_centroid_corrected']
    dy_to_odor = y_odor - df_worm_parameter['Y_rel_skel_pos_centroid_corrected']

    # Calculate angle to odor source
    angle_to_odor = np.degrees(np.arctan2(dy_to_odor, dx_to_odor))

    # Calculate bearing angle
    bearing_angle = angle_to_odor - df_worm_parameter['displacement_vector_degrees']

    # Normalize angle to [-180, 180] range
    #bearing_angle = (bearing_angle + 180) % 360 - 180

    df_worm_parameter['bearing_angle'] = bearing_angle

    return df_worm_parameter

def calculate_speed(df, fps):
    '''
    This function calculates the speed per second of the centroid position and adds the column speed
    :param df: pandas DataFrame with columns 'X_rel_skel_pos_centroid' and 'Y_rel_skel_pos_centroid'
    :param fps: frames per second of the video
    :return: DataFrame with an additional 'speed' column
    '''
    # Smoothing window size for position, to reduce noise
    position_smoothing_window = 2  # For averaging the latest 2 positions

    # Apply rolling mean to smooth positions
    df['X_smooth'] = df['X_rel_skel_pos_centroid'].rolling(window=position_smoothing_window, min_periods=1).mean()
    df['Y_smooth'] = df['Y_rel_skel_pos_centroid'].rolling(window=position_smoothing_window, min_periods=1).mean()

    # Calculate the difference between consecutive smoothed positions
    df['X_diff'] = df['X_smooth'].diff()
    df['Y_diff'] = df['Y_smooth'].diff()

    # Calculate the speed (distance traveled per frame) and convert to per second
    df['speed'] = ((df['X_diff'] ** 2 + df['Y_diff'] ** 2) ** 0.5) * fps

    # Further smooth the 'speed' column to reduce variability
    speed_smoothing_window_size = int(fps * 2)  # For a 2-second window
    df['speed'] = df['speed'].rolling(window=speed_smoothing_window_size, min_periods=1).mean()

    # Drop intermediate columns used for calculation
    df.drop(columns=['X_smooth', 'Y_smooth', 'X_diff', 'Y_diff'], inplace=True)

    return df


def calculate_radial_speed(df, fps):
    '''
    Calculates radial speed (speed towards the odor source) by using the change in distance to the odor over time
    in mm/second and adds the column radial_speed.

    :param df: pandas DataFrame with column 'distance_to_odor_centroid'
    :param fps: frames per second of the video
    :return: DataFrame with an additional 'radial_speed' column
    '''
    temp_df = df[['distance_to_odor_centroid']].copy()

    # Smoothing window size for distance, to reduce noise
    distance_smoothing_window = 2  # For averaging the latest 2 distances

    # Apply rolling mean to smooth distances
    temp_df['distance_smooth'] = temp_df['distance_to_odor_centroid'].rolling(window=distance_smoothing_window,
                                                                              min_periods=1).mean()

    # Calculate the difference between consecutive smoothed distances
    temp_df['distance_diff'] = temp_df['distance_smooth'].diff()

    # Calculate the radial speed (change in distance per frame) and convert to per second
    temp_df['radial_speed'] = temp_df['distance_diff'] * fps

    # Further smooth the 'radial_speed' column to reduce variability
    speed_smoothing_window_size = int(fps * 2)  # For a 2-second window
    temp_df['radial_speed'] = temp_df['radial_speed'].rolling(window=speed_smoothing_window_size, min_periods=1).mean()

    # Add the radial speed column to the original DataFrame
    df['radial_speed'] = temp_df['radial_speed']

    return df
