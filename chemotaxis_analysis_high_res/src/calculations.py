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
        factor_px_to_mm: float,
        video_origin: str
) -> pd.DataFrame:
    '''
    This function uses the relative stage position and the skeleton to calculate the real relative
    worm position inside the arena.

    Parameters:
    worm_pos (pd.DataFrame): DataFrame containing the worm's position.
    spline_X (pd.DataFrame): DataFrame with X coordinates of the worm's spline.
    spline_Y (pd.DataFrame): DataFrame with Y coordinates of the worm's spline.
    skel_pos (int): Index of the skeleton position (0-99) to use for correction.
                    -> if position = 999 , centroid is calculated as average of all positions (1-99)
    video_resolution_x (int): Width of the video in pixels.
    video_resolution_y (int): Height of the video in pixels.
    factor_px_to_mm (float): Conversion factor from pixels to millimeters.
    video_origin (str): Origin of the video. "vid" for the original logic, "crop" for corrected logic.

    Returns:
    pd.DataFrame: Updated DataFrame with the worm's corrected position.
    '''

    video_resolution_x = int(video_resolution_x)
    video_resolution_y = int(video_resolution_y)
    factor_px_to_mm = float(factor_px_to_mm)

    center_x = video_resolution_x / 2
    center_y = video_resolution_y / 2

    if skel_pos == 999:  # calculate centroid
        column_skel_pos_x = spline_X.mean(axis=1).to_numpy().astype(float)
        column_skel_pos_y = spline_Y.mean(axis=1).to_numpy().astype(float)
    else:
        column_skel_pos_x = spline_X.iloc[:, skel_pos].to_numpy().astype(float)
        column_skel_pos_y = spline_Y.iloc[:, skel_pos].to_numpy().astype(float)

    difference_x_px = column_skel_pos_x - center_x
    difference_y_px = column_skel_pos_y - center_y

    difference_center_x_mm = difference_x_px * factor_px_to_mm
    difference_center_y_mm = difference_y_px * factor_px_to_mm

    worm_pos = worm_pos.copy()

    if video_origin == "zim01":
        # Working logic with swapped Y sign
        if skel_pos == 999:
            worm_pos['X_rel_skel_pos_centroid'] = worm_pos['X_rel'] - difference_center_x_mm
            worm_pos['Y_rel_skel_pos_centroid'] = worm_pos['Y_rel'] + difference_center_y_mm
        else:
            worm_pos[f'X_rel_skel_pos_{skel_pos}'] = worm_pos['X_rel'] - difference_center_x_mm
            worm_pos[f'Y_rel_skel_pos_{skel_pos}'] = worm_pos['Y_rel'] + difference_center_y_mm

    if video_origin == "zim06":
        # Working logic with swapped Y sign
        if skel_pos == 999:
            worm_pos['X_rel_skel_pos_centroid'] = worm_pos['X_rel'] + difference_center_x_mm
            worm_pos['Y_rel_skel_pos_centroid'] = worm_pos['Y_rel'] - difference_center_y_mm
        else:
            worm_pos[f'X_rel_skel_pos_{skel_pos}'] = worm_pos['X_rel'] + difference_center_x_mm
            worm_pos[f'Y_rel_skel_pos_{skel_pos}'] = worm_pos['Y_rel'] - difference_center_y_mm

    elif video_origin == "crop":
        # Logic without Y flip
        if skel_pos == 999:
            worm_pos['X_rel_skel_pos_centroid'] = worm_pos['X_rel'] + difference_center_x_mm
            worm_pos['Y_rel_skel_pos_centroid'] = worm_pos['Y_rel'] + difference_center_y_mm
        else:
            worm_pos[f'X_rel_skel_pos_{skel_pos}'] = worm_pos['X_rel'] + difference_center_x_mm
            worm_pos[f'Y_rel_skel_pos_{skel_pos}'] = worm_pos['Y_rel'] + difference_center_y_mm

    return worm_pos

def correct_dlc_coordinates(
        worm_pos: pd.DataFrame,
        dlc_nose_coords: pd.DataFrame,
        dlc_tail_coords: pd.DataFrame,
        video_resolution_x: int,
        video_resolution_y: int,
        factor_px_to_mm: float,
        video_origin: str
) -> tuple:
    """
    Convert DLC coordinates from pixels to absolute millimeter positions inside the arena,
    correcting for the stage-encoded worm position and video origin.

    Parameters:
    worm_pos (pd.DataFrame): DataFrame containing the stage-based worm positions:
                             must have columns 'X_rel' and 'Y_rel' (in mm).
    dlc_nose_coords (pd.DataFrame): DataFrame with nose 'x','y' pixel coordinates.
    dlc_tail_coords (pd.DataFrame): DataFrame with tail 'x','y' pixel coordinates.
    video_resolution_x (int): Width of the video in pixels.
    video_resolution_y (int): Height of the video in pixels.
    factor_px_to_mm (float): Conversion factor from pixels to millimeters.
    video_origin (str): Origin of the video. One of 'zim01', 'zim06', or 'crop'.

    Returns:
    tuple: Two DataFrames (nose, tail) each with new columns:
           'X_rel_DLC_nose', 'Y_rel_DLC_nose' or
           'X_rel_DLC_tail', 'Y_rel_DLC_tail' giving absolute arena positions.
    """
    # Ensure numeric types
    video_resolution_x = int(video_resolution_x)
    video_resolution_y = int(video_resolution_y)
    factor_px_to_mm = float(factor_px_to_mm)

    # Center of frame in pixels
    center_x = video_resolution_x / 2.0
    center_y = video_resolution_y / 2.0

    # Copy to avoid modifying originals
    nose = dlc_nose_coords.copy()
    tail = dlc_tail_coords.copy()

    # Pixel offsets from center
    dx_n = nose['x'] - center_x
    dy_n = nose['y'] - center_y
    dx_t = tail['x'] - center_x
    dy_t = tail['y'] - center_y

    # Convert pixel offsets to mm
    mx_n = dx_n * factor_px_to_mm
    my_n = dy_n * factor_px_to_mm
    mx_t = dx_t * factor_px_to_mm
    my_t = dy_t * factor_px_to_mm

    # Apply origin-specific sign and add stage pos
    if video_origin == 'zim01':
        # X shifts subtract, Y shifts add
        nose['X_rel_DLC_nose'] = worm_pos['X_rel'] - mx_n
        nose['Y_rel_DLC_nose'] = worm_pos['Y_rel'] + my_n
        tail['X_rel_DLC_tail'] = worm_pos['X_rel'] - mx_t
        tail['Y_rel_DLC_tail'] = worm_pos['Y_rel'] + my_t

    elif video_origin == 'zim06':
        # X shifts add, Y shifts subtract
        nose['X_rel_DLC_nose'] = worm_pos['X_rel'] + mx_n
        nose['Y_rel_DLC_nose'] = worm_pos['Y_rel'] - my_n
        tail['X_rel_DLC_tail'] = worm_pos['X_rel'] + mx_t
        tail['Y_rel_DLC_tail'] = worm_pos['Y_rel'] - my_t

    elif video_origin == 'crop':
        # No inversion: both X and Y add
        nose['X_rel_DLC_nose'] = worm_pos['X_rel'] + mx_n
        nose['Y_rel_DLC_nose'] = worm_pos['Y_rel'] + my_n
        tail['X_rel_DLC_tail'] = worm_pos['X_rel'] + mx_t
        tail['Y_rel_DLC_tail'] = worm_pos['Y_rel'] + my_t

    else:
        raise ValueError(f"Unknown video_origin: {video_origin}")

    return nose, tail

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
    Calculates a new column with time in seconds passed from the "frame" column (if it exists) or from the index, depending on the structure of the DataFrame.
    :param df: The DataFrame containing tracking data
    :param fps: FPS (frames per second) of the recording
    :return: DataFrame with a new 'time_seconds' column
    '''
    print('calc time in seconds for:', df.head())
    fps = float(fps)

    if "frame" in df.columns:
        # If "frame" column exists, use it to calculate time in seconds
        df['time_seconds'] = df['frame'].astype(float) / fps
    else:
        # If no "frame" column, use index to calculate time in seconds
        if not pd.api.types.is_numeric_dtype(df.index):
            df.index = pd.to_numeric(df.index,
                                     errors='coerce')  # 'coerce' turns invalid values into NaNs, adjust as needed
        df['time_seconds'] = df.index.to_series().astype(float) / fps

    return df


def calculate_preceived_conc(distance: float, time_seconds: float, conc_array: np.ndarray, distance_array: np.ndarray,
                             diffusion_time_offset: int) -> float:
    # Check for NaN distance first
    if np.isnan(distance):
        return np.nan

    # Type enforcement
    distance = float(distance)
    time_seconds = float(time_seconds)
    conc_array = np.asarray(conc_array)
    distance_array = np.asarray(distance_array)

    # Convert time to array index, rounding to the nearest second
    sim_time_array = round(time_seconds)

    # Offset the diffusion simulation time if the assay started later than 1 hour
    sim_time_array = sim_time_array + diffusion_time_offset

    # Get all distance values in the chosen frame
    distances_in_frame = distance_array[sim_time_array]

    # Find the closest value to variable_value
    closest_distance = min(distances_in_frame, key=lambda x: abs(x - distance))

    # Finding the index of the closest value to closest_distance
    index_of_closest_distance = np.where(distances_in_frame == closest_distance)[0]

    # Extract the concentration value as a scalar and force to float
    conc_value = float(
        conc_array[sim_time_array][index_of_closest_distance][0])  # Take the first element if it's an array

    return conc_value

def calculate_displacement_vector(df_worm_parameter):
    '''
    Calculate and add the displacement vector components to the DataFrame.
    Parameters:
    df_worm_parameter (pd.DataFrame): DataFrame containing columns:
                                     'X_rel_skel_pos_centroid_corrected' and
                                     'Y_rel_skel_pos_centroid_corrected'
    Returns:
    pd.DataFrame: Original DataFrame with new columns added:
                  - 'dx_dt': Rate of change in x direction
                  - 'dy_dt': Rate of change in y direction
                  - 'displacement_vector_degrees': Angle of the displacement vector in degrees
                  - 'displacement_magnitude': Length of the displacement vector
    '''
    # Update required columns for single-level index
    required_columns = [
        'X_rel_skel_pos_centroid',
        'Y_rel_skel_pos_centroid'
    ]
    missing_columns = [col for col in required_columns if col not in df_worm_parameter.columns]
    if missing_columns:
        raise ValueError(f"DataFrame must contain columns: {missing_columns}")

    # Extract x and y coordinates using single-level column names
    x = df_worm_parameter['X_rel_skel_pos_centroid'].values
    y = df_worm_parameter['Y_rel_skel_pos_centroid'].values

    # Calculate gradients (dx/dt and dy/dt)
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)

    # Calculate direction in radians and then convert to degrees
    direction_radians = np.arctan2(dy_dt, dx_dt)
    direction_degrees = np.degrees(direction_radians)

    # Calculate the displacement magnitude (Euclidean norm)
    displacement_magnitude = np.sqrt(dx_dt**2 + dy_dt**2)

    # Add new columns to the DataFrame with single-level indexing
    df_worm_parameter['centroid_dx_dt'] = dx_dt
    df_worm_parameter['centroid_dy_dt'] = dy_dt
    df_worm_parameter['centroid_displacement_vector_degrees'] = direction_degrees
    df_worm_parameter['centroid_displacement_magnitude'] = displacement_magnitude

    return df_worm_parameter


def calculate_curving_angle(df_worm_parameter, window_size=1):
    '''
    Calculate the change in bearing angle over a specified range of frames.

    Parameters:
    df_worm_parameter (pd.DataFrame): DataFrame with 'displacement_vector_degrees' column
                                      (output of calculate_displacement_vector).
    bearing_range (int): Number of frames to look back when calculating bearing change.

    Returns:
    pd.DataFrame: Original DataFrame with new 'bearing_change' column added.
    '''
    if 'centroid_displacement_vector_degrees' not in df_worm_parameter.columns:
        raise ValueError(
            "DataFrame must contain 'centroid_displacement_vector_degrees' column. Run calculate_displacement_vector first.")

    displacement_vector_degrees = df_worm_parameter['centroid_displacement_vector_degrees'].values
    bearing_change = np.zeros_like(displacement_vector_degrees)

    for i in range(window_size, len(displacement_vector_degrees)):
        change = displacement_vector_degrees[i] - displacement_vector_degrees[i - window_size]
        # Ensure the change is in the range [-180, 180]
        bearing_change[i] = (change + 180) % 360 - 180

    df_worm_parameter['curving_angle'] = bearing_change

    return df_worm_parameter


def calculate_bearing_angle(df):
    """
    Calculate the bearing angle between the worm's movement direction and the direction to the odor source.

    The bearing angle is defined as the angle between:
    1. The vector of recent movement (from centroid displacement vector)
    2. The vector pointing from current position to odor position

    Parameters:
    df (pd.DataFrame): DataFrame containing required parameters with single-index columns

    Returns:
    pd.DataFrame: Original DataFrame with new bearing_angle_degrees column added
    """
    # Get movement direction (already calculated)
    movement_angle = df['centroid_displacement_vector_degrees']

    # Calculate vector to odor source
    x_to_odor = df['odor_x'] - df['X_rel_skel_pos_centroid']
    y_to_odor = df['odor_y'] - df['Y_rel_skel_pos_centroid']

    # Calculate angle to odor (in degrees)
    angle_to_odor = np.degrees(np.arctan2(y_to_odor, x_to_odor))

    # Calculate bearing angle (difference between movement direction and angle to odor)
    bearing_angle = angle_to_odor - movement_angle

    # Normalize angle to be between -180 and 180 degrees
    bearing_angle = ((bearing_angle + 180) % 360) - 180

    # Add to DataFrame
    df['bearing_angle'] = abs(bearing_angle)

    return df

def calculate_centroid_speed(df, fps):
    '''
    This function calculates the speed per second using the displacement magnitude
    and adds the column 'speed'.

    :param df: pandas DataFrame with column 'displacement_magnitude'
    :param fps: frames per second of the video
    :return: DataFrame with an additional 'speed' column
    '''

    # Ensure the required column exists
    if 'centroid_displacement_magnitude' not in df.columns:
        raise ValueError("DataFrame must contain 'centroid_displacement_magnitude' column")

    # Calculate the speed (displacement magnitude per frame) and convert it to per second by multiplying by fps
    df['speed_centroid'] = df['centroid_displacement_magnitude'] * fps

    return df

def calculate_center_speed(df, fps, center_point):
    '''
    This function calculates the speed per second using the displacement magnitude
    and adds the column 'speed'.

    :param df: pandas DataFrame with columns for X and Y positions
    :param fps: frames per second of the video
    :param center_point: the name of the center point (e.g., 'centroid', 'head')
    :return: DataFrame with additional columns for displacement vector and speed
    '''
    # Extract x and y coordinates
    x = df[f'X_rel_skel_pos_{center_point}'].values
    y = df[f'Y_rel_skel_pos_{center_point}'].values

    # Calculate gradients (dx/dt and dy/dt)
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)

    # Calculate direction in radians and then convert to degrees
    direction_radians = np.arctan2(dy_dt, dx_dt)
    direction_degrees = np.degrees(direction_radians)

    # Calculate the displacement magnitude (Euclidean norm)
    displacement_magnitude = np.sqrt(dx_dt**2 + dy_dt**2)

    # Add new columns to the DataFrame
    df[f'{center_point}_dx_dt'] = dx_dt
    df[f'{center_point}_dy_dt'] = dy_dt
    df[f'{center_point}_displacement_vector_degrees'] = direction_degrees
    df[f'{center_point}_displacement_magnitude'] = displacement_magnitude

    # Calculate the speed (displacement magnitude per frame) and convert it to per second by multiplying by fps
    df[f'speed_center_{center_point}'] = df[f'{center_point}_displacement_magnitude'] * fps

    return df


def calculate_radial_speed(df, fps):
    '''
    Calculates radial speed (speed towards/away from the odor source) using change in distance over time.
    Positive values indicate movement toward the odor source, negative values indicate movement away.

    Parameters:
    df (pd.DataFrame): DataFrame containing distance_to_odor_centroid column
    fps (float): Frames per second of the video

    Returns:
    pd.DataFrame: Original DataFrame with new 'radial_speed' column added
    '''
    # Calculate radial speed using gradient of distance
    #e.g Speed is 2 mm/frame × 10 frames/second = 20 mm/second 2mm/frame×10frames/second=20mm/second.
    df['radial_speed'] = np.gradient(df['distance_to_odor_centroid']) * fps
    df['radial_speed'] = -df['radial_speed']

    return df


def calculate_min_border_distance(df, arena_max_x, arena_max_y, x_col='X_rel_skel_pos_centroid', y_col='Y_rel_skel_pos_centroid',
                                  arena_min_x=0, arena_min_y=0):
    """
    Calculate the minimum distance from each point (x,y) to the closest border of a rectangular arena.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing coordinates
    arena_max_x : float
        Maximum x-coordinate (width) of the arena in mm
    arena_max_y : float
        Maximum y-coordinate (height) of the arena in mm
    x_col : str, optional
        Name of the column containing x coordinates, defaults to 'x'
    y_col : str, optional
        Name of the column containing y coordinates, defaults to 'Y_rel_skel_pos_centroid'
    arena_min_x : float, optional
        Minimum x-coordinate of the arena, defaults to 0
    arena_min_y : float, optional
        Minimum y-coordinate of the arena, defaults to 0

    Returns:
    --------
    pandas.Series
        Series containing minimum distance to any border
    """
    # Calculate distances to each border
    dist_to_left = df[x_col] - arena_min_x
    dist_to_right = arena_max_x - df[x_col]
    dist_to_top = df[y_col] - arena_min_y
    dist_to_bottom = arena_max_y - df[y_col]

    # Find minimum distance using pandas minimum function
    min_distances = pd.concat([dist_to_left, dist_to_right, dist_to_top, dist_to_bottom],
                              axis=1).min(axis=1)

    return min_distances





