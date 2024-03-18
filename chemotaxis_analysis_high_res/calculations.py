import pandas as pd
import numpy as np
from scipy.spatial import distance
import math

def interpolate_df(vector, length) -> np.ndarray:
    return np.interp(np.linspace(0, len(vector) - 1, length), np.arange(len(vector)), vector)

import pandas as pd

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
        column_skel_pos_x = spline_X.iloc[:, 0:100].mean(axis=1).to_numpy().astype(float)
        column_skel_pos_y = spline_Y.iloc[:, 0:100].mean(axis=1).to_numpy().astype(float)
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


def calculate_angle(x:float, y:float, n_x:float, n_y:float, o_x:float, o_y:float, angle_type) -> float:
    """
    Angle Calculation Between Two Vectors:

    Introduction:
    The angle between two vectors in a plane is a measure of the difference in their directions. This concept is widely used in physics, engineering, and mathematics to understand and quantify the orientation of different entities relative to each other.

    Definitions:
    1. Vector: A vector is a mathematical object that has magnitude (or length) and direction. It can be represented in a coordinate system by coordinates (x, y) in two-dimensional space or (x, y, z) in three-dimensional space.

    2. Dot Product: The dot product (also known as the scalar product) between two vectors A and B, represented as A·B, is a measure of the vectors' mutual alignment. It is defined as:
       A·B = |A| |B| cos(θ)
       where |A| and |B| are the magnitudes (lengths) of vectors A and B, respectively, and θ is the angle between them.

    3. Magnitude of a Vector: The magnitude (or length) of a vector A = (a1, a2) in two-dimensional space is given by:
       |A| = sqrt(a1² + a2²)

    Angle Between Vectors:
    The angle θ between two vectors can be found using the dot product and magnitudes of the vectors. If we have two vectors, A = (a1, a2) and B = (b1, b2), the angle θ between them can be calculated as follows:

    1. Calculate the dot product of vectors A and B:
       A·B = a1*b1 + a2*b2

    2. Calculate the magnitudes of vectors A and B:
       |A| = sqrt(a1² + a2²)
       |B| = sqrt(b1² + b2²)

    3. Find the cosine of the angle between A and B:
       cos(θ) = (A·B) / (|A| |B|)

    4. Determine the angle θ:
       θ = cos⁻¹[(A·B) / (|A| |B|)]

    """
    x, y, n_x, n_y, o_x, o_y = map(float, [x, y, n_x, n_y, o_x, o_y])

    a = (n_x, n_y)  # Past position of the object
    b = (x, y)  # Current position of the object
    c = (o_x, o_y) # future position

    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))

    ang = abs(ang)  # Take the absolute value of the angle

    ang = ang if ang <= 180 else 360 - ang  # Ensure the angle is within the range 0 to 180

    if (angle_type == 'curving_angle'):
        # Calculate the complementary angle
            ang = 180 - ang

    return ang


def calculate_speed(df, fps):
    '''
    This function calculates speed per seconds of the centroid position and adds the column speed
    :param df:
    :param fps:
    :return:
    '''
    # Assuming df is a pandas DataFrame with columns 'X_rel' and 'Y_rel'
    df['speed'] = ((df['X_rel_skel_pos_centroid'].diff() ** 2 + df['Y_rel_skel_pos_centroid'].diff() ** 2) ** 0.5) * fps

    # Assuming df is your DataFrame
    smoothing_window_size = int(fps*2)

    # Smooth the 'speed' column using a rolling window and taking the mean
    df['speed'] = df['speed'].rolling(window=smoothing_window_size, min_periods=1).mean()

    return df

def calculate_radial_speed(df, fps):
    '''
    calculates raidal speed (speed towards the odor source) by using the change in distance to the odor over time
    in mm/second and adda the column radial speed

    :param df:
    :param fps:
    :return:
    '''

    # Assuming df is your DataFrame
    smoothing_window_size = int(fps*2)

    df['radial_speed'] = df['distance_to_odor_centroid'].diff() * fps
    # Smooth the 'speed' column using a rolling window and taking the mean
    df['radial_speed'] = df['radial_speed'].rolling(window=smoothing_window_size, min_periods=1).mean()

    return df

#def clean_reversal():