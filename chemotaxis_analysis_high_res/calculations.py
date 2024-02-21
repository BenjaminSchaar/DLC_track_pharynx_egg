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
    This function uses the relative stage position and the skeletton to calculate the real relative
    worm position inside the arena!

    Parameters:
    worm_pos (pd.DataFrame): DataFrame containing the worm's position.
    spline_X (pd.DataFrame): DataFrame with X coordinates of the worm's spline.
    spline_Y (pd.DataFrame): DataFrame with Y coordinates of the worm's spline.
    skel_pos (int): Index of the skeleton position (0-99) to use for correction.
                    -> if position = 100 , centroid is calculate as average of all positions (1-99
    video_resolution_x (int): Width of the video in pixels.
    video_resolution_y (int): Height of the video in pixels.
    factor_px_to_mm (float): Conversion factor from pixels to millimeters.

    Returns:
    pd.DataFrame: Updated DataFrame with the worm's corrected position.
    '''

    center_x = video_resolution_x / 2
    center_y = video_resolution_y / 2

    if skel_pos == 100: #calculate centroid
        # Convert all columns to NumPy arrays for centroid calculation
        column_skel_pos_x = spline_X.iloc[:, skel_pos].mean(axis=1).to_numpy()
        column_skel_pos_y = spline_Y.iloc[:, skel_pos].mean(axis=1).to_numpy()

    else:
        # Convert relevant columns to NumPy arrays for efficient computation
        column_skel_pos_x = spline_X.iloc[:, skel_pos].to_numpy()
        column_skel_pos_y = spline_Y.iloc[:, skel_pos].to_numpy()

    # Calculate the differences in px and mm using NumPy operations
    difference_x_px = column_skel_pos_x - center_x
    difference_y_px = column_skel_pos_y - center_y
    difference_center_x_mm = difference_x_px * factor_px_to_mm
    difference_center_y_mm = difference_y_px * factor_px_to_mm

    if skel_pos == 100:
        # Update worm_pos DataFrame with calculated values and name df column centroid
        worm_pos['X_rel_skel_pos_centroid'] = worm_pos['X_rel'] - difference_center_y_mm
        worm_pos['Y_rel_skel_pos_centroid'] = worm_pos['Y_rel'] - difference_center_x_mm

    else:
        # Update worm_pos DataFrame with calculated values
        worm_pos[f'X_rel_skel_pos_{skel_pos}'] = worm_pos['X_rel'] - difference_center_y_mm
        worm_pos[f'Y_rel_skel_pos_{skel_pos}'] = worm_pos['Y_rel'] - difference_center_x_mm

    return worm_pos


# Define a function to calculate distance while handling NaN
def calculate_distance(row: pd.Series, x_col: str, y_col: str, x_odor: float, y_odor: float) -> float:
    '''
    calculates distance to odot from x and y coordinates row by row from df.apply function

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

    df['time_seconds'] = df.index / fps
    return df


def calculate_preceived_conc(distance: float, time_seconds: float, conc_array: np.ndarray, distance_array: np.ndarray) -> float:

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


def calculate_angle(x:float, y:float, n_x:float, n_y:float, o_x:float, o_y:float) -> float:
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

    The resulting angle θ is measured in radians and can be converted to degrees if required. The angle provides insight into the directional relationship between the two vectors: if θ = 0, the vectors are in the same direction; if θ = π/2 (or 90 degrees), they are perpendicular; and if θ = π (or 180 degrees), they are in opposite directions.

    Usage:
    Understanding the angle between vectors is crucial in many fields such as robotics, navigation, computer graphics, and more, as it helps in determining the orientation and turning requirements between different paths or orientations.

    Note: The cosine inverse function, cos⁻¹, also known as arc cosine (acos), returns the principal value of the angle in radians, which lies in the range [0, π] for all possible dot product values.

    """

    point_n_minus = (n_x, n_y)  # Past position of the object
    point_n = (x, y)  # Current position of the object
    point_n_plus = (o_x, o_y) #future position of the object

    # Calculate the movement vector
    vector_past = (point_n_minus[0] - point_n[0], point_n_minus[1] - point_n[1])

    # Calculate the vector pointing to the odor source from the current position
    vector_future = (point_n_plus[0] - point_n[0], point_n_plus[1] - point_n[1])

    # Calculate dot product and magnitudes
    dot_product = vector_past[0] * vector_future[0] + vector_past[1] * vector_future[1]
    magnitude_movement = math.sqrt(vector_past[0] ** 2 + vector_past[1] ** 2)
    magnitude_to_odor = math.sqrt(vector_future[0] ** 2 + vector_future[1] ** 2)

    # Check for division by zero before calculating the angle
    if magnitude_movement * magnitude_to_odor == 0:
        return np.nan  # or any other value to indicate an undefined angle

    # Calculate the angle in radians
    angle = math.acos(dot_product / (magnitude_movement * magnitude_to_odor))

    # Convert the angle to degrees
    angle_degrees = math.degrees(angle)

    # Ensure the angle is within the range [0, 180]
    if angle_degrees > 180:
        angle_degrees = 360 - angle_degrees

    return angle_degrees
