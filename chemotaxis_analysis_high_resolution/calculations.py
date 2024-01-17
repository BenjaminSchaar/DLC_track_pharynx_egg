import pandas as pd
import numpy as np
from scipy.spatial import distance


def interpolate_df(vector, indices, length):
    vector = np.interp(indices, np.linspace(0, 1, length), vector)
    return vector


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
    Corrects the position of a worm on a stage in a video frame.

    This function adjusts the worm's position in a video by calculating the differences
    between the worm's skeleton position and the center of the video frame, both in pixels
    and in millimeters. It updates the worm's position relative to the specified skeleton
    position.

    Parameters:
    worm_pos (pd.DataFrame): DataFrame containing the worm's position.
    spline_X (pd.DataFrame): DataFrame with X coordinates of the worm's spline.
    spline_Y (pd.DataFrame): DataFrame with Y coordinates of the worm's spline.
    skel_pos (int): Index of the skeleton position to use for correction.
    video_resolution_x (int): Width of the video in pixels.
    video_resolution_y (int): Height of the video in pixels.
    factor_px_to_mm (float): Conversion factor from pixels to millimeters.

    Returns:
    pd.DataFrame: Updated DataFrame with the worm's corrected position.
    '''

    center_x = video_resolution_x / 2
    center_y = video_resolution_y / 2

    # Convert relevant columns to NumPy arrays for efficient computation
    column_skel_pos_x = spline_X.iloc[:, skel_pos].to_numpy()
    column_skel_pos_y = spline_Y.iloc[:, skel_pos].to_numpy()

    # Calculate the differences in px and mm using NumPy operations
    difference_x_px = column_skel_pos_x - center_x
    difference_y_px = column_skel_pos_y - center_y
    difference_center_x_mm = difference_x_px * factor_px_to_mm
    difference_center_y_mm = difference_y_px * factor_px_to_mm

    # Update worm_pos DataFrame with calculated values
    worm_pos[f'X_rel_skel_pos_{skel_pos}'] = worm_pos['X_rel'] - difference_center_y_mm
    worm_pos[f'Y_rel_skel_pos_{skel_pos}'] = worm_pos['Y_rel'] - difference_center_x_mm

    return worm_pos


# Define a function to calculate distance while handling NaN
def calculate_distance(row: pd.series, x_col: str, y_col: str, x_odor: int, y_odor: int) -> float:
    x_rel, y_rel = row[x_col], row[y_col]
    if np.isnan(x_rel) or np.isnan(y_rel):
        return np.nan  # or you can return a default value, like -1 or 0
    return distance.euclidean((x_rel, y_rel), (x_odor, y_odor))


def calculate_time_in_seconds(df: pd.DataFrame, fps: int):
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