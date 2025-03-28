import pandas as pd
import argparse
import os
import numpy as np
import yaml
import sys

from chemotaxis_analysis_high_res.src.calculations import (
    interpolate_df,
    correct_stage_pos_with_skeleton,
    calculate_distance,
    calculate_time_in_seconds,
    calculate_speed,
    calculate_radial_speed,
    calculate_displacement_vector,
    calculate_curving_angle,
    calculate_bearing_angle,
    process_bearing_angles,
    calc_reorientation_columns,
)

from chemotaxis_analysis_high_res.src.plotting_visualisation import (
    plot_chemotaxis_overview,
    create_angle_animation,
    plot_ethogram,
    plot_ethogram_simple,
    plot_skeleton_spline,
    create_worm_animation,
    plot_angles_binned,
    plot_turns,
    plot_pumps,
    plot_dynamic_binned,
)

from chemotaxis_analysis_high_res.src.data_smothing import (
    replace_outliers_with_nan,
    apply_smoothing,
)


class CoordinateSystem:
    def __init__(self, top_left_pos, factor_px_to_mm):
        # Store the initial parameters
        self.top_left_x, self.top_left_y = top_left_pos
        self.factor_px_to_mm = factor_px_to_mm

    def transform_coordinates(self, df):
        """
        Transform coordinates exactly as in the Jupyter notebook implementation.
        """
        # Step 1: Convert X and Y to mm (note the swap in subtraction)
        df['X_mm'] = (df['X'] - self.top_left_y) * self.factor_px_to_mm
        df['Y_mm'] = (df['Y'] - self.top_left_x) * self.factor_px_to_mm

        # Step 2: Rotate X_mm and Y_mm by 90 degrees counterclockwise
        df['X_rel'] = -df['Y_mm']
        df['Y_rel'] = df['X_mm']

        # Step 3: Make X_mm_rotated positive
        df['X_rel'] = df['X_rel'].abs()

        # Remove intermediate columns
        df = df.drop(['X_mm', 'Y_mm'], axis=1)

        # Add rotated odor coordinates explicitly to DataFrame
        df['odor_x'] = self.top_left_x * self.factor_px_to_mm
        df['odor_y'] = self.top_left_y * self.factor_px_to_mm

        return df

def read_csv_files(beh_annotation_path:str, skeleton_spline_path:str, worm_pos_path:str, spline_X_path:str, spline_Y_path:str, turn_annotation_path:str):
    # Check if the file paths exist
    if not os.path.exists(beh_annotation_path):
        raise FileNotFoundError(f"The file '{beh_annotation_path}' does not exist.")
    if not os.path.exists(skeleton_spline_path):
        raise FileNotFoundError(f"The file '{skeleton_spline_path}' does not exist.")
    if not os.path.exists(worm_pos_path):
        raise FileNotFoundError(f"The file '{worm_pos_path}' does not exist.")
    if not os.path.exists(spline_X_path):
        raise FileNotFoundError(f"The file '{spline_X_path}' does not exist.")
    if not os.path.exists(spline_Y_path):
        raise FileNotFoundError(f"The file '{spline_Y_path}' does not exist.")
    if not os.path.exists(turn_annotation_path):
        raise FileNotFoundError(f"The file '{turn_annotation_path}' does not exist.")

    # Read CSV files into separate dataframes
    beh_annotation_df = pd.read_csv(beh_annotation_path, header=None)
    skeleton_spline_df = pd.read_csv(skeleton_spline_path, header=None)
    turn_annotation_df = pd.read_csv(turn_annotation_path)

    worm_pos_df = pd.read_csv(worm_pos_path)
    worm_pos_df = worm_pos_df.drop(columns=['time'], errors='ignore')  # deletes old time column before interplation step
    # Enforce integer type for 'frame' column if it exists
    if ('frame' in worm_pos_df.columns):
        worm_pos_df['frame'] = worm_pos_df['frame'].fillna(0).astype(int)

    spline_X_df = pd.read_csv(spline_X_path, header=None)
    spline_Y_df = pd.read_csv(spline_Y_path, header=None)

    # Convert all columns to numeric, if possible
    beh_annotation_df = beh_annotation_df.apply(pd.to_numeric, errors='coerce')
    turn_annotation_df = turn_annotation_df.apply(pd.to_numeric, errors='coerce')
    skeleton_spline_df = skeleton_spline_df.apply(pd.to_numeric, errors='coerce')
    worm_pos_df = worm_pos_df.apply(pd.to_numeric, errors='coerce')
    spline_X_df = spline_X_df.apply(pd.to_numeric, errors='coerce')
    spline_Y_df = spline_Y_df.apply(pd.to_numeric, errors='coerce')

    #check if worm_pos has same length as frame dependent data -> stage pos is tracked separate and can have different FPS
    #-> interpolate
    print("Stage Position Dataframe length before interpolation:", len(worm_pos_df))

    if(len(worm_pos_df) != len(spline_X_df)):
        worm_pos_df = worm_pos_df.apply(lambda x: interpolate_df(x, len(spline_X_df)), axis=0)
        print("Stage Position Dataframe length after interpolation:", len(worm_pos_df))
        print("Frame length of recorded video:", len(spline_X_df))

    return beh_annotation_df, skeleton_spline_df, worm_pos_df, spline_X_df, spline_Y_df, turn_annotation_df

def extract_coords(pos_string):
    # Remove the 'x=' and 'y=' parts and split by comma
    pos_string = pos_string.replace('x=', '').replace('y=', '')
    # Split the string by comma
    x_str, y_str = pos_string.split(',')
    # Convert to integers and return as tuple
    return int(x_str.strip()), int(y_str.strip())

def export_dataframe_to_csv(df: pd.DataFrame, output_path: str, file_name: str):
    """
    Export a pandas DataFrame to a CSV file.
    """
    if not file_name.endswith('.csv'):
        file_name += '.csv'
    full_path = os.path.join(output_path, file_name)
    df.to_csv(full_path, index=True)

def main(arg_list=None):
    parser = argparse.ArgumentParser(description='Read CSV files and plot data')
    parser.add_argument('--reversal_annotation', help='Full path to the reversal annotation CSV file', required=True)
    parser.add_argument('--skeleton_spline', help='Full path to the skeleton spline CSV file', required=True)
    parser.add_argument('--worm_pos', help='Full path to the worm pos text file', required=True)
    parser.add_argument('--skeleton_spline_X_coords', help='Full path to the skeleton_spline_X_coords CSV file', required=True)
    parser.add_argument('--skeleton_spline_Y_coords', help='Full path to the skeleton_spline_Y_coords CSV file', required=True)
    parser.add_argument('--factor_px_to_mm', help='conversion_facor px to mm',required=True)
    parser.add_argument('--video_resolution_x', help='video_resolution_x', required=True)
    parser.add_argument('--video_resolution_y', help='video_resolution_y', required=True)
    parser.add_argument('--fps', help='fps', required=True)
    parser.add_argument('--turn_annotation', help='Full path to the turn annotation CSV file', required=True)
    parser.add_argument('--top_left_pos', help='Tuple of x and y with top left arena position', required=True)

    args = parser.parse_args(arg_list)

    reversal_annotation_path = args.reversal_annotation
    turn_annotation_path = str(args.turn_annotation)
    skeleton_spline_path = args.skeleton_spline
    worm_pos_path = args.worm_pos
    spline_X_path = args.skeleton_spline_X_coords
    spline_Y_path = args.skeleton_spline_Y_coords
    factor_px_to_mm = float(args.factor_px_to_mm)
    video_resolution_x = int(args.video_resolution_x)
    video_resolution_y = int(args.video_resolution_y)
    fps = float(args.fps)

    # Set arena boundaries
    arena_min_x = 0
    arena_max_x = 38
    arena_min_y = 0
    arena_max_y = 40.5

    # Extracting the directory path and saving it to a new variable
    output_path = os.path.dirname(reversal_annotation_path)

    #-------------loading necessary files
    reversal_annotation, skeleton_spline, df_worm_parameter, spline_X, spline_Y, turn_annotation = read_csv_files(
        reversal_annotation_path, skeleton_spline_path, worm_pos_path, spline_X_path, spline_Y_path, turn_annotation_path
    )

    # Basic conversion to integer
    df_worm_parameter['frame'] = df_worm_parameter['frame'].astype(int)

    # Convert the position strings to tuples
    top_left_tuple = extract_coords(args.top_left_pos)

    # Initialize the system
    coord_system = CoordinateSystem(
        top_left_tuple,
        factor_px_to_mm
    )

    # Transform the coordinates
    df_worm_parameter = coord_system.transform_coordinates(df_worm_parameter)

    print(df_worm_parameter.head())

    # Create a copy of df_worm_parameter
    df_skel_all = df_worm_parameter.copy()  # create copy of df_worm_parameter for wormmovie later

    # Calculate corrected center position of the worm
    skel_pos_centroid = 100
    df_worm_parameter = correct_stage_pos_with_skeleton(
        df_worm_parameter,
        spline_X,
        spline_Y,
        skel_pos_centroid,  # 100 will calculate the centroid
        video_resolution_x,
        video_resolution_y,
        factor_px_to_mm,
        video_origin="crop"
    )

    skel_pos_0 = 0
    df_worm_parameter = correct_stage_pos_with_skeleton(
        df_worm_parameter,
        spline_X,
        spline_Y,
        skel_pos_0,  # 0 reflects nose position
        video_resolution_x,
        video_resolution_y,
        factor_px_to_mm,
        video_origin="crop"
    )

    #forward fill NAN where skeleton is NAN which equals a selftouch turn which equals same position
    df_worm_parameter['X_rel_skel_pos_centroid'] = df_worm_parameter['X_rel_skel_pos_centroid'].ffill()
    df_worm_parameter['Y_rel_skel_pos_centroid'] = df_worm_parameter['Y_rel_skel_pos_centroid'].ffill()

    #add column that shows time passed in seconds
    calculate_time_in_seconds(df_worm_parameter, fps)

    # Add angle calculations
    df_worm_parameter = calculate_displacement_vector(df_worm_parameter)
    df_worm_parameter = calculate_curving_angle(df_worm_parameter, window_size=1)
    df_worm_parameter = calculate_bearing_angle(df_worm_parameter)
    df_worm_parameter = process_bearing_angles(df_worm_parameter, window_size=50)

    # Add behavioral state data
    reversal_annotation = reversal_annotation.rename(columns={1: 'behaviour_state'})
    reversal_annotation = reversal_annotation.drop(0, axis=1)
    turn_annotation = turn_annotation.drop('Unnamed: 0', axis=1)

    df_worm_parameter = pd.merge(df_worm_parameter, reversal_annotation, left_index=True, right_index=True, how='left')
    df_worm_parameter = pd.merge(df_worm_parameter, turn_annotation, left_index=True, right_index=True, how='left')

    # Calculate reversal onset and end
    prior_state_behavior = df_worm_parameter['behaviour_state'].shift(periods=-1, fill_value=0)
    df_worm_parameter['reversal_onset'] = ((prior_state_behavior != -1) & (df_worm_parameter['behaviour_state'] == -1)).astype(int)
    df_worm_parameter['reversal_end'] = ((prior_state_behavior == -1) & (df_worm_parameter['behaviour_state'] != -1)).astype(int)

    # Calculate reversal frequency
    window_size = int(fps * 60)  # reversal frequency per minute
    df_worm_parameter['reversal_frequency'] = df_worm_parameter['reversal_onset'].rolling(window=window_size).sum()

    # Calculate speed and related metrics
    df_worm_parameter = calculate_speed(df_worm_parameter, fps)

    # Data smoothing and cleaning
    replace_outliers_with_nan(df_worm_parameter, 'speed',  2.576)

    df_worm_parameter = replace_outliers_with_nan(df_worm_parameter,['speed', 'reversal_frequency', 'curving_angle'], threshold=2.576)

    df_worm_parameter = apply_smoothing(df_worm_parameter,['speed', 'reversal_frequency', 'curving_angle'])

    df_worm_parameter.drop(['odor_x', 'odor_y'], axis=1, inplace=True)

    # Plotting
    plot_ethogram(reversal_annotation, output_path, file_name='ehtogram.png')

    plot_ethogram_simple(reversal_annotation, output_path, file_name='ehtogram_simple.png')

    plot_skeleton_spline(skeleton_spline, output_path, file_name='kymogram.png')

    plot_dynamic_binned(df_worm_parameter, 'reversal_frequency', output_path, 'reversal_frequency_over_time.png',
                        bin_count=100)

    plot_dynamic_binned(df_worm_parameter, 'speed', output_path, 'speed_over_time.png', bin_count=100)

    plot_turns(df_worm_parameter, output_path, file_name='turns.png')

    # Combine dataframes for final output
    df_combined = pd.concat([df_worm_parameter, skeleton_spline], axis=1)

    chemotaxis_columns = pd.MultiIndex.from_product(
        [['basic_parameter'], df_worm_parameter.columns]
    )

    spline_columns = pd.MultiIndex.from_product(
        [['Spline_K'], skeleton_spline.columns]
    )

    df_combined.columns = chemotaxis_columns.append(spline_columns)

    for skel_pos_abs in range(20):
        df_skel_pos_abs = correct_stage_pos_with_skeleton(
            df_skel_all,
            spline_X,
            spline_Y,
            skel_pos_abs,
            video_resolution_x,
            video_resolution_y,
            factor_px_to_mm,
            video_origin="crop"
        )

    df_skel_pos_abs.drop(['frame', 'X', 'Y', 'time_imputed_seconds', 'X_rel', 'Y_rel'], axis=1, inplace=True)

    skel_pos_columns = pd.MultiIndex.from_product(
        [['skel_pos_abs'], df_skel_pos_abs.columns]
    )

    df_combined = pd.concat([df_combined, df_skel_pos_abs], axis=1)

    all_columns = chemotaxis_columns.append(spline_columns).append(skel_pos_columns)
    df_combined.columns = all_columns

    df_combined.to_csv(os.path.join(output_path, 'basic_params.csv'), index=True)

if __name__ == "__main__":
    print("Shell commands passed:", sys.argv)
    main(sys.argv[1:])