import pandas as pd
import argparse
import os
import numpy as np
import yaml
import sys

from chemotaxis_analysis_high_res.src.coordinate_system import CoordinateSystem

from chemotaxis_analysis_high_res.src.calculations import (
    interpolate_df,
    correct_stage_pos_with_skeleton,
    calculate_distance,
    calculate_time_in_seconds,
    calculate_centroid_speed,
    calculate_center_speed,
    calculate_radial_speed,
    calculate_displacement_vector,
    calculate_curving_angle,
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
    plot_time_series,
)

from chemotaxis_analysis_high_res.src.data_smothing import (
    replace_outliers_with_nan,
    apply_smoothing,
    smooth_trajectory_savitzky_golay_filter,
)


def read_csv_files(beh_annotation_path: str, skeleton_spline_path: str, worm_pos_path: str, spline_X_path: str,
                   spline_Y_path: str, turn_annotation_path: str):
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
    worm_pos_df = worm_pos_df.drop(columns=['time'],
                                   errors='ignore')  # deletes old time column before interplation step
    # Enforce integer type for 'frame' column if it exists
    if ('frame' in worm_pos_df.columns):
        worm_pos_df['frame'] = worm_pos_df['frame'].fillna(0).astype(int)

    spline_X_df = pd.read_csv(spline_X_path, header=None)
    spline_Y_df = pd.read_csv(spline_Y_path, header=None)

    # Print the head of each dataframe
    print("Behavior Annotation DataFrame:")
    print(beh_annotation_df.head())

    print("Turn Annotation DataFrame:")
    print(turn_annotation_df.head())

    print("\nSkeleton Spline DataFrame:")
    print(skeleton_spline_df.head())

    print("\nWorm Position DataFrame:")
    print(worm_pos_df.head())

    print("\nSpline X DataFrame:")
    print(spline_X_df.head())

    print("\nSpline Y DataFrame:")
    print(spline_Y_df.head())

    # Convert all columns to numeric, if possible
    beh_annotation_df = beh_annotation_df.apply(pd.to_numeric, errors='coerce')
    turn_annotation_df = turn_annotation_df.apply(pd.to_numeric, errors='coerce')
    skeleton_spline_df = skeleton_spline_df.apply(pd.to_numeric, errors='coerce')
    worm_pos_df = worm_pos_df.apply(pd.to_numeric, errors='coerce')
    spline_X_df = spline_X_df.apply(pd.to_numeric, errors='coerce')
    spline_Y_df = spline_Y_df.apply(pd.to_numeric, errors='coerce')

    print("Number of rows in beh_annotation_df:", len(beh_annotation_df))
    print("Number of rows in turn_annotation_df:", len(turn_annotation_df))
    print("Number of rows in skeleton_spline_df:", len(skeleton_spline_df))
    print("Number of rows in worm_pos_df:", len(worm_pos_df))
    print("Number of rows in spline_X_df:", len(spline_X_df))
    print("Number of rows in spline_Y_df:", len(spline_Y_df))

    # Check if worm_pos has same length as frame dependent data -> stage pos is tracked separate and can have different FPS
    # -> interpolate
    print("Stage Position Dataframe length before interpolation:", len(worm_pos_df))

    if (len(worm_pos_df) != len(spline_X_df)):
        # Apply the interpolation for each column
        worm_pos_df = worm_pos_df.apply(lambda x: interpolate_df(x, len(spline_X_df)), axis=0)

        print("Stage Position Dataframe length after interpolation:", len(worm_pos_df))
        print("Stage Position Dataframe head after interpolation:", worm_pos_df.head())
        print("Frame length of recorded video:", len(spline_X_df))

    return beh_annotation_df, skeleton_spline_df, worm_pos_df, spline_X_df, spline_Y_df, turn_annotation_df


def export_dataframe_to_csv(df: pd.DataFrame, output_path: str, file_name: str):
    """
    Export a pandas DataFrame to a CSV file.

    Parameters:
    df (pandas.DataFrame): The DataFrame to export.
    output_path (str): The directory path where the file will be saved.
    file_name (str): The name of the file to save the DataFrame to, including the .csv extension.
    """
    # Ensure the file_name ends with '.csv'
    if not file_name.endswith('.csv'):
        file_name += '.csv'

    # Use os.path.join to combine the output path and file name
    full_path = os.path.join(output_path, file_name)

    # Export the DataFrame to a CSV file
    df.to_csv(full_path, index=True)  # Change 'index=False' to 'index=True' if you want to include the index.


def extract_coords(pos_string):
    # Remove the 'x=' and 'y=' parts and split by comma
    pos_string = pos_string.replace('x=', '').replace('y=', '')
    # Split the string by comma
    x_str, y_str = pos_string.split(',')
    # Convert to float instead of int and return as tuple
    return float(x_str.strip()), float(y_str.strip())


def main(arg_list=None):
    # --------------------------------------------------
    # 1. ARGUMENT PARSING
    # --------------------------------------------------
    parser = argparse.ArgumentParser(description='Read CSV files and plot data')
    parser.add_argument('--reversal_annotation', help='Full path to the reversal annotation CSV file', required=True)
    parser.add_argument('--skeleton_spline', help='Full path to the skeleton spline CSV file', required=True)
    parser.add_argument('--worm_pos', help='Full path to the worm pos text file', required=True)
    parser.add_argument('--skeleton_spline_X_coords', help='Full path to the skeleton_spline_X_coords CSV file',
                        required=True)
    parser.add_argument('--skeleton_spline_Y_coords', help='Full path to the skeleton_spline_Y_coords CSV file',
                        required=True)
    parser.add_argument('--factor_px_to_mm', help='conversion_facor px to mm', required=True)
    parser.add_argument('--video_resolution_x', help='video_resolution_x', required=True)
    parser.add_argument('--video_resolution_y', help='video_resolution_y', required=True)
    parser.add_argument('--fps', help='fps', required=True)
    parser.add_argument('--turn_annotation', help='Full path to the turn annotation CSV file', required=True)
    parser.add_argument('--top_left_pos', help='Tuple of x and y with top left arena position', required=True)
    parser.add_argument('--img_type',
                        help='Specify the type of recording: "vid" or "crop" (default: "crop")',
                        type=str,
                        choices=['vid', 'crop'],
                        default='crop',
                        required=False)

    args = parser.parse_args(arg_list)

    # --------------------------------------------------
    # 2. EXTRACTING ARGUMENTS & SETTING PARAMETERS
    # --------------------------------------------------
    # Extract file paths
    reversal_annotation_path = args.reversal_annotation
    turn_annotation_path = str(args.turn_annotation)
    skeleton_spline_path = args.skeleton_spline
    worm_pos_path = args.worm_pos
    spline_X_path = args.skeleton_spline_X_coords
    spline_Y_path = args.skeleton_spline_Y_coords

    # Extract numerical parameters
    factor_px_to_mm = float(args.factor_px_to_mm)
    video_resolution_x = int(args.video_resolution_x)
    video_resolution_y = int(args.video_resolution_y)
    fps = float(args.fps)
    img_type = args.img_type.lower()

    # Set arena boundaries
    arena_min_x = 0
    arena_max_x = 38
    arena_min_y = 0
    arena_max_y = 40.5

    # Extract output path from input file path
    output_path = os.path.dirname(reversal_annotation_path)

    # --------------------------------------------------
    # 3. DATA LOADING & PREPARATION
    # --------------------------------------------------
    # Load all CSV files
    reversal_annotation, skeleton_spline, df_worm_parameter, spline_X, spline_Y, turn_annotation = read_csv_files(
        reversal_annotation_path, skeleton_spline_path, worm_pos_path, spline_X_path, spline_Y_path,
        turn_annotation_path
    )

    if img_type == 'vid':
        # Add frame column if type = video to match cropper
        df_worm_parameter['frame'] = range(len(df_worm_parameter))

    # Convert frame column to integer
    df_worm_parameter['frame'] = df_worm_parameter['frame'].astype(int)

    # --------------------------------------------------
    # 4. COORDINATE SYSTEM TRANSFORMATION
    # --------------------------------------------------
    # Parse coordinate strings from arguments
    top_left_tuple = extract_coords(args.top_left_pos)

    # Initialize coordinate system with recording type
    if img_type == 'crop':
        # For crop mode, we need the pixel-to-mm conversion factor
        coord_system = CoordinateSystem(
            top_left_tuple,
            factor_px_to_mm,  # Using the variable directly
            recording_type='crop'
        )
    else:  # Must be 'vid' since argparse validates the choices
        # For vid mode, we also use the factor
        coord_system = CoordinateSystem(
            top_left_tuple,
            factor_px_to_mm,
            recording_type='vid'
        )

    df_worm_parameter = coord_system.transform_coordinates(df_worm_parameter)

    print(df_worm_parameter.head())

    # --------------------------------------------------
    # 5. SKELETON & POSITION CALCULATIONS
    # --------------------------------------------------
    # Create a copy of worm parameters for later use
    df_skel_all = df_worm_parameter.copy()  # create copy of df_worm_parameter for worm movie later

    # Calculate corrected centroid position
    df_worm_parameter = correct_stage_pos_with_skeleton(
        df_worm_parameter,
        spline_X,
        spline_Y,
        999,  # 999 will calculate the centroid -> column name will be 'X/Y_rel_skel_pos_centroid'
        video_resolution_x,
        video_resolution_y,
        factor_px_to_mm,
        img_type
    )

    # Calculate nose position (skeleton position 0)
    skel_pos_0 = 0
    df_worm_parameter = correct_stage_pos_with_skeleton(
        df_worm_parameter,
        spline_X,
        spline_Y,
        0,  # 0 reflects nose position
        video_resolution_x,
        video_resolution_y,
        factor_px_to_mm,
        img_type
    )

    center_point = int((len(spline_X.columns) / 2))
    print("Centerpoint of the skeleton used for speed calculation:", center_point)

    # Calculate center spline point position for body speed
    df_worm_parameter = correct_stage_pos_with_skeleton(
        df_worm_parameter,
        spline_X,
        spline_Y,
        center_point,  # uses center spline point and calculates abs positions
        video_resolution_x,
        video_resolution_y,
        factor_px_to_mm,
        img_type
    )

    print("added relative worm position:", df_worm_parameter)

    # --------------------------------------------------
    # 6. INTERPOLATION & SMOOTHING OF POSITION DATA
    # --------------------------------------------------
    # Forward fill NaN values where skeleton is NaN (self-touch turn equals same position)
    df_worm_parameter['X_rel_skel_pos_centroid'] = df_worm_parameter['X_rel_skel_pos_centroid'].ffill()
    df_worm_parameter['Y_rel_skel_pos_centroid'] = df_worm_parameter['Y_rel_skel_pos_centroid'].ffill()

    # Keep raw data for reference
    df_worm_parameter['X_rel_skel_pos_centroid_raw'] = df_worm_parameter['X_rel_skel_pos_centroid']
    df_worm_parameter['Y_rel_skel_pos_centroid_raw'] = df_worm_parameter['Y_rel_skel_pos_centroid']

    # Apply smoothing filter to trajectory
    df_worm_parameter['X_rel_skel_pos_centroid'] = smooth_trajectory_savitzky_golay_filter(
        df_worm_parameter['X_rel_skel_pos_centroid'],
        window_length=fps
    )
    df_worm_parameter['Y_rel_skel_pos_centroid'] = smooth_trajectory_savitzky_golay_filter(
        df_worm_parameter['Y_rel_skel_pos_centroid'],
        window_length=fps
    )

    # --------------------------------------------------
    # 7. TIME & BASIC CALCULATIONS
    # --------------------------------------------------
    # Add time column in seconds
    calculate_time_in_seconds(df_worm_parameter, fps)
    print("added column for time:", df_worm_parameter)

    # --------------------------------------------------
    # 8. ANGLE CALCULATIONS
    # --------------------------------------------------
    # Calculate displacement and curving angle (but not bearing angle)
    df_worm_parameter = calculate_displacement_vector(df_worm_parameter)
    df_worm_parameter = calculate_curving_angle(df_worm_parameter, window_size=1)

    print("Angles calculated.")
    print(df_worm_parameter.head())

    # --------------------------------------------------
    # 9. BEHAVIORAL STATE INTEGRATION
    # --------------------------------------------------
    # Prepare reversal annotation data
    print(reversal_annotation.head())
    reversal_annotation = reversal_annotation.rename(columns={1: 'behaviour_state'})
    reversal_annotation = reversal_annotation.drop(0, axis=1)

    # Prepare turn annotation data
    turn_annotation = turn_annotation.drop('Unnamed: 0', axis=1)

    # Merge behavioral annotations with worm parameter data
    df_worm_parameter = pd.merge(df_worm_parameter, reversal_annotation, left_index=True, right_index=True, how='left')
    df_worm_parameter = pd.merge(df_worm_parameter, turn_annotation, left_index=True, right_index=True, how='left')
    print(df_worm_parameter.head())

    # Calculate reversal onset and end
    prior_state_behavior = df_worm_parameter['behaviour_state'].shift(periods=-1, fill_value=0)
    df_worm_parameter['reversal_onset'] = (
            (prior_state_behavior != -1) & (df_worm_parameter['behaviour_state'] == -1)).astype(int)
    df_worm_parameter['reversal_end'] = (
            (prior_state_behavior == -1) & (df_worm_parameter['behaviour_state'] != -1)).astype(int)

    # Calculate reversal frequency per minute
    window_size = int(fps * 60)  # reversal frequency per minute
    df_worm_parameter['reversal_frequency'] = df_worm_parameter['reversal_onset'].rolling(window=window_size).sum()

    # --------------------------------------------------
    # 10. SPEED CALCULATIONS
    # --------------------------------------------------
    # Calculate speed (simplified from the original version)
    df_worm_parameter = calculate_centroid_speed(df_worm_parameter, fps)  # adds column centroid speed to df
    df_worm_parameter = calculate_center_speed(df_worm_parameter, fps, center_point)  # adds column center speed to df

    # --------------------------------------------------
    # 11. DATA CLEANING & SMOOTHING
    # --------------------------------------------------
    # Replace outliers with NaN values
    df_worm_parameter = replace_outliers_with_nan(df_worm_parameter,
                                                  ['speed', 'reversal_frequency', 'curving_angle'],
                                                  threshold=2.576)

    # Apply smoothing to key metrics
    df_worm_parameter = apply_smoothing(df_worm_parameter,
                                        ['speed', 'reversal_frequency', 'curving_angle'],
                                        fps)

    # --------------------------------------------------
    # 12. VISUALIZATION & PLOTTING
    # --------------------------------------------------
    # Generate various plots and visualizations
    plot_ethogram(reversal_annotation, output_path, file_name='ehtogram.png')
    plot_ethogram_simple(reversal_annotation, output_path, file_name='ehtogram_simple.png')
    plot_skeleton_spline(skeleton_spline, output_path, file_name='kymogram.png')

    plot_dynamic_binned(df_worm_parameter, 'reversal_frequency', output_path, 'reversal_frequency_over_time.png',
                        bin_count=100)
    plot_dynamic_binned(df_worm_parameter, 'speed', output_path, 'speed_over_time.png', bin_count=100)

    plot_turns(df_worm_parameter, output_path, file_name='turns.png')

    plot_time_series(df_worm_parameter,
                     ['speed_smoothed', 'reversal_frequency_smoothed', 'curving_angle_smoothed'],
                     fps, output_path, 3, figsize=(15, 10), save_suffix='basic_time_series_smoothed')

    # --------------------------------------------------
    # 13. DATA EXPORT & FINALIZATION
    # --------------------------------------------------
    # Create combined dataframe with hierarchical columns
    df_combined = pd.concat([df_worm_parameter, skeleton_spline], axis=1)

    # Create MultiIndex columns for initial data
    basic_columns = pd.MultiIndex.from_product(
        [['basic_parameter'], df_worm_parameter.columns]
    )

    spline_columns = pd.MultiIndex.from_product(
        [['Spline_K'], skeleton_spline.columns]
    )

    # Assign initial MultiIndex columns
    df_combined.columns = basic_columns.append(spline_columns)

    # Calculate and add absolute skeleton positions for each spline point
    num_spline_points = 20  # Using a fixed number for the simplified version

    for skel_pos_abs in range(num_spline_points):
        df_skel_pos_abs = correct_stage_pos_with_skeleton(
            df_skel_all,
            spline_X,
            spline_Y,
            skel_pos_abs,
            video_resolution_x,
            video_resolution_y,
            factor_px_to_mm,
            img_type
        )

    print('Worm Animation DF:', df_skel_pos_abs.head())

    # Clean up intermediate columns - only drop columns that exist
    columns_to_drop = ['frame', 'X', 'Y', 'time_imputed_seconds', 'X_rel', 'Y_rel', 'odor_x', 'odor_y']
    # Filter to only include columns that actually exist in the DataFrame
    existing_columns = [col for col in columns_to_drop if col in df_skel_pos_abs.columns]
    # Drop only the existing columns
    if existing_columns:
        df_skel_pos_abs.drop(existing_columns, axis=1, inplace=True)

    # Create MultiIndex columns for skeleton positions
    skel_pos_columns = pd.MultiIndex.from_product(
        [['skel_pos_abs'], df_skel_pos_abs.columns]
    )

    # Add skeleton position data to combined DataFrame
    df_combined = pd.concat([df_combined, df_skel_pos_abs], axis=1)

    # Combine all column hierarchies
    all_columns = basic_columns.append(spline_columns).append(skel_pos_columns)
    df_combined.columns = all_columns

    # Save final DataFrame to CSV
    df_combined.to_csv(os.path.join(output_path, 'basic_params.csv'), index=True)


if __name__ == "__main__":
    print("Shell commands passed:", sys.argv)
    main(sys.argv[1:])  # exclude the script name from the args when called from shell