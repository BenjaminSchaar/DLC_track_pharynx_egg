import argparse
import os
import numpy as np
import yaml
import sys
import pandas as pd

from chemotaxis_analysis_high_res.src.coordinate_system import CoordinateSystem

from chemotaxis_analysis_high_res.src.calculations import (
    interpolate_df,
    correct_stage_pos_with_skeleton,
    calculate_distance,
    calculate_time_in_seconds,
    calculate_preceived_conc,
    calculate_centroid_speed,
    calculate_center_speed,
    calculate_radial_speed,
    calculate_displacement_vector,
    calculate_curving_angle,
    calculate_bearing_angle,
    calculate_min_border_distance,
    correct_dlc_coordinates,
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
    create_improved_worm_animation,
    create_combined_visualization
)

from chemotaxis_analysis_high_res.src.data_smothing import (
    replace_outliers_with_nan,
    apply_smoothing,
    smooth_trajectory_savitzky_golay_filter,
)


def load_dlc_coordinates(dlc_file_path, nose_label, tail_label):
    """
    Load DLC h5 file and extract nose and tail coordinates.

    Parameters:
    dlc_file_path (str): Path to the DLC h5 file
    nose_label (str): Label for the nose in the DLC file
    tail_label (str): Label for the tail in the DLC file

    Returns:
    tuple: Two DataFrames containing (nose_coordinates, tail_coordinates)
    """
    try:
        # Load the h5 file
        df = pd.read_hdf(dlc_file_path)

        # Get the scorer (first level of multi-index)
        scorer = df.columns.get_level_values(0)[0]

        # Extract nose coordinates
        nose_coords = pd.DataFrame({
            'x': df[scorer][nose_label]['x'].values,
            'y': df[scorer][nose_label]['y'].values
        })

        # Extract tail coordinates
        tail_coords = pd.DataFrame({
            'x': df[scorer][tail_label]['x'].values,
            'y': df[scorer][tail_label]['y'].values
        })

        return nose_coords, tail_coords

    except Exception as e:
        print(f"Error loading DLC file: {e}")
        return None, None


def read_csv_files(beh_annotation_path, skeleton_spline_path, worm_pos_path, spline_X_path, spline_Y_path,
                   turn_annotation_path):
    """
    Read all required CSV files and perform initial preprocessing.

    Parameters:
    beh_annotation_path (str): Path to behavior annotation file
    skeleton_spline_path (str): Path to skeleton spline file
    worm_pos_path (str): Path to worm position file
    spline_X_path (str): Path to spline X coordinates file
    spline_Y_path (str): Path to spline Y coordinates file
    turn_annotation_path (str): Path to turn annotation file

    Returns:
    tuple: Six DataFrames containing the loaded data
    """
    # Check if the file paths exist
    for path, name in [
        (beh_annotation_path, 'behavior annotation'),
        (skeleton_spline_path, 'skeleton spline'),
        (worm_pos_path, 'worm position'),
        (spline_X_path, 'spline X'),
        (spline_Y_path, 'spline Y'),
        (turn_annotation_path, 'turn annotation')
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The {name} file '{path}' does not exist.")

    # Read CSV files into separate dataframes
    beh_annotation_df = pd.read_csv(beh_annotation_path, header=None)
    skeleton_spline_df = pd.read_csv(skeleton_spline_path, header=None)
    turn_annotation_df = pd.read_csv(turn_annotation_path)

    worm_pos_df = pd.read_csv(worm_pos_path)
    worm_pos_df = worm_pos_df.drop(columns=['time'], errors='ignore')  # Delete old time column before interpolation

    # Enforce integer type for 'frame' column if it exists
    if ('frame' in worm_pos_df.columns):
        worm_pos_df['frame'] = worm_pos_df['frame'].fillna(0).astype(int)

    spline_X_df = pd.read_csv(spline_X_path, header=None)
    spline_Y_df = pd.read_csv(spline_Y_path, header=None)

    # Print the head of each dataframe for debugging
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

    # Print the dimensions of each dataframe
    print("Number of rows in beh_annotation_df:", len(beh_annotation_df))
    print("Number of rows in turn_annotation_df:", len(turn_annotation_df))
    print("Number of rows in skeleton_spline_df:", len(skeleton_spline_df))
    print("Number of rows in worm_pos_df:", len(worm_pos_df))
    print("Number of rows in spline_X_df:", len(spline_X_df))
    print("Number of rows in spline_Y_df:", len(spline_Y_df))

    # Interpolate worm_pos if needed
    if (len(worm_pos_df) != len(spline_X_df)):
        print("Stage Position Dataframe length before interpolation:", len(worm_pos_df))
        worm_pos_df = worm_pos_df.apply(lambda x: interpolate_df(x, len(spline_X_df)), axis=0)
        print("Stage Position Dataframe length after interpolation:", len(worm_pos_df))
        print("Stage Position Dataframe head after interpolation:", worm_pos_df.head())
        print("Frame length of recorded video:", len(spline_X_df))

    return beh_annotation_df, skeleton_spline_df, worm_pos_df, spline_X_df, spline_Y_df, turn_annotation_df


def export_dataframe_to_csv(df, output_path, file_name):
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
    print(f"DataFrame exported to {full_path}")


def extract_coords(pos_string):
    """
    Extract x and y coordinates from a string.

    Parameters:
    pos_string (str): String containing coordinates in format 'x=X, y=Y' or 'x = X, y = Y'

    Returns:
    tuple: (x, y) coordinates as floats or None if invalid input
    """
    if pos_string is None:
        return None

    # Handle cases with or without spaces around the equals sign
    pos_string = pos_string.replace('x =', '').replace('x=', '')
    pos_string = pos_string.replace('y =', '').replace('y=', '')

    # Split the string by comma
    x_str, y_str = pos_string.split(',')

    # Convert to float and return as tuple
    return float(x_str.strip()), float(y_str.strip())


def parse_arguments(arg_list=None):
    """
    Parse command line arguments.

    Parameters:
    arg_list (list, optional): List of command line arguments to parse.
                               If None, sys.argv[1:] will be used.

    Returns:
    Namespace: An object containing parsed arguments as attributes
    """
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
    parser.add_argument('--conc_gradient_array', help='exported concentration_gradient.npy file for the odor used',
                        required=False, default=None)
    parser.add_argument('--distance_array', help='exported distance_array.npy file for the odor used',
                        required=False, default=None)
    parser.add_argument('--turn_annotation', help='Full path to the turn annotation CSV file', required=True)
    parser.add_argument('--top_left_pos', help='Tuple of x and y with top left arena position', required=True)
    parser.add_argument('--odor_pos', help='Tuple of x and y with odor position', required=False, default=None)
    parser.add_argument('--diffusion_time_offset',
                        help='offset in seconds for Diffusion simulation (default 1h = 3600 sec)', type=int,
                        default=3600, required=False)
    parser.add_argument('--img_type',
                        help='Specify the type of recording: "vid" or "crop" (default: "crop")',
                        type=str,
                        choices=['zim01', 'zim06', 'crop'],
                        default='crop',
                        required=False)
    parser.add_argument('--arena_x', help='arena X dimension (default: 38mm in wbfm)', required=False, default='38')
    parser.add_argument('--arena_y', help='arena Y dimension (default: 40.5mm in wbfm)', required=False, default='40.5')
    parser.add_argument('--DLC_coords', help='Full path to the DLC CSV file', required=False, default=None)
    parser.add_argument('--DLC_nose', help='Column name for nose coordinates in the DLC file', required=False,
                        default=None)
    parser.add_argument('--DLC_tail', help='Column name for tail coordinates in the DLC file', required=False,
                        default=None)

    return parser.parse_args(arg_list)


def extract_parameters(args):
    """
    Extract parameters from parsed arguments and set up derived parameters.

    Parameters:
    args (Namespace): Parsed command line arguments

    Returns:
    dict: Dictionary containing all parameters needed for analysis
    """
    params = {
        # File paths
        'reversal_annotation_path': args.reversal_annotation,
        'turn_annotation_path': str(args.turn_annotation),
        'skeleton_spline_path': args.skeleton_spline,
        'worm_pos_path': args.worm_pos,
        'spline_X_path': args.skeleton_spline_X_coords,
        'spline_Y_path': args.skeleton_spline_Y_coords,

        # Numerical parameters
        'factor_px_to_mm': float(args.factor_px_to_mm),
        'video_resolution_x': int(args.video_resolution_x),
        'video_resolution_y': int(args.video_resolution_y),
        'fps': float(args.fps),
        'diffusion_time_offset': int(args.diffusion_time_offset),
        'img_type': args.img_type.lower(),

        # Arena parameters
        'arena_min_x': 0,
        'arena_max_x': float(args.arena_x),
        'arena_min_y': 0,
        'arena_max_y': float(args.arena_y),

        # DLC parameters
        'dlc_file_path': args.DLC_coords,
        'dlc_nose_label': args.DLC_nose,
        'dlc_tail_label': args.DLC_tail,

        # Output path (derived from input path)
        'output_path': os.path.dirname(args.reversal_annotation)
    }

    # Check if odor-related parameters are provided
    params['has_odor_data'] = (args.odor_pos is not None and
                               args.conc_gradient_array is not None and
                               args.distance_array is not None)

    # Load concentration gradient arrays if provided
    params['conc_gradient_array'] = None
    params['distance_array'] = None

    if params['has_odor_data']:
        params['conc_gradient_array'] = np.load(args.conc_gradient_array)
        params['distance_array'] = np.load(args.distance_array)

        # Debug information about loaded arrays
        print("conc_gradient_array content:", params['conc_gradient_array'])
        print("conc_gradient_array type:", type(params['conc_gradient_array']))
        print("conc_gradient_array shape:", params['conc_gradient_array'].shape)

        print("distance_array content:", params['distance_array'])
        print("distance_array type:", type(params['distance_array']))
        print("distance_array shape:", params['distance_array'].shape)
    else:
        print("No odor data provided. Skipping odor-related calculations.")

    # Parse coordinate strings from arguments
    params['top_left_tuple'] = extract_coords(args.top_left_pos)
    params['odor_pos_tuple'] = extract_coords(args.odor_pos)  # Will be None if odor_pos is None

    # Check if DLC data is available
    params['has_dlc'] = (args.DLC_coords is not None and
                         args.DLC_nose is not None and
                         args.DLC_tail is not None)

    return params


def load_data_files(params):
    """
    Load all required data files.

    Parameters:
    params (dict): Dictionary of parameters

    Returns:
    dict: Dictionary containing loaded dataframes
    """
    # Load CSV files
    beh_annotation, skeleton_spline, df_worm_parameter, spline_X, spline_Y, turn_annotation = read_csv_files(
        params['reversal_annotation_path'],
        params['skeleton_spline_path'],
        params['worm_pos_path'],
        params['spline_X_path'],
        params['spline_Y_path'],
        params['turn_annotation_path']
    )

    # Add frame column for all types except 'crop'
    if params['img_type'] != 'crop':
        df_worm_parameter['frame'] = range(len(df_worm_parameter))

    # Convert frame column to integer
    if 'frame' in df_worm_parameter.columns:
        df_worm_parameter['frame'] = df_worm_parameter['frame'].astype(int)

    # Return all loaded data
    return {
        'beh_annotation': beh_annotation,
        'skeleton_spline': skeleton_spline,
        'df_worm_parameter': df_worm_parameter,
        'spline_X': spline_X,
        'spline_Y': spline_Y,
        'turn_annotation': turn_annotation
    }


def setup_coordinate_system(df_worm_parameter, params):
    """
    Initialize coordinate system and transform coordinates.

    Parameters:
    df_worm_parameter (pd.DataFrame): DataFrame with worm parameters
    params (dict): Dictionary of parameters

    Returns:
    pd.DataFrame: DataFrame with transformed coordinates
    """
    # Initialize coordinate system with recording type
    if params['img_type'] == 'crop':
        coord_system = CoordinateSystem(
            params['top_left_tuple'],
            params['factor_px_to_mm'],
            'crop',
            params['odor_pos_tuple']
        )
    elif params['img_type'] == 'zim01':
        coord_system = CoordinateSystem(
            params['top_left_tuple'],
            params['factor_px_to_mm'],
            'zim01',
            params['odor_pos_tuple']
        )
    elif params['img_type'] == 'zim06':
        coord_system = CoordinateSystem(
            params['top_left_tuple'],
            params['factor_px_to_mm'],
            'zim06',
            params['odor_pos_tuple']
        )
    else:
        raise ValueError(f"Unsupported image type: {params['img_type']}. Must be one of 'crop', 'zim01', or 'zim06'.")

    # Transform coordinates
    df_worm_parameter = coord_system.transform_coordinates(df_worm_parameter)

    # Capture odor position for later use
    params['x_odor'], params['y_odor'] = None, None
    if params['has_odor_data'] and 'odor_x' in df_worm_parameter.columns and 'odor_y' in df_worm_parameter.columns:
        params['x_odor'] = df_worm_parameter['odor_x'].iloc[0]  # Get from first row
        params['y_odor'] = df_worm_parameter['odor_y'].iloc[0]  # Get from first row
        print(f"Odor position (mm): x={params['x_odor']}, y={params['y_odor']}")

    print(df_worm_parameter.head())

    return df_worm_parameter


def calculate_angles(df_worm_parameter, params):
    """
    Calculate displacement vector, curving angle, and bearing angle.

    Parameters:
    df_worm_parameter (pd.DataFrame): DataFrame with worm parameters
    params (dict): Dictionary of parameters

    Returns:
    pd.DataFrame: DataFrame with calculated angles
    """
    # Calculate displacement vector (adds dx_dt, dy_dt, displacement_vector_degrees, displacement_magnitude)
    df_worm_parameter = calculate_displacement_vector(df_worm_parameter)

    # Calculate curving angle (adds curving_angle column)
    df_worm_parameter = calculate_curving_angle(df_worm_parameter, window_size=1)

    # Calculate bearing angle (only if odor position is available)
    if params['has_odor_data'] and params['x_odor'] is not None and params['y_odor'] is not None:
        df_worm_parameter = calculate_bearing_angle(df_worm_parameter)

    print("Angles calculated.")
    print(df_worm_parameter.head())

    return df_worm_parameter


def integrate_behavior_data(df_worm_parameter, beh_annotation, turn_annotation, params):
    """
    Integrate behavioral annotation data with worm parameter data.

    Parameters:
    df_worm_parameter (pd.DataFrame): DataFrame with worm parameters
    beh_annotation (pd.DataFrame): DataFrame with behavioral annotations
    turn_annotation (pd.DataFrame): DataFrame with turn annotations
    params (dict): Dictionary of parameters

    Returns:
    pd.DataFrame: DataFrame with integrated behavioral data
    """
    # Prepare reversal annotation data
    print(beh_annotation.head())
    beh_annotation = beh_annotation.rename(columns={1: 'behaviour_state'})
    beh_annotation = beh_annotation.drop(0, axis=1)

    # Prepare turn annotation data
    if 'Unnamed: 0' in turn_annotation.columns:
        turn_annotation = turn_annotation.drop('Unnamed: 0', axis=1)

    # Merge behavioral annotations with worm parameter data
    df_worm_parameter = pd.merge(df_worm_parameter, beh_annotation, left_index=True, right_index=True, how='left')
    df_worm_parameter = pd.merge(df_worm_parameter, turn_annotation, left_index=True, right_index=True, how='left')
    print(df_worm_parameter.head())

    # Calculate reversal onset and end
    prior_state_behavior = df_worm_parameter['behaviour_state'].shift(periods=-1, fill_value=0)
    df_worm_parameter['reversal_onset'] = (
                (prior_state_behavior != -1) & (df_worm_parameter['behaviour_state'] == -1)).astype(int)
    df_worm_parameter['reversal_end'] = (
                (prior_state_behavior == -1) & (df_worm_parameter['behaviour_state'] != -1)).astype(int)

    # Calculate reversal frequency per minute
    window_size = int(params['fps'] * 60)  # reversal frequency per minute
    df_worm_parameter['reversal_frequency'] = df_worm_parameter['reversal_onset'].rolling(window=window_size).sum()

    return df_worm_parameter


def calculate_speeds(df_worm_parameter, params):
    """
    Calculate different speed metrics.

    Parameters:
    df_worm_parameter (pd.DataFrame): DataFrame with worm parameters
    params (dict): Dictionary of parameters

    Returns:
    pd.DataFrame: DataFrame with calculated speeds
    """
    # Calculate centroid speed
    df_worm_parameter = calculate_centroid_speed(df_worm_parameter, params['fps'])

    # Calculate center speed
    df_worm_parameter = calculate_center_speed(df_worm_parameter, params['fps'], params['center_point'])

    # Calculate radial speed (only if odor position is available)
    if params['has_odor_data'] and params['x_odor'] is not None and params['y_odor'] is not None:
        df_worm_parameter = calculate_radial_speed(df_worm_parameter, params['fps'])

        # Calculate navigational index (NI)
        df_worm_parameter['NI'] = df_worm_parameter['radial_speed'] / df_worm_parameter['speed_centroid']

    return df_worm_parameter


def clean_and_smooth_data(df_worm_parameter, params):
    """
    Apply outlier replacement and data smoothing.

    Parameters:
    df_worm_parameter (pd.DataFrame): DataFrame with worm parameters
    params (dict): Dictionary of parameters

    Returns:
    pd.DataFrame: DataFrame with cleaned and smoothed data
    """
    # Define basic columns that should be smoothed
    columns_to_smooth = [
        'speed_centroid',
        f'speed_center_{params["center_point"]}',
        'reversal_frequency',
        'curving_angle',
        'distance_to_border_centroid',
        'distance_to_border_nose'
    ]

    # Add odor-related columns if they exist
    if params['has_odor_data'] and params['x_odor'] is not None and params['y_odor'] is not None:
        odor_columns = [
            'radial_speed',
            'bearing_angle',
            'NI',
            'distance_to_odor_centroid',
            'conc_at_centroid',
            f'conc_at_{params["skel_pos_0"]}',
            'dC_centroid',
            f'dC_{params["skel_pos_0"]}',
            f'distance_to_odor_{params["skel_pos_0"]}'
        ]
        # Add DLC odor columns if they exist
        dlc_odor_columns = [
            'distance_to_odor_DLC_nose', 'distance_to_odor_DLC_tail',
            'conc_at_DLC_nose', 'conc_at_DLC_tail',
            'dC_DLC_nose', 'dC_DLC_tail', 'd_DLC_nose_tail_C'
        ]
        odor_columns.extend(dlc_odor_columns)
        
        # Filter to include only columns that exist in the DataFrame
        odor_columns = [col for col in odor_columns if col in df_worm_parameter.columns]
        columns_to_smooth.extend(odor_columns)

    # Apply outlier replacement and smoothing
    df_worm_parameter = replace_outliers_with_nan(df_worm_parameter, columns_to_smooth, threshold=2.576)
    df_worm_parameter = apply_smoothing(df_worm_parameter, columns_to_smooth, params['fps'])

    return df_worm_parameter


def create_visualizations(df_worm_parameter, df_skel_all, data_files, params):
    """
    Generate visualizations based on processed data.

    Parameters:
    df_worm_parameter (pd.DataFrame): DataFrame with worm parameters
    df_skel_all (pd.DataFrame): DataFrame with all skeleton positions
    data_files (dict): Dictionary containing loaded dataframes
    params (dict): Dictionary of parameters

    Returns:
    None
    """
    # Define base time series columns for visualization
    time_series_columns = [
        'speed_centroid_smoothed',
        f'speed_center_{params["center_point"]}_smoothed',
        'reversal_frequency_smoothed',
        'curving_angle_smoothed',
        'distance_to_border_centroid',
        'distance_to_border_nose',
    ]

    # Add odor-related columns to time series if they exist
    if params['has_odor_data'] and params['x_odor'] is not None and params['y_odor'] is not None:
        odor_time_series = [
            'radial_speed_smoothed',
            'bearing_angle_smoothed',
            'NI_smoothed',
            'distance_to_odor_centroid_smoothed',
            'conc_at_centroid_smoothed',
            f'conc_at_{params["skel_pos_0"]}_smoothed',
            'dC_centroid_smoothed',
            f'dC_{params["skel_pos_0"]}_smoothed'
        ]
        # Filter to include only columns that exist in the DataFrame
        odor_time_series = [col for col in odor_time_series if col in df_worm_parameter.columns]
        time_series_columns.extend(odor_time_series)

    # Create combined visualization (PDF with multiple plots)
    create_combined_visualization(
        df=df_worm_parameter,
        beh_annotation=data_files['beh_annotation'],
        skeleton_spline=data_files['skeleton_spline'],
        output_path=params['output_path'],
        x_odor=params['x_odor'],
        y_odor=params['y_odor'],
        arena_min_x=params['arena_min_x'],
        arena_max_x=params['arena_max_x'],
        arena_min_y=params['arena_min_y'],
        arena_max_y=params['arena_max_y'],
        center_point=params['center_point'],
        fps=params['fps'],
        time_series_columns=time_series_columns,
        filename="chemotaxis_analysis.pdf"
    )

    # Create worm animation
    create_improved_worm_animation(
        df1=df_skel_all,
        df2=df_worm_parameter,
        output_path=params['output_path'],
        x_odor=params['x_odor'],
        y_odor=params['y_odor'],
        fps=params['fps'],
        arena_min_x=params['arena_min_x'],
        arena_max_x=params['arena_max_x'],
        arena_min_y=params['arena_min_y'],
        arena_max_y=params['arena_max_y'],
        nth_frame=int(params['fps'] / 4),
        nth_point=5,
        file_name="worm_movie.avi",
        conc_array=params['conc_gradient_array'],
        distance_array=params['distance_array'],
        diffusion_time_offset=params['diffusion_time_offset']
    )

    print("Animation creation complete.")

    return None


def export_final_data(df_worm_parameter, df_skel_all, data_files, params):
    """
    Export final processed data to CSV.

    Parameters:
    df_worm_parameter (pd.DataFrame): DataFrame with worm parameters
    df_skel_all (pd.DataFrame): DataFrame with all skeleton positions
    data_files (dict): Dictionary containing loaded dataframes
    params (dict): Dictionary of parameters

    Returns:
    None
    """
    # Clean up intermediate columns in df_skel_all
    columns_to_drop = ['frame', 'X', 'Y', 'time_imputed_seconds']
    # Add odor-related columns to drop list if they exist
    if params['has_odor_data'] and 'X_rel' in df_skel_all.columns:
        columns_to_drop.extend(['X_rel', 'Y_rel', 'odor_x', 'odor_y'])

    # Filter to only include columns that actually exist in the DataFrame
    existing_columns = [col for col in columns_to_drop if col in df_skel_all.columns]
    # Drop only the existing columns
    if existing_columns:
        df_skel_all.drop(existing_columns, axis=1, inplace=True)

    # Calculate and add absolute skeleton positions for each spline point
    print(f"Number of skeleton columns before processing: {data_files['spline_X'].shape[1]}")
    for skel_pos_abs in range(data_files['spline_X'].shape[1]):
        # Print progress for long-running operations
        if skel_pos_abs % 10 == 0:
            print(f"Processing skeleton position {skel_pos_abs}/{data_files['spline_X'].shape[1]}")

        # Keep using df_skel_all as it's already initialized above
        df_skel_all = correct_stage_pos_with_skeleton(
            df_skel_all,
            data_files['spline_X'],
            data_files['spline_Y'],
            skel_pos_abs,
            params['video_resolution_x'],
            params['video_resolution_y'],
            params['factor_px_to_mm'],
            params['img_type']
        )

    print("Skeleton positions calculated.")
    print(f"Number of columns in df_skel_all: {len(df_skel_all.columns)}")
    print(f"Available columns: {df_skel_all.columns.tolist()}")

    # Create combined dataframe with hierarchical columns
    df_combined = pd.concat([df_worm_parameter, data_files['skeleton_spline']], axis=1)

    # Create MultiIndex columns for initial data
    chemotaxis_columns = pd.MultiIndex.from_product(
        [['chemotaxis_parameter'], df_worm_parameter.columns]
    )

    spline_columns = pd.MultiIndex.from_product(
        [['Spline_K'], data_files['skeleton_spline'].columns]
    )

    # Create MultiIndex columns for skeleton positions
    skel_pos_columns = pd.MultiIndex.from_product(
        [['skel_pos_abs'], df_skel_all.columns]
    )

    # Add skeleton position data to combined DataFrame
    df_combined = pd.concat([df_combined, df_skel_all], axis=1)

    # Combine all column hierarchies
    all_columns = chemotaxis_columns.append(spline_columns).append(skel_pos_columns)
    df_combined.columns = all_columns

    # Save final DataFrame to CSV
    df_combined.to_csv(os.path.join(params['output_path'], 'chemotaxis_params.csv'), index=True)
    print(f"Final data exported to {os.path.join(params['output_path'], 'chemotaxis_params.csv')}")

    return None


def main(arg_list=None):
    """
    Main function that orchestrates the entire chemotaxis analysis process.

    Parameters:
    arg_list (list, optional): List of command-line arguments. If None, sys.argv[1:] is used.

    Returns:
    pd.DataFrame: Final processed DataFrame with chemotaxis parameters
    """
    try:
        # Parse command line arguments
        args = parse_arguments(arg_list)

        # Extract parameters from arguments
        params = extract_parameters(args)

        # Load all necessary data files
        data_files = load_data_files(params)

        # Initialize coordinate system and transform coordinates
        df_worm_parameter = setup_coordinate_system(data_files['df_worm_parameter'], params)

        # Process skeleton data and calculate corrected positions
        df_worm_parameter, df_skel_all = process_skeleton_data(
            df_worm_parameter,
            data_files['spline_X'],
            data_files['spline_Y'],
            params
        )

        # Process and smooth trajectory data
        df_worm_parameter = process_trajectory(df_worm_parameter)

        # Calculate distances (to odor, border, etc.)
        df_worm_parameter = calculate_distances(df_worm_parameter, params)

        # Add time and concentration data
        df_worm_parameter = add_time_and_concentration(df_worm_parameter, params)

        # Process DLC coordinates if available
        if params.get('has_dlc', False):
            df_worm_parameter = process_dlc_data(df_worm_parameter, params)

        # Calculate angles (displacement, curving, bearing)
        df_worm_parameter = calculate_angles(df_worm_parameter, params)

        # Integrate behavioral data
        df_worm_parameter = integrate_behavior_data(
            df_worm_parameter,
            data_files['beh_annotation'],
            data_files['turn_annotation'],
            params
        )

        # Calculate speed metrics
        df_worm_parameter = calculate_speeds(df_worm_parameter, params)

        # Apply smoothing and clean data
        df_worm_parameter = clean_and_smooth_data(df_worm_parameter, params)

        # Generate visualizations
        create_visualizations(
            df_worm_parameter,
            df_skel_all,
            data_files,
            params
        )

        # Export final data
        export_final_data(
            df_worm_parameter,
            df_skel_all,
            data_files,
            params
        )

        return df_worm_parameter

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        print("Shell commands passed:", sys.argv)
        main(sys.argv[1:])  # exclude the script name from the args when called from shell
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def process_skeleton_data(df_worm_parameter, spline_X, spline_Y, params):
    """
    Process skeleton data and calculate corrected positions.

    Parameters:
    df_worm_parameter (pd.DataFrame): DataFrame with worm parameters
    spline_X (pd.DataFrame): DataFrame with X spline coordinates
    spline_Y (pd.DataFrame): DataFrame with Y spline coordinates
    params (dict): Dictionary of parameters

    Returns:
    tuple: (df_worm_parameter, df_skel_all) with processed skeleton data
    """
    # Create a copy of worm parameters for later use
    df_skel_all = df_worm_parameter.copy()  # create copy of df_worm_parameter for worm movie later

    # Calculate corrected centroid position
    df_worm_parameter = correct_stage_pos_with_skeleton(
        df_worm_parameter,
        spline_X,
        spline_Y,
        999,  # 999 will calculate the centroid -> column name will be 'X/Y_rel_skel_pos_centroid'
        params['video_resolution_x'],
        params['video_resolution_y'],
        params['factor_px_to_mm'],
        params['img_type']
    )

    # Calculate nose position (skeleton position 0)
    skel_pos_0 = 0
    params['skel_pos_0'] = skel_pos_0
    df_worm_parameter = correct_stage_pos_with_skeleton(
        df_worm_parameter,
        spline_X,
        spline_Y,
        skel_pos_0,  # 0 reflects nose position
        params['video_resolution_x'],
        params['video_resolution_y'],
        params['factor_px_to_mm'],
        params['img_type']
    )

    # Calculate center point for body speed
    center_point = int((len(spline_X.columns) / 2))
    params['center_point'] = center_point
    print(f"Centerpoint of the skeleton used for speed calculation: {center_point}")

    df_worm_parameter = correct_stage_pos_with_skeleton(
        df_worm_parameter,
        spline_X,
        spline_Y,
        center_point,  # uses center spline point and calculates abs positions
        params['video_resolution_x'],
        params['video_resolution_y'],
        params['factor_px_to_mm'],
        params['img_type']
    )

    print("Added relative worm position:", df_worm_parameter.head())

    return df_worm_parameter, df_skel_all


def process_trajectory(df_worm_parameter):
    """
    Process and smooth trajectory data.

    Parameters:
    df_worm_parameter (pd.DataFrame): DataFrame with worm parameters

    Returns:
    pd.DataFrame: DataFrame with smoothed trajectory data
    """
    # Interpolate missing values in centroid position
    df_worm_parameter['X_rel_skel_pos_centroid'] = df_worm_parameter['X_rel_skel_pos_centroid'].interpolate(
        method='linear')
    df_worm_parameter['Y_rel_skel_pos_centroid'] = df_worm_parameter['Y_rel_skel_pos_centroid'].interpolate(
        method='linear')

    # Keep raw data for reference
    df_worm_parameter['X_rel_skel_pos_centroid_raw'] = df_worm_parameter['X_rel_skel_pos_centroid']
    df_worm_parameter['Y_rel_skel_pos_centroid_raw'] = df_worm_parameter['Y_rel_skel_pos_centroid']

    # Apply smoothing filter to trajectory (use fps as window length)
    fps = df_worm_parameter.get('fps', 30)  # Default to 30 fps if not provided
    df_worm_parameter['X_rel_skel_pos_centroid'] = smooth_trajectory_savitzky_golay_filter(
        df_worm_parameter['X_rel_skel_pos_centroid'],
        window_length=fps
    )
    df_worm_parameter['Y_rel_skel_pos_centroid'] = smooth_trajectory_savitzky_golay_filter(
        df_worm_parameter['Y_rel_skel_pos_centroid'],
        window_length=fps
    )

    return df_worm_parameter


def calculate_distances(df_worm_parameter, params):
    """
    Calculate distances to odor source and arena borders.

    Parameters:
    df_worm_parameter (pd.DataFrame): DataFrame with worm parameters
    params (dict): Dictionary of parameters

    Returns:
    pd.DataFrame: DataFrame with added distance calculations
    """
    # Calculate distances from different points to odor source (only if odor position is available)
    if params['has_odor_data'] and params['x_odor'] is not None and params['y_odor'] is not None:
        df_worm_parameter['distance_to_odor_stage'] = df_worm_parameter.apply(
            lambda row: calculate_distance(row, 'X_rel', 'Y_rel', params['x_odor'], params['y_odor']), axis=1)

        df_worm_parameter[f'distance_to_odor_centroid'] = df_worm_parameter.apply(
            lambda row: calculate_distance(row, 'X_rel_skel_pos_centroid', 'Y_rel_skel_pos_centroid',
                                           params['x_odor'], params['y_odor']), axis=1)

        df_worm_parameter[f'distance_to_odor_{params["skel_pos_0"]}'] = df_worm_parameter.apply(
            lambda row: calculate_distance(row, f'X_rel_skel_pos_{params["skel_pos_0"]}',
                                           f'Y_rel_skel_pos_{params["skel_pos_0"]}',
                                           params['x_odor'], params['y_odor']), axis=1)

        # Convert distance values to float
        df_worm_parameter['distance_to_odor_stage'] = df_worm_parameter['distance_to_odor_stage'].astype(float)
        df_worm_parameter['distance_to_odor_centroid'] = df_worm_parameter['distance_to_odor_centroid'].astype(float)
        df_worm_parameter[f'distance_to_odor_{params["skel_pos_0"]}'] = df_worm_parameter[
            f'distance_to_odor_{params["skel_pos_0"]}'].astype(float)

    # Calculate distances from different points to border (always performed)
    df_worm_parameter['distance_to_border_centroid'] = calculate_min_border_distance(
        df_worm_parameter,
        params['arena_max_x'],
        params['arena_max_y'],
        'X_rel_skel_pos_centroid',
        'Y_rel_skel_pos_centroid'
    )

    df_worm_parameter['distance_to_border_nose'] = calculate_min_border_distance(
        df_worm_parameter,
        params['arena_max_x'],
        params['arena_max_y'],
        f'X_rel_skel_pos_{params["skel_pos_0"]}',
        f'Y_rel_skel_pos_{params["skel_pos_0"]}',
    )

    return df_worm_parameter


def add_time_and_concentration(df_worm_parameter, params):
    """
    Add time and concentration calculations to the data.

    Parameters:
    df_worm_parameter (pd.DataFrame): DataFrame with worm parameters
    params (dict): Dictionary of parameters

    Returns:
    pd.DataFrame: DataFrame with added time and concentration data
    """
    # Add time column in seconds (always performed)
    calculate_time_in_seconds(df_worm_parameter, params['fps'])
    print("Added column for time:", df_worm_parameter.head())

    # Calculate concentration only if odor data is available
    if (params['has_odor_data'] and params['x_odor'] is not None and params['y_odor'] is not None
            and params['conc_gradient_array'] is not None and params['distance_array'] is not None):
        # Calculate concentration at centroid
        df_worm_parameter[f'conc_at_centroid'] = pd.to_numeric(df_worm_parameter.apply(
            lambda row: calculate_preceived_conc(
                row[f'distance_to_odor_centroid'], row['time_seconds'],
                params['conc_gradient_array'], params['distance_array'],
                params['diffusion_time_offset']), axis=1), errors='coerce'
        )

        # Calculate concentration at nose
        df_worm_parameter[f'conc_at_{params["skel_pos_0"]}'] = pd.to_numeric(df_worm_parameter.apply(
            lambda row: calculate_preceived_conc(
                row[f'distance_to_odor_{params["skel_pos_0"]}'], row['time_seconds'],
                params['conc_gradient_array'], params['distance_array'],
                params['diffusion_time_offset']), axis=1), errors='coerce'
        )

        # Calculate concentration derivatives (change over time)
        time_interval_dC_dT = int(params['fps'])  # Determine how far back in the past to compare
        df_worm_parameter[f'dC_centroid'] = df_worm_parameter[f'conc_at_centroid'].diff(
            periods=time_interval_dC_dT).astype(
            float)
        df_worm_parameter[f'dC_{params["skel_pos_0"]}'] = df_worm_parameter[f'conc_at_{params["skel_pos_0"]}'].diff(
            periods=1).astype(float)

    print("\nWorm DataFrame with Distance and Time:", df_worm_parameter.head())

    return df_worm_parameter


def process_dlc_data(df_worm_parameter, params):
    """
    Process DLC tracking data if available.

    Parameters:
    df_worm_parameter (pd.DataFrame): DataFrame with worm parameters
    params (dict): Dictionary of parameters

    Returns:
    pd.DataFrame: DataFrame with added DLC data
    """
    print(f"Processing DLC file: {params['dlc_file_path']}")

    # Load nose and tail coordinates from DLC file
    nose_coords, tail_coords = load_dlc_coordinates(
        params['dlc_file_path'],
        params['dlc_nose_label'],
        params['dlc_tail_label']
    )

    if nose_coords is not None and tail_coords is not None:
        # Convert DLC coordinates to mm and correct based on video origin
        nose_mm, tail_mm = correct_dlc_coordinates(
            nose_coords,
            tail_coords,
            params['video_resolution_x'],
            params['video_resolution_y'],
            params['factor_px_to_mm'],
            params['img_type']
        )

        # Check if lengths match (they should)
        if len(df_worm_parameter) != len(nose_mm):
            print(
                f"Warning: DLC data length ({len(nose_mm)}) doesn't match main data length ({len(df_worm_parameter)})")
            print("This is unexpected since both should come from the same video frames.")

        # Add DLC coordinates to the main DataFrame
        # Use the min length to avoid index errors in case the lengths differ slightly
        min_length = min(len(df_worm_parameter), len(nose_mm))

        df_worm_parameter['X_rel_DLC_nose'] = nose_mm['X_rel_DLC_nose'].values[:min_length]
        df_worm_parameter['Y_rel_DLC_nose'] = nose_mm['Y_rel_DLC_nose'].values[:min_length]
        df_worm_parameter['X_rel_DLC_tail'] = tail_mm['X_rel_DLC_tail'].values[:min_length]
        df_worm_parameter['Y_rel_DLC_tail'] = tail_mm['Y_rel_DLC_tail'].values[:min_length]

        print(f"Added DLC tracking data to main DataFrame")

        # Add distance calculations for DLC coordinates
        if params['has_odor_data'] and params['x_odor'] is not None and params['y_odor'] is not None:
            # Distance to odor calculations
            df_worm_parameter['distance_to_odor_DLC_nose'] = df_worm_parameter.apply(
                lambda row: calculate_distance(row, 'X_rel_DLC_nose', 'Y_rel_DLC_nose',
                                               params['x_odor'], params['y_odor']), axis=1)
            df_worm_parameter['distance_to_odor_DLC_nose'] = df_worm_parameter['distance_to_odor_DLC_nose'].astype(
                float)

            df_worm_parameter['distance_to_odor_DLC_tail'] = df_worm_parameter.apply(
                lambda row: calculate_distance(row, 'X_rel_DLC_tail', 'Y_rel_DLC_tail',
                                               params['x_odor'], params['y_odor']), axis=1)
            df_worm_parameter['distance_to_odor_DLC_tail'] = df_worm_parameter['distance_to_odor_DLC_tail'].astype(
                float)

        # Distance to border calculations (always performed)
        df_worm_parameter['distance_to_border_DLC_nose'] = calculate_min_border_distance(
            df_worm_parameter,
            params['arena_max_x'],
            params['arena_max_y'],
            'X_rel_DLC_nose',
            'Y_rel_DLC_nose'
        )

        df_worm_parameter['distance_to_border_DLC_tail'] = calculate_min_border_distance(
            df_worm_parameter,
            params['arena_max_x'],
            params['arena_max_y'],
            'X_rel_DLC_tail',
            'Y_rel_DLC_tail'
        )

        print(f"Calculated distances for DLC tracking points")

        # Calculate concentration at DLC points (if odor and concentration data available)
        if (params['has_odor_data'] and params['conc_gradient_array'] is not None
                and params['distance_array'] is not None):
            # Calculate concentration at DLC nose
            df_worm_parameter['conc_at_DLC_nose'] = pd.to_numeric(df_worm_parameter.apply(
                lambda row: calculate_preceived_conc(
                    row['distance_to_odor_DLC_nose'], row['time_seconds'],
                    params['conc_gradient_array'], params['distance_array'],
                    params['diffusion_time_offset']), axis=1), errors='coerce'
            )

            # Calculate concentration at DLC tail
            df_worm_parameter['conc_at_DLC_tail'] = pd.to_numeric(df_worm_parameter.apply(
                lambda row: calculate_preceived_conc(
                    row['distance_to_odor_DLC_tail'], row['time_seconds'],
                    params['conc_gradient_array'], params['distance_array'],
                    params['diffusion_time_offset']), axis=1), errors='coerce'
            )

            # Calculate frame-by-frame concentration difference (dC) for DLC nose
            df_worm_parameter['dC_DLC_nose'] = df_worm_parameter['conc_at_DLC_nose'].diff(periods=1).astype(float)

            # Calculate frame-by-frame concentration difference (dC) for DLC tail
            df_worm_parameter['dC_DLC_tail'] = df_worm_parameter['conc_at_DLC_tail'].diff(periods=1).astype(float)

            # Calculate concentration difference between nose and tail
            # Positive when nose is at higher concentration than tail
            df_worm_parameter['d_DLC_nose_tail_C'] = (
                    df_worm_parameter['conc_at_DLC_nose'] - df_worm_parameter['conc_at_DLC_tail']
            ).astype(float)

            print(f"Calculated concentration values and gradients for DLC tracking points")

    return df_worm_parameter