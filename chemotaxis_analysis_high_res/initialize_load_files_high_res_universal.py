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
    calc_signed_speed,
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

    # Print the head of each dataframe
    print("_Behavior Annotation DataFrame:")
    print(beh_annotation_df.head())

    # Print the head of each dataframe
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

    print("Number of rows in beh_annotation_df:", len(beh_annotation_df))
    print("Number of rows in turn_annotation_df:", len(turn_annotation_df))
    print("Number of rows in skeleton_spline_df:", len(skeleton_spline_df))
    print("Number of rows in worm_pos_df:", len(worm_pos_df))
    print("Number of rows in spline_X_df:", len(spline_X_df))
    print("Number of rows in spline_Y_df:", len(spline_Y_df))

    #check if worm_pos has same lenght as frame dependend data -> stage pos is tracked seperate and can have different FPS
    #-> interpolate
    print("Stage Position Dataframe lenght before interpolation:", len(worm_pos_df))

    if(len(worm_pos_df) != len(spline_X_df)):

        #worm_pos_df = worm_pos_df.iloc[0+len(worm_pos_df)].apply(lambda x: interpolate_df(x, np.linspace(0, 1, len(spline_X_df)), len(worm_pos_df)), axis=0)
        # Apply the interpolation for each column
        worm_pos_df = worm_pos_df.apply(lambda x: interpolate_df(x, len(spline_X_df)), axis=0)

        print("Stage Position Dataframe lenght after interpolation:", len(worm_pos_df))
        print("Stage Position Dataframe head after interpolation:", worm_pos_df.head())
        print("Frame lenght of recorded video:", len(spline_X_df))

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
    if pos_string is None:
        return None

    # Handle cases with or without spaces around the equals sign
    pos_string = pos_string.replace('x =', '').replace('x=', '')
    pos_string = pos_string.replace('y =', '').replace('y=', '')

    # Split the string by comma
    x_str, y_str = pos_string.split(',')

    # Convert to float and return as tuple
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
    parser.add_argument('--DLC_nose', help='Column name for nose coordinates in the DLC file', required=False, default=None)
    parser.add_argument('--DLC_tail', help='Column name for tail coordinates in the DLC file', required=False, default=None)

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
    diffusion_time_offset = int(args.diffusion_time_offset)
    img_type = args.img_type.lower()

    # Check if odor-related parameters are provided
    has_odor_data = (args.odor_pos is not None and
                     args.conc_gradient_array is not None and
                     args.distance_array is not None)

    # Load concentration gradient arrays if provided
    conc_gradient_array = None
    distance_array = None

    if has_odor_data:
        conc_gradient_array = np.load(args.conc_gradient_array)
        distance_array = np.load(args.distance_array)

        # Debug information about loaded arrays
        print("conc_gradient_array content:", conc_gradient_array)
        print("conc_gradient_array type:", type(conc_gradient_array))
        print("conc_gradient_array shape:", conc_gradient_array.shape)

        print("distance_array content:", distance_array)
        print("distance_array type:", type(distance_array))
        print("distance_array shape:", distance_array.shape)
    else:
        print("No odor data provided. Skipping odor-related calculations.")

    # Set arena boundaries
    arena_min_x = 0
    arena_max_x = float(args.arena_x)
    arena_min_y = 0
    arena_max_y = float(args.arena_y)

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

    if img_type != 'crop':
        # Add frame column for all types except 'crop'
        df_worm_parameter['frame'] = range(len(df_worm_parameter))

    # Convert frame column to integer
    df_worm_parameter['frame'] = df_worm_parameter['frame'].astype(int)

    # --------------------------------------------------
    # 4. COORDINATE SYSTEM TRANSFORMATION
    # --------------------------------------------------
    # Parse coordinate strings from arguments
    top_left_tuple = extract_coords(args.top_left_pos)
    odor_pos_tuple = extract_coords(args.odor_pos)  # Will be None if odor_pos is None

    # Initialize coordinate system with recording type
    if img_type == 'crop':
        coord_system = CoordinateSystem(
            top_left_tuple,
            factor_px_to_mm,
            'crop',
            odor_pos_tuple
        )
    elif img_type == 'zim01':
        coord_system = CoordinateSystem(
            top_left_tuple,
            factor_px_to_mm,
            'zim01',
            odor_pos_tuple
        )
    elif img_type == 'zim06':
        coord_system = CoordinateSystem(
            top_left_tuple,
            factor_px_to_mm,
            'zim06',
            odor_pos_tuple
        )
    else:
        raise ValueError(f"Unsupported image type: {img_type}. Must be one of 'crop', 'zim01', or 'zim06'.")

    df_worm_parameter = coord_system.transform_coordinates(df_worm_parameter)
    # Get odor position from DataFrame columns if available
    x_odor, y_odor = None, None
    if has_odor_data and 'odor_x' in df_worm_parameter.columns and 'odor_y' in df_worm_parameter.columns:
        x_odor = df_worm_parameter['odor_x'].iloc[0]  # Get from first row
        y_odor = df_worm_parameter['odor_y'].iloc[0]  # Get from first row
        print(f"Odor position (mm): x={x_odor}, y={y_odor}")
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
    # Interpolate missing values in centroid position
    df_worm_parameter['X_rel_skel_pos_centroid'] = df_worm_parameter['X_rel_skel_pos_centroid'].interpolate(
        method='linear')
    df_worm_parameter['Y_rel_skel_pos_centroid'] = df_worm_parameter['Y_rel_skel_pos_centroid'].interpolate(
        method='linear')

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
    # 7. DISTANCE CALCULATIONS
    # --------------------------------------------------
    # Calculate distances from different points to odor source (only if odor position is available)
    if has_odor_data and x_odor is not None and y_odor is not None:
        df_worm_parameter['distance_to_odor_stage'] = df_worm_parameter.apply(
            lambda row: calculate_distance(row, 'X_rel', 'Y_rel', x_odor, y_odor), axis=1)
        df_worm_parameter[f'distance_to_odor_centroid'] = df_worm_parameter.apply(
            lambda row: calculate_distance(row, 'X_rel_skel_pos_centroid', 'Y_rel_skel_pos_centroid', x_odor, y_odor),
            axis=1)
        df_worm_parameter[f'distance_to_odor_{skel_pos_0}'] = df_worm_parameter.apply(
            lambda row: calculate_distance(row, f'X_rel_skel_pos_{skel_pos_0}', f'Y_rel_skel_pos_{skel_pos_0}', x_odor,
                                        y_odor), axis=1)

        # Convert distance values to float
        df_worm_parameter['distance_to_odor_stage'] = df_worm_parameter['distance_to_odor_stage'].astype(float)
        df_worm_parameter['distance_to_odor_centroid'] = df_worm_parameter['distance_to_odor_centroid'].astype(float)
        df_worm_parameter[f'distance_to_odor_{skel_pos_0}'] = df_worm_parameter[f'distance_to_odor_{skel_pos_0}'].astype(
            float)

    # Calculate distances from different points to border (always performed)
    df_worm_parameter['distance_to_border_centroid'] = calculate_min_border_distance(
        df_worm_parameter,
        arena_max_x,
        arena_max_y,
        'X_rel_skel_pos_centroid',
        'Y_rel_skel_pos_centroid'
    )

    df_worm_parameter['distance_to_border_nose'] = calculate_min_border_distance(
        df_worm_parameter,
        arena_max_x,
        arena_max_y,
        f'X_rel_skel_pos_{skel_pos_0}',
        f'Y_rel_skel_pos_{skel_pos_0}',
    )

    # --------------------------------------------------
    # 8. TIME & CONCENTRATION CALCULATIONS
    # --------------------------------------------------
    # Add time column in seconds (always performed)
    calculate_time_in_seconds(df_worm_parameter, fps)
    print("added column for time:", df_worm_parameter)

    # Calculate concentration only if odor data is available
    if has_odor_data and x_odor is not None and y_odor is not None and conc_gradient_array is not None and distance_array is not None:
        # Calculate concentration at centroid
        df_worm_parameter[f'conc_at_centroid'] = pd.to_numeric(df_worm_parameter.apply(
            lambda row: calculate_preceived_conc(
                row[f'distance_to_odor_centroid'], row['time_seconds'], conc_gradient_array, distance_array,
                diffusion_time_offset), axis=1), errors='coerce'
        )

        # Calculate concentration at nose
        df_worm_parameter[f'conc_at_{skel_pos_0}'] = pd.to_numeric(df_worm_parameter.apply(
            lambda row: calculate_preceived_conc(
                row[f'distance_to_odor_{skel_pos_0}'], row['time_seconds'], conc_gradient_array, distance_array,
                diffusion_time_offset), axis=1), errors='coerce'
        )

        # Calculate concentration derivatives (change over time)
        time_interval_dC_dT = int(fps)  # Determine how far back in the past to compare
        df_worm_parameter[f'dC_centroid'] = df_worm_parameter[f'conc_at_centroid'].diff(periods=time_interval_dC_dT).astype(
            float)
        df_worm_parameter[f'dC_{skel_pos_0}'] = df_worm_parameter[f'conc_at_{skel_pos_0}'].diff(periods=1).astype(float)

    print("\nWorm DataFrame with Distance:")
    print(df_worm_parameter.head())

    # --------------------------------------------------
    # DLC PROCESSING
    # --------------------------------------------------
    # Check if DLC file is provided
    if args.DLC_coords is not None:
        print(f"Processing DLC file: {args.DLC_coords}")

        # Load nose and tail coordinates from DLC file
        nose_coords, tail_coords = load_dlc_coordinates(
            args.DLC_coords,
            args.DLC_nose,
            args.DLC_tail
        )

        if nose_coords is not None and tail_coords is not None:
            # Convert DLC coordinates to mm and correct based on video origin
            nose_mm, tail_mm = correct_dlc_coordinates(
                df_worm_parameter,
                nose_coords,
                tail_coords,
                args.video_resolution_x,
                args.video_resolution_y,
                args.factor_px_to_mm,
                args.img_type
            )

            # Check if lengths match (they should)
            if len(df_worm_parameter) != len(nose_mm):
                print(
                    f"Warning: DLC data length ({len(nose_mm)}) doesn't match main data length ({len(df_worm_parameter)})")
                print("This is unexpected since both should come from the same video frames.")
                # Consider adding error handling here if needed

            # Add DLC coordinates to the main DataFrame
            # Use the min length to avoid index errors in case the lengths differ slightly
            min_length = min(len(df_worm_parameter), len(nose_mm))

            df_worm_parameter['X_rel_DLC_nose'] = nose_mm['X_rel_DLC_nose'].values[:min_length]
            df_worm_parameter['Y_rel_DLC_nose'] = nose_mm['Y_rel_DLC_nose'].values[:min_length]
            df_worm_parameter['X_rel_DLC_tail'] = tail_mm['X_rel_DLC_tail'].values[:min_length]
            df_worm_parameter['Y_rel_DLC_tail'] = tail_mm['Y_rel_DLC_tail'].values[:min_length]

            print(f"Added DLC tracking data to main DataFrame")

            # Add distance calculations for DLC coordinates
            if has_odor_data and x_odor is not None and y_odor is not None:
                # Distance to odor calculations
                df_worm_parameter['distance_to_odor_DLC_nose'] = df_worm_parameter.apply(
                    lambda row: calculate_distance(row, 'X_rel_DLC_nose', 'Y_rel_DLC_nose', x_odor, y_odor), axis=1)
                df_worm_parameter['distance_to_odor_DLC_nose'] = df_worm_parameter['distance_to_odor_DLC_nose'].astype(
                    float)

                df_worm_parameter['distance_to_odor_DLC_tail'] = df_worm_parameter.apply(
                    lambda row: calculate_distance(row, 'X_rel_DLC_tail', 'Y_rel_DLC_tail', x_odor, y_odor), axis=1)
                df_worm_parameter['distance_to_odor_DLC_tail'] = df_worm_parameter['distance_to_odor_DLC_tail'].astype(
                    float)

            # Distance to border calculations (always performed)
            df_worm_parameter['distance_to_border_DLC_nose'] = calculate_min_border_distance(
                df_worm_parameter,
                arena_max_x,
                arena_max_y,
                'X_rel_DLC_nose',
                'Y_rel_DLC_nose'
            )

            df_worm_parameter['distance_to_border_DLC_tail'] = calculate_min_border_distance(
                df_worm_parameter,
                arena_max_x,
                arena_max_y,
                'X_rel_DLC_tail',
                'Y_rel_DLC_tail'
            )

            print(f"Calculated distances for DLC tracking points")

            # Calculate concentration at DLC points (if odor and concentration data available)
            if has_odor_data and conc_gradient_array is not None and distance_array is not None:
                # Calculate concentration at DLC nose
                df_worm_parameter['conc_at_DLC_nose'] = pd.to_numeric(df_worm_parameter.apply(
                    lambda row: calculate_preceived_conc(
                        row['distance_to_odor_DLC_nose'], row['time_seconds'], conc_gradient_array, distance_array,
                        diffusion_time_offset), axis=1), errors='coerce'
                )

                # Calculate concentration at DLC tail
                df_worm_parameter['conc_at_DLC_tail'] = pd.to_numeric(df_worm_parameter.apply(
                    lambda row: calculate_preceived_conc(
                        row['distance_to_odor_DLC_tail'], row['time_seconds'], conc_gradient_array, distance_array,
                        diffusion_time_offset), axis=1), errors='coerce'
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

    # --------------------------------------------------
    # 9. ANGLE CALCULATIONS
    # --------------------------------------------------
    # Calculate displacement, curving angle, and bearing angle
    df_worm_parameter = calculate_displacement_vector(df_worm_parameter)
    df_worm_parameter = calculate_curving_angle(df_worm_parameter, window_size=1)

    # Only calculate bearing angle if odor position is available
    if has_odor_data and x_odor is not None and y_odor is not None:
        df_worm_parameter = calculate_bearing_angle(df_worm_parameter)

    print("Angles calculated.")
    print(df_worm_parameter.head())

    # --------------------------------------------------
    # 10. BEHAVIORAL STATE INTEGRATION
    # --------------------------------------------------
    # Prepare reversal annotation data
    print(reversal_annotation.head())
    reversal_annotation = reversal_annotation.rename(columns={1: 'behaviour_state'})
    reversal_annotation = reversal_annotation.drop(0, axis=1)

    # Prepare turn annotation data
    if 'Unnamed: 0' in turn_annotation.columns:
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
    # 11. SPEED & NAVIGATION CALCULATIONS
    # --------------------------------------------------
    # Calculate different speed metrics
    df_worm_parameter = calculate_centroid_speed(df_worm_parameter, fps)  # adds column centroid speed to df
    df_worm_parameter = calculate_center_speed(df_worm_parameter, fps, center_point)  # adds column center speed to df

    df_worm_parameter = calc_signed_speed(df_worm_parameter, reversal_annotation, center_point)

    # Calculate radial speed only if odor position is available
    if has_odor_data and x_odor is not None and y_odor is not None:
        df_worm_parameter = calculate_radial_speed(df_worm_parameter, fps)  # adds column radial speed to df
        # Calculate navigational index (NI)
        df_worm_parameter['NI'] = (df_worm_parameter['radial_speed'] / df_worm_parameter['speed_centroid'])

    # --------------------------------------------------
    # 12. DATA CLEANING & SMOOTHING
    # --------------------------------------------------
    # Define basic columns that should be smoothed
    columns_to_smooth = [
        'speed_centroid',
        f'speed_center_{center_point}',
        'reversal_frequency',
        'curving_angle',
        'distance_to_border_centroid',
        'distance_to_border_nose'
    ]

    # Add odor-related columns if they exist
    if has_odor_data and x_odor is not None and y_odor is not None:
        odor_columns = [
            'radial_speed',
            'bearing_angle',
            'NI',
            'distance_to_odor_centroid',
            'conc_at_centroid',
            f'conc_at_{skel_pos_0}',
            'dC_centroid',
            f'dC_{skel_pos_0}',
            f'distance_to_odor_{skel_pos_0}'
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
    df_worm_parameter = apply_smoothing(df_worm_parameter, columns_to_smooth, fps)

    # --------------------------------------------------
    # 13. VISUALIZATION & PLOTTING
    # --------------------------------------------------
    # Define base time series columns
    time_series_columns = [
        'speed_centroid_smoothed',
        f'speed_center_{center_point}_smoothed',
        'reversal_frequency_smoothed',
        'curving_angle_smoothed',
        'distance_to_border_centroid',
        'distance_to_border_nose',
    ]

    # Add odor-related columns to time series if they exist
    if has_odor_data and x_odor is not None and y_odor is not None:
        odor_time_series = [
            'radial_speed_smoothed',
            'bearing_angle_smoothed',
            'NI_smoothed',
            'distance_to_odor_centroid_smoothed',
            'conc_at_centroid_smoothed',
            f'conc_at_{skel_pos_0}_smoothed',
            'dC_centroid_smoothed',
            f'dC_{skel_pos_0}_smoothed'
        ]
        # Filter to include only columns that exist in the DataFrame
        odor_time_series = [col for col in odor_time_series if col in df_worm_parameter.columns]
        time_series_columns.extend(odor_time_series)

    create_combined_visualization(
        df=df_worm_parameter,
        beh_annotation=reversal_annotation,
        skeleton_spline=skeleton_spline,
        output_path=output_path,
        x_odor=x_odor,  # Will be None if not provided
        y_odor=y_odor,  # Will be None if not provided
        arena_min_x=arena_min_x,
        arena_max_x=arena_max_x,
        arena_min_y=arena_min_y,
        arena_max_y=arena_max_y,
        center_point=center_point,
        fps=fps,
        time_series_columns=time_series_columns,
        filename="chemotaxis_analysis.pdf"
    )

    # --------------------------------------------------
    # 14. DATA EXPORT & FINALIZATION
    # --------------------------------------------------
    # Create combined dataframe with hierarchical columns
    df_combined = pd.concat([df_worm_parameter, skeleton_spline], axis=1)

    # Create MultiIndex columns for initial data
    chemotaxis_columns = pd.MultiIndex.from_product(
        [['chemotaxis_parameter'], df_worm_parameter.columns]
    )

    spline_columns = pd.MultiIndex.from_product(
        [['Spline_K'], skeleton_spline.columns]
    )

    # Assign initial MultiIndex columns
    df_combined.columns = chemotaxis_columns.append(spline_columns)

    # Calculate and add absolute skeleton positions for each spline point
    print(f"Number of skeleton columns before processing: {spline_X.shape[1]}")
    for skel_pos_abs in range(spline_X.shape[1]):
        # Print progress for long-running operations
        if skel_pos_abs % 10 == 0:
            print(f"Processing skeleton position {skel_pos_abs}/{spline_X.shape[1]}")

        # Keep using df_skel_all as it's already initialized above
        df_skel_all = correct_stage_pos_with_skeleton(
            df_skel_all,
            spline_X,
            spline_Y,
            skel_pos_abs,
            video_resolution_x,
            video_resolution_y,
            factor_px_to_mm,
            img_type
        )

    print("Skeleton positions calculated.")
    print(f"Number of columns in df_skel_all: {len(df_skel_all.columns)}")
    print(f"Available columns: {df_skel_all.columns.tolist()}")


    # Create worm animation
    create_improved_worm_animation(
        df1=df_skel_all,
        df2=df_worm_parameter,
        output_path=output_path,
        x_odor=x_odor,
        y_odor=y_odor,
        fps=fps,
        arena_min_x=arena_min_x,
        arena_max_x=arena_max_x,
        arena_min_y=arena_min_y,
        arena_max_y=arena_max_y,
        nth_frame=int(fps / 4),
        nth_point=5,
        file_name="worm_movie.avi",
        conc_array=conc_gradient_array,
        distance_array=distance_array,
        diffusion_time_offset=diffusion_time_offset
    )

    # If you want to check if the function executed successfully
    print("Animation creation complete.")


    # Clean up intermediate columns - only drop columns that exist
    columns_to_drop = ['frame', 'X', 'Y', 'time_imputed_seconds']
    # Add odor-related columns to drop list if they exist
    if has_odor_data and 'X_rel' in df_skel_all.columns:
        columns_to_drop.extend(['X_rel', 'Y_rel', 'odor_x', 'odor_y'])

    # Filter to only include columns that actually exist in the DataFrame
    existing_columns = [col for col in columns_to_drop if col in df_skel_all.columns]
    # Drop only the existing columns
    if existing_columns:
        df_skel_all.drop(existing_columns, axis=1, inplace=True)

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
    df_combined.to_csv(os.path.join(output_path, 'chemotaxis_params.csv'), index=True)


if __name__ == "__main__":
    try:
        print("Shell commands passed:", sys.argv)
        main(sys.argv[1:])  # exclude the script name from the args when called from shell
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)