import pandas as pd
import argparse
import os
import numpy as np
import yaml
import sys

from chemotaxis_analysis_high_res.calculations import (
    interpolate_df,
    correct_stage_pos_with_skeleton,
    calculate_distance,
    calculate_time_in_seconds,
    calculate_preceived_conc,
    calculate_speed,
    calculate_radial_speed,
    calculate_displacement_vector,
    calculate_curving_angle,
    calculate_bearing_angle,
)

from chemotaxis_analysis_high_res.plotting_visualisation import (
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

from chemotaxis_analysis_high_res.data_smothing import (
    replace_outliers_with_nan,
    update_behaviour_based_on_speed,
)


class CoordinateSystem:
    def __init__(self, top_left_pos, odor_pos, factor_px_to_mm):
        # Store the initial parameters
        self.top_left_x, self.top_left_y = top_left_pos
        self.odor_x, self.odor_y = odor_pos
        self.factor_px_to_mm = factor_px_to_mm

        # Calculate odor position relative to top-left and convert to mm
        odor_x_rel = self.odor_x - self.top_left_x
        odor_y_rel = self.odor_y - self.top_left_y
        self.odor_x_mm = odor_x_rel * self.factor_px_to_mm
        self.odor_y_mm = odor_y_rel * self.factor_px_to_mm

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
        df['odor_x'] = self.odor_x_mm
        df['odor_y'] = self.odor_y_mm

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

# Define a function to extract the x and y values from the yaml file
def extract_coords(coord_string:str):
    x, y = coord_string.split(',')
    x = float(x.strip().split('=')[1])
    y = float(y.strip().split('=')[1])
    return x, y

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
    # Convert to integers and return as tuple
    return int(x_str.strip()), int(y_str.strip())

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
    parser.add_argument('--conc_gradient_array', help='exportet concentration_gradient.npy file for the odor used', required=True)
    parser.add_argument('--distance_array', help='exportet distance_array.npy file for the odor used', required=True)
    parser.add_argument('--turn_annotation', help='Full path to the turn annotation CSV file', required=True)
    parser.add_argument('--top_left_pos', help='Tuple of x and y with top left arena position', required=True)
    parser.add_argument('--odor_pos', help='Tuple of x and y with odor position', required=True)
    parser.add_argument('--diffusion_time_offset', help='offset in seconds for Diffusion simulation (default 1h = 3600 sec)', type=int, default=3600, required=False)


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
    diffusion_time_offset = int(args.diffusion_time_offset)

    # Load arrays from .npy files
    conc_gradient_array = np.load(args.conc_gradient_array)
    distance_array = np.load(args.distance_array)

    print("conc_gradient_array content:", conc_gradient_array)
    print("conc_gradient_array type:", type(conc_gradient_array))
    print("conc_gradient_array shape:", conc_gradient_array.shape)

    print("distance_array content:", distance_array)
    print("distance_array type:", type(distance_array))
    print("distance_array shape:", distance_array.shape)

    # Set arena boundaries
    arena_min_x = 0
    arena_max_x = 38
    arena_min_y = 0
    arena_max_y = 40.5

    # Extracting the directory path and saving it to a new variable
    output_path = os.path.dirname(reversal_annotation_path)

    #-------------loading necessary files
    reversal_annotation, skeleton_spline, df_worm_parameter, spline_X, spline_Y, turn_annotation = read_csv_files(reversal_annotation_path, skeleton_spline_path, worm_pos_path, spline_X_path, spline_Y_path, turn_annotation_path)

    # Basic conversion to integer
    df['frame'] = df['frame'].astype(int)

    # Convert the position strings to tuples
    top_left_tuple = extract_coords(args.top_left_pos)
    odor_pos_tuple = extract_coords(args.odor_pos)

    # Initialize the system
    coord_system = CoordinateSystem(
        top_left_tuple,
        odor_pos_tuple,
        factor_px_to_mm
    )

    # Transform the coordinates
    df_worm_parameter = coord_system.transform_coordinates(df_worm_parameter)

    x_odor = coord_system.odor_x_mm
    y_odor = coord_system.odor_y_mm

    # Access transformed odor positions if needed
    print(f"Odor position (mm): x={x_odor}, y={y_odor}")
    print(df_worm_parameter.head())

    #_--------------------coorinate system augmentation finished!

    # Create a copy of df_worm_parameter
    df_skel_all = df_worm_parameter.copy()  # create copy of df_worm_parameter fo wormmovie later

    # Calculate corrected center position of the worm
    skel_pos_centroid = 100
    df_worm_parameter = correct_stage_pos_with_skeleton(
        df_worm_parameter,
        spline_X,
        spline_Y,
        skel_pos_centroid,  # 100 will calculate the centroid -> column name will be 'X/Y_rel_skel_pos_centroid'
        video_resolution_x,
        video_resolution_y,
        factor_px_to_mm,
        video_origin="crop"  # Set to "crop" for corrected logic
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
        video_origin="crop"  # Set to "crop" for corrected logic
    )

    print("added relative worm position:", df_worm_parameter)

    # calculate distances for stage, skeletton position 0 (nose) and 49 (center)
    df_worm_parameter['distance_to_odor_stage'] = df_worm_parameter.apply(lambda row: calculate_distance(row, 'X_rel', 'Y_rel', x_odor, y_odor), axis=1)
    df_worm_parameter[f'distance_to_odor_centroid'] = df_worm_parameter.apply(lambda row: calculate_distance(row, 'X_rel_skel_pos_centroid', 'Y_rel_skel_pos_centroid', x_odor, y_odor), axis=1)
    df_worm_parameter[f'distance_to_odor_{skel_pos_0}'] = df_worm_parameter.apply(lambda row: calculate_distance(row, f'X_rel_skel_pos_{skel_pos_0}', f'Y_rel_skel_pos_{skel_pos_0}', x_odor, y_odor), axis=1)

    # Convert to float
    df_worm_parameter['distance_to_odor_stage'] =  df_worm_parameter['distance_to_odor_stage'].astype(float)
    df_worm_parameter['distance_to_odor_centroid'] = df_worm_parameter['distance_to_odor_centroid'].astype(float)
    df_worm_parameter[f'distance_to_odor_{skel_pos_0}'] = df_worm_parameter[f'distance_to_odor_{skel_pos_0}'].astype(float)

    #add column that shows time passed in seconds
    calculate_time_in_seconds(df_worm_parameter, fps)

    print("added column for time:", df_worm_parameter)

    #calculate perceived concentration for nose and centroid position

    # Apply the function to create the 'Conc' column and ensure float conversion
    df_worm_parameter[f'conc_at_centroid'] = pd.to_numeric(df_worm_parameter.apply(
        lambda row: calculate_preceived_conc(
            row[f'distance_to_odor_centroid'], row['time_seconds'], conc_gradient_array, distance_array,
            diffusion_time_offset), axis=1), errors='coerce'
    )

    df_worm_parameter[f'conc_at_{skel_pos_0}'] = pd.to_numeric(df_worm_parameter.apply(
        lambda row: calculate_preceived_conc(
            row[f'distance_to_odor_{skel_pos_0}'], row['time_seconds'], conc_gradient_array, distance_array,
            diffusion_time_offset), axis=1), errors='coerce'
    )

    # Calculate delta concentration across the time interval
    time_interval_dC_dT = int(fps)  # Determine how far back in the past to compare

    # Compute differences and enforce float for the result
    df_worm_parameter[f'dC_centroid'] = df_worm_parameter[f'conc_at_centroid'].diff(periods=time_interval_dC_dT).astype(
        float)
    df_worm_parameter[f'dC_{skel_pos_0}'] = df_worm_parameter[f'conc_at_{skel_pos_0}'].diff(
        periods=1).astype(float)

    print("\nWorm  DataFrame wit Distance:")
    print(df_worm_parameter.head())

    '''
      Add angle calculations for bearing and curving angle to the dataframe

      For the function to calculate the angles the averaged centroid positions(worms trajectory) of the present, the past and the future is needed!
      How the function calculates the angle is explained in the function description!

      The shift in time is determined by variable n_shift and each integer reflects one frame of the video back

      What is counterintuitive is that the shift operation when positive shifts the whole df into the future meaning bringin 
      past datapoints to the present so to compare a value with a past value the shift needs to be positive and vice versa

      Here's a simple example to illustrate:

      Original data: [1, 2, 3, 4, 5]

      After applying a shift of +2: [NaN, NaN, 1, 2, 3]

      '''
    df_worm_parameter = calculate_displacement_vector(df_worm_parameter)
    df_worm_parameter = calculate_curving_angle(df_worm_parameter, window_size=1)
    df_worm_parameter = calculate_bearing_angle(df_worm_parameter)

    # Print confirmation and first few rows of the DataFrame
    print("Angles calculated.")
    print(df_worm_parameter.head())


    print(df_worm_parameter.head())  # This will print the first few rows of the DataFrame after adding the 'curving_angle' column
    '''
    - add behaviorial state dataframe to main dataframe 
    - calculates reversal start and end
    - calculates reversal frequency per minute
    '''
    print(reversal_annotation.head())
    # Renaming the second column from 1 to 'behaviour_state'
    reversal_annotation = reversal_annotation.rename(columns={1: 'behaviour_state'})
    reversal_annotation = reversal_annotation.drop(0, axis=1)

    turn_annotation = turn_annotation.drop('Unnamed: 0', axis=1)

    # Merge/join based on index
    df_worm_parameter = pd.merge(df_worm_parameter, reversal_annotation, left_index=True, right_index=True, how='left')
    df_worm_parameter = pd.merge(df_worm_parameter, turn_annotation, left_index=True, right_index=True, how='left')

    # Show the head of the merged DataFrame
    print(df_worm_parameter.head())

    # create column reversal onset and reversal end
    # Shift the 'behaviour_state' column by one to check the prior value
    prior_state = df_worm_parameter['behaviour_state'].shift(periods=-1, fill_value=0)

    # Check the conditions and assign values to the new column 'reversal_onset'
    df_worm_parameter['reversal_onset'] = ((prior_state != -1) & (df_worm_parameter['behaviour_state'] == -1)).astype(int)

    df_worm_parameter['reversal_end'] = ((prior_state == -1) & (df_worm_parameter['behaviour_state'] != -1)).astype(int)

    # Calculate the reversal frequenzy
    window_size = int(fps * 60)  # reversal frequenzy per minute

    df_worm_parameter['reversal_frequency'] = df_worm_parameter['reversal_onset'].rolling(window=window_size).sum()

    #Speed, radial Speed, NI

    df_worm_parameter = calculate_speed(df_worm_parameter, fps) #adds column speed to df

    df_worm_parameter = calculate_radial_speed(df_worm_parameter, fps) # adds column radial speed to df

    # calculating column navigational index based on speed and radial speed

    df_worm_parameter['NI'] = (df_worm_parameter['radial_speed'] / df_worm_parameter['speed'])

    '''
    data smoothing and cleaning part

    cals functions that smoothen and clean the data
    '''
    replace_outliers_with_nan(df_worm_parameter, 'speed', 2)
    replace_outliers_with_nan(df_worm_parameter, 'radial_speed', 2)
    replace_outliers_with_nan(df_worm_parameter, 'NI', 2)

    #-------------------------------
    #usses speed to update behavior state column
    update_behaviour_based_on_speed(df_worm_parameter, threshold=0.04)

    '''
    Plotting part

    cals functions that create various visualisations
    '''

    plot_ethogram(reversal_annotation, output_path, file_name='ehtogram.png')

    plot_ethogram_simple(reversal_annotation, output_path, file_name='ehtogram_simple.png')

    plot_skeleton_spline(skeleton_spline, output_path, file_name='kymogram.png')

    plot_dynamic_binned(df_worm_parameter, 'conc_at_centroid', output_path, 'conc_at_centroid_over_time.png',
                        bin_count=100)

    plot_dynamic_binned(df_worm_parameter, 'reversal_frequency', output_path, 'reversal_frequency_over_time.png',
                        bin_count=100)

    plot_dynamic_binned(df_worm_parameter, 'distance_to_odor_centroid', output_path,
                        'distance_to_odor_centroid_over_time.png', bin_count=100)

    plot_dynamic_binned(df_worm_parameter, 'NI', output_path, 'NI_over_time.png', bin_count=100)

    plot_dynamic_binned(df_worm_parameter, 'speed', output_path, 'speed_over_time.png', bin_count=100)

    plot_dynamic_binned(df_worm_parameter, 'NI', output_path, 'NI_over_time_with_speed_hue.png', hue_column='speed', bin_count=100)

    plot_turns(df_worm_parameter, output_path, file_name='turns.png')

    plot_chemotaxis_overview(df_worm_parameter, output_path, x_odor, y_odor, arena_min_x, arena_max_x, arena_min_y, arena_max_y, fps, file_name="chemotaxis_overview.png")


    '''
        concatenate df_worm_parameter and Spline_K before final output
    '''

    # Initial concatenation of worm parameter and skeleton spline data
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

    # Iterate over skeleton positions
    for skel_pos_abs in range(20):
        df_skel_pos_abs = correct_stage_pos_with_skeleton(
            df_skel_all,
            spline_X,
            spline_Y,
            skel_pos_abs,  # 100 will calculate the centroid -> column name will be 'X/Y_rel_skel_pos_centroid'
            video_resolution_x,
            video_resolution_y,
            factor_px_to_mm,
            video_origin="crop"  # Set to "crop" for corrected logic
        )

    print('Worm Animation DF:', df_skel_pos_abs.head())

    df_skel_pos_abs.drop(['frame', 'X', 'Y', 'time_imputed_seconds', 'X_rel', 'Y_rel', 'odor_x', 'odor_y'], axis=1, inplace=True)

    # Create MultiIndex columns for skeleton positions
    skel_pos_columns = pd.MultiIndex.from_product(
        [['skel_pos_abs'], df_skel_pos_abs.columns]
    )

    # Add skeleton position data to combined DataFrame
    df_combined = pd.concat([df_combined, df_skel_pos_abs], axis=1)

    # Combine all column hierarchies
    all_columns = chemotaxis_columns.append(spline_columns).append(skel_pos_columns)
    df_combined.columns = all_columns

    # Save final DataFrame
    df_combined.to_csv(os.path.join(output_path, 'chemotaxis_params.csv'), index=True)

    #create_worm_animation(df_skel_pos_abs, df_worm_parameter, output_path, x_odor, y_odor, fps, arena_min_x, arena_max_x, arena_min_y, arena_max_y, file_name='worm_movie.avi')


if __name__ == "__main__":

        print("Shell commands passed:", sys.argv)
        main(sys.argv[1:])  # exclude the script name from the args when called from shell