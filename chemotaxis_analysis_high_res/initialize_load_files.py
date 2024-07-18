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
    plot_skeleton_spline,
    plot_odor_concentration,
    plot_speed,
    plot_distance_to_odor,
    plot_NI,
    create_worm_animation,
    plot_angles_binned,
    plot_turns,
    plot_pumps,
)

from chemotaxis_analysis_high_res.data_smothing import (
    replace_outliers_with_nan,
    update_behaviour_based_on_speed,
)

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
    print("_Behavior Annotation DataFrame:")
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

def process_pharynx_pump_csv(file_path, fps):
    '''
    I need to implement this cleaner, this seems messy when it is just one function
    '''

    print(f"Processing pharynx CSV file at {file_path}")
    # Assuming the file has a header row
    df = pd.read_csv(file_path)

    # remove column names and set first row to new column name
    df.columns = df.iloc[0]
    df = df[1:]

    # Get the first row (which will become the second level of column names)
    second_level_names = df.iloc[0]

    # Create a MultiIndex for columns using the existing column names as the first level
    first_level_names = df.columns
    multi_index = pd.MultiIndex.from_arrays([first_level_names, second_level_names])

    # Set the new MultiIndex as the columns of the DataFrame
    df.columns = multi_index

    # Remove the first row from the DataFrame as it's now used for column names
    df = df.iloc[1:]

    # Removing the first column (index 0)
    df = df.drop(df.columns[0], axis=1)
    df = df.reset_index(drop=True)

    flow_df = df.xs(key='end', level=0, axis=1)
    flow_df['x'] = pd.to_numeric(flow_df['x'])
    flow_df['y'] = pd.to_numeric(flow_df['y'])

    nose_df = df.xs(key='nose', level=0, axis=1)
    nose_df['x'] = pd.to_numeric(nose_df['x'])
    nose_df['y'] = pd.to_numeric(nose_df['y'])

    x_diff_squared = (flow_df['x'] - nose_df['x'])
    y_diff_squared = (flow_df['y'] - nose_df['y'])

    # Calculate the Euclidean distance
    distance_series = np.sqrt(x_diff_squared + y_diff_squared)

    distance_df = pd.DataFrame({'distance': distance_series})

    def binarize_data(column, lower_threshold, upper_threshold):
        return [1 if lower_threshold <= x <= upper_threshold else 0 for x in column]

    lower_threshold_value = 6.5
    upper_threshold_value = 14

    distance_df['binarized'] = binarize_data(distance_df['distance'], lower_threshold_value, upper_threshold_value)


    # Define the number of frames per minutes (30 fps * 60 seconds)
    frames_per_min = int(fps * 60)
    print(frames_per_min)

    # Calculate the rolling sum of 'binary_data' with a window size of frames_per_10_seconds
    distance_df['pumping_frequency'] = distance_df['binarized'].rolling(frames_per_min, min_periods=1).sum()

    distance_df['pumping_frequency'] = distance_df['pumping_frequency'] / 2 #needed because of how DLC tracks pumping

    # Fill NaN values with 0
    distance_df['pumping_frequency'].fillna(0, inplace=True)

    # Convert the 'frequency' column to integers
    distance_df['pumping_frequency'] = distance_df['pumping_frequency'].astype(int)

    print("pharynx pumping binarized!")
    print(distance_df.head())

    return distance_df


# Define a function to extract the x and y values from the yaml file
def extract_coords(coord_string:str):
    x, y = coord_string.split(',')
    x = float(x.strip().split('=')[1])
    y = float(y.strip().split('=')[1])
    return x, y

# Define a function to convert X and Y values to the absolute grid
def convert_coordinates(row: pd.Series, x_zero: float, y_zero: float) -> pd.Series:
    row["X_rel"] = row["X"] - x_zero
    row["Y_rel"] = row["Y"] - y_zero
    return row


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
    df.to_csv(full_path, index=False)  # Change 'index=False' to 'index=True' if you want to include the index.

def main(arg_list=None):
    parser = argparse.ArgumentParser(description='Read CSV files and plot data')
    parser.add_argument('--beh_annotation', help='Full path to the behavior annotation CSV file', required=True)
    parser.add_argument('--skeleton_spline', help='Full path to the skeleton spline CSV file', required=True)
    parser.add_argument('--worm_pos', help='Full path to the worm pos text file', required=True)
    parser.add_argument('--worm_config', help='Full path to the odor pos file', required=True)
    parser.add_argument('--skeleton_spline_X_coords', help='Full path to the skeleton_spline_X_coords CSV file', required=True)
    parser.add_argument('--skeleton_spline_Y_coords', help='Full path to the skeleton_spline_Y_coords CSV file', required=True)
    parser.add_argument('--factor_px_to_mm', help='conversion_facor px to mm',required=True)
    parser.add_argument('--video_resolution_x', help='video_resolution_x', required=True)
    parser.add_argument('--video_resolution_y', help='video_resolution_y', required=True)
    parser.add_argument('--fps', help='fps', required=True)
    parser.add_argument('--conc_gradient_array', help='exportet concentration_gradient.npy file for the odor used', required=True)
    parser.add_argument('--distance_array', help='exportet distance_array.npy file for the odor used', required=True)
    parser.add_argument('--turn_annotation', help='Full path to the turn annotation CSV file', required=True)
    parser.add_argument('--pharynx_pump_csv', help='Full path to the CSV file from DLC tracking', required=False)
    parser.add_argument('--downsampled_avi', help='Full path to the compressed avi', required=True)

    args = parser.parse_args(arg_list)

    beh_annotation_path = args.beh_annotation
    turn_annotation_path = str(args.turn_annotation)
    skeleton_spline_path = args.skeleton_spline
    worm_pos_path = args.worm_pos
    worm_config_path = args.worm_config
    spline_X_path = args.skeleton_spline_X_coords
    spline_Y_path = args.skeleton_spline_Y_coords
    factor_px_to_mm = float(args.factor_px_to_mm)
    video_resolution_x = int(args.video_resolution_x)
    video_resolution_y = int(args.video_resolution_y)
    fps = float(args.fps)
    pharynx_pump_csv_path = str(args.pharynx_pump_csv)
    downsampled_avi_path = str(args.downsampled_avi)

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
    output_path = os.path.dirname(beh_annotation_path)

    #-------------loading necessary files
    beh_annotation, skeleton_spline, df_worm_parameter, spline_X, spline_Y, turn_annotation = read_csv_files(beh_annotation_path, skeleton_spline_path, worm_pos_path, spline_X_path, spline_Y_path, turn_annotation_path)

    # Check if the pharynx_pump_csv argument is provided and not None
    if args.pharynx_pump_csv is not None:
        pharynx_pumping_binary_df = process_pharynx_pump_csv(pharynx_pump_csv_path, fps)
    else:
        print("analysis without pharynx pumping data")


    #-----------------load config file for odor and arena positions
    with open(worm_config_path, 'r') as config_file:
        worm_config = yaml.safe_load(config_file)

    # Assign the value from the loaded YAML file to the variable, with a default of 0 if the key doesn't exist
    diffusion_time_offset = worm_config.get('diffusion_time_offset', 3600)

    print('diffusion_time_offset:', diffusion_time_offset)

    # Access the odor coordinates
    x_odor, y_odor = extract_coords(worm_config['odor_pos'])
    x_zero, y_zero = extract_coords(worm_config['top_left'])

    # Print the variables together
    print("Odor position: x =", x_odor, ", y =", y_odor)
    print("Top left position: x =", x_zero, ", y =", y_zero)

    # -------------shifts every value of x and y in the positive range, by addition of the lowest value to all values

    # Find the minimum values in the X and Y columns
    min_x = float(np.nanmin(df_worm_parameter['X']))
    min_y = float(np.nanmin(df_worm_parameter['Y']))

    # Determine the overall minimum values for x and y (including odor and zero points)
    overall_min_x = min(min_x, x_odor, x_zero)
    overall_min_y = min(min_y, y_odor, y_zero)

    # Calculate the necessary shift for each column
    shift_x = abs(overall_min_x) if overall_min_x < 0 else 0.0
    shift_y = abs(overall_min_y) if overall_min_y < 0 else 0.0

    # Apply the shift to the DataFrame and additional values if necessary
    if shift_x > 0:
        df_worm_parameter['X'] += shift_x
        x_odor += shift_x
        x_zero += shift_x

    if shift_y > 0:
        df_worm_parameter['Y'] += shift_y
        y_odor += shift_y
        y_zero += shift_y

    print(f"Shift X: {shift_x}, Shift Y: {shift_y}")

    print(f"Shift X: {shift_x}, Shift Y: {shift_y}")

    print("\nWorm Pos DataFrame shifted:")
    print(df_worm_parameter.head())

    # adjust odor point to relative grid via reference point
    x_odor = x_odor - x_zero
    y_odor = y_odor - y_zero

    x_odor = abs(x_odor)  # shift relative odor position to positive values

    print("relative x_odor:")
    print(x_odor)
    print("relative y_odor:")
    print(y_odor)

    # Add the new columns to the dataframe
    df_worm_parameter['x_odor'] = x_odor
    df_worm_parameter['y_odor'] = y_odor

    # Apply the conversion function to relative coordinates to each row, add x_rel and y_rel columns
    df_worm_parameter = df_worm_parameter.apply(lambda row: convert_coordinates(row, x_zero, y_zero), axis=1)

    df_worm_parameter['X_rel'] = df_worm_parameter['X_rel'].abs()  # shift relative stage position to positive values

    #finished initialisation and aligning

    # Create a copy of df_worm_parameter
    df_worm_movie = df_worm_parameter.copy()  # create copy of df_worm_parameter fo wormmovie later

    # calculate corrected center position of the worm
    skel_pos_centroid = 100
    df_worm_parameter = correct_stage_pos_with_skeleton(
        df_worm_parameter,
        spline_X,
        spline_Y,
        skel_pos_centroid, #100 will calculate the centroid -> column name will be 'X/Y_rel_skel_pos_centroid'
        video_resolution_x,
        video_resolution_y,
        factor_px_to_mm
    )

    skel_pos_0 = 0

    df_worm_parameter = correct_stage_pos_with_skeleton(
        df_worm_parameter,
        spline_X,
        spline_Y,
        skel_pos_0, # 0 reflects nose position
        video_resolution_x,
        video_resolution_y,
        factor_px_to_mm
    )

    #perform rolling mean on centroid columns to get the proper trajectory of the worm, window size can be defined

    centroid_rolling_mean_window_size = int(fps)

    # Calculate the rolling mean for the 'X_rel_skel_pos_centroid' column
    df_worm_parameter['X_rel_skel_pos_centroid_corrected'] = df_worm_parameter['X_rel_skel_pos_centroid'].rolling(window=centroid_rolling_mean_window_size).mean()
    df_worm_parameter['Y_rel_skel_pos_centroid_corrected'] = df_worm_parameter['Y_rel_skel_pos_centroid'].rolling(window=centroid_rolling_mean_window_size).mean()

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

    # Apply the function to create the 'Conc' column
    df_worm_parameter[f'conc_at_centroid'] = df_worm_parameter.apply(
        lambda row: calculate_preceived_conc(
            row[f'distance_to_odor_centroid'], row['time_seconds'], conc_gradient_array, distance_array, diffusion_time_offset),axis=1)

    df_worm_parameter[f'conc_at_{skel_pos_0}'] = df_worm_parameter.apply(
        lambda row: calculate_preceived_conc(
            row[f'distance_to_odor_{skel_pos_0}'], row['time_seconds'], conc_gradient_array,distance_array, diffusion_time_offset),axis=1)

    # Calculate delta concentration across the time interval
    time_interval_dC_dT = int(fps)  #need to figure out how far back in the past to compare it to

    df_worm_parameter[f'dC_centroid'] = df_worm_parameter[f'conc_at_centroid'].diff(periods=time_interval_dC_dT)
    df_worm_parameter[f'dC_{skel_pos_0}'] = df_worm_parameter[f'conc_at_{skel_pos_0}'].diff(periods=time_interval_dC_dT)

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
    df_worm_parameter = calculate_curving_angle(df_worm_parameter, bearing_range=1)
    df_worm_parameter = calculate_bearing_angle(df_worm_parameter, x_odor, y_odor)


    # Print confirmation and first few rows of the DataFrame
    print("Angles calculated.")
    print(df_worm_parameter.head())


    print(df_worm_parameter.head())  # This will print the first few rows of the DataFrame after adding the 'curving_angle' column
    '''
    - add behaviorial state dataframe to main dataframe 
    - calculates reversal start and end
    - calculates reversal frequency per minute
    '''
    # Renaming the second column from 1 to 'behaviour_state'
    beh_annotation = beh_annotation.rename(columns={1: 'behaviour_state'})

    # Merge/join based on index
    df_worm_parameter = pd.merge(df_worm_parameter, beh_annotation, left_index=True, right_index=True, how='left')
    df_worm_parameter = pd.merge(df_worm_parameter, turn_annotation, left_index=True, right_index=True, how='left')

    df_worm_parameter = df_worm_parameter.drop(columns=['Unnamed: 0']) #index colum from turn annotations

    if args.pharynx_pump_csv is not None:
        df_worm_parameter = pd.merge(df_worm_parameter, pharynx_pumping_binary_df, left_index=True, right_index=True, how='left')
    else:
        print("analysis without pharynx pumping data")

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
    # NI: -1 (away) to 1 (towards), 0 (perpendicular/random); measures movement efficiency

    df_worm_parameter['NI'] = (df_worm_parameter['radial_speed'] / df_worm_parameter['speed'])

    '''
    data smoothing and cleaning part

    cals functions that smoothen and clean the data
    '''
    replace_outliers_with_nan(df_worm_parameter, 'speed', 2)
    replace_outliers_with_nan(df_worm_parameter, 'radial_speed', 2)
    replace_outliers_with_nan(df_worm_parameter, 'NI', 2)

    #-------------------------------
    window_size_speed = int(fps) #2 seconds smoothing for speed

    df_worm_parameter['speed_s'] = df_worm_parameter['speed'].rolling(window=window_size_speed).mean()
    df_worm_parameter['radial_speed_s'] = df_worm_parameter['radial_speed'].rolling(window=window_size_speed).mean()
    df_worm_parameter['NI_s'] = df_worm_parameter['NI'].rolling(window=window_size_speed).mean()

    window_size_angle = int(fps)

    df_worm_parameter['bearing_angle_s'] = df_worm_parameter['bearing_angle'].rolling(window=window_size_angle).mean()
    df_worm_parameter['curving_angle_s'] = df_worm_parameter['curving_angle'].rolling(window=window_size_angle).mean()

    #up
    update_behaviour_based_on_speed(df_worm_parameter, threshold=0.04)

    '''
    Plotting part
    
    cals functions that create various visualisations
    '''

    plot_ethogram(beh_annotation, output_path, file_name = 'ehtogram.png')

    plot_skeleton_spline(skeleton_spline, output_path, file_name = 'kymogram.png')

    plot_odor_concentration(df_worm_parameter, output_path, file_name = 'perceived_conc.png')

    plot_speed(df_worm_parameter, output_path, file_name = 'speed.png')

    plot_NI(df_worm_parameter, output_path, file_name='NI.png')

    plot_distance_to_odor(df_worm_parameter, output_path, file_name = 'distance_to_odor.png')

    plot_chemotaxis_overview(df_worm_parameter, output_path, x_odor, y_odor, arena_min_x, arena_max_x, arena_min_y, arena_max_y, fps, file_name="chemotaxis_overview.png")

    #create_angle_animation(df_worm_parameter, output_path, x_odor, y_odor, fps, file_name='angle_animation.avi', nth_frame=1000)

    plot_angles_binned(df_worm_parameter, 'bearing_angle_s', 'curving_angle_s', output_path,  num_bins=20, file_name='curving_angle_binned_plot.png')

    plot_turns(df_worm_parameter, output_path, file_name='turns.png')

    if args.pharynx_pump_csv is not None:
        plot_pumps(df_worm_parameter, output_path, file_name='pharynx_pumps.png')
    else:
        print("analysis without pharynx pumping data")


    # Saving param df to a CSV file
    df_worm_parameter.to_csv(os.path.join(output_path, 'chemotaxis_params.csv'), index=True)

    #create animation of whole worm skelleton in arena
    # Assuming df_worm_parameter, spline_X, spline_Y, video_resolution_x, video_resolution_y, factor_px_to_mm are defined

    # Define skel_pos_0
    skel_pos_movie = 0
    
    # Iterate over skeleton positions from 0 to 100 inclusive
    for skel_pos_movie in range(101):
        print(skel_pos_movie)
        df_worm_movie = correct_stage_pos_with_skeleton(
            df_worm_movie,
            spline_X,
            spline_Y,
            skel_pos_movie,
            video_resolution_x,
            video_resolution_y,
            factor_px_to_mm
        )

    print('Worm Animation DF:', df_worm_movie.head())

    create_worm_animation(df_worm_movie, df_worm_parameter, output_path, x_odor, y_odor, fps, arena_min_x, arena_max_x, arena_min_y, arena_max_y, downsampled_avi_path, 1000,  file_name='worm_movie.avi')


if __name__ == "__main__":

        print("Shell commands passed:", sys.argv)
        main(sys.argv[1:])  # exclude the script name from the args when called from shell