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
    calculate_angle,
)

from chemotaxis_analysis_high_res.plotting_visualisation import (
    plot_chemotaxis_overview,
    create_angle_animation,
)

def read_csv_files(beh_annotation_path:str, skeleton_spline_path:str, worm_pos_path:str, spline_X_path:str, spline_Y_path:str):
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

    # Read CSV files into separate dataframes
    beh_annotation_df = pd.read_csv(beh_annotation_path, header=None)
    skeleton_spline_df = pd.read_csv(skeleton_spline_path, header=None)
    worm_pos_df = pd.read_csv(worm_pos_path)
    spline_X_df = pd.read_csv(spline_X_path, header=None)
    spline_Y_df = pd.read_csv(spline_Y_path, header=None)

    # Print the head of each dataframe
    print("Behavior Annotation DataFrame:")
    print(beh_annotation_df.head())

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
    skeleton_spline_df = skeleton_spline_df.apply(pd.to_numeric, errors='coerce')
    worm_pos_df.iloc[:, 1:] = worm_pos_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce') #don't change 1. column time to numeric
    spline_X_df = spline_X_df.apply(pd.to_numeric, errors='coerce')
    spline_Y_df = spline_Y_df.apply(pd.to_numeric, errors='coerce')

    # Print the head of each dataframe
    print("_Behavior Annotation DataFrame:")
    print(beh_annotation_df.head())

    print("\nSkeleton Spline DataFrame:")
    print(skeleton_spline_df.head())

    print("\nWorm Position DataFrame:")
    print(worm_pos_df.head())

    print("\nSpline X DataFrame:")
    print(spline_X_df.head())

    print("\nSpline Y DataFrame:")
    print(spline_Y_df.head())

    print("Number of rows in beh_annotation_df:", len(beh_annotation_df))
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
        print("Frame lenght of recorded video:", len(spline_X_df))

    return beh_annotation_df, skeleton_spline_df, worm_pos_df, spline_X_df, spline_Y_df

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
    parser.add_argument('--stage_pos', help='Full path to the odor pos file', required=True)
    parser.add_argument('--skeleton_spline_X_coords', help='Full path to the skeleton_spline_X_coords CSV file', required=True)
    parser.add_argument('--skeleton_spline_Y_coords', help='Full path to the skeleton_spline_Y_coords CSV file', required=True)
    parser.add_argument('--factor_px_to_mm', help='conversion_facor px to mm',required=True)
    parser.add_argument('--video_resolution_x', help='video_resolution_x', required=True)
    parser.add_argument('--video_resolution_y', help='video_resolution_y', required=True)
    parser.add_argument('--fps', help='fps', required=True)
    parser.add_argument('--conc_gradient_array', help='exportet concentration_gradient.npy file for the odor used', required=True)
    parser.add_argument('--distance_array', help='exportet distance_array.npy file for the odor used', required=True)

    args = parser.parse_args(arg_list)

    beh_annotation_path = args.beh_annotation
    skeleton_spline_path = args.skeleton_spline
    worm_pos_path = args.worm_pos
    stage_pos_path = args.stage_pos
    spline_X_path = args.skeleton_spline_X_coords
    spline_Y_path = args.skeleton_spline_Y_coords
    factor_px_to_mm = args.factor_px_to_mm
    video_resolution_x = int(args.video_resolution_x)
    video_resolution_y = int(args.video_resolution_y)
    fps = args.fps
    conc_gradient_array = args.conc_gradient_array
    distance_array = args.distance_array

    # Set arena boundaries
    arena_min_x = 0
    arena_max_x = 38
    arena_min_y = 0
    arena_max_y = 40.5

    # Extracting the directory path and saving it to a new variable
    output_path = os.path.dirname(beh_annotation_path)

    #-------------loading necessary files
    beh_annotation, skeleton_spline, df_worm_parameter, spline_X, spline_Y = read_csv_files(beh_annotation_path, skeleton_spline_path, worm_pos_path, spline_X_path, spline_Y_path)

    #-----------------load config file for odor and arena positions
    with open(stage_pos_path, 'r') as config_file:
        stage_pos = yaml.safe_load(config_file)

    # Access the odor coordinates
    x_odor, y_odor = extract_coords(stage_pos['odor_pos'])
    x_zero, y_zero = extract_coords(stage_pos['top_left'])

    # Print the variables together
    print("Odor position: x =", x_odor, ", y =", y_odor)
    print("Top left position: x =", x_zero, ", y =", y_zero)

    # -------------shifts every value of x and y in the positive range, by addition of the lowest value to all values
    # Finding the lowest negative value among X_rel and Y_rel columns

    lowest_neg_x = df_worm_parameter['X'][df_worm_parameter['X'] < 0].min()
    lowest_neg_y = df_worm_parameter['Y'][df_worm_parameter['Y'] < 0].min()

    # Saving the lowest negative value as move_grid_factor
    move_grid_factor = min(lowest_neg_x, lowest_neg_y, x_odor, y_odor, x_zero, y_zero)

    # Adding the lowest negative value to every value in x_rel, y_rel, and additional values
    df_worm_parameter['X'] += abs(move_grid_factor)
    df_worm_parameter['Y'] += abs(move_grid_factor)
    x_odor += abs(move_grid_factor)
    y_odor += abs(move_grid_factor)
    x_zero += abs(move_grid_factor)
    y_zero += abs(move_grid_factor)

    # Display the loaded dataframes
    print("Behavior Annotation DataFrame:")
    print(beh_annotation.head())

    print("\nSkeleton Spline DataFrame:")
    print(skeleton_spline.head())

    print("\nWorm Pos DataFrame:")
    print(df_worm_parameter.head())

    # adjust odor point to relative grid via reference point
    x_odor = x_odor - x_zero
    y_odor = y_odor - y_zero

    print("Adjusted x_odor:")
    print(x_odor)
    print("Adjusted y_odor:")
    print(y_odor)

    # Apply the conversion function to relative coordinates to each row, add x_rel and y_rel columns
    df_worm_parameter = df_worm_parameter.apply(lambda row: convert_coordinates(row, x_zero, y_zero), axis=1)

    df_worm_parameter['X_rel'] = df_worm_parameter['X_rel'].abs()  # shift relative stage position to positive values

    # Drop the 'time' column because it is not needed
    df_worm_parameter.drop(columns='time', inplace=True)

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

    centroid_rolling_mean_window_size = 10

    # Calculate the rolling mean for the 'X_rel_skel_pos_centroid' column
    df_worm_parameter['X_rel_skel_pos_centroid_corrected'] = df_worm_parameter['X_rel_skel_pos_centroid'].rolling(window=centroid_rolling_mean_window_size).mean()
    df_worm_parameter['Y_rel_skel_pos_centroid_corrected'] = df_worm_parameter['Y_rel_skel_pos_centroid'].rolling(window=centroid_rolling_mean_window_size).mean()

    print("added relative worm position:", df_worm_parameter)

    # calculate distances for stage, skeletton position 0 (nose) and 49 (center)
    df_worm_parameter['distance_to_odor_stage'] = df_worm_parameter.apply(lambda row: calculate_distance(row, 'X_rel', 'Y_rel', x_odor, y_odor), axis=1)
    df_worm_parameter[f'distance_to_odor_centroid'] = df_worm_parameter.apply(lambda row: calculate_distance(row, 'X_rel_skel_pos_centroid', 'Y_rel_skel_pos_centroid', x_odor, y_odor), axis=1)
    df_worm_parameter[f'distance_to_odor_{skel_pos_0}'] = df_worm_parameter.apply(lambda row: calculate_distance(row, f'X_rel_skel_pos_{skel_pos_0}', f'Y_rel_skel_pos_{skel_pos_0}', x_odor, y_odor), axis=1)

    #add column that shows time passed in seconds
    calculate_time_in_seconds(df_worm_parameter, fps)

    print("added column for time:", df_worm_parameter)

    #calculate perceived concentration for nose and centroid position

    # Apply the function to create the 'Conc' column
    df_worm_parameter[f'conc_at_centroid'] = df_worm_parameter.apply(
        lambda row: calculate_preceived_conc(row[f'distance_to_odor_centroid'], row['time_seconds'], conc_gradient_array, distance_array),
        axis=1)

    df_worm_parameter[f'conc_at_{skel_pos_0}'] = df_worm_parameter.apply(
        lambda row: calculate_preceived_conc(row[f'distance_to_odor_{skel_pos_0}'], row['time_seconds'], conc_gradient_array,
                                             distance_array),
        axis=1)

    # Calculate delta concentration across the time interval
    time_interval_dC_dT = 30  #need to figure out how far back in the past to compare it to

    df_worm_parameter[f'dC_centroid'] = df_worm_parameter[f'conc_at_centroid'].diff(periods=time_interval_dC_dT)
    df_worm_parameter[f'dC_{skel_pos_0}'] = df_worm_parameter[f'conc_at_{skel_pos_0}'].diff(periods=time_interval_dC_dT)

    print("\nWorm  DataFrame wit Distance:")
    print(df_worm_parameter.head())


    '''
    Add angle calculations for bearing and curving angle to the dataframe
    
    For the function to calculate the angles the averaged centroid positions(worms trajectory) of the present, the past and the future is needed!
    How the function calculates the angle is explained in the function description!
    
    The shift in time is determined by variable n_shift and each integer reflects one frame of the video back
    '''
    time_shifted_for_angles=1000
    # Replace NaN values with a placeholder value before applying the shift
    df_worm_parameter['X_shifted_negative'] = df_worm_parameter['X_rel_skel_pos_centroid_corrected'].shift(-time_shifted_for_angles).fillna(0)
    df_worm_parameter['Y_shifted_negative'] = df_worm_parameter['Y_rel_skel_pos_centroid_corrected'].shift(-time_shifted_for_angles).fillna(0)

    # Replace NaN values with a placeholder value before applying the shift
    df_worm_parameter['X_shifted_positive'] = df_worm_parameter['X_rel_skel_pos_centroid_corrected'].shift(+time_shifted_for_angles).fillna(0)
    df_worm_parameter['Y_shifted_positive'] = df_worm_parameter['Y_rel_skel_pos_centroid_corrected'].shift(+time_shifted_for_angles).fillna(0)

    # Applying the function to each row and creating a new column 'bearing_angle'
    df_worm_parameter['bearing_angle'] = df_worm_parameter.apply(lambda row: calculate_angle(row['X_rel_skel_pos_centroid_corrected'], row['Y_rel_skel_pos_centroid_corrected'], row['X_shifted_negative'],row['Y_shifted_negative'], x_odor, y_odor), axis=1)

    # Applying the function to each row and creating a new column 'curving_angle'
    df_worm_parameter['curving_angle'] = df_worm_parameter.apply(lambda row: calculate_angle(row['X_rel_skel_pos_centroid_corrected'], row['Y_rel_skel_pos_centroid_corrected'], row['X_shifted_negative'],row['Y_shifted_negative'], row['X_shifted_positive'],row['Y_shifted_positive']), axis=1)

    '''
    - add behaviorial state dataframe to main dataframe 
    - calculates reversal start and end
    - calculates reversal frequency per minute
    '''
    # Renaming the second column from '0' to 'behaviour_state'
    beh_annotation.rename(columns={'0': 'behaviour_state'}, inplace=True)

    # Merge/join based on index
    df_worm_parameter = pd.merge(df_worm_parameter, beh_annotation, left_index=True, right_index=True, how='inner')

    # create column reversal onset and reversal end
    # Shift the 'behaviour_state' column by one to check the prior value
    prior_state = df_worm_parameter['behaviour_state'].shift(periods=-1, fill_value=0)

    # Check the conditions and assign values to the new column 'reversal_onset'
    df_worm_parameter['reversal_onset'] = ((prior_state != -1) & (df_worm_parameter['behaviour_state'] == -1)).astype(int)

    df_worm_parameter['reversal_end'] = ((prior_state == -1) & (df_worm_parameter['behaviour_state'] != -1)).astype(int)

    # Calculate the reversal frequenzy
    window_size = fps * 60  # reversal frequenzy per minute

    df_worm_parameter['reversal_frequency'] = df_worm_parameter['reversal_onset'].rolling(window=window_size).sum()

    '''
    Plotting part
    
    cals functions that create various visualisations
    '''

    plot_chemotaxis_overview(df_worm_parameter, output_path, x_odor, y_odor, arena_min_x, arena_max_x, arena_min_y, arena_max_y, file_name="chemotaxis_overview.png")

    create_angle_animation(df_worm_parameter, output_path, x_odor, y_odor, fps, filename ='angle_animation.avi')

if __name__ == "__main__":

        print("Shell commands passed:", sys.argv)
        main(sys.argv[1:])  # exclude the script name from the args when called from shell