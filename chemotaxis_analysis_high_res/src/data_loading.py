import pandas as pd
import numpy as np
import os
import re
from .calculations import interpolate_df, correct_dlc_coordinates


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


def read_csv_files(beh_annotation_path: str, skeleton_spline_path: str, worm_pos_path: str,
                   spline_X_path: str, spline_Y_path: str, turn_annotation_path: str):
    # Check if the file paths exist
    required_files = [
        beh_annotation_path, skeleton_spline_path, worm_pos_path,
        spline_X_path, spline_Y_path, turn_annotation_path
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    # Read CSV files into separate dataframes
    beh_annotation_df = pd.read_csv(beh_annotation_path, header=None, encoding='latin-1')
    skeleton_spline_df = pd.read_csv(skeleton_spline_path, header=None, encoding='latin-1')
    turn_annotation_df = pd.read_csv(turn_annotation_path, encoding='latin-1')

    # Try UTF-8 first (standard), then fall back to latin-1 for compatibility
    try:
        worm_pos_df = pd.read_csv(worm_pos_path, encoding='utf-8')
        print(f"Successfully loaded {worm_pos_path} with UTF-8 encoding")
    except UnicodeDecodeError as e:
        print(f"UTF-8 decode failed for {worm_pos_path}: {e}")
        print(f"Falling back to latin-1 encoding...")
        try:
            worm_pos_df = pd.read_csv(worm_pos_path, encoding='latin-1')
            print(f"Successfully loaded {worm_pos_path} with latin-1 encoding")
        except Exception as fallback_error:
            raise RuntimeError(f"Failed to read {worm_pos_path} with both UTF-8 and latin-1 encodings. "
                             f"Original UTF-8 error: {e}. Latin-1 error: {fallback_error}") from fallback_error
    worm_pos_df = worm_pos_df.drop(columns=['time'],
                                   errors='ignore')  # deletes old time column before interpolation step

    # Enforce integer type for 'frame' column if it exists
    if 'frame' in worm_pos_df.columns:
        worm_pos_df['frame'] = worm_pos_df['frame'].fillna(0).astype(int)

    spline_X_df = pd.read_csv(spline_X_path, header=None, encoding='latin-1')
    spline_Y_df = pd.read_csv(spline_Y_path, header=None, encoding='latin-1')

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

    # Debug: Show worm_pos_df before numeric conversion
    print("\n=== DEBUGGING: worm_pos_df BEFORE numeric conversion ===")
    print("Columns:", worm_pos_df.columns.tolist())
    print("Data types:", worm_pos_df.dtypes)
    if len(worm_pos_df) > 0:
        print("First few rows:")
        print(worm_pos_df.head())
        # Show specific X,Y values if they exist
        for col in ['X', 'Y']:
            if col in worm_pos_df.columns:
                print(f"Sample {col} values:", worm_pos_df[col].head(3).tolist())

    beh_annotation_df = beh_annotation_df.apply(pd.to_numeric, errors='coerce')
    turn_annotation_df = turn_annotation_df.apply(pd.to_numeric, errors='coerce')
    skeleton_spline_df = skeleton_spline_df.apply(pd.to_numeric, errors='coerce')
    
    # Fix: Apply numeric conversion to worm_pos_df but skip X,Y columns if they exist
    coordinate_columns = ['X', 'Y']
    worm_pos_numeric_columns = [col for col in worm_pos_df.columns if col not in coordinate_columns]
    
    if worm_pos_numeric_columns:
        # Apply numeric conversion only to non-coordinate columns
        worm_pos_df[worm_pos_numeric_columns] = worm_pos_df[worm_pos_numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # For X,Y columns, convert them properly handling leading zeros
    for col in coordinate_columns:
        if col in worm_pos_df.columns:
            worm_pos_df[col] = pd.to_numeric(worm_pos_df[col], errors='coerce')
    
    spline_X_df = spline_X_df.apply(pd.to_numeric, errors='coerce')
    spline_Y_df = spline_Y_df.apply(pd.to_numeric, errors='coerce')

    # Debug: Show worm_pos_df after numeric conversion
    print("\n=== DEBUGGING: worm_pos_df AFTER numeric conversion ===")
    print("Data types:", worm_pos_df.dtypes)
    if len(worm_pos_df) > 0:
        print("First few rows:")
        print(worm_pos_df.head())
        # Show specific X,Y values after conversion
        for col in ['X', 'Y']:
            if col in worm_pos_df.columns:
                print(f"Sample {col} values after conversion:", worm_pos_df[col].head(3).tolist())
                print(f"{col} column NaN count:", worm_pos_df[col].isna().sum())

    print("Number of rows in each dataframe:")
    print(f"beh_annotation_df: {len(beh_annotation_df)}")
    print(f"turn_annotation_df: {len(turn_annotation_df)}")
    print(f"skeleton_spline_df: {len(skeleton_spline_df)}")
    print(f"worm_pos_df: {len(worm_pos_df)}")
    print(f"spline_X_df: {len(spline_X_df)}")
    print(f"spline_Y_df: {len(spline_Y_df)}")

    # Check if worm_pos has same length as frame dependent data -> stage pos is tracked separate and can have different FPS
    # -> interpolate
    print("Stage Position Dataframe length before interpolation:", len(worm_pos_df))

    if len(worm_pos_df) != len(spline_X_df):
        # Apply the interpolation for each column
        worm_pos_df = worm_pos_df.apply(lambda x: interpolate_df(x, len(spline_X_df)), axis=0)
        print("Stage Position Dataframe length after interpolation:", len(worm_pos_df))
        print("Stage Position Dataframe head after interpolation:", worm_pos_df.head())
        print("Frame length of recorded video:", len(spline_X_df))

    return beh_annotation_df, skeleton_spline_df, worm_pos_df, spline_X_df, spline_Y_df, turn_annotation_df


def add_dlc_coordinates(df: pd.DataFrame, dlc_file_path: str, nose_label: str, tail_label: str,
                        video_resolution_x: int, video_resolution_y: int,
                        factor_px_to_mm: float, img_type: str) -> pd.DataFrame:
    """
    Add DLC coordinates to the main DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Main DataFrame
    dlc_file_path : str
        Path to DLC h5 file
    nose_label, tail_label : str
        DLC labels for nose and tail
    video_resolution_x, video_resolution_y : int
        Video resolution
    factor_px_to_mm : float
        Pixel to mm conversion factor
    img_type : str
        Image type

    Returns:
    --------
    pd.DataFrame
        DataFrame with DLC coordinates added
    """
    print(f"Processing DLC file: {dlc_file_path}")

    # Load nose and tail coordinates from DLC file
    nose_coords, tail_coords = load_dlc_coordinates(dlc_file_path, nose_label, tail_label)

    if nose_coords is not None and tail_coords is not None:
        # Convert DLC coordinates to mm and correct based on video origin
        nose_mm, tail_mm = correct_dlc_coordinates(
            df, nose_coords, tail_coords, video_resolution_x, video_resolution_y,
            factor_px_to_mm, img_type
        )

        # Check if lengths match (they should)
        if len(df) != len(nose_mm):
            print(f"Warning: DLC data length ({len(nose_mm)}) doesn't match main data length ({len(df)})")
            print("This is unexpected since both should come from the same video frames.")

        # Add DLC coordinates to the main DataFrame
        # Use the min length to avoid index errors in case the lengths differ slightly
        min_length = min(len(df), len(nose_mm))

        df['X_rel_DLC_nose'] = nose_mm['X_rel_DLC_nose'].values[:min_length]
        df['Y_rel_DLC_nose'] = nose_mm['Y_rel_DLC_nose'].values[:min_length]
        df['X_rel_DLC_tail'] = tail_mm['X_rel_DLC_tail'].values[:min_length]
        df['Y_rel_DLC_tail'] = tail_mm['Y_rel_DLC_tail'].values[:min_length]

        print(f"Added DLC tracking data to main DataFrame")

    return df


def extract_coords(pos_string):
    if pos_string is None:
        return None

    # Extract all numbers (including decimals) from the string
    numbers = re.findall(r'-?\d+\.?\d*', pos_string)

    # Return first two numbers as x, y
    return float(numbers[0]), float(numbers[1])