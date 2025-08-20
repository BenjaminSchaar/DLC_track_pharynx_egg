import pandas as pd
import numpy as np
import os
import re
from .calculations import interpolate_df, correct_dlc_coordinates


def validate_file_readability(file_path):
    """
    Validate that a file is readable as text and not binary garbage.
    
    Parameters:
    -----------
    file_path : str
        Path to the file to validate
        
    Returns:
    --------
    bool
        True if file appears to be readable text, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            print(f"ERROR: File does not exist: {file_path}")
            return False
            
        file_size = os.path.getsize(file_path)
        print(f"Validating file: {file_path} (size: {file_size} bytes)")
        
        if file_size == 0:
            print(f"ERROR: File is empty: {file_path}")
            return False
            
        # Read first 100 bytes to check for binary content
        with open(file_path, 'rb') as f:
            first_bytes = f.read(100)
            
        print(f"First 50 bytes (hex): {first_bytes[:50].hex()}")
        print(f"First 50 bytes (repr): {repr(first_bytes[:50])}")
        
        # Check for null bytes (common in binary files)
        if b'\x00' in first_bytes:
            print(f"ERROR: File contains null bytes, appears to be binary: {file_path}")
            return False
            
        # Check if most bytes are printable ASCII/extended ASCII
        printable_count = sum(1 for byte in first_bytes if 32 <= byte <= 126 or byte in [9, 10, 13])
        printable_ratio = printable_count / len(first_bytes) if first_bytes else 0
        
        print(f"Printable character ratio: {printable_ratio:.2f}")
        
        if printable_ratio < 0.7:  # Less than 70% printable characters
            print(f"ERROR: File has low printable character ratio ({printable_ratio:.2f}), may be binary: {file_path}")
            return False
            
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to validate file {file_path}: {e}")
        return False


def read_csv_with_encoding_detection(file_path, **kwargs):
    """
    Read CSV file with multiple encoding attempts and validation.
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file
    **kwargs : dict
        Additional arguments for pd.read_csv
        
    Returns:
    --------
    pd.DataFrame
        Loaded DataFrame
        
    Raises:
    -------
    RuntimeError
        If file cannot be read with any encoding
    """
    encodings_to_try = ['utf-8', 'latin-1', 'windows-1252', 'cp1252', 'iso-8859-1']
    
    print(f"\nAttempting to read CSV: {file_path}")
    
    for encoding in encodings_to_try:
        try:
            print(f"  Trying encoding: {encoding}")
            df = pd.read_csv(file_path, encoding=encoding, **kwargs)
            print(f"  SUCCESS: Loaded with {encoding} encoding")
            print(f"  Shape: {df.shape}")
            if len(df.columns) > 0:
                print(f"  Columns: {df.columns.tolist()}")
            if len(df) > 0:
                print(f"  First row sample: {df.iloc[0].to_dict()}")
            return df
            
        except UnicodeDecodeError as e:
            print(f"  Failed with {encoding}: {e}")
            continue
        except Exception as e:
            print(f"  Failed with {encoding} (other error): {e}")
            continue
    
    # If we get here, all encodings failed
    raise RuntimeError(f"Failed to read {file_path} with any of the attempted encodings: {encodings_to_try}")


def validate_csv_structure(df, file_path, expected_columns=None):
    """
    Validate CSV structure and detect corrupted data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    file_path : str
        Original file path for error messages
    expected_columns : list, optional
        List of expected column names
        
    Raises:
    -------
    ValueError
        If CSV structure is invalid or corrupted
    """
    print(f"\nValidating CSV structure for: {file_path}")
    
    if df is None or len(df) == 0:
        raise ValueError(f"CSV file is empty or could not be loaded: {file_path}")
    
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Data types: {df.dtypes.to_dict()}")
    
    # Check for expected columns if provided
    if expected_columns:
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            print(f"  WARNING: Missing expected columns {missing_columns} in {file_path}")
    
    # Check for obvious corruption indicators
    for col in df.columns:
        if df[col].dtype == 'object':
            # Look for binary garbage in string columns
            sample_values = df[col].dropna().head(5).astype(str)
            for idx, val in sample_values.items():
                # Check for non-printable characters (excluding common whitespace)
                non_printable_count = sum(1 for c in str(val) if ord(c) < 32 and c not in '\t\n\r')
                if non_printable_count > 0:
                    raise ValueError(f"Corrupted data detected in {file_path}, column '{col}', row {idx}: "
                                   f"contains {non_printable_count} non-printable characters. "
                                   f"Sample value: {repr(val)}")
    
    print(f"  CSV structure validation passed for: {file_path}")


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
    """
    Read CSV files with comprehensive validation and error handling.
    
    Parameters:
    -----------
    beh_annotation_path, skeleton_spline_path, worm_pos_path,
    spline_X_path, spline_Y_path, turn_annotation_path : str
        Paths to the respective CSV files
        
    Returns:
    --------
    tuple of pd.DataFrame
        Loaded and validated DataFrames
        
    Raises:
    -------
    RuntimeError
        If ANY file cannot be read or validated properly
    """
    print("\n" + "="*80)
    print("STARTING ROBUST CSV FILE LOADING WITH VALIDATION")
    print("="*80)
    
    # Define file mappings for better error reporting
    file_mapping = {
        'behavior_annotation': beh_annotation_path,
        'skeleton_spline': skeleton_spline_path, 
        'worm_position': worm_pos_path,
        'spline_X': spline_X_path,
        'spline_Y': spline_Y_path,
        'turn_annotation': turn_annotation_path
    }
    
    # Step 1: Validate all files exist and are readable
    print("\nStep 1: Validating file accessibility...")
    failed_files = []
    
    for file_type, file_path in file_mapping.items():
        print(f"\nValidating {file_type}: {file_path}")
        if not validate_file_readability(file_path):
            failed_files.append((file_type, file_path))
    
    if failed_files:
        error_msg = "CRITICAL ERROR: The following files failed validation:\n"
        for file_type, file_path in failed_files:
            error_msg += f"  - {file_type}: {file_path}\n"
        error_msg += "\nExecution stopped to prevent processing corrupted data."
        raise RuntimeError(error_msg)
    
    print("\n✓ All files passed initial validation")
    
    # Step 2: Read CSV files with encoding detection and validation
    print("\nStep 2: Loading CSV files with encoding detection...")
    dataframes = {}
    
    try:
        # Read behavior annotation file
        print(f"\n--- Loading behavior annotation ---")
        dataframes['beh_annotation'] = read_csv_with_encoding_detection(
            beh_annotation_path, header=None
        )
        validate_csv_structure(dataframes['beh_annotation'], beh_annotation_path)
        
        # Read skeleton spline file  
        print(f"\n--- Loading skeleton spline ---")
        dataframes['skeleton_spline'] = read_csv_with_encoding_detection(
            skeleton_spline_path, header=None
        )
        validate_csv_structure(dataframes['skeleton_spline'], skeleton_spline_path)
        
        # Read turn annotation file
        print(f"\n--- Loading turn annotation ---")
        dataframes['turn_annotation'] = read_csv_with_encoding_detection(
            turn_annotation_path
        )
        validate_csv_structure(dataframes['turn_annotation'], turn_annotation_path)
        
        # Read worm position file (most critical - often the problem file)
        print(f"\n--- Loading worm position (CRITICAL FILE) ---")
        dataframes['worm_pos'] = read_csv_with_encoding_detection(worm_pos_path)
        validate_csv_structure(
            dataframes['worm_pos'], worm_pos_path, 
            expected_columns=['X', 'Y']  # Common expected columns
        )
        
        # Read spline X file
        print(f"\n--- Loading spline X ---")
        dataframes['spline_X'] = read_csv_with_encoding_detection(
            spline_X_path, header=None
        )
        validate_csv_structure(dataframes['spline_X'], spline_X_path)
        
        # Read spline Y file
        print(f"\n--- Loading spline Y ---")
        dataframes['spline_Y'] = read_csv_with_encoding_detection(
            spline_Y_path, header=None
        )
        validate_csv_structure(dataframes['spline_Y'], spline_Y_path)
        
    except Exception as e:
        error_msg = f"CRITICAL ERROR during CSV loading: {e}\n"
        error_msg += "Execution stopped to prevent processing corrupted data."
        raise RuntimeError(error_msg) from e
    
    print("\n✓ All CSV files loaded successfully")
    
    # Step 3: Post-processing and data type conversion
    print("\nStep 3: Data cleaning and type conversion...")
    
    # Clean up worm position dataframe
    worm_pos_df = dataframes['worm_pos']
    worm_pos_df = worm_pos_df.drop(columns=['time'], errors='ignore')
    
    # Enforce integer type for 'frame' column if it exists
    if 'frame' in worm_pos_df.columns:
        worm_pos_df['frame'] = worm_pos_df['frame'].fillna(0).astype(int)
    
    # Show data before numeric conversion
    print("\n=== worm_pos_df BEFORE numeric conversion ===")
    print("Columns:", worm_pos_df.columns.tolist())
    print("Data types:", worm_pos_df.dtypes)
    if len(worm_pos_df) > 0:
        print("First few rows:")
        print(worm_pos_df.head())
        for col in ['X', 'Y']:
            if col in worm_pos_df.columns:
                print(f"Sample {col} values:", worm_pos_df[col].head(3).tolist())
    
    # Apply numeric conversions
    try:
        dataframes['beh_annotation'] = dataframes['beh_annotation'].apply(pd.to_numeric, errors='coerce')
        dataframes['turn_annotation'] = dataframes['turn_annotation'].apply(pd.to_numeric, errors='coerce')
        dataframes['skeleton_spline'] = dataframes['skeleton_spline'].apply(pd.to_numeric, errors='coerce')
        
        # Special handling for worm position coordinates
        coordinate_columns = ['X', 'Y']
        worm_pos_numeric_columns = [col for col in worm_pos_df.columns if col not in coordinate_columns]
        
        if worm_pos_numeric_columns:
            worm_pos_df[worm_pos_numeric_columns] = worm_pos_df[worm_pos_numeric_columns].apply(pd.to_numeric, errors='coerce')
        
        for col in coordinate_columns:
            if col in worm_pos_df.columns:
                worm_pos_df[col] = pd.to_numeric(worm_pos_df[col], errors='coerce')
        
        dataframes['spline_X'] = dataframes['spline_X'].apply(pd.to_numeric, errors='coerce')
        dataframes['spline_Y'] = dataframes['spline_Y'].apply(pd.to_numeric, errors='coerce')
        
    except Exception as e:
        raise RuntimeError(f"CRITICAL ERROR during numeric conversion: {e}") from e
    
    # Show results after numeric conversion
    print("\n=== worm_pos_df AFTER numeric conversion ===")
    print("Data types:", worm_pos_df.dtypes)
    if len(worm_pos_df) > 0:
        print("First few rows:")
        print(worm_pos_df.head())
        for col in ['X', 'Y']:
            if col in worm_pos_df.columns:
                print(f"Sample {col} values after conversion:", worm_pos_df[col].head(3).tolist())
                nan_count = worm_pos_df[col].isna().sum()
                print(f"{col} column NaN count: {nan_count}")
                if nan_count > 0:
                    print(f"WARNING: {col} column has {nan_count} NaN values after conversion!")
    
    # Step 4: Final validation and interpolation
    print("\nStep 4: Final validation and interpolation...")
    spline_X_df = dataframes['spline_X']
    
    print("DataFrame lengths:")
    for name, df in [('beh_annotation', dataframes['beh_annotation']),
                     ('turn_annotation', dataframes['turn_annotation']),
                     ('skeleton_spline', dataframes['skeleton_spline']),
                     ('worm_pos', worm_pos_df),
                     ('spline_X', spline_X_df),
                     ('spline_Y', dataframes['spline_Y'])]:
        print(f"  {name}: {len(df)} rows")
    
    # Interpolate worm position if needed
    print(f"\nStage Position length before interpolation: {len(worm_pos_df)}")
    if len(worm_pos_df) != len(spline_X_df):
        print(f"Interpolating worm position from {len(worm_pos_df)} to {len(spline_X_df)} frames...")
        try:
            worm_pos_df = worm_pos_df.apply(lambda x: interpolate_df(x, len(spline_X_df)), axis=0)
            print(f"✓ Interpolation successful. New length: {len(worm_pos_df)}")
        except Exception as e:
            raise RuntimeError(f"CRITICAL ERROR during interpolation: {e}") from e
    
    print("\n" + "="*80)
    print("CSV LOADING COMPLETED SUCCESSFULLY")
    print("="*80)
    
    return (dataframes['beh_annotation'], dataframes['skeleton_spline'], worm_pos_df, 
            dataframes['spline_X'], dataframes['spline_Y'], dataframes['turn_annotation'])


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