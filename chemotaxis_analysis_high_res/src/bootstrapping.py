import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import random


def generate_random_odor_position(
        arena_min_x: float,
        arena_max_x: float,
        arena_min_y: float,
        arena_max_y: float,
        border_buffer: float = 10.0
) -> Tuple[float, float]:
    """
    Generate a random odor position within the arena boundaries.

    Parameters:
    -----------
    arena_min_x, arena_max_x : float
        Arena X boundaries
    arena_min_y, arena_max_y : float
        Arena Y boundaries
    border_buffer : float
        Distance from arena edges to avoid placing odor too close to borders

    Returns:
    --------
    tuple
        (x_odor, y_odor) coordinates
    """
    # Apply buffer to avoid edge effects
    min_x = arena_min_x + border_buffer
    max_x = arena_max_x - border_buffer
    min_y = arena_min_y + border_buffer
    max_y = arena_max_y - border_buffer

    # Generate random coordinates
    x_odor = random.uniform(min_x, max_x)
    y_odor = random.uniform(min_y, max_y)

    return x_odor, y_odor


def recalculate_odor_dependent_parameters(
        df_worm_parameter: pd.DataFrame,
        x_odor_new: float,
        y_odor_new: float,
        conc_gradient_array: np.ndarray,
        distance_array: np.ndarray,
        diffusion_time_offset: int,
        calculate_distance_func,
        calculate_preceived_conc_func,
        calculate_radial_speed_func,
        calculate_bearing_angle_func,
        fps: float,
        skel_pos_0: int = 0
) -> Dict[str, np.ndarray]:
    """
    Recalculate all odor-dependent parameters for a new odor position.

    Parameters:
    -----------
    df_worm_parameter : pd.DataFrame
        Main worm analysis DataFrame
    x_odor_new, y_odor_new : float
        New odor position coordinates
    conc_gradient_array : np.ndarray
        Concentration gradient array
    distance_array : np.ndarray
        Distance array for concentration calculations
    diffusion_time_offset : int
        Time offset for diffusion simulation
    calculate_distance_func : function
        Function to calculate distances
    calculate_preceived_conc_func : function
        Function to calculate perceived concentration
    calculate_radial_speed_func : function
        Function to calculate radial speed
    calculate_bearing_angle_func : function
        Function to calculate bearing angle
    fps : float
        Frames per second
    skel_pos_0 : int
        Skeleton position for nose (usually 0)

    Returns:
    --------
    dict
        Dictionary containing all recalculated odor-dependent parameters
    """

    # Create a copy of the dataframe to avoid modifying the original
    df_temp = df_worm_parameter.copy()

    # Store new odor position
    result = {
        'odor_x': x_odor_new,
        'odor_y': y_odor_new
    }

    # 1. RECALCULATE DISTANCES
    # Distance from stage position
    df_temp['distance_to_odor_stage'] = df_temp.apply(
        lambda row: calculate_distance_func(row, 'X_rel', 'Y_rel', x_odor_new, y_odor_new), axis=1
    ).astype(float)

    # Distance from centroid
    df_temp['distance_to_odor_centroid'] = df_temp.apply(
        lambda row: calculate_distance_func(row, 'X_rel_skel_pos_centroid', 'Y_rel_skel_pos_centroid', x_odor_new,
                                            y_odor_new), axis=1
    ).astype(float)

    # Distance from nose (skeleton position 0)
    df_temp[f'distance_to_odor_{skel_pos_0}'] = df_temp.apply(
        lambda row: calculate_distance_func(row, f'X_rel_skel_pos_{skel_pos_0}', f'Y_rel_skel_pos_{skel_pos_0}',
                                            x_odor_new, y_odor_new), axis=1
    ).astype(float)

    # Distance from DLC positions if available
    if 'X_rel_DLC_nose' in df_temp.columns:
        df_temp['distance_to_odor_DLC_nose'] = df_temp.apply(
            lambda row: calculate_distance_func(row, 'X_rel_DLC_nose', 'Y_rel_DLC_nose', x_odor_new, y_odor_new), axis=1
        ).astype(float)

    if 'X_rel_DLC_tail' in df_temp.columns:
        df_temp['distance_to_odor_DLC_tail'] = df_temp.apply(
            lambda row: calculate_distance_func(row, 'X_rel_DLC_tail', 'Y_rel_DLC_tail', x_odor_new, y_odor_new), axis=1
        ).astype(float)

    # 2. RECALCULATE CONCENTRATIONS
    # Concentration at centroid
    df_temp['conc_at_centroid'] = pd.to_numeric(df_temp.apply(
        lambda row: calculate_preceived_conc_func(
            row['distance_to_odor_centroid'], row['time_seconds'],
            conc_gradient_array, distance_array, diffusion_time_offset
        ), axis=1
    ), errors='coerce')

    # Concentration at nose
    df_temp[f'conc_at_{skel_pos_0}'] = pd.to_numeric(df_temp.apply(
        lambda row: calculate_preceived_conc_func(
            row[f'distance_to_odor_{skel_pos_0}'], row['time_seconds'],
            conc_gradient_array, distance_array, diffusion_time_offset
        ), axis=1
    ), errors='coerce')

    # Concentration at DLC positions if available
    if 'distance_to_odor_DLC_nose' in df_temp.columns:
        df_temp['conc_at_DLC_nose'] = pd.to_numeric(df_temp.apply(
            lambda row: calculate_preceived_conc_func(
                row['distance_to_odor_DLC_nose'], row['time_seconds'],
                conc_gradient_array, distance_array, diffusion_time_offset
            ), axis=1
        ), errors='coerce')

    if 'distance_to_odor_DLC_tail' in df_temp.columns:
        df_temp['conc_at_DLC_tail'] = pd.to_numeric(df_temp.apply(
            lambda row: calculate_preceived_conc_func(
                row['distance_to_odor_DLC_tail'], row['time_seconds'],
                conc_gradient_array, distance_array, diffusion_time_offset
            ), axis=1
        ), errors='coerce')

    # 3. RECALCULATE CONCENTRATION DERIVATIVES
    time_interval_dC_dT = int(fps)
    df_temp['dC_centroid'] = df_temp['conc_at_centroid'].diff(periods=time_interval_dC_dT).astype(float)
    df_temp[f'dC_{skel_pos_0}'] = df_temp[f'conc_at_{skel_pos_0}'].diff(periods=1).astype(float)

    if 'conc_at_DLC_nose' in df_temp.columns:
        df_temp['dC_DLC_nose'] = df_temp['conc_at_DLC_nose'].diff(periods=1).astype(float)

    if 'conc_at_DLC_tail' in df_temp.columns:
        df_temp['dC_DLC_tail'] = df_temp['conc_at_DLC_tail'].diff(periods=1).astype(float)

    # Concentration difference between nose and tail
    if 'conc_at_DLC_nose' in df_temp.columns and 'conc_at_DLC_tail' in df_temp.columns:
        df_temp['d_DLC_nose_tail_C'] = (
                df_temp['conc_at_DLC_nose'] - df_temp['conc_at_DLC_tail']
        ).astype(float)

    # 4. RECALCULATE NAVIGATION PARAMETERS
    # Update the DataFrame with new odor position for calculation functions
    df_temp['odor_x'] = x_odor_new
    df_temp['odor_y'] = y_odor_new

    # Radial speed
    df_temp = calculate_radial_speed_func(df_temp, fps)

    # Bearing angle
    df_temp = calculate_bearing_angle_func(df_temp)

    # Navigation Index (NI)
    if 'radial_speed' in df_temp.columns and 'speed_centroid' in df_temp.columns:
        df_temp['NI'] = (df_temp['radial_speed'] / df_temp['speed_centroid'])

    # 5. EXTRACT RESULTS
    # Distance columns
    distance_columns = [
        'distance_to_odor_stage', 'distance_to_odor_centroid', f'distance_to_odor_{skel_pos_0}'
    ]
    if 'distance_to_odor_DLC_nose' in df_temp.columns:
        distance_columns.append('distance_to_odor_DLC_nose')
    if 'distance_to_odor_DLC_tail' in df_temp.columns:
        distance_columns.append('distance_to_odor_DLC_tail')

    # Concentration columns
    concentration_columns = [
        'conc_at_centroid', f'conc_at_{skel_pos_0}', 'dC_centroid', f'dC_{skel_pos_0}'
    ]
    if 'conc_at_DLC_nose' in df_temp.columns:
        concentration_columns.extend(['conc_at_DLC_nose', 'dC_DLC_nose'])
    if 'conc_at_DLC_tail' in df_temp.columns:
        concentration_columns.extend(['conc_at_DLC_tail', 'dC_DLC_tail'])
    if 'd_DLC_nose_tail_C' in df_temp.columns:
        concentration_columns.append('d_DLC_nose_tail_C')

    # Navigation columns
    navigation_columns = ['radial_speed', 'bearing_angle', 'NI']

    # Store all calculated values
    for col in distance_columns + concentration_columns + navigation_columns:
        if col in df_temp.columns:
            result[col] = df_temp[col].values

    return result


def perform_bootstrap_analysis(
        df_worm_parameter: pd.DataFrame,
        arena_min_x: float,
        arena_max_x: float,
        arena_min_y: float,
        arena_max_y: float,
        conc_gradient_array: np.ndarray,
        distance_array: np.ndarray,
        diffusion_time_offset: int,
        calculate_distance_func,
        calculate_preceived_conc_func,
        calculate_radial_speed_func,
        calculate_bearing_angle_func,
        fps: float,
        n_iterations: int = 1000,
        random_seed: int = 42,
        skel_pos_0: int = 0,
        border_buffer: float = 2.0
) -> Dict:
    """
    Perform bootstrap analysis by generating random odor positions and recalculating parameters.

    Parameters:
    -----------
    df_worm_parameter : pd.DataFrame
        Main worm analysis DataFrame
    arena_min_x, arena_max_x, arena_min_y, arena_max_y : float
        Arena boundaries
    conc_gradient_array : np.ndarray
        Concentration gradient array
    distance_array : np.ndarray
        Distance array for concentration calculations
    diffusion_time_offset : int
        Time offset for diffusion simulation
    calculate_distance_func, calculate_preceived_conc_func, etc. : functions
        Analysis functions from your existing codebase
    fps : float
        Frames per second
    n_iterations : int
        Number of bootstrap iterations
    random_seed : int
        Random seed for reproducibility
    skel_pos_0 : int
        Skeleton position for nose
    border_buffer : float
        Buffer distance from arena edges

    Returns:
    --------
    dict
        Bootstrap results containing all iterations and summary statistics
    """

    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    print(f"Starting bootstrap analysis with {n_iterations} iterations...")

    # Store results for each iteration
    bootstrap_iterations = []

    # Store summary statistics
    summary_stats = {
        'iteration': [],
        'mean_NI': [],
        'mean_radial_speed': [],
        'mean_bearing_angle': [],
        'mean_distance_to_odor': [],
        'mean_conc_at_centroid': [],
        'odor_x': [],
        'odor_y': []
    }

    for i in range(n_iterations):
        # Generate random odor position
        x_odor_rand, y_odor_rand = generate_random_odor_position(
            arena_min_x, arena_max_x, arena_min_y, arena_max_y, border_buffer
        )

        # Recalculate all odor-dependent parameters
        iteration_results = recalculate_odor_dependent_parameters(
            df_worm_parameter=df_worm_parameter,
            x_odor_new=x_odor_rand,
            y_odor_new=y_odor_rand,
            conc_gradient_array=conc_gradient_array,
            distance_array=distance_array,
            diffusion_time_offset=diffusion_time_offset,
            calculate_distance_func=calculate_distance_func,
            calculate_preceived_conc_func=calculate_preceived_conc_func,
            calculate_radial_speed_func=calculate_radial_speed_func,
            calculate_bearing_angle_func=calculate_bearing_angle_func,
            fps=fps,
            skel_pos_0=skel_pos_0
        )

        # Store iteration results
        bootstrap_iterations.append(iteration_results)

        # Calculate summary statistics for this iteration
        summary_stats['iteration'].append(i)
        summary_stats['odor_x'].append(x_odor_rand)
        summary_stats['odor_y'].append(y_odor_rand)

        # Calculate means (handling NaN values)
        if 'NI' in iteration_results:
            summary_stats['mean_NI'].append(np.nanmean(iteration_results['NI']))
        else:
            summary_stats['mean_NI'].append(np.nan)

        if 'radial_speed' in iteration_results:
            summary_stats['mean_radial_speed'].append(np.nanmean(iteration_results['radial_speed']))
        else:
            summary_stats['mean_radial_speed'].append(np.nan)

        if 'bearing_angle' in iteration_results:
            summary_stats['mean_bearing_angle'].append(np.nanmean(iteration_results['bearing_angle']))
        else:
            summary_stats['mean_bearing_angle'].append(np.nan)

        if 'distance_to_odor_centroid' in iteration_results:
            summary_stats['mean_distance_to_odor'].append(np.nanmean(iteration_results['distance_to_odor_centroid']))
        else:
            summary_stats['mean_distance_to_odor'].append(np.nan)

        if 'conc_at_centroid' in iteration_results:
            summary_stats['mean_conc_at_centroid'].append(np.nanmean(iteration_results['conc_at_centroid']))
        else:
            summary_stats['mean_conc_at_centroid'].append(np.nan)

        # Progress update
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{n_iterations} bootstrap iterations")

    # Calculate original (real odor) summary statistics for comparison
    original_summary = {}
    if 'NI' in df_worm_parameter.columns:
        original_summary['mean_NI'] = np.nanmean(df_worm_parameter['NI'])
    if 'radial_speed' in df_worm_parameter.columns:
        original_summary['mean_radial_speed'] = np.nanmean(df_worm_parameter['radial_speed'])
    if 'bearing_angle' in df_worm_parameter.columns:
        original_summary['mean_bearing_angle'] = np.nanmean(df_worm_parameter['bearing_angle'])
    if 'distance_to_odor_centroid' in df_worm_parameter.columns:
        original_summary['mean_distance_to_odor'] = np.nanmean(df_worm_parameter['distance_to_odor_centroid'])
    if 'conc_at_centroid' in df_worm_parameter.columns:
        original_summary['mean_conc_at_centroid'] = np.nanmean(df_worm_parameter['conc_at_centroid'])

    print(f"Bootstrap analysis complete: {n_iterations} iterations generated")

    return {
        'iterations': bootstrap_iterations,
        'summary': summary_stats,
        'original_summary': original_summary,
        'n_iterations': n_iterations,
        'random_seed': random_seed
    }