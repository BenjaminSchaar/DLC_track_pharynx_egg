import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import random

from chemotaxis_analysis_high_res.src.calculations import (
    calculate_all_odor_parameters,
)


def generate_random_odor_position(
        df_worm: pd.DataFrame,
        arena_min_x: float,
        arena_max_x: float,
        arena_min_y: float,
        arena_max_y: float,
        variation_percent: float = 0.05
) -> Tuple[float, float]:
    """
    Generate a semi-random odor position along the worm's trajectory with ±5% variation.

    Parameters:
    -----------
    df_worm : pd.DataFrame
        DataFrame containing worm trajectory data
    arena_min_x, arena_max_x : float
        Arena X boundaries
    arena_min_y, arena_max_y : float
        Arena Y boundaries
    variation_percent : float
        Percentage of arena size to use for random variation (default 0.05 = 5%)

    Returns:
    --------
    tuple
        (x_odor, y_odor) coordinates
    """
    # Calculate arena dimensions for variation
    arena_width = arena_max_x - arena_min_x
    arena_height = arena_max_y - arena_min_y

    # Calculate variation range (±5% of arena size)
    x_variation = arena_width * variation_percent
    y_variation = arena_height * variation_percent

    # Randomly select a point from the worm's trajectory - FIXED COLUMN NAMES
    valid_indices = df_worm.dropna(subset=['X_rel_skel_pos_centroid', 'Y_rel_skel_pos_centroid']).index
    random_idx = random.choice(valid_indices)
    base_x = df_worm.loc[random_idx, 'X_rel_skel_pos_centroid']
    base_y = df_worm.loc[random_idx, 'Y_rel_skel_pos_centroid']

    # Add random variation
    x_offset = random.uniform(-x_variation, x_variation)
    y_offset = random.uniform(-y_variation, y_variation)

    x_odor = base_x + x_offset
    y_odor = base_y + y_offset

    # Ensure the position stays within arena boundaries
    x_odor = max(arena_min_x, min(arena_max_x, x_odor))
    y_odor = max(arena_min_y, min(arena_max_y, y_odor))

    return x_odor, y_odor


def recalculate_odor_dependent_parameters(
        df_worm_parameter: pd.DataFrame,
        x_odor_new: float,
        y_odor_new: float,
        conc_gradient_array: np.ndarray,
        distance_array: np.ndarray,
        diffusion_time_offset: int,
        fps: float,
        center_point: int,
        columns_to_smooth: list,
        skel_pos_0: int = 0,
        dC_lookback_frames: int = 1
) -> Dict[str, np.ndarray]:
    # Import the smoothing and outlier replacement functions
    from chemotaxis_analysis_high_res.src.data_smothing import apply_smoothing, replace_outliers_with_nan

    # Use the unified function to calculate odor parameters
    df_temp = calculate_all_odor_parameters(
        df=df_worm_parameter.copy(),
        x_odor=x_odor_new,
        y_odor=y_odor_new,
        conc_gradient_array=conc_gradient_array,
        distance_array=distance_array,
        diffusion_time_offset=diffusion_time_offset,
        fps=fps,
        skel_pos_0=skel_pos_0,
        dC_lookback_frames=dC_lookback_frames,
        inplace=False
    )

    # Apply outlier replacement and smoothing using the exact same columns from main wrapper
    # Filter to only include odor-related columns that exist in this DataFrame
    odor_columns_to_smooth = [col for col in columns_to_smooth
                              if col in df_temp.columns and
                              col.startswith(
                                  ('radial_speed', 'bearing_angle', 'NI', 'distance_to_odor', 'conc_at', 'dC_'))]

    # Apply outlier replacement and smoothing if we have columns to process
    if odor_columns_to_smooth:
        df_temp = replace_outliers_with_nan(df_temp, odor_columns_to_smooth, threshold=2.576)
        df_temp = apply_smoothing(df_temp, odor_columns_to_smooth, fps, center_point)

    # Extract results into dictionary format
    result = {
        'odor_x': x_odor_new,
        'odor_y': y_odor_new
    }

    # Add all the calculated columns to result (both raw and smoothed)
    for col in df_temp.columns:
        if col.startswith(('distance_to_odor', 'conc_at', 'dC_', 'radial_speed', 'bearing_angle', 'NI')):
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
        fps: float,
        center_point: int,
        columns_to_smooth: list,
        n_iterations: int = 1000,
        random_seed: int = 42,
        skel_pos_0: int = 0,
        dC_lookback_frames: int = 1
) -> Dict:
    """
    Perform bootstrap analysis by generating random odor positions and recalculating parameters.
    Now includes smoothing of calculated parameters using the same column list as main analysis.

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
    fps : float
        Frames per second
    n_iterations : int
        Number of bootstrap iterations
    random_seed : int
        Random seed for reproducibility
    skel_pos_0 : int
        Skeleton position for nose
    dC_lookback_frames : int
        Number of frames to look back for dC calculations
    center_point : int
        Center point of skeleton for dynamic column naming
    columns_to_smooth : list
        List of columns to smooth (passed from main wrapper)

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
        # Generate random odor position along worm trajectory
        x_odor_rand, y_odor_rand = generate_random_odor_position(
            df_worm_parameter, arena_min_x, arena_max_x, arena_min_y, arena_max_y, 0.05
        )

        # Recalculate all odor-dependent parameters WITH smoothing using exact same column list
        iteration_results = recalculate_odor_dependent_parameters(
            df_worm_parameter=df_worm_parameter,
            x_odor_new=x_odor_rand,
            y_odor_new=y_odor_rand,
            conc_gradient_array=conc_gradient_array,
            distance_array=distance_array,
            diffusion_time_offset=diffusion_time_offset,
            fps=fps,
            skel_pos_0=skel_pos_0,
            dC_lookback_frames=dC_lookback_frames,
            center_point=center_point,
            columns_to_smooth=columns_to_smooth
        )

        # Store iteration results
        bootstrap_iterations.append(iteration_results)

        # Calculate summary statistics for this iteration
        summary_stats['iteration'].append(i)
        summary_stats['odor_x'].append(x_odor_rand)
        summary_stats['odor_y'].append(y_odor_rand)

        # Calculate means using raw (unsmoothed) values for summary stats
        summary_stats['mean_NI'].append(np.nanmean(iteration_results['NI']) if 'NI' in iteration_results else np.nan)
        summary_stats['mean_radial_speed'].append(
            np.nanmean(iteration_results['radial_speed']) if 'radial_speed' in iteration_results else np.nan)
        summary_stats['mean_bearing_angle'].append(
            np.nanmean(iteration_results['bearing_angle']) if 'bearing_angle' in iteration_results else np.nan)
        summary_stats['mean_distance_to_odor'].append(np.nanmean(iteration_results[
                                                                     'distance_to_odor_centroid']) if 'distance_to_odor_centroid' in iteration_results else np.nan)
        summary_stats['mean_conc_at_centroid'].append(
            np.nanmean(iteration_results['conc_at_centroid']) if 'conc_at_centroid' in iteration_results else np.nan)

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