import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import random


from chemotaxis_analysis_high_res.src.calculations import (
   calculate_all_odor_parameters,
)


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
       fps: float,
       skel_pos_0: int = 0,
       dC_lookback_frames: int = 1
) -> Dict[str, np.ndarray]:
   # Use the unified function
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

   # Extract results into dictionary format
   result = {
       'odor_x': x_odor_new,
       'odor_y': y_odor_new
   }

   # Add all the calculated columns to result
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
       n_iterations: int = 1000,
       random_seed: int = 42,
       skel_pos_0: int = 0,
       border_buffer: float = 2.0,
       dC_lookback_frames: int = 1
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
   dC_lookback_frames : int
       Number of frames to look back for dC calculations

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
           fps=fps,
           skel_pos_0=skel_pos_0,
           dC_lookback_frames=dC_lookback_frames
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