# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a Python-based analysis pipeline for high-resolution chemotaxis analysis of C. elegans worms using DeepLabCut (DLC) tracking data. The system processes worm movement trajectories, performs coordinate transformations, and calculates various behavioral parameters like bearing angles, speeds, and chemotaxis indices.

## Main Analysis Pipeline

The primary entry points are the `initialize_load_files_high_res_universal*.py` scripts, which orchestrate the complete analysis workflow:

1. **initialize_load_files_high_res_universal_bootstrapping.py** - Main script with bootstrap analysis capabilities
2. **initialize_load_files_high_res_universal.py** - Core analysis pipeline without bootstrapping  
3. **initialize_load_files_high_res_universal_cleaned.py** - Cleaned version of the core pipeline

## Core Architecture

### Source Modules (`src/`)

- **calculations.py** - Core mathematical functions for trajectory analysis:
  - `interpolate_df()` - Array interpolation
  - `correct_stage_pos_with_skeleton()` - Position correction using skeleton data
  - `calculate_distance()`, `calculate_bearing_angle()`, `calculate_speed()` etc.
  - `calculate_all_odor_parameters()` - Comprehensive parameter calculation

- **coordinate_system.py** - `CoordinateSystem` class handling different recording types:
  - `transform_coordinates()` - Coordinate transformation based on recording type ('crop', 'zim01', 'zim06')
  - Handles pixel-to-millimeter conversion and coordinate rotation

- **bootstrapping.py** - Statistical analysis functions:
  - `generate_random_odor_position()` - Creates control conditions for bootstrap analysis
  - Bootstrap sampling for significance testing

- **data_smothing.py** - Data preprocessing:
  - `replace_outliers_with_nan()` - Outlier detection and removal
  - `apply_smoothing()` - Signal smoothing functions
  - `smooth_trajectory_savitzky_golay_filter()` - Savitzky-Golay filtering

- **plotting_visualisation.py** - Visualization and output:
  - `create_combined_visualization()` - Multi-panel PDF reports
  - `plot_chemotaxis_overview()` - Trajectory plots
  - `create_worm_animation()` - Video generation
  - `save_chemotaxis_analysis_h5()` - HDF5 data export

## Key Dependencies

The project relies on scientific Python stack:
- pandas, numpy - Data manipulation
- matplotlib, seaborn, plotly - Visualization  
- scipy - Statistical functions and filtering
- cv2 (OpenCV) - Video processing
- h5py - HDF5 file handling
- yaml - Configuration files

## Data Flow

1. **Input**: DeepLabCut tracking data (CSV files with skeleton coordinates)
2. **Coordinate Transformation**: Convert pixel coordinates to arena coordinates
3. **Parameter Calculation**: Compute speeds, angles, distances, and behavioral metrics
4. **Smoothing/Filtering**: Apply outlier removal and trajectory smoothing
5. **Analysis**: Calculate chemotaxis indices and statistical measures
6. **Output**: Generate visualizations, animations, and HDF5 data files

## Analysis Parameters

The system calculates comprehensive behavioral parameters:
- **Position**: Corrected coordinates using skeleton data
- **Movement**: Speed, displacement vectors, curving angles
- **Chemotaxis**: Bearing angles, radial speeds, distance to odor
- **Temporal**: Time-based analysis with frame-to-second conversion

## Configuration

- Uses YAML configuration files for experiment parameters
- Command-line argument parsing for batch processing
- Flexible coordinate system handling for different recording setups

## Mathematical Foundation

Detailed mathematical documentation is provided in `chemotaxis_calculations.md`, covering:
- Interpolation algorithms
- Coordinate transformations
- Vector calculations for movement analysis
- Statistical measures for chemotaxis quantification

## Development Notes

- Python 3.12+ compatible
- Modular design with clear separation of concerns
- Extensive parameter validation and error handling
- Support for different experimental setups through coordinate system abstraction