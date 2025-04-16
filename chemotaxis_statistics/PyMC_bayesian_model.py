#!/usr/bin/env python3
# This script performs Bayesian modeling to analyze relationships between
# worm behavior (chemotaxis) and neural activity data

import argparse  # For command-line argument parsing
import os  # For file/directory operations
import sys  # For system-level operations like sys.exit()
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
from scipy import interpolate  # For interpolation methods
import pymc as pm  # For Bayesian statistical modeling
import arviz as az  # For analyzing and visualizing Bayesian models
import ast  # For parsing Python literals from strings (used for command line arguments)
import pickle  # For serializing and deserializing Python objects


def equalize_dataframes(behavior_df, neuron_df, method='linear'):
    """
    Ensure that behavior and neuron dataframes have the same number of rows by interpolating
    the smaller dataframe to match the size of the larger one.

    Args:
        behavior_df: DataFrame containing behavior measurements
        neuron_df: DataFrame containing neural activity recordings
        method: Interpolation method to use (default: 'linear')

    Returns:
        Tuple of (behavior_df, neuron_df) with equal lengths
    """
    behavior_len = len(behavior_df)
    neuron_len = len(neuron_df)

    print("Before interpolation:")
    print(f"Behavior DataFrame shape: {behavior_df.shape}")
    print(f"Neuron DataFrame shape: {neuron_df.shape}")

    # If behavior data has more rows than neuron data, interpolate neuron data
    if behavior_len > neuron_len:
        print(f"Interpolating neuron DataFrame from {neuron_len} to {behavior_len} rows")
        # Create normalized x-coordinates for original and target data
        x_original = np.linspace(0, 1, neuron_len)
        x_target = np.linspace(0, 1, behavior_len)
        interpolated_neuron_df = pd.DataFrame(columns=neuron_df.columns, index=range(behavior_len))

        # Interpolate each column separately
        for column in neuron_df.columns:
            # Create interpolation function, ignoring NaN values
            f = interpolate.interp1d(x_original[~neuron_df[column].isna()],
                                     neuron_df[column].dropna().values,
                                     kind=method,
                                     bounds_error=False,
                                     fill_value="extrapolate")
            # Apply interpolation function to get new values
            interpolated_neuron_df[column] = f(x_target)
        return behavior_df, interpolated_neuron_df

    # If neuron data has more rows than behavior data, interpolate behavior data
    elif neuron_len > behavior_len:
        print(f"Interpolating behavior DataFrame from {behavior_len} to {neuron_len} rows")
        x_original = np.linspace(0, 1, behavior_len)
        x_target = np.linspace(0, 1, neuron_len)
        interpolated_behavior_df = pd.DataFrame(columns=behavior_df.columns, index=range(neuron_len))

        # Interpolate each column separately
        for column in behavior_df.columns:
            f = interpolate.interp1d(x_original[~behavior_df[column].isna()],
                                     behavior_df[column].dropna().values,
                                     kind=method,
                                     bounds_error=False,
                                     fill_value="extrapolate")
            interpolated_behavior_df[column] = f(x_target)
        return interpolated_behavior_df, neuron_df

    # If already the same length, no interpolation needed
    else:
        print("DataFrames already have the same length. No interpolation needed.")
        return behavior_df, neuron_df


def run_bayesian_model(behavior_df, neural_df, behavior_params, n_draws=1000, n_tune=1000):
    """
    Run a Bayesian model to analyze the relationship between behavior parameters and neural activity.

    This function creates a hierarchical Bayesian model that relates behavioral parameters
    to neural activity time series, incorporating temporal dynamics.

    Args:
        behavior_df: DataFrame containing behavior measurements
        neural_df: DataFrame containing neural activity recordings
        behavior_params: List of behavior parameters to include in the model
        n_draws: Number of posterior samples to draw (default: 1000)
        n_tune: Number of tuning steps (default: 1000)

    Returns:
        Tuple containing:
        - idata: InferenceData object with the posterior samples
        - result_df: DataFrame with summary statistics of parameter estimates
        - predicted_df: DataFrame with predicted neural activity
        - neuron_correlations: Dict of correlation matrices for each neuron
        - all_neurons_corr: Average correlation matrix across all neurons
    """
    # Extract behavior features and fill missing values with zeros
    odor_features_raw = behavior_df[behavior_params]
    odor_features = odor_features_raw.fillna(0).values

    # Get neural activity as a numpy array
    neural_activity = neural_df.values

    n_behavior_params = len(behavior_params)

    # Define the Bayesian model
    with pm.Model() as model:
        # Create variables for each behavior parameter
        param_vars = []
        param_names = []
        for param_name in behavior_params:
            # Clean parameter names by replacing dots and dashes with underscores
            clean_name = f"c_{param_name.replace('.', '_').replace('-', '_')}"
            # Create a normal prior for each parameter-neuron pair
            # Shape is (1, n_neurons)
            param = pm.Normal(clean_name, mu=0, sigma=0.5, shape=(1, neural_activity.shape[1]))
            param_vars.append(param)
            param_names.append(clean_name)

        # Create priors for persistence (s) and baseline (b) parameters
        # s controls how much the previous neural state influences the current state
        s = pm.HalfNormal('s', sigma=0.5, shape=(1, neural_activity.shape[1]))
        # b is a baseline activity level
        b = pm.Normal('b', mu=0, sigma=0.5, shape=(1, neural_activity.shape[1]))

        # Calculate the impact of odor features on neural activity
        odor_encoding = 0
        for i, param in enumerate(param_vars):
            odor_encoding = odor_encoding + param * odor_features[:, i:i + 1]

        # We expect odor_features and neural_activity to have the same shape[0] due to interpolation
        # If they don't, this indicates a failure in the interpolation step
        if odor_features.shape[0] != neural_activity.shape[0]:
            raise ValueError(
                f"Interpolation failed: odor_features shape {odor_features.shape} does not match neural_activity shape {neural_activity.shape}")

        # Use full size minus 1 for time lagging (t vs t-1)
        trimmed_size = odor_features.shape[0] - 1

        # Model neural activity as a function of:
        # 1. Current sensory input (odor_encoding) with weight 1/(s+1)
        # 2. Previous neural state with weight s/(s+1)
        # 3. Baseline activity (b)
        neural_activity_model = ((1 / (s + 1)) * odor_encoding[:trimmed_size] +
                                 (s / (s + 1)) * neural_activity[:trimmed_size, :] +
                                 b)

        # Define the likelihood function - comparing model predictions to actual observations
        # Note that we're predicting neural_activity[1:] based on neural_activity[:-1]
        # This implements a first-order autoregressive model
        likelihood = pm.Normal('neural_activity', mu=neural_activity_model, sigma=1.0,
                               observed=neural_activity[1:trimmed_size + 1, :])

        # Sample from the posterior distribution
        print(f"Sampling with {n_draws} draws and {n_tune} tuning steps...")
        idata = pm.sample(draws=n_draws,  # Number of samples to draw
                          tune=n_tune,  # Number of tuning steps (discarded)
                          return_inferencedata=True,  # Return arviz InferenceData object
                          progressbar=True,
                          init='adapt_diag',  # Initialization method
                          target_accept=0.9)  # Target acceptance rate

    # Extract posterior samples for analysis
    posterior_samples = az.extract(idata, group="posterior")

    # Create a results dataframe with parameter estimates for each neuron
    result_df = pd.DataFrame()
    result_df['neuron_id'] = [f"neuron_{i + 1:03d}" for i in range(neural_activity.shape[1])]

    # Debug shapes to help diagnose any issues
    print("Shape debugging information:")
    print(f"result_df shape: {result_df.shape}")
    for param_name in param_names:
        print(f"Parameter {param_name} shape: {posterior_samples[param_name].mean(axis=0).shape}")

    # Add mean and credible intervals for each parameter
    for clean_name, original_name in zip(param_names, behavior_params):
        # Fix: Use [0, :] to get array for all neurons
        result_df[f'{clean_name}_mean'] = posterior_samples[clean_name].mean(axis=0)[0, :]
        result_df[f'{clean_name}_lower'] = np.percentile(posterior_samples[clean_name].values, 2.5, axis=0)[0, :]
        result_df[f'{clean_name}_upper'] = np.percentile(posterior_samples[clean_name].values, 97.5, axis=0)[0, :]

    # Add results for persistence parameter (s)
    result_df['s_mean'] = posterior_samples['s'].mean(axis=0)[0, :]
    result_df['s_lower'] = np.percentile(posterior_samples['s'].values, 2.5, axis=0)[0, :]
    result_df['s_upper'] = np.percentile(posterior_samples['s'].values, 97.5, axis=0)[0, :]

    # Add results for baseline parameter (b)
    result_df['b_mean'] = posterior_samples['b'].mean(axis=0)[0, :]
    result_df['b_lower'] = np.percentile(posterior_samples['b'].values, 2.5, axis=0)[0, :]
    result_df['b_upper'] = np.percentile(posterior_samples['b'].values, 97.5, axis=0)[0, :]

    # Calculate correlations between parameters for each neuron
    neuron_correlations = {}
    params = param_names + ['s', 'b']
    for neuron_idx in range(neural_activity.shape[1]):
        # Extract parameter values for this neuron
        neuron_params = {param: posterior_samples[param].values[:, 0, neuron_idx] for param in params}
        param_df = pd.DataFrame(neuron_params)
        # Calculate correlation matrix
        neuron_correlations[f"neuron_{neuron_idx + 1:03d}"] = param_df.corr()

    # Calculate average correlation matrix across all neurons
    all_neurons_corr = pd.DataFrame(0, index=params, columns=params)
    for corr_matrix in neuron_correlations.values():
        all_neurons_corr += corr_matrix
    all_neurons_corr /= len(neuron_correlations)

    # Generate predicted neural activity using the estimated parameters
    predicted_df = pd.DataFrame(index=neural_df.index)
    for neuron_idx in range(neural_activity.shape[1]):
        neuron_id = f"neuron_{neuron_idx + 1:03d}"
        # Get mean parameter values for this neuron
        param_values = [posterior_samples[param_name].mean(axis=0)[0, neuron_idx] for param_name in param_names]
        s_neuron = posterior_samples['s'].mean(axis=0)[0, neuron_idx]
        b_neuron = posterior_samples['b'].mean(axis=0)[0, neuron_idx]

        # Initialize predicted activity array
        predicted_activity = np.zeros(neural_activity.shape[0])
        # Use actual first value as starting point
        predicted_activity[0] = neural_activity[0, neuron_idx]

        # Generate predictions for each time point based on the model
        for t in range(1, neural_activity.shape[0]):
            sensory_input = 0
            # Calculate total sensory input at time t-1
            for i, param_value in enumerate(param_values):
                if t - 1 < odor_features.shape[0]:
                    sensory_input += param_value * odor_features[t - 1, i]

            # Apply the autoregressive model:
            # new_state = sensory_component + history_component + baseline
            predicted_activity[t] = ((1 / (s_neuron + 1)) * sensory_input +
                                     (s_neuron / (s_neuron + 1)) * predicted_activity[t - 1] +
                                     b_neuron)

        # Store predictions in the dataframe
        predicted_df[neuron_id] = predicted_activity

    return idata, result_df, predicted_df, neuron_correlations, all_neurons_corr


def main(arg_list=None):
    """
    Main function to parse arguments and run the Bayesian modeling pipeline.
    """
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Run Bayesian modeling on worm behavior and neural data')
    parser.add_argument('--chemotaxis_file', required=True, help='Path to chemotaxis CSV file')
    parser.add_argument('--traces_file', required=True, help='Path to neural traces HDF5 file')
    parser.add_argument('--fps', type=float, default=83.0, help='Frames per second of behavior data')
    parser.add_argument('--output', default='bayesian_modeling', help='Subfolder for output results')
    parser.add_argument('--behavior_params', type=ast.literal_eval, required=True,
                        help='List of behavior parameters (e.g., \'["param1", "param2"]\')')
    parser.add_argument('--draws', type=int, default=1000,
                        help='Number of posterior samples to draw (default: 1000)')
    parser.add_argument('--tune', type=int, default=1000,
                        help='Number of tuning steps (default: 1000)')
    parser.add_argument('--max_timepoints', type=int, default=None,
                        help='Maximum number of timepoints to use (for testing)')

    if arg_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(arg_list)

    # Load behavior data (chemotaxis parameters)
    # Header=[0,1] suggests a multi-index header with two rows
    chemotaxis_params = pd.read_csv(args.chemotaxis_file, header=[0, 1], index_col=0)
    behavior = chemotaxis_params.loc[:, 'chemotaxis_parameter']

    # Load neural traces from HDF5 file
    traces = pd.read_hdf(args.traces_file, key='df_with_missing')

    # Optionally limit the number of timepoints (for testing)
    if args.max_timepoints is not None:
        print(f"Using only {args.max_timepoints} timepoints (testing mode)")
        behavior = behavior.iloc[:args.max_timepoints]
        traces = traces.iloc[:args.max_timepoints]

    # Ensure behavior and neural data have the same number of time points
    behavior, traces = equalize_dataframes(behavior, traces, method='linear')

    # Create output directory
    output_dir = os.path.join(os.path.dirname(args.chemotaxis_file), args.output)
    os.makedirs(output_dir, exist_ok=True)

    # Run the Bayesian model with the specified sampling parameters
    idata, result_df, predicted_df, neuron_correlations, all_neurons_corr = run_bayesian_model(
        behavior, traces, args.behavior_params, n_draws=args.draws, n_tune=args.tune)

    # Save results to files

    # Summary statistics for each neuron's parameters
    result_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    # Predicted neural activity time series with time in seconds
    predicted_df['time_sec'] = np.arange(len(predicted_df)) / args.fps
    predicted_df.to_csv(os.path.join(output_dir, "predicted_timeseries.csv"), index=False)

    # Save the full posterior samples
    with open(os.path.join(output_dir, "trace.pkl"), 'wb') as f:
        pickle.dump(idata, f)

    # Save correlation matrices
    corr_dir = os.path.join(output_dir, "correlations")
    os.makedirs(corr_dir, exist_ok=True)

    # Individual correlation matrices for each neuron
    for neuron_id, corr_matrix in neuron_correlations.items():
        corr_matrix.to_csv(os.path.join(corr_dir, f"{neuron_id}.csv"))

    # Average correlation matrix across all neurons
    all_neurons_corr.to_csv(os.path.join(corr_dir, "all_neurons_avg.csv"))

    # Save metadata about the analysis for reproducibility
    metadata = {
        'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'behavior_parameters': args.behavior_params,
        'fps': args.fps,
        'draws': args.draws,
        'tune': args.tune,
        'chemotaxis_file': os.path.basename(args.chemotaxis_file),
        'traces_file': os.path.basename(args.traces_file),
        'num_neurons': traces.shape[1],
        'num_timepoints': traces.shape[0]
    }
    pd.DataFrame([metadata]).to_csv(os.path.join(output_dir, "metadata.csv"), index=False)


if __name__ == "__main__":
    try:
        print("Shell commands passed:", sys.argv)
        main(sys.argv[1:])  # exclude the script name from the args when called from shell
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)