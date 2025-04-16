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
    """
    odor_features_raw = behavior_df[behavior_params]
    odor_features = odor_features_raw.fillna(0).values
    neural_activity = neural_df.values
    n_behavior_params = len(behavior_params)

    print("🧠 Starting model setup")
    print(f"- Behavior matrix shape: {odor_features.shape}")
    print(f"- Neural activity shape: {neural_activity.shape}")
    print(f"- Parameters to model: {behavior_params}")

    with pm.Model() as model:
        param_vars = []
        param_names = []

        for param_name in behavior_params:
            clean_name = f"c_{param_name.replace('.', '_').replace('-', '_')}"
            param = pm.Normal(clean_name, mu=0, sigma=0.5, shape=(1, neural_activity.shape[1]))
            param_vars.append(param)
            param_names.append(clean_name)

        s = pm.HalfNormal('s', sigma=0.5, shape=(1, neural_activity.shape[1]))
        b = pm.Normal('b', mu=0, sigma=0.5, shape=(1, neural_activity.shape[1]))

        odor_encoding = sum(param * odor_features[:, i:i + 1] for i, param in enumerate(param_vars))

        if odor_features.shape[0] != neural_activity.shape[0]:
            raise ValueError("Mismatch in timepoints between odor features and neural activity")

        trimmed_size = odor_features.shape[0] - 1

        neural_activity_model = (
            (1 / (s + 1)) * odor_encoding[:trimmed_size] +
            (s / (s + 1)) * neural_activity[:trimmed_size, :] +
            b
        )

        pm.Normal('neural_activity',
                  mu=neural_activity_model,
                  sigma=1.0,
                  observed=neural_activity[1:trimmed_size + 1, :])

        print(f"📦 Sampling with {n_draws} draws and {n_tune} tuning steps...")
        idata = pm.sample(draws=n_draws,
                          tune=n_tune,
                          return_inferencedata=True,
                          progressbar=True,
                          init='adapt_diag',
                          target_accept=0.9)

    print("📊 Sampling complete. Extracting posterior...")
    posterior_samples = az.extract(idata, group="posterior")

    print("📐 Posterior sample shapes and dims:")
    for param in param_names + ['s', 'b']:
        arr = posterior_samples[param]
        print(f"  {param}: shape = {arr.shape}, dims = {arr.dims}")

    # === PREDICTION  ===
    predicted_df = pd.DataFrame(index=neural_df.index)
    print(f"🔁 Predicting neural activity for {neural_activity.shape[1]} neurons...")

    for neuron_idx in range(neural_activity.shape[1]):
        neuron_id = f"neuron_{neuron_idx + 1:03d}"

        param_values = [posterior_samples[p].values.mean(axis=0)[0, neuron_idx] for p in param_names]
        s_neuron = posterior_samples['s'].values.mean(axis=0)[0, neuron_idx]
        b_neuron = posterior_samples['b'].values.mean(axis=0)[0, neuron_idx]

        pred = np.zeros(neural_activity.shape[0])
        pred[0] = neural_activity[0, neuron_idx]

        for t in range(1, neural_activity.shape[0]):
            sensory_input = sum(param_values[i] * odor_features[t - 1, i] for i in range(n_behavior_params))
            pred[t] = (1 / (s_neuron + 1)) * sensory_input + (s_neuron / (s_neuron + 1)) * pred[t - 1] + b_neuron

        predicted_df[neuron_id] = pred

    print("📈 Prediction complete.")

    # === EXPORT SUMMARY STATISTICS ===
    print("📦 Computing parameter summaries...")
    summary_df = pd.DataFrame(index=[f"neuron_{i+1:03d}" for i in range(neural_activity.shape[1])])

    for clean_name in param_names:
        samples = posterior_samples[clean_name].values[:, 0, :]  # (n_samples, n_neurons)
        summary_df[f'{clean_name}_mean'] = samples.mean(axis=0)
        summary_df[f'{clean_name}_lower'] = np.percentile(samples, 2.5, axis=0)
        summary_df[f'{clean_name}_upper'] = np.percentile(samples, 97.5, axis=0)

    for name in ['s', 'b']:
        samples = posterior_samples[name].values[:, 0, :]
        summary_df[f'{name}_mean'] = samples.mean(axis=0)
        summary_df[f'{name}_lower'] = np.percentile(samples, 2.5, axis=0)
        summary_df[f'{name}_upper'] = np.percentile(samples, 97.5, axis=0)

    # === SAVE FULL POSTERIOR SAMPLES TO DISK ===
    output_dir = os.path.dirname(predicted_df.columns.name or "bayesian_modeling")
    samples_dir = os.path.join(output_dir, "posterior_samples")
    os.makedirs(samples_dir, exist_ok=True)

    print("💾 Saving full posterior samples...")
    for param in param_names + ['s', 'b']:
        samples = posterior_samples[param].values[:, 0, :]  # shape: (n_samples, n_neurons)
        save_path = os.path.join(samples_dir, f"{param}_posterior_samples.npy")
        np.save(save_path, samples)

    # === CORRELATIONS ===
    print("🔗 Computing correlations...")
    neuron_correlations = {}
    all_param_names = param_names + ['s', 'b']
    for neuron_idx in range(neural_activity.shape[1]):
        vals = {p: posterior_samples[p].values[:, 0, neuron_idx] for p in all_param_names}
        neuron_correlations[f"neuron_{neuron_idx + 1:03d}"] = pd.DataFrame(vals).corr()

    all_neurons_corr = sum(neuron_correlations.values()) / len(neuron_correlations)

    print("✅ Bayesian modeling complete.")
    return idata, summary_df, predicted_df, neuron_correlations, all_neurons_corr


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
