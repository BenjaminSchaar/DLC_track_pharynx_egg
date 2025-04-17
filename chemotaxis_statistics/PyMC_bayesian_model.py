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
import matplotlib.pyplot as plt


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

    Returns:
        idata: InferenceData object
        summary_df: DataFrame with parameter summaries per neuron
        predicted_df: DataFrame with predicted neural activity
        neuron_correlations: Dict of correlation matrices per neuron
        all_neurons_corr: Averaged correlation matrix across neurons
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
    for p in param_names + ["s", "b"]:
        arr = posterior_samples[p]
        print(f"  {p}: shape = {arr.shape}, dims = {arr.dims}")

    # === PREDICTION ===
    predicted_df = pd.DataFrame(index=neural_df.index)
    print(f"🔁 Predicting neural activity for {neural_activity.shape[1]} neurons...")
    for neuron_idx in range(neural_activity.shape[1]):
        neuron_id = f"neuron_{neuron_idx + 1:03d}"
        param_values = [posterior_samples[p].values[0, neuron_idx, :].mean() for p in param_names]
        s_neuron = posterior_samples["s"].values[0, neuron_idx, :].mean()
        b_neuron = posterior_samples["b"].values[0, neuron_idx, :].mean()

        pred = np.zeros(neural_activity.shape[0])
        pred[0] = neural_activity[0, neuron_idx]

        for t in range(1, neural_activity.shape[0]):
            sensory_input = sum(param_values[i] * odor_features[t - 1, i] for i in range(n_behavior_params))
            pred[t] = (1 / (s_neuron + 1)) * sensory_input + (s_neuron / (s_neuron + 1)) * pred[t - 1] + b_neuron

        predicted_df[neuron_id] = pred

    print("📈 Prediction complete.")
    print("📦 Computing parameter summaries...")

    summary_df = pd.DataFrame(index=[f"neuron_{i+1:03d}" for i in range(neural_activity.shape[1])])

    for clean_name in param_names:
        samples = posterior_samples[clean_name].values[0, :, :]  # (neurons, samples)
        summary_df[f'{clean_name}_mean'] = samples.mean(axis=1)
        summary_df[f'{clean_name}_lower'] = np.percentile(samples, 2.5, axis=1)
        summary_df[f'{clean_name}_upper'] = np.percentile(samples, 97.5, axis=1)

    for name in ['s', 'b']:
        samples = posterior_samples[name].values[0, :, :]
        summary_df[f'{name}_mean'] = samples.mean(axis=1)
        summary_df[f'{name}_lower'] = np.percentile(samples, 2.5, axis=1)
        summary_df[f'{name}_upper'] = np.percentile(samples, 97.5, axis=1)

    # === SAVE FULL POSTERIOR SAMPLES TO DISK ===
    output_dir = os.path.dirname(predicted_df.columns.name or "bayesian_modeling")
    samples_dir = os.path.join(output_dir, "posterior_samples")
    os.makedirs(samples_dir, exist_ok=True)

    for param in param_names + ['s', 'b']:
        samples = posterior_samples[param].values[0, :, :]  # (neurons, samples)
        save_path = os.path.join(samples_dir, f"{param}_posterior_samples.npy")
        np.save(save_path, samples)

    # === CORRELATIONS ===
    neuron_correlations = {}
    all_param_names = param_names + ['s', 'b']
    for neuron_idx in range(neural_activity.shape[1]):
        vals = {p: posterior_samples[p].values[0, neuron_idx, :] for p in all_param_names}
        neuron_correlations[f"neuron_{neuron_idx + 1:03d}"] = pd.DataFrame(vals).corr()

    all_neurons_corr = sum(neuron_correlations.values()) / len(neuron_correlations)

    return idata, summary_df, predicted_df, neuron_correlations, all_neurons_corr

import matplotlib.pyplot as plt

def plot_behavior_encoding(summary_df, output_dir=".", figsize=(10, 8)):
    """
    Create a dot heatmap showing how strongly each neuron encodes each behavior parameter.
    Dot size reflects gain control parameter `s`.

    Args:
        summary_df: DataFrame containing parameter summaries per neuron
        output_dir: Directory to save the plot
        figsize: Size of the figure
    """
    # Extract columns
    param_cols = [col for col in summary_df.columns if col.endswith("_mean") and col.startswith("c_")]
    encoding_matrix = summary_df[param_cols].copy()
    s_vals = summary_df["s_mean"].values

    # Sort neurons by encoding strength
    encoding_strength = encoding_matrix.abs().sum(axis=1)
    sorted_idx = np.argsort(-encoding_strength)
    encoding_matrix = encoding_matrix.iloc[sorted_idx]
    s_vals_sorted = s_vals[sorted_idx]

    # Normalize `s` for dot size scaling
    s_norm = (s_vals_sorted - s_vals_sorted.min()) / (s_vals_sorted.max() - s_vals_sorted.min() + 1e-6)
    s_scaled = 20 + 180 * s_norm  # dot size from 20 to 200

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    x, y, sizes, colors = [], [], [], []

    for i, param in enumerate(encoding_matrix.columns):
        for j, neuron in enumerate(encoding_matrix.index):
            x.append(i)
            y.append(j)
            sizes.append(s_scaled[j])
            colors.append(encoding_matrix.iloc[j, i])

    scatter = ax.scatter(x, y, s=sizes, c=colors, cmap="YlGnBu", edgecolor="k")

    # Axis labels
    ax.set_xticks(range(len(param_cols)))
    ax.set_xticklabels([p.replace("c_", "") for p in param_cols], rotation=45, ha="right")
    ax.set_yticks(range(len(encoding_matrix)))
    ax.set_yticklabels(encoding_matrix.index)

    ax.set_xlabel("Behavior Parameters")
    ax.set_ylabel("Neurons (sorted by encoding strength)")
    ax.set_title("Behavior Encoding Strength per Neuron")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Encoding Strength (mean)")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "encoding_heatmap.png")
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Saved encoding heatmap to {plot_path}")
    plt.close()

def plot_neuron_predictions(traces_df, predicted_df, summary_df, output_dir, behavior_params):
    """
    Generate a PDF with per-neuron plots showing:
    - Original vs predicted activity
    - R2 score
    - Mini matrix of behavior encoding strengths (heatmap-style)
    - Title per neuron, sorted by prediction strength (R²)
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from sklearn.metrics import r2_score
    import numpy as np

    pdf_path = os.path.join(output_dir, "neuron_prediction_plots.pdf")

    # Compute R^2 scores for sorting
    r2_scores = []
    for i, neuron in enumerate(traces_df.columns):
        y_true = traces_df[neuron].values
        y_pred = predicted_df[f"neuron_{i+1:03d}"].values
        r2_scores.append((neuron, i, r2_score(y_true, y_pred)))

    # Sort neurons by descending R^2
    r2_scores.sort(key=lambda x: -x[2])

    with PdfPages(pdf_path) as pdf:
        for neuron, i, r2 in r2_scores:
            fig, ax = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={'height_ratios': [3, 1]})

            y_true = traces_df[neuron].values
            y_pred = predicted_df[f"neuron_{i+1:03d}"].values
            time = predicted_df['time_sec'].values

            # === Top Plot: Traces ===
            ax[0].plot(time, y_true, label="Observed", alpha=0.7)
            ax[0].plot(time, y_pred, label="Predicted", alpha=0.7)
            ax[0].set_title(f"{neuron} | $R^2$ = {r2:.3f}")
            ax[0].set_ylabel("Activity")
            ax[0].legend()

            # === Bottom Plot: Mini encoding matrix (1-row heatmap) ===
            weights = np.array([summary_df.loc[f"neuron_{i+1:03d}", f"c_{param}_mean"] for param in behavior_params])
            im = ax[1].imshow(weights[np.newaxis, :], cmap="YlGnBu", aspect="auto", vmin=-1, vmax=1)

            ax[1].set_yticks([])
            ax[1].set_xticks(range(len(behavior_params)))
            ax[1].set_xticklabels(behavior_params, rotation=45, ha="right")
            ax[1].set_title("Behavior Encoding Strength")

            plt.colorbar(im, ax=ax[1], orientation='horizontal', pad=0.2)
            fig.suptitle(f"Prediction Summary: {neuron}", fontsize=14, y=1.02)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"✅ Saved per-neuron prediction plots to {pdf_path}")


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

    #plotting part----------------

    plot_behavior_encoding(result_df, output_dir)

    plot_neuron_predictions(traces, predicted_df, result_df, output_dir, args.behavior_params)


if __name__ == "__main__":
    try:
        print("Shell commands passed:", sys.argv)
        main(sys.argv[1:])  # exclude the script name from the args when called from shell
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
