import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.spatial import distance

def read_csv_files(beh_annotation_path, skeleton_spline_path, worm_pos_path):
    # Check if the file paths exist
    if not os.path.exists(beh_annotation_path):
        raise FileNotFoundError(f"The file '{beh_annotation_path}' does not exist.")
    if not os.path.exists(skeleton_spline_path):
        raise FileNotFoundError(f"The file '{skeleton_spline_path}' does not exist.")
    if not os.path.exists(worm_pos_path):
        raise FileNotFoundError(f"The file '{worm_pos_path}' does not exist.")

    # Read CSV files into separate dataframes
    beh_annotation_df = pd.read_csv(beh_annotation_path, header=None)
    skeleton_spline_df = pd.read_csv(skeleton_spline_path, header=None)
    worm_pos_df = pd.read_csv(worm_pos_path)

    return beh_annotation_df, skeleton_spline_df, worm_pos_df



def plot_skeleton_spline(skeleton_spline, output_path):
    try:
        num_frames = len(skeleton_spline)
        num_lines = 4
        cut_frames = num_frames // num_lines
        fig, axs = plt.subplots(num_lines, 1, dpi=400, figsize=(10, 1 * num_lines))

        for i, ax in enumerate(axs):
            start_idx = i * cut_frames
            end_idx = start_idx + cut_frames
            ax.imshow(skeleton_spline.iloc[start_idx:end_idx].T, origin="upper", cmap='seismic', aspect=20, vmin=-0.06,
                      vmax=0.06)
            ax.set_xticks(np.linspace(0, cut_frames, 5))
            ax.set_xticklabels(np.linspace(start_idx, end_idx, 5).astype(int), fontsize=6)
            ax.set_yticklabels(ax.get_yticks(), fontsize=6)
            if i == num_lines - 1:
                ax.set_xlabel('Frame', fontsize=6)
            ax.set_ylabel('Body Part', fontsize=6)

        # Save the plot to a file
        plt.tight_layout()
        plot_name = 'kymogram.png'
        full_file_path = os.path.join(output_path, plot_name)
        plt.savefig(full_file_path)

        plt.clf()  # Clear the current figure after displaying the plot

    except Exception as e:
        print(f'Problem plotting the data: {e}')


def plot_ethogram(beh_annotation, output_path):
    try:
        num_frames = len(beh_annotation)
        num_lines = 4
        cut_frames = num_frames // num_lines
        fig, axs = plt.subplots(num_lines, 1, dpi=400, figsize=(10, 2 * num_lines))

        for i, ax in enumerate(axs):
            start_idx = i * cut_frames
            end_idx = start_idx + cut_frames
            ax.imshow(beh_annotation.iloc[start_idx:end_idx].T, origin="upper", cmap='seismic_r', aspect=20 * 100, vmin=-0.06,
                      vmax=0.06)
            ax.set_xticks(np.linspace(0, cut_frames, 5))
            ax.set_xticklabels(np.linspace(start_idx, end_idx, 5).astype(int))
            if i == num_lines - 1:
                ax.set_xlabel('Frame')
            ax.set_ylabel('Behavioral State')

        # Save the plot to a file
        plt.tight_layout()
        plot_name = 'ehtogram.png'
        full_file_path = os.path.join(output_path, plot_name)
        plt.savefig(full_file_path)

        plt.clf()  # Clear the current figure after displaying the plot

    except Exception as e:
        print(f'Problem plotting the ethogram: {e}')

def plot_worm_tracks(worm_pos, output_path, x_odor, y_odor):

    # Set arena boundaries
    arena_min_x = 0
    arena_max_x = -38
    arena_min_y = 0
    arena_max_y = 45
    # Set the figure size
    plt.figure(figsize=(12, 12))

    # Create a scatter plot for the tracks
    plt.scatter(worm_pos['X_rel'], worm_pos['Y_rel'], label='Tracks')

    # Plot the "odor" point
    plt.scatter(x_odor, y_odor, color='red', label='Odor Point', s=50)

    plt.xlim(arena_min_x, arena_max_x)
    plt.ylim(arena_min_y, arena_max_y)

    # Add grid lines
    plt.grid(True)

    # Add labels and legend
    plt.xlabel('X Relative')
    plt.ylabel('Y Relative')
    plt.title('Tracks and Odor Point')
    plt.legend()

    plot_name = 'worm_track_overview.png'
    full_file_path = os.path.join(output_path, plot_name)
    # Save the plot to a file (e.g., a PNG image)
    plt.savefig(full_file_path)

    plt.clf()  # Clear the current figure after displaying the plot


def plot_ED(worm_pos, output_path):

    # Plotting code
    plt.figure(figsize=(20, 6))
    plt.plot(worm_pos.index, worm_pos['distance'])

    # Set the y-axis limits
    plt.ylim(0, 40.5)

    # Label the axes
    plt.xlabel('Index')
    plt.ylabel('Distance (mm)')

    plot_name = 'distance_plot.png'
    full_file_path = os.path.join(output_path, plot_name)
    # Save the plot to a file (e.g., a PNG image)
    plt.savefig(full_file_path)

    plt.clf()  # Clear the current figure after displaying the plot

# Define a function to extract the x and y values from the yaml file
def extract_coords(coord_string):
    x, y = coord_string.split(',')
    x = float(x.strip().split('=')[1])
    y = float(y.strip().split('=')[1])
    return x, y

# Define a function to convert X and Y values to the absolute grid
def convert_coordinates(row, x_zero, y_zero):
    row["X_rel"] = row["X"] - x_zero
    row["Y_rel"] = row["Y"] - y_zero
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read CSV files and plot data')
    parser.add_argument('--beh_annotation', help='Full path to the behavior annotation CSV file', required=True)
    parser.add_argument('--skeleton_spline', help='Full path to the skeleton spline CSV file', required=True)
    parser.add_argument('--worm_pos', help='Full path to the worm pos text file', required=True)
    parser.add_argument('--stage_pos', help='Full path to the odor pos file', required=True)

    args = parser.parse_args()

    beh_annotation_path = args.beh_annotation
    skeleton_spline_path = args.skeleton_spline
    worm_pos_path = args.worm_pos
    stage_pos_path = args.stage_pos

    # Extracting the directory path and saving it to a new variable
    output_path = os.path.dirname(beh_annotation_path)

    beh_annotation, skeleton_spline, worm_pos = read_csv_files(beh_annotation_path, skeleton_spline_path, worm_pos_path)

    # Display the loaded dataframes
    print("Behavior Annotation DataFrame:")
    print(beh_annotation.head())

    print("\nSkeleton Spline DataFrame:")
    print(skeleton_spline.head())

    print("\nWorm Pos DataFrame:")
    print(worm_pos.head())

    #load config file for odor and arena positions

    # Load the YAML file
    with open(stage_pos_path, 'r') as config_file:
        stage_pos = yaml.safe_load(config_file)

    # Access the odor coordinates
    x_odor, y_odor = extract_coords(stage_pos['odor_pos'])
    x_zero, y_zero = extract_coords(stage_pos['top_left'])

    # adjust to relative grid via reference point
    x_odor = x_odor - x_zero
    y_odor = y_odor - y_zero

    print("Adjusted x_odor:")
    print(x_odor)
    print("Adjusted y_odor:")
    print(y_odor)

    # Apply the conversion function to each row
    worm_pos = worm_pos.apply(lambda row: convert_coordinates(row, x_zero, y_zero), axis=1)
    worm_pos['distance'] = worm_pos.apply(lambda row: distance.euclidean((row['X_rel'], row['Y_rel']), (x_odor, y_odor)),axis=1)

    print("\nWorm Pos DataFrame wit Distance:")
    print(worm_pos.head())

    plot_skeleton_spline(skeleton_spline, output_path)
    plot_ethogram(beh_annotation, output_path)
    plot_worm_tracks(worm_pos, output_path, x_odor, y_odor)
    plot_ED(worm_pos, output_path)
