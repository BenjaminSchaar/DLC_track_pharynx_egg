import cv2
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D
import plotly.express as px


def plot_chemotaxis_overview(df, output_path, x_odor, y_odor, arena_min_x, arena_max_x, arena_min_y, arena_max_y, center_point, fps, file_name):
    """
    Plot the tracks and odor point from a given DataFrame and save the plot as a PNG file.

    Parameters:
    - data: DataFrame containing the tracks and odor data.
    - x_odor: X-coordinate of the odor point.
    - y_odor: Y-coordinate of the odor point.
    - arena_min_x: Minimum X-coordinate of the arena.
    - arena_max_x: Maximum X-coordinate of the arena.
    - arena_min_y: Minimum Y-coordinate of the arena.
    - arena_max_y: Maximum Y-coordinate of the arena.
    - file_name: Name of the file to save the plot. Default is 'tracks_and_odor_point.png'.
    """
    fps = int(fps)

    # Combine the output path and file name
    full_path = os.path.join(output_path, file_name)

    plt.figure(figsize=(160, 160))

    # Create a scatter plot for the corrected tracks
    plt.scatter(df['X_rel_skel_pos_centroid'], df['Y_rel_skel_pos_centroid'], label='Tracks_centroid', s=1, c=(df['time_seconds'] / 60), cmap='plasma')
    plt.colorbar(label='Time(min)')

    # Create a scatter plot for the nose tracks
    plt.scatter(df['X_rel_skel_pos_0'], df['Y_rel_skel_pos_0'], label='Tracks_nose', s=1, c=(df['dC_0']))
    # Create a scatter plot for the nose tracks
    plt.scatter(df[f'X_rel_skel_pos_{center_point}'], df[f'Y_rel_skel_pos_{center_point}'], label='Tracks_center', s=1)

    # Filter tracks where dC/cT is positive
    positive_dC_cT = df[df['dC_centroid'] > (0.0000000000001)]
    plt.scatter(positive_dC_cT['X_rel_skel_pos_0'], positive_dC_cT['Y_rel_skel_pos_0'], color='yellow', label='dC/cT > 0', s=0.2, alpha=0.05)

    # Filter tracks where dC/cT is negative
    negative_dC_cT = df[df['dC_centroid'] < (0.0000000000001)]
    plt.scatter(negative_dC_cT['X_rel_skel_pos_0'], negative_dC_cT['Y_rel_skel_pos_0'], color='red', label='dC/cT < 0', s=0.2, alpha=0.05)

    # Plot the "odor" point
    plt.scatter(x_odor, y_odor, color='red', label='Odor Point', s=1000)

    plt.xlim(arena_min_x, arena_max_x)
    plt.ylim(arena_min_y, arena_max_y)

    plt.legend()
    # Add grid lines
    plt.grid(True)

    # Add labels and legend
    plt.xlabel('X Relative')
    plt.ylabel('Y Relative')
    plt.title('Tracks and Odor Point')

    print("The full file path is:", full_path)
    # Save the plot
    plt.savefig(full_path)
    plt.close()  # Close the plot to free memory


def plot_time_series(df, column_names, fps, output_path, rows_per_plot=3, figsize=(15, 10),
                     dpi=100, line_color='blue', grid=True, save_suffix='time_series'):
    """
    Plot multiple columns of a dataframe as time series, converting frames to minutes.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data to plot
    column_names : list
        List of column names to plot as separate time series
    fps : float
        Frames per second, used to convert frame numbers to minutes
    output_path : str
        Directory path where to save the plots
    rows_per_plot : int, default=3
        Number of subplots per figure
    figsize : tuple, default=(15, 10)
        Figure size (width, height) in inches
    dpi : int, default=100
        Resolution of the figure
    line_color : str, default='blue'
        Color of the line plot
    grid : bool, default=True
        Whether to show grid on plots
    save_suffix : str, default='time_series'
        Suffix to add to saved filenames

    Returns:
    --------
    None : Saves plots to the specified output path
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import math

    # Create a copy of the dataframe to avoid modifying the original
    plot_df = df.copy()

    # Create a time column in minutes
    plot_df['time_minutes'] = plot_df.index / (fps * 60)

    # Calculate number of figures needed
    num_plots = len(column_names)
    num_figures = math.ceil(num_plots / rows_per_plot)

    for fig_num in range(num_figures):
        # Create a new figure
        fig, axes = plt.subplots(min(rows_per_plot, num_plots - fig_num * rows_per_plot),
                                 1, figsize=figsize, dpi=dpi, sharex=True)

        # Handle the case when there's only one subplot
        if rows_per_plot == 1 or num_plots - fig_num * rows_per_plot == 1:
            axes = [axes]

        # Plot each column in this figure
        for i, col_idx in enumerate(range(fig_num * rows_per_plot,
                                          min((fig_num + 1) * rows_per_plot, num_plots))):
            col = column_names[col_idx]

            # Create the plot
            axes[i].plot(plot_df['time_minutes'], plot_df[col], color=line_color)

            # Set labels and title
            axes[i].set_ylabel(col)
            axes[i].set_title(f'{col} vs Time')

            # Add grid if requested
            if grid:
                axes[i].grid(True, linestyle='--', alpha=0.7)

            # Add a horizontal line at y=0 for reference if appropriate
            if abs(plot_df[col].min()) < abs(plot_df[col].max() * 0.1):
                axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.3)

        # Set the x-label for the bottom subplot
        axes[-1].set_xlabel('Time (minutes)')

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        fig_path = os.path.join(output_path, f'{save_suffix}_fig{fig_num + 1}.png')
        plt.savefig(fig_path)
        plt.close()

        print(f"Saved figure {fig_num + 1}/{num_figures} to {fig_path}")

    print(f"Completed plotting {num_plots} time series across {num_figures} figures.")

def create_angle_animation(df, output_path, x_odor, y_odor, fps, file_name, nth_frame):
    '''
    Create and save an animation showing angles from a DataFrame using OpenCV.

    :param df: DataFrame containing the data points for the animation.
    :param output_path: The directory to save the output file.
    :param x_odor: X-coordinate for the odor/source location.
    :param y_odor: Y-coordinate for the odor/source location.
    :param fps: Frames per second for the output video.
    :param file_name: Name of the output file.
    :param nth_frame: Plot every nth frame (default is 1, which plots every frame).
    '''
    # Combine the output path and file name
    full_path = os.path.join(output_path, file_name)
    print("The full file path is:", full_path)

    # Define the video's width, height, and codec
    width, height = 600, 600
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 'XVID' can also be used

    # Create VideoWriter object
    out = cv2.VideoWriter(full_path, fourcc, fps, (width, height))

    # Create a figure for plotting
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)  # Adjust figsize if needed
    canvas = FigureCanvas(fig)

    for frame in range(0, len(df), nth_frame):
        # Clear the previous frame
        ax.clear()

        # Extract data for current frame
        center_x_close = df.at[frame, 'X_rel_skel_pos_centroid']
        center_y_close = df.at[frame, 'Y_rel_skel_pos_centroid']

        # Time shift for past and future positions
        time = 1

        # Calculate shifted positions in the temporary DataFrame
        df['X_shifted_negative'] = df['X_rel_skel_pos_centroid'].shift(-time)
        df['Y_shifted_negative'] = df['Y_rel_skel_pos_centroid'].shift(-time)
        df['X_shifted_positive'] = df['X_rel_skel_pos_centroid'].shift(time)
        df['Y_shifted_positive'] = df['Y_rel_skel_pos_centroid'].shift(time)

        # Plot points and lines for the current frame
        current_point = ax.scatter(df.at[frame, 'X_rel'], df.at[frame, 'Y_rel'], color='blue', s=5, label='Current Position')
        past_point = ax.scatter(df.at[frame, 'X_shifted_negative'], df.at[frame, 'Y_shifted_negative'], color='lightblue', s=5, label='Past Position')
        future_point = ax.scatter(df.at[frame, 'X_shifted_positive'], df.at[frame, 'Y_shifted_positive'], color='purple', s=5, label='Future Position')
        odor_source = ax.scatter(x_odor, y_odor, color='red', s=5, label='Odor Source')
        ax.plot([x_odor, df.at[frame, 'X_rel']], [y_odor, df.at[frame, 'Y_rel']], color='red', linestyle='--', linewidth=0.8)
        ax.plot([df.at[frame, 'X_shifted_negative'], df.at[frame, 'X_rel']], [df.at[frame, 'Y_shifted_negative'], df.at[frame, 'Y_rel']], color='green', linestyle='--', linewidth=0.8)
        ax.plot([df.at[frame, 'X_shifted_positive'], df.at[frame, 'X_rel']], [df.at[frame, 'Y_shifted_positive'], df.at[frame, 'Y_rel']], color='blue', linestyle='--', linewidth=0.8)

        # Setting plot limits and labels
        ax.set_xlim(center_x_close - 5, center_x_close + 5)
        ax.set_ylim(center_y_close - 5, center_y_close + 5)
        ax.set_xlabel('Distance (mm)')
        ax.set_ylabel('Distance (mm)')
        ax.set_title(f'Bearing Angle Visualization for Frame {frame}')
        ax.grid(True)

        # Add legend
        ax.legend(loc='upper right', fontsize=8)

        # Text for angles
        bearing_angle_text = f'Bearing Angle: {df.at[frame, "bearing_angle"]:.2f}'
        curving_angle_text = f'Curving Angle: {df.at[frame, "curving_angle"]:.2f}'
        ax.text(0.05, 0.95, bearing_angle_text, transform=ax.transAxes, fontsize=10, color='black')
        ax.text(0.05, 0.90, curving_angle_text, transform=ax.transAxes, fontsize=10, color='black')

        # Convert the Matplotlib figure to an array
        canvas.draw()  # Draw the canvas
        frame_array = np.array(canvas.renderer.buffer_rgba())  # Get the RGBA buffer from the canvas
        frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGBA2BGR)  # Convert RGBA to BGR

        # Write the frame to the video
        out.write(frame_array)

    print("The full file path is:", full_path)
    # Release everything when job is finished
    out.release()
    plt.close(fig)  # Close the figure to free memorydef create_angle_animation(df, output_path, x_odor, y_odor, fps, file_name, nth_frame):
    '''
    Create and save an animation showing angles from a DataFrame using OpenCV.

    :param df: DataFrame containing the data points for the animation.
    :param output_path: The directory to save the output file.
    :param x_odor: X-coordinate for the odor/source location.
    :param y_odor: Y-coordinate for the odor/source location.
    :param fps: Frames per second for the output video.
    :param file_name: Name of the output file.
    :param nth_frame: Plot every nth frame (default is 1, which plots every frame).
    '''
    # Combine the output path and file name
    full_path = os.path.join(output_path, file_name)
    print("The full file path is:", full_path)

    # Define the video's width, height, and codec
    width, height = 600, 600
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 'XVID' can also be used

    # Create VideoWriter object
    out = cv2.VideoWriter(full_path, fourcc, fps, (width, height))

    # Create a figure for plotting
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)  # Adjust figsize if needed
    canvas = FigureCanvas(fig)

    for frame in range(0, len(df), nth_frame):
        # Clear the previous frame
        ax.clear()

        # Extract data for current frame
        center_x_close = df.at[frame, 'X_rel_skel_pos_centroid']
        center_y_close = df.at[frame, 'Y_rel_skel_pos_centroid']

        # Time shift for past and future positions
        time = 1

        # Calculate shifted positions in the temporary DataFrame
        df['X_shifted_negative'] = df['X_rel_skel_pos_centroid'].shift(-time)
        df['Y_shifted_negative'] = df['Y_rel_skel_pos_centroid'].shift(-time)
        df['X_shifted_positive'] = df['X_rel_skel_pos_centroid'].shift(time)
        df['Y_shifted_positive'] = df['Y_rel_skel_pos_centroid'].shift(time)

        # Plot points and lines for the current frame
        current_point = ax.scatter(df.at[frame, 'X_rel'], df.at[frame, 'Y_rel'], color='blue', s=5, label='Current Position')
        past_point = ax.scatter(df.at[frame, 'X_shifted_negative'], df.at[frame, 'Y_shifted_negative'], color='lightblue', s=5, label='Past Position')
        future_point = ax.scatter(df.at[frame, 'X_shifted_positive'], df.at[frame, 'Y_shifted_positive'], color='purple', s=5, label='Future Position')
        odor_source = ax.scatter(x_odor, y_odor, color='red', s=5, label='Odor Source')
        ax.plot([x_odor, df.at[frame, 'X_rel']], [y_odor, df.at[frame, 'Y_rel']], color='red', linestyle='--', linewidth=0.8)
        ax.plot([df.at[frame, 'X_shifted_negative'], df.at[frame, 'X_rel']], [df.at[frame, 'Y_shifted_negative'], df.at[frame, 'Y_rel']], color='green', linestyle='--', linewidth=0.8)
        ax.plot([df.at[frame, 'X_shifted_positive'], df.at[frame, 'X_rel']], [df.at[frame, 'Y_shifted_positive'], df.at[frame, 'Y_rel']], color='blue', linestyle='--', linewidth=0.8)

        # Setting plot limits and labels
        ax.set_xlim(center_x_close - 5, center_x_close + 5)
        ax.set_ylim(center_y_close - 5, center_y_close + 5)
        ax.set_xlabel('Distance (mm)')
        ax.set_ylabel('Distance (mm)')
        ax.set_title(f'Bearing Angle Visualization for Frame {frame}')
        ax.grid(True)

        # Add legend
        ax.legend(loc='upper right', fontsize=8)

        # Text for angles
        bearing_angle_text = f'Bearing Angle: {df.at[frame, "bearing_angle"]:.2f}'
        curving_angle_text = f'Curving Angle: {df.at[frame, "curving_angle"]:.2f}'
        ax.text(0.05, 0.95, bearing_angle_text, transform=ax.transAxes, fontsize=10, color='black')
        ax.text(0.05, 0.90, curving_angle_text, transform=ax.transAxes, fontsize=10, color='black')

        # Convert the Matplotlib figure to an array
        canvas.draw()  # Draw the canvas
        frame_array = np.array(canvas.renderer.buffer_rgba())  # Get the RGBA buffer from the canvas
        frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGBA2BGR)  # Convert RGBA to BGR

        # Write the frame to the video
        out.write(frame_array)

    print("The full file path is:", full_path)
    # Release everything when job is finished
    out.release()
    plt.close(fig)  # Close the figure to free memory


def plot_ethogram(beh_annotation, output_path, file_name):
    '''
    Inputs beh_annotation df and plots ethogram
    :param beh_annotation: DataFrame containing behavioral annotation data
    :param output_path: Path to save the output file
    :param file_name: Name of the output file
    :return: None
    '''
    try:
        num_frames = len(beh_annotation)
        num_lines = 4
        cut_frames = num_frames // num_lines
        fig, axs = plt.subplots(num_lines, 1, dpi=400, figsize=(10, 1 * num_lines))

        for i, ax in enumerate(axs):
            start_idx = i * cut_frames
            end_idx = start_idx + cut_frames
            im = ax.imshow(beh_annotation.iloc[start_idx:end_idx].T, origin="upper", cmap='seismic_r',
                           aspect='auto', vmin=-0.06, vmax=0.06, interpolation='nearest')
            ax.set_xticks(np.linspace(0, cut_frames, 5))
            ax.set_xticklabels(np.linspace(start_idx, end_idx, 5).astype(int))
            if i == num_lines - 1:
                ax.set_xlabel('Frame')

            # Remove y-axis labels
            ax.set_yticks([])

            # Remove white space between subplots
            ax.set_ylim(beh_annotation.shape[1] - 0.5, 0.5)

        # Adjust the layout to remove extra white space
        plt.tight_layout(h_pad=0, w_pad=0)

        # Save the plot to a file
        full_path = os.path.join(output_path, file_name)
        print("The full file path is:", full_path)
        plt.savefig(full_path, bbox_inches='tight', pad_inches=0.1)
        plt.show()
        plt.close(fig)  # Close the figure to free up memory

    except Exception as e:
        print(f'Problem plotting the ethogram: {e}')


def plot_ethogramm_simple(df, output_path, file_name):
    # Extract the behaviour_state column and convert it to a numpy array
    behavior_data = df['behaviour_state'].values

    # Reshape the data to a 2D array
    behavior_data_2d = behavior_data.reshape(-1, 1)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(16, 4))

    # Plot the data using imshow
    im = ax.imshow(behavior_data_2d.T, origin="upper", cmap='seismic_r',
                   aspect='auto', interpolation='nearest')

    # Remove y-axis ticks
    ax.set_yticks([])

    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Behavior State')
    ax.set_title('Worm Behavior State Over Time')

    # Add a colorbar
    plt.colorbar(im, label='Behavior State')

    # Adjust layout to make room for the text
    plt.tight_layout()

    # Save the figure
    full_path = os.path.join(output_path, file_name)
    print("The full file path is:", full_path)
    plt.savefig(full_path)

    # Show the plot
    plt.show()

    # Clear the current figure
    plt.clf()


def plot_ethogram_simple(df, output_path, file_name):
    # Extract the behaviour_state column and convert it to a numpy array
    behavior_data = df['behaviour_state'].values

    # Reshape the data to a 2D array
    behavior_data_2d = behavior_data.reshape(-1, 1)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(16, 4))

    # Plot the data using imshow
    im = ax.imshow(behavior_data_2d.T, origin="upper", cmap='seismic_r',
                   aspect='auto', interpolation='nearest')

    # Remove y-axis ticks
    ax.set_yticks([])

    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Behavior State')
    ax.set_title('Worm Behavior State Over Time')

    # Add a colorbar
    plt.colorbar(im, label='Behavior State')

    # Adjust layout to make room for the text
    plt.tight_layout()

    # Save the figure
    full_path = os.path.join(output_path, file_name)
    print("The full file path is:", full_path)
    plt.savefig(full_path)

    # Show the plot
    plt.show()

    # Clear the current figure
    plt.clf()

def plot_skeleton_spline(skeleton_spline, output_path, file_name):
    '''
    Inputs skelleton_spline_df and plots kymogramm

    :param skeleton_spline:
    :param output_path:
    :return:
    '''
    try:
        num_frames = len(skeleton_spline)

        # Dynamically adjust number of rows and plot stretch
        if num_frames < 10000:
            num_lines = 1
        elif num_frames < 20000:
            num_lines = 2
        elif num_frames < 30000:
            num_lines = 3
        else:
            num_lines = 4

        cut_frames = num_frames // num_lines
        fig, axs = plt.subplots(num_lines, 1, dpi=400, figsize=(10, 1 * num_lines))

        # Ensure axs is iterable by converting it to an array if it's not
        if num_lines == 1:
            axs = [axs]  # Make it a list if only one subplot

        for i, ax in enumerate(axs):
            start_idx = i * cut_frames
            end_idx = start_idx + cut_frames
            ax.imshow(skeleton_spline.iloc[start_idx:end_idx].T, origin="upper", cmap='seismic', aspect='auto', vmin=-0.06, vmax=0.06)
            ax.set_xticks(np.linspace(0, cut_frames, 5))
            ax.set_xticklabels(np.linspace(start_idx, end_idx, 5).astype(int), fontsize=6)
            ax.set_yticklabels(ax.get_yticks(), fontsize=6)
            if i == num_lines - 1:
                ax.set_xlabel('Frame', fontsize=6)
            ax.set_ylabel('Body Part', fontsize=6)

        # Save the plot to a file
        plt.tight_layout()

        full_path = os.path.join(output_path, file_name)
        print("The full file path is:", full_path)

        plt.savefig(full_path)

        plt.clf()  # Clear the current figure after displaying the plot

    except Exception as e:
        print(f'Problem plotting the data: {e}')


def create_worm_animation(df1, df2, output_path, x_odor, y_odor, fps, arena_min_x, arena_max_x, arena_min_y, arena_max_y, video_path, nth_frame, file_name):
    '''
    Create and save an animation showing angles from a DataFrame using OpenCV.

    :param df1: DataFrame containing the data points for the animation.
    :param df2: DataFrame containing additional data points (e.g., angles).
    :param output_path: The directory to save the output file.
    :param x_odor: X-coordinate for the odor/source location.
    :param y_odor: Y-coordinate for the odor/source location.
    :param fps: Frames per second for the output video.
    :param arena_min_x: Minimum x-coordinate for the arena.
    :param arena_max_x: Maximum x-coordinate for the arena.
    :param arena_min_y: Minimum y-coordinate for the arena.
    :param arena_max_y: Maximum y-coordinate for the arena.
    :param video_path: Path to the input video file.
    :param nth_frame: Plot every nth frame.
    :param file_name: Name of the output file.
    '''
    full_path = os.path.join(output_path, file_name)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    out = cv2.VideoWriter(full_path, fourcc, fps, (width, height))

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    canvas = FigureCanvas(fig)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % nth_frame == 0:
            if frame_count >= len(df1) or frame_count >= len(df2):
                print(f"Frame count {frame_count} is out of range for the dataframes.")
                break

            center_x_close = df1.at[frame_count, 'X_rel']
            center_y_close = df1.at[frame_count, 'Y_rel']

            ax.clear()

            for skel_number in range(101):
                if skel_number == 0:
                    ax.scatter(df1.at[frame_count, f'X_rel_skel_pos_{skel_number}'],
                               df1.at[frame_count, f'Y_rel_skel_pos_{skel_number}'],
                               s=10, c=df2.at[frame_count, 'dC_0'])
                elif skel_number == 100:
                    ax.scatter(df1.at[frame_count, 'X_rel_skel_pos_centroid'],
                               df1.at[frame_count, 'Y_rel_skel_pos_centroid'],
                               s=10, c=df2.at[frame_count, 'speed'])
                else:
                    ax.scatter(df1.at[frame_count, f'X_rel_skel_pos_{skel_number}'],
                               df1.at[frame_count, f'Y_rel_skel_pos_{skel_number}'],
                               s=10, c='blue')

            ax.scatter(x_odor, y_odor, color='red', label='Odor Point', s=100)

            ax.set_xlim(center_x_close - 0.7, center_x_close + 0.7)
            ax.set_ylim(center_y_close - 0.7, center_y_close + 0.7)
            ax.set_xlabel('Distance (mm)')
            ax.set_ylabel('Distance (mm)')
            ax.set_title(f'Simulated Worm at Frame: {frame_count}')
            ax.grid(True)

            ax.set_position([0, 0, 1, 1])
            ax.axis('off')
            ax.invert_xaxis()

            bearing_angle_text = f'Bearing Angle: {df2.at[frame_count, "bearing_angle"]:.2f}'
            curving_angle_text = f'Curving Angle: {df2.at[frame_count, "curving_angle"]:.2f}'
            ax.text(0.05, 0.95, bearing_angle_text, transform=ax.transAxes, fontsize=10, color='black')
            ax.text(0.05, 0.90, curving_angle_text, transform=ax.transAxes, fontsize=10, color='black')

            canvas.draw()
            frame_array = np.array(canvas.renderer.buffer_rgba())
            frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGBA2BGR)

            frame_array_resized = cv2.resize(frame_array, (width, height))

            overlay = cv2.addWeighted(frame, 0.7, frame_array_resized, 0.3, 0)

            out.write(overlay)

        frame_count += 1

    out.release()
    cap.release()
    plt.close(fig)
    print("The full file path is:", full_path)

def plot_angles_binned(df, x_col, y_col, output_path, num_bins=10, file_name='plot.png'):
    """
    Plots mean of the y_col binned according to x_col with SEM error bars and saves the plot to a file.
    All axes are set between -180 and +180 degrees.

    Parameters:
    - df : pandas.DataFrame
        DataFrame containing the data
    - x_col : str
        Column name to be used for binning
    - y_col : str
        Column name to be plotted as mean with error bars
    - num_bins : int
        Number of bins to divide the x_col data into
    - file_name : str
        Filename to save the plot. The extension determines the format (e.g., 'plot.png', 'plot.pdf')

    Returns:
    - None
    """
    # Step 1: Binning x_col and creating a new column with the midpoint value of each bin interval
    df[f'{x_col}_binned'] = pd.cut(df[x_col].fillna(0), bins=num_bins).apply(lambda x: x.mid).astype(float)

    # Step 2: Calculate mean, standard deviation and count of y_col for each bin
    grouped_data = df.groupby(f'{x_col}_binned')[y_col].agg(['mean', 'std', 'count']).reset_index()
    grouped_data['SEM'] = grouped_data['std'] / np.sqrt(grouped_data['count'])  # Calculating SEM
    grouped_data.columns = [f'{x_col}_binned', 'mean_value', 'std', 'count', 'SEM']

    # Step 3: Create a line plot with error bars for SEM
    fig = px.line(grouped_data, x=f'{x_col}_binned', y='mean_value',
                  title=f'Mean {y_col} at Binned {x_col}', error_y='SEM',
                  labels={'mean_value': f'Mean {y_col}', f'{x_col}_binned': f'{x_col} Binned'})

    # Adding markers to the line plot to indicate data points
    fig.add_scatter(x=grouped_data[f'{x_col}_binned'], y=grouped_data['mean_value'], mode='markers', error_y=dict(type='data', array=grouped_data['SEM']))

    # Update axes titles and ranges
    #fig.update_xaxes(title_text=f'{x_col} Binned', range=[-180, 180])
    #fig.update_yaxes(title_text=f'Mean {y_col}', range=[-180, 180])

    # Update layout to ensure axes are centered at 0
    fig.update_layout(
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')
    )

    full_path = os.path.join(output_path, file_name)
    print("The full file path is:", full_path)
    # Save the plot
    fig.write_image(full_path)

    return None


def plot_turns(df, output_path, file_name='plot.png'):
    """
    Function to plot 'turn' values over their respective indices from a DataFrame and optionally save the figure.

    Args:
    turns_df (DataFrame): A pandas DataFrame containing at least one column named 'turn'.
    save_path (str, optional): Path where the figure should be saved. If not specified, the figure is not saved.

    Returns:
    fig (plotly.graph_objects.Figure): The Plotly figure object for the plot.
    """

    # Convert time from seconds to minutes
    times = df['time_seconds'] / 60

    fig = px.scatter(df, x = times,  y='turn', title='Turns Over Index',
                     color='turn', symbol='turn',
                     category_orders={"turn": [1, 0, -1]},
                     color_discrete_map={1: 'green', 0: 'blue', -1: 'red'},
                     symbol_map={1: 'triangle-up', 0: 'circle', -1: 'triangle-down'}
                     )

    fig.update_layout(
        xaxis_title='Time(min)',
        yaxis_title='Turn Value',
        showlegend=False  # This removes the legend
    )


    full_path = os.path.join(output_path, file_name)
    print("The full file path is:", full_path)

    fig.write_image(full_path)
    print(f"Figure saved to {full_path}")

    return None

def plot_pumps(df, output_path, file_name):

    # Convert time from seconds to minutes
    times = df['time_seconds'] / 60
    # Create the line plot
    plt.figure(figsize=(30, 8))  # Optional: Adjust the figure size
    plt.plot(times, df['pumping_frequency'])

    # Optional: Add labels and a title
    plt.xlabel('Time')
    plt.ylabel('pumps/60 sec')
    plt.title('Frequency')

    # Enable grid
    plt.grid(True)

    #saving part

    full_path = os.path.join(output_path, file_name)
    print("The full file path is:", full_path)

    plt.savefig(full_path)

    plt.clf()  # Clear the current figure after displaying the plot


def plot_dynamic_binned(df, y_column, output_path, file_name, hue_column=None, bin_count=100):
    # Create a copy of the dataframe to avoid modifying the original
    df_plot = df.copy()

    # Define the figure size
    plt.figure(figsize=(16, 6))

    # Convert time from seconds to minutes
    times = df_plot['time_seconds'] / 60

    # Create bins for the time data
    bin_size = (times.max() - times.min()) / bin_count
    bins = np.arange(times.min(), times.max() + bin_size, bin_size)

    # Bin the data
    binned_data = df_plot.groupby(pd.cut(times, bins=bins))

    # Calculate mean time and y_column for each bin
    binned_times = binned_data['time_seconds'].mean() / 60  # Convert to minutes
    binned_y = binned_data[y_column].mean()

    # Creating the scatter plot
    if hue_column:
        binned_hue = binned_data[hue_column].mean()
        scatter = plt.scatter(binned_times, binned_y, c=binned_hue, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label=hue_column)
    else:
        scatter = plt.scatter(binned_times, binned_y, cmap='plasma', alpha=0.7)

    # Plotting the line
    plt.plot(binned_times, binned_y, alpha=0.5)

    # Set the title and labels
    plt.title(f'Binned {y_column} over Time')
    plt.xlabel('Time (minutes)')
    plt.ylabel(y_column)

    # Enable grid
    plt.grid(True)

    # Save the figure
    full_path = os.path.join(output_path, file_name)
    print("The full file path is:", full_path)
    plt.savefig(full_path)
    plt.clf()  # Clear the current figure

