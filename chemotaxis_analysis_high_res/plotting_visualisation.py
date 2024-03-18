import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
import pandas as pd


def plot_chemotaxis_overview(df, output_path, x_odor, y_odor, arena_min_x, arena_max_x, arena_min_y, arena_max_y, fps, file_name):
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
    print("The full file path is:", full_path)

    plt.figure(figsize=(160, 160))

    # Create a scatter plot for the corrected tracks
    plt.scatter(df['X_rel_skel_pos_centroid_corrected'], df['Y_rel_skel_pos_centroid_corrected'], label='Tracks_centroid', s=1, c=(df['time_seconds'] / 60), cmap='plasma')
    plt.colorbar(label='Time(min)')

    # Create a scatter plot for the nose tracks
    plt.scatter(df['X_rel_skel_pos_0'], df['Y_rel_skel_pos_0'], label='Tracks_nose', s=1, c=(df['dC_0']))

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

def create_angle_animation(df, output_path, x_odor, y_odor, fps, file_name):
    '''
    Create and save an animation showing angles from a DataFrame using OpenCV.

    :param df: DataFrame containing the data points for the animation.
    :param output_path: The directory to save the output file.
    :param x_odor: X-coordinate for the odor/source location.
    :param y_odor: Y-coordinate for the odor/source location.
    :param fps: Frames per second for the output video.
    :param file_name: Name of the output file.
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

    for frame in range(len(df)):
        # Clear the previous frame
        ax.clear()

        # Extract data for current frame
        center_x_close = df.at[frame, 'X_rel']
        center_y_close = df.at[frame, 'Y_rel']

        # Plot points and lines for the current frame
        ax.scatter(df.at[frame, 'X_rel'], df.at[frame, 'Y_rel'], color='blue', s=5)
        ax.scatter(df.at[frame, 'X_shifted_negative'], df.at[frame, 'Y_shifted_negative'], color='lightblue', s=5)
        ax.scatter(df.at[frame, 'X_shifted_positive'], df.at[frame, 'Y_shifted_positive'], color='purple', s=5)
        ax.scatter(x_odor, y_odor, color='red', s=5)
        ax.plot([x_odor, df.at[frame, 'X_rel']], [y_odor, df.at[frame, 'Y_rel']], color='red', linestyle='--', linewidth=0.8)
        ax.plot([df.at[frame, 'X_shifted_negative'], df.at[frame, 'X_rel']], [df.at[frame, 'Y_shifted_negative'], df.at[frame, 'Y_rel']], color='green', linestyle='--', linewidth=0.8)
        ax.plot([df.at[frame, 'X_shifted_positive'], df.at[frame, 'X_rel']], [df.at[frame, 'Y_shifted_positive'], df.at[frame, 'Y_rel']], color='blue', linestyle='--', linewidth=0.8)

        # Adding a legend
        plt.legend()

        # Setting plot limits and labels
        ax.set_xlim(center_x_close - 5, center_x_close + 5)
        ax.set_ylim(center_y_close - 5, center_y_close + 5)
        ax.set_xlabel('Distance (mm)')
        ax.set_ylabel('Distance (mm)')
        ax.set_title(f'Bearing Angle Visualization for Frame {frame}')
        ax.grid(True)


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
    Inputs beh_annotation df and plots erhtogramm

    :param beh_annotation:
    :param output_path:
    :return:
    '''
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
        full_path = os.path.join(output_path, file_name)
        print("The full file path is:", full_path)
        plt.savefig(full_path)

        plt.clf()  # Clear the current figure after displaying the plot

    except Exception as e:
        print(f'Problem plotting the ethogram: {e}')


def plot_skeleton_spline(skeleton_spline, output_path, file_name):
    '''
    Inputs skelleton_spline_df and plots kymogramm

    :param skeleton_spline:
    :param output_path:
    :return:
    '''
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

        full_path = os.path.join(output_path, file_name)
        print("The full file path is:", full_path)

        plt.savefig(full_path)

        plt.clf()  # Clear the current figure after displaying the plot

    except Exception as e:
        print(f'Problem plotting the data: {e}')


def plot_odor_concentration(df, output_path, file_name):
    # Define the figure size
    plt.figure(figsize=(16, 4))

    # Convert time from seconds to minutes
    times = df['time_seconds'] / 60

    # Creating the scatter plot with color based on time
    scatter = plt.scatter(times, df['conc_at_centroid'], c=times, cmap='plasma', alpha=0.1)

    # Plotting the line
    plt.plot(times, df['conc_at_centroid'], alpha=0.5)  # Set lower alpha to make line less prominent

    # Adding a color bar to understand the mapping from time to color
    plt.colorbar(scatter, label='Time (minutes)')

    # Set the title and labels
    plt.title('Experienced Odor Concentration')
    plt.xlabel('Time (minutes)')
    plt.ylabel('C (mol/l)')

    # Enable grid
    plt.grid(True)

    full_path = os.path.join(output_path, file_name)
    print("The full file path is:", full_path)

    plt.savefig(full_path)

    plt.clf()  # Clear the current figure after displaying the plot

def plot_speed(df, output_path, file_name):
    # Define the figure size
    plt.figure(figsize=(16, 4))

    # Convert time from seconds to minutes
    times = df['time_seconds'] / 60

    # Creating the scatter plot with color based on time
    scatter = plt.scatter(times, df['speed'], c=times, cmap='plasma', alpha=0.1)

    # Plotting the line
    plt.plot(times, df['speed'], alpha=0.5)  # Set lower alpha to make line less prominent

    # Adding a color bar to understand the mapping from time to color
    plt.colorbar(scatter, label='Time (minutes)')

    # Set the title and labels
    plt.title('Centroid Speed')
    plt.xlabel('Time (minutes)')
    plt.ylabel('mm/sec')

    # Enable grid
    plt.grid(True)

    full_path = os.path.join(output_path, file_name)
    print("The full file path is:", full_path)

    plt.savefig(full_path)

    plt.clf()  # Clear the current figure after displaying the plot


def plot_NI(df, output_path, file_name):

    # Define the figure size
    plt.figure(figsize=(16, 4))

    # Convert time from seconds to minutes
    times = df['time_seconds'] / 60

    # Creating the scatter plot with color based on time
    scatter = plt.scatter(times, df['NI'], c=times, cmap='plasma', alpha=0.1)

    # Plotting the line
    plt.plot(times, df['NI'], alpha=0.5)  # Set lower alpha to make line less prominent

    # Adding a color bar to understand the mapping from time to color
    plt.colorbar(scatter, label='Time (minutes)')

    # Set the title and labels
    plt.title('Centroid NI')
    plt.xlabel('Time (minutes)')
    plt.ylabel('NI')

    # Enable grid
    plt.grid(True)

    full_path = os.path.join(output_path, file_name)
    print("The full file path is:", full_path)

    plt.savefig(full_path)

    plt.clf()  # Clear the current figure after displaying the plot

def plot_distance_to_odor(df, output_path, file_name):
    # Define the figure size
    plt.figure(figsize=(16, 4))

    # Convert time from seconds to minutes
    times = df['time_seconds'] / 60

    # Creating the scatter plot with color based on time
    scatter = plt.scatter(times, df['distance_to_odor_centroid'], c=times, cmap='plasma', alpha=0.1)

    # Plotting the line
    plt.plot(times, df['distance_to_odor_centroid'], alpha=0.5)  # Set lower alpha to make line less prominent

    # Adding a color bar to understand the mapping from time to color
    plt.colorbar(scatter, label='Time (minutes)')

    # Set the title and labels
    plt.title('Centroid Distance to Odor')
    plt.xlabel('Time (minutes)')
    plt.ylabel('mm')

    # Enable grid
    plt.grid(True)

    full_path = os.path.join(output_path, file_name)
    print("The full file path is:", full_path)

    plt.savefig(full_path)

    plt.clf()  # Clear the current figure after displaying the plot

def plot_reversal_frequency(df, output_path, file_name):
    # Define the figure size
    plt.figure(figsize=(16, 4))

    # Convert time from seconds to minutes
    times = df['time_seconds'] / 60

    # Creating the scatter plot with color based on time
    scatter = plt.scatter(times, df['reversal_frequency'], c=times, cmap='plasma', alpha=0.1)

    # Plotting the line
    plt.plot(times, df['reversal_frequency'], alpha=0.5)  # Set lower alpha to make line less prominent

    # Adding a color bar to understand the mapping from time to color
    plt.colorbar(scatter, label='Time (minutes)')

    # Set the title and labels
    plt.title('Reversal Frequency')
    plt.xlabel('Time (minutes)')
    plt.ylabel('events/min')

    # Enable grid
    plt.grid(True)

    #saving part

    full_path = os.path.join(output_path, file_name)
    print("The full file path is:", full_path)

    plt.savefig(full_path)

    plt.clf()  # Clear the current figure after displaying the plot

def plot_curving_vs_bearing(df, output_path, file_name):

    # Define the figure size
    plt.figure(figsize=(16, 4))

    # Convert time from seconds to minutes
    times = df['time_seconds'] / 60

    # Creating the scatter plot with color based on time
    scatter = plt.scatter(df['bearing_angle'], df['curving_angle'], c=times, cmap='plasma', alpha=0.1)

    # Plotting the line
    plt.scatter(df['bearing_angle'], df['curving_angle'], alpha=0.5)  # Set lower alpha to make line less prominent

    # Adding a color bar to understand the mapping from time to color
    plt.colorbar(scatter, label='Time (minutes)')

    # Set the title and labels
    plt.title('Bearing Angle vs Curving Angle')
    plt.xlabel('bearing angle')
    plt.ylabel('curving angle')

    # Enable grid
    plt.grid(True)

    # saving part
    full_path = os.path.join(output_path, file_name)
    print("The full file path is:", full_path)

    plt.savefig(full_path)

    plt.clf()  # Clear the current figure after displaying the plot


def create_worm_animation(df1, df2, output_path, x_odor, y_odor, fps, arena_min_x, arena_max_x, arena_min_y, arena_max_y, file_name):
    '''
    Create and save an animation showing angles from a DataFrame using OpenCV.

    :param df1: DataFrame containing the data points for the animation.
    :param output_path: The directory to save the output file.
    :param x_odor: X-coordinate for the odor/source location.
    :param y_odor: Y-coordinate for the odor/source location.
    :param fps: Frames per second for the output video.
    :param file_name: Name of the output file.
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

    for frame in range(len(df1)):
        row_index = frame
        # Clear the previous frame
        ax.clear()

        for skel_pos in range(101):  # Iterate over all skeleton positions
            if skel_pos == 0:
                ax.scatter(df1[f'X_rel_skel_pos_{skel_pos}'].iloc[row_index],
                           df1[f'Y_rel_skel_pos_{skel_pos}'].iloc[row_index], label=f'Tracks_skel_pos_{skel_pos}', s=60, c=(df2['dC_0']))
            elif  skel_pos== 100:
                ax.scatter(df1[f'X_rel_skel_pos_centroid_corrected'].iloc[row_index],
                           df1[f'Y_rel_skel_pos_centroid_corrected'].iloc[row_index], label=f'Tracks_skel_pos_{skel_pos}', s=60, c=(df2['speed']))
            else:
                ax.scatter(df1[f'X_rel_skel_pos_{skel_pos}'].iloc[row_index],
                       df1[f'Y_rel_skel_pos_{skel_pos}'].iloc[row_index], label=f'Tracks_skel_pos_{skel_pos}', s=30)



        ax.scatter(x_odor, y_odor, color='red', label='Odor Point', s=100)

        ax.scatter(x_odor, y_odor, color='red', s=5)

        # Adding a legend
        plt.legend()

        # Setting plot limits and labels
        ax.set_xlim(arena_min_x, arena_max_x)
        ax.set_ylim(arena_min_y, arena_max_y)
        ax.set_xlabel('Distance (mm)')
        ax.set_ylabel('Distance (mm)')
        ax.set_title(f'Simulated Worm at Frame: {frame}')
        ax.grid(True)


        # Text for angles
        bearing_angle_text = f'Bearing Angle: {df1.at[frame, "bearing_angle"]:.2f}'
        curving_angle_text = f'Curving Angle: {df1.at[frame, "curving_angle"]:.2f}'
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

