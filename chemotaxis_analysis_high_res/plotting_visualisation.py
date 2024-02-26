import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def plot_chemotaxis_overview(df, output_path, x_odor, y_odor, arena_min_x, arena_max_x, arena_min_y, arena_max_y, file_name):
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

    # Combine the output path and file name
    full_path = os.path.join(output_path, file_name)
    print("The full file path is:", full_path)

    plt.figure(figsize=(160, 160))

    # Create a scatter plot for the corrected tracks
    plt.scatter(df['X_rel_skel_pos_centroid_corrected'], df['Y_rel_skel_pos_centroid_corrected'], label='Tracks_centroid', s=10, c=(df['speed']), cmap='plasma')
    plt.colorbar(label='Time(min)')

    # Create a scatter plot for the corrected tracks
    plt.scatter(df['X_rel_skel_pos_centroid_corrected'], df['Y_rel_skel_pos_centroid_corrected'], label='Tracks_centroid', s=50, c=(df['time_seconds'] / 60), cmap='plasma', alpha=0.3)
    plt.colorbar(label='Time(min)')

    # Create a scatter plot for the nose tracks
    plt.scatter(df['X_rel_skel_pos_0'], df['Y_rel_skel_pos_0'], label='Tracks_nose', s=10, c=('dC_0'))

    # Plot the "odor" point
    plt.scatter(x_odor, y_odor, color='red', label='Odor Point', s=1000)

    plt.xlim(arena_min_x, arena_max_x)
    plt.ylim(arena_min_y, arena_max_y)

    # Add grid lines
    plt.grid(True)

    # Add labels and legend
    plt.xlabel('X Relative')
    plt.ylabel('Y Relative')
    plt.title('Tracks and Odor Point')
    plt.legend()

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

    # Release everything when job is finished
    out.release()
    plt.close(fig)  # Close the figure to free memory


