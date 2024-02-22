import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation

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

    plt.figure(figsize=(160, 160))

    # Create a scatter plot for the corrected tracks
    plt.scatter(df['X_rel_skel_pos_centroid_corrected'], df['Y_rel_skel_pos_centroid_corrected'], label='Tracks_centroid', s=0.1, c=(df['time_seconds'] / 60), cmap='plasma')
    plt.colorbar(label='Time(min)')

    # Create a scatter plot for the nose tracks
    plt.scatter(df['X_rel_skel_pos_0'], df['Y_rel_skel_pos_0'], label='Tracks_nose', s=0.1, color='lightblue')

    # Plot the "odor" point
    plt.scatter(x_odor, y_odor, color='red', label='Odor Point', s=100)

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
    plt.savefig(file_name)
    plt.close()  # Close the plot to free memory

def create_angle_animation(df, output_path, x_odor, y_odor, fps, file_name):
    '''
    :param df:
    :param output_path:
    :param x_odor:
    :param y_odor:
    :param fps:
    :param output_filename:
    :return:
    '''

    # Combine the output path and file name
    full_path = os.path.join(output_path, file_name)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Function to update the plot for each frame
    def update(frame):
        ax.clear()
        row_number = frame

        # Extracting the central point coordinates
        center_x_close = df.at[row_number, 'X_rel']
        center_y_close = df.at[row_number, 'Y_rel']

        # Plotting the points for the angle
        ax.scatter(df.at[row_number, 'X_rel'], df.at[row_number, 'Y_rel'], color='blue', s=5)
        ax.scatter(df.at[row_number, 'X_shifted_negative'], df.at[row_number, 'Y_shifted_negative'], color='lightblue', s=5)
        ax.scatter(df.at[row_number, 'X_shifted_positive'], df.at[row_number, 'Y_shifted_positive'], color='purple', s=5)
        ax.scatter(x_odor, y_odor, color='red', s=5)

        # Drawing lines for the angle
        ax.plot([x_odor, df.at[row_number, 'X_rel']], [y_odor, df.at[row_number, 'Y_rel']], color='red', linestyle='--', linewidth=0.8)
        ax.plot([df.at[row_number, 'X_shifted_negative'], df.at[row_number, 'X_rel']], [df.at[row_number, 'Y_shifted_negative'], df.at[row_number, 'Y_rel']], color='green', linestyle='--', linewidth=0.8)
        ax.plot([df.at[row_number, 'X_shifted_positive'], df.at[row_number, 'X_rel']], [df.at[row_number, 'Y_shifted_positive'], df.at[row_number, 'Y_rel']], color='blue', linestyle='--', linewidth=0.8)

        # Setting the window limits around the point
        ax.set_xlim(center_x_close - 5, center_x_close + 5)
        ax.set_ylim(center_y_close - 5, center_y_close + 5)

        # Set labels and title
        ax.set_xlabel('Distance (mm)')
        ax.set_ylabel('Distance (mm)')
        ax.set_title(f'Bearing Angle Visualization for Row {row_number}')
        ax.grid(True)

        # Display bearing and curving angles as text
        bearing_angle_text = 'Bearing Angle: {:.2f}'.format(df.at[row_number, 'bearing_angle'])
        curving_angle_text = 'Curving Angle: {:.2f}'.format(df.at[row_number, 'curving_angle'])
        ax.text(0.05, 0.95, bearing_angle_text, transform=ax.transAxes, fontsize=10, color='black')
        ax.text(0.05, 0.90, curving_angle_text, transform=ax.transAxes, fontsize=10, color='black')

    # Create the animation
    animation = FuncAnimation(fig, update, frames=len(df), interval=100, repeat=False)

    # Save the animation as an AVI file
    animation.save(full_path, writer='pillow', fps=fps)

    # Close the plot to prevent it from displaying in the notebook or Python script
    plt.close()


