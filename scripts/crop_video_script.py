import cv2
import pandas as pd
import snakemake
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def read_csv(csv_path):
    # reads DLC csv in and puts the resulting df in a reasonable format to plot data from!

    file_name = csv_path

    # Assuming the file has a header row
    df = pd.read_csv(file_name)

    # remove column names and set first row to new column name
    df.columns = df.iloc[0]
    df = df[1:]

    # Get the first row (which will become the second level of column names)
    second_level_names = df.iloc[0]

    # Create a MultiIndex for columns using the existing column names as the first level
    first_level_names = df.columns
    multi_index = pd.MultiIndex.from_arrays([first_level_names, second_level_names])

    # Set the new MultiIndex as the columns of the DataFrame
    df.columns = multi_index

    # Remove the first row from the DataFrame as it's now used for column names
    df = df.iloc[1:]

    # Removing the first column (index 0)
    df = df.drop(df.columns[0], axis=1)
    df = df.reset_index(drop=True)

    nose_df = df.xs(key='nose', level=0, axis=1)
    pharynx_df = df.xs(key='pharynx', level=0, axis=1)

    # Make copies of the DataFrames
    nose_df_copy = nose_df.copy()
    pharynx_df_copy = pharynx_df.copy()

    # Convert columns to numeric in the copies
    nose_df_copy['x'] = pd.to_numeric(nose_df_copy['x'])
    nose_df_copy['y'] = pd.to_numeric(nose_df_copy['y'])
    pharynx_df_copy['x'] = pd.to_numeric(pharynx_df_copy['x'])
    pharynx_df_copy['y'] = pd.to_numeric(pharynx_df_copy['y'])

    # Calculate the average of x and y columns in the copies
    processed_df = pd.DataFrame()
    processed_df['x'] = (nose_df_copy['x'] + pharynx_df_copy['x']) / 2
    processed_df['y'] = (nose_df_copy['y'] + pharynx_df_copy['y']) / 2

    return processed_df


def crop_video(processed_df, video_path):

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    grayscale_values = []

    # Conversion factor from LQ coordinates to HQ coordinates: similar to  the factor of downsampled video
    conversion_hq = 3

    while True:
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Get the current frame number
        ret, frame = cap.read()  # Read the next frame

        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

        if not ret:
            break  # Break the loop when there are no more frames

        # Define the ROI coordinates (x, y, width, height)
        nose_x = value = processed_df['x'].iloc[frame_number] * conversion_hq
        nose_y = processed_df['y'].iloc[frame_number] * conversion_hq

        # Define the ROI coordinates (x, y, width, height)
        roi_width, roi_height = 256, 256
        roi_x = int(nose_x - roi_width // 2)
        roi_y = int(nose_y - roi_width // 2)

        # Extract the grayscale ROI from the original frame
        gray_roi = cv2.cvtColor(frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width], cv2.COLOR_BGR2GRAY)

        # Append the grayscale values of the ROI to the list
        grayscale_values.append(gray_roi)

    # Release the video capture object and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    return grayscale_values

def export_video(cropped_video_stack, output):

    # Create the figure and subplots outside the update function
    fig = plt.figure(figsize=(12, 5))

    # Define the update function
    def update(frame):
        # Clear the previous data on the subplots
        plt.clf()

        # Plot for array 1
        plt.imshow(cropped_video_stack[frame], cmap='gray')

        plt.axis('off')  # Turn off axis

    # Create the FuncAnimation object with 10 fps (100 ms delay)
    num_frames = len(cropped_video_stack)  # Assuming both arrays have the same number of frames
    ani = FuncAnimation(fig, update, frames=num_frames, repeat=False, blit=False, interval=(1000 / frame_rate))

    # To save the animation as an MP4 file
    ani.save(output, writer='ffmpeg')


if __name__ == "__main__":

    video_path = snakemake.params.video
    csv_path = snakemake.params.csv
    output = snakemake.params.cropped_video

    global frame_rate

    print("cropping video...")
    print("Video path:", video_path)
    print("CSV path:", csv_path)
    print("Output path:", output)

    #call functions that process data----------------------------

    processed_df = read_csv(csv_path)
    cropped_video_stack = crop_video(processed_df, video_path)
    export_video(cropped_video_stack, output)
