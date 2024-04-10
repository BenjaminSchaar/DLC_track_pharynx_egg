import cv2
import pandas as pd
import numpy as np
import snakemake
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import glob

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

    vulva_df = df.xs(key='vulva', level=0, axis=1)

    vulva_df_copy = vulva_df.copy()

    # Convert columns to numeric in the copies
    vulva_df_copy['x'] = pd.to_numeric(vulva_df_copy['x'])
    vulva_df_copy['y'] = pd.to_numeric(vulva_df_copy['y'])

    return vulva_df_copy


def crop_video(processed_df, video_path, roi_width, roi_height, frame_rate):

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create an empty NumPy array to store grayscale values
    new_roi_array = []

    # Conversion factor from LQ coordinates to HQ coordinates: similar to  the factor of downsampled video
    conversion_hq = 3

    while True:
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Get the current frame number
        ret, frame = cap.read()  # Read the next frame

        if not ret:
            break  # Break the loop when there are no more frames

        # Define the ROI coordinates (x, y, width, height)
        vulva_x = processed_df['x'].iloc[frame_number]*conversion_hq
        vulva_y = processed_df['y'].iloc[frame_number]*conversion_hq

        roi_x = int(vulva_x - (roi_width / 2))
        roi_y = int(vulva_y - (roi_width / 2))

        # Ensure ROI coordinates are within frame boundaries, else skip
        if (
                roi_x >= 0 and
                roi_y >= 0 and
                roi_x + roi_width <= frame.shape[1] and
                roi_y + roi_height <= frame.shape[0]
        ):
            # Extract the grayscale ROI from the original frame
            #frame_roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
            frame_roi = cv2.cvtColor(frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width], cv2.COLOR_BGR2GRAY)
            # Append the grayscale values of the ROI to the list
            new_roi_array.append(frame_roi)
        else:
            # Create an empty frame (all black) when the ROI is out of bounds
            empty_frame = np.zeros((roi_height, roi_width), dtype=np.uint8)
            # Append the empty frame to the list
            new_roi_array.append(empty_frame)
            # Print a message indicating that an empty frame is added
            print(f"Frame {frame_number}: ROI is out of bounds. Adding an empty frame...")

    # Release the video capture object and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    return new_roi_array

def export_video(cropped_video_stack, output, frame_rate, roi_width, roi_height):
    # Get the shape of the input frames


    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use the XVID codec for AVI
    out = cv2.VideoWriter(output, fourcc, frame_rate, (roi_width, roi_height))

    for frame in cropped_video_stack:
        # Apply histogram equalization
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # Write the frame to the video file
        out.write(frame)

    # Release the VideoWriter object
    out.release()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="track with DLC")
    parser.add_argument("--video", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", required=True)
    parser.add_argument("--crop_size", required=True)
    args = parser.parse_args()

    video_path = args.video
    csv_path = args.csv
    output = args.output
    frame_rate = int(args.fps)
    crop_size = int(args.crop_size)

    # Define the ROI coordinates (x, y, width, height) from the crop
    roi_width, roi_height = crop_size, crop_size

    print("cropping video...")
    print("Video path:", video_path)
    print("CSV path:", csv_path)
    print("Output path:", output)

    #call functions that process data----------------------------

    processed_df = read_csv(csv_path)
    cropped_video_stack = crop_video(processed_df, video_path, roi_width, roi_height, frame_rate)
    export_video(cropped_video_stack, output, frame_rate, roi_width, roi_height)
