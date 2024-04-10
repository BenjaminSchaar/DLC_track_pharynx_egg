import cv2
import argparse
import pickle
import os

def save_var_to_pickle(cfactor, path):
    # Remove the file extension from the path
    directory_path = os.path.dirname(path)
    # Construct the pickle file path by joining the root folder with the variable name and '.pickle'
    pickle_filename = "cfactor.pickle"
    pickle_path = os.path.join(directory_path, pickle_filename)

    try:
        # Open the file in binary mode
        with open(pickle_path, 'wb') as file:
            # Serialize and write the variable to the file
            pickle.dump(cfactor, file)
        print(f"The variable 'cfactor' has been saved successfully to {pickle_path}")
    except Exception as e:
        print("An error occurred while saving the variable: ", e)

def compress_avi(path, downsampled_video, frame_rate):

    print("avi being compressed")

    # Import video file
    video = cv2.VideoCapture(str(path))

    # Get the video frame dimensions
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cfactor = width/512

    save_var_to_pickle(cfactor, path)

    print("Compression-factor to 512x512:", cfactor)


    print("Set FPS:", frame_rate)

    # Calculate target resolution - onl one cfactor since sometimes videos are not perfect squares
    target_width = int(width/cfactor)
    target_height = int(height/cfactor)

    print("Target_width:", target_width)
    print("target_height:", target_height)

    # Create video writer object to save compressed video
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(downsampled_video, fourcc, frame_rate, (target_width, target_height))

    # Loop through all video frames
    while True:
        # Read the next frame
        ret, frame = video.read()

        # If there are no more frames, break out of the loop
        if not ret:
            break

        # Resize the frame to the target resolution
        frame = cv2.resize(frame, (target_width, target_height))

        # Write the resized frame to the output video file
        out.write(frame)

    # Release the video capture and writer objects
    video.release()
    out.release()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="downsample video for DLC")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", required=True)
    args = parser.parse_args()

    path = args.video
    output = args.output
    frame_rate = int(args.fps)

    print("Videopath:", path)
    print("Outputfile:", output)

    output_folder =  output.split('/')

    print("output_folder:",  output_folder[0])

    compress_avi(path, output, frame_rate)
