import cv2
import snakemake
import os

def compress_avi(path, cfactor, downsampled_video):

    print("avi being compressed")

    # Import video file
    video = cv2.VideoCapture(str(path))

    # Get the video frame dimensions
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    original_fps = video.get(cv2.CAP_PROP_FPS)

    print("Original FPS:", original_fps)

    # Calculate target resolution
    target_width = int(width/int(cfactor))
    target_height = int(height/int(cfactor))

    # Create video writer object to save compressed video
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(downsampled_video, fourcc, original_fps, (target_width, target_height))

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

    cfactor = 3 # compress from
    path = snakemake.params.video
    output = snakemake.params.downsampled_video

    print("Videopath:", path)
    print("Outputfile:", output)

    output_folder =  output.split('/')

    print("output_folder:",  output_folder[0])

    if not os.path.exists(output_folder[0]):
        os.makedirs(output_folder[0])
        print("directory created")


    compress_avi(path, cfactor, output)
