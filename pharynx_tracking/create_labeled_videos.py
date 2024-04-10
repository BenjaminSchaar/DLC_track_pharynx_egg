import argparse
import deeplabcut

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="track with DLC")
    parser.add_argument("--video", required=True)
    parser.add_argument("--use_DLC", required=True)
    args = parser.parse_args()

    video = args.video
    use_DLC = args.use_DLC

    print("Video Tracking folder:", video.rsplit("/", 1)[0])

    deeplabcut.create_labeled_video(use_DLC, video.rsplit("/", 1)[0], videotype='.mp4')
    #deeplabcut.create_labeled_video(use_DLC, "/scratch/neurobiology/zimmer/schaar/code/DLC_track_pharynx_egg/output/downsampled", videotype='.mp4')

