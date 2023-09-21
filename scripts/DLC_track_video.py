import os
import deeplabcut
import snakemake
import argparse

def start_tracking(video_file, config_path):

    print("Videofile:", video_file)
    print("Configpath:", config_path)

    deeplabcut.analyze_videos(config_path, video_file, save_as_csv=True, videotype=".avi",)

    print(f"Analyzed and labeled {video_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="track with DLC")
    parser.add_argument("--video", required=True)
    parser.add_argument("--use_DLC", required=True)
    args = parser.parse_args()

    video_file = args.video
    config_path = args.use_DLC

    start_tracking(video_file, config_path)



