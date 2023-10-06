import os
import deeplabcut
import snakemake
import argparse
import glob
import time

def rename_csv_for_snakemake(video_file, output):

    folder_path = video_file.rsplit("/", 1)[0]
    print("Folderpath: ", folder_path)
    csv_file_path = glob.glob(os.path.join(folder_path, '*.csv'))
    print(type(csv_file_path[0]))

    if csv_file_path:
        csv_file = csv_file_path[0]

        # Check if the old file exists before renaming
        if os.path.exists(csv_file):
            # Rename the file
            os.rename(csv_file, output)

        # Do something with the first CSV file
    else:
        print("No CSV files found in the specified folder.")




def start_tracking(video_file, config_path):

    print("Videofile:", video_file)
    print("Configpath:", config_path)

    deeplabcut.analyze_videos(config_path, video_file, save_as_csv=True, videotype=".avi",)

    print(f"Analyzed and labeled {video_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="track with DLC")
    parser.add_argument("--video", required=True)
    parser.add_argument("--use_DLC", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    video_file = args.video
    config_path = args.use_DLC
    output = args.output

    start_tracking(video_file, config_path)

    # Wait for 10 seconds for filesystem
    print("waiting up to 60 seconds for filesystem to generate files")
    time.sleep(60)

    rename_csv_for_snakemake(video_file, output)



