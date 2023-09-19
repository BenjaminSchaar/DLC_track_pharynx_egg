# Snakefile
import os

current_folder = os.getcwd()
avi_files = [file for file in os.listdir(current_folder) if file.endswith(".avi")]

# Define input and output files/directories
video_file = avi_files[0]


if not os.path.exists("output"):
    os.makedirs("output")
    print("directory created")

downsampled_video = "output/downsampled_video.avi"
deeplabcut_csv = "output/deeplabcut_output.csv"
cropped_video = "output/cropped_video.avi"
track_pharynx_csv = "output/track_pharynx.csv"

# Define paths for DLC_networks
track_nose_DLC = "/scratch/neurobiology/zimmer/schaar/code/DLC/track_nose_pharynx-Ben-2023-09-05"
track_pumping_DLC = "/scratch/neurobiology/zimmer/schaar/code/DLC/track_pharynx_pumping_HQ-Ben-2023-09-07"

rule downsample_video:
    input:
        video=video_file
    output:
        downsampled_video=downsampled_video
    shell:
        "python scripts/downsample_script.py {input.video} {output.downsampled}"

rule deeplabcut_tracking:
    input:
        video=downsampled_video,
        use_DLC=track_nose_DLC
    output:
        csv=deeplabcut_csv
    shell:
        "python scripts/DLC_track_video.py {input.video} {input.use_DLC} {output.csv}"

rule crop_video:
    input:
        video=video_file,
        csv=deeplabcut_csv
    output:
        cropped=cropped_video
    shell:
        "python scripts/crop_video_script.py {input.video} {input.csv} {output.cropped}"

rule pharynx_egg_tracking:
    input:
        video=cropped_video,
        use_DLC=deeplabcut_csv
    output:
        csv=track_pharynx_csv
    shell:
        "python scripts/DLC_track_video.py {input.video} {input.use_DLC} {output.csv}"
