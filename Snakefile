# Snakefile
import os

# Define input and output files/directories
video_file = "*.avi" #use path like ulises
downsampled_video = "output/downsampled_video.avi"
deeplabcut_csv = "output/deeplabcut_output.csv"
cropped_video = "output/cropped_video.avi"
track_pharynx_csv = "output/track_pharynx.csv"

# Define paths for DLC_networks
track_nose_DLC = ""
track_pumping_DLC =""

# Define rules for each step in the pipeline
rule all:
    input:
        dlc_tracking_output

rule downsample_video:
    input:
        video=video_file
    output:
        downsampled=downsampled_video
    shell:
        "python downsample_script.py {input.video} {output.downsampled}"

rule deeplabcut_tracking:
    input:
        video=downsampled_video
        use_DLC = track_nose_DLC
    output:
        csv=deeplabcut_csv
    shell:
        "python DLC_track_video.py {input.video} {input.use_DLC} {output.csv}"

rule crop_video:
    input:
        video=video_file,
        csv=deeplabcut_csv
    output:
        cropped=cropped_video
    shell:
        "python crop_video_script.py {input.video} {input.csv} {output.cropped}"

rule pharynx_egg_tracking:
    input:
        video=cropped_video,
        use_DLC=deeplabcut_csv
    output:
        csv=track_pharynx_csv
    shell:
        "python DLC_track_video.py {input.video} {input.use_DLC} {output.csv}"


