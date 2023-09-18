# Snakefile

# Define input and output files/directories
video_file = "input/video.mp4"
downsampled_video = "output/downsampled_video.mp4"
deeplabcut_csv = "output/deeplabcut_output.csv"
cropped_video = "output/cropped_video.mp4"
dlc_tracking_output = "output/dlc_tracking_output"

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
    output:
        csv=deeplabcut_csv
    shell:
        "python deeplabcut_script.py {input.video} {output.csv}"

rule crop_video:
    input:
        video=video_file,
        csv=deeplabcut_csv
    output:
        cropped=cropped_video
    shell:
        "python crop_video_script.py {input.video} {input.csv} {output.cropped}"

rule dlc_tracking:
    input:
        video=cropped_video,
        csv=deeplabcut_csv
    output:
        dlc=dlc_tracking_output
    shell:
        "python dlc_script.py {input.video} {input.csv} {output.dlc}"

# Add additional rules as needed
