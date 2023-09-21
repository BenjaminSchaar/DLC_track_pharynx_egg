# Snakefile
import os
import glob

current_folder = os.getcwd()
avi_files = [file for file in os.listdir(current_folder) if file.endswith(".avi")]

# Define input and output files/directories
video_file = avi_files[0]

if not os.path.exists("output"):
    os.makedirs("output")
    print("directory created")

if not os.path.exists("output/downsampled"):
    os.makedirs("output/downsampled")
    print("directory created")

if not os.path.exists("output/cropped"):
    os.makedirs("output/cropped")
    print("directory created")

#inputs/outputs---------------------------
downsampled_video = "output/downsampled/downsampled_video.avi"
track_nose_csv = "output/downsampled/downsampled.csv"
cropped_video = "output/cropped/cropped_video.avi"
track_pharynx_csv = "output/cropped/cropped.csv"
labeled_video_pharynx = "output/downsampled/pharynx_labeled.mp4"
labeled_video_nose = "output/cropped/nose_labeled.mp4"

# Params----------------------------------
track_nose_DLC = "/scratch/neurobiology/zimmer/schaar/code/DLC/track_nose_pharynx-Ben-2023-09-05/config.yaml"
track_pumping_DLC = "/scratch/neurobiology/zimmer/schaar/code/DLC/track_pharynx_pumping_HQ-Ben-2023-09-07/config.yaml"

frame_rate = int(30)
print(track_nose_csv)

# Define the rule all to specify all rules as the default target(s)
rule targets:
    input:
        final_csv=track_pharynx_csv

rule downsample_video:
    input:
        video=video_file
    output:
        downsampled_video=downsampled_video
    params:
        fps=frame_rate
    shell:
        "python scripts/downsample_script.py --video {input.video} --output {output.downsampled_video} --fps {params.fps}"

rule nose_tracking:
    input:
        downsampled_video=downsampled_video
    output:
        track_nose_csv=track_nose_csv
    params:
        use_DLC=track_nose_DLC
    shell:
        "python scripts/DLC_track_video.py --video {input.downsampled_video} --use_DLC {params.use_DLC} "


rule crop_video:
    input:
        track_nose_csv=track_nose_csv
    output:
        cropped=cropped_video
    params:
        video=video_file,
        fps=frame_rate
    shell:
        "python scripts/crop_video_script.py --video {params.video} --csv {input.track_nose_csv} --output {output.cropped} --fps {params.fps}"


rule pharynx_egg_tracking:
    input:
        video=cropped_video
    output:
        track_pharynx_csv=track_pharynx_csv
    params:
        use_DLC=track_pumping_DLC
    shell:
        "python scripts/DLC_track_video.py --video {input.video} --use_DLC {params.use_DLC} "

'''
rule create_labeled_video_nose:
    input:
        video=downsampled_video
    output:
        video_out=labeled_video_nose
    params:
        use_DLC=track_nose_DLC
    shell:
        "python scripts/create_labeled_videos.py --video {input.video} --use_DLC {params.use_DLC}"

rule create_labeled_video_pharynx:
    input:
        video=cropped_video
    output:
        video_out=labeled_video_pharynx
    params:
        use_DLC=track_pumping_DLC
    shell:
       "python scripts/create_labeled_videos.py --video {input.video} --use_DLC {params.use_DLC}"

'''