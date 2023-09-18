import os
import deeplabcut
import snakemake


video_file = snakemake.params.video
config_path = snakemake.params.use_DLC

deeplabcut.analyze(
    video_file,
    config_path=config_path,
    Shuffles=1,
    save_as_csv=true,
    cropping=True,
    Tframe_start=0,
    Tframe_end=0,
    trail_crop=False,
    update=False,
    rois=None,
    net_type=None,
    task="2d",
    net_avg=False,
    opts=None,
    start=0,
    stop=1,
    indices=None,
    videotype=".avi",
)

# After analysis, create labeled videos
deeplabcut.create_labeled_video(config_path, video_file)

print(f"Analyzed and labeled {video_file}")

