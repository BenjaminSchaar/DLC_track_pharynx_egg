#!/bin/bash
#SBATCH --output=run_ome_tiff_to_avi_simple.out
#SBATCH --error=run_ome_tiff_to_avi_simple.err

OPT="sbatch -p {cluster.partition} --cpus-per-task {cluster.cpus_per_task} --mem {cluster.mem} --output {cluster.output}"

# Run the actual script
snakemake --snakefile track_pharynx_egg_laying --cores 1 --latency-wait 60 --cluster "$OPT" --cluster-config track_pharynx_egg_laying_cluster_config.yaml -j 1


