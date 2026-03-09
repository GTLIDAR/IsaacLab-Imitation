#!/usr/bin/env bash
cat <<EOT > job.sh
#!/bin/bash

#SBATCH --gpus-per-node=l40s:1
#SBATCH -N1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=24G
#SBATCH --time=16:00:00
#SBATCH --job-name="training-$(date +"%Y-%m-%dT%H:%M")"
#SBATCH --output="output_%j.log"
#SBATCH --error="error_%j.log"

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "$1/docker/cluster/run_singularity.sh" "$1" "$2" "${@:3}"
EOT
sbatch < job.sh
rm job.sh
