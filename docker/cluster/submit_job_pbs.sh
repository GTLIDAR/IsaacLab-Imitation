#!/usr/bin/env bash

run_singularity_path="$1/docker/cluster/run_singularity.sh"
workspace_root="$1"
container_profile="$2"
shift 2

printf -v quoted_run_singularity_path '%q' "$run_singularity_path"
printf -v quoted_workspace_root '%q' "$workspace_root"
printf -v quoted_container_profile '%q' "$container_profile"
printf -v quoted_job_args '%q ' "$@"

# in the case you need to load specific modules on the cluster, add them here
# e.g., `module load eth_proxy`

# create job script with compute demands
### MODIFY HERE FOR YOUR JOB ###
cat <<EOT > job.sh
#!/bin/bash

#PBS -l select=1:ncpus=8:mpiprocs=1:ngpus=1
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -q gpu
#PBS -N isaaclab
#PBS -m bea -M "user@mail"

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash $quoted_run_singularity_path $quoted_workspace_root $quoted_container_profile $quoted_job_args
EOT

qsub job.sh
rm job.sh
