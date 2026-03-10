#!/usr/bin/env bash

echo "(run_singularity.py): Called on compute node from current isaaclab directory $1 with container profile $2 and arguments ${@:3}"

#==
# Helper functions
#==

setup_directories() {
    # Check and create directories
    for dir in \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/kit" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/ov" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/pip" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/glcache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/computecache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/triton" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/torchinductor" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/home" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/logs" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents" \
        "${CLUSTER_DATA_DIR}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo "Created directory: $dir"
        fi
    done
}


#==
# Main
#==


# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# load variables to set the Isaac Lab path on the cluster
source $SCRIPT_DIR/.env.cluster
source $SCRIPT_DIR/../.env.base

# Runtime home inside singularity container.
# Defaults to /home/$USER so Isaac Sim writes to a user path, but the path is
# backed by scratch via bind mount below.
container_home="${CLUSTER_CONTAINER_HOME:-/home/${USER}}"
container_triton_cache_dir="${container_home}/.cache/triton"
container_torchinductor_cache_dir="${container_home}/.cache/torchinductor"
allow_torch_compile_debug="${CLUSTER_ALLOW_TORCH_COMPILE_DEBUG:-0}"

# Construct PYTHONPATH entries from synced repos.
# NOTE: We intentionally avoid "IsaacLab/source" because it makes "isaaclab" a namespace package
# (no __file__), which can break wandb/pydantic introspection in torchrl logging.
extra_pythonpath_rel="${CLUSTER_EXTRA_PYTHONPATH_REL:-IsaacLab/source/isaaclab:IsaacLab/source/isaaclab_tasks:IsaacLab/source/isaaclab_assets:IsaacLab/source/isaaclab_rl:IsaacLab/source/isaaclab_mimic:source/isaaclab_imitation:unitree_rl_lab/source/unitree_rl_lab:RLOpt:ImitationLearningTools}"
container_pythonpath_prefix=""
IFS=':' read -ra extra_pythonpath_items <<< "$extra_pythonpath_rel"
for rel_path in "${extra_pythonpath_items[@]}"; do
    if [ -n "$rel_path" ]; then
        container_pythonpath_prefix="${container_pythonpath_prefix}/workspace/isaaclab/project/${rel_path}:"
    fi
done
container_pythonpath="${container_pythonpath_prefix}\${PYTHONPATH}"

# make sure that all directories exists in cache directory
setup_directories
# copy all cache files
cp -r $CLUSTER_ISAAC_SIM_CACHE_DIR $TMPDIR

# make sure logs directory exists (in the permanent isaaclab directory)
mkdir -p "$CLUSTER_ISAACLAB_DIR/logs"
touch "$CLUSTER_ISAACLAB_DIR/logs/.keep"

# copy the temporary isaaclab directory with the latest changes to the compute node
cp -r $1 $TMPDIR
# Get the directory name
dir_name=$(basename "$1")

# copy container to the compute node
tar -xf $CLUSTER_SIF_PATH/$2.tar  -C $TMPDIR

# create a persistant overlay using apptainer with fakeroot
apptainer overlay create --size 20240 $CLUSTER_ISAACLAB_DIR/$dir_name.img

# execute command in singularity container
# NOTE: ISAACLAB_PATH is normally set in `isaaclab.sh` but we directly call the isaac-sim python because we sync the entire
# Isaac Lab directory to the compute node and remote the symbolic link to isaac-sim
singularity exec \
    -B $TMPDIR/docker-isaac-sim/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw \
    -B $TMPDIR/docker-isaac-sim/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
    -B $TMPDIR/docker-isaac-sim/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
    -B ${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/triton:${container_triton_cache_dir}:rw \
    -B ${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/torchinductor:${container_torchinductor_cache_dir}:rw \
    -B $TMPDIR/docker-isaac-sim/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw \
    -B $TMPDIR/docker-isaac-sim/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw \
    -B $TMPDIR/docker-isaac-sim/documents:${DOCKER_USER_HOME}/Documents:rw \
    -B $TMPDIR/docker-isaac-sim/home:${container_home}:rw \
    -B $TMPDIR/$dir_name:/workspace/isaaclab/project:rw \
    -B $CLUSTER_ISAACLAB_DIR/logs:/workspace/isaaclab/project/logs:rw \
    -B ${CLUSTER_DATA_DIR}:/data:rw \
    --overlay $CLUSTER_ISAACLAB_DIR/$dir_name.img \
    --nv --containall $TMPDIR/$2.sif \
    bash -c "export ACCEPT_EULA=${ACCEPT_EULA:-Y} && export PRIVACY_CONSENT=${PRIVACY_CONSENT:-Y} && export OMNI_KIT_ACCEPT_EULA=YES && export HOME=${container_home} && export XDG_CACHE_HOME=${container_home}/.cache && export XDG_DATA_HOME=${container_home}/.local/share && export ISAACLAB_WORKSPACE_PATH=/workspace/isaaclab/project && export ISAACLAB_PATH=/workspace/isaaclab/project/IsaacLab && export ISAACSIM_PATH=/workspace/isaaclab/project/IsaacLab/_isaac_sim && export ISAACLAB_DATA_DIR=/data && export PYTHONPATH=${container_pythonpath} && export WANDB_API_KEY=$(cat ~/.wandb_api_key) && export TRITON_CACHE_DIR=${container_triton_cache_dir} && export TORCHINDUCTOR_CACHE_DIR=${container_torchinductor_cache_dir} && export RL_WARNINGS=${RL_WARNINGS:-False} && if [ \"${allow_torch_compile_debug}\" != \"1\" ]; then unset TORCH_LOGS; export TORCHDYNAMO_VERBOSE=0; export TORCH_COMPILE_DEBUG=0; fi && cd /workspace/isaaclab/project && /isaac-sim/python.sh ${CLUSTER_PYTHON_EXECUTABLE} ${@:3}"

# copy resulting cache files back to host
rsync -azPv $TMPDIR/docker-isaac-sim $CLUSTER_ISAAC_SIM_CACHE_DIR/..

# if defined, remove the temporary isaaclab directory pushed when the job was submitted
if $REMOVE_CODE_COPY_AFTER_JOB; then
    rm -rf $1
fi

# remove the temporary image file
if $REMOVE_OVERLAY_AFTER_JOB; then
    rm -f $CLUSTER_ISAACLAB_DIR/$dir_name.img
fi

echo "(run_singularity.py): Return"
