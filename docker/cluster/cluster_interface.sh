#!/usr/bin/env bash

#==
# Configurations
#==

# Exits if error occurs
set -e

# Set tab-spaces
tabs 4

# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Resolved "<local_path>:<remote_subdir>" sync specs used by the current job submission.
SYNC_EXTRA_REPO_SPECS=""

#==
# Functions
#==
# Function to display warnings in red
display_warning() {
    echo -e "\033[31mWARNING: $1\033[0m"
}

# Helper function to compare version numbers
version_gte() {
    # Returns 0 if the first version is greater than or equal to the second, otherwise 1
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n 1)" == "$2" ]
}

# Function to check docker versions
check_docker_version() {
    # check if docker is installed
    if ! command -v docker &> /dev/null; then
        echo "[Error] Docker is not installed! Please check the 'Docker Guide' for instruction." >&2;
        exit 1
    fi
    # Retrieve Docker version
    docker_version=$(docker --version | awk '{ print $3 }')
    apptainer_version=$(apptainer --version | awk '{ print $3 }')

    # Check if Docker version is exactly 24.0.7 or Apptainer version is exactly 1.2.5
    if [ "$docker_version" = "24.0.7" ] && [ "$apptainer_version" = "1.2.5" ]; then
        echo "[INFO]: Docker version ${docker_version} and Apptainer version ${apptainer_version} are tested and compatible."

    # Check if Docker version is >= 27.0.0 and Apptainer version is >= 1.3.4
    elif version_gte "$docker_version" "27.0.0" && version_gte "$apptainer_version" "1.3.4"; then
        echo "[INFO]: Docker version ${docker_version} and Apptainer version ${apptainer_version} are tested and compatible."

    # Else, display a warning for non-tested versions
    else
        display_warning "Docker version ${docker_version} and Apptainer version ${apptainer_version} are non-tested versions. There could be issues, please try to update them. More info: https://isaac-sim.github.io/IsaacLab/source/deployment/cluster.html"
    fi
}

# Checks if a docker image exists, otherwise prints warning and exists
check_image_exists() {
    image_name="$1"
    if ! docker image inspect $image_name &> /dev/null; then
        echo "[Error] The '$image_name' image does not exist!" >&2;
        echo "[Error] You might be able to build it with /IsaacLab/docker/container.py." >&2;
        exit 1
    fi
}

# Check if the singularity image exists on the remote host, otherwise print warning and exit
check_singularity_image_exists() {
    image_name="$1"
    if ! ssh "$CLUSTER_LOGIN" "[ -f $CLUSTER_SIF_PATH/$image_name.tar ]"; then
        echo "[Error] The '$image_name' image does not exist on the remote host $CLUSTER_LOGIN!" >&2;
        exit 1
    fi
}

sync_tree_to_cluster() {
    local src_path="$1"
    local dst_path="$2"
    local label="$3"

    if [ ! -d "$src_path" ]; then
        display_warning "Skipping sync for '$label': local path not found: $src_path"
        return
    fi

    echo "[INFO] Syncing $label from '$src_path' -> '$CLUSTER_LOGIN:$dst_path'"
    ssh "$CLUSTER_LOGIN" "mkdir -p '$dst_path'"
    rsync -rh \
        --exclude="*.git*" \
        --exclude="wandb" \
        --filter=':- .dockerignore' \
        "$src_path/" \
        "$CLUSTER_LOGIN:$dst_path/"
}

dir_has_entries() {
    local dir_path="$1"
    if [ ! -d "$dir_path" ]; then
        return 1
    fi
    [ -n "$(find "$dir_path" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]
}

repo_is_dirty() {
    local repo_path="$1"
    if ! git -C "$repo_path" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        return 1
    fi
    [ -n "$(git -C "$repo_path" status --porcelain 2>/dev/null)" ]
}

resolve_repo_sync_path() {
    local repo_label="$1"
    local workspace_path="$2"
    local sibling_path="$3"
    local override_var_name="$4"
    local override_path="${!override_var_name:-}"
    local resolved_path

    if [ -n "$override_path" ]; then
        if [ ! -d "$override_path" ]; then
            echo "[ERROR] $override_var_name is set but path does not exist: '$override_path'" >&2
            exit 1
        fi
        resolved_path="$(realpath "$override_path")"
        echo "[INFO] Using $repo_label from $override_var_name: '$resolved_path'" >&2
        echo "$resolved_path"
        return
    fi

    if dir_has_entries "$workspace_path"; then
        resolved_path="$(realpath "$workspace_path")"
        echo "$resolved_path"
        return
    fi

    if dir_has_entries "$sibling_path"; then
        resolved_path="$(realpath "$sibling_path")"
        echo "[INFO] Using sibling $repo_label repo because workspace path is empty: '$resolved_path'" >&2
        echo "$resolved_path"
        return
    fi

    resolved_path="$(realpath "$workspace_path" 2>/dev/null || echo "$workspace_path")"
    echo "$resolved_path"
}

build_default_sync_specs() {
    local local_workspace_root="$1"
    local sibling_workspace_root="$2"
    local isaaclab_local_path
    local rlopt_local_path
    local ilt_local_path

    isaaclab_local_path="$(resolve_repo_sync_path "IsaacLab" "$local_workspace_root/IsaacLab" "$sibling_workspace_root/IsaacLab" "CLUSTER_ISAACLAB_LOCAL_PATH")"
    rlopt_local_path="$(resolve_repo_sync_path "RLOpt" "$local_workspace_root/RLOpt" "$sibling_workspace_root/RLOpt" "CLUSTER_RLOPT_LOCAL_PATH")"
    ilt_local_path="$(resolve_repo_sync_path "ImitationLearningTools" "$local_workspace_root/ImitationLearningTools" "$sibling_workspace_root/ImitationLearningTools" "CLUSTER_IMITATION_TOOLS_LOCAL_PATH")"
    echo "$isaaclab_local_path:IsaacLab $rlopt_local_path:RLOpt $ilt_local_path:ImitationLearningTools"
}

append_repo_manifest_entry() {
    local manifest_file="$1"
    local repo_name="$2"
    local local_path="$3"
    local remote_subdir="$4"
    local resolved_local_path
    local head_sha
    local branch
    local state
    local changed_files

    resolved_local_path="$(realpath "$local_path" 2>/dev/null || echo "$local_path")"

    if git -C "$resolved_local_path" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        head_sha="$(git -C "$resolved_local_path" rev-parse HEAD 2>/dev/null || echo "N/A")"
        branch="$(git -C "$resolved_local_path" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "N/A")"
        changed_files="$(git -C "$resolved_local_path" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
        if [ "${changed_files:-0}" -gt 0 ]; then
            state="dirty"
        else
            state="clean"
        fi
    else
        head_sha="N/A"
        branch="N/A"
        state="not_git_repo"
        changed_files="N/A"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$repo_name" \
        "$remote_subdir" \
        "$resolved_local_path" \
        "$head_sha" \
        "$branch" \
        "$state" \
        "$changed_files" >> "$manifest_file"

    echo "[INFO] Repo snapshot: name='$repo_name' remote_subdir='$remote_subdir' sha='$head_sha' branch='$branch' state='$state' changed_files='$changed_files' local_path='$resolved_local_path'"
}

record_repo_sync_manifest() {
    local local_workspace_root
    local manifest_local_file
    local manifest_remote_file
    local local_path
    local remote_subdir

    local_workspace_root="$(realpath "$SCRIPT_DIR/../..")"
    manifest_local_file="$(mktemp "${TMPDIR:-/tmp}/isaaclab_cluster_repo_manifest.XXXXXX")"
    manifest_remote_file="$CLUSTER_ISAACLAB_DIR/repo_sync_manifest.tsv"

    {
        echo "# Cluster repo sync manifest"
        echo "generated_at_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        echo "submission_host=$(hostname)"
        echo "cluster_login=$CLUSTER_LOGIN"
        echo "cluster_workspace=$CLUSTER_ISAACLAB_DIR"
        echo
        printf "repo_name\tremote_subdir\tlocal_path\thead_sha\tbranch\tstate\tchanged_files\n"
    } > "$manifest_local_file"

    append_repo_manifest_entry "$manifest_local_file" "IsaacLabImitation" "$local_workspace_root" "."

    for spec in $SYNC_EXTRA_REPO_SPECS; do
        local_path="${spec%%:*}"
        remote_subdir="${spec#*:}"
        if [ -z "$local_path" ] || [ -z "$remote_subdir" ]; then
            continue
        fi
        append_repo_manifest_entry "$manifest_local_file" "$remote_subdir" "$local_path" "$remote_subdir"
    done

    ssh "$CLUSTER_LOGIN" "cat > '$manifest_remote_file'" < "$manifest_local_file"
    echo "[INFO] Saved repo sync manifest to '$CLUSTER_LOGIN:$manifest_remote_file'"
    rm -f "$manifest_local_file"
}

submit_job() {

    echo "[INFO] Arguments passed to job script ${@}"

    case $CLUSTER_JOB_SCHEDULER in
        "SLURM")
            job_script_file=submit_job_slurm.sh
            ;;
        "PBS")
            job_script_file=submit_job_pbs.sh
            ;;
        *)
            echo "[ERROR] Unsupported job scheduler specified: '$CLUSTER_JOB_SCHEDULER'. Supported options are: ['SLURM', 'PBS']"
            exit 1
            ;;
    esac

    ssh $CLUSTER_LOGIN "cd $CLUSTER_ISAACLAB_DIR && bash -l $CLUSTER_ISAACLAB_DIR/docker/cluster/$job_script_file \"$CLUSTER_ISAACLAB_DIR\" \"isaac-lab-$profile\" ${@}"
}

sync_extra_repos() {
    local local_workspace_root
    local sibling_workspace_root
    local local_specs
    local local_path
    local remote_subdir

    local_workspace_root="$(realpath "$SCRIPT_DIR/../..")"
    sibling_workspace_root="$(realpath "$local_workspace_root/..")"

    if [ -n "${CLUSTER_EXTRA_SYNC_SPECS:-}" ]; then
        local_specs="$CLUSTER_EXTRA_SYNC_SPECS"
        echo "[INFO] Using CLUSTER_EXTRA_SYNC_SPECS for additional repo sync."
    else
        local_specs="$(build_default_sync_specs "$local_workspace_root" "$sibling_workspace_root")"
    fi
    SYNC_EXTRA_REPO_SPECS="$local_specs"

    for spec in $local_specs; do
        local_path="${spec%%:*}"
        remote_subdir="${spec#*:}"
        if [ -z "$local_path" ] || [ -z "$remote_subdir" ]; then
            display_warning "Ignoring invalid CLUSTER_EXTRA_SYNC_SPECS entry: '$spec'"
            continue
        fi
        local_path="$(realpath "$local_path" 2>/dev/null || echo "$local_path")"
        if repo_is_dirty "$local_path"; then
            echo "[INFO] $remote_subdir has uncommitted local changes at '$local_path'; syncing working tree state."
        fi
        sync_tree_to_cluster "$local_path" "$CLUSTER_ISAACLAB_DIR/$remote_subdir" "$remote_subdir"
    done
}

#==
# Main
#==

#!/bin/bash

help() {
    echo -e "\nusage: $(basename "$0") [-h] <command> [<profile>] [<job_args>...] -- Utility for interfacing between IsaacLab and compute clusters."
    echo -e "\noptions:"
    echo -e "  -h              Display this help message."
    echo -e "\ncommands:"
    echo -e "  push [<profile>]              Push the docker image to the cluster."
    echo -e "  job [<profile>] [<job_args>]  Submit a job to the cluster."
    echo -e "\nwhere:"
    echo -e "  <profile>  is the optional container profile specification. Defaults to 'base'."
    echo -e "  <job_args> are optional arguments specific to the job command."
    echo -e "\n" >&2
}

# Parse options
while getopts ":h" opt; do
    case ${opt} in
        h )
            help
            exit 0
            ;;
        \? )
            echo "Invalid option: -$OPTARG" >&2
            help
            exit 1
            ;;
    esac
done
shift $((OPTIND -1))

# Check for command
if [ $# -lt 1 ]; then
    echo "Error: Command is required." >&2
    help
    exit 1
fi

command=$1
shift
profile="base"

case $command in
    push)
        if [ $# -gt 1 ]; then
            echo "Error: Too many arguments for push command." >&2
            help
            exit 1
        fi
        [ $# -eq 1 ] && profile=$1
        echo "Executing push command"
        [ -n "$profile" ] && echo "Using profile: $profile"
        if ! command -v apptainer &> /dev/null; then
            echo "[INFO] Exiting because apptainer was not installed"
            echo "[INFO] You may follow the installation procedure from here: https://apptainer.org/docs/admin/main/installation.html#install-ubuntu-packages"
            exit
        fi
        # Check if Docker image exists
        check_image_exists isaac-lab-$profile:latest
        # Check docker and apptainer version
        check_docker_version
        # source env file to get cluster login and path information
        source $SCRIPT_DIR/.env.cluster
        # make sure exports directory exists
        mkdir -p /$SCRIPT_DIR/exports
        # clear old exports for selected profile
        rm -rf /$SCRIPT_DIR/exports/isaac-lab-$profile*
        # create singularity image
        # NOTE: we create the singularity image as non-root user to allow for more flexibility. If this causes
        # issues, remove the --fakeroot flag and open an issue on the IsaacLab repository.
        cd /$SCRIPT_DIR/exports
        APPTAINER_NOHTTPS=1 apptainer build --sandbox isaac-lab-$profile.sif docker-daemon://isaac-lab-$profile:latest
        # tar image (faster to send single file as opposed to directory with many files)
        tar -cvf /$SCRIPT_DIR/exports/isaac-lab-$profile.tar isaac-lab-$profile.sif
        # make sure target directory exists
        ssh $CLUSTER_LOGIN "mkdir -p $CLUSTER_SIF_PATH"
        # send image to cluster
        scp $SCRIPT_DIR/exports/isaac-lab-$profile.tar $CLUSTER_LOGIN:$CLUSTER_SIF_PATH/isaac-lab-$profile.tar
        ;;
    job)
        if [ $# -ge 1 ]; then
            passed_profile=$1
            if [ -f "$SCRIPT_DIR/../.env.$passed_profile" ]; then
                profile=$passed_profile
                shift
            fi
        fi
        job_args="$@"
        echo "[INFO] Executing job command"
        [ -n "$profile" ] && echo -e "\tUsing profile: $profile"
        [ -n "$job_args" ] && echo -e "\tJob arguments: $job_args"
        source $SCRIPT_DIR/.env.cluster
        # Get current date and time
        current_datetime=$(date +"%Y%m%d_%H%M%S")
        # Append current date and time to CLUSTER_ISAACLAB_DIR
        CLUSTER_ISAACLAB_DIR="${CLUSTER_ISAACLAB_DIR}_${current_datetime}"
        # Check if singularity image exists on the remote host
        check_singularity_image_exists isaac-lab-$profile
        # make sure target directory exists
        ssh $CLUSTER_LOGIN "mkdir -p $CLUSTER_ISAACLAB_DIR"
        # Sync Isaac Lab imitation code
        echo "[INFO] Syncing IsaacLab-Imitation code..."
        rsync -rh --exclude="*.git*" --exclude="wandb" --filter=':- .dockerignore' /$SCRIPT_DIR/../.. $CLUSTER_LOGIN:$CLUSTER_ISAACLAB_DIR
        # Sync optional extra repos (default: IsaacLab + RLOpt + ImitationLearningTools)
        sync_extra_repos
        # Record exact repo SHAs and dirty state used in this submission.
        record_repo_sync_manifest
        # execute job script
        echo "[INFO] Executing job script..."
        # check whether the second argument is a profile or a job argument
        submit_job $job_args
        ;;
    *)
        echo "Error: Invalid command: $command" >&2
        help
        exit 1
        ;;
esac
