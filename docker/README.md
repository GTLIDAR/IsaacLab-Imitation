# Docker and Cluster Usage

## Local Docker

The container is managed through `container.py` (the deprecated `container.sh` wrapper calls this):

```bash
# Build the image
python docker/container.py build [--profile base|ros2]

# Start container in detached mode
python docker/container.py start [--profile base|ros2]

# Enter a running container
python docker/container.py enter [--profile base|ros2]

# Stop and remove
python docker/container.py stop [--profile base|ros2]
```

Container configuration lives in `docker/.env.base`. Key paths inside the container:

| Host | Container |
|------|-----------|
| `source/` | `/workspace/isaaclab/source` |
| `scripts/` | `/workspace/isaaclab/scripts` |
| `IsaacLab/` | `/workspace/isaaclab/IsaacLab` |
| `RLOpt/` | `/workspace/isaaclab/RLOpt` |

---

## Cluster Job Submission

All cluster operations go through `docker/cluster/cluster_interface.sh`.

### Prerequisites

- `apptainer` installed locally (for `push`)
- SSH alias configured for the cluster login node (e.g. `skynet` → `Host skynet` in `~/.ssh/config`)
- The Singularity image already pushed to the cluster (see [Pushing the Image](#pushing-the-image))

### Setup

Copy and fill in the cluster env file:

```bash
cp docker/cluster/.env.cluster docker/cluster/.env.cluster
# or for a named cluster:
cp docker/cluster/.env.cluster docker/cluster/.env.<clustername>
```

Key variables in `.env.cluster` (all paths are **relative to `$HOME` on the cluster**):

| Variable | Purpose |
|----------|---------|
| `CLUSTER_LOGIN` | SSH alias for the login node (e.g. `skynet`) |
| `CLUSTER_JOB_SCHEDULER` | `SLURM` or `PBS` |
| `CLUSTER_ISAACLAB_DIR` | Where code is synced on the cluster |
| `CLUSTER_ISAAC_SIM_CACHE_DIR` | Isaac Sim cache directory (must end in `docker-isaac-sim`) |
| `CLUSTER_SIF_PATH` | Directory holding the `.tar`-packed Singularity image |
| `CLUSTER_DATA_DIR` | Dataset root; bind-mounted into the container at `/data` |
| `CLUSTER_PYTHON_EXECUTABLE` | Script run inside the container (default: `scripts/rlopt/train.py`) |

### Pushing the Image

Build and push a local Docker image to the cluster as an Apptainer `.sif`:

```bash
# Build the Docker image first
python docker/container.py build

# Push to cluster (converts to Singularity format, tars, and scps)
./docker/cluster/cluster_interface.sh push [<profile>]
# e.g.
./docker/cluster/cluster_interface.sh push base
```

### Submitting Jobs

```bash
./docker/cluster/cluster_interface.sh job [<job_args>...]
```

`<job_args>` are forwarded verbatim to `CLUSTER_PYTHON_EXECUTABLE` inside the container. Example:

```bash
./docker/cluster/cluster_interface.sh job \
    --task Isaac-Imitation-G1-Latent-v0 \
    --algo IPMD \
    --headless \
    --num_envs 4096 \
    agent.trainer.progress_bar=False
```

**What happens on submission:**

1. Code is synced to `${CLUSTER_ISAACLAB_DIR}_<timestamp>` (git-first, falls back to rsync).
2. Optional extra repos (RLOpt, IsaacLab, ImitationLearningTools) are overlaid if configured.
3. A `repo_sync_manifest.tsv` is written to the remote workspace recording exact commit SHAs.
4. A SLURM/PBS job script is generated and submitted. The job:
   - Copies the workspace and Singularity image to `$TMPDIR` on the compute node.
   - Optionally auto-downloads and validates the G1 LAFAN1 dataset (see [Data Setup](#data-setup)).
   - Runs the training script inside the container.
   - Syncs Isaac Sim caches back to scratch on completion.

### Multi-Cluster Support

Pass `-c <clustername>` to select an alternate cluster profile. `-c` is consumed before the job arguments and never forwarded to the container.

```bash
# Use default .env.cluster + submit_job_slurm.sh
./docker/cluster/cluster_interface.sh job --task ...

# Use .env.ice + submit_job_slurm_ice.sh (if present, else falls back to submit_job_slurm.sh)
./docker/cluster/cluster_interface.sh -c ice job --task ...
```

**To add a new cluster:**

1. Create `docker/cluster/.env.<clustername>` (copy `.env.cluster`, update `CLUSTER_LOGIN` and paths).
2. Optionally create `docker/cluster/submit_job_slurm_<clustername>.sh` with cluster-specific `#SBATCH` directives. If absent, `submit_job_slurm.sh` is used.

### SLURM Script

`docker/cluster/submit_job_slurm.sh` contains the default `#SBATCH` resource requests:

```
#SBATCH --gpus-per-node=a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-gpu=48G
#SBATCH --time=16:00:00
#SBATCH --partition=wu-lab
#SBATCH --nodelist=dendrite,synapse
#SBATCH --qos=short
```

Edit this file (or create a cluster-specific one) to change resources.

### Data Setup

Set `CLUSTER_AUTO_SETUP_G1_DATA=1` (default) to have the job preflight:
- Check whether the G1 LAFAN1 NPZ dataset is present and complete.
- Download it from Hugging Face (`CLUSTER_G1_REPO_ID`, default `GeorgiaTech/g1_lafan1_50hz`) if needed.
- Regenerate the manifest according to `CLUSTER_G1_MANIFEST_REFRESH_POLICY`:
  - `auto` — regenerate if missing or stale relative to the NPZ tree
  - `never` — leave the manifest untouched (use for Unitree or hand-authored manifests)
  - `always` — regenerate on every job

A Hugging Face token with `read` scope is required for dataset download. Store it in a file on the cluster:

```bash
echo "hf_xxx" > ~/.hf_token
```

Then set in `.env.cluster`:
```
CLUSTER_HF_TOKEN_FILE=.hf_token
```

Similarly for Weights & Biases:
```
CLUSTER_WANDB_API_KEY_FILE=.wandb_api_key
```

### Repo Sync

Code is synced using a git-first strategy:
- If local HEAD is reachable from the remote, the cluster checks out exactly that commit via `git clone --depth 1`.
- If the local repo is ahead of or diverged from the remote, the remote branch is cloned and the local diff (`git diff --binary`) is applied on top.
- Falls back to `rsync` if any git step fails or git metadata is missing.
- Set `CLUSTER_GIT_SYNC_FIRST=0` to always use rsync.

**Syncing extra repos** (RLOpt, IsaacLab, ImitationLearningTools):

Uncomment the relevant lines in `.env.cluster`:

```bash
CLUSTER_RLOPT_LOCAL_PATH=/path/to/RLOpt
CLUSTER_ISAACLAB_LOCAL_PATH=/path/to/IsaacLab
CLUSTER_IMITATION_TOOLS_LOCAL_PATH=/path/to/ImitationLearningTools
```

Or override fully with an explicit spec list:

```bash
CLUSTER_EXTRA_SYNC_SPECS="/abs/path/to/RLOpt:RLOpt /abs/path/to/IsaacLab:IsaacLab"
```

### Incremental Sync

On each submission, `cluster_interface.sh` updates a `_latest` symlink pointing to the most recent workspace snapshot. Subsequent syncs use `rsync --link-dest` against this snapshot, so only changed files are transferred. Set `CLUSTER_INCREMENTAL_SYNC=0` to disable.

### Logs and Cleanup

- Job stdout/stderr land in `output_<jobid>.log` in `CLUSTER_ISAACLAB_DIR`.
- Training logs stream to `logs/` inside the container, which is bind-mounted to the permanent `CLUSTER_ISAACLAB_DIR/logs/` directory on scratch (persists across job cleanup).
- `REMOVE_CODE_COPY_AFTER_JOB=true` deletes the per-job code copy from scratch after the job finishes.
- `REMOVE_OVERLAY_AFTER_JOB=true` deletes the Apptainer overlay image after the job finishes.
