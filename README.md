# IsaacLab-Imitation

IsaacLab-Imitation is a multi-repo workspace for humanoid imitation learning on top of Isaac Lab. This repository
contains the Isaac Lab extension code for the imitation environments, while the workspace also vendors or assumes local
checkouts of the upstream `IsaacLab`, `RLOpt`, and `ImitationLearningTools` repositories (typically as git submodules).

The current focus is manager-based imitation environments for the Unitree G1 robot, with training flows built around
RLOpt and RSL-RL.

## What is in this repo

- `source/isaaclab_imitation`: the installable Isaac Lab extension package
- `scripts/rlopt`: training and playback entrypoints for RLOpt
- `scripts/rsl_rl`: training entrypoints for RSL-RL
- `scripts/zero_agent.py`, `scripts/random_agent.py`: smoke-test environment runners
- `IsaacLab/`, `RLOpt/`, `ImitationLearningTools/`: required submodule checkouts
- `docker/cluster`: cluster submission utilities

Registered task IDs currently include:

- `Isaac-Imitation-G1-LafanTrack-v0`
- `Isaac-Imitation-G1-Dance102-Compare-v0`
- `Isaac-Imitation-G1-v0`
- `Isaac-Imitation-G1-Dance102-v0`

## Workspace setup

Clone with submodules (recommended layout, with `IsaacLab` etc. as submodules inside this repo):

```bash
git clone --recurse-submodules git@github.com:GTLIDAR/IsaacLab-Imitation.git
cd IsaacLab-Imitation
```

If you already cloned without submodules:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

This workspace also expects the upstream `IsaacLab`, `unitree_rl_lab`, and `loco-mujoco` repositories to be available
either as submodules in this repo (under directories such as `IsaacLab/`) or as sibling checkouts next to it.

When using sibling checkouts, a typical layout looks like:

```text
/path/to/workspace-root/
  IsaacLab-Imitation/
  IsaacLab/
  unitree_rl_lab/
  loco-mujoco/
```

In that case, make sure your Python tooling and environment configuration (for example, `PYTHONPATH` or editable
installs) can see all of these repositories.

If you are following the submodule-based layout, `IsaacLab` and the other vendored repos will live directly under
`IsaacLab-Imitation/` instead.

This workspace also expects `unitree_rl_lab` specifically as a sibling checkout for training and cluster workflows:

```bash
cd ..
git clone https://github.com/unitreerobotics/unitree_rl_lab.git
cd IsaacLab-Imitation
```

You also need to complete the upstream `unitree_rl_lab` setup, not just clone it. Follow the installation and asset
setup steps in `../unitree_rl_lab/README.md` before running training or mimic-style data workflows in this repo.

More detail on remotes, submodules, and cluster sync lives in [REPO_SETUP.md](REPO_SETUP.md).

## Installation

Use the workspace installer:

```bash
./scripts/install_workspace.sh
```

The script does the following:

- verifies the active `python` is `3.11`
- installs `uv` with `conda install -y uv`
- initializes git submodules if `IsaacLab`, `RLOpt`, or `ImitationLearningTools` are incomplete
- installs `ImitationLearningTools` and `RLOpt` in editable mode
- installs `isaacsim[all,extscache]==5.1.0`
- installs `torch==2.7.0` and `torchvision==0.22.0` from the CUDA 12.8 wheel index
- runs `./isaaclab.sh -i none` inside `IsaacLab`
- installs `source/isaaclab_imitation` in editable mode

If you need the manual submodule setup details or cluster notes, see [REPO_SETUP.md](REPO_SETUP.md).

## Running training

Examples below assume you are running from the repository root.

Train a G1 imitation policy with RLOpt IPMD:

```bash
./IsaacLab/isaaclab.sh -p scripts/rlopt/train.py \
    --task Isaac-Imitation-G1-LafanTrack-v0 \
    --algo IPMD \
    --headless
```

Train with RLOpt PPO:

```bash
./IsaacLab/isaaclab.sh -p scripts/rlopt/train.py \
    --task Isaac-Imitation-G1-LafanTrack-v0 \
    --algo PPO \
    --headless
```

Train with RSL-RL:

```bash
./IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-Imitation-G1-LafanTrack-v0 \
    --headless
```

Common flags:

- `--task`: selects the registered Isaac Lab environment
- `--num_envs`: overrides the environment count from config
- `--max_iterations`: caps training iterations
- `--video`: records periodic rollout videos during training
- `--device cuda:0`: pins execution to a specific GPU

Logs are written under `logs/`.

## Data preparation

Motion loading in this repo is manifest-driven and repo-local under `data/`.

Tracked manifests:

- `source/isaaclab_imitation/isaaclab_imitation/manifests/g1_default_manifest.json`: 3-motion debug manifest
- `source/isaaclab_imitation/isaaclab_imitation/manifests/g1_dance102_manifest.json`: single-motion `dance_102` manifest
- `source/isaaclab_imitation/isaaclab_imitation/manifests/g1_lafan1_manifest.template.json`: tracked template for a
  full local G1 LAFAN1 manifest

Local manifests:

- `data/lafan1/manifests/g1_lafan1_manifest.json`: full local G1 LAFAN1 manifest
- `data/lafan1/manifests/g1_debug_manifest.json`: optional smaller local subset

The full local G1 set is not shipped in git. When you prepare local motions under `data/lafan1/npz/g1/`, the full
manifest should live under `data/lafan1/manifests/g1_lafan1_manifest.json`.

See `data/README.md` for the expected local directory layout and the common local-data commands.

### Recommended full-dataset flow

The simplest way to get the full local G1 dataset is the Hugging Face preparation wrapper:

```bash
conda run -n SkillLearning python scripts/setup_lafan1_dataset.py \
    --prepare-npz --headless
```

This downloads the public retargeted LAFAN1 G1 CSV set, converts it to NPZ, and writes:

```text
data/lafan1/raw/g1/
data/lafan1/npz/g1/
data/lafan1/manifests/g1_lafan1_manifest.json
```

The Hugging Face dataset stores the retargeted G1 motions at 30 FPS, so the wrapper passes `--input_fps 30`
automatically during conversion. Use `--robot_type h1`, `--robot_type h1_2`, or `--robot_type all` for other subsets.

### If You Already Have NPZ Files

If `data/lafan1/manifests/g1_lafan1_manifest.json` already exists, you do not need to regenerate it.

If you already have local NPZ files but no manifest yet, generate one directly:

```bash
conda run -n SkillLearning python scripts/write_lafan1_npz_manifest.py \
    --npz_dir data/lafan1/npz/g1 \
    --manifest_path data/lafan1/manifests/g1_lafan1_manifest.json
```

If you want to hand-edit a manifest instead of generating one, copy the tracked template:

```bash
mkdir -p data/lafan1/manifests
cp source/isaaclab_imitation/isaaclab_imitation/manifests/g1_lafan1_manifest.template.json \
   data/lafan1/manifests/g1_lafan1_manifest.json
```

For a smaller local subset:

```bash
conda run -n SkillLearning python scripts/write_lafan1_npz_manifest.py \
    --npz_dir data/lafan1/npz/g1 \
    --manifest_path data/lafan1/manifests/g1_debug_manifest.json \
    --select dance1_subject1 dance1_subject2 walk1_subject1
```

### If You Start From CSV Files

Prepare local CSV motions into NPZ plus a manifest with:

```bash
python scripts/prepare_lafan1_from_csv.py \
    --csv_dir /absolute/path/to/csv_motions \
    --npz_dir /absolute/path/to/data/lafan1/npz/g1 \
    --manifest_path /absolute/path/to/data/lafan1/manifests/g1_lafan1_manifest.json \
    --recursive
```

If you want one replay MP4 per converted motion, add `--record_videos` and `--video_dir`.

### Direct NPZ Sync With Hugging Face

If you only want the prepared NPZ subtree, use:

```bash
conda run -n SkillLearning python scripts/setup_g1_lafan1_npz_dataset.py
```

That syncs `npz/g1` from the dataset repo `GeorgiaTech/g1_lafan1_50hz` into:

```text
data/lafan1/npz/g1/
```

Upload mode pushes the same local NPZ tree back to Hugging Face:

```bash
conda run -n SkillLearning python scripts/setup_g1_lafan1_npz_dataset.py \
    --mode upload --token "$HF_TOKEN"
```

### Dance 102

For the built-in `dance_102` comparison manifest, the expected repo-local output is:

```text
data/dance_102/G1_Take_102.bvh_60hz.npz
```

If you have the CSV already, convert it with:

```bash
python scripts/csv_to_npz.py \
    -f /absolute/path/to/G1_Take_102.bvh_60hz.csv \
    --input_fps 60 \
    --output_name data/dance_102/G1_Take_102.bvh_60hz.npz
```

## Playback and smoke tests

Run a zero-action smoke test:

```bash
./IsaacLab/isaaclab.sh -p scripts/zero_agent.py \
    --task Isaac-Imitation-G1-LafanTrack-v0
```

Run a random-action smoke test:

```bash
./IsaacLab/isaaclab.sh -p scripts/random_agent.py \
    --task Isaac-Imitation-G1-LafanTrack-v0
```

Play back an RLOpt checkpoint:

```bash
./IsaacLab/isaaclab.sh -p scripts/rlopt/play.py \
    --task Isaac-Imitation-G1-LafanTrack-v0 \
    --checkpoint /absolute/path/to/checkpoint.pt
```

Replay all 40 local G1 LAFAN1 motions from the full manifest:

```bash
./IsaacLab/isaaclab.sh -p scripts/replay_reference.py \
    --task Isaac-Imitation-G1-LafanTrack-v0 \
    --motion_manifest data/lafan1/manifests/g1_lafan1_manifest.json \
    --motion_refresh_dataset \
    --reset_schedule round_robin \
    --num_envs 40 \
    --video \
    --video_length 500 \
    --headless
```

If you are already inside an Isaac Sim-enabled Python environment, the equivalent direct invocation is:

```bash
python scripts/replay_reference.py \
    --task Isaac-Imitation-G1-LafanTrack-v0 \
    --motion_manifest data/lafan1/manifests/g1_lafan1_manifest.json \
    --motion_refresh_dataset \
    --reset_schedule round_robin \
    --num_envs 40 \
    --video \
    --video_length 500 \
    --headless
```

Notes:

- use `data/lafan1/manifests/g1_lafan1_manifest.json` to load the full local 40-motion set
- use `source/isaaclab_imitation/isaaclab_imitation/manifests/g1_default_manifest.json` only for the small 3-motion debug set
- if your workspace uses a sibling checkout instead of a submodule for Isaac Lab, replace `./IsaacLab/isaaclab.sh` with
  `../IsaacLab/isaaclab.sh`
- `replay_reference.py` disables reward and termination terms by default, so long reference videos do not reset early
- pass `--keep_terminations` or `--keep_rewards` if you explicitly want the old RL-style behavior during replay
- `--num_envs 40` is the way to see all 40 loaded trajectories at once; using fewer environments still loads the manifest,
  but only that many trajectories are visible at a time

## Development workflow

This repo is easier to work on with terminal-first tooling than with heavy IDE indexing.

Recommended tools:

- `ruff` for linting and formatting
- `pyrefly` for type and import checking

Install them into your environment or with `uv tool`:

```bash
uv tool install ruff
uv tool install pyrefly
```

Useful commands:

```bash
ruff check .
ruff format .
pyrefly check
```

`pyrefly` is configured by [pyrefly.toml](pyrefly.toml) and already includes
the import roots for this repo plus sibling/source dependencies such as `IsaacLab` and `unitree_rl_lab`.

For VS Code, prefer the Ruff extension and terminal-based `pyrefly` checks. Pylance is not the recommended workflow for
this workspace because the Isaac / Omniverse dependency tree is large, generated settings tend to drift, and static
analysis is more reliable here when driven from the checked-in repo configuration.

## Formatting and hooks

A pre-commit configuration is included:

```bash
pip install pre-commit
pre-commit run --all-files
```

Note that the current hook set is inherited from upstream Isaac Lab conventions. For day-to-day work in this repo,
`ruff` and `pyrefly` are the recommended feedback loop.

## Cluster note

For cluster submission, local Isaac Lab Python installation is not required on the submission machine if jobs run inside
the provided container or Apptainer image. See `docker/cluster` and [REPO_SETUP.md](REPO_SETUP.md) for the expected sync
layout and environment variables.
