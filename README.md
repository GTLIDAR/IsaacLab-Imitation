# IsaacLab-Imitation

IsaacLab-Imitation is a multi-repo workspace for humanoid imitation learning on top of Isaac Lab. This repository
contains the Isaac Lab extension code for the imitation environments, while the workspace also vendors the
`IsaacLab`, `RLOpt`, and `ImitationLearningTools` repositories as git submodules.

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

Clone with submodules:

```bash
git clone --recurse-submodules git@github.com:GTLIDAR/IsaacLab-Imitation.git
cd IsaacLab-Imitation
```

If you already cloned without submodules:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

This workspace also expects `unitree_rl_lab` as a sibling checkout for training and cluster workflows:

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

`unitree_rl_lab` is part of the expected local setup for two reasons:

- some training and evaluation flows depend on its robot/task assets and config
- the dance mimic data conversion pipeline uses its `scripts/mimic/csv_to_npz.py` converter

For dance tasks that use mimic motion data, convert the source CSV into NPZ before running the task.

The bundled `dance_102` CSV lives at:

```text
../unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/dance_102/G1_Take_102.bvh_60hz.csv
```

Run the converter like this:

```bash
python ../unitree_rl_lab/scripts/mimic/csv_to_npz.py \
    --input_file ../unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/dance_102/G1_Take_102.bvh_60hz.csv
```

This generates the `.npz` file next to the CSV, in the same directory.

For `dance_102`, the generated output is:

```text
../unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/dance_102/G1_Take_102.bvh_60hz.npz
```

If you are preparing a larger set of CSV motions for the LAFAN-style loader in this repo, use:

```bash
python scripts/prepare_lafan1_from_csv.py \
    --csv_dir /absolute/path/to/csv_motions \
    --npz_dir /absolute/path/to/npz_motions \
    --manifest_path /absolute/path/to/g1_lafan1_manifest.json \
    --recursive
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
