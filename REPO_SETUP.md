# Repository Setup (Git Submodules)

This repo now tracks its dependent repos as submodules:

- `IsaacLab/`
- `RLOpt/`
- `ImitationLearningTools/`

`IsaacLab-Imitation` itself remains the top-level repo.

Additionally required (currently not a submodule):

- `unitree_rl_lab/` (sibling repository)

## 1. Clone with submodules

```bash
git clone --recurse-submodules git@github.com:GTLIDAR/IsaacLab-Imitation.git
cd IsaacLab-Imitation
```

If you already cloned before submodules were added:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

## 2. Verify remotes (from current git config)

Run:

```bash
git remote -v
git -C IsaacLab remote -v
git -C RLOpt remote -v
git -C ImitationLearningTools remote -v
git -C ../unitree_rl_lab remote -v
```

Expected default remotes:

- `IsaacLab-Imitation`: `origin -> git@github.com:GTLIDAR/IsaacLab-Imitation.git`
- `IsaacLab`: `origin -> git@github.com:GTLIDAR/IsaacLab.git`
- `RLOpt`: `origin -> git@github.com:fei-yang-wu/RLOpt.git`
- `ImitationLearningTools`: `origin -> git@github.com:GTLIDAR/ImitationLearningTools.git`
- `unitree_rl_lab`: `origin -> https://github.com/unitreerobotics/unitree_rl_lab.git`

## 2b. Set up unitree_rl_lab (required)

If `unitree_rl_lab` is missing as a sibling repo:

```bash
cd ..
git clone https://github.com/unitreerobotics/unitree_rl_lab.git
```

Then follow the upstream setup instructions in:

- `../unitree_rl_lab/README.md`

At minimum, run its installation step and required asset/environment setup before training.

Optional extra remotes used in this workspace:

```bash
git -C IsaacLab remote add upstream git@github.com:isaac-sim/IsaacLab.git
git -C RLOpt remote add gatech https://github.gatech.edu/GeorgiaTechLIDARGroup/RLOpt.git
```

## 3. Update submodules later

```bash
git submodule update --init --recursive
```

To move submodules to newer commits:

```bash
git submodule update --remote --recursive
git add IsaacLab RLOpt ImitationLearningTools
git commit -m "Update submodule pins"
```

## 4. Cluster note (no conda/venv needed for submission)

For cluster submission, you do not need a local conda/venv for IsaacLab Python packages.

- Job execution uses `/isaac-sim/python.sh` inside the container/Apptainer image.
- Local requirements for submission are mainly Docker, Apptainer, and SSH access to the cluster.

Typical flow:

```bash
cd docker/cluster
# edit .env.cluster for cluster paths/login/script
bash cluster_interface.sh push base
bash cluster_interface.sh job --task Isaac-Imitation-G1-Latent-v0 --algo IPMD --headless
```

**Multiple clusters:** per-cluster env files and submit scripts are supported. Create `docker/cluster/.env.<name>` and optionally `docker/cluster/submit_job_slurm_<name>.sh`, then pass `-c <name>`:

```bash
bash cluster_interface.sh -c ice job --task Isaac-Imitation-G1-Latent-v0 --algo IPMD --headless
```

If no `-c` is given, the script auto-selects `submit_job_slurm_${CLUSTER_LOGIN}.sh` (from `.env.cluster`) when that file exists, falling back to `submit_job_slurm.sh`. See `docker/README.md` for full details.

If your active development clone for `RLOpt` (or `IsaacLab`, `ImitationLearningTools`) is outside this repo, set path overrides in `docker/cluster/.env.cluster` so cluster jobs sync your working tree directly:

```bash
CLUSTER_RLOPT_LOCAL_PATH=/absolute/path/to/RLOpt
# Optional:
# CLUSTER_ISAACLAB_LOCAL_PATH=/absolute/path/to/IsaacLab
# CLUSTER_IMITATION_TOOLS_LOCAL_PATH=/absolute/path/to/ImitationLearningTools
```

These overrides are used when `CLUSTER_EXTRA_SYNC_SPECS` is not set. Only the uncommented overrides are synced as overlays. If none are set, the cluster job uses the submodule state from the main `IsaacLabImitation` checkout without extra repo sync. The paths are local paths on the submission machine.

Each `job` submission also writes a repo manifest to `<CLUSTER_ISAACLAB_DIR>/repo_sync_manifest.tsv` containing SHA/branch/dirty-state for all synced repos.
