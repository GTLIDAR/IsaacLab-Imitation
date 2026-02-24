# Repository Setup (Git Submodules)

This repo now tracks its dependent repos as submodules:

- `IsaacLab/`
- `RLOpt/`
- `ImitationLearningTools/`

`IsaacLab-Imitation` itself remains the top-level repo.

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
```

Expected default remotes:

- `IsaacLab-Imitation`: `origin -> git@github.com:GTLIDAR/IsaacLab-Imitation.git`
- `IsaacLab`: `origin -> git@github.com:GTLIDAR/IsaacLab.git`
- `RLOpt`: `origin -> git@github.com:fei-yang-wu/RLOpt.git`
- `ImitationLearningTools`: `origin -> git@github.com:GTLIDAR/ImitationLearningTools.git`

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
bash cluster_interface.sh job base --headless
```
