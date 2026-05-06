# Experiment Workflow

This page records the practical workflow for local tests, cluster jobs, and
experiment tracking for IPMD-family G1 imitation runs.

The rule is simple: local smoke first, then full cluster job. Do not submit a
cluster job until the local command proves the task, algorithm, manifest, and
repo overlay are wired correctly.

## Local Validation Ladder

Start with the cheapest check that matches the change.

### 1. Docs or shell-only changes

```bash
git diff --check
bash -n docker/cluster/cluster_interface.sh
bash -n experiments/ipmd_stability/run_local_debug_ablations.sh
bash -n experiments/ipmd_stability/submit_cluster_ablations.sh
```

### 2. Expert-batch or env sampling changes

Pure pytest path:

```bash
conda run -n SkillLearning pytest source/isaaclab_imitation/test_reference_patch_env.py
```

Isaac Lab launcher path, needed when imports require Isaac Sim / Omniverse:

```bash
TERM=xterm conda run -n SkillLearning ../IsaacLab/isaaclab.sh -p -m pytest \
    source/isaaclab_imitation/test_reference_patch_env.py
```

Use `./IsaacLab/isaaclab.sh` instead of `../IsaacLab/isaaclab.sh` when using the
in-repo submodule checkout.

### 3. Minimal train smoke

Use a small number of envs and one or a few rollout iterations to prove wiring:

```bash
TERM=xterm PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
conda run -n SkillLearning python scripts/rlopt/train.py \
    --task Isaac-Imitation-G1-v0 \
    --num_envs 16 \
    --headless \
    --algo IPMD \
    --max_iterations 2 \
    --log_interval 1000 \
    --kit_args=--/app/extensions/fsWatcherEnabled=false \
    env.lafan1_manifest_path=./data/unitree/manifests/g1_unitree_dance102_manifest.json \
    env.dataset_path=/tmp/iltools_g1_lafan1_tracking_g1_unitree_dance102_manifest_6d26546fd54a \
    env.refresh_zarr_dataset=False \
    agent.logger.backend= \
    agent.logger.exp_name=ipmd_local_smoke
```

The empty `agent.logger.backend=` override disables the external metrics backend
for quick local smoke tests. Remove it when you want W&B tracking.

For latent IPMD, switch the task:

```bash
--task Isaac-Imitation-G1-Latent-v0
```

For bilinear IPMD, switch the algorithm and enable offline pretrain if that is
the surface under test:

```bash
--algo IPMD_BILINEAR \
agent.bilinear.offline_pretrain.enabled=true \
agent.bilinear.offline_pretrain.num_updates=10
```

The `--kit_args=--/app/extensions/fsWatcherEnabled=false` override is useful on
local machines where Isaac Kit file watcher startup fails under resource
pressure.

## Existing Experiment Scripts

`experiments/ipmd_stability/run_local_debug_ablations.sh` runs local IPMD reward
stability sweeps. Useful knobs:

```bash
TASK=Isaac-Imitation-G1-v0 \
NUM_ENVS=128 \
TIMEOUT_SECONDS=300 \
SEEDS=2024 \
COMBOS="A B C" \
experiments/ipmd_stability/run_local_debug_ablations.sh
```

`experiments/ipmd_stability/submit_cluster_ablations.sh` submits cluster sweeps
for IPMD and baseline GAIL/AMP/ASE variants:

```bash
DRY_RUN=1 \
TASK=Isaac-Imitation-G1-v0 \
NUM_ENVS=2048 \
ALGO=ipmd \
SEEDS="2024" \
COMBOS="A B" \
experiments/ipmd_stability/submit_cluster_ablations.sh
```

Use `DRY_RUN=1` first to inspect the generated commands.

`experiments/vqvae_temporal_ablation.sh` is the current VQ-VAE/FSQ temporal
ablation helper. It supports `local`, `cluster`, and `print` modes:

```bash
experiments/vqvae_temporal_ablation.sh print all
experiments/vqvae_temporal_ablation.sh local vqvae_p10_d64
experiments/vqvae_temporal_ablation.sh cluster vqvae_p10_d64
```

## Full Cluster Jobs

Cluster submission entrypoint:

```bash
./docker/cluster/cluster_interface.sh job [profile] [job args...]
```

The default profile is `base`. A typical full latent IPMD job:

```bash
./docker/cluster/cluster_interface.sh job \
    --task Isaac-Imitation-G1-Latent-v0 \
    --num_envs 4096 \
    --headless \
    --video \
    --algo IPMD \
    --kit_args=--/app/extensions/fsWatcherEnabled=false \
    agent.logger.exp_name=ipmd_latent_full_4096
```

A bilinear job with offline pretraining enabled:

```bash
./docker/cluster/cluster_interface.sh job \
    --task Isaac-Imitation-G1-Latent-v0 \
    --num_envs 4096 \
    --headless \
    --video \
    --algo IPMD_BILINEAR \
    --kit_args=--/app/extensions/fsWatcherEnabled=false \
    agent.bilinear.offline_pretrain.enabled=true \
    agent.bilinear.offline_pretrain.num_updates=2000 \
    agent.logger.exp_name=ipmd_bilinear_offline_full_4096
```

Cluster jobs append the default full G1 manifest unless the submitted command
already includes `env.lafan1_manifest_path=...`. The default is controlled by
`docker/cluster/.env.cluster`.

For Dance102 or other single-manifest debugging, pass the manifest explicitly:

```bash
env.lafan1_manifest_path=./data/unitree/manifests/g1_unitree_dance102_manifest.json
```

## Sibling RLOpt Overlay

If a job needs local algorithm edits from sibling `../RLOpt`, make sure
`docker/cluster/.env.cluster` has the overlay path enabled:

```bash
CLUSTER_RLOPT_LOCAL_PATH=/home/fwu91/Documents/Research/SkillLearning/RLOpt
```

If that line is commented out, cluster jobs use the pinned `RLOpt/` submodule
state from `IsaacLab-Imitation`.

Every job writes a repo manifest to:

```text
<CLUSTER_ISAACLAB_DIR>/repo_sync_manifest.tsv
```

Use it to confirm the exact branch/SHA/dirty-state for `IsaacLab-Imitation` and
any overlaid repos.

## Tracking Experiments

Every `scripts/rlopt/train.py` run writes local metadata under:

```text
logs/rlopt/<algo>/<task>/<timestamp>/
```

Important files:

- `command.txt`: exact command used for the run.
- `params/env.yaml`: resolved environment config.
- `params/agent.yaml`: resolved RLOpt config.
- `rlopt.log`: durable training summaries from RLOpt.
- `videos/train/`: local rollout videos when `--video` is enabled.
- `models/`: checkpoints, when the agent saves them.

Use explicit experiment names:

```bash
agent.logger.exp_name=<short_descriptive_name>
agent.logger.group_name=<optional_group_name>
```

Default RLOpt logging uses W&B. For cluster jobs, provide the key on the cluster
host and let `run_singularity.sh` inject it into the container:

```bash
printf '%s\n' 'your_wandb_api_key' > ~/.wandb_api_key
chmod 600 ~/.wandb_api_key
```

Then set:

```bash
CLUSTER_WANDB_API_KEY_FILE=.wandb_api_key
```

Do not rely only on W&B. For debugging, inspect `rlopt.log`, `command.txt`, and
the YAML configs first.

Useful IPMD metrics to scan in `rlopt.log`:

- `episode/return` and `episode/length`
- `r_step`
- `reward_diff`
- `exp_r`
- `env_r`
- `reward_l2`
- `reward_gp`
- `v_loss`
- `entropy`
- `grad_norm`
- `lr`
- `clip`

Interpretation rule: separate standing/stability from imitation quality. A run
can improve episode length or standing while still failing to imitate the
reference motion.

## Local To Cluster Promotion

Use this promotion checklist:

1. Run the smallest local test that exercises the changed path.
2. Inspect `logs/rlopt/.../command.txt`, `params/agent.yaml`, and `rlopt.log`.
3. Use a distinct `agent.logger.exp_name`.
4. Confirm whether the job should use sibling `../RLOpt` overlay or submodule
   `RLOpt/`.
5. Run `DRY_RUN=1` for experiment scripts that support it.
6. Submit the cluster job.
7. Record the job id, repo manifest path, experiment name, task, algo, seed,
   manifest, and important overrides.

## Common Failure Modes

- Hydra receives `task_name=None`: inspect the generated cluster job command;
  scheduler wrappers must preserve one shell word per argument.
- Cluster job uses stale algorithm code: check `repo_sync_manifest.tsv` and
  whether `CLUSTER_RLOPT_LOCAL_PATH` was enabled.
- Dance102 smoke loads the full LAFAN cache: pass both
  `env.lafan1_manifest_path=...` and a matching explicit `env.dataset_path=...`.
- W&B panels look duplicated: inspect the actual local history/logs before
  changing logger code.
- Runtime succeeds but imitation quality is poor: inspect `reward_diff`,
  `exp_r`, videos, and reference comparison; do not treat standing alone as
  success.
