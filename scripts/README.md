# Scripts

Command-line entrypoints and maintenance utilities for this repo.

Use the subdirectory paths below in docs, launchers, and new code.

## Layout

- `data/`: dataset download, CSV/NPZ conversion, manifests, audits, BONES-SEED prep
- `rlopt/`: RLOpt train, play, evaluation, offline pretraining, and planner pipelines
- `eval/`: reference replay, policy comparison, and dynamics/backend probes
- `smoke/`: lightweight Isaac Lab smoke runners (`zero_agent`, `random_agent`, `list_envs`)
- `benchmark/`: MDP / physics-backend / renderer throughput benchmarks
- `rsl_rl/`, `sb3/`, `skrl/`: alternate RL library train/play entrypoints
- `install_workspace.sh`: compatibility wrapper around `pixi install`

## Dataset Preparation (`data/`)

- `download_g1_lafan1_data.sh`: fetch the packaged G1 LAFAN1 dataset
- `setup_lafan1_dataset.py`: prepare LAFAN1 source data
- `prepare_lafan1_from_csv.py`: convert LAFAN1 CSV data into G1 NPZ data
- `csv_to_npz.py`, `batch_csv_to_npz.py`: single and batched CSV-to-NPZ conversion
- `setup_g1_lafan1_npz_dataset.py`, `setup_g1_bones_seed_npz_dataset.py`:
  materialize packaged G1 NPZ datasets
- `write_lafan1_npz_manifest.py`, `merge_g1_motion_manifests.py`: build or merge manifests
- `anchor_npz_local_frame.py`, `repair_g1_lafan1_body_offsets.py`,
  `audit_g1_lafan1_body_frames.py`: repair or validate motion-frame data
- `prepare_bones_seed_phase5.py`, `audit_bones_seed_phase5.py`,
  `compare_bones_seed_exports.py`: Phase-5 preparation and audits
- `select_bones_seed_100.py`, `prepare_bones_seed_subset.py`,
  `filter_bones_seed_sonic_exclusions.py`: select or filter motion subsets
- `align_language_sidecar_to_manifest.py`,
  `enrich_bones_seed_language_sidecar.py`: maintain language sidecars

## Evaluation And Diagnostics (`eval/`)

- `replay_reference.py`, `replay_unitree_lerobot_reference.py`,
  `preview_unitree_lerobot_episode.py`: inspect and replay reference motions
- `compare_policy_reference.py`, `compare_policy_reference_all.py`,
  `run_lafan1_closed_loop_eval_all_tmux.sh`: compare policy rollouts
- `validate_lerobot_streaming_cache.py`: probe LeRobot streaming cache health
- `dump_backend_index_contract.py`, `diagnose_g1_dynamics.py`: backend and dynamics probes

## Smoke (`smoke/`)

- `zero_agent.py`, `random_agent.py`: lightweight Isaac Lab smoke runners
- `list_envs.py`: list registered Isaac Lab environments

## Benchmarks (`benchmark/`)

- `benchmark_g1_mdp.py`: benchmark G1 MDP logic
- `benchmark_physics_backends.py`: compare physics backend throughput
- `benchmark_renderers.py`: compare renderer behavior and throughput

## Tests

Tests for these utilities live in `../tests/`, not in this directory.
