# Current Status

Last refreshed: 2026-05-06.

This page is intentionally dated. Update it when a branch lands or when the
active experiment direction changes.

## Branch Snapshot

- Current branch: `feature/ipmd-offline-spectral-pretrain`.
- Latest pulled commit: `ab602cb Support offline IPMD bilinear expert batches`.
- The fast-forward pull updated:
  - `source/isaaclab_imitation/isaaclab_imitation/envs/imitation_rl_env.py`
  - `source/isaaclab_imitation/test_reference_patch_env.py`

## High-Value Repo Context

- `IsaacLab-Imitation` is the orchestration repo for Isaac Lab G1 imitation
  environments, task registration, RLOpt entrypoints, cluster scripts, data
  manifests, and experiment scripts.
- The active research focus is representation learning for IPMD-family inverse
  RL: learn useful state representations and reward structure from expert state
  trajectories, then adapt online with environment interaction.
- Sibling `/home/fwu91/Documents/Research/SkillLearning/RLOpt` remains the
  active edit target for algorithm/runtime work.
- The `RLOpt/` submodule in this repo is a pinned dependency snapshot for
  reproducibility.
- `scripts/rlopt/train.py` resolves `--algo` to task-specific
  `rlopt_<algo>_cfg_entry_point` registry entries.
- `ImitationRLEnv.sample_expert_batch(...)` is the env-owned expert sampling
  surface used by imitation algorithms.

## Current Task Surfaces

- `Isaac-Imitation-G1-v0`: vanilla G1 tracking task.
- `Isaac-Imitation-G1-Latent-v0`: latent-conditioned G1 task.
- `Isaac-Imitation-G1-Latent-VQVAE-v0`: latent VQ-VAE G1 task.
- `IPMD_BILINEAR`: routed through the bilinear config entrypoint.

## Context-Management Status

- `AGENTS.md` is the durable coding-agent rule file.
- `CLAUDE.md` is the Claude Code command and architecture shortcut file.
- `wiki/context-management.md` is the durable context policy.
- `wiki/ipmd-representation-learning.md` records the current algorithmic focus.
- `wiki/experiment-workflow.md` records the local/cluster/tracking workflow.
- This file holds dated status and should be refreshed or pruned as work
  changes.
