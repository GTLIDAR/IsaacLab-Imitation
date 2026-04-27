# G1 IPMD Grouped Estimated-Reward Report

Date: April 26, 2026

## Summary

G1 IPMD now has a reproducibly better pure estimated-reward setting:

- reward model: grouped reward heads
- policy reward source: pure estimated reward
- reward output: nonnegative sigmoid in `[0.0, 0.25]`
- head weights: six equal weights summing to 1
- regularization: `reward_l2_coeff=0.05`, `reward_grad_penalty_coeff=0.2`, `reward_logit_reg_coeff=0.001`

The key result is that `ep_len` improved strongly in two 2048-env, 20M-frame runs:

| Seed | Frames | Final `ep_len` | Final `est_r` | Final `exp_r` | Run log |
|---:|---:|---:|---:|---:|---|
| 2024 | 20.005M | 147.52 | 0.0273 | 0.2143 | `/home/fwu91/Documents/Research/SkillLearning/IsaacLab-Imitation/logs/rlopt/ipmd/Isaac-Imitation-G1-v0/2026-04-26_21-05-21/rlopt.log` |
| 2025 | 20.005M | 133.99 | 0.0273 | 0.2143 | `/home/fwu91/Documents/Research/SkillLearning/IsaacLab-Imitation/logs/rlopt/ipmd/Isaac-Imitation-G1-v0/2026-04-26_21-35-49/rlopt.log` |

This is good enough to stop treating scalar gradient penalty as the main knob. The main finding is reward sign and offset: PPO behaves much better when the learned reward ranks expert above policy while keeping policy rewards low-but-nonnegative.

## Implementation Status

RLOpt changes were made in the sibling repo:

`/home/fwu91/Documents/Research/SkillLearning/RLOpt`

The vendored `IsaacLab-Imitation/RLOpt` submodule was not edited.

Implemented in sibling RLOpt:

- `reward_model_type = "mlp" | "grouped"`
- default remains `"mlp"` for compatibility
- grouped mode uses one small head per semantic reward slice
- grouped mode uses `reference_command` as shared context
- per-head reward differences and gradient penalties are logged
- construction-time validation is used for grouped reward configuration

G1 IPMD config wiring was added in:

`/home/fwu91/Documents/Research/SkillLearning/IsaacLab-Imitation/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/agents/rlopt_ipmd_cfg.py`

Current G1 IPMD reward defaults:

```python
self.ipmd.reward_input_type = "s"
self.ipmd.reward_model_type = "grouped"
self.ipmd.reward_group_head_weights = [1.0 / 6.0] * 6
self.ipmd.use_estimated_rewards_for_ppo = True
self.ipmd.env_reward_weight = 0.0
self.ipmd.est_reward_weight = 1.0
self.ipmd.reward_output_activation = "sigmoid"
self.ipmd.reward_output_scale = 0.25
self.ipmd.estimated_reward_clamp_min = 0.0
self.ipmd.estimated_reward_clamp_max = 0.25
self.ipmd.reward_l2_coeff = 0.05
self.ipmd.reward_grad_penalty_coeff = 0.2
self.ipmd.reward_logit_reg_coeff = 0.001
```

## Reward Sign And Offset

The failed tanh runs used a signed reward range:

```text
reward_output_activation = "tanh"
reward_output_scale = 0.25
clamp = [-0.25, 0.25]
```

That lets the reward model assign:

```text
expert states: positive reward
policy states: negative reward
```

This can still be a good discriminator, but it is a bad PPO reward surface. PPO only collects policy rollouts. If most policy steps receive negative reward, shorter episodes can improve return by collecting fewer negative rewards. This matched the observed failure mode: `ep_len` collapsed even when the reward model separated expert and policy states.

The successful sigmoid runs used a nonnegative reward range:

```text
reward_output_activation = "sigmoid"
reward_output_scale = 0.25
clamp = [0.0, 0.25]
```

That keeps the ranking:

```text
expert states: high positive reward, about 0.214
policy states: low positive reward, about 0.027
```

but removes the incentive to terminate early. The reward model still separates expert from policy by about `0.187`, while PPO sees longer episodes as collecting more low positive reward instead of more negative reward.

## Experiments

All runs below used:

- task: `Isaac-Imitation-G1-v0`
- envs: 2048 for full runs
- target: about 20M frames
- manifest: `./data/lafan1/manifests/g1_lafan1_manifest.json`
- key metric: `ep_len`

### Failed: Grouped Tanh, Stronger Regularization

Run log:

`/home/fwu91/Documents/Research/SkillLearning/IsaacLab-Imitation/logs/rlopt/ipmd/Isaac-Imitation-G1-v0/2026-04-26_19-01-38/rlopt.log`

Settings:

- grouped heads
- equal head weights
- pure estimated reward
- tanh output, clamp `[-0.25, 0.25]`
- `reward_l2_coeff=0.5`
- `reward_grad_penalty_coeff=2.0`

Result:

| Frames | `ep_len` | `est_r` | `exp_r` |
|---:|---:|---:|---:|
| 0.492M | 20.76 | -0.0034 | 0.0034 |
| 1.475M | 12.11 | -0.0013 | 0.0013 |
| 20.005M | 3.22 | -0.0001 | 0.0001 |

Interpretation:

The reward avoided hard saturation, but it collapsed toward zero and did not provide a useful control signal. `ep_len` got worse over training.

### Failed: Grouped Tanh, Lower Regularization

Run log:

`/home/fwu91/Documents/Research/SkillLearning/IsaacLab-Imitation/logs/rlopt/ipmd/Isaac-Imitation-G1-v0/2026-04-26_20-57-51/rlopt.log`

Settings:

- grouped heads
- equal head weights
- pure estimated reward
- tanh output, clamp `[-0.25, 0.25]`
- `reward_l2_coeff=0.05`
- `reward_grad_penalty_coeff=0.2`
- `reward_logit_reg_coeff=0.001`

Result:

| Frames | `ep_len` | `est_r` | `exp_r` |
|---:|---:|---:|---:|
| 0.492M | 20.78 | -0.2369 | 0.2378 |
| 0.983M | 14.37 | -0.2400 | 0.2404 |
| 1.475M | 10.02 | -0.2401 | 0.2410 |

Interpretation:

The model quickly learned a strong separator, but the policy reward was negative and near the lower clamp. `ep_len` collapsed early, so the run was stopped.

### Failed Control: Env-Reward IPMD Baseline

Run log:

`/home/fwu91/Documents/Research/SkillLearning/IsaacLab-Imitation/logs/rlopt/ipmd/Isaac-Imitation-G1-v0/2026-04-26_21-01-39/rlopt.log`

Settings:

- PPO used environment reward
- estimated reward was not used for PPO

Result:

| Frames | `ep_len` | `env_r` |
|---:|---:|---:|
| 0.492M | 19.65 | -0.0724 |
| 0.983M | 15.19 | -0.0600 |
| 1.475M | 9.19 | -0.0506 |

Interpretation:

This also collapsed by `ep_len`, so the scalar environment reward is not a useful primary metric for this experiment. It also reinforces that `ep_len` is the better live control-quality signal here.

### Successful: Grouped Sigmoid, Seed 2024

Run log:

`/home/fwu91/Documents/Research/SkillLearning/IsaacLab-Imitation/logs/rlopt/ipmd/Isaac-Imitation-G1-v0/2026-04-26_21-05-21/rlopt.log`

Settings:

- grouped heads
- equal head weights
- pure estimated reward
- sigmoid output, clamp `[0.0, 0.25]`
- `reward_l2_coeff=0.05`
- `reward_grad_penalty_coeff=0.2`
- `reward_logit_reg_coeff=0.001`

Result:

| Frames | `ep_len` | `est_r` | `exp_r` | `reward_diff` |
|---:|---:|---:|---:|---:|
| 0.492M | 17.86 | 0.0313 | 0.2138 | -0.1825 |
| 1.475M | 23.99 | 0.0277 | 0.2178 | -0.1901 |
| 4.915M | 34.78 | 0.0273 | 0.2142 | -0.1869 |
| 10.322M | 54.47 | 0.0273 | 0.2143 | -0.1870 |
| 14.746M | 72.28 | 0.0273 | 0.2143 | -0.1870 |
| 20.005M | 147.52 | 0.0273 | 0.2143 | -0.1870 |

### Successful Replication: Grouped Sigmoid, Seed 2025

Run log:

`/home/fwu91/Documents/Research/SkillLearning/IsaacLab-Imitation/logs/rlopt/ipmd/Isaac-Imitation-G1-v0/2026-04-26_21-35-49/rlopt.log`

Settings:

- same as seed 2024
- reward settings came from config defaults, not CLI overrides

Result:

| Frames | `ep_len` | `est_r` | `exp_r` | `reward_diff` |
|---:|---:|---:|---:|---:|
| 0.492M | 20.55 | 0.0312 | 0.2140 | -0.1827 |
| 1.475M | 22.26 | 0.0279 | 0.2177 | -0.1898 |
| 4.915M | 35.27 | 0.0273 | 0.2142 | -0.1869 |
| 10.322M | 57.73 | 0.0272 | 0.2143 | -0.1870 |
| 14.746M | 100.77 | 0.0273 | 0.2143 | -0.1870 |
| 20.005M | 133.99 | 0.0273 | 0.2143 | -0.1870 |

## Verification

Code checks:

```bash
conda run -n SkillLearning ruff check RLOpt/rlopt/agent/ipmd/ipmd.py RLOpt/rlopt/agent/ipmd/utils.py RLOpt/tests/test_ipmd_components.py
conda run -n SkillLearning pytest RLOpt/tests/test_ipmd_components.py -W ignore::DeprecationWarning
conda run -n SkillLearning ruff check source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/agents/rlopt_ipmd_cfg.py
conda run -n SkillLearning ruff format --check source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/agents/rlopt_ipmd_cfg.py
git diff --check
```

Smoke run:

```bash
PYTHONPATH=/home/fwu91/Documents/Research/SkillLearning/RLOpt:${PYTHONPATH:-} \
conda run -n SkillLearning python scripts/rlopt/train.py \
  --task Isaac-Imitation-G1-v0 \
  --algo IPMD \
  --num_envs 32 \
  --max_iterations 2 \
  --headless \
  --log_interval 768 \
  env.lafan1_manifest_path=./data/lafan1/manifests/g1_lafan1_manifest_single.json \
  agent.logger.backend= \
  agent.logger.exp_name=ipmd_grouped_sigmoid_cfg_smoke \
  agent.seed=2024 \
  agent.trainer.progress_bar=false
```

Smoke result:

| Iter | Frames | `r_step` | `ep_len` | `reward_diff` | `exp_r` |
|---:|---:|---:|---:|---:|---:|
| 1 | 768 | 0.1089 | 11.8919 | -0.0205 | 0.1114 |
| 2 | 1536 | 0.0755 | 15.1571 | -0.0467 | 0.1166 |

## Conclusions

1. Grouped reward heads are useful because they make the reward model more structured and diagnostic.
2. Stronger scalar gradient penalty alone did not fix the control signal.
3. The decisive change was shifting the learned policy reward from signed tanh to nonnegative sigmoid.
4. `ep_len` should remain the primary live metric for this set of experiments.
5. `env_r` became more negative during successful runs, so it should not be used alone to judge policy quality here.

## Next Steps

1. Run playback from the successful checkpoints and visually inspect motion quality.
2. Save representative videos for seed 2024 and seed 2025.
3. If playback is acceptable, run a longer continuation or larger-scale cluster run.
4. If playback shows long-but-bad survival, add task-specific diagnostics beyond `ep_len`, such as reference tracking error summaries or termination breakdowns.
