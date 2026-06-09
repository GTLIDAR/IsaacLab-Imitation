# Hierarchical Planning over Spectral Skills (GR00T-style HL + Bilinear LL)

## Context

The repo already trains a bilinear/spectral world model and a skill-conditioned
low-level policy via RLOpt's IPMD-Bilinear pipeline (`Isaac-Imitation-G1-Latent-v0`,
checkpointed at 500m / 1b frames). What's missing is a **high-level manager**
that picks skill sequences in a planning loop.

This plan describes a GR00T-style hierarchical planner stacked on top of the
existing low-level. The high-level is vision-language enriched; the low-level
is proprioceptive-only and unchanged. The "skill code" `z` is the bottleneck
between the two — it's a direction in the spectral coefficient space the
low-level was trained against, and the high-level learns to navigate that space.

## Current Offline Skill-DiffSR Direction (June 8, 2026)

The active near-term path is an offline, state-only high-level skill
representation learner. This is deliberately earlier than the vision-language
planner described below. The immediate goal is to learn a macro skill code `z`
from expert future windows and verify that it carries useful transition
information before wiring it into low-level control.

Current v1 objective:

```text
s_t, window_t = s_{t+1:t+W}, s_next = s_{t+W}
z_t = E_skill(s_t, window_t)
DiffSR: phi_W(s_t, z_t) = g(z_t)^T F_state(s_t)
target = s_next
```

Important constraints for future agents:

- This phase is offline high-level representation learning only.
- Do not modify `IPMD_BILINEAR` for this path; it has a separate trainer.
- Do not align `z_t` with the low-level latent command space yet.
- Do not detach `z_t`; DiffSR loss must backpropagate through `E_skill`.
- State features are exactly `expert_motion`, `expert_anchor_pos_b`, and
  `expert_anchor_ori_b`, using the same torso-relative construction as the
  `expert_window` observation group.

Implemented files:

| Surface | File |
|---|---|
| Isaac macro sampler | `source/isaaclab_imitation/isaaclab_imitation/envs/imitation_rl_env.py` |
| TorchRL wrapper hook | `source/isaaclab_imitation/isaaclab_imitation/envs/rlopt.py` |
| RLOpt trainer | `RLOpt/rlopt/agent/hl_skill_diffsr.py` |
| Training script | `scripts/rlopt/train_hl_skill_diffsr.py` |
| RLOpt tests | `RLOpt/tests/test_hl_skill_diffsr.py` |
| Isaac sampler tests | `source/isaaclab_imitation/test_reference_patch_env.py` |

Default v1 settings:

```text
W = 25
z_dim = 256
DiffSR feature_dim = 128
DiffSR embed_dim = 512
batch_size = 8192
num_updates = 2000
train_split = train
eval_split = eval
eval_trajectory_fraction = 0.1
```

Training command used for the first full local pass:

```bash
pixi run -e isaaclab env TERM=xterm PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1   python scripts/rlopt/train_hl_skill_diffsr.py   --headless   --task Isaac-Imitation-G1-Latent-v0   --num_envs 16   env.lafan1_manifest_path=./data/unitree/manifests/g1_unitree_dance102_manifest.json
```

The first full pass completed with the original all-trajectory sampler behavior
before train/eval splits were added. Final diagnostics were healthy:

```text
loss_real_z_eval     ~= 1.31
loss_shuffled_z_eval ~= 50.25
loss_zero_z_eval     ~= 34.18
z_effective_rank     ~= 130 / 256
z_dim_std_mean       ~= 0.63
```

A later Dance102 rerun with `--reconstruction_eval` still used all trajectories
because the Dance102 manifest loads as one trajectory. Its grouped sampled
reconstruction diagnostic showed the sampled target error was mostly in
`expert_motion`, not anchor terms:

```text
loss_real_z_eval      ~= 1.43
loss_shuffled_z_eval  ~= 49.16
loss_zero_z_eval      ~= 39.66
sample_recon_mse      ~= 0.414
expert_motion_mse     ~= 0.412
anchor_pos_mse        ~= 0.000095
anchor_ori_mse        ~= 0.0021
```

The first true multi-trajectory LAFAN1 run used
`data/lafan1/manifests/g1_lafan1_manifest.json` with a separate zarr cache at
`data/lafan1/g1_hl_diffsr`. Do not use the existing `data/lafan1/g1` cache for
this experiment unless it has been rebuilt; it was stale and contained only the
old `dance102/trajectory_0` tree. The LAFAN1 cache built 40 trajectories and
441,080 transitions. Final held-out diagnostics were:

```text
loss_real_z_eval      ~= 3.23
loss_shuffled_z_eval  ~= 139.39
loss_zero_z_eval      ~= 47.76
z_effective_rank      ~= 108.6 / 256
z_dim_std_mean        ~= 0.298
sample_recon_mse      ~= 2.384
expert_motion_mse     ~= 2.365
anchor_pos_mse        ~= 0.0059
anchor_ori_mse        ~= 0.0135
```

For these diagnostics, `sample_recon_mse` is not a reconstruction over the full
25-step future window. The encoder consumes `s_{t+1:t+W}`, but DiffSR samples
only the macro target state `s_{t+W}`. The metric is the per-sample sum of
squared error over the one-step macro target state, averaged over the batch;
`sample_recon_dim_mse` is the scalar mean over batch and dimensions. Grouped
metrics such as `expert_motion_mse` apply the same target-state calculation to
that feature slice only.

Ablations on the same 40-trajectory LAFAN1 cache:

| Run dir | W | z_dim | real | shuffled | zero | rank | sample_recon_mse | motion_mse | anchor_pos_mse | anchor_ori_mse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `2026-06-08_13-26-41` | 25 | 256 | 3.23 | 139.39 | 47.76 | 108.6 | 2.384 | 2.365 | 0.0059 | 0.0135 |
| `2026-06-08_13-47-11` | 25 | 128 | 3.43 | 140.51 | 53.29 | 76.9 | 2.927 | 2.905 | 0.0071 | 0.0148 |
| `2026-06-08_13-49-34` | 50 | 256 | 3.79 | 144.18 | 54.24 | 108.5 | 2.945 | 2.887 | 0.0220 | 0.0359 |
| `2026-06-08_13-52-17` | 100 | 256 | 4.05 | 144.61 | 49.48 | 109.9 | 3.867 | 3.664 | 0.1451 | 0.0575 |

All four runs pass the core real-vs-shuffled/zero and non-collapse checks. The
128D compression is viable but loses reconstruction quality. Longer horizons
remain useful through W=100, with expected growth in sampled target error,
especially anchor position at W=100.

A seeded repeat pass then reran the two most useful candidates on the same
40-trajectory LAFAN1 cache, with `--reconstruction_eval`, `--seed {0,1,2}`,
and the post-train held-out eval path. The trainer now seeds Python, Torch,
CUDA, and the Isaac env config. It also logs `norm_sample_recon_*`
diagnostics, where each target-state feature dimension's squared error is
divided by the held-out batch variance before averaging. These normalized
metrics make the anchor terms easier to compare with `expert_motion`.

Run dirs:

- `logs/hl_skill_diffsr/lafan1_w25_z256_seed0_norm`
- `logs/hl_skill_diffsr/lafan1_w25_z256_seed1_norm`
- `logs/hl_skill_diffsr/lafan1_w25_z256_seed2_norm`
- `logs/hl_skill_diffsr/lafan1_w50_z256_seed0_norm`
- `logs/hl_skill_diffsr/lafan1_w50_z256_seed1_norm`
- `logs/hl_skill_diffsr/lafan1_w50_z256_seed2_norm`

Seeded LAFAN1 aggregate, mean +/- sample std over three seeds:

| W | z_dim | real | shuffled | zero | rank | z_std | sample_recon_mse | sample_recon_dim_mse | norm_recon_dim_mse | norm_motion_dim_mse | norm_anchor_pos_dim_mse | norm_anchor_ori_dim_mse |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 256 | 3.26 +/- 0.24 | 144.10 +/- 2.18 | 50.65 +/- 2.97 | 107.6 +/- 0.1 | 0.289 +/- 0.006 | 2.90 +/- 0.93 | 0.0433 +/- 0.0139 | 0.0377 +/- 0.0019 | 0.0344 +/- 0.0027 | 0.102 +/- 0.016 | 0.0382 +/- 0.0042 |
| 50 | 256 | 3.39 +/- 0.24 | 150.18 +/- 1.79 | 52.24 +/- 4.38 | 108.4 +/- 0.3 | 0.280 +/- 0.003 | 2.79 +/- 0.18 | 0.0416 +/- 0.0027 | 0.0410 +/- 0.0059 | 0.0365 +/- 0.0034 | 0.127 +/- 0.076 | 0.0413 +/- 0.0036 |

Both seeded candidates pass the stage acceptance checks. Keep `W=25,z_dim=256`
as the clean baseline: it has comparable spectral separation and rank, slightly
lower normalized target-state error, and much less anchor-position variance.
`W=50,z_dim=256` remains viable when the downstream high level needs a slower
macro timescale, but do not claim it is better from raw `sample_recon_mse`
alone because that sampled diagnostic is stochastic and scale-sensitive.

A follow-up eval-only linear probe was added to test whether frozen `z` carries
information about the whole future window, not just the terminal macro target.
The probe fits closed-form ridge regressions on train-split batches and reports
held-out MSE for `z`, shuffled `z`, state-only, and mean baselines. This does
not update `E_skill` or DiffSR.

Probe commands used seed-0 checkpoints with `--window_probe_train_batches 8`,
`--window_probe_eval_batches 4`, and `--reconstruction_eval`:

```bash
pixi run -e isaaclab env TERM=xterm PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
  python scripts/rlopt/train_hl_skill_diffsr.py \
  --headless --task Isaac-Imitation-G1-Latent-v0 --num_envs 16 --seed 0 \
  --checkpoint logs/hl_skill_diffsr/lafan1_w25_z256_seed0_norm/checkpoints/latest.pt \
  --eval_only --reconstruction_eval --window_probe_eval \
  --window_probe_train_batches 8 --window_probe_eval_batches 4 \
  --output_dir logs/hl_skill_diffsr/lafan1_w25_z256_seed0_window_probe_eval \
  env.lafan1_manifest_path=./data/lafan1/manifests/g1_lafan1_manifest.json \
  env.dataset_path=./data/lafan1/g1_hl_diffsr
```

Window-probe held-out diagnostics:

| Checkpoint | W | z_dim | z_dim_mse | shuffled_z_dim_mse | state_dim_mse | mean_dim_mse | z_norm_dim_mse | z_first_dim_mse | z_mid_dim_mse | z_final_dim_mse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lafan1_w25_z256_seed0_norm` | 25 | 256 | 0.842 | 1.918 | 1.080 | 1.354 | 0.533 | 0.920 | 0.917 | 0.037 |
| `lafan1_w50_z256_seed0_norm` | 50 | 256 | 1.068 | 1.740 | 1.218 | 1.363 | 0.697 | 1.113 | 1.200 | 0.031 |

Interpretation: the original offline DiffSR objective makes `z` strongly
terminal-target aware. Real `z` is better than shuffled `z`, state-only, and
mean baselines over the full window, so it is not empty. However, most of the
linear probe advantage is concentrated at `s_{t+W}`; early and mid-window states
remain much harder. Do not treat the original checkpoint as a full-window motion
compressor.

On June 8, 2026, the trainer gained `encoder_window_mode`. The legacy `full`
mode preserves old checkpoint compatibility:

```text
z = E_skill(s_t, s_{t+1:t+W})
DiffSR target = s_{t+W}
```

The new `intermediate` mode hides the terminal target from the encoder:

```text
z = E_skill(s_t, s_{t+1:t+W-1})
DiffSR target = s_{t+W}
```

This is the cleaner transition-learning setup because `z` cannot directly copy
the exact target state that DiffSR is trained to predict. A full local pass used:

```bash
pixi run -e isaaclab env TERM=xterm PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
  python scripts/rlopt/train_hl_skill_diffsr.py \
  --headless --task Isaac-Imitation-G1-Latent-v0 --num_envs 16 --seed 0 \
  --output_dir logs/hl_skill_diffsr/lafan1_w25_z256_seed0_intermediate \
  --reconstruction_eval --window_probe_eval \
  --window_probe_train_batches 8 --window_probe_eval_batches 4 \
  --horizon_steps 25 --encoder_window_mode intermediate --z_dim 256 \
  env.lafan1_manifest_path=./data/lafan1/manifests/g1_lafan1_manifest.json \
  env.dataset_path=./data/lafan1/g1_hl_diffsr
```

Single-seed comparison against the old W25/z256 seed-0 full-window checkpoint:

| Mode | real | shuffled | zero | rank | z_std | sample_recon_mse | norm_recon_dim_mse | norm_motion_dim_mse | norm_anchor_pos_dim_mse | probe_z_dim_mse | probe_first | probe_mid | probe_final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 3.02 | 144.31 | 53.91 | 107.6 | 0.284 | 2.16 | 0.0358 | 0.0326 | 0.0999 | 0.842 | 0.920 | 0.917 | 0.037 |
| intermediate | 4.95 | 122.51 | 45.48 | 111.4 | 0.296 | 6.18 | 0.0726 | 0.0587 | 0.3987 | 0.785 | 0.874 | 0.862 | 0.080 |

Interpretation: `intermediate` is harder for endpoint DiffSR, as expected, and
sampled terminal reconstruction worsens. It still passes the main acceptance
checks by a large margin: real `z` is far better than shuffled and zero, and the
latent rank does not collapse. It also improves the full-window linear probe
(`0.785` vs `0.842`) and the first/mid-step probe errors, which is consistent
with `z` representing the transition rather than only the terminal target. Use
`intermediate` as the next research branch for valid transition learning. Keep
`full` only as a legacy baseline and for evaluating old checkpoints.

Acceptance signal for this stage remains:

```text
loss_real_z_eval < loss_shuffled_z_eval
loss_real_z_eval < loss_zero_z_eval
z_dim_std_mean is nontrivial
z_effective_rank does not collapse below 10% of z_dim
```

Current debugging additions:

- The Isaac sampler now supports deterministic `train` / `eval` trajectory
  splits for held-out diagnostics.
- The sampler exposes per-timestep macro-state feature slices.
- `--reconstruction_eval` adds stochastic DiffSR sampled reconstruction errors:
  `sample_recon_l1`, `sample_recon_mse`, `sample_recon_dim_mse`, plus grouped
  versions for `expert_motion`, `expert_anchor_pos_b`, and
  `expert_anchor_ori_b`.
- The same eval now logs variance-normalized target-state errors:
  `norm_sample_recon_dim_mse`,
  `norm_sample_recon_expert_motion_dim_mse`,
  `norm_sample_recon_expert_anchor_pos_b_dim_mse`, and
  `norm_sample_recon_expert_anchor_ori_b_dim_mse`.
- `--window_probe_eval` adds eval-only closed-form linear probes from frozen
  `z` and frozen state features to the flattened future window. Use this only as
  a diagnostic for whether `z` contains whole-window information.
- `--encoder_window_mode intermediate` hides `s_{t+W}` from `E_skill` while
  keeping `s_{t+W}` as the DiffSR target. This is the preferred mode for the
  target-excluding transition-learning branch.
- These reconstruction and probe numbers are diagnostics only; the training
  objective is still the diffusion loss.

Intermediate W25/z256 was repeated over seeds 0, 1, and 2 on June 8, 2026.
The three-seed post-train aggregate was:

| Metric | Mean | Std |
|---|---:|---:|
| `loss_real_z_eval` | 4.823 | 0.293 |
| `loss_shuffled_z_eval` | 124.792 | 2.037 |
| `loss_zero_z_eval` | 47.133 | 1.829 |
| `z_effective_rank` | 111.573 | 0.289 |
| `z_dim_std_mean` | 0.294 | 0.0019 |
| `sample_recon_mse` | 5.952 | 0.222 |
| `norm_sample_recon_dim_mse` | 0.0643 | 0.0079 |
| `window_probe_z_dim_mse` | 0.775 | 0.0087 |
| `window_probe_z_shuffled_dim_mse` | 1.983 | 0.0123 |
| `window_probe_state_dim_mse` | 1.085 | 0.0048 |
| `window_probe_z_step_first_dim_mse` | 0.877 | 0.0116 |
| `window_probe_z_step_mid_dim_mse` | 0.850 | 0.0106 |
| `window_probe_z_step_final_dim_mse` | 0.0788 | 0.0011 |

Conclusion: the target-excluding intermediate setup is stable enough to use as
our current frozen high-level encoder branch. It is not a solved whole-window
reconstruction model, but it passes the offline DiffSR acceptance checks with
healthy rank and strong real-vs-shuffled separation.

On the same date, a first low-level offline-online bridge was added:

- The Isaac env exposes `current_expert_macro_transition_batch(horizon_steps,
  env_ids=None)` for live-env macro windows aligned to the current trajectory
  cursor.
- RLOpt IPMD accepts `agent.ipmd.command_source=hl_skill`.
- `hl_skill` loads a frozen high-level DiffSR checkpoint, checks that checkpoint
  `z_dim` matches `agent.ipmd.latent_dim`, computes current expert-window
  latents online, holds them for `latent_steps_min/max`, and publishes them to
  `latent_command`.
- This path deliberately uses 256D commands directly. It does not silently map
  the high-level latent to the old 64D command space.

Smoke command that completed one online low-level rollout/update:

```bash
pixi run -e isaaclab env TERM=xterm WANDB_MODE=offline PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1   python scripts/rlopt/train.py   --headless --task Isaac-Imitation-G1-Latent-v0 --algo IPMD   --num_envs 16 --seed 0 --max_iterations 1 --log_interval 1   env.lafan1_manifest_path=./data/lafan1/manifests/g1_lafan1_manifest.json   env.dataset_path=./data/lafan1/g1_hl_diffsr   env.latent_command_dim=256   agent.ipmd.latent_dim=256   agent.ipmd.command_source=hl_skill   agent.ipmd.hl_skill_checkpoint_path=logs/hl_skill_diffsr/lafan1_w25_z256_seed0_intermediate/checkpoints/latest.pt   agent.ipmd.hl_skill_horizon_steps=25   agent.ipmd.latent_steps_min=25   agent.ipmd.latent_steps_max=25   agent.ipmd.reward_loss_coeff=0.0   agent.ipmd.reward_l2_coeff=0.0   agent.ipmd.reward_grad_penalty_coeff=0.0   agent.ipmd.reward_logit_reg_coeff=0.0   agent.ipmd.reward_param_weight_decay_coeff=0.0
```

Smoke result: the env observation table showed `latent_command (256,)`, the
policy built with 346 inputs including `latent_command`, and training completed
`frames=384/384` for one iteration. This only validates plumbing; it is not a
performance result.

A short follow-up local validation used the same configuration with
`--max_iterations 50` and completed `frames=19200/19200` in 79.79 seconds. Final
progress-line values were `r_step=-0.0950`, `ep_len=7.5300`, `r_ep=-0.7430`, and
`pi_loss=0.0071`. Treat these as stability/plumbing diagnostics only, not policy
quality.

June 8, 2026 update: the frozen `hl_skill` command path now mirrors the
phase convention already implemented for the patch VQ-VAE learner. With
`agent.ipmd.latent_learning.command_phase_mode=sin_cos`, the command published
to the low-level policy is `cat([z_256, sin(2*pi*phase), cos(2*pi*phase)])`.
For the current 256D high-level checkpoint, set `env.latent_command_dim=258` and
`agent.ipmd.latent_dim=258`; leave `agent.ipmd.latent_learning.code_latent_dim`
unset in Hydra because Isaac's config updater rejects integer overrides into a
field whose current value is `None`. The code width is inferred as `258 - 2 =
256`, and the sampler also validates that the checkpoint `z_dim` plus phase
width equals `agent.ipmd.latent_dim`.

A 4096-env LaFAN1 health pass completed with `--max_iterations 2`,
`frames=196608/196608`, `latent_command (258,)`, and a policy input width of
348. A full default-budget run was then launched with 4096 envs and no
`--max_iterations`; the config aligned the default IPMD budget to
`4,999,938,048` frames with `frames_per_batch=98,304`. The active run is:

```text
PID: 2216479
launcher log: logs/rlopt/launch/hl_skill_lafan1_4096_sincos_full_20260608_1638.log
run dir: logs/rlopt/ipmd/Isaac-Imitation-G1-Latent-v0/2026-06-08_16-38-16
checkpoint: logs/hl_skill_diffsr/lafan1_w25_z256_seed0_intermediate/checkpoints/latest.pt
```

Important operational note: the G1 IPMD config currently sets
`save_interval=100`, but RLOpt interprets `save_interval` in frames, not rollout
iterations. At 4096 envs this saves every rollout and will create hundreds of GB
of checkpoints over a full 5B-frame run. The active run was relaunched with
`agent.save_interval=100000000`, so it checkpoints roughly every 100M frames.
At first metrics it reached `iter=51/50862`, `frames=5013504/4999938048`,
`r_step=-0.0334`, `r_ep=-0.1377`, `pi_loss=-0.0088`, and `fps=37847.1072`.

Next recommended steps:

1. Run a longer low-level frozen-encoder job with `command_source=hl_skill`,
   still using env rewards first and keeping IPMD reward-model updates disabled.
2. Log frozen-command diagnostics during low-level training, especially command
   RMS/std/effective rank, command refresh cadence, episode return, and tracking
   reward terms.
3. If online training is unstable, improve the high-level objective before
   scaling: try multi-horizon DiffSR or an auxiliary terminal/window objective.
4. Only add a 256D-to-64D adapter if we explicitly train and evaluate that
   adapter. Do not treat it as an implicit compatibility shim.

## Longer-Term VL Planner Context

The sections below describe the larger GR00T-style hierarchical planner idea.
Treat them as roadmap context, not the active v1 implementation path.

## Two-Timescale Planning and Control

The central decomposition is a two-timescale hierarchy. The high-level planner
operates at a coarse temporal scale and chooses a skill command or subgoal every
`K` environment steps, while the low-level controller runs at the native control
rate and stabilizes the robot toward the currently active command.

Let `t` denote the low-level control timestep and let `m` denote the high-level
decision index, with `t = mK`. The high-level observation may include proprio,
vision, and task context:

```math
o_m^{HL} = \left(x_{mK}, I_{mK}^{ego}, I_{mK}^{tp}, c\right),
\qquad
h_m = E_\theta(o_m^{HL}).
```

Here `x_t` is the robot proprioceptive state, `I_t^{ego}` and `I_t^{tp}` are the
ego and third-person images, `c` is optional task or language context, and
`h_m` is the high-level latent state used for planning. The high-level action is
not a torque or joint target. It is a motor-actionable latent command:

```math
z_m \in \mathcal{Z},
```

where `\mathcal{Z}` is the skill/subgoal space understood by the low-level
policy. Depending on the experiment, `z_m` can be interpreted either as a skill
command sampled from a learned command manifold, or as an encoded subgoal state
that the low-level should reach over the next `K` steps.

The low-level policy is goal-conditioned control:

```math
a_t \sim \pi_\phi(a_t \mid x_t, z_m),
\qquad
t \in \{mK, \ldots, (m+1)K - 1\}.
```

Thus the low-level remains blind: it only consumes proprioception and the active
latent command. All camera and task-level reasoning is pushed into the
high-level planner. The shared interface between the two layers is the command
space `\mathcal{Z}`, not the raw observation space.

The high-level model predicts the coarse effect of executing a low-level command
for one command period:

```math
\hat{h}_{m+1} = T_\psi(h_m, z_m).
```

At planning time, the manager searches over a horizon of high-level commands
`z_{m:m+H-1}` and rolls the learned coarse dynamics forward:

```math
\hat{h}_{m+i+1} = T_\psi(\hat{h}_{m+i}, z_{m+i}),
\qquad
\hat{h}_m = h_m.
```

Given a sequence of goal embeddings `h^g_{m+1:m+H}`, planning solves

```math
z^\star_{m:m+H-1}
= \arg\min_{z_{m:m+H-1}}
\sum_{i=1}^{H}
d\!\left(\hat{h}_{m+i}, h^g_{m+i}\right)
+ \lambda \sum_{i=0}^{H-1} \Omega(z_{m+i}),
```

where `d` measures goal mismatch in the high-level latent space and
`\Omega(z)` keeps candidate commands on the valid skill manifold. Only the first
planned command `z_m^\star` is executed; after `K` low-level steps, the system
re-encodes the current observation and replans.

This receding-horizon structure gives the desired division of labor:

- the **high level** performs long-horizon semantic and visual reasoning by
  selecting skill commands or subgoals;
- the **low level** performs fast whole-body stabilization and tracking
  conditioned on the active command;
- the **shared latent command space** makes the two layers compositional, because
  the high-level planner only proposes commands that the low-level controller was
  trained to interpret.

User-confirmed design choices (this design doc is the converged form after
iterative discussion):

| Concern | Choice |
|---|---|
| HL fusion of proprio + vision | late concat → MLP (GR00T-style) |
| HL state inputs | proprio + ego camera + third-person camera |
| VL backbone | frozen DINOv2 + small trainable adapter |
| `T_k` target | single fused `h_{t+k}` tensor |
| `T_k` parameterisation | plain MLP first; bilinear `h + W(h)·G(z)` as Phase-2 swap |
| Goal source | reference clip frames → proprio + rendered ego + rendered third-person |
| Rendering stage | live during rollout collection |
| Planner | CEM over `z_{1:H}` |
| Low-level | unchanged `BilinearPolicyHead(s_proprio, z)` |
| MVP success criterion | planned-z beats random-z on dance102 tracking |

## Architecture

```
Inputs at time t (HIGH-LEVEL):
  s_proprio_t  (existing observation)
  image_ego_t  (NEW: head-mounted camera)
  image_tp_t   (NEW: fixed third-person camera)

High-level state encoder:
  F_t = pool over embed_dim of  F(s_proprio_t)               # frozen bilinear F
  v_t = adapter( concat[ DINOv2(image_ego_t),
                         DINOv2(image_tp_t) ] )              # DINOv2 frozen; adapter trained
  h_t = MLP_fuse([F_t, v_t])                                  # late concat → MLP

Goal encoding (clip frame at offset i·k from current time):
  F_g^i = pool(F(s_proprio_g^i))                              # from LAFAN1 reference (free)
  v_g^i = adapter( concat[ DINOv2(rendered_ego_at_g^i),
                           DINOv2(rendered_tp_at_g^i) ] )
  h_g^i = MLP_fuse([F_g^i, v_g^i])                            # same fusion MLP

High-level world model:
  T_k(h, z) → h_{t+k}                                         # plain MLP for MVP
  # Phase 2: h_{t+k} = h + W(h)·G(z)  (bilinear / spectral)

Planner (re-invoked every k env steps):
  CEM over z_{1:H} sequences:
    unroll  ĥ_{i+1} = T_k(ĥ_i, z_i)
    cost    = Σ_i ‖ĥ_i − h_g^i‖²
  emit z* = first skill of best elite
  call env.set_agent_latent_command(z*)

Low-level (unchanged):
  BilinearPolicyHead(s_proprio, z*) for k env steps
```

Parameter ownership:

| Component | Trainable? | Trained when |
|---|---|---|
| Bilinear `F`, `g`, low-level policy | frozen | already done (IPMD checkpoint) |
| DINOv2 backbone | frozen | n/a |
| VL adapter | trainable | jointly with `T_k` in one offline stage |
| `MLP_fuse` | trainable | jointly with `T_k` |
| `T_k` | trainable | one offline stage on labeled rollouts |
| CEM planner | parameter-free | online at eval |

## The spectral hook

The "transition learned with spectral decomposition" framing shows up in two
places:

1. **At the low level (already exists).** `BilinearPolicyHead` consumes `z` via
   `F(s)z`; skills are directions in the spectral coefficient space of the
   bilinear factorisation `φ(s,a) = g(a)ᵀ F(s)` (`RLOpt/.../ipmd/module.py:138`).
2. **At the high level (Phase 2 swap).** `T_k` can be re-parameterised as
   `h_{t+k} = h + W(h)·G(z)` — the coarse-grained, h-space analog of the
   low-level bilinear factorisation. The plain-MLP `T_k` in the MVP is just the
   warm start; the bilinear swap is one-line because the external signature is
   the same.

## MVP scope ("beat random-z on dance102")

The smallest end-to-end experiment that would justify continued investment.

Pipeline:

1. **One-time data collection** — re-roll out the existing scratch_1b bilinear
   checkpoint on dance102 with both cameras enabled and posterior-inferred `z`
   saved per window. ~30k frames/clip × 2 cameras × 224² ≈ 5–15 GB per clip.
   Expect 2–4× slowdown vs. cameraless rollout.
2. **One-time training** — jointly fit VL adapter + `MLP_fuse` + `T_k` on
   `(h_t, z, h_{t+k})` triples. Small offline run on a single GPU.
3. **Two evals on held-out frames** — identical env/checkpoint/cameras, only
   the latent-command controller differs:
   - baseline: existing `RandomLatentCommandSampler`
   - this work: `HierarchicalSkillController` + `CEMSkillPlanner`
4. **Metric**: per-step tracking error, averaged over ≥5 seeds. Pass = planned
   beats random with statistical significance (paired t-test, p < 0.05).

Explicitly out of scope for MVP:

- Posterior-z baseline (queued as follow-up #1)
- Cross-clip generalisation
- Bilinear `T_k` swap (queued as follow-up #2)
- Language goals
- Learned manager network (BC / RL on top of CEM)

## Near-Term Low-Level Experiment: Future-Goal AE Commands

Before adding cameras or a learned high-level manager, test whether the
low-level bilinear policy can use a slower, continuous goal embedding as its
action:

1. Register `Isaac-Imitation-G1-Latent-Goal-v0`, which exposes a single
   future reference goal state through the `expert_goal` observation group.
2. Encode `expert_goal.{expert_motion, expert_anchor_pos_b, expert_anchor_ori_b}`
   with the continuous `patch_autoencoder` posterior.
3. Hold the inferred latent command for `k` env steps via
   `ipmd.latent_learning.posterior_command_period=k`.
4. Use a 128D AE latent command and a 64D bilinear feature basis, forcing the
   `BilinearPolicyHead` to learn a command projector from goal embedding to
   spectral direction.

First sweep:

| Run | Goal offset | Command period | Latent dim | SR dim | Notes |
|---|---:|---:|---:|---:|---|
| goal25_period25 | ~0.5s at 50 Hz | 25 | 128 | 64 | First debug / baseline |
| goal50_period25 | ~1.0s | 25 | 128 | 64 | Longer lookahead, same control cadence |
| goal100_period50 | ~2.0s | 50 | 128 | 64 | Slower high-level action |

The expected failure mode is command-space mismatch: if reconstruction-only
latents are useful but not aligned with the bilinear basis, the learned
projector should help. If even this fails, the low-level may need to train with
explicitly sampled manager actions rather than posterior-inferred references.

## Stages

### Stage 0 — Camera env additions (read-only env config change)

Modify `imitation_g1_latent_env_cfg.py` to add ego-centric and third-person
camera terms (IsaacLab `TiledCameraCfg` with the existing `scene` parameters).
Cameras emit RGB at 224² (resize later if needed).

### Stage 1 — Data collection with cameras + z labels

Extend `scripts/rlopt/record_policy_rollout.py`:
- write camera buffers to NPZ alongside existing `qpos`/`qvel`/etc.
- additionally run `_latent_learner.infer_batch_latents` (or VQ posterior) per
  window and save the inferred `z`
- preserve the existing schema so non-VL consumers don't break

Run once on the dance102 clip(s).

### Stage 2 — High-level encoder + transition

New module `RLOpt/rlopt/agent/imitation/hl_encoder.py`:
- `VLAdapter(nn.Module)` — concat DINOv2 features → trainable MLP → `v_t`
- `HighLevelEncoder(nn.Module)` — owns `MLP_fuse([pool(F(s_p)), v_t]) → h_t`
- caches `F` reference (frozen) and DINOv2 reference (frozen)

New module `RLOpt/rlopt/agent/imitation/latent_transition.py`:
- `LatentTransition(nn.Module)` — plain MLP `(h_t, z) → h_{t+k}` for MVP
- abstract signature so bilinear variant can drop in without touching callers

New script `scripts/rlopt/train_hl_world_model.py`:
- load NPZs from Stage 1, build `(h_t, z, h_{t+k})` triples
- joint MSE loss on `T_k(h_t, z) − h_{t+k}`
- `F`, DINOv2, low-level head all frozen
- logs to `logs/hl_world_model/`

### Stage 3 — Skill space abstraction

New module `RLOpt/rlopt/agent/imitation/skill_space.py`:
- `SkillSpace` ABC with `sample(n)` and `to_latent_command(z)`
- `DiscreteVQSkillSpace` — wraps an existing VQ codebook (or a quick k-means
  on Stage-1 inferred `z` as a stand-in)
- `ContinuousSphericalSkillSpace` — `F.normalize(randn(n, D), dim=-1)` (for
  future continuous mode)

MVP uses the discrete variant.

### Stage 4 — CEM planner

New module `RLOpt/rlopt/agent/imitation/skill_planner.py`:
- `CEMSkillPlanner(skill_space, latent_transition, hl_encoder, plan_horizon, n_samples, n_elites, n_iters)`
- `plan(h_t, h_goals) → z*`
- vectorised over `n_samples × num_envs`

### Stage 5 — Hierarchical controller

New module `RLOpt/rlopt/agent/imitation/hierarchical_commands.py`:
- `HierarchicalSkillController` mirrors `LatentCommandController` public surface
  (drop-in compatible with `LatentCommandCollectorPolicy`)
- per-env countdown; re-plan at boundary or on `done`
- calls existing `env.set_agent_latent_command(z*)`

### Stage 6 — Eval script

New script `scripts/rlopt/play_hierarchical.py`:
- args: `--checkpoint`, `--hl_ckpt`, `--task Isaac-Imitation-G1-Latent-v0`,
  `--motion_manifest`, `--baseline {planned,random}`
- runs each baseline over N seeds × full dance102 held-out window
- logs per-step tracking error and overall reward
- emits a small CSV for the paired t-test

## Critical files

**New (in this work):**
- `RLOpt/rlopt/agent/imitation/hl_encoder.py` — VL adapter + fusion encoder
- `RLOpt/rlopt/agent/imitation/latent_transition.py` — `T_k` MLP
- `RLOpt/rlopt/agent/imitation/skill_space.py` — pluggable skill abstraction
- `RLOpt/rlopt/agent/imitation/skill_planner.py` — CEM
- `RLOpt/rlopt/agent/imitation/hierarchical_commands.py` — manager loop
- `scripts/rlopt/train_hl_world_model.py` — offline training
- `scripts/rlopt/play_hierarchical.py` — evaluation

**Modified:**
- `source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/imitation_g1_latent_env_cfg.py`
  — add ego + third-person `TiledCameraCfg` terms
- `scripts/rlopt/record_policy_rollout.py` — emit camera frames + inferred `z`

**Unchanged (reuse map):**
- `BilinearSR.encode_state` (`RLOpt/.../ipmd/module.py:132`) — proprio side of `h`
- `BilinearPolicyHead.forward` (`RLOpt/.../ipmd_bilinear.py:262`) — low-level
- `env.set_agent_latent_command` (`imitation_rl_env.py:1437`) — env hook
- `LatentCommandController.publish_latents_to_env` (`latent_commands.py:213`)
- `PatchVQVAELatentLearner.infer_batch_latents` — for discrete skill labelling

## Verification

1. **Shape smoke**: load checkpoint + untrained HL stack; confirm
   `h_t = MLP_fuse([pool(F(s)), adapter([DINOv2(ego), DINOv2(tp)])])` has the
   expected shape end-to-end on a single GPU batch.
2. **Stage-1 sanity**: re-roll one dance102 clip; spot-check that camera NPZ
   arrays render correctly (eyeball saved PNG samples) and `inferred_z` has
   non-trivial variance across frames.
3. **Stage-2 fit**: training loss for `T_k` drops below an "identity" baseline
   (`h_{t+k} = h_t`) on held-out windows. If it doesn't, the VL adapter or
   pooling choice needs revisiting.
4. **Stage-4 toy**: synthetic test where `h_goals` are pulled from a recorded
   rollout a few `k`-steps ahead; CEM's final cost should be strictly lower
   than random skill sequences (≥ 2× margin).
5. **End-to-end (MVP exit criterion)**: `play_hierarchical.py` on held-out
   dance102 frames, planned beats random in mean tracking error over ≥ 5 seeds
   with paired t-test p < 0.05.
6. **Regression**: existing random-z rollout path still works unchanged
   (`set_agent_latent_command` API and observation spec preserved).

Smoke command (illustrative):

```bash
pixi run -e isaaclab python scripts/rlopt/play_hierarchical.py \
    --task Isaac-Imitation-G1-Latent-v0 \
    --checkpoint <bilinear_scratch_1b.pt> \
    --hl_ckpt logs/hl_world_model/latest/ckpt.pt \
    --baseline planned \
    env.lafan1_manifest_path=./data/unitree/manifests/g1_unitree_dance102_manifest.json
```

## Open questions (resolve as part of Stage 0/1)

1. **Camera resolution**: 224² (DINOv2-native) vs 84² (cheap). Start at 224
   and downsample if render cost dominates.
2. **Goal pose rendering** in IsaacLab: cheapest path is a second invisible
   "ghost" robot articulation set to the reference pose, re-rendered through
   the same cameras. Worth checking whether IsaacLab supports two robots in
   the same scene without physics interference (it does via
   `replicate_physics=False` or visualisation-only assets).
3. **Pooling of `F(s)`**: `F(s).mean(dim=1) ∈ ℝ^D` is the default. If Stage-3
   fit is poor, try flattened `F(s) ∈ ℝ^{E·D}`.
4. **k (skill duration)**: start at `k=5` env steps (matches existing
   `latent_steps_min/max` defaults in `RandomLatentCommandSampler`).
5. **Plan horizon H**: start at 6–8 high-level steps (~30–40 env steps).

## Follow-ups (post-MVP, in priority order)

1. Beat the existing posterior-z baseline (`_inject_posterior_latent_command`)
   — proves planning helps over greedy posterior inference.
2. Swap `T_k` to bilinear `h + W(h)·G(z)` — closes the "spectral decomposition
   all the way through" loop.
3. Cross-clip generalisation: train on dance102, evaluate on another LAFAN1 clip.
4. Language-conditioned goals via the VL backbone's text tower.
5. Manager network distilled from CEM (faster inference; optional RL
   fine-tune).
