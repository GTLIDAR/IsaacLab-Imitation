# IPMD-Bilinear Inverse-RL Reward — Work In Progress

Status as of 2026-05-23. Branch: `claude/nice-liskov-f6502b` (parent), `claude/ipmd-bilinear-irl` (RLOpt submodule).

## Goal

Re-enable the IPMD reward estimator on top of the bilinear spectral representation, so PPO can train on a mix of environment reward and a learned IRL reward instead of env reward only.

Constraints from the user:
1. Reward should be linear in the bilinear features `f(s)` and the action `a`.
2. `r >= 0` (PPO trajectory-based — negative reward rewards early termination).
3. `r <= R_max` (large reward blows up advantages → policy explodes).
4. Reward estimator updated much slower than the policy.
5. Validate locally on dance102 with the action-labeled rollout dataset.

## Design (post codex review)

### Reward parameterization

```
phi(s, a)   = g(a)^T F_ema(s) / sqrt(E)        # (B, D), both detached
phi_norm    = LayerNorm(phi)                   # affine-free
r_logit     = theta^T phi_norm + b             # theta ∈ R^D, b scalar
r(s, a)     = r_floor + (R_max - r_floor) * sigmoid(r_logit)
```

- Reuses the existing bilinear `g(a)` action encoder and `state_net_ema` for `F(s)`; both detached so reward gradients touch only `theta` and `b` (≈65 params).
- LayerNorm + sigmoid + floor handle bound/saturation/early-termination concerns in one place.
- No latent-`z` conditioning yet (variant A first; variant B = "use `z` as theta" deferred).

### IPM loss

Margin-softplus surrogate instead of the unbounded `mean(r_pi) − mean(r_exp)`:

```
gap        = mean(r_pi) − mean(r_exp) + reward_margin * R_max
loss_main  = softplus(gap / (reward_temp * R_max)) * (reward_temp * R_max)
```

Plus existing IPMD regularizers: `reward_l2`, `reward_logit_reg`, `reward_param_weight_decay`. Gradient penalty is set to 0 because `phi` is detached.

### Update ordering

`_prepare_rollout_rewards` deep-copies the reward estimator into a frozen snapshot, PPO scores the whole rollout against the snapshot, then `_run_reward_updates` trains the live params after the PPO epoch loop finishes. Avoids the within-epoch GAE/reward drift the original IPMD path has.

### Slow updates (existing knobs)

| Knob | Value |
| ---- | ----- |
| `ipmd.reward_lr` | `1e-4` (policy lr = `1e-3`) |
| `ipmd.reward_update_interval` | `4` (one reward step per 4 PPO updates) |
| `ipmd.reward_update_warmup_updates` | `200` |
| `ipmd.reward_updates_per_policy_update` | `1` |

No theta EMA — YAGNI for 65 params with the above schedule.

### Mixing (start small, ramp)

| Stage | `env_reward_weight` | `est_reward_weight` |
| ----- | ------------------- | ------------------- |
| Default in committed config | `1.0` | `0.1` |
| Bump if stable | `1.0` | `0.3` |
| Mixed | `0.5` | `0.5` |
| Pure IRL | `0.0` | `1.0` |

## Files changed

### RLOpt submodule (branch `claude/ipmd-bilinear-irl`, commit `e8710f5` + uncommitted follow-up)

- `RLOpt/rlopt/agent/ipmd/module.py` — new `BilinearLinearReward(nn.Module)`.
- `RLOpt/rlopt/agent/ipmd/ipmd_bilinear.py` — added bilinear-reward fields to `BilinearSRConfig`; overrode `_construct_reward_estimator`, `_reward_from_td`, `_backward_reward_terms`, `_prepare_rollout_rewards` on `IPMDBilinear`; added `_bilinear_phi_from_td` helper.
- Follow-up fixes from the Dance102 smoke: expert sampler now requests bilinear obs keys + `expert_action`; `_bilinear_phi_from_td` handles arbitrary rollout batch ranks; `bilinear_linear` reward updates skip the inherited pre-PPO update path and run after PPO consumes the frozen reward snapshot.

### Parent worktree (branch `claude/nice-liskov-f6502b`, uncommitted)

- `source/isaaclab_imitation/.../agents/rlopt_ipmd_bilinear_cfg.py` — flipped `_G1ImitationRLOptIPMDBilinearBaseConfig` to enable estimated rewards, set sigmoid/clamps to `[0, 1]`, set slow reward schedule, set `bilinear.reward_kind = "bilinear_linear"` + new reward fields. All bilinear subclasses inherit.
- `RLOpt` submodule pointer is dirty (points at unpublished `e8710f5` plus local RLOpt edits); not yet bumped.

## Validation status

- Module-level sanity check (loading from worktree RLOpt):

  ```text
  reward shape: torch.Size([8, 1]) range: 0.525 0.525
  reward_kind default: bilinear_linear
  reward_max: 1.0 | reward_floor: 0.05
  ```

  `r = 0.525` is `0.05 + 0.95 · sigmoid(0)` ✓.

- **Dance102 smoke run completed locally (2026-05-23 16:38).** Passing smoke used `SL_irl`, 256 envs, 2 rollout iterations, 12,288 frames total, logger disabled, and the parent checkout's action-labeled Dance102 manifest because this Claude worktree lacks the generated action-label NPZ. Console summary:

  ```text
  iter=1/2 | frames=6144/12288 | r_step=-0.0157 | ep_len=16.4568 | r_ep=-1.2040 | reward_diff=0.0000 | exp_r=0.5250 | fps=2546.8555
  iter=2/2 | frames=12288/12288 | r_step=-0.0230 | ep_len=29.6600 | r_ep=-2.2698 | reward_diff=0.0000 | exp_r=0.5250 | fps=4005.9660
  Training time: 5.47 seconds
  ```

  Smoke-only overrides disabled reward warmup and used one PPO epoch so the reward update path was exercised quickly.

- `conda run -n SL_irl ruff check RLOpt/rlopt/agent/ipmd/ipmd_bilinear.py source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/agents/rlopt_ipmd_bilinear_cfg.py` passes.

- **128-env Dance102 validation completed locally (2026-05-23 17:20).** Command used the real slow reward schedule, `agent.logger.backend=null`, `agent.save_interval=0`, `--num_envs 128`, `--max_iterations 1628`, and minibatches resized to `3072 = 24 * 128`. It ran 5,001,216 frames in 2,113.44 seconds. Selected summary points:

  ```text
  iter=40/1628   | frames=122880/5001216  | r_step=0.0260 | ep_len=31.9600  | r_ep=-0.8669 | reward_diff=0.0000  | exp_r=0.0000
  iter=520/1628  | frames=1597440/5001216 | r_step=0.0507 | ep_len=61.0900  | r_ep=0.5002  | reward_diff=-0.0140 | exp_r=0.0980
  iter=920/1628  | frames=2826240/5001216 | r_step=0.0547 | ep_len=75.5900  | r_ep=0.9882  | reward_diff=-0.0136 | exp_r=0.0946
  iter=1240/1628 | frames=3809280/5001216 | r_step=0.0521 | ep_len=100.8200 | r_ep=1.1867  | reward_diff=-0.0227 | exp_r=0.1016
  iter=1628/1628 | frames=5001216/5001216 | r_step=0.0524 | ep_len=113.6800 | r_ep=1.5157  | reward_diff=-0.0285 | exp_r=0.1069
  ```

  Interpretation: integration is stable at 128 envs for 5M frames. Episode length improved steadily, estimated/expert reward metrics moved slowly and stayed far from the `[0, 1]` bounds, and there was no visible reward/advantage explosion in the local summary stream. Still compare against an env-only baseline before calling the learning effect positive.

- **Follow-up reward-weight/proxy investigation (2026-05-23 evening).** The simple weight sweep showed larger IRL weights are not automatically better, and the env-reward proxy work added calibration diagnostics rather than replacing the main path yet. Summary:

  ```text
  est_reward_weight=0.3, 128 envs, stopped at 860k frames:
    iter=280/1628 | r_step=0.0990 | ep_len=51.0900 | r_ep=-1.5475 | reward_diff=-0.0087 | exp_r=0.0964
    => dominated by 0.1; high positive estimated reward swamps env reward early.

  est_reward_weight=0.15, 128 envs, 1.23M frames:
    iter=400/400 | r_step=0.0603 | ep_len=54.8000 | r_ep=-0.1416 | reward_diff=-0.0107 | exp_r=0.0950
    => worse than 0.1 at the same frame count (`r_ep=0.2679`).

  env-only screen, 128 envs, 1.23M frames:
    iter=400/400 | r_step=0.0142 | ep_len=31.9600 | r_ep=0.4253
    => better early env return, but much shorter episodes than 0.1.

  est_reward_weight=0.05, 128 envs, stopped at 983k frames:
    iter=320/400 | r_step=0.0266 | ep_len=41.8800 | r_ep=0.1646
    => middle ground but not clearly better than env-only/0.1.
  ```

  Added investigation-only env-reward proxy instrumentation in RLOpt:

  - `ipmd.reward_env_proxy_loss_coeff`: optional BCE loss that fits reward logits to normalized env-reward percentiles.
  - `ipmd.reward_env_proxy_target`: `batch_percentile` or `running_percentile`.
  - `ipmd.reward_env_proxy_buffer_size` / `reward_env_proxy_min_buffer_size` for running CDF targets.
  - Rollout diagnostics now include scale-free `reward/est_vs_env` percentile/rank metrics.
  - Running-percentile buffers are saved/restored in IPMD checkpoints.

  Proxy calibration screens used PPO env reward only (`est_reward_weight=0`) so they measured reward-head calibration without perturbing policy. Results:

  ```text
  batch-percentile proxy, 1.23M frames:
    proxy_rho peaked around 0.69 early, then decayed; final iter=400 r_ep=0.4655, proxy_rho=0.5200

  running-percentile proxy, 1.23M frames:
    similar behavior; final iter=400 r_ep=0.4655, proxy_rho=0.4737

  proxy checkpoint + est_reward_weight=0.05, reward_floor=0.05, +1.23M frames:
    final iter=400 r_ep=0.7056, ep_len=57.45, proxy_rho=0.4320

  proxy checkpoint + est_reward_weight=0.1, reward_floor=0.05, +614k frames:
    final iter=200 r_ep=0.2809, ep_len=54.53, proxy_rho=0.4196
    => too much proxy reward hurts true env return.

  running-proxy checkpoint + est_reward_weight=0.05, reward_floor=0.0, +614k frames:
    final iter=200 r_ep=0.4972, ep_len=48.56, proxy_rho=0.3512
    => zero floor helps vs floor=0.05 at comparable continuation points, but still does not beat the original 0.1 adversarial run.
  ```

  Current interpretation: bilinear features do contain env-reward ranking signal (`proxy_rho` reliably reaches ~0.6 early), but the live one-head proxy reward is non-stationary and has not beaten the original conservative adversarial `est_reward_weight=0.1` path. For proxy-shaped PPO, use `bilinear.reward_floor=0.0`; the default `0.05` floor behaves like an unconditional survival bonus.


## Environment setup

Cloned `SL` → `SL_irl` and re-pointed the editable `rlopt` install at the worktree's `RLOpt`. This was necessary because `SL` uses a scikit-build-core editable install that loads via `_rlopt_editable.pth` and resolves before `PYTHONPATH`, so changes inside the worktree's `RLOpt/` submodule are invisible to it.

```bash
# What was done
conda create -n SL_irl --clone SL -y
/home/fwu91/miniforge3/envs/SL_irl/bin/pip uninstall rlopt -y
/home/fwu91/miniforge3/envs/SL_irl/bin/pip install scikit-build-core  # build backend
cd .claude/worktrees/nice-liskov-f6502b/RLOpt
/home/fwu91/miniforge3/envs/SL_irl/bin/pip install -e . --no-deps --no-build-isolation
```

Now `import rlopt` in `SL_irl` resolves to the worktree's RLOpt. `isaaclab_imitation` still resolves to the parent checkout unless this worktree's package source is prepended:

```bash
PYTHONPATH=/home/fwu91/Documents/SL/IsaacLab-Imitation/.claude/worktrees/nice-liskov-f6502b/source/isaaclab_imitation
```

The `IsaacLab/` and `ImitationLearningTools/` submodule directories are not initialized in this Claude worktree. The local smoke used the installed Isaac Lab at `/home/fwu91/Documents/SL/IsaacLab` through `python scripts/rlopt/train.py`; initialize submodules before relying on the in-repo `./IsaacLab/isaaclab.sh` launcher.

## Reproduction (when ready to resume)

Hydra root is `{env: ..., agent: ...}`. Use `agent.*` for RLOpt config overrides (e.g. `agent.ipmd.est_reward_weight=0.1` or `agent.collector.total_frames=...`). Prefer `--num_envs` and `--max_iterations` for smoke budgets because `train.py` rescales `collector.frames_per_batch` after the env count is known.

Passing short smoke command:

```bash
conda run -n SL_irl env \
  PYTHONPATH=/home/fwu91/Documents/SL/IsaacLab-Imitation/.claude/worktrees/nice-liskov-f6502b/source/isaaclab_imitation \
  HYDRA_FULL_ERROR=1 \
  python scripts/rlopt/train.py \
  --task Isaac-Imitation-G1-Latent-v0 --algo IPMD_BILINEAR --headless \
  --num_envs 256 --max_iterations 2 --log_interval 1 \
  env.lafan1_manifest_path=/home/fwu91/Documents/SL/IsaacLab-Imitation/data/unitree/manifests/g1_unitree_dance102_rlopt_ipmd_500m_actions_manifest.json \
  env.reconstructed_reference_action=False \
  agent.logger.backend=null agent.save_interval=0 \
  agent.loss.epochs=1 agent.loss.mini_batch_size=6144 \
  agent.ipmd.expert_batch_size=6144 \
  agent.ipmd.reward_update_warmup_updates=0 \
  agent.ipmd.reward_update_interval=1 \
  agent.ipmd.est_reward_weight=0.1
```

For the intended ~5M-frame local run, keep the default slow reward schedule and set minibatches to one rollout batch. Known-good budgets:

- 128 envs: `--max_iterations 1628`, `agent.loss.mini_batch_size=3072`, `agent.ipmd.expert_batch_size=3072` (`24 * 128 * 1628 = 5,001,216` frames).
- 256 envs: `--max_iterations 814`, `agent.loss.mini_batch_size=6144`, `agent.ipmd.expert_batch_size=6144` (`24 * 256 * 814 = 5,001,216` frames).

Use `agent.logger.backend=null` unless WandB is logged in, and use `agent.save_interval=0` for validation-only runs that should not leave checkpoints.

Notes:

- `env.reconstructed_reference_action=False` forces the expert sampler to use the recorded actions from the manifest's NPZ (the 500M-step IPMD-latent policy labels) instead of synthesizing actions from next-pose differences. See `source/isaaclab_imitation/.../envs/imitation_rl_env.py:2843-2862` for the precedence logic.
- `agent.ipmd.est_reward_weight` ramps the IRL contribution; start at `0.1`.
- This worktree has the manifest but not `data/unitree/npz/g1/G1_Take_102.bvh_60hz.rlopt_ipmd_500m_actions.npz`; either point at the parent manifest as above or copy/regenerate the action-labeled NPZ locally.
- Watch `train/estimated_reward_mean`, `train/expert_reward_mean`, `train/loss_reward_diff`, `episode/length`, `train/step_reward_mean`. Success = ep_len ≥ baseline, reward histogram doesn't saturate at 0 or 1, gap settles near `-margin`, SR loss keeps decreasing.

## May 23 pure-estimated reward follow-up

Clarified objective: the previous best-known 20M-frame local run was still mixed reward, not pure IRL. It used `env_reward_weight=1.0` and `est_reward_weight=0.1`; the run was stopped after the clarification so checkpoints were preserved.

```text
mixed env+estimated run, 128 envs, run 2026-05-23_21-23-15:
  5M checkpoint: model_step_5001216.pt
  10M checkpoint: model_step_10002432.pt
  near stop: iter=3440 | frames=10,567,680 | r_step=0.0438 | ep_len=233.38 | r_ep=3.3597
  best observed before stop: iter=3400 | frames=10,444,800 | r_ep=4.0531
  => longer budget clearly helped, but this was not pure IRL because env reward was still in the PPO reward.
```

Changed the bilinear config defaults toward estimated-only PPO:

- `use_estimated_rewards_for_ppo=True`
- `env_reward_weight=0.0`
- `est_reward_weight=0.1` (pure estimated reward, scaled to the stable mixed-run magnitude)
- `estimated_reward_done_penalty=1.0`
- `reward_update_warmup_updates=0`
- `bilinear.reward_floor=0.0`

Pure estimated-only screens at 128 envs, 1.23M frames unless noted:

```text
raw adversarial, env=0, est=1.0, no proxy, stopped after 1.23M from a 20M launch, run 2026-05-23_22-40-40:
  final iter=400 | r_step=0.3975 | ep_len=44.68 | r_ep=-6.4308 | reward_diff=-0.0100 | proxy_rho=0.0503 | exp_r=0.0895
  => PPO optimized high stale estimated rewards, while true env return collapsed.

proxy-only reward head, env=0, est=1.0, running-percentile proxy, reward_loss=0, no terminal penalty, run 2026-05-23_22-51-18:
  final iter=400 | r_step=0.5040 | ep_len=42.55 | r_ep=-5.2704 | proxy_rho=0.2226 | exp_r=0.5462
  => proxy learned some ranking but did not produce useful behavior from random policy states.

proxy-only reward head + estimated terminal penalty, run 2026-05-23_23-02-43:
  final iter=400 | r_step=0.4651 | ep_len=45.13 | r_ep=-5.7039 | proxy_rho=0.1532 | exp_r=0.4867
  => terminal penalty alone did not rescue estimated-only PPO.

stronger adversarial reward updates, env=0, est=0.1, reward_lr=1e-3, update_interval=1, updates_per_policy_update=4, run 2026-05-23_23-13-41:
  final iter=400 | r_step=0.0368 | ep_len=42.14 | r_ep=-5.9775 | reward_diff=-0.2241 | proxy_rho=-0.1066 | exp_r=0.6113
  => stronger reward updates fixed expert-vs-policy orientation (`exp_r` high, reward_diff negative) but rollout reward was anti-correlated with env reward.

same strong adversarial setting + bc_coef=0.01, run 2026-05-23_23-24-36:
  final iter=400 | r_step=0.0329 | ep_len=39.30 | r_ep=-3.7230 | reward_diff=-0.2707 | proxy_rho=-0.0102 | exp_r=0.6212
  => weak BC helped relative to raw pure IRL but remained negative.

same strong adversarial setting + bc_coef=0.1, run 2026-05-23_23-35-11:
  final iter=400 | r_step=0.0315 | ep_len=37.44 | r_ep=-3.3070 | reward_diff=-0.2417 | proxy_rho=-0.0731 | exp_r=0.5778
  => stronger BC improved the screen slightly but still did not establish learning.
```

Interpretation: estimated-only PPO from random initialization is currently not viable for this bilinear reward. The adversarial reward can be oriented so expert reward is higher than policy reward, but the rollout reward remains poorly aligned or anti-correlated with env reward. The env-reward proxy can produce positive rank correlation, but from a bad policy distribution it rewards the best states within a bad batch rather than expert-like behavior. A credible pure/no-env-reward path likely needs a real expert-manifold warm start (offline BC pretrain or resume from a policy already near the dataset) before switching to estimated reward only, or a different reward formulation that gives a usable dense signal off-manifold.

## Open issues / next steps

1. **Pure estimated-only path** — do not spend more long-run budget on raw estimated-only from random initialization. The next credible no-env-reward experiment should warm start the policy/reward near the expert manifold first (offline BC pretrain, resume from a competent checkpoint, or a reward formulation that is dense off-manifold), then switch PPO to estimated reward only.
2. **Longer mixed-reward reference** — the best empirical curve is still the mixed env+estimated path (`env_reward_weight=1.0`, `est_reward_weight=0.1`). It reached `r_ep≈4` by 10.4M frames locally; use it only as a reference/control, not as the pure-IRL answer.
3. **Proxy follow-up** — if continuing env-proxy shaping, keep PPO estimated-only when testing the pure objective, but treat env-proxy reward-head supervision as diagnostic. From random-policy rollouts it did not produce useful behavior by 1.23M frames.
4. **Baseline comparison** — compare the 128-env mixed-reference run against a longer env-only baseline, ideally with the same env count, seed, manifest, and minibatch sizing.
5. **RLOpt commit/push + parent pointer** — commit the follow-up RLOpt fixes on `claude/ipmd-bilinear-irl`, push the submodule branch, then bump the parent submodule pointer. CLAUDE.md says: "use the in-repo submodules as the authoritative codebases, then update the top-level submodule pointers."
6. **Submodule hygiene** — initialize `IsaacLab/` and `ImitationLearningTools/` in this worktree before using repo-root validation commands that depend on `./IsaacLab/isaaclab.sh`.
7. **Codex's deferred items** (recommended once basics work):
   - Variant B: reward conditioned on the latent `z` (`r = R_max · sigmoid(<z, phi> + b)`).
   - Low-rank `theta(z)` for multi-motion training (`theta(z) = theta0 + U · LN(z)`).
8. **Risk** — the config flip changes the default for *all* G1 bilinear subclasses (`G1ImitationLatentRLOptIPMDBilinearConfig`, `…VQVAEConfig`, `…LatentGoalRLOptIPMDBilinearConfig`). Any in-flight job using those configs will pick up the IRL behavior on next launch.

## Coordination with codex

Codex (gpt-5.2) reviewed the initial plan and pushed back on several points that are now incorporated:

- Margin/softplus IPM loss instead of raw `r_pi − r_exp` (prevents sigmoid saturation).
- Detach `g(a)` not just `F(s)` (action labels come from a previous 500M-step policy; letting reward grads into `g(a)` would teach the discriminator to recognize the labeler distribution, not expert occupancy).
- Reward floor `r_floor ≈ 0.05` for early-termination resistance.
- Snapshot-then-update ordering.
- Affine-free LayerNorm on `phi` to stabilize scale.
- Defer latent-conditioned reward to v2.

Codex implemented the diff in `RLOpt/rlopt/agent/ipmd/{module.py,ipmd_bilinear.py}` and the config flip in `rlopt_ipmd_bilinear_cfg.py` via the `mcp__codex__codex` tool with `workspace-write` sandbox.
