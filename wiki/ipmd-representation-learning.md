# IPMD Representation Learning

This repo is currently being used to study representation learning for
IPMD-family inverse RL on G1 imitation tasks.

The working direction is not just "pretrain a decoder". The core goal is to
learn a useful state representation `f(s)` from expert state trajectories and
then use that representation inside IPMD-style inverse RL / adversarial reward
learning. Offline learning should warm-start the representation, and online
training should keep adapting it instead of freezing the encoder permanently by
default.

## Research Target

The target stack is:

1. Expert state data provides motion/reference trajectories.
2. An offline stage learns representation and reward-relevant structure from
   those trajectories.
3. Online IPMD uses environment rollouts plus expert batches to refine the
   reward/representation and train the policy.
4. Variants such as bilinear IPMD and VQ-VAE/FSQ latent IPMD test different
   representation parameterizations.

The key point is that the expert data is mostly state trajectory data. Do not
assume action labels exist. Any offline objective that needs expert actions
must first prove where those labels come from.

## Ownership Boundary

`IsaacLab-Imitation` owns:

- G1 task registration.
- Env/config surfaces for vanilla, latent, VQ-VAE, and bilinear experiments.
- Expert batch sampling from trajectory data.
- Dataset manifests and Zarr-cache routing.
- Local smoke scripts and cluster submission scripts.
- Experiment context and command documentation.

Sibling `../RLOpt` owns:

- IPMD algorithm logic.
- Reward estimator architecture and update cadence.
- Latent learner implementation and checkpointing.
- Bilinear representation model internals.
- Offline pretraining implementation details.

Do not put algorithm workarounds into `scripts/rlopt/train.py` when the behavior
belongs in `RLOpt/rlopt/agent/ipmd/`.

## Current Env-Owned Surfaces

The important env-side surfaces in this repo are in
`source/isaaclab_imitation/isaaclab_imitation/envs/imitation_rl_env.py`:

- `sample_expert_batch(...)`: expert batch API consumed by imitation algorithms.
- `_sample_expert_batch_impl(...)`: maps requested nested keys into expert
  tensors.
- `_sample_expert_trajectory_batch(...)`: draws random expert transitions
  without advancing live env state.
- `_build_reward_input_cache(...)` and `_reward_input_expert_terms(...)`:
  pre-materialize expert-side `reward_input` values.
- `_map_requested_expert_observations(...)`: maps requested `policy`, `critic`,
  `reward_input`, `expert_state`, and `expert_window` keys.
- `_sample_expert_window_slice(...)` and `_build_expert_window_terms(...)`:
  expose temporal windows for latent encoders and codebooks.

For representation learning, `sample_expert_batch(...)` is the main bridge from
Isaac Lab trajectory data to RLOpt offline/IRL objectives.

## Task Surfaces

Current task surfaces:

- `Isaac-Imitation-G1-v0`: vanilla G1 tracking.
- `Isaac-Imitation-G1-LafanTrack-v0`: legacy alias for vanilla tracking.
- `Isaac-Imitation-G1-Latent-v0`: latent-conditioned task for ASE/IPMD-style
  latent command experiments.
- `Isaac-Imitation-G1-Latent-VQVAE-v0`: latent VQ-VAE/FSQ skill-codebook task.

RLOpt algorithm selection is done through `--algo` in `scripts/rlopt/train.py`.
For IPMD-family work, the important values are:

- `IPMD`: base IPMD path.
- `IPMD_BILINEAR`: IPMD plus bilinear representation/reward variant.
- `IPMD_SR`: supported by the registry, but not the current main focus.

The task registry maps these through `rlopt_<algo>_cfg_entry_point` entries
under `source/isaaclab_imitation/.../config/g1/__init__.py`.

## Current Config Surfaces

Key config files:

- `rlopt_ipmd_cfg.py`: base vanilla/latent IPMD config.
- `rlopt_ipmd_bilinear_cfg.py`: bilinear IPMD config and offline pretrain
  toggles.
- `rlopt_ipmd_vqvae_cfg.py`: latent VQ-VAE/FSQ skill-codebook config.
- `imitation_g1_env_cfg.py`: vanilla env config and reward/expert observation
  groups.
- `imitation_g1_latent_env_cfg.py`: latent-command env config.
- `imitation_g1_latent_vqvae_env_cfg.py`: latent VQ-VAE env window config.

Important config concepts:

- `REWARD_INPUT_KEYS` define the expert/rollout reward estimator input surface.
- `LATENT_POSTERIOR_INPUT_KEYS` define what the posterior encoder consumes.
- `LATENT_POLICY_INPUT_KEYS` define what the policy sees when latent commands
  are enabled.
- `bilinear.offline_pretrain.enabled` controls the bilinear offline pretrain
  stage.
- `latent_learning.patch_past_steps` and `patch_future_steps` control expert
  window size.
- `ipmd.use_estimated_rewards_for_ppo` and `ipmd.env_reward_weight` determine
  how much PPO actually follows estimated rewards versus env rewards.

## Methodological Constraints

State-only expert data creates identifiability limits. Avoid claims that require
expert actions unless the data path explicitly provides them.

Good directions:

- State-representation learning from expert windows.
- IRL/adversarial reward objectives comparing rollout state distributions
  against expert state distributions.
- Reward structure that is semantically grouped, for example joint position,
  joint velocity, root state, and anchor terms.
- Offline warm-start followed by online fine-tuning.

Risky directions:

- Pure behavior cloning without real action labels.
- Treating next joint position as an action label without checking
  `JointPositionAction` scaling/offset semantics.
- Freezing the encoder after reconstruction if the goal is reusable `f(s)`.
- Repeating scalar reward regularization sweeps when the problem is reward
  parameterization or representation structure.

## Current Hypotheses

Use these as working hypotheses, not settled facts:

- A monolithic scalar reward MLP over flattened reward input can become a narrow
  separator rather than a smooth control-shaped reward basin.
- Grouped or structured rewards are likely more useful than only increasing
  scalar gradient penalty or L2 regularization.
- Bilinear IPMD is useful if the learned representation captures state structure
  that online reward learning can exploit.
- VQ-VAE/FSQ latent IPMD is useful if discrete or held skill codes stabilize
  the high-level command geometry.

## Implementation Rules

- Validate required observation keys, window sizes, and dimensional contracts at
  construction/config time where possible.
- Avoid defensive guards in the algorithmic hot path.
- Keep posterior and collector ownership in the IPMD agent, not in generic env
  helper layers.
- Keep vanilla and latent task surfaces distinct. IPMD can support both, but PPO
  should stay on vanilla surfaces and ASE should stay on latent surfaces.
- If a change is algorithmic, patch sibling `../RLOpt`; if it is expert batch,
  env observation, task registration, manifest, or cluster routing, patch this
  repo.

