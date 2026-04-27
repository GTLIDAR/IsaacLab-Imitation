# IPMD Goal-Conditioned Reward Contract

IPMD treats imitation as goal-conditioned RL. The environment owns the goal/task
representation and publishes it through the command manager; the algorithm only
chooses which observation keys to consume.

## Command Surface

The command group separates the policy-visible command from the true reference
goal:

- `command.policy_command`: what the policy sees. Vanilla G1 publishes the
  current reference goal. Latent G1 publishes the agent-owned latent command.
- `command.reference_command`: the frame-t expert goal payload.
- `command.reference_motion`: reference joint position and velocity.
- `command.reference_anchor_pos_b`: reference anchor position in the current
  anchor frame.
- `command.reference_anchor_ori_b`: reference anchor orientation in the current
  anchor frame.

For vanilla G1, `policy_command` and `reference_command` carry the same goal
payload. For latent G1, `policy_command` is the latent, while
`reference_command` remains the explicit expert goal used by asymmetric critics,
posterior inputs, and reward estimation.

## Reward State

The reward estimator receives one env-owned state group:

```text
reward_state = [
  reference_command_t,
  joint_pos,
  joint_vel,
  root_pos,
  root_quat,
  root_lin_vel,
  root_ang_vel,
]
```

RLOpt IPMD should consume this generically with:

```python
reward_input_keys = [("reward_state", ...)]
reward_input_type = "s"
```

IPMD no longer assembles policy-vs-desired reward branches. The environment maps
the same group for rollout batches and expert batches.

## IRL Comparison

The reward model compares policy and expert samples as state-only rewards:

```text
r([g_t, x_t])       # rollout state under frame-t goal
r([g_t*, x_t*])     # expert state under its frame-t goal
```

Here `g` is always the env-owned `reference_command`; the kinematic part differs
between the rollout robot state and the expert reference state. This keeps the
goal representation fixed by the environment and lets IPMD remain a generic
reward-estimation algorithm over configured observation keys.
