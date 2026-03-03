from __future__ import annotations

from typing import Literal

import torch
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, sample_uniform
from isaaclab_imitation.envs import ImitationRLEnv


@torch.compile
def _replace_nan_with_default(
    values: torch.Tensor, defaults: torch.Tensor
) -> torch.Tensor:
    return torch.where(torch.isnan(values), defaults, values)


def randomize_joint_default_pos(
    env: ImitationRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize nominal joint default positions (mimic parity with unitree_rl_lab)."""
    asset: Articulation = env.scene[asset_cfg.name]

    # Keep a nominal copy for downstream tooling (e.g., deploy/export).
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    if pos_distribution_params is None:
        return

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
    else:
        joint_ids = torch.tensor(
            asset_cfg.joint_ids, dtype=torch.int, device=asset.device
        )

    randomized_pos = _randomize_prop_by_op(
        asset.data.default_joint_pos.clone(),
        pos_distribution_params,
        env_ids,
        joint_ids,
        operation=operation,
        distribution=distribution,
    )

    env_ids_for_slice = env_ids[:, None] if joint_ids != slice(None) else env_ids
    selected_pos = randomized_pos[env_ids_for_slice, joint_ids]
    asset.data.default_joint_pos[env_ids_for_slice, joint_ids] = selected_pos

    joint_pos_action_term = env.action_manager.get_term("joint_pos")
    offset = getattr(joint_pos_action_term, "_offset", None)
    if isinstance(offset, torch.Tensor):
        offset[env_ids_for_slice, joint_ids] = selected_pos


def reset_joints_to_reference(
    env: ImitationRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints to the reference joint positions and velocities."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # get reference joint state
    reference_joint_pos = env.current_reference["joint_pos"]
    reference_joint_vel = env.current_reference["joint_vel"]

    joint_pos = _replace_nan_with_default(
        reference_joint_pos[env_ids], asset.data.default_joint_pos[env_ids]
    )
    joint_vel = _replace_nan_with_default(
        reference_joint_vel[env_ids], asset.data.default_joint_vel[env_ids]
    )

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    asset.write_data_to_sim()
    # Refresh cached kinematics buffers (e.g. root_lin_vel_b) after direct state writes.
    env.scene.update(dt=0.0)
    asset.update(dt=0.0)


def reset_root_and_joints_to_reference_with_randomization(
    env: ImitationRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    joint_position_range: tuple[float, float] = (0.0, 0.0),
):
    """Reset root + joints from current reference using Unitree MotionCommand semantics.

    Mirrors unitree_rl_lab MotionCommand._resample_command behavior on reset:
    - start from current reference root/joint state,
    - add sampled root pose/velocity deltas,
    - add sampled joint position deltas and clip to soft limits,
    - write both joint and root state to simulation.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device

    reference = env.current_reference

    # Root pose/velocity: prefer explicit root keys, fall back to first body state.
    root_pos = reference.get("root_pos")
    root_quat = reference.get("root_quat")
    root_lin_vel = reference.get("root_lin_vel")
    root_ang_vel = reference.get("root_ang_vel")
    if root_pos is None or root_quat is None:
        body_pos_w = reference.get("body_pos_w")
        body_quat_w = reference.get("body_quat_w")
        if body_pos_w is None or body_quat_w is None:
            raise KeyError(
                "Reference root state is missing (`root_*` and `body_*` keys unavailable)."
            )
        root_pos = body_pos_w[:, 0, :]
        root_quat = body_quat_w[:, 0, :]
    if root_lin_vel is None or root_ang_vel is None:
        body_lin_vel_w = reference.get("body_lin_vel_w")
        body_ang_vel_w = reference.get("body_ang_vel_w")
        if body_lin_vel_w is not None and body_ang_vel_w is not None:
            root_lin_vel = body_lin_vel_w[:, 0, :]
            root_ang_vel = body_ang_vel_w[:, 0, :]
        else:
            root_lin_vel = torch.zeros_like(root_pos)
            root_ang_vel = torch.zeros_like(root_pos)

    root_pos = root_pos.clone()
    root_quat = root_quat.clone()
    root_lin_vel = root_lin_vel.clone()
    root_ang_vel = root_ang_vel.clone()

    # Unitree motion command exports body positions in local env frame and adds env origin.
    root_pos[env_ids, :3] += env.scene.env_origins[env_ids]

    # Apply sampled pose perturbation.
    pose_range = pose_range or {}
    pose_bounds = torch.tensor(
        [
            pose_range.get(k, (0.0, 0.0))
            for k in ("x", "y", "z", "roll", "pitch", "yaw")
        ],
        device=device,
    )
    pose_delta = sample_uniform(
        pose_bounds[:, 0], pose_bounds[:, 1], (env_ids.numel(), 6), device=device
    )
    root_pos[env_ids] += pose_delta[:, 0:3]
    orientation_delta = quat_from_euler_xyz(
        pose_delta[:, 3], pose_delta[:, 4], pose_delta[:, 5]
    )
    root_quat[env_ids] = quat_mul(orientation_delta, root_quat[env_ids])

    # Apply sampled velocity perturbation.
    velocity_range = velocity_range or {}
    vel_bounds = torch.tensor(
        [
            velocity_range.get(k, (0.0, 0.0))
            for k in ("x", "y", "z", "roll", "pitch", "yaw")
        ],
        device=device,
    )
    vel_delta = sample_uniform(
        vel_bounds[:, 0], vel_bounds[:, 1], (env_ids.numel(), 6), device=device
    )
    root_lin_vel[env_ids] += vel_delta[:, :3]
    root_ang_vel[env_ids] += vel_delta[:, 3:]

    # Joint state from reference with NaN fallback to defaults.
    reference_joint_pos = _replace_nan_with_default(
        reference["joint_pos"].clone(), asset.data.default_joint_pos
    )
    reference_joint_vel = _replace_nan_with_default(
        reference["joint_vel"].clone(), asset.data.default_joint_vel
    )

    # Apply sampled joint position perturbation and clip to soft limits.
    if joint_position_range[0] != 0.0 or joint_position_range[1] != 0.0:
        joint_noise = sample_uniform(
            joint_position_range[0],
            joint_position_range[1],
            reference_joint_pos[env_ids].shape,
            device=device,
        )
        reference_joint_pos[env_ids] += joint_noise
    soft_joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    reference_joint_pos[env_ids] = torch.clip(
        reference_joint_pos[env_ids],
        soft_joint_pos_limits[:, :, 0],
        soft_joint_pos_limits[:, :, 1],
    )

    # Write root + joints exactly on reset env ids.
    asset.write_joint_state_to_sim(
        reference_joint_pos[env_ids],
        reference_joint_vel[env_ids],
        env_ids=env_ids,
    )
    root_state = torch.cat(
        [
            root_pos[env_ids],
            root_quat[env_ids],
            root_lin_vel[env_ids],
            root_ang_vel[env_ids],
        ],
        dim=-1,
    )
    asset.write_root_state_to_sim(root_state, env_ids=env_ids)
    asset.write_data_to_sim()

    # Refresh cached kinematics buffers after direct writes.
    env.scene.update(dt=0.0)
    asset.update(dt=0.0)
