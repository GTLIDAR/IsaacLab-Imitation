from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab_imitation.envs import ImitationRLEnv

from ._compiled import body_pose_in_anchor_frame, quat_to_rot6d_flat


def _select_last_dim(values: torch.Tensor, ids: torch.Tensor | slice) -> torch.Tensor:
    if isinstance(ids, slice):
        return values
    return values.index_select(-1, ids)


def reference_joint_pos(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    joint_ids = env._get_joint_ids_tensor_fast(asset_cfg.joint_ids)
    return _select_last_dim(env.current_reference["joint_pos"], joint_ids)


def reference_joint_vel(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    joint_ids = env._get_joint_ids_tensor_fast(asset_cfg.joint_ids)
    return _select_last_dim(env.current_reference["joint_vel"], joint_ids)


def reference_root_lin_vel(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return env.current_reference["root_lin_vel"]


def reference_root_ang_vel(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return env.current_reference["root_ang_vel"]


def reference_root_pos(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return env.current_reference["root_pos"]


def reference_root_quat(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return env.current_reference["root_quat"]


def reference_motion_command(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return env._get_reference_motion_command_fast(asset_cfg.joint_ids)


def agent_latent_command(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    del asset_cfg
    return env.get_agent_latent_command()


def reference_anchor_pos_b(
    env: ImitationRLEnv,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot_anchor_pos_w, robot_anchor_quat_w = env._get_robot_anchor_state_w_fast(
        anchor_body_name
    )
    ref_anchor_pos_w, ref_anchor_quat_w = env._get_reference_body_pose_w_fast(
        (anchor_body_name,)
    )
    anchor_pos_b, _ = body_pose_in_anchor_frame(
        robot_anchor_pos_w,
        robot_anchor_quat_w,
        ref_anchor_pos_w,
        ref_anchor_quat_w,
    )
    return anchor_pos_b[:, 0, :]


def reference_anchor_ori_b(
    env: ImitationRLEnv,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot_anchor_pos_w, robot_anchor_quat_w = env._get_robot_anchor_state_w_fast(
        anchor_body_name
    )
    ref_anchor_pos_w, ref_anchor_quat_w = env._get_reference_body_pose_w_fast(
        (anchor_body_name,)
    )
    _, anchor_ori_b = body_pose_in_anchor_frame(
        robot_anchor_pos_w,
        robot_anchor_quat_w,
        ref_anchor_pos_w,
        ref_anchor_quat_w,
    )
    return quat_to_rot6d_flat(anchor_ori_b[:, 0, :])


def robot_body_pos_b(
    env: ImitationRLEnv,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    body_pos_b, _ = env._get_robot_body_state_in_anchor_frame_fast(
        asset_cfg.body_ids, anchor_body_name
    )
    return body_pos_b.reshape(env.num_envs, -1)


def robot_body_ori_b(
    env: ImitationRLEnv,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    _, body_ori_b = env._get_robot_body_state_in_anchor_frame_fast(
        asset_cfg.body_ids, anchor_body_name
    )
    return quat_to_rot6d_flat(body_ori_b)
