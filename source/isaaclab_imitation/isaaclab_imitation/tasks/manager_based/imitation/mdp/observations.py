from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab_imitation.envs import ImitationRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    matrix_from_quat,
    subtract_frame_transforms,
)


@torch.compile
def _quat_to_rot6d_flat(quat: torch.Tensor) -> torch.Tensor:
    quat_mat = matrix_from_quat(quat)
    return quat_mat[..., :2].reshape(quat_mat.shape[0], -1)


@torch.compile
def _body_pose_in_anchor_frame(
    anchor_pos_w: torch.Tensor,
    anchor_quat_w: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_bodies = body_pos_w.shape[1]
    return subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )


def reference_joint_pos(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    ref_joint_pos = env.current_reference.get("joint_pos")

    return ref_joint_pos[..., asset_cfg.joint_ids]  # type: ignore


def reference_joint_vel(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    ref_joint_vel = env.current_reference.get("joint_vel")

    return ref_joint_vel[..., asset_cfg.joint_ids]  # type: ignore


def reference_root_lin_vel(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The root linear velocity of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    ref_root_lin_vel = env.current_reference.get("root_lin_vel")
    return ref_root_lin_vel


def reference_root_ang_vel(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The root angular velocity of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    ref_root_ang_vel = env.current_reference.get("root_ang_vel")
    return ref_root_ang_vel


def reference_root_pos(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The root position of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    ref_root_pos = env.current_reference.get("root_pos")
    return ref_root_pos


def reference_root_quat(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The root quaternion of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    ref_root_quat = env.current_reference.get("root_quat")
    return ref_root_quat


def _resolve_reference_body_indices(
    env: ImitationRLEnv, reference_body_names: Sequence[str], device: torch.device
) -> torch.Tensor:
    """Map reference body names to indices in replay metadata."""
    all_reference_body_names = getattr(env, "reference_body_names", None) or []

    def _has_only_generic_body_names(names: Sequence[str]) -> bool:
        return len(names) > 0 and all(name.startswith("body_") and name[5:].isdigit() for name in names)

    reference_names_were_generic = len(all_reference_body_names) == 0 or _has_only_generic_body_names(
        all_reference_body_names
    )

    ref_body_pos = env.current_reference.get("xpos")
    if ref_body_pos is None:
        ref_body_pos = env.current_reference.get("body_pos_w")
    reference_body_count = int(ref_body_pos.shape[1]) if ref_body_pos is not None and ref_body_pos.ndim >= 3 else None

    robot_lookup: dict[str, int] = {}
    robot_lookup_lower: dict[str, int] = {}

    # iltools loaders may emit generic body_0...body_N metadata.
    # Prefer semantic robot names whenever index-wise alignment is plausible.
    if reference_names_were_generic:
        try:
            asset: Articulation = env.scene["robot"]
            robot_body_names = list(asset.body_names)
            robot_lookup = {name: idx for idx, name in enumerate(robot_body_names)}
            robot_lookup_lower = {name.lower(): idx for idx, name in enumerate(robot_body_names)}

            if reference_body_count is not None and reference_body_count > 0 and len(robot_body_names) >= reference_body_count:
                inferred_reference_body_names = robot_body_names[:reference_body_count]
                if all_reference_body_names != inferred_reference_body_names:
                    env.reference_body_names = list(inferred_reference_body_names)
                    if hasattr(env, "_reference_body_index_cache"):
                        env._reference_body_index_cache = {}  # type: ignore[attr-defined]
                all_reference_body_names = inferred_reference_body_names
        except Exception:
            # Fall back to metadata-driven lookup below.
            pass

    if len(all_reference_body_names) == 0 and len(robot_lookup) == 0:
        raise RuntimeError(
            "Reference body names are unavailable in the environment metadata. "
            "Ensure dataset zarr metadata contains `body_names` or that reference body "
            "count matches robot body count for automatic fallback."
        )

    if not hasattr(env, "_reference_body_index_cache"):
        env._reference_body_index_cache = {}  # type: ignore[attr-defined]
    cache_key = tuple(reference_body_names)
    if cache_key in env._reference_body_index_cache:  # type: ignore[attr-defined]
        return env._reference_body_index_cache[cache_key]  # type: ignore[attr-defined]

    lookup = {name: idx for idx, name in enumerate(all_reference_body_names)}
    lookup_lower = {name.lower(): idx for idx, name in enumerate(all_reference_body_names)}
    max_reference_body_index = reference_body_count if reference_body_count is not None else len(all_reference_body_names)

    def _lookup_from_robot_names(name: str) -> int | None:
        if len(robot_lookup) == 0 or max_reference_body_index <= 0:
            return None
        if name in robot_lookup:
            idx = robot_lookup[name]
            return idx if idx < max_reference_body_index else None
        lowered = name.lower()
        if lowered in robot_lookup_lower:
            idx = robot_lookup_lower[lowered]
            return idx if idx < max_reference_body_index else None
        return None

    def _find_one(name: str) -> int:
        if name in lookup:
            return lookup[name]
        if name.lower() in lookup_lower:
            return lookup_lower[name.lower()]
        simplified = name.replace("_link", "")
        if simplified in lookup:
            return lookup[simplified]
        if simplified.lower() in lookup_lower:
            return lookup_lower[simplified.lower()]
        if reference_names_were_generic:
            robot_idx = _lookup_from_robot_names(name)
            if robot_idx is not None:
                return robot_idx
            robot_idx = _lookup_from_robot_names(simplified)
            if robot_idx is not None:
                return robot_idx
        raise KeyError(name)

    try:
        ref_indices = [_find_one(name) for name in reference_body_names]
    except KeyError as exc:
        missing_name = str(exc).strip("'")
        raise KeyError(
            f"Reference body '{missing_name}' not found in replay metadata. "
            f"First 20 available names: {all_reference_body_names[:20]}"
        ) from exc

    ref_indices_t = torch.tensor(ref_indices, dtype=torch.long, device=device)
    env._reference_body_index_cache[cache_key] = ref_indices_t  # type: ignore[attr-defined]
    return ref_indices_t


def _reference_body_pose_w(
    env: ImitationRLEnv, reference_body_names: Sequence[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reference body positions/quaternions in world frame."""
    device = env.device
    ref_body_ids = _resolve_reference_body_indices(env, reference_body_names, device)
    ref_pos = env.get_reference_data(key="xpos")[..., ref_body_ids, :]
    ref_quat = env.get_reference_data(key="xquat")[..., ref_body_ids, :]
    ref_pos_w, ref_quat_w = env._transform_reference_pose_to_world(ref_pos, ref_quat)
    return ref_pos_w, ref_quat_w  # type: ignore[return-value]


def reference_motion_command(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reference joint position/velocity command vector (qpos || qvel)."""
    ref_joint_pos = env.get_reference_data(key="joint_pos", joint_indices=asset_cfg.joint_ids)
    ref_joint_vel = env.get_reference_data(key="joint_vel", joint_indices=asset_cfg.joint_ids)
    return torch.cat([ref_joint_pos, ref_joint_vel], dim=-1)


def reference_anchor_pos_b(
    env: ImitationRLEnv,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reference anchor position expressed in the robot-anchor frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    anchor_body_idx = asset.body_names.index(anchor_body_name)
    robot_anchor_pos_w = asset.data.body_pos_w[:, anchor_body_idx]
    robot_anchor_quat_w = asset.data.body_quat_w[:, anchor_body_idx]

    ref_anchor_pos_w, ref_anchor_quat_w = _reference_body_pose_w(env, [anchor_body_name])
    ref_anchor_pos_w = ref_anchor_pos_w[:, 0, :]
    ref_anchor_quat_w = ref_anchor_quat_w[:, 0, :]

    anchor_pos_b, _ = subtract_frame_transforms(
        robot_anchor_pos_w, robot_anchor_quat_w, ref_anchor_pos_w, ref_anchor_quat_w
    )
    return anchor_pos_b


def reference_anchor_ori_b(
    env: ImitationRLEnv,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reference anchor orientation expressed in the robot-anchor frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    anchor_body_idx = asset.body_names.index(anchor_body_name)
    robot_anchor_pos_w = asset.data.body_pos_w[:, anchor_body_idx]
    robot_anchor_quat_w = asset.data.body_quat_w[:, anchor_body_idx]

    ref_anchor_pos_w, ref_anchor_quat_w = _reference_body_pose_w(env, [anchor_body_name])
    ref_anchor_pos_w = ref_anchor_pos_w[:, 0, :]
    ref_anchor_quat_w = ref_anchor_quat_w[:, 0, :]

    _, anchor_ori_b = subtract_frame_transforms(robot_anchor_pos_w, robot_anchor_quat_w, ref_anchor_pos_w, ref_anchor_quat_w)
    return _quat_to_rot6d_flat(anchor_ori_b)


def robot_body_pos_b(
    env: ImitationRLEnv,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot body positions in the robot-anchor frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    anchor_body_idx = asset.body_names.index(anchor_body_name)
    robot_anchor_pos_w = asset.data.body_pos_w[:, anchor_body_idx]
    robot_anchor_quat_w = asset.data.body_quat_w[:, anchor_body_idx]

    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    body_pos_b, _ = _body_pose_in_anchor_frame(robot_anchor_pos_w, robot_anchor_quat_w, body_pos_w, body_quat_w)
    return body_pos_b.reshape(env.num_envs, -1)


def robot_body_ori_b(
    env: ImitationRLEnv,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot body orientations in the robot-anchor frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    anchor_body_idx = asset.body_names.index(anchor_body_name)
    robot_anchor_pos_w = asset.data.body_pos_w[:, anchor_body_idx]
    robot_anchor_quat_w = asset.data.body_quat_w[:, anchor_body_idx]

    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    _, body_ori_b = _body_pose_in_anchor_frame(robot_anchor_pos_w, robot_anchor_quat_w, body_pos_w, body_quat_w)
    return _quat_to_rot6d_flat(body_ori_b)
