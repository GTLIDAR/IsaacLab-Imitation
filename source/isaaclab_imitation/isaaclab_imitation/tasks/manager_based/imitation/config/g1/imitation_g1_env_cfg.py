# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_imitation.tasks.manager_based.imitation.mdp as mdp
from isaaclab_imitation.tasks.manager_based.imitation.imitation_env_cfg import (
    ImitationLearningEnvCfg,
)

try:
    from unitree_rl_lab.assets.robots.unitree import (
        UNITREE_G1_29DOF_MIMIC_ACTION_SCALE,
        UNITREE_G1_29DOF_MIMIC_CFG,
    )
except ImportError:
    _this_file = Path(__file__).resolve()
    _unitree_source = None
    for _parent in _this_file.parents:
        _candidates = (
            _parent / "unitree_rl_lab" / "source" / "unitree_rl_lab",
            _parent / "unitree_rl_lab" / "source",
        )
        for _candidate in _candidates:
            if _candidate.is_dir():
                _unitree_source = _candidate
                break
        if _unitree_source is not None:
            break
    if _unitree_source is not None:
        _unitree_source_str = str(_unitree_source)
        if _unitree_source_str not in sys.path:
            sys.path.append(_unitree_source_str)
    try:
        from unitree_rl_lab.assets.robots.unitree import (
            UNITREE_G1_29DOF_MIMIC_ACTION_SCALE,
            UNITREE_G1_29DOF_MIMIC_CFG,
        )
    except ImportError as err:
        raise ImportError(
            "Failed to import Unitree 29-DoF mimic robot config from unitree_rl_lab. "
            "Install unitree_rl_lab or add it to PYTHONPATH to use Isaac-Imitation-G1-v0."
        ) from err


VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

G1_29DOF_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Body tracking set aligned with unitree_rl_lab/tasks/mimic/.../tracking_env_cfg.py.
G1_TRACKED_BODY_NAMES: list[str] = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

G1_EE_BODY_NAMES: list[str] = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

G1_UNDESIRED_CONTACT_PATTERN = (
    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
    r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
)

# Observation keys used by rlopt configs.
G1_POLICY_OBS_KEYS: list[str] = ["policy"]
G1_VALUE_OBS_KEYS: list[str] = ["critic"]
G1_REWARD_OBS_KEYS: list[str] = ["critic"]


def _resolve_workspace_path(*parts: str) -> str:
    """Resolve a file path by searching parent directories from this file."""
    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        candidate = parent.joinpath(*parts)
        if candidate.exists():
            return str(candidate)
    return str(Path(*parts))


DANCE_102_NPZ_PATH = _resolve_workspace_path(
    "unitree_rl_lab",
    "source",
    "unitree_rl_lab",
    "unitree_rl_lab",
    "tasks",
    "mimic",
    "robots",
    "g1_29dof",
    "dance_102",
    "G1_Take_102.bvh_60hz.npz",
)

DANCE_102_CSV_PATH = _resolve_workspace_path(
    "unitree_rl_lab",
    "source",
    "unitree_rl_lab",
    "unitree_rl_lab",
    "tasks",
    "mimic",
    "robots",
    "g1_29dof",
    "dance_102",
    "G1_Take_102.bvh_60hz.csv",
)

DANCE_102_MOTION_PATH = DANCE_102_NPZ_PATH
DEFAULT_LAFAN1_MOTION_PATH = DANCE_102_MOTION_PATH


def _read_npz_fps(npz_path: Path) -> float | None:
    """Read optional fps metadata from an npz file."""
    if not npz_path.is_file() or npz_path.suffix.lower() != ".npz":
        return None
    try:
        with np.load(npz_path) as npz_data:
            fps_value = npz_data.get("fps")
            if fps_value is None:
                return None
            return float(np.asarray(fps_value).reshape(-1)[0])
    except Exception:
        return None


def _normalize_lafan1_motion_entries(entries_like: Any, default_input_fps: float = 60.0) -> list[dict[str, Any]]:
    """Normalize manifest/json data into iltools lafan1_csv entry dicts."""

    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _normalize_one(entry_like: Any, fallback_name: str | None = None) -> dict[str, Any]:
        if isinstance(entry_like, str):
            path_obj = Path(entry_like).expanduser().resolve()
            name = fallback_name or path_obj.stem
            return {"name": name, "path": str(path_obj), "input_fps": float(default_input_fps)}

        if not isinstance(entry_like, dict):
            raise ValueError(f"Unsupported motion entry type: {type(entry_like)}")

        path_value = entry_like.get("path") or entry_like.get("file")
        if path_value is None:
            raise ValueError("Each motion entry must include `path` (or `file`).")
        path_obj = Path(str(path_value)).expanduser().resolve()
        name = str(entry_like.get("name") or fallback_name or path_obj.stem)
        input_fps = float(entry_like.get("input_fps", default_input_fps))
        normalized = {"name": name, "path": str(path_obj), "input_fps": input_fps}
        if "frame_range" in entry_like:
            normalized["frame_range"] = entry_like["frame_range"]
        return normalized

    if isinstance(entries_like, dict):
        if "motions" in entries_like:
            return _normalize_lafan1_motion_entries(entries_like["motions"], default_input_fps=default_input_fps)
        if "lafan1_csv" in entries_like:
            return _normalize_lafan1_motion_entries(entries_like["lafan1_csv"], default_input_fps=default_input_fps)
        dataset_cfg = entries_like.get("dataset")
        if isinstance(dataset_cfg, dict):
            trajectories_cfg = dataset_cfg.get("trajectories")
            if isinstance(trajectories_cfg, dict) and "lafan1_csv" in trajectories_cfg:
                return _normalize_lafan1_motion_entries(
                    trajectories_cfg["lafan1_csv"],
                    default_input_fps=default_input_fps,
                )
        if "path" in entries_like or "file" in entries_like:
            return [_normalize_one(entries_like)]

        # Mapping style: {motion_name: path_or_paths}
        normalized_entries: list[dict[str, Any]] = []
        for motion_name, path_spec in entries_like.items():
            for index, item in enumerate(_as_list(path_spec)):
                fallback_name = str(motion_name) if index == 0 else f"{motion_name}_{index}"
                normalized_entries.append(_normalize_one(item, fallback_name=fallback_name))
        return normalized_entries

    normalized_entries = []
    for entry in _as_list(entries_like):
        normalized_entries.append(_normalize_one(entry))
    return normalized_entries


def _load_lafan1_entries_from_manifest(
    *,
    manifest_path: str | None,
    manifest_data: Any | None,
) -> list[dict[str, Any]] | None:
    """Load normalized motion entries from cfg-provided manifest path/data."""
    if manifest_data is not None:
        return _normalize_lafan1_motion_entries(manifest_data)
    if manifest_path is None:
        return None

    manifest_file = Path(manifest_path).expanduser().resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"lafan1_manifest_path not found: {manifest_file}")
    with manifest_file.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return _normalize_lafan1_motion_entries(data)


def _infer_common_motion_fps(entries: list[dict[str, Any]]) -> float | None:
    """Infer a shared fps from npz sources if all provide the same fps metadata."""
    fps_values: list[float] = []
    for entry in entries:
        source_path = Path(str(entry["path"])).expanduser().resolve()
        fps_value = _read_npz_fps(source_path)
        if fps_value is None or fps_value <= 0.0:
            return None
        fps_values.append(float(fps_value))
    if len(fps_values) == 0:
        return None
    first = fps_values[0]
    if all(abs(value - first) < 1.0e-6 for value in fps_values):
        return first
    return None


def _dataset_path_from_entries(entries: list[dict[str, Any]]) -> str:
    """Create a stable cache path tied to the motion entry manifest."""
    signature = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    return f"/tmp/iltools_g1_lafan1_tracking_{digest}"


@configclass
class G1ActionsCfg:
    """Action settings for 29-DoF mimic G1."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=UNITREE_G1_29DOF_MIMIC_ACTION_SCALE,
        use_default_offset=True,
    )


@configclass
class G1ObservationCfg:
    """Observation settings aligned with the 29-DoF tracking environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""

        reference_motion = ObsTerm(
            func=mdp.reference_motion_command,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=G1_29DOF_JOINT_NAMES,
                )
            },
        )
        reference_anchor_ori_b = ObsTerm(
            func=mdp.reference_anchor_ori_b,
            params={"asset_cfg": SceneEntityCfg("robot"), "anchor_body_name": "torso_link"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic observations."""

        reference_motion = ObsTerm(
            func=mdp.reference_motion_command,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=G1_29DOF_JOINT_NAMES,
                )
            },
        )
        reference_anchor_pos_b = ObsTerm(
            func=mdp.reference_anchor_pos_b,
            params={"asset_cfg": SceneEntityCfg("robot"), "anchor_body_name": "torso_link"},
        )
        reference_anchor_ori_b = ObsTerm(
            func=mdp.reference_anchor_ori_b,
            params={"asset_cfg": SceneEntityCfg("robot"), "anchor_body_name": "torso_link"},
        )
        body_pos = ObsTerm(
            func=mdp.robot_body_pos_b,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=G1_TRACKED_BODY_NAMES),
                "anchor_body_name": "torso_link",
            },
        )
        body_ori = ObsTerm(
            func=mdp.robot_body_ori_b,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=G1_TRACKED_BODY_NAMES),
                "anchor_body_name": "torso_link",
            },
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1EventCfg:
    """Event settings aligned with the 29-DoF tracking environment."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {
                "x": (-0.025, 0.025),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
            },
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.01, 0.01),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints_to_reference = EventTerm(
        func=mdp.reset_joints_to_reference,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class G1RewardsCfg:
    """Reward terms aligned to the 29-DoF tracking environment."""

    # -- base
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    # -- tracking
    motion_global_anchor_pos = RewTerm(
        func=mdp.reference_global_anchor_position_error_exp,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "anchor_body_name": "torso_link", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.reference_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "anchor_body_name": "torso_link", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.reference_relative_body_position_error_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1_TRACKED_BODY_NAMES),
            "reference_body_names": G1_TRACKED_BODY_NAMES,
            "anchor_body_name": "torso_link",
            "std": 0.3,
        },
    )
    motion_body_ori = RewTerm(
        func=mdp.reference_relative_body_orientation_error_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1_TRACKED_BODY_NAMES),
            "reference_body_names": G1_TRACKED_BODY_NAMES,
            "anchor_body_name": "torso_link",
            "std": 0.4,
        },
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.reference_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1_TRACKED_BODY_NAMES),
            "reference_body_names": G1_TRACKED_BODY_NAMES,
            "std": 1.0,
        },
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.reference_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1_TRACKED_BODY_NAMES),
            "reference_body_names": G1_TRACKED_BODY_NAMES,
            "std": 3.14,
        },
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[G1_UNDESIRED_CONTACT_PATTERN],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class G1TerminationsCfg:
    """Termination terms aligned to the 29-DoF tracking environment."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "anchor_body_name": "torso_link",
            "threshold": 0.25,
        },
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "anchor_body_name": "torso_link",
            "threshold": 0.8,
        },
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_reference_body_pos_z_only,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1_EE_BODY_NAMES),
            "reference_body_names": G1_EE_BODY_NAMES,
            "threshold": 0.25,
        },
    )


@configclass
class ImitationG1BaseTrackingEnvCfg(ImitationLearningEnvCfg):
    """Shared 29-DoF G1 tracking config aligned with Unitree mimic tracking settings."""

    actions = G1ActionsCfg()
    observations = G1ObservationCfg()
    rewards = G1RewardsCfg()  # type: ignore
    terminations = G1TerminationsCfg()  # type: ignore
    events = G1EventCfg()

    device: str = "cuda"
    replay_reference: bool = False
    replay_only: bool = False
    reference_start_frame: int = 0

    _debug_rewards: bool = False

    visualize_reference_arrows: bool = True
    print_reference_velocity: bool = False
    print_reference_velocity_every: int = 50

    reference_joint_names: list[str] = G1_29DOF_JOINT_NAMES.copy()
    target_joint_names: list[str] = G1_29DOF_JOINT_NAMES.copy()

    def __post_init__(self) -> None:
        super().__post_init__()  # type: ignore

        self.scene.robot = UNITREE_G1_29DOF_MIMIC_CFG.replace(  # type: ignore
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
            self.scene.contact_forces.force_threshold = 10.0
            self.scene.contact_forces.debug_vis = True

        self.scene.height_scanner = None


@configclass
class ImitationG1LafanTrackEnvCfg(ImitationG1BaseTrackingEnvCfg):
    """General 29-DoF motion-tracking env using iltools LAFAN1 loader."""

    dataset_path: str = "/tmp/iltools_g1_lafan1_tracking"
    loader_type: str = "lafan1_csv"
    loader_kwargs: dict = {
        "dataset_name": "lafan1",
        "dataset": {
            "trajectories": {
                "lafan1_csv": [
                    {
                        "name": "motion_0",
                        "path": DEFAULT_LAFAN1_MOTION_PATH,
                        "input_fps": 60,
                    }
                ]
            }
        },
        "control_freq": 50.0,
        "sim": {"dt": 0.005},
        "decimation": 4,
        "joint_names": G1_29DOF_JOINT_NAMES,
    }
    reset_schedule: str = "random"
    refresh_zarr_dataset: bool = False
    require_npz_body_states: bool = True
    autodetect_motion_fps: bool = True
    # -- optional explicit cfg overrides (no env-var controls)
    lafan1_manifest_path: str | None = None
    lafan1_manifest_data: Any | None = None
    lafan1_dataset_path: str | None = None
    lafan1_refresh_zarr_dataset: bool | None = None
    lafan1_motions: list[str] | None = None
    lafan1_trajectories: list[str] | None = None
    lafan1_control_freq: float | None = None
    lafan1_reset_schedule: str | None = None
    lafan1_wrap_steps: bool | None = None
    lafan1_reference_start_frame: int | None = None

    def _lafan_source_entries(self) -> list[dict[str, Any]]:
        try:
            entries = self.loader_kwargs["dataset"]["trajectories"]["lafan1_csv"]
        except Exception as err:
            raise ValueError(
                "loader_kwargs must define dataset.trajectories.lafan1_csv with at least one source entry."
            ) from err
        if not isinstance(entries, list) or len(entries) == 0:
            raise ValueError(
                "loader_kwargs.dataset.trajectories.lafan1_csv must be a non-empty list."
            )
        return entries

    def _lafan_source_entry(self) -> dict[str, Any]:
        """Backward-compatible accessor for first source entry."""
        return self._lafan_source_entries()[0]

    def _validate_source_path(self, source_path: Path) -> None:
        if not source_path.is_file():
            raise FileNotFoundError(
                "LAFAN1 motion source is missing. "
                f"Expected: {source_path}. "
                "Set `lafan1_manifest_path`/`loader_kwargs` in cfg to an exported npz from "
                "unitree_rl_lab/scripts/mimic/csv_to_npz.py."
            )
        if self.require_npz_body_states and source_path.suffix.lower() != ".npz":
            raise ValueError(
                "This tracking env requires an npz source with body states "
                "(body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w). "
                f"Got: {source_path}. "
                "Generate npz with unitree_rl_lab/scripts/mimic/csv_to_npz.py."
            )

    def _apply_loader_cfg_overrides(self) -> None:
        """Apply cfg-driven overrides for multi-motion/scheduler workflows."""
        manifest_entries = _load_lafan1_entries_from_manifest(
            manifest_path=self.lafan1_manifest_path,
            manifest_data=self.lafan1_manifest_data,
        )
        if manifest_entries is not None:
            dataset_cfg = copy.deepcopy(self.loader_kwargs.get("dataset", {}))
            trajectories_cfg = copy.deepcopy(dataset_cfg.get("trajectories", {}))
            trajectories_cfg["lafan1_csv"] = manifest_entries
            dataset_cfg["trajectories"] = trajectories_cfg
            self.loader_kwargs["dataset"] = dataset_cfg
            self.loader_kwargs["dataset_name"] = self.loader_kwargs.get("dataset_name", "lafan1")
            # If manifest is provided and no explicit cfg motion filter is set, include all manifest motions.
            if self.lafan1_motions is None:
                self.motions = [entry["name"] for entry in manifest_entries]
            # Avoid stale zarr caches by default for manifest-driven runs.
            if self.lafan1_dataset_path is None:
                self.dataset_path = _dataset_path_from_entries(manifest_entries)
            if self.lafan1_refresh_zarr_dataset is None:
                self.refresh_zarr_dataset = True

        if self.lafan1_dataset_path is not None:
            self.dataset_path = str(Path(self.lafan1_dataset_path).expanduser())

        if self.lafan1_refresh_zarr_dataset is not None:
            self.refresh_zarr_dataset = bool(self.lafan1_refresh_zarr_dataset)

        if self.lafan1_motions is not None and len(self.lafan1_motions) > 0:
            self.motions = list(self.lafan1_motions)

        if self.lafan1_trajectories is not None and len(self.lafan1_trajectories) > 0:
            self.trajectories = list(self.lafan1_trajectories)

        if self.lafan1_reset_schedule is not None:
            normalized_schedule = self.lafan1_reset_schedule.strip().lower()
            allowed = {"random", "sequential", "round_robin"}
            if normalized_schedule not in allowed:
                raise ValueError(
                    f"Unsupported lafan1_reset_schedule='{self.lafan1_reset_schedule}'. "
                    f"Allowed values: {sorted(allowed)}."
                )
            self.reset_schedule = normalized_schedule

        if self.lafan1_wrap_steps is not None:
            self.wrap_steps = bool(self.lafan1_wrap_steps)

        if self.lafan1_reference_start_frame is not None:
            self.reference_start_frame = int(self.lafan1_reference_start_frame)

        if self.lafan1_control_freq is not None:
            self.loader_kwargs["control_freq"] = float(self.lafan1_control_freq)

    def _sync_loader_frequency_from_sources(self, source_entries: list[dict[str, Any]]) -> None:
        if not self.autodetect_motion_fps:
            return
        # Only auto-sync when all sources report the same fps metadata.
        common_fps = _infer_common_motion_fps(source_entries)
        if common_fps is None:
            return
        self.loader_kwargs["control_freq"] = float(common_fps)

    def __post_init__(self) -> None:
        super().__post_init__()

        # Avoid mutating class-level dicts/lists across instances.
        self.loader_kwargs = copy.deepcopy(self.loader_kwargs)
        if hasattr(self, "motions"):
            self.motions = list(self.motions)
        if hasattr(self, "trajectories"):
            self.trajectories = list(self.trajectories)

        self._apply_loader_cfg_overrides()

        source_entries = self._lafan_source_entries()
        for source in source_entries:
            source_path = Path(str(source["path"])).expanduser().resolve()
            source["path"] = str(source_path)
            self._validate_source_path(source_path)

        self._sync_loader_frequency_from_sources(source_entries)


@configclass
class ImitationG1Dance102CompareEnvCfg(ImitationG1LafanTrackEnvCfg):
    """Dance-102 comparison env for parity checks against unitree_rl_lab."""

    dataset_path: str = "/tmp/iltools_g1_dance_102_lafan1"
    loader_kwargs: dict = {
        "dataset_name": "lafan1",
        "dataset": {
            "trajectories": {
                "lafan1_csv": [
                    {
                        "name": "dance_102",
                        "path": DANCE_102_MOTION_PATH,
                        "input_fps": 60,
                    }
                ]
            }
        },
        "control_freq": 50.0,
        "sim": {"dt": 0.005},
        "decimation": 4,
        "joint_names": G1_29DOF_JOINT_NAMES,
    }
    motions: list[str] = ["dance_102"]
    trajectories: list[str] = ["trajectory_0"]
    reset_schedule: str = "sequential"
    refresh_zarr_dataset: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        source_path = Path(self._lafan_source_entry()["path"])
        if not source_path.exists():
            raise FileNotFoundError(
                "dance_102 motion npz file is required for full tracking terms. "
                f"Expected: {source_path}. "
                "Generate it with unitree_rl_lab/scripts/mimic/csv_to_npz.py from "
                f"{DANCE_102_CSV_PATH}."
            )


# Backward-compatible aliases.
ImitationG1EnvCfg = ImitationG1LafanTrackEnvCfg
ImitationG1Dance102LafanEnvCfg = ImitationG1Dance102CompareEnvCfg
