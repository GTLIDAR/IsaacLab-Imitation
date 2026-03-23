# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
from pathlib import Path

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ... import mdp
from ...imitation_env_cfg import ImitationLearningEnvCfg
from ...lafan1_manifest import (
    DEFAULT_G1_DANCE102_MANIFEST_PATH,
    DEFAULT_G1_LAFAN1_MANIFEST_PATH,
    dataset_path_from_entries,
    load_lafan1_manifest,
)

# Import g1 29DOF from unitree_rl_lab
from unitree_rl_lab.assets.robots.unitree import (
    UNITREE_G1_29DOF_MIMIC_ACTION_SCALE,
    UNITREE_G1_29DOF_MIMIC_CFG,
)


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


def _compose_obs_keys(
    group_name: str, term_names: list[str]
) -> list[str | tuple[str, ...]]:
    """Compose nested observation keys from group and term names."""
    return [(group_name, term_name) for term_name in term_names]


G1_POLICY_OBS_KEYS: list[str | tuple[str, ...]] = _compose_obs_keys(
    "policy",
    [
        "latent_command",
        "base_lin_vel",
        "base_ang_vel",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
    ],
)


G1_VALUE_OBS_KEYS: list[str | tuple[str, ...]] = _compose_obs_keys(
    "critic",
    [
        "latent_command",
        "body_pos",
        "body_ori",
        "base_lin_vel",
        "base_ang_vel",
        "joint_pos_rel",
        "joint_vel_rel",
        "joint_pos",
        "joint_vel",
        "last_action",
    ],
)

G1_REWARD_OBS_KEYS: list[str | tuple[str, ...]] = _compose_obs_keys(
    "reference",
    [
        "joint_pos",
        "joint_vel",
        "root_pos",
        "root_quat",
        "root_lin_vel",
        "root_ang_vel",
    ],
)


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

        latent_command = ObsTerm(func=mdp.agent_latent_command)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel_rel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic observations."""

        latent_command = ObsTerm(func=mdp.agent_latent_command)

        body_pos = ObsTerm(
            func=mdp.robot_body_pos_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=G1_TRACKED_BODY_NAMES,
                    preserve_order=True,
                ),
                "anchor_body_name": "torso_link",
            },
        )
        body_ori = ObsTerm(
            func=mdp.robot_body_ori_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=G1_TRACKED_BODY_NAMES,
                    preserve_order=True,
                ),
                "anchor_body_name": "torso_link",
            },
            history_length=3,
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=3)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=3)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, history_length=3)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, history_length=3)
        joint_pos = ObsTerm(func=mdp.joint_pos, history_length=3)
        joint_vel = ObsTerm(func=mdp.joint_vel, history_length=3)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = False

    @configclass
    class ReferenceCfg(ObsGroup):
        """Reference kinematic observations."""

        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        root_pos = ObsTerm(func=mdp.root_pos_w)
        root_quat = ObsTerm(func=mdp.root_quat_w)
        root_lin_vel = ObsTerm(func=mdp.root_lin_vel_w)
        root_ang_vel = ObsTerm(func=mdp.root_ang_vel_w)

        def __post_init__(self):
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    reference: ReferenceCfg = ReferenceCfg()


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

    reset_reference_state = EventTerm(
        func=mdp.reset_root_and_joints_to_reference_with_randomization,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.01, 0.01),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
            "velocity_range": VELOCITY_RANGE,
            "joint_position_range": (-0.1, 0.1),
        },
    )

    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(1.0, 3.0),
    #     params={"velocity_range": VELOCITY_RANGE},
    # )


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
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "anchor_body_name": "torso_link",
            "std": 0.3,
        },
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.reference_global_anchor_orientation_error_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "anchor_body_name": "torso_link",
            "std": 0.4,
        },
    )
    motion_body_pos = RewTerm(
        func=mdp.reference_relative_body_position_error_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=G1_TRACKED_BODY_NAMES,
                preserve_order=True,
            ),
            "reference_body_names": G1_TRACKED_BODY_NAMES,
            "anchor_body_name": "torso_link",
            "std": 0.3,
        },
    )
    motion_body_ori = RewTerm(
        func=mdp.reference_relative_body_orientation_error_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=G1_TRACKED_BODY_NAMES,
                preserve_order=True,
            ),
            "reference_body_names": G1_TRACKED_BODY_NAMES,
            "anchor_body_name": "torso_link",
            "std": 0.4,
        },
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.reference_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=G1_TRACKED_BODY_NAMES,
                preserve_order=True,
            ),
            "reference_body_names": G1_TRACKED_BODY_NAMES,
            "std": 1.0,
        },
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.reference_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=G1_TRACKED_BODY_NAMES,
                preserve_order=True,
            ),
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
                body_names=[
                    (
                        r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                        r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                    )
                ],
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
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=G1_EE_BODY_NAMES,
                preserve_order=True,
            ),
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
    latent_command_dim: int = 64

    _debug_rewards: bool = False

    # Master switch for all expensive visualizers/marker debug rendering.
    # Keep disabled by default for training/runtime performance.
    enable_visualizers: bool = False
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
            self.scene.contact_forces.debug_vis = bool(self.enable_visualizers)

        # Reference marker visualizers are also gated by the master toggle.
        self.visualize_reference_arrows = bool(
            self.enable_visualizers and self.visualize_reference_arrows
        )

        self.scene.height_scanner = None


@configclass
class ImitationG1LafanTrackEnvCfg(ImitationG1BaseTrackingEnvCfg):
    """General 29-DoF motion-tracking env driven by a LAFAN1 manifest."""

    dataset_path: str | None = None
    loader_type: str = "lafan1_csv"
    loader_kwargs: dict = {
        "dataset_name": "lafan1",
        "dataset": {"trajectories": {"lafan1_csv": []}},
        "control_freq": 50.0,
        "sim": {"dt": 0.005},
        "decimation": 4,
        "joint_names": G1_29DOF_JOINT_NAMES,
    }
    reset_schedule: str = "random"
    refresh_zarr_dataset: bool = True
    require_npz_body_states: bool = True
    lafan1_manifest_path: str = str(DEFAULT_G1_LAFAN1_MANIFEST_PATH)
    motions: list[str] | None = None
    trajectories: list[str] | None = None
    wrap_steps: bool = False
    reconstructed_reference_action: bool = True
    reconstructed_reference_action_mode = "next_pose"

    def _lafan_source_entries(self) -> list[dict[str, object]]:
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

    def _validate_source_path(self, source_path: Path) -> None:
        if not source_path.is_file():
            raise FileNotFoundError(
                "LAFAN1 motion source is missing. "
                f"Expected: {source_path}. "
                "Set `lafan1_manifest_path` to a manifest that points at repo-local NPZ motions."
            )
        if self.require_npz_body_states and source_path.suffix.lower() != ".npz":
            raise ValueError(
                "This tracking env requires an npz source with body states "
                "(body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w). "
                f"Got: {source_path}. "
                "Generate repo-local NPZ files before loading this manifest."
            )

    def _apply_manifest(self) -> None:
        _, manifest_entries = load_lafan1_manifest(self.lafan1_manifest_path)
        dataset_cfg = copy.deepcopy(self.loader_kwargs.get("dataset", {}))
        trajectories_cfg = copy.deepcopy(dataset_cfg.get("trajectories", {}))
        trajectories_cfg["lafan1_csv"] = manifest_entries
        dataset_cfg["trajectories"] = trajectories_cfg
        self.loader_kwargs["dataset"] = dataset_cfg
        self.loader_kwargs["dataset_name"] = "lafan1"
        self.loader_kwargs["sim"] = {"dt": float(self.sim.dt)}
        self.loader_kwargs["decimation"] = int(self.decimation)
        self.loader_kwargs["joint_names"] = list(self.reference_joint_names)

        if self.dataset_path is None:
            self.dataset_path = dataset_path_from_entries(manifest_entries)
        else:
            self.dataset_path = str(Path(self.dataset_path).expanduser().resolve())

        if self.motions is None:
            self.motions = [str(entry["name"]) for entry in manifest_entries]

    def __post_init__(self) -> None:
        super().__post_init__()

        self.loader_kwargs = copy.deepcopy(self.loader_kwargs)
        if self.motions is not None:
            self.motions = list(self.motions)
        if self.trajectories is not None:
            self.trajectories = list(self.trajectories)

        self._apply_manifest()

        allowed_reset_schedules = {"random", "sequential", "round_robin"}
        self.reset_schedule = self.reset_schedule.strip().lower()
        if self.reset_schedule not in allowed_reset_schedules:
            raise ValueError(
                f"Unsupported reset_schedule='{self.reset_schedule}'. "
                f"Allowed values: {sorted(allowed_reset_schedules)}."
            )

        source_entries = self._lafan_source_entries()
        for source in source_entries:
            source_path = Path(str(source["path"])).expanduser().resolve()
            source["path"] = str(source_path)
            self._validate_source_path(source_path)


@configclass
class ImitationG1Dance102CompareEnvCfg(ImitationG1LafanTrackEnvCfg):
    """Single-motion comparison env using a repo-local dance_102 manifest."""

    dataset_path: str | None = None
    lafan1_manifest_path: str = str(DEFAULT_G1_DANCE102_MANIFEST_PATH)
    motions: list[str] | None = ["dance_102"]
    trajectories: list[str] | None = ["trajectory_0"]
    reset_schedule: str = "sequential"
    refresh_zarr_dataset: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        source_path = Path(str(self._lafan_source_entries()[0]["path"]))
        if not source_path.exists():
            raise FileNotFoundError(
                "dance_102 motion npz file is required for this comparison env. "
                f"Expected repo-local file: {source_path}. "
                "Either place the motion at that path or override `lafan1_manifest_path`."
            )


# Backward-compatible aliases.
ImitationG1EnvCfg = ImitationG1LafanTrackEnvCfg
ImitationG1Dance102LafanEnvCfg = ImitationG1Dance102CompareEnvCfg
