import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias

import isaaclab.utils.math as math_utils
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from isaaclab.assets import Articulation
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import (
    FRAME_MARKER_CFG,
)
from tensordict import TensorDict

logger = logging.getLogger(__name__)
NestedKey: TypeAlias = str | tuple[str, ...]
_MDP_COMPILED: Any | None = None


def _get_mdp_compiled_module() -> Any:
    global _MDP_COMPILED
    if _MDP_COMPILED is None:
        from isaaclab_imitation.tasks.manager_based.imitation.mdp import _compiled

        _MDP_COMPILED = _compiled
    return _MDP_COMPILED


# Import the new manager and utilities
try:
    from iltools.datasets.lafan1.loader import Lafan1CsvLoader
    from iltools.datasets.loco_mujoco.loader import LocoMuJoCoLoader
    from iltools.datasets.manager import ParallelTrajectoryManager, ResetSchedule
    from iltools.datasets.utils import make_rb_from
except ImportError as e:
    raise ImportError(
        f"Failed to import required modules from iltools_datasets: {e}. Make sure ImitationLearningTools is installed."
    ) from e


class ImitationRLEnv(ManagerBasedRLEnv):
    """
    Simplified RL environment for imitation learning with clean dataset interface.

    Config attributes (cfg):
        dataset_path: str, path to Zarr dataset directory (or directory containing trajectories.zarr)
        reset_schedule: str, trajectory reset schedule ("random", "sequential", "round_robin", "custom")
        wrap_steps: bool, if True, wrap steps within trajectory (default: False)
        replay_only: bool, if True, ignore actions and force reference root/joint state each step
        loader_type: str, required if Zarr does not exist
            (supported: "loco_mujoco", "lafan1_csv", "lafan1")
        loader_kwargs: dict, required if Zarr does not exist (e.g., {"env_name": "UnitreeG1", "cfg": ...})
        reference_joint_names: list[str], joint names in reference data order
        target_joint_names: list[str], optional, joint names in target robot order (for mapping)
        datasets: str | list[str] | None, optional, dataset names to load from Zarr
        motions: str | list[str] | None, optional, motion names to load from Zarr
        trajectories: str | list[str] | None, optional, trajectory names to load from Zarr
        keys: str | list[str] | None, optional, keys to load from Zarr (default: all keys)
        refresh_zarr_dataset: bool, if True, delete existing zarr and rebuild it using the loader each run
        reference_start_frame: int, trajectory-local frame index used after each reset (default: 0)
        visualize_reference_arrows: bool, if True show reference velocity/position/heading arrows and
            desired/current frame markers for root and tracked bodies (default: False)

    Example config:
        dataset_path = '/path/to/zarr'
        reset_schedule = 'random'  # or 'sequential', 'round_robin', 'custom'
        wrap_steps = False
        loader_type = 'loco_mujoco'  # or 'lafan1_csv'
        loader_kwargs = {'env_name': 'UnitreeG1', 'cfg': {...}}
        reference_joint_names = ['left_hip_pitch_joint', ...]
    """

    def __init__(self, cfg: Any, render_mode: str | None = None, **kwargs: Any) -> None:
        """Initialize the simplified ImitationRLEnv."""
        # Get device
        device = cfg.sim.device
        num_envs = cfg.scene.num_envs

        # Get dataset path and determine if we need to create it
        dataset_path = getattr(cfg, "dataset_path", None)
        loader_type = getattr(cfg, "loader_type", None)
        loader_kwargs = getattr(cfg, "loader_kwargs", {})
        refresh_zarr_dataset = bool(getattr(cfg, "refresh_zarr_dataset", False))

        # Build or load the replay buffer and trajectory info
        if dataset_path is not None:
            dataset_path = Path(dataset_path)
            # Check if it's a directory containing trajectories.zarr or the zarr itself
            if dataset_path.is_dir():
                zarr_path = dataset_path / "trajectories.zarr"
                if not zarr_path.exists():
                    zarr_path = dataset_path  # Assume the directory itself is the zarr
            else:
                zarr_path = dataset_path

            # For debugging, optionally force dataset refresh on every run.
            if refresh_zarr_dataset:
                if loader_type is None:
                    raise ValueError(
                        "refresh_zarr_dataset=True requires loader_type + loader_kwargs "
                        "so the zarr dataset can be rebuilt."
                    )
                if zarr_path.exists():
                    if zarr_path.is_dir():
                        shutil.rmtree(zarr_path)
                    else:
                        zarr_path.unlink()

            # If zarr doesn't exist and loader is provided, create it
            if not zarr_path.exists() and loader_type is not None:
                if loader_type == "loco_mujoco":
                    from omegaconf import DictConfig

                    loader_cfg = DictConfig(loader_kwargs)
                    _ = LocoMuJoCoLoader(
                        env_name=loader_kwargs["env_name"],
                        cfg=loader_cfg,
                        build_zarr_dataset=True,
                        zarr_path=str(zarr_path),
                    )
                elif loader_type in ("lafan1_csv", "lafan1"):
                    from omegaconf import DictConfig

                    loader_cfg = DictConfig(loader_kwargs)
                    _ = Lafan1CsvLoader(
                        cfg=loader_cfg,
                        build_zarr_dataset=True,
                        zarr_path=str(zarr_path),
                    )
                else:
                    raise ValueError(
                        f"Unsupported loader_type: {loader_type}. "
                        "Supported loader types: loco_mujoco, lafan1_csv, lafan1."
                    )

            # Load replay buffer from Zarr
            datasets = getattr(cfg, "datasets", None)
            motions = getattr(cfg, "motions", None)
            traj_names = getattr(cfg, "trajectories", None)
            keys = getattr(cfg, "keys", None)

            rb, traj_info = make_rb_from(
                zarr_path=str(zarr_path),
                datasets=datasets,
                motions=motions,
                trajectories=traj_names,
                keys=keys,
                device=torch.device("cuda:0"),
                verbose_tree=False,
                prefetch=3,
            )
        else:
            raise ValueError(
                "Either dataset_path must be provided, or loader_type + loader_kwargs "
                "must be provided to create a new dataset."
            )

        # Map assignment_strategy to reset_schedule (for backward compatibility)
        assignment_strategy = getattr(cfg, "assignment_strategy", None)
        reset_schedule = getattr(cfg, "reset_schedule", None)
        if reset_schedule is None and assignment_strategy is not None:
            # Map old assignment_strategy to new reset_schedule
            mapping = {
                "random": ResetSchedule.RANDOM,
                "sequential": ResetSchedule.SEQUENTIAL,
                "round_robin": ResetSchedule.ROUND_ROBIN,
            }
            reset_schedule = mapping.get(assignment_strategy, ResetSchedule.RANDOM)
        if reset_schedule is None:
            reset_schedule = ResetSchedule.RANDOM
        # Get other config options
        wrap_steps = getattr(cfg, "wrap_steps", False)
        reference_start_frame = int(getattr(cfg, "reference_start_frame", 0))
        if reference_start_frame < 0:
            raise ValueError("reference_start_frame must be >= 0.")
        reference_joint_names = list(getattr(cfg, "reference_joint_names", []))
        target_joint_names = list(getattr(cfg, "target_joint_names", []))
        dataset_joint_names = self._read_reference_joint_names_from_zarr(zarr_path)
        if len(dataset_joint_names) > 0:
            if len(reference_joint_names) == 0:
                reference_joint_names = dataset_joint_names
            elif len(reference_joint_names) != len(dataset_joint_names):
                reference_joint_names = dataset_joint_names

        first_transition = rb[0]
        first_qpos = first_transition.get("qpos")
        if first_qpos is not None:
            expected_reference_joint_dim = int(first_qpos.shape[-1]) - 7
            if len(reference_joint_names) != expected_reference_joint_dim:
                raise ValueError(
                    "reference_joint_names length mismatch with replay buffer qpos. "
                    f"Expected {expected_reference_joint_dim} joints from qpos, got "
                    f"{len(reference_joint_names)} reference names."
                )

        assert len(reference_joint_names) > 0 and len(target_joint_names) > 0, (
            "Reference and target joint names must have the length greater than 0"
        )
        self._reference_has_aligned_next = (
            first_transition.get("next_qpos") is not None
            and first_transition.get("next_qvel") is not None
        )
        self._reconstructed_reference_action_enabled = bool(
            getattr(cfg, "reconstructed_reference_action", False)
        )
        self._reconstructed_reference_action_mode = str(
            getattr(cfg, "reconstructed_reference_action_mode", "next_pose")
        )
        if (
            self._reconstructed_reference_action_enabled
            and not self._reference_has_aligned_next
        ):
            raise ValueError(
                "reconstructed_reference_action=True requires transition-aligned next_* reference data. "
                "Rebuild the cached dataset with `refresh_zarr_dataset=True`."
            )

        # Initialize the trajectory manager
        self.trajectory_manager = ParallelTrajectoryManager(
            rb=rb,
            traj_info=traj_info,
            num_envs=num_envs,
            reset_schedule=reset_schedule,
            reset_start_step=reference_start_frame,
            wrap_steps=wrap_steps,
            device=device,
            reference_joint_names=reference_joint_names,
            target_joint_names=target_joint_names,
        )

        # Get initial reference data (this also initializes env assignments)
        self.current_reference: TensorDict = self.trajectory_manager.sample(
            advance=False
        )
        self._agent_latent_dim = int(getattr(cfg, "latent_command_dim", 16))
        self._agent_latent_command = torch.zeros(
            (num_envs, self._agent_latent_dim),
            device=device,
            dtype=torch.float32,
        )

        # Store reference joint mapping
        self.reference_joint_names = reference_joint_names
        self.reference_body_names: list[str] = []
        self.reference_site_names: list[str] = []
        self._joint_mapping_cache: torch.Tensor | None = None
        self._reference_vel_vis_enabled = bool(
            getattr(
                cfg,
                "visualize_reference_arrows",
                getattr(cfg, "visualize_reference_velocity", False),
            )
        )
        self._reference_vel_marker: VisualizationMarkers | None = None
        self._reference_pos_delta_marker: VisualizationMarkers | None = None
        self._initial_heading_marker: VisualizationMarkers | None = None
        self._goal_root_frame_marker: VisualizationMarkers | None = None
        self._current_root_frame_marker: VisualizationMarkers | None = None
        self._goal_body_frame_markers: list[VisualizationMarkers] = []
        self._current_body_frame_markers: list[VisualizationMarkers] = []
        self._vis_reference_body_ids: torch.Tensor | None = None
        self._vis_robot_body_ids: torch.Tensor | None = None
        self._vis_body_names: list[str] = []
        self._last_tracked_root_pos_w = torch.zeros((num_envs, 3), device=device)
        self._last_tracked_root_pos_valid = torch.zeros(
            (num_envs,), device=device, dtype=torch.bool
        )
        self.replay_reference = getattr(cfg, "replay_reference", False)
        self.replay_only = getattr(cfg, "replay_only", False)
        if self.replay_only and not self.replay_reference:
            self.replay_reference = True
        self._expert_sampler_warned_action_fallback = False
        self._expert_sampler_warned_unknown_terms: set[str] = set()
        self._reconstructed_reference_action_term: JointPositionAction | None = None
        self._reconstructed_reference_target_to_action_index: torch.Tensor | None = None
        self._reconstructed_reference_action_pd_ratio_target: torch.Tensor | None = None

        # Store initial poses for replay
        self._init_root_pos = torch.zeros((num_envs, 3), device=device)
        self._init_root_quat = torch.zeros((num_envs, 4), device=device)
        self._init_root_quat[:, 0] = 1.0
        # Reference root pose at reset frame for each env.
        # This aligns datasets whose xpos/xquat do not start near origin.
        self._reference_reset_root_pos = torch.zeros((num_envs, 3), device=device)
        self._reference_reset_root_quat = torch.zeros((num_envs, 4), device=device)
        self._reference_reset_root_quat[:, 0] = 1.0
        initial_reference_root_pos = self.current_reference.get("root_pos")
        initial_reference_root_quat = self.current_reference.get("root_quat")
        if initial_reference_root_pos is not None:
            self._reference_reset_root_pos.copy_(initial_reference_root_pos)
        if initial_reference_root_quat is not None:
            self._reference_reset_root_quat.copy_(initial_reference_root_quat)
        self._load_reference_metadata(zarr_path)

        # Initialize parent class
        super().__init__(cfg, render_mode, **kwargs)

        self.robot: Articulation = self.scene["robot"]
        self._expert_env_origins = self.scene.env_origins.clone()
        self._expert_default_joint_pos = self.robot.data.default_joint_pos.clone()
        self._expert_default_joint_vel = self.robot.data.default_joint_vel.clone()
        self._setup_reconstructed_reference_action_cache()
        self._finalize_reference_body_names()
        self._initialize_mdp_fast_paths()
        self._setup_reference_velocity_visualizer()
        self._update_reference_velocity_visualizer()

    @staticmethod
    def _read_reference_joint_names_from_zarr(zarr_path: Path) -> list[str]:
        """Read reference joint names from zarr metadata if available."""
        try:
            root = zarr.open(str(zarr_path), mode="r")
        except Exception:
            return []

        try:
            for key in list(root.group_keys()):  # type: ignore[attr-defined]
                group = root[key]
                joint_names = group.attrs.get("joint_names", None)
                if joint_names is not None:
                    return list(joint_names)
        except Exception:
            return []

        return []

    def _load_reference_metadata(self, zarr_path: Path) -> None:
        """Load reference body/site names from zarr metadata if available."""
        try:
            root = zarr.open(str(zarr_path), mode="r")
        except Exception:
            return

        dataset_group = None
        try:
            group_keys = list(root.group_keys())  # type: ignore[attr-defined]
            for key in group_keys:
                group = root[key]
                if "body_names" in group.attrs:
                    dataset_group = group
                    break
        except Exception:
            dataset_group = None

        if dataset_group is None:
            return

        body_names = dataset_group.attrs.get("body_names", [])
        site_names = dataset_group.attrs.get("site_names", [])
        self.reference_body_names = list(body_names) if body_names is not None else []
        self.reference_site_names = list(site_names) if site_names is not None else []

    def _resolve_static_joint_parameter(
        self, values: torch.Tensor, *, name: str
    ) -> torch.Tensor:
        """Collapse per-env joint parameters to a single vector for cached reconstruction."""
        tensor = values.to(device=self.device, dtype=torch.float32)
        if tensor.ndim == 1:
            return tensor
        if tensor.ndim != 2:
            raise ValueError(f"Unexpected {name} tensor shape {tuple(tensor.shape)}.")
        reference = tensor[0]
        if tensor.shape[0] > 1 and not torch.allclose(tensor, reference.unsqueeze(0)):
            logger.warning(
                "Reference action reconstruction expected env-invariant %s; using env 0 values.",
                name,
            )
        return reference

    def _compute_reconstructed_reference_action_targets(
        self,
        *,
        mode: str,
        kp_target: torch.Tensor | None,
        kd_target: torch.Tensor | None,
        chunk_size: int = 65536,
    ) -> torch.Tensor:
        """Precompute reference joint position targets for every replay transition."""
        tm = self.trajectory_manager
        num_transitions = len(tm.rb)
        cache = torch.empty(
            (num_transitions, len(tm.target_joint_names)),
            device=tm.storage_device,
            dtype=torch.float32,
        )
        ratio = None
        if mode == "pd_compensated":
            if kp_target is None or kd_target is None:
                raise ValueError("pd_compensated reconstruction requires Kp and Kd.")
            safe_kp = torch.where(
                kp_target.abs() > 1.0e-8, kp_target, torch.ones_like(kp_target)
            )
            ratio = torch.where(
                kp_target.abs() > 1.0e-8,
                kd_target / safe_kp,
                torch.zeros_like(kd_target),
            )
        elif mode != "next_pose":
            raise ValueError(
                "Unsupported reconstructed_reference_action_mode: "
                f"{mode!r}. Expected 'next_pose' or 'pd_compensated'."
            )

        for start in range(0, num_transitions, chunk_size):
            end = min(start + chunk_size, num_transitions)
            indices = torch.arange(
                start, end, device=tm.storage_device, dtype=torch.int64
            )
            reference = tm.rb[indices]
            if getattr(tm, "_device", None) is not None:
                reference = reference.to(tm._device)
            reference = tm._attach_reference_fields(reference, use_buffers=False)
            next_joint_pos = reference.get(("next", "joint_pos"))
            if next_joint_pos is None:
                raise ValueError(
                    "Transition-aligned next joint positions are missing from the replay buffer."
                )
            command_target = next_joint_pos
            if ratio is not None:
                command_target = command_target + reference["joint_vel"] * ratio
            cache[start:end] = command_target.to(device=tm.storage_device)

        return cache

    def _setup_reconstructed_reference_action_cache(self) -> None:
        """Initialize optional cached expert-action reconstruction after action manager setup."""
        self.trajectory_manager.set_reconstructed_action_targets(None)
        self._reconstructed_reference_action_term = None
        self._reconstructed_reference_target_to_action_index = None
        self._reconstructed_reference_action_pd_ratio_target = None
        if not self._reference_has_aligned_next:
            if self._reconstructed_reference_action_enabled:
                raise ValueError(
                    "reconstructed_reference_action=True requires transition-aligned next_* reference data. "
                    "Rebuild the cached dataset with `refresh_zarr_dataset=True`."
                )
            return

        try:
            action_term = self.action_manager.get_term("joint_pos")
        except Exception:
            if self._reconstructed_reference_action_enabled:
                raise
            return
        if not isinstance(action_term, JointPositionAction):
            if self._reconstructed_reference_action_enabled:
                raise TypeError(
                    "reconstructed_reference_action is only supported for JointPositionAction."
                )
            return

        target_joint_names = list(self.trajectory_manager.target_joint_names)
        target_name_to_index = {
            name: idx for idx, name in enumerate(target_joint_names)
        }
        action_joint_names = list(action_term._joint_names)
        missing_joint_names = [
            name for name in action_joint_names if name not in target_name_to_index
        ]
        if missing_joint_names:
            raise ValueError(
                "JointPositionAction joints are missing from target_joint_names: "
                f"{missing_joint_names}"
            )

        target_joint_ids, _ = self.robot.find_joints(
            target_joint_names, preserve_order=True
        )
        kp_target = None
        kd_target = None
        if self._reconstructed_reference_action_mode == "pd_compensated":
            kp_target = self._resolve_static_joint_parameter(
                self.robot.data.default_joint_stiffness[:, target_joint_ids],
                name="joint stiffness",
            )
            kd_target = self._resolve_static_joint_parameter(
                self.robot.data.default_joint_damping[:, target_joint_ids],
                name="joint damping",
            )
            safe_kp = torch.where(
                kp_target.abs() > 1.0e-8, kp_target, torch.ones_like(kp_target)
            )
            self._reconstructed_reference_action_pd_ratio_target = torch.where(
                kp_target.abs() > 1.0e-8,
                kd_target / safe_kp,
                torch.zeros_like(kd_target),
            )

        self._reconstructed_reference_action_term = action_term
        self._reconstructed_reference_target_to_action_index = torch.tensor(
            [target_name_to_index[name] for name in action_joint_names],
            device=self.device,
            dtype=torch.int64,
        )
        if not self._reconstructed_reference_action_enabled:
            return

        cached_targets = self._compute_reconstructed_reference_action_targets(
            mode=self._reconstructed_reference_action_mode,
            kp_target=kp_target,
            kd_target=kd_target,
        )
        self.trajectory_manager.set_reconstructed_action_targets(cached_targets)

    def _gather_action_term_parameter(
        self,
        value: torch.Tensor | float,
        *,
        env_ids: torch.Tensor,
        template: torch.Tensor,
    ) -> torch.Tensor:
        """Gather an action-term parameter for the sampled env ids."""
        if isinstance(value, torch.Tensor):
            if value.ndim == 2:
                return value.index_select(0, env_ids).to(
                    device=template.device, dtype=template.dtype
                )
            return value.to(device=template.device, dtype=template.dtype)
        return torch.full_like(template, float(value))

    def _raw_to_processed_action(
        self,
        raw_action: torch.Tensor,
        *,
        env_ids: torch.Tensor,
        action_term: JointPositionAction | None = None,
    ) -> torch.Tensor:
        """Apply the action term's affine transform and clipping to raw actions."""
        if action_term is None:
            action_term = self._reconstructed_reference_action_term
        if action_term is None:
            raise ValueError(
                "JointPositionAction term is unavailable for action processing."
            )

        raw_action = raw_action.to(device=self.device, dtype=torch.float32)
        env_ids = env_ids.to(device=self.device, dtype=torch.int64)
        offset = self._gather_action_term_parameter(
            action_term._offset, env_ids=env_ids, template=raw_action
        )
        scale = self._gather_action_term_parameter(
            action_term._scale, env_ids=env_ids, template=raw_action
        )
        processed_action = raw_action * scale + offset
        if getattr(action_term.cfg, "clip", None) is not None:
            clip = action_term._clip.index_select(0, env_ids).to(
                device=self.device, dtype=processed_action.dtype
            )
            processed_action = torch.clamp(
                processed_action, min=clip[..., 0], max=clip[..., 1]
            )
        return processed_action

    def _processed_to_raw_action(
        self,
        processed_action: torch.Tensor,
        *,
        env_ids: torch.Tensor,
        action_term: JointPositionAction | None = None,
    ) -> torch.Tensor:
        """Invert the unclipped affine transform from processed to raw action space."""
        if action_term is None:
            action_term = self._reconstructed_reference_action_term
        if action_term is None:
            raise ValueError(
                "JointPositionAction term is unavailable for action processing."
            )

        processed_action = processed_action.to(device=self.device, dtype=torch.float32)
        env_ids = env_ids.to(device=self.device, dtype=torch.int64)
        offset = self._gather_action_term_parameter(
            action_term._offset, env_ids=env_ids, template=processed_action
        )
        scale = self._gather_action_term_parameter(
            action_term._scale, env_ids=env_ids, template=processed_action
        )
        safe_scale = torch.where(scale.abs() > 1.0e-8, scale, torch.ones_like(scale))
        return torch.where(
            scale.abs() > 1.0e-8,
            (processed_action - offset) / safe_scale,
            torch.zeros_like(processed_action),
        )

    def _reconstruct_reference_action_from_reference(
        self,
        reference: TensorDict,
        *,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Reconstruct raw and processed reference actions from a live reference batch."""
        action_term = self._reconstructed_reference_action_term
        action_index = self._reconstructed_reference_target_to_action_index
        if action_term is None or action_index is None:
            return None

        next_joint_pos = reference.get(("next", "joint_pos"))
        if next_joint_pos is None:
            return None
        processed_reference_action = next_joint_pos.to(
            device=self.device, dtype=torch.float32
        ).index_select(-1, action_index)

        if self._reconstructed_reference_action_pd_ratio_target is not None:
            joint_vel = reference.get("joint_vel")
            if joint_vel is None:
                return None
            pd_ratio = self._reconstructed_reference_action_pd_ratio_target.to(
                device=self.device, dtype=processed_reference_action.dtype
            ).index_select(0, action_index)
            processed_reference_action = (
                processed_reference_action
                + joint_vel.to(
                    device=self.device, dtype=processed_reference_action.dtype
                ).index_select(-1, action_index)
                * pd_ratio
            )

        if env_ids is None:
            env_ids = torch.arange(
                processed_reference_action.shape[0],
                device=self.device,
                dtype=torch.int64,
            )
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.int64)

        if getattr(action_term.cfg, "clip", None) is not None:
            clip = action_term._clip.index_select(0, env_ids).to(
                device=self.device, dtype=processed_reference_action.dtype
            )
            processed_reference_action = torch.clamp(
                processed_reference_action, min=clip[..., 0], max=clip[..., 1]
            )

        raw_reference_action = self._processed_to_raw_action(
            processed_reference_action,
            env_ids=env_ids,
            action_term=action_term,
        )
        return raw_reference_action, processed_reference_action

    @staticmethod
    def _compute_action_alignment_metrics(
        policy_action: torch.Tensor,
        reference_action: torch.Tensor,
        *,
        prefix: str,
        include_reference_nan_frac: bool = False,
    ) -> dict[str, float]:
        """Aggregate alignment metrics between policy and reconstructed reference actions."""
        if policy_action.ndim == 1:
            policy_action = policy_action.unsqueeze(0)
        if reference_action.ndim == 1:
            reference_action = reference_action.unsqueeze(0)

        policy_action = policy_action.detach().to(dtype=torch.float32)
        reference_action = reference_action.detach().to(dtype=torch.float32)
        reference_nan_frac = float(
            (~torch.isfinite(reference_action)).float().mean().item()
        )

        policy_action = torch.nan_to_num(policy_action, nan=0.0, posinf=0.0, neginf=0.0)
        reference_action = torch.nan_to_num(
            reference_action, nan=0.0, posinf=0.0, neginf=0.0
        )

        policy_flat = policy_action.reshape(policy_action.shape[0], -1)
        reference_flat = reference_action.reshape(reference_action.shape[0], -1)
        diff_flat = policy_flat - reference_flat
        per_env_abs_mean = diff_flat.abs().mean(dim=-1)
        per_env_mse = diff_flat.square().mean(dim=-1)

        metrics = {
            f"{prefix}_mae": float(per_env_abs_mean.mean().item()),
            f"{prefix}_mse": float(per_env_mse.mean().item()),
            f"{prefix}_rmse": float(per_env_mse.sqrt().mean().item()),
            f"{prefix}_max_abs": float(diff_flat.abs().amax(dim=-1).mean().item()),
            f"{prefix}_cosine": float(
                F.cosine_similarity(policy_flat, reference_flat, dim=-1, eps=1.0e-8)
                .mean()
                .item()
            ),
            f"{prefix}_policy_abs_mean": float(policy_flat.abs().mean().item()),
            f"{prefix}_reference_abs_mean": float(reference_flat.abs().mean().item()),
        }
        if include_reference_nan_frac:
            metrics[f"{prefix}_reference_nan_frac"] = reference_nan_frac
        return metrics

    def _compute_rollout_reference_action_log(
        self, policy_raw_action: torch.Tensor
    ) -> dict[str, float]:
        """Compare the rollout action against the aligned reconstructed reference action."""
        if self.current_reference is None:
            return {}

        reconstructed = self._reconstruct_reference_action_from_reference(
            self.current_reference
        )
        if reconstructed is None:
            return {}
        reference_raw_action, reference_processed_action = reconstructed

        policy_raw_action = policy_raw_action.to(
            device=self.device, dtype=torch.float32
        )
        if policy_raw_action.shape != reference_raw_action.shape:
            return {}

        env_ids = torch.arange(
            policy_raw_action.shape[0], device=self.device, dtype=torch.int64
        )
        policy_processed_action = self._raw_to_processed_action(
            policy_raw_action,
            env_ids=env_ids,
        )

        metrics = self._compute_action_alignment_metrics(
            policy_raw_action,
            reference_raw_action,
            prefix="rollout_action/raw",
            include_reference_nan_frac=True,
        )
        metrics.update(
            self._compute_action_alignment_metrics(
                policy_processed_action,
                reference_processed_action,
                prefix="rollout_action/processed",
            )
        )
        return metrics

    @staticmethod
    def _compute_rollout_state_alignment_metrics(
        actual_state: torch.Tensor,
        reference_state: torch.Tensor,
        *,
        prefix: str,
    ) -> dict[str, float]:
        """Aggregate next-state tracking metrics against the aligned reference transition."""
        if actual_state.ndim == 1:
            actual_state = actual_state.unsqueeze(0)
        if reference_state.ndim == 1:
            reference_state = reference_state.unsqueeze(0)

        actual_state = actual_state.detach().to(dtype=torch.float32)
        reference_state = reference_state.detach().to(dtype=torch.float32)
        reference_nan_frac = float(
            (~torch.isfinite(reference_state)).float().mean().item()
        )

        actual_state = torch.nan_to_num(actual_state, nan=0.0, posinf=0.0, neginf=0.0)
        reference_state = torch.nan_to_num(
            reference_state, nan=0.0, posinf=0.0, neginf=0.0
        )
        diff = actual_state - reference_state
        per_env_abs_mean = diff.abs().reshape(diff.shape[0], -1).mean(dim=-1)
        per_env_mse = diff.square().reshape(diff.shape[0], -1).mean(dim=-1)
        return {
            f"{prefix}_mae": float(per_env_abs_mean.mean().item()),
            f"{prefix}_mse": float(per_env_mse.mean().item()),
            f"{prefix}_rmse": float(per_env_mse.sqrt().mean().item()),
            f"{prefix}_max_abs": float(
                diff.abs().reshape(diff.shape[0], -1).amax(dim=-1).mean().item()
            ),
            f"{prefix}_reference_nan_frac": reference_nan_frac,
        }

    def _compute_rollout_reference_state_log(self) -> dict[str, float]:
        """Compare the post-step robot state against the aligned reference next state."""
        if self.current_reference is None:
            return {}

        next_joint_pos = self.current_reference.get(("next", "joint_pos"))
        next_joint_vel = self.current_reference.get(("next", "joint_vel"))
        if next_joint_pos is None or next_joint_vel is None:
            return {}

        metrics = self._compute_rollout_state_alignment_metrics(
            self.robot.data.joint_pos,
            next_joint_pos.to(device=self.device, dtype=torch.float32),
            prefix="rollout_state/joint_pos",
        )
        metrics.update(
            self._compute_rollout_state_alignment_metrics(
                self.robot.data.joint_vel,
                next_joint_vel.to(device=self.device, dtype=torch.float32),
                prefix="rollout_state/joint_vel",
            )
        )
        return metrics

    def _sample_reconstructed_reference_actions(
        self,
        *,
        global_indices: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """Convert cached target joint positions into raw policy actions."""
        if self._reconstructed_reference_action_term is None:
            return None
        if self._reconstructed_reference_target_to_action_index is None:
            return None

        cached_targets = self.trajectory_manager.get_reconstructed_action_targets(
            global_indices
        )
        if cached_targets is None:
            return None

        env_ids = env_ids.to(device=self.device, dtype=torch.int64)
        q_cmd = cached_targets.to(device=self.device, dtype=torch.float32).index_select(
            -1, self._reconstructed_reference_target_to_action_index
        )
        action_term = self._reconstructed_reference_action_term

        if getattr(action_term.cfg, "clip", None) is not None:
            clip = action_term._clip.index_select(0, env_ids).to(
                device=self.device, dtype=q_cmd.dtype
            )
            q_cmd = torch.clamp(q_cmd, min=clip[..., 0], max=clip[..., 1])
        return self._processed_to_raw_action(
            q_cmd, env_ids=env_ids, action_term=action_term
        )

    def _finalize_reference_body_names(self) -> None:
        """Improve reference body-name mapping for datasets that only provide generic names."""
        ref_body_pos = self.current_reference.get("xpos")
        if ref_body_pos is None:
            ref_body_pos = self.current_reference.get("body_pos_w")
        if ref_body_pos is None or ref_body_pos.ndim < 3:
            return

        num_reference_bodies = int(ref_body_pos.shape[1])
        robot_body_names = list(self.robot.body_names)

        has_generic_names = len(self.reference_body_names) == 0 or all(
            name.startswith("body_") and name[5:].isdigit()
            for name in self.reference_body_names
        )
        if has_generic_names and len(robot_body_names) >= num_reference_bodies:
            self.reference_body_names = robot_body_names[:num_reference_bodies]

    @staticmethod
    def _normalize_body_name_for_matching(name: str) -> str:
        """Normalize body names for tolerant cross-dataset matching."""
        lowered = name.lower()
        if lowered.endswith("_link"):
            lowered = lowered[:-5]
        return lowered

    def _initialize_mdp_fast_paths(self) -> None:
        if not hasattr(self, "robot"):
            self.robot = self.scene["robot"]
        self._finalize_reference_body_names()
        self._mdp_cache_step = -1
        self._mdp_align_quat: torch.Tensor | None = None
        self._mdp_align_pos: torch.Tensor | None = None
        self._mdp_reference_root_cache: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None
        self._mdp_reference_cvel_cache: torch.Tensor | None = None
        self._mdp_reference_motion_cache: dict[tuple[int, ...], torch.Tensor] = {}
        self._mdp_reference_body_id_cache: dict[tuple[str, ...], torch.Tensor] = {}
        self._mdp_reference_body_pose_cache: dict[
            tuple[str, ...], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._mdp_reference_body_velocity_cache: dict[
            tuple[str, ...], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._mdp_robot_anchor_id_cache: dict[str, int] = {}
        self._mdp_robot_anchor_state_cache: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._mdp_robot_body_pose_w_cache: dict[
            object, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._mdp_robot_body_velocity_w_cache: dict[
            object, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._mdp_robot_body_anchor_frame_cache: dict[
            tuple[int, object], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._mdp_body_name_to_id = {
            name: idx for idx, name in enumerate(self.robot.body_names)
        }
        self._mdp_body_name_to_id_lower = {
            name.lower(): idx for idx, name in enumerate(self.robot.body_names)
        }
        self._mdp_body_name_to_id_normalized = {
            self._normalize_body_name_for_matching(name): idx
            for idx, name in enumerate(self.robot.body_names)
        }
        self._mdp_reference_body_name_to_id = {
            name: idx for idx, name in enumerate(self.reference_body_names)
        }
        self._mdp_reference_body_name_to_id_lower = {
            name.lower(): idx for idx, name in enumerate(self.reference_body_names)
        }
        self._mdp_reference_body_name_to_id_normalized = {
            self._normalize_body_name_for_matching(name): idx
            for idx, name in enumerate(self.reference_body_names)
        }
        self._mdp_body_id_tensor_cache: dict[tuple[int, ...], torch.Tensor] = {}
        self._mdp_joint_id_tensor_cache: dict[tuple[int, ...], torch.Tensor] = {}
        self._mdp_all_body_ids_key = tuple(range(len(self.robot.body_names)))
        self._mdp_reset_pose_bounds: torch.Tensor | None = None
        self._mdp_reset_velocity_bounds: torch.Tensor | None = None

        reference = self.current_reference
        self._mdp_reference_body_pos_key = (
            "xpos" if "xpos" in reference else "body_pos_w"
        )
        self._mdp_reference_body_quat_key = (
            "xquat" if "xquat" in reference else "body_quat_w"
        )
        self._mdp_reference_body_count = int(
            reference[self._mdp_reference_body_pos_key].shape[1]
        )
        self._mdp_reset_root_pose_source = (
            "root" if "root_pos" in reference and "root_quat" in reference else "body"
        )
        if "root_lin_vel" in reference and "root_ang_vel" in reference:
            self._mdp_reset_root_velocity_source = "root"
        elif "body_lin_vel_w" in reference and "body_ang_vel_w" in reference:
            self._mdp_reset_root_velocity_source = "body"
        else:
            self._mdp_reset_root_velocity_source = "zeros"

    def _ensure_mdp_fast_paths(self) -> None:
        if hasattr(self, "_mdp_cache_step"):
            return
        self._initialize_mdp_fast_paths()

    def _invalidate_mdp_cache(self) -> None:
        self._ensure_mdp_fast_paths()
        self._mdp_cache_step = -1
        self._mdp_align_quat = None
        self._mdp_align_pos = None
        self._mdp_reference_root_cache = None
        self._mdp_reference_cvel_cache = None
        self._mdp_reference_motion_cache.clear()
        self._mdp_reference_body_pose_cache.clear()
        self._mdp_reference_body_velocity_cache.clear()
        self._mdp_robot_anchor_state_cache.clear()
        self._mdp_robot_body_pose_w_cache.clear()
        self._mdp_robot_body_velocity_w_cache.clear()
        self._mdp_robot_body_anchor_frame_cache.clear()

    def _ensure_mdp_step_cache(self) -> None:
        self._ensure_mdp_fast_paths()
        if (
            self._mdp_cache_step == self.common_step_counter
            and self._mdp_align_quat is not None
        ):
            return
        align_quat, align_pos = self._get_reference_alignment_transform()
        self._mdp_align_quat = align_quat
        self._mdp_align_pos = align_pos
        self._mdp_reference_root_cache = None
        self._mdp_reference_cvel_cache = None
        self._mdp_reference_motion_cache.clear()
        self._mdp_reference_body_pose_cache.clear()
        self._mdp_reference_body_velocity_cache.clear()
        self._mdp_robot_anchor_state_cache.clear()
        self._mdp_robot_body_pose_w_cache.clear()
        self._mdp_robot_body_velocity_w_cache.clear()
        self._mdp_robot_body_anchor_frame_cache.clear()
        self._mdp_cache_step = self.common_step_counter

    def _get_reference_alignment_fast(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_mdp_step_cache()
        return self._mdp_align_quat, self._mdp_align_pos  # type: ignore[return-value]

    def _get_body_ids_tensor_fast(
        self, body_ids: Sequence[int] | slice
    ) -> torch.Tensor | slice:
        self._ensure_mdp_fast_paths()
        if isinstance(body_ids, slice):
            return body_ids
        key = tuple(int(body_id) for body_id in body_ids)
        body_ids_t = self._mdp_body_id_tensor_cache.get(key)
        if body_ids_t is None:
            body_ids_t = torch.tensor(key, dtype=torch.long, device=self.device)
            self._mdp_body_id_tensor_cache[key] = body_ids_t
        return body_ids_t

    def _get_joint_ids_tensor_fast(
        self, joint_ids: Sequence[int] | slice
    ) -> torch.Tensor | slice:
        self._ensure_mdp_fast_paths()
        if isinstance(joint_ids, slice):
            return joint_ids
        key = tuple(int(joint_id) for joint_id in joint_ids)
        joint_ids_t = self._mdp_joint_id_tensor_cache.get(key)
        if joint_ids_t is None:
            joint_ids_t = torch.tensor(key, dtype=torch.long, device=self.device)
            self._mdp_joint_id_tensor_cache[key] = joint_ids_t
        return joint_ids_t

    def _get_reference_root_state_w_fast(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_mdp_step_cache()
        if self._mdp_reference_root_cache is None:
            compiled = _get_mdp_compiled_module()
            reference = self.current_reference
            root_pos_w, root_quat_w = compiled.transform_root_pose_to_world(
                self._mdp_align_quat,
                self._mdp_align_pos,
                reference["root_pos"],
                reference["root_quat"],
            )
            root_lin_vel_w, root_ang_vel_w = compiled.transform_root_velocity_to_world(
                self._mdp_align_quat,
                reference["root_lin_vel"],
                reference["root_ang_vel"],
            )
            self._mdp_reference_root_cache = (
                root_pos_w,
                root_quat_w,
                root_lin_vel_w,
                root_ang_vel_w,
            )
        return self._mdp_reference_root_cache

    def _get_reference_cvel_fast(self) -> torch.Tensor:
        self._ensure_mdp_step_cache()
        if self._mdp_reference_cvel_cache is None:
            reference = self.current_reference
            self._mdp_reference_cvel_cache = torch.cat(
                [reference["body_ang_vel_w"], reference["body_lin_vel_w"]], dim=-1
            )
        return self._mdp_reference_cvel_cache

    def _get_reference_body_ids_fast(
        self, reference_body_names: Sequence[str]
    ) -> torch.Tensor:
        self._ensure_mdp_fast_paths()
        cache_key = tuple(reference_body_names)
        body_ids = self._mdp_reference_body_id_cache.get(cache_key)
        if body_ids is not None:
            return body_ids

        ref_indices: list[int] = []
        for name in cache_key:
            body_id = self._mdp_reference_body_name_to_id.get(name)
            if body_id is None:
                body_id = self._mdp_reference_body_name_to_id_lower.get(name.lower())
            if body_id is None:
                body_id = self._mdp_reference_body_name_to_id_normalized.get(
                    self._normalize_body_name_for_matching(name)
                )
            if body_id is None:
                body_id = self._mdp_body_name_to_id.get(name)
            if body_id is None:
                body_id = self._mdp_body_name_to_id_lower.get(name.lower())
            if body_id is None:
                body_id = self._mdp_body_name_to_id_normalized.get(
                    self._normalize_body_name_for_matching(name)
                )
            if body_id is not None and body_id >= self._mdp_reference_body_count:
                body_id = None
            if body_id is None:
                raise KeyError(
                    f"Reference body '{name}' not found in reference metadata."
                )
            ref_indices.append(body_id)

        body_ids = torch.tensor(ref_indices, dtype=torch.long, device=self.device)
        self._mdp_reference_body_id_cache[cache_key] = body_ids
        return body_ids

    def _get_reference_body_pose_w_fast(
        self, reference_body_names: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_mdp_step_cache()
        cache_key = tuple(reference_body_names)
        body_pose = self._mdp_reference_body_pose_cache.get(cache_key)
        if body_pose is None:
            compiled = _get_mdp_compiled_module()
            ref_body_ids = self._get_reference_body_ids_fast(cache_key)
            reference = self.current_reference
            ref_pos = reference[self._mdp_reference_body_pos_key].index_select(
                1, ref_body_ids
            )
            ref_quat = reference[self._mdp_reference_body_quat_key].index_select(
                1, ref_body_ids
            )
            body_pose = compiled.transform_body_pose_to_world(
                self._mdp_align_quat,
                self._mdp_align_pos,
                ref_pos,
                ref_quat,
            )
            self._mdp_reference_body_pose_cache[cache_key] = body_pose
        return body_pose

    def _get_reference_body_velocity_w_fast(
        self, reference_body_names: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_mdp_step_cache()
        cache_key = tuple(reference_body_names)
        body_velocity = self._mdp_reference_body_velocity_cache.get(cache_key)
        if body_velocity is None:
            compiled = _get_mdp_compiled_module()
            ref_body_ids = self._get_reference_body_ids_fast(cache_key)
            ref_cvel = self._get_reference_cvel_fast().index_select(1, ref_body_ids)
            body_velocity = compiled.transform_body_velocity_to_world(
                self._mdp_align_quat, ref_cvel
            )
            self._mdp_reference_body_velocity_cache[cache_key] = body_velocity
        return body_velocity

    def _get_robot_anchor_body_id_fast(self, anchor_body_name: str) -> int:
        self._ensure_mdp_fast_paths()
        anchor_body_id = self._mdp_robot_anchor_id_cache.get(anchor_body_name)
        if anchor_body_id is None:
            anchor_body_id = self._mdp_body_name_to_id.get(anchor_body_name)
            if anchor_body_id is None:
                anchor_body_id = self._mdp_body_name_to_id_lower.get(
                    anchor_body_name.lower()
                )
            if anchor_body_id is None:
                anchor_body_id = self._mdp_body_name_to_id_normalized[
                    self._normalize_body_name_for_matching(anchor_body_name)
                ]
            self._mdp_robot_anchor_id_cache[anchor_body_name] = anchor_body_id
        return anchor_body_id

    def _body_ids_cache_key(
        self, body_ids: Sequence[int] | torch.Tensor | slice
    ) -> object:
        if isinstance(body_ids, slice):
            return self._mdp_all_body_ids_key
        if isinstance(body_ids, torch.Tensor):
            if body_ids.device.type == "cpu":
                return tuple(int(body_id) for body_id in body_ids.tolist())
            return ("tensor", int(body_ids.data_ptr()), int(body_ids.numel()))
        return tuple(int(body_id) for body_id in body_ids)

    def _get_robot_anchor_state_w_fast(
        self, anchor_body_name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_mdp_step_cache()
        anchor_body_id = self._get_robot_anchor_body_id_fast(anchor_body_name)
        anchor_state = self._mdp_robot_anchor_state_cache.get(anchor_body_id)
        if anchor_state is None:
            anchor_state = (
                self.robot.data.body_pos_w[:, anchor_body_id],
                self.robot.data.body_quat_w[:, anchor_body_id],
            )
            self._mdp_robot_anchor_state_cache[anchor_body_id] = anchor_state
        return anchor_state

    def _get_robot_body_pose_w_fast(
        self, body_ids: Sequence[int] | torch.Tensor | slice
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_mdp_step_cache()
        body_ids_key = self._body_ids_cache_key(body_ids)
        body_pose = self._mdp_robot_body_pose_w_cache.get(body_ids_key)
        if body_pose is not None:
            return body_pose
        body_ids_t = self._get_body_ids_tensor_fast(body_ids)
        if isinstance(body_ids_t, slice):
            body_pose = (self.robot.data.body_pos_w, self.robot.data.body_quat_w)
        else:
            body_pose = (
                self.robot.data.body_pos_w.index_select(1, body_ids_t),
                self.robot.data.body_quat_w.index_select(1, body_ids_t),
            )
        self._mdp_robot_body_pose_w_cache[body_ids_key] = body_pose
        return body_pose

    def _get_robot_body_velocity_w_fast(
        self, body_ids: Sequence[int] | torch.Tensor | slice
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_mdp_step_cache()
        body_ids_key = self._body_ids_cache_key(body_ids)
        body_velocity = self._mdp_robot_body_velocity_w_cache.get(body_ids_key)
        if body_velocity is not None:
            return body_velocity
        body_ids_t = self._get_body_ids_tensor_fast(body_ids)
        if isinstance(body_ids_t, slice):
            body_velocity = (
                self.robot.data.body_ang_vel_w,
                self.robot.data.body_lin_vel_w,
            )
        else:
            body_velocity = (
                self.robot.data.body_ang_vel_w.index_select(1, body_ids_t),
                self.robot.data.body_lin_vel_w.index_select(1, body_ids_t),
            )
        self._mdp_robot_body_velocity_w_cache[body_ids_key] = body_velocity
        return body_velocity

    def _get_robot_body_state_in_anchor_frame_fast(
        self,
        body_ids: Sequence[int] | torch.Tensor | slice,
        anchor_body_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_mdp_step_cache()
        anchor_body_id = self._get_robot_anchor_body_id_fast(anchor_body_name)
        body_ids_key = self._body_ids_cache_key(body_ids)
        cache_key = (anchor_body_id, body_ids_key)
        body_state = self._mdp_robot_body_anchor_frame_cache.get(cache_key)
        if body_state is not None:
            return body_state

        compiled = _get_mdp_compiled_module()
        robot_anchor_pos_w, robot_anchor_quat_w = self._get_robot_anchor_state_w_fast(
            anchor_body_name
        )
        body_pos_w, body_quat_w = self._get_robot_body_pose_w_fast(body_ids)
        body_state = compiled.body_pose_in_anchor_frame(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            body_pos_w,
            body_quat_w,
        )
        self._mdp_robot_body_anchor_frame_cache[cache_key] = body_state
        return body_state

    def _get_reference_motion_command_fast(
        self, joint_ids: Sequence[int] | slice
    ) -> torch.Tensor:
        self._ensure_mdp_step_cache()
        if isinstance(joint_ids, slice):
            return torch.cat(
                [
                    self.current_reference["joint_pos"],
                    self.current_reference["joint_vel"],
                ],
                dim=-1,
            )

        joint_ids_t = self._get_joint_ids_tensor_fast(joint_ids)
        cache_key = tuple(int(joint_id) for joint_id in joint_ids)
        motion_command = self._mdp_reference_motion_cache.get(cache_key)
        if motion_command is None:
            motion_command = torch.cat(
                [
                    self.current_reference["joint_pos"].index_select(-1, joint_ids_t),
                    self.current_reference["joint_vel"].index_select(-1, joint_ids_t),
                ],
                dim=-1,
            )
            self._mdp_reference_motion_cache[cache_key] = motion_command
        return motion_command

    def get_agent_latent_command(
        self, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the current agent-published latent command buffer."""
        if env_ids is None:
            return self._agent_latent_command
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        return self._agent_latent_command.index_select(0, env_ids)

    def set_agent_latent_command(
        self, latent_command: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        """Publish the latest agent latent command into the env observation state."""
        latent_command = latent_command.to(device=self.device, dtype=torch.float32)
        if env_ids is None:
            if (
                latent_command.ndim != 2
                or latent_command.shape != self._agent_latent_command.shape
            ):
                raise ValueError(
                    "Latent command shape mismatch. "
                    f"Expected {tuple(self._agent_latent_command.shape)}, got {tuple(latent_command.shape)}."
                )
            self._agent_latent_command.copy_(latent_command)
            return

        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if latent_command.ndim != 2 or latent_command.shape != (
            env_ids.shape[0],
            self._agent_latent_dim,
        ):
            raise ValueError(
                "Latent command shape mismatch for indexed update. "
                f"Expected {(env_ids.shape[0], self._agent_latent_dim)}, got {tuple(latent_command.shape)}."
            )
        self._agent_latent_command.index_copy_(0, env_ids, latent_command)

    def reset_agent_latent_command(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset latent commands for the selected environments to zeros."""
        if env_ids is None:
            self._agent_latent_command.zero_()
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self._agent_latent_command.index_fill_(0, env_ids, 0.0)

    def _resolve_reference_body_visualization_pairs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]] | None:
        """Resolve pairs of (reference body idx, robot body idx) to visualize."""
        if len(self.reference_body_names) == 0:
            return None

        reference_body_pos = None
        reference_body_quat = None
        try:
            reference_body_pos = self.get_reference_data("xpos")
            reference_body_quat = self.get_reference_data("xquat")
        except KeyError:
            pass
        if reference_body_pos is None or reference_body_quat is None:
            return None
        num_reference_bodies = int(reference_body_pos.shape[1])

        robot_body_names = list(self.robot.body_names)
        robot_name_lookup = {name: idx for idx, name in enumerate(robot_body_names)}
        robot_name_lookup_lower = {
            name.lower(): idx for idx, name in enumerate(robot_body_names)
        }
        robot_normalized_lookup: dict[str, list[int]] = {}
        robot_normalized_names: list[str] = []

        for idx, body_name in enumerate(robot_body_names):
            normalized_name = self._normalize_body_name_for_matching(body_name)
            robot_normalized_names.append(normalized_name)
            robot_normalized_lookup.setdefault(normalized_name, []).append(idx)

        selected_ref_ids: list[int] = []
        selected_robot_ids: list[int] = []
        selected_names: list[str] = []
        used_robot_ids: set[int] = set()

        for ref_id, ref_body_name in enumerate(self.reference_body_names):
            if ref_id >= num_reference_bodies:
                continue
            robot_id: int | None = None

            if ref_body_name in robot_name_lookup:
                robot_id = robot_name_lookup[ref_body_name]
            else:
                ref_body_name_lower = ref_body_name.lower()
                if ref_body_name_lower in robot_name_lookup_lower:
                    robot_id = robot_name_lookup_lower[ref_body_name_lower]
                else:
                    normalized_ref_name = self._normalize_body_name_for_matching(
                        ref_body_name
                    )
                    normalized_matches = robot_normalized_lookup.get(
                        normalized_ref_name, []
                    )
                    if len(normalized_matches) > 0:
                        robot_id = normalized_matches[0]
                    else:
                        prefix_matches = [
                            idx
                            for idx, normalized_robot_name in enumerate(
                                robot_normalized_names
                            )
                            if normalized_robot_name.startswith(normalized_ref_name)
                            or normalized_ref_name.startswith(normalized_robot_name)
                        ]
                        if len(prefix_matches) > 0:
                            robot_id = prefix_matches[0]

            if robot_id is None:
                continue
            if robot_id in used_robot_ids:
                continue

            used_robot_ids.add(robot_id)
            selected_ref_ids.append(ref_id)
            selected_robot_ids.append(robot_id)
            selected_names.append(ref_body_name)

        if len(selected_ref_ids) == 0:
            return None
        return (
            torch.tensor(selected_ref_ids, dtype=torch.long, device=self.device),
            torch.tensor(selected_robot_ids, dtype=torch.long, device=self.device),
            selected_names,
        )

    def _get_reference_alignment_transform(
        self, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-env rigid transform from dataset world frame to simulation world frame.

        Alignment is yaw-only (rotation around z-axis), to avoid injecting small roll/pitch
        frame mismatches that can tilt reference motion upward/downward.
        """
        if env_ids is None:
            init_pos = self._init_root_pos
            init_quat = self._init_root_quat
            ref_reset_pos = self._reference_reset_root_pos
            ref_reset_quat = self._reference_reset_root_quat
        else:
            init_pos = self._init_root_pos[env_ids]
            init_quat = self._init_root_quat[env_ids]
            ref_reset_pos = self._reference_reset_root_pos[env_ids]
            ref_reset_quat = self._reference_reset_root_quat[env_ids]

        init_yaw_quat = math_utils.yaw_quat(init_quat)
        ref_reset_yaw_quat = math_utils.yaw_quat(ref_reset_quat)
        align_quat = math_utils.quat_mul(
            init_yaw_quat, math_utils.quat_inv(ref_reset_yaw_quat)
        )
        align_pos = init_pos - math_utils.quat_apply(align_quat, ref_reset_pos)
        return align_quat, align_pos

    def _transform_reference_pose_to_world(
        self,
        ref_pos: torch.Tensor,
        ref_quat: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply per-episode rigid transform from reference frame to world frame."""
        align_quat, align_pos = self._get_reference_alignment_transform(env_ids)

        if ref_pos.ndim == 2:
            pos_w = math_utils.quat_apply(align_quat, ref_pos) + align_pos
            if ref_quat is None:
                return pos_w, None
            quat_w = math_utils.quat_mul(align_quat, ref_quat)
            return pos_w, quat_w

        if ref_pos.ndim != 3:
            raise ValueError(
                f"Unsupported ref_pos shape for transform: {tuple(ref_pos.shape)}"
            )

        num_envs, num_items = ref_pos.shape[0], ref_pos.shape[1]
        align_quat_expand = (
            align_quat.unsqueeze(1).expand(-1, num_items, -1).reshape(-1, 4)
        )
        pos_w = math_utils.quat_apply(
            align_quat_expand, ref_pos.reshape(-1, 3)
        ).reshape(num_envs, num_items, 3)
        pos_w = pos_w + align_pos.unsqueeze(1)

        if ref_quat is None:
            return pos_w, None
        quat_w = math_utils.quat_mul(
            align_quat_expand, ref_quat.reshape(-1, 4)
        ).reshape(num_envs, num_items, 4)
        return pos_w, quat_w

    def _transform_reference_body_pose_to_init_alignment(
        self,
        ref_pos: torch.Tensor,
        ref_quat: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Map reference body pose with reset-time alignment so global motion is preserved."""
        return self._transform_reference_pose_to_world(
            ref_pos, ref_quat, env_ids=env_ids
        )

    def _index_copy_reference_rows_(
        self, dst: TensorDict, src: TensorDict, env_ids: torch.Tensor
    ) -> None:
        for key in src.keys():
            src_value = src.get(key)
            dst_value = dst.get(key)
            if isinstance(src_value, TensorDict) and isinstance(dst_value, TensorDict):
                self._index_copy_reference_rows_(dst_value, src_value, env_ids)
                continue
            if isinstance(src_value, torch.Tensor) and isinstance(
                dst_value, torch.Tensor
            ):
                dst_value.index_copy_(0, env_ids, src_value)
                continue
            dst.set(key, src_value)

    def _refresh_current_reference(
        self, env_ids: torch.Tensor | None = None, *, advance: bool = False
    ) -> None:
        reference = self.trajectory_manager.sample(env_ids=env_ids, advance=advance)
        if env_ids is None or self.current_reference is None:
            self.current_reference = reference
        else:
            self._index_copy_reference_rows_(self.current_reference, reference, env_ids)
        self._invalidate_mdp_cache()

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset the specified environments.

        Notes:
            IsaacLab managers, events, and sensors accept tensor indices and internally move
            them to the appropriate device. We normalize ``env_ids`` to a CUDA long tensor so
            that all internal buffers (which live on ``self.device``) and the trajectory
            manager see consistent indexing.
        """

        # Reset trajectory tracking (reassigns trajectories and resets steps)
        self.trajectory_manager.reset_envs(env_ids.clone())
        self.reset_agent_latent_command(env_ids)

        # Refresh only the resetting rows before reset events consume current_reference.
        self._refresh_current_reference(env_ids, advance=False)

        # Trigger the reset events (curriculum, sensors, managers, etc.) using tensor indices
        result = super()._reset_idx(env_ids)  # type: ignore[arg-type]

        # Store initial poses for replay/alignment.
        reset_root_state_w = self.robot.data.root_state_w.index_select(0, env_ids)
        self._init_root_pos.index_copy_(0, env_ids, reset_root_state_w[:, 0:3])
        self._init_root_quat.index_copy_(0, env_ids, reset_root_state_w[:, 3:7])

        reference_root_pos = self.current_reference.get("root_pos")
        reference_root_quat = self.current_reference.get("root_quat")
        if reference_root_pos is not None:
            self._reference_reset_root_pos.index_copy_(
                0, env_ids, reference_root_pos.index_select(0, env_ids)
            )
        if reference_root_quat is not None:
            self._reference_reset_root_quat.index_copy_(
                0, env_ids, reference_root_quat.index_select(0, env_ids)
            )
        if self.replay_reference:
            self._replay_reference(env_ids)

        tracked_root_pos_w = self._get_tracked_reference_root_pos_w()
        if tracked_root_pos_w is not None:
            self._last_tracked_root_pos_w.index_copy_(
                0, env_ids, tracked_root_pos_w.index_select(0, env_ids)
            )
            self._last_tracked_root_pos_valid.index_fill_(0, env_ids, True)

        return result

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Step the environment and update reference data."""
        # Standard RL stepping path.
        if not self.replay_only:
            # Get next reference data point (advance=True to move to next step)
            self._refresh_current_reference(advance=True)
            rollout_action_log = self._compute_rollout_reference_action_log(
                action.to(self.device)
            )
            step_return = super().step(action)
            rollout_state_log = self._compute_rollout_reference_state_log()
            if len(rollout_action_log) > 0 or len(rollout_state_log) > 0:
                self.extras.setdefault("log", {}).update(rollout_action_log)
                self.extras.setdefault("log", {}).update(rollout_state_log)
            # self._update_reference_velocity_visualizer()
            return step_return

        # Replay-only path: ignore physics stepping and evaluate rewards exactly
        # on the replayed reference state.
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        # Sample the current reference frame and advance the internal step by exactly one.
        # `sample(advance=True)` returns frame t and then increments to t+1.
        # This avoids double-advance while keeping reward computation aligned with frame t.
        reference_for_step = self.trajectory_manager.sample(env_ids=None, advance=True)
        self.current_reference = reference_for_step
        self._invalidate_mdp_cache()
        self._replay_reference(reference=reference_for_step)
        self.scene.update(dt=0.0)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # Clear any stale terminal info from previous steps.
        for key in ("final_obs", "final_info"):
            if key in self.extras:
                del self.extras[key]

        if len(reset_env_ids) > 0:
            reset_env_ids_list = reset_env_ids.tolist()
            # Populate Gymnasium-style terminal observation info for vector envs.
            # final_obs/final_info are object arrays with None for non-reset envs.
            final_obs = np.empty(self.num_envs, dtype=object)
            final_obs[:] = None
            final_info = np.empty(self.num_envs, dtype=object)
            final_info[:] = None

            def _slice_obs(obs: dict | torch.Tensor, env_id: int):
                if isinstance(obs, dict):
                    return {k: _slice_obs(v, env_id) for k, v in obs.items()}
                return obs[env_id].clone()

            for env_id in reset_env_ids_list:
                final_obs[env_id] = _slice_obs(self.obs_buf, env_id)
                final_info[env_id] = {}

            self.extras["final_obs"] = final_obs
            self.extras["final_info"] = final_info

            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids_list)

            self._reset_idx(reset_env_ids)

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids_list)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # Expose post-step reference (frame t+1) for observations/outputs, matching
        # ManagerBasedRLEnv command timing after command_manager.compute().
        self._refresh_current_reference(advance=False)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)
        self._update_reference_velocity_visualizer()
        self._update_env0_velocity_metrics()
        # return observations, rewards, resets and extras
        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    def get_reference_data(
        self, key: str | None = None, joint_indices: Sequence[int] | None = None
    ) -> TensorDict | torch.Tensor:
        """
        Get the current reference data.

        Args:
            key: Specific key to extract. If None, returns full TensorDict.

        Returns:
            Reference data for all environments
        """
        if self.current_reference is None:
            raise RuntimeError("No reference data available. Call reset() first.")

        if key is None:
            return self.current_reference

        data: torch.Tensor | TensorDict | None = None
        if key in self.current_reference:
            data = self.current_reference[key]
        elif key == "xpos" and "body_pos_w" in self.current_reference:
            data = self.current_reference["body_pos_w"]
        elif key == "xquat" and "body_quat_w" in self.current_reference:
            data = self.current_reference["body_quat_w"]
        elif key == "cvel":
            data = self._get_reference_cvel_fast()

        if data is None:
            available_keys = [str(k) for k in self.current_reference.keys()]
            raise KeyError(f"Key '{key}' not found. Available keys: {available_keys}")

        if joint_indices is not None:
            if isinstance(data, torch.Tensor):
                return data[..., joint_indices]
            else:
                # Handle TensorDict case - data should be a Tensor
                return data[..., joint_indices]  # type: ignore[return-value]
        else:
            return data  # type: ignore[return-value]

    @staticmethod
    def _normalize_nested_key(key: NestedKey) -> tuple[str, ...]:
        """Normalize a nested key to tuple form."""
        if isinstance(key, tuple):
            return key
        return (key,)

    @staticmethod
    def _denormalize_nested_key(key_parts: tuple[str, ...]) -> NestedKey:
        """Convert tuple-form key back to str when single-token."""
        if len(key_parts) == 1:
            return key_parts[0]
        return key_parts

    def _sample_reference_batch_for_expert(
        self, batch_size: int
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor]:
        """Sample random reference transitions without advancing env manager state."""
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")

        reference, env_ids_tm, global_indices = (
            self.trajectory_manager.sample_random_transitions(batch_size)
        )
        return (
            reference.to(self.device),
            env_ids_tm.to(self.device),
            global_indices.to(self.device),
        )

    def _reference_obs_by_term(
        self,
        reference: TensorDict,
        env_ids: torch.Tensor,
        *,
        prefix: tuple[str, ...] = (),
    ) -> dict[str, torch.Tensor]:
        """Convert sampled reference data to observation-compatible tensors."""
        env_ids = env_ids.to(dtype=torch.int64, device=self.device)
        key = (
            (lambda name: name) if len(prefix) == 0 else (lambda name: (*prefix, name))
        )
        root_pos_ref = reference.get(key("root_pos"))
        root_quat_ref = reference.get(key("root_quat"))
        root_lin_vel_ref = reference.get(key("root_lin_vel"))
        root_ang_vel_ref = reference.get(key("root_ang_vel"))
        joint_pos_ref = reference.get(key("joint_pos"))
        joint_vel_ref = reference.get(key("joint_vel"))
        if (
            root_pos_ref is None
            or root_quat_ref is None
            or root_lin_vel_ref is None
            or root_ang_vel_ref is None
            or joint_pos_ref is None
            or joint_vel_ref is None
        ):
            raise KeyError(
                f"Reference batch is missing fields for prefix {prefix or ('current',)}."
            )
        root_pos_w, root_quat_w_opt = self._transform_reference_pose_to_world(
            root_pos_ref, root_quat_ref, env_ids=env_ids
        )
        if root_quat_w_opt is None:
            raise RuntimeError(
                "Failed to transform reference quaternion for expert sampling."
            )
        root_quat_w = root_quat_w_opt
        scene = getattr(self, "scene", None)
        if scene is None:
            env_origins = self._expert_env_origins.index_select(0, env_ids)
        else:
            env_origins = scene.env_origins.index_select(0, env_ids)
        root_pos = root_pos_w - env_origins
        align_quat, _ = self._get_reference_alignment_transform(env_ids)
        root_lin_vel = math_utils.quat_apply(align_quat, root_lin_vel_ref)
        root_ang_vel = math_utils.quat_apply(align_quat, root_ang_vel_ref)
        robot = getattr(self, "robot", None)
        if robot is None:
            default_joint_pos = self._expert_default_joint_pos.index_select(0, env_ids)
            default_joint_vel = self._expert_default_joint_vel.index_select(0, env_ids)
        else:
            default_joint_pos = robot.data.default_joint_pos.index_select(0, env_ids)
            default_joint_vel = robot.data.default_joint_vel.index_select(0, env_ids)
        batch_size = int(env_ids.shape[0])
        identity_rot6d = torch.zeros(
            (batch_size, 6),
            device=self.device,
            dtype=root_quat_w.dtype,
        )
        identity_rot6d[:, 0] = 1.0
        identity_rot6d[:, 4] = 1.0
        zero_anchor_pos = torch.zeros(
            (batch_size, 3),
            device=self.device,
            dtype=root_pos.dtype,
        )
        action_manager = getattr(self, "action_manager", None)
        current_action = getattr(action_manager, "action", None)
        if isinstance(current_action, torch.Tensor):
            zero_last_action = torch.zeros_like(current_action.index_select(0, env_ids))
        else:
            action_space = getattr(
                self, "single_action_space", getattr(self, "action_space", None)
            )
            action_shape = (
                tuple(int(dim) for dim in action_space.shape)
                if action_space is not None and getattr(action_space, "shape", None)
                else (1,)
            )
            zero_last_action = torch.zeros(
                (batch_size, *action_shape),
                device=self.device,
                dtype=torch.float32,
            )

        return {
            "joint_pos": joint_pos_ref,
            "joint_vel": joint_vel_ref,
            "joint_pos_rel": joint_pos_ref - default_joint_pos,
            "joint_vel_rel": joint_vel_ref - default_joint_vel,
            "root_pos": root_pos,
            "root_quat": root_quat_w,
            "root_lin_vel": root_lin_vel,
            "root_ang_vel": root_ang_vel,
            "reference_motion": torch.cat([joint_pos_ref, joint_vel_ref], dim=-1),
            # Expert-state observations treat robot and reference as aligned.
            "reference_anchor_pos_b": zero_anchor_pos,
            "reference_anchor_ori_b": identity_rot6d,
            # Previous expert action is not available from the reference sampler here.
            "last_action": zero_last_action,
            # Backward-compatible aliases.
            "base_lin_vel": root_lin_vel,
            "base_ang_vel": root_ang_vel,
        }

    def _reference_to_requested_obs(
        self,
        reference: TensorDict,
        env_ids: torch.Tensor,
        obs_keys: Sequence[NestedKey],
        *,
        prefix: tuple[str, ...] = (),
    ) -> dict[NestedKey, torch.Tensor] | None:
        """Map requested nested observation keys to tensors from sampled reference."""
        term_values = self._reference_obs_by_term(reference, env_ids, prefix=prefix)
        mapped_values: dict[NestedKey, torch.Tensor] = {}
        unknown_terms: list[str] = []

        for obs_key in obs_keys:
            key_tuple = self._normalize_nested_key(obs_key)
            term_name = key_tuple[-1]
            value = term_values.get(term_name)
            if value is None:
                unknown_terms.append(term_name)
                continue
            mapped_values[obs_key] = value

        if len(unknown_terms) > 0:
            for term_name in unknown_terms:
                if term_name in self._expert_sampler_warned_unknown_terms:
                    continue
                logger.warning(
                    "Expert sampler cannot provide term '%s' from trajectory manager.",
                    term_name,
                )
                self._expert_sampler_warned_unknown_terms.add(term_name)
            return None

        return mapped_values

    def sample_expert_batch(
        self, batch_size: int, required_keys: Sequence[NestedKey]
    ) -> TensorDict | None:
        """Sample an expert batch for imitation algorithms from trajectory manager."""
        if batch_size <= 0:
            return None
        if len(required_keys) == 0:
            return TensorDict({}, batch_size=[batch_size], device=self.device)

        dedup_required_keys = list(dict.fromkeys(required_keys))
        current_obs_keys: list[NestedKey] = []
        next_obs_keys: list[NestedKey] = []
        needs_action = False

        for key in dedup_required_keys:
            key_tuple = self._normalize_nested_key(key)
            if key_tuple == ("action",):
                needs_action = True
                continue
            if len(key_tuple) > 0 and key_tuple[0] == "next":
                if len(key_tuple) < 2:
                    continue
                next_obs_keys.append(self._denormalize_nested_key(key_tuple[1:]))
                continue
            current_obs_keys.append(self._denormalize_nested_key(key_tuple))

        expert_batch = TensorDict({}, batch_size=[batch_size], device=self.device)
        current_reference: TensorDict | None = None
        current_env_ids: torch.Tensor | None = None
        current_global_indices: torch.Tensor | None = None

        needs_current_transition = (
            len(current_obs_keys) > 0
            or needs_action
            or (len(next_obs_keys) > 0 and self._reference_has_aligned_next)
        )

        if needs_current_transition:
            current_reference, current_env_ids, current_global_indices = (
                self._sample_reference_batch_for_expert(batch_size)
            )

        if len(current_obs_keys) > 0:
            assert current_reference is not None and current_env_ids is not None
            mapped_current = self._reference_to_requested_obs(
                current_reference,
                current_env_ids,
                current_obs_keys,
            )
            if mapped_current is None:
                return None
            for key, value in mapped_current.items():
                expert_batch.set(key, value)

        if len(next_obs_keys) > 0:
            if self._reference_has_aligned_next:
                assert current_reference is not None and current_env_ids is not None
                next_reference = current_reference
                next_env_ids = current_env_ids
                next_prefix = ("next",)
            else:
                next_reference, next_env_ids, _ = (
                    self._sample_reference_batch_for_expert(batch_size)
                )
                next_prefix = ()
            mapped_next = self._reference_to_requested_obs(
                next_reference,
                next_env_ids,
                next_obs_keys,
                prefix=next_prefix,
            )
            if mapped_next is None:
                return None
            for key, value in mapped_next.items():
                key_tuple = self._normalize_nested_key(key)
                expert_batch.set(("next", *key_tuple), value)

        if needs_action:
            sampled_action = None
            if (
                self._reconstructed_reference_action_enabled
                and current_env_ids is not None
                and current_global_indices is not None
            ):
                sampled_action = self._sample_reconstructed_reference_actions(
                    global_indices=current_global_indices,
                    env_ids=current_env_ids,
                )
            if current_reference is not None and "action" in current_reference.keys():
                sampled_action = (
                    sampled_action
                    if sampled_action is not None
                    else current_reference.get("action")
                )
            if sampled_action is None:
                if not self._expert_sampler_warned_action_fallback:
                    logger.warning(
                        "Expert sampler action unavailable; using zero-action fallback."
                    )
                    self._expert_sampler_warned_action_fallback = True
                action_space = getattr(
                    self, "single_action_space", getattr(self, "action_space", None)
                )
                action_shape = (
                    tuple(int(dim) for dim in action_space.shape)
                    if action_space is not None and getattr(action_space, "shape", None)
                    else (1,)
                )
                sampled_action = torch.zeros(
                    (batch_size, *action_shape),
                    device=self.device,
                    dtype=torch.float32,
                )
            sampled_action = sampled_action.to(self.device)
            expert_batch.set("action", sampled_action)
            expert_batch.set("expert_action", sampled_action)

        return expert_batch

    def _replay_reference(
        self, env_ids: torch.Tensor | None = None, reference: TensorDict | None = None
    ):
        """Replay the reference data. If env_ids is provided, only replay the reference data for the given environments.
        If env_ids is not provided, replay the reference data for all environments."""

        if env_ids is None:
            ref = self.current_reference if reference is None else reference
            defaults_pos = self.robot.data.default_joint_pos
            defaults_vel = self.robot.data.default_joint_vel
        else:
            env_ids_tensor = env_ids
            full_reference = self.current_reference if reference is None else reference
            ref = full_reference[env_ids_tensor]
            defaults_pos = self.robot.data.default_joint_pos[env_ids_tensor]
            defaults_vel = self.robot.data.default_joint_vel[env_ids_tensor]

        root_pos, root_quat_opt = self._transform_reference_pose_to_world(
            ref["root_pos"], ref["root_quat"], env_ids=env_ids
        )
        if root_quat_opt is None:
            raise RuntimeError(
                "Failed to transform reference root quaternion for replay."
            )
        root_quat = root_quat_opt
        align_quat, _ = self._get_reference_alignment_transform(env_ids)
        root_lin_vel = self._estimate_reference_root_lin_vel_w_from_pos(
            ref["root_pos"], env_ids=env_ids
        )
        root_ang_vel = math_utils.quat_apply(align_quat, ref["root_ang_vel"])
        root_pose = torch.cat([root_pos, root_quat], dim=-1)
        root_vel = torch.cat([root_lin_vel, root_ang_vel], dim=-1)
        # Extract joint data from reference TensorDict
        # ref is a TensorDict, so accessing keys returns tensors
        joint_pos_raw = ref["joint_pos"]  # type: ignore[assignment]
        joint_vel_raw = ref["joint_vel"]  # type: ignore[assignment]
        joint_pos = joint_pos_raw.clone()
        joint_vel = joint_vel_raw.clone()

        # Replace NaN positions with default values
        joint_pos = torch.where(torch.isnan(joint_pos), defaults_pos, joint_pos)
        joint_vel = torch.where(torch.isnan(joint_vel), defaults_vel, joint_vel)
        # Use link/com-specific writers so all articulation data buffers stay coherent.
        # `base_lin_vel` uses root_com_vel_w + root_link_quat_w internally.
        self.robot.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_com_velocity_to_sim(root_vel, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_data_to_sim()
        # Refresh cached kinematics buffers (e.g. root_lin_vel_b) after direct state writes.
        self.scene.update(dt=0.0)
        self.robot.update(dt=0.0)
        self._invalidate_mdp_cache()

    def _get_tracked_reference_root_pos_w(self) -> torch.Tensor | None:
        """Return tracked reference root positions in world frame for all environments."""
        if self.current_reference is None:
            return None

        reference_root_pos = self.current_reference.get("root_pos")
        if reference_root_pos is None:
            return None

        # Apply the full per-episode rigid transform (R, t) from reset frame to world frame.
        tracked_root_pos_w, _ = self._transform_reference_pose_to_world(
            reference_root_pos
        )
        return tracked_root_pos_w

    def _estimate_reference_root_lin_vel_w_from_pos(
        self,
        reference_root_pos: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        update_cache: bool = False,
    ) -> torch.Tensor:
        """Estimate reference root linear velocity in world frame from finite differences of root position."""
        if env_ids is None:
            tracked_root_pos_w, _ = self._transform_reference_pose_to_world(
                reference_root_pos
            )
            previous_pos_w = self._last_tracked_root_pos_w
            previous_valid = self._last_tracked_root_pos_valid
        else:
            env_ids_tensor = env_ids.to(dtype=torch.int64)
            tracked_root_pos_w, _ = self._transform_reference_pose_to_world(
                reference_root_pos, env_ids=env_ids_tensor
            )
            previous_pos_w = self._last_tracked_root_pos_w[env_ids_tensor]
            previous_valid = self._last_tracked_root_pos_valid[env_ids_tensor]

        reference_root_lin_vel_w = torch.zeros_like(tracked_root_pos_w)
        dt = float(self.step_dt)
        if dt > 0.0:
            reference_root_lin_vel_w[previous_valid] = (
                tracked_root_pos_w[previous_valid] - previous_pos_w[previous_valid]
            ) / dt

        if update_cache:
            if env_ids is None:
                self._last_tracked_root_pos_w.copy_(tracked_root_pos_w)
                self._last_tracked_root_pos_valid.fill_(True)
            else:
                env_ids_tensor = env_ids.to(dtype=torch.int64)
                self._last_tracked_root_pos_w[env_ids_tensor] = tracked_root_pos_w
                self._last_tracked_root_pos_valid[env_ids_tensor] = True

        return reference_root_lin_vel_w

    def _setup_reference_velocity_visualizer(self) -> None:
        """Create desired/current frame markers for root and tracked bodies."""
        if not self._reference_vel_vis_enabled:
            return

        # Desired reference body (root) location and current robot root — frame markers like unitree_rl_lab
        goal_cfg = FRAME_MARKER_CFG.copy()
        goal_cfg.prim_path = "/Visuals/Imitation/reference_root_goal"
        goal_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        self._goal_root_frame_marker = VisualizationMarkers(goal_cfg)
        self._goal_root_frame_marker.set_visibility(True)
        current_cfg = FRAME_MARKER_CFG.copy()
        current_cfg.prim_path = "/Visuals/Imitation/current_root"
        current_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        self._current_root_frame_marker = VisualizationMarkers(current_cfg)
        self._current_root_frame_marker.set_visibility(True)

        body_id_pairs = self._resolve_reference_body_visualization_pairs()
        if body_id_pairs is None:
            return

        self._vis_reference_body_ids, self._vis_robot_body_ids, self._vis_body_names = (
            body_id_pairs
        )
        for body_name in self._vis_body_names:
            current_body_cfg = FRAME_MARKER_CFG.copy()
            current_body_cfg.prim_path = f"/Visuals/Imitation/current_body/{body_name}"
            current_body_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            current_body_marker = VisualizationMarkers(current_body_cfg)
            current_body_marker.set_visibility(True)
            self._current_body_frame_markers.append(current_body_marker)

            goal_body_cfg = FRAME_MARKER_CFG.copy()
            goal_body_cfg.prim_path = (
                f"/Visuals/Imitation/reference_body_goal/{body_name}"
            )
            goal_body_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            goal_body_marker = VisualizationMarkers(goal_body_cfg)
            goal_body_marker.set_visibility(True)
            self._goal_body_frame_markers.append(goal_body_marker)

    def _update_reference_velocity_visualizer(self) -> None:
        """Update marker pose/scale from current reference linear velocity."""
        if not self._reference_vel_vis_enabled:
            return
        if self.current_reference is None:
            return
        if not self.robot.is_initialized:
            return

        tracked_root_pos_w = self._get_tracked_reference_root_pos_w()

        # Desired reference body (root) location and current robot root — frame markers like unitree_rl_lab
        if (
            self._goal_root_frame_marker is not None
            and self._current_root_frame_marker is not None
        ):
            ref_root_pos_w = tracked_root_pos_w
            align_quat, _ = self._get_reference_alignment_transform()
            ref_root_quat_w = math_utils.quat_mul(
                align_quat, self.current_reference["root_quat"]
            )
            if ref_root_pos_w is not None:
                self._goal_root_frame_marker.visualize(
                    translations=ref_root_pos_w, orientations=ref_root_quat_w
                )
            self._current_root_frame_marker.visualize(
                translations=self.robot.data.root_pos_w,
                orientations=self.robot.data.root_quat_w,
            )

        if (
            self._vis_reference_body_ids is not None
            and self._vis_robot_body_ids is not None
            and len(self._goal_body_frame_markers)
            == len(self._current_body_frame_markers)
            and len(self._goal_body_frame_markers) > 0
        ):
            reference_body_pos = None
            reference_body_quat = None
            try:
                reference_body_pos = self.get_reference_data("xpos")
                reference_body_quat = self.get_reference_data("xquat")
            except KeyError:
                pass

            if reference_body_pos is not None:
                ref_body_pos = reference_body_pos[..., self._vis_reference_body_ids, :]
                if reference_body_quat is not None:
                    ref_body_quat = reference_body_quat[
                        ..., self._vis_reference_body_ids, :
                    ]
                else:
                    # Keep position-only visualization alive when orientation keys are unavailable.
                    ref_body_quat = torch.zeros(
                        (*ref_body_pos.shape[:-1], 4), device=ref_body_pos.device
                    )
                    ref_body_quat[..., 0] = 1.0

                ref_body_pos_w, ref_body_quat_w_opt = (
                    self._transform_reference_body_pose_to_init_alignment(
                        ref_body_pos, ref_body_quat
                    )
                )
                if ref_body_quat_w_opt is None:
                    return
                ref_body_quat_w = ref_body_quat_w_opt
                num_bodies = ref_body_pos.shape[1]

                robot_body_pos_w = self.robot.data.body_pos_w[
                    :, self._vis_robot_body_ids
                ]
                robot_body_quat_w = self.robot.data.body_quat_w[
                    :, self._vis_robot_body_ids
                ]

                for body_index in range(num_bodies):
                    self._current_body_frame_markers[body_index].visualize(
                        robot_body_pos_w[:, body_index],
                        robot_body_quat_w[:, body_index],
                    )
                    self._goal_body_frame_markers[body_index].visualize(
                        ref_body_pos_w[:, body_index], ref_body_quat_w[:, body_index]
                    )

    def _update_env0_velocity_metrics(self) -> None:
        """Expose env[0] velocity tracking metrics in extras for easy logging."""
        if self.current_reference is None or self.num_envs < 1:
            return
        reference_root_pos = self.current_reference.get("root_pos")
        if reference_root_pos is None:
            return

        reference_root_lin_vel_w = self._estimate_reference_root_lin_vel_w_from_pos(
            reference_root_pos, update_cache=True
        )
        reference_vel_env0 = reference_root_lin_vel_w[0]
        actual_vel_env0 = self.robot.data.root_lin_vel_w[0]
        diff_vel_env0 = actual_vel_env0 - reference_vel_env0

        self.extras["metrics/env0/reference_root_lin_vel_x"] = reference_vel_env0[
            0
        ].item()
        self.extras["metrics/env0/reference_root_lin_vel_y"] = reference_vel_env0[
            1
        ].item()
        self.extras["metrics/env0/reference_root_lin_vel_z"] = reference_vel_env0[
            2
        ].item()
        self.extras["metrics/env0/actual_root_lin_vel_x"] = actual_vel_env0[0].item()
        self.extras["metrics/env0/actual_root_lin_vel_y"] = actual_vel_env0[1].item()
        self.extras["metrics/env0/actual_root_lin_vel_z"] = actual_vel_env0[2].item()
        self.extras["metrics/env0/root_lin_vel_diff_x"] = diff_vel_env0[0].item()
        self.extras["metrics/env0/root_lin_vel_diff_y"] = diff_vel_env0[1].item()
        self.extras["metrics/env0/root_lin_vel_diff_z"] = diff_vel_env0[2].item()
        self.extras["metrics/env0/root_lin_vel_diff_norm"] = torch.linalg.norm(
            diff_vel_env0
        ).item()
