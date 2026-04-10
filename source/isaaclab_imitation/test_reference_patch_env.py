from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("pxr")

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab_imitation.envs.imitation_rl_env import ImitationRLEnv


class _WindowTrajectoryManager:
    def __init__(self, *, max_step: int, env_steps: torch.Tensor) -> None:
        self._state_device = torch.device("cpu")
        self._max_step = int(max_step)
        self.env_step = env_steps.to(dtype=torch.long, device=self._state_device)

    def sample_slice(self, batch_size, env_ids, start_steps, mode):
        assert mode == "independent"
        assert start_steps.shape == (env_ids.shape[0], batch_size)
        clamped = start_steps.clamp(min=0, max=self._max_step)
        joint_pos = clamped.to(dtype=torch.float32).unsqueeze(-1)
        joint_vel = joint_pos + 10.0
        body_pos = torch.zeros(
            env_ids.shape[0], batch_size, 1, 3, device=self._state_device
        )
        body_pos[..., 0] = clamped.to(dtype=torch.float32)
        body_quat = torch.zeros(
            env_ids.shape[0], batch_size, 1, 4, device=self._state_device
        )
        body_quat[..., 0] = 1.0
        return TensorDict(
            {
                "joint_pos": joint_pos,
                "joint_vel": joint_vel,
                "xpos": body_pos,
                "xquat": body_quat,
            },
            batch_size=[env_ids.shape[0], batch_size],
            device=self._state_device,
        )


class _ResetTrajectoryManager:
    def __init__(self) -> None:
        self._state_device = torch.device("cpu")
        self.steps = None

    def reset_envs(self, env_ids, steps=None):
        self.env_ids = env_ids.clone()
        self.steps = None if steps is None else steps.clone()

    def sample(self, env_ids=None, advance=False):
        batch_size = 2 if env_ids is None else int(env_ids.shape[0])
        return TensorDict(
            {
                "root_pos": torch.zeros(batch_size, 3),
                "root_quat": torch.zeros(batch_size, 4),
            },
            batch_size=[batch_size],
        )


def _make_env_for_patch_tests() -> ImitationRLEnv:
    env = ImitationRLEnv.__new__(ImitationRLEnv)
    env.device = torch.device("cpu")
    env.num_envs = 2
    env.current_expert_frame = TensorDict(
        {
            "root_pos": torch.zeros(2, 3),
            "root_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
            "root_lin_vel": torch.zeros(2, 3),
            "root_ang_vel": torch.zeros(2, 3),
            "joint_pos": torch.zeros(2, 1),
            "joint_vel": torch.zeros(2, 1),
        },
        batch_size=[2],
        device=env.device,
    )
    env.trajectory_manager = _WindowTrajectoryManager(
        max_step=4,
        env_steps=torch.tensor([0, 4]),
    )
    env._latent_patch_past_steps = 1
    env._latent_patch_future_steps = 1
    env._expert_sampler_warned_unknown_terms = set()
    env._expert_env_origins = torch.zeros(2, 3)
    env._mdp_reference_body_pos_key = "xpos"
    env._mdp_reference_body_quat_key = "xquat"
    env._mdp_expert_window_obs_cache = {}
    env._ensure_mdp_step_cache = lambda: None
    env._get_joint_ids_tensor_fast = (
        lambda joint_ids: joint_ids
        if isinstance(joint_ids, slice)
        else torch.as_tensor(joint_ids, dtype=torch.long, device=env.device)
    )
    env._get_reference_body_ids_fast = lambda body_names: torch.tensor(
        [0], dtype=torch.long, device=env.device
    )
    env._get_robot_anchor_state_w_fast = lambda anchor_body_name: (
        torch.zeros(env.num_envs, 3, device=env.device),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device).repeat(
            env.num_envs, 1
        ),
    )
    env._transform_reference_pose_to_world = (
        lambda ref_pos, ref_quat=None, env_ids=None: (ref_pos, ref_quat)
    )
    return env


def test_expert_window_slice_clamps_left_and_right() -> None:
    env = _make_env_for_patch_tests()
    expert_window = env._sample_expert_window_slice(
        env_ids=torch.tensor([0, 1]),
        local_steps=torch.tensor([0, 4]),
        past_steps=2,
        future_steps=2,
    )
    values = expert_window.get("joint_pos").squeeze(-1)
    expected = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 2.0], [2.0, 3.0, 4.0, 4.0, 4.0]],
    )
    assert torch.equal(values, expected)


def test_get_current_expert_window_term_returns_motion_window() -> None:
    env = _make_env_for_patch_tests()
    motion = env.get_current_expert_window_term(
        term_name="expert_motion",
        past_steps=1,
        future_steps=1,
    )
    expected = torch.tensor(
        [[0.0, 10.0, 0.0, 10.0, 1.0, 11.0], [3.0, 13.0, 4.0, 14.0, 4.0, 14.0]]
    )
    assert torch.equal(motion, expected)


def test_reset_idx_randomizes_steps_within_configured_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ManagerBasedRLEnv, "_reset_idx", lambda self, env_ids: None)

    env = ImitationRLEnv.__new__(ImitationRLEnv)
    env.device = torch.device("cpu")
    env.trajectory_manager = _ResetTrajectoryManager()
    env._random_reset_step_min = 0
    env._random_reset_step_max = 200
    env.reset_agent_latent_command = lambda env_ids=None: None
    env._refresh_current_expert_frame = lambda env_ids=None, advance=False: None
    env.robot = SimpleNamespace(data=SimpleNamespace(root_state_w=torch.zeros(2, 13)))
    env._init_root_pos = torch.zeros(2, 3)
    env._init_root_quat = torch.zeros(2, 4)
    env._reference_reset_root_pos = torch.zeros(2, 3)
    env._reference_reset_root_quat = torch.zeros(2, 4)
    env.current_expert_frame = TensorDict(
        {
            "root_pos": torch.zeros(2, 3),
            "root_quat": torch.zeros(2, 4),
        },
        batch_size=[2],
    )
    env.replay_reference = False
    env._get_tracked_reference_root_pos_w = lambda: None
    env._last_tracked_root_pos_w = torch.zeros(2, 3)
    env._last_tracked_root_pos_valid = torch.zeros(2, dtype=torch.bool)

    env._reset_idx(torch.tensor([0, 1], dtype=torch.long))

    assert env.trajectory_manager.steps is not None
    assert env.trajectory_manager.steps.min().item() >= 0
    assert env.trajectory_manager.steps.max().item() <= 200


def test_reset_idx_uses_default_start_when_random_range_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ManagerBasedRLEnv, "_reset_idx", lambda self, env_ids: None)

    env = ImitationRLEnv.__new__(ImitationRLEnv)
    env.device = torch.device("cpu")
    env.trajectory_manager = _ResetTrajectoryManager()
    env._random_reset_step_min = 0
    env._random_reset_step_max = 0
    env.reset_agent_latent_command = lambda env_ids=None: None
    env._refresh_current_expert_frame = lambda env_ids=None, advance=False: None
    env.robot = SimpleNamespace(data=SimpleNamespace(root_state_w=torch.zeros(1, 13)))
    env._init_root_pos = torch.zeros(1, 3)
    env._init_root_quat = torch.zeros(1, 4)
    env._reference_reset_root_pos = torch.zeros(1, 3)
    env._reference_reset_root_quat = torch.zeros(1, 4)
    env.current_expert_frame = TensorDict(
        {
            "root_pos": torch.zeros(1, 3),
            "root_quat": torch.zeros(1, 4),
        },
        batch_size=[1],
    )
    env.replay_reference = False
    env._get_tracked_reference_root_pos_w = lambda: None
    env._last_tracked_root_pos_w = torch.zeros(1, 3)
    env._last_tracked_root_pos_valid = torch.zeros(1, dtype=torch.bool)

    env._reset_idx(torch.tensor([0], dtype=torch.long))

    assert env.trajectory_manager.steps is None
