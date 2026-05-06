from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

try:
    from isaaclab.app import AppLauncher
except ModuleNotFoundError as exc:
    pytest.skip(
        f"Isaac Sim app launcher is unavailable: {exc}", allow_module_level=True
    )

simulation_app = AppLauncher(headless=True).app

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv  # noqa: E402
from isaaclab_imitation.envs.imitation_rl_env import ImitationRLEnv  # noqa: E402


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
        body_pos[..., 0] = clamped.to(dtype=torch.float32).unsqueeze(-1)
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
        self.env_traj_rank = torch.zeros(2, dtype=torch.long)
        self.env_step = torch.zeros(2, dtype=torch.long)
        self._length = torch.tensor([300, 40], dtype=torch.long)

    def reset_envs(self, env_ids, steps=None):
        self.env_ids = env_ids.clone()
        self.env_traj_rank[env_ids] = env_ids % 2
        self.steps = None if steps is None else steps.clone()
        if steps is None:
            self.env_step[env_ids] = 0
        else:
            self.env_step[env_ids] = steps

    def _set_env_steps(self, env_ids, steps):
        self.env_ids = env_ids.clone()
        self.steps = steps.clone()
        self.env_step[env_ids] = steps

    def sample(self, env_ids=None, advance=False):
        batch_size = 2 if env_ids is None else int(env_ids.shape[0])
        return TensorDict(
            {
                "root_pos": torch.zeros(batch_size, 3),
                "root_quat": torch.zeros(batch_size, 4),
            },
            batch_size=[batch_size],
        )


class _StepTrajectoryManager:
    def __init__(self) -> None:
        self._state_device = torch.device("cpu")
        self.env_step = torch.zeros(1, dtype=torch.long)
        self.advance_calls = 0

    def sample(self, env_ids=None, advance=False):
        if env_ids is None:
            env_ids_t = torch.arange(self.env_step.shape[0], dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, dtype=torch.long)
        frame = self.env_step.index_select(0, env_ids_t).to(dtype=torch.float32)
        reference = TensorDict(
            {"frame": frame.unsqueeze(-1)},
            batch_size=[int(env_ids_t.shape[0])],
        )
        if advance:
            self.env_step[env_ids_t] += 1
            self.advance_calls += 1
        return reference


class _SyncTrajectoryManager:
    def __init__(self) -> None:
        self._state_device = torch.device("cpu")
        self.env_traj_rank = torch.tensor([0, 1, 2], dtype=torch.long)
        self.env_step = torch.tensor([10, 20, 30], dtype=torch.long)

    def set_env_cursor(self, *, env_ids, ranks, steps):
        self.env_traj_rank[env_ids] = ranks
        self.env_step[env_ids] = steps

    def sample(self, env_ids=None, advance=False):
        assert not advance
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long)
        return TensorDict(
            {
                "frame": self.env_step.index_select(0, env_ids_t)
                .to(dtype=torch.float32)
                .unsqueeze(-1),
                "rank": self.env_traj_rank.index_select(0, env_ids_t)
                .to(dtype=torch.float32)
                .unsqueeze(-1),
            },
            batch_size=[int(env_ids_t.shape[0])],
        )


class _FrameObservationManager:
    def __init__(self, env: ImitationRLEnv) -> None:
        self.env = env
        self.update_history_flags: list[bool] = []

    def compute(self, update_history=False):
        self.update_history_flags.append(bool(update_history))
        return {"policy": {"frame": self.env.current_expert_frame["frame"].clone()}}


class _TerminationManager:
    def __init__(self, terms: dict[str, tuple[bool, torch.Tensor]]) -> None:
        self._terms = terms
        self.active_terms = list(terms.keys())

    def get_term_cfg(self, term_name):
        return SimpleNamespace(time_out=self._terms[term_name][0])

    def get_term(self, term_name):
        return self._terms[term_name][1]


def _make_uninitialized_env(num_envs: int) -> ImitationRLEnv:
    env = ImitationRLEnv.__new__(ImitationRLEnv)
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env.scene = SimpleNamespace(
        num_envs=num_envs,
        env_origins=torch.zeros(num_envs, 3),
    )
    env._is_closed = True
    return env


def _make_env_for_patch_tests() -> ImitationRLEnv:
    env = _make_uninitialized_env(num_envs=2)
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
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device).repeat(env.num_envs, 1),
    )
    env._transform_reference_pose_to_world = (
        lambda ref_pos, ref_quat=None, env_ids=None: (ref_pos, ref_quat)
    )
    return env


def _make_env_for_expert_action_sampler(
    *,
    reconstructed_action: torch.Tensor | None,
    recorded_action: torch.Tensor | None = None,
) -> ImitationRLEnv:
    env = _make_uninitialized_env(num_envs=2)
    env._reference_has_aligned_next = True
    env._reconstructed_reference_action_enabled = reconstructed_action is not None
    env._latent_patch_past_steps = 0
    env._latent_patch_future_steps = 0
    env._expert_sampler_warned_unknown_terms = set()
    env._sample_expert_trajectory_batch = lambda batch_size: (
        TensorDict(
            {} if recorded_action is None else {"action": recorded_action[:batch_size]},
            batch_size=[batch_size],
            device=env.device,
        ),
        torch.arange(batch_size, device=env.device) % env.num_envs,
        torch.arange(batch_size, device=env.device),
    )
    env._expert_local_steps_from_global_indices = (
        lambda env_ids, global_indices: torch.zeros_like(global_indices)
    )
    if reconstructed_action is not None:
        env._sample_reconstructed_reference_actions = (
            lambda global_indices, env_ids: reconstructed_action[: env_ids.shape[0]]
        )
    env._map_requested_expert_observations = lambda *args, **kwargs: TensorDict(
        {},
        batch_size=[args[1].shape[0]],
        device=env.device,
    )
    return env


def test_expert_action_request_returns_reconstructed_reference_action() -> None:
    reconstructed_action = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32)
    env = _make_env_for_expert_action_sampler(reconstructed_action=reconstructed_action)

    batch = env.sample_expert_batch(3, ["expert_action"])

    assert torch.equal(batch["expert_action"], reconstructed_action)
    assert torch.equal(batch["action"], reconstructed_action)


def test_expert_action_request_raises_when_reconstruction_unavailable() -> None:
    env = _make_env_for_expert_action_sampler(reconstructed_action=None)

    with pytest.raises(RuntimeError, match="action/expert_action"):
        env.sample_expert_batch(3, ["expert_action"])


def test_standard_step_returns_next_reference_frame_without_double_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _base_step(self, action):
        assert self.current_expert_frame["frame"].item() == 0.0
        self.reward_buf = torch.tensor([10.0])
        self.reset_terminated = torch.tensor([False])
        self.reset_time_outs = torch.tensor([False])
        self.obs_buf = {"policy": {"frame": self.current_expert_frame["frame"].clone()}}
        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    monkeypatch.setattr(ManagerBasedRLEnv, "step", _base_step)

    env = _make_uninitialized_env(num_envs=1)
    env.replay_only = False
    env.extras = {}
    env.trajectory_manager = _StepTrajectoryManager()
    env.current_expert_frame = env.trajectory_manager.sample(advance=False)
    env._current_reference_local_step = torch.zeros(1, dtype=torch.long)
    env._invalidate_mdp_cache = lambda: None
    env.observation_manager = _FrameObservationManager(env)
    env._compute_rollout_reference_action_log = lambda action: {
        "timing/action_frame": float(env.current_expert_frame["frame"].item())
    }
    env._compute_rollout_reference_state_log = lambda: {
        "timing/state_frame": float(env.current_expert_frame["frame"].item())
    }

    initial_obs = env.observation_manager.compute(update_history=True)
    assert initial_obs["policy"]["frame"].item() == 0.0

    obs, reward, terminated, time_outs, extras = env.step(torch.zeros(1, 1))

    assert env.trajectory_manager.advance_calls == 1
    assert env.trajectory_manager.env_step.tolist() == [1]
    assert env.current_expert_frame["frame"].item() == 1.0
    assert obs["policy"]["frame"].item() == 1.0
    assert reward.item() == 10.0
    assert not terminated.item()
    assert not time_outs.item()
    assert extras["log"]["timing/action_frame"] == 0.0
    assert extras["log"]["timing/state_frame"] == 0.0


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


def test_sync_reference_cursor_updates_target_frame_only() -> None:
    env = _make_uninitialized_env(num_envs=3)
    env.trajectory_manager = _SyncTrajectoryManager()
    env.current_expert_frame = TensorDict(
        {
            "frame": torch.full((3, 1), -1.0),
            "rank": torch.full((3, 1), -1.0),
        },
        batch_size=[3],
    )
    env._current_reference_local_step = torch.zeros(3, dtype=torch.long)
    env._invalidate_mdp_cache = lambda: None
    env._get_tracked_reference_root_pos_w = lambda: None

    env.sync_reference_cursor_from_source_envs(
        source_env_ids=[2],
        target_env_ids=[0],
    )

    assert torch.equal(
        env.trajectory_manager.env_traj_rank,
        torch.tensor([2, 1, 2], dtype=torch.long),
    )
    assert torch.equal(
        env.trajectory_manager.env_step,
        torch.tensor([30, 20, 30], dtype=torch.long),
    )
    assert torch.equal(
        env.current_expert_frame["frame"].squeeze(-1),
        torch.tensor([30.0, -1.0, -1.0]),
    )
    assert torch.equal(
        env.current_expert_frame["rank"].squeeze(-1),
        torch.tensor([2.0, -1.0, -1.0]),
    )


def test_reference_transform_uses_env_origin_not_reset_alignment() -> None:
    env = _make_uninitialized_env(num_envs=2)
    env.scene = SimpleNamespace(
        num_envs=2,
        env_origins=torch.tensor(
            [[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]],
            dtype=torch.float32,
        ),
    )
    # Legacy reset-alignment state must not influence the Unitree-style transform.
    env._init_root_pos = torch.tensor(
        [[11.0, 0.0, 1.0], [2.0, 22.0, 1.0]],
        dtype=torch.float32,
    )
    env._init_root_quat = torch.tensor(
        [[0.70710677, 0.0, 0.0, 0.70710677], [1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    align_quat, align_pos = env._get_reference_alignment_transform()

    expected_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
    assert torch.equal(align_pos, env.scene.env_origins)
    assert torch.equal(align_quat, expected_quat)

    env_ids = torch.tensor([1], dtype=torch.long)
    ref_body_pos = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
        dtype=torch.float32,
    )
    ref_body_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    body_pos_w, body_quat_w = env._transform_reference_pose_to_world(
        ref_body_pos,
        ref_body_quat,
        env_ids=env_ids,
    )

    assert torch.equal(
        body_pos_w, ref_body_pos + env.scene.env_origins[1].view(1, 1, 3)
    )
    assert torch.equal(body_quat_w, ref_body_quat)


def test_reset_idx_randomizes_steps_within_configured_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ManagerBasedRLEnv, "_reset_idx", lambda self, env_ids: None)

    env = _make_uninitialized_env(num_envs=2)
    env.trajectory_manager = _ResetTrajectoryManager()
    env._random_reset_step_min = 0
    env._random_reset_step_max = 200
    env._random_reset_full_trajectory = False
    env.reset_agent_latent_command = lambda env_ids=None: None
    env._refresh_current_expert_frame = lambda env_ids=None, advance=False: None
    env.robot = SimpleNamespace(data=SimpleNamespace(root_state_w=torch.zeros(2, 13)))
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

    env = _make_uninitialized_env(num_envs=1)
    env.trajectory_manager = _ResetTrajectoryManager()
    env._random_reset_step_min = 0
    env._random_reset_step_max = 0
    env._random_reset_full_trajectory = False
    env.reset_agent_latent_command = lambda env_ids=None: None
    env._refresh_current_expert_frame = lambda env_ids=None, advance=False: None
    env.robot = SimpleNamespace(data=SimpleNamespace(root_state_w=torch.zeros(1, 13)))
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


def test_reset_idx_can_randomize_steps_across_full_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ManagerBasedRLEnv, "_reset_idx", lambda self, env_ids: None)

    env = _make_uninitialized_env(num_envs=2)
    env.trajectory_manager = _ResetTrajectoryManager()
    env._random_reset_step_min = 0
    env._random_reset_step_max = 0
    env._random_reset_full_trajectory = True
    env._adaptive_failure_reset_uniform_ratio = 0.1
    env._adaptive_failure_reset_alpha = 0.0
    env._adaptive_failure_reset_bin_count = 6
    env._adaptive_failure_reset_bin_failed_count = torch.zeros(6)
    env._adaptive_failure_reset_current_bin_failed = torch.zeros(6)
    env._current_reference_local_step = torch.zeros(2, dtype=torch.long)
    env.termination_manager = _TerminationManager({})
    env.reset_agent_latent_command = lambda env_ids=None: None
    env._refresh_current_expert_frame = lambda env_ids=None, advance=False: None
    env.robot = SimpleNamespace(data=SimpleNamespace(root_state_w=torch.zeros(2, 13)))
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
    assert env.trajectory_manager.steps[0].item() < 299
    assert env.trajectory_manager.steps[1].item() < 39


def test_adaptive_failure_bins_count_tracking_failures_only() -> None:
    env = _make_uninitialized_env(num_envs=4)
    env.trajectory_manager = _ResetTrajectoryManager()
    env.trajectory_manager.env_traj_rank = torch.zeros(4, dtype=torch.long)
    env.trajectory_manager.env_step = torch.zeros(4, dtype=torch.long)
    env.trajectory_manager._length = torch.tensor([100], dtype=torch.long)
    env._current_reference_local_step = torch.tensor([10, 20, 50, 75])
    env._adaptive_failure_reset_bin_count = 4
    env._adaptive_failure_reset_alpha = 1.0
    env._adaptive_failure_reset_bin_failed_count = torch.zeros(4)
    env._adaptive_failure_reset_current_bin_failed = torch.zeros(4)
    env.termination_manager = _TerminationManager(
        {
            "time_out": (True, torch.tensor([True, False, False, False])),
            "reference_finished": (False, torch.tensor([False, True, False, False])),
            "anchor_pos": (False, torch.tensor([False, False, True, False])),
            "base_too_low": (False, torch.tensor([False, False, False, True])),
        }
    )

    env._record_adaptive_failure_reset_bins(torch.tensor([0, 1, 2, 3]))

    assert torch.equal(
        env._adaptive_failure_reset_bin_failed_count,
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
    )


def test_adaptive_full_trajectory_reset_samples_after_reassignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ManagerBasedRLEnv, "_reset_idx", lambda self, env_ids: None)

    def _fake_multinomial(input, num_samples, replacement):
        assert num_samples == 2
        assert replacement
        return torch.tensor([3, 1], dtype=torch.long, device=input.device)

    def _fake_rand(size, *, device=None, dtype=None):
        if isinstance(size, int):
            size = (size,)
        return torch.full(size, 0.5, device=device, dtype=dtype or torch.float32)

    monkeypatch.setattr(torch, "multinomial", _fake_multinomial)
    monkeypatch.setattr(torch, "rand", _fake_rand)

    env = _make_uninitialized_env(num_envs=2)
    env.trajectory_manager = _ResetTrajectoryManager()
    env._random_reset_step_min = 0
    env._random_reset_step_max = 0
    env._random_reset_full_trajectory = True
    env._adaptive_failure_reset_uniform_ratio = 0.1
    env._adaptive_failure_reset_alpha = 0.0
    env._adaptive_failure_reset_bin_count = 4
    env._adaptive_failure_reset_bin_failed_count = torch.tensor([0.0, 0.0, 0.0, 100.0])
    env._adaptive_failure_reset_current_bin_failed = torch.zeros(4)
    env._current_reference_local_step = torch.zeros(2, dtype=torch.long)
    env.termination_manager = _TerminationManager({})
    env.reset_agent_latent_command = lambda env_ids=None: None
    env._refresh_current_expert_frame = lambda env_ids=None, advance=False: None
    env.robot = SimpleNamespace(data=SimpleNamespace(root_state_w=torch.zeros(2, 13)))
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

    assert torch.equal(env.trajectory_manager.env_traj_rank, torch.tensor([0, 1]))
    assert torch.equal(env.trajectory_manager.steps, torch.tensor([261, 14]))
