# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate manager parity (observations/rewards/terminations) for dance_102.

This compares:
- Isaac-Imitation-G1-Dance102-Compare-v0 (iltools LAFAN1 loader)
- Unitree-G1-29dof-Mimic-Dance-102 (unitree_rl_lab motion command)

The script reports:
1. Schema parity:
   - observation groups/term order/term dims
   - reward terms/weights
   - termination terms/time_out flags
2. Runtime parity:
   - policy/critic observation tensors (env_0)
   - total reward and per-term reward values (env_0)
   - terminated/truncated flags and per-term termination flags (env_0)
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


def _append_workspace_sources() -> None:
    this_file = Path(__file__).resolve()
    workspace_root = this_file.parents[2]
    candidate_paths = [
        workspace_root / "IsaacLab" / "source" / "isaaclab",
        workspace_root / "IsaacLab" / "source" / "isaaclab_tasks",
        workspace_root / "IsaacLab-Imitation" / "source" / "isaaclab_imitation",
        workspace_root / "unitree_rl_lab" / "source" / "unitree_rl_lab",
        workspace_root / "ImitationLearningTools",
    ]
    for candidate in candidate_paths:
        if candidate.is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.append(candidate_str)


_append_workspace_sources()

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Compare dance_102 manager parity with unitree_rl_lab.")
parser.add_argument(
    "--task_ours",
    type=str,
    default="Isaac-Imitation-G1-Dance102-Compare-v0",
    help="IsaacLab-Imitation dance comparison task id.",
)
parser.add_argument(
    "--task_unitree",
    type=str,
    default="Unitree-G1-29dof-Mimic-Dance-102",
    help="Original unitree_rl_lab dance task id.",
)
parser.add_argument("--motion_path", type=str, default=None, help="Optional motion npz to force in both envs.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--steps", type=int, default=200, help="Runtime parity rollout length.")
parser.add_argument(
    "--runtime_mode",
    type=str,
    choices=("manager_eval", "env_step"),
    default="manager_eval",
    help=(
        "Runtime parity mode. "
        "`manager_eval` compares managers on replayed matched states (recommended for function parity). "
        "`env_step` compares full rollout behavior."
    ),
)
parser.add_argument("--seed", type=int, default=42, help="Seed for both env configs.")
parser.add_argument(
    "--allow_torch_compile",
    action="store_true",
    default=False,
    help="Allow torch.compile kernels (disabled by default for parity runs).",
)
parser.add_argument("--max_runtime_s", type=float, default=900.0, help="Hard timeout in seconds. Set <= 0 to disable.")
parser.add_argument("--progress_every", type=int, default=20, help="Print progress every N steps. Set <= 0 to disable.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--obs_tol",
    type=float,
    default=1.0e-6,
    help="Observation max-abs threshold for runtime parity.",
)
parser.add_argument(
    "--reward_tol",
    type=float,
    default=1.0e-6,
    help="Reward max-abs threshold for runtime parity.",
)
parser.add_argument(
    "--topk_mismatch",
    type=int,
    default=5,
    help="Number of worst mismatch samples to include in runtime diagnostics.",
)
parser.add_argument(
    "--report_path",
    type=str,
    default=None,
    help="Optional JSON output path.",
)
parser.add_argument(
    "--collector_role",
    type=str,
    choices=("both", "ours", "unitree"),
    default="both",
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--stream_path",
    type=str,
    default=None,
    help=argparse.SUPPRESS,
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.allow_torch_compile:
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import gymnasium as gym
import numpy as np
import torch


OBS_NAME_ALIASES = {
    "reference_motion": "motion_command",
    "command": "motion_command",
    "motion_command": "motion_command",
    "reference_anchor_pos_b": "motion_anchor_pos_b",
    "reference_anchor_ori_b": "motion_anchor_ori_b",
}


def _canonical_obs_term_name(name: str) -> str:
    return OBS_NAME_ALIASES.get(name, name)


def _canonical_term_name(name: str) -> str:
    # Reward/termination names are already aligned in compare envs.
    return name


def _resolve_default_motion_path() -> Path:
    this_file = Path(__file__).resolve()
    workspace_root = this_file.parents[2]
    default_npz = (
        workspace_root
        / "unitree_rl_lab"
        / "source"
        / "unitree_rl_lab"
        / "unitree_rl_lab"
        / "tasks"
        / "mimic"
        / "robots"
        / "g1_29dof"
        / "dance_102"
        / "G1_Take_102.bvh_60hz.npz"
    )
    if args_cli.motion_path is not None:
        return Path(args_cli.motion_path).expanduser().resolve()
    return default_npz


def _safe_expand_dim_shape(shape_like: Any) -> list[int] | list[list[int]]:
    if isinstance(shape_like, tuple):
        return [int(x) for x in shape_like]
    if isinstance(shape_like, list):
        if len(shape_like) == 0:
            return []
        if isinstance(shape_like[0], tuple):
            return [[int(x) for x in item] for item in shape_like]
        return [int(x) for x in shape_like]
    return []


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0
        return float(value.reshape(-1)[0])
    return float(value)


def _to_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return False
        return bool(value.reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
        return bool(value.reshape(-1)[0])
    return bool(value)


def _clone_to_cpu(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().to("cpu").clone()


def _zero_action(env) -> torch.Tensor:
    action = torch.as_tensor(env.action_space.sample(), device=env.unwrapped.device)
    return torch.zeros_like(action)


def _check_deadline(deadline_s: float | None, stage: str, step: int, total_steps: int) -> None:
    if deadline_s is None:
        return
    if time.monotonic() > deadline_s:
        raise TimeoutError(f"Timed out during {stage} at step {step}/{total_steps}.")


def _start_timeout_watchdog(max_runtime_s: float) -> threading.Event | None:
    if max_runtime_s <= 0:
        return None

    stop_event = threading.Event()

    def _watchdog():
        if stop_event.wait(timeout=max_runtime_s):
            return
        print(
            f"[ERROR] Evaluation exceeded max_runtime_s={max_runtime_s:.1f}s. Force exiting.",
            file=sys.stderr,
            flush=True,
        )
        os._exit(124)

    thread = threading.Thread(target=_watchdog, name="manager-parity-timeout-watchdog", daemon=True)
    thread.start()
    return stop_event


def _disable_observation_noise(env_cfg) -> None:
    observations_cfg = getattr(env_cfg, "observations", None)
    if observations_cfg is None:
        return
    for group_name, group_cfg in observations_cfg.__dict__.items():
        if group_name.startswith("_") or group_cfg is None:
            continue
        if hasattr(group_cfg, "enable_corruption"):
            group_cfg.enable_corruption = False


def _disable_random_events(env_cfg) -> None:
    events_cfg = getattr(env_cfg, "events", None)
    if events_cfg is None:
        return
    for attr in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
        if hasattr(events_cfg, attr):
            setattr(events_cfg, attr, None)


def _configure_our_env_cfg(env_cfg, motion_path: Path) -> None:
    source_entry = {
        "name": "dance_102",
        "path": str(motion_path),
        "input_fps": 60,
    }
    env_cfg.loader_type = "lafan1_csv"
    env_cfg.loader_kwargs = {
        "dataset_name": "lafan1",
        "dataset": {"trajectories": {"lafan1_csv": [source_entry]}},
        "sim": {"dt": float(env_cfg.sim.dt)},
        "decimation": int(env_cfg.decimation),
        "joint_names": list(getattr(env_cfg, "reference_joint_names", []) or []),
    }
    try:
        with np.load(motion_path) as npz_data:
            if "fps" in npz_data.files:
                env_cfg.loader_kwargs["control_freq"] = float(np.asarray(npz_data["fps"]).reshape(-1)[0])
            else:
                env_cfg.loader_kwargs["control_freq"] = 1.0 / (float(env_cfg.sim.dt) * float(env_cfg.decimation))
    except Exception:
        env_cfg.loader_kwargs["control_freq"] = 1.0 / (float(env_cfg.sim.dt) * float(env_cfg.decimation))

    env_cfg.dataset_path = f"/tmp/iltools_eval_dance102_manager_parity_{motion_path.stem}"
    env_cfg.refresh_zarr_dataset = True
    env_cfg.motions = ["dance_102"]
    env_cfg.trajectories = ["trajectory_0"]
    env_cfg.reset_schedule = "sequential"
    env_cfg.reference_start_frame = 0
    env_cfg.replay_reference = False
    env_cfg.replay_only = False
    if hasattr(env_cfg, "enable_visualizers"):
        env_cfg.enable_visualizers = False
    env_cfg.visualize_reference_arrows = False
    env_cfg.visualize_reference_velocity = False

    _disable_observation_noise(env_cfg)
    _disable_random_events(env_cfg)


def _configure_unitree_env_cfg(env_cfg, motion_path: Path) -> None:
    if not hasattr(env_cfg, "commands") or not hasattr(env_cfg.commands, "motion"):
        raise RuntimeError(
            "Selected unitree env cfg does not expose `commands.motion`. "
            "Pass a Unitree mimic dance task via --task_unitree."
        )
    env_cfg.commands.motion.motion_file = str(motion_path)
    env_cfg.commands.motion.pose_range = {k: (0.0, 0.0) for k in ["x", "y", "z", "roll", "pitch", "yaw"]}
    env_cfg.commands.motion.velocity_range = {k: (0.0, 0.0) for k in ["x", "y", "z", "roll", "pitch", "yaw"]}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    env_cfg.commands.motion.adaptive_uniform_ratio = 1.0
    env_cfg.commands.motion.adaptive_alpha = 0.0
    if hasattr(env_cfg.commands.motion, "debug_vis"):
        env_cfg.commands.motion.debug_vis = False
    if hasattr(env_cfg.scene, "contact_forces") and hasattr(env_cfg.scene.contact_forces, "debug_vis"):
        env_cfg.scene.contact_forces.debug_vis = False

    _disable_observation_noise(env_cfg)
    _disable_random_events(env_cfg)


def _resolve_registered_task_id(task_name: str, kind: str) -> str:
    requested = task_name.split(":")[-1]
    try:
        gym.spec(requested)
        return requested
    except Exception:
        pass

    all_ids = sorted(str(task_id) for task_id in gym.registry.keys())
    all_ids_lower = [task_id.lower() for task_id in all_ids]

    if kind == "unitree":
        requested_tokens = [tok for tok in re.split(r"[^a-z0-9]+", requested.lower()) if tok]
        unitree_candidates = [task_id for task_id in all_ids if "unitree" in task_id.lower()]
        if len(unitree_candidates) == 0:
            raise RuntimeError(
                f"Requested unitree task '{requested}' is not registered and no Unitree tasks were found."
            )

        def _score(candidate: str) -> tuple[int, int, int]:
            candidate_lower = candidate.lower()
            overlap = sum(token in candidate_lower for token in requested_tokens)
            return overlap, -len(candidate), -all_ids.index(candidate)

        resolved = sorted(unitree_candidates, key=_score, reverse=True)[0]
        print(
            f"[INFO] Requested unitree task '{requested}' was not registered. "
            f"Using closest Unitree match: '{resolved}'."
        )
        return resolved

    preferred_keywords = ("isaac", "imitation", "g1", "dance", "compare")
    preferred_matches = [task_id for task_id in all_ids if all(keyword in task_id.lower() for keyword in preferred_keywords)]
    if preferred_matches:
        print(
            f"[INFO] Requested {kind} task '{requested}' was not registered. "
            f"Using closest registered match: '{preferred_matches[0]}'."
        )
        return preferred_matches[0]

    close_matches = difflib.get_close_matches(requested.lower(), all_ids_lower, n=10, cutoff=0.35)
    close_candidates = [all_ids[all_ids_lower.index(match)] for match in close_matches]
    raise RuntimeError(f"Task '{requested}' is not registered. Closest registered tasks: {close_candidates}")


def _extract_obs_schema(env) -> dict[str, Any]:
    obs_manager = env.unwrapped.observation_manager
    schema: dict[str, Any] = {}
    for group_name, term_names in obs_manager.active_terms.items():
        schema[group_name] = {
            "term_names": list(term_names),
            "term_names_canonical": [_canonical_obs_term_name(name) for name in term_names],
            "term_dims": [_safe_expand_dim_shape(shape) for shape in obs_manager.group_obs_term_dim[group_name]],
            "group_dim": _safe_expand_dim_shape(obs_manager.group_obs_dim[group_name]),
        }
    return schema


def _extract_reward_schema(env) -> dict[str, Any]:
    reward_manager = env.unwrapped.reward_manager
    terms: list[dict[str, Any]] = []
    for name in reward_manager.active_terms:
        cfg = reward_manager.get_term_cfg(name)
        func = cfg.func
        if hasattr(func, "__name__"):
            func_name = str(func.__name__)
        else:
            func_name = str(func.__class__.__name__)
        terms.append(
            {
                "name": name,
                "name_canonical": _canonical_term_name(name),
                "weight": float(cfg.weight),
                "func_name": func_name,
            }
        )
    return {"terms": terms}


def _extract_termination_schema(env) -> dict[str, Any]:
    termination_manager = env.unwrapped.termination_manager
    terms: list[dict[str, Any]] = []
    for name in termination_manager.active_terms:
        cfg = termination_manager.get_term_cfg(name)
        terms.append(
            {
                "name": name,
                "name_canonical": _canonical_term_name(name),
                "time_out": bool(cfg.time_out),
            }
        )
    return {"terms": terms}


def _extract_group_obs(obs: Any, group_name: str) -> torch.Tensor | None:
    if isinstance(obs, dict):
        value = obs.get(group_name)
        if isinstance(value, torch.Tensor):
            return value
        return None
    if isinstance(obs, torch.Tensor):
        return obs if group_name == "policy" else None
    return None


def _new_runtime_trace() -> dict[str, Any]:
    return {
        "policy_obs": [],
        "critic_obs": [],
        "reward_total": [],
        "terminated": [],
        "truncated": [],
        "reward_terms": {},
        "termination_terms": {},
    }


def _append_runtime_step(trace: dict[str, Any], env, obs: Any, reward: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor):
    policy_obs = _extract_group_obs(obs, "policy")
    critic_obs = _extract_group_obs(obs, "critic")
    if policy_obs is not None:
        trace["policy_obs"].append(_clone_to_cpu(policy_obs[0]))
    if critic_obs is not None:
        trace["critic_obs"].append(_clone_to_cpu(critic_obs[0]))
    trace["reward_total"].append(_to_float(reward[0]))
    trace["terminated"].append(_to_bool(terminated[0]))
    trace["truncated"].append(_to_bool(truncated[0]))

    reward_terms = env.unwrapped.reward_manager.get_active_iterable_terms(0)
    for term_name, value in reward_terms:
        canonical = _canonical_term_name(term_name)
        trace["reward_terms"].setdefault(canonical, []).append(float(value[0]))

    termination_terms = env.unwrapped.termination_manager.get_active_iterable_terms(0)
    for term_name, value in termination_terms:
        canonical = _canonical_term_name(term_name)
        trace["termination_terms"].setdefault(canonical, []).append(bool(value[0] >= 0.5))


def _collect_runtime_trace_env_step(env, steps: int, deadline_s: float | None, progress_every: int) -> dict[str, Any]:
    obs, _ = env.reset()
    action = _zero_action(env)
    trace = _new_runtime_trace()

    for step_idx in range(steps):
        _check_deadline(deadline_s, "runtime trace collection", step_idx, steps)
        obs, reward, terminated, truncated, _ = env.step(action)
        _append_runtime_step(trace, env, obs, reward, terminated, truncated)

        if progress_every > 0 and ((step_idx + 1) % progress_every == 0 or step_idx + 1 == steps):
            print(f"[INFO] Collected runtime trace: {step_idx + 1}/{steps} steps", flush=True)

    return trace


def _write_unitree_robot_state_from_command(env_unwrapped, command) -> None:
    robot = env_unwrapped.scene["robot"]
    root_pos = command.body_pos_w[:, 0]
    root_quat = command.body_quat_w[:, 0]
    root_lin_vel = command.body_lin_vel_w[:, 0]
    root_ang_vel = command.body_ang_vel_w[:, 0]
    root_state = torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1)
    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(command.joint_pos, command.joint_vel)
    robot.write_data_to_sim()
    env_unwrapped.scene.update(dt=0.0)
    robot.update(dt=0.0)


def _write_ours_robot_state_from_reference(env_unwrapped, reference) -> None:
    from isaaclab.utils import math as math_utils

    robot = env_unwrapped.scene["robot"]
    root_pos_w, root_quat_w_opt = env_unwrapped._transform_reference_pose_to_world(
        reference["root_pos"], reference["root_quat"]
    )
    if root_quat_w_opt is None:
        raise RuntimeError("Failed to transform reference root quaternion.")
    root_quat_w = root_quat_w_opt

    align_quat, _ = env_unwrapped._get_reference_alignment_transform()
    root_lin_vel_w = math_utils.quat_apply(align_quat, reference["root_lin_vel"])
    root_ang_vel_w = math_utils.quat_apply(align_quat, reference["root_ang_vel"])
    root_state = torch.cat([root_pos_w, root_quat_w, root_lin_vel_w, root_ang_vel_w], dim=-1)

    defaults_pos = robot.data.default_joint_pos
    defaults_vel = robot.data.default_joint_vel
    joint_pos = torch.where(torch.isnan(reference["joint_pos"]), defaults_pos, reference["joint_pos"])
    joint_vel = torch.where(torch.isnan(reference["joint_vel"]), defaults_vel, reference["joint_vel"])

    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.write_data_to_sim()
    env_unwrapped.scene.update(dt=0.0)
    robot.update(dt=0.0)


def _update_unitree_command_relative_buffers(command) -> None:
    from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat

    anchor_pos_w_repeat = command.anchor_pos_w[:, None, :].repeat(1, len(command.cfg.body_names), 1)
    anchor_quat_w_repeat = command.anchor_quat_w[:, None, :].repeat(1, len(command.cfg.body_names), 1)
    robot_anchor_pos_w_repeat = command.robot_anchor_pos_w[:, None, :].repeat(1, len(command.cfg.body_names), 1)
    robot_anchor_quat_w_repeat = command.robot_anchor_quat_w[:, None, :].repeat(1, len(command.cfg.body_names), 1)

    delta_pos_w = robot_anchor_pos_w_repeat.clone()
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

    command.body_quat_relative_w = quat_mul(delta_ori_w, command.body_quat_w)
    command.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, command.body_pos_w - anchor_pos_w_repeat)


def _collect_runtime_trace_manager_eval_ours(env, steps: int, deadline_s: float | None, progress_every: int) -> dict[str, Any]:
    env.reset()
    action = _zero_action(env)
    env_u = env.unwrapped
    trace = _new_runtime_trace()

    for step_idx in range(steps):
        _check_deadline(deadline_s, "manager_eval ours trace collection", step_idx, steps)
        env_u.action_manager.process_action(action.to(env_u.device))

        reference_for_step = env_u.trajectory_manager.sample(advance=True)
        env_u.current_reference = reference_for_step
        _write_ours_robot_state_from_reference(env_u, reference_for_step)

        env_u.episode_length_buf += 1
        env_u.common_step_counter += 1
        env_u.reset_buf = env_u.termination_manager.compute()
        env_u.reset_terminated = env_u.termination_manager.terminated
        env_u.reset_time_outs = env_u.termination_manager.time_outs
        reward = env_u.reward_manager.compute(dt=env_u.step_dt)

        # Keep manager ordering consistent with RL env.
        env_u.command_manager.compute(dt=env_u.step_dt)
        if "interval" in env_u.event_manager.available_modes:
            env_u.event_manager.apply(mode="interval", dt=env_u.step_dt)

        # Keep observations aligned on the same frame used for reward/termination (frame t).
        obs = env_u.observation_manager.compute(update_history=True)

        _append_runtime_step(trace, env, obs, reward, env_u.reset_terminated, env_u.reset_time_outs)

        if progress_every > 0 and ((step_idx + 1) % progress_every == 0 or step_idx + 1 == steps):
            print(f"[INFO] Collected manager_eval ours: {step_idx + 1}/{steps} steps", flush=True)

    return trace


def _collect_runtime_trace_manager_eval_unitree(
    env, steps: int, deadline_s: float | None, progress_every: int
) -> dict[str, Any]:
    env.reset()
    action = _zero_action(env)
    env_u = env.unwrapped
    command = env_u.command_manager.get_term("motion")
    trace = _new_runtime_trace()

    max_step = max(int(command.motion.time_step_total) - 1, 0)
    for step_idx in range(steps):
        _check_deadline(deadline_s, "manager_eval unitree trace collection", step_idx, steps)
        env_u.action_manager.process_action(action.to(env_u.device))

        command.time_steps[:] = min(step_idx, max_step)
        _write_unitree_robot_state_from_command(env_u, command)
        _update_unitree_command_relative_buffers(command)

        env_u.episode_length_buf += 1
        env_u.common_step_counter += 1
        env_u.reset_buf = env_u.termination_manager.compute()
        env_u.reset_terminated = env_u.termination_manager.terminated
        env_u.reset_time_outs = env_u.termination_manager.time_outs
        reward = env_u.reward_manager.compute(dt=env_u.step_dt)

        if "interval" in env_u.event_manager.available_modes:
            env_u.event_manager.apply(mode="interval", dt=env_u.step_dt)
        obs = env_u.observation_manager.compute(update_history=True)

        _append_runtime_step(trace, env, obs, reward, env_u.reset_terminated, env_u.reset_time_outs)

        if progress_every > 0 and ((step_idx + 1) % progress_every == 0 or step_idx + 1 == steps):
            print(f"[INFO] Collected manager_eval unitree: {step_idx + 1}/{steps} steps", flush=True)

    return trace


def _collect_runtime_trace(
    env, role: str, steps: int, deadline_s: float | None, progress_every: int, runtime_mode: str
) -> dict[str, Any]:
    if runtime_mode == "env_step":
        return _collect_runtime_trace_env_step(env, steps=steps, deadline_s=deadline_s, progress_every=progress_every)
    if runtime_mode == "manager_eval":
        if role == "ours":
            return _collect_runtime_trace_manager_eval_ours(
                env, steps=steps, deadline_s=deadline_s, progress_every=progress_every
            )
        return _collect_runtime_trace_manager_eval_unitree(
            env, steps=steps, deadline_s=deadline_s, progress_every=progress_every
        )
    raise ValueError(f"Unsupported runtime_mode: {runtime_mode}")


def _collect_payload_for_role(role: str, stream_path: Path | None = None) -> dict[str, Any]:
    if role not in ("ours", "unitree"):
        raise ValueError(f"Unsupported collector role: {role}")

    print(f"[INFO] Collector role '{role}' starting.", flush=True)
    deadline_s = None if args_cli.max_runtime_s <= 0 else (time.monotonic() + float(args_cli.max_runtime_s))
    motion_path = _resolve_default_motion_path()
    if not motion_path.is_file():
        raise FileNotFoundError(f"Dance motion npz not found at: {motion_path}")

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    env = None
    try:
        import isaaclab_imitation  # noqa: F401
        import isaaclab_tasks  # noqa: F401
        import unitree_rl_lab.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        resolved_task_ours = _resolve_registered_task_id(args_cli.task_ours, kind="ours")
        resolved_task_unitree = _resolve_registered_task_id(args_cli.task_unitree, kind="unitree")
        print(
            f"[INFO] Collector role '{role}' resolved tasks: ours={resolved_task_ours}, unitree={resolved_task_unitree}.",
            flush=True,
        )

        if role == "ours":
            cfg = parse_env_cfg(
                resolved_task_ours,
                device=args_cli.device,
                num_envs=args_cli.num_envs,
                use_fabric=not args_cli.disable_fabric,
            )
            cfg.seed = args_cli.seed
            _configure_our_env_cfg(cfg, motion_path)
            _check_deadline(deadline_s, "before creating our env", 0, max(args_cli.steps, 1))
            env = gym.make(resolved_task_ours, cfg=cfg)
            requested_task = args_cli.task_ours
            resolved_task = resolved_task_ours
        else:
            cfg = parse_env_cfg(
                resolved_task_unitree,
                device=args_cli.device,
                num_envs=args_cli.num_envs,
                use_fabric=not args_cli.disable_fabric,
            )
            cfg.seed = args_cli.seed
            _configure_unitree_env_cfg(cfg, motion_path)
            _check_deadline(deadline_s, "before creating unitree env", 0, max(args_cli.steps, 1))
            env = gym.make(resolved_task_unitree, cfg=cfg)
            requested_task = args_cli.task_unitree
            resolved_task = resolved_task_unitree

        print(f"[INFO] Collector role '{role}' created env.", flush=True)
        payload = {
            "role": role,
            "requested_task": requested_task,
            "resolved_task": resolved_task,
            "motion_path": str(motion_path),
            "num_envs": int(args_cli.num_envs),
            "steps": int(args_cli.steps),
            "runtime_mode": str(args_cli.runtime_mode),
            "observation_schema": _extract_obs_schema(env),
            "reward_schema": _extract_reward_schema(env),
            "termination_schema": _extract_termination_schema(env),
            "runtime_trace": _collect_runtime_trace(
                env,
                role=role,
                steps=args_cli.steps,
                deadline_s=deadline_s,
                progress_every=args_cli.progress_every,
                runtime_mode=str(args_cli.runtime_mode),
            ),
        }
        print(f"[INFO] Collector role '{role}' finished collection.", flush=True)

        if stream_path is not None:
            stream_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, stream_path)
            print(f"[INFO] Collector role '{role}' wrote stream file: {stream_path}", flush=True)
        return payload
    finally:
        if env is not None:
            print(f"[INFO] Collector role '{role}' closing env.", flush=True)
            env.close()
        print(f"[INFO] Collector role '{role}' closing simulation app.", flush=True)
        simulation_app.close()


def _run_collector_subprocess(role: str, stream_path: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--collector_role",
        role,
        "--stream_path",
        str(stream_path),
    ]
    print(f"[INFO] Launching collector: role={role}", flush=True)
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Collector subprocess failed for role='{role}' with exit code {completed.returncode}.")
    if not stream_path.is_file():
        raise RuntimeError(f"Collector subprocess for role='{role}' did not write stream file: {stream_path}")
    return torch.load(stream_path, map_location="cpu")


def _summarize(values: list[float]) -> dict[str, float]:
    if len(values) == 0:
        return {"mean": 0.0, "max": 0.0}
    arr = torch.tensor(values, dtype=torch.float32)
    return {"mean": float(arr.mean().item()), "max": float(arr.max().item())}


def _compare_list_exact(lhs: list[Any], rhs: list[Any]) -> dict[str, Any]:
    return {
        "match": bool(lhs == rhs),
        "lhs": lhs,
        "rhs": rhs,
    }


def _compare_observation_schema(ours_schema: dict[str, Any], unitree_schema: dict[str, Any]) -> dict[str, Any]:
    ours_groups = sorted(ours_schema.keys())
    unitree_groups = sorted(unitree_schema.keys())
    result: dict[str, Any] = {
        "group_names": _compare_list_exact(ours_groups, unitree_groups),
        "groups": {},
    }
    schema_pass = result["group_names"]["match"]

    common_groups = sorted(set(ours_groups).intersection(unitree_groups))
    for group_name in common_groups:
        ours_group = ours_schema[group_name]
        unitree_group = unitree_schema[group_name]
        group_result = {
            "term_names_canonical": _compare_list_exact(
                list(ours_group["term_names_canonical"]),
                list(unitree_group["term_names_canonical"]),
            ),
            "term_dims": _compare_list_exact(
                list(ours_group["term_dims"]),
                list(unitree_group["term_dims"]),
            ),
            "group_dim": _compare_list_exact(
                list(ours_group["group_dim"]) if isinstance(ours_group["group_dim"], list) else ours_group["group_dim"],
                list(unitree_group["group_dim"])
                if isinstance(unitree_group["group_dim"], list)
                else unitree_group["group_dim"],
            ),
        }
        group_pass = (
            group_result["term_names_canonical"]["match"]
            and group_result["term_dims"]["match"]
            and group_result["group_dim"]["match"]
        )
        group_result["pass"] = bool(group_pass)
        schema_pass = schema_pass and group_pass
        result["groups"][group_name] = group_result

    result["pass"] = bool(schema_pass)
    return result


def _compare_reward_schema(ours_schema: dict[str, Any], unitree_schema: dict[str, Any], weight_tol: float) -> dict[str, Any]:
    ours_terms = ours_schema["terms"]
    unitree_terms = unitree_schema["terms"]
    ours_names = [term["name_canonical"] for term in ours_terms]
    unitree_names = [term["name_canonical"] for term in unitree_terms]
    terms_match = _compare_list_exact(ours_names, unitree_names)

    per_term: dict[str, Any] = {}
    weights_pass = True
    for ours_term, unitree_term in zip(ours_terms, unitree_terms):
        name = ours_term["name_canonical"]
        weight_diff = abs(float(ours_term["weight"]) - float(unitree_term["weight"]))
        term_pass = weight_diff <= weight_tol
        weights_pass = weights_pass and term_pass
        per_term[name] = {
            "weight_ours": float(ours_term["weight"]),
            "weight_unitree": float(unitree_term["weight"]),
            "weight_abs_diff": float(weight_diff),
            "func_name_ours": ours_term["func_name"],
            "func_name_unitree": unitree_term["func_name"],
            "pass": bool(term_pass),
        }

    result = {
        "term_names": terms_match,
        "per_term": per_term,
        "pass": bool(terms_match["match"] and weights_pass),
    }
    return result


def _compare_termination_schema(ours_schema: dict[str, Any], unitree_schema: dict[str, Any]) -> dict[str, Any]:
    ours_terms = ours_schema["terms"]
    unitree_terms = unitree_schema["terms"]
    ours_names = [term["name_canonical"] for term in ours_terms]
    unitree_names = [term["name_canonical"] for term in unitree_terms]
    names_match = _compare_list_exact(ours_names, unitree_names)

    per_term: dict[str, Any] = {}
    flags_pass = True
    for ours_term, unitree_term in zip(ours_terms, unitree_terms):
        name = ours_term["name_canonical"]
        same_flag = bool(ours_term["time_out"] == unitree_term["time_out"])
        flags_pass = flags_pass and same_flag
        per_term[name] = {
            "time_out_ours": bool(ours_term["time_out"]),
            "time_out_unitree": bool(unitree_term["time_out"]),
            "pass": same_flag,
        }

    result = {
        "term_names": names_match,
        "per_term": per_term,
        "pass": bool(names_match["match"] and flags_pass),
    }
    return result


def _compare_tensor_series(lhs: list[torch.Tensor], rhs: list[torch.Tensor]) -> dict[str, Any]:
    return _compare_tensor_series_with_shift(lhs, rhs, shift=0, topk=args_cli.topk_mismatch)


def _pairing_bounds(lhs_len: int, rhs_len: int, shift: int) -> tuple[int, int, int]:
    lhs_start = max(0, -shift)
    rhs_start = max(0, shift)
    compared_steps = min(lhs_len - lhs_start, rhs_len - rhs_start)
    return lhs_start, rhs_start, max(0, compared_steps)


def _compare_tensor_series_with_shift(
    lhs: list[torch.Tensor], rhs: list[torch.Tensor], shift: int, topk: int
) -> dict[str, Any]:
    lhs_start, rhs_start, compared_steps = _pairing_bounds(len(lhs), len(rhs), shift)
    if compared_steps == 0:
        return {
            "compared_steps": 0,
            "shift": int(shift),
            "shape_match": False,
            "errors": {"mean": 0.0, "max": float("inf")},
            "pass": False,
        }

    errors: list[float] = []
    shape_match = True
    worst_step = {"lhs_step": -1, "rhs_step": -1, "abs_err": 0.0}
    top_mismatches: list[dict[str, Any]] = []
    for step_idx in range(compared_steps):
        lhs_idx = lhs_start + step_idx
        rhs_idx = rhs_start + step_idx
        lhs_tensor = lhs[lhs_idx]
        rhs_tensor = rhs[rhs_idx]
        if tuple(lhs_tensor.shape) != tuple(rhs_tensor.shape):
            shape_match = False
            break
        diff = torch.abs(lhs_tensor - rhs_tensor)
        step_max = float(torch.max(diff).item())
        errors.append(step_max)
        if step_max >= worst_step["abs_err"]:
            worst_step = {"lhs_step": int(lhs_idx), "rhs_step": int(rhs_idx), "abs_err": step_max}
        if topk > 0:
            flat_idx = int(torch.argmax(diff).item())
            unravel_idx = [int(x) for x in np.unravel_index(flat_idx, diff.shape)]
            lhs_value = float(lhs_tensor.reshape(-1)[flat_idx].item())
            rhs_value = float(rhs_tensor.reshape(-1)[flat_idx].item())
            top_mismatches.append(
                {
                    "lhs_step": int(lhs_idx),
                    "rhs_step": int(rhs_idx),
                    "index": unravel_idx,
                    "abs_err": step_max,
                    "lhs_value": lhs_value,
                    "rhs_value": rhs_value,
                }
            )

    if not shape_match:
        return {
            "compared_steps": int(compared_steps),
            "shift": int(shift),
            "shape_match": False,
            "lhs_shape": list(lhs[lhs_start].shape),
            "rhs_shape": list(rhs[rhs_start].shape),
            "errors": {"mean": 0.0, "max": float("inf")},
            "pass": False,
        }

    if topk > 0:
        top_mismatches = sorted(top_mismatches, key=lambda item: item["abs_err"], reverse=True)[:topk]

    return {
        "compared_steps": int(compared_steps),
        "shift": int(shift),
        "shape_match": True,
        "errors": _summarize(errors),
        "worst_step": worst_step,
        "top_mismatches": top_mismatches,
        "pass": True,
    }


def _compare_scalar_series(lhs: list[float], rhs: list[float]) -> dict[str, Any]:
    return _compare_scalar_series_with_shift(lhs, rhs, shift=0, topk=args_cli.topk_mismatch)


def _compare_scalar_series_with_shift(lhs: list[float], rhs: list[float], shift: int, topk: int) -> dict[str, Any]:
    lhs_start, rhs_start, compared_steps = _pairing_bounds(len(lhs), len(rhs), shift)
    errors: list[float] = []
    mismatch_samples: list[dict[str, Any]] = []
    for step_idx in range(compared_steps):
        lhs_idx = lhs_start + step_idx
        rhs_idx = rhs_start + step_idx
        lhs_value = float(lhs[lhs_idx])
        rhs_value = float(rhs[rhs_idx])
        error = abs(lhs_value - rhs_value)
        errors.append(error)
        if topk > 0:
            mismatch_samples.append(
                {
                    "lhs_step": int(lhs_idx),
                    "rhs_step": int(rhs_idx),
                    "lhs_value": lhs_value,
                    "rhs_value": rhs_value,
                    "abs_err": float(error),
                }
            )
    if topk > 0:
        mismatch_samples = sorted(mismatch_samples, key=lambda item: item["abs_err"], reverse=True)[:topk]
    return {
        "compared_steps": int(compared_steps),
        "shift": int(shift),
        "errors": _summarize(errors),
        "top_mismatches": mismatch_samples,
    }


def _compare_bool_series(lhs: list[bool], rhs: list[bool]) -> dict[str, Any]:
    return _compare_bool_series_with_shift(lhs, rhs, shift=0, topk=args_cli.topk_mismatch)


def _compare_bool_series_with_shift(lhs: list[bool], rhs: list[bool], shift: int, topk: int) -> dict[str, Any]:
    lhs_start, rhs_start, compared_steps = _pairing_bounds(len(lhs), len(rhs), shift)
    mismatch_indices: list[dict[str, Any]] = []
    mismatch_count = 0
    for step_idx in range(compared_steps):
        lhs_idx = lhs_start + step_idx
        rhs_idx = rhs_start + step_idx
        lhs_value = bool(lhs[lhs_idx])
        rhs_value = bool(rhs[rhs_idx])
        if lhs_value != rhs_value:
            mismatch_count += 1
            if topk <= 0 or len(mismatch_indices) < topk:
                mismatch_indices.append(
                    {
                        "lhs_step": int(lhs_idx),
                        "rhs_step": int(rhs_idx),
                        "lhs_value": lhs_value,
                        "rhs_value": rhs_value,
                    }
                )
    return {
        "compared_steps": int(compared_steps),
        "shift": int(shift),
        "mismatch_count": int(mismatch_count),
        "mismatch_samples": mismatch_indices,
    }


def _term_flat_dim(term_dim: Any) -> int | None:
    if isinstance(term_dim, list):
        if len(term_dim) == 1 and isinstance(term_dim[0], int):
            return int(term_dim[0])
        if len(term_dim) > 0 and all(isinstance(x, int) for x in term_dim):
            prod = 1
            for item in term_dim:
                prod *= int(item)
            return int(prod)
    if isinstance(term_dim, int):
        return int(term_dim)
    return None


def _obs_term_slices(group_schema: dict[str, Any]) -> list[dict[str, Any]] | None:
    term_names = list(group_schema.get("term_names_canonical", []))
    term_dims = list(group_schema.get("term_dims", []))
    if len(term_names) != len(term_dims):
        return None
    cursor = 0
    slices: list[dict[str, Any]] = []
    for name, dim_spec in zip(term_names, term_dims):
        flat_dim = _term_flat_dim(dim_spec)
        if flat_dim is None:
            return None
        slices.append(
            {
                "name": str(name),
                "start": int(cursor),
                "end": int(cursor + flat_dim),
                "flat_dim": int(flat_dim),
            }
        )
        cursor += flat_dim
    return slices


def _compare_obs_group_term_breakdown(
    lhs: list[torch.Tensor],
    rhs: list[torch.Tensor],
    ours_group_schema: dict[str, Any] | None,
    unitree_group_schema: dict[str, Any] | None,
    shift: int,
) -> dict[str, Any]:
    if ours_group_schema is None or unitree_group_schema is None:
        return {"available": False, "reason": "missing_group_schema"}

    ours_names = list(ours_group_schema.get("term_names_canonical", []))
    unitree_names = list(unitree_group_schema.get("term_names_canonical", []))
    if ours_names != unitree_names:
        return {"available": False, "reason": "term_name_mismatch", "ours": ours_names, "unitree": unitree_names}

    ours_dims = list(ours_group_schema.get("term_dims", []))
    unitree_dims = list(unitree_group_schema.get("term_dims", []))
    if ours_dims != unitree_dims:
        return {"available": False, "reason": "term_dim_mismatch", "ours": ours_dims, "unitree": unitree_dims}

    slices = _obs_term_slices(ours_group_schema)
    if slices is None:
        return {"available": False, "reason": "unsupported_term_dim"}

    lhs_start, rhs_start, compared_steps = _pairing_bounds(len(lhs), len(rhs), shift)
    if compared_steps <= 0:
        return {"available": False, "reason": "no_overlap"}

    per_term: dict[str, Any] = {}
    for term_slice in slices:
        term_name = term_slice["name"]
        term_start = int(term_slice["start"])
        term_end = int(term_slice["end"])
        errors: list[float] = []
        worst = {"lhs_step": -1, "rhs_step": -1, "abs_err": 0.0}
        for step_idx in range(compared_steps):
            lhs_idx = lhs_start + step_idx
            rhs_idx = rhs_start + step_idx
            lhs_tensor = lhs[lhs_idx]
            rhs_tensor = rhs[rhs_idx]
            if tuple(lhs_tensor.shape) != tuple(rhs_tensor.shape):
                return {
                    "available": False,
                    "reason": "shape_mismatch",
                    "lhs_shape": list(lhs_tensor.shape),
                    "rhs_shape": list(rhs_tensor.shape),
                    "lhs_step": int(lhs_idx),
                    "rhs_step": int(rhs_idx),
                }
            if term_end > int(lhs_tensor.numel()) or term_end > int(rhs_tensor.numel()):
                return {"available": False, "reason": "term_slice_out_of_bounds", "term_name": term_name}
            lhs_slice = lhs_tensor.reshape(-1)[term_start:term_end]
            rhs_slice = rhs_tensor.reshape(-1)[term_start:term_end]
            err = float(torch.max(torch.abs(lhs_slice - rhs_slice)).item()) if lhs_slice.numel() > 0 else 0.0
            errors.append(err)
            if err >= worst["abs_err"]:
                worst = {"lhs_step": int(lhs_idx), "rhs_step": int(rhs_idx), "abs_err": float(err)}

        per_term[term_name] = {
            "flat_dim": int(term_slice["flat_dim"]),
            "errors": _summarize(errors),
            "worst_step": worst,
        }

    ranked = sorted(
        [{"term": name, "max": float(report["errors"]["max"])} for name, report in per_term.items()],
        key=lambda item: item["max"],
        reverse=True,
    )
    return {
        "available": True,
        "shift": int(shift),
        "compared_steps": int(compared_steps),
        "per_term": per_term,
        "ranked_max": ranked,
    }


def _compare_runtime_at_shift(
    ours_trace: dict[str, Any],
    unitree_trace: dict[str, Any],
    shift: int,
    topk: int,
    ours_obs_schema: dict[str, Any] | None = None,
    unitree_obs_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    result["policy_obs"] = _compare_tensor_series_with_shift(
        ours_trace["policy_obs"], unitree_trace["policy_obs"], shift=shift, topk=topk
    )
    result["critic_obs"] = _compare_tensor_series_with_shift(
        ours_trace["critic_obs"], unitree_trace["critic_obs"], shift=shift, topk=topk
    )
    result["reward_total"] = _compare_scalar_series_with_shift(
        ours_trace["reward_total"], unitree_trace["reward_total"], shift=shift, topk=topk
    )
    result["terminated"] = _compare_bool_series_with_shift(
        ours_trace["terminated"], unitree_trace["terminated"], shift=shift, topk=topk
    )
    result["truncated"] = _compare_bool_series_with_shift(
        ours_trace["truncated"], unitree_trace["truncated"], shift=shift, topk=topk
    )

    ours_reward_terms = ours_trace["reward_terms"]
    unitree_reward_terms = unitree_trace["reward_terms"]
    common_reward_terms = sorted(set(ours_reward_terms.keys()).intersection(unitree_reward_terms.keys()))
    result["reward_terms"] = {}
    for term_name in common_reward_terms:
        result["reward_terms"][term_name] = _compare_scalar_series_with_shift(
            ours_reward_terms[term_name],
            unitree_reward_terms[term_name],
            shift=shift,
            topk=topk,
        )
    result["reward_terms_missing_in_ours"] = sorted(set(unitree_reward_terms.keys()) - set(ours_reward_terms.keys()))
    result["reward_terms_missing_in_unitree"] = sorted(set(ours_reward_terms.keys()) - set(unitree_reward_terms.keys()))

    ours_termination_terms = ours_trace["termination_terms"]
    unitree_termination_terms = unitree_trace["termination_terms"]
    common_termination_terms = sorted(set(ours_termination_terms.keys()).intersection(unitree_termination_terms.keys()))
    result["termination_terms"] = {}
    for term_name in common_termination_terms:
        result["termination_terms"][term_name] = _compare_bool_series_with_shift(
            ours_termination_terms[term_name],
            unitree_termination_terms[term_name],
            shift=shift,
            topk=topk,
        )
    result["termination_terms_missing_in_ours"] = sorted(
        set(unitree_termination_terms.keys()) - set(ours_termination_terms.keys())
    )
    result["termination_terms_missing_in_unitree"] = sorted(
        set(ours_termination_terms.keys()) - set(unitree_termination_terms.keys())
    )

    if ours_obs_schema is not None and unitree_obs_schema is not None:
        result["obs_term_breakdown"] = {
            "policy": _compare_obs_group_term_breakdown(
                ours_trace["policy_obs"],
                unitree_trace["policy_obs"],
                ours_obs_schema.get("policy"),
                unitree_obs_schema.get("policy"),
                shift=shift,
            ),
            "critic": _compare_obs_group_term_breakdown(
                ours_trace["critic_obs"],
                unitree_trace["critic_obs"],
                ours_obs_schema.get("critic"),
                unitree_obs_schema.get("critic"),
                shift=shift,
            ),
        }

    return result


def _compare_runtime(
    ours_trace: dict[str, Any],
    unitree_trace: dict[str, Any],
    ours_obs_schema: dict[str, Any] | None = None,
    unitree_obs_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _compare_runtime_at_shift(
        ours_trace=ours_trace,
        unitree_trace=unitree_trace,
        shift=0,
        topk=args_cli.topk_mismatch,
        ours_obs_schema=ours_obs_schema,
        unitree_obs_schema=unitree_obs_schema,
    )


def _runtime_pass(runtime_result: dict[str, Any], obs_tol: float, reward_tol: float) -> bool:
    if not runtime_result["policy_obs"]["pass"] or not runtime_result["critic_obs"]["pass"]:
        return False
    if runtime_result["policy_obs"]["errors"]["max"] > obs_tol:
        return False
    if runtime_result["critic_obs"]["errors"]["max"] > obs_tol:
        return False
    if runtime_result["reward_total"]["errors"]["max"] > reward_tol:
        return False
    if runtime_result["terminated"]["mismatch_count"] > 0:
        return False
    if runtime_result["truncated"]["mismatch_count"] > 0:
        return False
    if len(runtime_result["reward_terms_missing_in_ours"]) > 0:
        return False
    if len(runtime_result["reward_terms_missing_in_unitree"]) > 0:
        return False
    if len(runtime_result["termination_terms_missing_in_ours"]) > 0:
        return False
    if len(runtime_result["termination_terms_missing_in_unitree"]) > 0:
        return False
    for term_report in runtime_result["reward_terms"].values():
        if term_report["errors"]["max"] > reward_tol:
            return False
    for term_report in runtime_result["termination_terms"].values():
        if term_report["mismatch_count"] > 0:
            return False
    return True


def _write_report(report: dict[str, Any]) -> None:
    print(json.dumps(report, indent=2, sort_keys=True))
    if args_cli.report_path is not None:
        report_path = Path(args_cli.report_path).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[INFO] Wrote report to: {report_path}")


def main() -> None:
    watchdog_stop = _start_timeout_watchdog(float(args_cli.max_runtime_s))
    try:
        if args_cli.collector_role in ("ours", "unitree"):
            if args_cli.stream_path is None:
                raise ValueError("collector role requires --stream_path.")
            stream_path = Path(args_cli.stream_path).expanduser().resolve()
            _collect_payload_for_role(args_cli.collector_role, stream_path=stream_path)
            return

        with tempfile.TemporaryDirectory(prefix="dance102_manager_parity_") as temp_dir:
            temp_dir_path = Path(temp_dir)
            ours_stream_path = temp_dir_path / "ours_payload.pt"
            unitree_stream_path = temp_dir_path / "unitree_payload.pt"

            ours_payload = _run_collector_subprocess("ours", ours_stream_path)
            unitree_payload = _run_collector_subprocess("unitree", unitree_stream_path)

        schema_observation = _compare_observation_schema(
            ours_payload["observation_schema"], unitree_payload["observation_schema"]
        )
        schema_reward = _compare_reward_schema(
            ours_payload["reward_schema"], unitree_payload["reward_schema"], weight_tol=args_cli.reward_tol
        )
        schema_termination = _compare_termination_schema(
            ours_payload["termination_schema"], unitree_payload["termination_schema"]
        )
        schema_pass = bool(schema_observation["pass"] and schema_reward["pass"] and schema_termination["pass"])

        runtime_result = _compare_runtime(
            ours_payload["runtime_trace"],
            unitree_payload["runtime_trace"],
            ours_obs_schema=ours_payload["observation_schema"],
            unitree_obs_schema=unitree_payload["observation_schema"],
        )
        runtime_pass = _runtime_pass(runtime_result, obs_tol=args_cli.obs_tol, reward_tol=args_cli.reward_tol)

        report = {
            "task_ours": ours_payload["resolved_task"],
            "task_unitree": unitree_payload["resolved_task"],
            "task_ours_requested": ours_payload["requested_task"],
            "task_unitree_requested": unitree_payload["requested_task"],
            "motion_path": ours_payload["motion_path"],
            "num_envs": int(args_cli.num_envs),
            "steps": int(args_cli.steps),
            "runtime_mode": str(args_cli.runtime_mode),
            "schema": {
                "observation": schema_observation,
                "reward": schema_reward,
                "termination": schema_termination,
                "pass": bool(schema_pass),
            },
            "runtime": runtime_result,
            "thresholds": {
                "obs_tol": float(args_cli.obs_tol),
                "reward_tol": float(args_cli.reward_tol),
            },
            "schema_pass": bool(schema_pass),
            "runtime_pass": bool(runtime_pass),
        }
        report["pass"] = bool(schema_pass and runtime_pass)
        _write_report(report)
    finally:
        if watchdog_stop is not None:
            watchdog_stop.set()


if __name__ == "__main__":
    main()
