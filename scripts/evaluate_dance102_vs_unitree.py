# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate dance_102 parity between IsaacLab-Imitation and unitree_rl_lab env commands.

This compares the *reference motion stream* used by:
- Isaac-Imitation-G1-Dance102-Compare-v0 (iltools LAFAN1 loader)
- Unitree-G1-29dof-Mimic-Dance-102 (original unitree_rl_lab command term)

Example:
    python scripts/evaluate_dance102_vs_unitree.py \
        --motion_path /abs/path/G1_Take_102.bvh_60hz.npz \
        --steps 1200 --headless
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


def _append_workspace_sources() -> None:
    """Best-effort source path setup for local mono-workspace usage."""
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

# CLI
parser = argparse.ArgumentParser(description="Compare dance_102 command/reference parity with unitree_rl_lab.")
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
parser.add_argument(
    "--motion_path",
    type=str,
    default=None,
    help="Optional dance motion npz to force in both envs.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of simulated envs.")
parser.add_argument("--steps", type=int, default=1200, help="Number of comparison steps.")
parser.add_argument("--seed", type=int, default=42, help="Seed for both env configs.")
parser.add_argument(
    "--allow_torch_compile",
    action="store_true",
    default=False,
    help=(
        "Allow torch.compile kernels. Disabled by default to avoid long/unstable inductor compilation "
        "during parity checks."
    ),
)
parser.add_argument(
    "--max_runtime_s",
    type=float,
    default=900.0,
    help="Hard timeout in seconds for the full evaluation. Set <= 0 to disable.",
)
parser.add_argument(
    "--progress_every",
    type=int,
    default=20,
    help="Print progress every N steps while collecting streams. Set <= 0 to disable.",
)
parser.add_argument(
    "--enable_reference_visualizer",
    action="store_true",
    default=False,
    help="Enable imitation reference markers/frames. Disabled by default for faster parity evaluation.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--pos_tol",
    type=float,
    default=1.0e-6,
    help="Pass/fail threshold for position max abs error.",
)
parser.add_argument(
    "--quat_tol_deg",
    type=float,
    default=1.0e-5,
    help="Pass/fail threshold for quaternion angular error (degrees).",
)
parser.add_argument(
    "--vel_tol",
    type=float,
    default=1.0e-6,
    help="Pass/fail threshold for velocity max abs error.",
)
parser.add_argument(
    "--joint_tol",
    type=float,
    default=1.0e-6,
    help="Pass/fail threshold for joint position/velocity max abs error.",
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

# Disable torch compile by default for deterministic, fast parity checks.
if not args_cli.allow_torch_compile:
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import gymnasium as gym
import numpy as np
import torch


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


def _quat_angle_error_deg(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Quaternion angular error in degrees, sign-invariant (wxyz layout).

    Uses relative-quaternion atan2 formulation so equal-but-nonunit quaternions still report ~0 error.
    """
    w1, x1, y1, z1 = lhs.unbind(dim=-1)
    w2, x2, y2, z2 = rhs.unbind(dim=-1)

    # lhs * conjugate(rhs)
    w = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2
    x = -w1 * x2 + x1 * w2 - y1 * z2 + z1 * y2
    y = -w1 * y2 + x1 * z2 + y1 * w2 - z1 * x2
    z = -w1 * z2 - x1 * y2 + y1 * x2 + z1 * w2

    v_norm = torch.sqrt(torch.clamp(x * x + y * y + z * z, min=0.0))
    angle = 2.0 * torch.atan2(v_norm, torch.abs(w).clamp(min=1.0e-12))
    return torch.rad2deg(angle)


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
    # Preserve body keys for tracking terms by matching source fps when available.
    try:
        with np.load(motion_path) as npz_data:
            if "fps" in npz_data.files:
                env_cfg.loader_kwargs["control_freq"] = float(np.asarray(npz_data["fps"]).reshape(-1)[0])
            else:
                env_cfg.loader_kwargs["control_freq"] = 1.0 / (float(env_cfg.sim.dt) * float(env_cfg.decimation))
    except Exception:
        env_cfg.loader_kwargs["control_freq"] = 1.0 / (float(env_cfg.sim.dt) * float(env_cfg.decimation))

    env_cfg.dataset_path = f"/tmp/iltools_eval_dance102_{motion_path.stem}"
    env_cfg.refresh_zarr_dataset = True
    env_cfg.motions = ["dance_102"]
    env_cfg.trajectories = ["trajectory_0"]
    env_cfg.reset_schedule = "sequential"
    env_cfg.replay_reference = True
    env_cfg.replay_only = True
    # Parity checks do not require marker rendering and these markers can dominate startup time.
    enable_vis = bool(args_cli.enable_reference_visualizer)
    if hasattr(env_cfg, "enable_visualizers"):
        env_cfg.enable_visualizers = enable_vis
    env_cfg.visualize_reference_arrows = enable_vis
    env_cfg.visualize_reference_velocity = enable_vis


def _configure_unitree_env_cfg(env_cfg, motion_path: Path) -> None:
    if not hasattr(env_cfg, "commands") or not hasattr(env_cfg.commands, "motion"):
        raise RuntimeError(
            "Selected unitree env cfg does not expose `commands.motion`. "
            "Pass a Unitree mimic dance task via --task_unitree "
            "(for example: Unitree-G1-29dof-Mimic-Dance-102 or its versioned variant)."
        )
    env_cfg.commands.motion.motion_file = str(motion_path)
    # Keep command deterministic for comparison.
    env_cfg.commands.motion.pose_range = {k: (0.0, 0.0) for k in ["x", "y", "z", "roll", "pitch", "yaw"]}
    env_cfg.commands.motion.velocity_range = {
        k: (0.0, 0.0) for k in ["x", "y", "z", "roll", "pitch", "yaw"]
    }
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    env_cfg.commands.motion.adaptive_uniform_ratio = 1.0
    env_cfg.commands.motion.adaptive_alpha = 0.0
    if hasattr(env_cfg.commands.motion, "debug_vis"):
        env_cfg.commands.motion.debug_vis = False
    if hasattr(env_cfg.scene, "contact_forces") and hasattr(env_cfg.scene.contact_forces, "debug_vis"):
        env_cfg.scene.contact_forces.debug_vis = False

    # Avoid early resets affecting command progression.
    env_cfg.episode_length_s = 1.0e9
    if hasattr(env_cfg.terminations, "anchor_pos"):
        env_cfg.terminations.anchor_pos = None
    if hasattr(env_cfg.terminations, "anchor_ori"):
        env_cfg.terminations.anchor_ori = None
    if hasattr(env_cfg.terminations, "ee_body_pos"):
        env_cfg.terminations.ee_body_pos = None


def _zero_action(env) -> torch.Tensor:
    action = torch.as_tensor(env.action_space.sample(), device=env.unwrapped.device)
    return torch.zeros_like(action)


def _clone_to_cpu(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().to("cpu").clone()


def _normalize_name(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith("_link"):
        lowered = lowered[:-5]
    return lowered


def _resolve_name_reindex(source_names: list[str], target_names: list[str]) -> list[int] | None:
    """Return indices into source_names matching target_names by tolerant name lookup."""
    if len(source_names) == 0 or len(target_names) == 0:
        return None
    exact = {name: i for i, name in enumerate(source_names)}
    lower = {name.lower(): i for i, name in enumerate(source_names)}
    normalized: dict[str, int] = {}
    for i, name in enumerate(source_names):
        normalized.setdefault(_normalize_name(name), i)

    indices: list[int] = []
    for target_name in target_names:
        idx = exact.get(target_name)
        if idx is None:
            idx = lower.get(target_name.lower())
        if idx is None:
            idx = normalized.get(_normalize_name(target_name))
        if idx is None:
            return None
        indices.append(int(idx))
    return indices


def _check_deadline(deadline_s: float | None, stage: str, step: int, total_steps: int) -> None:
    if deadline_s is None:
        return
    now_s = time.monotonic()
    if now_s > deadline_s:
        raise TimeoutError(f"Timed out during {stage} at step {step}/{total_steps}.")


def _start_timeout_watchdog(max_runtime_s: float) -> threading.Event | None:
    """Force-terminate the process if wall-time exceeds max_runtime_s.

    This handles hangs inside C++/extension calls where Python-level timeout checks cannot run.
    """
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

    watchdog_thread = threading.Thread(target=_watchdog, name="eval-timeout-watchdog", daemon=True)
    watchdog_thread.start()
    return stop_event


def _collect_our_reference_stream(env, steps: int, deadline_s: float | None, progress_every: int):
    env.reset()
    action = _zero_action(env)

    stream = {
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    terminated_counts = 0
    truncated_counts = 0

    for step_idx in range(steps):
        _check_deadline(deadline_s, "our stream collection", step_idx, steps)
        _, _, terminated, truncated, _ = env.step(action)
        terminated_counts += int(terminated.sum().item())
        truncated_counts += int(truncated.sum().item())

        ref = env.unwrapped.get_reference_data()
        if "body_pos_w" not in ref.keys() or "body_quat_w" not in ref.keys():
            raise RuntimeError(
                "body_pos_w/body_quat_w not found in imitation reference stream. "
                "Ensure motion npz contains body states and loader control_freq matches source fps."
            )

        stream["joint_pos"].append(_clone_to_cpu(ref["joint_pos"]))
        stream["joint_vel"].append(_clone_to_cpu(ref["joint_vel"]))
        stream["body_pos_w"].append(_clone_to_cpu(ref["body_pos_w"]))
        stream["body_quat_w"].append(_clone_to_cpu(ref["body_quat_w"]))
        stream["body_lin_vel_w"].append(_clone_to_cpu(ref.get("body_lin_vel_w")))
        stream["body_ang_vel_w"].append(_clone_to_cpu(ref.get("body_ang_vel_w")))

        if progress_every > 0 and ((step_idx + 1) % progress_every == 0 or step_idx + 1 == steps):
            print(f"[INFO] Collected ours: {step_idx + 1}/{steps} steps")

    return stream, terminated_counts, truncated_counts


def _collect_unitree_command_stream(env, steps: int, deadline_s: float | None, progress_every: int):
    env.reset()
    command = env.unwrapped.command_manager.get_term("motion")
    command.time_steps.zero_()

    action = _zero_action(env)
    stream = {
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    terminated_counts = 0
    truncated_counts = 0

    for step_idx in range(steps):
        _check_deadline(deadline_s, "unitree stream collection", step_idx, steps)
        _, _, terminated, truncated, _ = env.step(action)
        terminated_counts += int(terminated.sum().item())
        truncated_counts += int(truncated.sum().item())

        command = env.unwrapped.command_manager.get_term("motion")
        stream["joint_pos"].append(_clone_to_cpu(command.joint_pos))
        stream["joint_vel"].append(_clone_to_cpu(command.joint_vel))
        stream["body_pos_w"].append(_clone_to_cpu(command.body_pos_w))
        stream["body_quat_w"].append(_clone_to_cpu(command.body_quat_w))
        stream["body_lin_vel_w"].append(_clone_to_cpu(command.body_lin_vel_w))
        stream["body_ang_vel_w"].append(_clone_to_cpu(command.body_ang_vel_w))

        if progress_every > 0 and ((step_idx + 1) % progress_every == 0 or step_idx + 1 == steps):
            print(f"[INFO] Collected unitree: {step_idx + 1}/{steps} steps")

    return stream, terminated_counts, truncated_counts


def _compute_stream_errors(our_stream, unitree_stream):
    start_our = 0
    start_unitree = 0
    compared_steps = min(
        len(our_stream["joint_pos"]) - start_our,
        len(unitree_stream["joint_pos"]) - start_unitree,
    )
    if compared_steps <= 0:
        raise RuntimeError(
            "Invalid stream lengths for strict same-step comparison: "
            f"ours={len(our_stream['joint_pos'])}, unitree={len(unitree_stream['joint_pos'])}."
        )

    our_joint_names = list(our_stream.get("joint_names", []) or [])
    unitree_joint_names = list(unitree_stream.get("joint_names", []) or [])
    joint_reindex = _resolve_name_reindex(our_joint_names, unitree_joint_names)
    use_joint_reindex = False
    if joint_reindex is not None and len(joint_reindex) > 0:
        first_our_joint_pos = our_stream["joint_pos"][start_our]
        first_unitree_joint_pos = unitree_stream["joint_pos"][start_unitree]
        if len(joint_reindex) == first_unitree_joint_pos.shape[-1]:
            direct_joint_err = float(torch.max(torch.abs(first_our_joint_pos - first_unitree_joint_pos)).item())
            mapped_joint_err = float(
                torch.max(torch.abs(first_our_joint_pos[..., joint_reindex] - first_unitree_joint_pos)).item()
            )
            if mapped_joint_err <= direct_joint_err:
                use_joint_reindex = True

    our_body_names = list(our_stream.get("body_names", []) or [])
    unitree_body_names = list(unitree_stream.get("body_names", []) or [])
    body_reindex = _resolve_name_reindex(our_body_names, unitree_body_names)

    joint_pos_err: list[float] = []
    joint_vel_err: list[float] = []
    body_pos_err: list[float] = []
    body_lin_vel_err: list[float] = []
    body_ang_vel_err: list[float] = []
    body_quat_err_deg: list[float] = []

    for compare_idx in range(compared_steps):
        our_step_idx = start_our + compare_idx
        unitree_step_idx = start_unitree + compare_idx
        our_joint_pos = our_stream["joint_pos"][our_step_idx]
        our_joint_vel = our_stream["joint_vel"][our_step_idx]
        unitree_joint_pos = unitree_stream["joint_pos"][unitree_step_idx]
        unitree_joint_vel = unitree_stream["joint_vel"][unitree_step_idx]
        if use_joint_reindex and len(joint_reindex) == unitree_joint_pos.shape[-1]:
            our_joint_pos = our_joint_pos[..., joint_reindex]
            our_joint_vel = our_joint_vel[..., joint_reindex]
        joint_pos_err.append(float(torch.max(torch.abs(our_joint_pos - unitree_joint_pos)).item()))
        joint_vel_err.append(float(torch.max(torch.abs(our_joint_vel - unitree_joint_vel)).item()))

        ref_body_pos = our_stream["body_pos_w"][our_step_idx]
        ref_body_quat = our_stream["body_quat_w"][our_step_idx]
        ref_body_lin_vel = our_stream["body_lin_vel_w"][our_step_idx]
        ref_body_ang_vel = our_stream["body_ang_vel_w"][our_step_idx]

        unitree_body_pos = unitree_stream["body_pos_w"][unitree_step_idx]
        unitree_body_quat = unitree_stream["body_quat_w"][unitree_step_idx]
        unitree_body_lin_vel = unitree_stream["body_lin_vel_w"][unitree_step_idx]
        unitree_body_ang_vel = unitree_stream["body_ang_vel_w"][unitree_step_idx]

        if body_reindex is not None and len(body_reindex) == unitree_body_pos.shape[1]:
            selected_body_pos = ref_body_pos[:, body_reindex]
            selected_body_quat = ref_body_quat[:, body_reindex]
            selected_body_lin_vel = ref_body_lin_vel[:, body_reindex] if ref_body_lin_vel is not None else None
            selected_body_ang_vel = ref_body_ang_vel[:, body_reindex] if ref_body_ang_vel is not None else None
        elif ref_body_pos.shape[1] == unitree_body_pos.shape[1]:
            selected_body_pos = ref_body_pos
            selected_body_quat = ref_body_quat
            selected_body_lin_vel = ref_body_lin_vel
            selected_body_ang_vel = ref_body_ang_vel
        else:
            raise RuntimeError(
                "Reference body array shape is incompatible with unitree tracked bodies and no name mapping was found. "
                f"reference bodies={ref_body_pos.shape[1]}, unitree tracked bodies={unitree_body_pos.shape[1]}."
            )

        body_pos_err.append(float(torch.max(torch.abs(selected_body_pos - unitree_body_pos)).item()))
        body_quat_err_deg.append(float(torch.max(_quat_angle_error_deg(selected_body_quat, unitree_body_quat)).item()))

        if selected_body_lin_vel is not None:
            body_lin_vel_err.append(float(torch.max(torch.abs(selected_body_lin_vel - unitree_body_lin_vel)).item()))
        if selected_body_ang_vel is not None:
            body_ang_vel_err.append(float(torch.max(torch.abs(selected_body_ang_vel - unitree_body_ang_vel)).item()))

    return {
        "compared_steps": int(compared_steps),
        "joint_pos_err": joint_pos_err,
        "joint_vel_err": joint_vel_err,
        "body_pos_err": body_pos_err,
        "body_quat_err_deg": body_quat_err_deg,
        "body_lin_vel_err": body_lin_vel_err,
        "body_ang_vel_err": body_ang_vel_err,
    }


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "max": 0.0}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {"mean": float(tensor.mean().item()), "max": float(tensor.max().item())}


def _resolve_registered_task_id(task_name: str, kind: str) -> str:
    """Resolve a task id against Gym registry with robust fallbacks."""
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
                f"Requested unitree task '{requested}' is not registered and no Unitree tasks were found.\n"
                "Ensure unitree_rl_lab is installed/importable, or pass a valid --task_unitree explicitly."
            )

        preferred_candidates = [
            task_id
            for task_id in unitree_candidates
            if "g1" in task_id.lower() and ("mimic" in task_id.lower() or "dance" in task_id.lower())
        ]
        search_pool = preferred_candidates if preferred_candidates else unitree_candidates

        def _score_unitree(candidate: str) -> tuple[int, int, int]:
            candidate_lower = candidate.lower()
            overlap = sum(token in candidate_lower for token in requested_tokens)
            return overlap, -len(candidate), -all_ids.index(candidate)

        resolved = sorted(search_pool, key=_score_unitree, reverse=True)[0]
        print(
            f"[INFO] Requested unitree task '{requested}' was not registered. "
            f"Using closest Unitree match: '{resolved}'."
        )
        return resolved

    else:
        preferred_keywords = ("isaac", "imitation", "g1", "dance", "compare")
        fallback_keywords = ("isaac", "imitation", "g1", "dance")

    preferred_matches = [
        task_id
        for task_id in all_ids
        if all(keyword in task_id.lower() for keyword in preferred_keywords)
    ]
    if preferred_matches:
        print(
            f"[INFO] Requested {kind} task '{requested}' was not registered. "
            f"Using closest registered match: '{preferred_matches[0]}'."
        )
        return preferred_matches[0]

    keyword_matches = [
        task_id
        for task_id in all_ids
        if any(keyword in task_id.lower() for keyword in fallback_keywords)
    ]
    if keyword_matches:
        requested_tokens = [tok for tok in re.split(r"[^a-z0-9]+", requested.lower()) if tok]

        def _score(candidate: str) -> tuple[int, int, int]:
            candidate_lower = candidate.lower()
            overlap = sum(token in candidate_lower for token in requested_tokens)
            return overlap, -len(candidate), -all_ids.index(candidate)

        resolved = sorted(keyword_matches, key=_score, reverse=True)[0]
        print(
            f"[INFO] Requested {kind} task '{requested}' was not registered. "
            f"Using fallback match: '{resolved}'."
        )
        return resolved

    close_matches = difflib.get_close_matches(requested.lower(), all_ids_lower, n=10, cutoff=0.35)
    close_candidates = [all_ids[all_ids_lower.index(match)] for match in close_matches]
    raise RuntimeError(
        f"Task '{requested}' is not registered and no {kind}-like tasks were found.\n"
        f"Closest registered tasks: {close_candidates}"
    )


def _collect_stream_for_role(role: str, stream_path: Path | None = None) -> dict:
    """Collect one stream in an isolated simulator process context."""
    if role not in ("ours", "unitree"):
        raise ValueError(f"Unsupported collector role: {role}")

    print(f"[INFO] Collector role '{role}' starting.", flush=True)
    deadline_s = None if args_cli.max_runtime_s <= 0 else (time.monotonic() + float(args_cli.max_runtime_s))
    motion_path = _resolve_default_motion_path()
    if not motion_path.is_file():
        raise FileNotFoundError(
            f"Dance motion npz not found at: {motion_path}\n"
            "Provide --motion_path or generate it with:\n"
            "python unitree_rl_lab/scripts/mimic/csv_to_npz.py "
            "-f unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/dance_102/G1_Take_102.bvh_60hz.csv"
        )

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    print(f"[INFO] Collector role '{role}' launched AppLauncher.", flush=True)

    env = None
    try:
        # Keep task registration scoped to the collector subprocess.
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
            print("[INFO] Collector role 'ours' created env.", flush=True)
            stream, terminated_counts, truncated_counts = _collect_our_reference_stream(
                env, args_cli.steps, deadline_s=deadline_s, progress_every=args_cli.progress_every
            )
            print("[INFO] Collector role 'ours' finished collection.", flush=True)
            resolved_task = resolved_task_ours
            requested_task = args_cli.task_ours
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
            print("[INFO] Collector role 'unitree' created env.", flush=True)
            stream, terminated_counts, truncated_counts = _collect_unitree_command_stream(
                env, args_cli.steps, deadline_s=deadline_s, progress_every=args_cli.progress_every
            )
            print("[INFO] Collector role 'unitree' finished collection.", flush=True)
            resolved_task = resolved_task_unitree
            requested_task = args_cli.task_unitree

        our_joint_names = []
        our_body_names = []
        if role == "ours":
            tm = getattr(env.unwrapped, "trajectory_manager", None)
            if tm is not None and getattr(tm, "target_joint_names", None) is not None:
                our_joint_names = list(tm.target_joint_names)
            elif getattr(env.unwrapped, "reference_joint_names", None) is not None:
                our_joint_names = list(env.unwrapped.reference_joint_names)
            our_body_names = list(getattr(env.unwrapped, "reference_body_names", []) or [])
        else:
            cmd = env.unwrapped.command_manager.get_term("motion")
            if getattr(cmd, "cfg", None) is not None and getattr(cmd.cfg, "body_names", None) is not None:
                our_body_names = list(cmd.cfg.body_names)
            if getattr(cmd, "cfg", None) is not None and getattr(cmd.cfg, "joint_names", None) is not None:
                our_joint_names = list(cmd.cfg.joint_names)
            elif getattr(cmd, "joint_names", None) is not None:
                our_joint_names = list(cmd.joint_names)
            else:
                our_joint_names = list(env.unwrapped.scene["robot"].joint_names)

        # Keep names inside the stream payload because downstream comparison operates on stream dicts.
        stream["joint_names"] = list(our_joint_names)
        stream["body_names"] = list(our_body_names)

        payload = {
            "role": role,
            "resolved_task": resolved_task,
            "requested_task": requested_task,
            "motion_path": str(motion_path),
            "steps": int(args_cli.steps),
            "num_envs": int(args_cli.num_envs),
            "joint_names": our_joint_names,
            "body_names": our_body_names,
            "stream": stream,
            "terminated_counts": int(terminated_counts),
            "truncated_counts": int(truncated_counts),
        }
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


def _run_collector_subprocess(role: str, stream_path: Path) -> dict:
    """Spawn a dedicated subprocess to collect a single role stream."""
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


def _write_report(report: dict) -> None:
    print(json.dumps(report, indent=2, sort_keys=True))
    if args_cli.report_path is not None:
        report_path = Path(args_cli.report_path).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[INFO] Wrote report to: {report_path}")


def main() -> None:
    watchdog_stop = _start_timeout_watchdog(float(args_cli.max_runtime_s))
    try:
        # Child mode: collect exactly one stream and return.
        if args_cli.collector_role in ("ours", "unitree"):
            if args_cli.stream_path is None:
                raise ValueError("collector role requires --stream_path.")
            stream_path = Path(args_cli.stream_path).expanduser().resolve()
            _collect_stream_for_role(args_cli.collector_role, stream_path=stream_path)
            return

        # Parent mode: collect each stream in an isolated subprocess to avoid simulator teardown deadlocks.
        with tempfile.TemporaryDirectory(prefix="dance102_parity_") as temp_dir:
            temp_dir_path = Path(temp_dir)
            ours_stream_path = temp_dir_path / "ours_stream.pt"
            unitree_stream_path = temp_dir_path / "unitree_stream.pt"

            ours_payload = _run_collector_subprocess("ours", ours_stream_path)
            unitree_payload = _run_collector_subprocess("unitree", unitree_stream_path)

        ours_steps = int(len(ours_payload["stream"]["joint_pos"]))
        unitree_steps = int(len(unitree_payload["stream"]["joint_pos"]))
        if ours_steps != unitree_steps:
            raise RuntimeError(
                f"Stream length mismatch: ours_steps={ours_steps}, unitree_steps={unitree_steps}. "
                "Both collectors must produce the same number of steps."
            )

        errors = _compute_stream_errors(ours_payload["stream"], unitree_payload["stream"])
        print(
            "[INFO] Stream alignment: strict same-step comparison "
            f"(lag_steps=0, compared_steps={errors['compared_steps']}).",
            flush=True,
        )
        terminated_counts = int(ours_payload["terminated_counts"]) + int(unitree_payload["terminated_counts"])
        truncated_counts = int(ours_payload["truncated_counts"]) + int(unitree_payload["truncated_counts"])

        report = {
            "task_ours": ours_payload["resolved_task"],
            "task_unitree": unitree_payload["resolved_task"],
            "task_ours_requested": ours_payload["requested_task"],
            "task_unitree_requested": unitree_payload["requested_task"],
            "motion_path": ours_payload["motion_path"],
            "steps": int(ours_steps),
            "compared_steps": int(errors["compared_steps"]),
            "comparison_mode": "strict_same_step",
            "num_envs": int(args_cli.num_envs),
            "joint_pos_abs": _summarize(errors["joint_pos_err"]),
            "joint_vel_abs": _summarize(errors["joint_vel_err"]),
            "body_pos_abs": _summarize(errors["body_pos_err"]),
            "body_quat_deg": _summarize(errors["body_quat_err_deg"]),
            "body_lin_vel_abs": _summarize(errors["body_lin_vel_err"]),
            "body_ang_vel_abs": _summarize(errors["body_ang_vel_err"]),
            "terminated_counts": terminated_counts,
            "truncated_counts": truncated_counts,
            "thresholds": {
                "joint_tol": float(args_cli.joint_tol),
                "pos_tol": float(args_cli.pos_tol),
                "quat_tol_deg": float(args_cli.quat_tol_deg),
                "vel_tol": float(args_cli.vel_tol),
            },
        }
        report["runtime_pass"] = bool(
            report["joint_pos_abs"]["max"] <= args_cli.joint_tol
            and report["joint_vel_abs"]["max"] <= args_cli.joint_tol
            and report["body_pos_abs"]["max"] <= args_cli.pos_tol
            and report["body_quat_deg"]["max"] <= args_cli.quat_tol_deg
            and report["body_lin_vel_abs"]["max"] <= args_cli.vel_tol
            and report["body_ang_vel_abs"]["max"] <= args_cli.vel_tol
        )
        report["pass"] = bool(report["runtime_pass"])

        _write_report(report)
    finally:
        if watchdog_stop is not None:
            watchdog_stop.set()


if __name__ == "__main__":
    main()
