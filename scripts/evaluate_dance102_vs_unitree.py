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
import json
import sys
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
    "--pos_tol",
    type=float,
    default=1.0e-5,
    help="Pass/fail threshold for position max abs error.",
)
parser.add_argument(
    "--quat_tol_deg",
    type=float,
    default=0.05,
    help="Pass/fail threshold for quaternion angular error (degrees).",
)
parser.add_argument(
    "--vel_tol",
    type=float,
    default=1.0e-4,
    help="Pass/fail threshold for velocity max abs error.",
)
parser.add_argument(
    "--joint_tol",
    type=float,
    default=1.0e-5,
    help="Pass/fail threshold for joint position/velocity max abs error.",
)
parser.add_argument(
    "--report_path",
    type=str,
    default=None,
    help="Optional JSON output path.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# runtime imports after AppLauncher
import gymnasium as gym
import numpy as np
import torch

import isaaclab_imitation  # noqa: F401
import isaaclab_tasks  # noqa: F401
import unitree_rl_lab  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


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
    """Quaternion angular error in degrees, sign-invariant."""
    dot = torch.sum(lhs * rhs, dim=-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.acos(dot))


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


def _configure_unitree_env_cfg(env_cfg, motion_path: Path) -> None:
    env_cfg.commands.motion.motion_file = str(motion_path)
    # Keep command deterministic for comparison.
    env_cfg.commands.motion.pose_range = {k: (0.0, 0.0) for k in ["x", "y", "z", "roll", "pitch", "yaw"]}
    env_cfg.commands.motion.velocity_range = {
        k: (0.0, 0.0) for k in ["x", "y", "z", "roll", "pitch", "yaw"]
    }
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    env_cfg.commands.motion.adaptive_uniform_ratio = 1.0
    env_cfg.commands.motion.adaptive_alpha = 0.0

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


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "max": 0.0}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {"mean": float(tensor.mean().item()), "max": float(tensor.max().item())}


def main() -> None:
    motion_path = _resolve_default_motion_path()
    if not motion_path.is_file():
        raise FileNotFoundError(
            "Dance motion npz not found. Provide --motion_path or generate it with:\n"
            "python /Users/fwu/Developer/SkillLearning/unitree_rl_lab/scripts/mimic/csv_to_npz.py "
            "-f /Users/fwu/Developer/SkillLearning/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/dance_102/G1_Take_102.bvh_60hz.csv"
        )

    our_cfg = parse_env_cfg(
        args_cli.task_ours,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    unitree_cfg = parse_env_cfg(
        args_cli.task_unitree,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    our_cfg.seed = args_cli.seed
    unitree_cfg.seed = args_cli.seed
    _configure_our_env_cfg(our_cfg, motion_path)
    _configure_unitree_env_cfg(unitree_cfg, motion_path)

    our_env = gym.make(args_cli.task_ours, cfg=our_cfg)
    unitree_env = gym.make(args_cli.task_unitree, cfg=unitree_cfg)

    try:
        our_env.reset()
        unitree_env.reset()

        # Force deterministic starting frame on unitree command stream.
        unitree_command = unitree_env.unwrapped.command_manager.get_term("motion")
        unitree_command.time_steps.zero_()

        our_action = _zero_action(our_env)
        unitree_action = _zero_action(unitree_env)

        joint_pos_err: list[float] = []
        joint_vel_err: list[float] = []
        body_pos_err: list[float] = []
        body_lin_vel_err: list[float] = []
        body_ang_vel_err: list[float] = []
        body_quat_err_deg: list[float] = []
        terminated_counts = 0
        truncated_counts = 0

        for _ in range(args_cli.steps):
            _, _, our_terminated, our_truncated, _ = our_env.step(our_action)
            _, _, unitree_terminated, unitree_truncated, _ = unitree_env.step(unitree_action)

            terminated_counts += int(our_terminated.sum().item()) + int(unitree_terminated.sum().item())
            truncated_counts += int(our_truncated.sum().item()) + int(unitree_truncated.sum().item())

            ref = our_env.unwrapped.get_reference_data()
            command = unitree_env.unwrapped.command_manager.get_term("motion")
            body_ids = command.body_indexes

            our_joint_pos = ref["joint_pos"]
            our_joint_vel = ref["joint_vel"]
            unitree_joint_pos = command.joint_pos
            unitree_joint_vel = command.joint_vel

            joint_pos_err.append(float(torch.max(torch.abs(our_joint_pos - unitree_joint_pos)).item()))
            joint_vel_err.append(float(torch.max(torch.abs(our_joint_vel - unitree_joint_vel)).item()))

            if "body_pos_w" not in ref.keys() or "body_quat_w" not in ref.keys():
                raise RuntimeError(
                    "body_pos_w/body_quat_w not found in imitation reference stream. "
                    "Ensure motion npz contains body states and loader control_freq matches source fps."
                )

            ref_body_pos = ref["body_pos_w"]
            ref_body_quat = ref["body_quat_w"]
            if int(body_ids.max().item()) < ref_body_pos.shape[1]:
                selected_body_pos = ref_body_pos[:, body_ids]
                selected_body_quat = ref_body_quat[:, body_ids]
                selected_body_lin_vel = ref.get("body_lin_vel_w")
                selected_body_ang_vel = ref.get("body_ang_vel_w")
                if selected_body_lin_vel is not None:
                    selected_body_lin_vel = selected_body_lin_vel[:, body_ids]
                if selected_body_ang_vel is not None:
                    selected_body_ang_vel = selected_body_ang_vel[:, body_ids]
            elif ref_body_pos.shape[1] == command.body_pos_w.shape[1]:
                selected_body_pos = ref_body_pos
                selected_body_quat = ref_body_quat
                selected_body_lin_vel = ref.get("body_lin_vel_w")
                selected_body_ang_vel = ref.get("body_ang_vel_w")
            else:
                raise RuntimeError(
                    "Reference body array shape is incompatible with unitree body indexes. "
                    f"reference bodies={ref_body_pos.shape[1]}, "
                    f"max unitree body index={int(body_ids.max().item())}, "
                    f"unitree tracked bodies={command.body_pos_w.shape[1]}."
                )

            our_body_pos = selected_body_pos
            our_body_quat = selected_body_quat
            unitree_body_pos = command.body_pos_w
            unitree_body_quat = command.body_quat_w

            body_pos_err.append(float(torch.max(torch.abs(our_body_pos - unitree_body_pos)).item()))
            body_quat_err_deg.append(float(torch.max(_quat_angle_error_deg(our_body_quat, unitree_body_quat)).item()))

            if selected_body_lin_vel is not None:
                body_lin_vel_err.append(
                    float(torch.max(torch.abs(selected_body_lin_vel - command.body_lin_vel_w)).item())
                )
            if selected_body_ang_vel is not None:
                body_ang_vel_err.append(
                    float(torch.max(torch.abs(selected_body_ang_vel - command.body_ang_vel_w)).item())
                )

        report = {
            "task_ours": args_cli.task_ours,
            "task_unitree": args_cli.task_unitree,
            "motion_path": str(motion_path),
            "steps": int(args_cli.steps),
            "num_envs": int(args_cli.num_envs),
            "joint_pos_abs": _summarize(joint_pos_err),
            "joint_vel_abs": _summarize(joint_vel_err),
            "body_pos_abs": _summarize(body_pos_err),
            "body_quat_deg": _summarize(body_quat_err_deg),
            "body_lin_vel_abs": _summarize(body_lin_vel_err),
            "body_ang_vel_abs": _summarize(body_ang_vel_err),
            "terminated_counts": int(terminated_counts),
            "truncated_counts": int(truncated_counts),
        }
        report["pass"] = bool(
            report["joint_pos_abs"]["max"] <= args_cli.joint_tol
            and report["joint_vel_abs"]["max"] <= args_cli.joint_tol
            and report["body_pos_abs"]["max"] <= args_cli.pos_tol
            and report["body_quat_deg"]["max"] <= args_cli.quat_tol_deg
            and report["body_lin_vel_abs"]["max"] <= args_cli.vel_tol
            and report["body_ang_vel_abs"]["max"] <= args_cli.vel_tol
        )

        print(json.dumps(report, indent=2, sort_keys=True))

        if args_cli.report_path is not None:
            report_path = Path(args_cli.report_path).expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"[INFO] Wrote report to: {report_path}")
    finally:
        our_env.close()
        unitree_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
