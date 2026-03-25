#!/usr/bin/env python3
# ruff: noqa: E402
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batch replay motion CSVs and export one NPZ per motion in a single Isaac Sim session."""

# Launch Isaac Sim Simulator first.

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

try:
    from tqdm.rich import TqdmExperimentalWarning, tqdm

    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
except ImportError:  # pragma: no cover - fallback only used in misconfigured envs
    from tqdm.auto import tqdm


def _append_workspace_sources() -> None:
    """Best-effort source path setup for local mono-workspace usage."""
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[1]
    workspace_root = repo_root.parent
    candidate_paths = [
        workspace_root / "IsaacLab" / "source" / "isaaclab",
        workspace_root / "IsaacLab" / "source" / "isaaclab_tasks",
        repo_root / "source" / "isaaclab_imitation",
        workspace_root / "unitree_rl_lab" / "source" / "unitree_rl_lab",
        repo_root / "unitree_rl_lab" / "source" / "unitree_rl_lab",
    ]
    for candidate in candidate_paths:
        if candidate.is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.append(candidate_str)


_append_workspace_sources()

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Batch replay motion CSV files and export one NPZ per motion."
)
parser.add_argument(
    "--jobs_json",
    type=str,
    required=True,
    help=(
        "JSON file containing a list of job objects with "
        "{'input_file', 'output_name', 'video_output?', 'frame_range?' }."
    ),
)
parser.add_argument(
    "--input_fps",
    type=float,
    default=60.0,
    help="The fps of the input motions.",
)
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "Frame range: START END (both inclusive). The frame index starts from 1. "
        "If not provided, all frames are loaded."
    ),
)
parser.add_argument(
    "--output_fps", type=float, default=50.0, help="The fps of the output motions."
)
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record one MP4 per motion using a per-env camera.",
)
parser.add_argument(
    "--overwrite_video",
    action="store_true",
    default=False,
    help="Overwrite existing MP4 outputs declared in the jobs JSON.",
)
parser.add_argument(
    "--video_width",
    type=int,
    default=640,
    help="Per-env video width in pixels.",
)
parser.add_argument(
    "--video_height",
    type=int,
    default=480,
    help="Per-env video height in pixels.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_from_matrix,
    quat_conjugate,
    quat_mul,
    quat_slerp,
)
from unitree_rl_lab.assets.robots.unitree import (
    UNITREE_G1_29DOF_CFG as ROBOT_CFG,
)


@dataclass(frozen=True)
class MotionJob:
    input_file: Path
    output_name: Path
    video_output: Path | None = None
    frame_range: tuple[int, int] | None = None


def _normalize_frame_range(value: object) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("frame_range must be [start, end].")
    start = int(value[0])
    end = int(value[1])
    if start < 1:
        raise ValueError("frame_range start must be >= 1.")
    if end < start:
        raise ValueError("frame_range end must be >= start.")
    return (start, end)


def _look_at_quat_world(
    camera_pos: tuple[float, float, float],
    target_pos: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    camera = torch.tensor(camera_pos, dtype=torch.float32)
    target = torch.tensor(target_pos, dtype=torch.float32)
    forward = target - camera
    forward = forward / torch.linalg.norm(forward)
    world_up = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32)
    left = torch.cross(world_up, forward, dim=0)
    left = left / torch.linalg.norm(left)
    up = torch.cross(forward, left, dim=0)
    rotation = torch.stack([forward, left, up], dim=1)
    quat = quat_from_matrix(rotation.unsqueeze(0))[0]
    return tuple(float(x) for x in quat.tolist())


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    lazy_sensor_update = not args_cli.video

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    if args_cli.video:
        video_camera: TiledCameraCfg = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/VideoCamera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(2.5, 2.5, 1.4),
                rot=_look_at_quat_world((2.5, 2.5, 1.4), (0.0, 0.0, 0.8)),
                convention="world",
            ),
            data_types=["rgb"],
            update_period=0,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 20.0),
            ),
            width=args_cli.video_width,
            height=args_cli.video_height,
        )


class MotionLoader:
    def __init__(
        self,
        motion_file: Path,
        input_fps: float,
        output_fps: float,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _count_csv_frames(self) -> int:
        with self.motion_file.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)

    def _load_motion(self) -> None:
        self.csv_max_frame = self._count_csv_frames()
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
            loaded_frame_range = (1, self.csv_max_frame)
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
            loaded_frame_range = (
                self.frame_range[0],
                self.frame_range[0] + int(motion.shape[0]) - 1,
            )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7][:, [3, 0, 1, 2]]
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(
            f"[INFO] Loaded {self.motion_file} | csv_max_frame={self.csv_max_frame} "
            f"| loaded_frame_range={loaded_frame_range[0]}-{loaded_frame_range[1]} "
            f"| loaded_input_frames={self.input_frames} | duration={self.duration:.3f} sec"
        )

    def _interpolate_motion(self) -> None:
        times = torch.arange(
            0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
        )
        self.output_frames = int(times.shape[0])
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )

    def _lerp(
        self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
    ) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    def _slerp(
        self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
    ) -> torch.Tensor:
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(
        self, times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(
            index_0 + 1,
            torch.full_like(index_0, self.input_frames - 1),
        )
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self) -> None:
        self.motion_base_lin_vels = torch.gradient(
            self.motion_base_poss, spacing=self.output_dt, dim=0
        )[0]
        self.motion_dof_vels = torch.gradient(
            self.motion_dof_poss, spacing=self.output_dt, dim=0
        )[0]
        self.motion_base_ang_vels = self._so3_derivative(
            self.motion_base_rots, self.output_dt
        )

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega


@dataclass
class BatchedMotionData:
    base_pos: torch.Tensor
    base_rot: torch.Tensor
    base_lin_vel: torch.Tensor
    base_ang_vel: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor
    lengths: torch.Tensor


def _load_jobs(jobs_json_path: Path) -> list[MotionJob]:
    payload = json.loads(jobs_json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) == 0:
        raise ValueError("--jobs_json must contain a non-empty list of jobs.")
    jobs: list[MotionJob] = []
    for job_like in payload:
        if not isinstance(job_like, dict):
            raise ValueError("Each job entry must be a JSON object.")
        input_file = Path(str(job_like["input_file"])).expanduser().resolve()
        output_name = Path(str(job_like["output_name"])).expanduser().resolve()
        if not input_file.is_file():
            raise FileNotFoundError(f"Input CSV not found: {input_file}")
        output_name.parent.mkdir(parents=True, exist_ok=True)
        video_output = None
        if "video_output" in job_like and job_like["video_output"] is not None:
            video_output = Path(str(job_like["video_output"])).expanduser().resolve()
            video_output.parent.mkdir(parents=True, exist_ok=True)
            if video_output.exists() and not args_cli.overwrite_video:
                raise FileExistsError(
                    f"Video output exists: {video_output}. Use --overwrite_video to replace it."
                )
        frame_range = _normalize_frame_range(job_like.get("frame_range"))
        jobs.append(
            MotionJob(
                input_file=input_file,
                output_name=output_name,
                video_output=video_output,
                frame_range=frame_range,
            )
        )
    return jobs


def _pad_sequence(sequence: torch.Tensor, target_length: int) -> torch.Tensor:
    if sequence.shape[0] == target_length:
        return sequence
    last = sequence[-1:].expand(target_length - sequence.shape[0], *sequence.shape[1:])
    return torch.cat([sequence, last], dim=0)


def _build_batched_motion_data(
    jobs: list[MotionJob],
    device: torch.device,
) -> BatchedMotionData:
    default_frame_range = (
        tuple(args_cli.frame_range) if args_cli.frame_range is not None else None
    )
    loaders = [
        MotionLoader(
            motion_file=job.input_file,
            input_fps=float(args_cli.input_fps),
            output_fps=float(args_cli.output_fps),
            device=device,
            frame_range=job.frame_range
            if job.frame_range is not None
            else default_frame_range,
        )
        for job in jobs
    ]
    max_frames = max(loader.output_frames for loader in loaders)
    lengths = torch.tensor(
        [loader.output_frames for loader in loaders], device=device, dtype=torch.long
    )
    base_pos = torch.stack(
        [_pad_sequence(loader.motion_base_poss, max_frames) for loader in loaders],
        dim=0,
    )
    base_rot = torch.stack(
        [_pad_sequence(loader.motion_base_rots, max_frames) for loader in loaders],
        dim=0,
    )
    base_lin_vel = torch.stack(
        [_pad_sequence(loader.motion_base_lin_vels, max_frames) for loader in loaders],
        dim=0,
    )
    base_ang_vel = torch.stack(
        [_pad_sequence(loader.motion_base_ang_vels, max_frames) for loader in loaders],
        dim=0,
    )
    dof_pos = torch.stack(
        [_pad_sequence(loader.motion_dof_poss, max_frames) for loader in loaders], dim=0
    )
    dof_vel = torch.stack(
        [_pad_sequence(loader.motion_dof_vels, max_frames) for loader in loaders], dim=0
    )
    return BatchedMotionData(
        base_pos=base_pos,
        base_rot=base_rot,
        base_lin_vel=base_lin_vel,
        base_ang_vel=base_ang_vel,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        lengths=lengths,
    )


def _save_outputs(
    jobs: list[MotionJob],
    lengths: np.ndarray,
    log_data: dict[str, np.ndarray],
) -> None:
    for env_id, job in enumerate(jobs):
        frame_count = int(lengths[env_id])
        np.savez(
            job.output_name,
            fps=[args_cli.output_fps],
            joint_pos=log_data["joint_pos"][env_id, :frame_count],
            joint_vel=log_data["joint_vel"][env_id, :frame_count],
            body_pos_w=log_data["body_pos_w"][env_id, :frame_count],
            body_quat_w=log_data["body_quat_w"][env_id, :frame_count],
            body_lin_vel_w=log_data["body_lin_vel_w"][env_id, :frame_count],
            body_ang_vel_w=log_data["body_ang_vel_w"][env_id, :frame_count],
        )
        print(f"[INFO] Saved NPZ: {job.output_name}")


def _open_video_writers(
    jobs: list[MotionJob],
) -> list[imageio.core.format.Writer | None]:
    if not args_cli.video:
        return [None] * len(jobs)
    writers: list[imageio.core.format.Writer | None] = []
    fps = max(1, int(round(float(args_cli.output_fps))))
    for job in jobs:
        if job.video_output is None:
            writers.append(None)
            continue
        if job.video_output.exists() and args_cli.overwrite_video:
            job.video_output.unlink()
        writers.append(
            imageio.get_writer(
                str(job.video_output),
                fps=fps,
                codec="libx264",
                macro_block_size=None,
            )
        )
        print(f"[INFO] Recording MP4: {job.video_output}")
    return writers


def _close_video_writers(writers: list[imageio.core.format.Writer | None]) -> None:
    for writer in writers:
        if writer is not None:
            writer.close()


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    jobs: list[MotionJob],
) -> None:
    batched = _build_batched_motion_data(jobs=jobs, device=sim.device)
    robot = scene["robot"]
    video_camera = scene.sensors.get("video_camera") if args_cli.video else None
    robot_joint_indexes = robot.find_joints(
        scene.cfg.robot.joint_sdk_names, preserve_order=True
    )[0]

    num_envs = len(jobs)
    max_frames = int(batched.lengths.max().item())
    print(
        f"[INFO] Replaying {num_envs} motion(s) for up to {max_frames} simulation steps "
        f"at output_fps={args_cli.output_fps}."
    )
    joint_shape = tuple(robot.data.joint_pos.shape[1:])
    body_pos_shape = tuple(robot.data.body_pos_w.shape[1:])
    body_quat_shape = tuple(robot.data.body_quat_w.shape[1:])
    body_vel_shape = tuple(robot.data.body_lin_vel_w.shape[1:])

    log_data = {
        "joint_pos": np.zeros((num_envs, max_frames, *joint_shape), dtype=np.float32),
        "joint_vel": np.zeros((num_envs, max_frames, *joint_shape), dtype=np.float32),
        "body_pos_w": np.zeros(
            (num_envs, max_frames, *body_pos_shape), dtype=np.float32
        ),
        "body_quat_w": np.zeros(
            (num_envs, max_frames, *body_quat_shape), dtype=np.float32
        ),
        "body_lin_vel_w": np.zeros(
            (num_envs, max_frames, *body_vel_shape), dtype=np.float32
        ),
        "body_ang_vel_w": np.zeros(
            (num_envs, max_frames, *body_vel_shape), dtype=np.float32
        ),
    }

    default_root_state = robot.data.default_root_state.clone()
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    if video_camera is not None:
        # Warm up tiled camera textures before writing frame 0 to avoid blank first frames.
        for _ in range(5):
            sim.render()
            scene.update(sim.get_physics_dt())
    video_writers = _open_video_writers(jobs)

    try:
        with tqdm(
            total=max_frames,
            desc="Batch replay",
            unit="step",
            dynamic_ncols=True,
        ) as progress_bar:
            for step in range(max_frames):
                active_mask = step < batched.lengths
                active_env_ids = active_mask.nonzero(as_tuple=False).squeeze(-1)

                root_states = default_root_state.clone()
                joint_pos = default_joint_pos.clone()
                joint_vel = default_joint_vel.clone()

                if active_env_ids.numel() > 0:
                    root_states_active = default_root_state[active_env_ids].clone()
                    root_states_active[:, :3] = batched.base_pos[active_env_ids, step]
                    root_states_active[:, :2] += scene.env_origins[active_env_ids, :2]
                    root_states_active[:, 3:7] = batched.base_rot[active_env_ids, step]
                    root_states_active[:, 7:10] = batched.base_lin_vel[
                        active_env_ids, step
                    ]
                    root_states_active[:, 10:] = batched.base_ang_vel[
                        active_env_ids, step
                    ]
                    root_states[active_env_ids] = root_states_active

                    joint_pos_active = default_joint_pos[active_env_ids].clone()
                    joint_vel_active = default_joint_vel[active_env_ids].clone()
                    joint_pos_active[:, robot_joint_indexes] = batched.dof_pos[
                        active_env_ids, step
                    ]
                    joint_vel_active[:, robot_joint_indexes] = batched.dof_vel[
                        active_env_ids, step
                    ]
                    joint_pos[active_env_ids] = joint_pos_active
                    joint_vel[active_env_ids] = joint_vel_active

                robot.write_root_state_to_sim(root_states)
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                sim.render()
                scene.update(sim.get_physics_dt())

                joint_pos_np = robot.data.joint_pos.cpu().numpy()
                joint_vel_np = robot.data.joint_vel.cpu().numpy()
                body_pos_np = robot.data.body_pos_w.cpu().numpy()
                body_quat_np = robot.data.body_quat_w.cpu().numpy()
                body_lin_vel_np = robot.data.body_lin_vel_w.cpu().numpy()
                body_ang_vel_np = robot.data.body_ang_vel_w.cpu().numpy()

                active_np = active_mask.cpu().numpy()
                log_data["joint_pos"][active_np, step] = joint_pos_np[active_np]
                log_data["joint_vel"][active_np, step] = joint_vel_np[active_np]
                log_data["body_pos_w"][active_np, step] = body_pos_np[active_np]
                log_data["body_quat_w"][active_np, step] = body_quat_np[active_np]
                log_data["body_lin_vel_w"][active_np, step] = body_lin_vel_np[active_np]
                log_data["body_ang_vel_w"][active_np, step] = body_ang_vel_np[active_np]

                if video_camera is not None:
                    rgb_frames = video_camera.data.output["rgb"]
                    if rgb_frames.shape[-1] > 3:
                        rgb_frames = rgb_frames[..., :3]
                    rgb_frames_np = rgb_frames.detach().cpu().numpy()
                    for env_id in active_env_ids.tolist():
                        writer = video_writers[env_id]
                        if writer is not None:
                            writer.append_data(rgb_frames_np[env_id])

                progress_bar.set_postfix_str(
                    f"active_envs={int(active_env_ids.numel())}",
                    refresh=False,
                )
                progress_bar.update()

        _save_outputs(
            jobs=jobs,
            lengths=batched.lengths.cpu().numpy(),
            log_data=log_data,
        )
    finally:
        _close_video_writers(video_writers)


def main() -> None:
    jobs = _load_jobs(Path(args_cli.jobs_json).expanduser().resolve())
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplayMotionsSceneCfg(num_envs=len(jobs), env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print(f"[INFO] Setup complete for {len(jobs)} motion(s).")
    run_simulator(sim=sim, scene=scene, jobs=jobs)


if __name__ == "__main__":
    main()
    # Isaac Sim shutdown can hang after offline batched export even after all NPZ files are written.
    # This script is used as a one-shot data conversion job, so exit the process explicitly.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
