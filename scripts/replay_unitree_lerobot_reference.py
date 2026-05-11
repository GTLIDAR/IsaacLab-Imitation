#!/usr/bin/env python3
# ruff: noqa: E402

"""Replay a Unitree LeRobot episode segment on the Isaac G1 model."""

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


def _append_workspace_sources() -> None:
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


def _default_video_output(repo_id: str, episode_index: int) -> str:
    repo_slug = repo_id.replace("/", "_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(
        Path("logs")
        / "unitree_lerobot_replay"
        / f"{repo_slug}_ep{episode_index}_{stamp}.mp4"
    )


parser = argparse.ArgumentParser(
    description=(
        "Stream a Unitree WBT LeRobot episode from Hugging Face and replay "
        "observation.state.robot_q_current on the Isaac G1 model."
    )
)
parser.add_argument(
    "--repo_id",
    type=str,
    default="unitreerobotics/G1_WBT_Brainco_Pickup_Pillow",
    help="Hugging Face dataset repo id.",
)
parser.add_argument("--split", type=str, default="train", help="Dataset split.")
parser.add_argument(
    "--episode_index", type=int, default=0, help="Episode index to replay."
)
parser.add_argument(
    "--state_field",
    type=str,
    default="observation.state.robot_q_current",
    help="36-wide Unitree configuration field to replay.",
)
parser.add_argument(
    "--quat_order",
    type=str,
    choices=("wxyz", "xyzw"),
    default="wxyz",
    help="Root quaternion order in --state_field.",
)
parser.add_argument(
    "--fps",
    type=float,
    default=30.0,
    help="Replay frame rate. Unitree WBT rows are currently treated as fixed-rate.",
)
parser.add_argument(
    "--max_frames",
    type=int,
    default=180,
    help="Maximum number of episode frames to stream and replay.",
)
parser.add_argument(
    "--max_scan_rows",
    type=int,
    default=100_000,
    help="Maximum streamed rows to scan while looking for --episode_index.",
)
parser.add_argument(
    "--video_output",
    type=str,
    default=None,
    help="Output MP4 path. Defaults under logs/unitree_lerobot_replay/.",
)
parser.add_argument(
    "--overwrite_video",
    action="store_true",
    default=False,
    help="Overwrite an existing --video_output.",
)
parser.add_argument(
    "--no_video",
    action="store_true",
    default=False,
    help="Replay without recording an MP4.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.fps <= 0:
    raise ValueError(f"--fps must be positive, got {args_cli.fps}")
if args_cli.max_frames < 2:
    raise ValueError(f"--max_frames must be at least 2, got {args_cli.max_frames}")
if args_cli.max_scan_rows < args_cli.max_frames:
    raise ValueError(
        f"--max_scan_rows must be >= --max_frames, got {args_cli.max_scan_rows}"
    )

if not args_cli.no_video:
    args_cli.enable_cameras = True
    if args_cli.video_output is None:
        args_cli.video_output = _default_video_output(
            args_cli.repo_id, args_cli.episode_index
        )
    video_output_path = Path(args_cli.video_output).expanduser().resolve()
    video_output_path.parent.mkdir(parents=True, exist_ok=True)
    if video_output_path.exists() and not args_cli.overwrite_video:
        raise FileExistsError(
            f"Video output exists: {video_output_path}. "
            "Use --overwrite_video to replace it."
        )
    args_cli.video_output = str(video_output_path)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from datasets import load_dataset

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import omni.kit.app

from unitree_rl_lab.assets.robots.unitree import (
    UNITREE_G1_29DOF_MIMIC_CFG as ROBOT_CFG,
)

if not args_cli.no_video:
    extension_manager = omni.kit.app.get_app().get_extension_manager()
    extension_manager.set_extension_enabled_immediate("omni.kit.capture.viewport", True)
    extension_manager.set_extension_enabled_immediate("omni.videoencoding", True)

    import omni.kit.viewport.utility as vp_utils
    from omni.kit.capture.viewport import (
        CaptureExtension,
        CaptureOptions,
        CaptureRangeType,
    )
else:
    vp_utils = None
    CaptureExtension = None
    CaptureOptions = None
    CaptureRangeType = None


@dataclass(frozen=True)
class UnitreeLeRobotMotion:
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    joint_pos: torch.Tensor
    fps: float
    repo_id: str
    split: str
    episode_index: int
    state_field: str

    @property
    def frames(self) -> int:
        return int(self.root_pos.shape[0])

    @property
    def dt(self) -> float:
        return 1.0 / self.fps


def _nested_row_get(row: dict, key: str):
    if key in row:
        return row[key]
    value = row
    for part in key.split("."):
        value = value[part]
    return value


def _row_episode_index(row: dict) -> int:
    return int(torch.as_tensor(_nested_row_get(row, "episode_index")).item())


def _row_frame_index(row: dict) -> int:
    return int(torch.as_tensor(_nested_row_get(row, "frame_index")).item())


def _normalize_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    quat_norm = quat.norm(dim=-1, keepdim=True)
    if torch.any(quat_norm == 0):
        raise ValueError("Root quaternion contains a zero-norm sample.")
    return quat / quat_norm


def _make_quat_continuous_wxyz(quat: torch.Tensor) -> torch.Tensor:
    quat = quat.clone()
    for index in range(1, quat.shape[0]):
        if torch.dot(quat[index - 1], quat[index]) < 0:
            quat[index] = -quat[index]
    return quat


def _load_unitree_episode() -> UnitreeLeRobotMotion:
    dataset = load_dataset(args_cli.repo_id, split=args_cli.split, streaming=True)
    rows = []
    scanned_rows = 0
    for row in dataset:
        scanned_rows += 1
        episode_index = _row_episode_index(row)
        if episode_index == args_cli.episode_index:
            rows.append(row)
            if len(rows) >= args_cli.max_frames:
                break
        elif rows:
            break
        if scanned_rows >= args_cli.max_scan_rows:
            break

    if len(rows) < 2:
        raise RuntimeError(
            f"Found {len(rows)} rows for episode {args_cli.episode_index} "
            f"after scanning {scanned_rows} rows from {args_cli.repo_id}/{args_cli.split}."
        )

    rows.sort(key=_row_frame_index)
    q_current = torch.stack(
        [
            torch.as_tensor(
                _nested_row_get(row, args_cli.state_field), dtype=torch.float32
            )
            for row in rows
        ],
        dim=0,
    )
    if q_current.ndim != 2 or q_current.shape[1] != 36:
        raise ValueError(
            f"{args_cli.state_field} must have shape [T, 36], got "
            f"{tuple(q_current.shape)}"
        )

    root_pos = q_current[:, 0:3]
    root_quat = q_current[:, 3:7]
    if args_cli.quat_order == "xyzw":
        root_quat = root_quat[:, [3, 0, 1, 2]]
    root_quat = _make_quat_continuous_wxyz(_normalize_quat_wxyz(root_quat))
    joint_pos = q_current[:, 7:]
    if joint_pos.shape[1] != 29:
        raise ValueError(
            f"G1 replay expects 29 joint positions, got {joint_pos.shape[1]}"
        )

    print(
        "[INFO]: Loaded Unitree LeRobot episode:",
        f"repo={args_cli.repo_id}",
        f"split={args_cli.split}",
        f"episode={args_cli.episode_index}",
        f"field={args_cli.state_field}",
        f"frames={q_current.shape[0]}",
        f"q_shape={tuple(q_current.shape)}",
    )
    print(
        "[INFO]: Root xyz first/mid/last:",
        root_pos[0].tolist(),
        root_pos[root_pos.shape[0] // 2].tolist(),
        root_pos[-1].tolist(),
    )

    return UnitreeLeRobotMotion(
        root_pos=root_pos,
        root_quat=root_quat,
        joint_pos=joint_pos,
        fps=float(args_cli.fps),
        repo_id=args_cli.repo_id,
        split=args_cli.split,
        episode_index=int(args_cli.episode_index),
        state_field=args_cli.state_field,
    )


unitree_motion = _load_unitree_episode()


@configclass
class ReplayUnitreeLeRobotSceneCfg(InteractiveSceneCfg):
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


def _quat_conjugate_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat[..., :1], -quat[..., 1:]], dim=-1)


def _quat_mul_wxyz(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = lhs.unbind(dim=-1)
    w2, x2, y2, z2 = rhs.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _axis_angle_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    quat = _normalize_quat_wxyz(quat)
    vector = quat[..., 1:]
    vector_norm = vector.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, quat[..., :1].clamp(-1.0, 1.0))
    axis = vector / vector_norm.clamp_min(1.0e-8)
    return axis * angle


class IsaacReplayMotion:
    def __init__(self, motion: UnitreeLeRobotMotion, device: torch.device):
        self.device = device
        self.fps = motion.fps
        self.dt = motion.dt
        self.root_pos = motion.root_pos.to(device)
        self.root_quat = motion.root_quat.to(device)
        self.joint_pos = motion.joint_pos.to(device)
        self.root_lin_vel = torch.gradient(self.root_pos, spacing=self.dt, dim=0)[0]
        self.joint_vel = torch.gradient(self.joint_pos, spacing=self.dt, dim=0)[0]
        self.root_ang_vel = self._so3_derivative(self.root_quat, self.dt)
        self.current_index = 0

    @property
    def frames(self) -> int:
        return int(self.root_pos.shape[0])

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev = rotations[:-2]
        q_next = rotations[2:]
        q_rel = _quat_mul_wxyz(q_next, _quat_conjugate_wxyz(q_prev))
        omega = _axis_angle_from_quat_wxyz(q_rel) / (2.0 * dt)
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(self):
        index = self.current_index
        self.current_index += 1
        reset = self.current_index >= self.frames
        if reset:
            self.current_index = 0
        state = (
            self.root_pos[index : index + 1],
            self.root_quat[index : index + 1],
            self.root_lin_vel[index : index + 1],
            self.root_ang_vel[index : index + 1],
            self.joint_pos[index : index + 1],
            self.joint_vel[index : index + 1],
        )
        return state, reset


def _pump_app_once() -> None:
    if hasattr(simulation_app, "update"):
        simulation_app.update()
        return
    omni.kit.app.get_app().update()


def _start_video_capture(output_frames: int):
    if args_cli.no_video:
        return None
    if (
        CaptureExtension is None
        or CaptureOptions is None
        or CaptureRangeType is None
        or vp_utils is None
    ):
        raise RuntimeError(
            "Video capture requested, but viewport capture extensions are unavailable."
        )

    viewport = vp_utils.get_active_viewport()
    if viewport is None:
        raise RuntimeError(
            "Video capture requested, but no active viewport is available."
        )

    output_path = Path(args_cli.video_output).expanduser().resolve()
    capture = CaptureExtension.get_instance()
    options = CaptureOptions()
    options.camera = viewport.camera_path.pathString
    options.output_folder = str(output_path.parent)
    options.file_name = output_path.stem
    options.file_type = output_path.suffix or ".mp4"
    options.range_type = CaptureRangeType.FRAMES
    options.start_frame = 1
    options.end_frame = int(output_frames)
    options.capture_every_Nth_frames = 1
    options.fps = int(args_cli.fps)
    options.overwrite_existing_frames = True
    capture.options = options

    if not capture.start():
        raise RuntimeError(f"Failed to start video capture for: {output_path}")

    print("[INFO]: Recording replay video to", output_path)
    return capture


def _wait_for_capture(capture) -> None:
    if capture is None:
        return
    updates = 0
    max_updates = 900
    while not capture.done and simulation_app.is_running() and updates < max_updates:
        _pump_app_once()
        updates += 1
    if capture.done:
        print("[INFO]: Video capture completed:", capture.get_outputs())
    else:
        raise TimeoutError("Timed out waiting for video capture to finish.")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    motion = IsaacReplayMotion(unitree_motion, device=sim.device)
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(
        scene.cfg.robot.joint_sdk_names, preserve_order=True
    )[0]

    first_lookat = motion.root_pos[0].cpu().numpy()
    sim.set_camera_view(first_lookat + np.array([2.0, 2.0, 0.7]), first_lookat)
    capture = _start_video_capture(motion.frames)

    while simulation_app.is_running():
        (
            (
                root_pos,
                root_quat,
                root_lin_vel,
                root_ang_vel,
                joint_pos_input,
                joint_vel_input,
            ),
            reset,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = root_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = root_quat
        root_states[:, 7:10] = root_lin_vel
        root_states[:, 10:] = root_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = joint_pos_input
        joint_vel[:, robot_joint_indexes] = joint_vel_input
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(lookat + np.array([2.0, 2.0, 0.7]), lookat)
        sim.render()
        scene.update(sim.get_physics_dt())

        if reset:
            break

    _wait_for_capture(capture)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.fps
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplayUnitreeLeRobotSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
