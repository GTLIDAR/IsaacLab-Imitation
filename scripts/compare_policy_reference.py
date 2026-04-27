#!/usr/bin/env python3
# ruff: noqa: E402

"""Side-by-side reference/policy playback for an RLOpt checkpoint."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Compare a policy-controlled robot against expert reference replay."
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during play."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--algo",
    "--algorithm",
    dest="algorithm",
    type=str.upper,
    default="IPMD",
    choices=[
        "PPO",
        "SAC",
        "FASTSAC",
        "IPMD",
        "IPMD_SR",
        "IPMD_BILINEAR",
        "GAIL",
        "AMP",
        "ASE",
    ],
    help="RLOpt algorithm (must match the checkpoint).",
)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)."
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment."
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import random
import time

import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.dict import print_dict
from isaaclab_imitation.envs.imitation_rl_env import ImitationRLEnv
from isaaclab_imitation.envs.rlopt import IsaacLabTerminalObsReader, IsaacLabWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rlopt.agent import AMP, ASE, GAIL, IPMD, IPMDBilinear, IPMDSR, PPO, SAC, FastSAC
from tensordict.nn import InteractionType
from torchrl.envs import Compose, RewardClipping, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import set_exploration_type, step_mdp

import isaaclab_tasks  # noqa: F401
import isaaclab_imitation.tasks  # noqa: F401

ALGORITHM_CLASS_MAP = {
    "PPO": PPO,
    "SAC": SAC,
    "FASTSAC": FastSAC,
    "IPMD": IPMD,
    "IPMD_SR": IPMDSR,
    "IPMD_BILINEAR": IPMDBilinear,
    "GAIL": GAIL,
    "AMP": AMP,
    "ASE": ASE,
}

ENTRY_POINT_ALGORITHM_MAP = {
    "rlopt_ppo_cfg_entry_point": "PPO",
    "rlopt_sac_cfg_entry_point": "SAC",
    "rlopt_fastsac_cfg_entry_point": "FASTSAC",
    "rlopt_ipmd_cfg_entry_point": "IPMD",
    "rlopt_ipmd_sr_cfg_entry_point": "IPMD_SR",
    "rlopt_ipmd_bilinear_cfg_entry_point": "IPMD_BILINEAR",
    "rlopt_gail_cfg_entry_point": "GAIL",
    "rlopt_amp_cfg_entry_point": "AMP",
    "rlopt_ase_cfg_entry_point": "ASE",
}

REFERENCE_ENV_ID = 0
POLICY_ENV_ID = 1
REFERENCE_MARKER_COLOR = (0.0, 0.35, 1.0)
POLICY_MARKER_COLOR = (1.0, 0.1, 0.0)
MARKER_HEIGHT_OFFSET = 1.35


def resolve_agent_cfg_entry_point(task_name: str | None, algorithm: str) -> str:
    """Resolve the agent config entry point based on algorithm and task registry."""
    if task_name is None:
        return f"rlopt_{algorithm.lower()}_cfg_entry_point"
    task_id = task_name.split(":")[-1]
    algo_entry_point = f"rlopt_{algorithm.lower()}_cfg_entry_point"
    try:
        spec = gym.spec(task_id)
    except Exception as exc:
        msg = f"Could not resolve task '{task_id}' from registry."
        raise ValueError(msg) from exc

    if spec.kwargs.get(algo_entry_point) is not None:
        print(f"[INFO] Using agent config entry point: {algo_entry_point}")
        return algo_entry_point

    supported_algorithms = sorted(
        ENTRY_POINT_ALGORITHM_MAP[key]
        for key in ENTRY_POINT_ALGORITHM_MAP
        if spec.kwargs.get(key) is not None
    )
    msg = (
        "Unsupported task/algo combination: "
        f"task '{task_id}' does not expose an RLOpt config for '{algorithm}'. "
        f"Supported RLOpt algorithms for this task: {supported_algorithms}."
    )
    raise ValueError(msg)


def _unwrap_imitation_env(env) -> ImitationRLEnv:
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, ImitationRLEnv):
            return current
        current_unwrapped = getattr(current, "unwrapped", None)
        if isinstance(current_unwrapped, ImitationRLEnv):
            return current_unwrapped
        current = (
            getattr(current, "base_env", None)
            or getattr(current, "env", None)
            or getattr(current, "_env", None)
        )
    raise TypeError("Could not unwrap an ImitationRLEnv from the provided environment.")


def _create_role_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/reference_policy_role_markers",
        markers={
            "reference": sim_utils.SphereCfg(
                radius=0.08,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=REFERENCE_MARKER_COLOR
                ),
            ),
            "policy": sim_utils.SphereCfg(
                radius=0.08,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=POLICY_MARKER_COLOR
                ),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def _update_role_markers(
    base_env: ImitationRLEnv, role_markers: VisualizationMarkers
) -> None:
    root_pos = base_env.robot.data.root_pos_w[[REFERENCE_ENV_ID, POLICY_ENV_ID]].clone()
    root_pos[:, 2] += MARKER_HEIGHT_OFFSET
    marker_indices = torch.tensor([0, 1], dtype=torch.long, device=base_env.device)
    role_markers.visualize(translations=root_pos, marker_indices=marker_indices)


agent_entry_point = resolve_agent_cfg_entry_point(args_cli.task, args_cli.algorithm)


@hydra_task_config(args_cli.task, agent_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg,
):
    """Play an RLOpt policy next to the expert reference motion."""
    sync_input_keys = getattr(agent_cfg, "sync_input_keys", None)
    if callable(sync_input_keys):
        sync_input_keys()

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    env_cfg.scene.num_envs = 2
    agent_cfg.env.num_envs = 2
    agent_cfg.env.env_name = args_cli.task
    agent_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    agent_cfg.collector.frames_per_batch *= env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    if args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required for compare_policy_reference.py.")
    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    log_dir = os.path.dirname(checkpoint_path)
    env_cfg.log_dir = log_dir

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if isinstance(env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported for RLOpt play.")

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "compare_policy_reference"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during reference/policy comparison.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = IsaacLabWrapper(env)
    env = env.set_info_dict_reader(
        IsaacLabTerminalObsReader(
            observation_spec=env.observation_spec, backend="gymnasium"
        )
    )
    env = TransformedEnv(
        base_env=env,
        transform=Compose(
            RewardSum(),
            StepCounter(1000),
            RewardClipping(-10.0, 5.0),
        ),
    )

    base_env = _unwrap_imitation_env(env)
    base_env.configure_reference_replay_targets(
        source_env_ids=[POLICY_ENV_ID],
        target_env_ids=[REFERENCE_ENV_ID],
    )
    role_markers = _create_role_markers()

    agent_class = ALGORITHM_CLASS_MAP[args_cli.algorithm]
    agent = agent_class(env=env, config=agent_cfg)

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    agent.load_model(checkpoint_path)

    collector_policy = agent.collector_policy
    collector_policy.eval()

    dt = getattr(base_env, "step_dt", None)

    td = env.reset()
    base_env.apply_reference_replay_targets()
    _update_role_markers(base_env, role_markers)
    timestep = 0
    print(
        "[INFO] Starting comparison loop. env 0 replays reference, env 1 runs policy."
    )
    print("[INFO] Visual markers: blue sphere = REFERENCE, red sphere = POLICY.")

    while simulation_app.is_running():
        start_time = time.time()
        with (
            torch.inference_mode(),
            set_exploration_type(InteractionType.DETERMINISTIC),
        ):
            td = collector_policy(td)
            action = td.get("action")
            if action is None:
                raise KeyError(
                    "Collector output is missing the top-level 'action' tensor."
                )
            action[REFERENCE_ENV_ID].zero_()
            td = env.step(td)
            _update_role_markers(base_env, role_markers)
            td = step_mdp(
                td, exclude_reward=True, exclude_done=False, exclude_action=True
            )

        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break

        if args_cli.real_time and dt is not None:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
