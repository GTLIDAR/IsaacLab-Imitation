"""This is copied from torchrl's IsaacLabWrapper https://github.com/pytorch/rl/blob/main/torchrl/envs/libs/isaac_lab.py."""

from __future__ import annotations

from collections import deque
from dataclasses import MISSING
from typing import Any

import gymnasium as gym
import torch
from torchrl.data.tensor_specs import Composite, Unbounded
from torchrl.envs.libs.gym import GymWrapper, _gym_to_torchrl_spec_transform, terminal_obs_reader

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass


def _flatten_obs(obs: dict) -> dict:
    """Flatten one level of nested observation dicts.

    This hoists leaf tensors in grouped observations (for example, ``policy``)
    into top-level keys when needed.
    """
    flat = {}
    for key, value in obs.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


class IsaacLabWrapper(GymWrapper):
    """A wrapper for IsaacLab environments.

    Args:
        env (scripts_isaaclab.envs.ManagerBasedRLEnv or equivalent): the environment instance to wrap.
        categorical_action_encoding (bool, optional): if ``True``, categorical
            specs will be converted to the TorchRL equivalent (:class:`torchrl.data.Categorical`),
            otherwise a one-hot encoding will be used (:class:`torchrl.data.OneHot`).
            Defaults to ``False``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`reset` is called.
            Defaults to ``False``.

    For other arguments, see the :class:`torchrl.envs.GymWrapper` documentation.

    Refer to `the Isaac Lab doc for installation instructions <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html>`_.

    Example:
        >>> # This code block ensures that the Isaac app is started in headless mode
        >>> from scripts_isaaclab.app import AppLauncher
        >>> import argparse

        >>> parser = argparse.ArgumentParser(description="Train an RL agent with TorchRL.")
        >>> AppLauncher.add_app_launcher_args(parser)
        >>> args_cli, hydra_args = parser.parse_known_args(["--headless"])
        >>> app_launcher = AppLauncher(args_cli)

        >>> # Imports and env
        >>> import gymnasium as gym
        >>> import isaaclab_tasks  # noqa: F401
        >>> from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
        >>> from torchrl.envs.libs.isaac_lab import IsaacLabWrapper

        >>> env = gym.make("Isaac-Ant-v0", cfg=AntEnvCfg())
        >>> env = IsaacLabWrapper(env)

    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,  # noqa: F821
        *,
        categorical_action_encoding: bool = False,
        allow_done_after_reset: bool = True,
        convert_actions_to_numpy: bool = False,
        device: torch.device | None = None,
        **kwargs,
    ):
        if device is None:
            device = torch.device("cuda:0")
        super().__init__(
            env,
            device=device,
            categorical_action_encoding=categorical_action_encoding,
            allow_done_after_reset=allow_done_after_reset,
            convert_actions_to_numpy=convert_actions_to_numpy,
            **kwargs,
        )
        self.log_infos = deque()

    @property
    def _is_batched(self) -> bool:
        return True

    def seed(self, seed: int | None):
        self._set_seed(seed)

    def _build_env(
        self,
        env,
        from_pixels: bool = False,
        pixels_only: bool = False,
    ) -> gym.core.Env:  # noqa: F821
        env = super()._build_env(
            env,
            from_pixels=from_pixels,
            pixels_only=pixels_only,
        )
        env.autoreset_mode = "SameStep"
        return env

    def _make_specs(self, env: gym.Env, batch_size=None) -> None:  # noqa: F821
        # Build specs from IsaacLab's unbatched spaces to preserve observation keys.
        if batch_size is None:
            batch_size = self.batch_size
        env_unwrapped = getattr(env, "unwrapped", env)

        action_space = getattr(env_unwrapped, "single_action_space", None)
        action_needs_batch = action_space is not None
        action_space = action_space if action_space is not None else env.action_space
        action_spec = _gym_to_torchrl_spec_transform(
            action_space,
            device=self.device,
            categorical_action_encoding=self._categorical_action_encoding,
        )
        if action_needs_batch:
            action_spec = action_spec.expand(*batch_size, *action_spec.shape)  # type: ignore

        obs_space = getattr(env_unwrapped, "single_observation_space", None)
        obs_needs_batch = obs_space is not None
        obs_space = obs_space if obs_space is not None else env.observation_space
        observation_spec = _gym_to_torchrl_spec_transform(
            obs_space,
            device=self.device,
            categorical_action_encoding=self._categorical_action_encoding,
        )
        if obs_needs_batch:
            observation_spec = observation_spec.expand(*batch_size, *observation_spec.shape)  # type: ignore
        if not isinstance(observation_spec, Composite):
            if self.from_pixels:
                observation_spec = Composite(pixels=observation_spec, shape=batch_size)  # type: ignore
            else:
                observation_spec = Composite(observation=observation_spec, shape=batch_size)  # type: ignore

        reward_space = self._reward_space(env)
        if reward_space is not None:
            reward_spec = _gym_to_torchrl_spec_transform(
                reward_space,
                device=self.device,
                categorical_action_encoding=self._categorical_action_encoding,
            )
        else:
            reward_spec = Unbounded(shape=[1], device=self.device).expand(*batch_size, 1)  # type: ignore
        if reward_space is not None:
            reward_spec = reward_spec.expand(*batch_size, *reward_spec.shape)  # type: ignore

        # Flatten one nested level to keep term keys at top-level TensorDict keys.
        flat_entries = {}
        needs_flatten = False
        for key in list(observation_spec.keys()):
            child = observation_spec[key]
            if isinstance(child, Composite):
                needs_flatten = True
                for subkey in child.keys():
                    flat_entries[subkey] = child[subkey]
            else:
                flat_entries[key] = child
        if needs_flatten:
            observation_spec = Composite(flat_entries, shape=observation_spec.shape)

        self.done_spec = self._make_done_spec()  # type: ignore
        self.action_spec = action_spec  # type: ignore
        self.reward_spec = reward_spec  # type: ignore
        self.observation_spec = observation_spec  # type: ignore

    def _output_transform(self, step_outputs_tuple):  # type: ignore
        # IsaacLab will modify the `terminated` and `truncated` tensors
        #  in-place. We clone them here to make sure data doesn't inadvertently get modified.
        # The variable naming follows torchrl's convention here.
        observations, reward, terminated, truncated, info = step_outputs_tuple
        if isinstance(info, dict) and "log" in info:
            self.log_infos.append(info["log"])

        if torch.isnan(reward).any():
            raise ValueError(
                "NaN values found in reward during step. "
                "This is likely due to an error in the environment or the model."
            )

        done = terminated | truncated
        reward = reward.clone().unsqueeze(-1).to(dtype=torch.float32)  # to get to (num_envs, 1)
        observations = _flatten_obs(CloneObsBuf(observations))

        # IsaacLab emits Gymnasium-style keys: final_obs / final_info.
        if isinstance(info, dict) and "final_obs" in info:
            info = {"final_obs": info["final_obs"]}
            return (
                observations,
                reward,
                terminated.clone().to(dtype=torch.bool),
                truncated.clone().to(dtype=torch.bool),
                done.clone().to(dtype=torch.bool),
                info,
            )
        else:
            return (
                observations,
                reward,
                terminated.clone().to(dtype=torch.bool),
                truncated.clone().to(dtype=torch.bool),
                done.clone().to(dtype=torch.bool),
                {},
            )

    def _reset_output_transform(self, reset_data):
        """Transform the output of the reset method."""
        observations, info = reset_data
        return (_flatten_obs(CloneObsBuf(observations)), {})


def CloneObsBuf(
    obs_buf: dict[str, torch.Tensor | dict],
) -> dict[str, torch.Tensor | dict]:
    """Clone the observation buffer.

    Args:
        obs_buf: Dictionary that can contain tensors or nested dictionaries of tensors.

    Returns:
        Cloned dictionary with the same structure as obs_buf.
    """
    cloned = {}
    for k, v in obs_buf.items():
        if isinstance(v, dict):
            # Recursively clone nested dictionaries
            cloned[k] = CloneObsBuf(v)
        elif isinstance(v, torch.Tensor):
            # Clone tensors
            cloned[k] = v.clone()
            assert v.dtype == torch.float32
        else:
            # For other types, just copy the reference
            cloned[k] = v
    return cloned


def CheckObsBufForNaN(obs_buf: dict[str, torch.Tensor | dict], prefix: str = "") -> None:
    """Recursively check nested observation dicts for NaNs."""
    for key, value in obs_buf.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            CheckObsBufForNaN(value, name)
        elif isinstance(value, torch.Tensor) and torch.isnan(value).any():
            first_row = value[0] if value.ndim > 0 else value
            print(f"NaN values found in observation {name} during step. First row: {first_row}")
            raise ValueError(
                f"NaN values found in observation {name} during step. "
                "This is likely due to an error in the environment or the model."
            )


class IsaacLabTerminalObsReader(terminal_obs_reader):
    """A terminal observation reader for IsaacLab environments.

    This reader extracts the terminal observation from the environment's info dictionary.
    It is used to read the terminal observation when the environment is reset."""

    def __init__(self, observation_spec: Composite, backend, name: str = "final"):
        super().__init__(observation_spec=observation_spec, backend=backend, name=name)
        # Provide info specs upfront to avoid dummy rollouts in set_info_dict_reader.
        self._info_spec = Composite({self.name: observation_spec.clone()}, shape=[])

    def __call__(self, info_dict, tensordict):
        # IsaacLab: info_dict["final_obs"] is np.ndarray(num_envs, dtype=object);
        # each entry is None or a nested dict produced by _slice_obs.
        # We flatten exactly like _flatten_obs so keys match the flattened spec.
        backend_key = self.backend_key[self.backend]
        final_obs_arr = info_dict.pop(backend_key, None)
        info_dict.pop(self.backend_info_key[self.backend], None)

        # Let the parent handle remaining info entries and validation.
        super().__call__(info_dict, tensordict)
        if not self._final_validated:
            self.info_spec[self.name] = self._obs_spec.update(self.info_spec)
            self._final_validated = True

        # Flatten per-env dicts once, then scatter into batched zero buffers.
        num_envs = len(final_obs_arr) if final_obs_arr is not None else 0
        flat_per_env: list[dict | None] = [None] * num_envs
        for i in range(num_envs):
            if final_obs_arr[i] is not None:
                flat_per_env[i] = _flatten_obs(final_obs_arr[i])

        for key in self.info_spec[self.name].keys():
            spec = self.info_spec[self.name, key]
            buffer = spec.zero()
            device = buffer.device
            for i in range(num_envs):
                flat = flat_per_env[i]
                if flat is None or key not in flat:
                    continue
                value = flat[key]
                if isinstance(value, torch.Tensor):
                    buffer[i] = value.to(device=device)
                else:
                    buffer[i] = torch.as_tensor(value, device=device)
            tensordict.set((self.name, key), buffer)
        return tensordict


@configclass
class RLOptPPOConfig:
    """Main configuration class for RLOpt PPO."""

    @configclass
    class EnvConfig:
        """Environment configuration for RLOpt PPO."""

        env_name: Any = MISSING
        """Name of the environment."""

        device: str = "cuda:0"
        """Device to run the environment on."""

        num_envs: Any = MISSING
        """Number of environments to simulate."""

    @configclass
    class CollectorConfig:
        """Data collector configuration for RLOpt PPO."""

        num_collectors: int = 1
        """Number of data collectors."""

        frames_per_batch: int = 2048 * 12
        """Number of frames per batch."""

        total_frames: int = 100_000_000
        """Total number of frames to collect."""

        set_truncated: bool = False
        """Whether to set truncated to True when the episode is done."""

    @configclass
    class LoggerConfig:
        """Logger configuration for RLOpt PPO."""

        backend: str = "wandb"
        """Logger backend to use."""

        project_name: str = "IsaacLab"
        """Project name for logging."""

        group_name: str | None = None
        """Group name for logging."""

        exp_name: Any = MISSING
        """Experiment name for logging."""

        test_interval: int = 1_000_000
        """Interval between test evaluations."""

        num_test_episodes: int = 5
        """Number of test episodes to run."""

        video: bool = False
        """Whether to record videos."""

        log_dir: str = "logs"
        """Directory to save logs."""

    @configclass
    class OptimConfig:
        """Optimizer configuration for RLOpt PPO."""

        lr: float = 3e-4
        """Learning rate."""

        weight_decay: float = 0.0
        """Weight decay for optimizer."""

        anneal_lr: bool = True
        """Whether to anneal learning rate."""

        device: str = "cuda:0"
        """Device for optimizer."""

        max_grad_norm: float = 0.5
        """Maximum gradient norm for clipping."""

    @configclass
    class LossConfig:
        """Loss function configuration for RLOpt PPO."""

        gamma: float = 0.99
        """Discount factor."""

        mini_batch_size: Any = MISSING
        """Mini-batch size for training."""

        epochs: int = 4
        """Number of training epochs."""

        gae_lambda: float = 0.95
        """GAE lambda parameter."""

        clip_epsilon: float = 0.2
        """Clipping epsilon for PPO."""

        clip_value: bool = True
        """Whether to clip value function."""

        anneal_clip_epsilon: bool = False
        """Whether to anneal clip epsilon."""

        critic_coeff: float = 1.0
        """Critic coefficient."""

        entropy_coeff: float = 0.005
        """Entropy coefficient."""

        loss_critic_type: str = "l2"
        """Type of critic loss."""

        max_loss: float | None = 50.0
        """Maximum loss value to prevent exploding gradients."""

    @configclass
    class CompileConfig:
        """Compilation configuration for RLOpt PPO."""

        compile: bool = False
        """Whether to compile the model."""

        compile_mode: str = "default"
        """Compilation mode."""

        cudagraphs: bool = False
        """Whether to use CUDA graphs."""

    @configclass
    class PolicyConfig:
        """Policy network configuration for RLOpt PPO."""

        num_cells: list[int] = [512, 256, 128]
        """Number of cells in each layer."""

    @configclass
    class ValueNetConfig:
        """Value network configuration for RLOpt PPO."""

        num_cells: list[int] = [512, 256, 128]
        """Number of cells in each layer."""

    @configclass
    class FeatureExtractorConfig:
        """Feature extractor configuration for RLOpt PPO."""

        num_cells: list[int] = [512, 256, 128]
        """Number of cells in each layer."""

        output_dim: int = 128
        """Output dimension of the feature extractor."""

    @configclass
    class TrainerConfig:
        """Trainer configuration for RLOpt PPO."""

        optim_steps_per_batch: int = 10
        """Number of optimization steps per batch."""

        clip_grad_norm: bool = True
        """Whether to clip gradient norm."""

        clip_norm: float = 0.5
        """Gradient clipping norm."""

        progress_bar: bool = True
        """Whether to show progress bar."""

        save_trainer_interval: int = 10_000
        """Interval for saving trainer."""

        log_interval: int = 1000
        """Interval for logging."""

        save_trainer_file: str | None = None
        """File to save trainer to."""

        frame_skip: int = 1
        """Frame skip for training."""

    env: EnvConfig = EnvConfig()
    """Environment configuration."""

    collector: CollectorConfig = CollectorConfig()
    """Data collector configuration."""

    logger: LoggerConfig = LoggerConfig()
    """Logger configuration."""

    optim: OptimConfig = OptimConfig()
    """Optimizer configuration."""

    loss: LossConfig = LossConfig()
    """Loss function configuration."""

    compile: CompileConfig = CompileConfig()
    """Compilation configuration."""

    policy: PolicyConfig = PolicyConfig()
    """Policy network configuration."""

    value_net: ValueNetConfig = ValueNetConfig()
    """Value network configuration."""

    feature_extractor: FeatureExtractorConfig = FeatureExtractorConfig()
    """Feature extractor configuration."""

    trainer: TrainerConfig = TrainerConfig()
    """Trainer configuration."""

    use_feature_extractor: bool = True
    """Whether to use a feature extractor."""

    device: str = "cuda:0"
    """Device for training."""

    seed: int = 0
    """Random seed."""

    save_interval: int = 500
    """Interval for saving the model."""

    save_path: str = "models"
    """Path to save the model."""

    policy_in_keys: list[str] = ["hidden"]
    """Keys to use for the policy."""

    value_net_in_keys: list[str] = ["hidden"]
    """Keys to use for the value network."""

    total_input_keys: list[str] = ["policy"]
    """Keys to use for the total input."""
