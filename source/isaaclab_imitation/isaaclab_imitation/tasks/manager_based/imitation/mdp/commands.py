from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class ImitationEnvCommand(CommandTerm):
    """Dynamic command term backed by ImitationRLEnv state."""

    cfg: "ImitationEnvCommandCfg"

    @property
    def command(self) -> torch.Tensor:
        getter = getattr(self._env, self.cfg.getter_name)
        return getter()

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        return {}

    def compute(self, dt: float) -> None:
        pass

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        pass

    def _update_command(self) -> None:
        pass


@configclass
class ImitationEnvCommandCfg(CommandTermCfg):
    """Command term that forwards to an env getter."""

    class_type: type = ImitationEnvCommand
    getter_name: str = "get_policy_command"
    resampling_time_range: tuple[float, float] = (1.0, 1.0)
