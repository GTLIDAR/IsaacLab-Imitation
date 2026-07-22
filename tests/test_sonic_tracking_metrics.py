"""Unit tests for SONIC-style tracking metric helpers (no Isaac required)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "isaaclab_imitation"
    / "isaaclab_imitation"
    / "metrics"
    / "sonic_tracking.py"
)
_SPEC = importlib.util.spec_from_file_location("sonic_tracking_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
sonic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sonic)


def test_mpjpe_local_zero_when_identical_relative_pose() -> None:
    root = torch.tensor([[1.0, 2.0, 0.9], [0.0, 0.0, 1.0]])
    offsets = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0]],
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0]],
        ]
    )
    robot = root.unsqueeze(1) + offsets
    ref_root = root + torch.tensor([[3.0, -1.0, 0.0], [0.5, 0.5, 0.0]])
    ref = ref_root.unsqueeze(1) + offsets
    err = sonic.mpjpe_local_mm(robot, ref, root, ref_root)
    assert torch.allclose(err, torch.zeros(2), atol=1e-3)


def test_mpjpe_local_scales_to_mm() -> None:
    root = torch.zeros(1, 3)
    robot = torch.tensor([[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]])
    ref = torch.zeros_like(robot)
    err = sonic.mpjpe_local_mm(robot, ref, root, root)
    assert abs(float(err.item()) - 5.0) < 1e-4


def test_root_failure_mask_thresholds() -> None:
    height = torch.tensor([0.0, 0.26, 0.1])
    ori = torch.tensor([0.0, 0.1, 1.1])
    failed = sonic.root_failure_mask(height, ori)
    assert failed.tolist() == [False, True, True]
    assert sonic.SONIC_ROOT_HEIGHT_FAIL_M == 0.25
    assert sonic.SONIC_ROOT_ORI_FAIL_RAD == 1.0


def test_summarize_sonic_episode_success_rate() -> None:
    ever_failed = torch.tensor([False, True, False, True])
    active = torch.tensor([True, True, True, False])
    stats = {
        "mpjpe_l_mm": {"mean": 12.5, "std": 1.0, "count": 3},
        "e_vel_mm_per_frame": {"mean": 3.0, "std": 0.0, "count": 3},
        "e_acc_mm_per_frame2": {"mean": 0.5, "std": 0.0, "count": 3},
    }
    summary = sonic.summarize_sonic_episode(
        ever_failed=ever_failed, active_mask=active, step_metric_stats=stats
    )
    assert abs(summary["success_rate"] - (2.0 / 3.0)) < 1e-6
    assert summary["mpjpe_l_mm"] == 12.5
    assert summary["e_vel"] == 3.0
    assert summary["e_acc"] == 0.5
