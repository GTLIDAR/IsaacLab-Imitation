# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare iltools LAFAN1 loader outputs against the original dance_102 motion arrays.

This is a loader-only check (no simulator). It builds a zarr dataset via
`Lafan1CsvLoader` and compares the exported arrays to the source npz used by
`unitree_rl_lab` dance tasks.

Example:
    python scripts/compare_dance102_loader.py \
        --motion_path /abs/path/G1_Take_102.bvh_60hz.npz
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def _append_workspace_sources() -> None:
    this_file = Path(__file__).resolve()
    workspace_root = this_file.parents[2]
    candidate_paths = [
        workspace_root / "ImitationLearningTools",
        workspace_root / "IsaacLab-Imitation" / "source" / "isaaclab_imitation",
    ]
    for candidate in candidate_paths:
        if candidate.is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.append(candidate_str)


_append_workspace_sources()


def _default_motion_path() -> Path:
    this_file = Path(__file__).resolve()
    workspace_root = this_file.parents[2]
    return (
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


def _read_source_fps(motion_path: Path, fallback_fps: float) -> float:
    with np.load(motion_path) as npz_data:
        if "fps" in npz_data.files:
            return float(np.asarray(npz_data["fps"]).reshape(-1)[0])
    return float(fallback_fps)


def _compute_error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    error = candidate - reference
    abs_error = np.abs(error)
    return {
        "mean_abs": float(abs_error.mean()),
        "max_abs": float(abs_error.max()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare iltools LAFAN1 loader output with source dance npz arrays.")
    parser.add_argument(
        "--motion_path",
        type=str,
        default=str(_default_motion_path()),
        help="Path to dance motion npz file.",
    )
    parser.add_argument("--dataset_name", type=str, default="lafan1")
    parser.add_argument("--motion_name", type=str, default="dance_102")
    parser.add_argument("--trajectory_name", type=str, default="trajectory_0")
    parser.add_argument("--input_fps", type=float, default=60.0)
    parser.add_argument(
        "--control_freq",
        type=float,
        default=None,
        help="Loader output frequency. Defaults to source npz fps.",
    )
    parser.add_argument("--sim_dt", type=float, default=0.005)
    parser.add_argument("--decimation", type=int, default=4)
    parser.add_argument(
        "--zarr_path",
        type=str,
        default=None,
        help="Temporary zarr cache path. Defaults to /tmp/iltools_loader_compare_<motion_stem>.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1.0e-6,
        help="Pass/fail threshold on per-key max abs error.",
    )
    parser.add_argument("--keep_zarr", action="store_true", default=False)
    parser.add_argument("--report_path", type=str, default=None)
    args = parser.parse_args()

    try:
        import zarr
        from iltools.datasets.lafan1.loader import Lafan1CsvLoader
    except Exception as err:
        raise ImportError(
            "Missing dependencies for loader comparison. "
            "Ensure iltools and zarr are available in the active environment."
        ) from err

    motion_path = Path(args.motion_path).expanduser().resolve()
    if not motion_path.is_file():
        raise FileNotFoundError(
            "Motion npz not found. Generate it with:\n"
            "python /Users/fwu/Developer/SkillLearning/unitree_rl_lab/scripts/mimic/csv_to_npz.py "
            "-f /Users/fwu/Developer/SkillLearning/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/dance_102/G1_Take_102.bvh_60hz.csv"
        )
    if motion_path.suffix.lower() != ".npz":
        raise ValueError(f"Expected an npz source, got: {motion_path}")

    source_fps = _read_source_fps(motion_path, fallback_fps=args.input_fps)
    control_freq = float(args.control_freq) if args.control_freq is not None else float(source_fps)

    if args.zarr_path is None:
        zarr_path = Path(f"/tmp/iltools_loader_compare_{motion_path.stem}.zarr")
    else:
        zarr_path = Path(args.zarr_path).expanduser().resolve()

    if zarr_path.exists():
        if zarr_path.is_dir():
            shutil.rmtree(zarr_path)
        else:
            zarr_path.unlink()

    loader_cfg = {
        "dataset_name": args.dataset_name,
        "dataset": {
            "trajectories": {
                "lafan1_csv": [
                    {
                        "name": args.motion_name,
                        "path": str(motion_path),
                        "input_fps": float(args.input_fps),
                    }
                ]
            }
        },
        "control_freq": control_freq,
        "sim": {"dt": float(args.sim_dt)},
        "decimation": int(args.decimation),
    }

    _ = Lafan1CsvLoader(cfg=loader_cfg, build_zarr_dataset=True, zarr_path=str(zarr_path))

    zarr_root = zarr.open(str(zarr_path), mode="r")
    traj_group = zarr_root[args.dataset_name][args.motion_name][args.trajectory_name]

    with np.load(motion_path) as npz_data:
        source_arrays = {key: np.asarray(npz_data[key]) for key in npz_data.files}

    keys_to_compare = [
        "qpos",
        "qvel",
        "root_pos",
        "root_quat",
        "root_lin_vel",
        "root_ang_vel",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ]

    key_reports: dict[str, dict] = {}
    all_pass = True
    for key in keys_to_compare:
        if key not in traj_group:
            continue
        if key not in source_arrays:
            continue

        candidate = np.asarray(traj_group[key][:])
        reference = np.asarray(source_arrays[key])

        compare_len = min(candidate.shape[0], reference.shape[0])
        candidate = candidate[:compare_len]
        reference = reference[:compare_len]

        if candidate.shape != reference.shape:
            key_reports[key] = {
                "compared": False,
                "reason": f"shape mismatch after time alignment: {candidate.shape} vs {reference.shape}",
            }
            all_pass = False
            continue

        metrics = _compute_error_metrics(reference=reference, candidate=candidate)
        key_pass = metrics["max_abs"] <= args.tol
        all_pass = all_pass and key_pass
        key_reports[key] = {
            "compared": True,
            "shape": list(candidate.shape),
            "mean_abs": metrics["mean_abs"],
            "rmse": metrics["rmse"],
            "max_abs": metrics["max_abs"],
            "pass": key_pass,
        }

    report = {
        "motion_path": str(motion_path),
        "zarr_path": str(zarr_path),
        "dataset_name": args.dataset_name,
        "motion_name": args.motion_name,
        "trajectory_name": args.trajectory_name,
        "source_fps": source_fps,
        "control_freq": control_freq,
        "resampled": bool(abs(control_freq - source_fps) > 1.0e-8),
        "tol": float(args.tol),
        "pass": bool(all_pass),
        "keys": key_reports,
    }

    print(json.dumps(report, indent=2, sort_keys=True))

    if args.report_path is not None:
        report_path = Path(args.report_path).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[INFO] Wrote report to: {report_path}")

    if not args.keep_zarr and zarr_path.exists():
        shutil.rmtree(zarr_path)


if __name__ == "__main__":
    main()
