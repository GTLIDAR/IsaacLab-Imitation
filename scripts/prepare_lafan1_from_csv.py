#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batch-convert CSV motions to NPZ and emit a LAFAN1 manifest JSON.

Workflow:
1) Find CSV files under a folder.
2) Convert CSV -> NPZ in one batched Isaac Sim session.
3) Optionally record one MP4 per motion using a per-env camera in that same session.
4) Write one manifest JSON consumable by ImitationG1LafanTrackEnvCfg.

Example:
    python scripts/prepare_lafan1_from_csv.py \
      --csv_dir /abs/path/csv_motions \
      --npz_dir /abs/path/npz_motions \
      --manifest_path /abs/path/g1_lafan1_manifest.json \
      --recursive --headless
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def _resolve_default_converter() -> Path:
    return Path(__file__).resolve().with_name("csv_to_npz.py")


def _resolve_default_batch_converter() -> Path:
    return Path(__file__).resolve().with_name("batch_csv_to_npz.py")


def _sanitize_motion_name(path_without_suffix: Path) -> str:
    name = path_without_suffix.as_posix()
    name = re.sub(r"[^A-Za-z0-9_\-/]+", "_", name)
    name = name.replace("/", "__").replace("-", "_")
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "motion"


def _discover_csv_files(csv_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted(path for path in csv_dir.glob(pattern) if path.is_file())


def _build_npz_path(csv_file: Path, csv_root: Path, npz_root: Path) -> Path:
    relative = csv_file.relative_to(csv_root)
    return (npz_root / relative).with_suffix(".npz")


def _run_conversion(
    *,
    csv_file: Path,
    npz_file: Path,
    converter_script: Path,
    python_exe: str,
    input_fps: float,
    output_fps: float,
    frame_range: tuple[int, int] | None,
    headless: bool,
    device: str | None,
    video_output: Path | None,
    overwrite_video: bool,
) -> None:
    cmd = [
        python_exe,
        str(converter_script),
        "-f",
        str(csv_file),
        "--input_fps",
        str(input_fps),
        "--output_name",
        str(npz_file),
        "--output_fps",
        str(output_fps),
    ]
    if frame_range is not None:
        cmd.extend(["--frame_range", str(frame_range[0]), str(frame_range[1])])
    if device is not None:
        cmd.extend(["--device", device])
    if headless:
        cmd.append("--headless")
    if video_output is not None:
        cmd.extend(["--video", "--video_output", str(video_output)])
        if overwrite_video:
            cmd.append("--overwrite_video")

    print(f"[INFO] Converting: {csv_file} -> {npz_file}")
    print(f"[CMD]  {' '.join(shlex.quote(x) for x in cmd)}")
    subprocess.run(cmd, check=True)


def _run_batch_conversion(
    *,
    jobs: list[dict[str, str]],
    batch_converter_script: Path,
    python_exe: str,
    input_fps: float,
    output_fps: float,
    frame_range: tuple[int, int] | None,
    headless: bool,
    device: str | None,
    record_videos: bool,
    overwrite_videos: bool,
    video_width: int,
    video_height: int,
) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="lafan1_batch_jobs_",
        delete=False,
        encoding="utf-8",
    ) as handle:
        json.dump(jobs, handle)
        temp_jobs_path = Path(handle.name)

    cmd = [
        python_exe,
        str(batch_converter_script),
        "--jobs_json",
        str(temp_jobs_path),
        "--input_fps",
        str(input_fps),
        "--output_fps",
        str(output_fps),
    ]
    if frame_range is not None:
        cmd.extend(["--frame_range", str(frame_range[0]), str(frame_range[1])])
    if device is not None:
        cmd.extend(["--device", device])
    if headless:
        cmd.append("--headless")
    if record_videos:
        cmd.extend(
            [
                "--video",
                "--video_width",
                str(video_width),
                "--video_height",
                str(video_height),
            ]
        )
        if overwrite_videos:
            cmd.append("--overwrite_video")

    print(f"[INFO] Batch converting {len(jobs)} motion(s) in one Isaac Sim session.")
    print(f"[CMD]  {' '.join(shlex.quote(x) for x in cmd)}")
    try:
        subprocess.run(cmd, check=True)
    finally:
        temp_jobs_path.unlink(missing_ok=True)


def _default_video_dir(npz_dir: Path) -> Path:
    if npz_dir.parent != npz_dir:
        return npz_dir.parent / "videos"
    return npz_dir / "videos"


def _build_video_output(csv_file: Path, csv_root: Path, video_root: Path) -> Path:
    relative = csv_file.relative_to(csv_root)
    return (video_root / relative).with_suffix(".mp4")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch convert CSV motions to NPZ and generate manifest JSON."
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        required=True,
        help="Input folder containing CSV motion files.",
    )
    parser.add_argument(
        "--npz_dir", type=str, required=True, help="Output folder for NPZ files."
    )
    parser.add_argument(
        "--manifest_path", type=str, required=True, help="Output manifest JSON path."
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Recursively scan csv_dir.",
    )
    parser.add_argument(
        "--converter_script",
        type=str,
        default=str(_resolve_default_converter()),
        help="Path to csv_to_npz.py converter script.",
    )
    parser.add_argument(
        "--batch_converter_script",
        type=str,
        default=str(_resolve_default_batch_converter()),
        help="Path to batch_csv_to_npz.py used for one-shot batched conversion.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to run converter.",
    )
    parser.add_argument(
        "--input_fps",
        type=float,
        default=60.0,
        help="Input CSV FPS passed to converter.",
    )
    parser.add_argument(
        "--output_fps",
        type=float,
        default=50.0,
        help="Output NPZ FPS passed to converter.",
    )
    parser.add_argument(
        "--frame_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Optional frame range passed to converter (1-indexed inclusive).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Pass --headless to converter.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional sim device for converter (e.g. cuda:0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing NPZ files.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="Skip conversion when target NPZ already exists.",
    )
    parser.add_argument(
        "--assume_npz_exists",
        action="store_true",
        default=False,
        help="Do not run converter; require NPZ files to already exist in npz_dir.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="lafan1",
        help="Dataset name written to manifest.",
    )
    parser.add_argument(
        "--motion_name_prefix",
        type=str,
        default="",
        help="Optional prefix for generated motion names in manifest.",
    )
    parser.add_argument(
        "--record_videos",
        action="store_true",
        default=False,
        help="Record one MP4 per motion during the batched conversion session.",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Output folder for recorded videos (defaults to a sibling 'videos' folder next to npz_dir).",
    )
    parser.add_argument(
        "--overwrite_videos",
        action="store_true",
        default=False,
        help="Overwrite existing videos with the same output path.",
    )
    parser.add_argument(
        "--video_width",
        type=int,
        default=640,
        help="Per-env video width in pixels when --record_videos is enabled.",
    )
    parser.add_argument(
        "--video_height",
        type=int,
        default=480,
        help="Per-env video height in pixels when --record_videos is enabled.",
    )
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir).expanduser().resolve()
    npz_dir = Path(args.npz_dir).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve()
    converter_script = Path(args.converter_script).expanduser().resolve()
    batch_converter_script = Path(args.batch_converter_script).expanduser().resolve()

    if not csv_dir.is_dir():
        raise NotADirectoryError(f"csv_dir does not exist: {csv_dir}")
    if not args.assume_npz_exists and not converter_script.is_file():
        raise FileNotFoundError(f"converter_script not found: {converter_script}")
    if not args.assume_npz_exists and not batch_converter_script.is_file():
        raise FileNotFoundError(
            f"batch_converter_script not found: {batch_converter_script}"
        )
    if args.assume_npz_exists and args.record_videos:
        raise ValueError(
            "--record_videos requires conversion, so it cannot be combined with --assume_npz_exists."
        )

    csv_files = _discover_csv_files(csv_dir, recursive=args.recursive)
    if len(csv_files) == 0:
        raise RuntimeError(f"No CSV files found under: {csv_dir}")

    npz_dir.mkdir(parents=True, exist_ok=True)
    video_dir = (
        Path(args.video_dir).expanduser().resolve()
        if args.video_dir is not None
        else _default_video_dir(npz_dir)
    )

    batch_jobs: list[dict[str, str]] = []
    frame_range = tuple(args.frame_range) if args.frame_range is not None else None

    for csv_file in csv_files:
        npz_file = _build_npz_path(
            csv_file=csv_file, csv_root=csv_dir, npz_root=npz_dir
        )
        npz_file.parent.mkdir(parents=True, exist_ok=True)

        if args.assume_npz_exists:
            if not npz_file.is_file():
                raise FileNotFoundError(
                    f"assume_npz_exists=True but target missing: {npz_file}. "
                    "Either disable --assume_npz_exists or pre-generate all NPZ files."
                )
        else:
            if npz_file.exists() and not args.overwrite:
                if args.skip_existing:
                    print(f"[INFO] Skipping existing NPZ: {npz_file}")
                else:
                    raise FileExistsError(
                        f"Target NPZ exists: {npz_file}. Use --overwrite or --skip_existing."
                    )
            else:
                job: dict[str, str] = {
                    "input_file": str(csv_file),
                    "output_name": str(npz_file),
                }
                if args.record_videos:
                    job["video_output"] = str(
                        _build_video_output(
                            csv_file=csv_file, csv_root=csv_dir, video_root=video_dir
                        )
                    )
                batch_jobs.append(job)

    if len(batch_jobs) > 0:
        _run_batch_conversion(
            jobs=batch_jobs,
            batch_converter_script=batch_converter_script,
            python_exe=args.python,
            input_fps=float(args.input_fps),
            output_fps=float(args.output_fps),
            frame_range=frame_range,
            headless=bool(args.headless),
            device=args.device,
            record_videos=bool(args.record_videos),
            overwrite_videos=bool(args.overwrite_videos),
            video_width=int(args.video_width),
            video_height=int(args.video_height),
        )

    manifest_entries: list[dict[str, object]] = []

    for csv_file in csv_files:
        npz_file = _build_npz_path(
            csv_file=csv_file, csv_root=csv_dir, npz_root=npz_dir
        )
        if not npz_file.is_file():
            raise FileNotFoundError(f"Expected converted NPZ not found: {npz_file}")

        relative_no_suffix = npz_file.relative_to(npz_dir).with_suffix("")
        motion_name = _sanitize_motion_name(relative_no_suffix)
        if args.motion_name_prefix:
            motion_name = f"{args.motion_name_prefix}{motion_name}"

        entry: dict[str, object] = {
            "name": motion_name,
            "path": str(npz_file),
            # Source is NPZ generated at output_fps.
            "input_fps": float(args.output_fps),
        }
        if args.frame_range is not None:
            entry["frame_range"] = [int(args.frame_range[0]), int(args.frame_range[1])]
        manifest_entries.append(entry)

    manifest = {
        "dataset_name": args.dataset_name,
        "dataset": {
            "trajectories": {
                "lafan1_csv": manifest_entries,
            }
        },
        "metadata": {
            "csv_dir": str(csv_dir),
            "npz_dir": str(npz_dir),
            "num_motions": len(manifest_entries),
            "input_fps": float(args.input_fps),
            "output_fps": float(args.output_fps),
            "converter_script": str(converter_script),
            "assume_npz_exists": bool(args.assume_npz_exists),
            "record_videos": bool(args.record_videos),
            "video_dir": str(video_dir) if args.record_videos else None,
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"[INFO] Wrote manifest: {manifest_path}")
    print(f"[INFO] Motion count: {len(manifest_entries)}")
    if args.record_videos:
        print(f"[INFO] Video root: {video_dir}")

    cfg_snippet = (
        "\n"
        "# Paste into ImitationG1LafanTrackEnvCfg\n"
        f'lafan1_manifest_path = "{manifest_path}"\n'
        "lafan1_refresh_zarr_dataset = True\n"
        'lafan1_reset_schedule = "random"\n'
    )
    print(cfg_snippet)


if __name__ == "__main__":
    main()
