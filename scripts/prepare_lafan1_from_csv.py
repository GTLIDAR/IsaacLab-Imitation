#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batch-convert CSV motions to NPZ and emit a LAFAN1 manifest JSON.

Workflow:
1) Find CSV files under a folder.
2) Convert each CSV -> NPZ using unitree_rl_lab/scripts/mimic/csv_to_npz.py.
3) Write one manifest JSON consumable by ImitationG1LafanTrackEnvCfg.

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
from pathlib import Path


def _resolve_default_converter() -> Path:
    this_file = Path(__file__).resolve()
    workspace_root = this_file.parents[2]
    return workspace_root / "unitree_rl_lab" / "scripts" / "mimic" / "csv_to_npz.py"


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

    print(f"[INFO] Converting: {csv_file} -> {npz_file}")
    print(f"[CMD]  {' '.join(shlex.quote(x) for x in cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch convert CSV motions to NPZ and generate manifest JSON.")
    parser.add_argument("--csv_dir", type=str, required=True, help="Input folder containing CSV motion files.")
    parser.add_argument("--npz_dir", type=str, required=True, help="Output folder for NPZ files.")
    parser.add_argument("--manifest_path", type=str, required=True, help="Output manifest JSON path.")
    parser.add_argument("--recursive", action="store_true", default=False, help="Recursively scan csv_dir.")
    parser.add_argument(
        "--converter_script",
        type=str,
        default=str(_resolve_default_converter()),
        help="Path to csv_to_npz.py converter script.",
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used to run converter.")
    parser.add_argument("--input_fps", type=float, default=60.0, help="Input CSV FPS passed to converter.")
    parser.add_argument("--output_fps", type=float, default=50.0, help="Output NPZ FPS passed to converter.")
    parser.add_argument(
        "--frame_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Optional frame range passed to converter (1-indexed inclusive).",
    )
    parser.add_argument("--headless", action="store_true", default=False, help="Pass --headless to converter.")
    parser.add_argument("--device", type=str, default=None, help="Optional sim device for converter (e.g. cuda:0).")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing NPZ files.")
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
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir).expanduser().resolve()
    npz_dir = Path(args.npz_dir).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve()
    converter_script = Path(args.converter_script).expanduser().resolve()

    if not csv_dir.is_dir():
        raise NotADirectoryError(f"csv_dir does not exist: {csv_dir}")
    if not args.assume_npz_exists and not converter_script.is_file():
        raise FileNotFoundError(f"converter_script not found: {converter_script}")

    csv_files = _discover_csv_files(csv_dir, recursive=args.recursive)
    if len(csv_files) == 0:
        raise RuntimeError(f"No CSV files found under: {csv_dir}")

    npz_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, object]] = []

    for csv_file in csv_files:
        npz_file = _build_npz_path(csv_file=csv_file, csv_root=csv_dir, npz_root=npz_dir)
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
                _run_conversion(
                    csv_file=csv_file,
                    npz_file=npz_file,
                    converter_script=converter_script,
                    python_exe=args.python,
                    input_fps=float(args.input_fps),
                    output_fps=float(args.output_fps),
                    frame_range=tuple(args.frame_range) if args.frame_range is not None else None,
                    headless=bool(args.headless),
                    device=args.device,
                )

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
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"[INFO] Wrote manifest: {manifest_path}")
    print(f"[INFO] Motion count: {len(manifest_entries)}")

    cfg_snippet = (
        "\n"
        "# Paste into ImitationG1LafanTrackEnvCfg\n"
        f"lafan1_manifest_path = \"{manifest_path}\"\n"
        "lafan1_refresh_zarr_dataset = True\n"
        "lafan1_reset_schedule = \"random\"\n"
    )
    print(cfg_snippet)


if __name__ == "__main__":
    main()
