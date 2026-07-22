"""Unit tests for LAFAN1 ablation mean±std aggregation."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AGG_PATH = REPO_ROOT / "experiments" / "lafan1_ablation" / "aggregate_results.py"

sys.path.insert(0, str(AGG_PATH.parent))
import aggregate_results as agg  # noqa: E402


def _write_summary(
    path: Path,
    *,
    interface: str,
    window: int,
    rank: int,
    trajectory: str,
    setting: str,
    success_rate: float,
    mpjpe_l_mm: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "interface": interface,
        "table": "table_a",
        "window": window,
        "rank": rank,
        "trajectory": trajectory,
        "setting": setting,
        "planner_unit": "trajectory",
        "status": "ok",
        "success_rate": success_rate,
        "mpjpe_l_mm": mpjpe_l_mm,
        "aggregate": {
            "success_rate": success_rate,
            "mpjpe_l_mm": mpjpe_l_mm,
            "e_vel": 1.0,
            "e_acc": 2.0,
            "survival_steps_mean": 100.0,
        },
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_mean_over_trajectories(tmp_path: Path) -> None:
    root = tmp_path / "table_a"
    for rank, succ, mpjpe in ((0, 0.5, 10.0), (1, 1.0, 20.0)):
        _write_summary(
            root
            / "W10"
            / "latent_cont"
            / "trajectories"
            / f"rank_{rank}_m"
            / "eval"
            / "finetuned"
            / "summary.json",
            interface="latent_cont",
            window=10,
            rank=rank,
            trajectory=f"m{rank}",
            setting="finetuned",
            success_rate=succ,
            mpjpe_l_mm=mpjpe,
        )
        _write_summary(
            root
            / "W10"
            / "latent_cont"
            / "trajectories"
            / f"rank_{rank}_m"
            / "eval"
            / "oracle"
            / "summary.json",
            interface="latent_cont",
            window=10,
            rank=rank,
            trajectory=f"m{rank}",
            setting="oracle",
            success_rate=1.0,
            mpjpe_l_mm=mpjpe / 2.0,
        )

    out = tmp_path / "out"
    paths = agg.aggregate_table(root, out, "table_a")
    mean_csv = paths["mean_by_setting"]
    assert mean_csv.is_file()

    with mean_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    finetuned = next(row for row in rows if row["setting"] == "finetuned")
    assert int(finetuned["n_trajectories"]) == 2
    assert abs(float(finetuned["success_rate_mean"]) - 0.75) < 1e-9
    assert abs(float(finetuned["success_rate_std"]) - 0.25) < 1e-9
    assert abs(float(finetuned["mpjpe_l_mm_mean"]) - 15.0) < 1e-9

    mean_wide = paths["mean_wide"]
    with mean_wide.open(encoding="utf-8", newline="") as handle:
        wide_rows = list(csv.DictReader(handle))
    assert len(wide_rows) == 1
    assert abs(float(wide_rows[0]["finetuned_success_rate_mean"]) - 0.75) < 1e-9
    assert abs(float(wide_rows[0]["oracle_success_rate_mean"]) - 1.0) < 1e-9
    assert abs(float(wide_rows[0]["finetuned_succ_oracle_ratio_mean"]) - 0.75) < 1e-9
