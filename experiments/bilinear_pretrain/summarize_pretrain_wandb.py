#!/usr/bin/env python3
"""Summarize offline bilinear SR pretraining curves from W&B."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import wandb


DEFAULT_PROJECT = "G1-Imitation-RLOpt-Pretrain"
DEFAULT_KEYS = (
    "offline/sr/loss/dynamics_loss",
    "offline/sr/sample/recon_mse",
    "offline/sr/sample/recon_l1",
    "offline/sr/info/obs_norm_count",
    "offline/sr/history_buffer_size",
    "offline/policy_bc/bc_nll",
    "offline/policy_bc/bc_log_prob_mean",
    "offline/policy_bc/bc_actor_grad_norm",
)


def _resolve_entity(api: wandb.Api, entity: str | None) -> str:
    if entity:
        return entity
    viewer = api.viewer
    if viewer is None or viewer.entity is None:
        msg = "Pass --entity or configure W&B credentials with a default entity."
        raise RuntimeError(msg)
    return str(viewer.entity)


def _format_float(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def _summarize_run(run: wandb.apis.public.Run, keys: Sequence[str], samples: int) -> None:
    loss_key = "offline/sr/loss/dynamics_loss"
    history = run.history(keys=list(keys), samples=samples, pandas=True)
    if history.empty or loss_key not in history:
        print(f"{run.id} | {run.state:8s} | no offline pretrain rows | {run.name}")
        return

    rows = history.dropna(subset=[loss_key])
    if rows.empty:
        print(f"{run.id} | {run.state:8s} | no offline pretrain rows | {run.name}")
        return

    first = rows.iloc[0]
    last = rows.iloc[-1]
    min_loss = rows[loss_key].min()
    step = int(last["_step"]) if "_step" in rows else -1
    print(
        f"{run.id} | {run.state:8s} | points={len(rows):02d} | "
        f"step={step:>5d} | "
        f"loss { _format_float(first[loss_key]) } -> { _format_float(last[loss_key]) } "
        f"(min { _format_float(min_loss) }) | "
        f"mse={_format_float(last.get('offline/sr/sample/recon_mse'))} | "
        f"l1={_format_float(last.get('offline/sr/sample/recon_l1'))} | "
        f"bc_nll={_format_float(last.get('offline/policy_bc/bc_nll'))} | "
        f"history={_format_float(last.get('offline/sr/history_buffer_size'))} | "
        f"{run.name}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=None)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--group", default=None)
    parser.add_argument("--name-contains", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = wandb.Api(timeout=60)
    entity = _resolve_entity(api, args.entity)
    path = f"{entity}/{args.project}"
    runs = list(api.runs(path, order="-created_at", per_page=args.limit))
    if args.group:
        runs = [run for run in runs if run.group == args.group]
    if args.name_contains:
        runs = [run for run in runs if args.name_contains in run.name]
    if not runs:
        msg = f"No runs matched path={path!r}, group={args.group!r}, name_contains={args.name_contains!r}."
        raise RuntimeError(msg)

    keys = ("_step", *DEFAULT_KEYS)
    print(f"path={path} group={args.group or '*'} name_contains={args.name_contains or '*'}")
    for run in runs[: args.limit]:
        _summarize_run(run, keys, args.samples)


if __name__ == "__main__":
    main()
