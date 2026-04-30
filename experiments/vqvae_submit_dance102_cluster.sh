#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

RLOPT_VQVAE_DIR="${RLOPT_VQVAE_DIR:-${WORKSPACE_ROOT}/RLOpt-vqvae}"

# Uses the VQVAE config defaults for total training horizon and checkpoint cadence.
CLUSTER_EXTRA_SYNC_SPECS="${CLUSTER_EXTRA_SYNC_SPECS:-${RLOPT_VQVAE_DIR}:RLOpt}" \
    "${REPO_ROOT}/docker/cluster/cluster_interface.sh" job \
    --task Isaac-Imitation-G1-Latent-VQVAE-v0 \
    --num_envs 4096 \
    --headless \
    --video \
    --algo IPMD \
    --kit_args=--/app/extensions/fsWatcherEnabled=false \
    env.lafan1_manifest_path=./data/unitree/manifests/g1_unitree_dance102_manifest.json \
    env.dataset_path=/tmp/iltools_g1_lafan1_tracking_g1_unitree_dance102_manifest_6d26546fd54a \
    env.refresh_zarr_dataset=False \
    agent.logger.exp_name=vqvae_dance102_4096_default
