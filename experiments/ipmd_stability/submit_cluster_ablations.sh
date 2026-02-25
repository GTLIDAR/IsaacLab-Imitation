#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TASK="${TASK:-Isaac-Imitation-G1-v0}"
NUM_ENVS="${NUM_ENVS:-2048}"
SEEDS_STR="${SEEDS:-2024 2025 2026}"
COMBOS_STR="${COMBOS:-A B C D E}"
# Optional profile name recognized by ./docker/.env.<profile> (for example: base).
CLUSTER_PROFILE="${CLUSTER_PROFILE:-}"

read -r -a SEED_LIST <<< "$SEEDS_STR"
read -r -a COMBO_LIST <<< "$COMBOS_STR"

A_OVERRIDES=(
    "agent.collector.init_random_frames=0"
    "agent.ipmd.reward_lr=1e-3"
    "agent.ipmd.reward_update_interval=1"
    "agent.ipmd.reward_margin=0.0"
    "agent.ipmd.reward_consistency_coeff=0.0"
    "agent.ipmd.use_reward_target_network=false"
    "agent.ipmd.use_reward_target_for_ppo=false"
    "agent.ipmd.normalize_reward_input=false"
    "agent.ipmd.reward_grad_penalty_coeff=0.0"
    "agent.ipmd.reward_logit_reg_coeff=0.0"
    "agent.ipmd.reward_param_weight_decay_coeff=0.0"
    "agent.ipmd.reward_replay_size=0"
    "agent.ipmd.reward_replay_ratio=0.0"
    "agent.ipmd.reward_replay_keep_prob=1.0"
    "agent.ipmd.reward_mix_alpha_start=0.0"
    "agent.ipmd.reward_mix_alpha_end=1.0"
    "agent.ipmd.reward_mix_anneal_updates=20000"
    "agent.ipmd.reward_mix_gate_estimated_std_min=0.05"
    "agent.ipmd.reward_mix_alpha_when_unstable=0.15"
    "agent.ipmd.reward_mix_gate_after_updates=500"
    "agent.ipmd.entropy_coeff_start=0.005"
    "agent.ipmd.entropy_coeff_end=0.005"
    "agent.ipmd.entropy_schedule_updates=0"
    "agent.ipmd.bc_loss_coeff=0.0"
    "agent.ipmd.bc_warmup_updates=0"
    "agent.ipmd.bc_final_coeff=0.0"
)

B_EXTRA=(
    "agent.ipmd.reward_lr=2e-4"
    "agent.ipmd.reward_update_interval=2"
)

C_EXTRA=(
    "agent.ipmd.normalize_reward_input=true"
    "agent.ipmd.reward_grad_penalty_coeff=0.2"
    "agent.ipmd.reward_logit_reg_coeff=0.02"
    "agent.ipmd.reward_param_weight_decay_coeff=1e-5"
    "agent.ipmd.reward_replay_size=200000"
    "agent.ipmd.reward_replay_ratio=0.5"
    "agent.ipmd.reward_replay_keep_prob=0.25"
)

D_EXTRA=(
    "agent.ipmd.use_reward_target_network=true"
    "agent.ipmd.use_reward_target_for_ppo=true"
    "agent.ipmd.reward_target_polyak=0.995"
    "agent.ipmd.reward_target_update_interval=1"
    "agent.ipmd.reward_margin=0.05"
    "agent.ipmd.reward_consistency_coeff=0.2"
)

E_EXTRA=(
    "agent.collector.init_random_frames=49152"
    "agent.ipmd.entropy_coeff_start=0.02"
    "agent.ipmd.entropy_coeff_end=0.005"
    "agent.ipmd.entropy_schedule_updates=15000"
    "agent.ipmd.bc_loss_coeff=0.02"
    "agent.ipmd.bc_warmup_updates=20000"
    "agent.ipmd.bc_final_coeff=0.0"
)

get_combo_overrides() {
    local combo="$1"
    OVERRIDES=("${A_OVERRIDES[@]}")
    case "$combo" in
        A)
            ;;
        B)
            OVERRIDES+=("${B_EXTRA[@]}")
            ;;
        C)
            OVERRIDES+=("${B_EXTRA[@]}" "${C_EXTRA[@]}")
            ;;
        D)
            OVERRIDES+=("${B_EXTRA[@]}" "${C_EXTRA[@]}" "${D_EXTRA[@]}")
            ;;
        E)
            OVERRIDES+=("${B_EXTRA[@]}" "${C_EXTRA[@]}" "${D_EXTRA[@]}" "${E_EXTRA[@]}")
            ;;
        *)
            echo "[ERROR] Unknown combo '$combo'. Supported combos: A B C D E"
            exit 1
            ;;
    esac
}

submit_one() {
    local combo="$1"
    local seed="$2"
    get_combo_overrides "$combo"

    local run_name="${combo}_seed${seed}"
    local cmd=(./docker/cluster/cluster_interface.sh job)

    # Follow cluster_interface usage: job [<profile>] [<job_args>...]
    if [[ -n "$CLUSTER_PROFILE" ]]; then
        cmd+=("$CLUSTER_PROFILE")
    fi

    cmd+=(
        --task "$TASK"
        --num_envs "$NUM_ENVS"
        --headless
        --algo ipmd
        "agent.seed=${seed}"
        "agent.logger.exp_name=${run_name}"
        "${OVERRIDES[@]}"
    )

    printf "\n[%s] Submitting %s\n" "$(date '+%F %T')" "$run_name"
    printf "[CMD] "
    printf "%q " "${cmd[@]}"
    printf "\n"
    "${cmd[@]}"
}

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Task=${TASK}, num_envs=${NUM_ENVS}, seeds='${SEEDS_STR}', combos='${COMBOS_STR}', profile='${CLUSTER_PROFILE:-<default>}'"

for combo in "${COMBO_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
        submit_one "$combo" "$seed"
    done
done

echo
echo "[INFO] Submitted all requested jobs."
