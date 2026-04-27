#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TASK="${TASK:-Isaac-Imitation-G1-v0}"
NUM_ENVS="${NUM_ENVS:-128}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
SEEDS_STR="${SEEDS:-2024}"
COMBOS_STR="${COMBOS:-A B C D E F}"

if ! command -v timeout >/dev/null 2>&1; then
    echo "[ERROR] 'timeout' command is required but not found."
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] 'conda' command is required but not found."
    exit 1
fi

read -r -a SEED_LIST <<< "$SEEDS_STR"
read -r -a COMBO_LIST <<< "$COMBOS_STR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SCRIPT_DIR/logs/local_debug_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
    scripts/rlopt/train.py
    --task "$TASK"
    --num_envs "$NUM_ENVS"
    --headless
    --algo ipmd
)

A_OVERRIDES=(
    "agent.collector.init_random_frames=0"
    "agent.ipmd.use_estimated_rewards_for_ppo=false"
    "agent.ipmd.reward_lr=3e-4"
    "agent.ipmd.reward_update_interval=1"
    "agent.ipmd.reward_updates_per_policy_update=1"
    "agent.ipmd.reward_update_warmup_updates=0"
    "agent.ipmd.reward_grad_penalty_coeff=0.0"
    "agent.ipmd.reward_logit_reg_coeff=0.0"
    "agent.ipmd.reward_param_weight_decay_coeff=0.0"
    "agent.ipmd.reward_l2_coeff=0.0"
    "agent.ipmd.env_reward_weight=1.0"
    "agent.ipmd.est_reward_weight=0.0"
    "agent.ipmd.bc_coef=0.0"
)

B_EXTRA=(
    "agent.ipmd.reward_update_interval=4"
)

C_EXTRA=(
    "agent.ipmd.reward_update_interval=8"
)

D_EXTRA=(
    "agent.ipmd.reward_update_interval=4"
    "agent.ipmd.reward_grad_penalty_coeff=0.2"
    "agent.ipmd.reward_logit_reg_coeff=0.01"
    "agent.ipmd.reward_param_weight_decay_coeff=1e-5"
)

E_EXTRA=(
    "agent.ipmd.reward_updates_per_policy_update=2"
    "agent.ipmd.reward_update_warmup_updates=0"
)

F_EXTRA=(
    "agent.ipmd.use_estimated_rewards_for_ppo=true"
    "agent.ipmd.est_reward_weight=0.1"
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
            OVERRIDES+=("${D_EXTRA[@]}")
            ;;
        E)
            OVERRIDES+=("${D_EXTRA[@]}" "${E_EXTRA[@]}")
            ;;
        F)
            OVERRIDES+=("${D_EXTRA[@]}" "${E_EXTRA[@]}" "${F_EXTRA[@]}")
            ;;
        *)
            echo "[ERROR] Unknown combo '$combo'. Supported combos: A B C D E F"
            exit 1
            ;;
    esac
}

force_kill_python_processes() {
    # Force cleanup after each run: timeout may leave child processes alive.
    pkill -9 python >/dev/null 2>&1 || true
}

run_one() {
    local combo="$1"
    local seed="$2"
    get_combo_overrides "$combo"

    local run_name="${combo}_seed${seed}"
    local log_file="$LOG_DIR/${run_name}.log"

    local cmd=(
        conda run -n SkillLearning python
        "${COMMON_ARGS[@]}"
        "agent.seed=${seed}"
        "agent.logger.exp_name=${run_name}"
        "${OVERRIDES[@]}"
    )

    printf "\n[%s] Running %s (timeout=%ss)\n" "$(date '+%F %T')" "$run_name" "$TIMEOUT_SECONDS"
    printf "[CMD] "
    printf "%q " "${cmd[@]}"
    printf "\n"

    set +e
    timeout --signal=TERM --kill-after=20s --preserve-status "${TIMEOUT_SECONDS}" "${cmd[@]}" >"$log_file" 2>&1
    local rc=$?
    set -e

    case "$rc" in
        0)
            echo "[DONE] ${run_name} completed (log: $log_file)"
            ;;
        124|137|143)
            echo "[TIMEOUT] ${run_name} hit timeout as expected (log: $log_file)"
            ;;
        *)
            echo "[FAIL] ${run_name} exited with code ${rc} (log: $log_file)"
            ;;
    esac

    force_kill_python_processes
    sleep 2
}

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Logs dir:  $LOG_DIR"
echo "[INFO] Task=${TASK}, num_envs=${NUM_ENVS}, seeds='${SEEDS_STR}', combos='${COMBOS_STR}'"
echo "[INFO] Combo A: reward probe, update every PPO step, no regularization, PPO still env-reward only"
echo "[INFO] Combo B: reward probe, update every 4 PPO steps, no regularization"
echo "[INFO] Combo C: reward probe, update every 8 PPO steps, no regularization"
echo "[INFO] Combo D: reward probe, update every 4 PPO steps, add grad/logit/weight regularization"
echo "[INFO] Combo E: combo D plus two reward steps when an update is due"
echo "[INFO] Combo F: combo E plus estimated reward mixed back into PPO at weight 0.1"

for combo in "${COMBO_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
        run_one "$combo" "$seed"
    done
done

echo
echo "[INFO] Finished local debug sweep. Check logs under: $LOG_DIR"
