#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TASK="${TASK:-Isaac-Imitation-G1-v0}"
NUM_ENVS="${NUM_ENVS:-2048}"
ALGO="${ALGO:-ipmd}"
PRESET="${PRESET:-baseline}"
SEEDS_STR="${SEEDS:-2024 2025 2026}"
COMBOS_STR="${COMBOS:-A B C D E F}"
EXPERT_RB_DIR="${EXPERT_RB_DIR:-}"
DRY_RUN="${DRY_RUN:-0}"
VIDEO="${VIDEO:-1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-2000}"
# Optional profile name recognized by ./docker/.env.<profile> (for example: base).
CLUSTER_PROFILE="${CLUSTER_PROFILE:-}"

ALGO_LOWER="$(echo "$ALGO" | tr '[:upper:]' '[:lower:]')"
read -r -a SEED_LIST <<< "$SEEDS_STR"
read -r -a COMBO_LIST <<< "$COMBOS_STR"

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

GAIL_BASELINE_OVERRIDES=(
    "agent.gail.discriminator_input_keys=[invrwd]"
    "agent.gail.discriminator_updates_per_policy_update=2"
    "agent.gail.expert_batch_size=24576"
    "agent.gail.discriminator_batch_size=24576"
    "agent.gail.normalize_discriminator_input=true"
    "agent.gail.discriminator_replay_size=200000"
    "agent.gail.discriminator_replay_ratio=0.5"
    "agent.gail.discriminator_replay_keep_prob=0.25"
    "agent.gail.discriminator_grad_penalty_coeff=0.2"
    "agent.gail.discriminator_logit_reg_coeff=0.01"
    "agent.gail.discriminator_weight_decay_coeff=1e-5"
    "agent.gail.normalize_discriminator_reward=true"
    "agent.gail.proportion_env_reward=0.1"
)

AMP_BASELINE_OVERRIDES=(
    "agent.gail.discriminator_input_keys=[invrwd]"
    "agent.gail.discriminator_updates_per_policy_update=2"
    "agent.gail.expert_batch_size=24576"
    "agent.gail.discriminator_batch_size=24576"
    "agent.gail.normalize_discriminator_input=true"
    "agent.gail.discriminator_replay_size=200000"
    "agent.gail.discriminator_replay_ratio=0.5"
    "agent.gail.discriminator_replay_keep_prob=0.25"
    "agent.gail.discriminator_grad_penalty_coeff=0.2"
    "agent.gail.discriminator_logit_reg_coeff=0.02"
    "agent.gail.discriminator_weight_decay_coeff=1e-5"
    "agent.gail.normalize_discriminator_reward=true"
    "agent.gail.proportion_env_reward=0.1"
    "agent.gail.amp_reward_clip=true"
    "agent.gail.amp_reward_scale=1.0"
)

ASE_BASELINE_OVERRIDES=(
    "agent.gail.discriminator_input_keys=[invrwd]"
    "agent.gail.discriminator_updates_per_policy_update=2"
    "agent.gail.expert_batch_size=24576"
    "agent.gail.discriminator_batch_size=24576"
    "agent.gail.normalize_discriminator_input=true"
    "agent.gail.discriminator_replay_size=200000"
    "agent.gail.discriminator_replay_ratio=0.5"
    "agent.gail.discriminator_replay_keep_prob=0.25"
    "agent.gail.discriminator_grad_penalty_coeff=0.2"
    "agent.gail.discriminator_logit_reg_coeff=0.02"
    "agent.gail.discriminator_weight_decay_coeff=1e-5"
    "agent.gail.normalize_discriminator_reward=true"
    "agent.gail.proportion_env_reward=0.1"
    "agent.gail.amp_reward_clip=true"
    "agent.gail.amp_reward_scale=1.0"
    "agent.ase.latent_dim=16"
    "agent.ase.latent_steps_min=30"
    "agent.ase.latent_steps_max=120"
    "agent.ase.mi_reward_weight=0.25"
    "agent.ase.mi_loss_coeff=1.0"
    "agent.ase.mi_grad_penalty_coeff=0.05"
    "agent.ase.mi_weight_decay_coeff=1e-5"
    "agent.ase.diversity_bonus_coeff=0.05"
    "agent.ase.latent_uniformity_coeff=0.005"
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

get_preset_overrides() {
    local algo="$1"
    local preset="$2"
    case "$algo" in
        gail)
            if [[ "$preset" != "baseline" ]]; then
                echo "[ERROR] Unknown GAIL preset '$preset' (supported: baseline)"
                exit 1
            fi
            OVERRIDES=("${GAIL_BASELINE_OVERRIDES[@]}")
            ;;
        amp)
            if [[ "$preset" != "baseline" ]]; then
                echo "[ERROR] Unknown AMP preset '$preset' (supported: baseline)"
                exit 1
            fi
            OVERRIDES=("${AMP_BASELINE_OVERRIDES[@]}")
            ;;
        ase)
            if [[ "$preset" != "baseline" ]]; then
                echo "[ERROR] Unknown ASE preset '$preset' (supported: baseline)"
                exit 1
            fi
            OVERRIDES=("${ASE_BASELINE_OVERRIDES[@]}")
            ;;
        *)
            echo "[ERROR] Unsupported algorithm '$algo'. Supported: ipmd gail amp ase"
            exit 1
            ;;
    esac
}

submit_one() {
    local algo="$1"
    local label="$2"
    local seed="$3"

    local run_name="${algo}_${label}_seed${seed}"
    local cmd=(./docker/cluster/cluster_interface.sh job)

    # Follow cluster_interface usage: job [<profile>] [<job_args>...]
    if [[ -n "$CLUSTER_PROFILE" ]]; then
        cmd+=("$CLUSTER_PROFILE")
    fi

    cmd+=(
        --task "$TASK"
        --num_envs "$NUM_ENVS"
        --headless
        --algo "$algo"
        "agent.seed=${seed}"
        "agent.logger.exp_name=${run_name}"
        "${OVERRIDES[@]}"
    )

    if [[ "$VIDEO" == "1" || "$VIDEO" == "true" ]]; then
        cmd+=(
            --video
            --video_length "$VIDEO_LENGTH"
            --video_interval "$VIDEO_INTERVAL"
        )
    fi

    if [[ "$algo" == "gail" || "$algo" == "amp" || "$algo" == "ase" ]]; then
        if [[ -z "$EXPERT_RB_DIR" ]]; then
            echo "[ERROR] EXPERT_RB_DIR must be set for ALGO=$algo"
            exit 1
        fi
        cmd+=(--expert_rb_dir "$EXPERT_RB_DIR")
    fi

    printf "\n[%s] Submitting %s\n" "$(date '+%F %T')" "$run_name"
    printf "[CMD] "
    printf "%q " "${cmd[@]}"
    printf "\n"
    if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    "${cmd[@]}"
}

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Task=${TASK}, num_envs=${NUM_ENVS}, algo=${ALGO_LOWER}, preset=${PRESET}, seeds='${SEEDS_STR}', combos='${COMBOS_STR}', video='${VIDEO}', video_length='${VIDEO_LENGTH}', video_interval='${VIDEO_INTERVAL}', dry_run='${DRY_RUN}', profile='${CLUSTER_PROFILE:-<default>}'"
echo "[INFO] Combo A: reward probe, update every PPO step, no regularization, PPO still env-reward only"
echo "[INFO] Combo B: reward probe, update every 4 PPO steps, no regularization"
echo "[INFO] Combo C: reward probe, update every 8 PPO steps, no regularization"
echo "[INFO] Combo D: reward probe, update every 4 PPO steps, add grad/logit/weight regularization"
echo "[INFO] Combo E: combo D plus two reward steps when an update is due"
echo "[INFO] Combo F: combo E plus estimated reward mixed back into PPO at weight 0.1"

if [[ "$ALGO_LOWER" == "ipmd" ]]; then
    for combo in "${COMBO_LIST[@]}"; do
        get_combo_overrides "$combo"
        for seed in "${SEED_LIST[@]}"; do
            submit_one "$ALGO_LOWER" "$combo" "$seed"
        done
    done
else
    get_preset_overrides "$ALGO_LOWER" "$PRESET"
    for seed in "${SEED_LIST[@]}"; do
        submit_one "$ALGO_LOWER" "$PRESET" "$seed"
    done
fi

echo
echo "[INFO] Submitted all requested jobs."
