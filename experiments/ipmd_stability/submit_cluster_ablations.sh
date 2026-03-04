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
COMBOS_STR="${COMBOS:-A B C D E F G}"
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
    "agent.ipmd.reward_lr=1e-5"
    "agent.ipmd.reward_update_interval=100"
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

F_EXTRA=(
    "agent.ipmd.reward_updates_per_policy_update=2"
    "agent.ipmd.reward_update_warmup_updates=500"
    "agent.ipmd.reward_balance_policy_and_expert=true"
    "agent.ipmd.reward_train_on_logits=true"
)

G_EXTRA=(
    "agent.ipmd.reward_input_noise_std=0.01"
    "agent.ipmd.reward_input_dropout_prob=0.05"
    "agent.ipmd.reward_replay_reset_interval_updates=5000"
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
    "agent.ase.mi_reward_coeff=0.25"
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
            OVERRIDES+=("${B_EXTRA[@]}" "${C_EXTRA[@]}" "${D_EXTRA[@]}")
            ;;
        E)
            OVERRIDES+=("${B_EXTRA[@]}" "${C_EXTRA[@]}" "${D_EXTRA[@]}" "${E_EXTRA[@]}")
            ;;
        F)
            OVERRIDES+=("${B_EXTRA[@]}" "${C_EXTRA[@]}" "${D_EXTRA[@]}" "${E_EXTRA[@]}" "${F_EXTRA[@]}")
            ;;
        G)
            OVERRIDES+=("${B_EXTRA[@]}" "${C_EXTRA[@]}" "${D_EXTRA[@]}" "${E_EXTRA[@]}" "${F_EXTRA[@]}" "${G_EXTRA[@]}")
            ;;
        *)
            echo "[ERROR] Unknown combo '$combo'. Supported combos: A B C D E F G"
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
