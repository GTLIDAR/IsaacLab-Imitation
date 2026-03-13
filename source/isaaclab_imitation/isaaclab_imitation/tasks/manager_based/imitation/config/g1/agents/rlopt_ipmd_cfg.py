from isaaclab.utils import configclass

from isaaclab_imitation.envs.rlopt import IPMDRLOptConfig
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_cfg import (
    G1_IPMD_REWARD_OBS_KEYS,
    G1_POLICY_OBS_KEYS,
    G1_VALUE_OBS_KEYS,
)


@configclass
class G1ImitationRLOptIPMDConfig(IPMDRLOptConfig):
    """RLOpt IPMD configuration for G1 imitation."""

    def __post_init__(self):
        super().__post_init__()

        assert isinstance(self, IPMDRLOptConfig)
        assert self.value_function is not None, (
            "Value function configuration must be provided."
        )

        self.policy.input_keys = list(G1_POLICY_OBS_KEYS)
        self.value_function.input_keys = list(G1_VALUE_OBS_KEYS)
        self.ipmd.reward_input_keys = list(G1_IPMD_REWARD_OBS_KEYS)

        # More initial exploration to improve policy-state coverage for inverse reward.
        self.collector.init_random_frames = 49152
        self.collector.frames_per_batch = 24
        self.replay_buffer.size = 4096 * 24

        self.loss.epochs = 1
        self.loss.mini_batch_size = 4096 * 24 // 4
        self.loss.loss_critic_type = "l2"

        self.ppo.clip_epsilon = 0.2
        self.ppo.gae_lambda = 0.95
        self.ppo.entropy_coeff = 0.005
        self.ppo.critic_coeff = 1.0
        self.ppo.clip_value = True
        self.ppo.normalize_advantage = True
        self.ppo.clip_log_std = False
        self.ppo.log_std_init = 0.0

        self.optim.lr = 1.0e-3
        self.optim.max_grad_norm = 1.0
        self.optim.scheduler = "adaptive"
        self.optim.desired_kl = 0.01

        self.loss.gamma = 0.99

        self.policy.num_cells = [512, 256, 128]
        self.value_function.num_cells = [512, 256, 128]

        self.collector.total_frames = 500_000_000
        self.save_interval = 500

        self.ipmd.reward_input_type = "s'"
        self.ipmd.use_estimated_rewards_for_ppo = True
        # self.ipmd.use_reward_target_network = True
        # self.ipmd.use_reward_target_for_ppo = True
        # self.ipmd.reward_target_polyak = 0.995
        # self.ipmd.reward_target_update_interval = 1

        # Decouple and slow reward updates relative to PPO (combo G: B + F).
        # self.ipmd.reward_optimizer = "adamw"
        # self.ipmd.reward_lr = 1.0e-5
        # self.ipmd.reward_weight_decay = 0.0
        # self.ipmd.reward_max_grad_norm = 1.0
        # self.ipmd.reward_update_interval = 100
        # self.ipmd.reward_updates_per_policy_update = 2
        # self.ipmd.reward_update_warmup_updates = 500
        # Keep expert reward minibatch aligned with PPO minibatch by default.
        self.ipmd.expert_batch_size = int(self.loss.mini_batch_size)
        self.ipmd.bc_coef = 0.1
        # self.ipmd.reward_balance_policy_and_expert = True
        self.compile.compile = False
        self.trainer.progress_bar = True
        self.ipmd.reward_output_scale = 1.0
        self.ipmd.estimated_reward_clamp_min = -1.0
        self.ipmd.estimated_reward_clamp_max = 1.0
        self.ipmd.estimated_reward_mix_coeff = 1.0
        self.collector.no_cuda_sync = True
        self.log_level = "critical"
        self.ipmd.reward_l2_coeff = 0.5

        # ------- unnecessary for ipmd simple --------- #
        # Trust-region / anti-collapse reward objective.
        # self.ipmd.reward_margin = 0.05
        # self.ipmd.reward_consistency_coeff = 0.2
        # self.ipmd.reward_train_on_logits = True

        # AMP-style regularization and replay for reward learning.
        # self.ipmd.normalize_reward_input = True
        # self.ipmd.reward_input_noise_std = 0.01
        # self.ipmd.reward_input_dropout_prob = 0.05
        # self.ipmd.reward_grad_penalty_coeff = 0.2
        # self.ipmd.reward_logit_reg_coeff = 0.02
        # self.ipmd.reward_param_weight_decay_coeff = 1.0e-5
        # self.ipmd.reward_replay_size = 200000
        # self.ipmd.reward_replay_ratio = 0.5
        # self.ipmd.reward_replay_keep_prob = 0.25
        # self.ipmd.reward_replay_reset_interval_updates = 5000

        # Curriculum: smoothly mix env imitation reward with learned reward.
        # self.ipmd.reward_mix_alpha_start = 0.5
        # self.ipmd.reward_mix_alpha_end = 0.5
        # self.ipmd.reward_mix_anneal_updates = 0
        # self.ipmd.reward_mix_gate_estimated_std_min = 0.05
        # self.ipmd.reward_mix_alpha_when_unstable = 0.15
        # self.ipmd.reward_mix_gate_after_updates = 500
        # self.ipmd.reward_mix_gate_abs_gap_max = 0.5
        # self.ipmd.reward_mix_alpha_when_gap_large = 0.1

        # Exploration and BC warm-start schedules.
        # self.ipmd.entropy_coeff_start = 0.02
        # self.ipmd.entropy_coeff_end = self.ppo.entropy_coeff
        # self.ipmd.entropy_schedule_updates = 15000
        # self.ipmd.policy_random_action_prob_start = 0.0
        # self.ipmd.policy_random_action_prob_end = 0.0
        # self.ipmd.policy_random_action_schedule_updates = 0
        # self.ipmd.reward_scheduler = None
        # self.ipmd.reward_scheduler_kwargs = {}
        # self.ipmd.reward_scheduler_step = "update"
        # self.ipmd.bc_coef = 0.00
        # self.ipmd.bc_warmup_updates = 0
        # self.ipmd.bc_final_coeff = 0.0
