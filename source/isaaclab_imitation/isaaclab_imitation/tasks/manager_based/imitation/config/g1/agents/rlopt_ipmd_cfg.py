from isaaclab.utils import configclass

from isaaclab_rl.rlopt import IPMDRLOptConfig

from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_cfg import (
    G1_POLICY_OBS_KEYS,
    G1_REWARD_OBS_KEYS,
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
        self.ipmd.reward_input_keys = list(G1_REWARD_OBS_KEYS)

        # More initial exploration to improve policy-state coverage for inverse reward.
        self.collector.init_random_frames = 49152
        self.collector.frames_per_batch = 24
        self.replay_buffer.size = 4096 * 24

        self.loss.epochs = 5
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

        self.collector.total_frames = 30000 * 4096 * 24
        self.save_interval = 500

        self.ipmd.reward_input_type = "s'"
        self.ipmd.use_estimated_rewards_for_ppo = True
        self.ipmd.use_reward_target_network = True
        self.ipmd.use_reward_target_for_ppo = True
        self.ipmd.reward_target_polyak = 0.995
        self.ipmd.reward_target_update_interval = 1

        # Decouple and slow reward updates relative to PPO.
        self.ipmd.reward_optimizer = "adamw"
        self.ipmd.reward_lr = 2.0e-4
        self.ipmd.reward_weight_decay = 0.0
        self.ipmd.reward_max_grad_norm = 1.0
        self.ipmd.reward_update_interval = 2

        # Trust-region / anti-collapse reward objective.
        self.ipmd.reward_margin = 0.05
        self.ipmd.reward_consistency_coeff = 0.2

        # AMP-style regularization and replay for reward learning.
        self.ipmd.normalize_reward_input = True
        self.ipmd.reward_grad_penalty_coeff = 0.2
        self.ipmd.reward_logit_reg_coeff = 0.02
        self.ipmd.reward_param_weight_decay_coeff = 1.0e-5
        self.ipmd.reward_replay_size = 200000
        self.ipmd.reward_replay_ratio = 0.5
        self.ipmd.reward_replay_keep_prob = 0.25

        # Curriculum: smoothly mix env imitation reward with learned reward.
        self.ipmd.reward_mix_alpha_start = 0.0
        self.ipmd.reward_mix_alpha_end = 1.0
        self.ipmd.reward_mix_anneal_updates = 20000
        self.ipmd.reward_mix_gate_estimated_std_min = 0.05
        self.ipmd.reward_mix_alpha_when_unstable = 0.15
        self.ipmd.reward_mix_gate_after_updates = 500

        # Exploration and BC warm-start schedules.
        self.ipmd.entropy_coeff_start = 0.02
        self.ipmd.entropy_coeff_end = self.ppo.entropy_coeff
        self.ipmd.entropy_schedule_updates = 15000
        self.ipmd.bc_loss_coeff = 0.02
        self.ipmd.bc_warmup_updates = 20000
        self.ipmd.bc_final_coeff = 0.0
        self.ipmd.expert_batch_size = 10000
