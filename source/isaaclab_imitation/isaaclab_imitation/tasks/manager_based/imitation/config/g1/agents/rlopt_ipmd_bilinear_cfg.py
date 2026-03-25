from isaaclab.utils import configclass

from isaaclab_imitation.envs.rlopt import IPMDBilinearRLOptConfig
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_cfg import (
    G1_IPMD_REWARD_OBS_KEYS,
    G1_POLICY_OBS_KEYS,
    G1_VALUE_OBS_KEYS,
)


@configclass
class G1ImitationRLOptIPMDBilinearConfig(IPMDBilinearRLOptConfig):
    """RLOpt IPMD + Bilinear SR configuration for G1 imitation."""

    def __post_init__(self):
        super().__post_init__()

        assert isinstance(self, IPMDBilinearRLOptConfig)
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
        self.ipmd.expert_batch_size = int(self.loss.mini_batch_size)
        self.ipmd.bc_coef = 0.0
        self.compile.compile = False
        self.trainer.progress_bar = True
        self.ipmd.reward_output_scale = 0.25
        self.ipmd.estimated_reward_clamp_min = 0.0
        self.ipmd.estimated_reward_clamp_max = 0.25
        self.ipmd.estimated_reward_mix_coeff = 0.3
        self.collector.no_cuda_sync = True
        self.log_level = "critical"
        self.ipmd.reward_l2_coeff = 0.5
