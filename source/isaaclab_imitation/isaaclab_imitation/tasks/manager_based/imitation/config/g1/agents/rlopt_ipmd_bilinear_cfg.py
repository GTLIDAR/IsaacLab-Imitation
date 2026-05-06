from isaaclab.utils import configclass

from isaaclab_imitation.envs.rlopt import IPMDBilinearRLOptConfig

from isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg import (
    LATENT_CRITIC_INPUT_KEYS,
    LATENT_POLICY_INPUT_KEYS,
    LATENT_POSTERIOR_INPUT_KEYS,
    LATENT_PRIOR_INPUT_KEYS,
    REWARD_INPUT_KEYS,
    VANILLA_CRITIC_INPUT_KEYS,
    VANILLA_POLICY_INPUT_KEYS,
)

BILINEAR_OBS_KEYS: list[tuple[str, str]] = [
    ("policy", "base_ang_vel"),
    ("policy", "joint_pos_rel"),
    ("policy", "joint_vel_rel"),
    ("policy", "last_action"),
]

BILINEAR_NEXT_OBS_KEYS: list[tuple[str, str]] = [
    ("policy", "base_ang_vel"),
    ("policy", "joint_pos_rel"),
    ("policy", "joint_vel_rel"),
]


@configclass
class _G1ImitationRLOptIPMDBilinearBaseConfig(IPMDBilinearRLOptConfig):
    """Shared RLOpt IPMD + Bilinear configuration for G1 imitation."""

    _default_use_latent_command: bool = False

    def sync_input_keys(self) -> None:
        use_latent_command = bool(self.ipmd.use_latent_command)
        self.policy.input_keys = (
            list(LATENT_POLICY_INPUT_KEYS)
            if use_latent_command
            else list(VANILLA_POLICY_INPUT_KEYS)
        )
        if self.value_function is not None:
            self.value_function.input_keys = (
                list(LATENT_CRITIC_INPUT_KEYS)
                if use_latent_command
                else list(VANILLA_CRITIC_INPUT_KEYS)
            )
        self.ipmd.reward_input_keys = list(REWARD_INPUT_KEYS)
        self.ipmd.latent_learning.posterior_input_keys = list(
            LATENT_POSTERIOR_INPUT_KEYS
        )
        self.ipmd.latent_learning.prior_input_keys = list(LATENT_PRIOR_INPUT_KEYS)
        self.ipmd.latent_key = ("policy", "latent_command")
        self.ipmd.use_latent_command = use_latent_command
        self.bilinear.obs_keys = list(BILINEAR_OBS_KEYS)
        self.bilinear.next_obs_keys = list(BILINEAR_NEXT_OBS_KEYS)

    def __post_init__(self):
        super().__post_init__()

        assert isinstance(self, IPMDBilinearRLOptConfig)
        assert self.value_function is not None, (
            "Value function configuration must be provided."
        )

        self.ipmd.use_latent_command = bool(self._default_use_latent_command)
        self.sync_input_keys()
        self.logger.group_name = ""

        # More initial exploration to improve policy-state coverage for inverse reward.
        self.collector.init_random_frames = 0
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

        self.collector.total_frames = 5_000_000_000
        self.save_interval = 100  # rollout iterations

        # Debug: latent posterior input mirrors the single-step vanilla tracker
        # policy reference payload directly: expert_motion (58) + anchor_ori (6).
        self.ipmd.latent_dim = 64
        self.ipmd.latent_steps_min = 1
        self.ipmd.latent_steps_max = 1
        self.ipmd.latent_learning.method = "patch_autoencoder"
        self.ipmd.latent_learning.encoder_hidden_dims = [256, 256]
        self.ipmd.latent_learning.encoder_activation = "elu"
        self.ipmd.latent_learning.prior_hidden_dims = [256, 256]
        self.ipmd.latent_learning.prior_activation = "elu"
        self.ipmd.latent_learning.patch_past_steps = 0
        self.ipmd.latent_learning.patch_future_steps = 0
        self.ipmd.latent_learning.lr = 3.0e-4
        self.ipmd.latent_learning.grad_clip_norm = 1.0
        self.ipmd.latent_learning.freeze_encoder = True
        self.ipmd.latent_learning.train_posterior_through_policy = True

        # Debug mode still trains the autoencoder on expert reference patches,
        # but the live latent_command path publishes the raw posterior features
        # directly so data flow can be checked independently of the encoder.
        self.ipmd.latent_learning.recon_coeff = 1.0
        self.ipmd.latent_learning.weight_decay_coeff = 0.0
        self.ipmd.latent_learning.kl_coeff = 0.0
        self.ipmd.latent_learning.probe_enabled = False
        self.ipmd.latent_learning.probe_condition_on_state = False
        self.ipmd.latent_learning.probe_target_keys = list(REWARD_INPUT_KEYS)
        self.ipmd.latent_learning.probe_hidden_dims = [256, 256]
        self.ipmd.latent_learning.probe_activation = "elu"
        self.ipmd.latent_learning.probe_lr = 3.0e-4
        self.ipmd.latent_learning.probe_grad_clip_norm = 1.0
        self.ipmd.latent_learning.probe_batch_size = 8192
        self.ipmd.env_reward_weight = 1.0

        # Keep the policy objective free of extra latent shaping.
        self.ipmd.diversity_bonus_coeff = 0.0
        self.ipmd.diversity_target = 0.0
        self.ipmd.latent_uniformity_temperature = 2.0

        self.ipmd.reward_input_type = "s"
        self.ipmd.use_estimated_rewards_for_ppo = False
        self.ipmd.expert_batch_size = int(self.loss.mini_batch_size)
        self.ipmd.bc_coef = 0.0
        self.compile.compile = False
        # self.trainer.progress_bar = False
        # self.trainer.log_interval = 10_000_000
        self.ipmd.reward_output_scale = 1.0
        self.ipmd.estimated_reward_clamp_min = -1.0
        self.ipmd.estimated_reward_clamp_max = 1.0
        self.ipmd.est_reward_weight = 1.0
        self.ipmd.reward_loss_coeff = 1.0
        self.ipmd.reward_l2_coeff = 0.0
        self.ipmd.reward_grad_penalty_coeff = 0.0
        self.collector.no_cuda_sync = True

        # Collector latents should consume the same observation-manager channel
        # stored in the rollout TensorDict. Expert data enters through
        # sample_expert_batch(...) during updates, not through live env getters.
        if self._default_use_latent_command:
            self.ipmd.command_source = "posterior"

        self.bilinear.feature_dim = self.ipmd.latent_dim
        self.bilinear.offline_pretrain.enabled = False
        self.bilinear.offline_pretrain.num_updates = 2000
        self.bilinear.offline_pretrain.batch_size = 8192
        self.bilinear.offline_pretrain.log_interval = 100


@configclass
class G1ImitationRLOptIPMDBilinearConfig(_G1ImitationRLOptIPMDBilinearBaseConfig):
    """Vanilla RLOpt IPMD + Bilinear configuration for G1 imitation."""

    _default_use_latent_command: bool = False


@configclass
class G1ImitationLatentRLOptIPMDBilinearConfig(_G1ImitationRLOptIPMDBilinearBaseConfig):
    """Latent-conditioned RLOpt IPMD + Bilinear configuration for G1 imitation."""

    _default_use_latent_command: bool = True
