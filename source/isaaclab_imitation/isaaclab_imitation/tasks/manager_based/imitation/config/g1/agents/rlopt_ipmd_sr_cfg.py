from isaaclab.utils import configclass

from isaaclab_imitation.envs.rlopt import IPMDSRRLOptConfig


VANILLA_POLICY_INPUT_KEYS: list[tuple[str, str]] = [
    ("policy", "expert_motion"),
    ("policy", "expert_anchor_ori_b"),
    ("policy", "base_ang_vel"),
    ("policy", "joint_pos_rel"),
    ("policy", "joint_vel_rel"),
    ("policy", "last_action"),
]

VANILLA_CRITIC_INPUT_KEYS: list[tuple[str, str]] = [
    ("critic", "expert_motion"),
    ("critic", "expert_anchor_pos_b"),
    ("critic", "expert_anchor_ori_b"),
    ("critic", "body_pos"),
    ("critic", "body_ori"),
    ("critic", "base_lin_vel"),
    ("critic", "base_ang_vel"),
    ("critic", "joint_pos_rel"),
    ("critic", "joint_vel_rel"),
    ("critic", "last_action"),
]

LATENT_POLICY_INPUT_KEYS: list[tuple[str, str]] = [
    ("policy", "latent_command"),
    ("policy", "projected_gravity"),
    ("policy", "base_lin_vel"),
    ("policy", "base_ang_vel"),
    ("policy", "joint_pos_rel"),
    ("policy", "joint_vel_rel"),
    ("policy", "last_action"),
]

LATENT_CRITIC_INPUT_KEYS: list[tuple[str, str]] = [
    ("critic", "latent_command"),
    ("critic", "expert_motion"),
    ("critic", "expert_anchor_pos_b"),
    ("critic", "expert_anchor_ori_b"),
    ("critic", "body_pos"),
    ("critic", "body_ori"),
    ("critic", "projected_gravity"),
    ("critic", "base_lin_vel"),
    ("critic", "base_ang_vel"),
    ("critic", "joint_pos_rel"),
    ("critic", "joint_vel_rel"),
    ("critic", "joint_pos"),
    ("critic", "joint_vel"),
    ("critic", "last_action"),
]

EXPERT_INPUT_KEYS: list[tuple[str, str]] = [
    ("expert_state", "joint_pos"),
    ("expert_state", "joint_vel"),
    ("expert_state", "root_pos"),
    ("expert_state", "root_quat"),
    ("expert_state", "root_lin_vel"),
    ("expert_state", "root_ang_vel"),
]


@configclass
class _G1ImitationRLOptIPMDSRBaseConfig(IPMDSRRLOptConfig):
    """Shared RLOpt IPMD + SR configuration for G1 imitation."""

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
        self.ipmd.reward_input_keys = list(EXPERT_INPUT_KEYS)
        self.ipmd.latent_key = ("policy", "latent_command")
        self.ipmd.use_latent_command = use_latent_command

    def __post_init__(self):
        super().__post_init__()

        assert isinstance(self, IPMDSRRLOptConfig)
        assert self.value_function is not None, (
            "Value function configuration must be provided."
        )

        self.ipmd.use_latent_command = bool(self._default_use_latent_command)
        self.ipmd.command_source = (
            "posterior" if self._default_use_latent_command else "random"
        )
        self.sync_input_keys()

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
        self.save_interval = 5_000_000  # samples

        self.ipmd.latent_dim = 64
        self.ipmd.latent_steps_min = 30
        self.ipmd.latent_steps_max = 120
        self.ipmd.latent_learning.method = "patch_autoencoder"
        self.ipmd.latent_learning.encoder_hidden_dims = [256, 256]
        self.ipmd.latent_learning.encoder_activation = "elu"
        self.ipmd.latent_learning.patch_past_steps = 1
        self.ipmd.latent_learning.patch_future_steps = 1
        self.ipmd.latent_learning.lr = 3.0e-4
        self.ipmd.latent_learning.grad_clip_norm = 1.0
        self.ipmd.latent_learning.recon_coeff = 1.0
        self.ipmd.latent_learning.uniformity_coeff = 0.0
        self.ipmd.latent_learning.weight_decay_coeff = 0.0

        self.ipmd.reward_input_type = "s'"
        self.ipmd.use_estimated_rewards_for_ppo = True
        self.ipmd.expert_batch_size = int(self.loss.mini_batch_size)
        self.ipmd.bc_coef = 0.1
        self.compile.compile = False
        self.trainer.progress_bar = True
        self.trainer.log_interval = 10_000_000  # samples
        self.ipmd.reward_output_scale = 1.0
        self.ipmd.estimated_reward_clamp_min = -1.0
        self.ipmd.estimated_reward_clamp_max = 1.0
        self.ipmd.est_reward_weight = 1.0
        self.collector.no_cuda_sync = True
        self.log_level = "critical"
        self.ipmd.reward_l2_coeff = 0.5


@configclass
class G1ImitationRLOptIPMDSRConfig(_G1ImitationRLOptIPMDSRBaseConfig):
    """Vanilla RLOpt IPMD + SR configuration for G1 imitation."""

    _default_use_latent_command: bool = False


@configclass
class G1ImitationLatentRLOptIPMDSRConfig(_G1ImitationRLOptIPMDSRBaseConfig):
    """Latent-conditioned RLOpt IPMD + SR configuration for G1 imitation."""

    _default_use_latent_command: bool = True
