from isaaclab.utils import configclass

from isaaclab_imitation.envs.rlopt import SACRLOptConfig


VANILLA_POLICY_INPUT_KEYS: list[tuple[str, str]] = [
    ("policy", "reference_motion"),
    ("policy", "reference_anchor_ori_b"),
    ("policy", "base_ang_vel"),
    ("policy", "joint_pos_rel"),
    ("policy", "joint_vel_rel"),
    ("policy", "last_action"),
]

VANILLA_CRITIC_INPUT_KEYS: list[tuple[str, str]] = [
    ("critic", "reference_motion"),
    ("critic", "reference_anchor_pos_b"),
    ("critic", "reference_anchor_ori_b"),
    ("critic", "body_pos"),
    ("critic", "body_ori"),
    ("critic", "base_lin_vel"),
    ("critic", "base_ang_vel"),
    ("critic", "joint_pos_rel"),
    ("critic", "joint_vel_rel"),
    ("critic", "last_action"),
]


@configclass
class G1ImitationRLOptSACConfig(SACRLOptConfig):
    """RLOpt SAC configuration for vanilla G1 imitation."""

    def __post_init__(self):
        super().__post_init__()

        assert self.q_function is not None, "Q function configuration must be provided."

        self.policy.input_keys = list(VANILLA_POLICY_INPUT_KEYS)
        self.q_function.input_keys = list(VANILLA_CRITIC_INPUT_KEYS)
        if self.value_function is not None:
            self.value_function.input_keys = list(VANILLA_CRITIC_INPUT_KEYS)

        self.collector.frames_per_batch = 24
        self.collector.init_random_frames = 0
        self.collector.total_frames = 100_000_000

        self.loss.epochs = 1
        self.loss.mini_batch_size = 256
        self.loss.gamma = 0.99

        self.policy.num_cells = [512, 256, 128]
        self.q_function.num_cells = [512, 256, 128]

        self.sac.alpha_init = 1.0
        self.sac.target_entropy = "auto"
        self.sac.num_qvalue_nets = 2

        self.optim.target_update_polyak = 0.995
        self.optim.lr = 3.0e-4

        self.save_interval = 500
        self.compile.compile = False
        self.collector.no_cuda_sync = True
