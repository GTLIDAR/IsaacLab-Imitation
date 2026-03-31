# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ... import mdp
from .imitation_g1_env_cfg import (
    G1ObservationCfg,
    ImitationG1LafanTrackEnvCfg,
    _g1_reference_anchor_obs_params,
    _g1_reference_motion_obs_params,
    _g1_tracked_body_obs_params,
    _g1_lafan_track_env_cfg_from_dict,
)


@configclass
class G1LatentObservationCfg:
    """Latent-conditioned observation settings for the 29-DoF tracking environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""

        latent_command = ObsTerm(func=mdp.agent_latent_command)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        body_pos = ObsTerm(
            func=mdp.robot_body_pos_b,
            params=_g1_tracked_body_obs_params(),
        )
        body_ori = ObsTerm(
            func=mdp.robot_body_ori_b,
            params=_g1_tracked_body_obs_params(),
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel_rel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic observations."""

        latent_command = ObsTerm(func=mdp.agent_latent_command)
        reference_motion = ObsTerm(
            func=mdp.reference_motion_command,
            params=_g1_reference_motion_obs_params(),
        )
        reference_anchor_pos_b = ObsTerm(
            func=mdp.reference_anchor_pos_b,
            params=_g1_reference_anchor_obs_params(),
        )
        reference_anchor_ori_b = ObsTerm(
            func=mdp.reference_anchor_ori_b,
            params=_g1_reference_anchor_obs_params(),
        )
        body_pos = ObsTerm(
            func=mdp.robot_body_pos_b,
            params=_g1_tracked_body_obs_params(),
        )
        body_ori = ObsTerm(
            func=mdp.robot_body_ori_b,
            params=_g1_tracked_body_obs_params(),
            history_length=3,
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=3)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=3)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=3)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, history_length=3)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, history_length=3)
        joint_pos = ObsTerm(func=mdp.joint_pos, history_length=3)
        joint_vel = ObsTerm(func=mdp.joint_vel, history_length=3)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = False

    ReferenceCfg = G1ObservationCfg.ReferenceCfg

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    reference: ReferenceCfg = ReferenceCfg()


@configclass
class ImitationG1LatentEnvCfg(ImitationG1LafanTrackEnvCfg):
    """Latent-conditioned G1 motion-tracking env driven by a LAFAN1 manifest."""

    observations = G1LatentObservationCfg()
    enable_latent_command: bool = True

    def __post_init__(self):
        super().__post_init__()
        # No reference-based terminations in latent mode
        self.terminations.anchor_pos = None
        self.terminations.anchor_ori = None
        self.terminations.ee_body_pos = None


ImitationG1LatentEnvCfg.from_dict = _g1_lafan_track_env_cfg_from_dict
