from __future__ import annotations

import pytest

try:
    from isaaclab.app import AppLauncher
except ModuleNotFoundError as exc:
    pytest.skip(
        f"Isaac Sim app launcher is unavailable: {exc}", allow_module_level=True
    )

try:
    simulation_app = AppLauncher(headless=True).app
except (ImportError, OSError, RuntimeError, SystemExit) as exc:
    pytest.skip(f"Isaac Sim bootstrap is unavailable: {exc}", allow_module_level=True)

try:
    from isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ase_cfg import (  # noqa: E501
        G1ImitationRLOptASEConfig,
    )
    from isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_bilinear_cfg import (  # noqa: E501
        G1ImitationLatentRLOptIPMDBilinearConfig,
    )
    from isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg import (  # noqa: E501
        G1ImitationLatentRLOptIPMDConfig,
    )
    from isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_vqvae_cfg import (  # noqa: E501
        G1ImitationLatentRLOptIPMDVQVAEConfig,
    )
except (ModuleNotFoundError, SystemExit) as exc:
    pytest.skip(
        f"IsaacLab config imports are unavailable: {exc}", allow_module_level=True
    )


def test_g1_latent_observations_expose_separate_command_group() -> None:
    try:
        from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_latent_env_cfg import (  # noqa: E501
            G1LatentObservationCfg,
        )
    except (ModuleNotFoundError, SystemExit) as exc:
        pytest.skip(f"G1 latent env config import is unavailable: {exc}")

    obs = G1LatentObservationCfg()

    assert hasattr(obs.command, "latent_command")
    assert obs.command.concatenate_terms is False
    assert not hasattr(obs.policy, "latent_command")
    for term_name in (
        "latent_command",
        "expert_motion",
        "expert_anchor_pos_b",
        "expert_anchor_ori_b",
    ):
        assert not hasattr(obs.critic, term_name)


def test_g1_latent_agent_configs_use_command_group_for_latent_key() -> None:
    command_key = ("command", "latent_command")
    stale_policy_key = ("policy", "latent_command")
    stale_critic_key = ("critic", "latent_command")

    for cfg in (
        G1ImitationLatentRLOptIPMDConfig(),
        G1ImitationLatentRLOptIPMDBilinearConfig(),
        G1ImitationLatentRLOptIPMDVQVAEConfig(),
    ):
        assert cfg.ipmd.latent_key == command_key
        assert command_key in cfg.policy.input_keys
        assert command_key in cfg.value_function.input_keys
        assert stale_policy_key not in cfg.policy.input_keys
        assert stale_critic_key not in cfg.value_function.input_keys

    ase_cfg = G1ImitationRLOptASEConfig()
    assert ase_cfg.ase.latent_key == command_key
    assert command_key in ase_cfg.policy.input_keys
    assert command_key in ase_cfg.value_function.input_keys
    assert stale_policy_key not in ase_cfg.policy.input_keys
    assert stale_critic_key not in ase_cfg.value_function.input_keys


def test_g1_latent_bilinear_config_uses_raw_policy_and_linear_privileged_value() -> (
    None
):
    cfg = G1ImitationLatentRLOptIPMDBilinearConfig()

    assert cfg.ipmd.latent_key == ("command", "latent_command")
    assert cfg.bilinear.policy_input_mode == "raw"
    assert cfg.bilinear.value_input_mode == "linear_fz"
    assert ("command", "latent_command") in cfg.policy.input_keys
    assert ("command", "latent_command") in cfg.value_function.input_keys
    assert cfg.bilinear.value_command_keys == [("command", "latent_command")]
    assert cfg.bilinear.value_state_keys == cfg.bilinear.obs_keys

    expert_motion_keys = {
        ("policy", "expert_motion"),
        ("critic", "expert_motion"),
        ("expert_goal", "expert_motion"),
    }
    assert expert_motion_keys.isdisjoint(cfg.policy.input_keys)
    assert expert_motion_keys.isdisjoint(cfg.value_function.input_keys)
    assert expert_motion_keys.isdisjoint(cfg.bilinear.obs_keys)
