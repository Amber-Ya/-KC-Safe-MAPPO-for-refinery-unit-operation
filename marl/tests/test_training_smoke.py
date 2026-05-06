from __future__ import annotations

import math

import config
from marl.algorithms.kc_safe_mappo import KCSafeMAPPOTrainer
from marl.configs.algo_config import ALGO_CONFIG
from marl.envs.refinery_env import RefinerySchedulingEnv
from marl.utils.config_adapter import ConfigAdapter


def test_training_smoke_outputs_finite_loss_and_checkpoint(tmp_path) -> None:
    env = RefinerySchedulingEnv(ConfigAdapter(config).build_env_config(), seed=3)
    algo_config = dict(ALGO_CONFIG)
    algo_config.update({"rollout_length": 8, "ppo_epochs": 1, "mini_batch_size": 32})
    trainer = KCSafeMAPPOTrainer(env, algo_config=algo_config, seed=3, device="cpu")
    logs = trainer.train(total_steps=16, output_dir=str(tmp_path))
    assert logs
    assert math.isfinite(logs[-1]["loss"])
    checkpoint = tmp_path / "kc_safe_mappo_checkpoint.pt"
    assert checkpoint.exists()
    trainer.load_checkpoint(str(checkpoint))
    assert (tmp_path / "training_log.csv").exists()
