"""Algorithm and reward hyperparameters for KC-Safe-MAPPO.

This file intentionally keeps learning parameters outside config.py, which is
reserved as the shared refinery case-data source.
"""

ALGO_CONFIG = {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_param": 0.2,
    "value_clip_param": 0.2,
    "entropy_coef": 0.05,
    "value_loss_coef": 0.5,
    "learning_rate": 3e-4,
    "max_grad_norm": 0.5,
    # One rollout = exactly one episode (horizon=24 steps).
    "rollout_length": 24,
    "ppo_epochs": 5,
    "mini_batch_size": 64,
    "hidden_dim": 256,
    "reward_scale": 10000.0,
}

REWARD_CONFIG = {
    "alpha": 0.95,
    "switch_cost": 50.0,
    # Shortage penalty must make 'do nothing' strictly more expensive than
    # running.  Full shutdown shortage ≈ 30.63 total demand units.
    # At 8000/unit → ~245K, well above run-cost of ~95K do-nothing baseline
    # and competitive with running costs of ~100-200K.
    "shortage_cost": 8000.0,
    # Safety layer already prevents most violations; keep this non-dominating
    # so learning focuses on profit/capacity tradeoffs.
    "violation_cost": 500.0,
    "crude_efficiency_bonus": 500.0,
    "reward_scale": ALGO_CONFIG["reward_scale"],
}

LOAD_DELTA_ACTIONS = {
    0: -1,
    1: 0,
    2: 1,
}

LOAD_LEVELS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
