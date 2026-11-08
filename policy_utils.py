"""학습된 정책을 불러와서 한 스텝씩 행동을 뽑는 도우미."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import make_command_ant


class Policy:
    """VecNormalize 통계까지 함께 불러와서, 원본 관측 → 행동으로 바꿔준다."""

    def __init__(self, run_dir: str | Path, checkpoint: str = "final_model"):
        run_dir = Path(run_dir)
        self.model = PPO.load(run_dir / f"{checkpoint}.zip", device="cpu")
        dummy = DummyVecEnv([lambda: make_command_ant()])
        self.norm = VecNormalize.load(str(run_dir / "vecnormalize.pkl"), dummy)
        self.norm.training = False
        self.norm.norm_reward = False

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_n = self.norm.normalize_obs(obs[None, :])
        action, _ = self.model.predict(obs_n, deterministic=True)
        return action[0]
