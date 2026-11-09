"""PPO로 명령 조건부 보행 정책을 학습한다.

예시:
    python train.py --timesteps 5000000 --n-envs 8 --run-name main
    tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize

from envs import make_command_ant


class TrackingLogger(BaseCallback):
    """속도 추종 오차와 제어 노력을 텐서보드에 기록한다."""

    def __init__(self):
        super().__init__()
        self._err, self._effort = [], []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "tracking_error" in info:
                self._err.append(info["tracking_error"])
                self._effort.append(info["effort"])
        return True

    def _on_rollout_end(self) -> None:
        if self._err:
            self.logger.record("command/tracking_error_mps", float(np.mean(self._err)))
            self.logger.record("command/effort", float(np.mean(self._effort)))
        self._err.clear()
        self._effort.clear()


class SaveNormCallback(BaseCallback):
    """체크포인트와 함께 VecNormalize 통계도 저장한다."""

    def __init__(self, save_freq: int, run_dir: Path):
        super().__init__()
        self.save_freq = save_freq
        self.run_dir = run_dir

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.training_env.save(str(self.run_dir / "vecnormalize.pkl"))
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=5_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default=None, help="이어 학습할 run 폴더")
    p.add_argument("--tracking-sigma", type=float, default=0.25,
                   help="속도 추종 보상 exp(-오차²/sigma)의 sigma (v2: 0.25, v3: 1.0)")
    p.add_argument("--tracking-weight", type=float, default=2.0,
                   help="속도 추종 보상 가중치 (v2·v3: 2.0, v4: 5.0)")
    p.add_argument("--alive-weight", type=float, default=1.0, help="생존 보상 가중치")
    args = p.parse_args()

    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    def make(rank: int):
        def _init():
            return make_command_ant(seed=args.seed + rank,
                                    tracking_sigma=args.tracking_sigma,
                                    tracking_weight=args.tracking_weight,
                                    alive_weight=args.alive_weight)
        return _init

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = VecMonitor(vec_cls([make(i) for i in range(args.n_envs)]))

    if args.resume:
        resume = Path(args.resume)
        venv = VecNormalize.load(str(resume / "vecnormalize.pkl"), venv)
        model = PPO.load(resume / "final_model.zip", env=venv, device="cpu",
                         tensorboard_log=str(run_dir))
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        n_steps = 1024             # 환경당 수집 스텝 → 한 번에 1024 * n_envs
        model = PPO(
            "MlpPolicy",
            venv,
            n_steps=n_steps,
            # v1에서는 batch 256이라 업데이트 한 번에 미니배치가 48개 × 10에폭 = 480번 갱신
            # → 정책이 너무 크게 바뀌어 approx_kl 4.1, clip_fraction 0.87로 붕괴했다.
            # 미니배치 4개 × 5에폭 = 20번 갱신으로 줄인다 (legged_gym 설정과 비슷).
            batch_size=n_steps * args.n_envs // 4,
            n_epochs=5,
            target_kl=0.02,        # KL이 커지면 그 업데이트의 남은 에폭을 건너뛴다
            learning_rate=lambda progress: 3e-4 * progress,  # 3e-4 → 0 선형 감소
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            tensorboard_log=str(run_dir),
            seed=args.seed,
            device="cpu",
            verbose=1,
        )

    save_every = max(200_000 // args.n_envs, 1)
    callbacks = [
        TrackingLogger(),
        CheckpointCallback(save_freq=save_every, save_path=str(run_dir / "checkpoints"),
                           name_prefix="ppo", save_vecnormalize=True),
        SaveNormCallback(save_every, run_dir),
    ]

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                reset_num_timesteps=not args.resume)
    model.save(run_dir / "final_model")
    venv.save(str(run_dir / "vecnormalize.pkl"))
    print(f"done in {(time.time() - t0) / 60:.1f} min → {run_dir}")


if __name__ == "__main__":
    main()
