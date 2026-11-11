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
from envs.run_config import check_resume_compatible, env_kwargs_from_config, load_env_kwargs
from squashed_policy import SquashedGaussianPolicy


class TrackingLogger(BaseCallback):
    """속도 추종 오차와 제어 노력을 텐서보드에 기록한다."""

    def __init__(self):
        super().__init__()
        self._err, self._effort, self._limit, self._vz2 = [], [], [], []
        self._power, self._tlim, self._tabs = [], [], []
        self._bonus, self._slip = [], []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "tracking_error" in info:
                self._err.append(info["tracking_error"])
                self._effort.append(info["effort"])
                self._limit.append(info.get("limit_penalty", 0.0))
                self._vz2.append(info.get("vz2", 0.0))
                self._power.append(info["power"])
                self._tlim.append(info["torque_limit_frac"])
                self._tabs.append(float(np.mean(np.abs(info["torque"]))))
                self._bonus.append(info.get("stance_bonus", 0.0))
                self._slip.append(info.get("slip_sq", 0.0))
        return True

    def _on_rollout_end(self) -> None:
        if self._err:
            self.logger.record("command/tracking_error_mps", float(np.mean(self._err)))
            self.logger.record("command/effort", float(np.mean(self._effort)))
            # 관절 한계 페널티 원값 P (rad, 가중치 곱하기 전). 보상에서 빠지는 양은 limit_weight × P
            self.logger.record("command/limit_penalty", float(np.mean(self._limit)))
            # 몸통 수직 속도 제곱 v_z² (가중치 곱하기 전). 보상에서 빠지는 양은 vz_weight × v_z²
            self.logger.record("command/vz2", float(np.mean(self._vz2)))
            # v8: 실제 적용 토크 기준 (모든 갈래 공통)
            self.logger.record("command/power", float(np.mean(self._power)))
            self.logger.record("command/torque_limit_frac", float(np.mean(self._tlim)))
            self.logger.record("command/mean_abs_torque", float(np.mean(self._tabs)))
            # v9: 디딤 보너스·미끄러짐 (가중치 곱하기 전, 제어 스텝당). 가중치 0인 run은 0
            self.logger.record("command/stance_bonus", float(np.mean(self._bonus)))
            self.logger.record("command/slip_sq", float(np.mean(self._slip)))
        self._err.clear()
        self._effort.clear()
        self._limit.clear()
        self._vz2.clear()
        for x in (self._power, self._tlim, self._tabs, self._bonus, self._slip):
            x.clear()


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
    p.add_argument("--limit-weight", type=float, default=0.0,
                   help="관절 한계 페널티 가중치 (기본 0 = 기존 보상, v6: 1.5)")
    p.add_argument("--limit-soft", type=float, default=0.8,
                   help="페널티 시작 위치: 중심 ± 이 값 × 반범위 (0.8 = 범위 끝 20%%)")
    p.add_argument("--vz-weight", type=float, default=0.0,
                   help="몸통 수직 속도 페널티 가중치 × v_z² (기본 0 = 기존 보상, v7A: 1.0, v7B: 3.0)")
    p.add_argument("--squash", action="store_true",
                   help="행동을 tanh로 (-1, 1)에 묶는 정책 사용 (squashed_policy.py, v5)")
    p.add_argument("--constant-lr", type=float, default=None,
                   help="학습률을 이 값으로 고정 (기본: 3e-4 → 0 선형 감소). --resume과 함께 쓰면 이어 학습에도 적용")
    p.add_argument("--action-mode", choices=["torque", "pd"], default="torque",
                   help="행동 방식 (v8): torque = 행동 × 150 N·m, pd = 관절 목표 각도 + 물리 스텝마다 PD")
    p.add_argument("--pd-kp", type=float, default=0.0, help="PD 비례 이득 [N·m/rad] (--action-mode pd)")
    p.add_argument("--pd-kd", type=float, default=0.0, help="PD 미분 이득 [N·m·s/rad] (--action-mode pd)")
    p.add_argument("--tracking-kernel", choices=["single", "two_term"], default="single",
                   help="추종 커널 (v8): single = exp(-e²/σ), two_term = 넓은 항 + 좁은 항(σ 0.25) 반씩")
    p.add_argument("--stance-weight", type=float, default=0.0,
                   help="디딤 보상 가중치 (v9, envs/stance.py). 0이면 기존 보상")
    p.add_argument("--slip-weight", type=float, default=0.0,
                   help="하중 발 미끄러짐 속도² 페널티 가중치 (v9). 0이면 기존 보상")
    p.add_argument("--log-std-init", type=float, default=None,
                   help="정책 탐색 노이즈 초기 log σ (기본: SB3 기본 0 = σ 1.0, v8C: ln 0.1)")
    p.add_argument("--save-init", action="store_true",
                   help="학습 전 정책을 runs/<run-name>_init/ 에 저장 (학습 전 측정용)")
    args = p.parse_args()

    env_kwargs = env_kwargs_from_config(vars(args))
    if args.resume:   # 이어 학습이 행동 방식·이득·커널을 바꾸면 정책이 배운 것의 뜻이 달라진다 → 멈춘다
        check_resume_compatible(load_env_kwargs(args.resume), env_kwargs)

    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    def make(rank: int):
        def _init():
            return make_command_ant(seed=args.seed + rank, **env_kwargs)
        return _init

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = VecMonitor(vec_cls([make(i) for i in range(args.n_envs)]))

    if args.resume:
        resume = Path(args.resume)
        venv = VecNormalize.load(str(resume / "vecnormalize.pkl"), venv)
        model = PPO.load(resume / "final_model.zip", env=venv, device="cpu",
                         tensorboard_log=str(run_dir))
        if args.constant_lr is not None:
            model.learning_rate = args.constant_lr
            model._setup_lr_schedule()
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        n_steps = 1024             # 환경당 수집 스텝 → 한 번에 1024 * n_envs
        model = PPO(
            SquashedGaussianPolicy if args.squash else "MlpPolicy",
            venv,
            n_steps=n_steps,
            # v1에서는 batch 256이라 업데이트 한 번에 미니배치가 48개 × 10에폭 = 480번 갱신
            # → 정책이 너무 크게 바뀌어 approx_kl 4.1, clip_fraction 0.87로 붕괴했다.
            # 미니배치 4개 × 5에폭 = 20번 갱신으로 줄인다 (legged_gym 설정과 비슷).
            batch_size=n_steps * args.n_envs // 4,
            n_epochs=5,
            target_kl=0.02,        # KL이 커지면 그 업데이트의 남은 에폭을 건너뛴다
            learning_rate=(args.constant_lr if args.constant_lr is not None
                           else lambda progress: 3e-4 * progress),  # 기본: 3e-4 → 0 선형 감소
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                               **({} if args.log_std_init is None else dict(log_std_init=args.log_std_init))),
            tensorboard_log=str(run_dir),
            seed=args.seed,
            device="cpu",
            verbose=1,
        )

    if args.save_init and not args.resume:   # 학습 전 정책: 평범한 run 폴더로 저장해 기존 평가 도구를 그대로 쓴다
        init_dir = Path("runs") / f"{run_name}_init"
        init_dir.mkdir(parents=True, exist_ok=True)
        model.save(init_dir / "final_model")
        venv.save(str(init_dir / "vecnormalize.pkl"))
        (init_dir / "config.json").write_text(json.dumps(dict(vars(args), run_name=f"{run_name}_init"), indent=2))

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
