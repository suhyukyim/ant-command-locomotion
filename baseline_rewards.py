"""기준선 보상 비교: 학습된 정책이 '아무것도 안 하기'나 '그냥 서 있기'보다 보상을 더 받는가?

v6B 보상(--tracking-sigma 1.0 --tracking-weight 5.0 --limit-weight 1.5 --limit-soft 0.8)으로,
학습 때와 같은 랜덤 명령(에피소드마다 랜덤, 250스텝마다 재샘플)에서 1000스텝 × 8 에피소드를 굴려
에피소드 보상 합(= 학습 로그 ep_rew_mean과 같은 단위)을 잰다.

정책 6가지:
  - v6Bb 노이즈 켬 / 끔             학습된 정책 (runs/main_v6Bb)
  - 토크 0                          a = 0
  - PD 서기                         관절을 범위 가운데에 두는 손으로 짠 PD (kp 50, kd 10), 결과를 토크 행동으로 넣음
  - 무작위, 행동 = 토크             a = tanh(σ z), σ = 1.0  (학습 시작 때 정책과 같은 노이즈 크기)
  - 무작위, 행동 = 관절 목표 각도   같은 a를 목표 각도(범위 가운데 + a × 반범위)로 해석해 PD로 토크를 만듦

주의: Ant의 행동 순서(hip_4, ankle_4, hip_1, …)는 관절 순서(hip_1, ankle_1, …)와 다르다.
PD 토크는 관절 순서로 계산한 뒤 `actuator_trnid`로 행동 순서로 바꾼다.

    python baseline_rewards.py
결과: runs/baselines/summary.json
"""

from __future__ import annotations

import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

V6B = dict(tracking_sigma=1.0, tracking_weight=5.0, limit_weight=1.5, limit_soft=0.8)
RUN = "runs/main_v6Bb"
EPISODES, STEPS, GEAR = 8, 1000, 150.0
KP, KD, SIGMA = 50.0, 10.0, 1.0

CASES = [
    ("v6Bb_noise_on", "학습된 v6Bb, 노이즈 켬 (PPO가 실제로 최적화하는 값)"),
    ("zero", "토크 0, 아무것도 안 함"),
    ("v6Bb_noise_off", "학습된 v6Bb, 노이즈 끔"),
    ("pd_stand", "손으로 짠 PD로 관절을 범위 가운데에 두고 서 있기"),
    ("random_torque", "학습 안 된 무작위 정책, 행동 = 토크 (σ 1.0)"),
    ("random_target", "학습 안 된 무작위 정책, 행동 = 관절 목표 각도 (σ 1.0)"),
]


def job(case):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    from envs import make_command_ant

    policy = None
    if case.startswith("v6Bb"):
        from policy_utils import Policy
        policy = Policy(RUN)

    rets, lens, torque = [], [], []
    for ep in range(EPISODES):
        rng = np.random.default_rng(500 + ep)
        torch.manual_seed(2000 + ep)
        env = make_command_ant(max_episode_steps=STEPS + 10, **V6B)
        obs, _ = env.reset(seed=100 + ep)
        m, d = env.unwrapped.model, env.unwrapped.data
        act_to_joint = m.actuator_trnid[:, 0] - 1          # 행동 i가 움직이는 관절(qpos 순서) 번호
        rng_j = m.jnt_range[1:9]
        center, half = rng_j.mean(axis=1), (rng_j[:, 1] - rng_j[:, 0]) / 2

        def pd(target):                                    # 관절 순서 목표 → 행동 순서 토크 행동
            tau = KP * (target - d.qpos[7:15]) - KD * d.qvel[6:14]
            return np.clip(tau[act_to_joint] / GEAR, -1, 1)

        R, n = 0.0, 0
        for _ in range(STEPS):
            if case == "v6Bb_noise_on":
                a = policy.act(obs, deterministic=False)
            elif case == "v6Bb_noise_off":
                a = policy.act(obs, deterministic=True)
            elif case == "zero":
                a = np.zeros(8)
            elif case == "pd_stand":
                a = pd(center)
            elif case == "random_torque":
                a = np.tanh(SIGMA * rng.standard_normal(8))
            elif case == "random_target":
                a = pd(center + np.tanh(SIGMA * rng.standard_normal(8)) * half)
            a = np.clip(np.asarray(a, dtype=np.float32), -1, 1)
            obs, r, term, _, _ = env.step(a)
            R += r
            n += 1
            torque.append(np.abs(a).mean() * GEAR)
            if term:
                break
        rets.append(R)
        lens.append(n)
        env.close()
    return case, dict(ep_rew_mean=float(np.mean(rets)), ep_rew_std=float(np.std(rets)),
                      ep_len_mean=float(np.mean(lens)), mean_abs_torque_nm=float(np.mean(torque)))


def main():
    with Pool(len(CASES)) as pool:
        res = dict(pool.map(job, [c for c, _ in CASES]))
    out = Path("runs/baselines")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(
        dict(reward="v6B", run=RUN, episodes=EPISODES, steps=STEPS, kp=KP, kd=KD, sigma=SIGMA, results=res), indent=1))
    print("| 정책 | 에피소드 보상 (평균 ± 표준편차) | 버틴 스텝 | 평균 \\|토크\\| (N·m) |")
    print("| --- | --- | --- | --- |")
    for c, label in CASES:
        x = res[c]
        print(f"| {label} | {x['ep_rew_mean']:.0f} ± {x['ep_rew_std']:.0f} | {x['ep_len_mean']:.0f} | "
              f"{x['mean_abs_torque_nm']:.1f} |")
    print(f"\nsaved → {out / 'summary.json'}")


if __name__ == "__main__":
    main()
