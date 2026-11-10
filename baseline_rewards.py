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


# ---------------------------------------------------------------- 공식 평가 조건 (v8)
REWARDS = {"v6B": V6B, "v6B_two_term": dict(V6B, tracking_kernel="two_term")}
SEED_RUNS = ["runs/main_v6Bb", "runs/main_v6Bb_s1", "runs/main_v6Bb_s2"]


def official_job(args):
    """(정책, 명령) 하나: reset seed 0~9 × 1000스텝의 에피소드 보상. evaluate_repeated.py와 같은 조건."""
    case, run, ci, reward, kp, kd = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    from envs import make_command_ant
    from evaluate_repeated import NOISE_SEED_OFFSET, commands
    cmd = commands()[ci]
    policy = None
    kw = dict(REWARDS[reward])
    if run:
        from policy_utils import Policy
        policy = Policy(run)
        kw = {**policy.env_kwargs, **kw}          # 행동 방식은 run 것 그대로, 보상만 바꿔 잰다
    if case == "pd_mode":
        kw.update(action_mode="pd", pd_kp=kp, pd_kd=kd)
    out = []
    for s in range(10):
        if case == "noise_on":
            torch.manual_seed(NOISE_SEED_OFFSET + s)
        env = make_command_ant(**{**kw, "max_episode_steps": STEPS})
        env.set_command(cmd[0], cmd[1], 0.5)
        obs, _ = env.reset(seed=s)
        m, d = env.unwrapped.model, env.unwrapped.data
        j = m.actuator_trnid[:, 0] - 1
        center = m.jnt_range[1:9].mean(axis=1)
        R, n, tq = 0.0, 0, []
        for _ in range(STEPS):
            if case in ("noise_on", "noise_off"):
                a = policy.act(obs, deterministic=case == "noise_off")
            elif case == "pd_stand":       # 손으로 짠 PD를 토크 행동으로 (kp 50, kd 10)
                a = np.clip((KP * (center - d.qpos[7:15]) - KD * d.qvel[6:14])[j] / GEAR, -1, 1)
            else:                          # zero, pd_mode (a = 0 = 관절 범위 가운데 목표)
                a = np.zeros(8)
            obs, r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
            R += r; n += 1; tq.append(float(np.mean(np.abs(info["torque"]))))
            if term or trunc:
                break
        out.append(dict(seed=s, ret=R, len=n, fell=n < STEPS, torque=float(np.mean(tq))))
        env.close()
    return case, run, ci, out


def official(reward, kp, kd):
    cases = [("zero", None), ("pd_stand", None)] + ([("pd_mode", None)] if kp else [])
    cases += [(m, r) for r in SEED_RUNS for m in ("noise_off", "noise_on")]
    jobs = [(c, r, ci, reward, kp, kd) for c, r in cases for ci in range(16)]
    with Pool() as pool:
        res = pool.map(official_job, jobs, chunksize=1)
    table = {}
    for case, run, ci, out in res:
        for e in out:
            table.setdefault((case, run), {})[(ci, e["seed"])] = e
    ref = table[("pd_stand", None)]
    rows = {}
    for (case, run), E in table.items():
        keys = sorted(E)
        r = np.array([E[k]["ret"] for k in keys])
        rows[f"{case}" + (f"|{Path(run).name}" if run else "")] = dict(
            ep_rew_mean=float(r.mean()),
            ep_rew_0_5=float(np.mean([E[k]["ret"] for k in keys if k[0] < 8])),
            ep_rew_1_0=float(np.mean([E[k]["ret"] for k in keys if k[0] >= 8])),
            ep_len_mean=float(np.mean([E[k]["len"] for k in keys])),
            fell=float(np.mean([E[k]["fell"] for k in keys])),
            mean_abs_torque_nm=float(np.mean([E[k]["torque"] for k in keys])),
            frac_below_pd_stand=float(np.mean([E[k]["ret"] < ref[k]["ret"] for k in keys])))
    for m in ("noise_off", "noise_on"):
        v = [rows[f"{m}|{Path(r).name}"]["ep_rew_mean"] for r in SEED_RUNS]
        rows[f"{m}|v6Bb_seed_mean"] = dict(ep_rew_mean=float(np.mean(v)), ep_rew_std=float(np.std(v, ddof=1)))
    out = Path("runs/baselines")
    out.mkdir(parents=True, exist_ok=True)
    name = f"official_{reward}" + (f"_kp{kp:g}_kd{kd:g}" if kp else "") + ".json"
    (out / name).write_text(json.dumps(dict(reward=reward, condition="명령 16개 × reset seed 0~9 × 1000스텝, 에너지 0.5",
                                            pd_kp=kp, pd_kd=kd, results=rows), indent=1))
    for k, v in rows.items():
        print(k, {kk: round(vv, 3) for kk, vv in v.items()})
    print("saved →", out / name)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--official", action="store_true", help="공식 평가 조건으로 (v8)")
    p.add_argument("--reward", choices=list(REWARDS), default="v6B")
    p.add_argument("--pd-kp", type=float, default=0.0, help="주면 PD 행동 방식 a = 0 서기도 잰다")
    p.add_argument("--pd-kd", type=float, default=0.0)
    a = p.parse_args()
    if a.official:
        return official(a.reward, a.pd_kp, a.pd_kd)
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
