"""디딤 패턴 vs 학습된 정책: 지금 보상(v6B)으로 같은 조건에서 채점하고 항별로 나눈다.

질문: 지금 보상은 디딤 보행(CMA-ES로 찾은 PD 고정 패턴, 디딤 0.10 s)을 학습된 B(디딤 0.03 s)보다
높게 쳐 주는가? 낮다면 보상이 디딤을 손해로 본다는 근거, 높다면 학습이 거기까지 못 간 것.

CMA 패턴은 앞(+x)으로만 가는 고정 패턴이라 앞 명령 두 개(0.5, 1.0 m/s)만 비교한다.
조건은 evaluate_repeated.py와 같다: reset seed 0~9 × 1000스텝, 에너지 0.5, 노이즈 켬 seed 오프셋 동일.
CMA 패턴은 열린 루프라 초당 약 5°씩 돌아 50초면 원을 그린다. 그래서 방향이 아직 크게 틀어지지 않은
3~10초 구간(스텝 60~199)의 스텝당 평균도 따로 낸다(w_ 접두사). 공정한 비교는 이쪽이다.

    python reward_compare.py
결과: runs/v8_pretest/reward_compare.json
"""

from __future__ import annotations

import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

V6B = dict(tracking_sigma=1.0, tracking_weight=5.0, limit_weight=1.5, limit_soft=0.8)
PD = dict(action_mode="pd", pd_kp=120.0, pd_kd=21.0)
STEPS, SEEDS, CMDS, W_END = 1000, range(10), (4, 12), 200
OUT = Path("runs/v8_pretest/reward_compare.json")
# v9 사전 시험: 디딤 보상 가중치 (환경 변수로 넘김, 워커 프로세스에도 전달됨). 0이면 v6B 그대로
EXTRA = dict(stance_weight=float(os.environ.get("STANCE_W", 0)), slip_weight=float(os.environ.get("SLIP_W", 0)))
if any(EXTRA.values()):
    OUT = Path(f"runs/v9_pretest/reward_compare_st{EXTRA['stance_weight']:g}_sl{EXTRA['slip_weight']:g}.json")


def cma_params(kp=120.0, restart=2, key="best_score_params"):
    d = json.loads(Path("runs/v8_pretest/cma_summary_long.json").read_text())
    c = next(c for c in d["candidates"] if c["kp"] == kp)
    return next(r for r in c["runs"] if r["restart"] == restart)[key]


CASES = {
    "cma_kp120_r2": ("디딤 패턴 (CMA kp120 재시작 2, 사전 시험 합격)", None, "cma", dict(restart=2)),
    "cma_kp120_r3": ("디딤 패턴 (CMA kp120 재시작 3, 디딤 0.11 s)", None, "cma", dict(restart=3)),
    "pd_stand": ("PD 서기 (목표 = 범위 가운데)", None, "zero", {}),
    "B_off": ("학습된 B, 노이즈 끔", "main_v8Bb", "off", {}),
    "B_on": ("학습된 B, 노이즈 켬", "main_v8Bb", "on", {}),
}
SFX = ("", "_s1", "_s2")


def job(args):
    name, sfx, ci = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    from cma_gait import action as cma_action
    from envs import make_command_ant
    from evaluate_repeated import NOISE_SEED_OFFSET, commands
    _, run, mode, extra = CASES[name]
    policy = None
    if run:
        from policy_utils import Policy
        policy = Policy(f"runs/{run}{sfx}")
    p = cma_params(**extra) if mode == "cma" else None
    cmd = commands()[ci]
    eps = []
    for s in SEEDS:
        if mode == "on":
            torch.manual_seed(NOISE_SEED_OFFSET + s)
        env = make_command_ant(**{**V6B, **PD, **EXTRA, "max_episode_steps": STEPS})
        env.set_command(cmd[0], cmd[1], 0.5)
        obs, _ = env.reset(seed=s)
        dt = env.unwrapped.dt
        tot = dict(ret=0.0, tracking=0.0, alive=0.0, ctrl=0.0, limit=0.0)
        vel, n = [], 0
        win = dict(ret=[], tracking=[], ctrl=[], limit=[], vx=[], bonus=[], slip=[])
        for t in range(STEPS):
            if mode == "cma":
                a = cma_action(t * dt, p)
            elif mode == "zero":
                a = np.zeros(8)
            else:
                a = policy.act(obs, deterministic=mode == "off")
            obs, r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
            u = env.unwrapped
            tot["ret"] += r
            tot["tracking"] += 5.0 * info["r_tracking"]
            tot["alive"] += 0.0 if term else 1.0
            ctrl_w = 0.02 + (1 - 0.5) * (0.25 - 0.02)
            tot["ctrl"] -= ctrl_w * info["effort"]
            tot["limit"] -= 1.5 * info["limit_penalty"]
            if t >= 60:
                vel.append(info["velocity"])
            if 60 <= t < W_END:
                win["ret"].append(r); win["tracking"].append(5.0 * info["r_tracking"])
                win["ctrl"].append(-ctrl_w * info["effort"]); win["limit"].append(-1.5 * info["limit_penalty"])
                win["bonus"].append(info["stance_bonus"]); win["slip"].append(info["slip_sq"])
                win["vx"].append(float(info["velocity"] @ np.asarray(cmd) / np.linalg.norm(cmd)))
            n += 1
            if term or trunc:
                break
        env.close()
        v = np.mean(vel, axis=0) if vel else np.zeros(2)
        along = float(v @ np.asarray(cmd) / np.linalg.norm(cmd))
        eps.append(dict(seed=s, len=n, fell=n < STEPS, along=along, **tot,
                        **{f"w_{k}": float(np.mean(v)) for k, v in win.items()}))
    return name, sfx, ci, eps


def main():
    jobs = [(n, sfx if CASES[n][1] else "", ci) for n in CASES for sfx in (SFX if CASES[n][1] else ("",)) for ci in CMDS]
    with Pool(min(len(jobs), os.cpu_count() or 4)) as pool:
        res = pool.map(job, jobs)
    out = {}
    for name, sfx, ci, eps in res:
        out.setdefault(name, {}).setdefault(str(ci), []).extend(dict(e, run_sfx=sfx) for e in eps)
    keys = ("ret", "tracking", "alive", "ctrl", "limit", "along", "w_ret", "w_tracking", "w_ctrl", "w_limit", "w_vx",
            "w_bonus", "w_slip")
    table = {}
    print(f"{'case':16s} cmd  " + " ".join(f"{k:>9s}" for k in keys) + "   fell")
    for name in CASES:
        for ci in CMDS:
            eps = out[name][str(ci)]
            m = {k: float(np.mean([e[k] for e in eps])) for k in keys}
            m["fell"] = float(np.mean([e["fell"] for e in eps]))
            if CASES[name][1]:
                m["ret_per_seed"] = [float(np.mean([e["ret"] for e in eps if e["run_sfx"] == s])) for s in SFX]
            table[f"{name}/{ci}"] = m
            print(f"{name:16s} {ci:3d}  " + " ".join(f"{m[k]:9.3f}" if k.startswith("w_") else f"{m[k]:9.2f}" for k in keys) + f"   {m['fell']:.2f}"
                  + (f"   seeds {np.round(m['ret_per_seed'], 0).tolist()}" if "ret_per_seed" in m else ""))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(reward="v6B", extra=EXTRA, pd=PD, steps=STEPS, seeds=list(SEEDS), commands=list(CMDS),
                                   cases={k: v[0] for k, v in CASES.items()}, table=table, episodes=out), indent=1))
    print("saved →", OUT)


if __name__ == "__main__":
    main()
