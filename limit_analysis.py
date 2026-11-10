"""관절이 한계에 붙어 있을 때, 정책이 그 한계 쪽으로 토크를 계속 거는가?

Ant의 행동은 목표 각도가 아니라 토크(모터 gear 150, 행동 ±1, 양수 = 관절 각도 증가 방향)다.
관절이 한계 근처(±2°)에 있는 스텝에서
  - 한계 쪽으로 미는 토크 비율 (아래 한계면 a < 0, 위 한계면 a > 0)
  - 그때의 평균 |a|
를 관절별로 잰다. 미는 비율이 높으면 정책(=보상)이 그 자세를 적극적으로 선택한다는 뜻이고,
낮으면 동역학상 그쪽으로 떨어지는 자세라는 뜻이다.

    python limit_analysis.py
결과: runs/limit_analysis/summary.json
"""

from __future__ import annotations

import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from evaluate_repeated import JOINTS, commands

RUNS = ["runs/main_v5b", "runs/main_v5b_s1", "runs/main_v5b_s2"]
STEPS, WARMUP, MARGIN = 1000, 60, 2.0
RESET_SEEDS = [0, 1]


def job(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    from envs import make_command_ant
    from policy_utils import Policy

    run, det = args
    policy = Policy(run)
    lo = hi = None
    near_lo, near_hi, acts = [], [], []
    for cmd in commands():
        for s in RESET_SEEDS:
            if not det:
                torch.manual_seed(10_000 + s)
            env = make_command_ant(max_episode_steps=STEPS)
            env.set_command(cmd[0], cmd[1], 0.5)
            obs, _ = env.reset(seed=s)
            m, d = env.unwrapped.model, env.unwrapped.data
            lo, hi = np.degrees(m.jnt_range[1:9, 0]), np.degrees(m.jnt_range[1:9, 1])
            for t in range(STEPS):
                a = np.clip(policy.act(obs, deterministic=det), -1, 1)
                q = np.degrees(d.qpos[7:15])          # 행동을 고르는 시점의 관절 각도
                obs, _, term, trunc, _ = env.step(a)
                if t >= WARMUP:
                    near_lo.append(np.abs(q - lo) < MARGIN)
                    near_hi.append(np.abs(q - hi) < MARGIN)
                    acts.append(a)
                if term or trunc:
                    break
            env.close()
    near_lo, near_hi, acts = map(np.array, (near_lo, near_hi, acts))
    out = {}
    for j, name in enumerate(JOINTS):
        at = near_lo[:, j] | near_hi[:, j]
        push = (near_lo[:, j] & (acts[:, j] < 0)) | (near_hi[:, j] & (acts[:, j] > 0))
        free = ~at
        out[name] = dict(
            at_limit=float(at.mean()),
            at_lower=float(near_lo[:, j].mean()),
            at_upper=float(near_hi[:, j].mean()),
            push_ratio_when_at_limit=float(push[at].mean()) if at.any() else None,
            mean_abs_action_at_limit=float(np.abs(acts[at, j]).mean()) if at.any() else None,
            mean_abs_action_free=float(np.abs(acts[free, j]).mean()) if free.any() else None,
        )
    return run, "deterministic" if det else "stochastic", out


def main():
    jobs = [(r, det) for r in RUNS for det in (True, False)]
    with Pool(len(jobs)) as pool:
        res = pool.map(job, jobs)
    summary = {}
    for run, mode, out in res:
        summary.setdefault(Path(run).name, {})[mode] = out
    path = Path("runs/limit_analysis")
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps(summary, indent=1))

    for mode in ("deterministic", "stochastic"):
        print(f"### {mode} (v5b seed 0 / 1 / 2)")
        print("| 관절 | 한계 근처 비율 | 아래 / 위 한계 | 한계에서 한계 쪽으로 미는 비율 | 한계에서 평균 \\|a\\| | 한계 밖 평균 \\|a\\| |")
        print("| --- | --- | --- | --- | --- | --- |")
        for name in JOINTS:
            rows = [summary[Path(r).name][mode][name] for r in RUNS]
            f = lambda k: " / ".join("–" if x[k] is None else f"{x[k]:.2f}" for x in rows)
            lu = " / ".join(f"{x['at_lower']:.2f}·{x['at_upper']:.2f}" for x in rows)
            print(f"| {name} | {f('at_limit')} | {lu} | {f('push_ratio_when_at_limit')} | "
                  f"{f('mean_abs_action_at_limit')} | {f('mean_abs_action_free')} |")
        print()
    print(f"saved → {path / 'summary.json'}")


if __name__ == "__main__":
    main()
