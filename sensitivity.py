"""초기 상태 민감성 실험: 초기 관절 위치/속도를 아주 조금(기본 1e-9) 흔들면 걷기/굳기가 바뀌는가?

같은 기계, 같은 정책, 같은 명령, 노이즈 끔(deterministic), 같은 평가 seed에서
reset 직후 관절 상태에만 eps 크기의 무작위 값을 더한다.
  - base : 흔들지 않음 (흔들기와 같은 코드 경로, eps=0 → evaluate.py 결과와 같아야 함)
  - qpos : 관절 각도 8개에 eps · N(0, 1)
  - qvel : 관절 각속도 8개에 eps · N(0, 1)
결과가 바뀌면 "정책의 결과가 초기 상태에 매우 민감하다"는 것까지만 말할 수 있다.
(CPU 차이가 기계 간 차이의 원인이라는 증명은 아님)

판정 (evaluate.py와 같은 300스텝, 워밍업 60):
  - 이동: 명령 방향 속도 > 0.05 m/s, 넘어지지 않음
  - 끝에서 굳음(임시 기준): 마지막 2초 동안 관절 8개가 모두 한계 2° 이내, 관절 각속도 평균 < 0.01 rad/s

    python sensitivity.py                  # v5, v5b × seed 0, 1, 2
결과: runs/sensitivity/sensitivity.json
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

RUNS = ["main_v5", "main_v5_s1", "main_v5_s2", "main_v5b", "main_v5b_s1", "main_v5b_s2"]
STEPS, WARMUP, TAIL = 300, 60, 40


def rollout(policy, cmd, kind="base", eps=0.0, pert_seed=0, eval_seed=0):
    import mujoco
    from envs import make_command_ant

    env = make_command_ant(max_episode_steps=STEPS + 10)
    env.set_command(cmd[0], cmd[1], 0.5)
    obs, _ = env.reset(seed=eval_seed)
    u = env.unwrapped
    m, d = u.model, u.data
    noise = eps * np.random.default_rng(pert_seed).standard_normal(8)
    if kind == "qpos":
        d.qpos[7:15] += noise
    elif kind == "qvel":
        d.qvel[6:14] += noise
    mujoco.mj_forward(m, d)
    obs = env._augment(u._get_obs())

    lo, hi = np.degrees(m.jnt_range[1:9, 0]), np.degrees(m.jnt_range[1:9, 1])
    vels, q, qd, fell = [], [], [], False
    for t in range(STEPS):
        obs, _, term, _, info = env.step(policy.act(obs, deterministic=True))
        if t >= WARMUP:
            vels.append(info["velocity"])
            q.append(np.degrees(d.qpos[7:15]))
            qd.append(np.abs(d.qvel[6:14]))
        if term:
            fell = True
            break
    env.close()
    c = np.asarray(cmd)
    along = float(np.mean(vels, axis=0) @ c / np.linalg.norm(c)) if vels else 0.0
    q_t, qd_t = np.array(q[-TAIL:]), np.array(qd[-TAIL:])
    at_limit = (np.abs(q_t - lo) < 2) | (np.abs(q_t - hi) < 2)
    frozen = bool(len(q_t) == TAIL and at_limit.all() and qd_t.mean() < 0.01)
    return dict(along=along, fell=fell, moving=bool(along > 0.05 and not fell), frozen_end=frozen)


def run_one(job):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    from policy_utils import Policy

    run, n_pert, eps = job
    policy = Policy(Path("runs") / run)
    ref = json.loads((Path("runs") / run / "results" / "summary.json").read_text())["tracking"]
    out = []
    for i, r in enumerate(ref):
        cmd = r["cmd"]
        rec = dict(idx=i, cmd=cmd, repo_along=r["along_cmd_speed"],
                   base=rollout(policy, cmd))
        for kind in ("qpos", "qvel"):
            rec[kind] = [rollout(policy, cmd, kind, eps, pert_seed=k) for k in range(n_pert)]
        out.append(rec)
    return run, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=RUNS)
    p.add_argument("--n-pert", type=int, default=10)
    p.add_argument("--eps", type=float, default=1e-9)
    p.add_argument("--out", default="runs/sensitivity")
    args = p.parse_args()

    with Pool(min(len(args.runs), 6)) as pool:
        results = dict(pool.map(run_one, [(r, args.n_pert, args.eps) for r in args.runs]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sensitivity.json").write_text(json.dumps(
        dict(eps=args.eps, n_pert=args.n_pert, runs=results), indent=1))

    print(f"eps = {args.eps:g}, 명령당 흔들기 {args.n_pert}번 × (qpos, qvel)")
    for run, recs in results.items():
        flips = {k: 0 for k in ("qpos", "qvel")}
        unstable = {k: 0 for k in ("qpos", "qvel")}
        for rec in recs:
            for k in ("qpos", "qvel"):
                n_diff = sum(x["moving"] != rec["base"]["moving"] for x in rec[k])
                flips[k] += n_diff
                unstable[k] += n_diff > 0
        print(f"[{run}] 기준 이동 {sum(r['base']['moving'] for r in recs)}/16, "
              f"끝에서 굳음 {sum(r['base']['frozen_end'] for r in recs)}/16 | "
              f"이동 여부가 기준과 달라진 흔들기: qpos {flips['qpos']}/{16 * args.n_pert}, "
              f"qvel {flips['qvel']}/{16 * args.n_pert} | "
              f"결과가 한 번이라도 바뀐 명령: qpos {unstable['qpos']}/16, qvel {unstable['qvel']}/16")
    print(f"saved → {out / 'sensitivity.json'}")


if __name__ == "__main__":
    main()
