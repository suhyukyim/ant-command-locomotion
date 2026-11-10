"""PD 사전 시험 (docs/V8_DESIGN.md 8절): 학습 전에 PD 이득을 정한다. 학습은 하지 않는다.

후보 (kp, kd): (30, 10), (60, 15), (120, 21). 후보마다
  T1 서기         a = 0 (관절 범위 가운데 목표), 50초 × reset seed 0~9
                  통과: 넘어짐 0, 한계 ±2° 비율 0, 평균 |토크| < 2 N·m
  T2 무작위 목표  a = tanh(σ z), σ 1.0 (학습 전 정책의 대역), 50초 × reset seed 0~9
                  통과: 넘어진 에피소드 ≤ 5%, 토크 제한 비율 < 5%
  T3 손으로 짠 걸음  시간만의 함수인 대각선 교대(trot) 목표 각도, 15초씩, 처음 3초 제외, +x 방향으로 잼
                  통과: 아래를 모두 채우는 패턴이 하나 이상
                    +x 속도 ≥ 0.3 m/s, 하중 디딤 평균 지속 ≥ 0.10 s, 하중 디딤 중 0.1 m/s 넘게 미끄러지는 비율 ≤ 0.5,
                    토크 제한 비율 < 5%, 넘어짐 없음
선택: 셋 다 통과한 후보 중 T3 최고 속도(통과 패턴 기준)가 가장 높은 것. 10% 이내로 비슷하면 kp가 낮은 쪽. 없으면 null.
하중 디딤·미끄러짐 정의는 stance_load_analysis.py와 같다 (발 하나 F_z ≥ 0.25 mg).
토크 제한 비율 = (관절, 물리 스텝) 중 |τ| ≥ 0.99 × 150 N·m 비율.

    python pd_pretest.py            # 결과 runs/v8_pretest/summary.json
    python pd_pretest.py --quick    # 흐름 확인용 (후보 1개, T3 패턴 2개)
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

CANDIDATES = [(30.0, 10.0), (60.0, 15.0), (120.0, 21.0)]
OUT = Path("runs/v8_pretest")
T12_STEPS, T3_STEPS, T3_SKIP = 1000, 300, 60          # 50초, 15초, 처음 3초
LOAD_THR, DT_PHYS = 0.25, 0.01
LIMIT_MARGIN = math.radians(2.0)
PASS = dict(t1_torque=2.0, t2_fell=0.05, torque_limit=0.05,
            t3_speed=0.3, t3_load_dur=0.10, t3_slip=0.5)
# T3 패턴 훑기: 주기, 엉덩이 진폭, 발목 진폭(행동 단위), 발목 위상차, 다리별 엉덩이 부호(진행 방향)
PERIODS, HIP_AMPS, ANK_AMPS = (0.4, 0.6, 0.8, 1.0), (0.3, 0.6, 0.9), (0.3, 0.6, 0.9)
ANK_PHASES = (math.pi / 2, -math.pi / 2)
HIP_SIGNS = list(itertools.product((1, -1), repeat=4))
# 다리 번호(1~4) → 행동 번호 (엉덩이, 발목). Ant 행동 순서: hip_4, ankle_4, hip_1, ankle_1, hip_2, ankle_2, hip_3, ankle_3
LEG_ACT = {4: (0, 1), 1: (2, 3), 2: (4, 5), 3: (6, 7)}
LEG_PHASE = {1: 0.0, 3: 0.0, 2: math.pi, 4: math.pi}   # 대각선 쌍(1·3 / 2·4)이 반 주기씩 교대


def make(kp, kd, steps):
    from envs import make_command_ant
    env = make_command_ant(action_mode="pd", pd_kp=kp, pd_kd=kd, max_episode_steps=steps)
    env.set_command(1.0, 0.0, 0.5)
    return env


def at_limit(env):
    m, d = env.unwrapped.model, env.unwrapped.data
    q, r = d.qpos[7:15], m.jnt_range[1:9]
    return float(np.mean((np.abs(q - r[:, 0]) < LIMIT_MARGIN) | (np.abs(q - r[:, 1]) < LIMIT_MARGIN)))


def t12_job(args):
    test, kp, kd, seed = args
    env = make(kp, kd, T12_STEPS)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    tq, lim, tl, n, fell = [], [], [], 0, False
    for _ in range(T12_STEPS):
        a = np.zeros(8) if test == "T1" else np.tanh(rng.standard_normal(8))
        _, _, term, trunc, info = env.step(a.astype(np.float32))
        tq.append(float(np.mean(np.abs(info["torque"])))); tl.append(info["torque_limit_frac"]); lim.append(at_limit(env))
        n += 1
        if term:
            fell = True
            break
        if trunc:
            break
    env.close()
    return test, kp, seed, dict(fell=fell, steps=n, mean_abs_torque=float(np.mean(tq)),
                                torque_limit_frac=float(np.mean(tl)), limit_occ=float(np.mean(lim)))


def pattern_action(t, period, ah, aa, dphi, signs):
    a = np.zeros(8)
    w = 2 * math.pi * t / period
    for k, leg in enumerate((1, 2, 3, 4)):
        hi, ai = LEG_ACT[leg]
        a[hi] = signs[k] * ah * math.sin(w + LEG_PHASE[leg])
        a[ai] = aa * math.sin(w + LEG_PHASE[leg] + dphi)
    return a


def t3_job(args):
    kp, kd, period, ah, aa, dphi, signs = args
    from propulsion_analysis import Recorder
    from stance_load_analysis import load_stats
    env = make(kp, kd, T3_STEPS + 10)
    rec = Recorder(env, np.array([1.0, 0.0]))
    env.reset(seed=0)
    dt = env.unwrapped.dt
    mg = float(env.unwrapped.model.body_mass.sum()) * 9.81
    vel, tl, lim, fell = [], [], [], False
    for t in range(T3_STEPS):
        rec.on = t >= T3_SKIP
        _, _, term, _, info = env.step(pattern_action(t * dt, period, ah, aa, dphi, signs).astype(np.float32))
        if t >= T3_SKIP:
            vel.append(info["velocity"]); tl.append(info["torque_limit_frac"]); lim.append(at_limit(env))
        if term:
            fell = True
            break
    env.close()
    res = dict(period=period, hip_amp=ah, ankle_amp=aa, ankle_phase=dphi, hip_signs=list(signs), fell=fell)
    if fell or not rec.rows:
        return kp, dict(res, speed_x=float("nan"), passed=False)
    con, fz, fa, slip, _, _ = (np.array(x) for x in zip(*rec.rows))
    ld = load_stats(dict(con=con, fz=fz / mg, fa=fa, slip=slip), LOAD_THR)
    v = np.mean(vel, axis=0)
    res.update(speed_x=float(v[0]), speed_any=float(np.linalg.norm(v)),
               load_dur_mean=float(ld[:, 0].mean() * DT_PHYS) if len(ld) else 0.0,
               load_slip_gt_0_1=float(np.nanmean(ld[:, 2])) if len(ld) else float("nan"),
               torque_limit_frac=float(np.mean(tl)), limit_occ=float(np.mean(lim)))
    res["passed"] = bool(res["speed_x"] >= PASS["t3_speed"] and res["load_dur_mean"] >= PASS["t3_load_dur"]
                         and res["load_slip_gt_0_1"] <= PASS["t3_slip"] and res["torque_limit_frac"] < PASS["torque_limit"])
    return kp, res


def choose(summary):
    ok = [c for c in summary["candidates"] if c["T1"]["passed"] and c["T2"]["passed"] and c["T3"]["passed"]]
    if not ok:
        return None
    best = max(c["T3"]["best_passing_speed"] for c in ok)
    close = [c for c in ok if c["T3"]["best_passing_speed"] >= 0.9 * best]
    c = min(close, key=lambda c: c["kp"])
    return dict(kp=c["kp"], kd=c["kd"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cands = CANDIDATES[1:2] if a.quick else CANDIDATES
    seeds = range(2) if a.quick else range(10)
    grid = list(itertools.product(PERIODS, HIP_AMPS, ANK_AMPS, ANK_PHASES, HIP_SIGNS))
    if a.quick:
        grid = grid[:2]
    jobs12 = [(t, kp, kd, s) for kp, kd in cands for t in ("T1", "T2") for s in seeds]
    jobs3 = [(kp, kd, *g) for kp, kd in cands for g in grid]
    with Pool() as pool:
        r12 = pool.map(t12_job, jobs12)
        r3 = pool.map(t3_job, jobs3, chunksize=4)
    summary = dict(criteria=PASS, load_thr_mg=LOAD_THR, candidates=[])
    for kp, kd in cands:
        t1 = [r for t, k, _, r in r12 if t == "T1" and k == kp]
        t2 = [r for t, k, _, r in r12 if t == "T2" and k == kp]
        t3 = [r for k, r in r3 if k == kp]
        passing = [r for r in t3 if r["passed"]]
        t1s = dict(fell=float(np.mean([r["fell"] for r in t1])), limit_occ=float(np.mean([r["limit_occ"] for r in t1])),
                   mean_abs_torque=float(np.mean([r["mean_abs_torque"] for r in t1])))
        t1s["passed"] = t1s["fell"] == 0 and t1s["limit_occ"] == 0 and t1s["mean_abs_torque"] < PASS["t1_torque"]
        t2s = dict(fell=float(np.mean([r["fell"] for r in t2])), mean_abs_torque=float(np.mean([r["mean_abs_torque"] for r in t2])),
                   torque_limit_frac=float(np.mean([r["torque_limit_frac"] for r in t2])),
                   limit_occ=float(np.mean([r["limit_occ"] for r in t2])))
        t2s["passed"] = t2s["fell"] <= PASS["t2_fell"] and t2s["torque_limit_frac"] < PASS["torque_limit"]
        ok3 = [r for r in t3 if not r["fell"] and not math.isnan(r["speed_x"])]
        best_any = max(ok3, key=lambda r: r["speed_x"]) if ok3 else None
        t3s = dict(n_patterns=len(t3), n_fell=sum(r["fell"] for r in t3), n_passed=len(passing),
                   passed=bool(passing),
                   best_passing_speed=max((r["speed_x"] for r in passing), default=0.0),
                   best_passing=max(passing, key=lambda r: r["speed_x"]) if passing else None,
                   fastest_any=best_any,
                   longest_load=max(ok3, key=lambda r: r["load_dur_mean"]) if ok3 else None)
        summary["candidates"].append(dict(kp=kp, kd=kd, T1=t1s, T2=t2s, T3=t3s))
    summary["chosen"] = choose(summary)
    out = OUT / ("quick.json" if a.quick else "summary.json")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("t3_quick.json" if a.quick else "t3_all.json")).write_text(json.dumps(r3, indent=0))
    out.write_text(json.dumps(summary, indent=1))
    for c in summary["candidates"]:
        print(f"kp {c['kp']:g} kd {c['kd']:g} | T1 {c['T1']} | T2 {c['T2']} | "
              f"T3 passed {c['T3']['n_passed']}/{c['T3']['n_patterns']}, best passing {c['T3']['best_passing_speed']:.3f} m/s")
        for k in ("best_passing", "fastest_any", "longest_load"):
            print(f"    {k}: {c['T3'][k]}")
    print("chosen:", summary["chosen"], "→", out)


if __name__ == "__main__":
    main()
