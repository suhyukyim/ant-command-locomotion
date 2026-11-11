"""CMA-ES로 "디딤을 유지하면서 낼 수 있는 최고 속도"를 찾는다 (v8 PD 사전 시험 T3 보강, 학습 없음).

pd_pretest.py의 T3는 사람이 고른 사인파 trot 1152개를 훑었고 통과 패턴이 없었다. 그 실패가 패턴 설계 탓인지,
이 Ant 몸에서 디딤 + 속도가 원래 어려운 탓인지 가리려고 더 넓은 패턴 공간을 최적화한다.

패턴 (시간만의 함수, 상태 되먹임 없음, 행동 = PD 목표 각도, 21개 파라미터):
  주파수 f (Hz), 다리마다 엉덩이 진폭·기본값, 발목 진폭·기본값, 다리 위상 (다리 1은 0), 발목 위상 지연(공통)
  a_엉덩이 = clip(기본 + 진폭 sin(2πft + φ_다리)),  a_발목 = clip(기본 + 진폭 sin(2πft + φ_다리 + δ))
평가: pd_pretest.t3와 같은 조건(15초, 처음 3초 제외, reset seed 0, +x 방향)과 같은 지표.
목적 함수 (최대화): +x 속도 × min(1, 디딤/0.10 s) × min(1, 0.5/미끄러짐 비율), 넘어지면 −1.
  → 디딤·미끄러짐 기준을 채우면 순수 속도, 못 채우면 그만큼 깎인다.
마지막에 최선 패턴을 reset seed 0~4로 다시 재서 확인한다.

    python cma_gait.py                 # 후보 3개 × 재시작 2번, 결과 runs/v8_pretest/cma_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from multiprocessing import Pool

import numpy as np

from pd_pretest import CANDIDATES, LEG_ACT, LOAD_THR, OUT, PASS, T3_SKIP, T3_STEPS, DT_PHYS, at_limit, make

LEGS = (1, 2, 3, 4)
DIM = 1 + 4 * 4 + 3 + 1


def decode(x):
    """[-1, 1] 근처의 탐색 변수 → 패턴 파라미터."""
    x = np.asarray(x, float)
    f = 0.5 + 3.5 * (np.tanh(x[0]) + 1) / 2                       # 0.5~4 Hz
    legs = x[1:17].reshape(4, 4)                                    # 엉덩이 진폭, 기본, 발목 진폭, 기본
    amp = lambda v: (np.tanh(v) + 1) / 2                            # 0~1
    off = lambda v: np.tanh(v)                                      # −1~1
    phases = np.r_[0.0, math.pi * np.tanh(x[17:20])]                # 다리 2~4 위상 −π~π
    delta = math.pi * np.tanh(x[20])
    return dict(freq=float(f), hip_amp=amp(legs[:, 0]).tolist(), hip_off=off(legs[:, 1]).tolist(),
                ank_amp=amp(legs[:, 2]).tolist(), ank_off=off(legs[:, 3]).tolist(),
                leg_phase=phases.tolist(), ankle_delay=float(delta))


def action(t, p):
    a = np.zeros(8)
    w = 2 * math.pi * p["freq"] * t
    for k, leg in enumerate(LEGS):
        hi, ai = LEG_ACT[leg]
        ph = w + p["leg_phase"][k]
        a[hi] = p["hip_off"][k] + p["hip_amp"][k] * math.sin(ph)
        a[ai] = p["ank_off"][k] + p["ank_amp"][k] * math.sin(ph + p["ankle_delay"])
    return np.clip(a, -1, 1)


def evaluate(args):
    x, kp, kd, seed = args
    from propulsion_analysis import Recorder
    from stance_load_analysis import load_stats
    p = decode(x)
    env = make(kp, kd, T3_STEPS + 10)
    rec = Recorder(env, np.array([1.0, 0.0]))
    env.reset(seed=seed)
    dt = env.unwrapped.dt
    mg = float(env.unwrapped.model.body_mass.sum()) * 9.81
    vel, tl, lim = [], [], []
    for t in range(T3_STEPS):
        rec.on = t >= T3_SKIP
        _, _, term, _, info = env.step(action(t * dt, p).astype(np.float32))
        if t >= T3_SKIP:
            vel.append(info["velocity"]); tl.append(info["torque_limit_frac"]); lim.append(at_limit(env))
        if term:
            env.close()
            return dict(fell=True, score=-1.0)
    env.close()
    con, fz, fa, slip, _, _ = (np.array(r) for r in zip(*rec.rows))
    ld = load_stats(dict(con=con, fz=fz / mg, fa=fa, slip=slip), LOAD_THR)
    dur = float(ld[:, 0].mean() * DT_PHYS) if len(ld) else 0.0
    sl = float(np.nanmean(ld[:, 2])) if len(ld) else 1.0
    sl = 1.0 if math.isnan(sl) else sl
    v = float(np.mean(vel, axis=0)[0])
    r = dict(fell=False, speed_x=v, load_dur_mean=dur, load_slip_gt_0_1=sl,
             torque_limit_frac=float(np.mean(tl)), limit_occ=float(np.mean(lim)),
             load_time_frac=float(ld[:, 0].sum() * DT_PHYS / (len(con) * DT_PHYS) / 4) if len(ld) else 0.0)
    r["meets_stance"] = dur >= PASS["t3_load_dur"] and sl <= PASS["t3_slip"] and r["torque_limit_frac"] < PASS["torque_limit"]
    r["passed"] = r["meets_stance"] and v >= PASS["t3_speed"]
    r["score"] = v * min(1.0, dur / PASS["t3_load_dur"]) * min(1.0, PASS["t3_slip"] / max(sl, 1e-6)) if v > 0 else v
    return r


def run_cma(pool, kp, kd, restart, gens, popsize, seed_offset=0, history=None):
    import cma
    es = cma.CMAEvolutionStrategy(np.zeros(DIM), 0.8, dict(popsize=popsize, seed=1000 + seed_offset + restart, verbose=-9))
    best, best_meet, best_meet_x = None, None, None
    for g in range(gens):
        xs = es.ask()
        rs = pool.map(evaluate, [(x, kp, kd, 0) for x in xs])
        es.tell(xs, [-r["score"] for r in rs])
        for x, r in zip(xs, rs):
            if best is None or r["score"] > best[1]["score"]:
                best = (np.array(x), r)
            if r.get("meets_stance") and (best_meet is None or r["speed_x"] > best_meet["speed_x"]):
                best_meet, best_meet_x = r, np.array(x)
        if history is not None:   # 수렴 기록: 세대별 그 세대 최고 점수, 지금까지 최고 점수, 지금까지 기준 충족 최고 속도
            history.append(dict(gen=g, gen_best=max(r["score"] for r in rs), best_so_far=best[1]["score"],
                                best_feasible_so_far=best_meet["speed_x"] if best_meet else None, sigma=float(es.sigma)))
        if g % 10 == 0 or g == gens - 1:
            bm = f"{best_meet['speed_x']:.3f}" if best_meet else "–"
            print(f"  kp {kp:g} restart {restart} gen {g}: best score {best[1]['score']:.3f}, "
                  f"best speed meeting stance {bm}", flush=True)
    return best, (best_meet_x, best_meet)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gens", type=int, default=60)
    p.add_argument("--popsize", type=int, default=32)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--kps", type=float, nargs="+", default=None, help="일부 후보만 (예: 30 120)")
    p.add_argument("--seed-offset", type=int, default=0, help="재시작 난수 seed = 1000 + offset + 재시작 번호")
    p.add_argument("--out-name", default="cma_summary.json")
    a = p.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    out = dict(criteria=PASS, gens=a.gens, popsize=a.popsize, seed_offset=a.seed_offset, candidates=[])
    cands = [c for c in CANDIDATES if a.kps is None or c[0] in a.kps]
    with Pool() as pool:
        for kp, kd in cands:
            runs = []
            for rs in range(a.restarts):
                hist = []
                (bx, br), (mx, mr) = run_cma(pool, kp, kd, rs, a.gens, a.popsize, a.seed_offset, hist)
                entry = dict(restart=rs, best_score=br, best_score_params=decode(bx), history=hist)
                if mx is not None:   # 디딤 기준을 채운 최고 속도 패턴을 초기 상태 5개로 다시 잰다
                    check = pool.map(evaluate, [(mx, kp, kd, s) for s in range(5)])
                    entry.update(best_meeting_stance=mr, best_meeting_stance_params=decode(mx),
                                 recheck_seeds_0_4=dict(
                                     speed_x=float(np.mean([c.get("speed_x", float("nan")) for c in check])),
                                     passed=sum(bool(c.get("passed")) for c in check),
                                     meets_stance=sum(bool(c.get("meets_stance")) for c in check),
                                     fell=sum(c["fell"] for c in check)))
                runs.append(entry)
            out["candidates"].append(dict(kp=kp, kd=kd, runs=runs))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / a.out_name).write_text(json.dumps(out, indent=1))
    print("saved →", OUT / a.out_name)
    for c in out["candidates"]:
        for r in c["runs"]:
            bm = r.get("best_meeting_stance")
            print(f"kp {c['kp']:g} restart {r['restart']}: best score {r['best_score']['score']:.3f} "
                  f"(speed {r['best_score'].get('speed_x', float('nan')):.3f}, stance {r['best_score'].get('load_dur_mean', 0):.3f} s, "
                  f"slip {r['best_score'].get('load_slip_gt_0_1', 1):.2f}) | fastest meeting stance: "
                  + (f"{bm['speed_x']:.3f} m/s, recheck {r['recheck_seeds_0_4']}" if bm else "none"))


if __name__ == "__main__":
    main()
