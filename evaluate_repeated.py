"""반복 평가: 명령마다 여러 초기 상태 × 노이즈 끔/켬으로 굴려서 성공률·관절 지표·굳음(freeze)을 잰다.

왜 필요한가: 초기 관절 상태를 1e-9만 흔들어도 명령별 걷기/굳기가 바뀐다 (sensitivity.py).
그래서 명령당 한 번 굴린 결과(evaluate.py)로는 정책 성능을 판단할 수 없다.
evaluate.py와 그 결과(summary.json)는 참고 기록으로 그대로 두고, 이 스크립트 결과를 공식 결과로 쓴다.

평가 조건
  - 명령: evaluate.py와 같은 16개 (0.5, 1.0 m/s × 8방향), 에너지 0.5
  - 초기 상태: env.reset(seed=0..9) (Ant-v5가 관절 위치·속도에 작은 무작위 값을 섞는다)
  - 노이즈 끔(deterministic) / 켬(stochastic): 같은 reset seed 사용, 켬의 노이즈 난수 seed = 10000 + reset seed
  - 길이: 1000스텝 = 50초 (학습 에피소드와 같음). 명령은 에피소드 내내 고정

에피소드 판정
  - 이동 성공: 3~50초 명령 방향 평균 속도 > 0.05 m/s, 넘어지지 않음
  - 추종 성공: 3~50초 명령 방향 평균 속도 ≥ 명령 속도의 50%, 넘어지지 않음
  - 굳음 스텝: 관절 8개 모두 한계 2° 이내 AND 관절 각속도 평균 < 0.01 rad/s AND 몸통 속도 < 0.02 m/s
  - 굳음 구간: 굳음 스텝이 10스텝(0.5초) 이상 연속
  - 결과 분류 (위에서부터 먼저 맞는 것)
      fell                 넘어짐
      initial_freeze       끝까지 못 빠져나온 굳음, 굳기 전 시작점 대비 명령 방향 순변위 < 0.1 m
      permanent_freeze     끝까지 못 빠져나온 굳음, 그 전에 0.1 m 이상 이동
      temporary_freeze     굳음 구간이 있었지만 다시 움직임 (영구 굳음 없음)
      sustained_locomotion 굳음 구간 없음 + 이동 성공
      stalled              굳음 구간 없음 + 이동 실패 (제자리에서 떨기 등)
  - 보행 지표: 관절별 한계(±2°) 근처 비율, 범위 끝 20% 비율, 움직임 범위(p5–p95), 네 발 모두 공중 비율,
    몸통 수직 속도² (월드 qvel[2]), 몸통 평균 높이, 몸통 / 윗다리(aux+leg) / 둘 동시 바닥 접촉 비율 (모두 3~50초),
    발 접촉 교대(diag_in_phase: 같은 대각선 두 발 접촉 상관, pair_alternation: 두 대각선 쌍의 반대 위상 정도)
  - 생존 곡선: "아직 영구 굳음에 빠지지 않은 비율" (Kaplan–Meier). 사건 = 영구 굳음 시작,
    50초까지 안 굳었거나 넘어진 에피소드는 그 시점에서 우측 검열

    python evaluate_repeated.py --check           # 실행 전 확인 (reset 다양성, 1e-9 흔들기 전달)
    python evaluate_repeated.py                   # v5, v5b × seed 0, 1, 2 전체
    python evaluate_repeated.py --runs runs/main_v5b --n-seeds 2 --steps 200   # 빠른 시험
결과: runs/<run>/results/summary_repeated.json, runs/repeated/ (비교표 comparison.md, survival.png, summary.json)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

RUNS = ["runs/main_v5", "runs/main_v5_s1", "runs/main_v5_s2",
        "runs/main_v5b", "runs/main_v5b_s1", "runs/main_v5b_s2"]
SPEEDS = [0.5, 1.0]
N_DIRS = 8
ENERGY = 0.5
WARMUP = 60                  # 3초
SHORT_END = 300              # 15초 (evaluate.py와 비교용 구간 3~15초)
NOISE_SEED_OFFSET = 10_000

CRITERIA = dict(
    move_speed=0.05,          # m/s
    track_ratio=0.5,          # 명령 속도 대비
    limit_margin_deg=2.0,
    soft_zone=0.8,            # 범위 끝 20% 구간: 중심 ± 0.8 × 반범위 밖 (v6 페널티 구간과 같음)
    freeze_joint_vel=0.01,    # rad/s, 관절 8개 평균
    freeze_torso_speed=0.02,  # m/s
    freeze_min_steps=10,      # 0.5초
    initial_freeze_disp=0.1,  # m, 시작점 대비 명령 방향 순변위
)
OUTCOMES = ["sustained_locomotion", "temporary_freeze", "permanent_freeze",
            "initial_freeze", "stalled", "fell"]
FEET = ["left_ankle_geom", "right_ankle_geom", "third_ankle_geom", "fourth_ankle_geom"]
# 윗다리 = 몸통과 다리를 잇는 연결부(aux) + 윗다리(leg). 발목(ankle)은 정상적인 발 접촉이라 제외
UPPER = ["aux_1_geom", "aux_2_geom", "aux_3_geom", "aux_4_geom",
         "left_leg_geom", "right_leg_geom", "back_leg_geom", "rightback_leg_geom"]
JOINTS = ["hip_1", "ankle_1", "hip_2", "ankle_2", "hip_3", "ankle_3", "hip_4", "ankle_4"]
# 대각선 다리 쌍: 다리 1(+x,+y)·3(−x,−y), 다리 2(−x,+y)·4(+x,−y). FEET 순서 기준 인덱스
DIAG_A, DIAG_B = (0, 2), (1, 3)


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def contact_pattern(fw):
    """발 접촉(스텝 × 4, 0/1)에서 교대 패턴 지표.
    trot(대각선 교대)이면 diag_in_phase > 0 (같은 대각선 두 발이 함께 닿음), pair_alternation > 0
    (두 대각선 쌍이 번갈아 닿음). 둘 다 None이면 접촉이 변하지 않음(계속 닿거나 계속 떠 있음)."""
    ca = [_corr(fw[:, DIAG_A[0]], fw[:, DIAG_A[1]]), _corr(fw[:, DIAG_B[0]], fw[:, DIAG_B[1]])]
    ca = [c for c in ca if c is not None]
    pair = _corr(fw[:, list(DIAG_A)].sum(axis=1), fw[:, list(DIAG_B)].sum(axis=1))
    n_down = fw.sum(axis=1)
    return dict(
        diag_in_phase=float(np.mean(ca)) if ca else None,
        pair_alternation=None if pair is None else -pair,
        feet_down_hist=[float(np.mean(n_down == k)) for k in range(5)],
    )


def commands():
    """evaluate.py와 같은 순서·값의 16개 명령."""
    out = []
    for s in SPEEDS:
        for a in np.linspace(-math.pi, math.pi, N_DIRS, endpoint=False):
            out.append([s * math.cos(a), s * math.sin(a)])
    return out


# ---------------------------------------------------------------- 한 에피소드
_POLICIES = {}


def _policy(run):
    if run not in _POLICIES:
        from policy_utils import Policy
        _POLICIES[run] = Policy(run)
    return _POLICIES[run]


def _runs_of_true(mask):
    """연속 True 구간들 [(시작, 끝(포함)), ...]."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [(s, e - 1) for s, e in zip(edges[::2], edges[1::2])]


def episode(run, cmd, reset_seed, deterministic, steps):
    import mujoco
    import torch
    from envs import make_command_ant

    noise_seed = None if deterministic else NOISE_SEED_OFFSET + reset_seed
    if noise_seed is not None:
        torch.manual_seed(noise_seed)
    policy = _policy(run)

    env = make_command_ant(**{**policy.env_kwargs, "max_episode_steps": steps})
    env.set_command(cmd[0], cmd[1], ENERGY)
    obs, _ = env.reset(seed=reset_seed)
    m, d = env.unwrapped.model, env.unwrapped.data
    dt = env.unwrapped.dt
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    floor, feet = gid("floor"), [gid(n) for n in FEET]
    torso_geom, upper = gid("torso_geom"), {gid(n) for n in UPPER}
    lo, hi = np.degrees(m.jnt_range[1:9, 0]), np.degrees(m.jnt_range[1:9, 1])
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    soft_lo, soft_hi = mid - CRITERIA["soft_zone"] * half, mid + CRITERIA["soft_zone"] * half
    c_hat = np.asarray(cmd) / np.linalg.norm(cmd)
    xy0 = env.torso_xy()

    vel, xy, q, qd, z, foot, act = [], [], [], [], [], [], []
    vz2, torso_c, upper_c = [], [], []
    tabs, tlim, pw = [], [], []          # v8: 실제 적용 토크 (관절 평균 |τ|), 토크 제한 비율, 일률
    fell = False
    for t in range(steps):
        a = policy.act(obs, deterministic=deterministic)
        obs, _, term, trunc, info = env.step(a)
        touching = set()
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            if g1 == floor:
                touching.add(g2)
            elif g2 == floor:
                touching.add(g1)
        vel.append(info["velocity"])
        xy.append(env.torso_xy())
        q.append(np.degrees(d.qpos[7:15]))
        qd.append(np.abs(d.qvel[6:14]))
        z.append(d.qpos[2])
        vz2.append(d.qvel[2] ** 2)               # 월드 좌표계 몸통 수직 속도 제곱
        foot.append([f in touching for f in feet])
        torso_c.append(torso_geom in touching)
        upper_c.append(bool(touching & upper))
        act.append(np.clip(a, -1, 1))
        tabs.append(float(np.mean(np.abs(info["torque"]))))
        tlim.append(info["torque_limit_frac"])
        pw.append(info["power"])
        if term:
            fell = True
            break
        if trunc:
            break
    env.close()

    vel, xy, q, qd = map(np.array, (vel, xy, q, qd))
    z, foot, act = np.array(z), np.array(foot, dtype=bool), np.array(act)
    vz2, torso_c, upper_c = np.array(vz2), np.array(torso_c, dtype=bool), np.array(upper_c, dtype=bool)
    T = len(vel)
    along = vel @ c_hat

    def mean_along(s, e):
        return float(along[s:e].mean()) if e > s else 0.0

    speed_3_15 = mean_along(WARMUP, min(SHORT_END, T))
    speed_3_50 = mean_along(WARMUP, T)

    # 굳음
    near = (np.abs(q - lo) < CRITERIA["limit_margin_deg"]) | (np.abs(q - hi) < CRITERIA["limit_margin_deg"])
    frozen_step = (near.all(axis=1)
                   & (qd.mean(axis=1) < CRITERIA["freeze_joint_vel"])
                   & (np.linalg.norm(vel, axis=1) < CRITERIA["freeze_torso_speed"]))
    spans = [(s, e) for s, e in _runs_of_true(frozen_step) if e - s + 1 >= CRITERIA["freeze_min_steps"]]
    permanent = bool(spans and spans[-1][1] == T - 1 and not fell)
    temp_spans = spans[:-1] if permanent else spans
    onset = spans[-1][0] if permanent else None
    disp_before = float((xy[onset] - xy0) @ c_hat) if permanent else float((xy[-1] - xy0) @ c_hat)

    moving = bool(speed_3_50 > CRITERIA["move_speed"] and not fell)
    tracking = bool(speed_3_50 >= CRITERIA["track_ratio"] * np.linalg.norm(cmd) and not fell)
    if fell:
        outcome = "fell"
    elif permanent:
        outcome = "initial_freeze" if disp_before < CRITERIA["initial_freeze_disp"] else "permanent_freeze"
    elif temp_spans:
        outcome = "temporary_freeze"
    elif moving:
        outcome = "sustained_locomotion"
    else:
        outcome = "stalled"

    w = slice(WARMUP, T)  # 지표는 워밍업 이후 구간
    fw = foot[w].astype(int)
    dur = max((T - WARMUP) * dt, dt)
    ok = T > WARMUP
    pattern = contact_pattern(fw) if ok else dict(diag_in_phase=None, pair_alternation=None, feet_down_hist=None)
    in_zone = (q[w] < soft_lo) | (q[w] > soft_hi)
    return dict(
        cmd=list(cmd), reset_seed=reset_seed, deterministic=deterministic, noise_seed=noise_seed,
        steps=T, fell=fell,
        speed_3_15=speed_3_15, speed_3_50=speed_3_50,
        moving=moving, tracking=tracking, outcome=outcome,
        freeze=bool(spans), permanent_freeze=permanent, initial_freeze=outcome == "initial_freeze",
        freeze_onset_s=None if onset is None else onset * dt,
        freeze_duration_s=None if onset is None else (T - onset) * dt,
        disp_before_freeze_m=disp_before,
        n_temporary_freezes=len(temp_spans),
        temporary_freeze_s=float(sum(e - s + 1 for s, e in temp_spans) * dt),
        first_freeze_s=spans[0][0] * dt if spans else None,
        joint_limit_occupancy=near[w].mean(axis=0).round(3).tolist() if T > WARMUP else None,
        joint_range_deg=np.ptp(q[w], axis=0).round(2).tolist() if T > WARMUP else None,
        joint_zone20_occupancy=in_zone.mean(axis=0).round(3).tolist() if ok else None,
        joint_rom_p5_p95_deg=(np.percentile(q[w], 95, axis=0)
                              - np.percentile(q[w], 5, axis=0)).round(2).tolist() if ok else None,
        diag_in_phase=pattern["diag_in_phase"],
        pair_alternation=pattern["pair_alternation"],
        feet_down_hist=pattern["feet_down_hist"],
        joint_velocity=float(qd[w].mean()) if T > WARMUP else None,
        trunk_height_std=float(z[w].std()) if T > WARMUP else None,
        trunk_height_range=float(np.ptp(z[w])) if T > WARMUP else None,
        foot_air_ratio=(1 - fw.mean(axis=0)).round(3).tolist() if T > WARMUP else None,
        all_feet_air_ratio=float((fw.sum(axis=1) == 0).mean()) if T > WARMUP else None,
        torso_vz2=float(vz2[w].mean()) if ok else None,
        trunk_height_mean=float(z[w].mean()) if ok else None,
        torso_contact_ratio=float(torso_c[w].mean()) if ok else None,
        upper_leg_contact_ratio=float(upper_c[w].mean()) if ok else None,
        torso_and_upper_contact_ratio=float((torso_c[w] & upper_c[w]).mean()) if ok else None,
        touchdowns_per_s=((np.diff(fw, axis=0) == 1).sum(axis=0) / dur).round(3).tolist() if T > WARMUP else None,
        action_saturation=float((np.abs(act[w]) > 0.99).mean()) if T > WARMUP else None,
        mean_abs_torque=float(np.mean(np.asarray(tabs)[w])) if T > WARMUP else None,
        torque_limit_frac=float(np.mean(np.asarray(tlim)[w])) if T > WARMUP else None,
        power=float(np.mean(np.asarray(pw)[w])) if T > WARMUP else None,
    )


def _job(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    run, idx, cmd, n_seeds, steps = args
    eps = []
    for det in (True, False):
        for s in range(n_seeds):
            r = episode(run, cmd, s, det, steps)
            r["cmd_idx"] = idx
            eps.append(r)
    return run, eps


# ---------------------------------------------------------------- 집계
def kaplan_meier(times, events, horizon):
    """우측 검열을 반영한 생존 곡선. times: 사건/검열 시각, events: True = 영구 굳음 발생."""
    times, events = np.asarray(times, float), np.asarray(events, bool)
    grid = np.unique(np.r_[0.0, times[events], horizon])
    surv, s = [], 1.0
    for t in grid:
        d = np.sum(events & (times == t))
        n = np.sum(times >= t)
        if n > 0 and d > 0:
            s *= 1 - d / n
        surv.append(s)
    return grid.tolist(), surv


def survival(eps, dt, steps, started_only=False):
    sel = [e for e in eps if not (started_only and e["initial_freeze"])]
    if not sel:
        return [], []
    times = [e["freeze_onset_s"] if e["permanent_freeze"] else e["steps"] * dt for e in sel]
    events = [e["permanent_freeze"] for e in sel]
    return kaplan_meier(times, events, steps * dt)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def aggregate(eps, dt, steps):
    out = dict(n=len(eps))
    out["move_rate"] = _mean([e["moving"] for e in eps])
    out["track_rate"] = _mean([e["tracking"] for e in eps])
    out["speed_3_15"] = _mean([e["speed_3_15"] for e in eps])
    out["speed_3_50"] = _mean([e["speed_3_50"] for e in eps])
    out["outcomes"] = {k: _mean([e["outcome"] == k for e in eps]) for k in OUTCOMES}
    out["freeze_rate"] = _mean([e["freeze"] for e in eps])
    onsets = [e["freeze_onset_s"] for e in eps if e["outcome"] == "permanent_freeze"]
    out["permanent_freeze_onset_s_median"] = float(np.median(onsets)) if onsets else None
    for k in ("joint_velocity", "trunk_height_std", "all_feet_air_ratio", "action_saturation",
              "diag_in_phase", "pair_alternation", "torso_vz2", "trunk_height_mean",
              "torso_contact_ratio", "upper_leg_contact_ratio", "torso_and_upper_contact_ratio",
              "mean_abs_torque", "torque_limit_frac", "power"):
        out[k] = _mean([e[k] for e in eps])
    for k in ("joint_limit_occupancy", "joint_range_deg", "foot_air_ratio", "touchdowns_per_s",
              "joint_zone20_occupancy", "joint_rom_p5_p95_deg", "feet_down_hist"):
        vals = [e[k] for e in eps if e[k] is not None]
        out[k] = np.mean(vals, axis=0).round(3).tolist() if vals else None
    g, s = survival(eps, dt, steps, started_only=True)
    out["started_still_walking_at_end"] = s[-1] if s else None
    return out


def summarize_run(run, eps, dt, steps, n_seeds):
    cmds = commands()
    res = dict(criteria=CRITERIA, steps=steps, dt=dt, n_seeds=n_seeds, energy=ENERGY,
               noise_seed_offset=NOISE_SEED_OFFSET, joints=JOINTS, feet=FEET)
    for mode, det in (("deterministic", True), ("stochastic", False)):
        me = [e for e in eps if e["deterministic"] == det]
        res[mode] = aggregate(me, dt, steps)
        res[mode]["per_command"] = []
        for i, c in enumerate(cmds):
            ce = [e for e in me if e["cmd_idx"] == i]
            a = dict(cmd=c, move_rate=_mean([e["moving"] for e in ce]),
                     track_rate=_mean([e["tracking"] for e in ce]),
                     speed_3_50_mean=_mean([e["speed_3_50"] for e in ce]),
                     speed_3_50_std=float(np.std([e["speed_3_50"] for e in ce])),
                     outcomes={k: sum(e["outcome"] == k for e in ce) for k in OUTCOMES})
            res[mode]["per_command"].append(a)
        res[mode]["survival_all"] = survival(me, dt, steps)
        res[mode]["survival_started"] = survival(me, dt, steps, started_only=True)
    res["episodes"] = eps
    return res


# ---------------------------------------------------------------- 출력
# 설정별 색 (고정 순서, 8개 넘으면 점선으로 반복)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def group_name(name):
    """run 이름에서 학습 seed 꼬리(_s1, _s2)를 떼어 설정 이름을 만든다."""
    import re
    return re.sub(r"_s\d+$", "", name)


def _avg(vals, start=None):
    """관절별 값 8개의 평균. start=0이면 엉덩이 4개, 1이면 발목 4개."""
    if not vals:
        return None
    return float(np.mean(vals if start is None else vals[start::2]))


def hip_ank(vals, fmt):
    """관절별 값 8개 → '엉덩이 평균 / 발목 평균'."""
    if not vals:
        return "–"
    return f"{fmt.format(np.mean(vals[0::2]))} / {fmt.format(np.mean(vals[1::2]))}"


def write_comparison(results, out: Path, steps, dt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    names = {r: Path(r).name for r in results}
    lines = []
    for mode, label in (("deterministic", "노이즈 끔"), ("stochastic", "노이즈 켬")):
        lines += [f"### {label}", "",
                  f"| run | 이동 성공률 | 추종 성공률 | 속도 3~15s | 속도 3~50s | 끝까지 이동 | 일시 굳음 | 걷다 영구 굳음 | 처음부터 굳음 | 제자리 | 넘어짐 | 걷기 시작 후 {steps * dt:.0f}초까지 유지 | 영구 굳음 시점 중간값 | 네 발 공중 | 관절 한계 비율 (관절별) | 끝 20% 비율 (엉덩이 / 발목) | 움직임 범위 p5–p95 (엉덩이 / 발목) | 대각선 동시 접촉 | 대각선 쌍 교대 |",
                  "| --- |" + " --- |" * 18]
        for r, res in results.items():
            a = res[mode]
            o = a["outcomes"]
            f = lambda x: "–" if x is None else f"{x:.2f}"
            lines.append(
                f"| {names[r]} | {f(a['move_rate'])} | {f(a['track_rate'])} | {f(a['speed_3_15'])} | {f(a['speed_3_50'])} | "
                f"{f(o['sustained_locomotion'])} | {f(o['temporary_freeze'])} | {f(o['permanent_freeze'])} | "
                f"{f(o['initial_freeze'])} | {f(o['stalled'])} | {f(o['fell'])} | {f(a['started_still_walking_at_end'])} | "
                f"{f(a['permanent_freeze_onset_s_median'])}{'' if a['permanent_freeze_onset_s_median'] is None else 's'} | {f(a['all_feet_air_ratio'])} | "
                f"{' '.join(f'{x:.2f}' for x in a['joint_limit_occupancy'] or [])} | "
                f"{hip_ank(a['joint_zone20_occupancy'], '{:.2f}')} | {hip_ank(a['joint_rom_p5_p95_deg'], '{:.0f}°')} | "
                f"{f(a['diag_in_phase'])} | {f(a['pair_alternation'])} |")
        lines.append("")
    # 설정별(학습 seed를 묶은) 평균 ± 표준편차
    groups = {}
    for r in results:
        groups.setdefault(group_name(names[r]), []).append(r)
    group_metrics = [
        ("이동 성공률", lambda a: a["move_rate"]),
        ("추종 성공률", lambda a: a["track_rate"]),
        ("속도 3~50s", lambda a: a["speed_3_50"]),
        ("영구 굳음", lambda a: a["outcomes"]["permanent_freeze"] + a["outcomes"]["initial_freeze"]),
        ("넘어짐", lambda a: a["outcomes"]["fell"]),
        ("네 발 공중", lambda a: a["all_feet_air_ratio"]),
        ("v_z²", lambda a: a.get("torso_vz2")),
        ("몸통 높이 (m)", lambda a: a.get("trunk_height_mean")),
        ("몸통 접촉", lambda a: a.get("torso_contact_ratio")),
        ("윗다리 접촉", lambda a: a.get("upper_leg_contact_ratio")),
        ("몸통·윗다리 동시", lambda a: a.get("torso_and_upper_contact_ratio")),
        ("한계 ±2° 비율", lambda a: _avg(a["joint_limit_occupancy"])),
        ("끝 20% 엉덩이", lambda a: _avg(a["joint_zone20_occupancy"], 0)),
        ("끝 20% 발목", lambda a: _avg(a["joint_zone20_occupancy"], 1)),
        ("엉덩이 범위 p5–p95 (°)", lambda a: _avg(a["joint_rom_p5_p95_deg"], 0)),
        ("발목 범위 p5–p95 (°)", lambda a: _avg(a["joint_rom_p5_p95_deg"], 1)),
        ("대각선 동시 접촉", lambda a: a["diag_in_phase"]),
        ("대각선 쌍 교대", lambda a: a["pair_alternation"]),
    ]
    head = ["", "## 설정별 (학습 seed 평균 ± 표준편차)", ""]
    for mode, label in (("deterministic", "노이즈 끔"), ("stochastic", "노이즈 켬")):
        head += [f"### {label}", "",
                 "| 설정 (run 수) | " + " | ".join(m for m, _ in group_metrics) + " |",
                 "| --- |" + " --- |" * len(group_metrics)]
        for gname, rs in groups.items():
            cells = []
            for _, fn in group_metrics:
                vals = [fn(results[r][mode]) for r in rs]
                vals = [v for v in vals if v is not None]
                cells.append("–" if not vals else
                             f"{np.mean(vals):.2f}" + (f" ± {np.std(vals):.2f}" if len(vals) > 1 else ""))
            head.append(f"| {gname} ({len(rs)}) | " + " | ".join(cells) + " |")
        head.append("")
    lines = head + ["## run별", ""] + lines
    (out / "comparison.md").write_text("\n".join(lines))
    (out / "summary.json").write_text(json.dumps(
        {names[r]: {m: {k: v for k, v in res[m].items() if k != "per_command"}
                    for m in ("deterministic", "stochastic")} for r, res in results.items()}, indent=1))

    # 생존 곡선: 설정마다 한 줄 (학습 seed들의 에피소드를 합쳐 Kaplan–Meier), 색 = 설정 (고정 순서)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (mode, label, det) in zip(axes, (("deterministic", "noise off", True), ("stochastic", "noise on", False))):
        for k, (gname, rs) in enumerate(groups.items()):
            eps = [e for r in rs for e in results[r]["episodes"] if e["deterministic"] == det]
            g, sv = survival(eps, dt, steps)
            ax.step(g, sv, where="post", color=PALETTE[k % len(PALETTE)], lw=1.8,
                    ls="-" if k < len(PALETTE) else "--", label=f"{gname} (n={len(rs)})")
        ax.set_title(label, fontsize=10, color="#222222", loc="left")
        ax.set_xlabel("time [s]", fontsize=9)
        ax.set_xlim(0, steps * dt)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color="#e4e3dd", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8, colors="#6b6b66")
    axes[0].set_ylabel("fraction not permanently frozen", fontsize=9)
    axes[1].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Survival until permanent freeze, per config (training seeds pooled; drop near t=0 = initial freeze)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "survival.png", dpi=150)
    plt.close(fig)
    print("\n".join(lines))


# ---------------------------------------------------------------- 실행 전 확인
def check():
    """(1) reset seed마다 초기 관절 상태가 실제로 다른가 (2) 1e-9 흔들기가 물리 상태에 남는가."""
    import mujoco
    from envs import make_command_ant

    print("[1] reset seed별 초기 관절 상태 (qpos 관절 8개, qvel 관절 8개)")
    env = make_command_ant()
    rows = []
    for s in range(10):
        env.reset(seed=s)
        d = env.unwrapped.data
        rows.append(np.r_[d.qpos[7:15], d.qvel[6:14]])
    rows = np.array(rows)
    print(f"  qpos dtype: {env.unwrapped.data.qpos.dtype}")
    print(f"  seed 0 qpos[:4] = {rows[0, :4].round(4)}, seed 1 qpos[:4] = {rows[1, :4].round(4)}")
    print(f"  seed 간 표준편차: 관절 위치 {rows[:, :8].std(axis=0).mean():.4f} rad, "
          f"관절 속도 {rows[:, 8:].std(axis=0).mean():.4f} rad/s")
    print(f"  서로 다른 초기 상태 수: {len({tuple(r) for r in rows})}/10")

    print("[2] 1e-9 흔들기 전달 확인")
    env.reset(seed=0)
    u = env.unwrapped
    base = u.data.qpos.copy()
    noise = 1e-9 * np.random.default_rng(0).standard_normal(8)
    u.data.qpos[7:15] += noise
    mujoco.mj_forward(u.model, u.data)
    diff = u.data.qpos[7:15] - base[7:15]
    print(f"  적용 직후 |차이| 최대 {np.abs(diff).max():.3e}, 의도한 값과의 오차 {np.abs(diff - noise).max():.3e}")
    obs32 = env._augment(u._get_obs()).astype(np.float32)
    env.reset(seed=0)
    obs32_base = env._augment(env.unwrapped._get_obs()).astype(np.float32)
    print(f"  정책 입력(float32) 관측값 차이: {np.abs(obs32 - obs32_base).max():.3e} "
          f"(0이면 첫 스텝에서는 정책이 차이를 못 보고 물리 상태로만 전달됨)")

    # 같은 행동 열을 넣었을 때 두 시뮬레이션의 차이가 커지는지
    a_env, b_env = make_command_ant(), make_command_ant()
    a_env.reset(seed=0); b_env.reset(seed=0)
    b_env.unwrapped.data.qpos[7:15] += noise
    mujoco.mj_forward(b_env.unwrapped.model, b_env.unwrapped.data)
    rng = np.random.default_rng(1)
    for t in range(1, 201):
        act = rng.uniform(-1, 1, 8)
        a_env.step(act); b_env.step(act)
        if t in (1, 10, 50, 100, 200):
            dq = np.abs(a_env.unwrapped.data.qpos - b_env.unwrapped.data.qpos).max()
            print(f"  같은 무작위 행동 {t:3d}스텝 후 상태 차이 최대 {dq:.3e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=RUNS)
    p.add_argument("--n-seeds", type=int, default=10, help="명령당 초기 상태 수 (reset seed 0..n-1)")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", default="runs/repeated")
    p.add_argument("--check", action="store_true", help="실행 전 확인만 하고 끝냄")
    args = p.parse_args()

    if args.check:
        check()
        return

    cmds = commands()
    jobs = [(r, i, c, args.n_seeds, args.steps) for r in args.runs for i, c in enumerate(cmds)]
    n_ep = len(jobs) * args.n_seeds * 2
    print(f"{len(args.runs)} runs × {len(cmds)} 명령 × {args.n_seeds} 초기 상태 × 노이즈 2종 = {n_ep} 에피소드, "
          f"{args.steps}스텝, 프로세스 {args.workers}개")
    t0 = time.time()
    by_run = {r: [] for r in args.runs}
    with Pool(args.workers) as pool:
        for k, (run, eps) in enumerate(pool.imap_unordered(_job, jobs), 1):
            by_run[run] += eps
            if k % max(len(jobs) // 20, 1) == 0 or k == len(jobs):
                el = time.time() - t0
                print(f"  {k}/{len(jobs)} 작업 완료, {el / 60:.1f}분 경과, 남은 예상 {el / k * (len(jobs) - k) / 60:.1f}분",
                      flush=True)

    from envs import make_command_ant
    dt = make_command_ant().unwrapped.dt
    results = {}
    for run, eps in by_run.items():
        eps.sort(key=lambda e: (e["cmd_idx"], not e["deterministic"], e["reset_seed"]))
        res = summarize_run(run, eps, dt, args.steps, args.n_seeds)
        path = Path(run) / "results" / "summary_repeated.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res, indent=1))
        results[run] = res
    write_comparison(results, Path(args.out), args.steps, dt)
    print(f"done in {(time.time() - t0) / 60:.1f} min → {args.out}, runs/*/results/summary_repeated.json")


if __name__ == "__main__":
    main()
