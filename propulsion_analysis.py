"""추진·접촉 분석: 정책이 발을 어떻게 딛고, 지면 반력을 어떻게 만드는지 물리 스텝(0.01 s) 단위로 잰다.

제어 스텝(0.05 s) = 물리 스텝 5개. env 내부의 mj_step(nstep=5)을 1스텝씩 5번으로 바꿔(결과는 같음)
물리 스텝마다 발-바닥 접촉, 접촉점 속도, 지면 반력을 기록한다.

조건: evaluate.py와 같음 (300 제어 스텝, 워밍업 60, 평가 seed 0, 에너지 0.5), 명령 16개 × 노이즈 끔/켬.

측정
  - 지면 반력: 접촉마다 mj_contactForce(접촉 좌표계) → 월드 좌표계로 변환, 로봇(발)이 받는 힘으로 부호 통일.
    MuJoCo의 접촉은 각각 별개의 접촉점이라 접촉별 힘을 더하면 중복 없음.
    수평 성분은 월드 좌표계에서 명령 방향 단위벡터 u로 투영(F_along). 몸이 돌아도 "명령 방향으로 미는지"를 본다.
    검증: 로봇 전체(발 외 몸통·윗다리 포함)의 수직 반력 시간 평균 ≈ mg, 수평 반력 시간 평균 ≈ 질량 × 평균 가속도 ≈ 0.
    주의: Ant-v5는 RK4 적분기라 mj_step 뒤 data의 접촉력은 마지막 RK 단계 값이다(이걸 쓰면 Fz 평균 1.07 mg, 충격량-운동량 불일치).
    그래서 매 물리 스텝 시작 상태에서 mj_forward로 다시 계산한 접촉력을 쓴다(Fz 평균 1.004 mg, 12초 충격량-운동량 오차는
    수평 총 충격량의 약 2%). RK4 네 단계의 가중 평균이 아니라 첫 단계 값이므로 근사다.
  - 디딤(stance): 한 발이 연속으로 바닥에 닿은 물리 스텝 구간. 지속 시간, 수직 충격량, 명령 방향 충격량.
  - 디딤 안의 힘 변화: 디딤을 앞/뒤 절반으로 나눠 명령 방향 충격량의 부호로 분류
      제동→추진(앞 −, 뒤 +), 추진만(+, +), 제동만(−, −), 추진→제동(+, −)
    그리고 3스텝 이상 디딤을 10구간으로 늘려 평균한 F_along / F_z 곡선(그림).
  - 미끄러짐: 접촉점 속도 = 발 geom 속도 + ω × (접촉점 − geom 중심)의 수평 크기. 디딤 첫·마지막 스텝을 뺀 중간값.
    마찰 사용률 |F_수평| / F_z (마찰 계수 1.0, 1에 가까우면 미끄러지는 중).
  - 그 밖에 (제어 스텝 단위): 관절 각속도 RMS, 발목 한계 비율, 이동 속도·직진성·회전.

예: python propulsion_analysis.py runs/main_v5b runs/main_v6Bb runs/main_v7Ab runs/main_v7Bb   (seed 접미사 _s1 _s2 자동 포함)
결과: runs/propulsion_v7/summary.json, runs/propulsion_v7/stance_profile.png, runs/propulsion_v7/table.md
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from envs import make_command_ant
from gait_analysis import FEET, STEPS, WARMUP, command_grid
from policy_utils import Policy

ANKLE_LIM = np.radians(70.0)
NBIN = 10
OUT = Path("runs/propulsion_v7")


class Recorder:
    """물리 스텝마다 발별 접촉·반력·접촉점 속도와 로봇 전체 반력을 기록."""

    def __init__(self, env, u):
        uw = env.unwrapped
        self.m, self.d, self.u = uw.model, uw.data, np.asarray(u)
        gid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n)
        self.floor, self.feet = gid("floor"), [gid(n) for n in FEET]
        self.on = False
        self.rows = []   # (con[4], fz[4], fa[4], slip[4], fh[4], total_f[3])
        self._f6, self._v6 = np.zeros(6), np.zeros(6)

        def step_sim(ctrl, n_frames):          # MujocoEnv._step_mujoco_simulation과 같은 동작, 1스텝씩
            uw.data.ctrl[:] = ctrl
            for _ in range(n_frames):
                if self.on:   # Ant는 RK4: mj_step 뒤의 접촉력은 마지막 RK 단계 값이라, 스텝 시작 상태에서 다시 계산해 기록
                    mujoco.mj_forward(uw.model, uw.data)   # 궤적은 바뀌지 않음 (확인: qpos 차이 0)
                    self.sample()
                mujoco.mj_step(uw.model, uw.data)
            mujoco.mj_rnePostConstraint(uw.model, uw.data)
        uw._step_mujoco_simulation = step_sim

    def sample(self):
        m, d = self.m, self.d
        con = np.zeros(4, bool); fz = np.zeros(4); fa = np.zeros(4); fh = np.zeros(4)
        slip = np.full(4, np.nan); total = np.zeros(3)
        for i in range(d.ncon):
            c = d.contact[i]
            if self.floor not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.floor else c.geom1
            mujoco.mj_contactForce(m, d, i, self._f6)
            fw = c.frame.reshape(3, 3).T @ self._f6[:3]     # 접촉 좌표계 → 월드 (geom1이 geom2에 주는 힘)
            if c.geom2 == self.floor:                         # 로봇이 받는 힘으로
                fw = -fw
            total += fw
            if other in self.feet:
                k = self.feet.index(other)
                con[k] = True
                fz[k] += fw[2]; fa[k] += fw[:2] @ self.u; fh[k] += np.linalg.norm(fw[:2])
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, other, self._v6, 0)
                vp = self._v6[3:6] + np.cross(self._v6[:3], c.pos - d.geom_xpos[other])
                slip[k] = np.linalg.norm(vp[:2])
        self.rows.append((con, fz, fa, slip, fh, total))


def run_one(policy, cmd, deterministic):
    env = make_command_ant(max_episode_steps=STEPS + 10)
    env.set_command(cmd[0], cmd[1], 0.5)
    u = np.asarray(cmd) / np.linalg.norm(cmd)
    rec = Recorder(env, u)
    obs, _ = env.reset(seed=0)
    d = rec.d
    C = dict(xy=[], v=[], q=[], qvel=[], yaw=[])
    for t in range(STEPS):
        rec.on = t >= WARMUP
        obs, _, term, _, info = env.step(policy.act(obs, deterministic=deterministic))
        if term:
            break
        if t >= WARMUP:
            C["xy"].append(d.qpos[:2].copy()); C["v"].append(info["velocity"])
            C["q"].append(d.qpos[7:15].copy()); C["qvel"].append(d.qvel[6:14].copy()); C["yaw"].append(d.qvel[5])
    dt_ctrl, dt = env.unwrapped.dt, env.unwrapped.model.opt.timestep
    mass = float(env.unwrapped.model.body_mass.sum())
    env.close()
    if len(C["xy"]) < 20:
        return None
    C = {k: np.array(v) for k, v in C.items()}
    con, fz, fa, slip, fh, total = (np.array(x) for x in zip(*rec.rows))
    stances = []
    for k in range(4):
        c = np.r_[False, con[:, k], False].astype(int)
        for a, b in zip(np.where(np.diff(c) == 1)[0], np.where(np.diff(c) == -1)[0]):
            n = b - a
            f_a, f_z = fa[a:b, k], fz[a:b, k]
            h = n // 2
            s = dict(dur=n * dt, imp_z=float(f_z.sum() * dt), imp_a=float(f_a.sum() * dt),
                     first=float(f_a[:max(h, 1)].sum() * dt), second=float(f_a[h:].sum() * dt) if n > 1 else 0.0,
                     slip_mid=float(np.nanmean(slip[a + 1:b - 1, k])) if n >= 3 else np.nan,
                     mu_use=float(np.median(fh[a:b, k] / np.maximum(f_z, 1e-6))))
            if n >= 3:
                x = np.linspace(0, 1, n); xb = np.linspace(0, 1, NBIN)
                s["prof_a"] = np.interp(xb, x, f_a).tolist(); s["prof_z"] = np.interp(xb, x, f_z).tolist()
            stances.append(s)
    hip, ank = slice(0, 8, 2), slice(1, 8, 2)
    path = np.linalg.norm(np.diff(C["xy"], axis=0), axis=1).sum()
    net = C["xy"][-1] - C["xy"][0]
    T = len(con) * dt
    ep = dict(
        speed_along=float(C["v"].mean(0) @ u),
        speed_total=float(np.linalg.norm(C["v"], axis=1).mean()),
        straightness=float(np.linalg.norm(net) / max(path, 1e-9)),
        still_frac=float((np.linalg.norm(C["v"], axis=1) < 0.05).mean()),
        net_yaw_deg=float(np.degrees(abs(C["yaw"].sum() * dt_ctrl))),
        hip_rms=float(np.sqrt((C["qvel"][:, hip] ** 2).mean())),
        ankle_rms=float(np.sqrt((C["qvel"][:, ank] ** 2).mean())),
        ankle_at_limit=float((np.abs(C["q"][:, ank]) > ANKLE_LIM - np.radians(2)).mean()),
        feet_on_ground=float(con.sum(1).mean()),
        fz_total_over_mg=float(total[:, 2].mean() / (mass * 9.81)),     # 검증: ≈ 1
        fz_feet_share=float(fz.sum() / max(total[:, 2].sum(), 1e-9)),   # 수직 반력 중 발이 받는 몫
        f_along_total_N=float(total[:, :2].mean(0) @ u),                 # 검증: ≈ 0
        prop_imp_per_s=float(np.clip(fa, 0, None).sum() * dt / T),
        brake_imp_per_s=float(np.clip(fa, None, 0).sum() * dt / T),
        stances_per_s=len(stances) / T,
    )
    return ep, stances


def summarize_stances(st):
    dur = np.array([s["dur"] for s in st])
    imp_a = np.array([s["imp_a"] for s in st]); imp_z = np.array([s["imp_z"] for s in st])
    first = np.array([s["first"] for s in st]); second = np.array([s["second"] for s in st])
    long_ = dur >= 0.03
    eps = 1e-4
    pat = lambda m: float(m[long_].mean()) if long_.any() else float("nan")
    slip = np.array([s["slip_mid"] for s in st]); mu = np.array([s["mu_use"] for s in st])
    return dict(
        n_stances=len(st),
        dur_median_s=float(np.median(dur)), dur_mean_s=float(dur.mean()),
        frac_dur_le_0_02=float((dur <= 0.02 + 1e-9).mean()),
        frac_dur_lt_0_05=float((dur < 0.05 - 1e-9).mean()),
        frac_dur_ge_0_10=float((dur >= 0.10 - 1e-9).mean()),
        share_z_imp_from_ge_0_05=float(imp_z[dur >= 0.05 - 1e-9].sum() / imp_z.sum()),
        share_prop_imp_from_ge_0_05=float(np.clip(imp_a, 0, None)[dur >= 0.05 - 1e-9].sum() / max(np.clip(imp_a, 0, None).sum(), 1e-9)),
        # 3스텝 이상 디딤의 반력 패턴 (앞 절반 / 뒤 절반 명령 방향 충격량)
        pat_brake_then_prop=pat((first < -eps) & (second > eps)),
        pat_prop_only=pat((first >= -eps) & (second > eps)),
        pat_brake_only=pat((first < -eps) & (second <= eps)),
        pat_prop_then_brake=pat((first > eps) & (second < -eps)),
        net_along_pos_frac_long=float((imp_a[long_] > 0).mean()) if long_.any() else float("nan"),
        slip_mid_mean=float(np.nanmean(slip)), slip_mid_gt_0_1=float(np.nanmean(slip[~np.isnan(slip)] > 0.1)),
        mu_use_median=float(np.median(mu)),
    )


def main():
    names = sys.argv[1:] or ["runs/main_v5b", "runs/main_v6Bb", "runs/main_v7Ab", "runs/main_v7Bb"]
    OUT.mkdir(parents=True, exist_ok=True)
    result, profiles = {}, {}
    for base in names:
        for mode, det in (("off", True), ("on", False)):
            per_seed, all_st = [], []
            for sfx in ("", "_s1", "_s2"):
                run = base + sfx
                pol = Policy(run)
                eps, st = [], []
                for c in command_grid():
                    r = run_one(pol, c, det)
                    if r:
                        eps.append(r[0]); st += r[1]
                agg = {k: float(np.nanmean([e[k] for e in eps])) for k in eps[0]}
                agg.update(summarize_stances(st))
                per_seed.append(agg); all_st += st
                print(f"{Path(run).name:14s} {mode}: dur_med {agg['dur_median_s']:.3f}s, ≤0.02s {agg['frac_dur_le_0_02']:.2f}, "
                      f"Fz/mg {agg['fz_total_over_mg']:.3f}, F_along {agg['f_along_total_N']:+.3f} N, "
                      f"brake→prop {agg['pat_brake_then_prop']:.2f}, slip {agg['slip_mid_mean']:.2f}", flush=True)
            key = f"{Path(base).name}/{mode}"
            ks = per_seed[0].keys()
            result[key] = dict(mean={k: float(np.mean([p[k] for p in per_seed])) for k in ks},
                               std={k: float(np.std([p[k] for p in per_seed], ddof=1)) for k in ks},
                               per_seed=per_seed)
            prof = [s for s in all_st if "prof_a" in s]
            profiles[key] = dict(a=np.mean([s["prof_a"] for s in prof], 0), z=np.mean([s["prof_z"] for s in prof], 0),
                                 n=len(prof))
    (OUT / "summary.json").write_text(json.dumps(
        dict(result=result, profiles={k: dict(a=v["a"].tolist(), z=v["z"].tolist(), n=v["n"]) for k, v in profiles.items()}),
        indent=1))
    plot(profiles, names)
    print("saved →", OUT)


def plot(profiles, names):
    colors = ["#6b6b66", "#2a78d6", "#eda100", "#eb6834"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    x = np.linspace(0, 100, NBIN)
    for col, mode in enumerate(("off", "on")):
        for base, c in zip(names, colors):
            p = profiles[f"{Path(base).name}/{mode}"]
            lab = f"{Path(base).name.replace('main_', '')} (n={p['n']})"
            axes[0, col].plot(x, p["a"], color=c, lw=2, label=lab)
            axes[1, col].plot(x, p["z"], color=c, lw=2, label=lab)
        axes[0, col].axhline(0, color="#222", lw=0.8)
        axes[0, col].set_title(f"noise {mode}: stances ≥ 0.03 s, time-normalized")
        axes[0, col].set_ylabel("F along command [N]  (+ = propulsion)")
        axes[1, col].set_ylabel("F vertical [N]")
        axes[1, col].set_xlabel("stance time [%]")
        axes[0, col].legend(fontsize=8, frameon=False)
        for a in axes[:, col]:
            a.grid(color="#e4e3dd"); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "stance_profile.png", dpi=110)


if __name__ == "__main__":
    main()
