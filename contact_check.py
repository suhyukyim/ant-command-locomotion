"""측정 검증 + 마찰·접촉 물리 점검 (보상·마찰 설정은 바꾸지 않음).

물리 스텝(0.01 s)마다, 발-바닥 접촉 하나하나에 대해 기록한다 (RK4라 스텝 시작 상태에서 mj_forward 후 기록).
  1-1 접촉점 속도 (해석식): 발 geom 중심 속도 + ω × (접촉점 − 중심)
      독립 검증: 접촉점에 있는 발의 '같은 물질점'을 발 좌표계에 고정해 다음 물리 스텝 위치를 구한 유한 차분 속도
  1-2 발 geom 중심 속도
  1-3/1-4 분해: 접촉점 속도 = 몸통 이동(병진) + 몸통 회전(ω_몸통 × r) + 다리 자체 운동(몸통에 대한 상대 속도)
      '상쇄 비율' = |접촉점 속도| / |몸통 강체 운동이 그 점에 주는 속도|. 0이면 다리가 몸통 운동을 완전히 상쇄해 발이 고정된 것.
  1-5/2-4 마찰력 방향 검증: 미끄러지는 접촉에서 접선력은 미끄러짐과 반대 방향이어야 함 (cos ≈ −1)
  2-2 마찰 사용률 |F_t| / F_n 과 미끄러짐의 관계 (부드러운 접촉이면 한계 아래에서도 미끄러짐이 생김)
  margin: 이 모델의 접촉 margin은 0.01 m라 실제로 닿기 전(dist > 0)에도 접촉으로 잡힌다 → dist 분포와 힘 확인

예: python contact_check.py   → runs/propulsion_v7/contact_check.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

from envs import make_command_ant
from gait_analysis import FEET, STEPS, WARMUP, command_grid
from policy_utils import Policy

OUT = Path("runs/propulsion_v7")
RUNS = ["main_v5b", "main_v6Bb", "main_v7Ab", "main_v7Bb"]


def rollout(policy, cmd, deterministic):
    env = make_command_ant(**{**policy.env_kwargs, "max_episode_steps": STEPS + 10})
    env.set_command(cmd[0], cmd[1], 0.5)
    uw = env.unwrapped
    m, d = uw.model, uw.data
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    floor, feet = gid("floor"), [gid(n) for n in FEET]
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    mg = float(m.body_mass.sum()) * 9.81
    u = np.asarray(cmd) / np.linalg.norm(cmd)
    rows, pending, on = [], [], [False]
    v6, f6 = np.zeros(6), np.zeros(6)

    def sample():
        pend = []
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, torso, v6, 0)
        w_t, v_t, c_t = v6[:3].copy(), v6[3:].copy(), d.xipos[torso].copy()   # body 속도는 질량 중심 기준
        # 검증: 질량 중심 속도 + ω × (body 원점 − 질량 중심) = free joint 선속도 qvel[0:3]
        chk = np.linalg.norm(v_t + np.cross(w_t, d.xpos[torso] - c_t) - d.qvel[:3])
        for i in range(d.ncon):
            c = d.contact[i]
            if floor not in (c.geom1, c.geom2):
                continue
            g = c.geom2 if c.geom1 == floor else c.geom1
            if g not in feet:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            R = c.frame.reshape(3, 3)
            fw = R.T @ f6[:3]
            if c.geom2 == floor:
                fw = -fw
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, g, v6, 0)
            w_g, v_g = v6[:3].copy(), v6[3:].copy()
            p = c.pos.copy()
            v_p = v_g + np.cross(w_g, p - d.geom_xpos[g])              # 접촉점 물질점 속도 (월드 = 지면 기준)
            v_body = v_t + np.cross(w_t, p - c_t)                       # 몸통 강체 운동이 그 점에 주는 속도
            v_rot = np.cross(w_t, p - c_t)
            p_local = d.geom_xmat[g].reshape(3, 3).T @ (p - d.geom_xpos[g])
            pend.append(dict(g=g, p=p, p_local=p_local, row=dict(
                foot=feet.index(g), dist=float(c.dist), fz=float(fw[2] / mg), ft=fw[:2].copy(),
                v_p=v_p[:2], v_center=v_g[:2], v_body=v_body[:2], v_trans=v_t[:2], v_rot=v_rot[:2],
                v_leg=(v_p - v_body)[:2], u=u, chk=chk)))
        return pend

    def finish(pend):
        for q in pend:   # 같은 물질점의 다음 위치 → 유한 차분 속도
            p_next = d.geom_xmat[q["g"]].reshape(3, 3) @ q["p_local"] + d.geom_xpos[q["g"]]
            q["row"]["v_fd"] = ((p_next - q["p"]) / m.opt.timestep)[:2]
            rows.append(q["row"])

    def step_sim(ctrl, n):
        d.ctrl[:] = ctrl
        for _ in range(n):
            pend = []
            if on[0]:
                mujoco.mj_forward(m, d)
                pend = sample()
            mujoco.mj_step(m, d)
            finish(pend)
        mujoco.mj_rnePostConstraint(m, d)
    # 이 검증은 물리 스텝 직후 값(유한 차분)이 필요해 루프를 직접 바꿔 끼운다. 토크 행동만 맞게 재현한다
    # (PD는 CommandAnt 루프에서 토크를 계산하므로 여기서는 지원하지 않음).
    if env.action_mode != "torque":
        raise NotImplementedError("contact_check.py는 토크 행동 run만 지원한다")
    uw._step_mujoco_simulation = step_sim

    obs, _ = env.reset(seed=0)
    for t in range(STEPS):
        on[0] = t >= WARMUP
        obs, _, term, _, _ = env.step(policy.act(obs, deterministic=deterministic))
        if term:
            break
    env.close()
    return rows


def summarize(rows):
    A = lambda k: np.array([r[k] for r in rows])
    n = lambda x: np.linalg.norm(x, axis=1)
    dist, fz, ft = A("dist"), A("fz"), A("ft")
    vp, vfd, vc, vb, vtr, vrot, vleg, u = (A(k) for k in ("v_p", "v_fd", "v_center", "v_body", "v_trans", "v_rot", "v_leg", "u"))
    touching = dist <= 0
    loaded = fz >= 0.25
    slip = n(vp)
    sel = loaded
    ftn = n(ft)
    sliding = loaded & (slip > 0.05) & (ftn > 1e-6)
    cos = (ft[sliding] * vp[sliding]).sum(1) / (ftn[sliding] * slip[sliding])
    ratio = ftn / np.maximum(fz * 8.936, 1e-9)        # |F_t| / F_n (fz는 mg 단위 → N)
    bins = [0, 0.25, 0.5, 0.75, 0.9, 1.01, 10]
    by_ratio = {f"{a:g}-{b:g}": dict(frac=float(((ratio >= a) & (ratio < b) & loaded).sum() / loaded.sum()),
                                     slip_median=float(np.median(slip[(ratio >= a) & (ratio < b) & loaded]))
                                     if (((ratio >= a) & (ratio < b) & loaded).sum() > 20) else float("nan"))
                for a, b in zip(bins[:-1], bins[1:])}
    along = lambda v: (v * u[sel]).sum(1)   # 모두 하중 접촉(sel)에만 쓰임
    return dict(
        n_contacts=len(rows),
        torso_vel_check_max=float(A("chk").max()),
        # margin
        frac_contacts_not_touching=float((~touching).mean()),
        fz_not_touching_median_mg=float(np.median(fz[~touching])) if (~touching).any() else float("nan"),
        frac_not_touching_with_fz_ge_0_25=float((loaded & ~touching).sum() / max(loaded.sum(), 1)),
        # 1-1, 1-2 (하중 접촉 기준)
        fd_vs_analytic_median_abs=float(np.median(n(vp[sel] - vfd[sel]))),
        fd_vs_analytic_rel=float(np.median(n(vp[sel] - vfd[sel]) / np.maximum(n(vfd[sel]), 1e-6))),
        slip_contact_point=float(np.median(slip[sel])), slip_fd=float(np.median(n(vfd[sel]))),
        speed_foot_center=float(np.median(n(vc[sel]))),
        # 1-3, 1-4
        body_point_speed=float(np.median(n(vb[sel]))),
        body_trans_speed=float(np.median(n(vtr[sel]))), body_rot_speed=float(np.median(n(vrot[sel]))),
        leg_rel_speed=float(np.median(n(vleg[sel]))),
        cancel_ratio=float(np.median(slip[sel] / np.maximum(n(vb[sel]), 1e-6))),
        leg_rel_along_cmd=float(np.mean(along(vleg[sel]))),            # < 0 이면 다리가 발을 몸 뒤로 쓸어 보냄
        body_point_along_cmd=float(np.mean(along(vb[sel]))),
        slip_along_cmd=float(np.mean(along(vp[sel]))),
        frac_leg_sweeps_back=float((along(vleg[sel]) < -0.05).mean()),
        # 2-4, 1-5
        cos_force_vs_slip_median=float(np.median(cos)), frac_cos_lt_minus_0_9=float((cos < -0.9).mean()),
        # 2-2
        friction_ratio_median=float(np.median(ratio[loaded])), friction_ratio_p90=float(np.quantile(ratio[loaded], 0.9)),
        frac_ratio_ge_0_9=float((ratio[loaded] >= 0.9).mean()),
        slip_by_friction_ratio=by_ratio,
        slip_when_ratio_lt_0_5=float(np.median(slip[loaded & (ratio < 0.5)])) if (loaded & (ratio < 0.5)).sum() > 20 else float("nan"),
    )


def main():
    res = {}
    for cfg in RUNS:
        for mode, det in (("off", True), ("on", False)):
            rows = []
            for sfx in ("", "_s1", "_s2"):
                pol = Policy(f"runs/{cfg}{sfx}")
                for c in command_grid():
                    rows += rollout(pol, c, det)
            res[f"{cfg}/{mode}"] = s = summarize(rows)
            print(cfg, mode, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k != "slip_by_friction_ratio"}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "contact_check.json").write_text(json.dumps(res, indent=1))
    print("saved →", OUT / "contact_check.json")


if __name__ == "__main__":
    main()
