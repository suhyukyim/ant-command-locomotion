"""디딤별 수평 충격량과 '체중을 싣는 디딤(하중 디딤)' 분석. propulsion_analysis.py의 후속.

1) 수평 충격량 (디딤 = 한 발이 연속으로 닿은 물리 스텝 구간, dt = 0.01 s)
     J_prop  = Σ max(F_a, 0) dt      명령 방향 추진 충격량
     J_brake = Σ min(F_a, 0) dt      제동 충격량 (음수)
     J_net   = Σ F_a dt              명령 방향 순충격량
     J_h     = Σ |F_수평| dt          방향 무관 수평 충격량 크기
   F_a = 월드 좌표계 수평 반력을 명령 방향 단위벡터에 투영한 값 (propulsion_analysis.py와 같은 Recorder).
2) 하중 디딤: 발 하나의 수직 반력 F_z가 기준 이상인 연속 구간. 기준은 F_z 분포를 보고 정한다(--load-thr, 체중 mg 배수).
   그 구간의 지속 시간, 접촉점 미끄러짐, 명령 방향 순충격량을 잰다.

예:
    python stance_load_analysis.py --record          # 롤아웃을 돌려 원자료 저장 (runs/propulsion_v7/raw.npz, git 제외)
    python stance_load_analysis.py --load-thr 0.25   # 저장된 원자료로 분석 → runs/propulsion_v7/stance_load.json, fz_distribution.png
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from envs import make_command_ant
from gait_analysis import STEPS, WARMUP, command_grid
from policy_utils import Policy
from propulsion_analysis import OUT, Recorder

CONFIGS = ["main_v5b", "main_v6Bb", "main_v7Ab", "main_v7Bb"]
SEEDS = ["", "_s1", "_s2"]
DT = 0.01
RAW = OUT / "raw.npz"


def rollout(policy, cmd, deterministic):
    env = make_command_ant(**{**policy.env_kwargs, "max_episode_steps": STEPS + 10})
    env.set_command(cmd[0], cmd[1], 0.5)
    u = np.asarray(cmd) / np.linalg.norm(cmd)
    rec = Recorder(env, u)
    obs, _ = env.reset(seed=0)
    vel = []
    for t in range(STEPS):
        rec.on = t >= WARMUP
        obs, _, term, _, info = env.step(policy.act(obs, deterministic=deterministic))
        if term:
            break
        if t >= WARMUP:
            vel.append(info["velocity"])
    mg = float(env.unwrapped.model.body_mass.sum()) * 9.81
    env.close()
    if len(rec.rows) < 100:          # 워밍업 전에 넘어진 에피소드는 제외
        return None
    con, fz, fa, slip, fh, _ = (np.array(x) for x in zip(*rec.rows))
    return dict(con=con, fz=fz / mg, fa=fa, slip=slip, fh=fh, speed=float(np.mean(vel, 0) @ u) if vel else np.nan)


def record():
    data = {}
    for cfg in CONFIGS:
        for sfx in SEEDS:
            pol = Policy(f"runs/{cfg}{sfx}")
            for mode, det in (("off", True), ("on", False)):
                for i, c in enumerate(command_grid()):
                    r = rollout(pol, c, det)
                    if r is None:
                        print("  skip (ended before warmup):", cfg + sfx, mode, i)
                        continue
                    key = f"{cfg}{sfx}|{mode}|{i}"
                    for k, v in r.items():
                        data[f"{key}|{k}"] = np.asarray(v, dtype=np.float32 if k != "con" else bool)
            print("recorded", cfg + sfx, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW, **data)
    print("saved →", RAW)


def episodes(raw, cfg, mode):
    for sfx in SEEDS:
        for i in range(16):
            key = f"{cfg}{sfx}|{mode}|{i}"
            if f"{key}|con" in raw:
                yield sfx, {k: raw[f"{key}|{k}"] for k in ("con", "fz", "fa", "slip", "fh", "speed")}


def segments(mask):
    c = np.r_[False, mask, False].astype(int)
    return zip(np.where(np.diff(c) == 1)[0], np.where(np.diff(c) == -1)[0])


def impulse_stats(ep):
    rows = []
    for k in range(4):
        for a, b in segments(ep["con"][:, k]):
            f = ep["fa"][a:b, k].astype(float)
            rows.append((b - a, np.clip(f, 0, None).sum() * DT, np.clip(f, None, 0).sum() * DT, f.sum() * DT,
                         ep["fh"][a:b, k].sum() * DT, ep["fz"][a:b, k].sum() * DT))
    return np.array(rows).reshape(-1, 6)


def load_stats(ep, thr):
    out = []
    for k in range(4):
        loaded = ep["con"][:, k] & (ep["fz"][:, k] >= thr)
        for a, b in segments(loaded):
            s = ep["slip"][a:b, k].astype(float)
            out.append((b - a, np.nanmean(s), np.nanmean(s > 0.1), ep["fa"][a:b, k].sum() * DT))
    return np.array(out).reshape(-1, 4)


def analyze(thr):
    raw = np.load(RAW)
    res = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    colors = dict(main_v5b="#6b6b66", main_v6Bb="#2a78d6", main_v7Ab="#eda100", main_v7Bb="#eb6834")
    bins = np.linspace(0, 4, 81)
    for col, mode in enumerate(("off", "on")):
        for cfg in CONFIGS:
            per_seed = {}
            fz_all = []
            for sfx, ep in episodes(raw, cfg, mode):
                fz_all.append(ep["fz"][ep["con"]])
                d = per_seed.setdefault(sfx, dict(imp=[], load=[], T=0.0, speed=[], ep=[]))
                imp, ld = impulse_stats(ep), load_stats(ep, thr)
                d["imp"].append(imp); d["load"].append(ld); d["T"] += len(ep["con"]) * DT
                d["speed"].append(float(ep["speed"]))
                # 에피소드 단위 지표 (명령별 속도와의 관계를 보기 위해)
                T = len(ep["con"]) * DT
                d["ep"].append((float(ep["speed"]), imp[:, 3].sum() / T, imp[:, 1].sum() / T,
                                ld[:, 0].sum() * DT / T if len(ld) else 0.0,
                                np.mean(ld[:, 0] * DT) if len(ld) else 0.0))
            fz_all = np.concatenate(fz_all)
            seeds = []
            for sfx, d in per_seed.items():
                imp = np.concatenate(d["imp"]); ld = np.concatenate(d["load"]); T = d["T"]
                epm = np.array(d["ep"])
                long_ = imp[:, 0] >= 3
                seeds.append(dict(
                    J_prop_per_stance=imp[:, 1].mean(), J_brake_per_stance=imp[:, 2].mean(),
                    J_net_per_stance=imp[:, 3].mean(), J_h_per_stance=imp[:, 4].mean(),
                    J_prop_per_s=imp[:, 1].sum() / T, J_brake_per_s=imp[:, 2].sum() / T,
                    J_net_per_s=imp[:, 3].sum() / T, J_h_per_s=imp[:, 4].sum() / T,
                    # 수평 충격량 중 명령 방향 순충격량으로 남는 몫 (상쇄 정도)
                    net_over_gross=imp[:, 3].sum() / max((imp[:, 1] - imp[:, 2]).sum(), 1e-9),
                    frac_net_pos=float((imp[:, 3] > 0).mean()),
                    # 순추진에 기여하는 디딤 길이: 길이별 J_net 합의 몫
                    net_share_1_2_steps=imp[~long_, 3].sum() / max(abs(imp[:, 3]).sum(), 1e-9),
                    load_time_frac=ld[:, 0].sum() * DT / T / 4,            # 발 하나당 하중을 싣는 시간 비율
                    load_per_s=len(ld) / T,
                    load_dur_mean=float(ld[:, 0].mean() * DT) if len(ld) else np.nan,
                    load_dur_ge_0_05=float((ld[:, 0] >= 5).mean()) if len(ld) else np.nan,
                    load_slip_mean=float(np.nanmean(ld[:, 1])) if len(ld) else np.nan,
                    load_slip_gt_0_1=float(np.nanmean(ld[:, 2])) if len(ld) else np.nan,
                    load_net_imp_per_s=float(ld[:, 3].sum() / T) if len(ld) else np.nan,
                    corr_speed_vs_load_time=float(np.corrcoef(epm[:, 0], epm[:, 3])[0, 1]),
                    corr_speed_vs_load_dur=float(np.corrcoef(epm[:, 0], epm[:, 4])[0, 1]),
                    corr_speed_vs_prop_per_s=float(np.corrcoef(epm[:, 0], epm[:, 2])[0, 1]),
                ))
            ks = seeds[0].keys()
            q = np.quantile(fz_all, [0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
            res[f"{cfg}/{mode}"] = dict(
                mean={k: float(np.nanmean([s[k] for s in seeds])) for k in ks},
                std={k: float(np.nanstd([s[k] for s in seeds], ddof=1)) for k in ks},
                fz_quantiles_mg=dict(zip(["p10", "p25", "p50", "p75", "p90", "p99"], q.round(3).tolist())),
                fz_frac_below_thr=float((fz_all < thr).mean()))
            colors.setdefault(cfg, f"C{len(colors) % 10}")
            h, _ = np.histogram(np.clip(fz_all, 0, 4), bins=bins, density=True)
            axes[col].step(bins[:-1], h, where="post", color=colors[cfg], lw=1.6,
                           label=f"{cfg.replace('main_', '')} (median {q[2]:.2f} mg)")
        axes[col].axvline(thr, color="#222", ls="--", lw=1)
        axes[col].set_title(f"noise {mode}: per-foot vertical force while in contact")
        axes[col].set_xlabel("F_z per foot [× body weight mg]")
        axes[col].set_yscale("log")
        axes[col].legend(fontsize=8, frameon=False)
        axes[col].grid(color="#e4e3dd"); axes[col].spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("density (log)")
    fig.tight_layout(); fig.savefig(OUT / "fz_distribution.png", dpi=110)
    (OUT / "stance_load.json").write_text(json.dumps(dict(load_thr_mg=thr, result=res), indent=1))
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--record", action="store_true")
    p.add_argument("--load-thr", type=float, default=0.25, help="하중 디딤 기준: 발 하나의 F_z ≥ thr × mg")
    p.add_argument("--runs", nargs="+", default=CONFIGS, help="설정 이름 (seed 접미사 _s1 _s2 자동 포함)")
    p.add_argument("--out", default=str(OUT), help="결과 폴더 (기본 runs/propulsion_v7)")
    a = p.parse_args()
    CONFIGS[:] = a.runs
    OUT = Path(a.out)
    RAW = OUT / "raw.npz"
    if a.record or not RAW.exists():
        record()
    r = analyze(a.load_thr)
    for k, v in r.items():
        print(k, v["fz_quantiles_mg"], {kk: round(vv, 3) for kk, vv in v["mean"].items()})
