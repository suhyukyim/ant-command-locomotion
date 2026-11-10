"""정책이 '진짜 걷는지' 확인한다: 영상 + 관절 각도 주기성 + 발 접촉 패턴 + 몸통 높이.

속도·관절 각속도만으로는 아래 셋을 구분할 수 없다.
  - 다리를 번갈아 디디며 주기적으로 걷기
  - 몸통/다리를 바닥에 끌며 밀고 가기
  - 불규칙하게 떨면서 조금씩 미끄러지기
그래서 evaluate.py와 같은 조건(노이즈 끔, 300스텝, 워밍업 60, 평가 seed 0)으로 굴리면서
  - 관절 각도의 자기상관 → 반복 주기와 주기성 강도 (1 = 완벽히 반복, 0 = 반복 없음)
  - 발(ankle geom)별 바닥 접촉 비율과 착지 횟수 → 발을 실제로 떼고 디디는지
  - 몸통·윗다리의 바닥 접촉, 몸통 높이 → 끌고 가는지
를 잰다.

예시:
    python gait_analysis.py                                  # 기본 비교 세트
    python gait_analysis.py --case runs/main_v5b:9 --no-video  # run:명령 번호(0~15, summary.json 순서)
결과: runs/gait/ (gait_<이름>.png, <이름>.gif, gait_summary.json)
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
import mujoco
import numpy as np

from envs import make_command_ant
from policy_utils import Policy

STEPS, WARMUP = 300, 60       # evaluate.py와 같게
FEET = ["left_ankle_geom", "right_ankle_geom", "third_ankle_geom", "fourth_ankle_geom"]
UPPER = ["left_leg_geom", "right_leg_geom", "back_leg_geom", "rightback_leg_geom",
         "aux_1_geom", "aux_2_geom", "aux_3_geom", "aux_4_geom"]
LEG_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # 다리 1~4 (고정 순서)
INK, MUTED, GRID = "#222222", "#6b6b66", "#e4e3dd"

# 기본 비교 세트: (이름, run, summary.json의 명령 번호, 설명)
#   명령 번호: 0~7 = 0.5 m/s, 8~15 = 1.0 m/s, 방향 −180°부터 45°씩
#   주의: 명령별 성공/정지는 기계(CPU)마다 달라질 수 있다 (README 참고). 그래서 제목에는 그 자리에서 잰 속도를 쓴다.
DEFAULT_CASES = [
    ("v5b_s0_1.0_270", "runs/main_v5b", 10, "v5b seed 0, cmd 1.0 m/s @ 270 deg"),
    ("v5b_s0_1.0_225", "runs/main_v5b", 9, "v5b seed 0, cmd 1.0 m/s @ 225 deg"),
    ("v5b_s0_0.5_90", "runs/main_v5b", 6, "v5b seed 0, cmd 0.5 m/s @ 90 deg"),
    ("v5b_s0_1.0_180", "runs/main_v5b", 8, "v5b seed 0, cmd 1.0 m/s @ 180 deg"),
]


def command_grid():
    """evaluate.py와 같은 16개 명령 (0~7 = 0.5 m/s, 8~15 = 1.0 m/s, 방향 −180°부터 45°씩)."""
    angles = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    return [[s * float(np.cos(a)), s * float(np.sin(a))] for s in (0.5, 1.0) for a in angles]


def record(policy, cmd, energy=0.5, seed=0, video=False):
    env = make_command_ant(max_episode_steps=STEPS + 10,
                           render_mode="rgb_array" if video else None)
    env.set_command(cmd[0], cmd[1], energy)
    obs, _ = env.reset(seed=seed)
    m, d = env.unwrapped.model, env.unwrapped.data
    if video:  # 기본 카메라는 바라보는 점이 고정이라 걸어가면 화면 밖으로 나간다 → 매 프레임 몸통을 보게
        env.render()
        cam = env.unwrapped.mujoco_renderer._get_viewer("rgb_array").cam
        cam.distance, cam.elevation = 3.5, -25.0
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    floor, torso = gid("floor"), gid("torso_geom")
    feet, upper = [gid(n) for n in FEET], {gid(n) for n in UPPER}

    log = dict(vel=[], qpos=[], height=[], foot=[], upper=[], torso=[])
    frames, fell = [], False
    for t in range(STEPS):
        action = policy.act(obs, deterministic=True)
        obs, _, term, _, info = env.step(action)
        if video and t % 2 == 0:
            cam.lookat[:] = d.qpos[:3]
            frames.append(env.render())
        if t >= WARMUP:
            touching = set()
            for i in range(d.ncon):
                c = d.contact[i]
                if c.geom1 == floor:
                    touching.add(c.geom2)
                elif c.geom2 == floor:
                    touching.add(c.geom1)
            log["vel"].append(info["velocity"])
            log["qpos"].append(d.qpos[7:15].copy())   # hip1, ankle1, ..., hip4, ankle4
            log["height"].append(d.qpos[2])
            log["foot"].append([f in touching for f in feet])
            log["upper"].append(bool(touching & upper))
            log["torso"].append(torso in touching)
        if term:
            fell = True
            break
    dt = env.unwrapped.dt
    env.close()
    return {k: np.array(v) for k, v in log.items()}, frames, fell, dt


def periodicity(q, dt, min_lag=4):
    """관절 각도의 정규화 자기상관에서 첫 번째 뚜렷한 봉우리 → (주기 s, 강도)."""
    x = q - q.mean(axis=0)
    var = (x ** 2).sum(axis=0)
    active = var > 1e-6 * len(x)                 # 거의 안 움직이는 관절은 제외
    if not active.any():
        return float("nan"), 0.0
    x = x[:, active]
    n = len(x)
    ac = np.array([(x[:n - k] * x[k:]).sum() / (x ** 2).sum() for k in range(n // 2)])
    peaks = [k for k in range(min_lag, len(ac) - 1) if ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]]
    if not peaks:
        return float("nan"), 0.0
    k = max(peaks, key=lambda i: ac[i]) if ac[peaks[0]] < 0.5 * max(ac[p] for p in peaks) else peaks[0]
    return k * dt, float(ac[k])


def metrics(log, cmd, dt):
    cmd = np.asarray(cmd)
    v = log["vel"].mean(axis=0)
    foot = log["foot"].astype(int)
    touchdowns = (np.diff(foot, axis=0) == 1).sum(axis=0)       # 발마다 0→1 전환 횟수
    dur = len(foot) * dt
    period, strength = periodicity(log["qpos"], dt)
    return dict(
        along_cmd_speed=float(v @ cmd / np.linalg.norm(cmd)),
        joint_range_deg=float(np.degrees(np.ptp(log["qpos"], axis=0)).mean()),
        period_s=period,
        periodicity=strength,
        foot_contact_ratio=foot.mean(axis=0).round(2).tolist(),
        touchdowns_per_s=(touchdowns / dur).round(2).tolist(),
        feet_on_ground_mean=float(foot.sum(axis=1).mean()),
        upper_leg_contact=float(log["upper"].mean()),
        torso_contact=float(log["torso"].mean()),
        torso_height_mean=float(log["height"].mean()),
        torso_height_min=float(log["height"].min()),
    )


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def plot(log, met, title, dt, path):
    t = WARMUP * dt + np.arange(len(log["height"])) * dt
    fig, axes = plt.subplots(4, 1, figsize=(9, 8.5), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 2, 1.4, 1.2]))
    q = np.degrees(log["qpos"])
    for leg in range(4):
        axes[0].plot(t, q[:, 2 * leg], color=LEG_COLORS[leg], lw=1.6, label=f"leg {leg + 1}")
        axes[1].plot(t, q[:, 2 * leg + 1], color=LEG_COLORS[leg], lw=1.6)
    axes[0].set_ylabel("hip angle [deg]", color=INK, fontsize=9)
    axes[1].set_ylabel("ankle angle [deg]", color=INK, fontsize=9)
    axes[0].legend(ncol=4, fontsize=8, frameon=False, loc="upper right",
                   bbox_to_anchor=(1, 1.22))

    # 접촉 다이어그램: 행마다 바닥에 닿은 구간을 막대로
    ax = axes[2]
    rows = [(f"foot {i + 1}", log["foot"][:, i], LEG_COLORS[i]) for i in range(4)]
    rows += [("upper leg", log["upper"], MUTED), ("torso", log["torso"], INK)]
    for r, (_, on, col) in enumerate(rows):
        y = len(rows) - 1 - r
        on = on.astype(bool)
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            ax.broken_barh([(t[s], (e - s) * dt)], (y - 0.35, 0.7), color=col, lw=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _, _ in rows][::-1], fontsize=8)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_ylabel("floor contact", color=INK, fontsize=9)

    axes[3].plot(t, log["height"], color=INK, lw=1.6)
    axes[3].set_ylabel("torso z [m]", color=INK, fontsize=9)
    axes[3].set_xlabel("time [s]", color=INK, fontsize=9)
    for a in axes:
        style(a)
    axes[2].grid(False)

    per = "none" if np.isnan(met["period_s"]) else f"{met['period_s']:.2f}s"
    fig.suptitle(f"{title}\nspeed along cmd {met['along_cmd_speed']:.2f} m/s · "
                 f"period {per} (strength {met['periodicity']:.2f}) · "
                 f"joint range {met['joint_range_deg']:.0f}° · "
                 f"touchdowns/s {sum(met['touchdowns_per_s']):.1f}",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", action="append", default=None,
                   help="run:명령번호 (여러 번 지정 가능). 생략하면 기본 비교 세트")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--out", default="runs/gait")
    args = p.parse_args()

    cases = DEFAULT_CASES
    if args.case:
        cases = []
        for c in args.case:
            run, idx = c.rsplit(":", 1)
            cases.append((f"{Path(run).name}_{idx}", run, int(idx), f"{run} cmd #{idx}"))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, run, idx, desc in cases:
        summary_path = Path(run) / "results" / "summary.json"
        if summary_path.exists():
            ref = json.loads(summary_path.read_text())["tracking"][idx]
        else:
            # v6부터는 evaluate.py를 돌리지 않아 summary.json이 없다 → evaluate.py와 같은 명령 순서로 만든다
            ref = dict(cmd=command_grid()[idx], along_cmd_speed=float("nan"))
        cmd = ref["cmd"]
        log, frames, fell, dt = record(Policy(run), cmd, video=not args.no_video)
        met = metrics(log, cmd, dt)
        met.update(run=run, cmd=cmd, fell=fell, eval_along_cmd_speed=ref["along_cmd_speed"])
        summary[name] = met
        plot(log, met, desc, dt, out / f"gait_{name}.png")
        if frames:
            import imageio.v2 as imageio
            # duration은 밀리초. 프레임은 2스텝마다 찍었으므로 실제 속도로 재생된다.
            imageio.mimsave(out / f"{name}.gif", [f[::2, ::2] for f in frames],
                            duration=dt * 2 * 1000, loop=0)
        per = "없음" if np.isnan(met["period_s"]) else f"{met['period_s']:.2f}s"
        print(f"[{name}] 속도 {met['along_cmd_speed']:.3f} (evaluate {ref['along_cmd_speed']:.3f}), "
              f"주기 {per} 강도 {met['periodicity']:.2f}, 관절 범위 {met['joint_range_deg']:.0f}°, "
              f"발 접촉 {met['foot_contact_ratio']}, 착지/s {met['touchdowns_per_s']}, "
              f"윗다리 접촉 {met['upper_leg_contact']:.2f}, 몸통 접촉 {met['torso_contact']:.2f}, "
              f"몸통 높이 {met['torso_height_mean']:.2f} (최저 {met['torso_height_min']:.2f})")
    (out / "gait_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
