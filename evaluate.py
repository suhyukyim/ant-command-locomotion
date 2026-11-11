"""학습된 정책을 평가하고 그래프를 만든다.

1. 명령 vs 실제 속도: 8방향 × 2속도 명령을 주고 실제로 얼마나 따라가는지
   - 노이즈 끔(deterministic, 평균 행동만)과 노이즈 켬(stochastic, 학습 때처럼) 두 가지로 잰다
2. 에너지 수준 비교: 같은 속도 명령에서 에너지 수준에 따라 제어 노력과 속도가 어떻게 바뀌는지
3. 행동 진단 (명령 1 m/s 앞, 노이즈 끔): 행동 포화, 행동의 시간 변화, 관절 각속도
   - 정책이 관절을 한계까지 밀어 몸을 굳히고 있는지 확인한다

예시:
    python evaluate.py --run runs/main
결과는 runs/main/results/ 에 저장된다.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from envs import make_command_ant
from policy_utils import Policy

SPEEDS = [0.5, 1.0]
N_DIRS = 8
# 노이즈 끔 기준, 행동이 한계(±1)에 붙어 있다고 보는 문턱
SAT_THRESHOLD = 0.99


def rollout(policy, vx, vy, energy, steps=300, warmup=60, seed=0, deterministic=True):
    """명령을 고정해 한 번 굴려보고, 워밍업 이후 통계를 돌려준다."""
    env = make_command_ant(**{**policy.env_kwargs, "max_episode_steps": steps + 10})
    env.set_command(vx, vy, energy)
    obs, _ = env.reset(seed=seed)
    data = env.unwrapped.data
    vels, efforts, powers, actions, joint_vel = [], [], [], [], []
    fell = False
    for t in range(steps):
        action = policy.act(obs, deterministic=deterministic)
        obs, _, term, trunc, info = env.step(action)
        if t >= warmup:
            vels.append(info["velocity"])
            efforts.append(info["effort"])
            powers.append(info.get("power", np.nan))
            actions.append(np.clip(action, -1.0, 1.0))
            joint_vel.append(np.abs(data.qvel[6:14]))  # 관절 8개 각속도
        if term:
            fell = True
            break
    env.close()
    v = np.mean(vels, axis=0) if vels else np.zeros(2)
    return dict(
        vel=v,
        effort=float(np.mean(efforts)) if efforts else float("nan"),
        power=float(np.mean(powers)) if powers else float("nan"),
        fell=fell,
        actions=np.array(actions),
        joint_vel=float(np.mean(joint_vel)) if joint_vel else float("nan"),
    )


def plot_tracking(policy, out: Path, energy=0.5, deterministic=True):
    angles = np.linspace(-math.pi, math.pi, N_DIRS, endpoint=False)
    rows = []
    fig, ax = plt.subplots(figsize=(6, 6))
    for s in SPEEDS:
        for a in angles:
            cmd = np.array([s * math.cos(a), s * math.sin(a)])
            r = rollout(policy, *cmd, energy, deterministic=deterministic)
            v, fell = r["vel"], r["fell"]
            err = float(np.linalg.norm(v - cmd))
            along = float(v @ cmd / np.linalg.norm(cmd))  # 명령 방향으로 얼마나 갔나
            rows.append(dict(cmd=cmd.tolist(), vel=v.tolist(), error=err,
                             along_cmd_speed=along, fell=fell))
            ax.scatter(*cmd, marker="x", color="gray", s=60, zorder=2)
            ax.annotate("", xy=v, xytext=cmd,
                        arrowprops=dict(arrowstyle="->", color="tab:red" if fell else "tab:blue"))
            ax.scatter(*v, color="tab:red" if fell else "tab:blue", s=25, zorder=3)
    ax.scatter([], [], marker="x", color="gray", label="command")
    ax.scatter([], [], color="tab:blue", label="achieved velocity")
    ax.scatter([], [], color="tab:red", label="fell")
    ax.set_xlabel("vx [m/s]")
    ax.set_ylabel("vy [m/s]")
    mode = "deterministic" if deterministic else "stochastic"
    ax.set_title(f"Commanded vs achieved velocity ({mode}, energy={energy})")
    ax.axhline(0, color="lightgray", lw=0.8)
    ax.axvline(0, color="lightgray", lw=0.8)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / ("tracking.png" if deterministic else "tracking_stochastic.png"), dpi=150)
    plt.close(fig)
    return rows


def plot_energy(policy, out: Path, cmd=(1.0, 0.0)):
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    efforts, speeds, powers = [], [], []
    for e in levels:
        r = rollout(policy, *cmd, e)
        efforts.append(r["effort"])
        powers.append(r["power"])
        speeds.append(float(np.linalg.norm(r["vel"])))
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(levels, efforts, "o-", color="tab:orange", label="effort Σ(τ/150)²")
    ax1.set_xlabel("energy level (0 = save energy, 1 = free)")
    ax1.set_ylabel("effort Σ(τ/150)² per step", color="tab:orange")
    ax2 = ax1.twinx()
    ax2.plot(levels, speeds, "s--", color="tab:blue", label="speed")
    ax2.axhline(np.linalg.norm(cmd), color="tab:blue", lw=0.8, alpha=0.4)
    ax2.set_ylabel("achieved speed [m/s]", color="tab:blue")
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0, top=max(1.2 * np.linalg.norm(cmd), max(speeds) * 1.1))
    ax1.set_title(f"Energy level vs effort (command = {cmd[0]:.1f} m/s forward)")
    fig.tight_layout()
    fig.savefig(out / "energy.png", dpi=150)
    plt.close(fig)
    return dict(levels=levels, effort=efforts, speed=speeds, power=powers)


def action_diagnostics(policy, cmd=(1.0, 0.0), energy=0.5):
    """노이즈 끔 상태에서 정책이 관절을 한계에 붙여 굳히고 있는지 본다."""
    r = rollout(policy, *cmd, energy, deterministic=True)
    acts = r["actions"]
    return dict(
        command=list(cmd),
        saturation_ratio=float(np.mean(np.abs(acts) > SAT_THRESHOLD)),
        saturation_per_joint=np.mean(np.abs(acts) > SAT_THRESHOLD, axis=0).round(3).tolist(),
        mean_abs_action=float(np.mean(np.abs(acts))),
        action_temporal_std=float(acts.std(axis=0).mean()),  # 0이면 매 스텝 같은 자세
        joint_velocity=r["joint_vel"],
    )


def summarize(rows):
    return dict(
        mean_tracking_error_mps=float(np.mean([r["error"] for r in rows])),
        mean_along_cmd_speed=float(np.mean([r["along_cmd_speed"] for r in rows])),
        falls=int(sum(r["fell"] for r in rows)),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", default="final_model")
    args = p.parse_args()

    torch.manual_seed(0)  # 노이즈 켬 평가를 재현 가능하게
    policy = Policy(args.run, args.checkpoint)
    out = Path(args.run) / "results"
    out.mkdir(exist_ok=True)

    tracking = plot_tracking(policy, out, deterministic=True)
    tracking_stoch = plot_tracking(policy, out, deterministic=False)
    energy = plot_energy(policy, out)
    diag = action_diagnostics(policy)
    det, sto = summarize(tracking), summarize(tracking_stoch)
    standing_error = float(np.mean([s for s in SPEEDS for _ in range(N_DIRS)]))

    summary = dict(
        # 기존 키 (노이즈 끔 기준)
        mean_tracking_error_mps=det["mean_tracking_error_mps"],
        falls=det["falls"],
        tracking=tracking,
        energy=energy,
        # 추가 키
        standing_still_error_mps=standing_error,
        deterministic=det,
        stochastic=sto,
        tracking_stochastic=tracking_stoch,
        action_diagnostics=diag,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"서 있기만 할 때의 오차(기준선): {standing_error:.3f} m/s")
    for name, s in [("노이즈 끔", det), ("노이즈 켬", sto)]:
        print(f"[{name}] 평균 오차 {s['mean_tracking_error_mps']:.3f} m/s, "
              f"명령 방향 속도 {s['mean_along_cmd_speed']:.3f} m/s, 넘어짐 {s['falls']}/{len(tracking)}")
    print(f"[행동 진단] 포화 비율 {diag['saturation_ratio']:.2f}, "
          f"행동 시간 변화 {diag['action_temporal_std']:.3f}, 관절 각속도 {diag['joint_velocity']:.3f} rad/s")
    print("effort by energy level:", [round(x, 3) for x in energy["effort"]])
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
