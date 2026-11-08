"""학습된 정책을 평가하고 그래프를 만든다.

1. 명령 vs 실제 속도: 8방향 × 2속도 명령을 주고 실제로 얼마나 따라가는지
2. 에너지 수준 비교: 같은 속도 명령에서 에너지 수준에 따라 제어 노력과 속도가 어떻게 바뀌는지

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

from envs import make_command_ant
from policy_utils import Policy


def rollout(policy, vx, vy, energy, steps=300, warmup=60, seed=0):
    """명령을 고정해 한 번 굴려보고, 워밍업 이후 평균 속도·노력·생존 여부를 돌려준다."""
    env = make_command_ant(max_episode_steps=steps + 10)
    env.set_command(vx, vy, energy)
    obs, _ = env.reset(seed=seed)
    vels, efforts = [], []
    fell = False
    for t in range(steps):
        obs, _, term, trunc, info = env.step(policy.act(obs))
        if t >= warmup:
            vels.append(info["velocity"])
            efforts.append(info["effort"])
        if term:
            fell = True
            break
    env.close()
    v = np.mean(vels, axis=0) if vels else np.zeros(2)
    return v, float(np.mean(efforts)) if efforts else float("nan"), fell


def plot_tracking(policy, out: Path, energy=0.5):
    speeds = [0.5, 1.0]
    angles = np.linspace(-math.pi, math.pi, 8, endpoint=False)
    rows = []
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, s in enumerate(speeds):
        for a in angles:
            cmd = np.array([s * math.cos(a), s * math.sin(a)])
            v, eff, fell = rollout(policy, *cmd, energy)
            err = float(np.linalg.norm(v - cmd))
            rows.append(dict(cmd=cmd.tolist(), vel=v.tolist(), error=err, fell=fell))
            ax.scatter(*cmd, marker="x", color="gray", s=60, zorder=2)
            ax.annotate("", xy=v, xytext=cmd,
                        arrowprops=dict(arrowstyle="->", color="tab:red" if fell else "tab:blue"))
            ax.scatter(*v, color="tab:red" if fell else "tab:blue", s=25, zorder=3)
    ax.scatter([], [], marker="x", color="gray", label="command")
    ax.scatter([], [], color="tab:blue", label="achieved velocity")
    ax.scatter([], [], color="tab:red", label="fell")
    ax.set_xlabel("vx [m/s]")
    ax.set_ylabel("vy [m/s]")
    ax.set_title(f"Commanded vs achieved velocity (energy={energy})")
    ax.axhline(0, color="lightgray", lw=0.8)
    ax.axvline(0, color="lightgray", lw=0.8)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "tracking.png", dpi=150)
    plt.close(fig)
    return rows


def plot_energy(policy, out: Path, cmd=(1.0, 0.0)):
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    efforts, speeds = [], []
    for e in levels:
        v, eff, _ = rollout(policy, *cmd, e)
        efforts.append(eff)
        speeds.append(float(np.linalg.norm(v)))
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(levels, efforts, "o-", color="tab:orange", label="control effort Σa²")
    ax1.set_xlabel("energy level (0 = save energy, 1 = free)")
    ax1.set_ylabel("control effort Σa² per step", color="tab:orange")
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
    return dict(levels=levels, effort=efforts, speed=speeds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", default="final_model")
    args = p.parse_args()

    policy = Policy(args.run, args.checkpoint)
    out = Path(args.run) / "results"
    out.mkdir(exist_ok=True)

    tracking = plot_tracking(policy, out)
    energy = plot_energy(policy, out)
    errs = [r["error"] for r in tracking]
    summary = dict(
        mean_tracking_error_mps=float(np.mean(errs)),
        falls=int(sum(r["fell"] for r in tracking)),
        tracking=tracking,
        energy=energy,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"mean tracking error: {summary['mean_tracking_error_mps']:.3f} m/s, "
          f"falls: {summary['falls']}/{len(tracking)}")
    print("effort by energy level:", [round(x, 3) for x in energy["effort"]])
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
