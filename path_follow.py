"""웨이포인트 경로 추종.

상위 로직은 단순하다: 다음 웨이포인트 방향으로 속도 명령을 준다.
하위 보행은 학습된 정책이 맡는다. (상위 명령 → 하위 정책 구조)

예시:
    python path_follow.py --run runs/main
    python path_follow.py --run runs/main --video   # GIF 저장 (화면 렌더링 필요)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from envs import make_command_ant
from policy_utils import Policy

PATHS = {
    "square": [(3, 0), (3, 3), (0, 3), (0, 0)],
    "zigzag": [(2, 1.5), (4, -1.5), (6, 1.5), (8, -1.5)],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--path", choices=PATHS, default="square")
    p.add_argument("--speed", type=float, default=0.8)
    p.add_argument("--energy", type=float, default=0.5)
    p.add_argument("--radius", type=float, default=0.4, help="웨이포인트 도착 판정 반경 [m]")
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--video", action="store_true")
    args = p.parse_args()

    policy = Policy(args.run)
    waypoints = np.array(PATHS[args.path], dtype=float)
    env = make_command_ant(
        render_mode="rgb_array" if args.video else None,
        max_episode_steps=args.max_steps + 10,
    )
    env.set_command(0.0, 0.0, args.energy)
    obs, _ = env.reset(seed=0)

    traj, frames = [env.torso_xy()], []
    idx = 0
    for t in range(args.max_steps):
        pos = env.torso_xy()
        to_wp = waypoints[idx] - pos
        dist = np.linalg.norm(to_wp)
        if dist < args.radius:
            idx += 1
            if idx == len(waypoints):
                print(f"모든 웨이포인트 도착: {t} steps ({t * env.unwrapped.dt:.1f} s)")
                break
            to_wp = waypoints[idx] - pos
            dist = np.linalg.norm(to_wp)
        # 가까워지면 감속해서 지나치지 않게
        speed = min(args.speed, 1.5 * dist)
        cmd = speed * to_wp / max(dist, 1e-6)
        env.set_command(cmd[0], cmd[1])

        obs, _, term, _, _ = env.step(policy.act(obs))
        traj.append(env.torso_xy())
        if args.video and t % 2 == 0:
            frames.append(env.render())
        if term:
            print(f"넘어짐: step {t}")
            break
    else:
        print(f"시간 초과: {idx}/{len(waypoints)}개 도착")

    traj = np.array(traj)
    out = Path(args.run) / "results"
    out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(traj[:, 0], traj[:, 1], color="tab:blue", lw=1.5, label="torso trajectory")
    route = np.vstack([[0, 0], waypoints])
    ax.plot(route[:, 0], route[:, 1], "--", color="gray", lw=1, label="planned route")
    for i, (x, y) in enumerate(waypoints):
        ax.add_patch(plt.Circle((x, y), args.radius, color="tab:orange", alpha=0.25))
        ax.text(x, y, str(i + 1), ha="center", va="center", fontsize=9)
    ax.scatter(*traj[0], color="tab:green", zorder=3, label="start")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Waypoint following ({args.path}, speed {args.speed} m/s)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / f"path_{args.path}.png", dpi=150)
    print(f"saved → {out / f'path_{args.path}.png'}")

    if args.video and frames:
        import imageio.v2 as imageio
        imageio.mimsave(out / f"path_{args.path}.gif", frames, duration=env.unwrapped.dt * 2)
        print(f"saved → {out / f'path_{args.path}.gif'}")
    env.close()


if __name__ == "__main__":
    main()
