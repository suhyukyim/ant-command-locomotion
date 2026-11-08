"""키보드로 Ant를 실시간 조종하는 데모 (Linux).

MuJoCo 뷰어 창이 뜨고, 키 입력은 이 스크립트를 실행한 *터미널*에서 받는다.
(뷰어 창 단축키와 겹치지 않게 하려고 터미널 입력을 쓴다)

    w / s : 앞(+x) / 뒤(-x)
    a / d : 왼쪽(+y) / 오른쪽(-y)
    + / - : 속도 올리기 / 내리기
    1 2 3 : 에너지 수준 0.0 / 0.5 / 1.0
    x     : 정지 명령
    r     : 리셋
    q     : 종료

예시:
    python keyboard_demo.py --run runs/main
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty

import mujoco.viewer
import numpy as np

from envs import make_command_ant
from policy_utils import Policy

DIRS = {"w": (1, 0), "s": (-1, 0), "a": (0, 1), "d": (0, -1)}
ENERGY = {"1": 0.0, "2": 0.5, "3": 1.0}


def read_key():
    """터미널에서 눌린 키 하나를 기다리지 않고 읽는다. 없으면 None."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    args = p.parse_args()

    policy = Policy(args.run)
    env = make_command_ant(max_episode_steps=10**9)
    direction, speed, energy = np.zeros(2), 0.8, 0.5
    env.set_command(0.0, 0.0, energy)
    obs, _ = env.reset(seed=0)
    model, data = env.unwrapped.model, env.unwrapped.data
    dt = env.unwrapped.dt

    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    print(__doc__)
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.trackbodyid = 1  # 몸통 따라가기
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            while viewer.is_running():
                t0 = time.time()
                key = read_key()
                if key == "q":
                    break
                if key in DIRS:
                    direction = np.array(DIRS[key], dtype=float)
                elif key == "x":
                    direction = np.zeros(2)
                elif key == "+":
                    speed = min(speed + 0.1, 1.5)
                elif key == "-":
                    speed = max(speed - 0.1, 0.1)
                elif key in ENERGY:
                    energy = ENERGY[key]
                elif key == "r":
                    obs, _ = env.reset()
                if key:
                    cmd = speed * direction
                    print(f"\rcmd=({cmd[0]:+.1f}, {cmd[1]:+.1f}) m/s  energy={energy:.1f}   ",
                          end="", flush=True)

                cmd = speed * direction
                env.set_command(cmd[0], cmd[1], energy)
                obs, _, term, _, _ = env.step(policy.act(obs))
                if term:
                    obs, _ = env.reset()
                viewer.sync()
                time.sleep(max(0.0, dt - (time.time() - t0)))
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        env.close()
        print()


if __name__ == "__main__":
    main()
