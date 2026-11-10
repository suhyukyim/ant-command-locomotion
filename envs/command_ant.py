"""명령 조건부 Ant 환경.

MuJoCo Ant-v5를 감싸서, 정책이 다음 두 가지 명령을 입력으로 받도록 만든다.

1. 속도 명령 (vx, vy): 월드 좌표계 기준 목표 이동 속도 [m/s]
2. 에너지 수준 e ∈ [0, 1]: 1에 가까울수록 힘을 많이 써도 되고,
   0에 가까울수록 제어 비용 페널티가 커져서 힘을 아껴 걸어야 한다.

모션 데이터 없이 보상만으로 학습한다(motion-free).
보상 = 속도 추종 + 생존 - 에너지 수준에 따라 달라지는 제어 비용
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def yaw_from_quat(q: np.ndarray) -> float:
    """MuJoCo 쿼터니언 (w, x, y, z)에서 yaw 각도를 구한다."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class CommandAnt(gym.Wrapper):
    """속도 명령과 에너지 수준을 관측에 붙이고, 보상을 직접 계산하는 래퍼."""

    def __init__(
        self,
        env: gym.Env,
        max_speed: float = 1.5,
        stand_prob: float = 0.1,
        resample_interval: int | None = 250,
        tracking_weight: float = 2.0,
        tracking_sigma: float = 0.25,
        alive_weight: float = 1.0,
        ctrl_weight_min: float = 0.02,
        ctrl_weight_max: float = 0.25,
        limit_weight: float = 0.0,
        limit_soft: float = 0.8,
    ):
        super().__init__(env)
        self.max_speed = max_speed
        self.stand_prob = stand_prob
        self.resample_interval = resample_interval
        self.tracking_weight = tracking_weight
        self.tracking_sigma = tracking_sigma
        self.alive_weight = alive_weight
        self.ctrl_weight_min = ctrl_weight_min
        self.ctrl_weight_max = ctrl_weight_max
        # 관절 한계 페널티 (v6). 관절마다 실제 하한·상한에서 중심과 반범위를 구하고,
        # 중심 ± limit_soft × 반범위를 넘어선 만큼(rad)을 더한다. limit_soft 0.8 = 범위 끝 20% 구간
        # (엉덩이 ±30° → ±24° 밖, 발목 30~70° → 34~66° 밖). limit_weight 0이면 기존 보상과 같다.
        self.limit_weight = limit_weight
        self.limit_soft = limit_soft
        rng = env.unwrapped.model.jnt_range[1:9]          # 관절 8개 (0번은 몸통 free joint)
        center, half = rng.mean(axis=1), (rng[:, 1] - rng[:, 0]) / 2
        self.soft_lo, self.soft_hi = center - limit_soft * half, center + limit_soft * half

        base = env.observation_space
        low = np.concatenate([base.low, [-np.inf, -np.inf, 0.0]]).astype(np.float32)
        high = np.concatenate([base.high, [np.inf, np.inf, 1.0]]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.command = np.zeros(2)  # 월드 좌표계 목표 속도
        self.energy = 0.5
        self.fixed = False  # True면 명령을 랜덤으로 바꾸지 않는다 (평가·데모용)
        self._steps = 0

    # ------------------------------------------------------------------
    # 명령 설정
    # ------------------------------------------------------------------
    def set_command(self, vx: float, vy: float, energy: float | None = None) -> None:
        """외부에서 명령을 고정한다. 이후 랜덤 재샘플링을 하지 않는다."""
        self.command = np.array([vx, vy], dtype=np.float64)
        if energy is not None:
            self.energy = float(np.clip(energy, 0.0, 1.0))
        self.fixed = True

    def _sample_command(self) -> None:
        rng = self.np_random
        if rng.random() < self.stand_prob:
            self.command = np.zeros(2)
        else:
            speed = rng.uniform(0.2, self.max_speed)
            angle = rng.uniform(-math.pi, math.pi)
            self.command = speed * np.array([math.cos(angle), math.sin(angle)])
        self.energy = float(rng.uniform(0.0, 1.0))

    # ------------------------------------------------------------------
    # 관측
    # ------------------------------------------------------------------
    @property
    def _data(self):
        return self.env.unwrapped.data

    def torso_xy(self) -> np.ndarray:
        return self._data.qpos[:2].copy()

    def torso_yaw(self) -> float:
        return yaw_from_quat(self._data.qpos[3:7])

    def _command_in_body_frame(self) -> np.ndarray:
        """월드 좌표계 명령을 몸통 yaw 기준으로 회전시킨다. 정책이 배우기 쉬워진다."""
        yaw = self.torso_yaw()
        c, s = math.cos(-yaw), math.sin(-yaw)
        vx, vy = self.command
        return np.array([c * vx - s * vy, s * vx + c * vy])

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        cmd_body = self._command_in_body_frame()
        return np.concatenate([obs, cmd_body, [self.energy]]).astype(np.float32)

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._steps = 0
        if not self.fixed:
            self._sample_command()
        return self._augment(obs), info

    def step(self, action):
        xy_before = self.torso_xy()
        obs, _, terminated, truncated, info = self.env.step(action)
        xy_after = self.torso_xy()
        dt = self.env.unwrapped.dt
        velocity = (xy_after - xy_before) / dt

        # 1) 속도 추종: 오차가 작을수록 1에 가까움
        err = float(np.sum((velocity - self.command) ** 2))
        tracking = math.exp(-err / self.tracking_sigma)

        # 2) 에너지: 에너지 수준이 낮을수록 페널티 가중치가 커짐
        ctrl_w = self.ctrl_weight_min + (1.0 - self.energy) * (
            self.ctrl_weight_max - self.ctrl_weight_min
        )
        effort = float(np.sum(np.square(action)))

        # 3) 관절 한계: 범위 끝 구간에 들어간 깊이(rad)의 합
        q = self._data.qpos[7:15]
        limit_penalty = float(np.sum(np.clip(q - self.soft_hi, 0.0, None)
                                     + np.clip(self.soft_lo - q, 0.0, None)))

        alive = 0.0 if terminated else 1.0
        reward = (
            self.tracking_weight * tracking
            + self.alive_weight * alive
            - ctrl_w * effort
            - self.limit_weight * limit_penalty
        )

        info.update(
            velocity=velocity,
            command=self.command.copy(),
            energy=self.energy,
            tracking_error=math.sqrt(err),
            effort=effort,
            r_tracking=tracking,
            limit_penalty=limit_penalty,
        )

        self._steps += 1
        if (
            not self.fixed
            and self.resample_interval
            and self._steps % self.resample_interval == 0
        ):
            # 에피소드 중간에 명령을 바꿔서 전환도 배우게 한다
            self._sample_command()

        return self._augment(obs), reward, terminated, truncated, info


def make_command_ant(
    seed: int | None = None,
    render_mode: str | None = None,
    max_episode_steps: int = 1000,
    **wrapper_kwargs,
) -> CommandAnt:
    env = gym.make(
        "Ant-v5",
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )
    env = CommandAnt(env, **wrapper_kwargs)
    if seed is not None:
        env.reset(seed=seed)
    return env
