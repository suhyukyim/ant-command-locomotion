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
import mujoco
import numpy as np
from gymnasium import spaces

# 토크 기준값 [N·m]. 모터 gear와 같다. 노력 E = Σ(τ / TAU_REF)² 의 분모이고 모든 갈래가 같은 값을 쓴다.
TAU_REF = 150.0


# 추종 커널 (v8). "single" = exp(−오차²/σ) (기존 K0), "two_term" = 넓은 항과 좁은 항(σ 0.25)을 반씩 (K1).
# 둘 다 오차 0에서 1이라, 가중치를 곱한 최댓값이 같다.
KERNELS = ("single", "two_term")
NARROW_SIGMA = 0.25


def tracking_reward(err2: float, weight: float, sigma: float, kernel: str) -> float:
    """속도 추종 보상. err2 = ‖v − v_명령‖² [m²/s²]."""
    if kernel == "single":
        return weight * math.exp(-err2 / sigma)
    if kernel == "two_term":
        return weight * (0.5 * math.exp(-err2 / sigma) + 0.5 * math.exp(-err2 / NARROW_SIGMA))
    raise ValueError(f"tracking_kernel={kernel!r}: {KERNELS} 중 하나")


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
        vz_weight: float = 0.0,
        action_mode: str = "torque",
        pd_kp: float = 0.0,
        pd_kd: float = 0.0,
        tracking_kernel: str = "single",
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
        # 몸통 수직 속도 페널티 (v7): vz_weight × v_z². v_z = qvel[2] = 몸통 free joint의 월드 좌표계
        # 선속도 z 성분 (기울어져도 월드 기준 위아래). vz_weight 0이면 기존 보상과 같다.
        self.vz_weight = vz_weight

        # 행동 방식 (v8): "torque" = 행동 × 150 N·m (기존), "pd" = 행동 = 관절 목표 각도, 물리 스텝마다 PD로 토크 계산
        if action_mode not in ("torque", "pd"):
            raise ValueError(f"action_mode={action_mode!r}: 'torque' 또는 'pd'")
        if action_mode == "pd" and pd_kp <= 0:
            raise ValueError(f"PD 행동 방식에는 pd_kp > 0이 필요하다 (pd_kp={pd_kp})")
        if tracking_kernel not in KERNELS:
            raise ValueError(f"tracking_kernel={tracking_kernel!r}: {KERNELS} 중 하나")
        self.action_mode = action_mode
        self.pd_kp = pd_kp
        self.pd_kd = pd_kd
        self.tracking_kernel = tracking_kernel

        # 물리 스텝 루프 (v8): Gymnasium의 _step_mujoco_simulation을 이 클래스의 루프로 바꿔 끼운다.
        # 물리 스텝마다 ctrl을 정한 뒤 mj_step 직전에 substep_callbacks를 순서대로 부른다
        # (기록기 등은 여기에 콜백을 등록한다).
        model = env.unwrapped.model
        # 행동 순서와 관절 순서는 다르다: 행동 i는 관절 actuator_trnid[i, 0] - 1을 움직인다.
        self._act_to_joint = model.actuator_trnid[:, 0].astype(int) - 1
        self._gear = model.actuator_gear[:, 0].copy()
        assert np.allclose(self._gear, TAU_REF), "모터 gear가 TAU_REF와 다르다"
        # PD: 행동 i의 목표 = 그 관절 범위 중심 + a_i × 반범위 (a = ±1 이면 관절 한계). 행동 순서로 저장
        jr = model.jnt_range[1:9]
        self._pd_center = jr.mean(axis=1)[self._act_to_joint]
        self._pd_half = ((jr[:, 1] - jr[:, 0]) / 2)[self._act_to_joint]
        self._pd_target = None
        self.substep_callbacks: list = []
        self._step_info: dict = {}
        env.unwrapped._step_mujoco_simulation = self._step_mujoco_simulation

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
    # 물리 스텝 루프
    # ------------------------------------------------------------------
    def _compute_ctrl(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """물리 스텝마다 부른다. (ctrl, 잘리기 전 요청 토크 [N·m]), 둘 다 행동 순서.
        토크 갈래는 행동 그대로 (기존과 비트 단위로 같게). PD 갈래는 제어 스텝 동안 고정된 목표로
        그 순간의 q, q̇에서 토크를 다시 계산하고 ±gear에서 자른다."""
        if self.action_mode == "torque":
            return action, action * self._gear
        d = self._data
        q = d.qpos[7:15][self._act_to_joint]
        qd = d.qvel[6:14][self._act_to_joint]
        req = self.pd_kp * (self._pd_target - q) - self.pd_kd * qd
        return np.clip(req, -self._gear, self._gear) / self._gear, req

    def _step_mujoco_simulation(self, ctrl, n_frames):
        """Gymnasium MujocoEnv와 같은 결과를 내되, 물리 스텝마다 적용 토크·일률을 모은다."""
        u = self.env.unwrapped
        model, data = u.model, u.data
        action = np.asarray(ctrl, dtype=np.float64)
        n = len(self._act_to_joint)
        torque_sum = np.zeros(n)        # 관절 순서
        req_sum = np.zeros(n)
        effort_sum = power_sum = 0.0
        n_limit = 0
        if self.action_mode == "pd":   # 목표 각도는 제어 스텝 시작 때 한 번 정하고 물리 스텝 5개 동안 유지
            self._pd_target = self._pd_center + np.clip(action, -1.0, 1.0) * self._pd_half
        for _ in range(n_frames):
            ctrl_now, req = self._compute_ctrl(action)
            data.ctrl[:] = ctrl_now
            req_sum[self._act_to_joint] += req
            for cb in self.substep_callbacks:
                cb()
            mujoco.mj_step(model, data)
            tau = np.empty(n)
            # actuator_force는 gear 곱하기 전 값(모터 게인 1)이라 관절에 걸리는 토크는 gear를 곱한 값이다
            tau[self._act_to_joint] = data.actuator_force * self._gear   # 행동 순서 -> 관절 순서
            torque_sum += tau
            effort_sum += float(np.sum((tau / TAU_REF) ** 2))
            power_sum += float(np.sum(np.abs(tau * data.qvel[6:6 + n])))
            n_limit += int(np.sum(np.abs(tau) >= 0.99 * TAU_REF))
        # Gymnasium과 같다: 접촉력 등 센서 값을 위해 마지막에 한 번
        mujoco.mj_rnePostConstraint(model, data)
        self._step_info = dict(
            torque=torque_sum / n_frames,
            torque_requested=req_sum / n_frames,
            effort=effort_sum / n_frames,
            power=power_sum / n_frames,
            torque_limit_frac=n_limit / (n * n_frames),
        )

    def _fallback_step_info(self) -> dict:
        """다른 코드가 _step_mujoco_simulation을 다시 바꿔 끼운 경우(예: 옛 Recorder)의 대비책.
        마지막 물리 스텝의 값 하나로 채운다 (토크 갈래에서는 제어 스텝 동안 일정하므로 같은 값)."""
        data = self._data
        n = len(self._act_to_joint)
        tau = np.empty(n)
        tau[self._act_to_joint] = data.actuator_force * self._gear
        return dict(
            torque=tau,
            torque_requested=tau,
            effort=float(np.sum((tau / TAU_REF) ** 2)),
            power=float(np.sum(np.abs(tau * data.qvel[6:6 + n]))),
            torque_limit_frac=float(np.mean(np.abs(tau) >= 0.99 * TAU_REF)),
        )

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
        self._step_info = {}
        obs, _, terminated, truncated, info = self.env.step(action)
        xy_after = self.torso_xy()
        dt = self.env.unwrapped.dt
        velocity = (xy_after - xy_before) / dt

        # 1) 속도 추종: 오차가 작을수록 1에 가까움
        err = float(np.sum((velocity - self.command) ** 2))
        tracking = tracking_reward(err, 1.0, self.tracking_sigma, self.tracking_kernel)   # 0~1

        # 2) 에너지: 에너지 수준이 낮을수록 페널티 가중치가 커짐
        ctrl_w = self.ctrl_weight_min + (1.0 - self.energy) * (
            self.ctrl_weight_max - self.ctrl_weight_min
        )
        si = self._step_info or self._fallback_step_info()
        effort = si["effort"]   # 적용 토크 기준 Σ(τ/150)² (물리 스텝 평균)

        # 3) 관절 한계: 범위 끝 구간에 들어간 깊이(rad)의 합
        q = self._data.qpos[7:15]
        limit_penalty = float(np.sum(np.clip(q - self.soft_hi, 0.0, None)
                                     + np.clip(self.soft_lo - q, 0.0, None)))

        # 4) 몸통 수직 속도
        vz2 = float(self._data.qvel[2] ** 2)

        alive = 0.0 if terminated else 1.0
        reward = (
            self.tracking_weight * tracking
            + self.alive_weight * alive
            - ctrl_w * effort
            - self.limit_weight * limit_penalty
            - self.vz_weight * vz2
        )

        info.update(
            velocity=velocity,
            command=self.command.copy(),
            energy=self.energy,
            tracking_error=math.sqrt(err),
            effort=effort,
            r_tracking=tracking,
            limit_penalty=limit_penalty,
            vz2=vz2,
            torque=si["torque"],
            torque_requested=si["torque_requested"],
            power=si["power"],
            torque_limit_frac=si["torque_limit_frac"],
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
