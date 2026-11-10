"""PD 행동 방식 (관절 목표 각도 + 물리 스텝마다 PD) 시험."""

import math

import numpy as np
import pytest

from envs.command_ant import make_command_ant

V6B = dict(tracking_sigma=1.0, tracking_weight=5.0, limit_weight=1.5, limit_soft=0.8)


def _pd_env(kp=60.0, kd=15.0, **kw):
    return make_command_ant(action_mode="pd", pd_kp=kp, pd_kd=kd, **kw)


def _hand_pd(env, action, kp, kd):
    """손 계산: 행동 순서의 PD 토크 (잘리기 전)."""
    m, d = env.unwrapped.model, env.unwrapped.data
    j = m.actuator_trnid[:, 0] - 1
    rng = m.jnt_range[1:9]
    center, half = rng.mean(axis=1), (rng[:, 1] - rng[:, 0]) / 2
    target = center[j] + np.asarray(action, float) * half[j]
    return kp * (target - d.qpos[7:15][j]) - kd * d.qvel[6:14][j], target


def test_pd_kp_must_be_positive():
    with pytest.raises(ValueError):
        make_command_ant(action_mode="pd", pd_kp=0.0, pd_kd=1.0)


def test_pd_torque_matches_hand_calc():
    env = _pd_env()
    env.reset(seed=1)
    a = np.array([0.3, -0.2, 0.1, 0.5, -0.4, 0.0, 0.2, -0.1])
    seen = []

    def cb():
        tau, _ = _hand_pd(env, a, 60.0, 15.0)
        seen.append((env.unwrapped.data.ctrl * 150.0, np.clip(tau, -150, 150)))
    env.substep_callbacks.append(cb)
    env.step(a)
    applied, expected = seen[0]
    assert np.max(np.abs(applied - expected)) < 1e-9


def test_pd_target_held_torque_recomputed():
    env = _pd_env()
    env.reset(seed=2)
    a = np.array([0.8, -0.6, 0.4, 0.9, -0.7, 0.2, -0.5, 0.6])
    targets, ctrls = [], []

    def cb():
        tau, target = _hand_pd(env, a, 60.0, 15.0)
        targets.append(env._pd_target.copy())
        assert np.max(np.abs(env.unwrapped.data.ctrl * 150 - np.clip(tau, -150, 150))) < 1e-9
        assert np.max(np.abs(env._pd_target - target)) < 1e-12
        ctrls.append(env.unwrapped.data.ctrl.copy())
    env.substep_callbacks.append(cb)
    env.step(a)
    assert len(targets) == 5
    assert all(np.array_equal(t, targets[0]) for t in targets)
    assert not all(np.array_equal(c, ctrls[0]) for c in ctrls)   # 토크는 물리 스텝마다 다시 계산된다


def test_pd_target_at_extremes():
    env = _pd_env(kp=500.0, kd=1.0)
    env.reset(seed=0)
    m = env.unwrapped.model
    j = m.actuator_trnid[:, 0] - 1
    for sign, col in ((1.0, 1), (-1.0, 0)):
        _, target = _hand_pd(env, np.full(8, sign), 500.0, 1.0)
        assert np.allclose(target, m.jnt_range[1:9, col][j])
    _, _, _, _, info = env.step(np.ones(8))
    assert np.all(np.abs(env.unwrapped.data.ctrl) <= 1.0)
    assert info["torque_limit_frac"] > 0


def test_info_requested_vs_applied():
    env = _pd_env(kp=500.0, kd=1.0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.ones(8))
    req, app = info["torque_requested"], info["torque"]
    assert req.shape == app.shape == (8,)
    assert np.max(np.abs(req)) > 150
    assert np.max(np.abs(app)) <= 150 + 1e-9


def test_torque_requested_equals_applied_in_torque_mode():
    env = make_command_ant()
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.tanh(np.linspace(-1, 1, 8)))
    assert np.max(np.abs(info["torque_requested"] - info["torque"])) < 1e-9


def test_pd_single_action_moves_its_joint():
    m0 = make_command_ant().unwrapped.model
    trn = m0.actuator_trnid[:, 0] - 1
    for i in range(8):
        env = _pd_env(**V6B)
        env.set_command(0, 0, 0.5)
        env.reset(seed=0)
        for _ in range(20):
            env.step(np.zeros(8))
        q0 = env.unwrapped.data.qpos[7:15].copy()
        a = np.zeros(8)
        a[i] = 0.5
        for _ in range(50):
            env.step(a)
        dq = env.unwrapped.data.qpos[7:15] - q0
        j = trn[i]
        assert dq[j] > 0, (i, dq)
        assert abs(dq[j]) > np.max(np.abs(np.delete(dq, j))), (i, dq)


@pytest.mark.parametrize("cmd_idx", [0, 8])
def test_pd_stand_reproduces_baseline(cmd_idx):
    s = 0.5 if cmd_idx < 8 else 1.0
    ang = np.linspace(-math.pi, math.pi, 8, endpoint=False)[cmd_idx % 8]
    env = _pd_env(kp=50.0, kd=10.0, max_episode_steps=1000, **V6B)
    env.set_command(s * math.cos(ang), s * math.sin(ang), 0.5)
    env.reset(seed=0)
    torques, n = [], 0
    for _ in range(1000):
        _, _, term, trunc, info = env.step(np.zeros(8))
        torques.append(np.mean(np.abs(info["torque"])))
        n += 1
        if term or trunc:
            break
    assert n == 1000 and not term
    assert np.mean(torques) < 1.0
