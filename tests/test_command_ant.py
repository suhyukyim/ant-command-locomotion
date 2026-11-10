"""CommandAnt 물리 스텝 루프와 적용 토크 시험."""

import numpy as np
import pytest

from envs.command_ant import make_command_ant
from tests._legacy_command_ant import make_command_ant as make_legacy

REWARD_KW = dict(tracking_sigma=1.0, tracking_weight=5.0, limit_weight=1.5, limit_soft=0.8)


def _compare(fixed: bool, steps: int = 2000, seed: int = 3):
    new, old = make_command_ant(**REWARD_KW), make_legacy(**REWARD_KW)
    if fixed:
        new.set_command(0.7, -0.4, 0.3)
        old.set_command(0.7, -0.4, 0.3)
    obs_n, _ = new.reset(seed=seed)
    obs_o, _ = old.reset(seed=seed)
    assert np.array_equal(obs_n, obs_o)
    rng = np.random.default_rng(0)
    max_dq, max_dr, n = 0.0, 0.0, 0
    for _ in range(steps):
        a = np.tanh(rng.normal(size=8) * 1.5)
        obs_n, r_n, te_n, tr_n, info_n = new.step(a)
        obs_o, r_o, te_o, tr_o, info_o = old.step(a)
        max_dq = max(max_dq, float(np.max(np.abs(new.unwrapped.data.qpos - old.unwrapped.data.qpos))))
        max_dr = max(max_dr, abs(r_n - r_o))
        assert np.array_equal(obs_n, obs_o)
        assert (te_n, tr_n) == (te_o, tr_o)
        for k in ("velocity", "tracking_error", "r_tracking", "limit_penalty", "vz2"):
            assert np.all(info_n[k] == info_o[k]), k
        n += 1
        if te_n or tr_n:
            new.reset()
            old.reset()
    assert n == steps
    assert max_dq == 0
    assert max_dr < 1e-12


@pytest.mark.parametrize("fixed", [True, False])
def test_torque_mode_matches_original(fixed):
    _compare(fixed)


def test_info_torque_is_joint_order():
    env = make_command_ant()
    env.reset(seed=0)
    a = np.zeros(8)
    a[0] = 0.2   # 행동 0 = hip_4 (관절 순서 인덱스 6)
    _, _, _, _, info = env.step(a)
    expect = np.zeros(8)
    expect[6] = 30.0
    assert np.max(np.abs(info["torque"] - expect)) < 1e-9


def test_effort_equals_sum_a2_in_torque_mode():
    env = make_command_ant()
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(50):
        a = np.tanh(rng.normal(size=8))
        _, _, te, _, info = env.step(a)
        assert abs(info["effort"] - float(np.sum(a ** 2))) < 1e-12
        if te:
            env.reset()


def test_power_and_limit_frac_present():
    env = make_command_ant()
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.ones(8))   # ctrlrange 끝 = 150 N·m
    assert info["torque_limit_frac"] == 1.0
    assert info["power"] >= 0.0


def test_substep_callbacks_called_five_times():
    env = make_command_ant()
    env.reset(seed=0)
    count = [0]
    env.substep_callbacks.append(lambda: count.__setitem__(0, count[0] + 1))
    env.step(np.zeros(8))
    assert count[0] == 5


def test_default_options():
    env = make_command_ant()
    assert (env.action_mode, env.pd_kp, env.pd_kd, env.tracking_kernel) == ("torque", 0.0, 0.0, "single")
    with pytest.raises(ValueError):
        make_command_ant(action_mode="velocity")


def test_step_survives_replaced_simulation_loop():
    """옛 Recorder처럼 _step_mujoco_simulation을 다시 바꿔 끼워도 step()이 동작한다."""
    import mujoco
    env = make_command_ant()
    env.reset(seed=0)
    uw = env.unwrapped

    def sim(ctrl, n_frames):
        uw.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            mujoco.mj_step(uw.model, uw.data)
        mujoco.mj_rnePostConstraint(uw.model, uw.data)

    uw._step_mujoco_simulation = sim
    a = np.full(8, 0.3)
    _, _, _, _, info = env.step(a)
    assert abs(info["effort"] - float(np.sum(a ** 2))) < 1e-12
