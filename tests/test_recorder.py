"""접촉 기록기(Recorder)가 substep_callbacks로 붙어도 궤적을 바꾸지 않고, 수정 전 기록기와 같은 값을 내는지."""

import numpy as np

from envs import make_command_ant
from propulsion_analysis import Recorder
from tests._legacy_recorder import Recorder as LegacyRecorder


def _rollout(env, rec_cls, steps, rec_on=True, policy=None, seed=0):
    rec = rec_cls(env, np.array([1.0, 0.0])) if rec_cls else None
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(7)
    for t in range(steps):
        if rec:
            rec.on = rec_on and t >= 10
        a = policy.act(obs, deterministic=True) if policy else np.tanh(rng.normal(size=8))
        obs, *_ = env.step(a)
    return env.unwrapped.data.qpos.copy(), (rec.rows if rec else None)


def test_recorder_uses_callbacks():
    env = make_command_ant()
    Recorder(env, np.array([1.0, 0.0]))
    assert len(env.substep_callbacks) == 1


def test_recorder_does_not_change_pd_trajectory():
    kw = dict(action_mode="pd", pd_kp=60.0, pd_kd=15.0)
    q_plain, _ = _rollout(make_command_ant(**kw), None, 200)
    q_rec, rows = _rollout(make_command_ant(**kw), Recorder, 200)
    assert np.array_equal(q_plain, q_rec)
    assert len(rows) == (200 - 10) * 5


def test_recorder_torque_matches_legacy():
    """v6Bb seed 0, 명령 (1, 0), 노이즈 끔: 새 기록기와 수정 전 기록기의 기록이 완전히 같다."""
    from policy_utils import Policy
    pol = Policy("runs/main_v6Bb")
    env_new = make_command_ant(**{**pol.env_kwargs, "max_episode_steps": 400})
    env_old = make_command_ant(**{**pol.env_kwargs, "max_episode_steps": 400})
    for env in (env_new, env_old):
        env.set_command(1.0, 0.0, 0.5)
    q_new, rows_new = _rollout(env_new, Recorder, 120, policy=pol)
    q_old, rows_old = _rollout(env_old, LegacyRecorder, 120, policy=pol)
    assert np.array_equal(q_new, q_old)
    assert len(rows_new) == len(rows_old) > 0
    for a, b in zip(rows_new, rows_old):
        for x, y in zip(a, b):
            assert np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)
