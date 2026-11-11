"""v9 디딤 보상 (envs/stance.py) 시험."""

import numpy as np
import pytest

from envs.stance import StanceTracker

DT = 0.01


def run(tr, pattern, along=1.0, cmd=1.0, slip=0.0):
    """pattern: 발 0의 하중 여부 시퀀스. 나머지 발은 늘 공중. 보너스 합을 돌려준다."""
    total = 0.0
    for on in pattern:
        loaded = np.array([on, False, False, False])
        b, _ = tr.update(loaded, np.full(4, slip), along, cmd)
        total += b
    return total


def test_long_stance_paid_on_liftoff():
    tr = StanceTracker(DT, min_dur=0.05, cap=0.25)
    assert run(tr, [True] * 10) == 0.0                       # 아직 안 뗐다
    assert run(tr, [False]) == pytest.approx(0.10 - 0.05)    # 10스텝 = 0.10 s


def test_short_contact_pays_nothing():
    tr = StanceTracker(DT, min_dur=0.05, cap=0.25)
    assert run(tr, [True, True, True, False] * 5) == 0.0     # 0.03 s 튀기


def test_cap():
    tr = StanceTracker(DT, min_dur=0.05, cap=0.25)
    assert run(tr, [True] * 100 + [False]) == pytest.approx(0.25 - 0.05)


def test_gate_by_progress_along_command():
    tr = StanceTracker(DT, min_dur=0.05, cap=0.25)
    assert run(tr, [True] * 10 + [False], along=0.5, cmd=1.0) == pytest.approx(0.05 * 0.5)
    assert run(tr, [True] * 10 + [False], along=-0.3, cmd=1.0) == 0.0      # 뒤로 가면 0
    assert run(tr, [True] * 10 + [False], along=0.0, cmd=0.0) == 0.0       # 서기 명령이면 0
    assert run(tr, [True] * 10 + [False], along=2.0, cmd=1.0) == pytest.approx(0.05)   # 1에서 자름


def test_slip_is_mean_square_over_loaded_feet():
    tr = StanceTracker(DT, min_dur=0.05, cap=0.25)
    _, s = tr.update(np.array([True, True, False, False]), np.array([0.2, 0.4, 9.0, np.nan]), 1.0, 1.0)
    assert s == pytest.approx(0.2 ** 2 + 0.4 ** 2)


def test_reset_drops_open_stance():
    tr = StanceTracker(DT, min_dur=0.05, cap=0.25)
    run(tr, [True] * 10)
    tr.reset()
    assert run(tr, [False]) == 0.0


def _roll(steps=200, **kw):
    from envs import make_command_ant
    env = make_command_ant(action_mode="pd", pd_kp=120.0, pd_kd=21.0, max_episode_steps=steps + 5, **kw)
    env.set_command(1.0, 0.0, 0.5)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    out = []
    for _ in range(steps):
        _, r, term, _, info = env.step(rng.uniform(-1, 1, 8).astype(np.float32))
        out.append((r, env.unwrapped.data.qpos.copy(), info))
        if term:
            break
    return out


def test_zero_weights_change_nothing():
    """가중치 0이면 v8과 같은 궤적·보상 (디딤 계산을 아예 안 함)."""
    a, b = _roll(), _roll(stance_weight=0.0, slip_weight=0.0)
    assert all(ra == rb and np.array_equal(qa, qb) for (ra, qa, _), (rb, qb, _) in zip(a, b))


def test_stance_terms_do_not_change_trajectory():
    """디딤 계산(mj_forward)은 상태를 바꾸지 않는다: 궤적은 같고 보상만 다르다."""
    a, b = _roll(), _roll(stance_weight=20.0, slip_weight=1.0)
    assert all(np.array_equal(qa, qb) for (_, qa, _), (_, qb, _) in zip(a, b))
    for (ra, _, _), (rb, _, ib) in zip(a, b):
        assert rb == pytest.approx(ra + 20.0 * ib["stance_bonus"] - 1.0 * ib["slip_sq"])
    assert sum(i["slip_sq"] for _, _, i in b) > 0
