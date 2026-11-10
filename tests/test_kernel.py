"""추종 커널 K0(한 항), K1(두 항) 시험. 기대값은 docs/V8_DESIGN.md 7절 표."""

import numpy as np
import pytest

from envs.command_ant import make_command_ant, tracking_reward

STAND = {  # 서 있을 때(오차 = 명령 속도)의 추종 보상, weight 5, sigma 1.0
    "single": {0.2: 4.80, 0.5: 3.89, 1.0: 1.84, 1.5: 0.53},
    "two_term": {0.2: 4.53, 0.5: 2.87, 1.0: 0.97, 1.5: 0.26},
}


@pytest.mark.parametrize("kernel", ["single", "two_term"])
def test_kernel_table(kernel):
    for speed, want in STAND[kernel].items():
        assert round(tracking_reward(speed ** 2, 5.0, 1.0, kernel), 2) == want
    assert tracking_reward(0.0, 5.0, 1.0, kernel) == pytest.approx(5.0)


def test_two_term_monotone():
    e = np.linspace(0, 3, 30001)
    r = np.array([tracking_reward(x * x, 5.0, 1.0, "two_term") for x in e])
    assert np.all(np.diff(r) < 0)


def test_unknown_kernel_rejected():
    with pytest.raises(ValueError):
        make_command_ant(tracking_kernel="three_term")


def test_env_uses_two_term_kernel():
    env = make_command_ant(tracking_kernel="two_term", tracking_sigma=1.0, tracking_weight=5.0)
    env.set_command(0.5, 0.0, 1.0)
    env.reset(seed=0)
    _, r, _, _, info = env.step(np.zeros(8))
    err2 = info["tracking_error"] ** 2
    assert info["r_tracking"] == pytest.approx(tracking_reward(err2, 1.0, 1.0, "two_term"))
