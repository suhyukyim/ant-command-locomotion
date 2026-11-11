"""v9 디딤 보상 (docs/V9_DESIGN.md).

물리 스텝마다 발별 '하중 디딤'(F_z ≥ 0.25 mg, v8 측정과 같은 기준) 여부를 받아, 디딤이 끝나는(발이 떨어지는)
순간에 보너스를 준다:

    보너스 = clip(디딤 시간 − min_dur, 0, cap − min_dur) × clip(디딤 동안 평균 명령 방향 속도 / |명령|, 0, 1)

- 0.03 s짜리 튀는 접촉은 min_dur(0.05 s) 미만이라 0.
- 서 있기만 하면 발을 떼지 않으니 0.
- 명령 방향으로 나아가지 않으면(제자리걸음, 뒤로, 서기 명령) 0.
미끄러짐: 하중이 실린 발의 접촉점 수평 속도² 합 (물리 스텝 값; 환경이 제어 스텝 평균을 낸다).
"""

from __future__ import annotations

import numpy as np


class StanceTracker:
    def __init__(self, dt: float, min_dur: float = 0.05, cap: float = 0.25, n_feet: int = 4):
        self.dt, self.min_dur, self.cap, self.n = dt, min_dur, cap, n_feet
        self.reset()

    def reset(self) -> None:
        self.steps = np.zeros(self.n, int)
        self.along = np.zeros(self.n)

    def update(self, loaded: np.ndarray, slip: np.ndarray, along: float, cmd_norm: float) -> tuple[float, float]:
        """물리 스텝 하나. (이번 스텝에 끝난 디딤의 보너스 합, 하중 발 미끄러짐 속도² 합)."""
        bonus = 0.0
        for k in range(self.n):
            if loaded[k]:
                self.steps[k] += 1
                self.along[k] += along
            elif self.steps[k]:
                dur = self.steps[k] * self.dt
                gate = 0.0 if cmd_norm < 1e-6 else float(np.clip(self.along[k] / self.steps[k] / cmd_norm, 0.0, 1.0))
                bonus += float(np.clip(dur - self.min_dur, 0.0, self.cap - self.min_dur)) * gate
                self.steps[k], self.along[k] = 0, 0.0
        s = np.where(loaded, np.nan_to_num(slip), 0.0)
        return bonus, float(np.sum(s ** 2))
