"""행동을 tanh로 (-1, 1) 안에 묶는 PPO 정책 (tanh-squashed Gaussian).

기본 MlpPolicy는 a ~ N(μ, σ²)를 뽑은 뒤 환경에 넘길 때 clip(a, -1, 1)로 자른다.
μ가 한계 밖(예: 2.9)으로 나가면 노이즈를 꺼도 켜도 대부분 1로 잘려서
  - 노이즈 끔: 매 스텝 같은 값 → 몸을 굳힌 자세
  - 노이즈 켬: 잘림 경계 근처에서만 무작위로 떨림 → 떨어서 이동
이 되는 현상이 v2~v4에서 관찰됐다.

이 정책은 u ~ N(μ, σ²), a = tanh(u)로 행동을 만든다.
  - 행동이 항상 (-1, 1) 안이라 잘림이 없다
  - 노이즈 끔 행동 tanh(μ)도 범위 안에서 연속적으로 표현된다
  - 로그 확률은 변수 변환을 반영한다: log π(a) = log N(u; μ, σ) − Σ log(1 − tanh(u)²)

SB3의 SquashedDiagGaussianDistribution(SAC가 쓰는 분포)을 ActorCriticPolicy에 끼운다.
SB3의 squash_output=True는 gSDE(다른 탐색 방식)와만 쓸 수 있어서 쓰지 않는다.
"""

from __future__ import annotations

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim


class SquashedGaussianPolicy(ActorCriticPolicy):
    def _build(self, lr_schedule) -> None:
        # 분포만 tanh 버전으로 바꾸고, 나머지(신경망 구조·초기화·옵티마이저)는 기본 정책과 같다.
        # SquashedDiagGaussianDistribution은 DiagGaussianDistribution의 하위 클래스라서
        # ActorCriticPolicy의 평균·log_std 처리 경로를 그대로 탄다.
        self.action_dist = SquashedDiagGaussianDistribution(get_action_dim(self.action_space))
        super()._build(lr_schedule)
