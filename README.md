# Command-Conditioned Ant Locomotion

MuJoCo Ant에게 **"이 방향으로, 이 속도로, 이 정도 에너지로 걸어"** 라는 명령을 주고,
**하나의 정책**이 모든 명령을 따르도록 PPO로 학습시킨 프로젝트입니다.
모션 데이터 없이 보상만으로 학습합니다.

<!-- 학습 후 여기에 데모 GIF를 넣으세요: runs/<run>/results/path_square.gif -->

## 동기

[FreeMusco](https://arxiv.org/abs/2511.14205)(SIGGRAPH Asia 2025)는 모션 데이터 없이 근골격 캐릭터의
보행을 학습하고, 목표 자세와 에너지 수준을 랜덤화해 다양한 동작을 얻은 뒤, 목표 지점 이동과 경로 추종을
다운스트림 태스크로 보여줍니다. [PhysicsFC](https://arxiv.org/abs/2504.21216)(SIGGRAPH 2025)는 사용자가
실시간으로 조작할 수 있는 물리 기반 캐릭터 컨트롤러를 다룹니다.

이 프로젝트는 그 아이디어들을 **작은 규모로 직접 실험해본 것**입니다. 논문 재현이 아니라,
"명령 조건부 정책 + 에너지 수준 조절 + 상위 명령으로 하위 정책 부리기"라는 구조를 손으로 만들어보며
이해하는 것이 목표였습니다.

## 무엇을 하나

| 구성 | 내용 |
| --- | --- |
| 시뮬레이터 | MuJoCo (Gymnasium `Ant-v5`), 제어 주기 0.05 s |
| 알고리즘 | PPO (Stable-Baselines3), 관측·보상 정규화 |
| 명령 입력 | 목표 속도 (vx, vy) [m/s], 에너지 수준 e ∈ [0, 1] |
| 학습 시 명령 | 에피소드마다 랜덤, 250스텝(12.5 s)마다 재샘플 → 명령 전환도 학습 |
| 데모 | 키보드 실시간 조종, 웨이포인트 경로 추종 |

### 관측

`Ant-v5` 기본 관측(105차원) 뒤에 세 값을 붙입니다.

- 목표 속도를 **몸통 yaw 기준으로 회전**시킨 값 (2차원): 월드 좌표 명령을 그대로 주는 것보다 정책이 배우기 쉽습니다.
- 에너지 수준 e (1차원)

### 보상

```
r = 2.0 · exp(-‖v − v_cmd‖² / 0.25)        # 속도 추종
  + 1.0 · alive                              # 넘어지지 않음
  − w(e) · ‖a‖²                              # 제어 비용
w(e) = 0.02 + (1 − e) · (0.25 − 0.02)        # 에너지 수준이 낮을수록 페널티 큼
```

속도 추종 항은 [Rudin et al. 2021](https://arxiv.org/abs/2109.11978)의 지수형 추종 보상 형태를 따랐습니다.

## 실행

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # GPU 없으면 CPU 버전
pip install -r requirements.txt

# 1) 학습 (8코어 CPU 기준 수 시간 걸릴 수 있음)
python train.py --timesteps 5000000 --n-envs 8 --run-name main
tensorboard --logdir runs        # command/tracking_error_mps 가 내려가는지 확인

# 2) 평가 그래프
python evaluate.py --run runs/main

# 3) 경로 추종 (+ GIF)
python path_follow.py --run runs/main --path square --video

# 4) 키보드 조종 (터미널에서 w/a/s/d, +/-, 1/2/3, x, r, q)
python keyboard_demo.py --run runs/main
```

학습을 이어서 하려면: `python train.py --resume runs/main --run-name main2 --timesteps 3000000`

## 결과

<!-- evaluate.py, path_follow.py 결과로 채우세요 -->

| 지표 | 값 |
| --- | --- |
| 평균 속도 추종 오차 (16개 명령) | ? m/s |
| 넘어진 횟수 | ? / 16 |
| 경로 추종 (square) | ? s |

- `results/tracking.png`: 명령 vs 실제 속도
- `results/energy.png`: 에너지 수준별 제어 노력과 속도
- `results/path_square.png`: 경로 추종 궤적

## 시도한 것 / 실패한 것

<!-- 직접 겪은 걸 적으세요. 교수님이 가장 관심 있게 볼 부분입니다. 예: -->
- 처음엔 생존 보상 0.5, 제어 비용 가중치 최대 0.5로 시작했는데, 랜덤 행동 기준 한 스텝 보상이 음수가 되어
  "빨리 넘어지는 게 이득"인 구조였습니다. 생존 보상을 1.0으로 올리고 제어 비용 상한을 0.25로 낮췄습니다.

## 한계와 다음 단계

- Ant는 사족 보행이라 사람형 캐릭터보다 균형 문제가 쉽습니다. 다음은 Humanoid로 확장.
- 동작 스타일(자연스러움)을 보상하지 않아서 걸음걸이가 부자연스러울 수 있습니다.
  모션 데이터를 쓰는 방식(DeepMimic, AMP)이나 대칭성 손실(Yu et al. 2018)과 비교해볼 수 있습니다.
- 명령 전환이 부드러운지는 따로 측정하지 않았습니다.

## 참고 문헌

- Kim, Jung, Lee. *PhysicsFC: Learning User-Controlled Skills for a Physics-Based Football Player Controller.* SIGGRAPH 2025. [arXiv](https://arxiv.org/abs/2504.21216)
- Kim, Lee. *FreeMusco: Motion-Free Learning of Latent Control for Morphology-Adaptive Locomotion in Musculoskeletal Characters.* SIGGRAPH Asia 2025. [arXiv](https://arxiv.org/abs/2511.14205)
- Rudin et al. *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning.* CoRL 2021. [arXiv](https://arxiv.org/abs/2109.11978)
- Schulman et al. *Proximal Policy Optimization Algorithms.* 2017. [arXiv](https://arxiv.org/abs/1707.06347)
- Yu, Turk, Liu. *Learning Symmetric and Low-Energy Locomotion.* SIGGRAPH 2018. [arXiv](https://arxiv.org/abs/1801.08093)
