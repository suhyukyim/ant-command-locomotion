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

지금까지 두 번 학습했습니다. v1은 학습이 무너졌고, v2는 안정적으로 학습됐지만 **넘어지지 않고 버티기만** 배웠습니다.
명령대로 걷는 것은 아직 성공하지 못했습니다.

### 학습 과정 비교 (텐서보드, 각 500만 스텝)

|                                       | v1                    | v2                    |
| ------------------------------------- | --------------------- | --------------------- |
| 학습 시간                             | 82분                  | 59분                  |
| 버틴 길이 `ep_len_mean` (최대 1000) | 371                   | **885**               |
| 속도 오차 `tracking_error_mps`      | 0.95                  | 0.84                  |
| `approx_kl` (정상 약 0.01–0.03)    | **4.11 (폭발)**       | 0.000001              |
| 행동 표준편차 `std`                 | 0.144 (붕괴)          | 0.703                 |

### v2 평가 (`python evaluate.py --run runs/main`)

`runs/main`이 v2 모델입니다. 속도 2개(0.5, 1.0 m/s) × 8방향 = 16개 명령을 주고, 명령마다 300스텝씩 굴려 평가했습니다.

| 지표 | 값 |
| --- | --- |
| 평균 속도 추종 오차 (16개 명령) | **0.747 m/s** |
| 넘어진 횟수 | 0 / 16 |
| 실제 평균 속도 | 16개 명령 모두 거의 0 m/s |

- **가만히 서 있을 때의 오차는 (0.5 × 8 + 1.0 × 8) / 16 = 0.75 m/s**입니다. 측정값 0.747이 이 값과 거의 같으므로,
  v2는 어떤 명령을 받아도 **제자리에 서 있기만** 한다는 뜻입니다.
- 에너지 수준 e를 0 → 1로 올리면 제어 노력(Σ action²)은 2.37 → 5.99로 커졌습니다. 다이얼에 반응은 하지만,
  걷는 데가 아니라 버티는 데 힘을 쓰고 있습니다(속도는 모든 e에서 ≈ 0).

결과 파일 (`runs/main/results/`):
- `tracking.png`: 명령 vs 실제 속도
- `energy.png`: 에너지 수준별 제어 노력과 속도
- `path_square.png`, `path_square.gif`: 정사각형 경로 추종 궤적
- `summary.json`: 위 수치 원본

## 시도한 것 / 실패한 것

### v0 → v1: 보상 부호 문제

- 처음엔 생존 보상 0.5, 제어 비용 가중치 최대 0.5로 시작했는데, 랜덤 행동 기준 한 스텝 보상이 음수가 되어
  "빨리 넘어지는 게 이득"인 구조였습니다. 생존 보상을 1.0으로 올리고 제어 비용 상한을 0.25로 낮췄습니다.

### v1: PPO 업데이트가 너무 커서 학습이 무너짐

| 설정 | 값 |
| --- | --- |
| 미니배치 | batch 256 → 업데이트당 **48개** |
| 에폭 | **10** |
| 업데이트 한 번에 가중치를 고친 횟수 | 48 × 10 = **480번** |
| 학습률 | 3e-4 고정 |

- **결과:** `approx_kl` 4.11, `clip_fraction` 0.87, 행동 표준편차가 0.144까지 줄며 붕괴. 버틴 길이 평균 371스텝.
- **원인:** PPO의 clip(확률비 0.8~1.2)은 데이터 하나하나가 정책을 미는 힘만 0으로 만듭니다. 그런데 신경망
  가중치는 모든 데이터가 함께 쓰기 때문에, clip에 걸린 데이터도 다른 데이터가 고치는 동안 계속 밀려납니다.
  480번 반복하면서 이 밀림이 쌓여 정책이 데이터를 모은 정책에서 멀리 떨어졌고, 모아 둔 데이터가 새 정책에 맞지
  않게 되어 학습이 무너졌습니다. 행동 표준편차가 작아지면 같은 크기의 업데이트에도 KL이 더 커지므로, 붕괴가
  스스로 가속됐습니다.

### v2: 안정화는 성공, 하지만 버티기만 배움

| 설정 | v1 → v2 | 이유 |
| --- | --- | --- |
| 미니배치 × 에폭 | 48 × 10 → **4 × 5 = 20번** | 밀림이 쌓일 기회를 줄임 (Rudin et al. 2021 설정 참고) |
| `target_kl` | 없음 → **0.02** | 정책 전체의 변화량(KL)이 커지면 남은 에폭을 건너뛰고 새 데이터를 모음 |
| 학습률 | 3e-4 고정 → **3e-4 → 0 선형 감소** | 후반 안정 |

- **결과:** KL 폭발이 사라지고 버틴 길이가 371 → 885로 늘었습니다. 하지만 평가해 보니 속도 오차 0.747로,
  서 있을 때(0.75)와 같았습니다.
- **원인 분석:** 속도 추종 보상 `exp(−오차² / 0.25)`는 sigma가 작아서, 서 있다가 0.1 m/s만큼 움직여도
  한 스텝 보상이 +0.04밖에 안 늘어납니다. 반면 움직이다 넘어지면 생존 보상(스텝당 1점)을 잃습니다. 그래서
  "서 있기" 주변의 보상 지형이 평지가 되어 국소 최적에 갇혔습니다. 여기에 학습률을 0까지 줄인 탓에, 학습 후반엔
  평지를 빠져나올 보폭마저 사라졌습니다.

  | 명령 1 m/s, sigma 0.25 | 오차² | tracking | 한 스텝 점수 |
  | --- | --- | --- | --- |
  | 가만히 서 있기 | 1 | 0.018 | ≈ 1.04 |
  | 0.1 m/s로 움직이기 | 0.81 | 0.039 | ≈ 1.08 |
  | 완벽하게 1 m/s | 0 | 1 | 3.0 |

## 한계와 다음 단계

### 다음 실험 (v3 후보, 하나씩 바꿔서 비교)

- **sigma 키우기:** 0.25 → 1.0. 서 있기 → 0.1 m/s 이득이 0.04 → 0.15로 약 4배가 되어 첫걸음 신호가 생깁니다.
  대신 후반에 대충 걷는 데 만족할 수 있습니다 (1 m/s: 3.0 vs 0.8 m/s: 2.92).
- **두 항 더하기:** `exp(−오차²/1.0) + exp(−오차²/0.25)`. 멀리선 넓은 항, 가까이선 좁은 항이 기울기를 줍니다.
  두 항 모두 오차가 줄면 커지므로 보상이 단조롭습니다.
- **커리큘럼:** sigma를 처음엔 크게, 평균 오차가 기준(예: 0.3 m/s) 아래로 내려가면 작게.
- **학습률:** 0까지 줄이지 않고 하한을 두거나 고정해서, 국소 최적에서 빠져나올 여지를 남깁니다.
- 기타 후보: 최근 1초 평균 속도로 추종 점수 계산(FreeMusco, Yu et al. 2018), 생존 보상 비중 낮추기.

### 한계

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
