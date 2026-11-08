# Ant 프로젝트: 왜 하나, 지금 어디까지 왔나

2026-11-01 기준

## 이 프로젝트를 하는 이유

이윤상 교수님(한양대 CGR Lab)께 겨울 학부인턴 컨택 메일을 보낼 때 "이런 걸 해봤습니다"라고 보여줄 거리를 만드는 게 목적이다.

- **랩이 하는 일을 직접 해봤다는 증거.** CGR Lab의 핵심은 물리 시뮬레이션과 강화학습으로 캐릭터·로봇을 움직이는 것이다. 이 프로젝트는 그걸 아주 작게 해본 것이다. 모캡 없이 학습하고, 에너지 수준을 조절하고, 경로를 따라가는 구조는 FreeMusco를, 사용자가 키보드로 조종하는 부분은 PhysicsFC를 축소한 형태다.
- **인턴 들어가서 바로 쓸 기초.** MuJoCo, PPO, 보상 설계, 학습 로그 읽기를 미리 익힌다.
- **목표는 완벽하게 걷는 Ant가 아니다.** 교수님이 학부생에게서 보는 건 결과물의 완성도보다 문제를 다루는 과정이다. "문제 발견 → 원인 가설 → 수정 → 검증" 흐름을 보여주는 게 핵심이다.
- **이 프로젝트의 가치는 내가 이해하고 설명할 수 있느냐에 달려 있다.** 그래서 `docs/ant-project-study-guide.html`로 공부한다.

## 지금까지 결과

|                                       | v1 (`runs/main`)    | v2 (`runs/main_v2`) |
| ------------------------------------- | --------------------- | --------------------- |
| 학습 시간                             | 82분                  | 59분                  |
| 버틴 길이 `ep_len_mean` (최대 1000) | 371                   | **885**         |
| 속도 오차 `tracking_error_mps`      | 0.95                  | 0.84                  |
| `approx_kl` (정상 약 0.01–0.03)    | **4.11 (폭발)** | 0.000001              |
| 행동 표준편차 `std`                 | 0.144 (붕괴)          | 0.703                 |

### v1: 학습이 무너짐

- 원인 가설: 업데이트 한 번에 경사 스텝 480번(미니배치 48개 × 10에폭) + 학습률 고정. 행동 표준편차가 0.144까지 줄면서, 같은 크기로 업데이트해도 KL이 약 48배 커져 폭발.
- 수정(v2): 미니배치 4개 × 5에폭 = 업데이트당 20번, `target_kl=0.02`, 학습률 3e-4 → 0 선형 감소. Rudin et al.(2021)의 PPO 설정을 참고했다.

### v2: 안정적으로 학습됐지만 명령은 아직 못 따름

- 학습은 무너지지 않았고, 넘어지는 일이 크게 줄었다.
- 하지만 속도 오차 0.84는 가만히 서 있을 때(약 0.8)와 비슷하다. "안 넘어지고 버티기"는 배웠지만 "명령대로 걷기"는 못 배운 상태다.
- 원인 추측: 살아 있기만 해도 매 스텝 1점을 받는데, 속도 추종 점수는 명령에 꽤 가까워져야 들어온다. 그래서 "버티기만 하자"는 국소 최적에 머물렀다. 학습 후반엔 학습률이 거의 0이라 더 나아가지도 못했다.

## 다음에 할 것

1. `python evaluate.py --run runs/main_v2`로 `tracking.png`, `summary.json` 확인 → 원인 가설 검증
2. 보상 수정 후보 (하나씩 바꿔서 비교):
   - 속도 추종을 덜 엄격하게: `tracking_sigma` 0.25 → 1.0 (멀리 있어도 점수를 조금씩 받게)
   - 순간 속도 대신 최근 1초 평균 속도로 추종 점수 계산 (FreeMusco, Yu et al. 2018 방식)
   - 생존 보상 비중 낮추기
   - 걸음 품질: 행동 변화율 벌점, 발 공중 시간 보상 (Rudin et al. 2021)
3. 텐서보드 `approx_kl` 그래프(v1 vs v2) 캡처 → README "시도한 것 / 실패한 것"에 정리
4. 컨택 메일은 프로젝트 완성을 기다리지 않고 보낸다. 지금의 v1 → v2 과정과 남은 문제만으로 충분하다.

## 결과 보는 법

```bash
cd ant-command-locomotion
source .venv/bin/activate

tensorboard --logdir runs                         # 학습 과정 (localhost:6006)
python evaluate.py --run runs/main_v2             # runs/main_v2/results/ 에 그래프
python path_follow.py --run runs/main_v2 --video  # 경로 추종 GIF
python keyboard_demo.py --run runs/main_v2        # 키보드 조종 (터미널에서 w/a/s/d, q 종료)
```

## docs 폴더 안내

- `PROJECT_STATUS.md`: 이 문서
- `ant-project-study-guide.html`: 프로젝트 공부 노트 (1단계 큰 그림, 2단계 `command_ant.py` 한 줄씩)
- `cgr-paper-notes.html`: 논문 9편 상세 분석 (PhysicsFC, FreeMusco, Rudin, PPO 등)
- `cgr-reading-list.html`: 논문 9편 요약과 링크

## 새 컴퓨터에서 처음 설정할 때

```bash
cd ant-command-locomotion
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

학습 결과(`runs/`)는 이 묶음에 들어 있지 않다. 노트북에 있는 `runs/main_v2` 폴더를 옮겨오거나, 새로 학습해야 한다.
