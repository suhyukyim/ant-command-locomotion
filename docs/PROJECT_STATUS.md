# Ant 프로젝트: 왜 하나, 지금 어디까지 왔나

2026-11-10 기준 (이전 판은 2026-11-01, v1·v2까지)

## 이 프로젝트를 하는 이유

이윤상 교수님(한양대 CGR Lab) 겨울 학부인턴 면담(2027년 2월)에서 "이런 걸 해봤습니다"라고 보여줄 포트폴리오다. 컨택 메일은 이미 보냈다.

- **랩이 하는 일을 직접 해봤다는 증거.** CGR Lab의 핵심은 물리 시뮬레이션과 강화학습으로 캐릭터·로봇을 움직이는 것이다. 이 프로젝트는 그걸 아주 작게 해본 것이다. 모캡 없이 학습하고, 에너지 수준을 조절하고, 경로를 따라가는 구조는 FreeMusco를, 사용자가 키보드로 조종하는 부분은 PhysicsFC를 축소한 형태다.
- **인턴 들어가서 바로 쓸 기초.** MuJoCo, PPO, 보상 설계, 학습 로그 읽기를 미리 익힌다.
- **목표는 완벽하게 걷는 Ant가 아니다.** 교수님이 학부생에게서 보는 건 결과물의 완성도보다 문제를 다루는 과정이다. "문제 발견 → 원인 가설 → 수정 → 검증" 흐름을 보여주는 게 핵심이다.
- **이 프로젝트의 가치는 내가 이해하고 설명할 수 있느냐에 달려 있다.** 그래서 `docs/STUDY_LOG.md`와 `docs/ant-project-study-guide.html`로 공부한다.

## 지금 상태 한눈에

| | 상태 |
| --- | --- |
| 기준 정책 | **v6Bb** (`runs/main_v6Bb`, 관절 한계 위치 페널티 + 이어 학습) |
| 넘어짐 / 굳음 | 거의 없음 (노이즈 켬 넘어짐 0.07, 영구 굳음 0) |
| 명령 방향 속도 | 0.21 m/s (노이즈 켬) / 0.23 m/s (끔). 명령 평균 0.75 m/s의 약 1/3 |
| 걸음 모양 | 다리 교대 보행 아님. 발이 0.02~0.03초만 닿는 충돌형 접촉 + 미끄러짐으로 튀어 다님 |
| 에너지 다이얼 (e) | v3부터 반응 없음 |
| 경로 추종 / 키보드 데모 | v2 이후 다시 돌리지 않음 |
| 자기 보상 기준 | **학습된 정책(노이즈 켬 2933)이 손으로 짠 "서 있기"(3842)보다 낮음** |

## 지금까지 흐름 (자세한 건 README)

| 단계 | 한 줄 |
| --- | --- |
| v1 | PPO 업데이트가 너무 커서 학습이 무너짐 (KL 4.11) |
| v2 | 업데이트 줄여 안정화. 하지만 버티기만 배움 (속도 오차 = 서 있기) |
| v3, v4 | 속도 추종 보상을 키움. 노이즈 켜야만 움직임 → 끔/켬 나눠 평가하기 시작 |
| v4b, v5 | 학습 부족과 행동 포화를 따로 검증. tanh(v5)로 노이즈 없이도 걷기 시작 |
| v5b + seed 반복 + 50초 반복 평가 | 이어 학습 효과는 seed 3개에서 재현. 하지만 노이즈 끔 64%가 결국 굳음 |
| v6 (A/B/C × seed 3) | 관절 한계 페널티로 굳음 0. 대신 더 튐 |
| v7 (A/B × seed 3) | 수직 속도 페널티. 발목을 잠가서 해결 → 속도 하락. 기준은 v6Bb 유지 |
| 접촉 분석 | 모든 정책이 정상 디딤 없이 충돌 + 미끄러짐으로 이동 |
| 기준선 비교 (2026-11-10~11) | 학습된 정책 < 서 있기. 용의자 둘: 행동 공간·탐색, 보상 모양(0.5 m/s 명령에서 서 있기가 최대의 80%) |

같은 날 `limit_analysis.py`의 행동↔관절 대응 버그를 찾아 고쳤다. 결론이 "8개 관절 모두 정책이 한계로 직접 민다"로 바뀌었다(README에 정정 표시).

## 다음에 할 것

1. **공부 따라잡기 (면담 전 우선).** `STUDY_LOG.md` 10번(왜 tanh인가)과 11번(기준선 표)을 내 말로 설명할 수 있게 하기. 실험이 설명보다 많이 앞서 있다.
2. **v8: 행동 공간 × 추종 커널 2×2 (+ 탐색 노이즈 갈래).** 설계·합격선은 `docs/V8_DESIGN.md`(학습 전에 고정). PD 사전 시험을 통과하지 못하면 학습하지 않고, 결과가 나빠도 합격선을 낮추지 않는다. 참고: Peng & van de Panne 2017, *Does the Choice of Action Space Matter?*
3. **걷기 시작한 뒤에.** 에너지 다이얼(노력 항을 행동 제곱이 아니라 실제 토크로), 방향별 실패에 대칭성 활용(Yu et al. 2018), 경로 추종·키보드 데모 다시 돌리기. Humanoid는 인턴 이후로 미뤄도 된다.

페널티를 하나 더 붙이는 쪽(v7 변형, 발목 한계 강화)은 기준선 비교로 보면 같은 식의 우회로를 다시 만날 가능성이 커서 뒤로 미뤘다.

## 결과 보는 법

```bash
cd ant-command-locomotion
source .venv/bin/activate

tensorboard --logdir runs                             # 학습 과정 (localhost:6006)
python evaluate_repeated.py                           # 공식 평가: 50초 반복 (runs/repeated/comparison.md)
python baseline_rewards.py                            # 기준선 보상 비교 (5초)
python gait_analysis.py --case runs/main_v6Bb:12 --out runs/gait_v6   # 보행 영상 + 그래프
python evaluate.py --run runs/main_v6Bb               # 15초 단일 평가 (참고용)
python keyboard_demo.py --run runs/main_v6Bb          # 키보드 조종 (터미널에서 w/a/s/d, q 종료)
```

## docs 폴더 안내

- `PROJECT_STATUS.md`: 이 문서
- `STUDY_LOG.md`: teach-me 공부 기록. 1~8번은 공부 완료(v1·v2), 9~12번은 v3 이후 공부용 노트(내 말로 칸 비어 있음)
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

학습된 모델(`runs/*/final_model.zip`, `vecnormalize.pkl`)은 저장소에 들어 있다. 체크포인트(`runs/*/checkpoints/`)와 학습 로그는 올리지 않는다.
