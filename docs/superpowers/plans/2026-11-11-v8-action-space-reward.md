# v8 행동 공간 × 보상 모양 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `docs/V8_DESIGN.md`의 2×2(+C) 실험을 돌릴 수 있게 환경·학습·분석 코드를 고치고, PD 사전 시험 → 학습 전 측정 → 학습 → 판정까지 실행한다.

**Architecture:** `CommandAnt`가 물리 스텝 루프를 직접 소유한다(토크/PD 공통, 물리 스텝마다 콜백). 행동 방식·PD 이득·커널은 `runs/<run>/config.json`에 저장되고, `Policy`가 그 설정에서 환경 인자를 꺼내 모든 분석 스크립트에 넘긴다. 학습 전 정책은 `runs/<run>_init/`이라는 평범한 run 폴더로 저장해 기존 평가 도구를 그대로 쓴다.

**Tech Stack:** Python 3.12, Gymnasium 1.4 `Ant-v5`, MuJoCo 3.17, Stable-Baselines3 2.9 (PPO), pytest(새로 추가).

**Spec:** `docs/V8_DESIGN.md` (확정, 2026-11-11). 구현자는 이 계획과 함께 반드시 읽는다. 절 번호는 이 문서 기준이다.

## Global Constraints

- 새 옵션을 주지 않으면 보상·관측·궤적이 지금 코드와 같다: 무작위 행동 2000스텝에서 qpos 차이 0, 보상 차이 1e-12 이하 (설계 12절 1번).
- 행동 순서(hip_4, ankle_4, hip_1, …)와 관절 순서(hip_1, ankle_1, …)를 이어 줄 때는 항상 `model.actuator_trnid[:, 0] - 1`을 쓴다. 숫자를 직접 적지 않는다.
- PD: `q_목표 = 범위 중심 + a × 반범위`, `τ = clip(kp(q_목표 − q) − kd q̇, −150, 150)`, 물리 스텝(0.01초)마다 계산, 8개 관절 같은 이득.
- 노력 항: 모든 갈래 `E = 물리 스텝 5개 평균 Σ(τ_적용 / 150)²`, 가중치 `w(e) = 0.02 + (1 − e) × 0.23`.
- K0 = `5·exp(−오차²/1.0)`, K1 = `2.5·exp(−오차²/1.0) + 2.5·exp(−오차²/0.25)`. 오차 = `‖v − v_명령‖`.
- 학습: tanh 정책, n_envs 8, 500만 스텝(3e-4 → 0) + 이어 학습 160만 스텝(1e-4 고정), seed 0·1·2, sigma 1.0, 추종 가중치 5, 한계 페널티 1.5 / 범위 끝 20%, v_z 페널티 0.
- C 갈래 초기 σ = 0.1 (`log_std_init = ln 0.1`).
- 결과를 본 뒤 커널·이득·합격선을 바꾸지 않는다. 바꿔야 하면 설계 문서 "변경 기록"에 남긴다.
- 커밋은 `v8` 브랜치에서 한다. 커밋 메시지 끝: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- 파이썬 실행은 `env -u PYTHONPATH .venv/bin/python` (저장소 스크립트 관례).

## Review Focus

1. **옛 run(행동 방식 키가 없는 config.json)을 평가할 때** 토크로 해석되고 결과가 지금과 같아야 한다 → Task 4 `test_old_config_defaults_to_torque`.
2. **이어 학습에서 행동 방식·이득·커널이 원래 run과 다를 때** 조용히 섞이지 않고 오류로 멈춰야 한다 → Task 4 `test_resume_mismatch_raises`.
3. **PD 갈래를 접촉 분석(`Recorder`)으로 잴 때** 기록을 켜도 궤적이 같아야 한다(PD가 물리 스텝 루프를 쓰므로 바꿔 끼우기가 부딪친다) → Task 5 `test_recorder_does_not_change_pd_trajectory`.
4. **행동이 ±1(관절 한계 목표)일 때** 목표 각도가 범위를 벗어나지 않고 토크는 ±150에서 잘려야 한다 → Task 2 `test_pd_target_at_extremes`.
5. **이득을 정하지 않고 학습 스크립트를 돌릴 때** 학습을 시작하지 않아야 한다 → Task 8 `run_v8.sh` 검사 단계.

---

## 실행 방식과 관문 (2026-11-11 사용자 결정)

서브에이전트 방식. 작업은 **순차**로만 넘긴다(같은 파일을 두 에이전트가 동시에 고치지 않는다).

| 관문 | 작업 | 독립 검토자가 볼 것 |
| --- | --- | --- |
| 1 | 0~3 환경 | 행동↔관절 대응, PD 주기(목표 유지 + 물리 스텝마다 토크), 요청·적용 토크, 노력 항이 v6Bb와 같은 값, 커널 수식 |
| 2 | 4~5 연결 | 옛 run이 토크로 재현되는지, PD run이 PD로 만들어지는지, 스크립트 호출표, 접촉 분석 결과 일치 |
| 3 | 6~8 학습 전 최종 | 기준선 재측정, 사전 시험 판정, 판정 스크립트. **사전 시험 실패면 학습하지 않는다** |
| 4 | 9 | 학습·평가. **결과가 나빠도 합격선을 낮추지 않는다** |

## 파일 구조

| 파일 | 책임 |
| --- | --- |
| `envs/command_ant.py` (수정) | 물리 스텝 루프, PD, 적용 토크·노력·일률, K0/K1 |
| `envs/run_config.py` (새로) | config.json ↔ 환경 인자 변환, 이어 학습 일치 검사 |
| `policy_utils.py` (수정) | `Policy.env_kwargs` |
| `train.py` (수정) | 새 옵션, 학습 전 정책 저장, 이어 학습 검사, 텐서보드 |
| 분석 스크립트 11개 (수정) | `**policy.env_kwargs`로 환경 생성 |
| `propulsion_analysis.py`, `stance_load_analysis.py` (수정) | `Recorder`를 콜백 방식으로, 대상 run·출력 폴더 CLI |
| `baseline_rewards.py` (수정) | 공식 평가 조건, 명령 속도별, seed 3개 |
| `pd_pretest.py` (새로) | 설계 8절 T1·T2·T3 |
| `v8_judge.py` (새로) | 설계 10절 합격선 표 |
| `run_v8.sh` (새로) | 학습 12쌍 + 학습 전 정책 + 평가 |
| `tests/` (새로) | pytest |

---

### Task 0: 브랜치와 테스트 도구

**Files:** Modify `requirements.txt`; Create `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 1:** `git switch -c v8` (출발점: 현재 `cleanup-baselines-docs`, 04dad9e). `docs/V8_DESIGN.md`와 이 계획 파일을 첫 커밋으로 넣는다.
- [ ] **Step 2:** `requirements.txt`에 `pytest` 추가, `.venv/bin/pip install pytest`.
- [ ] **Step 3:** `tests/conftest.py`: 저장소 루트를 `sys.path`에 넣고, `warnings` 필터로 Gymnasium 경고를 끈다.
- [ ] **Step 4:** `env -u PYTHONPATH .venv/bin/python -m pytest tests -q` → `no tests ran`.
- [ ] **Step 5:** 커밋 `Add v8 design, plan, and pytest setup`.

### Task 1: 물리 스텝 루프와 적용 토크 (토크 갈래, 동작 불변)

**Files:** Modify `envs/command_ant.py`; Test `tests/test_command_ant.py`

**Interfaces:**
- Produces:
  - `CommandAnt.__init__(..., action_mode: str = "torque", pd_kp: float = 0.0, pd_kd: float = 0.0, tracking_kernel: str = "single")` (Task 2, 3에서 채움)
  - `CommandAnt.substep_callbacks: list[Callable[[], None]]` — 물리 스텝마다 `ctrl`을 정한 뒤, `mj_step` 직전에 순서대로 부른다.
  - `step()`의 `info`에 추가: `torque` (np.ndarray (8,), **관절 순서**, 물리 스텝 평균 적용 토크 N·m), `effort` (= 물리 스텝 평균 Σ(τ/150)²로 정의 변경), `power` (물리 스텝 평균 Σ|τ q̇|, W), `torque_limit_frac` (관절×물리 스텝 중 |τ| ≥ 0.99 × 150 비율).
- 구현: `env.unwrapped._step_mujoco_simulation`을 래퍼의 루프로 바꿔 끼운다. 루프: 물리 스텝 5번 × (ctrl 계산 → 콜백 → `mj_step(model, data)` → `actuator_force` 누적), 끝에 `mj_rnePostConstraint`(Gymnasium과 같음). 적용 토크는 `data.actuator_force`를 `actuator_trnid`로 관절 순서에 놓은 값.

- [ ] **Step 1: 실패하는 테스트**
  - `test_torque_mode_matches_original`: 같은 seed로 `make_command_ant()`와, 루프를 바꾸지 않은 원래 `CommandAnt`(테스트 안에서 `gym.make("Ant-v5")`에 지금 커밋의 보상식을 손으로 계산)에 같은 무작위 행동 2000스텝 → qpos 최대 차이 `== 0`, 보상 차이 `< 1e-12`. (원래 동작을 비교 대상으로 남기려면, 바꾸기 전 커밋의 `CommandAnt`를 `tests/_legacy_command_ant.py`로 복사해 둔다.)
  - `test_info_torque_is_joint_order`: 행동 0만 0.2, 나머지 0 → `info["torque"]`에서 hip_4(관절 순서 인덱스 6)만 30.0 N·m(오차 1e-9).
  - `test_effort_equals_sum_a2_in_torque_mode`: 무작위 tanh 행동에서 `abs(info["effort"] - sum(a**2)) < 1e-12`.
  - `test_substep_callbacks_called_five_times`: 콜백 카운터가 `step()` 한 번에 5.
- [ ] **Step 2:** `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_command_ant.py -q` → 실패.
- [ ] **Step 3:** 위 Interfaces대로 구현.
- [ ] **Step 4:** 테스트 통과.
- [ ] **Step 5:** 커밋 `Own the physics-step loop in CommandAnt; record applied torque, effort, power`.

### Task 2: PD 행동 방식

**Files:** Modify `envs/command_ant.py`; Test `tests/test_pd.py`

**Interfaces:**
- Consumes: Task 1의 루프.
- Produces: `action_mode="pd"`일 때 물리 스텝마다 `ctrl = clip(kp(q_목표 − q) − kd q̇, −150, 150) / 150`(행동 순서로 배치). 목표는 `step()` 시작 때 한 번 계산해 5 물리 스텝 동안 고정. `pd_kp <= 0`이면 `ValueError`.

- [ ] **Step 1: 실패하는 테스트**
  - `test_pd_torque_matches_hand_calc`: 임의 상태·행동에서 첫 물리 스텝의 `data.ctrl * 150`이 손 계산한 PD 토크(행동 순서)와 1e-9 이내.
  - `test_pd_target_at_extremes`: a = +1 → 관절 순서 목표 = `jnt_range[1:9, 1]`, a = −1 → `[:, 0]`; 큰 오차에서 `|ctrl| <= 1`이고 `torque_limit_frac > 0`.
  - `test_pd_single_action_moves_its_joint`: 서기 상태에서 행동 i만 +0.5 → 50 제어 스텝 뒤 `actuator_trnid`가 가리키는 관절 각도가 증가하고, 다른 관절 변화보다 크다 (i = 0..7).
  - `test_pd_target_held_torque_recomputed`: 한 `step()` 안 5 물리 스텝에서 (콜백으로 관찰) 목표 각도는 5번 모두 같고, 각 물리 스텝의 `ctrl`은 그 순간의 q, q̇로 손 계산한 PD 값과 1e-9 이내(즉 5개 값이 서로 다를 수 있다).
  - `test_info_requested_vs_applied`: `info["torque_requested"]`(잘리기 전, 관절 순서, 물리 스텝 평균)와 `info["torque"]`(잘린 뒤)를 모두 내보내고, 큰 오차에서 |요청| > 150, |적용| ≤ 150.
  - `test_pd_stand_reproduces_baseline`: kp 50, kd 10, a = 0, 명령 16개 중 0번·8번 × reset seed 0, v6B 보상 → 1000스텝 평균 |토크| < 1 N·m, 넘어짐 없음.
- [ ] **Step 2:** 실패 확인. **Step 3:** 구현. **Step 4:** 통과 + Task 1 테스트도 통과.
- [ ] **Step 5:** 커밋 `Add joint-target PD action mode`.

### Task 3: 두 항 추종 커널

**Files:** Modify `envs/command_ant.py`; Test `tests/test_kernel.py`

**Interfaces:**
- Produces: `tracking_kernel="two_term"`이면 `r_추종 = tracking_weight × (0.5 exp(−e²/tracking_sigma) + 0.5 exp(−e²/0.25))`. 상수 `NARROW_SIGMA = 0.25`. `"single"`은 지금과 같음. 다른 값은 `ValueError`. `info["r_tracking"]`은 0~1로 정규화된 값(가중치 곱하기 전).
- 함수: `tracking_reward(err2: float, weight: float, sigma: float, kernel: str) -> float` (모듈 수준, 테스트용).

- [ ] **Step 1: 실패하는 테스트** `test_kernel_table`: weight 5, sigma 1.0, 오차 = 명령 속도(서 있기)일 때 설계 7절 표의 "정지" 열과 소수 둘째 자리까지 같다 — K0: 0.2→4.80, 0.5→3.89, 1.0→1.84, 1.5→0.53; K1: 4.53, 2.87, 0.97, 0.26. 오차 0에서 둘 다 5.0. `test_two_term_monotone`: 오차 0~3을 3만 점으로 훑어 K1이 엄격히 감소.
- [ ] **Step 2–4:** 실패 확인 → 구현 → 통과.
- [ ] **Step 5:** 커밋 `Add two-term tracking kernel`.

### Task 4: run 설정 전달 (학습·이어 학습·평가)

**Files:** Create `envs/run_config.py`; Modify `train.py`, `policy_utils.py`, `evaluate.py`, `evaluate_repeated.py`, `gait_analysis.py`, `propulsion_analysis.py`, `stance_load_analysis.py`, `contact_check.py`, `limit_analysis.py`, `sensitivity.py`, `baseline_rewards.py`, `path_follow.py`, `keyboard_demo.py`; Test `tests/test_run_config.py`

**Interfaces:**
- Produces (`envs/run_config.py`):
  - `ENV_KEYS = ("tracking_sigma", "tracking_weight", "alive_weight", "limit_weight", "limit_soft", "vz_weight", "action_mode", "pd_kp", "pd_kd", "tracking_kernel")`
  - `env_kwargs_from_config(cfg: dict) -> dict` — 없는 키는 `CommandAnt` 기본값(옛 run은 `action_mode="torque"`, `tracking_kernel="single"`).
  - `load_env_kwargs(run_dir: str | Path) -> dict` — `config.json`을 읽는다.
  - `check_resume_compatible(old: dict, new: dict) -> None` — `action_mode`, `pd_kp`, `pd_kd`, `tracking_kernel`이 다르면 `ValueError`(메시지에 키 이름).
  - `Policy.env_kwargs: dict` (= `load_env_kwargs(run_dir)`). `Policy` 안의 dummy 환경도 이것으로 만든다.
- `train.py` 새 옵션: `--action-mode {torque,pd}`, `--pd-kp`, `--pd-kd`, `--tracking-kernel {single,two_term}`, `--log-std-init FLOAT`(없으면 SB3 기본 0), `--save-init`. 설정 저장 키 이름은 `ENV_KEYS`와 같게(argparse의 `--pd-kp` → `pd_kp`).
- `--save-init`: 학습 전에 `runs/<run_name>_init/`에 `final_model.zip`, `vecnormalize.pkl`, `config.json`(같은 설정, `run_name`만 `_init`)을 저장한다.
- `--resume` 시 원래 run의 `config.json`과 `check_resume_compatible`.
- 텐서보드 `TrackingLogger`에 `command/power`, `command/torque_limit_frac`, `command/mean_abs_torque` 추가.
- 분석 스크립트: 정책을 굴리는 모든 `make_command_ant(...)` 호출에 `**policy.env_kwargs`를 넘긴다(명시 인자와 겹치면 명시 인자가 이긴다: `{**policy.env_kwargs, **explicit}`). `baseline_rewards.py`처럼 정책 없이 환경을 만드는 곳은 Task 6에서 다룬다.

- [ ] **Step 1: 실패하는 테스트**
  - `test_old_config_defaults_to_torque`: `runs/main_v6Bb/config.json` → `action_mode == "torque"`, `tracking_kernel == "single"`, `tracking_sigma == 1.0`, `limit_weight == 1.5`.
  - `test_resume_mismatch_raises`: torque ↔ pd, single ↔ two_term, kp 30 ↔ 60 각각 `ValueError`; 같으면 통과.
  - `test_policy_env_kwargs_roundtrip`: `train.py --action-mode pd --pd-kp 60 --pd-kd 15 --tracking-kernel two_term --squash --timesteps 2048 --n-envs 1 --save-init --run-name _test_v8` 실행(tmp에 `runs` 심볼릭 링크 없이 실제 `runs/_test_v8*`에 쓰고 테스트 끝에 지움) → `Policy("runs/_test_v8").env_kwargs["action_mode"] == "pd"`, `runs/_test_v8_init/final_model.zip` 존재.
  - `test_log_std_init`: `--log-std-init -2.302585` 로 2048스텝 → 저장된 정책 `log_std` 평균이 −2.30 ± 0.05.
- [ ] **Step 2:** 실패 확인.
- [ ] **Step 3:** 구현. 분석 스크립트 11개 수정 후 `grep -n "make_command_ant(" *.py`로 정책을 굴리는 호출에 빠진 곳이 없는지 확인.
  - `test_v6Bb_eval_reproduces`: 커밋된 `runs/main_v6Bb/results/summary_repeated.json`에서 에피소드 3개(명령 0·12, reset seed 0, 노이즈 끔 2개 + 켬 1개)를 고르고, `evaluate_repeated.episode`로 다시 돌린 `speed_3_50`, `fell`, `outcome`이 기록과 같다(속도 1e-9 이내).
  - `test_pd_run_builds_pd_env`: `test_policy_env_kwargs_roundtrip`에서 만든 run으로 `make_command_ant(**Policy(...).env_kwargs).action_mode == "pd"`이고 kp가 60.
- [ ] **Step 4:** 테스트 통과. 스크립트 확인: `grep -n "make_command_ant(" *.py`의 모든 호출을 표로 적어(파일, 줄, `env_kwargs`를 넘기는지, 안 넘기면 이유) 검토자에게 넘긴다. 정책 없이 환경을 만드는 곳(`evaluate_repeated.check`, `baseline_rewards`의 기준선 정책)만 예외다.
- [ ] **Step 5:** 커밋 `Carry action mode, PD gains, and kernel through run config`.

### Task 5: 접촉 기록기를 콜백 방식으로

**Files:** Modify `propulsion_analysis.py`, `stance_load_analysis.py`, `contact_check.py`; Test `tests/test_recorder.py`

**Interfaces:**
- Consumes: `CommandAnt.substep_callbacks` (Task 1).
- Produces: `Recorder(env, u)`가 `_step_mujoco_simulation`을 바꿔 끼우지 않고, 자신의 `on_substep()`을 `env.substep_callbacks`에 등록한다(`on`이 참일 때만 `mj_forward` + `sample`). `stance_load_analysis.py`에 `--runs`(run 접두사 목록, 기본은 지금 4개)와 `--out`(기본 `runs/propulsion_v7`), `propulsion_analysis.py`에 `--out`.

- [ ] **Step 1: 실패하는 테스트** `test_recorder_does_not_change_pd_trajectory`: PD(kp 60, kd 15), 무작위 행동 200스텝, 기록 켬/끔 → qpos 차이 0. `test_recorder_torque_matches_committed`: v6Bb seed 0, 명령 12, 노이즈 끔 → `stance_load_analysis.load_stats` 평균 지속 시간이 지금 `runs/propulsion_v7/stance_load.json` 값(시드 0만 다시 계산)과 같다.
- [ ] **Step 2–4:** 실패 확인 → 구현 → 통과.
- [ ] **Step 5:** 커밋 `Recorder uses substep callbacks; add run/out options`.

### Task 6: 기준선 보상 (공식 평가 조건) 과 README 정정

**Files:** Modify `baseline_rewards.py`, `README.md`, `docs/PROJECT_STATUS.md`; Output `runs/baselines/official.json`

**Interfaces:**
- Produces: `python baseline_rewards.py --official [--reward v6B|v6B_two_term] [--pd-kp KP --pd-kd KD]` → 명령 16개 × reset seed 0~9 × 1000스텝, 에너지 0.5. 정책: 토크 0, 손으로 짠 PD 서기(kp 50, kd 10, 토크 모드로 넣음), `action_mode="pd"` + a = 0 (주어진 이득), v6Bb seed 3개 노이즈 끔/켬(노이즈 seed 10000 + reset seed). 출력: 정책별 평균, 0.5 / 1.0 m/s 명령별 평균, (v6Bb) seed 평균 ± 표준편차, 같은 (명령, seed) 쌍에서 PD 서기보다 낮은 비율. 기존 학습 분포 측정(인자 없음)은 그대로.

- [ ] **Step 1:** 구현 후 실행: `env -u PYTHONPATH .venv/bin/python baseline_rewards.py --official`.
- [ ] **Step 2: 확인** — 설계 2절 값과 비교: PD 서기 3864, 토크 0 3409, v6Bb 끔 3432 ± 274, 켬 3003 ± 153, 0.5 m/s PD 서기 4890. 어느 값이 1% 넘게 다르면 설계 문서 2절 표를 저장소 값으로 고치고 "변경 기록"에 적는다.
- [ ] **Step 3:** `--reward v6B_two_term`으로 실행 → PD 서기 공식 조건 약 2916(설계 7절 계산값)과 비교해 기록.
- [ ] **Step 4:** README 요약 4·5번, "기준선 보상 비교" 해석, "다음에 할 일" 1순위를 "용의자 둘(행동 공간·탐색 / 보상 모양), v8은 2×2 (`docs/V8_DESIGN.md`)"로 고친다. 처음 진단("보상보다 행동 공간")은 지우지 말고 "2026-11-11 수정" 표시와 함께 이유(0.5 m/s 명령에서 서 있기가 최대의 80%)를 남긴다. PROJECT_STATUS.md 표의 마지막 줄과 "다음에 할 것" 2번도 같게.
- [ ] **Step 5:** 커밋 `Official-condition baselines; README now names two suspects and the v8 plan`.

### Task 7: PD 사전 시험 실행과 이득 확정

**Files:** Create `pd_pretest.py`; Output `runs/v8_pretest/summary.json`; Modify `docs/V8_DESIGN.md` (8절 아래 결과)

**Interfaces:**
- Consumes: `make_command_ant(action_mode="pd", pd_kp, pd_kd, ...)`, `Recorder`, `stance_load_analysis.load_stats`.
- Produces: `python pd_pretest.py` → 후보 (30, 10), (60, 15), (120, 21) 각각 T1·T2·T3 지표와 통과 여부, 선택 규칙 적용 결과(`chosen: {kp, kd}` 또는 `chosen: null`).
- T1/T2: 설계 8절 표 그대로(50초 × reset seed 0~9, T2는 `np.random.default_rng(reset seed)`로 σ 1.0 행동).
- T3: 시간만의 함수인 대각선 교대(trot) 패턴. 엉덩이 `a = A_h sin(2πt/T + φ_다리)`, 발목 `a = A_a sin(2πt/T + φ_다리 + Δ)`; 대각선 쌍(다리 1·3 / 2·4)은 위상 π 차이. 훑기: T ∈ {0.4, 0.6, 0.8, 1.0} s, A_h ∈ {0.3, 0.6, 0.9}, A_a ∈ {0.3, 0.6, 0.9}, Δ ∈ {+π/2, −π/2}, 그리고 엉덩이 부호를 다리마다 뒤집는 진행 방향 변형(+x로 가는 패턴이 하나는 들어가도록; 구현자가 정함). 각 15초, 처음 3초 제외, 명령 (1, 0)으로 Recorder 투영. 지표: +x 속도, 하중 디딤 평균 지속(F_z ≥ 0.25 mg), 하중 디딤 중 미끄러짐 > 0.1 m/s 비율, 관절 한계 ±2° 비율, `torque_limit_frac`, 넘어짐.

- [ ] **Step 1:** 구현 후 스모크: `pd_pretest.py --quick`(후보 1개, T3 패턴 2개) 끝까지 실행.
- [ ] **Step 2:** 전체 실행 `env -u PYTHONPATH .venv/bin/python pd_pretest.py`.
- [ ] **Step 3: 판정** — `chosen`이 null이면 **여기서 멈추고 사용자에게 보고한다**(설계 8절 규칙 3). 고른 후보의 최선 패턴이 10절 3·4번 합격선(디딤 0.10초, 미끄러짐 0.50)을 못 넘으면 역시 멈추고 보고한다(설계 "합격선 점검").
- [ ] **Step 4:** 설계 문서 8절 아래에 날짜, 후보별 표, 고른 kp/kd를 적는다.
- [ ] **Step 5:** 커밋 `PD gain pretest; fix kp/kd for v8`.

### Task 8: 학습 스크립트와 판정 스크립트

**Files:** Create `run_v8.sh`, `v8_judge.py`; Test `tests/test_judge.py`

**Interfaces:**
- `run_v8.sh`: `run_v7.sh`와 같은 틀(건너뛰기, 로그, `PY`, `SEEDS`, `PUSH`). `KP`, `KD` 환경 변수가 없으면 학습 전에 `exit 1`. 갈래:
  - B: `--action-mode pd --pd-kp $KP --pd-kd $KD`
  - D: B + `--tracking-kernel two_term`
  - E: `--tracking-kernel two_term`
  - C: `--log-std-init -2.302585`
  - 공통: `--squash --tracking-sigma 1.0 --tracking-weight 5.0 --limit-weight 1.5 --limit-soft 0.8 --n-envs 8`, 첫 학습에 `--save-init`, 이어 학습은 `--resume --constant-lr 1e-4 --timesteps 1600000`.
  - 순서: seed 0의 B·D·E·C → seed 1 → seed 2. 끝나면 `evaluate_repeated.py --out runs/repeated_v8`(대상: v6Bb, v8{B,D,E,C}b, 각 `_init`, seed 3개), `stance_load_analysis.py --record --runs main_v6Bb main_v8Bb main_v8Db main_v8Eb main_v8Cb --out runs/propulsion_v8`, 그다음 `v8_judge.py`.
- `v8_judge.py`: 입력 `runs/repeated_v8`의 run별 `summary_repeated.json`과 `runs/propulsion_v8/stance_load.json`. 출력 `runs/repeated_v8/judgement.md`: 설계 10절 합격선 표(갈래별 노이즈 끔 값, 노이즈 켬 보조 기준, 항목별 ✅/❌), 판정 라벨 셋 중 하나(`걷는다` / `결정론적으로만 걷는다` / `서 있기만 한다`) 또는 `그 밖: <못 넘은 항목>`, 1번 지표의 0.5 / 1.0 m/s 명령별 값, 학습 전(`_init`) 대비 표.
  - 함수 `judge(off: dict, on: dict) -> tuple[str, dict[str, bool]]` — 입력 키: `track_rate, speed, move_rate, permanent_freeze, load_dur, load_frac, all_air, load_slip, limit_occ, fell`.

- [ ] **Step 1: 실패하는 테스트** `test_judge_labels`: (a) 모든 값이 합격선을 겨우 넘는 off/on → `"걷는다"`; (b) off는 합격, on의 넘어짐 0.08 → `"결정론적으로만 걷는다"`; (c) off speed 0.03, move_rate 0.1 → `"서 있기만 한다"`; (d) off에서 load_slip 0.6만 실패 → `"그 밖"`이고 실패 항목에 `"미끄러짐"`. 경계값은 설계 10절 그대로(이상/이하 포함).
- [ ] **Step 2–4:** 실패 확인 → 구현 → 통과.
- [ ] **Step 5:** `SEEDS=0 PUSH=0 KP=60 KD=15` 와 학습 스텝을 줄이는 `TIMESTEPS=20000 CONT_TIMESTEPS=4096` 환경 변수(스크립트에 추가)로 전체 흐름을 끝까지 돌려 `judgement.md`가 생기는지 확인한 뒤 만든 `runs/main_v8*` 시험 폴더를 지운다.
- [ ] **Step 6:** 커밋 `Add run_v8.sh and v8 judgement script`.

### Task 9: 실행과 결과 기록

**Files:** Output `runs/main_v8*`, `runs/repeated_v8/`, `runs/propulsion_v8/`, 영상; Modify `README.md`, `docs/PROJECT_STATUS.md`

- [ ] **Step 1:** `tmux new -s v8` 안에서 `KP=<Task 7 값> KD=<값> PUSH=0 bash run_v8.sh`. 예상 시간은 **새 학습 12쌍 전체(B·D·E·C × seed 3개, 각 500만 + 160만 스텝)** 기준 3.5~4시간(v6·v7 실측: 쌍당 약 17분, PD는 물리 스텝 루프가 파이썬이라 더 느릴 수 있음) + 평가 약 20분. seed 0의 4쌍이 끝난 시점(약 1.2시간)의 실제 시간으로 남은 시간을 다시 계산해 사용자에게 알린다.
- [ ] **Step 2:** 실패한 run이 있으면 로그를 보고 사용자에게 보고한다(다시 돌리기 전에).
- [ ] **Step 3:** `gait_analysis.py`로 갈래마다 seed 0 명령 12번(앞 1.0 m/s)과 6번(0.5 m/s 90°) 영상 → `runs/gait_v8/`. `evaluate.py --run`으로 에너지 반응(`energy.png`, e = 0 / 0.5 / 1의 노력; 일률 `info["power"]` 평균을 summary.json에 추가) 기록. 판정에는 쓰지 않는다(설계 10절). `baseline_rewards.py --official --reward v6B_two_term --pd-kp KP --pd-kd KD`로 D·E 보상 기준의 PD 서기·토크 0도 잰다.
- [ ] **Step 4:** README에 v8 절 추가(v7 절 형식): 설계 링크, 갈래 표, `judgement.md` 표, 학습 전 대비, 설계 11절 해석표의 해당 줄(맞는 줄이 없으면 없다고 적는다), 영상. 못 넘은 항목도 적는다. PROJECT_STATUS.md 갱신.
- [ ] **Step 5:** 커밋 `v8 results`. push는 사용자에게 물어본 뒤.
