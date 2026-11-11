#!/usr/bin/env bash
# v8 (docs/V8_DESIGN.md): 행동 공간 × 추종 커널 2×2 + 탐색 노이즈 갈래 × 학습 seed 0, 1, 2 를 한 번에:
#   학습 500만 스텝 (+ 학습 전 정책 저장) → 학습률 1e-4 고정 이어 학습 160만 스텝
#   → 반복 평가 (v6Bb = 갈래 A, 학습 전 정책 포함) → 하중 디딤 분석 → 판정 → commit (+ push)
#
#   B: PD 목표 각도, 기존 커널        D: PD 목표 각도, 두 항 커널
#   E: 토크, 두 항 커널               C: 토크, 기존 커널, 초기 탐색 σ 0.1
#   공통: v6B와 같음 (tanh, sigma 1.0, 추종 가중치 5, 관절 한계 페널티 범위 끝 20% · 가중치 1.5)
#
# PD 이득은 사전 시험(pd_pretest.py, cma_gait.py)으로 정한 값을 반드시 넘긴다. 없으면 학습하지 않는다.
#   KP=120 KD=21 bash run_v8.sh
# 순서: seed 0의 B·D·E·C → seed 1 → seed 2. 다시 실행하면 final_model.zip이 있는 run은 건너뛴다.
# 옵션: PY=.venv/bin/python, SEEDS="0", PUSH=0 (기본 0: push는 사람이 확인 후),
#       TIMESTEPS / CONT_TIMESTEPS (흐름 시험용으로 줄일 때만)

set -u
cd "$(dirname "$0")"
PY="${PY:-.venv/bin/python}"
SEEDS="${SEEDS:-0 1 2}"
PUSH="${PUSH:-0}"
TIMESTEPS="${TIMESTEPS:-5000000}"
CONT_TIMESTEPS="${CONT_TIMESTEPS:-1600000}"
if [ -z "${KP:-}" ] || [ -z "${KD:-}" ]; then
  echo "KP, KD가 없다. PD 사전 시험으로 정한 이득을 넘길 것 (예: KP=120 KD=21 bash run_v8.sh)" >&2
  exit 1
fi
COMMON="--squash --tracking-sigma 1.0 --tracking-weight 5.0 --limit-weight 1.5 --limit-soft 0.8 --n-envs 8"
PD="--action-mode pd --pd-kp $KP --pd-kd $KD"
declare -A ARM=(
  [B]="$PD"
  [D]="$PD --tracking-kernel two_term"
  [E]="--tracking-kernel two_term"
  [C]="--log-std-init -2.302585"
)
LOG=runs/v8_all.log
mkdir -p runs

py() { env -u PYTHONPATH "$PY" "$@"; }
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

T0=$(date +%s)
say "start (seeds: $SEEDS, KP $KP, KD $KD, timesteps $TIMESTEPS + $CONT_TIMESTEPS)"
for s in $SEEDS; do
  sfx=""; [ "$s" != "0" ] && sfx="_s$s"
  for name in B D E C; do
    base="main_v8${name}${sfx}"
    cont="main_v8${name}b${sfx}"
    ARGS="$COMMON ${ARM[$name]}"
    if [ -f "runs/$base/final_model.zip" ]; then
      say "skip $base (already done)"
    else
      say "train $base ($name, seed $s)"
      if ! py train.py --timesteps "$TIMESTEPS" --seed "$s" --run-name "$base" --save-init $ARGS \
          > "runs/$base.log" 2>&1; then
        say "FAILED $base (see runs/$base.log)"; continue
      fi
      say "done $base: $(tail -n 1 "runs/$base.log")"
    fi
    if [ -f "runs/$cont/final_model.zip" ]; then
      say "skip $cont (already done)"
    else
      say "train $cont (resume $base, constant lr 1e-4)"
      if ! py train.py --resume "runs/$base" --constant-lr 1e-4 --timesteps "$CONT_TIMESTEPS" --seed "$s" \
          --run-name "$cont" $ARGS > "runs/$cont.log" 2>&1; then
        say "FAILED $cont (see runs/$cont.log)"; continue
      fi
      say "done $cont: $(tail -n 1 "runs/$cont.log")"
    fi
  done
done

RUNS=""
CONFS="main_v6Bb"
for r in main_v6Bb main_v6Bb_s1 main_v6Bb_s2; do RUNS="$RUNS runs/$r"; done
for name in B D E C; do
  [ -f "runs/main_v8${name}b/final_model.zip" ] && CONFS="$CONFS main_v8${name}b"
  for s in $SEEDS; do
    sfx=""; [ "$s" != "0" ] && sfx="_s$s"
    for r in "main_v8${name}b${sfx}" "main_v8${name}${sfx}_init"; do
      [ -f "runs/$r/final_model.zip" ] && RUNS="$RUNS runs/$r"
    done
  done
done
say "evaluate:$RUNS"
if ! py evaluate_repeated.py --runs $RUNS --out runs/repeated_v8 > runs/v8_eval.log 2>&1; then
  say "FAILED evaluation (see runs/v8_eval.log)"; exit 1
fi
say "stance-load analysis: $CONFS"
if ! py stance_load_analysis.py --record --runs $CONFS --out runs/propulsion_v8 > runs/v8_stance.log 2>&1; then
  say "FAILED stance-load analysis (see runs/v8_stance.log)"; exit 1
fi
py v8_judge.py >> "$LOG" 2>&1 || say "FAILED judgement"
say "total $(( ($(date +%s) - T0) / 60 )) min"

if [ "$PUSH" = "1" ]; then
  git add runs/main_v8* runs/repeated_v8 runs/propulsion_v8 runs/*/results/summary_repeated.json
  git commit -q -m "v8 training, repeated evaluation, and judgement" && git push -q && say "pushed" || say "commit/push failed"
fi
