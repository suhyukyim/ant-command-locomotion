#!/usr/bin/env bash
# v7 A/B (몸통 수직 속도 페널티 강도 비교) × 학습 seed 0, 1, 2 를 한 번에:
#   학습 500만 스텝 → 학습률 1e-4 고정 이어 학습 160만 스텝 → 반복 평가 (v5b, v6Bb 포함 18 run) → commit + push
#
#   공통: v6B와 같음 (tanh, sigma 1.0, 추종 가중치 5, 관절 한계 페널티 범위 끝 20% · 가중치 1.5)
#   A: 수직 속도 페널티 1.0 × v_z²
#   B: 수직 속도 페널티 3.0 × v_z²
#
# run 이름: main_v7A (seed 0), main_v7A_s1, main_v7A_s2, 이어 학습은 main_v7Ab, main_v7Ab_s1, ...
# 순서: seed 0의 A·B → seed 1 → seed 2 (중간에 멈춰도 설정 비교가 가능하도록)
# 다시 실행하면 final_model.zip이 이미 있는 run은 건너뛴다 (중간에 끊긴 run은 처음부터 다시).
#
# 사용:
#   tmux new -s v7            # 원격 연결이 끊겨도 계속 돌도록
#   bash run_v7.sh            # 진행 상황: tail -f runs/v7_all.log, 학습별 출력: runs/<run>.log
#   PY=.venv/bin/python bash run_v7.sh   # 파이썬 경로를 지정할 때
#   PUSH=0 bash run_v7.sh                # 끝나고 commit/push 하지 않을 때

set -u
cd "$(dirname "$0")"
PY="${PY:-python}"
SEEDS="${SEEDS:-0 1 2}"
PUSH="${PUSH:-1}"
CONFIGS="A:1.0 B:3.0"
COMMON="--tracking-sigma 1.0 --tracking-weight 5.0 --n-envs 8"
LOG=runs/v7_all.log
LIMIT="--limit-weight 1.5 --limit-soft 0.8"
mkdir -p runs

py() { env -u PYTHONPATH "$PY" "$@"; }
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

T0=$(date +%s)
say "start (seeds: $SEEDS)"
for s in $SEEDS; do
  sfx=""; [ "$s" != "0" ] && sfx="_s$s"
  for cfg in $CONFIGS; do
    IFS=: read -r name vzw <<< "$cfg"
    base="main_v7${name}${sfx}"
    cont="main_v7${name}b${sfx}"
    VZ="--vz-weight $vzw"

    if [ -f "runs/$base/final_model.zip" ]; then
      say "skip $base (already done)"
    else
      say "train $base (vz weight $vzw, seed $s)"
      if ! py train.py --squash --timesteps 5000000 --seed "$s" --run-name "$base" $COMMON $LIMIT $VZ \
          > "runs/$base.log" 2>&1; then
        say "FAILED $base (see runs/$base.log)"; continue
      fi
      say "done $base: $(tail -n 1 "runs/$base.log")"
    fi

    if [ -f "runs/$cont/final_model.zip" ]; then
      say "skip $cont (already done)"
    else
      say "train $cont (resume $base, constant lr 1e-4)"
      if ! py train.py --resume "runs/$base" --constant-lr 1e-4 --timesteps 1600000 --seed "$s" \
          --run-name "$cont" $COMMON $LIMIT $VZ > "runs/$cont.log" 2>&1; then
        say "FAILED $cont (see runs/$cont.log)"; continue
      fi
      say "done $cont: $(tail -n 1 "runs/$cont.log")"
    fi
  done
done

# 반복 평가: 끝난 run만 (v5b, v6Bb도 새 지표(v_z², 몸통 높이·접촉)를 얻도록 함께). 결과는 runs/repeated_v7/
# (v6 비교표 runs/repeated/는 그대로 둔다)
RUNS=""
for r in main_v5b main_v5b_s1 main_v5b_s2 main_v6Bb main_v6Bb_s1 main_v6Bb_s2 \
         main_v7A main_v7Ab main_v7B main_v7Bb \
         main_v7A_s1 main_v7Ab_s1 main_v7B_s1 main_v7Bb_s1 \
         main_v7A_s2 main_v7Ab_s2 main_v7B_s2 main_v7Bb_s2; do
  [ -f "runs/$r/final_model.zip" ] && RUNS="$RUNS runs/$r"
done
say "evaluate:$RUNS"
if ! py evaluate_repeated.py --runs $RUNS --out runs/repeated_v7 > runs/v7_eval.log 2>&1; then
  say "FAILED evaluation (see runs/v7_eval.log)"; exit 1
fi
say "evaluation done ($(tail -n 1 runs/v7_eval.log))"
say "total $(( ($(date +%s) - T0) / 60 )) min"

if [ "$PUSH" = "1" ]; then
  git add runs/main_v7* runs/*/results/summary_repeated.json runs/repeated_v7
  git commit -q -m "v7 A/B (vertical velocity penalty) training and repeated evaluation" \
    && git push -q && say "pushed" || say "commit/push failed — run git status"
fi
