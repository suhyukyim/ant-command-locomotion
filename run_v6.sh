#!/usr/bin/env bash
# v6 A/B/C (관절 한계 페널티 범위·강도 비교) × 학습 seed 0, 1, 2 를 한 번에:
#   학습 500만 스텝 → 학습률 1e-4 고정 이어 학습 160만 스텝 → 반복 평가 (v5/v5b 포함) → commit + push
#
#   A: 범위 끝 10% (--limit-soft 0.9), 가중치 3.0   좁은 구간, 강한 페널티
#   B: 범위 끝 20% (--limit-soft 0.8), 가중치 1.5   넓은 구간, 완만한 페널티
#   C: 범위 끝 20% (--limit-soft 0.8), 가중치 3.0   넓은 구간, 강한 페널티
#
# run 이름: main_v6A (seed 0), main_v6A_s1, main_v6A_s2, 이어 학습은 main_v6Ab, main_v6Ab_s1, ...
# 순서: seed 0의 A·B·C → seed 1의 A·B·C → seed 2 (중간에 멈춰도 설정 비교가 가능하도록)
# 다시 실행하면 final_model.zip이 이미 있는 run은 건너뛴다 (중간에 끊긴 run은 처음부터 다시).
#
# 사용:
#   tmux new -s v6            # 원격 연결이 끊겨도 계속 돌도록
#   bash run_v6.sh            # 진행 상황: tail -f runs/v6_all.log, 학습별 출력: runs/<run>.log
#   PY=.venv/bin/python bash run_v6.sh   # 파이썬 경로를 지정할 때
#   PUSH=0 bash run_v6.sh                # 끝나고 commit/push 하지 않을 때

set -u
cd "$(dirname "$0")"
PY="${PY:-python}"
SEEDS="${SEEDS:-0 1 2}"
PUSH="${PUSH:-1}"
CONFIGS="A:0.9:3.0 B:0.8:1.5 C:0.8:3.0"
COMMON="--tracking-sigma 1.0 --tracking-weight 5.0 --n-envs 8"
LOG=runs/v6_all.log
mkdir -p runs

py() { env -u PYTHONPATH "$PY" "$@"; }
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

T0=$(date +%s)
say "start (seeds: $SEEDS)"
for s in $SEEDS; do
  sfx=""; [ "$s" != "0" ] && sfx="_s$s"
  for cfg in $CONFIGS; do
    IFS=: read -r name soft w <<< "$cfg"
    base="main_v6${name}${sfx}"
    cont="main_v6${name}b${sfx}"
    LIMIT="--limit-weight $w --limit-soft $soft"

    if [ -f "runs/$base/final_model.zip" ]; then
      say "skip $base (already done)"
    else
      say "train $base (soft $soft, weight $w, seed $s)"
      if ! py train.py --squash --timesteps 5000000 --seed "$s" --run-name "$base" $COMMON $LIMIT \
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
          --run-name "$cont" $COMMON $LIMIT > "runs/$cont.log" 2>&1; then
        say "FAILED $cont (see runs/$cont.log)"; continue
      fi
      say "done $cont: $(tail -n 1 "runs/$cont.log")"
    fi
  done
done

# 반복 평가: 끝난 run만 (v5/v5b도 새 지표를 얻도록 함께)
RUNS=""
for r in main_v5 main_v5_s1 main_v5_s2 main_v5b main_v5b_s1 main_v5b_s2 \
         main_v6A main_v6Ab main_v6B main_v6Bb main_v6C main_v6Cb \
         main_v6A_s1 main_v6Ab_s1 main_v6B_s1 main_v6Bb_s1 main_v6C_s1 main_v6Cb_s1 \
         main_v6A_s2 main_v6Ab_s2 main_v6B_s2 main_v6Bb_s2 main_v6C_s2 main_v6Cb_s2; do
  [ -f "runs/$r/final_model.zip" ] && RUNS="$RUNS runs/$r"
done
say "evaluate:$RUNS"
if ! py evaluate_repeated.py --runs $RUNS > runs/v6_eval.log 2>&1; then
  say "FAILED evaluation (see runs/v6_eval.log)"; exit 1
fi
say "evaluation done ($(tail -n 1 runs/v6_eval.log))"
say "total $(( ($(date +%s) - T0) / 60 )) min"

if [ "$PUSH" = "1" ]; then
  git add runs/main_v6* runs/*/results/summary_repeated.json runs/repeated
  git commit -q -m "v6 A/B/C (joint-limit penalty) training and repeated evaluation" \
    && git push -q && say "pushed" || say "commit/push failed — run git status"
fi
